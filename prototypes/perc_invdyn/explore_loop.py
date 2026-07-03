"""Runnable explore -> act -> learn loop (no-reset edition).

Closes the loop the prototypes were building toward: STEP the live env so as to collect
the best data for the inverse-dynamics learner, then LEARN P/B from it, repeat.

  (1) DECIDE   select_action() scores the available actions by expected learning value
               (observability + frontier disagreement + coverage + contrast) -- explore_score.
  (2) ACT      env.step(a); append (X_t, a, X_{t+1}) to the buffer.
  (3) LEARN    every k steps, GEPA-optimize P/B on the buffer with a held-out gate;
               refresh the frontier (= the disagreement ensemble for step 1).

NO RESET (assumed): a fresh EPISODE is the only coarse reset -> start a new one on
done/stuck. The counter-decoy is NOT masked here; P is left to learn it away (purity:
the scorer never edits P's output). See explore_score.py for the acquisition rationale.

Env: BALROG `make_env` via Hydra compose. Validated bootable on autumn/DQ8GC (the one
clearly-solvable inverse-dynamics game; invdyn-perception-validated). Raw text obs =
obs["text"]["long_term_context"] -- identical to the trajectory.csv the learner trains on,
so collected transitions are in-distribution with the offline pipeline.

Usage (tiny, cheap default):
  uv run prototypes/perc_invdyn/explore_loop.py \
      --env autumn --task DQ8GC --budget 40 --warmup 12 --k-relearn 14 \
      --max-metric-calls 80 --task-model google/gemini-2.5-flash

Cost knobs: select_action issues |actions|x|frontier| forward LLM calls per decided step;
warmup is zero-LLM (round-robin actions). GEPA cost scales with --max-metric-calls.
"""

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import explore_score as X
from explore_score import _perc, _hash_state, select_action, missing_at_context
from validate import Transition, make_config, run_perceive
import gepa_optimize as G
import gepa


# ---------------------------------------------------------------------------
# Live env wrapper: exposes exactly what the loop needs over BALROG's EnvWrapper.
# ---------------------------------------------------------------------------
# Episode-control actions: end/restart the episode rather than transition the world. NOT
# dynamics actions, and stepping `reset` mid-collection would corrupt the buffer -> excluded
# from the derived dynamics action set. This tiny set is the only hand-named action knowledge
# (universal control verbs, not task semantics); everything else comes from the env.
_META_ACTIONS = {"reset", "quit", "restart", "exit"}


class LiveEnv:
    """Boots one BALROG env and exposes observe / available_actions / step / new_episode.
    available_actions() is DERIVED from the env's language_action_space: keep the single-token
    concrete actions (left/right/up/down/noop, ARC ACTION1..), drop parametric templates
    (multi-token, e.g. 'click ROW COL ...' / 'ACTION6 x= y=' -- can't be passed verbatim and
    the loop can't enumerate their args) and the _META_ACTIONS. An explicit whitelist (verbs)
    may be passed to override the derivation."""

    def __init__(self, env_name, task, cfg, action_whitelist: set[str] | None = None):
        # Dispatch the right factory per env family (mirrors stepwise_eb_learn): autumn and
        # arc_agi have dedicated factories; everything else goes through BALROG make_env.
        def _factory():
            if env_name == "arc_agi":
                from arc_agi_env import make_arc_env
                return make_arc_env(task, cfg)
            if env_name == "autumn":
                from autumn_env import make_autumn_env
                return make_autumn_env(task, cfg)
            from balrog.environments import make_env
            return make_env(env_name, task, cfg)
        self._make = _factory
        self.whitelist = action_whitelist  # None -> derive from the env each step
        self.env = None
        self._x = None
        self._ep = -1

    def _text(self, obs) -> str:
        return obs["text"]["long_term_context"]

    def new_episode(self, seed) -> str:
        self.env = self._make()
        obs, _ = self.env.reset(seed=seed)
        self._x = self._text(obs)
        self._ep += 1
        return self._x

    def observe(self) -> str:
        return self._x

    def _raw_actions(self) -> list[str]:
        acts = getattr(self.env, "language_action_space", None) or []
        vals = getattr(acts, "_values", None)
        return list(vals if vals is not None else acts)

    def available_actions(self) -> list[str]:
        acts = self._raw_actions()
        if self.whitelist is not None:  # explicit override (by verb / first token)
            return [a for a in acts if a.split()[0] in self.whitelist]
        # derive: single-token concrete actions that aren't episode-control.
        return [a for a in acts if len(a.split()) == 1 and a.lower() not in _META_ACTIONS]

    def step(self, action: str):
        obs, reward, term, trunc, info = self.env.step(action)
        x_t1 = self._text(obs)
        tr = Transition(self._x, x_t1, action.split()[0])  # collapse to verb (learner's target)
        self._x = x_t1
        return tr, bool(term or trunc)


# ---------------------------------------------------------------------------
# LEARN step: GEPA-optimize P/B on the current buffer, return the new frontier
# (top-N candidates = the disagreement ensemble) + the held-out ID accuracy metric.
# ---------------------------------------------------------------------------
def _collapse_pool(buffer: list[Transition]) -> list[str]:
    return sorted({tr.action for tr in buffer})


async def _id_accuracy(task_cfg, cand, test, _unused=None) -> float:
    acc, _ = await G.eval_on(task_cfg, G._clean_component("perception", cand.get("perception", "")),
                             cand.get("world_knowledge", ""), test)
    return acc


def relearn(task_cfg, refl_cfg, buffer, args, rng, ensemble_size=3):
    """Run GEPA on the buffer (balanced train/val/test); return (frontier, metric, info).
    frontier = up to `ensemble_size` distinct top-val candidates. metric = best candidate's
    inverse-dynamics accuracy on the held-out test split (neither train nor val touched it)."""
    pool = _collapse_pool(buffer)
    if len(pool) < 2 or len(buffer) < args.min_buffer:
        return None, None, {"skipped": f"buffer={len(buffer)} pool={len(pool)}"}

    data = list(buffer)
    rng.shuffle(data)
    k = args.k_choices
    test_n = min(args.test_n, max(2, len(data) // 4))
    rest, test_tr = G.balanced_split(data, test_n, 10**9, rng)
    val_n = min(args.val_n, max(2, len(rest) // 3))
    train_pool, val_tr = G.balanced_split(rest, val_n, 10**9, rng)
    train_tr = train_pool[: args.train_n]
    if not train_tr or not val_tr or not test_tr:
        return None, None, {"skipped": "empty split"}

    train = G.bake_choices(train_tr, pool, k, rng)
    val = G.bake_choices(val_tr, pool, k, rng)
    test = G.bake_choices(test_tr, pool, k, rng)

    adapter = G.InvDynAdapter(task_cfg, pool, concurrency=args.concurrency,
                              fd_scorer=args.fd_scorer, fd_weight=args.fd_weight)
    # Start EMPTY: no hand-written seed P. The writer builds perception from scratch,
    # guided only by the env-scoped OBSERVATION SCHEMA (format, no dynamics/semantics).
    seed_candidate = {"perception": "", "world_knowledge": ""}
    result = gepa.optimize(
        seed_candidate=seed_candidate, trainset=train, valset=val, adapter=adapter,
        reflection_lm=G.make_reflection_lm(refl_cfg),
        reflection_prompt_template=G.build_reflection_templates(args.env),
        candidate_selection_strategy="pareto", module_selector="round_robin",
        reflection_minibatch_size=min(len(train), 12),
        max_metric_calls=args.max_metric_calls, display_progress_bar=False,
        seed=args.seed, cache_evaluation=True, raise_on_exception=False,
        track_best_outputs=True,
    )
    # frontier = distinct candidates ranked by val aggregate score (best first).
    order = sorted(range(len(result.candidates)),
                   key=lambda i: result.val_aggregate_scores[i], reverse=True)
    frontier, seen = [], set()
    for i in order:
        c = result.candidates[i]
        key = (c.get("perception", ""), c.get("world_knowledge", ""))
        if key in seen:
            continue
        seen.add(key)
        frontier.append(c)
        if len(frontier) >= ensemble_size:
            break

    metric = asyncio.run(_id_accuracy(task_cfg, result.best_candidate, test))
    info = {"buffer": len(buffer), "pool": pool, "train": len(train), "val": len(val),
            "test": len(test), "candidates": result.num_candidates,
            "metric_calls": result.total_metric_calls, "cost": round(adapter.total_cost, 4),
            "frontier_size": len(frontier)}
    return frontier, metric, info


# ---------------------------------------------------------------------------
# The loop.
# ---------------------------------------------------------------------------
def _stuck(recent: list[str], n: int = 6) -> bool:
    return len(recent) >= n and len(set(recent[-n:])) == 1


def _phash(code: str, raw: str) -> str:
    """Hash of P's output on a raw frame (run_perceive returns (output, error))."""
    return _hash_state(run_perceive(code, raw)[0])


def _select_sync(task_cfg, frontier, x_t, actions, action_counts, recent_p, contrast, concurrency):
    """Run one async select_action in its own event loop (the outer loop is sync so GEPA's
    internal asyncio.run can't be nested). The semaphore is created INSIDE the coroutine so
    it binds to this call's loop."""
    async def _run():
        sem = asyncio.Semaphore(concurrency)
        return await select_action(
            cfg=task_cfg, frontier=frontier, x_t=x_t, actions=actions,
            action_counts=action_counts, recent_hashes=set(recent_p[-5:]), sem=sem,
            contrast_actions=contrast)
    return asyncio.run(_run())


def explore_learn(args):
    cfgdir = str((Path(__file__).resolve().parents[2] / "BALROG/balrog/config"))
    overrides = [f"envs.names={args.env}"]
    if args.env == "autumn":
        overrides.append("envs.autumn_kwargs.render_mode=text")  # text obs (no image rendering)
    with initialize_config_dir(config_dir=cfgdir, version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.tasks[f"{args.env}_tasks"] = [args.task]

    # actions: derived from the env by default (None); --actions overrides with explicit verbs.
    whitelist = set(filter(None, args.actions.split(","))) if args.actions else None
    env = LiveEnv(args.env, args.task, cfg, whitelist)
    task_cfg = make_config(args.task_model, args.client)
    refl_cfg = make_config(args.reflection_model or args.task_model, args.client)
    rng = random.Random(args.seed)

    outd = Path(args.out_dir or (Path(__file__).resolve().parents[2] / "logs" / "perc_invdyn"
                / f"loop_{args.env}_{args.task}_{time.strftime('%Y%m%d-%H%M%S')}"))
    outd.mkdir(parents=True, exist_ok=True)
    print(f"[out] {outd}")

    buffer: list[Transition] = []
    frontier = [{"perception": "", "world_knowledge": ""}]  # empty seed, no hand-written P
    action_counts, log = X.Counter(), []
    # recent_raw: hashes of the RAW frame -> P-independent stuck detection (loop plumbing,
    # not learning). recent_p: hashes of P's OUTPUT -> the scorer's revisit penalty, which
    # must live in the same space as the predicted z_hat it is compared against.
    recent_raw, recent_p = [], []
    total_select_cost = 0.0

    x_t = env.new_episode(seed=args.seed)
    recent_raw.append(_hash_state(x_t)); recent_p.append(_phash(_perc(frontier[0]), x_t))

    for step in range(args.budget):
        # LEARN first, on schedule -> never skipped by a stuck/done `continue` below.
        if step >= args.warmup and step % args.k_relearn == 0:
            new_frontier, metric, info = relearn(task_cfg, refl_cfg, buffer, args, rng,
                                                 ensemble_size=args.ensemble)
            if new_frontier:
                frontier = new_frontier
            rec = {"step": step, "buffer": len(buffer), "action_balance": dict(action_counts),
                   "id_acc": metric, "select_cost": round(total_select_cost, 4), **(info or {})}
            log.append(rec)
            (outd / "loop_log.json").write_text(json.dumps(log, indent=2))
            print(f"[relearn @ {step}] id_acc={metric} | {info}", flush=True)

        if _stuck(recent_raw):
            x_t = env.new_episode(seed=args.seed + env._ep + 1)
            recent_raw = [_hash_state(x_t)]; recent_p = [_phash(_perc(frontier[0]), x_t)]
            print(f"[step {step}] STUCK -> new episode {env._ep}", flush=True)
            continue

        actions = env.available_actions()
        if step < args.warmup or len(frontier) < 1:
            # warmup: round-robin the actions -> guarantees action balance + diverse,
            # observable data for the first GEPA fit (greedy/repeat would pin on a wall).
            a = actions[step % len(actions)]
            why = "warmup"
        else:
            contrast = missing_at_context(x_t, actions, _perc(frontier[0]), buffer)
            best, ranked, cost = _select_sync(
                task_cfg, frontier, x_t, actions, action_counts, recent_p, contrast,
                args.concurrency)
            a, why = best.action, (f"obs={best.observability:.2f} dis={best.disagreement:.2f} "
                                   f"cov={best.coverage:.2f} con={best.contrast:.0f}")
            total_select_cost += cost

        tr, done = env.step(a)
        buffer.append(tr)
        action_counts[a.split()[0]] += 1
        x_t = env.observe()
        recent_raw.append(_hash_state(x_t)); recent_p.append(_phash(_perc(frontier[0]), x_t))
        print(f"[step {step}] a={a!r:12} done={done} | {why}", flush=True)

        if done:
            x_t = env.new_episode(seed=args.seed + env._ep + 1)
            recent_raw = [_hash_state(x_t)]; recent_p = [_phash(_perc(frontier[0]), x_t)]

    # persist final P/B
    best = frontier[0]
    (outd / "final_perception.py").write_text(G._clean_component("perception", best.get("perception", "")))
    (outd / "final_beliefs.txt").write_text(best.get("world_knowledge", ""))
    print(f"[done] {len(buffer)} transitions, {env._ep+1} episodes, "
          f"select_cost=${total_select_cost:.4f}. artifacts -> {outd}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="autumn")
    ap.add_argument("--task", default="DQ8GC")
    ap.add_argument("--actions", default=None,
                    help="explicit comma-separated action verbs; default None = derive the "
                         "dynamics action set from the env's language_action_space")
    ap.add_argument("--budget", type=int, default=40, help="total env steps")
    ap.add_argument("--warmup", type=int, default=12, help="random/heuristic steps before model steering")
    ap.add_argument("--k-relearn", type=int, default=14, help="run GEPA every k steps")
    ap.add_argument("--min-buffer", type=int, default=12, help="skip relearn below this buffer size")
    ap.add_argument("--ensemble", type=int, default=3, help="frontier size used as the disagreement ensemble")
    # GEPA / data knobs (mirror gepa_optimize)
    ap.add_argument("--train-n", type=int, default=14)
    ap.add_argument("--val-n", type=int, default=12)
    ap.add_argument("--test-n", type=int, default=10)
    ap.add_argument("--k-choices", type=int, default=5)
    ap.add_argument("--max-metric-calls", type=int, default=80)
    ap.add_argument("--fd-scorer", choices=["none", "textdiff", "judge"], default="none")
    ap.add_argument("--fd-weight", type=float, default=0.5)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--task-model", default="google/gemini-2.5-flash")
    ap.add_argument("--reflection-model", default=None)
    ap.add_argument("--client", default="openrouter")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    explore_learn(args)


if __name__ == "__main__":
    main()
