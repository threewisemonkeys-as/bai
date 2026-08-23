#!/usr/bin/env python3
"""Multi-step forward-dynamics and goal-conditioned planning eval on test50 drives.

Generalizes the one-step test50 ID/FD evals to horizon h:

* msfd: the model sees the K-step history ending at s_t plus the TRUE next h
  actions and predicts the state after ALL of them (raw mode: canonical grid;
  learned mode: perception features, target = P(x_{t+h})).
* plan: the model sees the same history plus the GOAL state s_{t+h} (raw grid,
  or P(goal) features in learned mode) and emits a fully-parameterized action
  sequence of length <= h. The plan is executed in the Autumn engine from the
  replayed drive state at step t; success iff the rendered grid after the plan's
  final action equals the recorded goal grid.

Windows are sampled from the verified seed-0 source drives behind the test50
pools (resolve_sources from the sim rescore). Windows whose goal equals the
start grid and windows solvable by pure passivity (noop^h reproduces the goal)
are excluded, so "do nothing" can never score. The logged actions reach the
goal by construction (every drive is engine-verified frame-by-frame), giving a
1.0 ceiling; random plans of length h (3 per window) give the floor.

    uv run python offline_learning/scripts/eval_multistep_fd_plan.py
"""
from __future__ import annotations

import os as _bos, sys as _bsys  # offline_learning/ on sys.path (flat-import the kept libs)
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
import argparse
import asyncio
import json
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import program_runtime as prt
import test50_sim_tools as T
from clean_sweep import GAMES
from eval_test50_idfd import _reasoning_config, evaluator_llm_call  # import patches gepa_optimize._llm_call
from forward_objective import textdiff_delta_f1
from invdyn_core import DEFAULT_KNOWLEDGE, _tlabel, exact_match_f1
from rescore_test50_id_sim import grids_equal, resolve_sources
from validate import _parse_tag, make_config, run_perceive
from worldcoder_optimize import _clean_program

ACTION_RE = re.compile(r"^(?:up|down|left|right|noop|click \d{1,2} \d{1,2})$")


MSFD_RAW_TMPL = """You predict a FUTURE raw grid of a grid environment, {h} step(s) ahead.

=== DEFAULT KNOWLEDGE (always-true facts about this environment) ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

Below is a trajectory of consecutive RAW GRIDS in canonical JSON, ending at the
CURRENT grid. The action between each prior pair is shown. Use the whole history
to infer any passive, periodic, momentum, selection, or delayed dynamics.

{transcript}

The next {h} action(s) taken from the CURRENT grid, in order, are:
{actions}

Predict the complete raw grid after ALL {h} action(s) have been executed. Apply
each action in order, including any passive dynamics that advance on every step.
Return every row and every cell, including unchanged cells, in exactly the same
canonical JSON representation as the grids above: one compact JSON array,
double-quoted strings, no spaces or markdown. Do not emit an explanation.

<next_state>complete canonical JSON grid after all {h} action(s)</next_state>"""


MSFD_WIN_TMPL = """You predict FUTURE state features of a grid environment, {h} step(s) ahead.

=== DEFAULT KNOWLEDGE (always-true facts about this environment) ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== WORLD KNOWLEDGE (general facts; may be empty) ===
{beliefs}
=== END WORLD KNOWLEDGE ===

Below is a trajectory of consecutive states (summarized as features by a
perception module) ending at the CURRENT state, with the action taken between
each pair. Use the WHOLE history to capture dynamics that depend on more than
the current state alone.

{transcript}

The next {h} action(s) taken from the CURRENT state, in order, are:
{actions}

Predict the features of the state after ALL {h} action(s) have been executed, in
EXACTLY the same format and vocabulary the perception module uses above (same
keys, same coordinate/colour conventions). Apply every action in order,
including its side effects and any passive dynamics between steps. Do NOT add
commentary.

<next_state>predicted features after all {h} action(s), same format as CURRENT</next_state>"""


PLAN_RAW_TMPL = """You control a grid environment and must reach a GOAL state.

=== DEFAULT KNOWLEDGE (always-true facts about this environment) ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

Below is a trajectory of consecutive RAW GRIDS in canonical JSON, ending at the
CURRENT grid. The action between each prior pair is shown. Use the whole history
to infer the dynamics (passive drift, momentum, selection, delayed effects).

{transcript}

=== GOAL raw grid ===
{goal}
=== END GOAL ===

Plan a sequence of AT MOST {cap} action(s) so that, starting from the CURRENT
state and executing your actions in order, the grid after your FINAL action is
EXACTLY the GOAL grid. {timing}; the environment's passive dynamics keep running on every step
(including noop), so timing can matter. Every action must be fully specified:
one of up, down, left, right, noop, or click ROW COL with 0-indexed integers.

Respond as:
<reasoning>how your actions transform CURRENT into GOAL, step by step</reasoning>
<plan>
one action per line, at most {cap} line(s)
</plan>"""


PLAN_WIN_TMPL = """You control a grid environment and must reach a GOAL state.

=== DEFAULT KNOWLEDGE (always-true facts about this environment) ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== WORLD KNOWLEDGE (general facts; may be empty) ===
{beliefs}
=== END WORLD KNOWLEDGE ===

Below is a trajectory of consecutive states (summarized as features by a
perception module) ending at the CURRENT state, with the action taken between
each pair. Use the whole history to infer the dynamics (passive drift, momentum,
selection, delayed effects).

{transcript}

=== GOAL state features (same perception module) ===
{goal}
=== END GOAL ===

Plan a sequence of AT MOST {cap} action(s) so that, starting from the CURRENT
state and executing your actions in order, the state after your FINAL action is
EXACTLY the GOAL state. {timing}; the environment's passive dynamics keep running on every step
(including noop), so timing can matter. Every action must be fully specified:
one of up, down, left, right, noop, or click ROW COL with 0-indexed integers.

Respond as:
<reasoning>how your actions transform CURRENT into GOAL, step by step</reasoning>
<plan>
one action per line, at most {cap} line(s)
</plan>"""


# The plan cap defaults to h (cap == steps_ahead reproduces the historical
# prompts byte-for-byte). With --plan-cap N the horizon only dictates window
# difficulty (how far ahead the goal was recorded); the budget is fixed at N.
def plan_timing(steps_ahead: int) -> str:
    return (f"The GOAL was recorded exactly {steps_ahead} step(s) after the\n"
            "CURRENT state")


# Drives are stored/replayed in the interpreter's `click COL ROW` convention; every
# action SHOWN to the LLM here is canonicalized to `click ROW COL` (matching
# DEFAULT_KNOWLEDGE + the perception's (row,col) cells). De-canonicalization back to
# storage/interpreter convention happens only at SeqSim.run's env.step boundary.
def numbered(actions: list[str]) -> str:
    return "\n".join(f"{i + 1}. {a}" for i, a in enumerate(actions))


def raw_transcript(ctx_grids: list[tuple[str, str]], start_grid: str) -> str:
    lines, n = [], len(ctx_grids)
    for k, (grid, action) in enumerate(ctx_grids):
        idx = -(n - k)
        lines.append(f"STATE[{_tlabel(idx)}] RAW GRID:\n{grid}")
        lines.append(f"  action({_tlabel(idx)} -> {_tlabel(idx + 1)}): {action}")
    lines.append(f"STATE[t] RAW GRID (CURRENT):\n{start_grid}")
    return "\n".join(lines)


def feat_transcript(ctx_z: list[tuple[str, str]], z_t: str) -> str:
    lines, n = [], len(ctx_z)
    for k, (z, action) in enumerate(ctx_z):
        idx = -(n - k)
        lines.append(f"STATE[{_tlabel(idx)}] features:\n{z or '(empty)'}")
        lines.append(f"  action({_tlabel(idx)} -> {_tlabel(idx + 1)}): {action}")
    lines.append(f"STATE[t] (CURRENT) features:\n{z_t or '(empty)'}")
    return "\n".join(lines)


def parse_plan(text: str, dims: tuple[int, int]) -> tuple[list[str] | None, str | None]:
    """Extract a validated action sequence from <plan>. Returns (plan, error)."""
    body = _parse_tag(text, "plan")
    if body is None:
        return None, "no-plan-tag"
    plan = []
    for line in body.splitlines():
        s = re.sub(r"^(?:\d+[.)]|[-*])\s*", "", line.strip().strip("`")).strip()
        if not s:
            continue
        s = re.sub(r"\s+", " ", s.lower())
        if not ACTION_RE.match(s):
            return None, f"invalid-action:{s!r}"
        if s.startswith("click "):
            _, r, c = s.split()
            if int(r) >= dims[0] or int(c) >= dims[1]:
                return None, f"click-out-of-bounds:{s!r}"
        plan.append(s)
    if not plan:
        return None, "empty-plan"
    return plan, None


class SeqSim:
    """Branch executor over one source drive. Either a recorded drive CSV (replayed
    against a seed-0 reset, verified) or engine-generated rows under `seed` (already
    engine-truth, no verification needed)."""

    def __init__(self, game: str, drive_csv: Path = None, rows: list[dict] = None,
                 seed: int = 0, label: str = None):
        self.game = game
        self.seed = seed
        self.drive_csv = drive_csv if drive_csv is not None else label
        self.rows = rows if rows is not None else T.read_rows(drive_csv)
        self.actions = [(r.get("Action") or "").strip() for r in self.rows]
        self.obs = [r.get("Observation") or "" for r in self.rows]
        self.grids = [T.grid_json(o) if o.strip() else None for o in self.obs]
        # Generated drives ARE the engine's output under `seed`, so prefix-replay
        # reproduces them by construction -- verification only applies to recorded
        # (seed-0) drives replayed against a seed-0 reset.
        self.problems = [] if rows is not None else T.verify_drive_in_sim(game, self.rows)

    def run(self, t: int, actions: list[str]) -> list[str | None]:
        """Rendered grid after each action, executed from the drive state at step
        t (prefix replay from a reset at this drive's seed). None after a terminal step."""
        env, _obs = T.make_env(self.game, max_steps=t + len(actions) + 5, seed=self.seed)
        for k in range(t):
            _obs, _, term, _, _ = env.step(self.actions[k])
            if term:
                return [None] * len(actions)
        out, alive = [], True
        for action in actions:  # `actions` is a PLAN in canonical (row,col) -> de-canon for the engine
            if not alive:
                out.append(None)
                continue
            _obs, _, term, _, _ = env.step(action)
            out.append(T.obs_grid(_obs))
            if term:
                alive = False
        return out


def sample_windows(game: str, sims: list[SeqSim], h: int, n: int, context_k: int,
                   rng: random.Random) -> tuple[list[dict], dict]:
    """Sample up to n non-trivial h-step windows across the game's drives."""
    cands = []
    for si, sim in enumerate(sims):
        for t in range(len(sim.rows) - h):
            if all(sim.actions[t + j] for j in range(h)) and \
               all(sim.grids[t + j] is not None for j in range(h + 1)):
                cands.append((si, t))
    rng.shuffle(cands)
    windows, stats = [], {"candidates": len(cands), "static": 0, "noop_solvable": 0}
    for si, t in cands:
        if len(windows) >= n:
            break
        sim = sims[si]
        start, goal = sim.grids[t], sim.grids[t + h]
        if grids_equal(start, goal):
            stats["static"] += 1
            continue
        noop_grids = sim.run(t, ["noop"] * h)
        if grids_equal(noop_grids[-1], goal):
            stats["noop_solvable"] += 1
            continue
        ctx = []
        for j in range(t - 1, max(-1, t - 1 - context_k), -1):
            if not sim.actions[j] or sim.grids[j] is None:
                break
            ctx.insert(0, j)
        windows.append({
            "sim_index": si, "drive": str(sim.drive_csv), "t": t, "h": h,
            "actions": sim.actions[t:t + h], "ctx_steps": ctx,
            "start_grid": start, "goal_grid": goal,
        })
    return windows, stats


def random_plan(game: str, h: int, dims: tuple[int, int], rng: random.Random) -> list[str]:
    verbs = [v for v in GAMES[game][0].split(",") if v]
    plan = []
    for _ in range(h):
        verb = rng.choice(verbs)
        if verb == "click":
            plan.append(f"click {rng.randrange(dims[0])} {rng.randrange(dims[1])}")
        else:
            plan.append(verb)
    return plan


def generate_drive(game: str, seed: int, length: int, rng: random.Random) -> list[dict]:
    """Roll out a random-policy trajectory under `seed` via the REAL engine, returning
    rows [{Step, Action, Observation}] (Observation = the raw grid JSON) compatible
    with SeqSim. Used by the --env-seed path so both the window source AND the
    in-engine planning execution run on the real (non-seed-0-degenerate) game."""
    verbs = [v for v in GAMES[game][0].split(",") if v]
    env, obs = T.make_env(game, max_steps=length + 5, seed=seed)
    rows, dims = [], None
    for s in range(length):
        grid = T.obs_grid(obs)
        if dims is None:
            g = json.loads(grid)
            dims = (len(g), len(g[0]))
        verb = rng.choice(verbs)
        act = f"click {rng.randrange(dims[0])} {rng.randrange(dims[1])}" if verb == "click" else verb
        rows.append({"Step": s, "Action": act, "Observation": grid,
                     "Reasoning": "gen", "Auxiliary_Observation": "", "Reward": 0.0, "Done": False})
        obs, _, term, _, _ = env.step(act)
        if term:
            break
    return rows


async def call_retry(cfg, sem, prompt: str, attempts: int):
    errors, total_cost, text = [], 0.0, ""
    for attempt in range(1, attempts + 1):
        try:
            async with sem:
                text, cost = await evaluator_llm_call(cfg, prompt)
            text = text or ""
            total_cost += float(cost or 0.0)
            if text.strip():
                return text, total_cost, errors
            errors.append(f"attempt {attempt}: empty response")
        except Exception as exc:  # noqa: BLE001 - record and retry any provider error
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
        if attempt < attempts:
            await asyncio.sleep(min(2 ** attempt, 8))
    return text, total_cost, errors


def program_history(win: dict):
    """ctx_raw [(grid_json, storage_action)] -> [(Grid, Action)] for T-hat."""
    return [(json.loads(g), prt.parse_action(a))
            for g, a in win["ctx_raw"]]


def find_artifact(artifact_dir: Path, kind: str, seed: int) -> Path | None:
    """Locate a ship artifact regardless of the optimizer-name infix
    (best_<kind>_<gepa|rexpure|...>_seed<seed>.<ext>). Tries the known infixes for
    determinism, then falls back to a glob so any future optimizer name resolves."""
    ext = "py" if kind == "perception" else "txt"
    for infix in ("rexpure", "gepa"):
        p = artifact_dir / f"best_{kind}_{infix}_seed{seed}.{ext}"
        if p.exists():
            return p
    hits = sorted(artifact_dir.glob(f"best_{kind}_*_seed{seed}.{ext}"))
    return hits[0] if hits else None


async def eval_game(game: str, artifact_dir: Path, cfg, args, seed: int) -> dict:
    code_path = find_artifact(artifact_dir, "perception", seed)
    if code_path is None:
        raise FileNotFoundError(
            f"{game}: no best_perception_*_seed{seed}.py in {artifact_dir}")
    code = code_path.read_text()
    beliefs_path = find_artifact(artifact_dir, "beliefs", seed)
    beliefs = beliefs_path.read_text() if beliefs_path else ""
    program_path = args.program_artifacts.get(game)
    program_rt = None
    if program_path is not None:
        program_rt = prt.ProgramRuntime(
            _clean_program(Path(program_path).read_text()),
            timeout_s=args.program_timeout)

    # Effective engine seed for THIS game. Games in --gen-games are forced to fresh
    # generated drives under --gen-seed (truly-unseen, non-degenerate; used for the
    # RNG games s2kt7/83wkq whose recorded seed-0 drives are degenerate); every other
    # game follows the global --env-seed (0 => recorded test50 source drives).
    eff_seed = args.gen_seed if game in args.gen_games else args.env_seed
    if eff_seed:  # non-zero => generate fresh drives on the real game under this seed
        rng = random.Random(f"gendrive:{eff_seed}:{game}")
        sims = []
        for di in range(args.n_drives):
            dseed = eff_seed + di  # per-drive seeds (all non-zero) for variety
            rows = generate_drive(game, dseed, args.drive_length, rng)
            if len(rows) >= 2:
                sims.append(SeqSim(game, rows=rows, seed=dseed, label=f"gen_seed{dseed}"))
        if not sims:
            raise RuntimeError(f"{game}: generated no usable drives under eff_seed={eff_seed}")
    else:  # eff_seed == 0: recorded (seed-0) drives, verified against a seed-0 reset
        sources = resolve_sources(game, args.data_root)
        drive_csvs = sorted({p for paths in sources.values() for p in paths})
        sims = [SeqSim(game, p) for p in drive_csvs]
        bad = {str(s.drive_csv): s.problems for s in sims if s.problems}
        if bad:
            raise RuntimeError(f"{game}: drive verification failed: {bad}")

    windows = []
    sampling = {}
    for h in args.horizons:
        rng = random.Random(f"msplan:{args.seed}:{game}:{h}")
        wins, stats = sample_windows(game, sims, h, args.windows, args.context_k, rng)
        sampling[str(h)] = {**stats, "sampled": len(wins)}
        windows.extend(wins)

    # Perception features for every referenced (drive, step), computed once.
    perc_cache: dict[tuple[int, int], tuple[str, str | None]] = {}

    def perceive(si: int, step: int):
        key = (si, step)
        if key not in perc_cache:
            perc_cache[key] = run_perceive(code, sims[si].obs[step])
        return perc_cache[key]

    for win in windows:
        si, t, h = win["sim_index"], win["t"], win["h"]
        sim = sims[si]
        win["ctx_raw"] = [(sim.grids[j], sim.actions[j]) for j in win["ctx_steps"]]
        z_ctx, err = [], None
        for j in win["ctx_steps"]:
            z, e = perceive(si, j)
            err = err or e
            z_ctx.append((z, sim.actions[j]))
        z_t, e_t = perceive(si, t)
        z_goal, e_g = perceive(si, t + h)
        win["z_ctx"], win["z_t"], win["z_goal"] = z_ctx, z_t, z_goal
        win["perception_error"] = err or e_t or e_g
        g = json.loads(win["start_grid"])
        win["dims"] = (len(g), len(g[0]))

    sem = asyncio.Semaphore(args.concurrency)

    async def one_window(wi: int, win: dict):
        h = win["h"]
        cap = args.plan_cap or h
        prompts = {
            "fd_raw": MSFD_RAW_TMPL.format(
                h=h, default_knowledge=DEFAULT_KNOWLEDGE,
                transcript=raw_transcript(win["ctx_raw"], win["start_grid"]),
                actions=numbered(win["actions"])),
            "fd_learned": MSFD_WIN_TMPL.format(
                h=h, default_knowledge=DEFAULT_KNOWLEDGE,
                beliefs=beliefs.strip() or "(empty)",
                transcript=feat_transcript(win["z_ctx"], win["z_t"]),
                actions=numbered(win["actions"])),
            "plan_raw": PLAN_RAW_TMPL.format(
                cap=cap, timing=plan_timing(h), default_knowledge=DEFAULT_KNOWLEDGE,
                transcript=raw_transcript(win["ctx_raw"], win["start_grid"]),
                goal=win["goal_grid"]),
            "plan_learned": PLAN_WIN_TMPL.format(
                cap=cap, timing=plan_timing(h), default_knowledge=DEFAULT_KNOWLEDGE,
                beliefs=beliefs.strip() or "(empty)",
                transcript=feat_transcript(win["z_ctx"], win["z_t"]),
                goal=win["z_goal"] or "(empty)"),
        }
        if args.no_llm:
            return wi, {k: ("", 0.0, ["no-llm"]) for k in prompts}, prompts
        results = await asyncio.gather(
            *(call_retry(cfg, sem, prompts[k], args.attempts) for k in prompts)
        )
        return wi, dict(zip(prompts, results)), prompts

    responses = await asyncio.gather(*(one_window(i, w) for i, w in enumerate(windows)))

    fd_rows, plan_rows, cost = [], [], 0.0
    for wi, calls, prompts in sorted(responses, key=lambda r: r[0]):
        win = windows[wi]
        sim, t, h = sims[win["sim_index"]], win["t"], win["h"]
        cap = args.plan_cap or h
        base = {
            "window": wi, "h": h, "drive": win["drive"], "t": t,
            "actions": win["actions"], "perception_error": win["perception_error"],
        }
        for mode, start, target in (
            ("raw", win["start_grid"], win["goal_grid"]),
            ("learned", win["z_t"], win["z_goal"]),
        ):
            text, call_cost, errors = calls[f"fd_{mode}"]
            cost += call_cost
            pred = _parse_tag(text, "next_state") or text.strip()
            fd_rows.append({
                **base, "mode": mode, "pred": pred,
                "prompt": prompts[f"fd_{mode}"], "response": text,
                "retry_errors": errors, "cost": call_cost,
                "exact": exact_match_f1(pred, target),
                "partial": textdiff_delta_f1(start, pred, target),
                "stale_exact": exact_match_f1(start, target),
                "stale_partial": textdiff_delta_f1(start, start, target),
            })
        llm_plans = {}
        for mode in ("raw", "learned"):
            text, call_cost, errors = calls[f"plan_{mode}"]
            cost += call_cost
            plan, perr = parse_plan(text, win["dims"])
            llm_plans[mode] = plan if perr is None else None
            row = {
                **base, "mode": mode, "plan": plan, "plan_error": perr,
                "prompt": prompts[f"plan_{mode}"], "response": text,
                "retry_errors": errors, "cost": call_cost,
                "success": False, "reached_at": None,
            }
            if plan is not None and len(plan) > cap:
                row["plan_error"] = f"budget-exceeded:{len(plan)}>{cap}"
            elif plan is not None and not args.no_llm:
                grids = sim.run(t, plan)
                row["success"] = grids_equal(grids[-1], win["goal_grid"])
                for j, g in enumerate(grids):
                    if grids_equal(g, win["goal_grid"]):
                        row["reached_at"] = j + 1
                        break
            plan_rows.append(row)

        if program_rt is not None:
            # ---- program modes: zero LLM calls -------------------------------
            hist = program_history(win)
            start_g = json.loads(win["start_grid"])
            goal_g = json.loads(win["goal_grid"])
            goal_c = prt.canon_grid(goal_g)
            acts = [prt.parse_action(a) for a in win["actions"]]

            # msfd-program: REAL closed-loop h-step rollout of T-hat.
            t0p = time.time()
            rolled = prt.rollout(program_rt, hist, start_g, acts,
                                 context_k=args.context_k)
            endpoint = rolled[-1] if rolled else None
            pred_c = prt.canon_grid(endpoint) if endpoint is not None else ""
            fd_rows.append({
                **base, "mode": "program", "pred": pred_c,
                "prompt": "", "response": "(program closed-loop rollout)",
                "retry_errors": [], "cost": 0.0,
                "exact": 1.0 if pred_c and grids_equal(pred_c, win["goal_grid"]) else 0.0,
                "partial": textdiff_delta_f1(win["start_grid"], pred_c, win["goal_grid"]),
                "stale_exact": exact_match_f1(win["start_grid"], win["goal_grid"]),
                "stale_partial": textdiff_delta_f1(
                    win["start_grid"], win["start_grid"], win["goal_grid"]),
                "rollout_failed_at": (rolled.index(None) + 1
                                      if None in rolled else None),
                "program_seconds": round(time.time() - t0p, 2),
            })

            # plan-program: search over T-hat (BFS / beam), then execute in engine.
            verbs = [v for v in GAMES[game][0].split(",") if v]
            universe = prt.build_action_universe(verbs, start_g, goal_g)
            t0p = time.time()
            # allow_empty=False: success is scored on grids[-1], so a zero-length plan
            # has no endpoint to score (and would IndexError); make the search find a
            # >=1-step plan that holds the goal instead of short-circuiting to [].
            found = prt.plan_search(
                program_rt, hist, start_g, goal_g, universe, cap,
                beam=args.search_beam, node_budget=args.search_budget,
                context_k=args.context_k, allow_empty=False)
            search_secs = round(time.time() - t0p, 2)
            plan_str = [prt.unparse_action(a) for a in found] if found is not None else None
            row = {
                **base, "mode": "program", "plan": plan_str,
                "plan_error": None if plan_str is not None else "no-plan-found",
                "prompt": "", "response": f"(plan_search, {search_secs}s, "
                f"{program_rt.n_calls} T-hat calls total)",
                "retry_errors": [], "cost": 0.0,
                "success": False, "reached_at": None,
                "search_seconds": search_secs,
            }
            if plan_str is not None:
                grids = sim.run(t, plan_str)
                row["success"] = grids_equal(grids[-1], win["goal_grid"])
                for j, g in enumerate(grids):
                    if grids_equal(g, win["goal_grid"]):
                        row["reached_at"] = j + 1
                        break
            plan_rows.append(row)

            # plan-hybrid: T-hat verifies/selects among the LLM plans + the search
            # plan (no extra LLM calls); execute the best simulated candidate.
            cands = []
            for src in ("raw", "learned"):
                if llm_plans.get(src) and len(llm_plans[src]) <= cap:
                    cands.append((src, llm_plans[src]))
            if plan_str:
                cands.append(("search", plan_str))
            chosen, chosen_src, verified = None, None, False
            for src, cand in cands:
                cacts = [prt.parse_action(a) for a in cand]
                sim_grids = prt.rollout(program_rt, hist, start_g, cacts,
                                        context_k=args.context_k)
                end = sim_grids[-1] if sim_grids else None
                ok = end is not None and prt.canon_grid(end) == goal_c
                if ok and (chosen is None or len(cand) < len(chosen)):
                    chosen, chosen_src, verified = cand, src, True
            if chosen is None and cands:
                chosen, chosen_src = cands[0][1], cands[0][0] + "-unverified"
            row = {
                **base, "mode": "hybrid", "plan": chosen,
                "plan_error": None if chosen else "no-candidates",
                "prompt": "", "response": f"(T-hat selected {chosen_src}; "
                f"verified={verified}; candidates={[s for s, _ in cands]})",
                "retry_errors": [], "cost": 0.0,
                "success": False, "reached_at": None,
                "hybrid_source": chosen_src, "hybrid_verified": verified,
            }
            if chosen:
                grids = sim.run(t, chosen)
                row["success"] = grids_equal(grids[-1], win["goal_grid"])
                for j, g in enumerate(grids):
                    if grids_equal(g, win["goal_grid"]):
                        row["reached_at"] = j + 1
                        break
            plan_rows.append(row)
        rng = random.Random(f"msplan-rand:{args.seed}:{game}:{h}:{win['drive']}:{t}")
        hits = 0
        for _ in range(args.random_plans):
            grids = sim.run(t, random_plan(game, h, win["dims"], rng))
            hits += grids_equal(grids[-1], win["goal_grid"])
        win["random_success"] = hits / args.random_plans if args.random_plans else 0.0

    def fd_summary(rows):
        return {
            "n": len(rows),
            "exact": sum(r["exact"] for r in rows) / len(rows) if rows else None,
            "partial": sum(r["partial"] for r in rows) / len(rows) if rows else None,
            "stale_exact": sum(r["stale_exact"] for r in rows) / len(rows) if rows else None,
            "perception_errors": sum(bool(r["perception_error"]) for r in rows),
        }

    def plan_summary(rows, wins):
        valid = [r for r in rows if r["plan"] is not None and not r["plan_error"]]
        return {
            "n": len(rows),
            "success": sum(r["success"] for r in rows) / len(rows) if rows else None,
            "invalid": sum(r["plan"] is None or bool(r["plan_error"]) for r in rows),
            "mean_plan_len": (sum(len(r["plan"]) for r in valid) / len(valid)) if valid else None,
            "shorter_than_h": sum(len(r["plan"]) < r["h"] for r in valid),
            "random_success": (sum(w["random_success"] for w in wins) / len(wins)) if wins else None,
        }

    fd_modes = sorted({r["mode"] for r in fd_rows})
    plan_modes = sorted({r["mode"] for r in plan_rows})
    summary = {}
    for h in args.horizons:
        wins_h = [w for w in windows if w["h"] == h]
        summary[str(h)] = {
            "windows": len(wins_h),
            "fd": {m: fd_summary([r for r in fd_rows if r["h"] == h and r["mode"] == m])
                   for m in fd_modes},
            "plan": {m: plan_summary([r for r in plan_rows if r["h"] == h and r["mode"] == m], wins_h)
                     for m in plan_modes},
        }
    if program_rt is not None:
        program_rt.close()

    window_records = [
        {k: w[k] for k in ("drive", "t", "h", "actions", "start_grid", "goal_grid",
                           "z_t", "z_goal", "perception_error", "random_success")
         if k in w}
        for w in windows
    ]
    return {
        "game": game, "artifact_dir": str(artifact_dir), "sampling": sampling,
        "env_seed": eff_seed, "drive_source": "generated" if eff_seed else "recorded",
        "program_artifact": str(program_path) if program_path else None,
        "drives": [str(s.drive_csv) for s in sims], "cost": cost,
        "summary": summary, "windows": window_records,
        "fd_rows": fd_rows, "plan_rows": plan_rows,
    }


def render_report(payload: dict) -> str:
    horizons = payload["config"]["horizons"]
    hcols = "".join(f" h={h} |" for h in horizons)
    cfg = payload["config"]
    gen_games = cfg.get("gen_games") or []
    # Describe drive sourcing accurately per game rather than assuming seed-0.
    src_by_game = {res["game"]: res.get("drive_source", "?") for res in payload["results"]}
    gen_list = [g for g in src_by_game if src_by_game[g] == "generated"]
    rec_list = [g for g in src_by_game if src_by_game[g] == "recorded"]
    src_line = "Drive sourcing: "
    if rec_list:
        src_line += f"recorded test50 source drives for {', '.join(rec_list)}"
    if gen_list:
        src_line += ("; " if rec_list else "") + (
            f"FRESH engine-generated drives (unseen seed {cfg.get('gen_seed')}, "
            f"non-degenerate) for {', '.join(gen_list)}")
    src_line += "."
    lines = [
        "# Multi-step FD + goal-conditioned planning", "",
        src_line,
        "Goal==start and noop-solvable windows excluded, so passivity never scores. FD: predict the",
        "state after the true h-action sequence (exact match; raw grids vs learned",
        "P features). Plan: emit <=h fully-parameterized actions; executed in the",
        "Autumn engine from the replayed state at t; success iff the grid after the",
        "final action equals the recorded goal. `rand` = mean success of random",
        f"length-h plans ({payload['config']['random_plans']} per window). The logged",
        "actions always succeed by construction (ceiling 1.0).", "",
    ]
    if payload["config"].get("plan_cap"):
        lines += [f"NOTE: fixed plan cap = {payload['config']['plan_cap']} actions at EVERY "
                  "horizon (h only sets how far ahead the goal was recorded).", ""]

    # mode-agnostic columns (raw/learned LLM, program = T-hat rollout/search, hybrid).
    order = ("raw", "learned", "program", "hybrid")
    fd_modes = [m for m in order if any(
        m in res["summary"][str(h)]["fd"] for res in payload["results"] for h in horizons)]
    plan_modes = [m for m in order if any(
        m in res["summary"][str(h)]["plan"] for res in payload["results"] for h in horizons)]

    def cell(v):
        return "—" if v is None else f"{v:.2f}"

    lines += [
        f"## Multi-step FD exact", "",
        "| game |" + "".join(f" {m}:{hcols}" for m in fd_modes),
        "|---|" + "---:|" * (len(fd_modes) * len(horizons)),
    ]
    for res in payload["results"]:
        s = res["summary"]
        row = f"| {res['game']} |"
        for mode in fd_modes:
            for h in horizons:
                fd = s[str(h)]["fd"].get(mode)
                row += f" {cell(fd['exact'] if fd else None)} |"
        lines.append(row)
    lines += [
        "", "## Planning success (executed in engine)", "",
        f"| game | rand:{hcols}" + "".join(f" {m}:{hcols}" for m in plan_modes),
        "|---|" + "---:|" * ((1 + len(plan_modes)) * len(horizons)),
    ]
    for res in payload["results"]:
        s = res["summary"]
        row = f"| {res['game']} |"
        for h in horizons:
            row += f" {cell(s[str(h)]['plan']['raw']['random_success'])} |"
        for mode in plan_modes:
            for h in horizons:
                p = s[str(h)]["plan"].get(mode)
                row += f" {cell(p['success'] if p else None)} |"
        lines.append(row)

    lines += ["", "## Averages over games", ""]
    metrics = [(f"FD exact {m}",
                lambda s, h, m=m: (s[str(h)]["fd"].get(m) or {}).get("exact"))
               for m in fd_modes]
    metrics.append(("Plan success rand",
                    lambda s, h: s[str(h)]["plan"]["raw"]["random_success"]))
    metrics += [(f"Plan success {m}",
                 lambda s, h, m=m: (s[str(h)]["plan"].get(m) or {}).get("success"))
                for m in plan_modes]
    for label, path_fn in metrics:
        vals = []
        for h in horizons:
            per_game = [path_fn(res["summary"], h) for res in payload["results"]
                        if path_fn(res["summary"], h) is not None]
            vals.append(sum(per_game) / len(per_game) if per_game else None)
        lines.append(
            f"- {label}: " + ", ".join(f"h={h}: {cell(v)}" for h, v in zip(horizons, vals))
        )
    if payload.get("skipped"):
        lines += ["", "Skipped: " + ", ".join(f"{g} ({w})" for g, w in payload["skipped"].items())]
    return "\n".join(lines) + "\n"


async def main_async(args):
    id_payload = json.loads(args.id_json.read_text())
    config_in = id_payload["config"]
    args.data_root = Path(config_in["data_root"])
    args.context_k = config_in["context_k"]
    args.seed = config_in["seed"]
    artifact_dirs = {r["game"]: Path(r["artifact_dir"]) for r in id_payload["results"]}
    args.program_artifacts = {}
    for spec in args.program_artifact or []:
        game, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--program-artifact expects game=path, got {spec!r}")
        args.program_artifacts[game] = path
    if args.program_root:
        root = Path(args.program_root)
        for d in sorted(root.glob(f"*_seed{config_in['seed']}")):
            g = d.name.rsplit("_seed", 1)[0]
            p = d / f"best_transition_wc_seed{config_in['seed']}.py"
            if p.exists():
                args.program_artifacts.setdefault(g, str(p))
    if args.program_artifacts:
        print(f"[program] artifacts: {args.program_artifacts}", flush=True)
    games = [g for g in config_in["games"] if not args.games or g in args.games]
    task_model = args.model or config_in["task_model"]
    cfg = make_config(task_model, config_in["client"])

    # Engine seed: -1 (default) => pick a random NON-ZERO seed and generate fresh drives
    # on the real game; 0 => historical recorded seed-0 drives (degenerate for RNG games);
    # N>0 => fixed non-zero seed. Chosen once and logged for reproducibility.
    if args.env_seed is None or args.env_seed < 0:
        args.env_seed = random.randrange(1, 1_000_000)
        print(f"[env-seed] picked random non-zero engine seed = {args.env_seed} "
              f"(generating {args.n_drives} fresh drives/game of length {args.drive_length})", flush=True)
    elif args.env_seed == 0:
        print("[env-seed] 0 => recorded seed-0 drives (WARNING: degenerate for RNG games)", flush=True)
    else:
        print(f"[env-seed] fixed engine seed = {args.env_seed} (fresh generated drives)", flush=True)
    if args.gen_games:
        print(f"[gen-games] {sorted(args.gen_games)} FORCED to fresh generated drives under "
              f"base seed {args.gen_seed} (unseen, non-degenerate) regardless of --env-seed", flush=True)

    out_json = args.out.with_suffix(".json")
    results, skipped = [], {}
    if args.resume and out_json.exists():
        prior = json.loads(out_json.read_text())
        if prior.get("config", {}).get("horizons") == args.horizons and \
           prior.get("config", {}).get("windows") == args.windows and \
           (prior.get("config", {}).get("plan_cap") or 0) == args.plan_cap:
            results = prior.get("results", [])
    done = {r["game"] for r in results}

    config = {
        "id_source": str(args.id_json), "seed": args.seed,
        "context_k": args.context_k, "horizons": args.horizons,
        "windows": args.windows, "random_plans": args.random_plans,
        "plan_cap": args.plan_cap,
        "task_model": task_model, "client": config_in["client"],
        "reasoning": _reasoning_config(),
        "program_artifacts": dict(args.program_artifacts),
        "search_beam": args.search_beam, "search_budget": args.search_budget,
        "data_root": str(args.data_root),
        "env_seed": args.env_seed,
        "gen_games": sorted(args.gen_games), "gen_seed": args.gen_seed,
        "n_drives": args.n_drives, "drive_length": args.drive_length,
        "protocol": (
            f"windows from FRESH random-policy drives generated on the real game under "
            f"engine seed {args.env_seed} (per-drive {args.env_seed}+i); goal==start and "
            f"noop^h-solvable excluded; plans executed in-engine from prefix-replayed state at t"
            if args.env_seed else
            "windows from recorded seed-0 drives (degenerate for RNG games); goal==start and "
            "noop^h-solvable excluded; plans executed in-engine from prefix-replayed state at t"
        ),
    }
    started = time.time()
    for game in games:
        if game in done:
            print(f"[resume] {game}", flush=True)
            continue
        t0 = time.time()
        try:
            res = await eval_game(game, artifact_dirs[game], cfg, args, args.seed)
        except (RuntimeError, FileNotFoundError) as exc:
            skipped[game] = str(exc)
            print(f"[skip] {game}: {exc}", flush=True)
            continue
        results.append(res)
        parts = []
        # Modes are listed in sorted() order -- "learned" before "raw", and program modes
        # after. The label below is generated from that same order rather than hardcoded:
        # it used to read "(raw/learned)" while printing learned-first, which inverts every
        # arm comparison for anyone reading the progress line instead of the json.
        fd_modes = sorted(res["summary"][str(args.horizons[0])]["fd"])
        plan_modes = sorted(res["summary"][str(args.horizons[0])]["plan"])
        for h in args.horizons:
            s = res["summary"][str(h)]
            fd_bits = "/".join(
                f"{(s['fd'][m]['exact'] if s['fd'][m]['exact'] is not None else -1):.2f}"
                for m in fd_modes)
            plan_bits = "/".join(
                f"{(s['plan'][m]['success'] if s['plan'][m]['success'] is not None else -1):.2f}"
                for m in plan_modes)
            parts.append(f"h{h}: fd {fd_bits} plan {plan_bits}")
        print(f"[done] {game}: {' | '.join(parts)} "
              f"(fd {'/'.join(fd_modes)}; plan {'/'.join(plan_modes)}; "
              f"{time.time() - t0:.0f}s, USD {res['cost']:.3f})", flush=True)
        payload = {"config": config, "elapsed_seconds": time.time() - started,
                   "results": results, "skipped": skipped}
        out_json.write_text(json.dumps(payload, indent=2))

    payload = {"config": config, "elapsed_seconds": time.time() - started,
               "results": results, "skipped": skipped}
    out_json.write_text(json.dumps(payload, indent=2))
    report = render_report(payload)
    args.out.with_suffix(".md").write_text(report)
    print(report, flush=True)
    print(f"wrote {out_json} and {args.out.with_suffix('.md')}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id-json", type=Path,
                    default=REPO / "logs/id_eval_test50_raw_vs_learned.json")
    ap.add_argument("--out", type=Path, default=REPO / "logs/multistep_fd_plan_eval")
    ap.add_argument("--model", type=str, default="",
                    help="override task model (default: task_model from the ID json)")
    ap.add_argument("--games", type=str, default="",
                    help="comma-separated subset; default = all games in the ID json")
    ap.add_argument("--horizons", type=str, default="1,2,4,8")
    ap.add_argument("--windows", type=int, default=10,
                    help="windows per game per horizon")
    ap.add_argument("--random-plans", type=int, default=3)
    ap.add_argument("--plan-cap", type=int, default=0,
                    help="fixed max plan length for ALL horizons (0 = cap at h; "
                    "with a cap, h only dictates how far ahead the goal was recorded)")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--no-llm", action="store_true",
                    help="build+verify windows and baselines only; no LLM calls "
                    "(program modes still run -- they are LLM-free)")
    ap.add_argument("--program-artifact", action="append", metavar="GAME=PATH",
                    help="WorldCoder program world model per game; adds fd mode "
                    "'program' (closed-loop T-hat rollout) + plan modes 'program' "
                    "(search over T-hat) and 'hybrid' (T-hat verifies LLM plans)")
    ap.add_argument("--program-root", default=None,
                    help="root with <game>_seed<N>/best_transition_wc_seed<N>.py")
    ap.add_argument("--program-timeout", type=float, default=1.0)
    ap.add_argument("--search-beam", type=int, default=64)
    ap.add_argument("--search-budget", type=int, default=5000,
                    help="max T-hat calls per plan_search window")
    ap.add_argument("--env-seed", type=int, default=-1,
                    help="global engine seed. -1 (default): pick a RANDOM non-zero seed and "
                    "generate fresh real-game drives. 0: recorded seed-0 drives "
                    "(degenerate for RNG games). N>0: fixed non-zero seed.")
    ap.add_argument("--gen-games", type=str, default="",
                    help="comma-separated games FORCED to fresh generated drives under "
                    "--gen-seed, overriding --env-seed (for RNG games like s2kt7/83wkq "
                    "whose recorded seed-0 drives are degenerate). Other games follow --env-seed.")
    ap.add_argument("--gen-seed", type=int, default=700001,
                    help="base engine seed for --gen-games drives (must be disjoint from "
                    "each game's training seeds so windows are truly unseen)")
    ap.add_argument("--n-drives", type=int, default=6,
                    help="generated drives per game (--env-seed != 0)")
    ap.add_argument("--drive-length", type=int, default=40,
                    help="steps per generated drive (--env-seed != 0)")
    args = ap.parse_args()
    args.games = set(filter(None, args.games.split(",")))
    args.gen_games = set(filter(None, args.gen_games.split(",")))
    args.horizons = [int(h) for h in args.horizons.split(",") if h]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
