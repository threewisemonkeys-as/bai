#!/usr/bin/env python3
"""Paired, SCORED replay of the reflection step: does a candidate reflection model write
BETTER children than the incumbent, from identical evidence?

Why not just run the optimizer N times per arm? Because changing the reflection model
changes the proposal, which changes which parent is selected next, which changes
everything downstream -- so an end-to-end arm difference mixes "better proposals" with
"different random walk". At 30 nodes and ~0.06 test-acc noise per run you need dozens of
runs to see through that (cf. aug7 repro: 0.79 vs 0.82 on the SAME config).

This removes the walk. Every arm is handed the SAME logged parent state and the SAME
reflective-dataset prompt from a completed run, proposes a child, and the child is scored
by the SAME frozen F on the SAME train batch -- the exact composite the search optimizes.
The result is a PAIRED sample (one delta per parent), so n = #parents x #games instead of
n = #runs, and the parent's own difficulty cancels out of every delta.

What it can and cannot tell you:
  CAN  -- is arm X's one-step proposal quality worse/equal/better than the incumbent's,
          with enough n to resolve small effects; where it regresses (crashes, constant P,
          empty beliefs, hedging); whether the DIAGNOSIS or the PROPOSER call is the part
          that needs thinking (--arm-analysis re-runs the diagnosis per arm too).
  CANNOT -- compounding/trajectory effects, or whether an arm that proposes differently
          would have reached better regions. That needs the end-to-end blocked runs; this
          is the cheap screen that decides which arms deserve them.

Prompts come from the logged reflection_calls.jsonl of a completed run, matched back to
their parent candidate via process_log.jsonl (which records the selected parent index and
the proposed text per iteration) + candidates.jsonl (the pool).

    uv run python offline_learning/scripts/replay_proposal_ab.py \
        --run-root logs/aug8_hardmin_gptoss20b --games bt3gb,dq8gc \
        --n-parents 8 --arms control,dsnothink,gptoss120b,ling30flash
"""
from __future__ import annotations

import os as _bos, sys as _bsys  # offline_learning/ on sys.path (flat-import the kept libs)
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

import invdyn_core as core  # noqa: E402
from invdyn_core import (  # noqa: E402
    InvDynAdapter, _clean_component, extract_proposed_text, make_reflection_lm,
)
import rexpure_optimize as rex  # noqa: E402
from rexpure_optimize import build_parser as rexpure_argparser  # noqa: E402
from validate import make_config  # noqa: E402

NOTHINK = '{"enabled": false}'
EFFLOW = '{"effort": "low"}'

# arm -> (model, provider order, reasoning json)
ARMS = {
    "control": ("deepseek/deepseek-v4-flash", "deepseek,baidu,fireworks", None),
    "dsnothink": ("deepseek/deepseek-v4-flash", "deepseek,baidu,fireworks", NOTHINK),
    "gptoss120b": ("openai/gpt-oss-120b", "cerebras,groq,sambanova", EFFLOW),
    "ling30flash": ("inclusionai/ling-3.0-flash", "novita,deepinfra", NOTHINK),
}


# --------------------------------------------------------------------- parents
def load_parents(run_dir: Path, n: int, rng: random.Random) -> list[dict]:
    """(parent candidate, component, logged proposer prompt, logged child score) per
    iteration of a completed run. The prompt already has the diagnosis text baked in, so
    replaying it isolates the PROPOSER call; --arm-analysis re-derives it per arm."""
    rd = run_dir / "rexpure_run_seed1"
    plog = [json.loads(l) for l in (rd / "process_log.jsonl").open()]
    refl = [json.loads(l) for l in (rd / "reflection_calls.jsonl").open()]
    cands = {c["idx"]: c for c in (json.loads(l) for l in (rd / "candidates.jsonl").open())}

    # reflection_calls has no iteration field; it is appended in iteration order, one per
    # proposed component, so zip it against the accepted-or-not iterations that proposed.
    proposing = [r for r in plog if r.get("components")]
    out = []
    for rec, call in zip(proposing, refl):
        parent = cands.get(rec.get("selected"))
        if parent is None:
            continue
        out.append({
            "iteration": rec["i"],
            "component": call["component"],
            "parent_idx": rec["selected"],
            "parent": {"perception": parent.get("perception", ""),
                       "world_knowledge": parent.get("world_knowledge", "")},
            "prompt": call["prompt"],
            "logged_child_score": rec.get("new_score"),
        })
    if n and n < len(out):
        out = rng.sample(out, n)
    return sorted(out, key=lambda r: r["iteration"])


# --------------------------------------------------------------------- data
def args_from_cmd(cmd: list[str]):
    """Re-parse a run's saved argv through rexpure_optimize's OWN parser, so the replay
    inherits every data/scoring flag exactly (no hand-copied defaults to drift)."""
    ap = rexpure_argparser()
    return ap.parse_args(cmd[2:])  # drop [python, rexpure_optimize.py]


def build_run_data(cmd: list[str]):
    """Rebuild the source run's train batch via the optimizer's own build_data, then PROVE
    it is the same batch by matching the train fingerprint the run checkpointed."""
    args = args_from_cmd(cmd)
    rng = random.Random(args.seed)
    train, test, pool, context_k, _wl, _trs, id_n = rex.build_data(args, rng)
    return args, train, test, pool, context_k, id_n


def make_adapter(args, action_pool, context_k, train, id_n, concurrency):
    """An adapter configured EXACTLY like the source run's, so the score a replayed child
    gets is the number the search itself would have given it."""
    task_cfg = make_config(args.task_model, args.client,
                           provider_order=args.task_provider_order,
                           reasoning_json=args.task_reasoning_json)
    return InvDynAdapter(
        task_cfg, action_pool, analysis_cfg=task_cfg, concurrency=concurrency,
        fd_scorer=args.fd_scorer, fd_weight=args.fd_weight, fd_reflect=args.fd_reflect,
        analyze_mistakes=False, context_k=context_k, reuse_traces=True,
        gate_train_x=[i["tr"].x_t for i in train] + [i["tr"].x_t1 for i in train],
        id_set_loss=args.id_set_loss, id_eps=args.id_eps, id_n_actions=id_n,
        credited_scoring=args.credited_scoring, composite=args.composite,
        softmin_tau=args.softmin_tau, contrastive_fd=args.contrastive_fd,
        cfd_raw_targets=args.cfd_raw_targets, image_cls=core.Image,
    )


# --------------------------------------------------------------------- reflection
def arm_prompt(parent, component, parent_batch, run_args, arm_cfg, concurrency, action_pool,
               context_k, id_n, train):
    """Rebuild the reflective dataset with THIS ARM running the diagnosis, and render the
    proposer prompt from it. Without this the replay reuses the control's logged diagnosis
    and only tests the proposer half of the stack -- which matters because the diagnosis is
    both the reasoning-heaviest call and (per the bt3gb reflection forensics) the one the
    proposer largely transcribes."""
    a = InvDynAdapter(
        arm_cfg, action_pool, analysis_cfg=arm_cfg, concurrency=concurrency,
        fd_scorer=run_args.fd_scorer, fd_weight=run_args.fd_weight,
        fd_reflect=run_args.fd_reflect, analyze_mistakes=True,
        analyze_mode=run_args.analyze_mode, context_k=context_k, reuse_traces=True,
        gate_train_x=[i["tr"].x_t for i in train] + [i["tr"].x_t1 for i in train],
        id_set_loss=run_args.id_set_loss, id_eps=run_args.id_eps, id_n_actions=id_n,
        composite=run_args.composite, softmin_tau=run_args.softmin_tau,
        contrastive_fd=run_args.contrastive_fd, cfd_raw_targets=run_args.cfd_raw_targets,
        image_cls=core.Image,
    )
    refl = a.make_reflective_dataset(parent, parent_batch, [component])
    records = refl.get(component)
    if not records:
        return None, a.total_cost
    templates = core.build_reflection_templates(None)
    return core.render_reflection_prompt(templates[component], parent.get(component, ""),
                                         records), a.total_cost


# --------------------------------------------------------------------- scoring
def score_child(adapter, train, parent, component, proposal_text):
    """Splice the proposal into the parent and score the child on the full train batch.
    Returns (mean composite, diagnostics) -- a crash / constant-P / empty proposal is a
    REAL 0.0 here, exactly as the search would score it, not an excluded row."""
    text = _clean_component(component, extract_proposed_text(proposal_text or ""))
    child = dict(parent)
    child[component] = text
    if not text.strip():
        return 0.0, {"empty_proposal": True}
    batch = adapter.evaluate(train, child, capture_traces=False)
    scores = [s for s in batch.scores if s is not None]
    gate, gate_z = adapter._constant_p_gate(child.get("perception", ""))
    return (sum(scores) / len(scores) if scores else 0.0), {
        "empty_proposal": False, "const_p_gate": bool(gate),
        "chars": len(text), "zero_rows": sum(1 for s in scores if s == 0.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="logs/aug8_hardmin_gptoss20b")
    ap.add_argument("--games", default="bt3gb,dq8gc")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--n-parents", type=int, default=8,
                    help="parents sampled per game (0 = all ~29)")
    ap.add_argument("--reps", type=int, default=1,
                    help="proposals per (arm, parent). >1 separates arm effect from the "
                         "proposer's own sampling noise, which is the dominant variance")
    ap.add_argument("--arm-analysis", action="store_true",
                    help="each arm also runs its OWN diagnosis call and renders its own "
                         "proposer prompt (tests the whole reflection stack, not just the "
                         "proposer). Costs one extra parent eval per parent + one diagnosis "
                         "per (arm, parent); without it every arm reuses the logged diagnosis")
    ap.add_argument("--score-test", action="store_true",
                    help="also score each child on the run's held-out test batch -- train "
                         "fit is what the search optimizes, this is what you actually care "
                         "about. Roughly doubles F cost")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="logs/aug9_proposal_ab")
    ap.add_argument("--report-only", action="store_true",
                    help="skip generation; just re-report over --out/rows.jsonl (used to "
                         "pool the per-game shards a parallel sweep writes)")
    args = ap.parse_args()

    if args.report_only:
        report(Path(ROOT / args.out) / "rows.jsonl", [a for a in args.arms.split(",") if a in ARMS])
        return

    outd = ROOT / args.out
    outd.mkdir(parents=True, exist_ok=True)
    rows_path = outd / "rows.jsonl"
    done = set()
    if rows_path.exists():  # resumable: skip (game, iteration, arm, rep) already scored
        for line in rows_path.open():
            r = json.loads(line)
            done.add((r["game"], r["iteration"], r["arm"], r["rep"]))
        print(f"[resume] {len(done)} rows already scored")

    arms = [a for a in args.arms.split(",") if a in ARMS]
    rng = random.Random(args.seed)

    for game in args.games.split(","):
        run_dir = ROOT / args.run_root / f"{game}_seed1"
        launch = json.loads((run_dir / "launch.json").read_text())
        parents = load_parents(run_dir, args.n_parents, rng)
        run_args, train, test, pool, context_k, id_n = build_run_data(launch["cmd"])

        # HARD CHECK: same batch as the run, or every score below is incomparable.
        want = json.loads((run_dir / "rexpure_run_seed1" / "resume_state.json").read_text())
        got = core._train_fingerprint(train)
        if got != want["train_fingerprint"]:
            raise SystemExit(f"{game}: rebuilt train batch {got} != run's {want['train_fingerprint']} "
                             "-- the replay would be scoring on a different set")
        adapter = make_adapter(run_args, pool, context_k, train, id_n, args.concurrency)
        print(f"\n=== {game}: {len(parents)} parents x {len(arms)} arms x {args.reps} rep(s) "
              f"| train={len(train)} (fingerprint {got} == run) ctx_k={context_k}")

        for p in parents:
            # one parent eval, shared by every arm's diagnosis (traces are what the
            # reflective dataset is built from)
            parent_batch = (adapter.evaluate(train, p["parent"], capture_traces=True)
                            if args.arm_analysis else None)
            for arm in arms:
                model, pin, reasoning = ARMS[arm]
                cfg = make_config(model, "openrouter", provider_order=pin,
                                  hedge_delay_s=0, timeout_s=300, reasoning_json=reasoning)
                lm = make_reflection_lm(cfg)
                for rep in range(args.reps):
                    if (game, p["iteration"], arm, rep) in done:
                        continue
                    t0 = time.perf_counter()
                    try:
                        prompt = p["prompt"]
                        if args.arm_analysis:
                            prompt, _c = arm_prompt(
                                p["parent"], p["component"], parent_batch, run_args, cfg,
                                args.concurrency, pool, context_k, id_n, train)
                            if prompt is None:
                                raise RuntimeError("no reflective records for component")
                        proposal = lm(prompt)
                        wall = time.perf_counter() - t0
                        score, diag = score_child(adapter, train, p["parent"],
                                                  p["component"], proposal)
                        test_score = None
                        if args.score_test and not diag.get("empty_proposal"):
                            child = dict(p["parent"])
                            child[p["component"]] = _clean_component(
                                p["component"], extract_proposed_text(proposal or ""))
                            tb = adapter.evaluate(test, child, capture_traces=False)
                            ts = [x for x in tb.scores if x is not None]
                            test_score = sum(ts) / len(ts) if ts else 0.0
                        err = None
                    except Exception as e:  # noqa: BLE001
                        wall, score, test_score, diag, err = (
                            time.perf_counter() - t0, None, None, {},
                            f"{type(e).__name__}: {str(e)[:160]}")
                    row = {"game": game, "iteration": p["iteration"], "arm": arm, "rep": rep,
                           "component": p["component"], "parent_idx": p["parent_idx"],
                           "score": score, "test_score": test_score,
                           "logged_child_score": p["logged_child_score"],
                           "propose_wall_s": round(wall, 1), "error": err, **diag}
                    with rows_path.open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(f"  it={p['iteration']:3d} {p['component'][:10]:10s} {arm:12s} "
                          f"train={score if score is None else round(score, 3)}"
                          + (f" test={round(test_score, 3)}" if test_score is not None else "")
                          + f" ({wall:.1f}s){' ' + err if err else ''}", flush=True)

    report(rows_path, arms)


def report(rows_path: Path, arms: list[str]) -> None:
    rows = [json.loads(l) for l in rows_path.open()]
    ok = [r for r in rows if r.get("score") is not None]
    print(f"\n=== {len(ok)}/{len(rows)} scored proposals ===")
    print(f"{'arm':12s} {'n':>4s} {'train':>7s} {'test':>7s} {'zeros':>6s} "
          f"{'constP':>7s} {'empty':>6s} {'wall':>7s}")
    for arm in arms:
        a = [r for r in ok if r["arm"] == arm]
        if not a:
            continue
        sc = [r["score"] for r in a]
        ts = [r["test_score"] for r in a if r.get("test_score") is not None]
        print(f"{arm:12s} {len(a):4d} {statistics.mean(sc):7.3f} "
              f"{statistics.mean(ts) if ts else float('nan'):7.3f} "
              f"{sum(1 for s in sc if s == 0):6d} {sum(1 for r in a if r.get('const_p_gate')):7d} "
              f"{sum(1 for r in a if r.get('empty_proposal')):6d} "
              f"{statistics.mean(r['propose_wall_s'] for r in a):6.1f}s")

    # PAIRED deltas vs control: the whole point -- same parent, same prompt, same scorer.
    base = arms[0]
    for endpoint in ("score", "test_score"):
        by_key = {}
        for r in ok:
            if r.get(endpoint) is not None:
                by_key.setdefault((r["game"], r["iteration"], r["rep"]), {})[r["arm"]] = r[endpoint]
        if len(by_key) < 1:
            continue
        _paired(arms, base, by_key, endpoint)


def _paired(arms, base, by_key, endpoint):
    print(f"\npaired {endpoint} vs {base} (same parent+prompt+scorer):")
    print(f"{'arm':12s} {'pairs':>6s} {'mean_delta':>11s} {'sd':>7s} {'se':>7s} "
          f"{'win':>5s} {'tie':>5s} {'loss':>5s}")
    for arm in arms[1:]:
        d = [v[arm] - v[base] for v in by_key.values() if arm in v and base in v]
        if not d:
            continue
        sd = statistics.stdev(d) if len(d) > 1 else float("nan")
        se = sd / len(d) ** 0.5 if len(d) > 1 else float("nan")
        print(f"{arm:12s} {len(d):6d} {statistics.mean(d):+11.3f} {sd:7.3f} {se:7.3f} "
              f"{sum(1 for x in d if x > 1e-9):5d} {sum(1 for x in d if abs(x) <= 1e-9):5d} "
              f"{sum(1 for x in d if x < -1e-9):5d}")
    print("  95% CI = mean_delta +/- 1.96*se; a non-inferiority call needs the CI's lower "
          "bound above your tolerated regression, not just a p-value.")


if __name__ == "__main__":
    main()
