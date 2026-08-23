"""Evaluate forward prediction on raw frames and learned P+B representations.

This is the forward-prediction counterpart to the clean-sweep inverse-dynamics
test traces.  It reproduces each run's held-out cross-trajectory test split and
evaluates two regimes on the same transitions:

* ``raw``: F sees canonical raw-grid history and the true action, then predicts
  the complete canonical raw grid X[t+1].  It does not receive learned beliefs,
  matching the existing raw-frame inverse-dynamics reference.
* ``learned``: F sees the learned perception history and learned world beliefs,
  then predicts P(X[t+1]) using the existing GEPA forward predictor.

Both predictions receive the two deterministic metrics already used by GEPA:

* ``exact``: stripped character-for-character equality with the target.
* ``partial``: ``textdiff_delta_f1`` on the predicted versus true state change.

The raw target is a compact, canonical JSON serialization of the parsed grid.
Observation-wrapper metadata is intentionally excluded: the target is the raw
frame itself, not headers such as ``Step`` or ``Available actions``.

Default invocation evaluates every completed clean-data3 TEST50 pool, resolving
learned artifacts from the phase-1 or phase-2 sweep as appropriate:

    uv run python offline_learning/scripts/eval_forward_modes.py
"""

from __future__ import annotations

import os as _bos, sys as _bsys  # offline_learning/ on sys.path (flat-import the kept libs)
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
import argparse
import asyncio
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

from clean_sweep import GAMES
from forward_objective import textdiff_delta_f1
from invdyn_core import (
    DEFAULT_KNOWLEDGE,
    build_window,
    exact_match_f1,
    parse_grid,
    predict_next_state,
    predict_next_state_from_window,
)
from validate import _llm_call, _parse_tag, load_transitions, make_config, run_perceive
from validate_beliefs import balanced_split


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TEST50_GAMES = (
    "27vwc 83wkq ada85 aw9wd bt3gb dgg2c e3v6m eahcw ice nrdf6 "
    "ntq4y qqm74 s2kt7 va6fq"
).split()
PHASE1_GAMES = {
    "7www9", "7xf97", "bt3gb", "dgg2c", "dq8gc", "e3v6m", "f5w3n",
    "n2ntd", "nrdf6", "qfsvc", "vqjh6",
}
SPECIAL_SWEEP_ROOTS = {
    "aw9wd": REPO / "logs" / "clean_sweep_gepa_padded_ctxk9",
    "dgg2c": REPO / "logs" / "clean_sweep_gepa_cd3_warmstart",
    "nrdf6": REPO / "logs" / "clean_sweep_gepa_cd3_warmstart2",
}


RAW_FORWARD_TMPL = """You predict the complete NEXT raw grid of a grid environment.

=== DEFAULT KNOWLEDGE (always-true facts about this environment) ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

Below is a trajectory of consecutive RAW GRIDS in canonical JSON. The action
between each prior pair is shown. Use the whole history to infer any passive,
periodic, momentum, selection, or delayed dynamics.

{transcript}

The action now taken from the CURRENT raw grid is: {action}

Predict the complete resulting NEXT raw grid. Return every row and every cell,
including unchanged cells. Use exactly the same canonical JSON representation as
the grids above: one compact JSON array, double-quoted strings, no spaces or
markdown. Do not emit an explanation.

<next_state>complete canonical JSON next grid</next_state>"""


def canonical_grid(raw: str) -> str:
    """Observation wrapper -> compact canonical full-grid JSON."""
    grid = parse_grid(raw)
    if grid is None:
        raise ValueError("could not parse raw grid")
    return json.dumps(grid, ensure_ascii=False, separators=(",", ":"))


def collapse_actions(transitions) -> None:
    """Apply the same opt-in parameter collapse as gepa_optimize.py."""
    for tr in transitions:
        tr.action = tr.action.split()[0]
        tr.ctx_prev = [(raw, action.split()[0]) for raw, action in tr.ctx_prev]
        tr.ctx_next = [(action.split()[0], raw) for action, raw in tr.ctx_next]


def reproduce_test_split(
    game: str, data_root: Path, test_dir: str, seed: int, context_k: int, test_n: int
):
    """Reproduce gepa_optimize's cross-trajectory split, including RNG consumption."""
    action_csv, keep_params, _mmc, _legacy_rounds = GAMES[game]
    whitelist = set(filter(None, action_csv.split(","))) or None
    train = load_transitions([data_root / game / "train"], whitelist, context_k=context_k)
    test = load_transitions([data_root / game / test_dir], whitelist, context_k=context_k)
    if not keep_params:
        collapse_actions(train)
        collapse_actions(test)

    rng = random.Random(seed)
    train_pool = list(train)
    rng.shuffle(train_pool)  # gepa shuffles train before test; preserve RNG alignment
    test_pool = list(test)
    rng.shuffle(test_pool)
    _, selected = balanced_split(test_pool, test_n, 10**9, rng)
    return selected, len(train), len(test)


def verify_against_trace(run_dir: Path, seed: int, transitions) -> None:
    """Fail fast if split reproduction drifts from the saved inverse test trace."""
    path = run_dir / f"test_trace_gepa_seed{seed}.json"
    if not path.exists():
        return
    saved = json.loads(path.read_text())
    expected = [record["truth"] for record in saved.get("records", [])]
    actual = [tr.action for tr in transitions]
    if expected != actual:
        raise AssertionError(
            f"test split does not match {path}: expected actions {expected}, got {actual}"
        )


def raw_transcript(tr) -> str:
    """Canonical raw-grid history ending at X[t], aligned with ctx_prev actions."""
    frames = [(raw, action) for raw, action in tr.ctx_prev] + [(tr.x_t, None)]
    first = -(len(frames) - 1)
    parts = []
    for i, (raw, action) in enumerate(frames):
        label = first + i
        suffix = " (CURRENT)" if label == 0 else ""
        parts.append(f"STATE[t{label:+d}] RAW GRID{suffix}:\n{canonical_grid(raw)}")
        if action is not None:
            parts.append(f"  action(t{label:+d} -> t{label + 1:+d}): {action}")
    return "\n".join(parts)


async def predict_raw(cfg, tr, sem):
    prompt = RAW_FORWARD_TMPL.format(
        default_knowledge=DEFAULT_KNOWLEDGE,
        transcript=raw_transcript(tr),
        action=tr.action,
    )
    async with sem:
        text, cost = await _llm_call(cfg, prompt)
    text = text or ""
    pred = _parse_tag(text, "next_state") or text.strip()
    return pred, cost, text


async def evaluate_item(cfg, mode: str, tr, code: str, beliefs: str, sem, context_k: int):
    if mode == "raw":
        start, target = canonical_grid(tr.x_t), canonical_grid(tr.x_t1)
        pred, cost, response = await predict_raw(cfg, tr, sem)
        perception_error = None
    else:
        if context_k > 0:
            win, perception_error = build_window(code, tr)
            start, target = win["z_t"], win["z_t1"]
            pred, cost, response, _prompt = await predict_next_state_from_window(
                cfg, win, tr.action, beliefs, sem
            )
        else:
            start, err0 = run_perceive(code, tr.x_t)
            target, err1 = run_perceive(code, tr.x_t1)
            perception_error = err0 or err1
            pred, cost = await predict_next_state(cfg, start, tr.action, beliefs, sem)
            response = pred

    return {
        "mode": mode,
        "action": tr.action,
        "start": start,
        "target": target,
        "pred": pred,
        "response": response,
        "changed": start.strip() != target.strip(),
        "exact": exact_match_f1(pred, target),
        "partial": textdiff_delta_f1(start, pred, target),
        "stale_exact": exact_match_f1(start, target),
        "stale_partial": textdiff_delta_f1(start, start, target),
        "perception_error": perception_error,
        "cost": cost,
    }


def mean(rows, key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else math.nan


def summarize(rows) -> dict:
    dynamic = [row for row in rows if row["changed"]]
    return {
        "n": len(rows),
        "changed_n": len(dynamic),
        "unique_targets": len({row["target"] for row in rows}),
        "perception_errors": sum(bool(row["perception_error"]) for row in rows),
        "exact": mean(rows, "exact"),
        "partial": mean(rows, "partial"),
        "stale_exact": mean(rows, "stale_exact"),
        "stale_partial": mean(rows, "stale_partial"),
        "dynamic_exact": mean(dynamic, "exact"),
        "dynamic_partial": mean(dynamic, "partial"),
        "cost": sum(row["cost"] for row in rows),
        "action_counts": dict(Counter(row["action"] for row in rows)),
    }


async def evaluate_game(args, game: str, cfg, sem) -> dict:
    if args.sweep_root is not None:
        run_dir = args.sweep_root / f"{game}_seed{args.seed}"
    elif game in SPECIAL_SWEEP_ROOTS:
        run_dir = SPECIAL_SWEEP_ROOTS[game] / f"{game}_seed{args.seed}"
    else:
        phase_root = args.phase1_root if game in PHASE1_GAMES else args.phase2_root
        run_dir = phase_root / f"{game}_seed{args.seed}"
    code_path = run_dir / f"best_perception_gepa_seed{args.seed}.py"
    beliefs_path = run_dir / f"best_beliefs_gepa_seed{args.seed}.txt"
    if not code_path.exists():
        raise FileNotFoundError(code_path)

    transitions, train_total, test_total = reproduce_test_split(
        game, args.data_root, args.test_dir, args.seed, args.context_k, args.test_n
    )
    if args.test_dir == "test":
        verify_against_trace(run_dir, args.seed, transitions)
    code = code_path.read_text()
    beliefs = beliefs_path.read_text() if beliefs_path.exists() else ""

    tasks = [
        evaluate_item(cfg, mode, tr, code, beliefs, sem, args.context_k)
        for mode in ("raw", "learned")
        for tr in transitions
    ]
    rows = await asyncio.gather(*tasks)
    for idx, row in enumerate(rows):
        row["idx"] = idx % len(transitions)

    by_mode = {mode: [row for row in rows if row["mode"] == mode] for mode in ("raw", "learned")}
    return {
        "game": game,
        "seed": args.seed,
        "artifact_dir": str(run_dir),
        "train_total": train_total,
        "test_total": test_total,
        "test_n": len(transitions),
        "summary": {mode: summarize(mode_rows) for mode, mode_rows in by_mode.items()},
        "rows": rows,
    }


def aggregate(results: list[dict], mode: str) -> dict:
    rows = [row for result in results for row in result["rows"] if row["mode"] == mode]
    return summarize(rows)


def render_report(payload: dict) -> str:
    results = payload["results"]
    lines = [
        "# Forward prediction: raw frames vs learned perception + beliefs",
        "",
        "Held-out cross-trajectory test split; `exact` is stripped full-text equality; ",
        "`partial` is the existing `textdiff_delta_f1` change score. Raw predictions target ",
        "the complete canonical JSON grid. Learned predictions target `P(X[t+1])`.",
        "",
        "| game | mode | n | changed | unique targets | exact | partial | stale exact | stale partial | dynamic exact | dynamic partial | cost |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        for mode in ("raw", "learned"):
            s = result["summary"][mode]
            lines.append(
                f"| {result['game']} | {mode} | {s['n']} | {s['changed_n']} | "
                f"{s['unique_targets']} | {s['exact']:.3f} | {s['partial']:.3f} | "
                f"{s['stale_exact']:.3f} | {s['stale_partial']:.3f} | "
                f"{s['dynamic_exact']:.3f} | {s['dynamic_partial']:.3f} | ${s['cost']:.4f} |"
            )
    lines += ["", "## Micro-average over all test items", ""]
    for mode in ("raw", "learned"):
        s = payload["aggregate"][mode]
        lines.append(
            f"- {mode}: exact={s['exact']:.3f}, partial={s['partial']:.3f}, "
            f"dynamic exact={s['dynamic_exact']:.3f}, dynamic partial={s['dynamic_partial']:.3f}, "
            f"changed={s['changed_n']}/{s['n']}, cost=${s['cost']:.4f}"
        )
    lines += [
        "",
        "`stale` predicts no change by copying the current representation. A learned P with ",
        "zero changed targets can therefore score perfectly under both metrics; inspect `changed` ",
        "and the dynamic-only columns before interpreting a high learned score.",
    ]
    return "\n".join(lines)


async def main_async(args) -> dict:
    cfg = make_config(args.task_model, args.client)
    sem = asyncio.Semaphore(args.concurrency)
    started = time.time()
    tasks = {asyncio.create_task(evaluate_game(args, game, cfg, sem)): game for game in args.games}
    results = []
    for task in asyncio.as_completed(tasks):
        result = await task
        results.append(result)
        raw = result["summary"]["raw"]
        learned = result["summary"]["learned"]
        print(
            f"[{result['game']}] raw exact/partial={raw['exact']:.3f}/{raw['partial']:.3f} | "
            f"learned={learned['exact']:.3f}/{learned['partial']:.3f} | "
            f"P changed={learned['changed_n']}/{learned['n']}",
            flush=True,
        )
    order = {game: i for i, game in enumerate(args.games)}
    results.sort(key=lambda result: order[result["game"]])
    return {
        "config": {
            "games": args.games,
            "seed": args.seed,
            "test_n": args.test_n,
            "context_k": args.context_k,
            "task_model": args.task_model,
            "client": args.client,
            "data_root": str(args.data_root),
            "test_dir": args.test_dir,
            "sweep_root": str(args.sweep_root) if args.sweep_root is not None else None,
            "phase1_root": str(args.phase1_root),
            "phase2_root": str(args.phase2_root),
            "raw_uses_beliefs": False,
            "raw_target": "canonical full-grid JSON",
            "metrics": {"exact": "exact_match_f1", "partial": "textdiff_delta_f1"},
        },
        "elapsed_seconds": time.time() - started,
        "results": results,
        "aggregate": {mode: aggregate(results, mode) for mode in ("raw", "learned")},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", default=",".join(TEST50_GAMES))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--test-n", type=int, default=50)
    parser.add_argument("--test-dir", default="test50")
    parser.add_argument("--context-k", type=int, default=9)
    parser.add_argument("--task-model", default="deepseek/deepseek-v4-flash")
    parser.add_argument("--client", default="openrouter")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--data-root", type=Path, default=HERE / "clean_data3")
    parser.add_argument("--sweep-root", type=Path, default=None,
                        help="optional single sweep root override for every game")
    parser.add_argument("--phase1-root", type=Path,
                        default=REPO / "logs" / "clean_sweep_gepa_cd3_phase1")
    parser.add_argument("--phase2-root", type=Path,
                        default=REPO / "logs" / "clean_sweep_gepa_cd3_phase2")
    parser.add_argument(
        "--out", type=Path,
        default=REPO / "logs" / "forward_eval_test50_raw_vs_learned",
        help="output prefix; writes .json and .md",
    )
    args = parser.parse_args()
    args.games = [game for game in args.games.split(",") if game]

    payload = asyncio.run(main_async(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(payload, indent=2, default=str))
    report = render_report(payload)
    args.out.with_suffix(".md").write_text(report)
    print(report)
    print(f"\nwrote {args.out.with_suffix('.json')} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
