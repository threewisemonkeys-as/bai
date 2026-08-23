#!/usr/bin/env python3
"""Re-evaluate the exported ada85 best candidate on TEST50 with full drive context.

The TEST50 pool consists of curated slices from five deterministic source drives.  This
runner joins every target back to its source by (episode provenance, source Step), preserves
the completed run's exact held-out order and choice sets, and evaluates ID plus learned FD
without running GEPA again.
"""
from __future__ import annotations

import os as _bos, sys as _bsys  # offline_learning/ on sys.path (flat-import the kept libs)
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
import argparse
import asyncio
import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from forward_objective import textdiff_delta_f1
from invdyn_core import (
    build_window,
    exact_match_f1,
    predict_action_from_window,
    predict_next_state_from_window,
)
from validate import Transition, load_transitions, make_config
from validate_beliefs import balanced_split


EPISODE_SOURCE = {
    0: "srcA", 1: "srcA", 2: "srcA",
    3: "srcB", 4: "srcB", 5: "srcB",
    6: "srcC", 7: "srcC",
    8: "srcD", 9: "srcD",
    10: "srcE", 11: "srcE", 12: "srcE", 13: "srcE",
}


def read_rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def target_key(tr: Transition) -> tuple[str, str, str]:
    return tr.x_t, tr.action, tr.x_t1


def load_full_context_targets(test_root: Path, source_root: Path, context_k: int):
    """Load targets in validate.load_transitions order and attach source-drive context."""
    whitelist = {"noop", "click"}
    source_cache = {}
    source_rows = {}
    for source in sorted(set(EPISODE_SOURCE.values())):
        source_dir = source_root / source
        source_cache[source] = load_transitions([source_dir], whitelist, context_k=context_k)
        source_rows[source] = read_rows(source_dir / "episode_0" / "trajectory.csv")

    targets = []
    provenance = {}
    for csv_path in test_root.glob("episode_*/trajectory.csv"):
        episode = int(csv_path.parent.name.split("_")[1])
        source = EPISODE_SOURCE[episode]
        rows = read_rows(csv_path)
        src_rows = source_rows[source]
        src_transitions = source_cache[source]
        for i in range(len(rows) - 1):
            row, nxt = rows[i], rows[i + 1]
            action = (row.get("Action") or "").strip()
            if not action or action.split()[0] not in whitelist:
                continue
            step = int(row["Step"])
            if step >= len(src_transitions):
                raise AssertionError(f"{source} has no transition at step {step}")
            source_tr = src_transitions[step]
            target = Transition(row["Observation"], nxt["Observation"], action)
            if target_key(target) != target_key(source_tr):
                raise AssertionError(
                    f"episode {episode} step {step} does not match {source} byte-for-byte"
                )
            target.ctx_prev = list(source_tr.ctx_prev)
            target.ctx_next = list(source_tr.ctx_next)
            target.source_episode = episode
            target.source_step = step
            target.source_drive = source
            targets.append(target)
            provenance[id(target)] = {
                "curated_episode": episode,
                "source_drive": source,
                "source_step": step,
            }

    original = load_transitions([test_root], whitelist, context_k=context_k)
    if [target_key(t) for t in targets] != [target_key(t) for t in original]:
        raise AssertionError("full-context target order differs from validate.load_transitions")
    return targets, provenance


def reproduce_selection(targets, seed: int, train_count: int, test_n: int):
    rng = random.Random(seed)
    train_rng_consumption = list(range(train_count))
    rng.shuffle(train_rng_consumption)
    pool = list(targets)
    rng.shuffle(pool)
    _, selected = balanced_split(pool, test_n, 10**9, rng)
    return selected


async def retry_call(fn, attempts: int = 3):
    errors = []
    for attempt in range(1, attempts + 1):
        try:
            return await fn(), errors
        except Exception as exc:  # provider errors are transient; preserve every attempt
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            if attempt == attempts:
                raise
            await asyncio.sleep(min(2 ** attempt, 8))


async def evaluate(args, selected, baked, provenance, code: str, beliefs: str):
    cfg = make_config(args.task_model, args.client)
    sem = asyncio.Semaphore(args.concurrency)

    async def inverse_one(idx, inst):
        tr, choices = inst["tr"], inst["choices"]
        win, perception_error = build_window(code, tr)
        result, retry_errors = await retry_call(
            lambda: predict_action_from_window(cfg, win, beliefs, choices, sem)
        )
        pred, response, cost, prompt = result
        return {
            "idx": idx,
            **provenance[id(tr)],
            "truth": tr.action,
            "pred": pred,
            "correct": pred == tr.action,
            "choices": list(choices),
            "ctx_prev": len(tr.ctx_prev),
            "ctx_next": len(tr.ctx_next),
            "window_states": len(tr.ctx_prev) + 2 + len(tr.ctx_next),
            "z_t": win["z_t"],
            "z_t1": win["z_t1"],
            "prompt": prompt,
            "response": response,
            "perception_error": perception_error,
            "retry_errors": retry_errors,
            "cost": cost,
        }

    async def forward_one(idx, tr):
        win, perception_error = build_window(code, tr)
        result, retry_errors = await retry_call(
            lambda: predict_next_state_from_window(cfg, win, tr.action, beliefs, sem)
        )
        pred, cost, response, prompt = result
        start, target = win["z_t"], win["z_t1"]
        return {
            "idx": idx,
            **provenance[id(tr)],
            "action": tr.action,
            "ctx_prev": len(tr.ctx_prev),
            "ctx_next": len(tr.ctx_next),
            "window_states": len(tr.ctx_prev) + 2 + len(tr.ctx_next),
            "start": start,
            "target": target,
            "pred": pred,
            "changed": start.strip() != target.strip(),
            "exact": exact_match_f1(pred, target),
            "partial": textdiff_delta_f1(start, pred, target),
            "stale_exact": exact_match_f1(start, target),
            "stale_partial": textdiff_delta_f1(start, start, target),
            "prompt": prompt,
            "response": response,
            "perception_error": perception_error,
            "retry_errors": retry_errors,
            "cost": cost,
        }

    inv_tasks = [inverse_one(i, inst) for i, inst in enumerate(baked)]
    fd_tasks = [forward_one(i, tr) for i, tr in enumerate(selected)]
    all_rows = await asyncio.gather(*(inv_tasks + fd_tasks))
    return all_rows[: len(selected)], all_rows[len(selected):]


def mean(rows, key):
    return sum(float(r[key]) for r in rows) / len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", type=Path, required=True)
    ap.add_argument("--test-root", type=Path, required=True)
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--train-count", type=int, default=30)
    ap.add_argument("--test-n", type=int, default=50)
    ap.add_argument("--context-k", type=int, default=9)
    ap.add_argument("--task-model", default="deepseek/deepseek-v4-flash")
    ap.add_argument("--client", default="openrouter")
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    targets, provenance = load_full_context_targets(
        args.test_root, args.source_root, args.context_k
    )
    selected = reproduce_selection(targets, args.seed, args.train_count, args.test_n)

    original_path = args.artifact_dir / f"test_trace_gepa_seed{args.seed}.json"
    raw_path = args.artifact_dir / f"test_trace_raw_seed{args.seed}.json"
    original = json.loads(original_path.read_text())
    raw = json.loads(raw_path.read_text())
    original_records = original["records"]
    raw_records = raw["records"]
    if len(selected) != len(original_records) or len(selected) != len(raw_records):
        raise AssertionError("selected test size differs from saved traces")
    for i, (tr, inv, raw_rec) in enumerate(zip(selected, original_records, raw_records)):
        if tr.action != inv["truth"] or tr.action != raw_rec["truth"]:
            raise AssertionError(f"action mismatch at held-out index {i}")
        if tr.x_t[:6000] != raw_rec["z_t"] or tr.x_t1[:6000] != raw_rec["z_t1"]:
            raise AssertionError(f"raw target mismatch at held-out index {i}")

    baked = [
        {"tr": tr, "choices": list(rec["choices"])}
        for tr, rec in zip(selected, original_records)
    ]
    code = (args.artifact_dir / f"best_perception_gepa_seed{args.seed}.py").read_text()
    beliefs = (args.artifact_dir / f"best_beliefs_gepa_seed{args.seed}.txt").read_text()

    started = time.time()
    inverse, forward = asyncio.run(evaluate(args, selected, baked, provenance, code, beliefs))
    elapsed = time.time() - started

    inv_acc = mean(inverse, "correct")
    fd_exact = mean(forward, "exact")
    fd_partial = mean(forward, "partial")
    changed = [r for r in forward if r["changed"]]
    summary = {
        "game": "ada85",
        "seed": args.seed,
        "candidate": "exported GEPA best",
        "task_model": args.task_model,
        "context_k": args.context_k,
        "test_n": len(selected),
        "target_identity_verified_against_original_trace": True,
        "source_rows_verified_by_reconstruction": 64,
        "window_state_counts": dict(sorted(Counter(r["window_states"] for r in inverse).items())),
        "avg_ctx_prev": mean(inverse, "ctx_prev"),
        "avg_ctx_next": mean(inverse, "ctx_next"),
        "inverse_accuracy": inv_acc,
        "fd_exact": fd_exact,
        "fd_partial": fd_partial,
        "fd_changed_n": len(changed),
        "fd_dynamic_exact": mean(changed, "exact") if changed else None,
        "fd_dynamic_partial": mean(changed, "partial") if changed else None,
        "composite_id_fd_exact": 0.5 * (inv_acc + fd_exact),
        "inverse_cost": sum(r["cost"] for r in inverse),
        "forward_cost": sum(r["cost"] for r in forward),
        "total_cost": sum(r["cost"] for r in inverse + forward),
        "elapsed_seconds": elapsed,
        "original_truncated_context_inverse_accuracy": original["acc"],
    }

    inv_path = args.artifact_dir / f"test_trace_gepa_seed{args.seed}_fulltraj.json"
    fd_path = args.artifact_dir / f"test_trace_fd_seed{args.seed}_fulltraj.json"
    summary_path = args.artifact_dir / f"test50_fulltraj_summary_seed{args.seed}.json"
    inv_path.write_text(json.dumps({"summary": summary, "records": inverse}, indent=2))
    fd_path.write_text(json.dumps({"summary": summary, "records": forward}, indent=2))
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {inv_path}")
    print(f"wrote {fd_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
