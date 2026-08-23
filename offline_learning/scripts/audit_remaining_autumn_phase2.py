"""Audit phase-2 augmented training and reconstructed full-context test inputs."""

from __future__ import annotations

import os as _bos, sys as _bsys  # offline_learning/ on sys.path (flat-import the kept libs)
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
import json
from pathlib import Path

from validate import backfill_context_from_source, load_transitions


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "offline_learning"
INPUT = ROOT / "logs/fulltraj_context_remaining_autumn_phase2_inputs"
CONFIG = {
    "83wkq": ({"noop", "click"}, HERE / "clean_data3/83wkq/train", HERE / "clean_data2/83wkq/train"),
    "27vwc": ({"noop", "click"}, HERE / "clean_data3/27vwc/train", HERE / "clean_data2/27vwc/train"),
    "qqm74": ({"left", "right", "up", "down", "noop", "click"}, HERE / "clean_data3/qqm74/train", HERE / "clean_data2/qqm74/train"),
    "eahcw": ({"left", "right", "up", "down", "noop", "click"}, INPUT / "eahcw/train20", HERE / "clean_data2/eahcw/train"),
    "s2kt7": ({"noop", "click"}, HERE / "clean_data3/s2kt7/train", HERE / "clean_data3/s2kt7/train_regen"),
    "ice": ({"left", "right", "up", "down", "noop", "click"}, HERE / "clean_data3/ice/train", HERE / "clean_data2/ice/train"),
}


def key(transition):
    return transition.x_t, transition.action, transition.x_t1


def means(transitions):
    count = max(1, len(transitions))
    return (
        round(sum(len(t.ctx_prev) for t in transitions) / count, 2),
        round(sum(len(t.ctx_next) for t in transitions) / count, 2),
    )


def main() -> None:
    results = []
    for game, (whitelist, train_target, train_source) in CONFIG.items():
        current = load_transitions([train_target], whitelist, context_k=9)
        older = load_transitions([INPUT / game / "old_test10"], whitelist, context_k=9)
        backfill_context_from_source(current, [train_source], whitelist, context_k=9)
        backfill_context_from_source(
            older, [HERE / f"clean_data2/{game}/test"], whitelist, context_k=9
        )

        manifest = json.loads((INPUT / game / "test50_context_manifest.json").read_text())
        full_test = []
        for entry in manifest:
            target = load_transitions([ROOT / entry["target_root"]], whitelist, context_k=9)
            backfill_context_from_source(
                target, [ROOT / entry["source_root"]], whitelist, context_k=9
            )
            full_test.extend(target)
        original = load_transitions([HERE / f"clean_data3/{game}/test50"], whitelist, context_k=9)
        if [key(t) for t in full_test] != [key(t) for t in original]:
            raise RuntimeError(f"{game}: test regrouping changed target identity/order")
        if len(full_test) != 50:
            raise RuntimeError(f"{game}: expected 50 test targets, found {len(full_test)}")
        combined = current + older
        windows = [len(t.ctx_prev) + 2 + len(t.ctx_next) for t in full_test]
        results.append(
            {
                "game": game,
                "train_current": len(current),
                "train_older": len(older),
                "train_total": len(combined),
                "train_context_prev_next": means(combined),
                "test_targets": len(full_test),
                "test_context_prev_next": means(full_test),
                "test_window_min_max": [min(windows), max(windows)],
            }
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
