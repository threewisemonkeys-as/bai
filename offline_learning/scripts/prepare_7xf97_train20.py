"""Materialize the intended seed-1, action-balanced 20-item 7xf97 train subset.

The curated clean_data3/7xf97/train directory contains 24 transitions.  The
augmented full-trajectory experiments are specified as 20 current transitions
plus the older held-out 10, so select 20 deterministically and retain each
selected transition as a two-row target slice.  Full temporal context is
backfilled separately from train_regen by gepa_optimize.py.
"""

from __future__ import annotations

import os as _bos, sys as _bsys  # offline_learning/ on sys.path (flat-import the kept libs)
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
import csv
import json
import random
from collections import Counter
from pathlib import Path

from validate import load_transitions
from validate_beliefs import balanced_split


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "offline_learning/clean_data3/7xf97/train"
OUTPUT = ROOT / "logs/fulltraj_context_remaining_autumn_inputs/7xf97/train20"
MANIFEST = OUTPUT.parent / "train20_selection.json"
WHITELIST = {"left", "right", "up", "down", "noop", "click"}


def main() -> None:
    transitions = load_transitions([SOURCE], WHITELIST, context_k=0)
    if len(transitions) != 24:
        raise RuntimeError(f"expected 24 curated transitions, found {len(transitions)}")

    # balanced_split returns its selected, round-robin action-balanced items as
    # the second result. Collapse click coordinates exactly as the experiment
    # does before selection.
    for transition in transitions:
        transition.action = transition.action.split()[0]
    _, selected = balanced_split(transitions, 20, 10**9, random.Random(1))
    selected_keys = Counter((t.x_t, t.x_t1, t.action) for t in selected)

    matches: list[tuple[Path, int, dict, dict]] = []
    for csv_path in sorted(SOURCE.glob("episode_*/trajectory.csv")):
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for index in range(len(rows) - 1):
            row, next_row = rows[index], rows[index + 1]
            key = (
                row.get("Observation") or "",
                next_row.get("Observation") or "",
                (row.get("Action") or "").strip().split()[0],
            )
            if selected_keys[key] > 0:
                selected_keys[key] -= 1
                matches.append((csv_path, index, row, next_row))

    missing = sum(selected_keys.values())
    if missing or len(matches) != 20:
        raise RuntimeError(f"matched {len(matches)} selected transitions; missing={missing}")
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty {OUTPUT}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    for output_index, (csv_path, row_index, row, next_row) in enumerate(matches):
        episode_dir = OUTPUT / f"episode_{output_index}"
        episode_dir.mkdir()
        output_csv = episode_dir / "trajectory.csv"
        with output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows([row, next_row])
        records.append(
            {
                "target": str(output_csv.relative_to(ROOT)),
                "source": str(csv_path.relative_to(ROOT)),
                "source_row": row_index,
                "action": (row.get("Action") or "").strip().split()[0],
            }
        )

    manifest = {
        "seed": 1,
        "method": "balanced_split holdout selection after action collapse",
        "source_transition_count": len(transitions),
        "selected_transition_count": len(records),
        "action_counts": dict(sorted(Counter(r["action"] for r in records).items())),
        "records": records,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
