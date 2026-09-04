#!/usr/bin/env python3
"""Build the representation-neutral 15-game planning set.

The 28 accepted rows for ice, disease, mario, and ants retain their starts, goals, seeds,
and horizons. Redundant legacy witness actions are replaced length-preservingly with noops
and recorded in migration metadata. New routes are Python-goal-compressed by the raw
interpreter and then materialized in the prefix-aware v2 schema. Run
validate_planning_v2.py afterward; it uses the independent wrapper driver.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parents[1]), str(HERE.parents[0]), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from offline_learning.planning_nl_goals import (  # noqa: E402
    CHECKER_VERSION, get_python_goal, legacy_checker_id,
)
from offline_learning.planning_v2 import (
    SCHEMA_VERSION, SELECTED_GAMES, materialize, quiescent_after, raw_trace, success,
)
from offline_learning.planning_v2_specs import all_specs

REPO = Path(__file__).resolve().parents[2]
DEFAULT_LEGACY = REPO / "logs/2026-08-18/curated/problems.json"
DEFAULT_OUT = REPO / "logs/2026-08-29/planning_v2/problems.json"
LEGACY_GAMES = {"bt3gb", "dq8gc", "n2ntd", "s2kt7"}


def normalize_legacy_plan(row: dict) -> tuple[list[str], list[dict]]:
    """Replace individually unnecessary moves with timing noops, restarting to a fixpoint."""
    plan = list(row["plan"])
    repairs = []
    probe = {"goal": row["goal"], "frame_success_mode": "any"}
    while True:
        for i, action in enumerate(plan):
            if action == "noop":
                continue
            candidate = plan[:i] + ["noop"] + plan[i + 1:]
            trace = raw_trace(row["program"], row["seed"], candidate)
            if success(probe, "frame", trace[0], trace[1:])[0]:
                repairs.append({
                    "index": i,
                    "from": action,
                    "to": "noop",
                    "reason": "length-preserving exact-goal substitution",
                })
                plan = candidate
                break
        else:
            return plan, repairs


def migrate_legacy(row: dict) -> dict:
    """Add v2 fields and normalize only provably redundant legacy witness actions."""
    out = dict(row)
    checker_id = legacy_checker_id(row["game"], row["id"])
    python_goal = get_python_goal(checker_id)
    if python_goal.seed is not None and python_goal.seed != row["seed"]:
        raise ValueError(
            f"{row['game']}:{row['id']}: checker seed {python_goal.seed} "
            f"!= problem seed {row['seed']}"
        )
    plan, repairs = normalize_legacy_plan(row)
    trace = raw_trace(row["program"], row["seed"], plan)
    reached, reached_at = success(
        {"goal": row["goal"], "frame_success_mode": "any"},
        "frame", trace[0], trace[1:],
    )
    if not reached or len(plan) != row["h"]:
        raise ValueError(f"legacy normalization broke {row['game']}:{row['id']}")
    uid = f"{row['game']}:{row['id']}:s{row['seed']}"
    frame_quiescent = quiescent_after(
        row["program"], row["seed"], [], plan, "raw",
    )
    out.pop("quiescent", None)
    for old, new in (
        ("random_success", "frame_random_success"),
        ("random_trials", "frame_random_trials"),
        ("random_success_cap50", "frame_random_success_cap50"),
        ("random_trials_cap50", "frame_random_trials_cap50"),
    ):
        if old in out:
            out[new] = out.pop(old)
    for field in ("nl_random_success", "nl_random_trials",
                  "nl_random_success_cap50", "nl_random_trials_cap50"):
        out.setdefault(field, None)
    out.update({
        "schema_version": SCHEMA_VERSION,
        "task_uid": uid,
        "template_id": f"{row['game']}:{row['id']}",
        "prefix": [],
        "frame_success_mode": "any",


        "nl_goal": python_goal.nl,
        "nl_checker": checker_id,
        "nl_checker_version": CHECKER_VERSION,
        "nl_success_mode": python_goal.success_mode,
        "nl_require_quiescent": python_goal.require_quiescent,
        "nl_reference_plan": list(python_goal.reference_plan or plan),
        "source": "curated-v1-accepted-migration",
        "stochastic": False,
        "plan": plan,
        "n_decisions": sum(action != "noop" for action in plan),
        "frame_reference_reached_at": reached_at,
        "frame_reference_quiescent": frame_quiescent,
        "nl_reference_quiescent": (
            frame_quiescent if list(python_goal.reference_plan or plan) == plan else None
        ),
        "frame_noop_success": 0.0,
        "nl_noop_success": None,
        "migration_status": "normalized-reference-plan" if repairs else "retained",
    })
    if repairs:
        out["legacy_original_plan"] = list(row["plan"])
        out["migration_repairs"] = repairs
    return out


def build(legacy_path: Path, random_trials: int, compress: bool) -> list[dict]:
    old = json.loads(legacy_path.read_text())
    legacy = [migrate_legacy(row) for row in old if row["game"] in LEGACY_GAMES]
    if len(legacy) != 28:
        raise ValueError(f"expected 28 accepted problems across the original four, got {len(legacy)}")
    repaired = [row for row in legacy if row["migration_status"] == "normalized-reference-plan"]
    if len(repaired) != 5:
        raise ValueError(f"expected five legacy witness-plan repairs, got {len(repaired)}")

    new = []
    specs = all_specs()
    for i, spec in enumerate(specs, 1):
        spec = dict(spec)
        spec["compress"] = compress
        row = materialize(spec, random_trials=random_trials)
        new.append(row)
        print(f"  [{i:02d}/{len(specs)}] {row['task_uid']:<42} h={row['h']:<3} "
              f"dec={row['n_decisions']:<2}", flush=True)
    return legacy + new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--random-trials", type=int, default=24)
    parser.add_argument("--no-compress", action="store_true",
                        help="debug only: materialize authored plans without Python-goal compression")
    args = parser.parse_args()

    problems = build(args.legacy, args.random_trials, not args.no_compress)
    counts = Counter(p["game"] for p in problems)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selected_games": SELECTED_GAMES,
        "n": len(problems),
        "counts_by_game": {game: counts[game] for game in SELECTED_GAMES},
        "construction": {
            "legacy": (
                "28 accepted v1 rows retained without regenerating goals; five redundant "
                "witness plans normalized length-preservingly with recorded repairs"
            ),
            "new": "raw-interpreter replay + registered Python goal compression",
            "nl_checkers": CHECKER_VERSION,
            "validation": "run scripts/validate_planning_v2.py (wrapper driver)",
            "random_trials": args.random_trials,
        },
        "problems": problems,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"\nwrote {len(problems)} problems across {len(counts)} games -> {args.out}")
    print("  " + "  ".join(f"{g}:{counts[g]}" for g in SELECTED_GAMES))


if __name__ == "__main__":
    main()
