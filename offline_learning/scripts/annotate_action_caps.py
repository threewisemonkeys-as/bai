#!/usr/bin/env python3
"""Measure each planning problem's ANY-STEP reference reach and stamp it on the set.

The evaluators can then derive a per-problem or per-game action budget
(`eval_curated_plan.action_cap`) with no engine work at eval time.

The reach measured here is deliberately not `{pres}_reference_reached_at`: that field is
recorded under the row's own success mode, so a `final`-mode row stores its plan length
even when the goal first holds much earlier -- dino's "survive two cactus passes" stores
30 and actually first holds at 10. Online scoring is any-step by construction, so the
any-step reach is what a budget should be scaled against.

Writes `{pres}_anystep_reached_at` in place (backup first, atomic replace) and prints the
caps the rule implies under each mode.

    uv run python offline_learning/scripts/annotate_action_caps.py \
        --problems logs/2026-08-29/planning_v2/problems.json \
        --goal-presentation nl
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OFF = HERE.parent
REPO = OFF.parent
for _p in (str(OFF), str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval_curated_plan import (  # noqa: E402
    CAP_MODES, action_cap, configure_evaluation_goal, execute_and_score,
    load_eval_problems,
)


def measure(row: dict, presentation: str) -> tuple[bool, int | None, str | None]:
    """(oracle passed, any-step reach, error) for one row under one presentation."""
    try:
        p = configure_evaluation_goal(dict(row), presentation, "any")
        ok, at = execute_and_score(p, p["_eval_oracle_plan"])
    except Exception as exc:                      # report every bad row, don't abort
        return False, None, f"{type(exc).__name__}: {exc}"
    if not ok:
        return False, None, "reference plan does not satisfy its goal under any-step"
    return True, at, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems",
                    default=str(REPO / "logs/2026-08-29/planning_v2/problems.json"))
    ap.add_argument("--goal-presentation", choices=("frame", "nl", "both"), default="both")
    ap.add_argument("--games", default="")
    ap.add_argument("--dry-run", action="store_true", help="print, write nothing")
    a = ap.parse_args()

    path = Path(a.problems)
    meta, problems = load_eval_problems(path)
    if a.games:
        want = {g.strip() for g in a.games.split(",") if g.strip()}
        problems = [p for p in problems if p["game"] in want]
    presentations = (("frame", "nl") if a.goal_presentation == "both"
                     else (a.goal_presentation,))

    reach: dict[str, dict[str, int]] = defaultdict(dict)
    failures = []
    for presentation in presentations:
        for row in problems:
            if presentation == "nl" and not row.get("nl_checker"):
                continue
            ok, at, err = measure(row, presentation)
            if not ok:
                failures.append(f"{row['task_uid']} [{presentation}]: {err}")
                continue
            reach[row["task_uid"]][presentation] = at
            print(f"{row['task_uid']:44s} {presentation:5s} reach={at:3d} "
                  f"cap={action_cap(at):3d}", flush=True)

    if failures:
        print(f"\n{len(failures)} row(s) could not be measured:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)

    for presentation in presentations:
        rows = [p for p in problems if presentation in reach.get(p["task_uid"], {})]
        if not rows:
            continue
        caps = {p["task_uid"]: action_cap(reach[p["task_uid"]][presentation]) for p in rows}
        by_game: dict[str, int] = defaultdict(int)
        for p in rows:
            by_game[p["game"]] = max(by_game[p["game"]], caps[p["task_uid"]])
        flat = 50 * len(rows)
        print(f"\n=== {presentation}: {len(rows)} rows")
        for g in sorted(by_game):
            n = sum(1 for p in rows if p["game"] == g)
            print(f"  {g:13s} n={n:2d}  per-game cap {by_game[g]:3d}")
        for mode, total in (("per-game", sum(by_game[p["game"]] for p in rows)),
                            ("per-problem", sum(caps.values()))):
            print(f"  worst-case actions, {mode:11s} {total:5d} "
                  f"({100 * total / flat - 100:+.0f}% vs flat 50 = {flat})")

    if a.dry_run:
        print("\ndry run: nothing written")
        return

    payload = json.loads(path.read_text())
    written = 0
    for row in payload["problems"]:
        for presentation, at in reach.get(row["task_uid"], {}).items():
            row[f"{presentation}_anystep_reached_at"] = at
            written += 1
    payload.setdefault("meta", {})["anystep_reach_annotated"] = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "presentations": list(presentations),
        "rows": len(reach),
    }
    backup = path.with_suffix(f".pre-anystep-reach-backup.json")
    if not backup.exists():
        shutil.copy(path, backup)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, path)          # atomic: a concurrent eval never sees a torn file
    print(f"\nwrote {written} field(s) across {len(reach)} rows -> {path}"
          f"\nbackup {backup}")


if __name__ == "__main__":
    main()
