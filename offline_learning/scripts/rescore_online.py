#!/usr/bin/env python3
"""Re-score a finished ONLINE planning run against the CURRENT Python goal checkers.

Checkers get repaired after a run has already been scored (the f5w3n `shoot-enemy`
occlusion exploit, retired 2026-09-01, is the motivating case). The stored rollouts
are replayable offline -- every round carries `grid_after` -- so a row can be re-judged
without spending a single token, PROVIDED the stored trace is complete.

Three verdicts, and the third is the one that matters:

  rescored   the checker still exists and the trace is complete (the rollout ran to
             budget/termination), so the new pass/fail is sound in both directions.
  retired    the row's `nl_checker` is no longer in the registry. It was withdrawn,
             not repaired -- there is nothing to re-score against, so the row is
             DROPPED rather than silently zeroed.
  truncated  the old checker fired, so the rollout STOPPED there. If the fixed checker
             disagrees we only know the goal was not met by action k -- the run would
             have continued to the cap had it been scored correctly. Re-scoring such a
             row to a fail would invent a result. It is reported and left for a re-run.

Exact-frame rows carry no Python checker and are passed through untouched.

    uv run python offline_learning/scripts/rescore_online.py \
        --run logs/2026-08-30/planning_v2_online_ds \
        --problems logs/2026-08-29/planning_v2/problems.pre-exactlyN-backup.json,\
logs/2026-08-29/planning_v2/problems.json \
        --games f5w3n --write

Without --write it is a dry run and prints the diff only.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from offline_learning.planning_nl_goals import (  # noqa: E402
    GOALS_BY_ID, freeze_grid,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_curated_online import report  # noqa: E402


# --------------------------------------------------------------------- replay
def replay(attempt: dict, start_grid, goal) -> tuple[bool, int | None, int, bool]:
    """Re-judge one stored attempt. Returns (success, reached_at, n_actions, complete).

    Mirrors `llm_rollout_v2`: grids start with the problem's start frame, each executed
    round appends its `grid_after`, and the any-step test runs after every action with
    the trajectory so far. `complete` is False when the stored trace ends on a success
    (the old checker cut the rollout short) -- see the module docstring.
    """
    grids = [freeze_grid(start_grid)]
    executed: list[str] = []
    for rnd in attempt.get("rounds", []):
        if rnd.get("executed") is None or rnd.get("grid_after") is None:
            break  # invalid-plan round: nothing was executed
        grids.append(freeze_grid(json.loads(rnd["grid_after"])))
        executed.append(rnd["executed"])
        if goal.check(grids, list(executed)):
            return True, len(executed), len(executed), True
    truncated = bool(attempt.get("success"))
    return False, None, len(executed), not truncated


# ------------------------------------------------------------------ problems
def load_problem_index(paths: list[Path]) -> dict:
    """task_uid -> problem, first file wins (pass the run's own problem set first)."""
    idx: dict[str, dict] = {}
    for path in paths:
        for p in json.loads(path.read_text())["problems"]:
            idx.setdefault(p["task_uid"], p)
    return idx


def presentation_of(row: dict) -> str:
    """Runs written before the goal_presentation rename stored goal_mode instead."""
    return row.get("goal_presentation") or row.get("eval_goal_mode") or row.get("goal_mode")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path,
                    help="run dir holding <game>/online.json")
    ap.add_argument("--problems", required=True,
                    help="comma-separated problems.json files; earlier files win")
    ap.add_argument("--games", default="", help="comma-separated subset (default: all)")
    ap.add_argument("--write", action="store_true",
                    help="rewrite online.json/.md in place (a backup is taken first)")
    a = ap.parse_args()

    probs = load_problem_index([Path(x) for x in a.problems.split(",") if x.strip()])
    want = {g.strip() for g in a.games.split(",") if g.strip()}
    changed_any = False

    for jf in sorted(a.run.glob("*/online.json")):
        game = jf.parent.name
        if want and game not in want:
            continue
        doc = json.loads(jf.read_text())
        rows, kept, notes = doc["rows"], [], []
        for row in rows:
            uid = row["task_uid"]
            if presentation_of(row) == "frame":
                kept.append(row)
                notes.append((uid, "exact-frame", "no Python checker", []))
                continue
            problem = probs.get(uid)
            if problem is None:
                kept.append(row)
                notes.append((uid, "MISSING", "not in any --problems file", []))
                continue
            checker_id = problem.get("nl_checker")
            goal = GOALS_BY_ID.get(checker_id)
            if goal is None:
                notes.append((uid, "RETIRED", f"{checker_id} no longer registered", []))
                continue  # dropped
            arm_notes, drop_row = [], False
            for arm in ("raw", "lmwm", "wc"):
                cell = row.get(arm, {})
                if cell.get("status") != "evaluated":
                    continue
                for att in cell["attempts"]:
                    old = bool(att["success"])
                    new, at, used, complete = replay(att, problem["start"], goal)
                    if new == old:
                        continue
                    if not complete:
                        arm_notes.append(
                            f"{arm}: pass@{att['reached_at']} -> TRUNCATED "
                            f"(only {used} actions stored; needs a re-run)")
                        drop_row = True
                        continue
                    arm_notes.append(f"{arm}: {int(old)} -> {int(new)}"
                                     + (f" @{at}" if new else ""))
                    att["success"], att["reached_at"] = new, at
                    if not new:
                        att["failed_reason"] = "rescored-fail"
                    for rnd, k in zip(att["rounds"], range(1, len(att["rounds"]) + 1)):
                        rnd["reached_goal"] = bool(new and at is not None and k >= at)
                cell["pass_rate"] = sum(t["success"] for t in cell["attempts"]) / len(cell["attempts"])
                cell["pass_any"] = any(t["success"] for t in cell["attempts"])
            if drop_row:
                notes.append((uid, "TRUNCATED", "old checker cut the rollout short", arm_notes))
                continue  # dropped: cannot be judged from the stored trace
            kept.append(row)
            notes.append((uid, "rescored" if arm_notes else "unchanged", checker_id, arm_notes))

        print(f"\n=== {game}  ({len(rows)} rows -> {len(kept)} kept)")
        for uid, verdict, why, arm_notes in notes:
            flag = "  " if verdict in {"unchanged", "exact-frame"} else "* "
            print(f"{flag}{uid:34s} {verdict:11s} {why}")
            for an in arm_notes:
                print(f"    {an}")
        if len(kept) == len(rows) and not any(n[3] for n in notes):
            print("  no change")
            continue
        changed_any = True
        if not a.write:
            print("  (dry run -- pass --write to apply)")
            continue

        bak = jf.with_name(f"online.pre-rescore-{date.today():%Y%m%d}.json")
        if not bak.exists():
            bak.write_text(jf.read_text())
        for row in kept:  # runs predating the rename stored it as goal_mode
            row.setdefault("goal_presentation", presentation_of(row))
        doc["rows"] = kept
        doc["cost"] = sum(t["cost"] for r in kept for arm in ("raw", "lmwm", "wc")
                          for t in r.get(arm, {}).get("attempts", []))
        cfg = doc["config"]
        cfg["rescored"] = {
            "date": f"{date.today():%Y-%m-%d}",
            "checker_version_source": "offline_learning/planning_nl_goals.py",
            "dropped": [uid for uid, v, _, _ in notes if v in {"RETIRED", "TRUNCATED"}],
        }
        jf.write_text(json.dumps(doc, indent=1))
        args_shim = SimpleNamespace(
            goal_presentation=cfg.get("goal_presentation", "declared"),
            cap_mode=cfg.get("cap_mode"), attempts=cfg.get("attempts_planned", 1),
            wc_budget=cfg.get("wc_budget"), warm_start=cfg.get("warm_start"),
        )
        llm_shim = SimpleNamespace(label=cfg.get("model"), model=cfg.get("model"),
                                   backend=cfg.get("backend"))
        md = report(kept, llm_shim, {}, 0.0, doc["cost"], args_shim,
                    cfg.get("rollouts_done", 0), cfg.get("rollouts_total", 0))
        jf.with_suffix(".md").write_text(
            md + f"\n\n> Re-scored {date.today():%Y-%m-%d} against the current checkers; "
                 f"backup at `{bak.name}`. Dropped: "
            + (", ".join(cfg["rescored"]["dropped"]) or "none") + "\n")
        print(f"  wrote {jf} + {jf.with_suffix('.md')} (backup {bak.name})")

    if not changed_any:
        print("\nnothing to change")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
