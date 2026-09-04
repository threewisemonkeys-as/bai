#!/usr/bin/env python3
"""Cap-matched random floors for the v2 planning set (audit F3/F5).

The stored per-presentation random floors were measured with plans of exactly h actions,
but the evaluator lets a model submit up to PLAN_CAP=50 actions and (by default) scores
any-step -- a 50-action submission gets up to 50 chances against a floor measured at h.
This script recomputes each row's floor with random plans of the FULL evaluator cap,
scored through the evaluator's own `configure_evaluation_goal` + `execute_and_score`
path (same success-mode override, same driver), so the floor and the model scores obey
byte-identical rules. Results land in-place under explicit `frame_` or `nl_` field names;
the original rand@h fields are left untouched and the input file is backed up first.

    uv run python offline_learning/scripts/recompute_random_floors.py \
        --problems logs/2026-08-29/planning_v2/problems.json \
        --goal-presentation frame --trials 200 --jobs 12
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OFF = HERE.parent
REPO = OFF.parent
for _p in (str(OFF), str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import random  # noqa: E402

from offline_learning.planning_v2 import random_plan, stable_seed  # noqa: E402

from eval_curated_plan import (  # noqa: E402
    CAP_MODES, PLAN_CAP, configure_evaluation_goal, execute_and_score,
    load_eval_problems, resolve_action_caps, select_goal_presentation,
)


def floor_one(row: dict, trials: int, cap: int, presentation: str,
              success_mode: str, driver: str) -> tuple:
    """(task_uid, hits, mean reached_at over hits) for one problem's cap-length floor."""
    p = configure_evaluation_goal(row, presentation, success_mode)
    assert p is not None
    rng = random.Random(stable_seed(f"{p['task_uid']}:{presentation}:cap{cap}"))
    hits, reached = 0, []
    for _ in range(trials):
        plan = random_plan(p["game"], cap, rng)
        ok, at = execute_and_score(p, plan, driver=driver)
        if ok:
            hits += 1
            reached.append(at)
    return p["task_uid"], hits, (sum(reached) / len(reached) if reached else None)


def _worker(args):
    return floor_one(*args)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default=str(REPO / "logs/2026-08-29/planning_v2/problems.json"))
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--goal-presentation", choices=("frame", "nl"), required=True)
    ap.add_argument("--cap", type=int, default=PLAN_CAP,
                    help="flat budget to measure at under --cap-mode fixed")
    ap.add_argument("--cap-mode", choices=CAP_MODES, default="fixed",
                    help="match the evaluator's budget rule: 'fixed' measures every row "
                    "at --cap; the scaled modes measure each row at the budget "
                    "eval_curated_plan.action_cap gives it")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--success-mode", choices=("any", "reference"), default="any")
    ap.add_argument("--driver", choices=("wrapper", "raw"), default="wrapper",
                    help="wrapper = byte-identical to the evaluator's scoring path")
    ap.add_argument("--dry-run", action="store_true", help="print floors, write nothing")
    a = ap.parse_args()

    path = Path(a.problems)
    meta, raw_rows = load_eval_problems(path)
    # load_eval_problems normalizes; keep the on-disk rows for the rewrite
    disk = json.loads(path.read_text())

    if a.cap_mode == "fixed":
        caps = {row["task_uid"]: a.cap for row in raw_rows}
    else:
        configured = select_goal_presentation(raw_rows, a.goal_presentation,
                                              a.success_mode)
        caps = resolve_action_caps(configured, a.cap_mode, a.cap)
        print(f"caps ({a.cap_mode}): {min(caps.values())}-{max(caps.values())} "
              f"over {len(caps)} rows", flush=True)

    t0 = time.time()
    jobs = [(row, a.trials, caps[row["task_uid"]], a.goal_presentation, a.success_mode,
             a.driver) for row in raw_rows]
    results = {}
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for i, (uid, hits, mean_at) in enumerate(ex.map(_worker, jobs), 1):
            results[uid] = (hits, mean_at)
            old = next(r.get(f"{a.goal_presentation}_random_success") for r in raw_rows
                       if r["task_uid"] == uid)
            # A row measured under the other representation has no rand@h floor to
            # compare against -- every problem now carries both goals, but the
            # original floors were only ever measured under the one it was asked in.
            was = f"{old:.3f}" if old is not None else "unmeasured"
            print(f"[{i}/{len(jobs)}] {uid}: cap{caps[uid]} floor {hits}/{a.trials} = "
                  f"{hits / a.trials:.3f} (rand@h {was})"
                  + (f", mean hit at step {mean_at:.1f}" if mean_at else ""),
                  flush=True)
    elapsed = time.time() - t0

    nonzero = sum(1 for hits, _ in results.values() if hits)
    print(f"\n{len(results)} problems in {elapsed / 60:.1f} min; "
          f"{nonzero} with a nonzero floor")
    if a.dry_run:
        return

    tag = "cap50" if a.cap_mode == "fixed" else a.cap_mode
    backup = path.with_name(path.stem + f".pre-{tag}-backup.json")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"backed up original to {backup}")
    for row in disk["problems"]:
        uid = row.get("task_uid") or f"{row['game']}:{row['id']}:s{row['seed']}"
        hits, _ = results[uid]
        # `_random_floors` is the general record, keyed by the budget the floor was
        # measured at, so floors for several cap modes coexist and `_floor` can refuse
        # one that does not match the budget in force. The cap50 fields stay the
        # flat-regime name they always were.
        floors_at = row.setdefault(f"{a.goal_presentation}_random_floors", {})
        floors_at[str(caps[uid])] = {
            "success": hits / a.trials, "trials": a.trials, "cap_mode": a.cap_mode,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        }
        if a.cap_mode == "fixed" and a.cap == PLAN_CAP:
            row[f"{a.goal_presentation}_random_success_cap50"] = hits / a.trials
            row[f"{a.goal_presentation}_random_trials_cap50"] = a.trials
    floors = disk.setdefault("construction", {}).setdefault(
        "random_floor_cap50_by_presentation", {}
    )
    floors[a.goal_presentation] = {
        "cap": a.cap, "cap_mode": a.cap_mode,
        "caps": {u: c for u, c in caps.items()} if a.cap_mode != "fixed" else None,
        "trials": a.trials, "goal_presentation": a.goal_presentation,
        "success_mode": a.success_mode,
        "driver": a.driver, "scored_via": "eval_curated_plan.execute_and_score",
        "rng": "stable_seed(task_uid + ':<presentation>:cap<cap>')",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
    }
    path.write_text(json.dumps(disk, indent=1))
    print(f"updated {path}")


if __name__ == "__main__":
    main()
