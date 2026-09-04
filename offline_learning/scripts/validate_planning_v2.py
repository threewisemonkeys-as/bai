#!/usr/bin/env python3
"""Independent, prefix-aware audit for representation-neutral planning problems."""
from __future__ import annotations

import argparse
from dataclasses import replace as dc_replace
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parents[1]), str(HERE.parents[0]), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from offline_learning.planning_nl_goals import (  # noqa: E402
    CHECKER_VERSION, checker_holds_at_start, score_python_goal, validate_problem_goal,
)
from offline_learning.planning_v2 import (
    SCHEMA_VERSION, SELECTED_GAMES, STOCHASTIC_GAMES, load_problem_file,
    quiescent_after, random_plan, raw_trace, rollout, stable_seed, success, wrapper_trace,
)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PROBLEMS = REPO / "logs/2026-08-29/planning_v2/problems.json"

CHECKS = ["A1-schema", "A2-drivers", "A3-reference", "A4-nontrivial",
          "A5-noop-fails", "A5b-noop-anystep-fails", "A7-task-delete",
          "A8-substitute", "A9-quiescence", "A10-random-floor"]


def run(problem: dict[str, Any], plan: list[str]) -> tuple[Any, list[Any]]:
    start, frames, _ = rollout(problem["program"], problem["seed"], problem["prefix"],
                               plan, "wrapper")
    return start, frames


def _stable_after_final(problem: dict[str, Any], plan: list[str], require: bool) \
        -> bool | None:
    if not require:
        return None
    return quiescent_after(
        problem["program"], problem["seed"], problem["prefix"], plan, "wrapper",
    )


def frame_success(problem: dict[str, Any], start, frames, plan: list[str]) \
        -> tuple[bool, int | None]:
    return success(problem, "frame", start, frames, plan)


def task_success(problem: dict[str, Any], start, frames, plan: list[str]) -> bool:
    """Score every task's NL objective with its registered Python program."""
    goal = validate_problem_goal(problem)
    stable = _stable_after_final(problem, plan, goal.require_quiescent)
    return score_python_goal(
        goal, start, frames, plan, stable_after_final=stable,
    )[0]


def start_satisfies(problem: dict[str, Any], start) -> bool:
    return checker_holds_at_start(validate_problem_goal(problem), start)


def audit_one(problem: dict[str, Any], verify_random: bool) -> dict[str, Any]:
    uid = problem.get("task_uid", "<missing>")
    prefix, plan = list(problem.get("prefix", [])), list(problem.get("plan", []))
    nl_plan = list(problem.get("nl_reference_plan", plan))
    checks: dict[str, bool] = {}
    detail: dict[str, Any] = {}
    warnings: list[str] = []

    checks["A1-schema"] = bool(
        problem.get("schema_version") == SCHEMA_VERSION and uid != "<missing>"
        and problem.get("game") in SELECTED_GAMES and problem.get("nl_goal")
        and problem.get("h") == len(plan)
        and problem.get("frame_success_mode") in {"any", "final"}
        and problem.get("nl_checker")
        and problem.get("nl_checker_version") == CHECKER_VERSION
        and problem.get("nl_success_mode") in {"any", "final"}
        and isinstance(problem.get("nl_require_quiescent"), bool)
        and isinstance(problem.get("nl_reference_plan"), list)
        and not any(key in problem for key in (
            "goal_spec", "author_goal_spec", "author_require_quiescent",
        )))

    all_actions = prefix + plan
    raw = raw_trace(problem["program"], problem["seed"], all_actions)
    wrapped = wrapper_trace(problem["program"], problem["seed"], all_actions)
    if nl_plan == plan:
        nl_raw, nl_wrapped = raw, wrapped
    else:
        nl_actions = prefix + nl_plan
        nl_raw = raw_trace(problem["program"], problem["seed"], nl_actions)
        nl_wrapped = wrapper_trace(problem["program"], problem["seed"], nl_actions)
    checks["A2-drivers"] = raw == wrapped and nl_raw == nl_wrapped

    start = wrapped[len(prefix)]
    frames = wrapped[len(prefix) + 1:]
    nl_start = nl_wrapped[len(prefix)]
    nl_frames = nl_wrapped[len(prefix) + 1:]
    ref_ok, reached_at = frame_success(problem, start, frames, plan)
    task_ok = task_success(problem, nl_start, nl_frames, nl_plan)
    checks["A3-reference"] = bool(
        start == problem["start"] and nl_start == start and ref_ok and task_ok
    )
    detail["reached_at"] = reached_at
    detail["nl_reference_h"] = len(nl_plan)

    checks["A4-nontrivial"] = bool(plan and nl_plan and not start_satisfies(problem, start))

    frame_noops = ["noop"] * len(plan)
    noop_start, noop_frames = run(problem, frame_noops)
    noop_frame = frame_success(problem, noop_start, noop_frames, frame_noops)[0]
    task_noops = ["noop"] * len(nl_plan)
    task_noop_start, task_noop_frames = run(problem, task_noops)
    noop_task = task_success(problem, task_noop_start, task_noop_frames, task_noops)
    checks["A5-noop-fails"] = not noop_frame and not noop_task
    detail["noop_frame_success"] = noop_frame
    detail["noop_task_success"] = noop_task

    # A5b: screen noops under the EVAL-TIME rule (any-step at the 50-action cap,
    # quiescence waived, as eval_curated_online scores) -- A5 runs each checker in its
    # authored final mode, so a predicate a one-tick transient satisfies (the 2026-09-01
    # shoot-enemy occlusion exploit) passes A5 yet is random-trivial at eval time.
    cap_noops = ["noop"] * 50
    cap_start, cap_frames = run(problem, cap_noops)
    any_goal = dc_replace(validate_problem_goal(problem),
                          success_mode="any", require_quiescent=False)
    noop_any_task = score_python_goal(any_goal, cap_start, cap_frames, cap_noops)[0]
    any_probe = dict(problem, frame_success_mode="any")
    noop_any_frame = success(any_probe, "frame", cap_start, cap_frames, cap_noops)[0]
    checks["A5b-noop-anystep-fails"] = not noop_any_task and not noop_any_frame
    detail["noop_anystep_task_success"] = noop_any_task
    detail["noop_anystep_frame_success"] = noop_any_frame

    frame_deletions = []
    for i in range(len(plan)):
        candidate = plan[:i] + plan[i + 1:]
        cstart, cframes = run(problem, candidate)
        if frame_success(problem, cstart, cframes, candidate)[0]:
            frame_deletions.append(i)

    task_deletions = []
    for i in range(len(nl_plan)):
        candidate = nl_plan[:i] + nl_plan[i + 1:]
        cstart, cframes = run(problem, candidate)
        if task_success(problem, cstart, cframes, candidate):
            task_deletions.append(i)
    checks["A6-frame-delete"] = not frame_deletions
    enforce_nl_minimality = problem.get("source") != "curated-v1-accepted-migration"
    checks["A7-task-delete"] = not task_deletions or not enforce_nl_minimality
    detail["frame_deletions"] = frame_deletions
    detail["task_deletions"] = task_deletions

    substitutions = []
    for i, action in enumerate(nl_plan):
        if action == "noop":
            continue
        candidate = nl_plan[:i] + ["noop"] + nl_plan[i + 1:]
        cstart, cframes = run(problem, candidate)
        if task_success(problem, cstart, cframes, candidate):
            substitutions.append(i)
    checks["A8-substitute"] = not substitutions or not enforce_nl_minimality
    detail["noop_substitutions"] = substitutions
    if not enforce_nl_minimality and (task_deletions or substitutions):
        warnings.append("accepted v1 NL witness is not semantically minimal")

    extra = wrapper_trace(problem["program"], problem["seed"], prefix + plan + ["noop"])
    actual_quiescent = extra[-1] == extra[-2]
    checks["A9-quiescence"] = actual_quiescent == problem["frame_reference_quiescent"]
    detail["actual_quiescent"] = actual_quiescent

    floor_ok = True
    floor_detail = {}
    for presentation in ("frame", "nl"):
        trials = problem.get(f"{presentation}_random_trials")
        if not verify_random or not trials:
            floor_detail[presentation] = None
            continue
        rng = random.Random(stable_seed(f"{uid}:{presentation}:h{len(plan)}"))
        hits = 0
        for _ in range(int(trials)):
            candidate = random_plan(problem["game"], len(plan), rng)
            cstart, cframes = run(problem, candidate)
            if presentation == "frame":
                hit = frame_success(problem, cstart, cframes, candidate)[0]
            else:
                hit = task_success(problem, cstart, cframes, candidate)
            hits += hit
        rate = hits / int(trials)
        floor_detail[presentation] = rate
        floor_ok &= rate == problem.get(f"{presentation}_random_success")
    checks["A10-random-floor"] = floor_ok
    detail["wrapper_random_success"] = floor_detail

    required = list(CHECKS)
    failed = [name for name in required if not checks[name]]
    return {"task_uid": uid, "game": problem["game"], "id": problem["id"],
            "source": problem.get("source"), "ok": not failed, "failed": failed,
            "checks": checks, "detail": detail, "warnings": warnings}


def global_checks(problems: list[dict[str, Any]]) -> dict[str, Any]:
    uids = [p.get("task_uid") for p in problems]
    counts = Counter(p["game"] for p in problems)
    templates: dict[tuple[str, str], set[int]] = defaultdict(set)
    for p in problems:
        templates[(p["game"], p["template_id"])].add(p["seed"])
    multi = {
        game: max((len(seeds) for (g, _), seeds in templates.items() if g == game), default=0)
        for game in STOCHASTIC_GAMES
    }
    checker_errors = []
    for problem in problems:
        try:
            validate_problem_goal(problem)
        except (KeyError, ValueError) as exc:
            checker_errors.append(f"{problem.get('task_uid')}: {exc}")
    checker_ids = {p.get("nl_checker") for p in problems}
    checks = {
        "unique_task_uids": len(uids) == len(set(uids)),
        "all_15_games_present": set(counts) == set(SELECTED_GAMES),
        "legacy_28_retained": sum(p.get("source") == "curated-v1-accepted-migration"
                                  for p in problems) == 28,
        "every_problem_has_both_goal_representations": all(
            p.get("goal") and p.get("nl_goal") and p.get("nl_checker")
            for p in problems
        ),
        "stochastic_multi_seed": all(value >= 3 for value in multi.values()),
        "all_nl_checkers_registered": not checker_errors,
        "expected_68_checker_programs": len(checker_ids) == 68 and None not in checker_ids,
        "declarative_goal_payloads_absent": all(
            not any(key in p for key in (
                "goal_spec", "author_goal_spec", "author_require_quiescent",
            )) for p in problems
        ),
    }
    return {"ok": all(checks.values()), "checks": checks,
            "counts_by_game": {g: counts[g] for g in SELECTED_GAMES},
            "max_seeds_per_stochastic_template": multi,
            "checker_errors": checker_errors}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("problems", nargs="?", type=Path, default=DEFAULT_PROBLEMS)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--skip-random", action="store_true")
    args = parser.parse_args()

    meta, problems = load_problem_file(args.problems)
    rows = []
    for i, problem in enumerate(problems, 1):
        result = audit_one(problem, not args.skip_random)
        rows.append(result)
        suffix = "ok" if result["ok"] else "FAIL " + ",".join(result["failed"])
        print(f"  [{i:02d}/{len(problems)}] {result['task_uid']:<44} {suffix}", flush=True)

    global_result = global_checks(problems)
    failures = [r for r in rows if not r["ok"]]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "validated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "problems_file": str(args.problems),
        "problem_file_metadata": meta,
        "ok": not failures and global_result["ok"],
        "n": len(rows), "passed": len(rows) - len(failures),
        "global": global_result, "rows": rows,
    }
    out = args.out or args.problems.with_name("validation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"\n{payload['passed']}/{payload['n']} row audits pass; "
          f"global={'ok' if global_result['ok'] else 'FAIL'} -> {out}")
    if failures:
        for row in failures:
            print(f"  FAILED {row['task_uid']}: {', '.join(row['failed'])}")
    if not global_result["ok"]:
        print("  GLOBAL", {k: v for k, v in global_result["checks"].items() if not v})
    sys.exit(0 if payload["ok"] else 1)


if __name__ == "__main__":
    main()
