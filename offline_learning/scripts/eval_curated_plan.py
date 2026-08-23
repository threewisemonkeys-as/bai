#!/usr/bin/env python3
"""OFFLINE (open-loop) planning eval on the curated problem set.

Given the START state and the GOAL state, ask for a whole plan, execute it in the engine,
and check whether the goal frame is ever reached.  Five independent attempts per LLM arm
per problem.

Config is inherited from `eval_coverage_plan` by IMPORT, not by copy: the prompt templates,
the LLM call, the plan parser, the transcript builders and the wc search constants are the
same objects the coverage run used, so those cannot drift.  Three deliberate differences,
all settled with the user:

  * PLAN_CAP is 50, not 20.  Seven curated problems have h > 20 (all-coins-kill 34,
    staircase 33, all-coins 32, ...) and would be unsolvable by construction under the old
    cap.  h itself is still never disclosed.
  * NO HISTORY.  The coverage prompt opened with up to CONTEXT_K=9 preceding
    (state, action) pairs from a human drive; curated problems start from a bare engine
    reset, so there is nothing to show and the transcript is the CURRENT state alone.
    This is strictly harder for `raw`, whose only dynamics signal was that history.
  * FIVE attempts per LLM arm.  `wc` is a deterministic program search with no sampling,
    so it runs once and is reported as such.

Scored per game and per tier, never pooled into one number, with the per-problem
`random_success` floor carried alongside.

    uv run python offline_learning/scripts/eval_curated_plan.py \
        --problems logs/2026-08-18/curated/problems.json \
        --out logs/2026-08-18/curated/eval/offline
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OFF = HERE.parent
REPO = OFF.parent
for _p in (str(OFF), str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import program_runtime as prt  # noqa: E402
from validate import _parse_tag, run_perceive  # noqa: E402
from worldcoder_optimize import _clean_program  # noqa: E402
from offline_learning.coverage_plan import exec_plan  # noqa: E402
from offline_learning.human_replay import GAMES as HGAMES  # noqa: E402

import eval_coverage_plan as ecp  # noqa: E402
from eval_coverage_plan import (  # noqa: E402
    CONTEXT_K, DEFAULT_KNOWLEDGE, PLAN_RAW_TMPL, PLAN_WIN_TMPL, WC_BEAM, WC_BUDGET,
    feat_transcript, llm_call, parse_plan, raw_transcript, resolve_llm_config,
)

PLAN_CAP = 50               # was 20; see module docstring
ATTEMPTS = 5
LLM_ARMS = ["raw", "lmwm"]
ARMS = LLM_ARMS + ["wc"]
TIERS = ["L1", "L2", "L3", "L4"]


def gstr(grid: list[list[str]]) -> str:
    """Canonical grid string, byte-identical to what `_grid()` pulls out of a wrapper
    observation -- json.dumps' default ', ' separator is what the wrapper emits."""
    return json.dumps(grid)


def reached(grids: list[str | None], goal: str) -> tuple[bool, int | None]:
    for j, g in enumerate(grids):
        if g == goal:
            return True, j + 1
    return False, None


async def eval_game(game: str, problems: list[dict], sem, llm, artifact_root: Path,
                    a_reason: bool = True, a_keep: bool = False) -> dict:
    rex = artifact_root / "rexpure" / f"{game}_s1"
    perc_code = (rex / "best_perception_rexpure_seed1.py").read_text()
    beliefs = (rex / "best_beliefs_rexpure_seed1.txt").read_text()
    wc_path = artifact_root / "worldcoder" / f"{game}_s1/best_transition_wc_seed1.py"
    rt = prt.ProgramRuntime(_clean_program(wc_path.read_text()), timeout_s=1.0)
    verbs = HGAMES[game][2]

    pcache: dict[str, str] = {}

    def perceive(g: str) -> str:
        if g not in pcache:
            pcache[g] = run_perceive(perc_code, g)[0]
        return pcache[g]

    for p in problems:
        p["_start"] = gstr(p["start"])
        p["_goal"] = gstr(p["goal"])
        p["_z_t"] = perceive(p["_start"])
        p["_z_goal"] = perceive(p["_goal"])
        p["_dims"] = (len(p["start"]), len(p["start"][0]))

    async def attempts(p: dict, arm: str):
        if arm == "raw":
            prompt = PLAN_RAW_TMPL.format(
                cap=PLAN_CAP, default_knowledge=DEFAULT_KNOWLEDGE,
                transcript=raw_transcript([], p["_start"]), goal=p["_goal"])
        else:
            prompt = PLAN_WIN_TMPL.format(
                cap=PLAN_CAP, default_knowledge=DEFAULT_KNOWLEDGE,
                beliefs=beliefs.strip() or "(empty)",
                transcript=feat_transcript([], p["_z_t"]),
                goal=p["_z_goal"] or "(empty)")
        return await asyncio.gather(*(llm_call(prompt, sem, llm) for _ in range(ATTEMPTS)))

    jobs = [(p, arm) for p in problems for arm in LLM_ARMS]
    got = await asyncio.gather(*(attempts(p, arm) for p, arm in jobs))
    calls: dict[tuple[str, str], list] = {}
    for (p, arm), res in zip(jobs, got):
        calls[(p["id"], arm)] = res

    rows, cost = [], 0.0
    for p in problems:
        row = {k: p[k] for k in ("game", "id", "tier", "objective", "h", "n_decisions",
                                 "seed", "quiescent", "random_success")}
        row["start_grid"], row["goal_grid"] = p["_start"], p["_goal"]
        for arm in LLM_ARMS:
            tries = []
            for text, think, c, errs in calls[(p["id"], arm)]:
                cost += c
                plan, perr = parse_plan(text, p["_dims"])
                if plan is not None and len(plan) > PLAN_CAP:
                    plan, perr = None, f"budget-exceeded:{len(plan)}>{PLAN_CAP}"
                ok, at, grids = False, None, None
                if plan is not None:
                    grids = exec_plan(p["program"], p["seed"], [], plan)
                    ok, at = reached(grids, p["_goal"])
                rec = {"success": ok, "reached_at": at,
                       "plan_len": len(plan) if plan else None,
                       "plan_error": perr, "retry_errors": errs, "plan": plan}
                if a_reason:
                    # Two different things, and both are kept.  `reasoning` is the model's
                    # own <reasoning> block: its stated justification, written for a reader.
                    # `thinking` is the provider's hidden chain, which on a reasoning model
                    # IS the deliberation -- the visible block is a summary composed after
                    # the fact.  It is stored capped (see eval_coverage_plan.REASONING_CAP)
                    # rather than dropped; --no-keep-thinking omits it entirely.
                    rec["reasoning"] = _parse_tag(text, "reasoning")
                    if a_keep:
                        rec.update(ecp.thinking_record(think))
                tries.append(rec)
            row[arm] = {"attempts": tries,
                        "pass_rate": sum(t["success"] for t in tries) / len(tries),
                        "pass_any": any(t["success"] for t in tries)}
        # wc: deterministic program search, no sampling -> one attempt
        start_g, goal_g = json.loads(p["_start"]), json.loads(p["_goal"])
        universe = prt.build_action_universe(verbs, start_g, goal_g)
        found = prt.plan_search(rt, [], start_g, goal_g, universe, PLAN_CAP,
                                beam=WC_BEAM, node_budget=WC_BUDGET, context_k=CONTEXT_K,
                                allow_empty=False)
        plan = [prt.unparse_action(a) for a in found] if found is not None else None
        ok, at = False, None
        if plan is not None:
            ok, at = reached(exec_plan(p["program"], p["seed"], [], plan), p["_goal"])
        row["wc"] = {"attempts": [{"success": ok, "reached_at": at,
                                   "plan_len": len(plan) if plan else None,
                                   "plan_error": None if plan else "no-plan-found",
                                   "retry_errors": [], "plan": plan}],
                     "pass_rate": float(ok), "pass_any": ok}
        rows.append(row)
    rt.close()
    return {"game": game, "rows": rows, "cost": cost}


def report(all_rows: list[dict], llm, elapsed: float, cost: float) -> str:
    L = ["# Curated planning eval - OFFLINE (open-loop)", "",
         f"Planner: {llm.label} | plan cap {PLAN_CAP} | {ATTEMPTS} attempts per LLM arm "
         f"| wc budget {WC_BUDGET} (deterministic, 1 attempt) | no history, CURRENT state "
         f"only | {len(all_rows)} problems | {elapsed / 60:.0f} min | ${cost:.2f}", "",
         "`pass@1` is the mean over attempts, `pass@5` is any-of-five. `rand` is the "
         "per-problem random-plan floor. Scored per game, never pooled.", ""]

    L += ["## Per game", "",
          "| game | n | raw @1 | raw @5 | lmwm @1 | lmwm @5 | wc | rand |",
          "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for game in dict.fromkeys(r["game"] for r in all_rows):
        s = [r for r in all_rows if r["game"] == game]
        L.append(f"| {game} ({HGAMES[game][1]}) | {len(s)} | " + " | ".join(
            [f"{sum(r['raw']['pass_rate'] for r in s) / len(s):.2f}",
             f"{sum(r['raw']['pass_any'] for r in s) / len(s):.2f}",
             f"{sum(r['lmwm']['pass_rate'] for r in s) / len(s):.2f}",
             f"{sum(r['lmwm']['pass_any'] for r in s) / len(s):.2f}",
             f"{sum(r['wc']['pass_rate'] for r in s) / len(s):.2f}",
             f"{sum(r['random_success'] for r in s) / len(s):.2f}"]) + " |")

    L += ["", "## Per tier", "",
          "| tier | n | raw @1 | raw @5 | lmwm @1 | lmwm @5 | wc |", "|---|--:|--:|--:|--:|--:|--:|"]
    for tier in TIERS:
        s = [r for r in all_rows if r["tier"] == tier]
        if not s:
            continue
        L.append(f"| {tier} | {len(s)} | " + " | ".join(
            [f"{sum(r[a]['pass_rate'] for r in s) / len(s):.2f}" if k == 0 else
             f"{sum(r[a]['pass_any'] for r in s) / len(s):.2f}"
             for a in ("raw", "lmwm") for k in (0, 1)]
            + [f"{sum(r['wc']['pass_rate'] for r in s) / len(s):.2f}"]) + " |")

    L += ["", "## Per problem", "",
          "| game | tier | id | h | dec | raw @1 | lmwm @1 | wc | rand |",
          "|---|---|---|--:|--:|--:|--:|--:|--:|"]
    for r in all_rows:
        L.append(f"| {r['game']} | {r['tier']} | `{r['id']}` | {r['h']} | "
                 f"{r['n_decisions']} | {r['raw']['pass_rate']:.1f} | "
                 f"{r['lmwm']['pass_rate']:.1f} | {r['wc']['pass_rate']:.0f} | "
                 f"{r['random_success']:.2f} |")

    bad = defaultdict(int)
    for r in all_rows:
        for a in ARMS:
            for t in r[a]["attempts"]:
                if t["plan_error"]:
                    bad[f"{a}:{t['plan_error'].split(':')[0]}"] += 1
    if bad:
        L += ["", "## Unusable responses", ""]
        L += [f"- {k}: {v}" for k, v in sorted(bad.items())]
    return "\n".join(L) + "\n"


async def main_async(a):
    problems = json.loads(Path(a.problems).read_text())
    if a.games:
        want = set(a.games.split(","))
        problems = [p for p in problems if p["game"] in want]
    if a.limit:
        per = defaultdict(int)
        keep = []
        for p in problems:
            if per[p["game"]] < a.limit:
                keep.append(p)
                per[p["game"]] += 1
        problems = keep
    llm = resolve_llm_config(a)
    sem = asyncio.Semaphore(a.concurrency)
    root = Path(a.artifact_root)
    by_game: dict[str, list] = defaultdict(list)
    for p in problems:
        by_game[p["game"]].append(p)

    t0 = time.time()
    out = await asyncio.gather(*(eval_game(g, ps, sem, llm, root,
                                           a.reasoning_trace, a.keep_thinking)
                                 for g, ps in by_game.items()))
    rows = [r for o in out for r in o["rows"]]
    cost = sum(o["cost"] for o in out)
    md = report(rows, llm, time.time() - t0, cost)
    print(md)
    stem = Path(a.out)
    stem.parent.mkdir(parents=True, exist_ok=True)
    from collections import Counter as _C
    served = _C(str(c.get("provider")) for c in ecp.CALL_STATS)
    walls = sorted(c["wall_s"] for c in ecp.CALL_STATS)
    stem.with_suffix(".json").write_text(json.dumps(
        {"config": {"model": llm.model, "backend": llm.backend, "plan_cap": PLAN_CAP,
                    "attempts": ATTEMPTS, "context_k": 0, "wc_budget": WC_BUDGET,
                    "label": llm.label, "concurrency": a.concurrency,
                    # which hosts ACTUALLY served, and how slow they were: a routing
                    # change that is not recorded here cannot be attributed afterwards
                    "providers_served": dict(served.most_common()),
                    "call_p50_s": walls[len(walls)//2] if walls else None,
                    "call_mean_s": (sum(walls)/len(walls)) if walls else None},
         "rows": rows, "cost": cost}, indent=1))
    stem.with_suffix(".md").write_text(md)
    print(f"wrote {stem}.json / {stem}.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="logs/2026-08-18/curated/problems.json")
    ap.add_argument("--out", default="logs/2026-08-18/curated/eval/offline")
    ap.add_argument("--games", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--artifact-root", default=str(ecp.ARTIFACT_ROOT))
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--llm-backend", choices=("claude", "openrouter"), default="openrouter")
    ap.add_argument("--llm-url", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--provider-order", default=None)
    ecp.add_llm_tuning_args(ap)
    ap.add_argument("--reasoning-trace", action=argparse.BooleanOptionalAction, default=True,
                    help="persist each attempt's <reasoning> block")
    ap.add_argument("--keep-thinking", action=argparse.BooleanOptionalAction, default=True,
                    help="persist the provider's hidden reasoning tokens, capped by "
                         "LLM_REASONING_CAP (default 8000 chars/call)")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
