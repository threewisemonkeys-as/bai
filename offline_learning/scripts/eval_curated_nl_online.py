#!/usr/bin/env python3
"""ONLINE (receding-horizon / MPC) planning eval with NATURAL-LANGUAGE goals.

Closed-loop counterpart of `eval_curated_nl.py`, on the SAME five problems with the SAME
sentences, budget and attempt count, so every attempt is paired with its open-loop result.
Per round: the model plans at most (budget - n) actions toward the goal from the CURRENT
observed state, ONLY THE FIRST is executed, the resulting state joins the transcript, and it
replans.

Why this is the interesting arm here.  The open-loop run's dominant failure was not bad
planning but bad PREDICTION OF DURATION -- s2kt7 clicked correctly every time and then waited
five ticks when the ants needed fourteen, with forty-five actions of budget unspent.  Nothing
in an open-loop protocol can fix that: the agent commits to a length before it sees anything.
Closing the loop lets it look at the board and wait again.  The offline/online gap on these
five is therefore a fairly direct measure of how much of the failure was forward-model error
rather than planning error.

The rollout loop is written here rather than imported from `eval_coverage_online.llm_rollout`,
which tests `new_grid == goal_grid` every round and formats a goal frame into every prompt --
neither of which exists once the goal is a sentence.  Everything around the loop IS imported
(`Branch`, the engine executor, the retry suffix, the perception compiler, the LLM call and
plan parser), so the protocol is the same code the frame-goal online run used.

SUCCESS IS THE SAME ANY-STEP CHECKER as offline, evaluated on the frames actually executed
after every round.  Note what that means in a closed loop: the harness stops the rollout the
moment the checker accepts, so an agent that stumbles into the goal is indistinguishable from
one that knew it was there.  There is no submit action in either arm -- adding one would
separate those two, but it would also break pairing with the offline run unless both get it.

    uv run python offline_learning/scripts/eval_curated_nl_online.py \
        --model google/gemini-3.7-flash --provider-order google \
        --out logs/2026-08-19/nl_pilot/eval/online
"""
from __future__ import annotations

import argparse
import asyncio
import json
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

from offline_learning.human_replay import GAMES as HGAMES  # noqa: E402
from offline_learning.nl_goals import GOALS, NLGoal  # noqa: E402

import eval_coverage_plan as ecp  # noqa: E402
from eval_coverage_plan import (  # noqa: E402
    CONTEXT_K, DEFAULT_KNOWLEDGE, feat_transcript, llm_call, parse_plan, raw_transcript,
    resolve_llm_config, thinking_record,
)
from eval_coverage_online import (  # noqa: E402
    Branch, RETRY_SUFFIX, compile_perceive, engine,
)
from eval_curated_plan import ATTEMPTS, gstr  # noqa: E402
from eval_curated_nl import PLAN_NL_RAW_TMPL, PLAN_NL_WIN_TMPL, parse_reasoning  # noqa: E402

MAX_ACTIONS = 50            # same budget as the open-loop run, for pairing

# The frame-goal warm start says "still reaches the GOAL exactly", which is a claim about a
# picture the planner no longer has.
WARM_NL_TMPL = """=== CANDIDATE PLAN (proposed one step earlier; may be stale) ===
{cand}
=== END CANDIDATE PLAN ===
First check the candidate against the CURRENT state: if executing it in order still achieves
the GOAL described above, return it unchanged; revise or replace it only if it no longer
does. Your plan must have at most {remaining} action(s).
"""


def _tup(grid_str: str):
    return tuple(tuple(r) for r in json.loads(grid_str))


async def nl_rollout(goal: NLGoal, row: dict, perceive, beliefs: str, llm, sem,
                     args, arm: str = "lmwm") -> dict:
    """One receding-horizon rollout.  `arm` chooses what the planner sees each round: the raw
    grid, or the learned perception module's features plus its beliefs.  Everything else --
    budget, warm start, the corrective re-ask, the checker -- is shared, so the arms differ
    only in the state representation."""
    budget = args.max_actions
    dims = (len(row["start"]), len(row["start"][0]))
    branch = await engine(Branch, row["program"], goal.seed, [], budget)
    try:
        cur_grid = await engine(branch.grid)
        frames = [_tup(cur_grid)]
        actions: list[str] = []
        hist: list[tuple[str, str]] = []
        cur_z = perceive(cur_grid)[0] if arm == "lmwm" else cur_grid
        rounds: list[dict] = []
        cost, n = 0.0, 0
        success, sat_at, failed = False, None, None
        carry: list[str] = []

        while n < budget:
            remaining = budget - n
            if arm == "raw":
                prompt = PLAN_NL_RAW_TMPL.format(
                    cap=remaining, default_knowledge=DEFAULT_KNOWLEDGE,
                    transcript=raw_transcript(hist[-CONTEXT_K:], cur_z), goal=goal.nl)
            else:
                prompt = PLAN_NL_WIN_TMPL.format(
                    cap=remaining, default_knowledge=DEFAULT_KNOWLEDGE,
                    beliefs=beliefs.strip() or "(empty)",
                    transcript=feat_transcript(hist[-CONTEXT_K:], cur_z), goal=goal.nl)
            if args.warm_start and carry:
                warm = WARM_NL_TMPL.format(cand="\n".join(carry), remaining=remaining)
                prompt = prompt.replace("\nRespond as:\n", f"\n{warm}\nRespond as:\n", 1)

            text, think, c1, errs = await llm_call(prompt, sem, llm)
            cost += c1
            plan, perr = parse_plan(text, dims)
            if plan is not None and len(plan) > remaining:
                plan, perr = None, f"budget-exceeded:{len(plan)}>{remaining}"
            if plan is None:                       # one corrective re-ask, as offline
                fix = prompt + RETRY_SUFFIX.format(error=perr, remaining=remaining)
                text, think, c2, errs2 = await llm_call(fix, sem, llm)
                cost += c2
                errs = errs + errs2
                plan, perr = parse_plan(text, dims)
                if plan is not None and len(plan) > remaining:
                    plan, perr = None, f"budget-exceeded:{len(plan)}>{remaining}"

            block, rlines = parse_reasoning(text)
            if plan is None:
                rounds.append({"n": n, "remaining": remaining, "plan": None,
                               "executed": None, "plan_error": perr, "retry_errors": errs,
                               "reasoning": block, "why": "", "grid_after": None,
                               **thinking_record(think)})
                failed = "invalid-plan"
                break

            action = plan[0]
            carry = plan[1:]
            await engine(branch.step, action)
            new_grid = await engine(branch.grid)
            hist.append((cur_z, action))
            cur_z = perceive(new_grid)[0] if arm == "lmwm" else new_grid
            cur_grid = new_grid
            frames.append(_tup(new_grid))
            actions.append(action)
            n += 1
            ok = goal.check(frames, actions)
            rounds.append({"n": n - 1, "remaining": remaining, "plan": plan,
                           "executed": action, "plan_error": None, "retry_errors": errs,
                           "reasoning": block, "why": rlines.get(1, ""),
                           "grid_after": new_grid, "terminated": branch.terminated,
                           "satisfied": ok, **thinking_record(think)})
            if ok:
                success, sat_at = True, n
                break
            if branch.terminated:
                failed = "terminated"
                break
        if not success and failed is None:
            failed = "budget-exhausted"
    finally:
        await engine(branch.close)

    goal_frame = gstr(row["goal"])
    return {"success": success, "satisfied_at": sat_at, "actions_used": n,
            "failed_reason": failed, "cost": cost, "rounds": rounds,
            "frame_hit": any(r.get("grid_after") == goal_frame for r in rounds),
            "plan": actions}


def load_ckpt(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue                      # torn final line from a hard kill
            out[d["key"]] = d["result"]
    return out


async def checkpointed(coro, path: Path, key: str, lock):
    """Write each rollout the moment it lands.  A rollout here is up to 50 LLM calls; a run
    killed at the end of a terminal gather would otherwise return nothing at all."""
    res = await coro
    async with lock:
        with path.open("a") as fh:
            fh.write(json.dumps({"key": key, "result": res}) + "\n")
    return res


def report(rows: list[dict], off_idx: dict, llm, args, elapsed: float, cost: float,
           arms: list[str]) -> str:
    L = ["# Curated planning eval - NATURAL-LANGUAGE goals, ONLINE (receding horizon)", "",
         f"Planner: {llm.label} | rollout cap {args.max_actions} actions | plan cap = "
         f"{args.max_actions}-n each round | {args.attempts} attempt(s) | warm_start="
         f"{args.warm_start} | goal stated in words, no target frame shown | ANY-STEP "
         f"predicate scoring | {len(rows)} problems | {elapsed / 60:.0f} min | ${cost:.2f}",
         "",
         "Each round the model plans toward the goal, ONLY the first action is executed, then "
         "it replans from the observed state. `(off)` is the paired OPEN-LOOP result on the "
         "identical problems, sentences and attempt count -- the gap is what closing the loop "
         "bought. `used` is the mean number of actions executed.", "",
         "| game | id | h | " + " | ".join(
             f"{a} on | {a} (off) | {a} used" for a in arms) + " | rand@50 |",
         "|---|---|--:|" + "--:|" * (3 * len(arms) + 1)]
    for r in rows:
        cellsm = []
        for arm in arms:
            d = r.get(arm)
            off = off_idx.get((r["id"], arm))
            cellsm.append(
                "-- | -- | --" if d is None else
                f"{d['pass_rate']:.2f} | {'--' if off is None else f'{off:.2f}'} | "
                f"{d['mean_used']:.1f}")
        L.append(f"| {r['game']} | `{r['id']}` | {r['h']} | " + " | ".join(cellsm) +
                 f" | {'--' if r['rand'] is None else f'{r['rand']:.3f}'} |")
    fails = defaultdict(int)
    for r in rows:
        for arm in arms:
            for a in r.get(arm, {}).get("attempts", []):
                if not a["success"]:
                    fails[f"{arm}:{a['failed_reason'] or '?'}"] += 1
    if fails:
        L += ["", "## Why the failures ended", ""]
        L += [f"- {k}: {v}" for k, v in sorted(fails.items())]
    return "\n".join(L) + "\n"


async def main_async(a):
    curated = {r["id"]: r for r in json.loads(Path(a.problems).read_text())}
    goals = [g for g in GOALS if not a.pid or g.pid in a.pid]
    llm = resolve_llm_config(a)
    sem = asyncio.Semaphore(a.concurrency)
    root = Path(a.artifact_root)

    val = {}
    if Path(a.validation).exists():
        val = {r["pid"]: (r["N5"] or {}).get("floor_at_cap")
               for r in json.loads(Path(a.validation).read_text())}
    arms = [x for x in a.arms.split(",") if x]
    off_idx: dict[tuple[str, str], float] = {}
    if Path(a.offline).exists():
        off = json.loads(Path(a.offline).read_text())
        off_idx = {(r["id"], arm): r[arm]["pass_rate"]
                   for r in off["rows"] for arm in arms if arm in r}

    res = {}
    for g in goals:
        rex = root / "rexpure" / f"{g.game}_s1"
        res[g.game] = (compile_perceive((rex / "best_perception_rexpure_seed1.py").read_text()),
                       (rex / "best_beliefs_rexpure_seed1.txt").read_text())

    ckpt_path = Path(a.out).with_suffix(".ckpt.jsonl")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_ckpt(ckpt_path) if a.resume else {}
    if done:
        print(f"resuming: {len(done)} rollouts already on disk", flush=True)
    lock = asyncio.Lock()

    # The key carries the ARM as well as the attempt index, so a run at --attempts 1 can be
    # topped up to 5 later and only the four missing rollouts are executed.
    jobs = [(g, arm, k) for g in goals for arm in arms for k in range(a.attempts)
            if f"{g.pid}|{arm}|{k}" not in done]
    t0 = time.time()
    print(f"{len(jobs)} rollouts to run (cap {a.max_actions} actions each)", flush=True)
    out = await asyncio.gather(*(
        checkpointed(nl_rollout(g, curated[g.pid], *res[g.game], llm, sem, a, arm),
                     ckpt_path, f"{g.pid}|{arm}|{k}", lock) for g, arm, k in jobs))
    for (g, arm, k), r in zip(jobs, out):
        done[f"{g.pid}|{arm}|{k}"] = r

    rows, cost = [], 0.0
    for g in goals:
        row = {"game": g.game, "id": g.pid, "tier": g.tier, "nl": g.nl, "seed": g.seed,
               "h": curated[g.pid]["h"], "objective": curated[g.pid]["objective"],
               "rand": val.get(g.pid)}
        for arm in arms:
            att = [done[f"{g.pid}|{arm}|{k}"] for k in range(a.attempts)
                   if f"{g.pid}|{arm}|{k}" in done]
            if not att:
                continue
            cost += sum(x["cost"] for x in att)
            row[arm] = {
                "pass_rate": sum(x["success"] for x in att) / len(att),
                "pass_any": any(x["success"] for x in att),
                "frame_rate": sum(x["frame_hit"] for x in att) / len(att),
                "mean_used": sum(x["actions_used"] for x in att) / len(att),
                "attempts": att,
            }
        rows.append(row)

    md = report(rows, off_idx, llm, a, time.time() - t0, cost, arms)
    print(md)
    stem = Path(a.out)
    stem.with_suffix(".json").write_text(json.dumps(
        {"config": {"model": llm.model, "backend": llm.backend, "arms": arms,
                    "max_actions": a.max_actions, "attempts": a.attempts,
                    "warm_start": a.warm_start, "scoring": "any-step-predicate",
                    "mode": "online"},
         "rows": rows, "cost": cost}, indent=1))
    stem.with_suffix(".md").write_text(md)
    print(f"wrote {stem}.json / {stem}.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="logs/2026-08-18/curated/problems.json")
    ap.add_argument("--validation", default="logs/2026-08-19/nl_pilot/validation.json")
    ap.add_argument("--offline", default="logs/2026-08-19/nl_pilot/eval/offline.json")
    ap.add_argument("--out", default="logs/2026-08-19/nl_pilot/eval/online")
    ap.add_argument("--pid", action="append")
    ap.add_argument("--max-actions", type=int, default=MAX_ACTIONS)
    ap.add_argument("--arms", default="lmwm", help="comma-separated: lmwm[,raw]")
    ap.add_argument("--attempts", type=int, default=ATTEMPTS,
                    help="rollouts per problem per arm; the checkpoint keys on the attempt "
                         "index, so a later run with a larger value tops up")
    ap.add_argument("--warm-start", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--artifact-root", default=str(ecp.ARTIFACT_ROOT))
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--llm-backend", choices=("claude", "openrouter"), default="openrouter")
    ap.add_argument("--llm-url", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--provider-order", default=",".join(ecp.DEFAULT_PROVIDER_ORDER))
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
