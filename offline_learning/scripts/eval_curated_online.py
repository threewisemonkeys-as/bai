#!/usr/bin/env python3
"""ONLINE (receding-horizon / MPC) planning eval on the curated problem set.

Closed-loop counterpart of eval_curated_plan.py, on the SAME problems so every arm x
problem x attempt is paired with its open-loop result.  Per round: the model plans a
sequence of at most (budget - n) actions to the GOAL from the CURRENT state, ONLY the first
action is executed, the resulting state joins the history, and it replans.  Success iff the
rendered grid ever equals the goal frame.

The rollout functions themselves -- `llm_rollout`, `wc_rollout`, `Branch` -- are IMPORTED
from `eval_coverage_online`, not reimplemented, so the protocol (corrective re-ask on an
unusable plan, cold start, wc's exact-prediction carry, engine serialization) is the same
code the coverage run used.  This module only supplies the problems.

Same three deviations as the offline curated eval, all settled with the user: budget 50
rather than 20 (7 problems have h > 20); no pre-window history, because curated problems
start from a bare engine reset; 5 attempts per LLM arm, with `wc` run once because a
program search does not sample.

    uv run python offline_learning/scripts/eval_curated_online.py \
        --problems logs/2026-08-18/curated/problems.json \
        --offline logs/2026-08-18/curated/eval/offline.json \
        --out logs/2026-08-18/curated/eval/online
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
from worldcoder_optimize import _clean_program  # noqa: E402
from offline_learning.human_replay import GAMES as HGAMES  # noqa: E402

import eval_coverage_plan as ecp  # noqa: E402
import eval_coverage_online as eco  # noqa: E402
from eval_coverage_online import compile_perceive, llm_rollout, wc_rollout  # noqa: E402
from eval_coverage_plan import WC_BUDGET, resolve_llm_config  # noqa: E402
from eval_curated_plan import ATTEMPTS, LLM_ARMS, PLAN_CAP, TIERS, gstr  # noqa: E402

ARMS = LLM_ARMS + ["wc"]

# Rounds happen inside the imported `llm_rollout`, and the checkpoint only records TERMINAL
# state, so a run's real progress is invisible: 133 rollouts can each be at round 47 of 50
# and the completion count still reads zero.  Counting calls as they are issued is the only
# in-flight signal available without editing the shared reference module.
_CALLS = {"n": 0}


def count_calls(fn):
    async def wrapped(*args, **kw):
        _CALLS["n"] += 1
        return await fn(*args, **kw)
    return wrapped


async def heartbeat(period: int, total_rollouts: int, state: dict, t0: float):
    """Log issued-calls and completions periodically so an ETA can be measured, not guessed."""
    last = 0
    while True:
        await asyncio.sleep(period)
        n, done = _CALLS["n"], state["done"]
        rate = (n - last) / (period / 60)
        last = n
        left = total_rollouts - done
        # every unfinished rollout needs at most (cap - rounds_done) more calls; with no
        # per-rollout visibility the honest bound is the remaining budget across them
        print(f"  .. {done}/{total_rollouts} rollouts, {n} calls issued, "
              f"{rate:.0f} calls/min, {left} running "
              f"(<= {left * 50 - (n - done * 8)} calls left, "
              f"<= {max(0, (left * 50 - (n - done * 8))) / max(rate, 1) / 60:.1f} h)",
              flush=True)


def _ck_key(pid: str, arm: str, k: int) -> str:
    return f"{pid}|{arm}|{k}"


async def checkpointed(coro, path: Path, pid: str, arm: str, k: int, lock,
                       keep_thinking: bool = False):
    """Run one rollout and append its result the moment it lands.

    Rollouts here are long (up to 50 rounds x ~80 s per LLM call) and mostly FAIL, so they
    burn the full budget -- a single terminal gather can outlive any wall-clock limit and
    return nothing at all. Writing per rollout means a killed run resumes instead of
    restarting."""
    res = await coro
    if not keep_thinking:
        for rd in res.get("rounds", []):
            rd.pop("thinking", None)
    async with lock:
        with path.open("a") as fh:
            fh.write(json.dumps({"key": _ck_key(pid, arm, k), "result": res}) + "\n")
    return res


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue          # a torn final line from a hard kill
            out[d["key"]] = d["result"]
    return out


def build_resources(game: str, root: Path) -> dict:
    rex = root / "rexpure" / f"{game}_s1"
    wc = root / "worldcoder" / f"{game}_s1/best_transition_wc_seed1.py"
    return {"prog": HGAMES[game][0],
            "perceive": compile_perceive((rex / "best_perception_rexpure_seed1.py").read_text()),
            "beliefs": (rex / "best_beliefs_rexpure_seed1.txt").read_text(),
            "rt": prt.ProgramRuntime(_clean_program(wc.read_text()), timeout_s=1.0),
            "verbs": list(HGAMES[game][2])}


def prepare(problems: list[dict], perceive) -> None:
    """Attach the fields `llm_rollout` / `wc_rollout` expect.  Curated problems start from a
    reset, so the prefix and both context lists are empty -- the transcript each round is
    the CURRENT state plus whatever the rollout has itself executed."""
    for p in problems:
        p["start_grid"] = gstr(p["start"])
        p["goal_grid"] = gstr(p["goal"])
        p["_prefix"] = []
        p["_ctx_raw"] = []
        p["_ctx_z"] = []
        p["_z_t"] = perceive(p["start_grid"])[0]
        p["_z_goal"] = perceive(p["goal_grid"])[0]
        p["_dims"] = (len(p["start"]), len(p["start"][0]))


def report(rows, llm, off_idx, elapsed, cost, args, done=0, total=0) -> str:
    def on1(s, a):
        return sum(r[a]["pass_rate"] for r in s) / len(s)

    def on5(s, a):
        return sum(r[a]["pass_any"] for r in s) / len(s)

    def off1(s, a):
        v = [off_idx[(r["game"], r["id"])][a]["pass_rate"] for r in s
             if (r["game"], r["id"]) in off_idx]
        return sum(v) / len(v) if v else float("nan")

    L = ["# Curated planning eval - ONLINE (receding horizon)", "",
         f"Planner: {llm.label} | rollout cap {args.max_actions} actions | plan cap = "
         f"{args.max_actions}-n each round | {ATTEMPTS} attempts per LLM arm | wc budget "
         f"{args.wc_budget} (deterministic, 1 attempt) | warm_start={args.warm_start} | no "
         f"pre-window history | {len(rows)} problems | {elapsed / 60:.0f} min | ${cost:.2f}",
         "",
         "Each round the model plans to the goal, ONLY the first action is executed, then it "
         "replans from the observed state; success = the goal frame is ever reached inside "
         "the action budget. `(off)` = the paired OPEN-LOOP result on the identical "
         "problems and attempt count. Scored per game, never pooled.", ""]

    L += ["## Per game: online pass@1 (offline pass@1)", "",
          "| game | n | raw | lmwm | wc | rand |", "|---|--:|--:|--:|--:|--:|"]
    for game in dict.fromkeys(r["game"] for r in rows):
        s = [r for r in rows if r["game"] == game]
        L.append(f"| {game} ({HGAMES[game][1]}) | {len(s)} | "
                 + " | ".join(f"{on1(s, a):.2f} ({off1(s, a):.2f})" for a in ARMS)
                 + f" | {sum(r['random_success'] for r in s) / len(s):.2f} |")

    L += ["", "## Per game: pass@any (any attempt so far succeeded)", "",
          "| game | n | raw | lmwm |", "|---|--:|--:|--:|"]
    for game in dict.fromkeys(r["game"] for r in rows):
        s = [r for r in rows if r["game"] == game]
        L.append(f"| {game} | {len(s)} | " + " | ".join(f"{on5(s, a):.2f}" for a in LLM_ARMS) + " |")

    L += ["", "## Per tier: online pass@1 (offline pass@1)", "",
          "| tier | n | raw | lmwm | wc |", "|---|--:|--:|--:|--:|"]
    for tier in TIERS:
        s = [r for r in rows if r["tier"] == tier]
        if s:
            L.append(f"| {tier} | {len(s)} | "
                     + " | ".join(f"{on1(s, a):.2f} ({off1(s, a):.2f})" for a in ARMS) + " |")

    L += ["", "## Per problem (online pass@1)", "",
          "| game | tier | id | h | raw | lmwm | wc |", "|---|---|---|--:|--:|--:|--:|"]
    for r in rows:
        L.append(f"| {r['game']} | {r['tier']} | `{r['id']}` | {r['h']} | "
                 + " | ".join(f"{r[a]['pass_rate']:.1f}" for a in ARMS) + " |")

    L += ["", "## Diagnostics", ""]
    for a in ARMS:
        att = [t for r in rows for t in r[a]["attempts"]]
        wins = [t for t in att if t["success"]]
        why = defaultdict(int)
        for t in att:
            if not t["success"]:
                why[t["failed_reason"] or "?"] += 1
        mean = sum(t["actions_used"] for t in wins) / len(wins) if wins else float("nan")
        L.append(f"- **{a}**: {len(wins)}/{len(att)} rollouts reached the goal; mean actions "
                 f"to goal (successes) {mean:.2f}; failures "
                 + ", ".join(f"{k} {v}" for k, v in sorted(why.items())))
    return "\n".join(L) + "\n"


def emit(problems, per, llm, off_idx, elapsed, a, done, total) -> None:
    """Write the report from however many attempts have landed.  Overwrites in place, so
    the newest files are always the best estimate available."""
    rows, cost = [], 0.0
    for p in problems:
        row = {k: p[k] for k in ("game", "id", "tier", "objective", "h", "n_decisions",
                                 "seed", "quiescent", "random_success")}
        for arm in ARMS:
            tries = per[(p["id"], arm)]
            if not tries:
                continue
            cost += sum(t["cost"] for t in tries)
            row[arm] = {"attempts": tries,
                        "pass_rate": sum(t["success"] for t in tries) / len(tries),
                        "pass_any": any(t["success"] for t in tries)}
        if all(arm in row for arm in ARMS):
            rows.append(row)
    if not rows:
        return
    md = report(rows, llm, off_idx, elapsed, cost, a, done, total)
    stem = Path(a.out)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(
        {"config": {"model": llm.model, "backend": llm.backend,
                    "max_actions": a.max_actions, "rollouts_done": done,
                    "rollouts_total": total, "attempts_planned": ATTEMPTS, "context_k": 0,
                    "wc_budget": a.wc_budget, "warm_start": a.warm_start},
         "rows": rows, "cost": cost}, indent=1))
    stem.with_suffix(".md").write_text(md)
    print(f"[{done}/{total}] wrote {stem}.md ({elapsed / 60:.0f} min, ${cost:.2f})",
          flush=True)


async def main_async(a):
    problems = json.loads(Path(a.problems).read_text())
    if a.games:
        want = set(a.games.split(","))
        problems = [p for p in problems if p["game"] in want]
    if a.limit:
        per, keep = defaultdict(int), []
        for p in problems:
            if per[p["game"]] < a.limit:
                keep.append(p)
                per[p["game"]] += 1
        problems = keep
    llm = resolve_llm_config(a)
    llm_sem = asyncio.Semaphore(a.concurrency)
    root = Path(a.artifact_root)
    eco.ARTIFACT_ROOT = root

    off_idx = {}
    if a.offline and Path(a.offline).exists():
        for r in json.loads(Path(a.offline).read_text())["rows"]:
            off_idx[(r["game"], r["id"])] = r

    by_game: dict[str, list] = defaultdict(list)
    for p in problems:
        by_game[p["game"]].append(p)

    t0 = time.time()
    ckpt_path = Path(a.out).with_suffix(".ckpt.jsonl")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    have = load_checkpoint(ckpt_path) if a.resume else {}
    lock = asyncio.Lock()

    resources = {}
    for game, ps in by_game.items():
        resources[game] = build_resources(game, root)
        prepare(ps, resources[game]["perceive"])

    def make(p, arm, k):
        R = resources[p["game"]]
        return (wc_rollout(p, R["rt"], R["verbs"], R["prog"], a) if arm == "wc"
                else llm_rollout(arm, p, R["prog"], R["perceive"], R["beliefs"],
                                 llm, llm_sem, a))

    # FLAT schedule, with the report re-emitted as results land.
    #
    # A per-attempt WAVE schedule was tried and is much worse: a wave holds only ~60
    # rollouts, each rollout is sequential (round N+1 needs round N executed), so a rollout
    # can only ever have ONE call in flight.  As a wave drains, concurrency collapses onto
    # its hardest problems.  Measured on wave 1: 1938 rounds = 46 min of work, 572 min of
    # wall clock -- 92% idle at the barrier.
    #
    # Launching everything at once keeps all remaining rollouts live, so the semaphore stays
    # the binding constraint instead of the wave size.  Emitting after each completion gives
    # the same continuously-improving report the waves were for, at no scheduling cost --
    # every problem already has a wave-1 result checkpointed, so every emit is complete.
    per: dict[tuple[str, str], list] = defaultdict(list)
    todo = []
    for p in problems:
        for arm, k in [(x, i) for x in LLM_ARMS for i in range(ATTEMPTS)] + [("wc", 0)]:
            key = _ck_key(p["id"], arm, k)
            if key in have:
                per[(p["id"], arm)].append(have[key])
            else:
                todo.append((p, arm, k))

    state = {"done": 0}

    async def run_one(p, arm, k):
        res = await checkpointed(make(p, arm, k), ckpt_path, p["id"], arm, k, lock,
                                 a.keep_thinking)
        per[(p["id"], arm)].append(res)
        state["done"] += 1
        if state["done"] % a.emit_every == 0 or state["done"] == len(todo):
            emit(problems, per, llm, off_idx, time.time() - t0, a,
                 state["done"], len(todo))
        return res

    eco.llm_call = count_calls(eco.llm_call)
    hb = asyncio.create_task(heartbeat(a.heartbeat, len(todo), state, t0)) \
        if a.heartbeat else None
    print(f"{len(todo)} rollouts to run ({sum(len(v) for v in per.values())} resumed), "
          f"concurrency {a.concurrency}, report every {a.emit_every}", flush=True)
    emit(problems, per, llm, off_idx, time.time() - t0, a, 0, len(todo))
    try:
        await asyncio.gather(*(run_one(p, arm, k) for (p, arm, k) in todo))
    finally:
        if hb:
            hb.cancel()

    for R in resources.values():
        R["rt"].close()
    print(f"all {ATTEMPTS} waves complete")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="logs/2026-08-18/curated/problems.json")
    ap.add_argument("--offline", default="logs/2026-08-18/curated/eval/offline.json")
    ap.add_argument("--out", default="logs/2026-08-18/curated/eval/online")
    ap.add_argument("--games", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--artifact-root", default=str(ecp.ARTIFACT_ROOT))
    # 16 matched the reference run; raised because these rollouts are far longer than the
    # coverage ones and wall-clock, not the provider, is the binding constraint. Affects
    # scheduling only -- never what the model sees.
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--heartbeat", type=int, default=120,
                    help="seconds between in-flight progress lines (0 disables)")
    ap.add_argument("--emit-every", type=int, default=5,
                    help="rewrite the report every N completed rollouts")
    ap.add_argument("--max-actions", type=int, default=PLAN_CAP)
    ap.add_argument("--wc-budget", type=int, default=WC_BUDGET)
    ap.add_argument("--warm-start", action=argparse.BooleanOptionalAction, default=False)
    # ON by default: without it `_trace()` returns {} and no justification is persisted at
    # all, which is how the first seed shipped with trajectories but no reasoning. The
    # provider's raw hidden chain-of-thought is 95-99% of the output tokens (~10k/call) and
    # would balloon the JSON, so it is dropped unless --keep-thinking.
    ap.add_argument("--reasoning-trace", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--keep-thinking", action=argparse.BooleanOptionalAction, default=False,
                    help="also persist the provider's hidden reasoning tokens (very large)")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--policy", default="logged")
    ap.add_argument("--llm-backend", choices=("claude", "openrouter"), default="openrouter")
    ap.add_argument("--llm-url", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--provider-order", default=None)
    ecp.add_llm_tuning_args(ap)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
