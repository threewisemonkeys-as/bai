#!/usr/bin/env python3
"""Online (receding-horizon / MPC) planning eval on the coverage problem set.

Closed-loop counterpart of eval_coverage_plan.py, run on the SAME problems
(logs/coverage_plan_problems.json) so every arm x problem is paired with its
open-loop (offline) result. The protocol, per problem and arm:

    branch = env prefix-replayed to the window's step t (persistent, study seed)
    for n in 0 .. max_actions-1:                 # rollout capped at max_actions (20)
        the model plans a sequence of AT MOST (max_actions - n) actions to the
        GOAL from the CURRENT state (K pre-window context steps + every executed
        step, as one uniform history ending at CURRENT -- the horizon h is NEVER
        disclosed, identical to the offline coverage prompt);
        execute ONLY the first action in the engine; the resulting state joins
        the history; success iff the rendered grid EVER equals the goal grid.
    fail when the budget is spent, a plan is unusable twice, or the env
    terminates before the goal.

Same three arms as offline:
  raw   raw grids + goal grid (LLM).
  lmwm  rexpure perception features + learned beliefs + goal features (LLM).
  wc    worldcoder program searched each round from the OBSERVED grid, first
        action executed (NO LLM). Re-search is skipped only when the program
        predicted the just-executed transition exactly (true == T-hat) -- then
        the deterministic world makes the carried remainder identical to a fresh
        search, so this is a pure compute saving, not a warm start. When the
        program mispredicts, it re-searches from the true state: that is the only
        place closed-loop feedback can help wc (an imperfect program corrects).

Success is judged on raw engine grids (as offline). The plan cap counts down as
(max_actions - actions_taken), exactly the user's "20 - n" rule. Cold-start by
default (each round is an independent offline problem); --warm-start carries the
previous plan's unexecuted remainder as an MPC candidate (prior work: cold-start
replanning dithers on hidden-latent games).

    # OpenRouter deepseek (paired with the offline deepseek run):
    uv run python offline_learning/scripts/eval_coverage_online.py

    # Claude Sonnet via the local proxy (terminal 1: claude_cli_proxy.py):
    uv run python offline_learning/scripts/eval_coverage_online.py --llm-backend claude

    # engine/loop smoke test, no LLM (logged policy must score high on act):
    uv run python offline_learning/scripts/eval_coverage_online.py --no-llm --policy logged --limit 4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
OFF = HERE.parent
REPO = OFF.parent
for _p in (str(OFF), str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import program_runtime as prt
from autumn_env import AutumnBenchEnvWrapper
from validate import _parse_tag
from worldcoder_optimize import _clean_program
from offline_learning.coverage_plan import load_coverage, random_plan
from offline_learning.human_replay import GAMES as HGAMES, _grid, _obs_cell

import eval_coverage_plan as ecp
from eval_coverage_plan import (
    ARMS, BUCKETS, CONTEXT_K, DEFAULT_KNOWLEDGE, LLMConfig, PLAN_RAW_TMPL,
    PLAN_WIN_TMPL, WC_BEAM, WC_BUDGET, _base, _succ, cell, feat_transcript,
    llm_call, parse_plan, raw_transcript, resolve_llm_config, thinking_record,
)

MAX_ACTIONS = 20  # rollout cap; also the round-0 plan cap (== offline PLAN_CAP)
# Learned-artifact root; --artifact-root repoints both learned arms at another
# training run without touching the problem set or the raw arm.
ARTIFACT_ROOT = REPO / "logs/2026-08-11/human_unified"

# All engine work funnels through one thread: env instances stay serialized (no
# interpreter thread-safety assumptions) while the event loop stays free for LLM
# I/O. Program (wc) searches run on their own single worker so a multi-second
# search never blocks another job's ~1.5 ms env steps; one shared rt per game,
# so serializing them also matches the offline eval's sequential wc.
ENGINE = ThreadPoolExecutor(max_workers=1)
WC_ENGINE = ThreadPoolExecutor(max_workers=1)


async def engine(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(ENGINE, partial(fn, *args))


async def wc_engine(fn, *args, **kw):
    return await asyncio.get_running_loop().run_in_executor(
        WC_ENGINE, partial(fn, *args, **kw))


RETRY_SUFFIX = """

Your previous response was unusable ({error}). Respond again in EXACTLY the
required format: <reasoning>...</reasoning> then <plan> with one fully-specified
action per line (up, down, left, right, noop, or click ROW COL), AT MOST
{remaining} line(s), nothing else."""

WARM_TMPL = """=== CANDIDATE PLAN (proposed one step earlier; may be stale) ===
{cand}
=== END CANDIDATE PLAN ===
First check the candidate against the CURRENT state: if executing it in order
still reaches the GOAL exactly, return it unchanged; revise or replace it only
if it no longer does. Your plan must have at most {remaining} action(s).
"""


class Branch:
    """Persistent env branched off a coverage drive at step t (row-major clicks,
    executed straight through the wrapper). Every method must
    run on the ENGINE executor."""

    def __init__(self, prog: str, seed: int, prefix: list[str], budget: int):
        self.env = AutumnBenchEnvWrapper(
            env_name=prog, task_type="interactive",
            max_episode_steps=len(prefix) + budget + 8, seed=seed, render_mode="text")
        self.obs, _info = self.env.reset(seed=seed)
        self.terminated = False
        for a in prefix:
            self.obs, _r, term, _t, _i = self.env.step(a)
            if term:
                self.terminated = True
                break

    def step(self, action: str) -> None:
        self.obs, _r, term, _t, _i = self.env.step(action)
        self.terminated = bool(term)

    def grid(self) -> str:
        return _grid(_obs_cell(self.obs))

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass


def compile_perceive(code: str):
    """Compile the perception module once; return perceive(grid_str)->(feat,err)."""
    if not code.strip():
        return lambda _g: ("", None)
    ns: dict = {}
    exec(code, ns)  # noqa: S102 - trusted learned artifact
    fn = ns.get("perceive")
    if not callable(fn):
        return lambda _g: ("", "no callable perceive()")

    def perceive(grid_str: str):
        try:
            out = fn([grid_str])
            return (out if isinstance(out, str) else str(out)), None
        except Exception as e:  # noqa: BLE001
            return "", f"{type(e).__name__}: {e}"

    return perceive


def _trace(store: bool, text: str, think: str) -> dict:
    """Per-round reasoning capture: `reasoning` = the model's visible <reasoning>
    block, `thinking` = the provider's hidden reasoning tokens (empty when not
    returned), capped by the shared rule in `eval_coverage_plan`. Returns {} when capture is
    off so old rows stay unchanged."""
    if not store:
        return {}
    return {"reasoning": _parse_tag(text, "reasoning") or None, **thinking_record(think)}


# --------------------------------------------------------------- LLM arm rollout
async def llm_rollout(arm: str, p: dict, prog: str, perceive, beliefs: str,
                      llm: LLMConfig, llm_sem, args) -> dict:
    seed, prefix, dims = p["seed"], p["_prefix"], p["_dims"]
    goal_grid, z_goal = p["goal_grid"], p["_z_goal"]
    budget = args.max_actions
    branch = await engine(Branch, prog, seed, prefix, budget)
    try:
        cur_grid = await engine(branch.grid)
        start_match = (cur_grid == p["start_grid"])
        hist_raw = list(p["_ctx_raw"])           # [(grid, action), ...]
        hist_z = list(p["_ctx_z"])
        cur_z = p["_z_t"]
        rounds: list[dict] = []
        cost, n = 0.0, 0
        success = False
        reached_at = None
        failed = None
        carry: list[str] = []
        while n < budget:
            remaining = budget - n
            if arm == "raw":
                prompt = PLAN_RAW_TMPL.format(
                    cap=remaining, default_knowledge=DEFAULT_KNOWLEDGE,
                    transcript=raw_transcript(hist_raw[-CONTEXT_K:], cur_grid), goal=goal_grid)
            else:
                prompt = PLAN_WIN_TMPL.format(
                    cap=remaining, default_knowledge=DEFAULT_KNOWLEDGE,
                    beliefs=beliefs.strip() or "(empty)",
                    transcript=feat_transcript(hist_z[-CONTEXT_K:], cur_z),
                    goal=z_goal or "(empty)")
            if args.warm_start and carry:
                warm = WARM_TMPL.format(cand="\n".join(carry), remaining=remaining)
                prompt = prompt.replace("\nRespond as:\n", f"\n{warm}\nRespond as:\n", 1)

            if args.no_llm:
                plan, perr, errs, text, think = scripted_plan(args, p, n), None, [], "", ""
            else:
                text, think, c1, errs = await llm_call(prompt, llm_sem, llm)
                cost += c1
                plan, perr = parse_plan(text, dims)
                if plan is not None and len(plan) > remaining:
                    plan, perr = None, f"budget-exceeded:{len(plan)}>{remaining}"
                if plan is None:                  # one corrective re-ask
                    fix = prompt + RETRY_SUFFIX.format(error=perr, remaining=remaining)
                    text, think, c2, errs2 = await llm_call(fix, llm_sem, llm)
                    cost += c2
                    errs = errs + errs2
                    plan, perr = parse_plan(text, dims)
                    if plan is not None and len(plan) > remaining:
                        plan, perr = None, f"budget-exceeded:{len(plan)}>{remaining}"

            trace = _trace(args.reasoning_trace, text, think)
            if plan is None:
                rounds.append({"n": n, "remaining": remaining, "plan": None,
                               "plan_error": perr, "retry_errors": errs, **trace})
                failed = "invalid-plan"
                break

            action = plan[0]
            carry = plan[1:]
            await engine(branch.step, action)
            new_grid = await engine(branch.grid)
            hist_raw.append((cur_grid, action))
            row = {"n": n, "remaining": remaining, "plan": plan, "executed": action,
                   "grid_after": new_grid, "terminated": branch.terminated,
                   "plan_error": None, "retry_errors": errs, **trace}
            if arm == "lmwm":
                z_after, z_err = perceive(new_grid)
                hist_z.append((cur_z, action))
                cur_z = z_after
                row["z_after"] = z_after
                row["z_error"] = z_err
            cur_grid = new_grid
            n += 1
            reached = (new_grid == goal_grid)
            row["reached_goal"] = reached
            rounds.append(row)
            if reached:
                success, reached_at = True, n
                break
            if branch.terminated:
                failed = "terminated"
                break
        if not success and failed is None:
            failed = "budget-exhausted"
    finally:
        await engine(branch.close)
    return {"success": success, "reached_at": reached_at, "actions_used": n,
            "failed_reason": failed, "cost": cost, "start_match": start_match,
            "rounds": rounds}


def scripted_plan(args, p: dict, n: int) -> list[str]:
    """LLM stand-in for --no-llm harness tests."""
    if args.policy == "logged":
        rest = p["gt_actions"][n:]
        return list(rest) if rest else ["noop"]
    rng = random.Random(f"{p['game']}:{p['seed']}:{p['t']}:{n}")
    return random_plan(p["game"], 1, p["_dims"], rng)


# ---------------------------------------------------------------- wc arm rollout
def _wc_round(rt, hist_prog, cur_g, goal_g, verbs, remaining, budget):
    """One program search from the observed grid; returns (plan_actions|None).

    allow_empty=False: a planning solution must be at least one action, even when
    goal==start (the whole `maintain` bucket). Success is scored on the grid AFTER an
    executed action, so a zero-length plan is not a solution here -- the search must
    find a >=1-step plan that HOLDS the goal (noop, at depth 1). Matches the offline
    eval and `parse_plan`, which already rejects an empty <plan> block.
    """
    universe = prt.build_action_universe(verbs, cur_g, goal_g)
    return prt.plan_search(rt, hist_prog, cur_g, goal_g, universe, remaining,
                           beam=WC_BEAM, node_budget=budget, context_k=CONTEXT_K,
                           allow_empty=False)


async def wc_rollout(p: dict, rt, verbs, prog: str, args) -> dict:
    seed, prefix = p["seed"], p["_prefix"]
    goal_grid = p["goal_grid"]
    goal_g = json.loads(goal_grid)
    goal_c = prt.canon_grid(goal_g)
    budget = args.max_actions
    branch = await engine(Branch, prog, seed, prefix, budget)
    try:
        cur_grid = await engine(branch.grid)
        start_match = (cur_grid == p["start_grid"])
        hist_prog = [(json.loads(g), prt.parse_action(a)) for g, a in p["_ctx_raw"]]
        rounds: list[dict] = []
        n, searches = 0, 0
        success = False
        reached_at = None
        failed = None
        carry: list = []          # remaining Action tuples from the last search
        predicted_c = None        # T-hat canon grid for the last executed action
        while n < budget:
            remaining = budget - n
            cur_g = json.loads(cur_grid)
            if carry and predicted_c is not None and prt.canon_grid(cur_g) == predicted_c:
                found, how = carry, "carry"     # program was locally exact -> reuse
            else:
                found = await wc_engine(_wc_round, rt, hist_prog, cur_g, goal_g,
                                        verbs, remaining, args.wc_budget)
                how, searches = "searched", searches + 1
            if not found:                       # None (or [], which is not a solution)
                rounds.append({"n": n, "remaining": remaining, "plan": None,
                               "how": how, "plan_error": "no-plan-found"})
                failed = "invalid-plan"
                break
            action = found[0]
            carry = list(found[1:])
            pred = prt.rollout(rt, hist_prog, cur_g, [action], context_k=CONTEXT_K)[0]
            predicted_c = prt.canon_grid(pred) if pred is not None else None
            astr = prt.unparse_action(action)
            await engine(branch.step, astr)
            new_grid = await engine(branch.grid)
            hist_prog.append((cur_g, action))
            cur_grid = new_grid
            n += 1
            reached = (new_grid == goal_grid)
            diverged = (predicted_c != prt.canon_grid(json.loads(new_grid)))
            rounds.append({"n": n, "remaining": remaining, "plan": [prt.unparse_action(a) for a in found],
                           "executed": astr, "how": how, "grid_after": new_grid,
                           "model_diverged": diverged, "terminated": branch.terminated,
                           "reached_goal": reached})
            if reached:
                success, reached_at = True, n
                break
            if branch.terminated:
                failed = "terminated"
                break
        if not success and failed is None:
            failed = "budget-exhausted"
    finally:
        await engine(branch.close)
    return {"success": success, "reached_at": reached_at, "actions_used": n,
            "failed_reason": failed, "cost": 0.0, "searches": searches,
            "start_match": start_match, "rounds": rounds}


# --------------------------------------------------------------- per-game setup
PKEY_FIELDS = ("game", "seed", "t", "bucket", "mechanic", "h")
BASE_FIELDS = ("game", "bucket", "mechanic", "kind", "h", "seed", "t", "synthetic",
               "random_success", "noop_success")
GAME_ORDER = ["bt3gb", "dq8gc", "n2ntd", "s2kt7", "83wkq"]


def pkey(p: dict) -> tuple:
    return tuple(p[f] for f in PKEY_FIELDS)


def build_resources(game: str) -> dict:
    cov = load_coverage(game)
    rex = ARTIFACT_ROOT / "rexpure" / f"{game}_s1"
    perc_code = (rex / "best_perception_rexpure_seed1.py").read_text()
    beliefs = (rex / "best_beliefs_rexpure_seed1.txt").read_text()
    perceive = compile_perceive(perc_code)
    wc_path = ARTIFACT_ROOT / "worldcoder" / f"{game}_s1/best_transition_wc_seed1.py"
    rt = prt.ProgramRuntime(_clean_program(wc_path.read_text()), timeout_s=1.0)
    return {"prog": cov["program"], "drives": cov["drives_by_seed"], "perceive": perceive,
            "beliefs": beliefs, "rt": rt, "verbs": [v for v in HGAMES[game][2]]}


def prepare_problems(problems: list[dict], drives: dict, perceive) -> None:
    """Attach per-problem context (prefix, K raw+feature history, goal features,
    dims) -- identical assembly to eval_coverage_plan.eval_game."""
    for p in problems:
        seed, t = p["seed"], p["t"]
        grids, acts = drives[seed]["grids"], drives[seed]["actions"]
        ctx = []
        for j in range(t - 1, max(-1, t - 1 - CONTEXT_K), -1):
            if grids[j] is None or not acts[j]:
                break
            ctx.insert(0, j)
        p["_prefix"] = acts[:t]
        p["_ctx_raw"] = [(grids[j], acts[j]) for j in ctx]
        p["_ctx_z"] = [(perceive(grids[j])[0], acts[j]) for j in ctx]
        p["_z_t"] = perceive(p["start_grid"])[0]
        p["_z_goal"] = perceive(p["goal_grid"])[0]
        g = json.loads(p["start_grid"])
        p["_dims"] = (len(g), len(g[0]))


async def run_arm(arm: str, p: dict, R: dict, llm: LLMConfig, llm_sem, args) -> dict:
    if arm == "wc":
        return await wc_rollout(p, R["rt"], R["verbs"], R["prog"], args)
    return await llm_rollout(arm, p, R["prog"], R["perceive"], R["beliefs"], llm, llm_sem, args)


# ------------------------------------------------------------------- reporting
def _online(rows, arm):
    v = [r[arm]["success"] for r in rows if arm in r]
    return sum(v) / len(v) if v else None


def _offline(rows, arm, offline_idx):
    v = []
    for r in rows:
        o = offline_idx.get((r["game"], r["seed"], r["t"], r["bucket"], r["mechanic"], r["h"]))
        if o and arm in o:
            v.append(o[arm]["success"])
    return sum(v) / len(v) if v else None


def report(all_rows, llm, offline_idx, args) -> str:
    L = ["# Coverage-anchored ONLINE (receding-horizon) planning eval\n",
         f"Planner: {llm.label} | rollout cap {args.max_actions} actions | plan cap = "
         f"{args.max_actions}-n each round | wc budget {args.wc_budget} | "
         f"warm_start={args.warm_start} | {len(all_rows)} problems.\n",
         "Each round the model plans to the goal, ONLY the first action is executed, "
         "then it replans from the observed state; success = goal grid ever reached "
         "within the action budget. `off` columns = the paired OPEN-LOOP (offline) "
         "success on the identical problems. Scored per bucket, NEVER pooled.\n"]

    L.append("## Per bucket: online (off) vs baselines\n")
    L.append("| bucket | n | raw on(off) | lmwm on(off) | wc on(off) | noop | random |")
    L.append("|---|--:|--:|--:|--:|--:|--:|")
    for b in BUCKETS:
        rs = [r for r in all_rows if r["bucket"] == b]
        if not rs:
            continue
        parts = []
        for arm in ARMS:
            parts.append(f"{cell(_online(rs, arm))} ({cell(_offline(rs, arm, offline_idx))})")
        L.append(f"| {b} | {len(rs)} | " + " | ".join(parts)
                 + f" | {cell(_base(rs, 'noop_success'))} | {cell(_base(rs, 'random_success'))} |")

    L.append("\n## Per bucket x horizon (online raw / lmwm / wc)\n")
    hs = sorted({r["h"] for r in all_rows})
    L.append("| bucket | " + " | ".join(f"h={h}" for h in hs) + " |")
    L.append("|---|" + "---|" * len(hs))
    for b in BUCKETS:
        cells = []
        for h in hs:
            rs = [r for r in all_rows if r["bucket"] == b and r["h"] == h]
            cells.append("/".join(cell(_online(rs, a)) for a in ARMS) if rs else " -- ")
        L.append(f"| {b} | " + " | ".join(cells) + " |")

    L.append("\n## Online vs offline lift, per arm (act bucket)\n")
    L.append("| game | raw off->on | lmwm off->on | wc off->on |")
    L.append("|---|--:|--:|--:|")
    for game in ["bt3gb", "dq8gc", "n2ntd", "s2kt7", "83wkq"]:
        grs = [r for r in all_rows if r["game"] == game and r["bucket"] == "act"]
        if not grs:
            continue
        parts = []
        for arm in ARMS:
            on, off = _online(grs, arm), _offline(grs, arm, offline_idx)
            parts.append(f"{cell(off)}->{cell(on)}" if off is not None else f"-- ->{cell(on)}")
        L.append(f"| {game} ({HGAMES[game][1]}) | " + " | ".join(parts) + " |")

    # diagnostics
    L.append("\n## Diagnostics\n")
    for arm in ARMS:
        rs = [r for r in all_rows if arm in r]
        succ = [r for r in rs if r[arm]["success"]]
        mean_used = sum(r[arm]["actions_used"] for r in succ) / len(succ) if succ else None
        inv = sum(r[arm]["failed_reason"] == "invalid-plan" for r in rs)
        term = sum(r[arm]["failed_reason"] == "terminated" for r in rs)
        L.append(f"- **{arm}**: mean actions to goal (successes) {cell(mean_used)}; "
                 f"invalid-plan {inv}; env-terminations {term}")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------- driver
async def main_async(args):
    llm = None if args.no_llm else resolve_llm_config(args)
    llm = llm or LLMConfig(backend="none", url="", model=args.policy)
    data = json.loads(Path(args.problems).read_text())
    problems = data["problems"]
    if args.games:
        problems = [p for p in problems if p["game"] in args.games]
    by_game: dict[str, list] = defaultdict(list)
    for p in problems:
        by_game[p["game"]].append(p)
    if args.limit:
        by_game = {g: ps[:args.limit] for g, ps in by_game.items()}
    games_present = [g for g in GAME_ORDER if g in by_game]

    offline_idx: dict[tuple, dict] = {}
    off_path = Path(args.offline)
    if off_path.exists():
        for res in json.loads(off_path.read_text()).get("results", []):
            for r in res["rows"]:
                offline_idx[(r["game"], r["seed"], r["t"], r["bucket"], r["mechanic"], r["h"])] = r

    out = Path(args.out or REPO / ("logs/coverage_online_eval"
               if args.llm_backend == "openrouter" and not args.no_llm
               else "logs/coverage_online_eval_" + (args.policy if args.no_llm else args.llm_backend)))
    out.parent.mkdir(parents=True, exist_ok=True)
    oj = out.with_suffix(".json")

    cfg = {"backend": llm.backend, "model": llm.model, "max_actions": args.max_actions,
           "wc_budget": args.wc_budget, "warm_start": args.warm_start, "arms": args.arms,
           "context_k": CONTEXT_K, "no_llm": args.no_llm, "policy": args.policy,
           "reasoning_trace": args.reasoning_trace, "offline": str(off_path)}

    # ---- resume: per-(problem, arm) granularity so an overnight crash costs
    # at most `checkpoint_every` jobs, not a whole (up to 134-problem) game.
    rows_by_key: dict[tuple, dict] = {}
    tot_cost = 0.0
    if args.resume and oj.exists():
        prior = json.loads(oj.read_text())
        pc = prior.get("config", {})
        if (pc.get("model"), pc.get("warm_start"), pc.get("max_actions"), pc.get("arms")) != \
           (llm.model, args.warm_start, args.max_actions, args.arms):
            raise ValueError(f"refusing to resume {oj} with a different config; use --no-resume or --out")
        for row in prior.get("rows", []):
            rows_by_key[tuple(row[f] for f in PKEY_FIELDS)] = row
        tot_cost = prior.get("cost", 0.0)
    done_pairs = {(k, arm) for k, row in rows_by_key.items()
                  for arm in args.arms if row.get(arm) is not None}

    # build resources + contexts, seed missing base rows, list outstanding jobs
    resources: dict[str, dict] = {}
    jobs: list[tuple] = []
    for g in games_present:
        R = build_resources(g)
        resources[g] = R
        prepare_problems(by_game[g], R["drives"], R["perceive"])
        for p in by_game[g]:
            k = pkey(p)
            rows_by_key.setdefault(k, {b: p[b] for b in BASE_FIELDS})
            for arm in args.arms:
                if (k, arm) not in done_pairs:
                    jobs.append((k, arm, p))

    total_pairs = sum(len(by_game[g]) for g in games_present) * len(args.arms)
    print(f"[online] {len(games_present)} games, {total_pairs} (problem,arm) pairs; "
          f"{len(done_pairs)} resumed, {len(jobs)} to run | model={llm.model} "
          f"| conc={args.concurrency} | cap={args.max_actions}", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    llm_sem = asyncio.Semaphore(args.concurrency * 4)
    t0 = time.time()

    def checkpoint():
        oj.write_text(json.dumps({"config": cfg, "cost": tot_cost,
                                  "elapsed_s": time.time() - t0,
                                  "rows": list(rows_by_key.values())}, indent=1))

    async def guarded(k, arm, p):
        async with sem:
            out_ = await run_arm(arm, p, resources[p["game"]], llm, llm_sem, args)
        return k, arm, out_

    tasks = [asyncio.create_task(guarded(*j)) for j in jobs]
    n_done = 0
    for fut in asyncio.as_completed(tasks):
        k, arm, out_ = await fut
        rows_by_key[k][arm] = out_
        tot_cost += out_.get("cost", 0.0)
        n_done += 1
        if n_done % args.checkpoint_every == 0 or n_done == len(tasks):
            checkpoint()
            all_rows = list(rows_by_key.values())
            done_now = [r for r in all_rows if all(r.get(a) is not None for a in args.arms)]
            live = " ".join(f"{a} {(_online(done_now, a) or 0):.2f}" for a in args.arms)
            print(f"[ckpt] {n_done}/{len(tasks)} jobs | full-rows {len(done_now)} | "
                  f"{live} | ${tot_cost:.3f} | {time.time()-t0:.0f}s", flush=True)
    for R in resources.values():
        R["rt"].close()
    checkpoint()

    all_rows = list(rows_by_key.values())
    md = report(all_rows, llm, offline_idx, args)
    out.with_suffix(".md").write_text(md)
    print(md, flush=True)
    print(f"\nTOTAL ${tot_cost:.3f} | {time.time()-t0:.0f}s | wrote {oj} and {out.with_suffix('.md')}",
          flush=True)


def main():
    global ARTIFACT_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default=str(REPO / "logs/coverage_plan_problems.json"))
    ap.add_argument("--offline", default=str(REPO / "logs/coverage_plan_eval.json"),
                    help="paired open-loop results for the off() columns")
    ap.add_argument("--out", default="", help="output stem (default: backend-specific)")
    ap.add_argument("--games", type=str, default="")
    ap.add_argument("--artifact-root", default=str(ARTIFACT_ROOT),
                    help="root holding rexpure/<game>_s1 and worldcoder/<game>_s1")
    ap.add_argument("--limit", type=int, default=0, help="cap problems per game")
    ap.add_argument("--arms", type=str, default="raw,lmwm,wc")
    ap.add_argument("--max-actions", type=int, default=MAX_ACTIONS)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--wc-budget", type=int, default=WC_BUDGET)
    ap.add_argument("--warm-start", action=argparse.BooleanOptionalAction, default=False,
                    help="carry the previous plan's remainder as an MPC candidate (default off)")
    ap.add_argument("--reasoning-trace", action=argparse.BooleanOptionalAction, default=True,
                    help="store per-round visible <reasoning> + hidden thinking tokens")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--checkpoint-every", type=int, default=20,
                    help="flush json every N completed (problem,arm) jobs (crash safety)")
    ap.add_argument("--no-llm", action="store_true", help="scripted policy instead of the LLM")
    ap.add_argument("--policy", choices=("logged", "random"), default="logged")
    ap.add_argument("--llm-backend", choices=("claude", "openrouter"), default="openrouter")
    ap.add_argument("--llm-url", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--provider-order", default=",".join(ecp.DEFAULT_PROVIDER_ORDER))
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    args = ap.parse_args()
    ARTIFACT_ROOT = Path(args.artifact_root)
    args.games = set(filter(None, args.games.split(",")))
    args.arms = [a for a in args.arms.split(",") if a]
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
