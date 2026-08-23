#!/usr/bin/env python3
"""Online (receding-horizon) goal-conditioned planning eval on test50 drives.

MPC-style counterpart of the offline planning task in eval_multistep_fd_plan.py,
run on the SAME windows (loaded from its results json, so every comparison is
paired). Every round is an INDEPENDENT offline planning problem — the exact
offline templates and transcript builders are reused verbatim. Per window/mode:

    branch = env prefix-replayed to step t (persistent for the window)
    for r in 0..h-1:
        prompt = offline plan prompt with horizon h-r: the K pre-window context
                 steps plus all executed steps as one uniform history ending at
                 the current state, the GOAL, "at most h-r actions"
        plan   = validated <plan> (one corrective re-ask on an unusable plan)
        execute plan[0] only; the resulting state joins the rolling history
        success iff the rendered grid ever equals the recorded goal grid
    fail when the budget is exhausted, the plan is unusable twice, or the env
    terminates before the goal.

The round-0 prompt is byte-identical to the offline prompt for that window; the
model is never told about the interactive protocol or which history actions
were its own. "GOAL recorded exactly h-r steps after CURRENT" stays temporally
true after any action sequence (the branch advances one engine tick per step).

Warm start (default on, --no-warm-start to ablate): rounds r>0 additionally see
the previous round's unexecuted plan remainder as a CANDIDATE PLAN block with
the instruction to keep it unless it no longer reaches the GOAL — the standard
MPC shifted-solution warm start. This reintroduces cross-round dependence by
design (cold-start re-solving measurably dithers: plan churn destroyed winning
round-0 plans, and hidden-latent games induced noop procrastination loops).

Success is always judged on raw engine grids (as offline); learned mode differs
only in what the model sees (P features of history/current/goal, computed live
for off-drive states via the same Observation cell format the drives record).

    uv run python offline_learning/scripts/eval_online_plan.py
    # harness tests without LLM calls:
    uv run python offline_learning/scripts/eval_online_plan.py --no-llm --policy logged
"""
from __future__ import annotations

import os as _bos, sys as _bsys  # offline_learning/ on sys.path (flat-import the kept libs)
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
import argparse
import asyncio
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import program_runtime as prt
import test50_sim_tools as T
from autumn_drive import _observation_cell
from clean_sweep import GAMES
from eval_multistep_fd_plan import (PLAN_RAW_TMPL, PLAN_WIN_TMPL, SeqSim,
                                    call_retry, feat_transcript, parse_plan,
                                    plan_timing, raw_transcript)
from eval_test50_idfd import evaluator_llm_call  # noqa: F401 - patches gepa_optimize._llm_call
from invdyn_core import DEFAULT_KNOWLEDGE
from rescore_test50_id_sim import grids_equal
from validate import make_config, run_perceive
from worldcoder_optimize import _clean_program

# All engine work funnels through one thread: instances stay serialized (no
# interpreter thread-safety assumptions) and the event loop stays free for LLM
# I/O. Steps cost ~1.5 ms, so one worker is nowhere near the bottleneck.
ENGINE = ThreadPoolExecutor(max_workers=1)


async def engine(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(ENGINE, partial(fn, *args))


RETRY_SUFFIX = """

Your previous response was unusable ({error}). Respond again in EXACTLY the
required format: <reasoning>...</reasoning> followed by <plan> containing one
fully-specified action per line (up, down, left, right, noop, or click ROW COL),
AT MOST {remaining} line(s), nothing else."""


# MPC-style warm start: rounds r>0 carry the previous plan's unexecuted
# remainder as a candidate. Inserted immediately before the "Respond as:"
# block of the (otherwise verbatim offline) prompt.
WARM_TMPL = """=== CANDIDATE PLAN (proposed one step earlier; may be stale) ===
{cand}
=== END CANDIDATE PLAN ===
First check the candidate against the CURRENT state: if executing it in order
still reaches the GOAL exactly, return it unchanged as your plan; revise or
replace it only if it no longer does. Your plan must have at most {remaining}
action(s).
"""


class Branch:
    """Persistent env branched off a verified drive at step t. All methods must
    be called on the ENGINE executor."""

    def __init__(self, sim: SeqSim, t: int, budget: int):
        # reset at the DRIVE's engine seed (generated drives are non-zero; recorded
        # drives are seed 0) -- same convention as SeqSim.run.
        self.env, self.obs = T.make_env(sim.game, max_steps=t + budget + 5,
                                        seed=sim.seed)
        self.terminated = False
        for k in range(t):
            self.obs, _, term, _, _ = self.env.step(sim.actions[k])
            if term:
                raise RuntimeError(f"terminated during prefix replay at step {k}")

    def step(self, action: str) -> None:
        # plan actions are canonical (row,col); de-canon for the engine. Prefix replay
        # in __init__ uses raw sim.actions (interpreter-conv) as-is.
        self.obs, _, term, _, _ = self.env.step(action)
        self.terminated = bool(term)

    def grid(self) -> str:
        return T.obs_grid(self.obs)

    def cell(self) -> str:
        return _observation_cell(self.obs)

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass


def program_round(rt: prt.ProgramRuntime, hist_prog: list, cur_grid: list,
                  goal: list, carry: list, remaining: int, game: str,
                  context_k: int, beam: int, node_budget: int):
    """One MPC round of the program mode (sync; run on an executor). Warm start:
    if the carried remainder still reaches the GOAL under a T-hat rollout
    (ever-hit, matching the online success criterion), adopt it unchanged --
    the program-native analog of WARM_TMPL. Else search over T-hat.
    Returns (plan | None, descriptor)."""
    goal_c = prt.canon_grid(goal)
    if carry:
        acts = [prt.parse_action(a) for a in carry[:remaining]]
        rolled = prt.rollout(rt, hist_prog, cur_grid, acts, context_k=context_k)
        if any(g is not None and prt.canon_grid(g) == goal_c for g in rolled):
            return list(carry[:remaining]), "carry-verified"
    verbs = [v for v in GAMES[game][0].split(",") if v]
    universe = prt.build_action_universe(verbs, cur_grid, goal)
    found = prt.plan_search(rt, hist_prog, cur_grid, goal, universe, remaining,
                            beam=beam, node_budget=node_budget,
                            context_k=context_k)
    if found is None:
        return None, "search-exhausted"
    return [prt.unparse_action(a) for a in found], "searched"


def scripted_plan(policy: str, win: dict, r: int, rng: random.Random,
                  game: str) -> list[str]:
    """LLM stand-ins for --no-llm harness tests."""
    if policy == "logged":
        return win["actions"][r:]
    verbs = [v for v in GAMES[game][0].split(",") if v]
    verb = rng.choice(verbs)
    if verb == "click":
        return [f"click {rng.randrange(win['dims'][0])} {rng.randrange(win['dims'][1])}"]
    return [verb]


async def run_job(game: str, win: dict, mode: str, sim: SeqSim, code: str,
                  beliefs: str, cfg, args, llm_sem, program_rt=None) -> dict:
    h, goal, dims = win["h"], win["goal_grid"], win["dims"]
    budget = args.max_rounds or h
    branch = await engine(Branch, sim, win["t"], budget)
    try:
        if not grids_equal(await engine(branch.grid), win["start_grid"]):
            raise RuntimeError(f"branch replay grid mismatch at t={win['t']}")
        rng = random.Random(f"online-{args.policy}:{game}:{win['drive']}:{win['t']}:{h}:{mode}")
        # Growing history of (state, outgoing action) pairs: the K pre-window
        # context steps plus every executed step, presented uniformly (executed
        # steps are indistinguishable from pre-window context).
        hist_raw = list(win["ctx_raw"])
        hist_z = list(win["z_ctx"])
        cur_grid, cur_z = win["start_grid"], win["z_t"]
        # program mode: parsed twins of the raw history/state for T-hat calls.
        goal_parsed = json.loads(goal) if mode == "program" else None
        hist_prog = ([(json.loads(g), prt.parse_action(a))
                      for g, a in win["ctx_raw"]] if mode == "program" else None)
        rounds, cost = [], 0.0
        carry: list[str] = []
        success = failed_reason = prompt_r0 = None
        for r in range(budget):
            remaining = budget - r
            if mode == "program":
                plan, how = await asyncio.get_running_loop().run_in_executor(
                    None, program_round, program_rt, hist_prog,
                    json.loads(cur_grid), goal_parsed, carry, remaining, game,
                    args.context_k, args.search_beam, args.search_budget)
                perr = None if plan else "no-plan-found"
                row = {"round": r, "remaining": remaining, "plan": plan,
                       "plan_error": perr, "retried": False, "retry_errors": [],
                       "response": f"({how}; {program_rt.n_calls} T-hat calls total)",
                       "candidate": list(carry) or None}
                if plan is None:
                    rounds.append(row)
                    failed_reason = "invalid-plan"
                    break
                action = plan[0]
                carry = plan[1:] if args.warm_start else []
                await engine(branch.step, action)
                grid = await engine(branch.grid)
                hist_prog.append((json.loads(cur_grid),
                                  prt.parse_action(action)))
                hist_raw.append((cur_grid, action))
                cur_grid = grid
                row.update(executed=action, grid_after=grid,
                           terminated=branch.terminated)
                reached = grids_equal(grid, goal)
                row["reached_goal"] = reached
                rounds.append(row)
                if reached:
                    success = True
                    break
                if branch.terminated:
                    failed_reason = "terminated"
                    break
                continue
            # Goal-timing countdown tracks the DATA (goal recorded h steps after
            # the window start), independent of the round budget. Past the
            # recorded moment (only reachable with --max-rounds > h) say so.
            timing = plan_timing(h - r) if h - r >= 1 else (
                f"The GOAL was recorded exactly {h} step(s) after the state where this\n"
                "planning episode began; that moment has passed, but the GOAL state may\n"
                "still be reachable")
            if mode == "raw":
                transcript = raw_transcript(hist_raw, cur_grid)
                prompt = PLAN_RAW_TMPL.format(
                    default_knowledge=DEFAULT_KNOWLEDGE, transcript=transcript,
                    goal=goal, cap=remaining, timing=timing)
            else:
                transcript = feat_transcript(hist_z, cur_z)
                prompt = PLAN_WIN_TMPL.format(
                    default_knowledge=DEFAULT_KNOWLEDGE,
                    beliefs=beliefs.strip() or "(empty)", transcript=transcript,
                    goal=win["z_goal"] or "(empty)", cap=remaining, timing=timing)
            if carry:
                warm = WARM_TMPL.format(cand="\n".join(carry), remaining=remaining)
                prompt = prompt.replace("\nRespond as:\n", f"\n{warm}\nRespond as:\n", 1)
            if r == 0:
                prompt_r0 = prompt

            retried, retry_errors, response = False, [], ""
            if args.no_llm:
                plan, perr = scripted_plan(args.policy, win, r, rng, game), None
            else:
                response, ccost, retry_errors = await call_retry(cfg, llm_sem, prompt, args.attempts)
                cost += ccost
                plan, perr = parse_plan(response, dims)
                if plan is not None and len(plan) > remaining:
                    plan, perr = None, f"budget-exceeded:{len(plan)}>{remaining}"
                if plan is None:
                    retried = True
                    fix = prompt + RETRY_SUFFIX.format(error=perr, remaining=remaining)
                    response, ccost, errs2 = await call_retry(cfg, llm_sem, fix, args.attempts)
                    cost += ccost
                    retry_errors += errs2
                    plan, perr = parse_plan(response, dims)
                    if plan is not None and len(plan) > remaining:
                        plan, perr = None, f"budget-exceeded:{len(plan)}>{remaining}"

            row = {"round": r, "remaining": remaining, "plan": plan,
                   "plan_error": perr, "retried": retried,
                   "retry_errors": retry_errors, "response": response,
                   "candidate": list(carry) or None}
            if args.no_llm:
                row["prompt"] = prompt
            if plan is None:
                rounds.append(row)
                failed_reason = "invalid-plan"
                break

            action = plan[0]
            carry = plan[1:] if args.warm_start else []
            await engine(branch.step, action)
            grid = await engine(branch.grid)
            hist_raw.append((cur_grid, action))
            cur_grid = grid
            row.update(executed=action, grid_after=grid,
                       terminated=branch.terminated)
            if mode == "learned":
                z_after, z_err = await engine(run_perceive, code,
                                              await engine(branch.cell))
                hist_z.append((cur_z, action))
                cur_z = z_after
                row.update(z_after=z_after, z_error=z_err)
            reached = grids_equal(grid, goal)
            row["reached_goal"] = reached
            rounds.append(row)
            if reached:
                success = True
                break
            if branch.terminated:
                failed_reason = "terminated"
                break
        if not success and failed_reason is None:
            failed_reason = "budget-exhausted"
    finally:
        await engine(branch.close)

    stable = total = 0
    for a, b in zip(rounds, rounds[1:]):
        if a.get("plan") and len(a["plan"]) >= 2 and b.get("plan"):
            total += 1
            stable += a["plan"][1] == b["plan"][0]
    adopted = sum(1 for rr in rounds if rr.get("candidate") and rr.get("plan") == rr["candidate"])
    cand_rounds = sum(1 for rr in rounds if rr.get("candidate"))
    return {
        "adopted_rounds": adopted, "candidate_rounds": cand_rounds,
        "h": h, "budget": budget, "drive": win["drive"], "t": win["t"], "mode": mode,
        "actions": win["actions"], "success": bool(success),
        "rounds_used": len(rounds) if success else None,
        "failed_reason": failed_reason,
        "offline_success": (win["offline"].get(mode) or {}).get("endpoint"),
        "offline_success_everhit": (win["offline"].get(mode) or {}).get("everhit"),
        "random_success": win.get("random_success"),
        "perception_error": win.get("perception_error"),
        "stable_pairs": stable, "total_pairs": total, "cost": cost,
        "prompt_r0": prompt_r0, "rounds": rounds,
    }


def prepare_windows(game: str, res_off: dict, sims: list[SeqSim], code: str,
                    args) -> tuple[list[dict], int]:
    """Rebuild eval-ready windows from the offline result's window records:
    recover sim_index, context steps, raw context, and perception features."""
    by_drive = {str(s.drive_csv): i for i, s in enumerate(sims)}
    # Offline endpoint success (grids_equal(final, goal)) plus ever-hit credit
    # (reached_at is not None) — the latter matches the online success criterion.
    offline = {}
    for row in res_off["plan_rows"]:
        offline.setdefault((row["drive"], row["t"], row["h"]), {})[row["mode"]] = {
            "endpoint": row["success"], "everhit": row["reached_at"] is not None}

    perc_cache: dict[tuple[int, int], tuple[str, str | None]] = {}

    def perceive(si: int, step: int):
        key = (si, step)
        if key not in perc_cache:
            perc_cache[key] = run_perceive(code, sims[si].obs[step])
        return perc_cache[key]

    windows, z_mismatch, per_h = [], 0, {}
    for rec in res_off["windows"]:
        h = rec["h"]
        if args.horizons and h not in args.horizons:
            continue
        if args.limit and per_h.get(h, 0) >= args.limit:
            continue
        si = by_drive[rec["drive"]]
        sim, t = sims[si], rec["t"]
        if not (grids_equal(sim.grids[t], rec["start_grid"])
                and grids_equal(sim.grids[t + h], rec["goal_grid"])):
            raise RuntimeError(f"{game}: drive/window grid mismatch at t={t}, h={h}")
        ctx = []
        for j in range(t - 1, max(-1, t - 1 - args.context_k), -1):
            if not sim.actions[j] or sim.grids[j] is None:
                break
            ctx.insert(0, j)
        win = dict(rec)
        win.update(sim_index=si, ctx_steps=ctx,
                   ctx_raw=[(sim.grids[j], sim.actions[j]) for j in ctx],
                   offline=offline.get((rec["drive"], t, h), {}))
        g = json.loads(rec["start_grid"])
        win["dims"] = (len(g), len(g[0]))
        z_ctx, err = [], None
        for j in ctx:
            z, e = perceive(si, j)
            err = err or e
            z_ctx.append((z, sim.actions[j]))
        z_t, e_t = perceive(si, t)
        z_goal, e_g = perceive(si, t + h)
        z_mismatch += (z_t != rec.get("z_t")) + (z_goal != rec.get("z_goal"))
        win.update(z_ctx=z_ctx, z_t=z_t, z_goal=z_goal,
                   perception_error=err or e_t or e_g)
        per_h[h] = per_h.get(h, 0) + 1
        windows.append(win)
    return windows, z_mismatch


async def eval_game_online(game: str, res_off: dict, cfg, args, seed: int) -> dict:
    artifact_dir = Path(res_off["artifact_dir"])
    code = (artifact_dir / f"best_perception_gepa_seed{seed}.py").read_text()
    beliefs_path = artifact_dir / f"best_beliefs_gepa_seed{seed}.txt"
    beliefs = beliefs_path.read_text() if beliefs_path.exists() else ""

    env_seed = res_off.get("env_seed") or 0
    if env_seed:
        # offline eval used FRESH generated drives (labels "gen_seed<N>", no CSV):
        # regenerate them deterministically from the recorded config -- same rng
        # stream and per-drive seeds as eval_multistep_fd_plan.eval_game.
        from eval_multistep_fd_plan import generate_drive
        grng = random.Random(f"gendrive:{env_seed}:{game}")
        sims = []
        for di in range(args.off_n_drives):
            dseed = env_seed + di
            rows = generate_drive(game, dseed, args.off_drive_length, grng)
            if len(rows) >= 2:
                sims.append(SeqSim(game, rows=rows, seed=dseed, label=f"gen_seed{dseed}"))
        labels = {str(s.drive_csv) for s in sims}
        if not set(res_off["drives"]) <= labels:
            raise RuntimeError(
                f"{game}: regenerated drives {sorted(labels)} do not cover "
                f"offline drives {res_off['drives']}")
    else:
        sims = [SeqSim(game, Path(p)) for p in res_off["drives"]]
        bad = {str(s.drive_csv): s.problems for s in sims if s.problems}
        if bad:
            raise RuntimeError(f"{game}: drive verification failed: {bad}")

    windows, z_mismatch = prepare_windows(game, res_off, sims, code, args)

    # Live-observation parity: replaying the drive prefix must reproduce the
    # recorded Observation cell byte-for-byte (else live perception would see a
    # different format than the P module was trained on).
    live_cell_matches = None
    if windows:
        probe = windows[0]
        br = await engine(Branch, sims[probe["sim_index"]], probe["t"], 0)
        live_cell_matches = (await engine(br.cell)) == sims[probe["sim_index"]].obs[probe["t"]]
        await engine(br.close)

    job_sem = asyncio.Semaphore(args.concurrency)
    llm_sem = asyncio.Semaphore(args.concurrency * 4)  # jobs already bound calls

    game_modes = list(args.modes)
    program_rt = None
    program_path = args.program_artifacts.get(game)
    if "program" in game_modes:
        if program_path is None:
            print(f"[warn] {game}: mode 'program' requested but no --program-artifact; "
                  "skipping that mode for this game", flush=True)
            game_modes = [m for m in game_modes if m != "program"]
        else:
            program_rt = prt.ProgramRuntime(
                _clean_program(Path(program_path).read_text()),
                timeout_s=args.program_timeout)

    async def guarded(win, mode):
        async with job_sem:
            return await run_job(game, win, mode, sims[win["sim_index"]],
                                 code, beliefs, cfg, args, llm_sem,
                                 program_rt=program_rt)

    try:
        jobs = await asyncio.gather(*(guarded(w, m) for w in windows for m in game_modes))
    finally:
        if program_rt is not None:
            program_rt.close()

    def summarize(rows: list[dict], wins: list[dict]) -> dict:
        succ = [j for j in rows if j["success"]]
        off = [j["offline_success"] for j in rows if j["offline_success"] is not None]
        offeh = [j["offline_success_everhit"] for j in rows
                 if j.get("offline_success_everhit") is not None]
        pairs = sum(j["total_pairs"] for j in rows)
        return {
            "n": len(rows),
            "success": len(succ) / len(rows) if rows else None,
            "offline_success": sum(off) / len(off) if off else None,
            "offline_everhit": sum(offeh) / len(offeh) if offeh else None,
            "random_success": (sum(w.get("random_success") or 0.0 for w in wins) / len(wins)) if wins else None,
            "mean_rounds_used": (sum(j["rounds_used"] for j in succ) / len(succ)) if succ else None,
            "invalid": sum(j["failed_reason"] == "invalid-plan" for j in rows),
            "terminated": sum(j["failed_reason"] == "terminated" for j in rows),
            "replan_stability": (sum(j["stable_pairs"] for j in rows) / pairs) if pairs else None,
            "warm_adoption": (sum(j.get("adopted_rounds", 0) for j in rows) /
                              sum(j.get("candidate_rounds", 0) for j in rows))
                             if sum(j.get("candidate_rounds", 0) for j in rows) else None,
        }

    summary = {}
    for h in sorted({w["h"] for w in windows}):
        wins_h = [w for w in windows if w["h"] == h]
        summary[str(h)] = {
            "windows": len(wins_h),
            "plan": {m: summarize([j for j in jobs if j["h"] == h and j["mode"] == m], wins_h)
                     for m in game_modes},
        }
    return {
        "game": game, "artifact_dir": str(artifact_dir),
        "program_artifact": str(program_path) if program_path else None,
        "drives": res_off["drives"], "z_mismatch": z_mismatch,
        "live_cell_matches": live_cell_matches,
        "cost": sum(j["cost"] for j in jobs), "summary": summary, "jobs": jobs,
    }


def render_report(payload: dict) -> str:
    horizons = payload["config"]["horizons"]
    modes = payload["config"]["modes"]
    hcols = "".join(f" h={h} |" for h in horizons)

    def cell(v):
        return "—" if v is None else f"{v:.2f}"

    def get(res, h, mode, key):
        s = res["summary"].get(str(h))
        return s["plan"][mode].get(key) if s and mode in s["plan"] else None

    lines = [
        "# Online (receding-horizon) planning on test50 source drives", "",
        "Same windows as the offline eval (paired); at each of up to h rounds the",
        "model plans to the goal, only its FIRST action is executed in the engine,",
        "and it replans from the observed result. Success iff the rendered grid",
        "ever equals the recorded goal within the h-step budget. `off` = offline",
        "(open-loop) success on the same windows rescored with the SAME ever-hit",
        "criterion (offline reached_at is not None); `rand` = offline random floor.", "",
    ]
    if payload["config"].get("max_rounds"):
        lines += [f"NOTE: fixed replanning budget = {payload['config']['max_rounds']} rounds at "
                  "EVERY horizon (h only sets how far ahead the goal was recorded).", ""]
    for mode in modes:
        lines += [f"## Planning success — {mode}", "",
                  f"| game | rand:{hcols} off:{hcols} online:{hcols}",
                  "|---|" + "---:|" * (3 * len(horizons))]
        for res in payload["results"]:
            row = f"| {res['game']} |"
            for key in ("random_success", "offline_everhit", "success"):
                for h in horizons:
                    row += f" {cell(get(res, h, mode, key))} |"
            lines.append(row)
        lines.append("")

    zblind = set(payload["config"]["zblind"])
    for label, keep in (("all games", lambda g: True),
                        (f"excluding z-blind {sorted(zblind)}", lambda g: g not in zblind)):
        lines += [f"## Averages over games ({label})", ""]
        for mode in modes:
            for key, name in (("random_success", "rand"), ("offline_everhit", "offline"),
                              ("success", "online")):
                vals = []
                for h in horizons:
                    per = [get(r, h, mode, key) for r in payload["results"] if keep(r["game"])]
                    per = [v for v in per if v is not None]
                    vals.append(sum(per) / len(per) if per else None)
                lines.append(f"- {mode} {name}: " +
                             ", ".join(f"h={h}: {cell(v)}" for h, v in zip(horizons, vals)))
        lines.append("")

    lines += ["## Online diagnostics (all games)", ""]
    for mode in modes:
        for key, name in (("mean_rounds_used", "mean rounds to goal (successes)"),
                          ("replan_stability", "replan stability (plan[1] kept as next plan[0])")):
            vals = []
            for h in horizons:
                per = [get(r, h, mode, key) for r in payload["results"]]
                per = [v for v in per if v is not None]
                vals.append(sum(per) / len(per) if per else None)
            lines.append(f"- {mode} {name}: " +
                         ", ".join(f"h={h}: {cell(v)}" for h, v in zip(horizons, vals)))
    inv = sum(get(r, h, m, "invalid") or 0 for r in payload["results"]
              for h in horizons for m in modes)
    term = sum(get(r, h, m, "terminated") or 0 for r in payload["results"]
               for h in horizons for m in modes)
    lines += ["", f"Invalid-plan failures (after one corrective re-ask): {inv}; "
                  f"env terminations before goal: {term}."]
    if payload.get("skipped"):
        lines += ["", "Skipped: " + ", ".join(f"{g} ({w})" for g, w in payload["skipped"].items())]
    return "\n".join(lines) + "\n"


async def main_async(args):
    off_payload = json.loads(args.plan_json.read_text())
    config_in = off_payload["config"]
    args.context_k = config_in["context_k"]
    seed = config_in["seed"]
    if not args.horizons:
        args.horizons = config_in["horizons"]
    cfg = make_config(config_in["task_model"], config_in["client"])
    # generated-drive reconstruction params (env_seed runs; see eval_game_online)
    args.off_n_drives = config_in.get("n_drives", 6)
    args.off_drive_length = config_in.get("drive_length", 40)
    off_results = {r["game"]: r for r in off_payload["results"]}
    games = [g for g in off_results if not args.games or g in args.games]

    args.program_artifacts = {}
    for spec in args.program_artifact or []:
        game, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--program-artifact expects game=path, got {spec!r}")
        args.program_artifacts[game] = path
    # fall back to the offline eval's recorded artifacts (paired by construction)
    for g, p in (config_in.get("program_artifacts") or {}).items():
        args.program_artifacts.setdefault(g, p)
    if "program" in args.modes and args.program_artifacts:
        print(f"[program] artifacts: {args.program_artifacts}", flush=True)

    out_json = args.out.with_suffix(".json")
    run_key = {"horizons": args.horizons, "limit": args.limit, "modes": args.modes,
               "policy": args.policy if args.no_llm else "llm",
               "warm_start": args.warm_start, "max_rounds": args.max_rounds,
               "program_artifacts": dict(sorted(args.program_artifacts.items())),
               "search_beam": args.search_beam, "search_budget": args.search_budget}
    results, skipped = [], {}
    if args.resume and out_json.exists():
        prior = json.loads(out_json.read_text())
        prior_key = {k: prior.get("config", {}).get(k) for k in run_key}
        if prior_key.get("max_rounds") is None:  # pre---max-rounds files: budget was h
            prior_key["max_rounds"] = 0
        if prior_key == run_key:
            results = prior.get("results", [])
    done = {r["game"] for r in results}

    config = {
        "plan_source": str(args.plan_json), "seed": seed,
        "context_k": args.context_k, **run_key,
        "task_model": config_in["task_model"], "client": config_in["client"],
        "zblind": args.zblind,
        "protocol": "receding-horizon with independent rounds: each round is the "
                    "offline plan prompt (verbatim templates) with horizon h-r and "
                    "one uniform history = K pre-window context steps + all executed "
                    "steps, ending at the current state; execute first action in "
                    "engine; success iff grid ever equals goal within h steps; one "
                    "corrective re-ask per unusable plan; warm_start additionally "
                    "shows rounds r>0 the previous plan's unexecuted remainder as a "
                    "CANDIDATE block (keep unless it no longer reaches the GOAL)",
    }
    started = time.time()
    for game in games:
        if game in done:
            print(f"[resume] {game}", flush=True)
            continue
        t0 = time.time()
        try:
            res = await eval_game_online(game, off_results[game], cfg, args, seed)
        except (RuntimeError, FileNotFoundError) as exc:
            skipped[game] = str(exc)
            print(f"[skip] {game}: {exc}", flush=True)
            continue
        results.append(res)
        parts = []
        for h in args.horizons:
            s = res["summary"].get(str(h))
            if not s:
                continue
            vals = "/".join(
                f"{s['plan'][m]['success']:.2f}" if s["plan"][m]["success"] is not None else "—"
                for m in args.modes)
            offs = "/".join(
                f"{s['plan'][m]['offline_everhit']:.2f}" if s["plan"][m].get("offline_everhit") is not None else "—"
                for m in args.modes)
            parts.append(f"h{h}: {vals} (off {offs})")
        print(f"[done] {game}: {' | '.join(parts)} ({'/'.join(args.modes)}, "
              f"{time.time() - t0:.0f}s, USD {res['cost']:.3f}, "
              f"cell_match={res['live_cell_matches']}, z_mismatch={res['z_mismatch']})",
              flush=True)
        payload = {"config": config, "elapsed_seconds": time.time() - started,
                   "results": results, "skipped": skipped}
        out_json.write_text(json.dumps(payload, indent=2))

    payload = {"config": config, "elapsed_seconds": time.time() - started,
               "results": results, "skipped": skipped}
    out_json.write_text(json.dumps(payload, indent=2))
    report = render_report(payload)
    args.out.with_suffix(".md").write_text(report)
    print(report, flush=True)
    print(f"wrote {out_json} and {args.out.with_suffix('.md')}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-json", type=Path,
                    default=REPO / "logs/multistep_fd_plan_eval.json",
                    help="offline eval results; supplies windows, artifacts, model config")
    ap.add_argument("--out", type=Path, default=REPO / "logs/online_plan_eval")
    ap.add_argument("--games", type=str, default="")
    ap.add_argument("--horizons", type=str, default="",
                    help="comma-separated subset; default = all in the offline json")
    ap.add_argument("--limit", type=int, default=0,
                    help="max windows per game per horizon (0 = all)")
    ap.add_argument("--modes", type=str, default="raw,learned")
    ap.add_argument("--concurrency", type=int, default=16,
                    help="concurrent (window, mode) jobs; <=1 LLM call in flight per job")
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--no-llm", action="store_true",
                    help="run the full loop with a scripted policy instead of the LLM")
    ap.add_argument("--policy", choices=("logged", "random"), default="logged",
                    help="scripted policy for --no-llm: logged must score 1.0 everywhere")
    ap.add_argument("--max-rounds", type=int, default=0,
                    help="fixed replanning budget for ALL horizons (0 = h rounds; "
                    "with a cap, h only dictates how far ahead the goal was recorded)")
    ap.add_argument("--warm-start", action=argparse.BooleanOptionalAction, default=True,
                    help="carry the previous round's unexecuted plan remainder into "
                         "the next prompt as a candidate (MPC warm start)")
    ap.add_argument("--zblind", type=str, default="aw9wd,nrdf6,s2kt7")
    ap.add_argument("--program-artifact", action="append", metavar="GAME=PATH",
                    help="WorldCoder program WM per game for --modes program "
                    "(default: the offline eval's recorded program_artifacts)")
    ap.add_argument("--program-timeout", type=float, default=1.0)
    ap.add_argument("--search-beam", type=int, default=64)
    ap.add_argument("--search-budget", type=int, default=5000,
                    help="max T-hat calls per plan_search round")
    args = ap.parse_args()
    args.games = set(filter(None, args.games.split(",")))
    args.horizons = [int(h) for h in args.horizons.split(",") if h]
    args.modes = [m for m in args.modes.split(",") if m]
    args.zblind = sorted(filter(None, args.zblind.split(",")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
