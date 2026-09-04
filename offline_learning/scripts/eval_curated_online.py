#!/usr/bin/env python3
"""ONLINE (receding-horizon / MPC) planning eval for planning-problems v2.

Closed-loop counterpart of eval_curated_plan.py, on the SAME problems so every arm x
problem x attempt pairs with its open-loop result. Per round: the model plans at most
(budget - n) actions to the goal from the CURRENT state, ONLY the first action is
executed, the observed state joins the history, and it replans.

The budget is a flat 50 actions by default. ``--cap-mode per-game|per-problem`` scales it
off each row's measured any-step reference reach instead (2x up to 10 actions, 1.5x above,
per-game taking the max over the game's rows), which makes a pass rate mean the same thing
on a 4-action row and a 40-action one. Scaled budgets need random floors measured at the
same budget -- the run refuses to start without them -- and a rollout's checkpoint key
carries its budget, so results from different budgets never resume into each other.

V2 semantics, shared with the offline evaluator by IMPORT (loader, goal configuration,
prompt templates via `build_prompt`, success-mode override):

  * the replay address is (seed, prefix): every rollout branches the env through the
    stored prefix and hard-checks that the replayed START matches before any paid call;
  * exact-frame rows score grid==goal; python rows score the registered checker on the
    growing executed trajectory; NL prompts show the sentence, never the diagnostic
    reference frame, and WorldCoder is not-applicable on python rows;
  * online scoring is inherently ANY-STEP (success = the goal first holds after an
    executed action). Quiescence-requiring checkers (reachable only via
    --goal-presentation nl) have the stability requirement waived and the row says so.
  * a game whose artifacts are missing has that arm skipped with a warning (F1), never
    a run-fatal exception.

Engine mechanics (Branch, executor threads, corrective re-ask, warm-start carry, the wc
per-round search + exact-prediction carry) are IMPORTED from eval_coverage_online -- the
same code the coverage and v1 curated runs used.

    uv run python offline_learning/scripts/eval_curated_online.py \
        --problems logs/2026-08-29/planning_v2/problems.json \
        --artifact-root logs/2026-08-24/human_curated \
        --goal-presentation frame \
        --offline logs/2026-08-29/planning_v2/eval/offline.json \
        --out logs/2026-08-29/planning_v2/eval/online
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter, defaultdict
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
from offline_learning.planning_nl_goals import freeze_grid, validate_problem_goal  # noqa: E402

import eval_coverage_plan as ecp  # noqa: E402
import eval_coverage_online as eco  # noqa: E402
from eval_coverage_online import (  # noqa: E402
    Branch, RETRY_SUFFIX, WARM_TMPL, _trace, compile_perceive, engine, wc_rollout,
)
from eval_coverage_plan import (  # noqa: E402
    CONTEXT_K, WC_BUDGET, llm_call, parse_plan, resolve_llm_config,
)
from eval_curated_plan import (  # noqa: E402
    ATTEMPTS, CAP_MODES, DEFAULT_ARMS, DEFAULT_ARTIFACT_ROOT, DEFAULT_PROBLEMS, LLM_ARMS,
    PLAN_CAP, TIERS, _cap_label, _floor, add_icl_args, apply_action_caps, build_prompt,
    gstr, icl_config, load_eval_problems, load_icl_block, select_goal_presentation,
)

ARMS = LLM_ARMS + ["wc"]
DEFAULT_OUT = REPO / "logs/2026-08-29/planning_v2/eval/online"

_CALLS = {"n": 0}


def count_calls(fn):
    async def wrapped(*args, **kw):
        _CALLS["n"] += 1
        return await fn(*args, **kw)
    return wrapped


async def heartbeat(period: int, total_rollouts: int, state: dict, t0: float):
    """Issued-calls + completions periodically, so an ETA can be measured, not guessed."""
    last = 0
    while True:
        await asyncio.sleep(period)
        n, done = _CALLS["n"], state["done"]
        rate = (n - last) / (period / 60)
        last = n
        left = total_rollouts - done
        print(f"  .. {done}/{total_rollouts} rollouts, {n} calls issued, "
              f"{rate:.0f} calls/min, {left} running", flush=True)


# ------------------------------------------------------------------ goal tests
def make_goal_test(p: dict):
    """Per-round success test on the executed trajectory; returns (test, waived).

    `test(frozen_grids, executed_actions, new_grid_str)` is called after each executed
    action with the trajectory INCLUDING that action's frame. Online scoring is any-step
    by construction; a quiescence-requiring checker has stability waived (`waived`=True)
    because checking it would require executing an extra probe action."""
    presentation = p["_eval_presentation"]
    if presentation == "frame":
        goal_grid = gstr(p["goal"])
        return (lambda grids, executed, new_grid: new_grid == goal_grid), False
    goal = p.get("_eval_python_goal") or validate_problem_goal(p)

    def test(grids, executed, new_grid):
        return bool(goal.check(grids, list(executed)))

    return test, bool(goal.require_quiescent)


# ------------------------------------------------------------------ LLM rollout
async def llm_rollout_v2(arm: str, p: dict, prog: str, perceive, beliefs: str,
                         llm, llm_sem, args, icl_block: str = "") -> dict:
    seed, prefix, dims = p["seed"], p["_prefix"], p["_dims"]
    budget = p["_eval_action_cap"]
    goal_test, quiescence_waived = make_goal_test(p)
    branch = await engine(Branch, prog, seed, prefix, budget)
    try:
        cur_grid = await engine(branch.grid)
        start_match = (cur_grid == p["start_grid"])
        if not start_match:
            # the preflight replays every problem before paid calls, so reaching this
            # means the engine drifted mid-run -- fail the rollout, not the run
            return {"success": False, "reached_at": None, "actions_used": 0,
                    "failed_reason": "start-mismatch", "cost": 0.0,
                    "start_match": False, "quiescence_waived": quiescence_waived,
                    "rounds": []}
        grids = [freeze_grid(p["start"])]
        executed: list[str] = []
        hist_raw: list[tuple[str, str]] = []
        hist_z: list[tuple[str, str]] = []
        cur_z = p["_z_t"]
        rounds: list[dict] = []
        cost, n = 0.0, 0
        success, reached_at, failed = False, None, None
        carry: list[str] = []
        while n < budget:
            remaining = budget - n
            prompt = build_prompt(
                p, arm, cur_grid, start_features=cur_z,
                goal_features=p["_z_goal"], beliefs=beliefs,
                hist_raw=hist_raw[-CONTEXT_K:], hist_z=hist_z[-CONTEXT_K:],
                cap=remaining, icl_block=icl_block,
            )
            if args.warm_start and carry:
                warm = WARM_TMPL.format(cand="\n".join(carry), remaining=remaining)
                prompt = prompt.replace("\nRespond as:\n", f"\n{warm}\nRespond as:\n", 1)

            sent_prompt = prompt
            text, think, c1, errs = await llm_call(sent_prompt, llm_sem, llm)
            cost += c1
            plan, perr = parse_plan(text, dims)
            if plan is not None and len(plan) > remaining:
                plan, perr = None, f"budget-exceeded:{len(plan)}>{remaining}"
            if plan is None:                  # one corrective re-ask
                fix = prompt + RETRY_SUFFIX.format(error=perr, remaining=remaining)
                sent_prompt = fix
                text, think, c2, errs2 = await llm_call(sent_prompt, llm_sem, llm)
                cost += c2
                errs = errs + errs2
                plan, perr = parse_plan(text, dims)
                if plan is not None and len(plan) > remaining:
                    plan, perr = None, f"budget-exceeded:{len(plan)}>{remaining}"

            trace = _trace(args.reasoning_trace, text, think)
            if plan is None:
                rounds.append({"n": n, "remaining": remaining, "plan": None,
                               "plan_error": perr, "retry_errors": errs,
                               "prompt": sent_prompt, "response": text, **trace})
                failed = "invalid-plan"
                break

            action = plan[0]
            carry = plan[1:]
            await engine(branch.step, action)
            new_grid = await engine(branch.grid)
            hist_raw.append((cur_grid, action))
            executed.append(action)
            grids.append(freeze_grid(json.loads(new_grid)))
            row = {"n": n, "remaining": remaining, "plan": plan, "executed": action,
                   "grid_after": new_grid, "terminated": branch.terminated,
                   "plan_error": None, "retry_errors": errs,
                   "prompt": sent_prompt, "response": text, **trace}
            if arm == "lmwm":
                z_after, z_err = perceive(new_grid)
                hist_z.append((cur_z, action))
                cur_z = z_after
                row["z_after"] = z_after
                row["z_error"] = z_err
            cur_grid = new_grid
            n += 1
            reached = goal_test(grids, executed, new_grid)
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
            "quiescence_waived": quiescence_waived, "rounds": rounds}


# ------------------------------------------------------- oracle preflight (free)
async def oracle_preflight(problems: list[dict], resources: dict) -> None:
    """Replay every reference plan through Branch + the online goal test before any
    paid call: proves prefix start-reproduction and goal wiring on the online path."""
    async def one(p):
        prog = resources[p["game"]]["prog"]
        goal_test, _w = make_goal_test(p)
        plan = list(p.get("_eval_oracle_plan", p["plan"]))
        branch = await engine(Branch, prog, p["seed"], p["_prefix"], len(plan) + 1)
        try:
            cur = await engine(branch.grid)
            if cur != p["start_grid"]:
                return f"{p['task_uid']}: replayed prefix does not reproduce START"
            grids = [freeze_grid(p["start"])]
            executed = []
            hit = False
            for a in plan:
                await engine(branch.step, a)
                g = await engine(branch.grid)
                executed.append(a)
                grids.append(freeze_grid(json.loads(g)))
                if goal_test(grids, executed, g):
                    hit = True
                    break
                if branch.terminated:
                    break
            if not hit:
                return f"{p['task_uid']}: reference plan never satisfies the goal online"
        finally:
            await engine(branch.close)
        return None

    failures = [f for f in await asyncio.gather(*(one(p) for p in problems)) if f]
    if failures:
        raise RuntimeError("online oracle preflight failed:\n  " + "\n  ".join(failures))
    print(f"online oracle preflight: {len(problems)} passed", flush=True)


# ---------------------------------------------------------------- checkpointing
def _ck_key(uid: str, arm: str, k: int, cap: int | None = None) -> str:
    """A rollout is only reusable under the budget it was run with, so a scaled cap is
    part of its identity. Flat-50 runs keep the original key, so their checkpoints
    (and the runs resuming from them) are unaffected."""
    return f"{uid}|{arm}|{k}" if cap is None else f"{uid}|{arm}|{k}|cap{cap}"


def _ck_cap(p: dict, args) -> int | None:
    """The cap to key a rollout by: None under the flat regime, the budget otherwise."""
    return None if args.cap_mode == "fixed" else p["_eval_action_cap"]


async def checkpointed(coro, path: Path, uid: str, arm: str, k: int, lock,
                       keep_thinking: bool = False, cap: int | None = None):
    res = await coro
    if not keep_thinking:
        for rd in res.get("rounds", []):
            rd.pop("thinking", None)
    async with lock:
        with path.open("a") as fh:
            fh.write(json.dumps({"key": _ck_key(uid, arm, k, cap), "result": res}) + "\n")
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


# ------------------------------------------------------------------- resources
def build_resources(game: str, root: Path, arms: list[str],
                    icl_cfg: dict | None = None) -> tuple[dict, dict]:
    """Load a game's artifacts; a missing artifact skips that arm with a warning."""
    skipped: dict[str, str] = {}
    R = {"prog": HGAMES[game][0], "verbs": list(HGAMES[game][2]),
         "perceive": compile_perceive(""), "beliefs": "", "rt": None,
         "icl": "", "icl_meta": {}}
    if "lmwm" in arms:
        rex = root / "rexpure" / f"{game}_s1"
        pp = rex / "best_perception_rexpure_seed1.py"
        bp = rex / "best_beliefs_rexpure_seed1.txt"
        missing = next((x for x in (pp, bp) if not x.is_file()), None)
        if missing is not None:
            skipped["lmwm"] = f"missing artifact: {missing}"
            print(f"WARNING: {game}: skipping arm lmwm -- {skipped['lmwm']}", flush=True)
        else:
            R["perceive"] = compile_perceive(pp.read_text())
            R["beliefs"] = bp.read_text()
    if "icl" in arms:
        block, meta = load_icl_block(game, root, icl_cfg)
        if not block:
            skipped["icl"] = f"no training pool: {meta.get('error')}"
            print(f"WARNING: {game}: skipping arm icl -- {skipped['icl']}", flush=True)
        else:
            R["icl"], R["icl_meta"] = block, meta
            print(f"[icl] {game}: {meta['n_transitions']} transitions, "
                  f"~{meta['est_tokens']} tokens ({meta['render']}"
                  + (f", ctx {meta['icl_context_k']}" if meta["icl_context_k"] else "")
                  + ")", flush=True)
    if "wc" in arms:
        wc = root / "worldcoder" / f"{game}_s1/best_transition_wc_seed1.py"
        if not wc.is_file():
            skipped["wc"] = f"missing WorldCoder artifact: {wc}"
            print(f"WARNING: {game}: skipping arm wc -- {skipped['wc']}", flush=True)
        else:
            R["rt"] = prt.ProgramRuntime(_clean_program(wc.read_text()), timeout_s=1.0)
    return R, skipped


def prepare(problems: list[dict], perceive) -> None:
    """Attach the fields the rollouts expect. V2 states are replay addresses: the
    stored prefix is threaded into every Branch, and exact rows perceive their goal
    frame while python rows show only the sentence (offline parity)."""
    for p in problems:
        exact = p["_eval_presentation"] == "frame"
        p["start_grid"] = gstr(p["start"])
        p["goal_grid"] = gstr(p["goal"]) if exact else ""
        p["_prefix"] = list(p["prefix"])
        p["_ctx_raw"] = []
        p["_ctx_z"] = []
        p["_z_t"] = perceive(p["start_grid"])[0]
        p["_z_goal"] = perceive(p["goal_grid"])[0] if exact else ""
        p["_dims"] = (len(p["start"]), len(p["start"][0]))


# --------------------------------------------------------------------- report
def report(rows, llm, off_idx, elapsed, cost, args, done=0, total=0) -> str:
    def cell(v, d=2):
        return "N/A" if v is None else f"{v:.{d}f}"

    def on1(s, a):
        v = [r[a]["pass_rate"] for r in s if a in r and r[a].get("pass_rate") is not None]
        return sum(v) / len(v) if v else None

    def on_any(s, a):
        v = [r[a]["pass_any"] for r in s if a in r and r[a].get("pass_any") is not None]
        return sum(v) / len(v) if v else None

    def off1(s, a):
        v = []
        for r in s:
            o = off_idx.get(r["task_uid"])
            if o and a in o and o[a].get("pass_rate") is not None:
                v.append(o[a]["pass_rate"])
        return sum(v) / len(v) if v else None

    presentation = args.goal_presentation
    waived = sum(1 for r in rows if r.get("quiescence_waived"))
    L = ["# Curated planning eval v2 - ONLINE (receding horizon)", "",
         f"Planner: {llm.label if llm else 'none'} | rollout cap {_cap_label(rows, args.cap_mode)} "
         f"actions | plan cap = cap-n each round | {args.attempts} attempts per "
         f"LLM arm | wc budget {args.wc_budget} (deterministic, 1 attempt) | "
         f"warm_start={args.warm_start} | {len(rows)} problems "
         f"({len(rows)} {presentation}) | "
         f"{elapsed / 60:.0f} min | ${cost:.2f}", "",
         "Each round the model plans to the goal, ONLY the first action is executed, then "
         "it replans from the observed state. Online scoring is ANY-STEP: success = the "
         "selected goal (exact frame, or the registered Python checker on the executed "
         "trajectory) first holds after an executed action, within the action budget."
         + (f" {waived} quiescence-requiring rows have stability waived online."
            if waived else ""), "",
         "`(off)` = the paired OPEN-LOOP result on the identical problems. `rand` is the "
         "cap-matched any-step random floor where recomputed, else the stored rand@h.", ""]

    games = list(dict.fromkeys(r["game"] for r in rows))
    hdr = lambda cols: ["| " + " | ".join(cols) + " |",
                        "|" + "|".join(["---"] + ["--:"] * (len(cols) - 1)) + "|"]
    L += ["## Per game: online pass@1 (offline pass@1)", ""] \
        + hdr(["game", "n"] + ARMS + ["rand"])
    for game in games:
        s = [r for r in rows if r["game"] == game]
        cells = []
        for a in ARMS:
            on, off = on1(s, a), off1(s, a)
            cells.append(f"{cell(on)}" + (f" ({cell(off)})" if off is not None else ""))
        L.append(f"| {game} ({HGAMES[game][1]}) | {len(s)} | " + " | ".join(cells)
                 + f" | {cell(sum(_floor(r) or 0 for r in s) / len(s))} |")

    L += ["", "## Per game: pass@any (any attempt succeeded)", ""] \
        + hdr(["game", "n"] + LLM_ARMS)
    for game in games:
        s = [r for r in rows if r["game"] == game]
        L.append(f"| {game} | {len(s)} | "
                 + " | ".join(cell(on_any(s, a)) for a in LLM_ARMS) + " |")

    L += ["", "## Per tier: online pass@1 (offline pass@1)", ""] \
        + hdr(["tier", "n"] + ARMS)
    for tier in TIERS:
        s = [r for r in rows if r["tier"] == tier]
        if s:
            cells = []
            for a in ARMS:
                on, off = on1(s, a), off1(s, a)
                cells.append(f"{cell(on)}" + (f" ({cell(off)})" if off is not None else ""))
            L.append(f"| {tier} | {len(s)} | " + " | ".join(cells) + " |")

    L += ["", "## Per problem (online pass@1)", ""] \
        + hdr(["task", "tier", "mode", "h", "prefix"] + ARMS + ["rand"])
    for r in rows:
        cells = [cell(r[a]["pass_rate"], 1) if a in r and r[a].get("pass_rate") is not None
                 else ("skip" if a in r and r[a].get("status", "").startswith("skipped")
                       else "N/A")
                 for a in ARMS]
        L.append(f"| `{r['task_uid']}` | {r['tier']} | {r['goal_presentation']} | {r['h']} | "
                 f"{r['prefix_len']} | " + " | ".join(cells) + f" | {cell(_floor(r))} |")

    L += ["", "## Diagnostics", ""]
    for a in ARMS:
        att = [t for r in rows if a in r for t in r[a].get("attempts", [])]
        if not att:
            continue
        wins = [t for t in att if t["success"]]
        why = defaultdict(int)
        for t in att:
            if not t["success"]:
                why[t.get("failed_reason") or "?"] += 1
        mean = sum(t["actions_used"] for t in wins) / len(wins) if wins else None
        L.append(f"- **{a}**: {len(wins)}/{len(att)} rollouts reached the goal; mean "
                 f"actions to goal (successes) {cell(mean)}; failures "
                 + (", ".join(f"{k} {v}" for k, v in sorted(why.items())) or "none"))
    return "\n".join(L) + "\n"


def emit(problems, per, skipped_by_game, llm, off_idx, elapsed, a, done, total) -> None:
    """Write report + JSON from however many attempts have landed (overwrites in place)."""
    rows, cost = [], 0.0
    for p in problems:
        presentation = p["_eval_presentation"]
        row = {k: p.get(k) for k in (
            "game", "id", "task_uid", "template_id", "tier", "objective", "h",
            "n_decisions", "seed", "source", "stochastic",
        )}
        row["goal_presentation"] = presentation
        row["eval_success_mode"] = p["_eval_success_mode"]
        row["eval_success_override"] = p.get("_eval_success_override")
        row["nl_goal"] = p.get("_eval_nl_goal", p["nl_goal"])
        row["random_success"] = p.get(f"{presentation}_random_success")
        row["random_trials"] = p.get(f"{presentation}_random_trials")
        row["random_success_cap50"] = p.get(f"{presentation}_random_success_cap50")
        row["random_trials_cap50"] = p.get(f"{presentation}_random_trials_cap50")
        # the floor actually comparable to this row's scores: measured at the budget
        # the rollouts ran under, whatever --cap-mode chose
        row["action_cap"] = p["_eval_action_cap"]
        row["random_floor"] = _floor(p)
        row["noop_success"] = p.get(f"{presentation}_noop_success")
        row["prefix_len"] = len(p["prefix"])
        row["quiescence_waived"] = bool(
            p["_eval_presentation"] == "nl"
            and (p.get("_eval_python_goal") or validate_problem_goal(p)).require_quiescent
        )
        got_any = False
        for arm in ARMS:
            reason = skipped_by_game.get(p["game"], {}).get(arm)
            if reason is not None:
                row[arm] = {"status": "skipped-missing-artifact", "reason": reason,
                            "attempts": [], "pass_rate": None, "pass_any": None}
                continue
            if arm == "wc" and presentation != "frame":
                row[arm] = {"status": "not-applicable",
                            "reason": "WorldCoder search requires an exact target frame",
                            "attempts": [], "pass_rate": None, "pass_any": None}
                continue
            tries = per.get((p["task_uid"], arm), [])
            if not tries:
                continue
            got_any = True
            cost += sum(t["cost"] for t in tries)
            row[arm] = {"status": "evaluated", "attempts": tries,
                        "pass_rate": sum(t["success"] for t in tries) / len(tries),
                        "pass_any": any(t["success"] for t in tries)}
        if got_any:
            rows.append(row)
    if not rows:
        return
    md = report(rows, llm, off_idx, elapsed, cost, a, done, total)
    stem = Path(a.out)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(
        {"config": {"model": llm.model if llm else None,
                    "backend": llm.backend if llm else None,
                    "problems": str(Path(a.problems)),
                    "artifact_root": a.artifact_root,
                    "goal_presentation": a.goal_presentation,
                    "success_mode": "online-any-step",
                    "max_actions": a.max_actions, "cap_mode": a.cap_mode,
                    "action_caps": {p["task_uid"]: p["_eval_action_cap"]
                                    for p in problems},
                    "rollouts_done": done,
                    "rollouts_total": total, "attempts_planned": a.attempts,
                    "context_k": CONTEXT_K, "wc_budget": a.wc_budget,
                    "warm_start": a.warm_start,
                    "max_floor": a.max_floor,
                    "arms": [x.strip() for x in a.arms.split(",") if x.strip()],
                    "icl": ({k: str(v) for k, v in icl_config(a).items()}
                            if "icl" in a.arms else None),
                    "skipped_arms": skipped_by_game},
         "rows": rows, "cost": cost}, indent=1))
    stem.with_suffix(".md").write_text(md)
    print(f"[{done}/{total}] wrote {stem}.md ({elapsed / 60:.0f} min, ${cost:.2f})",
          flush=True)


async def main_async(a):
    global llm_call
    meta, problems = load_eval_problems(a.problems)
    if a.games:
        want = {g.strip() for g in a.games.split(",") if g.strip()}
        unknown = sorted(want - {p["game"] for p in problems})
        if unknown:
            raise ValueError(f"requested games absent from input: {unknown}")
        problems = [p for p in problems if p["game"] in want]
    problems = select_goal_presentation(problems, a.goal_presentation, a.success_mode)
    caps = apply_action_caps(problems, a.cap_mode, a.max_actions)
    if a.cap_mode != "fixed":
        unmatched = [p["task_uid"] for p in problems if _floor(p) is None]
        if unmatched:
            raise ValueError(
                f"--cap-mode {a.cap_mode} needs random floors measured at the same "
                f"budget; {len(unmatched)} row(s) have none (run "
                f"recompute_random_floors.py --cap-mode {a.cap_mode}): "
                + ", ".join(unmatched[:6]) + (" ..." if len(unmatched) > 6 else ""))
        if a.cap_mode == "per-game":
            by_game = {p["game"]: caps[p["task_uid"]] for p in problems}
            detail = ", ".join(f"{g}={c}" for g, c in sorted(by_game.items()))
        else:
            detail = (f"{min(caps.values())}-{max(caps.values())} "
                      f"over {len(caps)} rows")
        print(f"action caps ({a.cap_mode}): {detail}", flush=True)
    excluded_saturated = []
    if a.max_floor is not None and a.max_floor >= 0:
        keep = []
        for p in problems:
            if (_floor(p) or 0.0) > a.max_floor:
                excluded_saturated.append(p["task_uid"])
            else:
                keep.append(p)
        problems = keep
        if excluded_saturated:
            print(f"excluding {len(excluded_saturated)} saturated problems "
                  f"(random floor > {a.max_floor}): " + ", ".join(excluded_saturated),
                  flush=True)
    if a.limit:
        per_game, keep = defaultdict(int), []
        for p in problems:
            if per_game[p["game"]] < a.limit:
                keep.append(p)
                per_game[p["game"]] += 1
        problems = keep
    if not problems:
        raise ValueError("filters selected no planning problems")

    arms = list(dict.fromkeys(x.strip() for x in a.arms.split(",") if x.strip()))
    unknown_arms = sorted(set(arms) - set(ARMS))
    if not arms or unknown_arms:
        raise ValueError(f"invalid --arms {a.arms!r}; choose from {','.join(ARMS)}")

    root = Path(a.artifact_root)
    by_game: dict[str, list] = defaultdict(list)
    for p in problems:
        by_game[p["game"]].append(p)

    resources, skipped_by_game = {}, {}
    for game, ps in by_game.items():
        R, skipped = build_resources(game, root, arms, icl_config(a))
        resources[game] = R
        if skipped:
            skipped_by_game[game] = skipped
        prepare(ps, R["perceive"])

    if a.oracle_preflight:
        await oracle_preflight(problems, resources)
    if a.oracle_only:
        print(json.dumps({
            "input": str(Path(a.problems)),
            "goal_presentation": a.goal_presentation,
            "selected": len(problems),
            "skipped_arms": skipped_by_game,
        }, indent=2))
        return

    llm_requested = any(arm in LLM_ARMS for arm in arms)
    llm = resolve_llm_config(a) if llm_requested else None
    llm_sem = asyncio.Semaphore(a.concurrency)

    off_idx = {}
    if a.offline and Path(a.offline).exists():
        off_payload = json.loads(Path(a.offline).read_text())
        off_presentation = off_payload.get("config", {}).get("goal_presentation")
        if off_presentation != a.goal_presentation:
            raise ValueError(
                f"offline results use goal_presentation={off_presentation!r}; "
                f"online evaluation requested {a.goal_presentation!r}"
            )
        for r in off_payload["rows"]:
            off_idx[r["task_uid"]] = r

    t0 = time.time()
    ckpt_path = Path(a.out).with_suffix(f".{a.goal_presentation}.ckpt.jsonl")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    have = load_checkpoint(ckpt_path) if a.resume else {}
    lock = asyncio.Lock()

    def runnable(p, arm):
        if arm in skipped_by_game.get(p["game"], {}):
            return False
        if arm == "wc" and p["_eval_presentation"] != "frame":
            return False
        return True

    def make(p, arm):
        R = resources[p["game"]]
        if arm == "wc":
            # wc_rollout is the shared coverage implementation and budgets off
            # args.max_actions; give it a view of args carrying THIS problem's cap
            wc_args = argparse.Namespace(**{**vars(a),
                                            "max_actions": p["_eval_action_cap"]})
            return wc_rollout(p, R["rt"], R["verbs"], R["prog"], wc_args)
        return llm_rollout_v2(arm, p, R["prog"], R["perceive"], R["beliefs"],
                              llm, llm_sem, a, R["icl"])

    # FLAT schedule (see the v1 module history: waves collapse concurrency onto the
    # hardest problems -- measured 92% idle); report re-emitted as results land.
    per: dict[tuple[str, str], list] = defaultdict(list)
    todo = []
    for p in problems:
        for arm, k in [(x, i) for x in LLM_ARMS if x in arms
                       for i in range(a.attempts)] + ([("wc", 0)] if "wc" in arms else []):
            if not runnable(p, arm):
                continue
            key = _ck_key(p["task_uid"], arm, k, _ck_cap(p, a))
            if key in have:
                per[(p["task_uid"], arm)].append(have[key])
            else:
                todo.append((p, arm, k))

    state = {"done": 0}

    async def run_one(p, arm, k):
        res = await checkpointed(make(p, arm), ckpt_path, p["task_uid"], arm, k, lock,
                                 a.keep_thinking, _ck_cap(p, a))
        per[(p["task_uid"], arm)].append(res)
        state["done"] += 1
        if state["done"] % a.emit_every == 0 or state["done"] == len(todo):
            emit(problems, per, skipped_by_game, llm, off_idx, time.time() - t0, a,
                 state["done"], len(todo))
        return res

    llm_call = count_calls(llm_call)      # llm_rollout_v2 calls this module's binding
    hb = asyncio.create_task(heartbeat(a.heartbeat, len(todo), state, t0)) \
        if a.heartbeat else None
    print(f"{len(todo)} rollouts to run ({sum(len(v) for v in per.values())} resumed), "
          f"concurrency {a.concurrency}, report every {a.emit_every}", flush=True)
    emit(problems, per, skipped_by_game, llm, off_idx, time.time() - t0, a, 0, len(todo))
    try:
        await asyncio.gather(*(run_one(p, arm, k) for (p, arm, k) in todo))
    finally:
        if hb:
            hb.cancel()

    for R in resources.values():
        if R["rt"] is not None:
            R["rt"].close()
    print("all rollouts complete")


def main():
    ap = argparse.ArgumentParser(
        description="Receding-horizon eval of planning-problems v2 from replayed states."
    )
    ap.add_argument("--problems", default=str(DEFAULT_PROBLEMS))
    ap.add_argument("--offline", default="",
                    help="offline eval JSON for the paired (off) columns")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    ap.add_argument("--games", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                    help=f"comma-separated subset of {','.join(ARMS)}; 'icl' is the raw "
                         "planner with the world model's training transitions in context "
                         "and is off by default")
    add_icl_args(ap)
    ap.add_argument("--attempts", type=int, default=ATTEMPTS,
                    help="LLM attempts per problem x arm (wc stays single-shot)")
    ap.add_argument(
        "--goal-presentation", choices=("frame", "nl"), required=True)
    ap.add_argument(
        "--max-floor", type=float, default=0.95,
        help="exclude problems whose random floor (cap-matched when present) exceeds "
        "this; negative disables")
    ap.add_argument(
        "--success-mode", choices=("any", "reference"), default="any",
        help="passed to the shared goal configuration; online scoring is any-step "
        "either way (see module docstring)")
    ap.add_argument("--oracle-preflight", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--oracle-only", action="store_true",
                    help="run the engine-only preflight and exit (no LLM calls)")
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--heartbeat", type=int, default=120)
    ap.add_argument("--emit-every", type=int, default=5)
    ap.add_argument("--max-actions", type=int, default=PLAN_CAP,
                    help="rollout action budget under --cap-mode fixed")
    ap.add_argument("--cap-mode", choices=CAP_MODES, default="fixed",
                    help="how the rollout action budget is set: 'fixed' is --max-actions "
                    "for every row; 'per-game' and 'per-problem' scale it off the "
                    "measured any-step reference reach (2x up to 10 actions, 1.5x above), "
                    "per-game taking the max over the game's rows. Scaled modes need "
                    "floors recomputed at the same budget")
    ap.add_argument("--wc-budget", type=int, default=WC_BUDGET)
    ap.add_argument("--warm-start", action=argparse.BooleanOptionalAction, default=True,
                    help="ON: the online-plan-eval finding -- cold-start replanning "
                    "churns; carrying the stale candidate fixes it")
    ap.add_argument("--reasoning-trace", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--keep-thinking", action=argparse.BooleanOptionalAction,
                    default=False)
    ap.add_argument("--llm-backend", choices=("claude", "openrouter"), default="openrouter")
    ap.add_argument("--llm-url", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--provider-order", default=None)
    ecp.add_llm_tuning_args(ap)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
