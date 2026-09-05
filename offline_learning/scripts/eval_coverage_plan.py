#!/usr/bin/env python3
"""Multi-horizon planning eval on the coverage-anchored problem set (3 labelled buckets).

Consumes `logs/coverage_plan_problems.json` (built by offline_learning/coverage_plan.py).
Three arms, each plan EXECUTED in the Autumn engine from the prefix-replayed state at the
problem's step t (per-window study seed); success iff the grid after the final action == goal:

  raw   : the eval model sees raw grids + the goal grid.
  lmwm  : rexpure perception features + learned beliefs (best_*_rexpure_seed1) + goal features.
  wc    : worldcoder program (best_transition_wc_seed1) searched with plan_search (NO LLM).

Planner: deepseek-v4-flash through OpenRouter by default, pinned to
deepseek,baidu,fireworks. Claude Sonnet through the local Claude CLI proxy is available with
``--llm-backend claude``. Plan cap = 20 for EVERY problem (the prompt says only "at most
20 actions" -- the horizon h is NOT disclosed to the model, so it cannot use the temporal
distance as a shortcut). wc node_budget = 5000, beam = 64, context_k = 9.

Scored per bucket x mechanic x horizon, NEVER pooled into one number; the always-noop and
random baselines (carried on each problem) are printed beside the arms so a wait/maintain
score can't masquerade as skill.

    uv run python offline_learning/scripts/eval_coverage_plan.py  # OpenRouter default

    # Claude: terminal 1
    uv run python offline_learning/scripts/claude_cli_proxy.py

    # Claude: terminal 2
    uv run python offline_learning/scripts/eval_coverage_plan.py --llm-backend claude
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
OFF = HERE.parent
REPO = OFF.parent
for _p in (str(OFF), str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import httpx
from dotenv import load_dotenv

import program_runtime as prt
from invdyn_core import DEFAULT_KNOWLEDGE, _tlabel
from validate import _parse_tag, run_perceive
from worldcoder_optimize import _clean_program
from offline_learning.coverage_plan import exec_plan, load_coverage, random_plan  # noqa
from offline_learning.human_replay import GAMES as HGAMES

load_dotenv()
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
CLAUDE_PROXY_URL = "http://127.0.0.1:8000/v1/chat/completions"
OPENROUTER_MODEL = "deepseek/deepseek-v4-flash"
CLAUDE_MODEL = "sonnet"
DEFAULT_PROVIDER_ORDER = ("deepseek", "baidu", "fireworks")
PLAN_CAP = 20
WC_BUDGET = 5000
WC_BEAM = 64
CONTEXT_K = 9
ACTION_RE = re.compile(r"^(?:up|down|left|right|noop|click \d{1,2} \d{1,2})$")
# Appended by every successful llm_call: {wall_s, provider, output_tokens, reasoning_tokens,
# prompt_tokens, cost, streamed, ttfb_s, max_gap_s, finish_reason}. Read by the eval scripts
# to report the planner's real p50 latency; ttfb_s/max_gap_s/finish_reason are the streaming
# health signals (slow start vs. mid-stream stall vs. an output-capped, truncated plan).
CALL_STATS: list[dict] = []


@dataclass(frozen=True)
class LLMConfig:
    backend: str
    url: str
    model: str
    provider_order: tuple[str, ...] = ()
    # `order` is a FAILOVER list: every call goes to order[0] and only falls through on
    # error, so it concentrates a fan-out on one host. `only` is a whitelist that still
    # load-balances, and `sort` routes by a metric -- those are the throughput knobs.
    provider_only: tuple[str, ...] = ()
    provider_sort: str = ""
    api_key: str | None = None
    # OpenRouter `reasoning` block as JSON (e.g. '{"effort": "low"}' or '{"enabled": false}').
    # Kept as a string so the config stays hashable/serializable.
    reasoning_json: str = ""
    # False pins the order exclusively; True lets OpenRouter route past it. The historical
    # default was True, which silently turns a dead pin into default routing.
    allow_fallbacks: bool = True

    @property
    def label(self) -> str:
        bits = [f"{self.model} via {self.backend}"]
        if self.provider_order:
            bits.append("@" + ",".join(self.provider_order)
                        + ("" if self.allow_fallbacks else " (exclusive)"))
        if self.reasoning_json:
            bits.append(f"reasoning={self.reasoning_json}")
        return " ".join(bits)


# ---- prompt templates (verbatim from eval_multistep_fd_plan) --------------------
PLAN_RAW_TMPL = """You control a grid environment and must reach a GOAL state.

=== DEFAULT KNOWLEDGE (always-true facts about this environment) ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

Below is a trajectory of consecutive RAW GRIDS in canonical JSON, ending at the
CURRENT grid. The action between each prior pair is shown. Use the whole history
to infer the dynamics (passive drift, momentum, selection, delayed effects).

{transcript}

=== GOAL raw grid ===
{goal}
=== END GOAL ===

Plan a sequence of AT MOST {cap} action(s) so that, starting from the CURRENT
state and executing your actions in order, the grid after your FINAL action is
EXACTLY the GOAL grid. The environment's passive dynamics keep running on every step
(including noop), so timing can matter. Every action must be fully specified:
one of up, down, left, right, noop, or click ROW COL with 0-indexed integers.

Respond as:
<reasoning>how your actions transform CURRENT into GOAL, step by step</reasoning>
<plan>
one action per line, at most {cap} line(s)
</plan>"""

PLAN_WIN_TMPL = """You control a grid environment and must reach a GOAL state.

=== DEFAULT KNOWLEDGE (always-true facts about this environment) ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== WORLD KNOWLEDGE (general facts; may be empty) ===
{beliefs}
=== END WORLD KNOWLEDGE ===

Below is a trajectory of consecutive states (summarized as features by a
perception module) ending at the CURRENT state, with the action taken between
each pair. Use the whole history to infer the dynamics (passive drift, momentum,
selection, delayed effects).

{transcript}

=== GOAL state features (same perception module) ===
{goal}
=== END GOAL ===

Plan a sequence of AT MOST {cap} action(s) so that, starting from the CURRENT
state and executing your actions in order, the state after your FINAL action is
EXACTLY the GOAL state. The environment's passive dynamics keep running on every step
(including noop), so timing can matter. Every action must be fully specified:
one of up, down, left, right, noop, or click ROW COL with 0-indexed integers.

Respond as:
<reasoning>how your actions transform CURRENT into GOAL, step by step</reasoning>
<plan>
one action per line, at most {cap} line(s)
</plan>"""


# Actions are stored row-major (`click ROW COL`) and executed straight through
# AutumnBenchEnvWrapper (which transposes internally to the native interpreter).
def numbered(actions: list[str]) -> str:
    return "\n".join(f"{i + 1}. {a}" for i, a in enumerate(actions))


def raw_transcript(ctx: list[tuple[str, str]], start: str) -> str:
    lines, n = [], len(ctx)
    for k, (grid, action) in enumerate(ctx):
        idx = -(n - k)
        lines.append(f"STATE[{_tlabel(idx)}] RAW GRID:\n{grid}")
        lines.append(f"  action({_tlabel(idx)} -> {_tlabel(idx + 1)}): {action}")
    lines.append(f"STATE[t] RAW GRID (CURRENT):\n{start}")
    return "\n".join(lines)


def feat_transcript(ctx: list[tuple[str, str]], z_t: str) -> str:
    lines, n = [], len(ctx)
    for k, (z, action) in enumerate(ctx):
        idx = -(n - k)
        lines.append(f"STATE[{_tlabel(idx)}] features:\n{z or '(empty)'}")
        lines.append(f"  action({_tlabel(idx)} -> {_tlabel(idx + 1)}): {action}")
    lines.append(f"STATE[t] (CURRENT) features:\n{z_t or '(empty)'}")
    return "\n".join(lines)


def parse_plan(text: str, dims: tuple[int, int]) -> tuple[list[str] | None, str | None]:
    body = _parse_tag(text, "plan")
    if body is None:
        return None, "no-plan-tag"
    plan = []
    for line in body.splitlines():
        s = re.sub(r"^(?:\d+[.)]|[-*])\s*", "", line.strip().strip("`")).strip()
        if not s:
            continue
        s = re.sub(r"\s+", " ", s.lower())
        if not ACTION_RE.match(s):
            return None, f"invalid-action:{s!r}"
        if s.startswith("click "):
            _, r, c = s.split()
            if int(r) >= dims[0] or int(c) >= dims[1]:
                return None, f"click-out-of-bounds:{s!r}"
        plan.append(s)
    if not plan:
        return None, "empty-plan"
    return plan, None


# ---- LLM (OpenAI-compatible chat-completions) -----------------------------------
# The planner streams, and not because anyone reads the tokens as they arrive: the
# deadline depends on it. A NON-streamed OpenRouter completion sends no bytes until
# generation finishes, so httpx's per-read timeout acts as a hard TOTAL deadline --
# the old flat `timeout=600` silently capped the planner at roughly 80k output tokens
# (decode measured at a constant ~133 tok/s on this route, independent of prompt size,
# so the wall is an output budget). Long-horizon rounds cross that, and a trip is not a
# slow call: llm_call returns "", parse_plan rejects it, and llm_rollout_v2 records
# failed_reason="invalid-plan" -- an infra zero indistinguishable from a planning failure
# in the results table. Under SSE the server emits `: OPENROUTER PROCESSING` keep-alives
# through prefill and a chunk per token after, so the read timeout becomes what it should
# have been all along: a stall detector, not a length limit.
#
# LLM_STALL_TIMEOUT_S  seconds of silence before a call is declared dead (default 180).
# LLM_TOTAL_TIMEOUT_S  overall wall cap so a runaway generation cannot hang a run
#                      forever; 0 disables it (default 1800).
# LLM_STREAM=0         fall back to the old non-streamed request.
STALL_TIMEOUT_S = float(os.environ.get("LLM_STALL_TIMEOUT_S", "180"))
TOTAL_TIMEOUT_S = float(os.environ.get("LLM_TOTAL_TIMEOUT_S", "1800"))


def _stream_enabled(llm: "LLMConfig") -> bool:
    # the local Claude CLI proxy answers with a single JSON body and speaks no SSE
    return llm.backend == "openrouter" and os.environ.get("LLM_STREAM", "1") != "0"


def _harvest(d: dict, acc: dict) -> None:
    """Fold one streamed chunk (or a whole non-streamed body) into the accumulator."""
    if d.get("error"):
        raise RuntimeError(str(d["error"]))
    acc["provider"] = d.get("provider") or acc["provider"]
    if d.get("usage"):
        acc["usage"] = d["usage"]
    for ch in d.get("choices") or []:
        part = ch.get("delta") or ch.get("message") or {}
        acc["content"].append(part.get("content") or "")
        acc["reasoning"].append(part.get("reasoning") or part.get("reasoning_content") or "")
        acc["finish"] = ch.get("finish_reason") or acc["finish"]


async def _stream_call(c: httpx.AsyncClient, llm: "LLMConfig", body: dict,
                       headers: dict) -> dict:
    acc = {"content": [], "reasoning": [], "usage": {}, "provider": None, "finish": None}
    t0 = time.time()
    ttfb, last, max_gap = None, t0, 0.0
    async with c.stream("POST", llm.url, json={**body, "stream": True},
                        headers=headers) as r:
        if r.status_code >= 400:
            detail = (await r.aread()).decode("utf-8", "replace")[:300]
            raise RuntimeError(f"HTTP {r.status_code}: {detail}")
        async for line in r.aiter_lines():
            now = time.time()
            if ttfb is None:
                ttfb = now - t0          # a slow START is a queued provider ...
            else:
                max_gap = max(max_gap, now - last)   # ... a slow MIDDLE is a stall
            last = now
            if not line.startswith("data: "):
                continue                     # `: OPENROUTER PROCESSING` keep-alive
            payload = line[6:]
            if payload.strip() == "[DONE]":
                break
            try:
                _harvest(json.loads(payload), acc)
            except json.JSONDecodeError:
                continue                     # a split frame; the next line completes it
    acc["ttfb_s"] = round(ttfb or 0.0, 1)
    acc["max_gap_s"] = round(max_gap, 1)
    return acc


async def _post_call(c: httpx.AsyncClient, llm: "LLMConfig", body: dict,
                     headers: dict) -> dict:
    acc = {"content": [], "reasoning": [], "usage": {}, "provider": None, "finish": None}
    r = await c.post(llm.url, json=body, headers=headers)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    _harvest(r.json(), acc)
    acc["ttfb_s"], acc["max_gap_s"] = None, None
    return acc


async def llm_call(prompt: str, sem: asyncio.Semaphore, llm: LLMConfig,
                   attempts: int = 4) -> tuple[str, str, float, list]:
    """Returns (content, reasoning, cost, errors). `reasoning` is the provider's
    hidden thinking (OpenRouter surfaces it as message.reasoning / reasoning_content
    for reasoning models, and as delta.reasoning when streamed); empty string when the
    provider doesn't return it."""
    body: dict = {"model": llm.model, "messages": [{"role": "user", "content": prompt}]}
    headers: dict[str, str] = {}
    if llm.backend == "openrouter":
        prov: dict = {}
        if llm.provider_order:
            prov["order"] = list(llm.provider_order)
            prov["allow_fallbacks"] = llm.allow_fallbacks
        if llm.provider_only:
            prov["only"] = list(llm.provider_only)
        if llm.provider_sort:
            prov["sort"] = llm.provider_sort
        if prov:
            body["provider"] = prov
        body["usage"] = {"include": True}
        if llm.reasoning_json:
            body["reasoning"] = json.loads(llm.reasoning_json)
    if llm.api_key:
        headers["Authorization"] = f"Bearer {llm.api_key}"
    streamed = _stream_enabled(llm)
    timeout = httpx.Timeout(STALL_TIMEOUT_S, connect=30.0)
    errors: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            async with sem:
                t_call = time.time()
                async with httpx.AsyncClient(timeout=timeout) as c:
                    run = (_stream_call if streamed else _post_call)(c, llm, body, headers)
                    try:
                        acc = await (asyncio.wait_for(run, TOTAL_TIMEOUT_S)
                                     if TOTAL_TIMEOUT_S > 0 else run)
                    except asyncio.TimeoutError:
                        # distinct from a ReadTimeout: the stream was alive and producing,
                        # it just never finished inside the overall cap
                        raise RuntimeError(
                            f"total timeout: no completion in {TOTAL_TIMEOUT_S:.0f}s "
                            f"(LLM_TOTAL_TIMEOUT_S)") from None
            text = "".join(acc["content"])
            reasoning = "".join(acc["reasoning"])
            usage = acc["usage"] or {}
            cost = float(usage.get("cost") or 0.0)
            if text.strip():
                # per-call telemetry: which endpoint actually served this, and how long it
                # took. Planner wall time is a serial chain of these, so a run that cannot
                # say what its p50 was cannot be tuned. ttfb_s/max_gap_s separate "the
                # provider is slow to start" from "the stream stalled mid-generation", and
                # finish_reason=="length" is a TRUNCATED plan, not a slow one.
                CALL_STATS.append({
                    "wall_s": time.time() - t_call, "provider": acc["provider"],
                    "output_tokens": usage.get("completion_tokens") or 0,
                    "reasoning_tokens": ((usage.get("completion_tokens_details") or {})
                                         .get("reasoning_tokens") or 0),
                    "prompt_tokens": usage.get("prompt_tokens") or 0, "cost": cost,
                    "streamed": streamed, "ttfb_s": acc["ttfb_s"],
                    "max_gap_s": acc["max_gap_s"], "finish_reason": acc["finish"]})
                return text, reasoning, cost, errors
            errors.append(f"attempt {attempt}: empty (finish={acc['finish']})")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
        if attempt < attempts:
            await asyncio.sleep(min(2 ** attempt, 8))
    return "", "", 0.0, errors


def transport_health(stats: list[dict], start: int = 0) -> str:
    """One line of LLM-transport health over CALL_STATS[start:], for the run heartbeat.

    A planner run can only be watched through its calls: p50/p90 wall says whether the
    route is degrading, ttfb separates a slow provider from a slow generation, gap catches
    a stream that stalled mid-answer, and finish_reason!="stop" is a TRUNCATED plan --
    which scores as a planning failure unless someone notices it was an output cap."""
    rows = stats[start:]
    if not rows:
        return "no completed calls yet"
    def q(key, p):
        vals = sorted(r[key] for r in rows if r.get(key) is not None)
        return vals[min(int(p * len(vals)), len(vals) - 1)] if vals else float("nan")
    trunc = sum(1 for r in rows if r.get("finish_reason") not in (None, "stop"))
    gap = max((r["max_gap_s"] for r in rows if r.get("max_gap_s") is not None), default=0.0)
    return (f"{len(rows)} calls: wall p50 {q('wall_s', .5):.0f}s p90 {q('wall_s', .9):.0f}s "
            f"max {q('wall_s', 1.0):.0f}s | ttfb p50 {q('ttfb_s', .5):.1f}s | "
            f"worst stream gap {gap:.1f}s | out p90 {q('output_tokens', .9):.0f} tok | "
            f"non-stop finish {trunc}")


# Hidden reasoning is stored capped, not dropped: on a reasoning model it IS the
# deliberation.  Measured on deepseek-v4-flash planning calls, the hidden trace runs past
# 8000 chars on every call while the model's own <reasoning> block averages a few hundred --
# so a log that keeps only the visible block has kept a post-hoc summary and thrown the
# reasoning away.  LLM_REASONING_CAP=0 disables the cap.
REASONING_CAP = int(os.environ.get("LLM_REASONING_CAP", "8000"))


def thinking_record(thinking: str) -> dict:
    """Standard fields for the provider's hidden chain on one logged LLM call.

    Every eval that persists an LLM call should splat this into its record, so the key name
    and the truncation rule are the same everywhere.  `thinking_chars` is the UNTRUNCATED
    length, so a capped record still says how much was dropped."""
    thinking = thinking or ""
    return {"thinking": thinking if REASONING_CAP <= 0 else thinking[:REASONING_CAP],
            "thinking_chars": len(thinking)}


def _reached(grids: list[str | None], goal: str) -> tuple[bool, int | None]:
    for j, g in enumerate(grids):
        if g == goal:
            return True, j + 1
    return False, None


# ---- per-game eval --------------------------------------------------------------
async def eval_game(game: str, problems: list[dict], concurrency: int, llm: LLMConfig) -> dict:
    cov = load_coverage(game)
    prog, drives = cov["program"], cov["drives_by_seed"]
    rex = ARTIFACT_ROOT / "rexpure" / f"{game}_s1"
    perc_code = (rex / "best_perception_rexpure_seed1.py").read_text()
    beliefs = (rex / "best_beliefs_rexpure_seed1.txt").read_text()
    wc_path = ARTIFACT_ROOT / "worldcoder" / f"{game}_s1/best_transition_wc_seed1.py"
    rt = prt.ProgramRuntime(_clean_program(wc_path.read_text()), timeout_s=1.0)
    verbs = HGAMES[game][2]

    pcache: dict[str, str] = {}

    def perceive(grid: str) -> str:
        if grid not in pcache:
            z, _e = run_perceive(perc_code, grid)
            pcache[grid] = z
        return pcache[grid]

    # assemble per-problem context + prompts
    for p in problems:
        seed, t, h = p["seed"], p["t"], p["h"]
        d = drives[seed]
        grids, acts = d["grids"], d["actions"]
        ctx = []
        for j in range(t - 1, max(-1, t - 1 - CONTEXT_K), -1):
            if grids[j] is None or not acts[j]:
                break
            ctx.insert(0, j)
        p["_prefix"] = acts[:t]
        p["_ctx_raw"] = [(grids[j], acts[j]) for j in ctx]
        p["_ctx_z"] = [(perceive(grids[j]), acts[j]) for j in ctx]
        p["_z_t"] = perceive(p["start_grid"])
        p["_z_goal"] = perceive(p["goal_grid"])
        g = json.loads(p["start_grid"])
        p["_dims"] = (len(g), len(g[0]))

    sem = asyncio.Semaphore(concurrency)

    async def one(p: dict):
        raw_prompt = PLAN_RAW_TMPL.format(
            cap=PLAN_CAP, default_knowledge=DEFAULT_KNOWLEDGE,
            transcript=raw_transcript(p["_ctx_raw"], p["start_grid"]), goal=p["goal_grid"])
        lmwm_prompt = PLAN_WIN_TMPL.format(
            cap=PLAN_CAP, default_knowledge=DEFAULT_KNOWLEDGE,
            beliefs=beliefs.strip() or "(empty)",
            transcript=feat_transcript(p["_ctx_z"], p["_z_t"]), goal=p["_z_goal"] or "(empty)")
        (rt_txt, rt_think, rc, re_), (lt_txt, lt_think, lc, le_) = await asyncio.gather(
            llm_call(raw_prompt, sem, llm), llm_call(lmwm_prompt, sem, llm))
        return p, {"raw": (rt_txt, rt_think, rc, re_),
                   "lmwm": (lt_txt, lt_think, lc, le_)}

    llm_results = await asyncio.gather(*(one(p) for p in problems))

    rows, cost = [], 0.0
    wc_secs_tot, wc_calls_tot = 0.0, 0
    for p, calls in llm_results:
        seed, t, h, goal = p["seed"], p["t"], p["h"], p["goal_grid"]
        row = {k: p[k] for k in ("game", "bucket", "mechanic", "kind", "h", "seed", "t",
                                 "synthetic", "random_success", "noop_success",
                                 "start_grid", "goal_grid", "gt_actions")}
        # LLM arms
        for arm in ("raw", "lmwm"):
            text, think, ccost, errs = calls[arm]
            cost += ccost
            plan, perr = parse_plan(text, p["_dims"])
            ok, reached, grids_out = False, None, None
            if plan is not None and len(plan) > PLAN_CAP:
                perr = f"budget-exceeded:{len(plan)}>{PLAN_CAP}"
            elif plan is not None:
                grids_out = exec_plan(prog, seed, p["_prefix"], plan)
                ok, reached = _reached(grids_out, goal)
            row[arm] = {"success": ok, "reached_at": reached,
                        "plan_len": len(plan) if plan else None,
                        "plan_error": perr, "retry_errors": errs,
                        "plan": plan, "grids": grids_out,
                        "reasoning": _parse_tag(text, "reasoning") or "",
                        **thinking_record(think)}
        # wc arm (program search, no LLM)
        hist = [(json.loads(g), prt.parse_action(a)) for g, a in p["_ctx_raw"]]
        start_g, goal_g = json.loads(p["start_grid"]), json.loads(goal)
        universe = prt.build_action_universe(verbs, start_g, goal_g)
        c0, t0 = rt.n_calls, time.time()
        # allow_empty=False: the maintain bucket is goal==start by construction, and
        # success here is "the grid after the FINAL action == goal" -- a zero-length
        # plan has no final action, so make the search find a >=1-step plan that HOLDS
        # the goal (noop, found at depth 1) instead of short-circuiting to [].
        found = prt.plan_search(rt, hist, start_g, goal_g, universe, PLAN_CAP,
                                beam=WC_BEAM, node_budget=WC_BUDGET, context_k=CONTEXT_K,
                                allow_empty=False)
        wc_secs_tot += time.time() - t0
        wc_calls_tot += rt.n_calls - c0
        plan_str = [prt.unparse_action(a) for a in found] if found is not None else None
        ok, reached, grids_out = False, None, None
        if plan_str is not None:
            grids_out = exec_plan(prog, seed, p["_prefix"], plan_str)
            ok, reached = _reached(grids_out, goal)
        row["wc"] = {"success": ok, "reached_at": reached,
                     "plan_len": len(plan_str) if plan_str is not None else None,
                     "plan_error": None if plan_str is not None else "no-plan-found",
                     "plan": plan_str, "grids": grids_out}
        rows.append(row)
    rt.close()
    return {"game": game, "rows": rows, "cost": cost,
            "wc_secs": wc_secs_tot, "wc_calls": wc_calls_tot}


# ---- reporting ------------------------------------------------------------------
ARMS = ["raw", "lmwm", "wc"]
# Learned-artifact root; --artifact-root repoints both arms at another training run
# (e.g. a rebalanced retrain) without touching the problem set or the raw arm.
ARTIFACT_ROOT = REPO / "logs/2026-08-11/human_unified"
BUCKETS = ["act", "wait", "maintain"]


def _succ(rows, arm):
    return sum(r[arm]["success"] for r in rows) / len(rows) if rows else None


def _base(rows, key):
    return sum(r[key] for r in rows) / len(rows) if rows else None


def cell(v):
    return " -- " if v is None else f"{v:.2f}"


def report(all_rows: list[dict], llm: LLMConfig) -> str:
    L = []
    L.append("# Coverage-anchored multi-horizon planning eval\n")
    L.append(f"Planner: {llm.label} | plan cap {PLAN_CAP} | wc budget {WC_BUDGET} "
             f"| {len(all_rows)} problems | scored per bucket, NEVER pooled.\n")

    L.append("## Per bucket (arms vs baselines)\n")
    L.append("| bucket | n | raw | lmwm | wc | noop | random |")
    L.append("|---|--:|--:|--:|--:|--:|--:|")
    for b in BUCKETS:
        rs = [r for r in all_rows if r["bucket"] == b]
        L.append(f"| {b} | {len(rs)} | {cell(_succ(rs,'raw'))} | {cell(_succ(rs,'lmwm'))} | "
                 f"{cell(_succ(rs,'wc'))} | {cell(_base(rs,'noop_success'))} | "
                 f"{cell(_base(rs,'random_success'))} |")

    L.append("\n## Per bucket x horizon (raw / lmwm / wc)\n")
    hs = sorted({r["h"] for r in all_rows})
    L.append("| bucket | " + " | ".join(f"h={h}" for h in hs) + " |")
    L.append("|---|" + "---|" * len(hs))
    for b in BUCKETS:
        cells = []
        for h in hs:
            rs = [r for r in all_rows if r["bucket"] == b and r["h"] == h]
            if not rs:
                cells.append(" -- ")
            else:
                cells.append(f"{cell(_succ(rs,'raw'))}/{cell(_succ(rs,'lmwm'))}/{cell(_succ(rs,'wc'))}")
        L.append(f"| {b} | " + " | ".join(cells) + " |")

    L.append("\n## act bucket, per game x mechanic (raw / lmwm / wc)\n")
    for game in ["bt3gb", "dq8gc", "n2ntd", "s2kt7", "83wkq"]:
        grs = [r for r in all_rows if r["game"] == game and r["bucket"] == "act"]
        if not grs:
            continue
        L.append(f"\n**{game}** ({HGAMES[game][1]})\n")
        L.append("| mechanic | n | raw | lmwm | wc |")
        L.append("|---|--:|--:|--:|--:|")
        for m in sorted({r["mechanic"] for r in grs}):
            rs = [r for r in grs if r["mechanic"] == m]
            L.append(f"| {m} | {len(rs)} | {cell(_succ(rs,'raw'))} | {cell(_succ(rs,'lmwm'))} "
                     f"| {cell(_succ(rs,'wc'))} |")
    return "\n".join(L) + "\n"


async def main_async(args):
    llm = resolve_llm_config(args)
    data = json.loads(Path(args.problems).read_text())
    problems = data["problems"]
    if args.games:
        problems = [p for p in problems if p["game"] in args.games]
    by_game: dict[str, list] = defaultdict(list)
    for p in problems:
        by_game[p["game"]].append(p)
    if args.limit:
        by_game = {g: ps[:args.limit] for g, ps in by_game.items()}

    default_out = ("logs/coverage_plan_eval" if args.llm_backend == "openrouter"
                   else "logs/coverage_plan_eval_claude")
    out = Path(args.out or REPO / default_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results, all_rows, tot_cost = [], [], 0.0
    if args.resume and out.with_suffix(".json").exists():
        prior = json.loads(out.with_suffix(".json").read_text())
        prior_config = prior.get("config", {})
        prior_identity = (prior_config.get("backend"), prior_config.get("url"),
                          prior_config.get("model"))
        current_identity = (llm.backend, llm.url, llm.model)
        if prior_identity != current_identity:
            raise ValueError(
                f"refusing to resume {out.with_suffix('.json')} with a different planner: "
                f"saved={prior_identity}, requested={current_identity}. Use a new --out or --no-resume."
            )
        results = prior.get("results", [])
        all_rows = [r for res in results for r in res["rows"]]
        tot_cost = sum(res["cost"] for res in results)
    done = {res["game"] for res in results}

    t0 = time.time()
    for game in ["bt3gb", "dq8gc", "n2ntd", "s2kt7", "83wkq"]:
        if game not in by_game or game in done:
            continue
        gt = time.time()
        res = await eval_game(game, by_game[game], args.concurrency, llm)
        results.append({k: res[k] for k in ("game", "cost", "wc_secs", "wc_calls")} | {"rows": res["rows"]})
        all_rows += res["rows"]
        tot_cost += res["cost"]
        nd = {b: sum(1 for r in res["rows"] if r["bucket"] == b) for b in BUCKETS}
        line = " | ".join(f"{a} {(_succ(res['rows'],a) or 0):.2f}" for a in ARMS)
        print(f"[done] {game}: {line} | buckets {nd} | wc {res['wc_secs']:.0f}s "
              f"({res['wc_calls']} calls) | ${res['cost']:.3f} | {time.time()-gt:.0f}s", flush=True)
        payload = {"config": {"backend": llm.backend, "url": llm.url, "model": llm.model,
                              "plan_cap": PLAN_CAP, "wc_budget": WC_BUDGET,
                              "provider_order": list(llm.provider_order), "context_k": CONTEXT_K},
                   "cost": tot_cost, "elapsed_s": time.time() - t0, "results": results}
        out.with_suffix(".json").write_text(json.dumps(payload, indent=1))

    md = report(all_rows, llm)
    out.with_suffix(".md").write_text(md)
    print(md, flush=True)
    print(f"\nTOTAL ${tot_cost:.3f} | {time.time()-t0:.0f}s | wrote {out.with_suffix('.json')} "
          f"and {out.with_suffix('.md')}", flush=True)


def resolve_llm_config(args) -> LLMConfig:
    if args.llm_backend == "claude":
        return LLMConfig(
            backend="claude",
            url=args.llm_url or CLAUDE_PROXY_URL,
            model=args.model or CLAUDE_MODEL,
        )
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} must be set for the OpenRouter backend")
    order_arg = args.provider_order
    if order_arg is None:                    # not passed
        # a whitelist or a sort is a complete routing spec; pairing it with the legacy
        # default order would send a dead pin alongside it
        order_arg = "" if (getattr(args, "provider_only", "") or
                           getattr(args, "provider_sort", "")) else ",".join(DEFAULT_PROVIDER_ORDER)
    provider_order = tuple(filter(None, (p.strip() for p in order_arg.split(","))))
    return LLMConfig(
        backend="openrouter",
        url=args.llm_url or OPENROUTER_URL,
        model=args.model or OPENROUTER_MODEL,
        provider_order=provider_order,
        provider_only=tuple(filter(None, (x.strip() for x in
                                          getattr(args, "provider_only", "").split(",")))),
        provider_sort=getattr(args, "provider_sort", "") or "",
        api_key=api_key,
        reasoning_json=getattr(args, "reasoning_json", "") or "",
        allow_fallbacks=getattr(args, "allow_fallbacks", True),
    )


def add_llm_tuning_args(ap) -> None:
    """Planner speed knobs, shared by every eval that plans with an LLM."""
    ap.add_argument("--reasoning-json", default="",
                    help='OpenRouter reasoning block, e.g. \'{"effort": "low"}\' or '
                         '\'{"enabled": false}\'. Hidden thinking tokens dominate per-call '
                         'latency, so this is the main wall-clock knob.')
    ap.add_argument("--provider-only", default="",
                    help="comma-separated provider WHITELIST. Unlike --provider-order this "
                         "still load-balances across the listed hosts, which is what "
                         "throughput at fan-out depends on.")
    ap.add_argument("--provider-sort", default="",
                    help='route by a metric instead of a list, e.g. "throughput"')
    ap.add_argument("--allow-fallbacks", action=argparse.BooleanOptionalAction, default=True,
                    help="let OpenRouter route past --provider-order (default). Pass "
                         "--no-allow-fallbacks to pin exclusively -- a dead slug in the "
                         "order silently becomes default routing otherwise.")


def main():
    global ARTIFACT_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default=str(REPO / "logs/coverage_plan_problems.json"))
    ap.add_argument("--out", default="", help="output stem (default: backend-specific under logs/)")
    ap.add_argument("--games", type=str, default="", help="comma-separated subset")
    ap.add_argument("--artifact-root", default=str(ARTIFACT_ROOT),
                    help="root holding rexpure/<game>_s1 and worldcoder/<game>_s1")
    ap.add_argument("--limit", type=int, default=0, help="cap problems per game (smoke test)")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--llm-backend", choices=("claude", "openrouter"), default="openrouter")
    ap.add_argument("--llm-url", default="", help="chat-completions URL override")
    ap.add_argument("--model", default="", help="planner model/Claude CLI model alias override")
    ap.add_argument("--provider-order", default=None,
                    help="OpenRouter provider FAILOVER list, comma-separated. Every call "
                         "goes to the first entry; this is not a load balancer. Defaults to "
                         "the legacy pin only when neither --provider-only nor "
                         "--provider-sort is given.")
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY",
                    help="API-key environment variable for OpenRouter")
    add_llm_tuning_args(ap)
    args = ap.parse_args()
    ARTIFACT_ROOT = Path(args.artifact_root)
    args.games = set(filter(None, args.games.split(",")))
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
