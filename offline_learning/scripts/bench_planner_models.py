#!/usr/bin/env python3
"""Speed/quality bench for the PLANNER model used by the curated planning evals.

The curated online eval costs ~10 h of wall clock because every rollout is a serial
chain of up to 50 LLM calls, so total wall is set by per-call latency far more than by
anything else we control.  This script measures, on a fixed subset of the real curated
problems and with the REAL planning prompts (imported, not paraphrased):

  stage providers : per (model, provider) clean latency, tok/s, hidden-reasoning volume
  stage arms      : per (model, provider, reasoning-effort) latency AND plan quality,
                    scored by executing the plan in the Autumn engine exactly as the
                    offline eval does
  stage burst     : sustained throughput at the concurrency the real eval uses -- a
                    provider that is fast single-stream but throttles bursts is useless
                    here (Cerebras did exactly that in an earlier benchmark)

Every arm sees byte-identical prompts (built once, cached) and the same job list in the
same order, so differences are the model's.

    uv run python offline_learning/scripts/bench_planner_models.py providers
    uv run python offline_learning/scripts/bench_planner_models.py arms
    uv run python offline_learning/scripts/bench_planner_models.py burst --arm gptoss120b-cerebras-low
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
OFF = HERE.parent
REPO = OFF.parent
for _p in (str(OFF), str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import httpx  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from validate import _parse_tag, run_perceive  # noqa: E402
from offline_learning.coverage_plan import exec_plan  # noqa: E402
from eval_coverage_plan import (  # noqa: E402
    DEFAULT_KNOWLEDGE, PLAN_RAW_TMPL, PLAN_WIN_TMPL, feat_transcript, parse_plan,
    raw_transcript,
)
from eval_coverage_online import RETRY_SUFFIX  # noqa: E402
from eval_curated_plan import PLAN_CAP, gstr  # noqa: E402

load_dotenv(REPO / ".env")
URL = "https://openrouter.ai/api/v1/chat/completions"
PROBLEMS = REPO / "logs/2026-08-18/curated/problems.json"
ARTIFACTS = REPO / "logs/2026-08-11/human_unified"
OUT = REPO / "logs/2026-08-19/planner_bench"

# Subset: one prompt shape per game, difficulty spread, and deliberately biased toward
# problems the production planner got PARTIAL credit on -- a subset where every problem is
# 0.0 or 1.0 for both arms cannot rank models.  (offline pass@1 raw/lmwm in comments.)
SUBSET = [
    "n2ntd/high-ground",       # L1  0.0 / 0.6
    "n2ntd/coin-ground",       # L2  0.0 / 0.6
    "bt3gb/park-cloud",        # L1  0.0 / 1.0
    "bt3gb/ice-tower",         # L3  0.0 / 0.4
    "dq8gc/swap-drive",        # L2  0.6 / 0.6
    "dq8gc/gather",            # L4  0.2 / 0.0
    "s2kt7/one-eaten",         # L2  0.2 / 0.4
    "83wkq/spawn-two",         # L2  1.0 / 1.0  (sanity anchor: a broken arm drops here)
]

NOTHINK = {"enabled": False}


@dataclass(frozen=True)
class Arm:
    label: str
    model: str
    providers: tuple[str, ...] = ()      # `order`: a strict FAILOVER list, NOT a load
                                         # balancer -- every call tries [0] first
    only: tuple[str, ...] = ()           # `only`: a whitelist that STILL load-balances
    sort: str = ""                       # e.g. "throughput"; mutually exclusive with order
    reasoning: dict | None = None
    fallbacks: bool = False              # True only to reproduce the production pin
    note: str = ""

    def body(self, prompt: str) -> dict:
        b: dict = {"model": self.model,
                   "messages": [{"role": "user", "content": prompt}],
                   "usage": {"include": True}}
        prov: dict = {}
        if self.providers:
            prov["order"] = list(self.providers)
            prov["allow_fallbacks"] = self.fallbacks
        if self.only:
            prov["only"] = list(self.only)
        if self.sort:
            prov["sort"] = self.sort
        if prov:
            b["provider"] = prov
        if self.reasoning is not None:
            b["reasoning"] = dict(self.reasoning)
        return b


# ---- call ---------------------------------------------------------------------------
async def call(client: httpx.AsyncClient, prompt: str, arm: Arm,
               retries: int = 1) -> dict:
    body = arm.body(prompt)
    err = None
    for attempt in range(retries + 1):
        t0 = time.perf_counter()
        try:
            r = await client.post(URL, json=body)
            d = r.json()
        except Exception as exc:                                   # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            continue
        wall = time.perf_counter() - t0
        if d.get("error"):
            err = str(d["error"])[:300]
            # a provider that rejects the reasoning field should be recorded as such,
            # not silently re-run without it (that would benchmark a different arm)
            if "reasoning" in err.lower() or "Reasoning" in err:
                return {"error": err, "wall_s": wall, "reasoning_rejected": True}
            continue
        msg = d["choices"][0]["message"]
        u = d.get("usage") or {}
        det = u.get("completion_tokens_details") or {}
        out_tok = u.get("completion_tokens") or 0
        return {
            "wall_s": wall,
            "text": msg.get("content") or "",
            "thinking": msg.get("reasoning") or msg.get("reasoning_content") or "",
            "prompt_tokens": u.get("prompt_tokens") or 0,
            "output_tokens": out_tok,
            "reasoning_tokens": det.get("reasoning_tokens") or 0,
            "cost": float(u.get("cost") or 0.0),
            "provider": d.get("provider"),
            "tok_s": (out_tok / wall) if wall > 0 else 0.0,
            "retry": attempt,          # NOT "attempt": that key is the sample index
        }
    return {"error": err or "unknown", "wall_s": 0.0}


# ---- prompt bank --------------------------------------------------------------------
def build_bank() -> list[dict]:
    """One entry per (problem, arm-kind).  Built once so every model sees the same bytes."""
    cache = OUT / "prompt_bank.json"
    if cache.exists():
        return json.loads(cache.read_text())
    problems = {f"{p['game']}/{p['id']}": p for p in json.loads(PROBLEMS.read_text())}
    bank = []
    for key in SUBSET:
        p = problems[key]
        game = p["game"]
        rex = ARTIFACTS / "rexpure" / f"{game}_s1"
        perc = (rex / "best_perception_rexpure_seed1.py").read_text()
        beliefs = (rex / "best_beliefs_rexpure_seed1.txt").read_text()
        start, goal = gstr(p["start"]), gstr(p["goal"])
        z_t = run_perceive(perc, start)[0]
        z_goal = run_perceive(perc, goal)[0]
        common = {"key": key, "game": game, "id": p["id"], "tier": p["tier"],
                  "h": p["h"], "program": p["program"], "seed": p["seed"],
                  "goal_grid": goal, "dims": [len(p["start"]), len(p["start"][0])]}
        bank.append(common | {"arm_kind": "raw", "prompt": PLAN_RAW_TMPL.format(
            cap=PLAN_CAP, default_knowledge=DEFAULT_KNOWLEDGE,
            transcript=raw_transcript([], start), goal=goal)})
        bank.append(common | {"arm_kind": "lmwm", "prompt": PLAN_WIN_TMPL.format(
            cap=PLAN_CAP, default_knowledge=DEFAULT_KNOWLEDGE,
            beliefs=beliefs.strip() or "(empty)",
            transcript=feat_transcript([], z_t), goal=z_goal or "(empty)")})
    OUT.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(bank, indent=1))
    return bank


def score(rec: dict, item: dict) -> dict:
    """Parse + execute the plan exactly as eval_curated_plan does."""
    if "error" in rec:
        return {"success": False, "plan_error": "call-failed", "plan_len": None}
    plan, perr = parse_plan(rec["text"], tuple(item["dims"]))
    if plan is not None and len(plan) > PLAN_CAP:
        plan, perr = None, f"budget-exceeded:{len(plan)}>{PLAN_CAP}"
    if plan is None:
        return {"success": False, "plan_error": perr, "plan_len": None}
    grids = exec_plan(item["program"], item["seed"], [], plan)
    ok = any(g == item["goal_grid"] for g in grids)
    return {"success": ok, "plan_error": None, "plan_len": len(plan)}


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    walls = sorted(r["wall_s"] for r in ok)

    def pct(p):
        return walls[min(len(walls) - 1, int(p * len(walls)))] if walls else 0.0
    return {
        "n": len(rows), "n_ok": len(ok), "n_err": len(rows) - len(ok),
        "p50_s": pct(0.5), "p90_s": pct(0.9),
        "mean_s": statistics.mean(walls) if walls else 0.0,
        "max_s": max(walls) if walls else 0.0,
        "out_tok": statistics.mean(r["output_tokens"] for r in ok) if ok else 0,
        "rsn_tok": statistics.mean(r["reasoning_tokens"] for r in ok) if ok else 0,
        "in_tok": statistics.mean(r["prompt_tokens"] for r in ok) if ok else 0,
        "tok_s": statistics.median(r["tok_s"] for r in ok) if ok else 0,
        "cost_call": statistics.mean(r["cost"] for r in ok) if ok else 0,
        "cost_total": sum(r["cost"] for r in ok),
        "providers": sorted({str(r.get("provider")) for r in ok}),
    }


def client(timeout: float) -> httpx.AsyncClient:
    key = os.environ["OPENROUTER_API_KEY"]
    return httpx.AsyncClient(headers={"Authorization": f"Bearer {key}"},
                             timeout=timeout, limits=httpx.Limits(
                                 max_connections=256, max_keepalive_connections=64))


# ---- stage: providers ---------------------------------------------------------------
async def stage_providers(a):
    """Per-endpoint clean latency.  Calls WITHIN one endpoint are sequential (so the
    number is latency, not queueing); different endpoints run concurrently because they
    are different backends and cannot contend with each other."""
    bank = build_bank()
    probes = [b for b in bank if b["key"] in ("bt3gb/park-cloud", "dq8gc/gather")]
    spec = json.loads(Path(a.providers_json).read_text())
    res: dict = {}
    lock = asyncio.Lock()

    async def endpoint(c, model, prov, reasoning):
        arm = Arm(f"{model}@{prov or 'auto'}", model, (prov,) if prov else (), reasoning)
        rows = []
        for i in range(a.calls):
            rows.append(await call(c, probes[i % len(probes)]["prompt"], arm, retries=0))
        s = summarize(rows)
        async with lock:
            res[arm.label] = {"model": model, "provider": prov or "auto", "reasoning": reasoning,
                              "summary": s, "rows": rows}
            if s["n_ok"]:
                print(f"{arm.label:<50} p50={s['p50_s']:7.1f}s tok/s={s['tok_s']:6.0f} "
                      f"out={s['out_tok']:6.0f} rsn={s['rsn_tok']:6.0f} "
                      f"${s['cost_call']:.5f}/call  {s['providers']}", flush=True)
            else:
                print(f"{arm.label:<50} FAILED: {rows[0].get('error','?')[:110]}", flush=True)
            (OUT / a.out).write_text(json.dumps(res, indent=1))

    async with client(a.timeout) as c:
        tasks = [endpoint(c, e["model"], prov, e.get("reasoning"))
                 for e in spec for prov in e["providers"]]
        await asyncio.gather(*tasks)
    print(f"\n-> {OUT / a.out}")


# ---- stage: arms --------------------------------------------------------------------
async def stage_arms(a):
    bank = build_bank()
    arms = [Arm(**x) for x in json.loads(Path(a.arms_json).read_text())]
    if a.only:
        want = set(a.only.split(","))
        arms = [x for x in arms if x.label in want]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / a.out
    res = json.loads(path.read_text()) if (a.resume and path.exists()) else {}
    jobs = [(item, k) for item in bank for k in range(a.attempts)]
    async with client(a.timeout) as c:
        for arm in arms:
            if a.resume and res.get(arm.label, {}).get("summary", {}).get("n_ok"):
                print(f"[skip] {arm.label} (already in {path.name})", flush=True)
                continue
            # an arm whose every call failed (rate limit, dead key) carries no result, so
            # resume must RE-RUN it rather than treat the failure as done
            sem = asyncio.Semaphore(a.concurrency)
            t0 = time.time()

            done = {"n": 0}

            async def one(item, k, arm=arm, sem=sem, done=done):
                async with sem:
                    rec = await call(c, item["prompt"], arm, retries=a.retries)
                sc = score(rec, item)
                if a.reask and sc["plan_error"] and "error" not in rec:
                    # the ONLINE harness gives one corrective re-ask per round, so an arm
                    # that reasons itself out of emitting a <plan> is not simply dead
                    # there -- and the round pays for both calls, so the walls add.
                    fix = item["prompt"] + RETRY_SUFFIX.format(
                        error=sc["plan_error"], remaining=PLAN_CAP)
                    async with sem:
                        rec2 = await call(c, fix, arm, retries=a.retries)
                    if "error" not in rec2:
                        sc2 = score(rec2, item)
                        rec["wall_s"] += rec2["wall_s"]
                        rec["cost"] += rec2["cost"]
                        rec["output_tokens"] += rec2["output_tokens"]
                        rec["reasoning_tokens"] += rec2["reasoning_tokens"]
                        rec["reasked"] = True
                        sc = sc2 | {"recovered": not sc2["plan_error"]}
                # an arm that prints only when all 48 land is indistinguishable from a
                # hung one, which is exactly how the last long run wasted three hours
                done["n"] += 1
                if done["n"] % a.progress_every == 0:
                    print(f"    .. {arm.label}: {done['n']}/{len(jobs)} calls "
                          f"({time.time() - t0:.0f}s)", flush=True)
                rec.pop("text", None)
                rec.pop("thinking", None)
                return {"key": item["key"], "arm_kind": item["arm_kind"], "attempt": k,
                        "tier": item["tier"], **rec, **sc}

            rows = await asyncio.gather(*(one(i, k) for i, k in jobs))
            wall = time.time() - t0
            s = summarize(rows)
            good = [r for r in rows if "error" not in r]
            s["pass1"] = (sum(r["success"] for r in good) / len(good)) if good else 0.0
            s["pass1_raw"] = _rate(good, "raw")
            s["pass1_lmwm"] = _rate(good, "lmwm")
            s["invalid"] = sum(1 for r in good if r["plan_error"]) / max(1, len(good))
            s["reasked"] = sum(1 for r in good if r.get("reasked")) / max(1, len(good))
            valid = [r for r in good if not r["plan_error"]]
            s["pass1_given_valid"] = (sum(r["success"] for r in valid) / len(valid)
                                      if valid else 0.0)
            s["wall_s"] = wall
            res[arm.label] = {"arm": arm.__dict__, "summary": s, "rows": rows}
            print(f"{arm.label:<34} p50={s['p50_s']:6.1f}s p90={s['p90_s']:6.1f}s "
                  f"tok/s={s['tok_s']:5.0f} out={s['out_tok']:5.0f} rsn={s['rsn_tok']:5.0f} "
                  f"pass@1={s['pass1']:.2f} (raw {s['pass1_raw']:.2f}/lmwm {s['pass1_lmwm']:.2f}) "
                  f"|valid={s['pass1_given_valid']:.2f} bad={s['invalid']:.2f} "
                  f"reask={s['reasked']:.2f} err={s['n_err']} ${s['cost_total']:.3f} "
                  f"[{wall:.0f}s]", flush=True)
            path.write_text(json.dumps(res, indent=1))
    print(f"\n-> {path}")


def _rate(rows, kind):
    s = [r for r in rows if r["arm_kind"] == kind]
    return sum(r["success"] for r in s) / len(s) if s else 0.0


# ---- stage: burst -------------------------------------------------------------------
async def stage_burst(a):
    """Fire N identical-shape planning calls at once: does the provider hold up at the
    concurrency the real eval runs at, or does it 429 / queue?"""
    bank = build_bank()
    arms = {x["label"]: Arm(**x) for x in json.loads(Path(a.arms_json).read_text())}
    labels = a.arm.split(",") if a.arm else list(arms)
    out = {}
    async with client(a.timeout) as c:
        for lab in labels:
            arm = arms[lab]
            t0 = time.time()
            done = {"n": 0}

            async def one(i, arm=arm, done=done):
                r = await call(c, bank[i % len(bank)]["prompt"], arm, retries=0)
                done["n"] += 1
                if done["n"] % 12 == 0:
                    print(f"    .. {arm.label}: {done['n']}/{a.n} ({time.time()-t0:.0f}s)",
                          flush=True)
                return r
            rows = await asyncio.gather(*(one(i) for i in range(a.n)))
            wall = time.time() - t0
            s = summarize(rows)
            errs = [r["error"][:80] for r in rows if "error" in r]
            out[lab] = {"n": a.n, "wall_s": wall, "summary": s, "errors": errs}
            print(f"{lab:<34} {a.n} concurrent -> {wall:6.1f}s wall | p50={s['p50_s']:6.1f}s "
                  f"p90={s['p90_s']:6.1f}s max={s['max_s']:6.1f}s | ok={s['n_ok']}/{a.n} "
                  f"| {a.n / wall:.2f} calls/s", flush=True)
            for e in errs[:3]:
                print(f"     err: {e}", flush=True)
            (OUT / a.out).write_text(json.dumps(out, indent=1))
    print(f"\n-> {OUT / a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("providers")
    p.add_argument("--providers-json", default=str(OUT / "provider_spec.json"))
    p.add_argument("--out", default="providers.json")
    p.add_argument("--calls", type=int, default=2)
    p.add_argument("--reasoning", default="default",
                   help="'default' or a JSON object applied to every probe")
    p.add_argument("--timeout", type=float, default=300)
    p.set_defaults(fn=stage_providers)

    p = sub.add_parser("arms")
    p.add_argument("--arms-json", default=str(OUT / "arms.json"))
    p.add_argument("--out", default="arms.json")
    p.add_argument("--only", default="")
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--timeout", type=float, default=420)
    p.add_argument("--progress-every", type=int, default=8)
    p.add_argument("--reask", action=argparse.BooleanOptionalAction, default=False,
                   help="model the ONLINE harness's one corrective re-ask on an unusable "
                        "plan (walls and cost of both calls are summed into the round)")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.set_defaults(fn=stage_arms)

    p = sub.add_parser("burst")
    p.add_argument("--arms-json", default=str(OUT / "arms.json"))
    p.add_argument("--arm", default="")
    p.add_argument("--n", type=int, default=48)
    p.add_argument("--out", default="burst.json")
    p.add_argument("--timeout", type=float, default=420)
    p.set_defaults(fn=stage_burst)

    a = ap.parse_args()
    asyncio.run(a.fn(a))


if __name__ == "__main__":
    main()
