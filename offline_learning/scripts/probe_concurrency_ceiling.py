#!/usr/bin/env python3
"""How many F requests can we actually keep in flight before goodput stops improving?

Every sweep so far has been capped at 50-60 in-flight on an INHERITED claim ("survives
64-concurrent") that was never measured. That cap is now costing real wall time: a
--propose-batch 5 sweep over 5 games had to drop --concurrency to 2 (5x5x2 = 50), which
turns an 8.4s eval into ~50s. If the provider sustains more, every sweep gets faster.

The retry counters from the completed runs point at RATE, not parallelism, as the binding
limit: all arms sat at 60 in-flight, but retries ranged 1 -> 2342, scaling with how fast
each arm pushed requests (gpt-oss-120b's seconds-fast reflection produced the tightest F
bursts and the most retries). So this measures goodput and tokens/min, not just "did it
429".

Method: fire N genuinely concurrent REAL inverse-dynamics prompts (pulled from a completed
run's predictions.jsonl, so the token profile matches production), 3 back-to-back waves per
level to expose per-minute windows rather than a single forgiving burst. Raw httpx with NO
retry and NO hedge -- the production plumbing would mask 429s behind backoff, which is the
very thing being measured. A cooldown between levels lets the rate window drain so level
N's usage does not contaminate level N+1.

Reports per level: success rate, 429 count, p50/p95 latency, goodput (successful calls/s)
and tokens/s. The ceiling is where goodput FLATTENS, not where errors first appear.

    uv run python offline_learning/scripts/probe_concurrency_ceiling.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "offline_learning"))

import httpx  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

URL = "https://openrouter.ai/api/v1/chat/completions"
SRC = ROOT / "logs/aug8_hardmin_gptoss20b/bt3gb_seed1/rexpure_run_seed1/predictions.jsonl"
MODEL = "openai/gpt-oss-20b"
PROVIDER = ["groq"]
REASONING = {"effort": "low"}


def load_prompts(n: int) -> list[str]:
    """Real F prompts (inverse-dynamics) so token counts match a production eval."""
    out = []
    with SRC.open() as f:
        for line in f:
            try:
                p = json.loads(line).get("inv_prompt")
            except json.JSONDecodeError:
                continue
            if p:
                out.append(p)
            if len(out) >= n:
                break
    if not out:
        raise SystemExit(f"no inv_prompt rows in {SRC}")
    return out


async def one(client, prompt) -> dict:
    t0 = time.perf_counter()
    try:
        r = await client.post(URL, json={
            "model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "usage": {"include": True}, "reasoning": REASONING,
            "provider": {"order": PROVIDER, "allow_fallbacks": False},
        })
        wall = time.perf_counter() - t0
        if r.status_code == 429:
            return {"ok": False, "code": 429, "wall": wall,
                    "retry_after": r.headers.get("retry-after")}
        d = r.json()
        if "error" in d or "choices" not in d:
            return {"ok": False, "code": r.status_code, "wall": wall,
                    "err": str(d.get("error", ""))[:100]}
        u = d.get("usage", {}) or {}
        return {"ok": True, "wall": wall,
                "in_tok": u.get("prompt_tokens", 0), "out_tok": u.get("completion_tokens", 0),
                "cost": u.get("cost", 0.0)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "code": None, "wall": time.perf_counter() - t0,
                "err": f"{type(e).__name__}: {str(e)[:80]}"}


async def wave(client, prompts, n) -> tuple[list[dict], float]:
    """N genuinely simultaneous requests (no semaphore) -- this IS the in-flight level."""
    t0 = time.perf_counter()
    rows = await asyncio.gather(*[one(client, prompts[i % len(prompts)]) for i in range(n)])
    return rows, time.perf_counter() - t0


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="16,32,64,96,128,192,256")
    ap.add_argument("--waves", type=int, default=3, help="back-to-back waves per level")
    ap.add_argument("--cooldown", type=float, default=30.0,
                    help="seconds between levels so the per-minute window drains")
    ap.add_argument("--stop-error-rate", type=float, default=0.35,
                    help="abort escalation once a level fails this fraction of requests")
    ap.add_argument("--out", default="logs/aug9_concurrency_ceiling")
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    prompts = load_prompts(64)
    outd = ROOT / args.out
    outd.mkdir(parents=True, exist_ok=True)
    key = os.environ["OPENROUTER_API_KEY"]
    print(f"[probe] {MODEL} @ {','.join(PROVIDER)} | {len(prompts)} real F prompts "
          f"(~{statistics.mean(len(p) for p in prompts)/4:.0f} tok each)")
    print(f"{'N':>5s} {'ok%':>6s} {'429':>5s} {'err':>5s} {'p50':>7s} {'p95':>7s} "
          f"{'goodput':>9s} {'tok/s':>8s} {'$':>7s}")

    results = []
    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {key}"},
                                 timeout=180, limits=httpx.Limits(
                                     max_connections=None, max_keepalive_connections=None)) as client:
        for n in levels:
            rows, wall = [], 0.0
            for _ in range(args.waves):
                r, w = await wave(client, prompts, n)
                rows += r
                wall += w
            ok = [x for x in rows if x["ok"]]
            n429 = sum(1 for x in rows if x.get("code") == 429)
            nerr = len(rows) - len(ok) - n429
            walls = sorted(x["wall"] for x in ok)
            tok = sum(x.get("in_tok", 0) + x.get("out_tok", 0) for x in ok)
            rec = {
                "n": n, "requests": len(rows), "ok": len(ok), "n429": n429, "err": nerr,
                "ok_rate": len(ok) / len(rows), "wall_total": round(wall, 1),
                "p50": round(statistics.median(walls), 2) if walls else None,
                "p95": round(walls[int(0.95 * (len(walls) - 1))], 2) if walls else None,
                "goodput": round(len(ok) / wall, 2), "tok_per_s": round(tok / wall),
                "cost": round(sum(x.get("cost", 0.0) for x in ok), 4),
                "retry_after": [x.get("retry_after") for x in rows if x.get("retry_after")][:3],
            }
            results.append(rec)
            print(f"{n:5d} {100*rec['ok_rate']:5.0f}% {n429:5d} {nerr:5d} "
                  f"{rec['p50'] or 0:6.2f}s {rec['p95'] or 0:6.2f}s "
                  f"{rec['goodput']:8.2f}/s {rec['tok_per_s']:8d} {rec['cost']:7.4f}", flush=True)
            (outd / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            if 1 - rec["ok_rate"] >= args.stop_error_rate:
                print(f"[probe] {100*(1-rec['ok_rate']):.0f}% failures at N={n} -> stopping escalation")
                break
            if n != levels[-1]:
                await asyncio.sleep(args.cooldown)

    best = max(results, key=lambda r: r["goodput"])
    print(f"\npeak goodput {best['goodput']}/s at N={best['n']} "
          f"({100*best['ok_rate']:.0f}% ok, {best['tok_per_s']} tok/s)")
    print("the usable ceiling is where goodput FLATTENS -- past that, extra in-flight "
          "requests only add queueing latency and retries.")
    print(f"-> {outd/'results.json'}")


if __name__ == "__main__":
    asyncio.run(main())
