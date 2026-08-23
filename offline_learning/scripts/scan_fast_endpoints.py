#!/usr/bin/env python3
"""Full-catalog scan for planner candidates: which OpenRouter endpoints are actually FAST?

The curated planning eval's wall clock is a serial chain of LLM calls, so the only
catalog property that matters is decode throughput (plus enough context and a price we
can afford).  This walks every model's /endpoints and ranks endpoints by measured p50
throughput over the last 30m, which is the closest thing the catalog has to the quantity
we care about.  Output feeds the live probe in bench_planner_models.py -- catalog stats
are a filter, never evidence.

    uv run python offline_learning/scripts/scan_fast_endpoints.py --min-tps 200
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import httpx
from dotenv import load_dotenv

load_dotenv(REPO / ".env")
BASE = "https://openrouter.ai/api/v1"
OUT = REPO / "logs/2026-08-19/planner_bench"


async def endpoints(c, mid):
    try:
        r = await c.get(f"{BASE}/models/{mid}/endpoints")
        return mid, (r.json().get("data") or {}).get("endpoints") or []
    except Exception:                                        # noqa: BLE001
        return mid, []


async def main_async(a):
    key = os.environ["OPENROUTER_API_KEY"]
    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {key}"},
                                 timeout=60) as c:
        models = (await c.get(f"{BASE}/models")).json()["data"]
        ids = [m["id"] for m in models
               if not m["id"].startswith("~") and ":free" not in m["id"]
               and (m.get("context_length") or 0) >= a.min_ctx
               and "image" not in m["id"] and "video" not in m["id"]]
        sem = asyncio.Semaphore(16)

        async def one(mid):
            async with sem:
                return await endpoints(c, mid)
        got = await asyncio.gather(*(one(m) for m in ids))

    rows = []
    for mid, eps in got:
        for e in eps:
            tps = (e.get("throughput_last_30m") or {}).get("p50") or 0
            ttft = (e.get("latency_last_30m") or {}).get("p50") or 0
            pi = float((e.get("pricing") or {}).get("prompt") or 0) * 1e6
            po = float((e.get("pricing") or {}).get("completion") or 0) * 1e6
            if tps < a.min_tps or po > a.max_out or (e.get("uptime_last_30m") or 0) < a.min_uptime:
                continue
            rows.append({"model": mid, "tag": e.get("tag") or e.get("provider_name"),
                         "provider": (e.get("tag") or "").split("/")[0] or e.get("provider_name"),
                         "tps": tps, "ttft_ms": ttft, "in_per_m": pi, "out_per_m": po,
                         "ctx": e.get("context_length"),
                         "uptime": round(e.get("uptime_last_30m") or 0, 1),
                         "reasoning": "reasoning" in (e.get("supported_parameters") or [])})
    rows.sort(key=lambda r: -r["tps"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fast_endpoints.json").write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} endpoints >= {a.min_tps} tok/s, <= ${a.max_out}/M out\n")
    print(f"{'model':<44}{'tag':<24}{'tps':>6}{'ttft':>7}{'$in':>7}{'$out':>7}  reas")
    for r in rows[:a.top]:
        print(f"{r['model']:<44}{r['tag']:<24}{r['tps']:>6.0f}{r['ttft_ms']:>7.0f}"
              f"{r['in_per_m']:>7.2f}{r['out_per_m']:>7.2f}  {'Y' if r['reasoning'] else '-'}")
    print(f"\n-> {OUT / 'fast_endpoints.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-tps", type=float, default=200)
    ap.add_argument("--max-out", type=float, default=3.0, help="$/M completion ceiling")
    ap.add_argument("--min-ctx", type=int, default=32768)
    ap.add_argument("--min-uptime", type=float, default=95)
    ap.add_argument("--top", type=int, default=80)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
