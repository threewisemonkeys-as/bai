"""Stage-A live probe: which OpenRouter model/provider combos could beat
gpt-oss-120b@cerebras+efflow (35s wall / ~$0.00026 per call / 0.78 test ID) on the
eval-call role?

Paper filter (2026-08-07 full-catalog endpoint scan): text models with an endpoint
at prompt<=$0.15/M, completion<=$0.45/M, uptime>=95, ctx>=32k -> 52 models; the
plausible-quality subset below gets a live test with the REAL windowed set-ID
prompt (same builder as probe_qwen37flash_speed). Reasoning disabled where the
endpoint supports the param (on 400 error the call retries without it and the
fallback is recorded).

Per config x N_CALLS: wall, output/reasoning tokens, OpenRouter-billed cost,
<actions> parse validity, provider actually used. Survivors (p50 fast AND
cost/call within envelope) graduate to real bench arms.

Results -> logs/aug7_evalmodel_probe/results.json
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "offline_learning"))
sys.path.insert(0, str(ROOT / "offline_learning" / "scripts"))

import httpx
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from probe_qwen37flash_speed import build_prompts  # noqa: E402
from invdyn_core import _extract_action_set  # noqa: E402

URL = "https://openrouter.ai/api/v1/chat/completions"
OUT = ROOT / "logs/aug7_evalmodel_probe"
N_CALLS = 3
TIMEOUT_S = 120

NOTHINK = {"reasoning": {"enabled": False}}
EFFNONE = {"reasoning": {"effort": "none"}}
EFFLOW = {"reasoning": {"effort": "low"}}

# (label, model, provider order or None=unpinned, body extra)
CANDIDATES = [
    ("BASELINE gptoss120b+low", "openai/gpt-oss-120b", ["cerebras", "groq", "sambanova"], EFFLOW),
    ("gptoss20b groq +low", "openai/gpt-oss-20b", ["groq"], EFFLOW),
    ("llama4-scout groq", "meta-llama/llama-4-scout", ["groq"], {}),
    ("llama31-8b groq", "meta-llama/llama-3.1-8b-instruct", ["groq"], {}),
    ("ling26-flash novita", "inclusionai/ling-2.6-flash", ["novita"], {}),
    ("ling30-flash nothink", "inclusionai/ling-3.0-flash", ["novita", "deepinfra"], NOTHINK),
    ("qwen3-30b-a3b instruct", "qwen/qwen3-30b-a3b-instruct-2507", ["coreweave", "nebius", "siliconflow"], {}),
    ("gemma3-27b", "google/gemma-3-27b-it", ["deepinfra", "nebius", "novita"], {}),
    ("gemma4-26b-a4b nothink", "google/gemma-4-26b-a4b-it", ["cloudflare", "deepinfra", "novita"], NOTHINK),
    ("mistral-small-3.2", "mistralai/mistral-small-3.2-24b-instruct", ["deepinfra", "parasail"], {}),
    ("gemini25-flashlite off", "google/gemini-2.5-flash-lite", None, EFFNONE),
    ("gpt5-nano effnone", "openai/gpt-5-nano", None, EFFNONE),
    ("gpt56-luna effnone", "openai/gpt-5.6-luna", None, EFFNONE),
    ("dsv4-flash nothink", "deepseek/deepseek-v4-flash", ["baidu", "cloudflare", "coreweave"], NOTHINK),
    ("seed16-flash nothink", "bytedance-seed/seed-1.6-flash", None, NOTHINK),
    ("step35-flash nothink", "stepfun/step-3.5-flash", ["siliconflow"], NOTHINK),
    ("nex-n2-mini nothink", "nex-agi/nex-n2-mini", None, NOTHINK),
]


async def one_call(client, prompt, choices, model, order, body_extra):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "usage": {"include": True},
        **body_extra,
    }
    if order:
        body["provider"] = {"order": order, "allow_fallbacks": False}
    dropped_reasoning = False
    for attempt in (1, 2):
        t0 = time.perf_counter()
        try:
            r = await client.post(URL, json=body)
            d = r.json()
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}", "wall_s": round(time.perf_counter() - t0, 2)}
        wall = time.perf_counter() - t0
        if "error" in d:
            # retry once without the reasoning field (some endpoints 400 on it)
            if attempt == 1 and "reasoning" in body:
                body.pop("reasoning")
                dropped_reasoning = True
                continue
            return {"error": str(d["error"])[:200], "wall_s": round(wall, 2)}
        msg = d["choices"][0]["message"]
        usage = d.get("usage", {})
        det = usage.get("completion_tokens_details") or {}
        return {
            "wall_s": round(wall, 2),
            "output_tokens": usage.get("completion_tokens", 0),
            "reasoning_tokens": det.get("reasoning_tokens", 0) or 0,
            "cost": usage.get("cost", 0.0),
            "provider": d.get("provider"),
            "valid": bool(_extract_action_set(msg.get("content") or "", choices)),
            "dropped_reasoning": dropped_reasoning,
        }


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prompts = build_prompts()[:N_CALLS]
    key = os.environ["OPENROUTER_API_KEY"]
    results = {}
    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {key}"}, timeout=TIMEOUT_S
    ) as client:
        for label, model, order, extra in CANDIDATES:
            rows = [await one_call(client, p, c, model, order, extra) for p, c in prompts]
            results[label] = {"model": model, "order": order, "extra": extra, "calls": rows}
            ok = [r for r in rows if "error" not in r]
            if ok:
                walls = sorted(r["wall_s"] for r in ok)
                note = " [reasoning field rejected]" if any(r.get("dropped_reasoning") for r in ok) else ""
                print(f"{label:<26} p50={walls[len(walls)//2]:6.1f}s "
                      f"out={statistics.mean(r['output_tokens'] for r in ok):6.0f} "
                      f"rtok={statistics.mean(r['reasoning_tokens'] for r in ok):5.0f} "
                      f"$/call={statistics.mean(r['cost'] for r in ok):.5f} "
                      f"valid={sum(r['valid'] for r in ok)}/{len(rows)} "
                      f"prov={ok[0]['provider']}{note}", flush=True)
            else:
                print(f"{label:<26} ALL FAILED: {rows[0].get('error', '?')[:110]}", flush=True)
            (OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nresults -> {OUT / 'results.json'}")


if __name__ == "__main__":
    asyncio.run(main())
