"""Round 2 of the qwen3.7-flash speed probe: direct OpenRouter HTTP (no litellm).

Round 1 (probe_qwen37flash_speed.py) showed ~2.7-3.7k billed output tokens per eval
call vs ~250 tokens of visible text at 0 reported reasoning_tokens: ~90% of output
is hidden thinking that Alibaba bills but does not break out, and the `reasoning`
param sent through litellm's responses bridge (extra_body) changed nothing -- the
same bridge that silently drops extra_body["provider"].

This round bypasses litellm and POSTs /api/v1/chat/completions directly to find the
knob that actually kills the thinking tokens:

  baseline       -- no reasoning field (round-1 default, sanity anchor)
  enabled_false  -- OpenRouter unified reasoning {"enabled": false}
  effort_low     -- reasoning {"effort": "low"}
  native_off     -- Alibaba-native enable_thinking: false (OpenRouter passes
                    unknown body params through to the provider)
  brevity        -- prompt tail capping <reasoning> at 40 words (model-agnostic
                    fallback if no API knob works)

Same real windowed set-ID prompts as round 1. Results -> logs/aug7_qwen_speed_probe/
results_round2.json.
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
MODEL = "qwen/qwen3.7-flash"
N_ITEMS = 6
CONCURRENCY = 3
TIMEOUT_S = 240

BREVITY_TAIL = (
    "\n\nIMPORTANT: think briefly. Keep <reasoning> to at most 40 words -- state "
    "only the decisive feature change. Do NOT enumerate every state."
)

CONFIGS = [
    ("baseline", {}, ""),
    ("enabled_false", {"reasoning": {"enabled": False}}, ""),
    ("effort_low", {"reasoning": {"effort": "low"}}, ""),
    ("native_off", {"enable_thinking": False}, ""),
    ("brevity", {}, BREVITY_TAIL),
]


async def one_call(client, prompt, choices, body_extra, tail):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt + tail}],
        "provider": {"order": ["alibaba"], "allow_fallbacks": False},
        "usage": {"include": True},
        **body_extra,
    }
    t0 = time.perf_counter()
    try:
        r = await client.post(URL, json=body)
        wall = time.perf_counter() - t0
        d = r.json()
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}", "wall_s": round(time.perf_counter() - t0, 2)}
    if "error" in d:
        return {"error": str(d["error"])[:300], "wall_s": round(wall, 2)}
    msg = d["choices"][0]["message"]
    usage = d.get("usage", {})
    details = usage.get("completion_tokens_details") or {}
    return {
        "wall_s": round(wall, 2),
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "reasoning_tokens": details.get("reasoning_tokens", 0),
        "cost": usage.get("cost", 0.0),
        "visible_chars": len(msg.get("content") or ""),
        "api_reasoning_chars": len(msg.get("reasoning") or ""),
        "valid": bool(_extract_action_set(msg.get("content") or "", choices)),
    }


async def main() -> None:
    prompts = build_prompts()[:N_ITEMS]
    key = os.environ["OPENROUTER_API_KEY"]
    out_path = ROOT / "logs/aug7_qwen_speed_probe/results_round2.json"
    results = {}
    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {key}"}, timeout=TIMEOUT_S
    ) as client:
        for name, body_extra, tail in CONFIGS:
            sem = asyncio.Semaphore(CONCURRENCY)

            async def guarded(p, c):
                async with sem:
                    return await one_call(client, p, c, body_extra, tail)

            t0 = time.perf_counter()
            rows = await asyncio.gather(*(guarded(p, c) for p, c in prompts))
            batch = time.perf_counter() - t0
            results[name] = {"body_extra": body_extra, "tail": bool(tail),
                             "batch_wall_s": round(batch, 1), "calls": list(rows)}
            ok = [r for r in rows if "error" not in r]
            for r in rows:
                if "error" in r:
                    print(f"[{name}] ERROR after {r['wall_s']}s: {r['error']}", flush=True)
            if ok:
                walls = sorted(r["wall_s"] for r in ok)
                print(
                    f"[{name:<13}] p50={walls[len(walls)//2]:5.1f}s max={walls[-1]:5.1f}s "
                    f"out_tok={statistics.mean(r['output_tokens'] for r in ok):6.0f} "
                    f"reason_tok={statistics.mean(r['reasoning_tokens'] for r in ok):6.0f} "
                    f"vis_chars={statistics.mean(r['visible_chars'] for r in ok):5.0f} "
                    f"api_reason_chars={statistics.mean(r['api_reasoning_chars'] for r in ok):6.0f} "
                    f"cost=${statistics.mean(r['cost'] for r in ok):.5f} "
                    f"valid={sum(r['valid'] for r in ok)}/{len(rows)}",
                    flush=True,
                )
            out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nresults -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
