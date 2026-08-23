"""Why is qwen3.7-flash slow on the eval calls, and can OpenRouter serve it faster?

Context (aug6 evalmodel bench): qwen3.7-flash @ alibaba matched gpt-oss-120b on
test50 ID quality at -75% cost but took 744s vs 51s wall, with the 30s hedge firing
on 88/210 requests. Alibaba is the ONLY host (no routing escape, no variant slugs),
prompts are ~2k tokens (prefill can't explain 30s), and the model is hybrid-thinking
with `reasoning` in supported_parameters -- so the levers to probe are:

  default   -- exactly yesterday's config (thinking at provider default)
  repeat    -- same prompts again (Alibaba implicit caching: cache_read $0.006/M)
  nothink   -- reasoning {"enabled": false}
  lowthink  -- reasoning {"effort": "low"}

Prompts are REAL windowed set-ID decode prompts: the aug6 qwen arm's test-trace
features + shipped beliefs fed through invdyn_core's _inverse_transcript /
INV_WINDOW_SET_TMPL, so length and shape match production eval calls.

Measures per call: wall s, input/output/reasoning/cached tokens, OpenRouter-billed
cost, parseable <actions> answer. Results -> logs/aug7_qwen_speed_probe/.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "offline_learning"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import litellm  # noqa: E402

from invdyn_core import (  # noqa: E402
    DEFAULT_KNOWLEDGE,
    INV_WINDOW_SET_TMPL,
    _extract_action_set,
    _inverse_transcript,
)
from explore.mixed_improve import (  # noqa: E402
    _extract_cached_tokens,
    _get_response_cost,
    build_llm_input,
    extract_llm_response_text,
)

MODEL = "openrouter/qwen/qwen3.7-flash"
PIN = {"order": ["alibaba"], "allow_fallbacks": False}
TRACE = ROOT / "logs/aug6_evalmodel_bench/qwen37flash_alibaba/test_trace_gepa_seed1.json"
OUT = ROOT / "logs/aug7_qwen_speed_probe"
N_ITEMS = 10
CONCURRENCY = 5
TIMEOUT_S = 240

CONFIGS = [
    ("default", None),
    ("repeat", None),  # same prompts as `default` -> implicit-cache probe
    ("nothink", {"enabled": False}),
    ("lowthink", {"effort": "low"}),
]


def build_prompts() -> list[tuple[str, list[str]]]:
    t = json.loads(TRACE.read_text())
    recs, beliefs = t["records"], t["beliefs"]
    pool = sorted({a for r in recs for a in r["choices"]})
    prompts = []
    for i in range(N_ITEMS):
        r = recs[i * len(recs) // N_ITEMS]
        # 9 prev / 8 next context states from neighboring records' real features,
        # real actions cycled from the run's action pool (K=9 production shape).
        prev = [
            (recs[(i + j) % len(recs)]["z_t"], pool[(i + j) % len(pool)])
            for j in range(9)
        ]
        nxt = [
            (pool[(i + j + 3) % len(pool)], recs[(i + j) % len(recs)]["z_t1"])
            for j in range(8)
        ]
        win = {"prev": prev, "z_t": r["z_t"], "z_t1": r["z_t1"], "nxt": nxt}
        prompt = INV_WINDOW_SET_TMPL.format(
            beliefs=beliefs.strip() or "(empty)",
            default_knowledge=DEFAULT_KNOWLEDGE,
            transcript=_inverse_transcript(win),
            choices="\n".join(f"- {c}" for c in r["choices"]),
        )
        prompts.append((prompt, r["choices"]))
    return prompts


def _usage_int(usage, *path) -> int:
    cur = usage
    for key in path:
        nxt = getattr(cur, key, None)
        if nxt is None and isinstance(cur, dict):
            nxt = cur.get(key)
        if nxt is None:
            return 0
        cur = nxt
    return int(cur or 0)


async def one_call(prompt: str, choices: list[str], reasoning: dict | None, sem) -> dict:
    extra_body: dict = {"usage": {"include": True}}
    if reasoning is not None:
        extra_body["reasoning"] = reasoning
    async with sem:
        t0 = time.perf_counter()
        try:
            resp = await litellm.aresponses(
                model=MODEL,
                input=build_llm_input(prompt),
                timeout=TIMEOUT_S,
                provider=PIN,
                extra_body=extra_body,
            )
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}", "wall_s": round(time.perf_counter() - t0, 2)}
        wall = time.perf_counter() - t0
    text = extract_llm_response_text(resp) or ""
    usage = getattr(resp, "usage", None)
    return {
        "wall_s": round(wall, 2),
        "input_tokens": _usage_int(usage, "input_tokens"),
        "output_tokens": _usage_int(usage, "output_tokens"),
        "reasoning_tokens": _usage_int(usage, "output_tokens_details", "reasoning_tokens"),
        "cached_tokens": _extract_cached_tokens(usage) if usage is not None else 0,
        "cost": _get_response_cost(resp, MODEL.removeprefix("openrouter/")),
        "visible_chars": len(text),
        "valid": bool(_extract_action_set(text, choices)),
    }


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prompts = build_prompts()
    plens = [len(p) for p, _ in prompts]
    print(f"{len(prompts)} prompts, chars min/med/max = "
          f"{min(plens)}/{sorted(plens)[len(plens)//2]}/{max(plens)}")
    results: dict[str, list[dict]] = {}
    for name, reasoning in CONFIGS:
        sem = asyncio.Semaphore(CONCURRENCY)
        t0 = time.perf_counter()
        rows = await asyncio.gather(
            *(one_call(p, c, reasoning, sem) for p, c in prompts)
        )
        batch_wall = time.perf_counter() - t0
        results[name] = {"reasoning": reasoning, "batch_wall_s": round(batch_wall, 1),
                         "calls": list(rows)}
        ok = [r for r in rows if "error" not in r]
        errs = [r for r in rows if "error" in r]
        if ok:
            walls = sorted(r["wall_s"] for r in ok)
            p50 = walls[len(walls) // 2]
            p90 = walls[min(len(walls) - 1, int(len(walls) * 0.9))]
            print(f"[{name:<9}] batch={batch_wall:5.1f}s p50={p50:5.1f}s p90={p90:5.1f}s "
                  f"out_tok={statistics.mean(r['output_tokens'] for r in ok):6.0f} "
                  f"reason_tok={statistics.mean(r['reasoning_tokens'] for r in ok):6.0f} "
                  f"cached_tok={statistics.mean(r['cached_tokens'] for r in ok):6.0f} "
                  f"cost/call=${statistics.mean(r['cost'] for r in ok):.5f} "
                  f"valid={sum(r['valid'] for r in ok)}/{len(rows)}", flush=True)
        for r in errs:
            print(f"[{name}] ERROR after {r['wall_s']}s: {r['error'][:200]}", flush=True)
        (OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nresults -> {OUT / 'results.json'}")


if __name__ == "__main__":
    asyncio.run(main())
