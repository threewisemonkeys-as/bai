"""Stage-A probe for the REFLECTION role: can anything on OpenRouter beat
deepseek-v4-flash (pinned deepseek,baidu,fireworks) on the proposer/analysis calls?

The reflection role is the wall-clock bottleneck of a rex_pure run: per node the
loop does ONE analysis call then ONE proposer call, both serial and both blocking
the next eval. In logs/aug8_hardmin_gptoss20b the proposer prompt is ~80k chars
(~22k tokens) in / ~2k tokens out, and the observed analysis->propose gap is ~80s
median -- i.e. output-token-bound at the ~30-70 tok/s dsv4flash serves.

This replays the REAL logged prompts (proposer-perception, proposer-beliefs,
analysis) from that run against candidate model/provider/effort combos and
records wall, tokens, cost, routed provider, and a role-specific validity gate:

  perception : fenced block -> _clean_component -> exec -> perceive() must exist,
               run without raising on N real observations of that game, return
               non-empty <=2000-char strings, and NOT be constant across states
               (the constant-P failure the run's own gate fires on).
  beliefs    : non-empty extracted text within a sane length band.
  analysis   : <common_root_causes> + every per-mistake <mN_kind> tag the caller
               parses back out.

Survivors graduate to a real A/B arm (same launcher pattern as
launch_aug7_altmodels_bench.py, swapping only --reflection-model/-provider-order).

Results -> logs/aug8_reflection_probe/results.json (+ a printed table).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "offline_learning"))

import httpx  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from invdyn_core import _clean_component, extract_proposed_text  # noqa: E402
from validate import load_transitions, run_perceive  # noqa: E402

URL = "https://openrouter.ai/api/v1/chat/completions"
RUN_ROOT = ROOT / "logs/aug8_hardmin_gptoss20b"
OUT = ROOT / "logs/aug8_reflection_probe"
# A reflection call slower than this is useless for the role no matter how good it is,
# and an unbounded wait lets one stalled endpoint hold the whole sweep hostage (the first
# run of this script sat 77 min on 3 hung arms with nothing written).
TIMEOUT_S = 240
CANDIDATE_DEADLINE_S = 900  # hard cap per candidate; partial rows are still reported

NOTHINK = {"reasoning": {"enabled": False}}
EFFNONE = {"reasoning": {"effort": "none"}}
EFFLOW = {"reasoning": {"effort": "low"}}
EFFMED = {"reasoning": {"effort": "medium"}}

# (label, model, provider order (None = unpinned), body extra)
# Provider pins chosen from the 2026-08-08 OpenRouter endpoint scan (p50 throughput).
CANDIDATES = [
    ("BASELINE dsv4flash (deepseek,baidu,fireworks)", "deepseek/deepseek-v4-flash",
     ["deepseek", "baidu", "fireworks"], {}),
    ("dsv4flash @baidu,parasail", "deepseek/deepseek-v4-flash",
     ["baidu", "parasail"], {}),
    ("gptoss120b @cerebras +med", "openai/gpt-oss-120b",
     ["cerebras", "groq", "sambanova"], EFFMED),
    ("gptoss120b @cerebras +low", "openai/gpt-oss-120b",
     ["cerebras", "groq", "sambanova"], EFFLOW),
    ("ling30flash @novita nothink", "inclusionai/ling-3.0-flash",
     ["novita", "deepinfra"], NOTHINK),
    ("qwen37flash effnone", "qwen/qwen3.7-flash", None, EFFNONE),
    ("glm52 @coreweave", "z-ai/glm-5.2", ["coreweave", "friendli"], {}),
    ("minimax-m2.5 @friendli", "minimax/minimax-m2.5", ["friendli", "inceptron"], {}),
    ("gpt56luna +low", "openai/gpt-5.6-luna", None, EFFLOW),
    ("laguna-s-2.1", "poolside/laguna-s-2.1", None, {}),
    ("kimi-k2.7-code @modelrun", "moonshotai/kimi-k2.7-code",
     ["modelrun", "coreweave"], {}),
    ("qwen3-coder-next @ionstream", "qwen/qwen3-coder-next", ["ionstream"], {}),
    ("mimo-v2.5 @parasail", "xiaomi/mimo-v2.5", ["parasail", "venice"], {}),
    ("seed-2.0-mini", "bytedance-seed/seed-2.0-mini", None, {}),
    # The baseline's latency turned out to be 76-97% hidden reasoning tokens (10.6k of
    # them to emit a ~320-token diagnosis), so the same model with thinking OFF is the
    # cheapest possible fix -- no model change, one flag.
    ("dsv4flashNT nothink", "deepseek/deepseek-v4-flash",
     ["deepseek", "baidu", "fireworks"], NOTHINK),
    ("seed20miniNT nothink", "bytedance-seed/seed-2.0-mini", None, NOTHINK),
    # --- "expensive but fast" tier: thinking stays ON, we buy throughput instead. The
    # aug9 e2e run showed thinking-OFF costs -0.180 test acc, so the target is a model that
    # does the reasoning FAST, not one that skips it. Picked from the 2026-08-09 no-price-cap
    # endpoint scan (>=150 tok/s p50, uptime>=95%).
    ("FAST gptoss120b @cerebras +high", "openai/gpt-oss-120b",
     ["cerebras", "groq"], {"reasoning": {"effort": "high"}}),
    ("FAST gptoss120b @cerebras +med", "openai/gpt-oss-120b",
     ["cerebras", "groq"], EFFMED),
    ("FAST qwen3-32b @groq", "qwen/qwen3-32b", ["groq"], {}),
    ("FAST minimax-m2.7 @groq", "minimax/minimax-m2.7", ["groq"], {}),
    ("FAST kimi-k2.6 @coreweave", "moonshotai/kimi-k2.6", ["coreweave", "together"], {}),
    ("FAST gpt-5.6-luna-pro", "openai/gpt-5.6-luna-pro", None, EFFMED),
    ("FAST mercury-2", "inception/mercury-2", ["inception"], {}),
]

# game -> train dirs holding the raw observations perceive() is validated on
GAME_OBS = {
    "bt3gb": ROOT / "offline_learning/clean_data3/bt3gb/train",
    "dq8gc": ROOT / "offline_learning/clean_data3/dq8gc/train",
    "n2ntd": ROOT / "offline_learning/clean_data3/n2ntd/train",
}
N_OBS = 8  # observations perceive() is exercised on


# --------------------------------------------------------------------------- prompts
def _load_calls(game: str, fname: str) -> list[dict]:
    p = RUN_ROOT / f"{game}_seed1/rexpure_run_seed1/{fname}"
    return [json.loads(l) for l in p.open()] if p.exists() else []


def build_prompt_set(games: list[str]) -> list[dict]:
    """One perception + one beliefs proposer prompt + one analysis prompt per game,
    taken from mid-run (where the parent already has real content)."""
    out = []
    for g in games:
        refl = _load_calls(g, "reflection_calls.jsonl")
        ana = _load_calls(g, "analysis_calls.jsonl")
        perc = [c for c in refl if c["component"] == "perception" and c["response"]]
        wk = [c for c in refl if c["component"] == "world_knowledge" and c["response"]]
        if perc:
            out.append({"id": f"{g}/perception", "game": g, "role": "perception",
                        "prompt": perc[len(perc) // 2]["prompt"]})
        if wk:
            out.append({"id": f"{g}/beliefs", "game": g, "role": "beliefs",
                        "prompt": wk[len(wk) // 2]["prompt"]})
        if ana:
            out.append({"id": f"{g}/analysis", "game": g, "role": "analysis",
                        "prompt": ana[len(ana) // 2]["prompt"]})
    return out


# --------------------------------------------------------------------------- validity
_OBS_CACHE: dict[str, list[str]] = {}


def sample_obs(game: str) -> list[str]:
    if game not in _OBS_CACHE:
        d = GAME_OBS.get(game)
        trs = load_transitions([d], None) if d and d.exists() else []
        _OBS_CACHE[game] = [t.x_t for t in trs[:N_OBS]] or []
    return _OBS_CACHE[game]


def check_perception(text: str, game: str) -> dict:
    code = _clean_component("perception", extract_proposed_text(text or ""))
    if not code.strip():
        return {"valid": False, "why": "no code extracted"}
    obs = sample_obs(game)
    if not obs:
        return {"valid": False, "why": "no sample observations on disk"}
    outs, err = [], None
    for o in obs:
        z, e = run_perceive(code, o)
        if e:
            err = e
            break
        outs.append(z)
    if err:
        return {"valid": False, "why": f"perceive raised: {err[:120]}"}
    if any(not z.strip() for z in outs):
        return {"valid": False, "why": "perceive returned empty"}
    uniq = len(set(outs))
    if uniq < 2:
        return {"valid": False, "why": "constant across states (would trip the P gate)"}
    return {"valid": True, "why": "", "uniq_out": uniq, "n_obs": len(outs),
            "max_len": max(len(z) for z in outs),
            "over_2000": sum(1 for z in outs if len(z) > 2000)}


def check_beliefs(text: str) -> dict:
    t = _clean_component("world_knowledge", extract_proposed_text(text or "")).strip()
    if not t:
        return {"valid": False, "why": "empty proposal"}
    if len(t) < 80:
        return {"valid": False, "why": f"too short ({len(t)} chars)"}
    return {"valid": True, "why": "", "chars": len(t)}


def check_analysis(text: str, prompt: str) -> dict:
    t = text or ""
    want = set(re.findall(r"<(m\d+_(?:inv|fwd))>", prompt))
    if "<common_root_causes>" not in t:
        return {"valid": False, "why": "no <common_root_causes>"}
    missing = [w for w in want if f"<{w}>" not in t]
    if missing:
        return {"valid": False, "why": f"{len(missing)}/{len(want)} per-mistake tags missing"}
    return {"valid": True, "why": "", "tags": len(want)}


def validate(role: str, text: str, item: dict) -> dict:
    if role == "perception":
        return check_perception(text, item["game"])
    if role == "beliefs":
        return check_beliefs(text)
    return check_analysis(text, item["prompt"])


# --------------------------------------------------------------------------- calls
async def one_call(client, item, model, order, extra) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": item["prompt"]}],
            "usage": {"include": True}, **extra}
    if order:
        body["provider"] = {"order": order, "allow_fallbacks": False}
    dropped = False
    for attempt in (1, 2):
        t0 = time.perf_counter()
        try:
            r = await client.post(URL, json=body)
            d = r.json()
        except Exception as e:  # noqa: BLE001
            return {"id": item["id"], "role": item["role"], "error": f"{type(e).__name__}: {e}",
                    "wall_s": round(time.perf_counter() - t0, 2)}
        wall = time.perf_counter() - t0
        if "error" in d or "choices" not in d:
            if attempt == 1 and "reasoning" in body:  # some endpoints 400 on it
                body.pop("reasoning")
                dropped = True
                continue
            return {"id": item["id"], "role": item["role"],
                    "error": str(d.get("error", d))[:200], "wall_s": round(wall, 2)}
        msg = d["choices"][0]["message"]
        usage = d.get("usage", {}) or {}
        det = usage.get("completion_tokens_details") or {}
        text = msg.get("content") or ""
        v = validate(item["role"], text, item)
        return {"id": item["id"], "role": item["role"], "wall_s": round(wall, 2),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "reasoning_tokens": det.get("reasoning_tokens", 0) or 0,
                "cost": usage.get("cost", 0.0), "provider": d.get("provider"),
                "dropped_reasoning": dropped, "resp_chars": len(text), **v}


async def run_candidate(client, label, model, order, extra, items, reps) -> list[dict]:
    """Prompts are issued SEQUENTIALLY so each wall is a clean single-call latency
    (the loop's reflection calls are serial too). Bounded by CANDIDATE_DEADLINE_S: a
    candidate that blows it reports the rows it did finish instead of stalling the sweep."""
    rows = []
    t0 = time.perf_counter()
    for _ in range(reps):
        for it in items:
            if time.perf_counter() - t0 > CANDIDATE_DEADLINE_S:
                rows.append({"id": it["id"], "role": it["role"],
                             "error": f"candidate deadline {CANDIDATE_DEADLINE_S}s exceeded"})
                continue
            rows.append(await one_call(client, it, model, order, extra))
    return rows


def summarize(label, model, order, rows) -> dict:
    ok = [r for r in rows if "error" not in r]
    walls = [r["wall_s"] for r in ok]
    per_role = {}
    for role in ("perception", "beliefs", "analysis"):
        rr = [r for r in ok if r["role"] == role]
        if rr:
            per_role[role] = {
                "n": len(rr),
                "wall_med": round(statistics.median(r["wall_s"] for r in rr), 1),
                "valid": sum(1 for r in rr if r.get("valid")),
                "out_tok_med": int(statistics.median(r["output_tokens"] for r in rr)),
                "reason_tok_med": int(statistics.median(r["reasoning_tokens"] for r in rr)),
            }
    tps = [r["output_tokens"] / r["wall_s"] for r in ok if r["wall_s"] > 0 and r["output_tokens"]]
    return {
        "label": label, "model": model, "provider_order": order,
        "n_calls": len(rows), "n_errors": len(rows) - len(ok),
        "wall_med": round(statistics.median(walls), 1) if walls else None,
        "wall_mean": round(statistics.mean(walls), 1) if walls else None,
        "wall_total": round(sum(walls), 1),
        "cost_total": round(sum(r.get("cost", 0.0) for r in ok), 5),
        "cost_per_call": round(sum(r.get("cost", 0.0) for r in ok) / max(1, len(ok)), 5),
        "valid": sum(1 for r in ok if r.get("valid")), "n_ok": len(ok),
        "tok_per_s_med": round(statistics.median(tps), 1) if tps else None,
        "providers": sorted({str(r.get("provider")) for r in ok}),
        "per_role": per_role,
        "fail_reasons": [f"{r['id']}: {r.get('why')}" for r in ok if not r.get("valid")][:6],
        "errors": [r.get("error", "")[:120] for r in rows if "error" in r][:4],
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="bt3gb,dq8gc,n2ntd")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--only", default=None, help="substring filter over candidate labels")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    items = build_prompt_set(args.games.split(","))
    print(f"[probe] {len(items)} prompts x {args.reps} rep(s):")
    for it in items:
        print(f"   {it['id']:22s} {len(it['prompt']):7d} chars")

    cands = [c for c in CANDIDATES if not args.only or args.only.lower() in c[0].lower()]
    outd = Path(args.out)
    outd.mkdir(parents=True, exist_ok=True)
    key = os.environ["OPENROUTER_API_KEY"]

    summaries, raw = [], {}
    prompts_meta = [{k: v for k, v in i.items() if k != "prompt"} for i in items]
    part = outd / "results.partial.jsonl"
    part.unlink(missing_ok=True)

    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {key}"},
                                 timeout=TIMEOUT_S) as client:
        async def one_arm(lab, m, o, e):
            """Run + PERSIST one candidate the moment it finishes, so a slow arm can never
            cost us the arms that already completed."""
            try:
                rows = await run_candidate(client, lab, m, o, e, items, args.reps)
                summary = summarize(lab, m, o, rows)
            except Exception as exc:  # noqa: BLE001
                rows, summary = [], {"label": lab, "model": m, "fatal": str(exc)[:200]}
            raw[lab] = rows
            summaries.append(summary)
            with part.open("a") as f:
                f.write(json.dumps(summary) + "\n")
            print(f"[done] {lab}: wall_med={summary.get('wall_med')} "
                  f"valid={summary.get('valid')}/{summary.get('n_ok')}", flush=True)

        # candidates run CONCURRENTLY (distinct providers); prompts within one serial.
        await asyncio.gather(*[one_arm(lab, m, o, e) for lab, m, o, e in cands])

    (outd / "results.json").write_text(json.dumps(
        {"prompts": prompts_meta, "summaries": summaries}, indent=2) + "\n")
    (outd / "raw.json").write_text(json.dumps(raw, indent=2)[:20_000_000] + "\n")

    ok = [s for s in summaries if s.get("wall_med") is not None]
    ok.sort(key=lambda s: s["wall_med"])
    print(f"\n{'candidate':46s} {'wall_med':>8s} {'tok/s':>7s} {'valid':>7s} {'$/call':>8s}  provider")
    for s in ok:
        print(f"{s['label'][:46]:46s} {s['wall_med']:8.1f} {s['tok_per_s_med'] or 0:7.1f} "
              f"{s['valid']:3d}/{s['n_ok']:<3d} {s['cost_per_call']:8.5f}  {','.join(s['providers'])[:28]}")
        for f in s["fail_reasons"]:
            print(f"      X {f}")
        for e in s["errors"]:
            print(f"      ! {e}")
    for s in summaries:
        if s.get("wall_med") is None:
            print(f"{s['label'][:46]:46s}   FAILED  {s.get('fatal') or s.get('errors')}")
    print(f"\n-> {outd/'results.json'}")


if __name__ == "__main__":
    asyncio.run(main())
