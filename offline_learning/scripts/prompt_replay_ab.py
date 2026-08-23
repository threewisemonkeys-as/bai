"""Replay crucial wk reflection/analysis prompts from the three bt3gb runs under the
NEW templates (falsification-first analyzer + audit/tabulate/rewrite reflection) and
compare against the OLD prompts, to test whether the prompt changes flip the observed
failure modes (fossil retention, epicycles, truth-inversion, analyzer capture).

For each instance we run TWO arms through the same _llm_call plumbing the runs used
(gpt-oss-120b via OpenRouter, cerebras/groq/sambanova pin):
  old  -- the stored prompt verbatim (fresh sample = sampling-noise control; the
          historical response is also dumped alongside as *_hist.md)
  new  -- the stored prompt with the template text surgically replaced by the NEW
          template text imported from gepa_optimize (assertions guarantee each
          replacement fires exactly once, so template drift errors out loudly).

Outputs: logs/aug3_prompt_replay/<instance>__{old,new,hist}.md (+ prompts/ for audit).
"""

from __future__ import annotations

import os as _bos, sys as _bsys  # offline_learning/ on sys.path (flat-import the kept libs)
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("OPENROUTER_PROVIDER_ORDER", "cerebras,groq,sambanova")

import invdyn_core as go  # noqa: E402  (new templates live here)
from explore.mixed_improve import _llm_call, run_async  # noqa: E402
from validate import make_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs/aug3_prompt_replay"
BASE = ROOT / "logs/jul30_idcfd/bt3gb_seed1/gepa_run_seed1"
S30 = ROOT / "logs/aug2_valab/bt3gb_strat30_seed1/gepa_run_seed1"
T60 = ROOT / "logs/aug2_valab/bt3gb_tied60_seed1/gepa_run_seed1"

MODEL = "openai/gpt-oss-120b"
WK_TARGET = "the WORLD KNOWLEDGE block (its dynamics / convention rules)"

# (name, run_dir, iteration) -- wk reflection calls to replay
REFLECTION = [
    ("base32_antirule", BASE, 32),
    ("base36_offscreen", BASE, 36),
    ("base48_truthinv", BASE, 48),
    ("s30_16_wronginduction", S30, 16),
    ("s30_28_retainverbatim", S30, 28),
    ("s30_32_gravitycorrupt", S30, 32),
    ("t60_20_frameshift", T60, 20),
    ("t60_28_success_regression", T60, 28),
]
# (name, run_dir, iteration) -- combined wk analysis calls to replay
ANALYZER = [
    ("base32_antirule", BASE, 32),
    ("base36_offscreen", BASE, 36),
    ("base48_truthinv", BASE, 48),
    ("s30_16_wronginduction", S30, 16),
    ("s30_28_retainverbatim", S30, 28),
    ("s30_32_gravitycorrupt", S30, 32),
    ("t60_20_frameshift", T60, 20),
    ("t60_28_success_regression", T60, 28),
]

# ---- OLD template fragments as they appear verbatim in the stored prompts ----
OLD_REFL_TAIL = (
    "Rewrite the FULL world knowledge block: concise, general, sufficient to map "
    "feature changes -- across the whole shown window, not just the center pair -- "
    "to action names. Provide it within ``` blocks."
)
OLD_AZ_WK_HDR = (
    "=== WORLD KNOWLEDGE (general facts the predictors were given; may be empty) ==="
)
OLD_AZ_DIAG = (
    "Diagnose the ROOT CAUSE(S) and give specific, GENERAL feedback to improve "
    f"{WK_TARGET} so these classes of mistake are avoided. First identify any pattern "
    "shared across several mistakes, then give one concrete, actionable fix per "
    "mistake (do not merely restate the prediction)."
)


def _slice(text: str, start_marker: str, end_marker: str | None = None) -> str:
    i = text.find(start_marker)
    assert i >= 0, f"marker not found: {start_marker[:60]!r}"
    if end_marker is None:
        return text[i:]
    j = text.find(end_marker, i)
    assert j > i, f"end marker not found: {end_marker[:60]!r}"
    return text[i:j]


# ---- NEW fragments pulled from the live templates (no drift possible) ----
NEW_REFL_TAIL = _slice(
    go.REFLECTION_TEMPLATES["world_knowledge"], "The current block was written"
)
assert NEW_REFL_TAIL.endswith("motivating it."), NEW_REFL_TAIL[-80:]

NEW_AZ_TRUTH = _slice(
    go.ANALYZE_COMBINED_TMPL, "The TRUE labels below", "\n\n=== DEFAULT KNOWLEDGE"
)
NEW_AZ_WK_HDR = _slice(
    go.ANALYZE_COMBINED_TMPL, "=== WORLD KNOWLEDGE (general facts", "\n{beliefs}"
)
NEW_AZ_DIAG = _slice(
    go.ANALYZE_COMBINED_TMPL, "Diagnose in this order:", "\n\nRespond in EXACTLY"
).replace("{target}", WK_TARGET)


def _replace_once(text: str, old: str, new: str, what: str) -> str:
    n = text.count(old)
    assert n == 1, f"{what}: expected exactly 1 occurrence, found {n}"
    return text.replace(old, new)


def new_reflection_prompt(old_prompt: str) -> str:
    return _replace_once(old_prompt, OLD_REFL_TAIL, NEW_REFL_TAIL, "reflection tail")


def new_analyzer_prompt(old_prompt: str) -> str:
    p = _replace_once(
        old_prompt,
        "\n\n=== DEFAULT KNOWLEDGE",
        f"\n\n{NEW_AZ_TRUTH}\n\n=== DEFAULT KNOWLEDGE",
        "truth para insert",
    )
    p = _replace_once(p, OLD_AZ_WK_HDR, NEW_AZ_WK_HDR, "wk header")
    p = _replace_once(p, OLD_AZ_DIAG, NEW_AZ_DIAG, "diagnose para")
    return p


def load_reflection(run_dir: Path, iteration: int) -> dict:
    with (run_dir / "reflection_calls.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if r["call"] == iteration and r["component"] == "world_knowledge":
                return r
    raise KeyError(f"no wk reflection call {iteration} in {run_dir}")


def load_analysis(run_dir: Path, iteration: int) -> dict:
    with (run_dir / "analysis_calls.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if r["iteration"] == iteration and r["component"] == "world_knowledge":
                return r
    raise KeyError(f"no wk analysis call {iteration} in {run_dir}")


async def run_all() -> None:
    cfg = make_config(MODEL, "openrouter")
    sem = asyncio.Semaphore(8)
    (OUT / "prompts").mkdir(parents=True, exist_ok=True)
    jobs = []

    async def one(name: str, arm: str, prompt: str):
        async with sem:
            text, cost = await _llm_call(cfg, prompt)
        (OUT / f"{name}__{arm}.md").write_text(text or "(EMPTY RESPONSE)")
        print(f"done {name}__{arm}  ${cost:.4f}  {len(text or '')} ch", flush=True)

    for kind, roster, loader, transform in [
        ("refl", REFLECTION, load_reflection, new_reflection_prompt),
        ("az", ANALYZER, load_analysis, new_analyzer_prompt),
    ]:
        for name, run_dir, iteration in roster:
            rec = loader(run_dir, iteration)
            full = f"{kind}_{name}"
            old_prompt = rec["prompt"]
            new_prompt = transform(old_prompt)
            (OUT / "prompts" / f"{full}__old.txt").write_text(old_prompt)
            (OUT / "prompts" / f"{full}__new.txt").write_text(new_prompt)
            (OUT / f"{full}__hist.md").write_text(rec["response"] or "")
            jobs.append(one(full, "old", old_prompt))
            jobs.append(one(full, "new", new_prompt))

    await asyncio.gather(*jobs)


if __name__ == "__main__":
    run_async(run_all())
    print(f"all replays written to {OUT}")
