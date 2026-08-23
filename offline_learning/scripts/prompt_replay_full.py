"""Arm C of the prompt replay: full-pipeline test. Take the NEW-template reflection
prompts and replace the embedded OLD analyzer text (parsed from the historical
analysis_calls response, which is what was injected) with the NEW analyzer outputs
produced by prompt_replay_ab.py -- i.e., what the reflector would actually see with
both prompt changes live. Handles raw and JSON-escaped occurrences.
"""

from __future__ import annotations

import os as _bos, sys as _bsys  # offline_learning/ on sys.path (flat-import the kept libs)
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv

load_dotenv()
import os

os.environ.setdefault("OPENROUTER_PROVIDER_ORDER", "cerebras,groq,sambanova")

from explore.mixed_improve import _llm_call, run_async  # noqa: E402
from validate import make_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs/aug3_prompt_replay"
BASE = ROOT / "logs/jul30_idcfd/bt3gb_seed1/gepa_run_seed1"
S30 = ROOT / "logs/aug2_valab/bt3gb_strat30_seed1/gepa_run_seed1"
T60 = ROOT / "logs/aug2_valab/bt3gb_tied60_seed1/gepa_run_seed1"

CASES = [  # the reflection replays where the new template alone did not flip the outcome
    ("base36_offscreen", BASE, 36),
    ("base48_truthinv", BASE, 48),
    ("s30_16_wronginduction", S30, 16),
    ("s30_28_retainverbatim", S30, 28),
    ("t60_20_frameshift", T60, 20),
]


def parse_tags(text: str) -> dict[str, str]:
    out = {}
    for m in re.finditer(r"<(m\d+_\w+|common_root_causes)>(.*?)</\1>", text, re.S):
        out[m.group(1)] = m.group(2).strip()
    return out


def splice(prompt: str, old_az: dict, new_az: dict) -> tuple[str, int, int]:
    hits = misses = 0
    for key, old_fb in old_az.items():
        new_fb = new_az.get(key)
        if not new_fb or not old_fb:
            continue
        replaced = False
        for enc in (lambda s: s, lambda s: json.dumps(s)[1:-1]):
            o = enc(old_fb)
            if o in prompt:
                prompt = prompt.replace(o, enc(new_fb))
                replaced = True
                break
        hits += replaced
        misses += not replaced
    return prompt, hits, misses


def load_analysis(run_dir: Path, iteration: int) -> dict:
    with (run_dir / "analysis_calls.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if r["iteration"] == iteration and r["component"] == "world_knowledge":
                return r
    raise KeyError(iteration)


async def main() -> None:
    cfg = make_config("openai/gpt-oss-120b", "openrouter")
    sem = asyncio.Semaphore(8)
    jobs = []

    async def one(name: str, prompt: str):
        async with sem:
            text, cost = await _llm_call(cfg, prompt)
        (OUT / f"refl_{name}__full.md").write_text(text or "(EMPTY RESPONSE)")
        print(f"done refl_{name}__full  ${cost:.4f}  {len(text or '')} ch", flush=True)

    for name, run_dir, iteration in CASES:
        new_prompt = (OUT / "prompts" / f"refl_{name}__new.txt").read_text()
        old_az = parse_tags(load_analysis(run_dir, iteration)["response"])
        new_az = parse_tags((OUT / f"az_{name}__new.md").read_text())
        spliced, hits, misses = splice(new_prompt, old_az, new_az)
        print(f"{name}: spliced {hits} analysis blocks ({misses} not found in prompt)")
        (OUT / "prompts" / f"refl_{name}__full.txt").write_text(spliced)
        jobs.append(one(name, spliced))

    await asyncio.gather(*jobs)


if __name__ == "__main__":
    run_async(main())
