"""Re-simulate experiment selection at specific steps using a MODIFIED prompt:
we do NOT mention the active experiment at all, and instead present an
anonymized, shuffled list of the candidate *experiments* (the per-question
formulated plans that score_topk already produced and stored in
experiment_scoring_log.json) and ask the model to choose the single best one to
run next.

Context (default knowledge, beliefs, recent history, current state, images) is
reused byte-identically from the step's stored score_topk experiment_prompt, so
the only change vs. the real run is the selection mechanism: free choice among
experiment plans instead of b_diff scoring.
"""
import asyncio
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from omegaconf import OmegaConf
from PIL import Image

from llm_utils import extract_xml_key
from mixed_improve import _llm_call

load_dotenv()

RUN = Path(
    "logs/dev/may29/20260529-114550/eb_learn__arc_agi__gemini-2p5-flash__ft09/"
    "2026-05-29_11-45-53_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn"
)
EP = RUN / "episode_0"
CONFIG = OmegaConf.load(RUN / "config.yaml")
N_SAMPLES = 8
SEED = 7


def section(text: str, start_marker: str, end_marker: str) -> str:
    i = text.index(start_marker)
    j = text.index(end_marker) + len(end_marker)
    return text[i:j]


def build_prompt(step: str):
    el = json.loads((EP / f"step_{step}" / "experiment_log.json").read_text())
    sc = json.loads((EP / f"step_{step}" / "experiment_scoring_log.json").read_text())
    stored = el["experiment_prompt"]

    head = section(
        stored,
        "=== DEFAULT KNOWLEDGE ===",
        "=== END CURRENT STATE (agent has not yet acted) ===",
    )

    # Build the anonymized, shuffled experiment list from scoring candidates.
    cands = [
        {
            "question": c.get("question", ""),
            "plan": (c.get("plan") or "").strip(),
            "b_diff_score": c.get("score"),
            "is_active": c.get("source_index") is None and c.get("kind") == "active"
            or c.get("source_index") is None,
        }
        for c in sc["candidates"]
        if (c.get("plan") or "").strip()
    ]
    rng = random.Random(SEED)
    rng.shuffle(cands)

    lines = []
    for i, c in enumerate(cands, 1):
        lines.append(f"E{i}: {c['plan']}")
    experiments_block = "\n\n".join(lines)

    task = """=== CANDIDATE EXPERIMENTS ===
""" + experiments_block + """
=== END CANDIDATE EXPERIMENTS ===

Below is a list of candidate experiments, each a concrete action plan the agent could execute next from the current state. Choose the SINGLE experiment that is most valuable to run next for understanding how the game works and making progress toward completing a level.

Format your response as:
<think>
Briefly weigh which experiment yields the most decisive evidence for progressing in the game.
</think>
<selected_experiment>E#</selected_experiment>"""

    prompt = (
        "You are choosing the next experiment for an agent learning how a game works.\n\n"
        + head
        + "\n\n"
        + task
    )

    imgs = []
    for rel in el.get("experiment_image_paths", []):
        p = EP / f"step_{step}" / rel
        if p.exists():
            imgs.append(Image.open(p).convert("RGB"))
    return prompt, imgs, cands, el.get("selected_question")


def classify(question: str) -> str:
    q = question.lower()
    if "target" in q and "pattern" in q:
        return "TARGET-PATTERN (decisive)"
    if "central" in q and ("relevant" in q or "target" in q):
        return "CENTRAL-ELEMENT (mechanism)"
    if "pattern" in q and ("level completion" in q or "level progression" in q or "progression or completion" in q):
        return "PATTERN->COMPLETION (mechanism)"
    if "another 3x3 grid" in q or "another grid" in q or "one 3x3 grid cause a change in another" in q:
        return "CROSS-GRID (mechanism)"
    if re.search(r"action1|action2|action3|action4|action5|action7|simple action", q):
        return "ACTION-AVAILABILITY (dead-end)"
    if "undo" in q or "reset" in q or "hint" in q or "tutorial" in q:
        return "UNDO/RESET/HINT"
    if "border" in q:
        return "BORDER/HIGHLIGHT"
    return "OTHER-MECHANIC"


async def main():
    for step in ["009", "012"]:
        prompt, imgs, cands, actual = build_prompt(step)
        print("\n" + "#" * 80)
        print(f"# STEP {step} — SELECT-AMONG-EXPERIMENTS (no active mentioned) | {len(cands)} options, {len(imgs)} imgs")
        print(f"# score_topk ACTUALLY chose: {actual!r}")
        print("#" * 80)
        cat_counter = Counter()
        for s in range(N_SAMPLES):
            resp, cost = await _llm_call(CONFIG, prompt, images=imgs or None)
            m = re.search(r"E(\d+)", extract_xml_key(resp, "selected_experiment") or resp)
            if not m:
                print(f"  [sample {s+1}] <unparsed>")
                continue
            idx = int(m.group(1)) - 1
            if not (0 <= idx < len(cands)):
                print(f"  [sample {s+1}] E{idx+1} out of range")
                continue
            c = cands[idx]
            cat = classify(c["question"])
            cat_counter[cat] += 1
            print(f"  [sample {s+1}] E{idx+1} [{cat}] (b_diff_score={c['b_diff_score']})")
            print(f"             Q: {c['question'][:110]}")
        print(f"\n  >>> category tally: {dict(cat_counter)}")


if __name__ == "__main__":
    asyncio.run(main())
