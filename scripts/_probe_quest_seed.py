"""Standalone probe: does seeding generate_theories with a novel question make
the ensemble cover the cold-freezes-lava mechanism (and thus score high MI),
where the baseline ensemble does not?

Uses a CLEAN minihack-Quest belief state from the 50-step matrix run, which
NEVER observed lava cooling -- so the mechanism is genuinely outside the
ensemble. Generates theories baseline vs. seeded, then scores the test
question by theory-entropy MI for each ensemble.
"""
import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf

import mixed_improve
from theory_exploration import generate_theories, score_questions_theory_entropy
from stepwise_eb_learn_improve import EBQAPair

RUN = (
    "logs/matrix_v5/v5_50steps/eb_learn__minihack__gemini-2p5-flash/"
    "2026-06-03_16-24-58_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn"
)
CLEAN_BELIEFS = f"{RUN}/episode_2/beliefs.txt"
# A theory-gen prompt from the same run, to lift the exact DEFAULT KNOWLEDGE block.
PROMPT_SRC = f"{RUN}/episode_0/step_000/theory_log.json"

# The novel hypothesis the question generator DID produce (anticipatory), phrased
# as a clean binary test question.
TEST_Q = (
    "If the agent zaps a wand of cold at a molten-lava tile (shown as 'F' or '}'), "
    "does that tile become solid and safe to walk on (traversable)?"
)


def load_dk() -> str:
    import json
    d = json.load(open(PROMPT_SRC))
    prompt = d["theories"]["prompt"]
    m = re.search(r"=== DEFAULT KNOWLEDGE ===\n(.*?)\n=== CURRENT WORLD KNOWLEDGE",
                  prompt, re.S)
    return m.group(1).strip() if m else ""


def mentions_cold_lava(text: str) -> bool:
    t = text.lower()
    return ("lava" in t) and any(k in t for k in
            ("freez", "cold", "solidif", "cool", "frost", "obsidian"))


async def run():
    beliefs = open(CLEAN_BELIEFS).read()
    dk = load_dk()
    print(f"clean beliefs: {len(beliefs)} chars | DK: {len(dk)} chars")
    print(f"belief state mentions cold/freeze+lava already? "
          f"{mentions_cold_lava(beliefs)}  (must be False for a fair test)\n")

    config = OmegaConf.create(
        {"client": {"client_name": "openrouter", "model_id": "google/gemini-2.5-flash"}}
    )
    mixed_improve._MOCK_MODE = False
    mixed_improve._META_TEMPERATURE = 0.0

    async def gen(seed):
        return await generate_theories(
            config=config, beliefs=beliefs, default_knowledge=dk,
            steps_context="", current_observation=None, current_image=None,
            num_theories=5, decay=0.6, seed_questions=seed,
        )

    async def score(theories):
        qa = [EBQAPair(question=TEST_Q, answer=None, evidence="", source_step=0)]
        scores, cost, log = await score_questions_theory_entropy(
            config=config, theories=theories, qa_pairs=qa,
            default_knowledge=dk, max_concurrent=5,
        )
        per = log["per_question"][0]
        return per["score"], per["p_yes_per_theory"], cost

    for label, seed in [("BASELINE (no seed)", None),
                        ("SEEDED (test question injected)", [TEST_Q])]:
        theories, gcost, _ = await gen(seed)
        print("=" * 78)
        print(f"{label}  — {len(theories)} theories, gen cost ${gcost:.4f}")
        print("=" * 78)
        for t in theories:
            flag = "  <<< COVERS cold-lava" if mentions_cold_lava(t.world_knowledge) else ""
            head = t.world_knowledge.strip().splitlines()[0][:100]
            print(f"  rank{t.rank} w={t.weight:.2f}{flag}\n      {head}")
        mi, pyes, scost = await score(theories)
        n_yes = sum(1 for p in pyes if p > 0.5)
        print(f"\n  TEST Q: {TEST_Q}")
        print(f"  per-theory P(YES): {[round(p,2) for p in pyes]}  "
              f"({n_yes} theories predict YES)")
        print(f"  >>> MI(test question) = {mi:.4f} bits   (score cost ${scost:.4f})\n")


if __name__ == "__main__":
    asyncio.run(run())
