"""Re-simulate experiment selection for specific steps using the SIMPLE
selection prompt (single-mode `formulate_experiment_from_question`, which lists
all available unanswered questions and lets the LLM pick one to investigate),
instead of the `score_topk` / b_diff scoring that the original run used.

We reuse the *byte-identical* context (default knowledge, beliefs, recent
history, current state, and the exact AVAILABLE UNANSWERED QUESTIONS list) that
was captured in each step's stored score_topk `experiment_prompt`, so the only
thing that changes is the selection mechanism. Images are reloaded from disk.

This is a counterfactual on the original run's question pool: it answers
"given exactly the questions and context the agent had at this step, would the
simple-selection prompt have picked the target-pattern hypothesis that b_diff
scoring buried?"
"""
import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from omegaconf import OmegaConf
from PIL import Image

import explore.mixed_improve as mixed_improve
from explore.mixed_improve import _llm_call

load_dotenv()

RUN = Path(
    "logs/dev/may29/20260529-114550/eb_learn__arc_agi__gemini-2p5-flash__ft09/"
    "2026-05-29_11-45-53_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn"
)
EP = RUN / "episode_0"
CONFIG = OmegaConf.load(RUN / "config.yaml")
N_SAMPLES = 8

# Active experiment/question going INTO each step's formulation (= prior step's
# selected experiment, read from the prior step's step_log.json).
ACTIVE = {
    "009": {
        "question": "Does the specific arrangement or pattern of red and blue squares within a 3x3 grid, or across multiple grids, contribute to level progression or completion?",
        "plan": "Continue the current experiment by systematically clicking the remaining squares of the highlighted bottom-right 3x3 grid using ACTION6, moving across rows (e.g., (48,40), then (56,40), then (40,48), etc.). After each click, observe if the specific pattern of red and blue squares formed contributes to level progression or completion.",
    },
    "012": {
        "question": "Do actions ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, or ACTION7 cause any visible changes to the game state (e.g., border, grid elements, new views)?",
        "plan": "The last attempt to use ACTION1 failed, confirming it and other simple actions are currently unavailable for direct execution. Continue with ACTION6-based experiments to influence the game state, and at each step, carefully check the auxiliary observation for any changes in the availability of ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, or ACTION7.",
    },
}


def section(text: str, start_marker: str, end_marker: str) -> str:
    i = text.index(start_marker)
    j = text.index(end_marker) + len(end_marker)
    return text[i:j]


def build_single_mode_prompt(step: str, with_active: bool = True) -> tuple[str, list]:
    import json

    el = json.loads((EP / f"step_{step}" / "experiment_log.json").read_text())
    stored = el["experiment_prompt"]

    # Reuse the exact context block (DEFAULT KNOWLEDGE -> END CURRENT STATE) and
    # the exact question list, verbatim from the stored score_topk prompt.
    head = section(
        stored,
        "=== DEFAULT KNOWLEDGE ===",
        "=== END CURRENT STATE (agent has not yet acted) ===",
    )
    questions_block = section(
        stored,
        "=== AVAILABLE UNANSWERED QUESTIONS ===",
        "=== END AVAILABLE UNANSWERED QUESTIONS ===",
    )

    # Find the active question's Q-number in the available list, if present.
    act = ACTIVE[step]
    qnum = None
    for line in questions_block.splitlines():
        m = re.match(r"(Q\d+): (.*?) -> UNANSWERED", line)
        if not m:
            continue
        # best-effort match: shared key phrase
        if (
            m.group(2).strip().lower() == act["question"].strip().lower()
            or ("arrangement or pattern" in m.group(2) and "arrangement or pattern" in act["question"])
        ):
            qnum = m.group(1)
            break
    active_q_text = (f"{qnum}: " if qnum else "") + act["question"]
    if with_active:
        active_section = (
            "=== ACTIVE QUESTION AND EXPERIMENT ===\n"
            f"Active question:\n{active_q_text}\n\n"
            "Active question status in current Q&A:\nUNANSWERED\n\n"
            f"Active experiment plan:\n{act['plan']}\n"
            "=== END ACTIVE QUESTION AND EXPERIMENT ==="
        )
    else:
        # Free-selection variant: no active experiment, forcing the LLM to pick
        # one question from the pool (isolates the selection decision).
        active_section = (
            "=== ACTIVE QUESTION AND EXPERIMENT ===\n"
            "(no active question or experiment)\n"
            "=== END ACTIVE QUESTION AND EXPERIMENT ==="
        )

    task_tail = """Your task:
1. Decide whether the active question is already answered.
2. If there is an active experiment and the active question is not answered yet:
   - Return null if the current experiment plan is still the right next experiment.
   - Formulate a revised experiment for the active question if the same question should keep being investigated but the current plan should change.
3. If the active question is answered, or there is no active question, select one question from AVAILABLE UNANSWERED QUESTIONS to investigate next.
4. Any experiment plan must be specific, actionable, 1-3 sentences, and directly aimed at collecting evidence for the question named in the <q> tag.

Format your response as:
<think>
Is the active question answered? Should the current experiment be kept, revised for the same active question, or replaced with an experiment for a newly selected question?
</think>
<experiment>
If keeping the current experiment unchanged:
null

If revising the active question's experiment or formulating a new selected-question experiment:
<q n="Q#">
<experiment_plan>[1-3 sentence actionable experiment to answer the question named in the q tag]</experiment_plan>
</q>
</experiment>"""

    prompt = (
        "You are selecting the next question and designing the associated experiment for learning more about the game being played.\n\n"
        + head
        + "\n\n"
        + active_section
        + "\n\n"
        + questions_block
        + "\n\n"
        + task_tail
    )

    # Reload the exact images used at this step.
    imgs = []
    for rel in el.get("experiment_image_paths", []):
        p = EP / f"step_{step}" / rel
        if p.exists():
            imgs.append(Image.open(p).convert("RGB"))
    return prompt, imgs


def parse_choice(resp: str, questions_block: str) -> tuple[str | None, str | None]:
    body = mixed_improve.extract_xml_key(resp, "experiment") if hasattr(mixed_improve, "extract_xml_key") else None
    from llm_utils import extract_xml_key

    body = extract_xml_key(resp, "experiment") or resp
    if body.strip().lower() == "null":
        return "NULL (keep active experiment)", None
    m = re.search(r'<q\s+n="?(Q\d+)"?\s*>(.*?)</q>', body, re.S)
    if not m:
        return None, None
    qnum = m.group(1)
    plan = extract_xml_key(m.group(2), "experiment_plan") or ""
    # look up the question text
    qtext = ""
    for line in questions_block.splitlines():
        mm = re.match(rf"{qnum}: (.*?) -> UNANSWERED", line)
        if mm:
            qtext = mm.group(1)
            break
    return f"{qnum}: {qtext}", plan.strip()


async def main():
    import json

    for step in ["009", "012"]:
        el = json.loads((EP / f"step_{step}" / "experiment_log.json").read_text())
        questions_block = section(
            el["experiment_prompt"],
            "=== AVAILABLE UNANSWERED QUESTIONS ===",
            "=== END AVAILABLE UNANSWERED QUESTIONS ===",
        )
        print("\n" + "#" * 80)
        print(f"# STEP {step} — SIMPLE-SELECTION re-simulation")
        print(f"# score_topk ACTUALLY chose: {el.get('selected_question')!r}")
        print("#" * 80)
        for with_active in (True, False):
            prompt, imgs = build_single_mode_prompt(step, with_active=with_active)
            label = "A) faithful (active experiment present)" if with_active else "B) free-selection (no active experiment)"
            print(f"\n=== variant {label} | {len(imgs)} images ===")
            for s in range(N_SAMPLES):
                resp, cost = await _llm_call(CONFIG, prompt, images=imgs or None)
                choice, plan = parse_choice(resp, questions_block)
                print(f"  [sample {s + 1}] CHOSE: {choice}")
                if plan:
                    print(f"             PLAN: {plan}")


if __name__ == "__main__":
    asyncio.run(main())
