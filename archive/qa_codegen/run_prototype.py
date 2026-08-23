"""Drive the code-synthesis QA prototype over the may29 ft09 logs.

For each stored QA pair, ask the LLM (same model that produced the logs:
google/gemini-2.5-flash) to synthesise a predicate over the structured
trajectory, run it, and compare its verdict to the stored log-reading answer.

Usage:  uv run python prototypes/qa_codegen/run_prototype.py
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import litellm
from dotenv import load_dotenv

from harness import (  # local import (run from this dir) ...
    PREDICATE_API_DOC,
    extract_code,
    parse_trajectory,
    run_predicate,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

RUN = Path(__file__).resolve().parents[2] / (
    "logs/dev/may29/20260529-114550/eb_learn__arc_agi__gemini-2p5-flash__ft09/"
    "2026-05-29_11-45-53_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn/episode_0"
)
MODEL = "openrouter/google/gemini-2.5-flash"


def stored_to_verdict(ans) -> str:
    return {True: "YES", False: "NO", None: "MAYBE"}[ans]


async def synth(question: str) -> tuple[str, str]:
    prompt = (
        "We are reverse-engineering a grid puzzle game (ARC-AGI 3) from logged "
        "play. Write a Python predicate that answers the QUESTION below by "
        "inspecting the structured trajectory.\n\n"
        f"{PREDICATE_API_DOC}\n\n"
        f"QUESTION: {question}\n"
    )
    resp = await asyncio.to_thread(
        litellm.completion,
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        num_retries=4,
    )
    text = resp.choices[0].message.content
    code = extract_code(text) or ""
    return text, code


def ground_truth_facts(steps) -> dict:
    """A few facts computed directly here, to help adjudicate disagreements."""
    levels = [s.levels_completed for s in steps if s.levels_completed is not None]
    red_to_blue = any(
        any(old == 8 and new == 9 for _, _, old, new in s.changed_cells()) for s in steps
    )
    bg_click_caused_change = [
        s.step
        for s in steps
        if s.clicked_cell_pre() == 5 and s.any_change()
    ]
    bg_click_total = sum(1 for s in steps if s.clicked_cell_pre() == 5)
    return {
        "levels_ever_increased": bool(levels and max(levels) > levels[0]),
        "max_levels_completed": max(levels) if levels else None,
        "any_red->blue_revert": red_to_blue,
        "background(5)_clicks": bg_click_total,
        "background_clicks_that_changed_grid": bg_click_caused_change,
    }


async def main() -> None:
    buffer = json.loads((RUN / "trajectory_buffer.json").read_text())
    qa = json.loads((RUN / "qa_pairs.json").read_text())
    steps = parse_trajectory(buffer)

    print(f"Parsed {len(steps)} transitions; "
          f"{sum(s.pre is not None for s in steps)} have a pre-grid, "
          f"{sum(s.post is not None for s in steps)} have a post-grid.\n")
    print("=== DIRECTLY-COMPUTED GROUND-TRUTH FACTS ===")
    for k, v in ground_truth_facts(steps).items():
        print(f"  {k}: {v}")
    print()

    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(exist_ok=True)

    rows = []
    agree = disagree = abstain = 0
    for i, q in enumerate(qa):
        question = q["question"]
        stored = stored_to_verdict(q.get("answer"))
        text, code = await synth(question)
        if not code:
            verdict, err = "MAYBE", "no code extracted"
        else:
            verdict, err = run_predicate(code, steps)
        (out_dir / f"q{i:02d}.py").write_text(code or text)

        if verdict == "MAYBE":
            tag, abstain = "ABSTAIN", abstain + 1
        elif verdict == stored:
            tag, agree = "AGREE  ", agree + 1
        else:
            tag, disagree = "DISAGREE", disagree + 1

        rows.append(
            {"i": i, "question": question, "stored": stored,
             "code_verdict": verdict, "tag": tag.strip(), "error": err}
        )
        print(f"[{tag}] Q{i:02d} stored={stored:5s} code={verdict:5s} "
              f"{'ERR' if err else ''}")
        print(f"         {question[:96]}")
        if err:
            print(f"         ! {err.splitlines()[-1][:120]}")

    print(f"\n=== SUMMARY ===  agree={agree} disagree={disagree} "
          f"abstain={abstain}  (of {len(qa)})")
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"Synthesised predicates + results written to {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())
