"""Offline validation: score a logged step's unanswered question bank with the
theory-entropy scorer, to check whether the 'correct trick' question rises to
the top (where B-difference buried it).

Usage:
  uv run scripts/validate_theory_scorer.py --episode-dir <episode_0> --step 9 \
      [--match "target pattern" --match center]
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf
from PIL import Image

from explore.stepwise_eb_learn_improve import EBQAPair, deserialize_eb_qa_pairs
from explore.theory_exploration import (
    generate_crux_questions,
    generate_theories,
    score_questions_theory_entropy,
)

DK_START = "=== DEFAULT KNOWLEDGE ==="
DK_END = "=== END DEFAULT KNOWLEDGE ==="


def extract_default_knowledge(step_dir: Path) -> str:
    d = json.loads((step_dir / "experiment_log.json").read_text())
    for key in ("question_gen_prompt", "experiment_prompt"):
        p = d.get(key) or ""
        i, j = p.find(DK_START), p.find(DK_END)
        if i != -1 and j != -1:
            return p[i + len(DK_START):j].strip()
    return ""


async def main_async(args):
    episode_dir = args.episode_dir.resolve()
    step_dir = episode_dir / f"step_{args.step:03d}"
    config = OmegaConf.load(episode_dir.parent / "config.yaml")

    beliefs = (step_dir / "beliefs.txt").read_text()
    default_knowledge = extract_default_knowledge(step_dir)
    img_file = step_dir / "obs_before.png"
    image = Image.open(img_file).convert("RGB") if img_file.exists() else None

    qa_pairs = deserialize_eb_qa_pairs(json.loads((step_dir / "qa_pairs.json").read_text()))
    n_unanswered = sum(1 for q in qa_pairs if q.answer is None)
    print(f"step {args.step}: {len(qa_pairs)} questions ({n_unanswered} unanswered)")

    theories, t_cost, _ = await generate_theories(
        config=config, beliefs=beliefs, default_knowledge=default_knowledge,
        current_observation=None, current_image=image,
        num_theories=config.eval.evolve.get("num_theories", 5),
        decay=config.eval.evolve.get("theory_weight_decay", 0.6),
    )
    print(f"\n=== {len(theories)} THEORIES (weight | first line) ===")
    for t in theories:
        first = t.world_knowledge.strip().splitlines()[0][:90]
        print(f"  r{t.rank} w={t.weight:.3f}  {first}")

    crux, c_cost, _ = await generate_crux_questions(
        config=config, theories=theories, beliefs=beliefs,
        default_knowledge=default_knowledge,
        num_crux=config.eval.evolve.get("num_crux_questions", 5),
    )
    base = len(qa_pairs)
    for q in crux:
        qa_pairs.append(EBQAPair(question=q, answer=None, evidence="", source_step=args.step))
    print(f"\n=== {len(crux)} CRUX QUESTIONS (seeded) ===")
    for q in crux:
        print(f"  - {q}")

    scores, s_cost, log = await score_questions_theory_entropy(
        config=config, theories=theories, qa_pairs=qa_pairs,
        default_knowledge=default_knowledge,
    )

    print("\n=== TOP 15 QUESTIONS BY THEORY-ENTROPY (MI bits) ===")
    ranked = log["per_question"]
    for i, r in enumerate(ranked[:15], 1):
        tag = " [CRUX]" if r["idx"] >= base else ""
        hit = ""
        if any(m.lower() in r["question"].lower() for m in (args.match or [])):
            hit = "  <<< MATCH"
        print(f"{i:2d}. MI={r['score']:.4f}{tag} {r['question'][:95]}{hit}")

    if args.match:
        print("\n=== RANK OF MATCHING (trick) QUESTIONS ===")
        for rank, r in enumerate(ranked, 1):
            if any(m.lower() in r["question"].lower() for m in args.match):
                print(f"  rank {rank}/{len(ranked)}  MI={r['score']:.4f}  {r['question'][:95]}")

    print(f"\ncost: theories ${t_cost:.4f} + crux ${c_cost:.4f} + scoring ${s_cost:.4f} = ${t_cost+c_cost+s_cost:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-dir", required=True, type=Path)
    ap.add_argument("--step", required=True, type=int)
    ap.add_argument("--match", action="append", help="substring(s) marking the 'trick' question")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
