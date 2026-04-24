#!/usr/bin/env python3
"""Offline B-difference scoring over a finished stepwise_eb_learn run.

Walks every experiment-generation step of a run, reconstructs the
unanswered question pool and the beliefs that were current at that step,
then calls ``question_scoring.score_questions_b_diff`` to compute the
B-difference score per unanswered question. Per-step scores and an
aggregate comparison against the live LLM-trim decisions are written
next to the existing step logs.

Usage:
    uv run scripts/rescore_questions_offline.py <run_dir> \
        [--episode N] [--method full|light] [--max-concurrent 8] \
        [--steps 5-10] [--dry-run]

Example:
    uv run scripts/rescore_questions_offline.py \
        logs/dev/apr20/2026-04-20_22-54-50_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn \
        --episode 0 --method light
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path

import yaml
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from question_scoring import score_questions_b_diff  # noqa: E402
from stepwise_eb_learn_improve import (  # noqa: E402
    EBQAPair,
    deserialize_eb_qa_pairs,
)


# ---------------------------------------------------------------------------
# Pre-trim QA parsing
# ---------------------------------------------------------------------------


_QA_LINE_RE = re.compile(
    r"^Q(\d+):\s*(.+?)\s*->\s*(YES|NO|UNANSWERED)(?:\s*\(evidence:\s*(.*?)\))?\s*$",
    re.DOTALL,
)


def _parse_pre_trim_qa_from_prompt(prompt: str, fallback_step: int = 0) -> list[EBQAPair]:
    """Parse the === CURRENT QUESTIONS === section of a trim_log.json prompt.

    The prompt text encoded each question as produced by
    stepwise_eb_learn_improve._format_qa_list:
        Q{n}: {question} -> YES|NO|UNANSWERED (evidence: ...)
    """
    start = prompt.find("=== CURRENT QUESTIONS ===")
    end = prompt.find("=== END CURRENT QUESTIONS ===")
    if start == -1 or end == -1:
        return []
    body = prompt[start + len("=== CURRENT QUESTIONS ===") : end].strip()

    # Split on the start of each Q{n}: line so multi-line evidence stays attached.
    chunks = re.split(r"(?=^Q\d+:\s)", body, flags=re.MULTILINE)
    qa_pairs: list[EBQAPair] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        m = _QA_LINE_RE.match(chunk)
        if not m:
            continue
        _n, question, answer_word, evidence = m.groups()
        answer: bool | None
        if answer_word == "YES":
            answer = True
        elif answer_word == "NO":
            answer = False
        else:
            answer = None
        qa_pairs.append(
            EBQAPair(
                question=question.strip(),
                answer=answer,
                evidence=(evidence or "").strip(),
                source_step=fallback_step,
            )
        )
    return qa_pairs


# ---------------------------------------------------------------------------
# Artifact loaders
# ---------------------------------------------------------------------------


def _load_step_log(step_dir: Path) -> dict | None:
    path = step_dir / "step_log.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _load_trim_log(step_dir: Path) -> dict | None:
    path = step_dir / "trim_log.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _load_step_qa(step_dir: Path) -> list[EBQAPair]:
    path = step_dir / "qa_pairs.json"
    if not path.exists():
        return []
    return deserialize_eb_qa_pairs(json.loads(path.read_text()))


def _load_beliefs_for_step(episode_dir: Path, step_num: int) -> str:
    """Return beliefs as they stood *at the start* of step ``step_num``.

    Beliefs at step_N/beliefs.txt are post-improve (end of step_N), so the
    trim phase inside step_N saw step_(N-1)/beliefs.txt (or input_beliefs
    for step 0).
    """
    if step_num == 0:
        fallback = episode_dir / "input_beliefs.txt"
        if fallback.exists():
            return fallback.read_text()
        return ""
    prev = episode_dir / f"step_{step_num - 1:03d}" / "beliefs.txt"
    if prev.exists():
        return prev.read_text()
    # Fall back to episode-level input beliefs if an earlier step is missing.
    fallback = episode_dir / "input_beliefs.txt"
    if fallback.exists():
        return fallback.read_text()
    return ""


def _reconstruct_pre_scoring_qa(
    step_dir: Path, step_log: dict, fallback_step: int
) -> tuple[list[EBQAPair], str]:
    """Return (qa_list, source_tag).

    Prefers the pre-trim list parsed out of trim_log.json. Falls back to
    the step's own qa_pairs.json (post-update; a decent approximation
    when trim did not fire this step).
    """
    trim_log = _load_trim_log(step_dir)
    if trim_log and "prompt" in trim_log:
        qa = _parse_pre_trim_qa_from_prompt(trim_log["prompt"], fallback_step=fallback_step)
        if qa:
            return qa, "trim_log.prompt"
    # Fall back to post-update state of this step.
    return _load_step_qa(step_dir), "qa_pairs.json"


# ---------------------------------------------------------------------------
# Comparison against the live LLM trim
# ---------------------------------------------------------------------------


def _extract_llm_trim_kept_questions(trim_log: dict) -> list[str]:
    """From a trim_log.json produced by the live LLM trim, return the set
    of question texts the LLM chose to keep (answered or unanswered).
    """
    response = trim_log.get("response", "")
    kept: list[str] = []
    trimmed_block_start = response.find("<trimmed_questions>")
    trimmed_block_end = response.find("</trimmed_questions>")
    if trimmed_block_start == -1 or trimmed_block_end == -1:
        return kept
    body = response[trimmed_block_start:trimmed_block_end]
    for m in re.finditer(r"<q\s+n=\"\d+\">(.*?)</q>", body, re.DOTALL):
        q_content = m.group(1)
        q_match = re.search(r"<question>(.*?)</question>", q_content, re.DOTALL)
        if q_match:
            kept.append(q_match.group(1).strip())
    return kept


def _overlap_at_k(
    ranked_questions: list[str],
    baseline_questions: list[str],
    k: int,
) -> float:
    if k <= 0 or not ranked_questions:
        return 0.0
    top_k = set(q.strip().lower() for q in ranked_questions[:k])
    baseline = set(q.strip().lower() for q in baseline_questions)
    if not baseline:
        return 0.0
    inter = top_k & baseline
    denom = min(k, len(baseline))
    return len(inter) / denom if denom else 0.0


# ---------------------------------------------------------------------------
# Per-step entry
# ---------------------------------------------------------------------------


async def score_step(
    *,
    config,
    episode_dir: Path,
    step_dir: Path,
    method: str,
    max_concurrent: int,
    include_policy: bool,
    cap_unanswered: int,
) -> dict | None:
    step_log = _load_step_log(step_dir)
    if step_log is None:
        return None
    if not step_log.get("did_gen_questions"):
        return None

    step_num = int(step_log.get("global_step", step_log.get("step", 0)))
    qa_pairs, qa_source = _reconstruct_pre_scoring_qa(
        step_dir, step_log, fallback_step=step_num
    )
    if not qa_pairs:
        return {"step": step_num, "skipped": "no qa_pairs reconstructed"}

    unanswered = [qa for qa in qa_pairs if qa.answer is None]
    if len(unanswered) < 2:
        return {
            "step": step_num,
            "skipped": f"fewer than 2 unanswered ({len(unanswered)})",
            "qa_source": qa_source,
        }

    beliefs = _load_beliefs_for_step(episode_dir, step_num)

    scores, cost, score_log = await score_questions_b_diff(
        config=config,
        beliefs=beliefs,
        qa_pairs=qa_pairs,
        method=method,
        include_policy=include_policy,
        max_concurrent=max_concurrent,
    )

    # Ranked questions (high score first).
    ranked_entries = sorted(
        (
            {
                "idx": i,
                "question": qa_pairs[i].question,
                "source_step": qa_pairs[i].source_step,
                "score": scores[i],
            }
            for i in scores
        ),
        key=lambda d: (d["score"], d["source_step"]),
        reverse=True,
    )
    ranked_questions = [r["question"] for r in ranked_entries]

    # Compare with live LLM trim (if this step ran one).
    trim_log = _load_trim_log(step_dir)
    llm_kept_all = _extract_llm_trim_kept_questions(trim_log) if trim_log else []
    qa_by_question = {qa.question.strip().lower(): qa for qa in qa_pairs}
    llm_kept_unanswered = [
        q
        for q in llm_kept_all
        if q.strip().lower() in qa_by_question
        and qa_by_question[q.strip().lower()].answer is None
    ]
    top_k = cap_unanswered
    b_diff_top_k = ranked_questions[:top_k]
    overlap = _overlap_at_k(b_diff_top_k, llm_kept_unanswered, top_k)

    out: dict = {
        "step": step_num,
        "method": method,
        "qa_source": qa_source,
        "num_qa": len(qa_pairs),
        "num_unanswered_scored": len(scores),
        "ranked_unanswered": ranked_entries,
        "b_diff_top_k_questions": b_diff_top_k,
        "llm_trim_kept_unanswered": llm_kept_unanswered,
        "did_trim": bool(step_log.get("did_trim")),
        "overlap_at_k": overlap,
        "cap_unanswered": top_k,
        "cost_usd": cost,
        "active_experiment": step_log.get("active_experiment"),
        "scoring_log": score_log,
    }

    out_path = step_dir / f"scoring_offline_{method}.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    return out


# ---------------------------------------------------------------------------
# Config / CLI
# ---------------------------------------------------------------------------


def _load_run_config(run_dir: Path):
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found under {run_dir}")
    raw = yaml.safe_load(config_path.read_text())
    return OmegaConf.create(raw)


def _parse_step_range(spec: str | None) -> tuple[int, int] | None:
    if not spec:
        return None
    if "-" in spec:
        a, b = spec.split("-", 1)
        return int(a), int(b)
    n = int(spec)
    return n, n


async def main_async(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"Run dir not found: {run_dir}", file=sys.stderr)
        return 2

    config = _load_run_config(run_dir)
    eb_cfg = config.eval.evolve
    cap_unanswered = int(eb_cfg.get("max_unanswered_qa_pairs", 10))
    include_policy = bool(eb_cfg.get("include_policy", False))

    episode_name = f"episode_{args.episode}"
    episode_dir = run_dir / episode_name
    if not episode_dir.exists():
        print(f"Episode dir not found: {episode_dir}", file=sys.stderr)
        return 2

    step_dirs = sorted(
        [p for p in episode_dir.iterdir() if p.is_dir() and p.name.startswith("step_")],
        key=lambda p: int(p.name.split("_")[1]),
    )
    step_range = _parse_step_range(args.steps)
    if step_range:
        lo, hi = step_range
        step_dirs = [
            p for p in step_dirs if lo <= int(p.name.split("_")[1]) <= hi
        ]

    print(
        f"Scoring run={run_dir.name} episode={args.episode} "
        f"method={args.method} steps={len(step_dirs)} dry_run={args.dry_run}"
    )

    if args.dry_run:
        for sd in step_dirs:
            sl = _load_step_log(sd)
            if sl and sl.get("did_gen_questions"):
                tl = _load_trim_log(sd)
                qa_src = "trim_log.prompt" if tl else "qa_pairs.json"
                print(
                    f"  {sd.name}: did_trim={sl.get('did_trim')}, "
                    f"unanswered={sl.get('num_unanswered_questions')}, qa_source={qa_src}"
                )
        return 0

    summary: list[dict] = []
    total_cost = 0.0
    for sd in step_dirs:
        try:
            row = await score_step(
                config=config,
                episode_dir=episode_dir,
                step_dir=sd,
                method=args.method,
                max_concurrent=args.max_concurrent,
                include_policy=include_policy,
                cap_unanswered=cap_unanswered,
            )
        except Exception as e:
            logging.exception("scoring failed for %s", sd)
            summary.append({"step_dir": sd.name, "error": repr(e)})
            continue
        if row is None:
            continue
        if "skipped" in row:
            summary.append({"step": row["step"], "skipped": row["skipped"]})
            continue
        total_cost += row.get("cost_usd", 0.0)
        summary.append(
            {
                "step": row["step"],
                "did_trim": row["did_trim"],
                "num_unanswered": row["num_unanswered_scored"],
                "overlap_at_k": row["overlap_at_k"],
                "cost_usd": row["cost_usd"],
            }
        )
        print(
            f"  step_{row['step']:03d}: unanswered={row['num_unanswered_scored']} "
            f"did_trim={row['did_trim']} overlap@k={row['overlap_at_k']:.3f} "
            f"cost=${row['cost_usd']:.4f}"
        )

    agg = {
        "run_dir": str(run_dir),
        "episode": args.episode,
        "method": args.method,
        "cap_unanswered": cap_unanswered,
        "num_steps": len(summary),
        "total_cost_usd": total_cost,
        "mean_overlap_at_k": (
            sum(
                r.get("overlap_at_k", 0.0)
                for r in summary
                if "overlap_at_k" in r and r.get("did_trim")
            )
            / max(1, sum(1 for r in summary if "overlap_at_k" in r and r.get("did_trim")))
        ),
        "rows": summary,
    }
    out_path = run_dir / f"rescore_summary_ep{args.episode}_{args.method}.json"
    out_path.write_text(json.dumps(agg, indent=2, default=str))
    print(f"\nWrote {out_path}")
    print(
        f"Total cost: ${total_cost:.4f} | "
        f"Mean overlap@k (trim steps only): {agg['mean_overlap_at_k']:.3f}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline B-difference scoring of a stepwise_eb_learn run."
    )
    parser.add_argument("run_dir", help="Path to the run directory.")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--method", choices=("full", "light"), default="light",
        help="Belief-update strategy passed to score_questions_b_diff."
    )
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument(
        "--steps", default=None,
        help="Optional step filter, e.g. '5' or '5-12' (inclusive)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report which steps would be scored without making LLM calls."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
