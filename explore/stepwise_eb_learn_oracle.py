"""Oracle-driven stepwise EB-learning for AutumnBench.

Mirrors the question/answer/belief-update loop of `stepwise_eb_learn.py` but
replaces the entire environment-rollout path with direct LLM oracle calls. This
lets us A/B test question-selection strategies without paying the cost of full
game rollouts. There is no env, no perception, no experiment formulation.

Per step:
    1. Generate N candidate questions from current beliefs.
    2. Select ONE question from the unanswered bank via a pluggable selector (SCORERS).
    3. Answer it via `autumn_oracle.answer_question`.
    4. Update beliefs from the newly-answered Q/A.
    5. (Optional) Score beliefs against the program summary (0-10).

Outputs land under `logs/oracle_runs/<task>/<run_id>/`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Optional

from omegaconf import OmegaConf

from autumn_oracle import (
    STRONG_MODEL,
    answer_question,
    load_dsl_docs,
    load_program_source,
    score_beliefs_against_summary,
    task_oracle_dir,
    task_program_path,
)
from explore.mixed_improve import improve_beliefs_simple
from explore.stepwise_eb_learn_improve import (
    EBQAPair,
    generate_questions_from_beliefs,
    select_qa_pairs_for_experiment,
    serialize_eb_qa_pairs,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = REPO_ROOT / "logs" / "oracle_runs"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class OracleRunConfig:
    task: str
    num_steps: int = 12
    candidates_per_step: int = 6
    selector: str = "llm"
    learner_model: str = "openrouter/openai/gpt-5-mini"
    oracle_model: str = STRONG_MODEL
    scorer_model: str = STRONG_MODEL
    score_every_step: bool = True
    seed: int = 0
    notes: str = ""


def _build_litellm_config(model_name: str):
    """Synthesize the minimal `config` dict that `_llm_call` expects."""
    client_name, _, model_id = model_name.partition("/")
    if not model_id:
        raise ValueError(f"learner_model '{model_name}' must be 'provider/model_id'.")
    return OmegaConf.create({"client": {"client_name": client_name, "model_id": model_id}})


def _default_knowledge() -> str:
    from autumn_env import INSTRUCTION_PROMPT
    return INSTRUCTION_PROMPT


# ---------------------------------------------------------------------------
# Selectors — each returns at most ONE question index to send to the oracle
# ---------------------------------------------------------------------------


SelectorFn = Callable[..., Awaitable[tuple[Optional[int], float, dict]]]


async def _select_first(
    *, current_qa: list[EBQAPair], **_: object
) -> tuple[Optional[int], float, dict]:
    unanswered = [i for i, q in enumerate(current_qa) if q.answer is None]
    return (unanswered[0] if unanswered else None), 0.0, {
        "selector": "first",
        "pool": len(unanswered),
    }


async def _select_random(
    *, current_qa: list[EBQAPair], rng: random.Random, **_: object
) -> tuple[Optional[int], float, dict]:
    unanswered = [i for i, q in enumerate(current_qa) if q.answer is None]
    return (rng.choice(unanswered) if unanswered else None), 0.0, {
        "selector": "random",
        "pool": len(unanswered),
    }


async def _select_llm(
    *,
    config,
    current_qa: list[EBQAPair],
    beliefs: str,
    default_knowledge: str,
    **_: object,
) -> tuple[Optional[int], float, dict]:
    """LLM picks ONE unanswered question from the bank."""
    _, indices, cost, log = await select_qa_pairs_for_experiment(
        config=config,
        current_qa=current_qa,
        max_answered_qa_pairs=0,
        max_unanswered_qa_pairs=1,
        default_knowledge=default_knowledge,
        beliefs=beliefs,
    )
    log["selector"] = "llm"
    return (indices[0] if indices else None), cost, log


SCORERS: dict[str, SelectorFn] = {
    "first": _select_first,
    "random": _select_random,
    "llm": _select_llm,
}


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    step_index: int
    new_questions: list[str] = field(default_factory=list)
    selected_source_index: int | None = None
    selected_question: str = ""
    answer: dict = field(default_factory=dict)
    belief_score: int | None = None
    belief_score_rationale: str = ""
    step_cost: float = 0.0
    beliefs_after: str = ""


def _format_answered_summary(record: dict) -> str:
    """Render the single answered Q/A for the belief-update prompt.

    The agent sees ONLY the yes/no answer — not the oracle's rationale.
    """
    if not record:
        return ""
    ans = record["answer"].upper()
    return (
        "=== ANSWERED QUESTION (this step) ===\n"
        f"- Q: {record['question']}\n"
        f"  A: {ans}\n"
        "=== END ANSWERED QUESTION ==="
    )


def _apply_answer_to_bank(
    bank: list[EBQAPair], src_idx: int, record: dict
) -> None:
    """Mutate `bank` in place. 'unknown' stays unanswered but records rationale."""
    ans = record["answer"]
    if ans == "yes":
        bank[src_idx].answer = True
    elif ans == "no":
        bank[src_idx].answer = False
    else:
        bank[src_idx].answer = None
    bank[src_idx].evidence = (
        f"oracle: {ans}" + (f" — {record['rationale']}" if record.get("rationale") else "")
    )


async def run_oracle_episode(cfg: OracleRunConfig) -> dict:
    out_dir = RUNS_ROOT / cfg.task / f"{cfg.selector}_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    program_src = load_program_source(cfg.task)
    dsl_docs = load_dsl_docs()
    summary_path = task_oracle_dir(cfg.task) / "summary.md"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing {summary_path}. Run `uv run scripts/bootstrap_oracle.py {cfg.task}` first."
        )
    summary_text = summary_path.read_text()
    default_knowledge = _default_knowledge()
    learner_config = _build_litellm_config(cfg.learner_model)
    rng = random.Random(cfg.seed)

    selector_fn = SCORERS.get(cfg.selector)
    if selector_fn is None:
        raise ValueError(f"Unknown selector '{cfg.selector}'. Known: {list(SCORERS)}")

    bank: list[EBQAPair] = []
    beliefs = ""
    step_records: list[StepRecord] = []
    total_cost = 0.0

    for step in range(cfg.num_steps):
        print(f"[{cfg.task}/{cfg.selector}] step {step + 1}/{cfg.num_steps}")
        rec = StepRecord(step_index=step)
        llm_calls: list[dict] = []  # full prompt/response trace for this step

        bank_before_count = len(bank)

        # 1. generate candidate questions
        new_qs, gen_cost, gen_prompt, gen_raw = await generate_questions_from_beliefs(
            config=learner_config,
            beliefs=beliefs,
            current_qa=bank,
            default_knowledge=default_knowledge,
            num_questions=cfg.candidates_per_step,
            current_step=step,
        )
        bank.extend(new_qs)
        rec.new_questions = [q.question for q in new_qs]
        rec.step_cost += gen_cost
        llm_calls.append({
            "kind": "generate_questions",
            "model": cfg.learner_model,
            "cost": gen_cost,
            "prompt": gen_prompt,
            "response": gen_raw,
        })

        # 2. select ONE
        sel_idx, sel_cost, sel_log = await selector_fn(
            config=learner_config,
            current_qa=bank,
            beliefs=beliefs,
            default_knowledge=default_knowledge,
            rng=rng,
        )
        rec.step_cost += sel_cost
        rec.selected_source_index = sel_idx
        llm_calls.append({
            "kind": "select_question",
            "model": cfg.learner_model if cfg.selector == "llm" else "(deterministic)",
            "cost": sel_cost,
            "prompt": sel_log.get("prompt", ""),
            "response": sel_log.get("response", ""),
            "log": {k: v for k, v in sel_log.items() if k not in ("prompt", "response")},
        })

        if sel_idx is None:
            print("  no unanswered question available; stopping early")
            (out_dir / f"step_{step:02d}").mkdir(exist_ok=True)
            (out_dir / f"step_{step:02d}" / "llm_calls.json").write_text(
                json.dumps(llm_calls, indent=2)
            )
            step_records.append(rec)
            break

        selected_qa = bank[sel_idx]
        rec.selected_question = selected_qa.question

        # 3. oracle answer
        oa = await answer_question(
            cfg.task,
            selected_qa.question,
            model=cfg.oracle_model,
            program_src=program_src,
            summary=summary_text,
            dsl_docs=dsl_docs,
        )
        answer_record = {
            "question": selected_qa.question,
            "answer": oa["answer"],
            "rationale": oa["rationale"],
            "cost": oa["cost"],
            "cached": oa["cached"],
        }
        rec.answer = answer_record
        rec.step_cost += oa["cost"]
        _apply_answer_to_bank(bank, sel_idx, answer_record)
        llm_calls.append({
            "kind": "oracle_answer",
            "model": cfg.oracle_model,
            "cost": oa["cost"],
            "cached": oa["cached"],
            "prompt": oa.get("prompt", ""),
            "response": oa.get("raw", ""),
        })

        # 4. belief update (only when we got a yes/no — unknown adds no evidence)
        if oa["answer"] in ("yes", "no"):
            bel_trace: dict = {}
            beliefs, bel_cost = await improve_beliefs_simple(
                config=learner_config,
                beliefs=beliefs,
                default_knowledge=default_knowledge,
                episode_summaries=_format_answered_summary(answer_record),
                trace=bel_trace,
            )
            rec.step_cost += bel_cost
            llm_calls.append({
                "kind": "belief_update",
                "model": cfg.learner_model,
                "cost": bel_cost,
                "prompt": bel_trace.get("prompt", ""),
                "response": bel_trace.get("response", ""),
            })
        rec.beliefs_after = beliefs

        # 5. score
        if cfg.score_every_step or step == cfg.num_steps - 1:
            score_result = await score_beliefs_against_summary(
                cfg.task,
                beliefs,
                model=cfg.scorer_model,
                summary=summary_text,
                program_src=program_src,
            )
            rec.belief_score = score_result["score"]
            rec.belief_score_rationale = score_result["rationale"]
            rec.step_cost += score_result["cost"]
            llm_calls.append({
                "kind": "score_beliefs",
                "model": cfg.scorer_model,
                "cost": score_result["cost"],
                "prompt": score_result.get("prompt", ""),
                "response": score_result.get("raw", ""),
            })

        total_cost += rec.step_cost
        step_records.append(rec)

        step_dir = out_dir / f"step_{step:02d}"
        step_dir.mkdir(exist_ok=True)
        (step_dir / "beliefs.md").write_text(beliefs + "\n")
        (step_dir / "qa.jsonl").write_text(
            "\n".join(json.dumps(d) for d in serialize_eb_qa_pairs(bank)) + "\n"
        )
        (step_dir / "step.json").write_text(json.dumps({
            **asdict(rec),
            "bank_size_before": bank_before_count,
            "bank_size_after": len(bank),
            "unanswered_pool_size": sum(1 for q in bank if q.answer is None),
        }, indent=2))
        (step_dir / "llm_calls.json").write_text(json.dumps(llm_calls, indent=2))
        score_txt = f" score={rec.belief_score}/10" if rec.belief_score is not None else ""
        print(
            f"  candidates={len(new_qs)} picked=Q{sel_idx + 1} "
            f"answer={oa['answer']} cost=${rec.step_cost:.4f}{score_txt}"
        )

    summary = {
        "task": cfg.task,
        "selector": cfg.selector,
        "num_steps": cfg.num_steps,
        "steps_run": len(step_records),
        "final_belief_score": step_records[-1].belief_score if step_records else None,
        "scores_by_step": [r.belief_score for r in step_records],
        "total_cost": total_cost,
        "bank_size": len(bank),
        "n_answered": sum(1 for q in bank if q.answer is not None),
        "out_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[{cfg.task}/{cfg.selector}] done  total_cost=${total_cost:.4f}  out={out_dir}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> OracleRunConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", help="AutumnBench task name (e.g. 'ice').")
    p.add_argument("--num-steps", type=int, default=12)
    p.add_argument("--candidates-per-step", type=int, default=6)
    p.add_argument(
        "--selector",
        default="llm",
        choices=sorted(SCORERS.keys()),
        help="Question-selection strategy.",
    )
    p.add_argument(
        "--learner-model",
        default="openrouter/openai/gpt-5-mini",
        help="LiteLLM model id for question generation / belief updates / selection.",
    )
    p.add_argument("--oracle-model", default=STRONG_MODEL)
    p.add_argument("--scorer-model", default=STRONG_MODEL)
    p.add_argument(
        "--no-score-every-step",
        action="store_true",
        help="Only score beliefs at the final step.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--notes", default="")
    args = p.parse_args(argv)
    return OracleRunConfig(
        task=args.task,
        num_steps=args.num_steps,
        candidates_per_step=args.candidates_per_step,
        selector=args.selector,
        learner_model=args.learner_model,
        oracle_model=args.oracle_model,
        scorer_model=args.scorer_model,
        score_every_step=not args.no_score_every_step,
        seed=args.seed,
        notes=args.notes,
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = _parse_args(sys.argv[1:] if argv is None else argv)
    if not task_program_path(cfg.task).exists():
        sys.exit(f"No .sexp program for task '{cfg.task}' at {task_program_path(cfg.task)}")
    asyncio.run(run_oracle_episode(cfg))


if __name__ == "__main__":
    main()
