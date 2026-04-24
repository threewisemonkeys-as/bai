"""B-difference scoring for unanswered questions.

Given current beliefs B and a pool of unanswered questions Q, score each
q_i by how much its resolution would shift predictions over the other
unanswered questions:

    B_pos := beliefs updated assuming q_i = YES
    B_neg := beliefs updated assuming q_i = NO
    A_pos := predicted answers over Q \\ {q_i} under B_pos
    A_neg := predicted answers over Q \\ {q_i} under B_neg
    score(q_i) := Hamming(A_pos, A_neg) / |A_pos|

Two methods are supported for constructing B_pos / B_neg:
- ``full``: rewrite <world_knowledge>/<policy> via LLM (3 calls per q_i)
- ``light``: append an explicit assumption to B, no rewrite (1 call per q_i)

The only LLM-caller shared with the rest of the codebase is
``mixed_improve._llm_call`` so mock mode, logging, and cost accounting
remain consistent.
"""

import asyncio
import logging
import re
from typing import TYPE_CHECKING

from omegaconf import DictConfig

from llm_utils import extract_xml_key
from mixed_improve import _llm_call

if TYPE_CHECKING:
    from stepwise_eb_learn_improve import EBQAPair


# ---------------------------------------------------------------------------
# Belief-block templates (duplicated from stepwise_eb_learn to avoid a
# circular import — kept intentionally minimal, matching the main pipeline's
# <updated_beliefs> schema).
# ---------------------------------------------------------------------------


def _beliefs_block_template(include_policy: bool) -> str:
    if include_policy:
        return """<updated_beliefs>
<world_knowledge>
- ...
</world_knowledge>
<policy>
- ...
</policy>
</updated_beliefs>"""
    return """<updated_beliefs>
<world_knowledge>
- ...
</world_knowledge>
</updated_beliefs>"""


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _format_answered(qa_pairs: list["EBQAPair"]) -> str:
    lines = []
    for qa in qa_pairs:
        if qa.answer is None:
            continue
        status = "YES" if qa.answer else "NO"
        ev = qa.evidence.strip() if qa.evidence else ""
        ev_part = f" (evidence: {ev})" if ev else ""
        lines.append(f"- {qa.question} -> {status}{ev_part}")
    return "\n".join(lines) if lines else "(none)"


def _format_assumed(question: str, assumed_yes: bool) -> str:
    status = "YES" if assumed_yes else "NO"
    return f"- {question} -> {status} (HYPOTHETICAL, not yet confirmed by evidence)"


def _format_default_knowledge_section(default_knowledge: str) -> str:
    if not default_knowledge:
        return ""
    return f"""
=== DEFAULT KNOWLEDGE ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===
"""


# ---------------------------------------------------------------------------
# Belief-update prompts
# ---------------------------------------------------------------------------


async def _rewrite_beliefs_full(
    config: DictConfig,
    beliefs: str,
    answered_qa: list["EBQAPair"],
    assumed_question: str,
    assumed_yes: bool,
    include_policy: bool,
    default_knowledge: str = "",
) -> tuple[str, float, dict]:
    """Full method: ask LLM to rewrite beliefs given the hypothetical answer."""
    beliefs_block = _beliefs_block_template(include_policy)
    section_name = "world_knowledge and policy" if include_policy else "world_knowledge"
    default_knowledge_section = _format_default_knowledge_section(default_knowledge)

    prompt = f"""You are updating an agent's beliefs about an environment based on a hypothetical answer.

{default_knowledge_section}

=== CURRENT BELIEFS ===
{beliefs if beliefs else "(empty — no beliefs yet)"}
=== END CURRENT BELIEFS ===

=== CONFIRMED Q&A ===
{_format_answered(answered_qa)}
=== END CONFIRMED Q&A ===

=== HYPOTHETICAL Q&A ===
{_format_assumed(assumed_question, assumed_yes)}
=== END HYPOTHETICAL Q&A ===

Your task: Produce an updated version of the beliefs that ALSO incorporates the hypothetical Q&A as if it were confirmed. Revise or add facts/policies that follow from the hypothetical answer, and remove any that contradict it. Keep the revision minimal but faithful.

Format your response as:
<think>
What must change in the {section_name} if the hypothetical is true?
</think>

{beliefs_block}"""

    text, cost = await _llm_call(config, prompt)
    updated = extract_xml_key(text, "updated_beliefs")
    artifact = {
        "kind": "rewrite_beliefs_full",
        "assumed_question": assumed_question,
        "assumed_yes": assumed_yes,
        "default_knowledge_length": len(default_knowledge),
        "prompt": prompt,
        "response": text,
        "used_fallback_light": updated is None,
    }
    if updated is None:
        logging.warning(
            f"[question_scoring] _rewrite_beliefs_full: no <updated_beliefs> block found; "
            f"falling back to light assumption append"
        )
        updated_beliefs = _append_assumption_light(beliefs, assumed_question, assumed_yes)
        artifact["updated_beliefs"] = updated_beliefs
        return updated_beliefs, cost, artifact
    updated_beliefs = updated.strip()
    artifact["updated_beliefs"] = updated_beliefs
    return updated_beliefs, cost, artifact


def _append_assumption_light(
    beliefs: str,
    assumed_question: str,
    assumed_yes: bool,
) -> str:
    """Light method: prepend the hypothetical assumption to beliefs verbatim."""
    status = "YES" if assumed_yes else "NO"
    assumption = (
        f"<hypothetical_assumption>\n"
        f"Assume the answer to this question is {status}: {assumed_question}\n"
        f"</hypothetical_assumption>"
    )
    base = beliefs if beliefs else "(empty — no beliefs yet)"
    return f"{assumption}\n\n{base}"


# ---------------------------------------------------------------------------
# Answer-vector prediction
# ---------------------------------------------------------------------------


def _normalize_question_number(raw: str) -> int | None:
    value = raw.strip().upper()
    if value.startswith("Q"):
        value = value[1:]
    if not value.isdigit():
        return None
    return int(value)


def _answer_label_to_float(answer: str | None) -> float:
    """Map a parsed answer label to a vector value.

    Use exact labels/tokens rather than substring checks because UNKNOWN
    contains "NO".
    """
    if answer is None:
        return 0.5
    normalized = answer.strip().upper()
    if re.search(r"\bUNKNOWN\b", normalized):
        return 0.5
    has_yes = re.search(r"\bYES\b", normalized) is not None
    has_no = re.search(r"\bNO\b", normalized) is not None
    if has_yes and not has_no:
        return 1.0
    if has_no and not has_yes:
        return 0.0
    return 0.5


def _parse_answer_blocks(text: str) -> dict[int, str | None]:
    """Parse LLM answer blocks keyed by 1-based question number.

    The prompt asks for <q n="Q1">, but models sometimes emit variants like
    <Q n="1">, <Q n=1>, or closing tags such as </Q n=1>. We only need the
    opening tag and the content before the next question block.
    """
    openings = list(re.finditer(r"<Q\b(?P<attrs>[^>]*)>", text, re.IGNORECASE))
    answers_by_number: dict[int, str | None] = {}
    sequential_answers: list[str | None] = []

    for idx, match in enumerate(openings):
        attrs = match.group("attrs")
        next_start = openings[idx + 1].start() if idx + 1 < len(openings) else len(text)
        body = text[match.end() : next_start]
        answer = extract_xml_key(body, "answer")
        sequential_answers.append(answer)

        n_match = re.search(
            r"""\bn\s*=\s*(?:"([^"]+)"|'([^']+)'|([^\s>]+))""",
            attrs,
            re.IGNORECASE,
        )
        if not n_match:
            continue
        raw_number = next(group for group in n_match.groups() if group is not None)
        number = _normalize_question_number(raw_number)
        if number is not None and number not in answers_by_number:
            answers_by_number[number] = answer

    if answers_by_number:
        return answers_by_number
    return {idx: answer for idx, answer in enumerate(sequential_answers, 1)}


async def _predict_answer_vector(
    config: DictConfig,
    beliefs: str,
    questions: list[str],
    default_knowledge: str = "",
) -> tuple[list[float], float, str, str, dict]:
    """Ask LLM to answer a batch of questions under the given beliefs.

    Returns: (answers_as_floats, cost, prompt, response)
    Each float is 1.0 for YES, 0.0 for NO, 0.5 for UNKNOWN / missing.
    """
    if not questions:
        empty_artifact = {
            "kind": "predict_answer_vector",
            "default_knowledge_length": len(default_knowledge),
            "beliefs": beliefs,
            "questions": [],
            "prompt": "",
            "response": "",
            "answers": [],
        }
        return [], 0.0, "", "", empty_artifact

    question_blocks = [f"Q{i}: {q}" for i, q in enumerate(questions, 1)]
    questions_text = "\n".join(question_blocks)
    default_knowledge_section = _format_default_knowledge_section(default_knowledge)

    prompt = f"""Based on the beliefs below, answer each question with YES, NO, or UNKNOWN (if the beliefs do not settle the question).

{default_knowledge_section}

=== BELIEFS ===
{beliefs if beliefs else "(empty)"}
=== END BELIEFS ===

{questions_text}

For each question, respond in this format:
<q n="Q1">
<answer>YES or NO or UNKNOWN</answer>
</q>"""

    text, cost = await _llm_call(config, prompt)

    parsed_answers = _parse_answer_blocks(text)
    answers: list[float] = []
    for i, _ in enumerate(questions, 1):
        answers.append(_answer_label_to_float(parsed_answers.get(i)))
    artifact = {
        "kind": "predict_answer_vector",
        "default_knowledge_length": len(default_knowledge),
        "beliefs": beliefs,
        "questions": list(questions),
        "prompt": prompt,
        "response": text,
        "answers": list(answers),
    }
    return answers, cost, prompt, text, artifact


def _hamming_fraction(a: list[float], b: list[float]) -> float:
    if not a:
        return 0.0
    assert len(a) == len(b), f"vector length mismatch: {len(a)} vs {len(b)}"
    diffs = sum(abs(x - y) for x, y in zip(a, b))
    return diffs / len(a)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def score_questions_b_diff(
    config: DictConfig,
    beliefs: str,
    qa_pairs: list["EBQAPair"],
    *,
    method: str = "full",
    include_policy: bool = True,
    max_concurrent: int = 8,
    candidate_indices: list[int] | None = None,
    default_knowledge: str = "",
) -> tuple[dict[int, float], float, dict]:
    """Score each unanswered question by B-difference magnitude.

    Returns: (scores_by_index, total_cost, log)
    where scores_by_index maps the qa_pairs index to a float in [0, 1].
    Only unanswered questions (answer is None) are scored. When
    candidate_indices is supplied, only that subset is scored, but each
    candidate's A_pos/A_neg distance is still computed over the full
    unanswered question pool in qa_pairs.
    """
    if method not in ("full", "light"):
        raise ValueError(f"Unknown method: {method!r} (expected 'full' or 'light')")

    unanswered_indices = [i for i, qa in enumerate(qa_pairs) if qa.answer is None]
    unanswered_set = set(unanswered_indices)
    if candidate_indices is None:
        score_indices = unanswered_indices
    else:
        seen: set[int] = set()
        score_indices = []
        for i in candidate_indices:
            if i in seen or i not in unanswered_set:
                continue
            seen.add(i)
            score_indices.append(i)
    answered_qa = [qa for qa in qa_pairs if qa.answer is not None]

    log: dict = {
        "method": method,
        "include_policy": include_policy,
        "default_knowledge_length": len(default_knowledge),
        "num_unanswered_scored": len(score_indices),
        "num_unanswered_projection": len(unanswered_indices),
        "candidate_indices": score_indices,
        "projection_unanswered_indices": unanswered_indices,
        "num_answered_context": len(answered_qa),
        "per_question": [],
    }

    if len(score_indices) == 0:
        log["note"] = "no unanswered candidate questions to score"
        return {}, 0.0, log

    if len(unanswered_indices) < 2:
        # With 0 or 1 unanswered questions the answer vector is empty, so
        # the score degenerates to 0. Return zeros and skip LLM calls.
        scores = {i: 0.0 for i in score_indices}
        log["note"] = "fewer than 2 unanswered questions — skipped scoring"
        return scores, 0.0, log

    sem = asyncio.Semaphore(max(1, max_concurrent))

    async def score_one(idx_in_qa: int) -> tuple[int, float, dict]:
        q_i = qa_pairs[idx_in_qa]
        other_indices = [j for j in unanswered_indices if j != idx_in_qa]
        other_questions = [qa_pairs[j].question for j in other_indices]

        async with sem:
            if method == "full":
                (
                    b_pos,
                    cost_rewrite_pos,
                    rewrite_pos_artifact,
                ), (
                    b_neg,
                    cost_rewrite_neg,
                    rewrite_neg_artifact,
                ) = await asyncio.gather(
                    _rewrite_beliefs_full(
                        config,
                        beliefs,
                        answered_qa,
                        q_i.question,
                        True,
                        include_policy,
                        default_knowledge,
                    ),
                    _rewrite_beliefs_full(
                        config,
                        beliefs,
                        answered_qa,
                        q_i.question,
                        False,
                        include_policy,
                        default_knowledge,
                    ),
                )
            else:  # light
                b_pos = _append_assumption_light(beliefs, q_i.question, True)
                b_neg = _append_assumption_light(beliefs, q_i.question, False)
                cost_rewrite_pos = 0.0
                cost_rewrite_neg = 0.0
                rewrite_pos_artifact = {
                    "kind": "append_assumption_light",
                    "assumed_question": q_i.question,
                    "assumed_yes": True,
                    "default_knowledge_length": len(default_knowledge),
                    "prompt": None,
                    "response": None,
                    "used_fallback_light": False,
                    "updated_beliefs": b_pos,
                }
                rewrite_neg_artifact = {
                    "kind": "append_assumption_light",
                    "assumed_question": q_i.question,
                    "assumed_yes": False,
                    "default_knowledge_length": len(default_knowledge),
                    "prompt": None,
                    "response": None,
                    "used_fallback_light": False,
                    "updated_beliefs": b_neg,
                }

            (
                a_pos,
                cost_pred_pos,
                _,
                _,
                predict_pos_artifact,
            ), (
                a_neg,
                cost_pred_neg,
                _,
                _,
                predict_neg_artifact,
            ) = await asyncio.gather(
                _predict_answer_vector(config, b_pos, other_questions, default_knowledge),
                _predict_answer_vector(config, b_neg, other_questions, default_knowledge),
            )

        score = _hamming_fraction(a_pos, a_neg)
        q_cost = cost_rewrite_pos + cost_rewrite_neg + cost_pred_pos + cost_pred_neg
        q_log = {
            "idx": idx_in_qa,
            "question": q_i.question,
            "source_step": q_i.source_step,
            "score": score,
            "cost": q_cost,
            "a_pos": a_pos,
            "a_neg": a_neg,
            "other_indices": other_indices,
            "other_questions": other_questions,
            "b_pos_excerpt": b_pos[:400],
            "b_neg_excerpt": b_neg[:400],
            "b_pos": b_pos,
            "b_neg": b_neg,
            "rewrite_pos": rewrite_pos_artifact,
            "rewrite_neg": rewrite_neg_artifact,
            "predict_pos": predict_pos_artifact,
            "predict_neg": predict_neg_artifact,
        }
        return idx_in_qa, score, q_log

    results = await asyncio.gather(*(score_one(i) for i in score_indices))

    scores: dict[int, float] = {}
    total_cost = 0.0
    for idx, score, q_log in results:
        scores[idx] = score
        total_cost += q_log["cost"]
        log["per_question"].append(q_log)

    log["total_cost"] = total_cost
    log["per_question"].sort(key=lambda d: d["score"], reverse=True)

    logging.info(
        f"[question_scoring] scored {len(scores)} questions via method={method}; "
        f"cost=${total_cost:.6f}"
    )
    return scores, total_cost, log
