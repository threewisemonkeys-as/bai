"""LLM improvement functions for stepwise EB-learning.

EB-learning is experiment-driven: the agent generates questions about the
environment, designs experiments to answer them, and updates its Q&A knowledge
base from trajectory evidence.  These functions handle question generation,
experiment formulation, and Q&A updates — plus serialization for the nullable-
answer EBQAPair dataclass.

Improvement tracks (beliefs, perception, QA-based improvement) are reused
directly from stepwise_b_learn_improve and b_learn_improve.
"""

import logging
import re
from dataclasses import asdict, dataclass

from omegaconf import DictConfig

from llm_utils import extract_xml_key
from mixed_improve import (
    QAPair,
    _llm_call,
    _run_perception_on_observation,
)
from stepwise_b_learn import format_current_state


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------


@dataclass
class EBQAPair:
    """A question-answer pair where the answer can be None (unanswered)."""
    question: str
    answer: bool | None  # None = unanswered, True = YES, False = NO
    evidence: str         # empty string if unanswered
    source_step: int


def serialize_eb_qa_pairs(qa_pairs: list[EBQAPair]) -> list[dict]:
    return [asdict(qa) for qa in qa_pairs]


def deserialize_eb_qa_pairs(data: list[dict]) -> list[EBQAPair]:
    return [EBQAPair(**d) for d in data]


def eb_qa_to_qa(eb_qa: EBQAPair) -> QAPair:
    """Convert an answered EBQAPair to a QAPair for use with existing Track 2 functions."""
    assert eb_qa.answer is not None, "Cannot convert unanswered EBQAPair to QAPair"
    return QAPair(
        question=eb_qa.question,
        answer=eb_qa.answer,
        evidence=eb_qa.evidence,
        source_step=eb_qa.source_step,
    )


# ---------------------------------------------------------------------------
# Raw-obs stripping helper
# ---------------------------------------------------------------------------


def _strip_raw_state_text(steps_context: str) -> str:
    """Replace <pre_state> and <post_state> text content with a placeholder.

    Image annotations on the opening tag (e.g. ``<pre_state> (image 3)``) are
    preserved so the LLM can still cross-reference screenshots.
    """
    steps_context = re.sub(
        r"(<pre_state[^>]*>[^\n]*)\n.*?(\n</pre_state>)",
        r"\1\n(see attached image)\2",
        steps_context,
        flags=re.DOTALL,
    )
    steps_context = re.sub(
        r"(<post_state[^>]*>[^\n]*)\n.*?(\n</post_state>)",
        r"\1\n(see attached image)\2",
        steps_context,
        flags=re.DOTALL,
    )
    return steps_context


def _strip_raw_grid_text(text: str) -> str:
    """Remove raw grid dumps from persisted evidence before reusing it in prompts."""
    text = re.sub(
        r"<grid_\d+>\n.*?\n</grid_\d+>",
        "(raw grid omitted)",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"=+ Start of Direct Observation =+\n.*?\n=+ End of Direct Observation =+",
        "(raw observation omitted)",
        text,
        flags=re.DOTALL,
    )
    return text


# ---------------------------------------------------------------------------
# Question formatting helpers
# ---------------------------------------------------------------------------


def _format_qa_list(qa_pairs: list[EBQAPair]) -> str:
    """Format current Q for inclusion in prompts."""
    if not qa_pairs:
        return "(no questions yet)"
    lines = []
    for i, qa in enumerate(qa_pairs, 1):
        if qa.answer is None:
            status = "UNANSWERED"
        elif qa.answer:
            status = "YES"
        else:
            status = "NO"
        evidence = _strip_raw_grid_text(qa.evidence) if qa.evidence else ""
        evidence_part = f" (evidence: {evidence})" if evidence else ""
        lines.append(f"Q{i}: {qa.question} -> {status}{evidence_part}")
    return "\n".join(lines)


def _parse_1_based_indices(text: str, max_index: int) -> list[int]:
    """Parse unique 1-based Q indices from an LLM response fragment."""
    indices: list[int] = []
    seen: set[int] = set()
    for match in re.finditer(r"\d+", text or ""):
        idx = int(match.group(0)) - 1
        if 0 <= idx < max_index and idx not in seen:
            seen.add(idx)
            indices.append(idx)
    return indices


def _extract_attr(attrs: str, name: str) -> str | None:
    match = re.search(
        rf"""\b{name}\s*=\s*(?:"([^"]+)"|'([^']+)'|([^\s>]+))""",
        attrs,
        re.IGNORECASE,
    )
    if not match:
        return None
    return next(group for group in match.groups() if group is not None)


def _normalize_question_index(raw: str | None, max_index: int) -> int | None:
    """Parse a 1-based question reference like Q3 or 3 into a 0-based index."""
    if raw is None:
        return None
    match = re.search(r"(?:\bQ\s*)?(\d+)", raw, re.IGNORECASE)
    if not match:
        return None
    idx = int(match.group(1)) - 1
    if 0 <= idx < max_index:
        return idx
    return None


def _iter_q_blocks(text: str, max_index: int) -> list[tuple[int | None, str]]:
    """Return <q n="Q1">...</q> blocks, accepting n="1" as well."""
    blocks: list[tuple[int | None, str]] = []
    for match in re.finditer(
        r"<q\b(?P<attrs>[^>]*)>(?P<body>.*?)</q>",
        text or "",
        re.DOTALL | re.IGNORECASE,
    ):
        attrs = match.group("attrs")
        idx = _normalize_question_index(_extract_attr(attrs, "n"), max_index)
        blocks.append((idx, match.group("body")))
    return blocks


def _parse_q_tag_indices(text: str, max_index: int) -> list[int]:
    """Parse indices from <q n="Q1" /> or legacy <q source_index="1" /> tags."""
    indices: list[int] = []
    seen: set[int] = set()
    for match in re.finditer(r"<q\b(?P<attrs>[^>]*)/?>", text or "", re.IGNORECASE):
        attrs = match.group("attrs")
        idx = _normalize_question_index(_extract_attr(attrs, "n"), max_index)
        if idx is None:
            idx = _normalize_question_index(
                _extract_attr(attrs, "source_index"), max_index,
            )
        if idx is not None and idx not in seen:
            seen.add(idx)
            indices.append(idx)
    return indices


# ---------------------------------------------------------------------------
# Question generation
# ---------------------------------------------------------------------------


async def generate_questions_from_steps(
    config: DictConfig,
    beliefs: str,
    perception_code: str,
    steps_context: str,
    current_qa: list[EBQAPair],
    current_observation: str | None,
    current_aux_observation: str | None,
    default_knowledge: str,
    num_questions: int,
    current_step: int = 0,
    current_image=None,
    steps_context_images: list | None = None,
    hide_raw_obs: bool = False,
    include_recent_history: bool = True,
) -> tuple[list[EBQAPair], float, str, str]:
    """Generate new unanswered questions about the environment.

    Prompts the LLM with the agent's current beliefs, perception, optional
    trajectory history, current state, and existing Q.  Returns N new EBQAPair
    with answer=None.

    Returns: (new_questions, cost, prompt, raw_response)
    """
    step_history = steps_context
    if hide_raw_obs and include_recent_history and steps_context_images:
        step_history = _strip_raw_state_text(step_history)
    num_steps_images = (
        len(steps_context_images)
        if include_recent_history and steps_context_images
        else 0
    )

    current_image_index = num_steps_images + 1 if current_image is not None else None
    current_obs_section = format_current_state(
        observation=current_observation,
        aux_observation=current_aux_observation,
        perception_code=perception_code,
        image=current_image,
        image_index=current_image_index,
        hide_raw_obs=hide_raw_obs,
    )

    qa_list_text = _format_qa_list(current_qa)
    recent_history_section = ""
    if include_recent_history:
        recent_history_section = f"""
Each ``<pre_state>`` (and ``<post_state>``, when present) below is annotated with an ``(image K)`` marker referring to the K-th (1-indexed) screenshot attached to this message — use these to cross-reference the textual observation with the actual visual state. ``<pre_state>`` is the observation before the step's action; ``<post_state>`` is the final observation of a past episode segment.

=== RECENT HISTORY OF STATES AND ACTIONS ===
{step_history if step_history else "(no steps recorded yet)"}
=== END RECENT HISTORY ===
"""

    prompt = f"""You are generating questions to guide an agent's exploration of an environment.

=== DEFAULT KNOWLEDGE ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== CURRENT BELIEFS ===
{beliefs if beliefs else "(empty - no beliefs yet)"}
=== END CURRENT BELIEFS ===
{recent_history_section}
{current_obs_section}
=== CURRENT QUESTIONS ===
{qa_list_text}
=== END CURRENT QUESTIONS ===

Your task: Generate new binary (yes/no) questions about how the environment works.

Guidelines:
- Questions should be general in scope, asking about how the world works.
- Do not duplicate questions already in the current questions list.
- Focus on questions whose answers would be most useful for improving the agent's current beliefs.
- Each question must be a specific yes/no question, not open-ended.

Format your response as:
<think>
What aspects of the environment are we most uncertain about? What questions would help us learn the most?
</think>
<questions>
<q n="Q1">
<question>[A specific yes/no question about how the environment works]</question>
</q>
<q n="Q2">
<question>[Another specific yes/no question about how the environment works]</question>
</q>
...
(Generate questions)
</questions>"""

    images: list = []
    if include_recent_history and steps_context_images:
        images.extend(steps_context_images)
    if current_image is not None:
        images.append(current_image)

    text, cost = await _llm_call(config, prompt, images=images or None)

    questions_text = extract_xml_key(text, "questions")
    new_questions: list[EBQAPair] = []
    # Build set of existing question texts for deduplication
    existing_questions = {q.question.strip().lower() for q in current_qa}
    if questions_text:
        parsed_question_texts = [
            extract_xml_key(q_body, "question")
            for _, q_body in _iter_q_blocks(questions_text, num_questions)
        ]
        if not any(parsed_question_texts):
            parsed_question_texts = [
                match.group(1).strip()
                for match in re.finditer(
                    r"Q\s*\d+:\s*(.+?)(?=Q\s*\d+:|$)",
                    questions_text,
                    re.DOTALL | re.IGNORECASE,
                )
            ]
        for raw_q in parsed_question_texts:
            q = (raw_q or "").strip()
            if q:
                if len(q) > 300:
                    q = q[:300].rsplit(" ", 1)[0] + "..."
                # Skip duplicates of existing or already-added questions
                q_lower = q.strip().lower()
                if q_lower in existing_questions:
                    continue
                existing_questions.add(q_lower)
                new_questions.append(EBQAPair(
                    question=q,
                    answer=None,
                    evidence="",
                    source_step=current_step,
                ))

    logging.info(f"Generated {len(new_questions)} new questions from step history")
    return new_questions, cost, prompt, text


# ---------------------------------------------------------------------------
# Experiment formulation from questions
# ---------------------------------------------------------------------------


async def formulate_experiment_from_question(
    config: DictConfig,
    beliefs: str,
    perception_code: str,
    steps_context: str,
    current_qa: list[EBQAPair],
    current_experiment: str | None,
    current_observation: str | None,
    current_aux_observation: str | None,
    default_knowledge: str,
    current_image=None,
    steps_context_images: list | None = None,
    hide_raw_obs: bool = False,
    target_question_index: int | None = None,
) -> tuple[str | None, int | None, float, str, str]:
    """Select an unanswered question from Q and formulate an experiment to answer it.

    Returns: (experiment_text, selected_question_index, cost, prompt, raw_response)
    If the LLM returns "null", returns (None, None, cost, prompt, response) to keep
    the current experiment.
    """
    step_history = steps_context
    if hide_raw_obs and steps_context_images:
        step_history = _strip_raw_state_text(step_history)
    num_steps_images = len(steps_context_images) if steps_context_images else 0

    current_image_index = num_steps_images + 1 if current_image is not None else None
    current_obs_section = format_current_state(
        observation=current_observation,
        aux_observation=current_aux_observation,
        perception_code=perception_code,
        image=current_image,
        image_index=current_image_index,
        hide_raw_obs=hide_raw_obs,
        section_title="CURRENT STATE (agent has not yet acted)",
    )

    current_exp_text = current_experiment if current_experiment else "(no experiment set yet)"
    target_question_section = ""
    available_questions_section = ""
    task_instruction = (
        "Your task: Decide whether to keep the current experiment or formulate a new one for one unanswered question."
    )
    target_question_instruction = (
        "Otherwise, select one UNANSWERED question from AVAILABLE QUESTIONS and formulate a specific, actionable experiment (1-3 sentences) that the agent can carry out to answer it."
    )
    think_instruction = (
        "Has the current experiment been sufficiently tested? If not, which unanswered question should be answered next?"
    )
    question_index_format = (
        '<q n="Q1">\n'
        "<experiment_plan>[1-3 sentence actionable experiment to answer the selected question]</experiment_plan>"
        "\n</q>"
    )
    unanswered_question_lines = [
        f"Q{i + 1}: {qa.question}"
        for i, qa in enumerate(current_qa)
        if qa.answer is None
    ]
    if unanswered_question_lines:
        available_questions_section = f"""
=== AVAILABLE QUESTIONS ===
{chr(10).join(unanswered_question_lines)}
=== END AVAILABLE QUESTIONS ===
"""
    if (
        target_question_index is not None
        and 0 <= target_question_index < len(current_qa)
        and current_qa[target_question_index].answer is None
    ):
        target_question_section = f"""
=== TARGET QUESTION ===
Q{target_question_index + 1}: {current_qa[target_question_index].question}
=== END TARGET QUESTION ===
"""
        task_instruction = (
            "Your task: Decide whether to keep the current experiment or formulate a new one for the TARGET QUESTION."
        )
        target_question_instruction = (
            "Otherwise, formulate a specific, actionable experiment (1-3 sentences) to answer the TARGET QUESTION above.\n"
            f'Use <q n="Q{target_question_index + 1}"> to identify the fixed target question.'
        )
        think_instruction = (
            "Has the current experiment been sufficiently tested? If not, what experiment would answer the TARGET QUESTION?"
        )
        question_index_format = (
            f'<q n="Q{target_question_index + 1}">\n'
            "<experiment_plan>[1-3 sentence actionable experiment to answer the TARGET QUESTION]</experiment_plan>"
            "\n</q>"
        )

    prompt = f"""You are designing the next experiment for an agent interacting with an environment.

=== DEFAULT KNOWLEDGE ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== CURRENT BELIEFS ===
{beliefs if beliefs else "(empty - no beliefs yet)"}
=== END CURRENT BELIEFS ===

Each ``<pre_state>`` (and ``<post_state>``, when present) below is annotated with an ``(image K)`` marker referring to the K-th (1-indexed) screenshot attached to this message — use these to cross-reference the textual observation with the actual visual state. ``<pre_state>`` is the observation before the step's action; ``<post_state>`` is the final observation of a past episode segment.

=== RECENT HISTORY OF STATES AND ACTIONS ===
{step_history}
=== END RECENT HISTORY ==={current_obs_section}
{target_question_section}
{available_questions_section if not target_question_section else ""}

=== CURRENT EXPERIMENT ===
{current_exp_text}
=== END CURRENT EXPERIMENT ===

{task_instruction}

If the current experiment is still being tested (the agent hasn't had enough steps to gather evidence), return null.
{target_question_instruction}

Format your response as:
<think>
{think_instruction}
</think>
<experiment>
If keeping the current experiment:
null

If formulating a new experiment:
{question_index_format}
</experiment>"""

    images: list = []
    if steps_context_images:
        images.extend(steps_context_images)
    if current_image is not None:
        images.append(current_image)

    text, cost = await _llm_call(config, prompt, images=images or None)

    experiment_text_raw = extract_xml_key(text, "experiment")
    if not experiment_text_raw or experiment_text_raw.strip().lower() == "null":
        return None, None, cost, prompt, text

    question_index = None
    experiment_plan = None
    q_blocks = _iter_q_blocks(experiment_text_raw, len(current_qa))
    if q_blocks:
        question_index, q_body = q_blocks[0]
        experiment_plan = extract_xml_key(q_body, "experiment_plan")

    question_index_str = extract_xml_key(experiment_text_raw, "question_index")
    if question_index_str:
        question_index = _normalize_question_index(question_index_str, len(current_qa))

    if question_index is not None and current_qa[question_index].answer is not None:
        logging.warning(
            f"Experiment question_index {question_index + 1} is not unanswered, ignoring index"
        )
        question_index = None
    elif question_index is None and question_index_str:
        logging.warning(
            f"Experiment question_index {question_index_str!r} is out of range "
            f"(total={len(current_qa)}), ignoring index"
        )

    if experiment_plan is None:
        experiment_plan = extract_xml_key(experiment_text_raw, "experiment_plan")
    if not experiment_plan:
        # Fallback: treat the whole experiment block as the plan
        experiment_plan = experiment_text_raw.strip()

    if len(experiment_plan) > 300:
        experiment_plan = experiment_plan[:300].rsplit(" ", 1)[0] + "..."

    if (
        target_question_index is not None
        and 0 <= target_question_index < len(current_qa)
        and current_qa[target_question_index].answer is None
    ):
        question_index = target_question_index

    logging.info(f"Formulated experiment from question Q{(question_index or 0) + 1}: {experiment_plan[:80]}...")
    return experiment_plan, question_index, cost, prompt, text


# ---------------------------------------------------------------------------
# Q&A update from trajectory
# ---------------------------------------------------------------------------


async def update_qa_from_trajectory(
    config: DictConfig,
    current_qa: list[EBQAPair],
    steps_context: str,
    current_step: int = 0,
    steps_context_images: list | None = None,
    hide_raw_obs: bool = False,
) -> tuple[list[EBQAPair], float, dict]:
    """Update Q&A pairs from trajectory evidence.

    A single LLM call that:
    1. Answers unanswered questions if the trajectory provides evidence
    2. Corrects existing answers if trajectory contradicts them
    3. Adds new questions discovered from the trajectory
    Existing questions are preserved even if the model omits them from its
    response; pruning is handled by the separate trim phase.

    Returns: (updated_qa_pairs, cost, extraction_log)
    """
    if not steps_context:
        return current_qa, 0.0, {}

    display_steps_context = steps_context
    if hide_raw_obs and steps_context_images:
        display_steps_context = _strip_raw_state_text(display_steps_context)

    qa_list_text = _format_qa_list(current_qa)

    prompt = f"""You are analyzing an agent's trajectory to update our knowledge base of questions and answers about the environment.

Each ``<pre_state>`` (and ``<post_state>``, when present) in the sequence below is annotated with an ``(image K)`` marker referring to the K-th (1-indexed) screenshot attached to this message — use these to cross-reference the textual observation with the actual visual state. ``<pre_state>`` is the observation before the step's action; ``<post_state>`` is the observation after the step's action (shown for the final step of each episode segment).

=== SEQUENCE OF STEPS ===
{display_steps_context}
=== END SEQUENCE OF STEPS ===

=== CURRENT QUESTIONS ===
{qa_list_text}
=== END CURRENT QUESTIONS ===

Your task: Update the questions list based on evidence from the trajectory.

1. For each UNANSWERED question: If the trajectory provides clear, unambiguous evidence, answer it (YES or NO) with supporting evidence quoted from the trajectory.
2. For each ANSWERED question: If the trajectory provides evidence that contradicts the current answer, update the answer and evidence. Otherwise keep it unchanged.
3. If the trajectory reveals important aspects of how the environment works that aren't covered by existing questions, add NEW questions (with answers if evidence is available, otherwise as UNANSWERED).

Only answer questions when the trajectory provides clear evidence. Do not guess or infer beyond what is directly observed.

Format your response as:
<think>
Review the trajectory and each question. What can we learn?
</think>
<updated_questions>
<q n="Q1">
<question>[question text]</question>
<evidence>[evidence from trajectory, or empty if unanswered]</evidence>
<answer>YES or NO or UNANSWERED</answer>
</q>
<q n="Q2">
...
</q>
...
(Use existing Q numbers for existing questions. Number new questions as Q{len(current_qa) + 1}, Q{len(current_qa) + 2}, etc. Include all existing questions plus any new ones.)
</updated_questions>"""

    text, cost = await _llm_call(
        config, prompt, images=steps_context_images or None,
    )

    extraction_log = {
        "prompt": prompt,
        "response": text,
    }

    updated_questions_text = extract_xml_key(text, "updated_questions")
    if not updated_questions_text:
        extraction_log["parse_error"] = "No <updated_questions> block found"
        return current_qa, cost, extraction_log

    parsed_by_index: dict[int, EBQAPair] = {}
    parsed_by_question: dict[str, EBQAPair] = {}
    parsed_new_questions: list[EBQAPair] = []
    for parsed_idx, q_content in _iter_q_blocks(updated_questions_text, 10**9):
        question = extract_xml_key(q_content, "question")
        answer_str = extract_xml_key(q_content, "answer")
        evidence = extract_xml_key(q_content, "evidence") or ""

        if not question:
            continue

        answer: bool | None = None
        if answer_str:
            answer_upper = answer_str.strip().upper()
            if answer_upper == "YES":
                answer = True
            elif answer_upper == "NO":
                answer = False
            # else UNANSWERED -> None

        existing_by_index = (
            current_qa[parsed_idx]
            if parsed_idx is not None and 0 <= parsed_idx < len(current_qa)
            else None
        )
        existing_by_text = None
        if existing_by_index is None:
            for existing in current_qa:
                if existing.question.strip().lower() == question.strip().lower():
                    existing_by_text = existing
                    break

        source_question = (
            existing_by_index.question
            if existing_by_index is not None
            else question.strip()
        )
        source_step = (
            existing_by_index.source_step
            if existing_by_index is not None
            else existing_by_text.source_step
            if existing_by_text is not None
            else current_step
        )

        parsed = EBQAPair(
            question=source_question,
            answer=answer,
            evidence=evidence.strip(),
            source_step=source_step,
        )

        if existing_by_index is not None:
            parsed_by_index[parsed_idx] = parsed
        elif existing_by_text is not None:
            parsed_by_question[existing_by_text.question.strip().lower()] = parsed
        else:
            parsed_new_questions.append(parsed)

    if not parsed_by_index and not parsed_by_question and not parsed_new_questions:
        extraction_log["parse_error"] = "No valid <q> entries parsed"
        return current_qa, cost, extraction_log

    updated_qa: list[EBQAPair] = []
    existing_keys: set[str] = set()
    for idx, existing in enumerate(current_qa):
        key = existing.question.strip().lower()
        existing_keys.add(key)
        parsed = parsed_by_index.get(idx) or parsed_by_question.get(key)
        if parsed is None:
            updated_qa.append(existing)
        elif existing.answer is not None and parsed.answer is None:
            # Do not let an update pass accidentally erase an established answer.
            updated_qa.append(existing)
        else:
            updated_qa.append(parsed)

    for qa in parsed_new_questions:
        key = qa.question.strip().lower()
        if key not in existing_keys:
            updated_qa.append(qa)
            existing_keys.add(key)

    prev_unanswered = sum(1 for q in current_qa if q.answer is None)
    new_unanswered = sum(1 for q in updated_qa if q.answer is None)
    prev_by_question = {q.question.strip().lower(): q for q in current_qa}
    newly_answered = sum(
        1
        for q in updated_qa
        if q.answer is not None
        and (prev := prev_by_question.get(q.question.strip().lower())) is not None
        and prev.answer is None
    )
    extraction_log["prev_count"] = len(current_qa)
    extraction_log["new_count"] = len(updated_qa)
    extraction_log["prev_unanswered"] = prev_unanswered
    extraction_log["new_unanswered"] = new_unanswered
    extraction_log["newly_answered"] = newly_answered
    extraction_log["new_questions"] = sum(
        1 for q in updated_qa if q.question.strip().lower() not in prev_by_question
    )

    logging.info(
        f"Q&A update: {len(current_qa)} -> {len(updated_qa)} questions "
        f"(unanswered: {prev_unanswered} -> {new_unanswered})"
    )

    return updated_qa, cost, extraction_log


# ---------------------------------------------------------------------------
# Q&A trimming
# ---------------------------------------------------------------------------


async def trim_qa_pairs(
    config: DictConfig,
    current_qa: list[EBQAPair],
    max_answered_qa_pairs: int,
    max_unanswered_qa_pairs: int,
    current_step: int = 0,
) -> tuple[list[EBQAPair], float, dict]:
    """Trim the Q&A list to the answered and unanswered caps.

    Asks the LLM to decide which questions to keep based on usefulness.
    Should only be called when either status-specific cap is exceeded.

    Returns: (trimmed_qa_pairs, cost, trim_log)
    """
    qa_list_text = _format_qa_list(current_qa)
    num_answered = sum(1 for q in current_qa if q.answer is not None)
    num_unanswered = sum(1 for q in current_qa if q.answer is None)

    prompt = f"""You are maintaining a knowledge base of questions and answers about an environment.

The knowledge base currently has {len(current_qa)} questions:
- {num_answered} ANSWERED questions
- {num_unanswered} UNANSWERED questions

We need to trim it so that it has at most:
- {max_answered_qa_pairs} ANSWERED questions
- {max_unanswered_qa_pairs} UNANSWERED questions

=== CURRENT QUESTIONS ===
{qa_list_text}
=== END CURRENT QUESTIONS ===

Your task: Select the most useful questions to keep while satisfying both caps. Drop questions that are:
- Redundant (covered by other questions)
- No longer useful or too narrow in scope
- Superseded by better questions on the same topic

Prefer to keep:
- Answered questions with clear evidence (they represent confirmed knowledge)
- Unanswered questions that would be most valuable to answer next
- Questions that cover distinct, important aspects of the environment

Format your response as:
<think>
Which questions are most valuable? Which can be dropped?
</think>
<trimmed_questions>
<q n="Q1">
<question>[question text]</question>
<answer>YES or NO or UNANSWERED</answer>
<evidence>[evidence, or empty if unanswered]</evidence>
</q>
<q n="Q2">
...
</q>
...
(Use the original Q numbers from CURRENT QUESTIONS. Include at most {max_answered_qa_pairs} ANSWERED questions and at most {max_unanswered_qa_pairs} UNANSWERED questions)
</trimmed_questions>"""

    text, cost = await _llm_call(config, prompt)

    trim_log: dict = {
        "prompt": prompt,
        "response": text,
        "pre_trim_count": len(current_qa),
        "pre_trim_answered": num_answered,
        "pre_trim_unanswered": num_unanswered,
        "max_answered_qa_pairs": max_answered_qa_pairs,
        "max_unanswered_qa_pairs": max_unanswered_qa_pairs,
    }

    trimmed_text = extract_xml_key(text, "trimmed_questions")
    if not trimmed_text:
        trim_log["parse_error"] = "No <trimmed_questions> block found"
        return current_qa, cost, trim_log

    trimmed_qa: list[EBQAPair] = []
    for parsed_idx, q_content in _iter_q_blocks(trimmed_text, len(current_qa)):
        question = extract_xml_key(q_content, "question")
        answer_str = extract_xml_key(q_content, "answer")
        evidence = extract_xml_key(q_content, "evidence") or ""

        if not question:
            continue

        if parsed_idx is not None and 0 <= parsed_idx < len(current_qa):
            trimmed_qa.append(current_qa[parsed_idx])
            continue

        answer: bool | None = None
        if answer_str:
            answer_upper = answer_str.strip().upper()
            if answer_upper == "YES":
                answer = True
            elif answer_upper == "NO":
                answer = False

        # Preserve source_step from existing Q; use current_step for unknown
        source_step = current_step
        for existing in current_qa:
            if existing.question.strip().lower() == question.strip().lower():
                source_step = existing.source_step
                break

        trimmed_qa.append(EBQAPair(
            question=question.strip(),
            answer=answer,
            evidence=evidence.strip(),
            source_step=source_step,
        ))

    if not trimmed_qa:
        trim_log["parse_error"] = "No valid <q> entries parsed"
        return current_qa, cost, trim_log

    trim_log["post_trim_count"] = len(trimmed_qa)
    trim_log["post_trim_answered"] = sum(1 for q in trimmed_qa if q.answer is not None)
    trim_log["post_trim_unanswered"] = sum(1 for q in trimmed_qa if q.answer is None)
    trim_log["dropped_count"] = len(current_qa) - len(trimmed_qa)

    logging.info(
        f"Q&A trim: {len(current_qa)} -> {len(trimmed_qa)} questions "
        f"(answered: {num_answered} -> {trim_log['post_trim_answered']}, "
        f"unanswered: {num_unanswered} -> {trim_log['post_trim_unanswered']}, "
        f"dropped {len(current_qa) - len(trimmed_qa)})"
    )

    return trimmed_qa, cost, trim_log


# ---------------------------------------------------------------------------
# Probe-based Q&A maintenance and selection
# ---------------------------------------------------------------------------


async def deduplicate_qa_pairs(
    config: DictConfig,
    current_qa: list[EBQAPair],
) -> tuple[list[EBQAPair], float, dict]:
    """Drop duplicate questions while preserving the full non-duplicate bank.

    This intentionally does not trim for usefulness. It only asks which
    questions are semantic duplicates of another currently accumulated
    question.
    """
    qa_list_text = _format_qa_list(current_qa)
    prompt = f"""You are maintaining a knowledge base of binary questions about an environment.

=== CURRENT QUESTIONS ===
{qa_list_text}
=== END CURRENT QUESTIONS ===

Your task: identify only duplicate or near-duplicate questions that should be dropped. Do not drop a question because it is low priority, narrow, old, or currently unanswered.

When two questions are duplicates, prefer to keep:
- An answered question with clear evidence over an unanswered question
- The clearer or more general wording
- The earlier question if both are otherwise equivalent

Format your response as:
<think>
Which questions are duplicates of another question in the list?
</think>
<duplicate_drop_indices>
<q n="Q2" />
<q n="Q5" />
(Use the original Q numbers from CURRENT QUESTIONS. Write NONE if no questions should be dropped.)
</duplicate_drop_indices>"""

    text, cost = await _llm_call(config, prompt)
    drop_text = extract_xml_key(text, "duplicate_drop_indices")
    dedup_log: dict = {
        "method": "deduplicate_only",
        "prompt": prompt,
        "response": text,
        "pre_dedup_count": len(current_qa),
        "pre_dedup_answered": sum(1 for q in current_qa if q.answer is not None),
        "pre_dedup_unanswered": sum(1 for q in current_qa if q.answer is None),
    }

    if drop_text is None:
        dedup_log["parse_error"] = "No <duplicate_drop_indices> block found"
        dedup_log["post_dedup_count"] = len(current_qa)
        dedup_log["dropped_indices"] = []
        dedup_log["dropped_questions"] = []
        return current_qa, cost, dedup_log

    drop_indices = _parse_q_tag_indices(drop_text, len(current_qa))
    if not drop_indices:
        drop_indices = _parse_1_based_indices(drop_text, len(current_qa))
    drop_set = set(drop_indices)
    deduped = [q for i, q in enumerate(current_qa) if i not in drop_set]

    dedup_log["post_dedup_count"] = len(deduped)
    dedup_log["post_dedup_answered"] = sum(1 for q in deduped if q.answer is not None)
    dedup_log["post_dedup_unanswered"] = sum(1 for q in deduped if q.answer is None)
    dedup_log["dropped_count"] = len(current_qa) - len(deduped)
    dedup_log["dropped_indices"] = drop_indices
    dedup_log["dropped_questions"] = [current_qa[i].question for i in drop_indices]

    logging.info(
        f"Q&A dedup: {len(current_qa)} -> {len(deduped)} questions "
        f"(dropped {len(current_qa) - len(deduped)} duplicates)"
    )
    return deduped, cost, dedup_log


async def select_qa_pairs_for_experiment(
    config: DictConfig,
    current_qa: list[EBQAPair],
    max_answered_qa_pairs: int,
    max_unanswered_qa_pairs: int,
    default_knowledge: str = "",
    beliefs: str = "",
) -> tuple[list[EBQAPair], list[int], float, dict]:
    """Select a capped question subset for experiment formulation.

    The returned subset is a view over current_qa, represented both as copied
    EBQAPair objects and as source indices into current_qa. It should be used
    for experiment selection only, not persisted as the maintained question
    bank.
    """
    num_answered = sum(1 for q in current_qa if q.answer is not None)
    num_unanswered = sum(1 for q in current_qa if q.answer is None)
    unanswered_source_indices = [
        i for i, qa in enumerate(current_qa) if qa.answer is None
    ]
    selection_log: dict = {
        "method": "llm_top_k_probe_selection",
        "pre_selection_count": len(current_qa),
        "pre_selection_answered": num_answered,
        "pre_selection_unanswered": num_unanswered,
        "max_answered_qa_pairs": max_answered_qa_pairs,
        "max_unanswered_qa_pairs": max_unanswered_qa_pairs,
        "default_knowledge_length": len(default_knowledge),
        "beliefs_length": len(beliefs),
        "candidate_source_indices": unanswered_source_indices,
    }

    if not unanswered_source_indices:
        selection_log.update({
            "note": "no unanswered questions available for experiment selection",
            "selected_source_indices": [],
            "post_selection_count": 0,
            "post_selection_answered": 0,
            "post_selection_unanswered": 0,
        })
        return [], [], 0.0, selection_log

    if num_unanswered <= max_unanswered_qa_pairs:
        indices = unanswered_source_indices
        selection_log.update({
            "note": "unanswered question bank within selection cap; selected all unanswered questions",
            "selected_source_indices": indices,
            "post_selection_count": len(indices),
            "post_selection_answered": 0,
            "post_selection_unanswered": len(indices),
        })
        return [current_qa[i] for i in indices], indices, 0.0, selection_log

    qa_list_text = "\n".join(
        f"Q{source_idx + 1}: {current_qa[source_idx].question} -> UNANSWERED"
        for source_idx in unanswered_source_indices
    )
    default_knowledge_section = ""
    if default_knowledge:
        default_knowledge_section = f"""
=== DEFAULT KNOWLEDGE ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===
"""
    beliefs_section = ""
    if beliefs:
        beliefs_section = f"""
=== CURRENT BELIEFS ===
{beliefs}
=== END CURRENT BELIEFS ===
"""
    prompt = f"""You are selecting the next experiment target for an agent learning an environment.

{default_knowledge_section}

{beliefs_section}

=== AVAILABLE QUESTIONS ===
{qa_list_text}
=== END AVAILABLE QUESTIONS ===

Select up to {max_unanswered_qa_pairs} questions that will -
1. be most valuable to answer next
2. cover distinct aspects of the environment

Use each question's Q number in the <q n="..."> attribute. Format your response as:
<think>
Which questions should be selected?
</think>
<selected_questions>
<q n="Q1" />
...
</selected_questions>"""

    text, cost = await _llm_call(config, prompt)
    selection_log["prompt"] = prompt
    selection_log["response"] = text

    selected_text = extract_xml_key(text, "selected_questions")
    selected_indices: list[int] = []
    unanswered_source_set = set(unanswered_source_indices)
    if selected_text:
        selected_indices = [
            idx for idx in _parse_q_tag_indices(selected_text, len(current_qa))
            if idx in unanswered_source_set
        ]
        if not selected_indices:
            selected_indices = [
                i for i in _parse_1_based_indices(selected_text, len(current_qa))
                if i in unanswered_source_set
            ]

    if not selected_indices:
        selection_log["parse_error"] = "No valid selected source indices parsed"
        selected_indices = unanswered_source_indices

    selected_indices = selected_indices[:max_unanswered_qa_pairs]
    selected_qa = [current_qa[i] for i in selected_indices]

    selection_log["selected_source_indices"] = selected_indices
    selection_log["selected_questions"] = [
        {
            "source_index": i,
            "question": current_qa[i].question,
            "answer": current_qa[i].answer,
            "source_step": current_qa[i].source_step,
        }
        for i in selected_indices
    ]
    selection_log["post_selection_count"] = len(selected_qa)
    selection_log["post_selection_answered"] = sum(
        1 for q in selected_qa if q.answer is not None
    )
    selection_log["post_selection_unanswered"] = sum(
        1 for q in selected_qa if q.answer is None
    )

    logging.info(
        f"Q&A probe selection: {len(current_qa)} -> {len(selected_qa)} prompt questions "
        f"(answered {selection_log['post_selection_answered']}, "
        f"unanswered {selection_log['post_selection_unanswered']})"
    )
    return selected_qa, selected_indices, cost, selection_log


# ---------------------------------------------------------------------------
# Scored Q&A trimming (B-difference based)
# ---------------------------------------------------------------------------


async def trim_qa_pairs_scored(
    config: DictConfig,
    current_qa: list[EBQAPair],
    max_answered_qa_pairs: int,
    max_unanswered_qa_pairs: int,
    beliefs: str,
    *,
    method: str,
    max_concurrent: int = 8,
    include_policy: bool = True,
    current_step: int = 0,
) -> tuple[list[EBQAPair], float, dict]:
    """Trim Q&A by B-difference scoring on unanswered questions.

    Delegates to ``trim_qa_pairs`` for the answered-cap case (when the
    answered count exceeds its cap) so we don't reinvent an answered-question
    selector. For the unanswered pool, computes B-difference scores and
    keeps the top ``max_unanswered_qa_pairs`` entries.

    Returns: (trimmed_qa, total_cost, trim_log)
    """
    # Lazy import to avoid a top-level cycle with question_scoring.
    from question_scoring import score_questions_b_diff

    total_cost = 0.0
    trim_log: dict = {
        "method": f"b_diff_{method}",
        "pre_trim_count": len(current_qa),
        "pre_trim_answered": sum(1 for q in current_qa if q.answer is not None),
        "pre_trim_unanswered": sum(1 for q in current_qa if q.answer is None),
        "max_answered_qa_pairs": max_answered_qa_pairs,
        "max_unanswered_qa_pairs": max_unanswered_qa_pairs,
    }

    # Step A: if answered cap exceeded, reuse the existing LLM trim for the
    # answered side. We call it with the original (answered_cap, inf-ish)
    # pair and then separately handle unanswered below. The prompt already
    # handles both caps, but we only want to shrink answered here.
    working_qa = list(current_qa)
    if trim_log["pre_trim_answered"] > max_answered_qa_pairs:
        answered_only = [q for q in working_qa if q.answer is not None]
        trimmed_answered, ans_cost, ans_log = await trim_qa_pairs(
            config=config,
            current_qa=answered_only,
            max_answered_qa_pairs=max_answered_qa_pairs,
            max_unanswered_qa_pairs=0,
            current_step=current_step,
        )
        total_cost += ans_cost
        trim_log["answered_trim"] = ans_log
        # Reassemble with existing unanswered in place
        unanswered_in_order = [q for q in working_qa if q.answer is None]
        working_qa = trimmed_answered + unanswered_in_order

    # Step B: score + drop unanswered if over cap
    unanswered = [q for q in working_qa if q.answer is None]
    if len(unanswered) > max_unanswered_qa_pairs:
        scores, score_cost, score_log = await score_questions_b_diff(
            config=config,
            beliefs=beliefs,
            qa_pairs=working_qa,
            method=method,
            include_policy=include_policy,
            max_concurrent=max_concurrent,
        )
        total_cost += score_cost

        # Rank unanswered by score desc, break ties by source_step desc (prefer newer)
        ranked = sorted(
            [i for i, q in enumerate(working_qa) if q.answer is None],
            key=lambda i: (scores.get(i, 0.0), working_qa[i].source_step),
            reverse=True,
        )
        keep_indices = set(ranked[:max_unanswered_qa_pairs])
        ranked_entries = [
            {
                "idx": i,
                "question": working_qa[i].question,
                "source_step": working_qa[i].source_step,
                "score": scores.get(i, 0.0),
            }
            for i in ranked
        ]
        kept_unanswered_indices = ranked[:max_unanswered_qa_pairs]
        dropped_unanswered_indices = ranked[max_unanswered_qa_pairs:]
        trim_log["scoring"] = {
            **score_log,
            "ranked_unanswered": ranked_entries,
            "kept_unanswered_indices": kept_unanswered_indices,
            "kept_unanswered_questions": [
                working_qa[i].question for i in kept_unanswered_indices
            ],
            "dropped_unanswered_indices": dropped_unanswered_indices,
            "dropped_unanswered_questions": [
                working_qa[i].question for i in dropped_unanswered_indices
            ],
            "answered_context": [
                {
                    "idx": i,
                    "question": q.question,
                    "answer": q.answer,
                    "evidence": q.evidence,
                    "source_step": q.source_step,
                }
                for i, q in enumerate(working_qa)
                if q.answer is not None
            ],
            "unanswered_pool": [
                {
                    "idx": i,
                    "question": q.question,
                    "source_step": q.source_step,
                }
                for i, q in enumerate(working_qa)
                if q.answer is None
            ],
        }

        working_qa = [
            q for i, q in enumerate(working_qa)
            if q.answer is not None or i in keep_indices
        ]
    else:
        trim_log["scoring"] = {
            "note": "unanswered within cap — no scoring performed",
            "num_unanswered": len(unanswered),
        }

    trim_log["post_trim_count"] = len(working_qa)
    trim_log["post_trim_answered"] = sum(1 for q in working_qa if q.answer is not None)
    trim_log["post_trim_unanswered"] = sum(1 for q in working_qa if q.answer is None)
    trim_log["dropped_count"] = len(current_qa) - len(working_qa)
    trim_log["total_cost"] = total_cost

    logging.info(
        f"Q&A trim (scored/{method}): {len(current_qa)} -> {len(working_qa)} questions "
        f"(answered: {trim_log['pre_trim_answered']} -> {trim_log['post_trim_answered']}, "
        f"unanswered: {trim_log['pre_trim_unanswered']} -> {trim_log['post_trim_unanswered']}, "
        f"dropped {trim_log['dropped_count']}, cost ${total_cost:.6f})"
    )

    return working_qa, total_cost, trim_log
