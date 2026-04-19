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
    """Replace <raw_state> and <resulting_state> text content with a placeholder.

    Image annotations on the opening tag (e.g. ``<raw_state> (image 3)``) are
    preserved so the LLM can still cross-reference screenshots.
    """
    steps_context = re.sub(
        r"(<raw_state[^>]*>[^\n]*)\n.*?(\n</raw_state>)",
        r"\1\n(see attached image)\2",
        steps_context,
        flags=re.DOTALL,
    )
    steps_context = re.sub(
        r"(<resulting_state[^>]*>[^\n]*)\n.*?(\n</resulting_state>)",
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
) -> tuple[list[EBQAPair], float, str, str]:
    """Generate new unanswered questions about the environment.

    Prompts the LLM with the agent's current beliefs, perception, trajectory
    history, current state, and existing Q.  Returns N new EBQAPair with
    answer=None.

    Returns: (new_questions, cost, prompt, raw_response)
    """
    step_history = steps_context
    if hide_raw_obs and steps_context_images:
        step_history = _strip_raw_state_text(step_history)
    num_steps_images = len(steps_context_images) if steps_context_images else 0

    current_obs_section = ""
    if current_observation:
        current_perception = (
            _run_perception_on_observation(perception_code, current_observation)
            if perception_code
            else ""
        )
        current_state_img_tag = ""
        if current_image is not None:
            current_state_img_tag = f" (image {num_steps_images + 1})"
        raw_state_content = "(see attached image)" if (hide_raw_obs and current_image is not None) else current_observation
        current_obs_section = f"""
=== CURRENT STATE ===
<raw_state>{current_state_img_tag}
{raw_state_content}
</raw_state>

<auxiliary_observation>
{current_aux_observation or ""}
</auxiliary_observation>

<perception_output>
{current_perception if current_perception else "(no perception module)"}
</perception_output>
=== END CURRENT STATE ===
"""

    qa_list_text = _format_qa_list(current_qa)

    prompt = f"""You are generating questions to guide an agent's exploration of an environment.

=== DEFAULT KNOWLEDGE ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== CURRENT BELIEFS ===
{beliefs if beliefs else "(empty - no beliefs yet)"}
=== END CURRENT BELIEFS ===

Each ``<raw_state>`` (and ``<resulting_state>``, when present) below is annotated with an ``(image K)`` marker referring to the K-th (1-indexed) screenshot attached to this message — use these to cross-reference the textual observation with the actual visual state.

=== RECENT HISTORY OF STATES AND ACTIONS ===
{step_history if step_history else "(no steps recorded yet)"}
=== END RECENT HISTORY ===

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
QUESTION 1: [A specific yes/no question about how the environment works]
...
(Generate questions)
</questions>"""

    images: list = []
    if steps_context_images:
        images.extend(steps_context_images)
    if current_image is not None:
        images.append(current_image)

    text, cost = await _llm_call(config, prompt, images=images or None)

    questions_text = extract_xml_key(text, "questions")
    new_questions: list[EBQAPair] = []
    # Build set of existing question texts for deduplication
    existing_questions = {q.question.strip().lower() for q in current_qa}
    if questions_text:
        for match in re.finditer(
            r"QUESTION\s+\d+:\s*(.+?)(?=QUESTION\s+\d+:|$)",
            questions_text,
            re.DOTALL,
        ):
            q = match.group(1).strip()
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

    current_obs_section = ""
    if current_observation:
        current_perception = (
            _run_perception_on_observation(perception_code, current_observation)
            if perception_code
            else ""
        )
        current_state_img_tag = ""
        if current_image is not None:
            current_state_img_tag = f" (image {num_steps_images + 1})"
        raw_state_content = "(see attached image)" if (hide_raw_obs and current_image is not None) else current_observation
        current_obs_section = f"""
=== CURRENT STATE (agent has not yet acted) ===
<raw_state>{current_state_img_tag}
{raw_state_content}
</raw_state>

<auxiliary_observation>
{current_aux_observation or ""}
</auxiliary_observation>

<perception_output>
{current_perception if current_perception else "(no perception module)"}
</perception_output>
=== END CURRENT STATE ===
"""

    qa_list_text = _format_qa_list(current_qa)
    current_exp_text = current_experiment if current_experiment else "(no experiment set yet)"

    prompt = f"""You are designing the next experiment for an agent interacting with an environment.

=== DEFAULT KNOWLEDGE ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== CURRENT BELIEFS ===
{beliefs if beliefs else "(empty - no beliefs yet)"}
=== END CURRENT BELIEFS ===

Each ``<raw_state>`` (and ``<resulting_state>``, when present) below is annotated with an ``(image K)`` marker referring to the K-th (1-indexed) screenshot attached to this message — use these to cross-reference the textual observation with the actual visual state.

=== RECENT HISTORY OF STATES AND ACTIONS ===
{step_history}
=== END RECENT HISTORY ==={current_obs_section}

=== CURRENT QUESTIONS ===
{qa_list_text}
=== END CURRENT QUESTIONS ===

=== CURRENT EXPERIMENT ===
{current_exp_text}
=== END CURRENT EXPERIMENT ===

Your task: Decide whether to keep the current experiment or formulate a new one based on an unanswered question.

If the current experiment is still being tested (the agent hasn't had enough steps to gather evidence), return null.
Otherwise, select an UNANSWERED question from the list above and design a specific, actionable experiment (1-3 sentences) that the agent can carry out to answer that question.

Format your response as:
<think>
Has the current experiment been sufficiently tested? If so, which unanswered question should we focus on next?
</think>
<experiment>
If keeping the current experiment:
null

If formulating a new experiment:
<question_index>[The Q number of the selected unanswered question]</question_index>
<experiment_plan>[1-3 sentence actionable experiment to answer the selected question]</experiment_plan>
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

    question_index_str = extract_xml_key(experiment_text_raw, "question_index")
    experiment_plan = extract_xml_key(experiment_text_raw, "experiment_plan")

    question_index = None
    if question_index_str:
        # Parse "Q3" or "3" style references
        idx_match = re.search(r"\d+", question_index_str)
        if idx_match:
            raw_idx = int(idx_match.group()) - 1  # Convert to 0-indexed
            # Validate: must be in range and point to an unanswered question
            if 0 <= raw_idx < len(current_qa) and current_qa[raw_idx].answer is None:
                question_index = raw_idx
            else:
                logging.warning(
                    f"Experiment question_index {raw_idx + 1} is out of range or not unanswered "
                    f"(total={len(current_qa)}), ignoring index"
                )

    if not experiment_plan:
        # Fallback: treat the whole experiment block as the plan
        experiment_plan = experiment_text_raw.strip()

    if len(experiment_plan) > 300:
        experiment_plan = experiment_plan[:300].rsplit(" ", 1)[0] + "..."

    logging.info(f"Formulated experiment from question Q{(question_index or 0) + 1}: {experiment_plan[:80]}...")
    return experiment_plan, question_index, cost, prompt, text


# ---------------------------------------------------------------------------
# Q&A update from trajectory
# ---------------------------------------------------------------------------


async def update_qa_from_trajectory(
    config: DictConfig,
    current_qa: list[EBQAPair],
    steps_context: str,
    max_total_qa_pairs: int,
    current_step: int = 0,
    steps_context_images: list | None = None,
    hide_raw_obs: bool = False,
) -> tuple[list[EBQAPair], float, dict]:
    """Update Q&A pairs from trajectory evidence.

    A single LLM call that:
    1. Answers unanswered questions if the trajectory provides evidence
    2. Corrects existing answers if trajectory contradicts them
    3. Adds new questions discovered from the trajectory

    Returns: (updated_qa_pairs, cost, extraction_log)
    """
    if not steps_context:
        return current_qa, 0.0, {}

    display_steps_context = steps_context
    if hide_raw_obs and steps_context_images:
        display_steps_context = _strip_raw_state_text(display_steps_context)

    qa_list_text = _format_qa_list(current_qa)

    prompt = f"""You are analyzing an agent's trajectory to update our knowledge base of questions and answers about the environment.

Each ``<raw_state>`` (and ``<resulting_state>``, when present) in the sequence below is annotated with an ``(image K)`` marker referring to the K-th (1-indexed) screenshot attached to this message — use these to cross-reference the textual observation with the actual visual state.

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
4. If any existing questions are redundant, no longer useful, or superseded by better questions, you may DROP them.

Only answer questions when the trajectory provides clear evidence. Do not guess or infer beyond what is directly observed.

Format your response as:
<think>
Review the trajectory and each question. What can we learn?
</think>
<updated_questions>
<q n="1">
<question>[question text]</question>
<evidence>[evidence from trajectory, or empty if unanswered]</evidence>
<answer>YES or NO or UNANSWERED</answer>
</q>
<q n="2">
...
</q>
...
(Include all existing questions plus any new ones)
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

    # Parse each <q> entry
    updated_qa: list[EBQAPair] = []
    for q_match in re.finditer(
        r'<q\s+n="(\d+)">(.*?)</q>',
        updated_questions_text,
        re.DOTALL,
    ):
        q_content = q_match.group(2)
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

        # Try to find matching source_step from existing Q; use current_step for new questions
        source_step = current_step
        for existing in current_qa:
            if existing.question.strip().lower() == question.strip().lower():
                source_step = existing.source_step
                break

        updated_qa.append(EBQAPair(
            question=question.strip(),
            answer=answer,
            evidence=evidence.strip(),
            source_step=source_step,
        ))

    if not updated_qa:
        extraction_log["parse_error"] = "No valid <q> entries parsed"
        return current_qa, cost, extraction_log

    prev_unanswered = sum(1 for q in current_qa if q.answer is None)
    new_unanswered = sum(1 for q in updated_qa if q.answer is None)
    extraction_log["prev_count"] = len(current_qa)
    extraction_log["new_count"] = len(updated_qa)
    extraction_log["prev_unanswered"] = prev_unanswered
    extraction_log["new_unanswered"] = new_unanswered
    extraction_log["newly_answered"] = prev_unanswered - new_unanswered

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
    max_total_qa_pairs: int,
    current_step: int = 0,
) -> tuple[list[EBQAPair], float, dict]:
    """Trim the Q&A list down to at most *max_total_qa_pairs* entries.

    Asks the LLM to decide which questions to keep based on usefulness.
    Should only be called when ``len(current_qa) > max_total_qa_pairs``.

    Returns: (trimmed_qa_pairs, cost, trim_log)
    """
    qa_list_text = _format_qa_list(current_qa)

    prompt = f"""You are maintaining a knowledge base of questions and answers about an environment.

The knowledge base currently has {len(current_qa)} questions, but we need to trim it to at most {max_total_qa_pairs}.

=== CURRENT QUESTIONS ===
{qa_list_text}
=== END CURRENT QUESTIONS ===

Your task: Select the {max_total_qa_pairs} most useful questions to keep. Drop questions that are:
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
<q n="1">
<question>[question text]</question>
<answer>YES or NO or UNANSWERED</answer>
<evidence>[evidence, or empty if unanswered]</evidence>
</q>
<q n="2">
...
</q>
...
(Include at most {max_total_qa_pairs} questions)
</trimmed_questions>"""

    text, cost = await _llm_call(config, prompt)

    trim_log: dict = {
        "prompt": prompt,
        "response": text,
        "pre_trim_count": len(current_qa),
        "max_total_qa_pairs": max_total_qa_pairs,
    }

    trimmed_text = extract_xml_key(text, "trimmed_questions")
    if not trimmed_text:
        trim_log["parse_error"] = "No <trimmed_questions> block found"
        return current_qa, cost, trim_log

    trimmed_qa: list[EBQAPair] = []
    for q_match in re.finditer(
        r'<q\s+n="(\d+)">(.*?)</q>',
        trimmed_text,
        re.DOTALL,
    ):
        q_content = q_match.group(2)
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
    trim_log["dropped_count"] = len(current_qa) - len(trimmed_qa)

    logging.info(
        f"Q&A trim: {len(current_qa)} -> {len(trimmed_qa)} questions "
        f"(dropped {len(current_qa) - len(trimmed_qa)})"
    )

    return trimmed_qa, cost, trim_log
