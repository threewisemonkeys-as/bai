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

from goal_prompts import is_goal_aware
from llm_utils import extract_xml_key
from mixed_improve import (
    QAPair,
    _llm_call,
    _run_perception_on_observation,
)
from stepwise_b_learn import format_current_state

# ---------------------------------------------------------------------------
# Shared provenance note
# ---------------------------------------------------------------------------

# Injected into any prompt that presents the agent's trajectory for analysis, so
# the model does not mistake the agent's own chain-of-thought for environment
# ground truth. This is the guard against "speculation laundering" — the agent
# stating an assumed goal/win-condition in <agent_reasoning>, which then gets
# harvested as if it were an observation.
TRAJECTORY_REASONING_NOTE = (
    "PROVENANCE: Within each step, `<pre_state>`, `<post_state>`, and "
    "`<auxiliary_observation>` are the actual environment output — ground truth. "
    "`<agent_reasoning>` is the agent's own chain-of-thought from that step: its "
    "plans, hypotheses, and assumed goals at the time. It is NOT an observation "
    "and may be wrong. Never treat a claim made inside `<agent_reasoning>` (for "
    "example a stated objective or win condition) as evidence about how the game "
    "works; evidence comes only from observed state changes (`<pre_state>` -> "
    "`<post_state>`) and `<auxiliary_observation>` values such as the score/level "
    "counter."
)


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------


@dataclass
class EBQAPair:
    """A question-answer pair where the answer can be None (unanswered)."""

    question: str
    answer: bool | None  # None = unanswered, True = YES, False = NO
    evidence: str  # empty string if unanswered
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


def _strip_raw_pre_state_text(steps_context: str) -> str:
    """Replace <pre_state> text content with a placeholder.

    Image annotations on the opening tag (e.g. ``<pre_state> (image 3)``) are
    preserved when present, but this is also used for text-only prompts where
    no screenshot is attached.
    """
    return re.sub(
        r"(<pre_state[^>]*>[^\n]*)\n.*?(\n</pre_state>)",
        r"\1\n(raw pre-state observation hidden)\2",
        steps_context,
        flags=re.DOTALL,
    )


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
    for match in re.finditer(
        r"(?<![A-Za-z_])(?:Q\s*)?(\d+)(?![A-Za-z_])",
        text or "",
        re.IGNORECASE,
    ):
        idx = int(match.group(1)) - 1
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
    match = re.fullmatch(r"\s*(?:Q\s*)?(\d+)\s*", raw, re.IGNORECASE)
    if not match:
        return None
    idx = int(match.group(1)) - 1
    if 0 <= idx < max_index:
        return idx
    return None


def _normalize_candidate_index(raw: str | None, max_index: int) -> int | None:
    """Parse a 1-based candidate experiment reference like E3 or 3 into a 0-based index."""
    if raw is None:
        return None
    match = re.fullmatch(r"\s*(?:E\s*)?(\d+)\s*", raw, re.IGNORECASE)
    if not match:
        return None
    idx = int(match.group(1)) - 1
    if 0 <= idx < max_index:
        return idx
    return None


def _iter_q_blocks_with_refs(
    text: str,
    max_index: int,
) -> list[tuple[int | None, str | None, str]]:
    """Return <q n="Q1">...</q> blocks with parsed and raw references."""
    blocks: list[tuple[int | None, str | None, str]] = []
    for match in re.finditer(
        r"<q\b(?P<attrs>[^>]*)>(?P<body>.*?)</q>",
        text or "",
        re.DOTALL | re.IGNORECASE,
    ):
        attrs = match.group("attrs")
        raw_ref = _extract_attr(attrs, "n")
        idx = _normalize_question_index(raw_ref, max_index)
        blocks.append((idx, raw_ref, match.group("body")))
    return blocks


def _iter_q_blocks(text: str, max_index: int) -> list[tuple[int | None, str]]:
    """Return <q n="Q1">...</q> blocks, accepting n="1" as well."""
    return [(idx, body) for idx, _, body in _iter_q_blocks_with_refs(text, max_index)]


def _parse_q_tag_indices(text: str, max_index: int) -> list[int]:
    """Parse indices from <q n="Q1" /> or legacy <q source_index="1" /> tags."""
    indices: list[int] = []
    seen: set[int] = set()
    for match in re.finditer(r"<q\b(?P<attrs>[^>]*)/?>", text or "", re.IGNORECASE):
        attrs = match.group("attrs")
        idx = _normalize_question_index(_extract_attr(attrs, "n"), max_index)
        if idx is None:
            idx = _normalize_question_index(
                _extract_attr(attrs, "source_index"),
                max_index,
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
    if hide_raw_obs and include_recent_history:
        step_history = _strip_raw_pre_state_text(step_history)
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

{TRAJECTORY_REASONING_NOTE}

=== RECENT HISTORY OF STATES AND ACTIONS ===
{step_history if step_history else "(no steps recorded yet)"}
=== END RECENT HISTORY ===
"""

    prompt = f"""You are playing a game with the goal of understanding how it works. Your task is to generate questions about how the game works based on what you know already and what you have observed so far. These questions will guide your exploration of the game.

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
- Do not create questions which would be mostly answered if any of the current questions were answered.
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
                new_questions.append(
                    EBQAPair(
                        question=q,
                        answer=None,
                        evidence="",
                        source_step=current_step,
                    )
                )

    logging.info(f"Generated {len(new_questions)} new questions from step history")
    return new_questions, cost, prompt, text


# ---------------------------------------------------------------------------
# Beliefs-only question generation (oracle pipeline)
# ---------------------------------------------------------------------------


async def generate_questions_from_beliefs(
    config: DictConfig,
    beliefs: str,
    current_qa: list[EBQAPair],
    default_knowledge: str,
    num_questions: int,
    current_step: int = 0,
) -> tuple[list[EBQAPair], float, str, str]:
    """Generate new yes/no questions about the environment from beliefs alone.

    No observations, perception, or step history — used by the oracle-driven
    selection pipeline (stepwise_eb_learn_oracle.py) where there is no env.

    Returns: (new_questions, cost, prompt, raw_response)
    """
    qa_list_text = _format_qa_list(current_qa)
    prompt = f"""You are learning how an environment works by generating questions whose answers (provided later by an oracle) will update your beliefs about the game. You have no observations of the game itself — only your current beliefs and the standing question bank.

=== DEFAULT KNOWLEDGE ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== CURRENT BELIEFS ===
{beliefs if beliefs else "(empty - no beliefs yet)"}
=== END CURRENT BELIEFS ===

=== CURRENT QUESTIONS ===
{qa_list_text}
=== END CURRENT QUESTIONS ===

Your task: Generate {num_questions} new binary (yes/no) questions about how the environment works.

Guidelines:
- Questions should be general in scope, asking about how the world works.
- Do not duplicate or near-duplicate existing questions in the bank.
- Focus on areas your current beliefs are silent on, vague about, or possibly wrong about.
- Each question must be a specific yes/no question, not open-ended.
- Each question should be answerable by an oracle that has read the underlying game program.

Format your response as:
<think>
Where are the biggest gaps or uncertainties in the current beliefs? What yes/no questions would close them?
</think>
<questions>
<q n="Q1">
<question>[A specific yes/no question about how the environment works]</question>
</q>
<q n="Q2">
<question>[Another specific yes/no question about how the environment works]</question>
</q>
...
</questions>"""

    text, cost = await _llm_call(config, prompt)

    questions_text = extract_xml_key(text, "questions")
    new_questions: list[EBQAPair] = []
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
            if not q:
                continue
            if len(q) > 300:
                q = q[:300].rsplit(" ", 1)[0] + "..."
            q_lower = q.strip().lower()
            if q_lower in existing_questions:
                continue
            existing_questions.add(q_lower)
            new_questions.append(
                EBQAPair(
                    question=q,
                    answer=None,
                    evidence="",
                    source_step=current_step,
                )
            )

    logging.info(f"Generated {len(new_questions)} new questions from beliefs")
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
    current_experiment_question: str | None,
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
    if hide_raw_obs:
        step_history = _strip_raw_pre_state_text(step_history)
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

    active_question_status_text = "(no active question)"
    active_question_index: int | None = None
    if current_experiment_question:
        active_question_status_text = "not found in current questions"
        active_question_key = current_experiment_question.strip().lower()
        for i, qa in enumerate(current_qa):
            if qa.question.strip().lower() != active_question_key:
                continue
            active_question_index = i
            if qa.answer is None:
                active_question_status_text = "UNANSWERED"
            else:
                answer_text = "YES" if qa.answer else "NO"
                evidence = _strip_raw_grid_text(qa.evidence) if qa.evidence else ""
                active_question_status_text = f"{answer_text}" + (
                    f" (evidence: {evidence})" if evidence else ""
                )
            break

    if current_experiment:
        current_exp_question_text = (
            f"Q{active_question_index + 1}: {current_experiment_question}"
            if active_question_index is not None and current_experiment_question
            else current_experiment_question
            if current_experiment_question
            else "(question not recorded)"
        )
        current_exp_text = f"""Active question:
{current_exp_question_text}

Active question status in current Q&A:
{active_question_status_text}

Active experiment plan:
{current_experiment}"""
    else:
        current_exp_text = "(no active question or experiment)"

    valid_target_question_index = (
        target_question_index is not None
        and 0 <= target_question_index < len(current_qa)
        and current_qa[target_question_index].answer is None
    )
    unanswered_question_lines = [
        f"Q{i + 1}: {qa.question}"
        for i, qa in enumerate(current_qa)
        if qa.answer is None
    ]
    available_questions_text = (
        "\n".join(unanswered_question_lines) if unanswered_question_lines else "(none)"
    )

    if valid_target_question_index:
        target_question_text = (
            f"Q{target_question_index + 1}: "
            f"{current_qa[target_question_index].question}"
        )
        target_non_null_format = ""
        if (
            active_question_index is not None
            and current_qa[active_question_index].answer is None
            and active_question_index != target_question_index
        ):
            target_non_null_format += f"""If revising the active question's experiment:
<q n="Q{active_question_index + 1}">
<experiment_plan>[1-3 sentence revised experiment to answer the active question]</experiment_plan>
</q>

"""
        target_non_null_format += f"""If formulating a target-question experiment:
<q n="Q{target_question_index + 1}">
<experiment_plan>[1-3 sentence actionable experiment to answer the TARGET QUESTION]</experiment_plan>
</q>"""
        prompt = f"""You are designing the next experiment for an agent interacting with an environment. The experiment must answer either the ongoing active question or the fixed target question.

=== DEFAULT KNOWLEDGE ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== CURRENT BELIEFS ===
{beliefs if beliefs else "(empty - no beliefs yet)"}
=== END CURRENT BELIEFS ===

Each ``<pre_state>`` (and ``<post_state>``, when present) below is annotated with an ``(image K)`` marker referring to the K-th (1-indexed) screenshot attached to this message — use these to cross-reference the textual observation with the actual visual state. ``<pre_state>`` is the observation before the step's action; ``<post_state>`` is the final observation of a past episode segment.

{TRAJECTORY_REASONING_NOTE}

=== RECENT HISTORY OF STATES AND ACTIONS ===
{step_history}
=== END RECENT HISTORY ==={current_obs_section}

=== ACTIVE QUESTION AND EXPERIMENT ===
{current_exp_text}
=== END ACTIVE QUESTION AND EXPERIMENT ===

=== TARGET QUESTION ===
{target_question_text}
=== END TARGET QUESTION ===

Your task:
1. Decide whether the active question is already answered.
2. If there is an active experiment and the active question is not answered yet:
   - Return null if the current experiment plan is still the right next experiment.
   - Formulate a revised experiment for the active question if the same question should keep being investigated but the current plan should change.
3. If the active question is answered, or there is no active question, formulate a new experiment for the TARGET QUESTION.
4. Any experiment plan must be specific, actionable, 1-3 sentences, and directly aimed at collecting evidence for the question named in the <q> tag.

Format your response as:
<think>
Is the active question answered? Should the current experiment be kept, revised for the same active question, or replaced with an experiment for the target question?
</think>
<experiment>
If keeping the current experiment unchanged:
null

{target_non_null_format}
</experiment>"""
    else:
        prompt = f"""You are selecting the next question and designing the associated experiment for learning more about the game being played.

=== DEFAULT KNOWLEDGE ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== CURRENT BELIEFS ===
{beliefs if beliefs else "(empty - no beliefs yet)"}
=== END CURRENT BELIEFS ===

Each ``<pre_state>`` (and ``<post_state>``, when present) below is annotated with an ``(image K)`` marker referring to the K-th (1-indexed) screenshot attached to this message — use these to cross-reference the textual observation with the actual visual state. ``<pre_state>`` is the observation before the step's action; ``<post_state>`` is the final observation of a past episode segment.

{TRAJECTORY_REASONING_NOTE}

=== RECENT HISTORY OF STATES AND ACTIONS ===
{step_history}
=== END RECENT HISTORY ==={current_obs_section}

=== ACTIVE QUESTION AND EXPERIMENT ===
{current_exp_text}
=== END ACTIVE QUESTION AND EXPERIMENT ===

=== AVAILABLE UNANSWERED QUESTIONS ===
{available_questions_text}
=== END AVAILABLE UNANSWERED QUESTIONS ===

Your task:
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

    if question_index is None and valid_target_question_index:
        question_index = target_question_index

    if question_index is None:
        logging.warning(
            "Experiment response did not identify a valid unanswered selected question; "
            "keeping current experiment"
        )
        return None, None, cost, prompt, text

    if experiment_plan is None:
        experiment_plan = extract_xml_key(experiment_text_raw, "experiment_plan")
    if not experiment_plan:
        # Fallback: treat the whole experiment block as the plan
        experiment_plan = experiment_text_raw.strip()

    logging.info(
        f"Formulated experiment from question Q{(question_index or 0) + 1}: {experiment_plan[:80]}..."
    )
    return experiment_plan, question_index, cost, prompt, text


async def formulate_experiment_for_question(
    config: DictConfig,
    beliefs: str,
    perception_code: str,
    steps_context: str,
    target_question: str,
    current_observation: str | None,
    current_aux_observation: str | None,
    default_knowledge: str,
    current_image=None,
    steps_context_images: list | None = None,
    hide_raw_obs: bool = False,
) -> tuple[str, float, str, str]:
    """Formulate an experiment plan to answer a specific, pre-selected question.

    Unlike :func:`formulate_experiment_from_question`, there is no null/"keep"
    option and no question-selection logic: this is called only after question
    selection has already chosen a NEW target question, so the experiment is
    always formulated for that exact question. Whether to formulate at all (vs.
    keep the current experiment) is decided by the caller based on whether the
    selected question differs from the active one.

    Returns: ``(experiment_plan, cost, prompt, raw_response)``.
    """
    step_history = steps_context
    if hide_raw_obs:
        step_history = _strip_raw_pre_state_text(step_history)
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

    prompt = f"""You are designing the next experiment for an agent interacting with an environment. Question selection has already chosen the question below as the most informative one to investigate next. Your only job is to formulate a concrete experiment that will collect evidence to answer it.

=== DEFAULT KNOWLEDGE ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== CURRENT BELIEFS ===
{beliefs if beliefs else "(empty - no beliefs yet)"}
=== END CURRENT BELIEFS ===

Each ``<pre_state>`` (and ``<post_state>``, when present) below is annotated with an ``(image K)`` marker referring to the K-th (1-indexed) screenshot attached to this message — use these to cross-reference the textual observation with the actual visual state. ``<pre_state>`` is the observation before the step's action; ``<post_state>`` is the final observation of a past episode segment.

{TRAJECTORY_REASONING_NOTE}

=== RECENT HISTORY OF STATES AND ACTIONS ===
{step_history}
=== END RECENT HISTORY ==={current_obs_section}

=== TARGET QUESTION ===
{target_question}
=== END TARGET QUESTION ===

Formulate a specific, actionable experiment (1-3 sentences) that, when executed by the agent, will produce direct evidence to answer the TARGET QUESTION.

Format your response as:
<think>
What concrete sequence of actions / observations would yield evidence for the target question?
</think>
<experiment_plan>[1-3 sentence actionable experiment to answer the target question]</experiment_plan>"""

    images: list = []
    if steps_context_images:
        images.extend(steps_context_images)
    if current_image is not None:
        images.append(current_image)

    text, cost = await _llm_call(config, prompt, images=images or None)

    experiment_plan = extract_xml_key(text, "experiment_plan")
    if not experiment_plan:
        # Fallbacks: an <experiment> wrapper, else the raw response text.
        experiment_plan = extract_xml_key(text, "experiment") or text.strip()
    experiment_plan = (experiment_plan or "").strip()
    logging.info(
        f"Formulated experiment for selected question: {experiment_plan[:80]}..."
    )
    return experiment_plan, cost, prompt, text


# ---------------------------------------------------------------------------
# Experiment scoring against unanswered questions (score_topk selection mode)
# ---------------------------------------------------------------------------


async def score_experiments_against_questions(
    config: DictConfig,
    candidates: list[dict],
    unanswered_qa: list[EBQAPair],
    unanswered_source_indices: list[int],
    beliefs: str,
    default_knowledge: str,
) -> tuple[list[int], list[dict[int, bool]], float, str, str, bool]:
    """Score every candidate experiment against the unanswered bank in one LLM call.

    For each (candidate, unanswered question) pair, the LLM marks YES if the
    experiment would plausibly produce direct evidence for that question.
    The score for a candidate is the number of YES marks across questions.

    Questions are labeled by their bank position (``Q{source_index + 1}``);
    candidates are labeled by 1-based position (``E{i + 1}``) in
    ``candidates``. Each candidate dict must expose ``"plan"`` and may carry
    ``"question"`` (the associated question text).

    Returns: ``(scores, per_question_yesno_list, cost, prompt, raw_response, parsed_ok)``.
    ``per_question_yesno_list[i]`` is keyed by source index into ``qa_pairs``;
    pairs the LLM does not mark default to NO. ``parsed_ok`` is True when at
    least one candidate index was successfully extracted from the response;
    callers can use it to fall back to a safe default when scoring is unusable.
    """
    if not candidates:
        return [], [], 0.0, "", "", True
    if not unanswered_qa:
        return (
            [0] * len(candidates),
            [{} for _ in candidates],
            0.0,
            "",
            "",
            True,
        )

    qa_list_text = "\n".join(
        f"Q{src_idx + 1}: {qa.question}"
        for qa, src_idx in zip(unanswered_qa, unanswered_source_indices)
    )

    experiment_blocks: list[str] = []
    for i, cand in enumerate(candidates):
        question_line = cand.get("question") or "(no associated question)"
        plan_line = cand.get("plan") or "(no plan)"
        experiment_blocks.append(
            f"E{i + 1}:\n  Associated question: {question_line}\n  Plan: {plan_line}"
        )
    experiments_text = "\n\n".join(experiment_blocks)

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

    prompt = f"""You are scoring candidate experiments by judging which open questions each one could answer.
{default_knowledge_section}
{beliefs_section}
=== CANDIDATE EXPERIMENTS ===
{experiments_text}
=== END CANDIDATE EXPERIMENTS ===

=== UNANSWERED QUESTIONS ===
{qa_list_text}
=== END UNANSWERED QUESTIONS ===

For EACH candidate experiment and EACH unanswered question, decide whether running the experiment is likely to produce direct evidence that would answer the question (YES or NO). Mark YES only when the experiment would plausibly produce direct evidence for that question; mark NO if the experiment is unrelated or only tangentially related.

Format your response as:
<think>
For each experiment, what specific evidence would it produce, and which questions would that evidence answer?
</think>
<scores>
<e n="E1">
<q n="Q1"><answer>YES or NO</answer></q>
<q n="Q2"><answer>YES or NO</answer></q>
...
</e>
<e n="E2">
<q n="Q1"><answer>YES or NO</answer></q>
...
</e>
...
</scores>"""

    text, cost = await _llm_call(config, prompt)

    per_candidate_yesno: list[dict[int, bool]] = [
        {src_idx: False for src_idx in unanswered_source_indices} for _ in candidates
    ]
    unanswered_set = set(unanswered_source_indices)
    scores_text = extract_xml_key(text, "scores") or ""
    parsed_any_candidate = False
    if scores_text:
        max_q = max(unanswered_source_indices) + 1 if unanswered_source_indices else 0
        for e_match in re.finditer(
            r"<e\b(?P<attrs>[^>]*)>(?P<body>.*?)</e>",
            scores_text,
            re.DOTALL | re.IGNORECASE,
        ):
            attrs = e_match.group("attrs")
            e_body = e_match.group("body")
            cand_idx = _normalize_candidate_index(
                _extract_attr(attrs, "n"),
                len(candidates),
            )
            if cand_idx is None:
                continue
            parsed_any_candidate = True
            for src_idx, body in _iter_q_blocks(e_body, max_q):
                if src_idx is None or src_idx not in unanswered_set:
                    continue
                answer_text = extract_xml_key(body, "answer") or body
                is_yes = bool(answer_text) and answer_text.strip().upper().startswith(
                    "YES"
                )
                per_candidate_yesno[cand_idx][src_idx] = is_yes

    scores = [sum(1 for v in yesno.values() if v) for yesno in per_candidate_yesno]
    logging.info(
        f"Scored {len(candidates)} experiments against "
        f"{len(unanswered_qa)} unanswered questions: scores={scores} "
        f"parsed_ok={parsed_any_candidate}"
    )
    return scores, per_candidate_yesno, cost, prompt, text, parsed_any_candidate


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
    if hide_raw_obs:
        display_steps_context = _strip_raw_pre_state_text(display_steps_context)

    qa_list_text = _format_qa_list(current_qa)

    prompt = f"""You are analyzing a gameplay trajectory to update our knowledge base of questions and answers about the game being played.

Each ``<pre_state>`` (and ``<post_state>``, when present) in the sequence below is annotated with an ``(image K)`` marker referring to the K-th (1-indexed) screenshot attached to this message — use these to cross-reference the textual observation with the actual visual state. ``<pre_state>`` is the observation before the step's action; ``<post_state>`` is the observation after the step's action (shown for the final step of each episode segment).

{TRAJECTORY_REASONING_NOTE}

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

Only answer questions when the trajectory provides clear evidence. Do not guess or infer beyond what is directly observed. Evidence must come from observed state — `<pre_state>`/`<post_state>` changes and `<auxiliary_observation>` values — never from `<agent_reasoning>`. In particular, do NOT mark a question about the goal or win condition as answered just because the agent's reasoning asserted a goal: a win condition is only confirmed by an observed change in the score/level counter (e.g. the "Levels completed" value increasing). If the only support for a claim is the agent's own reasoning, leave the question UNANSWERED.

When writing <evidence>, state the concrete observed outcome first — the raw before→after change and any counter/score/level values exactly as seen — and then the conclusion it directly supports. Do not hedge with speculation ("perhaps", "or maybe", "this might mean"): if the observation is ambiguous or only partially supports a conclusion, leave the question UNANSWERED rather than recording a vague or softened conclusion. Record the raw anomaly even when you cannot fully explain it (e.g. "all cells turned red but the level counter stayed 0/6").

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
        config,
        prompt,
        images=steps_context_images or None,
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

        trimmed_qa.append(
            EBQAPair(
                question=question.strip(),
                answer=answer,
                evidence=evidence.strip(),
                source_step=source_step,
            )
        )

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
    """Merge questions about the same dynamic into replacement questions.

    This intentionally does not trim for usefulness. It only asks which groups
    of questions cover the same underlying environment dynamic, then replaces
    each group with one synthesized binary question about that dynamic.
    """
    qa_list_text = _format_qa_list(current_qa)
    prompt = f"""You are maintaining a knowledge base of binary questions about an environment.

=== CURRENT QUESTIONS ===
{qa_list_text}
=== END CURRENT QUESTIONS ===

Your task: identify groups of questions that ask about the same underlying environment dynamic, then write one replacement question for each group.

Two questions belong in the same group when answering one would mostly answer the others because they are probing the same rule or mechanic.

Do not group questions merely because they are both low priority, old, currently unanswered, or mention the same object while asking about genuinely different dynamics.

For each group:
- Include two or more original Q numbers.
- Create a consice single specific yes/no replacement question.
- The replacement should cover the shared underlying dynamic without becoming vague or open-ended.
- Do not include questions that are already distinct enough to remain separate.

Format your response as:
<think>
Which questions ask about the same underlying dynamics, and what replacement question best covers each group?
</think>
<dynamic_replacement_groups>
<group>
<members>
<q n="Q2" />
<q n="Q5" />
</members>
<replacement_question>[A specific yes/no question that replaces this group]</replacement_question>
</group>
...
(Use the original Q numbers from CURRENT QUESTIONS. Write NONE if no groups should be replaced.)
</dynamic_replacement_groups>"""

    text, cost = await _llm_call(config, prompt)
    groups_text = extract_xml_key(text, "dynamic_replacement_groups")
    dedup_log: dict = {
        "method": "dynamic_replacement_groups",
        "prompt": prompt,
        "response": text,
        "pre_dedup_count": len(current_qa),
        "pre_dedup_answered": sum(1 for q in current_qa if q.answer is not None),
        "pre_dedup_unanswered": sum(1 for q in current_qa if q.answer is None),
    }

    if groups_text is None:
        dedup_log["parse_error"] = "No <dynamic_replacement_groups> block found"
        dedup_log["post_dedup_count"] = len(current_qa)
        dedup_log["replacement_groups"] = []
        dedup_log["replaced_indices"] = []
        dedup_log["replaced_questions"] = []
        dedup_log["dropped_count"] = 0
        dedup_log["dropped_indices"] = []
        dedup_log["dropped_questions"] = []
        return current_qa, cost, dedup_log

    replacement_by_first_idx: dict[int, EBQAPair] = {}
    grouped_indices: set[int] = set()
    replacement_groups: list[dict] = []
    skipped_groups: list[dict] = []

    for group_match in re.finditer(
        r"<group\b[^>]*>(?P<body>.*?)</group>",
        groups_text or "",
        re.DOTALL | re.IGNORECASE,
    ):
        group_content = group_match.group("body")
        replacement_question = extract_xml_key(
            group_content,
            "replacement_question",
        )
        member_indices = _parse_q_tag_indices(group_content, len(current_qa))
        member_indices = sorted(set(member_indices))

        if len(member_indices) < 2:
            skipped_groups.append(
                {
                    "reason": "fewer than two valid members",
                    "member_indices": member_indices,
                    "replacement_question": replacement_question,
                }
            )
            continue
        if not replacement_question or not replacement_question.strip():
            skipped_groups.append(
                {
                    "reason": "missing replacement question",
                    "member_indices": member_indices,
                }
            )
            continue
        if any(idx in grouped_indices for idx in member_indices):
            skipped_groups.append(
                {
                    "reason": "overlaps earlier replacement group",
                    "member_indices": member_indices,
                    "replacement_question": replacement_question,
                }
            )
            continue

        replacement_question = replacement_question.strip()
        replacement_key = replacement_question.lower()
        exact_existing = next(
            (
                current_qa[idx]
                for idx in member_indices
                if current_qa[idx].question.strip().lower() == replacement_key
            ),
            None,
        )
        if exact_existing is not None:
            replacement_qa = EBQAPair(
                question=exact_existing.question,
                answer=exact_existing.answer,
                evidence=exact_existing.evidence,
                source_step=exact_existing.source_step,
            )
            answer_preserved = exact_existing.answer is not None
        else:
            replacement_qa = EBQAPair(
                question=replacement_question,
                answer=None,
                evidence="",
                source_step=min(current_qa[idx].source_step for idx in member_indices),
            )
            answer_preserved = False

        first_idx = min(member_indices)
        replacement_by_first_idx[first_idx] = replacement_qa
        grouped_indices.update(member_indices)
        replacement_groups.append(
            {
                "member_indices": member_indices,
                "member_questions": [
                    current_qa[idx].question for idx in member_indices
                ],
                "replacement_question": replacement_qa.question,
                "answer_preserved": answer_preserved,
            }
        )

    if not replacement_groups:
        dedup_log["post_dedup_count"] = len(current_qa)
        dedup_log["post_dedup_answered"] = sum(
            1 for q in current_qa if q.answer is not None
        )
        dedup_log["post_dedup_unanswered"] = sum(
            1 for q in current_qa if q.answer is None
        )
        dedup_log["replacement_groups"] = []
        dedup_log["skipped_groups"] = skipped_groups
        dedup_log["replaced_indices"] = []
        dedup_log["replaced_questions"] = []
        dedup_log["dropped_count"] = 0
        dedup_log["dropped_indices"] = []
        dedup_log["dropped_questions"] = []
        return current_qa, cost, dedup_log

    deduped: list[EBQAPair] = []
    for idx, qa in enumerate(current_qa):
        if idx in replacement_by_first_idx:
            deduped.append(replacement_by_first_idx[idx])
        elif idx in grouped_indices:
            continue
        else:
            deduped.append(qa)

    dedup_log["post_dedup_count"] = len(deduped)
    dedup_log["post_dedup_answered"] = sum(1 for q in deduped if q.answer is not None)
    dedup_log["post_dedup_unanswered"] = sum(1 for q in deduped if q.answer is None)
    dedup_log["replacement_group_count"] = len(replacement_groups)
    dedup_log["replaced_question_count"] = len(grouped_indices)
    dedup_log["net_reduced_count"] = len(current_qa) - len(deduped)
    dedup_log["dropped_count"] = len(current_qa) - len(deduped)
    dedup_log["replacement_groups"] = replacement_groups
    dedup_log["skipped_groups"] = skipped_groups
    dedup_log["replaced_indices"] = sorted(grouped_indices)
    dedup_log["replaced_questions"] = [
        current_qa[idx].question for idx in sorted(grouped_indices)
    ]
    retained_replacement_indices = set(replacement_by_first_idx)
    dropped_indices = sorted(grouped_indices - retained_replacement_indices)
    dedup_log["dropped_indices"] = dropped_indices
    dedup_log["dropped_questions"] = [
        current_qa[idx].question for idx in dropped_indices
    ]

    logging.info(
        f"Q&A dedup: {len(current_qa)} -> {len(deduped)} questions "
        f"(replaced {len(grouped_indices)} questions with "
        f"{len(replacement_groups)} dynamic-level questions)"
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
        selection_log.update(
            {
                "note": "no unanswered questions available for experiment selection",
                "selected_source_indices": [],
                "post_selection_count": 0,
                "post_selection_answered": 0,
                "post_selection_unanswered": 0,
            }
        )
        return [], [], 0.0, selection_log

    if num_unanswered <= max_unanswered_qa_pairs:
        indices = unanswered_source_indices
        selection_log.update(
            {
                "note": "unanswered question bank within selection cap; selected all unanswered questions",
                "selected_source_indices": indices,
                "post_selection_count": len(indices),
                "post_selection_answered": 0,
                "post_selection_unanswered": len(indices),
            }
        )
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
    selection_criterion = (
        "most useful at achieving the environment objective"
        if is_goal_aware(config)
        else "most useful for understanding how the environment works"
    )
    prompt = f"""You are selecting the next question target for learning about how a game works.

{default_knowledge_section}

{beliefs_section}

=== AVAILABLE QUESTIONS ===
{qa_list_text}
=== END AVAILABLE QUESTIONS ===

Select up to {max_unanswered_qa_pairs} questions that will be {selection_criterion} while covering distinct aspects of the environment.

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
            idx
            for idx in _parse_q_tag_indices(selected_text, len(current_qa))
            if idx in unanswered_source_set
        ]
        if not selected_indices:
            selected_indices = [
                i
                for i in _parse_1_based_indices(selected_text, len(current_qa))
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


async def select_qa_pairs_and_formulate_experiments(
    config: DictConfig,
    current_qa: list[EBQAPair],
    max_unanswered_qa_pairs: int,
    beliefs: str,
    perception_code: str,
    steps_context: str,
    current_observation: str | None,
    current_aux_observation: str | None,
    default_knowledge: str,
    current_image=None,
    steps_context_images: list | None = None,
    hide_raw_obs: bool = False,
    filter_questions: bool = True,
) -> tuple[list[EBQAPair], list[int], list[dict], float, dict]:
    """Select top-k unanswered questions and formulate one experiment per question.

    This is the combined selection/formulation prompt used by ``score_topk``:
    instead of first selecting top-k and then making one LLM call per selected
    question, one LLM call returns both the selected Q numbers and their plans.

    When ``filter_questions`` is False, the LLM is asked to formulate a plan for
    every unanswered question instead of filtering down to a top-k subset; the
    ``max_unanswered_qa_pairs`` cap is ignored in that mode.

    Returns: (selected_qa, selected_source_indices, candidates, cost, log).
    Candidate ``source_index`` values are indices into ``current_qa``.
    """
    step_history = steps_context
    if hide_raw_obs:
        step_history = _strip_raw_pre_state_text(step_history)
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

    num_answered = sum(1 for q in current_qa if q.answer is not None)
    num_unanswered = sum(1 for q in current_qa if q.answer is None)
    unanswered_source_indices = [
        i for i, qa in enumerate(current_qa) if qa.answer is None
    ]
    selection_log: dict = {
        "method": (
            "llm_top_k_probe_selection_with_experiments"
            if filter_questions
            else "llm_formulate_all_unanswered_experiments"
        ),
        "filter_questions": filter_questions,
        "pre_selection_count": len(current_qa),
        "pre_selection_answered": num_answered,
        "pre_selection_unanswered": num_unanswered,
        "max_unanswered_qa_pairs": max_unanswered_qa_pairs,
        "default_knowledge_length": len(default_knowledge),
        "beliefs_length": len(beliefs),
        "candidate_source_indices": unanswered_source_indices,
    }

    if not unanswered_source_indices:
        selection_log.update(
            {
                "note": "no unanswered questions available for experiment selection",
                "selected_source_indices": [],
                "post_selection_count": 0,
                "post_selection_answered": 0,
                "post_selection_unanswered": 0,
                "candidates": [],
            }
        )
        return [], [], [], 0.0, selection_log

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

    selection_criterion = (
        "most useful at achieving the environment objective"
        if is_goal_aware(config)
        else "most useful for understanding how the environment works"
    )
    if filter_questions:
        task_instruction = (
            f"Select up to {max_unanswered_qa_pairs} questions that will be "
            f"{selection_criterion} while covering "
            "distinct aspects of the environment.\n\n"
            "For each selected question, also formulate a 1-3 sentence actionable "
            "experiment plan that, when executed by the agent from the current "
            "state, will collect direct evidence to answer that exact question. "
            "The plan must be specific and concrete — name the actions or state "
            "changes the agent should attempt — not a vague intent."
        )
        think_hint = (
            "Which questions should be selected, and what direct evidence would "
            "resolve each one?"
        )
    else:
        task_instruction = (
            "For every question listed in AVAILABLE UNANSWERED QUESTIONS, "
            "formulate a 1-3 sentence actionable experiment plan that, when "
            "executed by the agent from the current state, will collect direct "
            "evidence to answer that exact question. Do not omit any listed "
            "question. The plan must be specific and concrete — name the "
            "actions or state changes the agent should attempt — not a vague "
            "intent."
        )
        think_hint = (
            "What direct evidence would resolve each unanswered question?"
        )

    prompt = f"""You are designing experiments for an agent learning an environment.

{default_knowledge_section}

{beliefs_section}

Each ``<pre_state>`` (and ``<post_state>``, when present) below is annotated with an ``(image K)`` marker referring to the K-th (1-indexed) screenshot attached to this message — use these to cross-reference the textual observation with the actual visual state. ``<pre_state>`` is the observation before the step's action; ``<post_state>`` is the final observation of a past episode segment.

{TRAJECTORY_REASONING_NOTE}

=== RECENT HISTORY OF STATES AND ACTIONS ===
{step_history}
=== END RECENT HISTORY ==={current_obs_section}

=== AVAILABLE UNANSWERED QUESTIONS ===
{qa_list_text}
=== END AVAILABLE UNANSWERED QUESTIONS ===

{task_instruction}

Use each question's Q number in the <q n="..."> attribute. Format your response as:
Only use Q numbers that appear in AVAILABLE UNANSWERED QUESTIONS. Do not invent new labels such as Q_new1; if a useful new question is not listed, omit it.
<think>
{think_hint}
</think>
<selected_experiments>
<q n="Q1">
<experiment_plan>[1-3 sentence actionable experiment plan for Q1]</experiment_plan>
</q>
<q n="Q2">
<experiment_plan>[1-3 sentence actionable experiment plan for Q2]</experiment_plan>
</q>
...
</selected_experiments>"""

    images: list = []
    if steps_context_images:
        images.extend(steps_context_images)
    if current_image is not None:
        images.append(current_image)

    text, cost = await _llm_call(config, prompt, images=images or None)
    selection_log["prompt"] = prompt
    selection_log["response"] = text

    selected_text = extract_xml_key(text, "selected_experiments")
    selected_indices: list[int] = []
    candidates: list[dict] = []
    invalid_q_refs: list[str | None] = []
    unanswered_source_set = set(unanswered_source_indices)
    if selected_text:
        for idx, raw_ref, q_body in _iter_q_blocks_with_refs(
            selected_text,
            len(current_qa),
        ):
            if idx is None:
                invalid_q_refs.append(raw_ref)
                continue
            if idx not in unanswered_source_set:
                continue
            if idx in selected_indices:
                continue
            if filter_questions and len(selected_indices) >= max_unanswered_qa_pairs:
                break
            selected_indices.append(idx)
            experiment_plan = extract_xml_key(q_body, "experiment_plan")
            if experiment_plan:
                experiment_plan = experiment_plan.strip()
            if not experiment_plan:
                continue
            candidates.append(
                {
                    "kind": "fresh",
                    "source_index": idx,
                    "question": current_qa[idx].question,
                    "plan": experiment_plan,
                    "topk_rank": len(selected_indices) - 1,
                    "formulation_prompt": prompt,
                    "formulation_response": text,
                    "formulation_cost": 0.0,
                }
            )

    if invalid_q_refs:
        selection_log["invalid_selected_experiment_refs"] = invalid_q_refs
        selection_log["parse_warning"] = (
            "Ignored malformed selected_experiments q references; "
            "expected exact bank labels like Q4 or 4."
        )

    if not selected_indices:
        selection_log["parse_error"] = "No valid selected experiments parsed"
        if filter_questions:
            selected_indices = unanswered_source_indices[:max_unanswered_qa_pairs]
        else:
            selected_indices = list(unanswered_source_indices)

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
    selection_log["candidates"] = [
        {
            "source_index": c["source_index"],
            "question": c["question"],
            "plan": c["plan"],
            "topk_rank": c["topk_rank"],
        }
        for c in candidates
    ]
    selection_log["post_selection_count"] = len(selected_qa)
    selection_log["post_selection_answered"] = sum(
        1 for q in selected_qa if q.answer is not None
    )
    selection_log["post_selection_unanswered"] = sum(
        1 for q in selected_qa if q.answer is None
    )

    logging.info(
        f"Q&A probe selection+formulation: {len(current_qa)} -> {len(selected_qa)} "
        f"prompt questions with {len(candidates)} experiment plans"
    )
    return selected_qa, selected_indices, candidates, cost, selection_log


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
            q
            for i, q in enumerate(working_qa)
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
