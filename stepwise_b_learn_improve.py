"""LLM improvement functions for stepwise B-learning.

These are variants of the functions in b_learn_improve.py that take raw step
sequences instead of episode summaries. The step sequence provides the LLM with
the actual (state, P(state), reasoning+action) tuples from the trajectory.
"""

import logging
import re

from omegaconf import DictConfig

from b_learn_improve import (
    FeedbackResult,
    QAFeedbackResult,
    _improve_with_perception_validation,
    _improve_with_perception_validation_conversational,
)
from llm_utils import extract_xml_key
from mixed_improve import (
    QAPair,
    CriticalMoment,
    _llm_call,
    _run_perception_on_observation,
    _build_execution_report,
)


PERCEPTION_INSTRUCTIONS = """For the perception module:
- It should be a valid Python function `perceive(observation_text: str) -> str`.
- Input `observation_text` contains everything from the direct environment observation as a string.
- Output should only contain features important for decision-making in the environment.
- Ensure the output does not exceed 2000 characters. Remove features that the agent does not use for decision-making.
- The output should be consistent with the current world knowledge and policy and should not make any additional or contradictory assumptions to them.
- Ensure that the perception module is working correctly — that it is correctly extracting the intended information from the raw environment state and presenting it clearly.
"""

RESPONSE_FORMAT = """Format your response as:
<think>
Analyze the step sequence and determine what needs to change.
</think>

<updated_beliefs>
<world_knowledge>
- [fact about mechanics, environmental properties, cause-and-effect relationships, etc ...]
- ...
</world_knowledge>
<policy>
- [what to do in specific situations, priorities, strategies for completing the objective etc ...]
- ...
</policy>
</updated_beliefs>

<updated_perception>
```python
def perceive(observation_text: str) -> str:
    # Your implementation here
    pass
```
</updated_perception>

<status>CONTINUE or SUBMIT</status>

Set status to SUBMIT if you believe your current beliefs and perception are sufficient given the available evidence. Set status to CONTINUE if you want to receive re-evaluation results and iterate further. When in doubt, prefer CONTINUE."""


PERCEPTION_ONLY_RESPONSE_FORMAT = """Format your response as:
<think>
Analyze the perception input/output examples and determine what the perception module should extract differently.
</think>

<updated_perception>
```python
def perceive(observation_text: str) -> str:
    # Your implementation here
    pass
```
</updated_perception>

<status>CONTINUE or SUBMIT</status>

Set status to SUBMIT if you believe your current perception module is extracting information well. Set status to CONTINUE if you want to see updated examples and iterate further. When in doubt, prefer CONTINUE."""


BELIEFS_ONLY_RESPONSE_FORMAT = """Format your response as:
<think>
Analyze the step sequence and determine what world knowledge and policy need to change.
</think>

<updated_beliefs>
<world_knowledge>
- [fact about mechanics, environmental properties, cause-and-effect relationships, etc ...]
- ...
</world_knowledge>
<policy>
- [what to do in specific situations, priorities, strategies for completing the objective etc ...]
- ...
</policy>
</updated_beliefs>

<perception_analysis>
Analysis of how the perception module could be improved:
- What extracted information was misleading or incorrect?
- What kind of information can be extracted that would help the agent make better decisions?
</perception_analysis>"""


def build_perception_followup_message(
    perception: str,
    sample_observations: list[tuple[str, int]] | None,
    prev_obs_section: str,
    current_turn: int = 0,
    max_turns: int = 0,
    sample_histories: list[list[str]] | None = None,
    history_window: int | None = None,
    display_tail: int | None = None,
    perception_instructions: str | None = None,
    response_format: str | None = None,
) -> str:
    """Build followup message for perception-only track showing updated perception outputs."""
    if not sample_observations:
        return "No sample observations available for re-evaluation. Please review your previous changes and decide whether to SUBMIT or CONTINUE."
    if perception_instructions is None:
        perception_instructions = PERCEPTION_INSTRUCTIONS
    if response_format is None:
        response_format = PERCEPTION_ONLY_RESPONSE_FORMAT

    new_obs_section = _build_obs_section(
        perception, sample_observations,
        sample_histories=sample_histories, history_window=history_window,
        display_tail=display_tail,
    )

    turns_remaining = max_turns - current_turn
    turns_note = f"You have {turns_remaining} turn(s) remaining (turn {current_turn}/{max_turns}). " if max_turns > 0 else ""

    return f"""Here are the results of your updated perception module on the same sample observations:

{new_obs_section}

{perception_instructions}

{turns_note}Review whether the perception changes improved, degraded, or had no effect on the output quality.
Then provide an updated perception module (or keep it unchanged if satisfied).

{response_format}"""


def build_steps_beliefs_followup_message() -> str:
    """Build followup message for steps-beliefs track."""
    return f"""Review your previous changes to world knowledge and policy.
Based on the step sequence provided earlier, refine your beliefs further, or SUBMIT if satisfied.

{BELIEFS_ONLY_RESPONSE_FORMAT}"""


def build_perception_with_analysis_prompt(
    beliefs: str,
    perception: str,
    default_knowledge: str,
    obs_section: str,
    perception_analysis: str,
    max_iterations: int,
    perception_instructions: str | None = None,
    response_format: str | None = None,
) -> str:
    """Build the initial prompt for the post-beliefs perception improvement loop (Track 1c).

    Identical in structure to the Track 1a perception prompt but includes
    the perception_analysis from Track 1b as additional guidance.

    `perception_instructions` and `response_format` default to the single-obs
    module-level constants but may be overridden (e.g. by stepwise_eb_learn
    to teach the LLM the list-based perceive signature).
    """
    if perception_instructions is None:
        perception_instructions = PERCEPTION_INSTRUCTIONS
    if response_format is None:
        response_format = PERCEPTION_ONLY_RESPONSE_FORMAT
    analysis_section = ""
    if perception_analysis and perception_analysis.strip():
        analysis_section = f"""
The following analysis was generated by reviewing the agent's step sequence and identifies specific areas where the perception module could be improved:
=== PERCEPTION IMPROVEMENT ANALYSIS ===

{perception_analysis.strip()}
=== END PERCEPTION IMPROVEMENT ANALYSIS ===
"""

    return f"""We are interacting with an environment and trying to figure out how it works. We maintain a perception module that extracts useful features from raw environment observations to help the agent make decisions.

The agent receives the following default instructions/knowledge:
=== DEFAULT KNOWLEDGE ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

We maintain the following current beliefs about the environment:
=== CURRENT BELIEFS ===
{beliefs if beliefs else "(empty - no beliefs yet)"}
=== END CURRENT BELIEFS ===

The following code is used to extract useful features from the raw environment observations:
=== CURRENT PERCEPTION MODULE ===
{perception if perception else "(empty - no perception module yet)"}
=== END CURRENT PERCEPTION MODULE ===

The following are examples of sampled inputs to and outputs from the current perception module:
=== SAMPLED INPUTS AND OUTPUTS ===
{obs_section}
=== END OF SAMPLED INPUTS AND OUTPUT ===

{analysis_section}

Your task is to improve the perception module so that it extracts useful information from the raw environment observations.

{perception_instructions}

Only update the perception module if it can be improved based on the examples above. Do not modify world knowledge or policy.

This is a multi-turn conversation. After each response, you will receive updated perception examples to review. You can iterate up to {max_iterations} rounds of improvement.


{response_format}"""


def _render_history_input(history: list[str], display_tail: int) -> str:
    """Render a list history as elision marker + last K frames.

    The last frame is always the current observation (marked current="true").
    If len(history) <= K (or K <= 0), shows all frames without elision.
    """
    n = len(history)
    if n == 0:
        return "(empty history)"
    k = max(0, int(display_tail))
    if k <= 0 or n <= k:
        tail_idxs = list(range(n))
        elided = 0
    else:
        tail_idxs = list(range(n - k, n))
        elided = n - k

    def _frame(i: int) -> str:
        rel = i - (n - 1)
        attrs = f'index="{i}" relative="{rel}"'
        if i == n - 1:
            attrs += ' current="true"'
        return f"<frame {attrs}>\n{history[i]}\n</frame>"

    parts: list[str] = []
    if elided > 0:
        parts.append(
            f"<!-- {elided} earlier frame(s) elided; perceive() receives the full history -->"
        )
    parts.extend(_frame(i) for i in tail_idxs)
    return "\n".join(parts)


def _build_obs_section(
    perception: str,
    sample_observations: list[tuple[str, int]] | None,
    sample_histories: list[list[str]] | None = None,
    history_window: int | None = None,
    display_tail: int | None = None,
) -> str:
    """Build the sampled perception examples section for prompts.

    If sample_histories and history_window are provided (aligned with
    sample_observations), perception is invoked via _run_perception_on_history
    on each step's windowed history. Otherwise legacy single-obs behavior.

    When history mode is active and display_tail is set, <perception_input>
    shows only the last K frames with an elision marker noting that perceive()
    saw earlier frames too. If display_tail is None, only the current raw_obs
    is shown (legacy display).
    """
    if not sample_observations:
        return ""
    from mixed_improve import _run_perception_on_history
    use_history = (
        sample_histories is not None
        and history_window is not None
        and len(sample_histories) == len(sample_observations)
    )
    tail_note = ""
    if use_history and display_tail is not None:
        tail_note = (
            f"Note: Each <perception_input> below shows only the last {display_tail} frame(s) "
            f"of the observation history for display purposes (the most recent frame is marked "
            f'current="true"). The actual perceive() call receives the full history '
            f"(up to {history_window} frames).\n\n"
        )

    obs_blocks = []
    for idx, (raw_obs, step_num) in enumerate(sample_observations):
        if use_history:
            history = sample_histories[idx]
            perc_out = _run_perception_on_history(perception, history, history_window)
            if display_tail is not None:
                input_block = _render_history_input(history, display_tail)
            else:
                input_block = raw_obs
        else:
            perc_out = _run_perception_on_observation(perception, raw_obs)
            input_block = raw_obs
        obs_blocks.append(
            f"<perception_example step=\"{step_num}\">\n"
            f"<perception_input>\n{input_block}\n</perception_input>\n"
            f"<perception_output>\n{perc_out if perc_out else '(empty)'}\n</perception_output>\n"
            f"</perception_example>"
        )
    return (
        "\n<sampled_perception_examples>\n"
        + tail_note
        + "\n\n".join(obs_blocks)
        + "\n</sampled_perception_examples>\n"
    )


def _build_execution_report_section(
    perception: str,
    sample_observations: list[tuple[str, int]] | None,
    sample_histories: list[list[str]] | None = None,
    history_window: int | None = None,
) -> str:
    """Build execution report section for prompts.

    History-aware variant currently renders the same legacy execution report
    (raw_obs → single-obs perception output). The history-aware perception
    output already appears in `_build_obs_section`, which is where the LLM
    reviews the current module's behavior.
    """
    if not sample_observations:
        return ""
    exec_report = _build_execution_report(perception, sample_observations)
    if not exec_report:
        return ""
    return f"""
=== CURRENT PERCEPTION EXECUTION OUTPUT ===
Below is the actual output of the current perception module on real observations:

{exec_report}
=== END CURRENT PERCEPTION EXECUTION OUTPUT ===
"""


def parse_submit_signal(response_text: str) -> bool:
    """Check if LLM indicated it's satisfied with current state."""
    status = extract_xml_key(response_text, "status")
    return status is not None and status.strip().upper() == "SUBMIT"


def build_steps_followup_message(
    perception: str,
    sample_observations: list[tuple[str, int]] | None,
    prev_obs_section: str,
) -> str:
    """Build followup message for steps track showing updated perception outputs."""
    if not sample_observations:
        return "No sample observations available for re-evaluation. Please review your previous changes and decide whether to SUBMIT or CONTINUE."

    new_obs_section = _build_obs_section(perception, sample_observations)

    return f"""Here are the results of your updated perception module on the same sample observations:

{new_obs_section}

Review whether the perception changes improved, degraded, or had no effect on the output quality.
Then provide updated beliefs and perception (or keep them unchanged if satisfied).

{RESPONSE_FORMAT}"""


def build_qa_followup_message(
    qa_feedback_results: list,
    prev_num_correct: int,
    prev_num_incorrect: int,
    response_format: str | None = None,
) -> str:
    """Build followup message for QA track showing re-evaluation delta."""
    if response_format is None:
        response_format = RESPONSE_FORMAT
    num_correct = sum(1 for fr in qa_feedback_results if fr.verdict == "CORRECT")
    num_incorrect = sum(1 for fr in qa_feedback_results if fr.verdict == "INCORRECT")
    num_inconclusive = sum(1 for fr in qa_feedback_results if fr.verdict == "INCONCLUSIVE")

    delta_correct = num_correct - prev_num_correct
    delta_incorrect = num_incorrect - prev_num_incorrect
    delta_sign_c = "+" if delta_correct >= 0 else ""
    delta_sign_i = "+" if delta_incorrect >= 0 else ""

    # Show remaining failures
    failure_blocks = []
    for i, fr in enumerate(qa_feedback_results, 1):
        if fr.verdict == "INCORRECT":
            actual = "YES" if fr.forward.qa_pair.answer else "NO"
            failure_blocks.append(
                f"  - Q{i}: \"{fr.forward.qa_pair.question}\" "
                f"Agent said {fr.forward.predicted_answer} (should be {actual}). "
                f"Feedback: {fr.feedback}"
            )
    failures_text = "\n".join(failure_blocks) if failure_blocks else "  (none)"

    return f"""=== QA RE-EVALUATION RESULTS ===
Correct: {num_correct}/{len(qa_feedback_results)} ({delta_sign_c}{delta_correct} vs previous)
Incorrect: {num_incorrect}/{len(qa_feedback_results)} ({delta_sign_i}{delta_incorrect} vs previous)
Inconclusive: {num_inconclusive}/{len(qa_feedback_results)}

Remaining failures:
{failures_text}
=== END QA RE-EVALUATION RESULTS ===

Based on these updated results, refine your beliefs and perception further, or SUBMIT if satisfied.

{response_format}"""


def build_moments_followup_message(
    feedback_results: list,
    prev_num_correct: int,
    prev_num_incorrect: int,
) -> str:
    """Build followup message for moments track showing re-evaluation delta."""
    num_correct = sum(1 for fr in feedback_results if fr.verdict == "CORRECT")
    num_incorrect = sum(1 for fr in feedback_results if fr.verdict == "INCORRECT")
    num_inconclusive = sum(1 for fr in feedback_results if fr.verdict == "INCONCLUSIVE")

    delta_correct = num_correct - prev_num_correct
    delta_incorrect = num_incorrect - prev_num_incorrect
    delta_sign_c = "+" if delta_correct >= 0 else ""
    delta_sign_i = "+" if delta_incorrect >= 0 else ""

    # Show remaining failures
    failure_blocks = []
    for i, fr in enumerate(feedback_results, 1):
        if fr.verdict == "INCORRECT":
            m = fr.forward.moment
            good = ", ".join(m.good_actions) if m.good_actions else "NONE"
            failure_blocks.append(
                f"  - M{i}: Goal=\"{m.goal}\" Agent predicted \"{fr.forward.predicted_action}\" "
                f"(desired: {good}). Feedback: {fr.feedback}"
            )
    failures_text = "\n".join(failure_blocks) if failure_blocks else "  (none)"

    return f"""=== MOMENT RE-EVALUATION RESULTS ===
Correct: {num_correct}/{len(feedback_results)} ({delta_sign_c}{delta_correct} vs previous)
Incorrect: {num_incorrect}/{len(feedback_results)} ({delta_sign_i}{delta_incorrect} vs previous)
Inconclusive: {num_inconclusive}/{len(feedback_results)}

Remaining failures:
{failures_text}
=== END MOMENT RE-EVALUATION RESULTS ===

Based on these updated results, refine your beliefs and perception further, or SUBMIT if satisfied.

{RESPONSE_FORMAT}"""


# ============================================================
# Track 1: Steps-based improvement
# ============================================================


async def improve_from_steps(
    config: DictConfig,
    beliefs: str,
    perception: str,
    steps_context: str,
    sample_observations: list[tuple[str, int]] | None,
    default_knowledge: str,
) -> tuple[str, str, float, str, str]:
    """Improve beliefs and perception based on the raw sequence of environment steps.

    Presents the LLM with:
    - Current perception code and its output on sampled states
    - Current world knowledge and policy
    - The actual step sequence (state, P(state), reasoning+action)

    Returns: (new_beliefs, new_perception, total_cost, prompt_used, llm_response)
    """
    if not steps_context:
        return beliefs, perception, 0.0, "", ""

    obs_section = _build_obs_section(perception, sample_observations)

    base_prompt = f"""You are improving an agent's knowledge and perception based on its recent interactions with an environment.

The agent receives the following default instructions/knowledge:
=== DEFAULT KNOWLEDGE ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== CURRENT BELIEFS ===
{beliefs if beliefs else "(empty - no beliefs yet)"}
=== END CURRENT BELIEFS ===

=== CURRENT PERCEPTION MODULE ===
{perception if perception else "(empty - no perception module yet)"}
=== END CURRENT PERCEPTION MODULE ===
{obs_section}
=== SEQUENCE OF STEPS ===
Below is the actual sequence of the agent's interactions with the environment.
Each step shows: the raw state observation, the perception module's output on that state, the agent's reasoning, and the action taken.

{steps_context}
=== END SEQUENCE OF STEPS ===

Your task: Based on the step sequence, improve the agent's world knowledge, policy, and perception module.

Analyze:
1. What patterns of success or failure appear in the step sequence?
2. Is the perception module extracting the right information from observations?
3. Are the world knowledge facts accurate based on observed evidence?
4. Is the policy leading to good decisions?

Guidelines:
- Keep each beliefs section to at most 8 concise bullet points. Merge redundant points.
- Balance safety with progress toward the objective.
- Remove beliefs that are contradicted by evidence in the step sequence.

{PERCEPTION_INSTRUCTIONS}

{RESPONSE_FORMAT}"""

    return await _improve_with_perception_validation(
        config, beliefs, perception, base_prompt, sample_observations,
    )


# ============================================================
# Track 2: QA-based improvement (with steps context)
# ============================================================


async def improve_from_qa_feedback_with_steps(
    config: DictConfig,
    beliefs: str,
    perception: str,
    qa_feedback_results: list[QAFeedbackResult],
    default_knowledge: str,
    steps_context: str,
    sample_observations: list[tuple[str, int]] | None = None,
) -> tuple[str, str, float, str, str]:
    """Improve beliefs and perception based on QA feedback, using step sequence as context.

    Returns: (new_beliefs, new_perception, total_cost, prompt_used, llm_response)
    """
    if not qa_feedback_results:
        return beliefs, perception, 0.0, "", ""

    # Build QA feedback items
    qa_blocks = []
    for i, fr in enumerate(qa_feedback_results, 1):
        actual = "YES" if fr.forward.qa_pair.answer else "NO"
        qa_blocks.append(
            f"<qa_feedback n=\"{i}\">\n"
            f"<question>{fr.forward.qa_pair.question}</question>\n"
            f"<correct_answer>{actual}</correct_answer>\n"
            f"<evidence>{fr.forward.qa_pair.evidence}</evidence>\n"
            f"<predicted_answer>{fr.forward.predicted_answer}</predicted_answer>\n"
            f"<agent_reasoning>{fr.forward.reasoning}</agent_reasoning>\n"
            f"<verdict>{fr.verdict}</verdict>\n"
            f"<feedback>{fr.feedback}</feedback>\n"
            f"</qa_feedback>"
        )
    qa_text = "\n\n".join(qa_blocks)

    num_correct = sum(1 for fr in qa_feedback_results if fr.verdict == "CORRECT")
    num_incorrect = sum(1 for fr in qa_feedback_results if fr.verdict == "INCORRECT")

    execution_report_section = _build_execution_report_section(perception, sample_observations)

    base_prompt = f"""You are improving an agent's knowledge and perception based on testing its understanding of the environment via question-answering.

The agent receives the following default instructions/knowledge:
=== DEFAULT KNOWLEDGE ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== CURRENT BELIEFS ===
{beliefs if beliefs else "(empty - no beliefs yet)"}
=== END CURRENT BELIEFS ===

=== CURRENT PERCEPTION MODULE ===
{perception if perception else "(empty - no perception module yet)"}
=== END CURRENT PERCEPTION MODULE ===
{execution_report_section}
We tested the agent's understanding by asking it factual questions about the environment.
The agent answered based only on its current world knowledge.

Results: {num_correct} correct, {num_incorrect} incorrect out of {len(qa_feedback_results)} evaluated.

<qa_feedback_results>
{qa_text}
</qa_feedback_results>

=== SEQUENCE OF STEPS (for additional context) ===
{steps_context if steps_context else "(no steps recorded yet)"}
=== END SEQUENCE OF STEPS ===

Your task: Based on the QA feedback, improve the agent's world knowledge, policy, and perception module.

For INCORRECT predictions, analyze:
1. Was the agent's world knowledge missing the relevant fact? If so, add it.
2. Was the agent's world knowledge wrong? If so, correct it.
3. Does the perception module need to extract different information to support this knowledge? If so, update it.

Guidelines:
- Keep each beliefs section to at most 8 concise bullet points. Merge redundant points.
- Balance safety with progress toward the objective.
- Remove beliefs that are contradicted by the QA evidence.

{PERCEPTION_INSTRUCTIONS}

{RESPONSE_FORMAT}"""

    return await _improve_with_perception_validation(
        config, beliefs, perception, base_prompt, sample_observations,
    )


# ============================================================
# Track 3: Critical moment improvement (with steps context)
# ============================================================


async def improve_from_feedback_with_steps(
    config: DictConfig,
    beliefs: str,
    perception: str,
    feedback_results: list[FeedbackResult],
    default_knowledge: str,
    steps_context: str,
    sample_observations: list[tuple[str, int]] | None = None,
) -> tuple[str, str, float, str, str]:
    """Improve beliefs and perception based on critical moment feedback, using step sequence as context.

    Returns: (new_beliefs, new_perception, total_cost, prompt_used, llm_response)
    """
    if not feedback_results:
        return beliefs, perception, 0.0, "", ""

    # Build feedback items
    feedback_blocks = []
    for i, fr in enumerate(feedback_results, 1):
        m = fr.forward.moment
        good = ", ".join(m.good_actions) if m.good_actions else "NONE"
        bad = ", ".join(m.bad_actions) if m.bad_actions else "NONE"
        feedback_blocks.append(
            f"<moment_feedback n=\"{i}\">\n"
            f"<goal>{m.goal}</goal>\n"
            f"<raw_observation>\n{m.raw_observation[:1500]}\n</raw_observation>\n"
            f"<perception_output>\n{fr.forward.perception_output if fr.forward.perception_output else '(empty)'}\n</perception_output>\n"
            f"<predicted_action>{fr.forward.predicted_action}</predicted_action>\n"
            f"<agent_reasoning>{fr.forward.reasoning}</agent_reasoning>\n"
            f"<verdict>{fr.verdict}</verdict>\n"
            f"<feedback>{fr.feedback}</feedback>\n"
            f"<desired_actions>{good}</desired_actions>\n"
            f"<undesired_actions>{bad}</undesired_actions>\n"
            f"</moment_feedback>"
        )
    feedback_text = "\n\n".join(feedback_blocks)

    num_correct = sum(1 for fr in feedback_results if fr.verdict == "CORRECT")
    num_incorrect = sum(1 for fr in feedback_results if fr.verdict == "INCORRECT")

    execution_report_section = _build_execution_report_section(perception, sample_observations)

    base_prompt = f"""You are improving an agent's knowledge and perception based on feedback from testing its decision-making.

The agent receives the following default instructions/knowledge:
=== DEFAULT KNOWLEDGE ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== CURRENT BELIEFS ===
{beliefs if beliefs else "(empty - no beliefs yet)"}
=== END CURRENT BELIEFS ===

=== CURRENT PERCEPTION MODULE ===
{perception if perception else "(empty - no perception module yet)"}
=== END CURRENT PERCEPTION MODULE ===
{execution_report_section}
We tested the agent by simulating its forward pipeline on critical moments from the environment.
For each moment, the agent saw the raw observation, processed it through the perception module,
and chose an action based on its knowledge and policy.

Results: {num_correct} correct, {num_incorrect} incorrect out of {len(feedback_results)} evaluated.

<moment_feedback_results>
{feedback_text}
</moment_feedback_results>

=== SEQUENCE OF STEPS (for additional context) ===
{steps_context if steps_context else "(no steps recorded yet)"}
=== END SEQUENCE OF STEPS ===

Your task: Based on the feedback, improve the agent's world knowledge, policy, and perception module.

For INCORRECT predictions, analyze:
1. Was the perception module missing critical information from the raw observation? If so, fix the perception module.
2. Was the agent's world knowledge wrong or incomplete? If so, update world_knowledge.
3. Was the agent's policy/strategy wrong? If so, update the policy.

Guidelines:
- Keep each beliefs section to at most 8 concise bullet points. Merge redundant points.
- Balance safety with progress toward the objective.
- Remove beliefs that are contradicted by the feedback evidence.

{PERCEPTION_INSTRUCTIONS}

{RESPONSE_FORMAT}"""

    return await _improve_with_perception_validation(
        config, beliefs, perception, base_prompt, sample_observations,
    )


# ============================================================
# Experiment generation (with step history)
# ============================================================


def _extract_obs_message(raw_obs: str) -> str:
    """Extract just the message section from a raw long_term_context observation."""
    match = re.search(r"message:\n(.*?)(?=\n\ncursor:|\Z)", raw_obs, re.DOTALL)
    return match.group(1).strip() if match else raw_obs.strip()



async def generate_experiments_from_steps(
    config: DictConfig,
    beliefs: str,
    qa_pairs: list[QAPair],
    critical_moments: list[CriticalMoment],
    perception_code: str,
    current_experiment: str | None,
    num_experiments: int,
    steps_context: str,
    current_observation: str | None = None,
    current_aux_observation: str | None = None,
    default_knowledge: str = "",
    past_experiments: list[str] | None = None,
) -> tuple[list[str], float, str, str]:
    """Generate experiments using step history, split beliefs, and current goal.

    Returns: (experiments, cost, prompt, raw_response)
    """
    # --- Section 4: Known facts ---
    known_facts = ""
    if qa_pairs:
        lines = []
        for i, qa in enumerate(qa_pairs):
            lines.append(f"  Q{i+1}: {qa.question} -> {'YES' if qa.answer else 'NO'}")
        known_facts += "Known Q&A pairs:\n" + "\n".join(lines) + "\n\n"
    if critical_moments:
        lines = []
        for i, m in enumerate(critical_moments):
            lines.append(
                f"  M{i+1}: State: {m.state} | Goal: {m.goal} "
                f"| Good: [{', '.join(m.good_actions)}] "
                f"| Bad: [{', '.join(m.bad_actions)}]"
            )
        known_facts += "Known critical moments:\n" + "\n".join(lines) + "\n\n"

    # --- Section 5: Step history ---
    step_history = steps_context

    # --- Section 6: Current experiment ---
    current_exp_text = current_experiment if current_experiment else "(no experiment set yet)"

    # --- Build prompt ---
    beliefs_section = f"""=== CURRENT BELIEFS ===
{beliefs if beliefs else "(empty - no beliefs yet)"}
=== END CURRENT BELIEFS ==="""

    from stepwise_b_learn import format_current_state  # deferred to avoid cycle
    current_obs_section = format_current_state(
        observation=current_observation,
        aux_observation=current_aux_observation,
        perception_code=perception_code,
        section_title="CURRENT STATE (agent has not yet acted)",
    )

    # --- Default knowledge section ---
    default_knowledge_section = ""
    if default_knowledge:
        default_knowledge_section = f"""
=== DEFAULT KNOWLEDGE ===
{default_knowledge}
=== END EFAULT KNOWLEDGE ===
"""

    # --- Past experiments section ---
    past_experiments_section = ""
    if past_experiments:
        past_lines = [f"  {i+1}. {exp}" for i, exp in enumerate(past_experiments)]
        past_experiments_section = f"""
=== PAST EXPERIMENTS (already tried) ===
{chr(10).join(past_lines)}
=== END PAST EXPERIMENTS ===
"""

    prompt = f"""You are designing the next experiment for an agent interacting with an environment.
{default_knowledge_section}

{beliefs_section}

=== RECENT HISTORY OF STATES AND ACTIONS ===
{step_history}
=== END RECENT HISTORY ==={current_obs_section}{past_experiments_section}
=== CURRENT EXPERIMENT ===
{current_exp_text}
=== END CURRENT EXPERIMENT ===

Your task: Decide if the current experiment has been tested and if so generate {num_experiments} possible experiment(s) to test next. 
If no experiment is set yet then create one.

Guidelines:
- Each experiment should be a specific, actionable hypothesis to test, 1-3 sentences long.
- Design experiments that will lead to learning new things about the world, compared to what we already know in the current world knowledge.
- Avoid repeating past experiments that have already been tried (see list above if available). Focus on novel hypotheses.

Format your response as:
<think>
...
</think>
<experiments>
If the current experiment needs to be maintained for the next step - 
null

If generating new experiments - 

EXPERIMENT 1: [First experiment to test]

EXPERIMENT 2: [Second experiment to test]
...
(Generate {num_experiments} experiments)
</experiments>"""

    text, cost = await _llm_call(config, prompt)

    experiments_text = extract_xml_key(text, "experiments")
    experiments = []
    if experiments_text:
        if experiments_text.strip().lower() == "null":
            experiments = [current_exp_text]
        else:
            for match in re.finditer(
                r"EXPERIMENT\s+\d+:\s*(.+?)(?=EXPERIMENT\s+\d+:|$)",
                experiments_text,
                re.DOTALL,
            ):
                exp = match.group(1).strip()
                if exp:
                    if len(exp) > 300:
                        exp = exp[:300].rsplit(" ", 1)[0] + "..."
                    experiments.append(exp)

    logging.info(f"Generated {len(experiments)} experiments from step history")
    return experiments, cost, prompt, text
