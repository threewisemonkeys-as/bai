"""LLM prompt functions for B-learning: forward pass, feedback, and improvement."""

import asyncio
import logging
import re
from dataclasses import dataclass, asdict

from omegaconf import DictConfig

from mixed_improve import (
    CriticalMoment,
    _llm_call,
    _llm_call_conversational,
    _run_perception_on_observation,
    _action_in_set,
    process_trajectories,
    generate_experiments_from_gaps,
    serialize_moments,
    deserialize_moments,
    serialize_qa_pairs,
    deserialize_qa_pairs,
    _build_execution_report,
    QAPair,
)
from improve import validate_perception_code
from llm_utils import extract_xml_key


@dataclass
class ForwardResult:
    moment: CriticalMoment
    predicted_action: str
    reasoning: str
    perception_output: str  # P(s) for this moment


@dataclass
class FeedbackResult:
    forward: ForwardResult
    feedback: str           # explanation of why correct/incorrect
    verdict: str            # "CORRECT" | "INCORRECT" | "INCONCLUSIVE"


async def _forward_pass_batch(
    config: DictConfig,
    beliefs: str,
    perception: str,
    moments: list[CriticalMoment],
    perc_outputs: list[str],
    batch_offset: int,
) -> tuple[list[ForwardResult], float]:
    """Run forward pass on a single batch of moments."""
    moment_blocks = []
    for i, (m, perc_out) in enumerate(zip(moments, perc_outputs), 1):
        moment_blocks.append(
            f"--- Situation {i} ---\n"
            f"Goal: {m.goal}\n"
            f"Raw observation:\n{m.raw_observation[:2000]}\n"
            f"Perception output:\n{perc_out if perc_out else '(empty — perception produced no output)'}\n"
            f"---"
        )
    moments_text = "\n\n".join(moment_blocks)

    response_format = "\n".join(
        f"<S{i}>\n<reasoning>Your step-by-step reasoning</reasoning>\n<action>single action command</action>\n</S{i}>"
        for i in range(1, len(moments) + 1)
    )

    prompt = f"""You are an agent interacting with an environment. Here is what you know:

=== BELIEFS ===
{beliefs if beliefs else "(empty - no beliefs yet)"}
=== END BELIEFS ===

For each situation below, you see the raw observation from the environment and the
output of your perception module. Based on your knowledge, policy, and the observation,
reason about what to do and provide a single action command.

{moments_text}

For each situation, respond with reasoning and an action in this format:
{response_format}"""

    text, cost = await _llm_call(config, prompt)

    results = []
    for i, (m, perc_out) in enumerate(zip(moments, perc_outputs), 1):
        s_block = extract_xml_key(text, f"S{i}")
        if s_block:
            reasoning = extract_xml_key(s_block, "reasoning") or ""
            action = extract_xml_key(s_block, "action") or "MISSING"
        else:
            reasoning = ""
            action = "MISSING"
        results.append(ForwardResult(
            moment=m,
            predicted_action=action.strip(),
            reasoning=reasoning.strip(),
            perception_output=perc_out,
        ))

    return results, cost


async def forward_pass(
    config: DictConfig,
    beliefs: str,
    perception: str,
    moments: list[CriticalMoment],
    max_per_batch: int = 15,
) -> tuple[list[ForwardResult], float]:
    """Run the forward pipeline on critical moments to predict actions.

    Batches moments and runs batches in parallel.

    Returns: (list[ForwardResult], total_cost)
    """
    if not moments:
        return [], 0.0

    # Pre-compute perception outputs
    perc_outputs = [
        _run_perception_on_observation(perception, m.raw_observation)
        for m in moments
    ]

    # Split into batches
    batches = []
    for start in range(0, len(moments), max_per_batch):
        end = start + max_per_batch
        batches.append((
            moments[start:end],
            perc_outputs[start:end],
            start,
        ))

    # Run batches in parallel
    tasks = [
        _forward_pass_batch(config, beliefs, perception, batch_moments, batch_perc, offset)
        for batch_moments, batch_perc, offset in batches
    ]
    batch_results = await asyncio.gather(*tasks)

    all_results = []
    total_cost = 0.0
    for results, cost in batch_results:
        all_results.extend(results)
        total_cost += cost

    return all_results, total_cost


async def _get_feedback_batch(
    config: DictConfig,
    forward_results: list[ForwardResult],
) -> tuple[list[FeedbackResult], float, str, str]:
    """Get feedback on a single batch of forward results.

    Returns: (results, cost, prompt_used, llm_response)
    """
    item_blocks = []
    for i, fr in enumerate(forward_results, 1):
        m = fr.moment
        good = ", ".join(m.good_actions) if m.good_actions else "NONE"
        bad = ", ".join(m.bad_actions) if m.bad_actions else "NONE"
        item_blocks.append(
            f"--- Situation {i} ---\n"
            f"Goal: {m.goal}\n"
            f"State: {m.state}\n"
            f"Raw observation (excerpt):\n{m.raw_observation[:1500]}\n"
            f"Predicted action: {fr.predicted_action}\n"
            f"Reasoning: {fr.reasoning}\n"
            f"Desired actions: {good}\n"
            f"Undesired actions: {bad}\n"
            f"Evidence: {m.evidence}\n"
            f"---"
        )
    items_text = "\n\n".join(item_blocks)

    response_format = "\n".join(
        f"<F{i}>\n<verdict>CORRECT or INCORRECT or INCONCLUSIVE</verdict>\n"
        f"<feedback>Explanation</feedback>\n</F{i}>"
        for i in range(1, len(forward_results) + 1)
    )

    prompt = f"""You are evaluating whether an agent's predicted actions are correct for various situations in an environment.

For each situation below, you are given:
- The goal and state description
- The agent's predicted action and reasoning
- The desired action(s) (ground truth of what the agent should do)
- The undesired action(s) (ground truth of what the agent should NOT do)
- Evidence from the trajectory explaining why

Evaluate whether the predicted action is:
- CORRECT: matches or is equivalent to a desired action
- INCORRECT: matches an undesired action, or is clearly wrong given the evidence
- INCONCLUSIVE: cannot determine from the available information

Provide specific feedback explaining WHY the action is correct or incorrect.
For incorrect actions, explain what the agent misunderstood and what knowledge or perception was missing or wrong.

{items_text}

For each situation, respond in this format:
{response_format}"""

    text, cost = await _llm_call(config, prompt)

    results = []
    for i, fr in enumerate(forward_results, 1):
        f_block = extract_xml_key(text, f"F{i}")
        if f_block:
            verdict_str = extract_xml_key(f_block, "verdict") or "INCONCLUSIVE"
            feedback_str = extract_xml_key(f_block, "feedback") or ""
        else:
            verdict_str = "INCONCLUSIVE"
            feedback_str = ""

        # Normalize verdict
        verdict_upper = verdict_str.strip().upper()
        if "CORRECT" in verdict_upper and "INCORRECT" not in verdict_upper:
            verdict = "CORRECT"
        elif "INCORRECT" in verdict_upper:
            verdict = "INCORRECT"
        else:
            verdict = "INCONCLUSIVE"

        results.append(FeedbackResult(
            forward=fr,
            feedback=feedback_str.strip(),
            verdict=verdict,
        ))

    return results, cost, prompt, text


async def get_feedback(
    config: DictConfig,
    forward_results: list[ForwardResult],
    max_per_batch: int = 15,
) -> tuple[list[FeedbackResult], float, list[str], list[str]]:
    """Get feedback on forward pass predictions.

    Batches results and runs batches in parallel.

    Returns: (list[FeedbackResult], total_cost, prompts_used, responses)
    """
    if not forward_results:
        return [], 0.0, [], []

    # Split into batches
    batches = []
    for start in range(0, len(forward_results), max_per_batch):
        end = start + max_per_batch
        batches.append(forward_results[start:end])

    tasks = [_get_feedback_batch(config, batch) for batch in batches]
    batch_results = await asyncio.gather(*tasks)

    all_results = []
    total_cost = 0.0
    all_prompts = []
    all_responses = []
    for results, cost, prompt, response in batch_results:
        all_results.extend(results)
        total_cost += cost
        all_prompts.append(prompt)
        all_responses.append(response)

    return all_results, total_cost, all_prompts, all_responses


async def improve_from_feedback(
    config: DictConfig,
    beliefs: str,
    perception: str,
    feedback_results: list[FeedbackResult],
    default_knowledge: str,
    episode_summaries: str,
    sample_observations: list[tuple[str, int]] | None = None,
) -> tuple[str, str, float, str, str]:
    """Improve beliefs and perception based on feedback from forward pass testing.

    Takes non-inconclusive feedback results and generates updated beliefs
    (world_knowledge + policy) and perception code.

    Validates perception code with retry (up to 3 attempts).
    Falls back to old perception if all retries fail.

    Returns: (new_beliefs, new_perception, total_cost, prompt_used, llm_response)
    """
    if not feedback_results:
        return beliefs, perception, 0.0, "", ""

    # Build feedback items for the prompt
    feedback_blocks = []
    for i, fr in enumerate(feedback_results, 1):
        m = fr.forward.moment
        good = ", ".join(m.good_actions) if m.good_actions else "NONE"
        bad = ", ".join(m.bad_actions) if m.bad_actions else "NONE"
        feedback_blocks.append(
            f"--- Feedback {i} ---\n"
            f"Goal: {m.goal}\n"
            f"Raw observation:\n{m.raw_observation[:1500]}\n"
            f"Perception output:\n{fr.forward.perception_output if fr.forward.perception_output else '(empty)'}\n"
            f"Agent's predicted action: {fr.forward.predicted_action}\n"
            f"Agent's reasoning: {fr.forward.reasoning}\n"
            f"Verdict: {fr.verdict}\n"
            f"Feedback: {fr.feedback}\n"
            f"Desired actions: {good}\n"
            f"Undesired actions: {bad}\n"
            f"---"
        )
    feedback_text = "\n\n".join(feedback_blocks)

    num_correct = sum(1 for fr in feedback_results if fr.verdict == "CORRECT")
    num_incorrect = sum(1 for fr in feedback_results if fr.verdict == "INCORRECT")

    # Build execution report for current perception
    execution_report_section = ""
    if sample_observations:
        exec_report = _build_execution_report(perception, sample_observations)
        if exec_report:
            execution_report_section = f"""
=== CURRENT PERCEPTION EXECUTION OUTPUT ===
Below is the actual output of the current perception module on real observations:

{exec_report}
=== END CURRENT PERCEPTION EXECUTION OUTPUT ===
"""

    perception_instructions = """For the perception module:
- It must be a Python function `perceive(observation_text: str) -> str`.
- Input `observation_text` contains everything from the direct observation as a string.
- The code must be valid Python.
- Output should be a textual description of the environment state useful for decision-making.
- Output should contain all information necessary for progressing and be presented clearly."""

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

{feedback_text}

Recent episode summaries for additional context:
{episode_summaries}

Your task: Based on the feedback, improve the agent's world knowledge, policy, and perception module.

For INCORRECT predictions, analyze:
1. Was the perception module missing critical information from the raw observation? If so, fix the perception module.
2. Was the agent's world knowledge wrong or incomplete? If so, update world_knowledge.
3. Was the agent's policy/strategy wrong? If so, update the policy.

Guidelines:
- Keep each beliefs section to at most 8 concise bullet points. Merge redundant points.
- Balance safety with progress toward the objective.
- Remove beliefs that are contradicted by the feedback evidence.

{perception_instructions}

Format your response as:
<think>
Analyze the feedback patterns and determine what needs to change.
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
</updated_perception>"""

    max_retries = 3
    perception_error = None
    new_beliefs = beliefs
    new_perception = perception
    total_cost = 0.0

    for attempt in range(max_retries):
        if perception_error:
            prompt = f"""{base_prompt}

=== PERCEPTION CODE ERROR (RETRY {attempt}/{max_retries}) ===
Your previous perception code had an error:
{perception_error}

Please fix the error in your perception code.
=== END OF PERCEPTION CODE ERROR ===
"""
        else:
            prompt = base_prompt

        text, cost = await _llm_call(config, prompt)
        total_cost += cost

        # Extract beliefs
        updated_beliefs = extract_xml_key(text, "updated_beliefs")
        if updated_beliefs:
            new_beliefs = updated_beliefs.strip()

        # Extract perception code
        candidate_perception = extract_xml_key(text, "updated_perception")
        if not candidate_perception:
            logging.warning(f"Failed to extract updated_perception (attempt {attempt + 1})")
            perception_error = "No <updated_perception> block found in response."
            continue

        candidate_perception = candidate_perception.strip()
        # Strip markdown markers
        if candidate_perception.startswith("```python"):
            candidate_perception = candidate_perception[len("```python"):].strip()
        elif candidate_perception.startswith("```"):
            candidate_perception = candidate_perception[len("```"):].strip()
        if candidate_perception.endswith("```"):
            candidate_perception = candidate_perception[:-len("```")].strip()

        # Validate perception code
        test_obs = [raw for raw, _ in sample_observations[:3]] if sample_observations else None
        is_valid, error_msg = validate_perception_code(candidate_perception, test_observations=test_obs)
        if not is_valid:
            perception_error = error_msg
            logging.warning(f"Perception code validation failed (attempt {attempt + 1}/{max_retries}): {error_msg}")
            continue

        # Execution verification: check for output degradation
        if sample_observations:
            new_outputs = [
                _run_perception_on_observation(candidate_perception, raw_obs)
                for raw_obs, _ in sample_observations[:3]
            ]
            degraded_samples = []
            for raw_obs, step_num in sample_observations[:3]:
                new_out = _run_perception_on_observation(candidate_perception, raw_obs)
                raw_has_map = "map:" in raw_obs and len(raw_obs) > 500
                new_line_count = len(new_out.strip().splitlines()) if new_out else 0
                if raw_has_map and new_line_count <= 3:
                    degraded_samples.append(
                        f"Sample step {step_num}: raw observation has {len(raw_obs)} chars "
                        f"but perception output is only {new_line_count} line(s):\n"
                        f"{new_out if new_out else '(empty)'}"
                    )

            if degraded_samples:
                new_report = _build_execution_report(candidate_perception, sample_observations)
                perception_error = (
                    f"Perception output appears degraded on real observations:\n\n"
                    + "\n\n".join(degraded_samples) + "\n\n"
                    f"Full execution I/O:\n{new_report}\n\n"
                    f"Please fix the perception code to correctly parse these observations."
                )
                logging.warning(f"Perception degradation detected (attempt {attempt + 1}/{max_retries})")
                continue

        new_perception = candidate_perception
        logging.info(f"Perception code validated successfully on attempt {attempt + 1}")
        return new_beliefs, new_perception, total_cost, base_prompt, text

    # All retries failed for perception — keep new beliefs but old perception
    logging.error(f"All {max_retries} attempts to generate valid perception failed. Keeping previous perception.")
    return new_beliefs, perception, total_cost, base_prompt, text


async def extract_and_consolidate_moments(
    config: DictConfig,
    beliefs: str,
    perception: str,
    output_dir: str,
    existing_moments: list[CriticalMoment],
    default_knowledge: str,
    step: int,
) -> tuple[str, list[list[CriticalMoment]], list[CriticalMoment], list[int], float]:
    """Extract moments from trajectories and consolidate with existing ones.

    Wraps process_trajectories with use_qa=False, use_moments=True.

    Returns: (episode_summaries, per_traj_moments, new_moments, remove_indices, cost)
    """
    (
        episode_summaries,
        _per_traj_qa,
        per_traj_moments,
        _new_qa,
        new_moments,
        _remove_qa_indices,
        remove_moment_indices,
        cost,
        _extraction_records,
    ) = await process_trajectories(
        config=config,
        beliefs=beliefs,
        perception=perception,
        output_dir=output_dir,
        existing_qa=[],
        existing_moments=existing_moments,
        default_knowledge=default_knowledge,
        step=step,
        use_qa=False,
        use_moments=True,
    )

    return episode_summaries, per_traj_moments, new_moments, remove_moment_indices, cost


def serialize_feedback_results(results: list[FeedbackResult]) -> list[dict]:
    """Serialize feedback results for JSON storage."""
    serialized = []
    for fr in results:
        serialized.append({
            "moment_state": fr.forward.moment.state,
            "moment_goal": fr.forward.moment.goal,
            "moment_good_actions": fr.forward.moment.good_actions,
            "moment_bad_actions": fr.forward.moment.bad_actions,
            "predicted_action": fr.forward.predicted_action,
            "reasoning": fr.forward.reasoning,
            "perception_output": fr.forward.perception_output,
            "verdict": fr.verdict,
            "feedback": fr.feedback,
        })
    return serialized


# ============================================================
# Knowledge extraction (QA + moments)
# ============================================================

async def extract_and_consolidate_knowledge(
    config: DictConfig,
    beliefs: str,
    perception: str,
    output_dir: str,
    existing_qa: list[QAPair],
    existing_moments: list[CriticalMoment],
    default_knowledge: str,
    step: int,
) -> tuple[str, list[list[QAPair]], list[list[CriticalMoment]],
           list[QAPair], list[CriticalMoment],
           list[int], list[int], float, list[dict]]:
    """Extract QA pairs and moments from trajectories and consolidate.

    Wraps process_trajectories with use_qa=True, use_moments=True.

    Returns: (episode_summaries, per_traj_qa, per_traj_moments,
              new_qa, new_moments, remove_qa_indices, remove_moment_indices, cost,
              extraction_records)
    """
    return await process_trajectories(
        config=config,
        beliefs=beliefs,
        perception=perception,
        output_dir=output_dir,
        existing_qa=existing_qa,
        existing_moments=existing_moments,
        default_knowledge=default_knowledge,
        step=step,
        use_qa=True,
        use_moments=True,
    )


# ============================================================
# Summary-based improvement
# ============================================================

async def improve_from_summaries(
    config: DictConfig,
    beliefs: str,
    perception: str,
    episode_summaries: str,
    sample_observations: list[tuple[str, int]] | None,
    default_knowledge: str,
) -> tuple[str, str, float, str, str]:
    """Improve beliefs and perception based on trajectory summaries and sampled observations.

    Presents the LLM with:
    - Current perception code and its output on sampled states
    - Current world knowledge and policy
    - All trajectory summaries

    Returns: (new_beliefs, new_perception, total_cost, prompt_used, llm_response)
    """
    if not episode_summaries:
        return beliefs, perception, 0.0, "", ""

    # Build sampled observation section
    obs_section = ""
    if sample_observations:
        obs_blocks = []
        for raw_obs, step_num in sample_observations:
            perc_out = _run_perception_on_observation(perception, raw_obs)
            obs_blocks.append(
                f"<perception_example step=\"{step_num}\">\n"
                f"<perception_input>\n{raw_obs}\n</perception_input>\n"
                f"<perception_output>\n{perc_out if perc_out else '(empty)'}\n</perception_output>\n"
                f"</perception_example>"
            )
        obs_section = (
            "\n<sampled_perception_examples>\n"
            + "\n\n".join(obs_blocks)
            + "\n</sampled_perception_examples>\n"
        )

    perception_instructions = """For the perception module:
- It must be a Python function `perceive(observation_text: str) -> str`.
- Input `observation_text` contains everything from the direct observation as a string.
- The code must be valid Python.
- Output should be a textual description of the environment state useful for decision-making.
- Output should contain all information necessary for progressing and be presented clearly."""

    base_prompt = f"""You are improving an agent's knowledge and perception based on trajectory summaries from its interactions with an environment.

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
=== TRAJECTORY SUMMARIES ===
{episode_summaries}
=== END TRAJECTORY SUMMARIES ===

Your task: Based on the trajectory summaries and observation samples, improve the agent's world knowledge, policy, and perception module.

Analyze:
1. What patterns of success or failure appear across trajectories?
2. Is the perception module extracting the right information from observations?
3. Are the world knowledge facts accurate based on observed evidence?
4. Is the policy leading to good decisions?

Guidelines:
- Keep each beliefs section to at most 8 concise bullet points. Merge redundant points.
- Balance safety with progress toward the objective.
- Remove beliefs that are contradicted by trajectory evidence.

{perception_instructions}

Format your response as:
<think>
Analyze the trajectory summaries and observation samples to determine what needs to change.
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
</updated_perception>"""

    return await _improve_with_perception_validation(
        config, beliefs, perception, base_prompt, sample_observations,
    )


# ============================================================
# QA-based improvement
# ============================================================

@dataclass
class QAForwardResult:
    qa_pair: QAPair
    predicted_answer: str   # "YES" or "NO"
    reasoning: str


@dataclass
class QAFeedbackResult:
    forward: QAForwardResult
    feedback: str           # explanation of why correct/incorrect
    verdict: str            # "CORRECT" | "INCORRECT" | "INCONCLUSIVE"


async def qa_forward_pass(
    config: DictConfig,
    beliefs: str,
    qa_pairs: list[QAPair],
    max_per_batch: int = 15,
) -> tuple[list[QAForwardResult], float, list[str], list[str]]:
    """Predict answers to QA questions based on current beliefs.

    Returns: (list[QAForwardResult], total_cost, prompts_used, responses)
    """
    if not qa_pairs:
        return [], 0.0, [], []

    # Split into batches
    batches = []
    for start in range(0, len(qa_pairs), max_per_batch):
        end = start + max_per_batch
        batches.append(qa_pairs[start:end])

    tasks = [_qa_forward_batch(config, beliefs, batch) for batch in batches]
    batch_results = await asyncio.gather(*tasks)

    all_results = []
    total_cost = 0.0
    all_prompts = []
    all_responses = []
    for results, cost, prompt, response in batch_results:
        all_results.extend(results)
        total_cost += cost
        all_prompts.append(prompt)
        all_responses.append(response)

    return all_results, total_cost, all_prompts, all_responses


async def _qa_forward_batch(
    config: DictConfig,
    beliefs: str,
    qa_pairs: list[QAPair],
) -> tuple[list[QAForwardResult], float, str, str]:
    """Predict answers for a batch of QA pairs.

    Returns: (results, cost, prompt_used, llm_response)
    """
    question_blocks = []
    for i, qa in enumerate(qa_pairs, 1):
        question_blocks.append(f"Q{i}: {qa.question}")
    questions_text = "\n".join(question_blocks)

    prompt = f"""You are an agent that has been interacting with an environment. Based on your current beliefs, answer each question below with YES or NO.

=== CURRENT BELIEFS ===
{beliefs if beliefs else "(empty - no beliefs yet)"}
=== END CURRENT BELIEFS ===

For each question, reason step-by-step about what your belief says (or doesn't say), then give your answer.

{questions_text}

For each question, respond in this format:
<Q n=<question number>>
<reasoning>Your step-by-step reasoning based on your knowledge</reasoning>
<answer>YES or NO</answer>
</Q n=<question number>>
"""

    text, cost = await _llm_call(config, prompt)

    results = []
    for i, qa in enumerate(qa_pairs, 1):
        m = re.search(rf"<Q\s+n={i}>\s*(.*?)\s*</Q\s+n={i}>", text, re.DOTALL)
        q_block = m.group(1) if m else None
        if q_block:
            reasoning = extract_xml_key(q_block, "reasoning") or ""
            answer = extract_xml_key(q_block, "answer") or "MISSING"
        else:
            reasoning = ""
            answer = "MISSING"

        # Normalize answer
        answer_upper = answer.strip().upper()
        if "YES" in answer_upper:
            predicted = "YES"
        elif "NO" in answer_upper:
            predicted = "NO"
        else:
            predicted = "MISSING"

        results.append(QAForwardResult(
            qa_pair=qa,
            predicted_answer=predicted,
            reasoning=reasoning.strip(),
        ))

    return results, cost, prompt, text


async def qa_get_feedback(
    config: DictConfig,
    qa_forward_results: list[QAForwardResult],
    max_per_batch: int = 15,
) -> tuple[list[QAFeedbackResult], float, list[str], list[str]]:
    """Get feedback on QA predictions by comparing to ground truth.

    Returns: (list[QAFeedbackResult], total_cost, prompts_used, responses)
    """
    if not qa_forward_results:
        return [], 0.0, [], []

    batches = []
    for start in range(0, len(qa_forward_results), max_per_batch):
        end = start + max_per_batch
        batches.append(qa_forward_results[start:end])

    tasks = [_qa_feedback_batch(config, batch) for batch in batches]
    batch_results = await asyncio.gather(*tasks)

    all_results = []
    total_cost = 0.0
    all_prompts = []
    all_responses = []
    for results, cost, prompt, response in batch_results:
        all_results.extend(results)
        total_cost += cost
        all_prompts.append(prompt)
        all_responses.append(response)

    return all_results, total_cost, all_prompts, all_responses


async def _qa_feedback_batch(
    config: DictConfig,
    qa_forward_results: list[QAForwardResult],
) -> tuple[list[QAFeedbackResult], float, str, str]:
    """Get feedback on a batch of QA predictions.

    Returns: (results, cost, prompt_used, llm_response)
    """
    item_blocks = []
    for i, fr in enumerate(qa_forward_results, 1):
        actual = "YES" if fr.qa_pair.answer else "NO"
        item_blocks.append(
            f"--- Question {i} ---\n"
            f"Question: {fr.qa_pair.question}\n"
            f"Agent's reasoning: {fr.reasoning}\n"
            f"Agent's predicted answer: {fr.predicted_answer}\n"
            f"Correct answer: {actual}\n"
            f"Evidence for correct answer: {fr.qa_pair.evidence}\n"
            f"---"
        )
    items_text = "\n\n".join(item_blocks)

    prompt = f"""You are evaluating whether an agent's predicted answers to questions about an environment are correct.

For each question below, you are given:
- The question about the environment
- The agent's reasoning and predicted answer
- The correct answer with supporting evidence

Evaluate whether the prediction is:
- CORRECT: the agent's answer matches the correct answer
- INCORRECT: the agent's answer contradicts the correct answer
- INCONCLUSIVE: the agent's prediction was MISSING or the question is ambiguous

For INCORRECT predictions, explain what knowledge the agent was missing or had wrong.

{items_text}

For each question Qn, respond with feedback Fn in this format:
<F n=<question number>>
<verdict>CORRECT or INCORRECT or INCONCLUSIVE</verdict>
<feedback>Explanation of what the agent's knowledge got right or wrong</feedback>
</F n=<question number>>"""

    text, cost = await _llm_call(config, prompt)

    results = []
    for i, fr in enumerate(qa_forward_results, 1):
        # The prompt requests tags like `<F n=1>...</F n=1>`, not `<F1>...</F1>`.
        # Parse that exact shape so feedback verdicts don't all fall back to
        # INCONCLUSIVE when the model follows instructions correctly.
        m = re.search(rf"<F\s+n={i}>\s*(.*?)\s*</F\s+n={i}>", text, re.DOTALL)
        f_block = m.group(1) if m else None
        if f_block:
            verdict_str = extract_xml_key(f_block, "verdict") or "INCONCLUSIVE"
            feedback_str = extract_xml_key(f_block, "feedback") or ""
        else:
            verdict_str = "INCONCLUSIVE"
            feedback_str = ""

        verdict_upper = verdict_str.strip().upper()
        if "CORRECT" in verdict_upper and "INCORRECT" not in verdict_upper:
            verdict = "CORRECT"
        elif "INCORRECT" in verdict_upper:
            verdict = "INCORRECT"
        else:
            verdict = "INCONCLUSIVE"

        results.append(QAFeedbackResult(
            forward=fr,
            feedback=feedback_str.strip(),
            verdict=verdict,
        ))

    return results, cost, prompt, text


async def improve_from_qa_feedback(
    config: DictConfig,
    beliefs: str,
    perception: str,
    qa_feedback_results: list[QAFeedbackResult],
    default_knowledge: str,
    episode_summaries: str,
    sample_observations: list[tuple[str, int]] | None = None,
) -> tuple[str, str, float, str, str]:
    """Improve beliefs and perception based on QA feedback.

    Returns: (new_beliefs, new_perception, total_cost, prompt_used, llm_response)
    """
    if not qa_feedback_results:
        return beliefs, perception, 0.0, "", ""

    # Build QA feedback items
    qa_blocks = []
    for i, fr in enumerate(qa_feedback_results, 1):
        actual = "YES" if fr.forward.qa_pair.answer else "NO"
        qa_blocks.append(
            f"--- QA Feedback {i} ---\n"
            f"Question: {fr.forward.qa_pair.question}\n"
            f"Correct answer: {actual}\n"
            f"Evidence: {fr.forward.qa_pair.evidence}\n"
            f"Agent's predicted answer: {fr.forward.predicted_answer}\n"
            f"Agent's reasoning: {fr.forward.reasoning}\n"
            f"Verdict: {fr.verdict}\n"
            f"Feedback: {fr.feedback}\n"
            f"---"
        )
    qa_text = "\n\n".join(qa_blocks)

    num_correct = sum(1 for fr in qa_feedback_results if fr.verdict == "CORRECT")
    num_incorrect = sum(1 for fr in qa_feedback_results if fr.verdict == "INCORRECT")

    # Build execution report for current perception
    execution_report_section = ""
    if sample_observations:
        exec_report = _build_execution_report(perception, sample_observations)
        if exec_report:
            execution_report_section = f"""
=== CURRENT PERCEPTION EXECUTION OUTPUT ===
{exec_report}
=== END CURRENT PERCEPTION EXECUTION OUTPUT ===
"""

    perception_instructions = """For the perception module:
- It must be a Python function `perceive(observation_text: str) -> str`.
- Input `observation_text` contains everything from the direct observation as a string.
- The code must be valid Python.
- Output should be a textual description of the environment state useful for decision-making.
- Output should contain all information necessary for progressing and be presented clearly."""

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

{qa_text}

Your task: Based on the QA feedback, improve the agent's world knowledge, policy, and perception module.

For INCORRECT predictions, analyze:
1. Was the agent's world knowledge missing the relevant fact? If so, add it.
2. Was the agent's world knowledge wrong? If so, correct it.
3. Does the perception module need to extract different information to support this knowledge? If so, update it.

Guidelines:
- Keep each beliefs section to at most 8 concise bullet points. Merge redundant points.
- Balance safety with progress toward the objective.
- Remove beliefs that are contradicted by the QA evidence.

{perception_instructions}

Format your response as:
<think>
Analyze the QA feedback patterns and determine what needs to change.
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
</updated_perception>"""

    return await _improve_with_perception_validation(
        config, beliefs, perception, base_prompt, sample_observations,
    )


# ============================================================
# Shared perception validation helper
# ============================================================

async def _improve_with_perception_validation(
    config: DictConfig,
    beliefs: str,
    perception: str,
    base_prompt: str,
    sample_observations: list[tuple[str, int]] | None = None,
    max_retries: int = 3,
) -> tuple[str, str, float, str, str]:
    """Run an improve prompt with perception validation and retry.

    Shared logic for improve_from_summaries, improve_from_qa_feedback, etc.

    Returns: (new_beliefs, new_perception, total_cost, prompt_used, llm_response)
    """
    perception_error = None
    new_beliefs = beliefs
    new_perception = perception
    total_cost = 0.0

    for attempt in range(max_retries):
        if perception_error:
            prompt = f"""{base_prompt}

=== PERCEPTION CODE ERROR (RETRY {attempt}/{max_retries}) ===
Your previous perception code had an error:
{perception_error}

Please fix the error in your perception code.
=== END OF PERCEPTION CODE ERROR ===
"""
        else:
            prompt = base_prompt

        text, cost = await _llm_call(config, prompt)
        total_cost += cost

        # Extract beliefs
        updated_beliefs = extract_xml_key(text, "updated_beliefs")
        if updated_beliefs:
            new_beliefs = updated_beliefs.strip()

        # Extract perception code
        candidate_perception = extract_xml_key(text, "updated_perception")
        if not candidate_perception:
            logging.warning(f"Failed to extract updated_perception (attempt {attempt + 1})")
            perception_error = "No <updated_perception> block found in response."
            continue

        candidate_perception = candidate_perception.strip()
        if candidate_perception.startswith("```python"):
            candidate_perception = candidate_perception[len("```python"):].strip()
        elif candidate_perception.startswith("```"):
            candidate_perception = candidate_perception[len("```"):].strip()
        if candidate_perception.endswith("```"):
            candidate_perception = candidate_perception[:-len("```")].strip()

        # Validate perception code
        test_obs = [raw for raw, _ in sample_observations[:3]] if sample_observations else None
        is_valid, error_msg = validate_perception_code(candidate_perception, test_observations=test_obs)
        if not is_valid:
            perception_error = error_msg
            logging.warning(f"Perception code validation failed (attempt {attempt + 1}/{max_retries}): {error_msg}")
            continue

        # Execution verification: check for output degradation
        if sample_observations:
            degraded_samples = []
            for raw_obs, step_num in sample_observations[:3]:
                new_out = _run_perception_on_observation(candidate_perception, raw_obs)
                raw_has_map = "map:" in raw_obs and len(raw_obs) > 500
                new_line_count = len(new_out.strip().splitlines()) if new_out else 0
                if raw_has_map and new_line_count <= 3:
                    degraded_samples.append(
                        f"Sample step {step_num}: raw observation has {len(raw_obs)} chars "
                        f"but perception output is only {new_line_count} line(s):\n"
                        f"{new_out if new_out else '(empty)'}"
                    )

            if degraded_samples:
                new_report = _build_execution_report(candidate_perception, sample_observations)
                perception_error = (
                    f"Perception output appears degraded on real observations:\n\n"
                    + "\n\n".join(degraded_samples) + "\n\n"
                    f"Full execution I/O:\n{new_report}\n\n"
                    f"Please fix the perception code to correctly parse these observations."
                )
                logging.warning(f"Perception degradation detected (attempt {attempt + 1}/{max_retries})")
                continue

        new_perception = candidate_perception
        logging.info(f"Perception code validated successfully on attempt {attempt + 1}")
        return new_beliefs, new_perception, total_cost, base_prompt, text

    logging.error(f"All {max_retries} attempts to generate valid perception failed. Keeping previous perception.")
    return new_beliefs, perception, total_cost, base_prompt, text


async def _improve_with_perception_validation_conversational(
    config: DictConfig,
    beliefs: str,
    perception: str,
    conversation_history: list[dict],
    user_message: str,
    sample_observations: list[tuple[str, int]] | None = None,
    max_retries: int = 3,
    images: list | None = None,
) -> tuple[str, str, float, list[dict], str]:
    """Run an improve prompt with perception validation using multi-turn conversation.

    Like _improve_with_perception_validation but maintains conversation history.
    Perception validation retries are appended as additional turns.

    If ``images`` is provided, they are attached to the initial user_message only.
    Retry turns (perception errors) are text-only; the images remain visible via
    the conversation history.

    Returns: (new_beliefs, new_perception, total_cost, updated_history, llm_response)
    """
    perception_error = None
    new_beliefs = beliefs
    new_perception = perception
    total_cost = 0.0
    current_history = conversation_history
    current_message = user_message
    text = ""
    initial_images = images

    for attempt in range(max_retries):
        if perception_error:
            current_message = (
                f"Your previous perception code had an error:\n"
                f"{perception_error}\n\n"
                f"Please fix the error in your perception code and provide "
                f"updated beliefs and perception in the same format as before."
            )

        # Only attach images on the first attempt; retries are text-only.
        turn_images = initial_images if attempt == 0 else None

        text, cost, current_history = await _llm_call_conversational(
            config, current_history, current_message, images=turn_images,
        )
        total_cost += cost

        # Extract beliefs
        updated_beliefs = extract_xml_key(text, "updated_beliefs")
        if updated_beliefs:
            new_beliefs = updated_beliefs.strip()

        # Extract perception code
        candidate_perception = extract_xml_key(text, "updated_perception")
        if not candidate_perception:
            logging.warning(f"Failed to extract updated_perception (attempt {attempt + 1})")
            perception_error = "No <updated_perception> block found in response."
            continue

        candidate_perception = candidate_perception.strip()
        if candidate_perception.startswith("```python"):
            candidate_perception = candidate_perception[len("```python"):].strip()
        elif candidate_perception.startswith("```"):
            candidate_perception = candidate_perception[len("```"):].strip()
        if candidate_perception.endswith("```"):
            candidate_perception = candidate_perception[:-len("```")].strip()

        # Validate perception code
        test_obs = [raw for raw, _ in sample_observations[:3]] if sample_observations else None
        is_valid, error_msg = validate_perception_code(candidate_perception, test_observations=test_obs)
        if not is_valid:
            perception_error = error_msg
            logging.warning(f"Perception code validation failed (attempt {attempt + 1}/{max_retries}): {error_msg}")
            continue

        # Execution verification: check for output degradation
        if sample_observations:
            degraded_samples = []
            for raw_obs, step_num in sample_observations[:3]:
                new_out = _run_perception_on_observation(candidate_perception, raw_obs)
                raw_has_map = "map:" in raw_obs and len(raw_obs) > 500
                new_line_count = len(new_out.strip().splitlines()) if new_out else 0
                if raw_has_map and new_line_count <= 3:
                    degraded_samples.append(
                        f"Sample step {step_num}: raw observation has {len(raw_obs)} chars "
                        f"but perception output is only {new_line_count} line(s):\n"
                        f"{new_out if new_out else '(empty)'}"
                    )

            if degraded_samples:
                new_report = _build_execution_report(candidate_perception, sample_observations)
                perception_error = (
                    f"Perception output appears degraded on real observations:\n\n"
                    + "\n\n".join(degraded_samples) + "\n\n"
                    f"Full execution I/O:\n{new_report}\n\n"
                    f"Please fix the perception code to correctly parse these observations."
                )
                logging.warning(f"Perception degradation detected (attempt {attempt + 1}/{max_retries})")
                continue

        new_perception = candidate_perception
        logging.info(f"Perception code validated successfully on attempt {attempt + 1}")
        return new_beliefs, new_perception, total_cost, current_history, text

    logging.error(f"All {max_retries} attempts to generate valid perception failed. Keeping previous perception.")
    return new_beliefs, perception, total_cost, current_history, text


async def _improve_beliefs_only_conversational(
    config: DictConfig,
    beliefs: str,
    conversation_history: list[dict],
    user_message: str,
    images: list | None = None,
) -> tuple[str, float, list[dict], str]:
    """Run a beliefs-only improve prompt using multi-turn conversation.

    Only extracts <updated_beliefs>; no perception extraction or validation.
    Optional ``images`` are attached to the new user_message turn.

    Returns: (new_beliefs, total_cost, updated_history, llm_response)
    """
    text, cost, current_history = await _llm_call_conversational(
        config, conversation_history, user_message, images=images,
    )

    updated_beliefs = extract_xml_key(text, "updated_beliefs")
    new_beliefs = updated_beliefs.strip() if updated_beliefs else beliefs

    return new_beliefs, cost, current_history, text


def serialize_qa_feedback_results(results: list[QAFeedbackResult]) -> list[dict]:
    """Serialize QA feedback results for JSON storage."""
    serialized = []
    for fr in results:
        serialized.append({
            "question": fr.forward.qa_pair.question,
            "correct_answer": "YES" if fr.forward.qa_pair.answer else "NO",
            "evidence": fr.forward.qa_pair.evidence,
            "predicted_answer": fr.forward.predicted_answer,
            "reasoning": fr.forward.reasoning,
            "verdict": fr.verdict,
            "feedback": fr.feedback,
        })
    return serialized
