"""LLM prompt functions for mixed explore: Q&A extraction, scoring, and belief improvement."""

import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import litellm
from omegaconf import DictConfig


# ---------------------------------------------------------------------------
# Mock mode: when enabled, _llm_call / _llm_call_conversational skip the real
# LLM API call and return a canned response sniffed from the prompt's format
# instructions.  Prompts are constructed normally, so callers (and logs) see
# the exact text that would have been sent.
# ---------------------------------------------------------------------------

_MOCK_MODE = False


def set_mock_mode(enabled: bool) -> None:
    """Enable or disable mock mode globally (affects all _llm_call* calls)."""
    global _MOCK_MODE
    _MOCK_MODE = enabled


def is_mock_mode() -> bool:
    return _MOCK_MODE


def _mock_llm_response(prompt: str) -> str:
    """Return a plausible LLM response for the given prompt by sniffing format hints.

    Used only when _MOCK_MODE is True.  The goal is to produce output that the
    downstream parsers accept so the full pipeline exercises its control flow.
    """
    p = prompt

    # --- Question generation (stepwise_eb_learn_improve.generate_questions_from_steps) ---
    if "generating questions to guide an agent" in p or "Generate new binary" in p:
        n = random.randint(2, 4)
        lines = [f"QUESTION {i + 1}: [mock] Generated question {i + 1} at random?" for i in range(n)]
        return (
            "<think>[mock] reasoning about what to ask next</think>\n"
            "<questions>\n" + "\n".join(lines) + "\n</questions>"
        )

    # --- Experiment formulation (formulate_experiment_from_question) ---
    if "designing the next experiment" in p:
        if random.random() < 0.3:
            return "<think>[mock] keep current experiment</think>\n<experiment>null</experiment>"
        # Pick an existing question number from the prompt if possible
        q_numbers = re.findall(r"^Q(\d+):", p, re.MULTILINE)
        q_idx = random.choice(q_numbers) if q_numbers else "1"
        return (
            "<think>[mock] propose new experiment</think>\n"
            "<experiment>\n"
            f"<question_index>{q_idx}</question_index>\n"
            "<experiment_plan>[mock] Try a candidate action and observe the outcome.</experiment_plan>\n"
            "</experiment>"
        )

    # --- Update QA from trajectory (update_qa_from_trajectory) ---
    if "<updated_questions>" in p:
        qs = re.findall(r"^Q(\d+):\s*(.+?)(?:\s*->|\s*$)", p, re.MULTILINE)
        inner_blocks = []
        for idx_str, q_text in qs[:20]:
            # Drop the status tail if present
            q_text = re.split(r"\s*->\s*", q_text, maxsplit=1)[0].strip()
            status = random.choice(["YES", "NO", "UNANSWERED"])
            ev = "[mock] evidence from trajectory" if status != "UNANSWERED" else ""
            inner_blocks.append(
                f'<q n="{idx_str}">\n'
                f"<question>{q_text}</question>\n"
                f"<evidence>{ev}</evidence>\n"
                f"<answer>{status}</answer>\n"
                "</q>"
            )
        if not inner_blocks or random.random() < 0.3:
            new_idx = len(qs) + 1
            inner_blocks.append(
                f'<q n="{new_idx}">\n'
                f"<question>[mock] newly discovered question {new_idx}?</question>\n"
                f"<evidence></evidence>\n"
                f"<answer>UNANSWERED</answer>\n"
                "</q>"
            )
        return (
            "<think>[mock] reviewed trajectory</think>\n"
            "<updated_questions>\n" + "\n".join(inner_blocks) + "\n</updated_questions>"
        )

    # --- Trim QA (trim_qa_pairs) ---
    if "<trimmed_questions>" in p:
        qs = re.findall(r"^Q(\d+):\s*(.+?)(?:\s*->|\s*$)", p, re.MULTILINE)
        max_match = re.search(r"trim it to at most (\d+)", p)
        max_total = int(max_match.group(1)) if max_match else len(qs)
        kept = random.sample(qs, min(max_total, len(qs))) if qs else []
        inner_blocks = []
        for idx_str, q_text in kept:
            q_text = re.split(r"\s*->\s*", q_text, maxsplit=1)[0].strip()
            status = random.choice(["YES", "NO", "UNANSWERED"])
            ev = "[mock] kept evidence" if status != "UNANSWERED" else ""
            inner_blocks.append(
                f'<q n="{idx_str}">\n'
                f"<question>{q_text}</question>\n"
                f"<answer>{status}</answer>\n"
                f"<evidence>{ev}</evidence>\n"
                "</q>"
            )
        if not inner_blocks:
            inner_blocks.append(
                '<q n="1">\n<question>[mock] placeholder</question>\n'
                '<answer>UNANSWERED</answer>\n<evidence></evidence>\n</q>'
            )
        return (
            "<think>[mock] trimming</think>\n"
            "<trimmed_questions>\n" + "\n".join(inner_blocks) + "\n</trimmed_questions>"
        )

    # --- QA forward pass (_qa_forward_batch) ---
    if "answer each question below with YES or NO" in p:
        nums = re.findall(r"^Q(\d+):", p, re.MULTILINE)
        blocks = []
        for n in nums:
            ans = random.choice(["YES", "NO"])
            blocks.append(
                f"<Q n={n}>\n<reasoning>[mock] reasoning for Q{n}</reasoning>\n<answer>{ans}</answer>\n</Q n={n}>"
            )
        return "\n".join(blocks) if blocks else "<Q n=1>\n<reasoning>[mock]</reasoning>\n<answer>YES</answer>\n</Q n=1>"

    # --- QA feedback (_qa_feedback_batch) ---
    if "evaluating whether an agent's predicted answers" in p:
        nums = re.findall(r"--- Question (\d+) ---", p)
        blocks = []
        for n in nums:
            v = random.choices(
                ["CORRECT", "INCORRECT", "INCONCLUSIVE"], weights=[5, 4, 1]
            )[0]
            blocks.append(
                f"<F n={n}>\n<verdict>{v}</verdict>\n<feedback>[mock] feedback for Q{n}</feedback>\n</F n={n}>"
            )
        return "\n".join(blocks) if blocks else "<F n=1>\n<verdict>INCONCLUSIVE</verdict>\n<feedback>[mock]</feedback>\n</F n=1>"

    # --- Perception error / retry message ---
    if "previous perception code had an error" in p:
        return _mock_improve_response(p, include_beliefs=False, include_perception=True)

    # --- Improve prompts (conversational) ---
    has_beliefs_fmt = "<updated_beliefs>" in p
    has_perc_fmt = "<updated_perception>" in p
    if has_beliefs_fmt or has_perc_fmt:
        return _mock_improve_response(p, include_beliefs=has_beliefs_fmt, include_perception=has_perc_fmt)

    # --- Fallback ---
    return "<think>[mock] response for unrecognized prompt</think>"


def _mock_improve_response(prompt: str, include_beliefs: bool, include_perception: bool) -> str:
    """Build an improve-style response with optional beliefs/perception blocks and a status tag."""
    parts = ["<think>[mock] improve analysis</think>"]
    # Some improve prompts (beliefs-only Track 1a) expect a <perception_analysis> sibling
    if "<perception_analysis>" in prompt:
        parts.append("<perception_analysis>[mock] perception analysis notes</perception_analysis>")
    if include_beliefs:
        parts.append(
            "<updated_beliefs>\n"
            "<world_knowledge>\n"
            f"- [mock] world fact {random.randint(1000, 9999)}\n"
            "- [mock] additional environment mechanic observed\n"
            "</world_knowledge>\n"
            "<policy>\n"
            f"- [mock] policy item {random.randint(1000, 9999)}\n"
            "</policy>\n"
            "</updated_beliefs>"
        )
    if include_perception:
        parts.append(
            "<updated_perception>\n"
            "```python\n"
            "def perceive(observation_history: list[str]) -> str:\n"
            f"    # [mock] perception v{random.randint(100, 999)}\n"
            "    if not observation_history:\n"
            "        return \"\"\n"
            "    obs = observation_history[-1]\n"
            "    return obs[:2000]\n"
            "```\n"
            "</updated_perception>"
        )
    # Only improve prompts with a <status> tag in their format request it
    if "<status>" in prompt:
        status = "SUBMIT" if random.random() < 0.4 else "CONTINUE"
        parts.append(f"<status>{status}</status>")
    return "\n".join(parts)

from improve import (
    _get_response_cost,
    get_episode_outcome_header,
    trim_to_model_context_lim,
    _get_beliefs_perception_summary_async,
    extract_obs_perc_examples,
    validate_perception_code,
)
from llm_utils import (
    build_llm_input,
    build_llm_input_multiturn,
    append_assistant_message,
    extract_llm_response_text,
    extract_xml_key,
)


def _normalize_action(action: str) -> str:
    """Normalize an action string for comparison."""
    return action.lower().strip()


def _action_in_set(predicted: str, action_set: list[str]) -> bool:
    """Check if predicted action matches any action in a set (case-insensitive, substring-aware)."""
    p = _normalize_action(predicted)
    if not p:
        return False
    for a in action_set:
        a_norm = _normalize_action(a)
        if not a_norm:
            continue
        if p == a_norm or p in a_norm or a_norm in p:
            return True
    return False


@dataclass
class QAPair:
    question: str
    answer: bool  # True=YES, False=NO
    evidence: str
    source_step: int


@dataclass
class CriticalMoment:
    state: str
    goal: str
    good_actions: list[str]  # Actions the agent SHOULD take (evidence-backed)
    bad_actions: list[str]   # Actions the agent should NOT take (evidence-backed)
    evidence: str
    source_step: int
    raw_observation: str = ""  # Full raw observation at this game step
    traj_step_number: int = -1  # Step number in trajectory CSV (for lookup, not serialized to scoring)


def _get_model_name(config: DictConfig) -> str:
    if config.client.client_name == "vllm":
        return f"hosted_vllm/{config.client.model_id}"
    return f"{config.client.client_name}/{config.client.model_id}"


async def _llm_call(
    config: DictConfig,
    prompt: str,
    images: list | None = None,
) -> tuple[str, float]:
    """Make an async LLM call and return (response_text, cost).

    Optional ``images`` are attached to the user turn as input_image parts.
    When mock mode is enabled, prompts are still fully constructed and logged
    but the response is synthesized locally with zero cost.
    """
    num_imgs = sum(1 for i in (images or []) if i is not None)
    if _MOCK_MODE:
        logging.info(f"LLM prompt [MOCK] (images={num_imgs}):\n{prompt}")
        text = _mock_llm_response(prompt)
        logging.info(f"LLM response [MOCK]:\n{text}")
        return text, 0.0

    input_data = build_llm_input(prompt, images=images)
    model_name = _get_model_name(config)
    logging.info(f"LLM prompt (images={num_imgs}):\n{prompt}")
    response = await asyncio.to_thread(
        litellm.responses,
        model=model_name,
        input=input_data,
        num_retries=5,
    )
    text = extract_llm_response_text(response)
    logging.info(f"LLM response:\n{text}")
    cost = _get_response_cost(response, config.client.model_id)
    return text, cost


async def _llm_call_conversational(
    config: DictConfig,
    conversation_history: list[dict],
    user_message: str,
    images: list | None = None,
) -> tuple[str, float, list[dict]]:
    """Multi-turn LLM call that maintains conversation history.

    Appends user_message (with optional images) to history, calls the LLM,
    appends assistant reply.  In mock mode, the prompt is still built and
    logged, and the conversation history is still extended with both the user
    message and a synthesized assistant reply.

    Returns: (response_text, cost, updated_conversation_history)
    """
    num_imgs = sum(1 for i in (images or []) if i is not None)

    if _MOCK_MODE:
        logging.info(
            f"LLM conversational prompt [MOCK] (turn {len(conversation_history) + 1}, images={num_imgs}):\n{user_message}"
        )
        text = _mock_llm_response(user_message)
        logging.info(f"LLM conversational response [MOCK]:\n{text}")
        updated_history = build_llm_input_multiturn(conversation_history, user_message, images=images)
        updated_history = append_assistant_message(updated_history, text)
        return text, 0.0, updated_history

    input_data = build_llm_input_multiturn(conversation_history, user_message, images=images)
    model_name = _get_model_name(config)
    logging.info(f"LLM conversational prompt (turn {len(input_data)}, images={num_imgs}):\n{user_message}")
    response = await asyncio.to_thread(
        litellm.responses,
        model=model_name,
        input=input_data,
        num_retries=5,
    )
    text = extract_llm_response_text(response)
    logging.info(f"LLM conversational response:\n{text}")
    cost = _get_response_cost(response, config.client.model_id)
    # Build updated history with the user message (w/ images) and assistant response
    updated_history = build_llm_input_multiturn(conversation_history, user_message, images=images)
    updated_history = append_assistant_message(updated_history, text)
    return text, cost, updated_history


def _parse_qa_pairs(text: str, step: int) -> list[QAPair]:
    """Parse Q&A pairs from XML block."""
    qa_block = extract_xml_key(text, "qa_pairs")
    if not qa_block:
        return []

    pairs = []
    # Parse individual qa entries: <qa> blocks
    for qa_match in re.finditer(r"<qa>(.*?)</qa>", qa_block, re.DOTALL):
        qa_text = qa_match.group(1)
        question = extract_xml_key(qa_text, "question")
        answer_str = extract_xml_key(qa_text, "answer")
        evidence = extract_xml_key(qa_text, "evidence")
        if question and answer_str and evidence:
            answer = answer_str.strip().upper() == "YES"
            pairs.append(QAPair(
                question=question.strip(),
                answer=answer,
                evidence=evidence.strip(),
                source_step=step,
            ))
    return pairs


def _parse_action_list(text: str) -> list[str]:
    """Parse a comma-separated list of actions, filtering out empty/descriptive ones."""
    actions = []
    for a in text.split(","):
        a = a.strip()
        if a and len(a) <= 50 and '\n' not in a and a.upper() != "NONE":
            actions.append(a)
    return actions


def _parse_critical_moments(text: str, step: int) -> list[CriticalMoment]:
    """Parse critical moments from XML block.

    Returns moments with step_number stored temporarily in state field prefix
    for later raw_observation lookup. The step_number is extracted separately.
    """
    moments_block = extract_xml_key(text, "critical_moments")
    if not moments_block:
        return []

    moments = []
    for m_match in re.finditer(r"<moment>(.*?)</moment>", moments_block, re.DOTALL):
        m_text = m_match.group(1)
        step_number_str = extract_xml_key(m_text, "step_number")
        state = extract_xml_key(m_text, "state")
        goal = extract_xml_key(m_text, "goal")
        good_actions_str = extract_xml_key(m_text, "good_actions") or ""
        bad_actions_str = extract_xml_key(m_text, "bad_actions") or ""
        evidence = extract_xml_key(m_text, "evidence")
        if state and goal and evidence:
            good_actions = _parse_action_list(good_actions_str)
            bad_actions = _parse_action_list(bad_actions_str)
            # Skip moments with no actionable information
            if not good_actions and not bad_actions:
                continue
            # Parse trajectory step number (for raw observation lookup)
            traj_step_number = -1
            if step_number_str:
                try:
                    traj_step_number = int(step_number_str.strip())
                except ValueError:
                    pass
            moments.append(CriticalMoment(
                state=state.strip(),
                goal=goal.strip(),
                good_actions=good_actions,
                bad_actions=bad_actions,
                evidence=evidence.strip(),
                source_step=step,
                raw_observation="",  # Populated later by _attach_raw_observations
                traj_step_number=traj_step_number,
            ))
    return moments


def _read_csv_observations(trajectory_path: str) -> dict[int, str]:
    """Read a trajectory CSV and return a map of step_number -> raw observation text."""
    import csv as csv_module
    obs_map = {}
    try:
        with open(trajectory_path, newline='') as f:
            reader = csv_module.DictReader(f, escapechar="\u02d8", quoting=csv_module.QUOTE_MINIMAL)
            for row in reader:
                try:
                    step_num = int(row.get('Step', -1))
                    obs_map[step_num] = row.get('Observation', '')
                except (ValueError, TypeError):
                    continue
    except Exception as e:
        logging.warning(f"Failed to read CSV for observation lookup: {trajectory_path}: {e}")
    return obs_map


def _attach_raw_observations(moments: list[CriticalMoment], trajectory_path: str) -> None:
    """Attach raw observations from the trajectory CSV to moments that have step numbers."""
    moments_needing_obs = [m for m in moments if m.traj_step_number >= 0 and not m.raw_observation]
    if not moments_needing_obs:
        return

    obs_map = _read_csv_observations(trajectory_path)
    for m in moments_needing_obs:
        if m.traj_step_number in obs_map:
            m.raw_observation = obs_map[m.traj_step_number]
        else:
            logging.warning(f"Step {m.traj_step_number} not found in {trajectory_path}")


def extract_perception_input(observation: str) -> str:
    """Extract just the Direct Observation section from a CSV observation string.

    The CSV Observation column may contain:
    [Perception Module output] + [Auxiliary Observation] + [Direct Observation]

    The perception function (perceive()) only receives the Direct Observation
    content (obs["text"]["long_term_context"]). This function extracts that section.

    If no Direct Observation markers are found (e.g. step 1 with no perception),
    returns the full observation unchanged.
    """
    start_marker = "Start of Direct Observation"
    end_marker = "End of Direct Observation"
    start_idx = observation.find(start_marker)
    if start_idx == -1:
        return observation
    # Skip past the marker line (find the newline after the marker)
    content_start = observation.find("\n", start_idx)
    if content_start == -1:
        return observation
    content_start += 1  # skip the newline
    end_idx = observation.find(end_marker, content_start)
    if end_idx == -1:
        return observation[content_start:]
    # Walk back to strip the "========== " prefix line
    line_start = observation.rfind("\n", content_start, end_idx)
    if line_start == -1:
        return observation[content_start:end_idx].strip()
    return observation[content_start:line_start].strip()


def _run_perception_on_observation(perception: str, raw_observation: str) -> str:
    """Run the perception module on a raw observation and return its output.

    Returns empty string if perception is empty or execution fails.
    """
    if not perception or not perception.strip() or not raw_observation:
        return ""
    try:
        namespace = {}
        exec(perception, namespace)
        if "perceive" in namespace and callable(namespace["perceive"]):
            result = namespace["perceive"](raw_observation)
            return result if isinstance(result, str) else ""
    except Exception as e:
        logging.warning(f"Perception execution failed: {e}")
    return ""


def _run_perception_on_history(
    perception: str,
    history: list,
    window: int,
) -> str:
    """Run perception on an observation history, auto-detecting signature.

    If perceive expects a list, passes the last `window` items. Legacy single-arg
    perceive gets history[-1]. Returns empty string on failure or empty history.
    """
    from stepwise_explore import _perceive_signature_mode
    if not perception or not perception.strip() or not history:
        return ""
    try:
        namespace = {}
        exec(perception, namespace)
        fn = namespace.get("perceive")
        if not callable(fn):
            return ""
        mode = _perceive_signature_mode(fn)
        if mode == "history":
            windowed = history[-window:] if window and window > 0 else list(history)
            result = fn(windowed)
        else:
            result = fn(history[-1])
        return result if isinstance(result, str) else ""
    except Exception as e:
        logging.warning(f"Perception execution failed: {e}")
    return ""


async def score_perception_on_moments(
    config: DictConfig,
    moments: list[CriticalMoment],
    perception: str,
    beliefs: str = "",
    max_moments: int = 10,
) -> tuple[str, float, float, dict]:
    """Test whether perception output alone is sufficient for correct action prediction.

    For each qualifying moment, runs the perception module on the raw observation,
    then asks the LLM to predict good/bad actions from perception output + goal only
    (without seeing the answers). Scores predictions against ground truth and formats
    failures as targeted feedback for improve_perception.

    Returns: (feedback_string, moment_score, cost, debug_info)
    """
    # Filter to moments with non-empty raw_observation
    qualifying = [m for m in moments if m.raw_observation and m.raw_observation.strip()]
    if not qualifying:
        return "", 1.0, 0.0, {}

    # Take evenly spaced moments if more than max_moments
    if len(qualifying) > max_moments:
        indices = [int(i * (len(qualifying) - 1) / (max_moments - 1)) for i in range(max_moments)]
        qualifying = [qualifying[i] for i in indices]

    # Run perception on each moment's raw observation
    perc_outputs = []
    for m in qualifying:
        perc_output = _run_perception_on_observation(perception, m.raw_observation)
        perc_outputs.append(perc_output)

    # Build prediction prompt
    moment_blocks = []
    for i, (m, perc_output) in enumerate(zip(qualifying, perc_outputs), 1):
        moment_blocks.append(
            f"M{i}: Goal: {m.goal}\n"
            f"Perception output: {perc_output if perc_output else '(empty — perception produced no output)'}"
        )

    moments_text = "\n\n".join(moment_blocks)
    response_format = "\n".join(f"M{i}_do: <action>\nM{i}_avoid: <action>" for i in range(1, len(qualifying) + 1))

    prompt = f"""We are interacting with an environment. Here is what we know about it:
=== CURRENT BELIEFS ===
{beliefs if beliefs else "(empty - no beliefs yet)"}
=== END CURRENT BELIEFS ===

For each moment below, you are given the goal and what the perception module
showed the agent. Based on the perception output and your knowledge of the
environment, predict:
- What single action command the agent should take
- What single action command the agent should avoid

{moments_text}

Respond in this exact format (one action per line, no explanations):
{response_format}"""

    text, cost = await _llm_call(config, prompt)

    # Parse predictions and score against ground truth
    moment_total_score = 0.0
    failures = []
    num_correct = 0
    n = len(qualifying)

    for i, (m, perc_output) in enumerate(zip(qualifying, perc_outputs), 1):
        # Parse predicted do/avoid actions
        do_match = re.search(rf"M{i}_do:\s*(.+)", text)
        avoid_match = re.search(rf"M{i}_avoid:\s*(.+)", text)
        predicted_do = do_match.group(1).strip() if do_match else "MISSING"
        predicted_avoid = avoid_match.group(1).strip() if avoid_match else "MISSING"

        # Score do prediction: +1 if in good_actions, -1 if in bad_actions, 0 otherwise
        do_in_good = _action_in_set(predicted_do, m.good_actions)
        do_in_bad = _action_in_set(predicted_do, m.bad_actions)
        if do_in_good:
            do_score = 1
        elif do_in_bad:
            do_score = -1
        else:
            do_score = 0

        # Score avoid prediction: +1 if in bad_actions (correctly avoided), -1 if in good_actions, 0 otherwise
        avoid_in_bad = _action_in_set(predicted_avoid, m.bad_actions)
        avoid_in_good = _action_in_set(predicted_avoid, m.good_actions)
        if avoid_in_bad:
            avoid_score = 1
        elif avoid_in_good:
            avoid_score = -1
        else:
            avoid_score = 0

        score_val = do_score + avoid_score
        moment_total_score += score_val

        is_correct = do_score > 0 and avoid_score >= 0
        if is_correct:
            num_correct += 1

        # Collect failures for feedback
        if not is_correct:
            good = ", ".join(m.good_actions) if m.good_actions else "NONE"
            bad = ", ".join(m.bad_actions) if m.bad_actions else "NONE"
            failures.append(
                f"--- Failed Moment {i} (step {m.traj_step_number}) ---\n"
                f"Goal: {m.goal}\n"
                f"Perception output: {perc_output if perc_output else '(empty)'}\n"
                f"Your predicted action: {predicted_do} | Correct: {good}\n"
                f"Your predicted avoid: {predicted_avoid} | Correct: {bad}\n"
                f"Evidence: {m.evidence}\n"
                f"Raw observation (ground truth):\n{m.raw_observation}"
            )

    # Normalize score to [0, 1]: map from [-2n, 2n] to [0, 1]
    moment_score = (moment_total_score + 2 * n) / (4 * n)

    # Build feedback string
    summary = f"Perception moment score: {num_correct}/{n} correct ({num_correct/n:.0%})"
    logging.info(f"Perception scoring: {summary}")

    if failures:
        feedback = summary + "\n\n" + "\n\n".join(failures)
    else:
        feedback = summary

    debug_info = {
        "scoring_prompt": prompt,
        "scoring_response": text,
        "num_moments": n,
        "num_correct": num_correct,
        "feedback": feedback,
    }

    return feedback, moment_score, cost, debug_info


async def extract_qa_and_moments_from_trajectory(
    config: DictConfig,
    traj_text: str,
    trajectory_path: str,
    step: int,
    use_qa: bool,
    use_moments: bool,
    existing_qa: list[QAPair] | None = None,
) -> tuple[list[QAPair], list[CriticalMoment], float, str, str]:
    """Extract Q&A pairs and critical moments from a single raw trajectory.

    Returns: (qa_pairs, moments, cost, prompt, response)
    """
    if not use_qa and not use_moments:
        return [], [], 0.0, "", ""

    extraction_sections = []
    if use_qa:
        extraction_sections.append("""1. Q&A Pairs: Yes/no questions about general world mechanics or rules, with answers supported by direct evidence from this trajectory.
   - The question should be about how the WORLD works — environment mechanics, rules, cause-and-effect relationships — not about what the agent specifically did in this episode.
   - The evidence must quote or closely paraphrase the actual observations from the trajectory that support the answer.
   - Only include pairs where this trajectory provides clear, unambiguous evidence for the answer.
   Format each as:
   <qa>
   <question>A yes/no question about a general world mechanic or rule</question>
   <answer>YES or NO</answer>
   <evidence>Quote or closely paraphrase the specific observations from the trajectory that support this answer</evidence>
   </qa>""")
    if use_moments:
        extraction_sections.append("""2. Critical Moments: Decision points where the trajectory provides evidence about which actions are good or bad.
   - The step number must be the exact Step number from the trajectory CSV where this moment occurs.
   - The state description must use the actual observations from the trajectory, not abstract or assumed descriptions.
   - good_actions: A comma-separated list of action commands the agent SHOULD take in this state, based on evidence from the trajectory. Write "NONE" if there is no positive evidence.
   - bad_actions: A comma-separated list of action commands the agent should NOT take in this state, based on evidence from the trajectory (e.g., the agent took this action and it led to death or a bad outcome). Write "NONE" if there is no negative evidence.
   - Each action must be a single valid action command exactly as typed at the prompt. Do NOT write descriptions.
   - At least one of good_actions or bad_actions must be non-empty (not "NONE").
   - The evidence must reference specific observations showing WHY these actions are good or bad.
   - Do NOT describe states or actions using terminology or concepts not present in the trajectory observations.
   Format each as:
   <moment>
   <step_number>The exact Step number from the trajectory CSV</step_number>
   <raw_state>The actual observed game state at this moment, described using the trajectory's own terms</raw_state>
   <goal>Brief goal (one sentence)</goal>
   <good_actions>action1, action2 (or NONE)</good_actions>
   <bad_actions>action1, action2 (or NONE)</bad_actions>
   <evidence>Specific observations from the trajectory showing why these actions are good or bad</evidence>
   </moment>""")

    extraction_text = "\n\n".join(extraction_sections)

    qa_format = "<qa_pairs>\n(your Q&A pairs here)\n</qa_pairs>\n\n" if use_qa else ""
    moments_format = "<critical_moments>\n(your critical moments here)\n</critical_moments>\n\n" if use_moments else ""

    # Build existing QA section if provided
    existing_qa_section = ""
    if existing_qa:
        qa_lines = []
        for i, qa in enumerate(existing_qa, 1):
            answer_str = "YES" if qa.answer else "NO"
            qa_lines.append(f"  {i}. Q: {qa.question} — A: {answer_str} (evidence: {qa.evidence})")
        existing_qa_section = f"""
=== EXISTING Q&A PAIRS ===
The following Q&A pairs have already been extracted from previous steps. Do NOT duplicate these — only extract NEW knowledge not already covered.
{chr(10).join(qa_lines)}
=== END EXISTING Q&A PAIRS ===
"""

    prompt = f"""You are analyzing a trajectory from an agent interacting with an environment.

=== SEQUENCE OF STEPS ===
Below is the sequence of the agent's interactions with the environment.
Each step shows: the raw state observation, the perception module's output on that state, the agent's reasoning, and the action taken.

{traj_text}
=== END SEQUENCE OF STEPS ===
{existing_qa_section}
Your task is to extract structured knowledge from this trajectory. Use the trajectory as evidence to learn general facts about how the world works.

{extraction_text}

Important guidelines:
- Q&A pairs should capture reusable world knowledge (mechanics, rules, interactions) — not episode-specific agent behavior.
- Evidence must point to specific observations in this trajectory, but the questions themselves should be about general world properties.
- Use the trajectory's own language and terms when describing states, actions, and outcomes.
- When the trajectory is presented as a step sequence, treat the raw state observation, auxiliary observation, and perception output together as the evidence available at that step.
- Do NOT speculate beyond what the trajectory evidence supports.
- If something is ambiguous or could have multiple explanations, skip it entirely.
- Prefer fewer high-quality, well-grounded items over many speculative ones.

Format your response as:
<think>
Identify observations in the trajectory that reveal general world mechanics or rules.
</think>

{qa_format}{moments_format}"""

    text, cost = await _llm_call(config, prompt)
    qa_pairs = _parse_qa_pairs(text, step) if use_qa else []
    moments = _parse_critical_moments(text, step) if use_moments else []

    # Filter out moments from the last step (typically death/end-game screen)
    if moments:
        obs_map = _read_csv_observations(trajectory_path)
        if obs_map:
            last_step_num = max(obs_map.keys())
            before = len(moments)
            moments = [m for m in moments if m.traj_step_number != last_step_num]
            filtered = before - len(moments)
            if filtered:
                logging.info(f"Filtered out {filtered} moment(s) from last step ({last_step_num}) — end-game screen")

    # Attach raw observations from the CSV to moments
    if moments:
        _attach_raw_observations(moments, trajectory_path)
        attached = sum(1 for m in moments if m.raw_observation)
        logging.info(f"Attached raw observations to {attached}/{len(moments)} moments")

    logging.info(f"Extracted {len(qa_pairs)} Q&A pairs and {len(moments)} moments from {trajectory_path}")
    return qa_pairs, moments, cost, prompt, text


REVIEW_BATCH_SIZE = 20  # Max existing items (QA + moments combined) per review batch


async def _review_existing_batch(
    config: DictConfig,
    batch_qa: list[tuple[int, QAPair]],
    batch_moments: list[tuple[int, CriticalMoment]],
    new_evidence_text: str,
) -> tuple[list[int], list[int], float]:
    """Review a batch of existing items against new trajectory evidence.

    Returns: (qa_indices_to_remove, moment_indices_to_remove, cost)
    """
    existing_qa_text = ""
    if batch_qa:
        lines = []
        for orig_idx, qa in batch_qa:
            lines.append(f"  EQ{orig_idx+1}: Q: {qa.question} | A: {'YES' if qa.answer else 'NO'} | Evidence: {qa.evidence}")
        existing_qa_text = "Existing Q&A pairs to review:\n" + "\n".join(lines)

    existing_moments_text = ""
    if batch_moments:
        lines = []
        for orig_idx, m in batch_moments:
            lines.append(f"  EM{orig_idx+1}: State: {m.state} | Goal: {m.goal} | Good: [{', '.join(m.good_actions)}] | Bad: [{', '.join(m.bad_actions)}] | Evidence: {m.evidence}")
        existing_moments_text = "Existing critical moments to review:\n" + "\n".join(lines)

    prompt = f"""You are reviewing existing structured knowledge against new trajectory evidence.
All knowledge must be grounded in direct observations — not assumptions, generalizations, or inferences.

{existing_qa_text}

{existing_moments_text}

New trajectory evidence:
{new_evidence_text}

Check each existing item against the new trajectory evidence. An existing item should be REMOVED if:
- New evidence directly contradicts it (e.g., an answer is shown to be wrong).
- The original evidence was based on a coincidence or misinterpretation now clarified by new data.
- It is ambiguous and the new evidence does not resolve the ambiguity.
- It contains assumptions or generalizations that go beyond what the original evidence actually showed (e.g., claims something "always" or "never" happens based on limited observation).
Do NOT remove items just because they are not mentioned in new trajectories — only remove if actively contradicted, shown to be false, or poorly grounded.

Format your response as:
<think>
For each item, analyze whether the new evidence contradicts or undermines it.
</think>

<remove_existing_qa>
(List the IDs of existing Q&A pairs to REMOVE, e.g., "EQ2, EQ5". Write "NONE" if no pairs should be removed.)
</remove_existing_qa>

<remove_existing_moments>
(List the IDs of existing moments to REMOVE, e.g., "EM1, EM3". Write "NONE" if no moments should be removed.)
</remove_existing_moments>"""

    text, cost = await _llm_call(config, prompt)

    remove_qa_indices = []
    remove_qa_resp = extract_xml_key(text, "remove_existing_qa")
    if remove_qa_resp and "NONE" not in remove_qa_resp.upper():
        valid_indices = {orig_idx for orig_idx, _ in batch_qa}
        for match in re.finditer(r"EQ(\d+)", remove_qa_resp):
            idx = int(match.group(1)) - 1
            if idx in valid_indices:
                remove_qa_indices.append(idx)

    remove_moment_indices = []
    remove_moments_resp = extract_xml_key(text, "remove_existing_moments")
    if remove_moments_resp and "NONE" not in remove_moments_resp.upper():
        valid_indices = {orig_idx for orig_idx, _ in batch_moments}
        for match in re.finditer(r"EM(\d+)", remove_moments_resp):
            idx = int(match.group(1)) - 1
            if idx in valid_indices:
                remove_moment_indices.append(idx)

    return remove_qa_indices, remove_moment_indices, cost


async def consolidate_qa_and_moments(
    config: DictConfig,
    new_per_traj_qa: list[list[QAPair]],
    new_per_traj_moments: list[list[CriticalMoment]],
    existing_qa: list[QAPair],
    existing_moments: list[CriticalMoment],
) -> tuple[list[QAPair], list[CriticalMoment], list[int], list[int], float]:
    """Consolidate new Q&A pairs and moments, deduplicating against existing sets.

    Reviews existing items in batches (parallel LLM calls) against new trajectory
    evidence, then consolidates new items in a single call.

    Returns:
        (new_qa_to_add, new_moments_to_add,
         existing_qa_indices_to_remove, existing_moment_indices_to_remove, cost)
    """
    # Flatten new items
    all_new_qa = [qa for traj_qa in new_per_traj_qa for qa in traj_qa]
    all_new_moments = [m for traj_m in new_per_traj_moments for m in traj_m]

    if not all_new_qa and not all_new_moments and not existing_qa and not existing_moments:
        return [], [], [], [], 0.0

    total_cost = 0.0

    # Format new items as evidence text (shared across review batches and consolidation)
    new_evidence_lines = []
    for i, qa in enumerate(all_new_qa):
        new_evidence_lines.append(f"  NQ{i+1}: Q: {qa.question} | A: {'YES' if qa.answer else 'NO'} | Evidence: {qa.evidence}")
    for i, m in enumerate(all_new_moments):
        new_evidence_lines.append(f"  NM{i+1}: State: {m.state} | Goal: {m.goal} | Good: [{', '.join(m.good_actions)}] | Bad: [{', '.join(m.bad_actions)}] | Evidence: {m.evidence}")
    new_evidence_text = "\n".join(new_evidence_lines) if new_evidence_lines else "No new items extracted."

    # ---- PART 1: Review existing items in batches ----
    all_remove_qa_indices = []
    all_remove_moment_indices = []

    total_existing = len(existing_qa) + len(existing_moments)
    if (existing_qa or existing_moments) and (all_new_qa or all_new_moments):
        # Build indexed lists
        indexed_qa = list(enumerate(existing_qa))
        indexed_moments = list(enumerate(existing_moments))

        # Interleave QA and moments into batches of REVIEW_BATCH_SIZE
        all_indexed = [("qa", idx, item) for idx, item in indexed_qa] + \
                      [("moment", idx, item) for idx, item in indexed_moments]

        batches = []
        for batch_start in range(0, len(all_indexed), REVIEW_BATCH_SIZE):
            batch = all_indexed[batch_start:batch_start + REVIEW_BATCH_SIZE]
            batch_qa = [(idx, item) for kind, idx, item in batch if kind == "qa"]
            batch_moments = [(idx, item) for kind, idx, item in batch if kind == "moment"]
            batches.append((batch_qa, batch_moments))

        if len(batches) == 1:
            # Single batch — no need for parallel overhead
            rm_qa, rm_mom, cost = await _review_existing_batch(
                config, batches[0][0], batches[0][1], new_evidence_text,
            )
            all_remove_qa_indices.extend(rm_qa)
            all_remove_moment_indices.extend(rm_mom)
            total_cost += cost
        else:
            # Multiple batches — run in parallel
            logging.info(f"Reviewing {total_existing} existing items in {len(batches)} batches "
                         f"(batch size {REVIEW_BATCH_SIZE})")
            review_tasks = [
                _review_existing_batch(config, bq, bm, new_evidence_text)
                for bq, bm in batches
            ]
            review_results = await asyncio.gather(*review_tasks)
            for rm_qa, rm_mom, cost in review_results:
                all_remove_qa_indices.extend(rm_qa)
                all_remove_moment_indices.extend(rm_mom)
                total_cost += cost

    # ---- PART 2: Consolidate new items ----
    kept_qa = []
    kept_moments = []

    if all_new_qa or all_new_moments:
        # Format existing items (after planned removals) for dedup context
        surviving_qa_indices = set(range(len(existing_qa))) - set(all_remove_qa_indices)
        surviving_moment_indices = set(range(len(existing_moments))) - set(all_remove_moment_indices)

        existing_qa_text = ""
        if surviving_qa_indices:
            lines = []
            for i in sorted(surviving_qa_indices):
                qa = existing_qa[i]
                lines.append(f"  EQ{i+1}: Q: {qa.question} | A: {'YES' if qa.answer else 'NO'}")
            existing_qa_text = "Existing Q&A pairs (for dedup — do not modify these):\n" + "\n".join(lines)

        existing_moments_text = ""
        if surviving_moment_indices:
            lines = []
            for i in sorted(surviving_moment_indices):
                m = existing_moments[i]
                lines.append(f"  EM{i+1}: State: {m.state} | Goal: {m.goal} | Good: [{', '.join(m.good_actions)}] | Bad: [{', '.join(m.bad_actions)}]")
            existing_moments_text = "Existing moments (for dedup — do not modify these):\n" + "\n".join(lines)

        new_qa_text = ""
        if all_new_qa:
            lines = []
            for i, qa in enumerate(all_new_qa):
                lines.append(f"  NQ{i+1}: Q: {qa.question} | A: {'YES' if qa.answer else 'NO'} | Evidence: {qa.evidence}")
            new_qa_text = "New Q&A pairs extracted from latest trajectories:\n" + "\n".join(lines)

        new_moments_text = ""
        if all_new_moments:
            lines = []
            for i, m in enumerate(all_new_moments):
                lines.append(f"  NM{i+1}: State: {m.state} | Goal: {m.goal} | Good: [{', '.join(m.good_actions)}] | Bad: [{', '.join(m.bad_actions)}] | Evidence: {m.evidence}")
            new_moments_text = "New critical moments extracted from latest trajectories:\n" + "\n".join(lines)

        consolidation_prompt = f"""You are consolidating new structured knowledge extracted from multiple trajectories.
All knowledge must be grounded in direct observations — not assumptions, generalizations, or inferences.

{existing_qa_text}

{existing_moments_text}

{new_qa_text}

{new_moments_text}

Your task — CONSOLIDATE NEW ITEMS:
1. Remove duplicates within the new items (same fact discovered in multiple episodes).
2. Remove new items that duplicate existing knowledge.
3. Filter out ambiguous or contradictory new items.
4. Filter out items that contain assumptions or generalizations not directly supported by their evidence. The evidence field should quote or closely paraphrase actual trajectory observations — reject items where the evidence is vague or the question/state description uses terms not found in the trajectory.
5. Return only the consolidated new items to add.

Format your response as:
<think>
Analyze the evidence for each new item. Verify grounding in actual observations. Identify duplicates.
</think>

<consolidated_qa>
(List the IDs of new Q&A pairs to KEEP, e.g., "NQ1, NQ3, NQ5". Write "NONE" if no new pairs should be added.)
</consolidated_qa>

<consolidated_moments>
(List the IDs of new moments to KEEP, e.g., "NM1, NM2". Write "NONE" if no new moments should be added.)
</consolidated_moments>"""

        text, cost = await _llm_call(config, consolidation_prompt)
        total_cost += cost

        qa_keep_text = extract_xml_key(text, "consolidated_qa")
        if qa_keep_text and "NONE" not in qa_keep_text.upper():
            for match in re.finditer(r"NQ(\d+)", qa_keep_text):
                idx = int(match.group(1)) - 1
                if 0 <= idx < len(all_new_qa):
                    kept_qa.append(all_new_qa[idx])

        moments_keep_text = extract_xml_key(text, "consolidated_moments")
        if moments_keep_text and "NONE" not in moments_keep_text.upper():
            for match in re.finditer(r"NM(\d+)", moments_keep_text):
                idx = int(match.group(1)) - 1
                if 0 <= idx < len(all_new_moments):
                    kept_moments.append(all_new_moments[idx])

    logging.info(f"Consolidation: {len(all_new_qa)} -> {len(kept_qa)} new Q&A pairs, "
                 f"{len(all_new_moments)} -> {len(kept_moments)} new moments")
    logging.info(f"Review: removing {len(all_remove_qa_indices)} existing Q&A pairs, "
                 f"{len(all_remove_moment_indices)} existing moments "
                 f"(reviewed {total_existing} in {max(1, (total_existing + REVIEW_BATCH_SIZE - 1) // REVIEW_BATCH_SIZE)} batches)")
    return kept_qa, kept_moments, all_remove_qa_indices, all_remove_moment_indices, total_cost


async def score_beliefs(
    config: DictConfig,
    beliefs: str,
    qa_pairs: list[QAPair],
    critical_moments: list[CriticalMoment],
    default_knowledge: str,
    perception: str = "",
    use_raw_observations: bool = True,
) -> tuple[float, float, list[dict], float]:
    """Score beliefs against Q&A pairs and critical moments.

    For moments with raw observations (when use_raw_observations=True),
    presents the full game observation (and perception output if a perception
    module is provided) so the LLM predicts actions from realistic game state.

    Returns: (qa_score, moment_score, details, cost)
    """
    if not qa_pairs and not critical_moments:
        return 1.0, 1.0, [], 0.0

    # Build prediction tasks
    prediction_tasks = []
    if qa_pairs:
        prediction_tasks.append("Q&A Predictions - For each question, predict YES or NO based on the beliefs:")
        for i, qa in enumerate(qa_pairs):
            prediction_tasks.append(f"  Q{i+1}: {qa.question}")

    if critical_moments:
        prediction_tasks.append("\nMoment Predictions - For each situation, predict the single game command you would take:")
        for i, m in enumerate(critical_moments):
            if use_raw_observations and m.raw_observation:
                # Present the full raw observation as the agent would see it
                obs_block = f"\n--- Observation ---\n{m.raw_observation}\n--- End Observation ---"
                perc_block = ""
                if perception:
                    perc_output = _run_perception_on_observation(perception, m.raw_observation)
                    if perc_output:
                        perc_block = f"\n--- Perception Output ---\n{perc_output}\n--- End Perception Output ---"
                prediction_tasks.append(
                    f"  M{i+1}: Given the following game state, to achieve '{m.goal}', "
                    f"what single game command should be taken?"
                    f"{obs_block}{perc_block}"
                )
            else:
                # Fallback to summary state if no raw observation
                prediction_tasks.append(f"  M{i+1}: In state '{m.state}', to achieve '{m.goal}', what action should be taken?")

    tasks_text = "\n".join(prediction_tasks)

    prompt = f"""You are evaluating a set of beliefs about an environment by testing predictions.

The agent receives the following default instructions/knowledge by default:
=== DEFAULT KNOWLEDGE ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== CURRENT BELIEFS ===
{beliefs if beliefs else "(empty - no beliefs yet)"}
=== END OF CURRENT BELIEFS ===

Based ONLY on the beliefs and default knowledge above, make predictions for each item below.
Do NOT use any external knowledge — only what is stated in the beliefs.
For moment predictions (M items), respond with a single action command exactly as it would be typed at the prompt.

{tasks_text}

Format your response as:
<think>
For each item, reason about what the beliefs say (or don't say) and make a prediction.
</think>

<predictions>
(For each item, write the ID and your prediction on a separate line)
(For Q items: "Q1: YES" or "Q1: NO")
(For M items: "M1: action_name")
</predictions>"""

    text, cost = await _llm_call(config, prompt)

    predictions_text = extract_xml_key(text, "predictions") or ""

    # Score Q&A pairs
    qa_correct = 0
    details = []
    for i, qa in enumerate(qa_pairs):
        label = f"Q{i+1}"
        pattern = rf"{label}:\s*(YES|NO)"
        match = re.search(pattern, predictions_text, re.IGNORECASE)
        predicted = match.group(1).upper() if match else "MISSING"
        actual = "YES" if qa.answer else "NO"
        correct = predicted == actual
        if correct:
            qa_correct += 1
        details.append({
            "type": "qa",
            "id": label,
            "question": qa.question,
            "predicted": predicted,
            "actual": actual,
            "correct": correct,
        })

    # Score critical moments with tri-state: +1 (in good_actions), 0 (neutral), -1 (in bad_actions)
    moment_total_score = 0.0
    for i, m in enumerate(critical_moments):
        label = f"M{i+1}"
        pattern = rf"{label}:\s*(.+)"
        match = re.search(pattern, predictions_text)
        predicted = match.group(1).strip() if match else "MISSING"

        in_good = _action_in_set(predicted, m.good_actions)
        in_bad = _action_in_set(predicted, m.bad_actions)
        if in_good:
            score_val = 1
        elif in_bad:
            score_val = -1
        else:
            score_val = 0

        moment_total_score += score_val
        details.append({
            "type": "moment",
            "id": label,
            "state": m.state,
            "goal": m.goal,
            "predicted": predicted,
            "good_actions": m.good_actions,
            "bad_actions": m.bad_actions,
            "score": score_val,
            "correct": score_val > 0,  # For backwards compat with improve_beliefs display
            "raw_observation": m.raw_observation if use_raw_observations else "",
        })

    qa_score = qa_correct / len(qa_pairs) if qa_pairs else 1.0
    # Normalize moment score to [0, 1] range: map from [-N, N] to [0, 1]
    if critical_moments:
        n = len(critical_moments)
        moment_score = (moment_total_score + n) / (2 * n)
    else:
        moment_score = 1.0

    logging.info(f"Scoring: Q&A {qa_correct}/{len(qa_pairs)} ({qa_score:.2%}), "
                 f"Moments {moment_total_score}/{len(critical_moments)} ({moment_score:.2%})")
    return qa_score, moment_score, details, cost


async def improve_beliefs_from_scores(
    config: DictConfig,
    beliefs: str,
    qa_pairs: list[QAPair],
    critical_moments: list[CriticalMoment],
    score_details: list[dict],
    default_knowledge: str,
    episode_summaries: str,
    perception: str = "",
) -> tuple[str, float]:
    """Improve beliefs based on scoring results and episode summaries.

    For incorrect moment predictions, includes the full raw observation and
    perception output so the LLM can see exactly what the agent saw.
    """
    # Format scoring results
    correct_items = [d for d in score_details if d["correct"]]
    incorrect_items = [d for d in score_details if not d["correct"]]

    # Separate moment details by score category
    good_items = [d for d in score_details if d["correct"]]  # QA correct or moment score > 0
    bad_items = [d for d in score_details if not d["correct"]]

    correct_text = ""
    if good_items:
        lines = []
        for d in good_items:
            if d["type"] == "qa":
                lines.append(f"  {d['id']}: {d['question']} -> {d['actual']} (CORRECT)")
            else:
                lines.append(f"  {d['id']}: State: {d['state']}, Goal: {d['goal']} -> Predicted: {d['predicted']} (GOOD ACTION)")
        correct_text = "Correct/good predictions:\n" + "\n".join(lines)

    incorrect_text = ""
    if bad_items:
        lines = []
        for d in bad_items:
            if d["type"] == "qa":
                lines.append(f"  {d['id']}: {d['question']} -> Predicted: {d['predicted']}, Actual: {d['actual']} (WRONG)")
            else:
                score_val = d.get("score", 0)
                good_acts = d.get("good_actions", [])
                bad_acts = d.get("bad_actions", [])
                if score_val < 0:
                    label = "BAD ACTION"
                else:
                    label = "NEUTRAL"
                line = (f"  {d['id']}: Goal: {d['goal']} -> Predicted: {d['predicted']} ({label})"
                        f"\n    Should do: [{', '.join(good_acts)}] | Should NOT do: [{', '.join(bad_acts)}]")
                # Include raw observation + perception output for problematic moments
                raw_obs = d.get("raw_observation", "")
                if raw_obs:
                    line += f"\n    --- Game Observation ---\n    {raw_obs}"
                    if perception:
                        perc_output = _run_perception_on_observation(perception, raw_obs)
                        if perc_output:
                            line += f"\n    --- Perception Output ---\n    {perc_output}"
                    line += f"\n    --- End ---"
                else:
                    line += f"\n    State summary: {d['state']}"
                lines.append(line)
        incorrect_text = "Incorrect/problematic predictions:\n" + "\n".join(lines)

    total = len(score_details)
    num_correct = len(good_items)

    prompt = f"""You are improving beliefs about an environment based on scoring results.

The agent receives the following default instructions/knowledge by default:
=== DEFAULT KNOWLEDGE ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== CURRENT BELIEFS ===
{beliefs if beliefs else "(empty - no beliefs yet)"}
=== END OF CURRENT BELIEFS ===

The beliefs were tested against known facts. Results ({num_correct}/{total} correct):

{correct_text}

{incorrect_text}

Recent episode summaries for additional context:
{episode_summaries}

Your task: Update the beliefs to be consistent with ALL known facts, paying special attention to the incorrect predictions. The beliefs should:
- Correct any information that led to wrong predictions
- Incorporate knowledge needed for correct predictions
- Be grounded in the evidence from trajectories
- Be concise and actionable
- Keep each section to at most 8 concise bullet points. Merge redundant points.
- Balance safety with progress toward the objective. An overly cautious agent that never moves forward is worse than one that takes calculated risks.
- Remove beliefs that have not led to improved game outcomes.

Format your response as:
<think>
Analyze which beliefs need to change based on the incorrect predictions.
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
</updated_beliefs>"""

    text, cost = await _llm_call(config, prompt)

    updated_beliefs = extract_xml_key(text, "updated_beliefs")
    if not updated_beliefs:
        logging.warning("Failed to extract updated_beliefs from response, keeping current beliefs")
        return beliefs, cost

    return updated_beliefs.strip(), cost


async def improve_beliefs_simple(
    config: DictConfig,
    beliefs: str,
    default_knowledge: str,
    episode_summaries: str,
) -> tuple[str, float]:
    """Fallback belief improvement when both toggles are off. Directly improves from summaries."""
    prompt = f"""You are improving beliefs about an environment based on episode trajectories.

The agent receives the following default instructions/knowledge by default:
=== DEFAULT KNOWLEDGE ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== CURRENT BELIEFS ===
{beliefs if beliefs else "(empty - no beliefs yet)"}
=== END OF CURRENT BELIEFS ===

{episode_summaries}

Your task: Based on the episode summaries, update the beliefs to incorporate new knowledge and correct any incorrect beliefs.

For beliefs:
- Beliefs are split into two sections:
  * <world_knowledge>: mechanics, environmental properties, cause-and-effect relationships.
  * <policy>: what to do in specific situations, priorities, strategies for completing the objective.
- They should be brief with each point being a few sentences.
- Correct any wrong or misleading beliefs in either section.
- They should be grounded in the evidence present from the trajectories.
- They should be as simple as possible.
- Keep each section to at most 8 concise bullet points. Merge redundant points.
- Balance safety with progress toward the objective. An overly cautious agent that never moves forward is worse than one that takes calculated risks.

Format your response as:
<think>
Analyze what we learned from the trajectories.
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
</updated_beliefs>"""

    text, cost = await _llm_call(config, prompt)

    updated_beliefs = extract_xml_key(text, "updated_beliefs")
    if not updated_beliefs:
        logging.warning("Failed to extract updated_beliefs from response, keeping current beliefs")
        return beliefs, cost

    return updated_beliefs.strip(), cost


async def generate_experiments_from_gaps(
    config: DictConfig,
    beliefs: str,
    qa_pairs: list[QAPair],
    critical_moments: list[CriticalMoment],
    score_details: list[dict],
    episode_summaries: str,
    default_knowledge: str,
    num_experiments: int,
) -> tuple[list[str], float]:
    """Generate experiments targeting knowledge gaps."""
    # Format known facts
    known_facts = ""
    if qa_pairs:
        lines = []
        for i, qa in enumerate(qa_pairs):
            lines.append(f"  Q{i+1}: {qa.question} -> {'YES' if qa.answer else 'NO'}")
        known_facts += "Known Q&A pairs:\n" + "\n".join(lines) + "\n\n"

    if critical_moments:
        lines = []
        for i, m in enumerate(critical_moments):
            lines.append(f"  M{i+1}: State: {m.state} | Goal: {m.goal} | Good: [{', '.join(m.good_actions)}] | Bad: [{', '.join(m.bad_actions)}]")
        known_facts += "Known critical moments:\n" + "\n".join(lines) + "\n\n"

    # Format scoring gaps if available
    gaps_text = ""
    if score_details:
        incorrect = [d for d in score_details if not d["correct"]]
        if incorrect:
            lines = []
            for d in incorrect:
                if d["type"] == "qa":
                    lines.append(f"  {d['id']}: {d['question']} (beliefs predicted {d['predicted']}, actual {d['actual']})")
                else:
                    good_acts = d.get("good_actions", [])
                    bad_acts = d.get("bad_actions", [])
                    lines.append(f"  {d['id']}: State: {d['state']}, Goal: {d['goal']} (predicted '{d['predicted']}', good: [{', '.join(good_acts)}], bad: [{', '.join(bad_acts)}])")
            gaps_text = "Knowledge gaps (incorrect predictions):\n" + "\n".join(lines)

    prompt = f"""You are designing experiments to test in an environment to fill knowledge gaps.

The agent receives the following default instructions/knowledge by default:
=== DEFAULT KNOWLEDGE ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== CURRENT BELIEFS ===
{beliefs if beliefs else "(empty - no beliefs yet)"}
=== END OF CURRENT BELIEFS ===

{known_facts}{gaps_text}

{episode_summaries}

Your task: Design {num_experiments} experiments to test in the next step. Each experiment should be a specific, actionable strategy or mechanic to test during gameplay.

Guidelines:
- Target knowledge gaps — things we don't know or got wrong.
- Experiments should be conditional on environment state (e.g., "If you see X, try doing Y").
- Each experiment should be testable in a single episode.
- Each experiment must be 1-3 sentences maximum. State the hypothesis and the action to test, nothing more. Do NOT include multi-step procedures or nested conditionals.
- Focus on gaining knowledge that helps achieve the main goal.

Format your response as:
<think>
Identify the most important knowledge gaps and design experiments to fill them.
</think>

<experiments>
EXPERIMENT 1: [First experiment to test]

EXPERIMENT 2: [Second experiment to test]
...
(Generate {num_experiments} experiments)
</experiments>"""

    text, cost = await _llm_call(config, prompt)

    experiments_text = extract_xml_key(text, "experiments")
    experiments = []
    if experiments_text:
        for match in re.finditer(r"EXPERIMENT\s+\d+:\s*(.+?)(?=EXPERIMENT\s+\d+:|$)", experiments_text, re.DOTALL):
            exp = match.group(1).strip()
            if exp:
                # Cap experiment length to keep them concise
                if len(exp) > 300:
                    exp = exp[:300].rsplit(' ', 1)[0] + "..."
                experiments.append(exp)

    logging.info(f"Generated {len(experiments)} experiments")
    return experiments, cost, prompt, text


async def process_trajectories(
    config: DictConfig,
    beliefs: str,
    perception: str,
    output_dir: str,
    existing_qa: list[QAPair],
    existing_moments: list[CriticalMoment],
    default_knowledge: str,
    step: int,
    use_qa: bool,
    use_moments: bool,
) -> tuple[str, list[list[QAPair]], list[list[CriticalMoment]], list[QAPair], list[CriticalMoment], list[int], list[int], float, list[dict]]:
    """Process all trajectories: summarize + extract Q&A/moments in parallel.

    Returns:
        (episode_summaries_xml, per_traj_qa, per_traj_moments,
         new_consolidated_qa, new_consolidated_moments,
         existing_qa_indices_to_remove, existing_moment_indices_to_remove, total_cost,
         extraction_records)
    """
    csv_paths = sorted(Path(output_dir).rglob("*.csv"))

    if not csv_paths:
        logging.warning(f"No CSV trajectories found in {output_dir}")
        return "<episode_summaries>\nNo episodes found.\n</episode_summaries>", [], [], [], [], [], [], 0.0, []

    # Build parallel tasks: summaries and extractions
    summary_tasks = []
    extraction_tasks = []
    extraction_indices = []  # Track which CSV each extraction task corresponds to

    for csv_path in csv_paths:
        outcome_header = get_episode_outcome_header(str(csv_path))
        traj_text = csv_path.read_text()
        traj_text = trim_to_model_context_lim(traj_text, config.client.model_id)

        # Summary task
        summary_tasks.append(
            _get_beliefs_perception_summary_async(
                config, beliefs, perception, outcome_header, traj_text,
                str(csv_path), default_knowledge,
            )
        )

        # Extraction task (from raw trajectory)
        if use_qa or use_moments:
            extraction_tasks.append(
                extract_qa_and_moments_from_trajectory(
                    config, traj_text, str(csv_path),
                    step, use_qa, use_moments,
                )
            )
            extraction_indices.append(len(summary_tasks) - 1)

    # Run all tasks in parallel
    all_tasks = summary_tasks + extraction_tasks
    logging.info(f"Processing {len(csv_paths)} trajectories: {len(summary_tasks)} summaries, "
                 f"{len(extraction_tasks)} extractions")
    results = await asyncio.gather(*all_tasks)

    # Separate results
    summary_results = results[:len(summary_tasks)]
    extraction_results = results[len(summary_tasks):]

    # Build episode summaries XML and collect extraction records
    episode_summaries = "<episode_summaries>\nThese are summaries of different episodes of running the agent."
    total_cost = 0.0
    extraction_records = []
    for i, (summary, cost, s_prompt, s_response) in enumerate(summary_results):
        episode_summaries += f"\n<episode_summary episode_idx={i+1}>\n{summary.strip()}\n</episode_summary>"
        total_cost += cost
        extraction_records.append({
            "trajectory": str(csv_paths[i]),
            "summary_prompt": s_prompt,
            "summary_response": s_response,
        })
    episode_summaries += "\n</episode_summaries>"

    # Collect per-trajectory extractions
    per_traj_qa = []
    per_traj_moments = []
    for i, (qa_list, moments_list, cost, e_prompt, e_response) in enumerate(extraction_results):
        per_traj_qa.append(qa_list)
        per_traj_moments.append(moments_list)
        total_cost += cost
        # Match extraction back to its CSV via extraction_indices
        csv_idx = extraction_indices[i]
        if csv_idx < len(extraction_records):
            extraction_records[csv_idx]["extraction_prompt"] = e_prompt
            extraction_records[csv_idx]["extraction_response"] = e_response

    # Consolidate if we have extractions (or existing items to review)
    new_qa = []
    new_moments = []
    remove_qa_indices = []
    remove_moment_indices = []
    has_new_extractions = any(per_traj_qa) or any(per_traj_moments)
    has_existing = existing_qa or existing_moments
    if (use_qa or use_moments) and (has_new_extractions or has_existing):
        new_qa, new_moments, remove_qa_indices, remove_moment_indices, consolidation_cost = (
            await consolidate_qa_and_moments(
                config, per_traj_qa, per_traj_moments, existing_qa, existing_moments,
            )
        )
        total_cost += consolidation_cost

    return (episode_summaries, per_traj_qa, per_traj_moments,
            new_qa, new_moments, remove_qa_indices, remove_moment_indices, total_cost,
            extraction_records)


def _build_execution_report(
    perception: str,
    sample_observations: list[tuple[str, int]],
    max_samples: int = 3,
) -> str:
    """Run perception on sample observations and build a human-readable I/O report.

    Args:
        perception: Perception module source code.
        sample_observations: List of (raw_observation, step_number) tuples.
        max_samples: Maximum number of samples to include.

    Returns:
        Formatted string showing input→output pairs, or empty string if no samples.
    """
    if not sample_observations or not perception or not perception.strip():
        return ""

    samples = sample_observations[:max_samples]
    blocks = []
    for raw_obs, step_num in samples:
        output = _run_perception_on_observation(perception, raw_obs)
        blocks.append(
            f"<execution_sample step=\"{step_num}\">\n"
            f"<input>\n{raw_obs}\n</input>\n"
            f"<output>\n{output if output else '(empty — perception produced no output)'}\n</output>\n"
            f"</execution_sample>"
        )
    return "\n".join(blocks)


async def improve_perception(
    config: DictConfig,
    perception: str,
    beliefs: str,
    episode_summaries: str,
    perception_examples: str,
    default_knowledge: str,
    initial_perception: str = "",
    initial_perception_examples: str = "",
    moment_perception_feedback: str = "",
    sample_observations: list[tuple[str, int]] | None = None,
) -> tuple[str, float, str]:
    """Improve the perception module based on episode summaries and perception examples.

    Uses a retry loop with validation (up to 3 attempts). Falls back to previous
    perception if all attempts fail.

    When initial_perception is provided (and differs from current perception),
    it is included in the prompt so the LLM can see the perception module that
    was active when the episode summaries were collected.

    When sample_observations is provided, the current perception module is executed
    on these real observations and the I/O pairs are shown in the prompt, so the
    LLM can see exactly what the code actually produces. After generating a
    candidate, the new code is also executed on the same observations and the
    output is verified — if it appears degraded, the execution results are fed
    back for a retry.

    Returns: (updated_perception, cost, reasoning, prompt_used)
    """
    perception_instructions = """For the perception module:
- It should be a Python function `perceive(observation_text: str) -> str`.
- Input `observation_text` contains everything from the direct observation as a string.
- The code must be valid Python.
- Ensure that the perception module is working correctly in that it is correctly extracting the intended information from the environment state and presenting it in the features from perception module section.
- Output should be a textual description of the environment state that is useful for progressing in the environment.
- Output should contain all information that is necessary for progressing in the environment and should be presented in a clear and descriptive way."""

    # Build initial perception context if it differs from the current one
    show_initial = (
        initial_perception
        and initial_perception.strip()
        and initial_perception.strip() != (perception or "").strip()
    )
    if show_initial:
        initial_context = f"""
The episode summaries below were collected using an earlier version of the perception module.
Use it and its examples to understand what the agent actually saw during those episodes.

=== INITIAL PERCEPTION MODULE (used during data collection) ===
{initial_perception}
=== END OF INITIAL PERCEPTION MODULE ===

{initial_perception_examples}
"""
    else:
        initial_context = ""

    moment_feedback_section = ""
    if moment_perception_feedback:
        moment_feedback_section = f"""
=== PERCEPTION TEST RESULTS ===
We tested whether an LLM could predict correct actions from perception output alone.
Below are the moments where prediction failed — the perception module did not surface
enough information for correct decisions. Fix the perception module so it surfaces the
information needed for correct decisions.

{moment_perception_feedback}
=== END PERCEPTION TEST RESULTS ===
"""

    # Build execution report for current perception on real observations
    execution_report_section = ""
    if sample_observations:
        exec_report = _build_execution_report(perception, sample_observations)
        if exec_report:
            execution_report_section = f"""
=== CURRENT PERCEPTION EXECUTION OUTPUT ===
Below is the ACTUAL output of the current perception module when run on real
observations from the environment. Review this carefully — if the output is
missing information that is clearly present in the raw observation (e.g., map
features, nearby objects, hazards), the perception code likely has bugs such as
regex patterns that don't match the actual observation format.

{exec_report}
=== END CURRENT PERCEPTION EXECUTION OUTPUT ===
"""

    base_prompt = f"""We are interacting with an environment and trying to figure out how it works.

The agent receives the following default instructions/knowledge by default:
=== DEFAULT KNOWLEDGE ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

We maintain the following current beliefs about the environment:
=== CURRENT BELIEFS ===
{beliefs if beliefs else "(empty - no beliefs yet)"}
=== END OF CURRENT BELIEFS ===

The following code is the current version used to extract useful features from the raw observations:
=== CURRENT PERCEPTION MODULE ===
{perception if perception else "(empty - no perception module yet)"}
=== END OF CURRENT PERCEPTION MODULE ===

We have collected the following experience by running the agent:
=== COLLECTED EXPERIENCE ===
{episode_summaries}
{perception_examples}
=== END OF COLLECTED EXPERIENCES ===
{initial_context}
{moment_feedback_section}
{execution_report_section}
Your task is to:
1. Analyze the results and the perception examples (input/output pairs).
2. If execution output is provided above, carefully compare it against the raw observations — identify any information present in the raw observation that is missing from the perception output. This often indicates bugs in the parsing code (e.g., regex patterns that don't match the actual format, hardcoded values that vary between runs, missing flags like re.DOTALL for multi-line matching).
3. Update the perception module to make sure it is correct and that it extracts useful information from the direct observation and presents it in a clear and descriptive way. Pay special attention to the perception test results (if provided) — these show specific situations where an LLM could not predict correct actions from perception output alone, with the raw observation showing what information was available but not surfaced.

{perception_instructions}

Format your response as:
<think>
Analyze the perception module's current behavior and design improvements.
</think>

<updated_perception>
```python
def perceive(observation_text: str) -> str:
    # Your implementation here
    pass
```
</updated_perception>"""

    max_retries = 3
    perception_error = None
    updated_perception = perception
    total_cost = 0.0
    reasoning = ""
    prompt_used = base_prompt

    for attempt in range(max_retries):
        if perception_error:
            prompt = f"""{base_prompt}

=== PERCEPTION CODE ERROR (RETRY {attempt}/{max_retries}) ===
Your previous perception code had an error and failed to execute:
{perception_error}

Please fix the error in your perception code.
=== END OF PERCEPTION CODE ERROR ===
"""
        else:
            prompt = base_prompt

        prompt_used = prompt
        text, cost = await _llm_call(config, prompt)
        total_cost += cost

        # Extract reasoning
        think_text = extract_xml_key(text, "think") or ""

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

        # Validate — use real observations when available
        test_obs = [raw for raw, _ in sample_observations[:3]] if sample_observations else None
        is_valid, error_msg = validate_perception_code(candidate_perception, test_observations=test_obs)
        if not is_valid:
            perception_error = error_msg
            logging.warning(f"Perception code validation failed (attempt {attempt + 1}/{max_retries}): {error_msg}")
            continue

        # Execution verification: run candidate on real observations and check
        # for output degradation (much shorter than what the raw obs warrants)
        if sample_observations:
            new_report = _build_execution_report(candidate_perception, sample_observations)
            old_report = _build_execution_report(perception, sample_observations) if perception else ""

            # Check if new perception produces substantially less output than old
            new_outputs = [
                _run_perception_on_observation(candidate_perception, raw_obs)
                for raw_obs, _ in sample_observations[:3]
            ]
            old_outputs = [
                _run_perception_on_observation(perception, raw_obs)
                for raw_obs, _ in sample_observations[:3]
            ] if perception else [""] * len(new_outputs)

            # Flag if new output is empty or drastically shorter on any sample
            degraded_samples = []
            for i, (new_out, old_out, (raw_obs, step_num)) in enumerate(
                zip(new_outputs, old_outputs, sample_observations[:3])
            ):
                raw_has_map = "map:" in raw_obs and len(raw_obs) > 500
                new_line_count = len(new_out.strip().splitlines()) if new_out else 0
                if raw_has_map and new_line_count <= 3:
                    degraded_samples.append(
                        f"Sample step {step_num}: raw observation has {len(raw_obs)} chars "
                        f"including map data, but perception output is only {new_line_count} line(s):\n"
                        f"{new_out if new_out else '(empty)'}"
                    )

            if degraded_samples:
                degradation_detail = "\n\n".join(degraded_samples)
                perception_error = (
                    f"Execution verification failed — your new perception code compiles and runs, "
                    f"but produces very little output on real observations, suggesting parsing bugs "
                    f"(e.g., regex patterns that don't match the actual observation format).\n\n"
                    f"{degradation_detail}\n\n"
                    f"Full execution I/O on sample observations:\n{new_report}\n\n"
                    f"Common causes:\n"
                    f"- Hardcoded strings that vary between runs (e.g., character class names like "
                    f"'Agent the Evoker' vs 'Agent the Tenderfoot')\n"
                    f"- Missing re.DOTALL flag when matching across newlines\n"
                    f"- Regex boundary patterns that assume blank lines (\\n\\n) but actual lines "
                    f"have trailing spaces\n"
                    f"Please fix the perception code to correctly parse these observations."
                )
                logging.warning(
                    f"Perception execution verification failed (attempt {attempt + 1}/{max_retries}): "
                    f"{len(degraded_samples)} degraded sample(s)"
                )
                continue

        updated_perception = candidate_perception
        reasoning = think_text.strip()
        logging.info(f"Perception code validated successfully on attempt {attempt + 1}")
        return updated_perception, total_cost, reasoning, prompt_used

    logging.error(f"All {max_retries} attempts to generate valid perception code failed. Keeping previous perception.")
    return perception, total_cost, reasoning, prompt_used


def serialize_qa_pairs(qa_pairs: list[QAPair]) -> list[dict]:
    return [asdict(qa) for qa in qa_pairs]


def deserialize_qa_pairs(data: list[dict]) -> list[QAPair]:
    return [QAPair(**d) for d in data]


def serialize_moments(moments: list[CriticalMoment]) -> list[dict]:
    return [asdict(m) for m in moments]


def deserialize_moments(data: list[dict]) -> list[CriticalMoment]:
    moments = []
    for d in data:
        # Handle old data without raw_observation/traj_step_number fields
        d.setdefault("raw_observation", "")
        d.setdefault("traj_step_number", -1)
        # Handle old data with single "action" field -> convert to action sets
        if "action" in d and "good_actions" not in d:
            d["good_actions"] = [d.pop("action")]
            d.setdefault("bad_actions", [])
        d.setdefault("good_actions", [])
        d.setdefault("bad_actions", [])
        # Remove legacy field if still present
        d.pop("action", None)
        moments.append(CriticalMoment(**d))
    return moments
