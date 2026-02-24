import asyncio
import json
import logging
import re
from pathlib import Path

import litellm
from omegaconf import DictConfig

from balrog.pricing import calculate_cost
from llm_utils import build_llm_input, extract_llm_response_text, extract_xml_kv, validate_response_fields


# Map NLE end_status codes to human-readable descriptions
# Source: nle/env/base.py:152-166 and nle/env/tasks.py:102-106
END_REASON_MAP = {
    -1: "ABORTED - Episode truncated (max steps reached)",
    0: "RUNNING - Episode ended while still in progress",
    1: "DEATH - Agent died during the episode",
    2: "TASK_SUCCESSFUL - Agent completed the task goal",
}


def _get_response_cost(response, model_id: str) -> float:
    """Extract cost from a litellm response object using BALROG pricing."""
    try:
        usage = response.usage
        if usage is None:
            return 0.0
        input_tokens = getattr(usage, 'input_tokens', 0) or 0
        output_tokens = getattr(usage, 'output_tokens', 0) or 0
        return calculate_cost(model_id, input_tokens, output_tokens)
    except Exception:
        return 0.0


def get_episode_outcome_header(trajectory_path: str) -> str:
    """Read episode metadata from JSON file and create a clear outcome header.

    Args:
        trajectory_path: Path to the trajectory CSV file

    Returns:
        Formatted string describing the episode outcome
    """
    # Find corresponding JSON file (same name but .json extension)
    csv_path = Path(trajectory_path)
    json_path = csv_path.with_suffix(".json")

    if not json_path.exists():
        return "=== EPISODE OUTCOME: Unknown (metadata file not found) ===\n"

    try:
        with open(json_path) as f:
            metadata = json.load(f)

        progression = metadata.get("progression", 0.0)
        num_steps = metadata.get("num_steps", "unknown")
        end_reason_code = metadata.get("end_reason", None)
        episode_return = metadata.get("episode_return", 0.0)
        task = metadata.get("task", "unknown")

        # Convert end_reason code to human-readable string
        if isinstance(end_reason_code, int):
            end_reason_str = END_REASON_MAP.get(end_reason_code, f"Unknown (code={end_reason_code})")
            # Include raw code for transparency
            end_reason_str = f"{end_reason_str} (code={end_reason_code})"
        elif isinstance(end_reason_code, str):
            end_reason_str = end_reason_code
        else:
            end_reason_str = f"Unknown (raw={end_reason_code})"

        # Determine success/failure status
        if progression >= 1.0:
            outcome_status = "SUCCESS - Goal achieved!"
        else:
            outcome_status = "FAILURE"

        # Provide goal context based on task name
        if "Quest" in task or "Staircase" in task:
            goal_reminder = "GOAL: Find and reach the stairs down (>) to descend to the next level."
        else:
            goal_reminder = "GOAL: Complete the task objective."

        header = f"""Task: {task}
{goal_reminder}
Status: {outcome_status}
End Reason: {end_reason_str}
Steps Taken: {num_steps}
Episode Return: {episode_return:.2f}
"""
        return header

    except (json.JSONDecodeError, KeyError) as e:
        logging.warning(f"Failed to parse episode metadata from {json_path}: {e}")
        return "=== EPISODE OUTCOME: Unknown (failed to parse metadata) ===\n"


def trim_to_model_context_lim(text: str, model_name: str, buffer: int = 0, prefix: bool = True) -> str:
    if "qwen" in model_name.lower():
        return text[-100000:]
    else:
        return text[-350000:]


def _extract_between(text: str, start_marker: str, end_marker: str) -> str:
    """Extract content between two delimiter strings."""
    if start_marker not in text:
        return ""
    after = text.split(start_marker, 1)[1]
    if end_marker not in after:
        return after.strip()
    return after.split(end_marker, 1)[0].strip()


def _sample_indices(n: int, k: int) -> list[int]:
    """Return k evenly-spaced indices in [0, n)."""
    if k >= n:
        return list(range(n))
    return [int(i * (n - 1) / (k - 1)) for i in range(k)] if k > 1 else [0]


def extract_obs_perc_examples(
    output_dir: str,
    perception: str,
    max_trajs: int = 3,
    max_steps_per_traj: int = 1,
) -> str:
    """Extract concrete (raw_observation, perception_output) pairs from trajectory CSVs.

    Shows the verbatim input fed to perceive() and its verbatim output, so the
    improve LLM can see exactly what the code receives and produces.

    Returns a formatted string to inject into the improve prompt, or "" if perception
    is empty or no perception sections are found in the CSVs.
    """
    import csv as csv_module

    if not perception or not perception.strip():
        return ""

    PERC_START = "========== Start of features from Perception Module =========="
    PERC_END   = "========== End of features from Perception Module =========="
    OBS_START  = "========== Start of Direct Game Observation =========="
    OBS_END    = "========== End of Direct Game Observation =========="

    examples = []
    csv_paths = sorted(Path(output_dir).rglob("*.csv"))[:max_trajs]

    for traj_idx, csv_path in enumerate(csv_paths):
        try:
            with open(csv_path, newline='') as f:
                reader = csv_module.DictReader(f, escapechar="\u02d8", quoting=csv_module.QUOTE_MINIMAL)
                rows = list(reader)
        except Exception as e:
            logging.warning(f"Failed to read {csv_path}: {e}")
            continue

        if not rows:
            continue

        # len(rows) - 1 is so that we dont include last state
        # which is usually just the end status screen
        for row_idx in _sample_indices(len(rows) - 1, max_steps_per_traj):
            obs_text = rows[row_idx].get('Observation', '')
            if PERC_START not in obs_text:
                continue  # perception not active for this step

            perc_out = _extract_between(obs_text, PERC_START, PERC_END)
            raw_obs  = _extract_between(obs_text, OBS_START,  OBS_END)

            if perc_out and raw_obs:
                step_num = rows[row_idx].get('Step', str(row_idx))
                examples.append((traj_idx + 1, step_num, raw_obs, perc_out))

    if not examples:
        if perception and perception.strip():
            logging.warning(
                "Perception code is non-empty but no perception output found in any trajectory CSV. "
                "The perception module may have failed silently during rollouts."
            )
            return (
                "<perception_examples>\n"
                "WARNING: The perception code was provided but produced NO output in any trajectory.\n"
                "This means the perception module either failed silently (e.g. a runtime error disabled it)\n"
                "or it ran but returned an empty string for every observation.\n"
                "Please review and fix the perception code so it reliably produces output.\n"
                "</perception_examples>"
            )
        return ""

    lines = [
        "<perception_examples>",
        "These show the exact input the current perception code receives and the exact output it produces.",
        "Use them to verify correctness and identify bugs or missing information.",
        "",
    ]
    for ep_num, step_num, raw_obs, perc_out in examples:
        lines.append(f"<example episode={ep_num} step={step_num}>")
        lines.append("INPUT to perceive() — verbatim Direct Game Observation:")
        lines.append(f"<perceive_input>")
        lines.append(raw_obs.strip())
        lines.append("")
        lines.append(f"</perceive_input>\n")
        lines.append("OUTPUT of perceive():")
        lines.append("<perceive_output>")
        lines.append(perc_out.strip())
        lines.append("")
        lines.append(f"</perceive_output>")
        lines.append("</example>")
        lines.append("")

    lines.append("</perception_examples>")
    return '\n'.join(lines)


async def _get_beliefs_perception_summary_async(
    config: DictConfig,
    beliefs: str,
    perception: str,
    outcome_header: str,
    traj_text: str,
    trajectory_path: str,
    default_knowledge: str,
) -> str:
    """Get a summary focused on beliefs and perception evaluation (async).

    Args:
        config: Configuration containing model information
        beliefs: Current beliefs about the game
        perception: Current perception module code
        outcome_header: Formatted episode outcome header
        traj_text: Trajectory text content
        trajectory_path: Path to the trajectory file (for logging)

    Returns:
        Summary text focused on beliefs and perception
    """
    prompt = f"""We are playing a game and trying to figure out how it works.

The agent receives the following default instructions/knowledge by default:
=== DEFAULT KNOWLEDGE ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

We maintain the following current beliefs about the game:
=== CURRENT BELIEFS ===
{beliefs if beliefs else "(empty - no beliefs yet)"}
=== END OF CURRENT BELIEFS ===

The following code is used to extract useful features from the raw game observations:
=== PERCEPTION MODULE ===
{perception if perception else "(empty - no perception module yet)"}
=== END OF PERCEPTION MODULE ===

We have played the game using these beliefs and perception.  
Here is the episode outcome and trajectory:
=== EPISODE OUTCOME ===
{outcome_header}
=== END OF EPISODE OUTCOME ===
=== TRAJECTORY ===
{traj_text}
=== END OF TRAJECTORY ===

Your summary should contain analysis of the behaviour of the perception module:
- The perception module is provided with everything inside the Direct Game Observation as an input string.
- It should extract features that are useful to playing the game.
- Ensure that the perception module is working correctly in that the intended information in the perception code is correctly being presented in the features from perception module section.

Your summary should be grounded in the episode outcome above:

If the episode was a FAILURE (0% progress or death):
- What was the PRIMARY CAUSE of failure? Trace back from the end state to identify the critical mistake(s).
- What beliefs led to bad decisions? 
- What beliefs were missing that would have prevented this outcome?
- Did the perception module include any misleading information that led to this outcome?
- What additional information could be included by the perception module that would have prevented this outcome?

Even if the episode was a FAILURE, the agent may have made partial progress towards the goal.
- Decide if the agent was able to make progress towards the goal?
- What allowed progress to be made? This is important to identifying beliefs that might lead to success.
- What prevented full completion? Identify the specific bottleneck or mistake.
- Did the perception module help with the successful parts? What did it do incorrectly? 
- What information could be included by the perception module that would have resulted in further progress?

If the episode was a SUCCESS:
- What key actions led to success? What beliefs can we infer about the world from this?
- What information from perception (if any) was most valuable?
- Was there unnecessary inefficiency that could be improved?

Provide a summary highlighting:
- Root cause analysis: Why did the episode end this way?
- Belief learning: What beliefs can we infer from the trajectory, especially those that show us how to make progress towards completing the task. 
- Belief update: How should we update our beliefs so that we can make progress towards completing the task. Were there beliefs that were incorrect or misleading?
- Perception correctness: Regardless of the outcome of the episode, verify whether the perception module is working correctly. Check that the output of the perception module is correctly mapping the corresponding direct game observation into the intended features.
- Perception analysis: What information was presented in the explicit features from perception module section. What part of that information was helpful, what information was misleading / incorrect and what additional information would have helped if extracted by the perception module?

Format your response in XML style as -
<think>
Analyze the trajectory in light of the episode outcome. Focus on causality - what led to this specific result?
</think>
<summary>
Summary with the sections above clearly addressed
</summary>
"""

    # Build input for LLM
    input_data = build_llm_input(prompt)

    # Call LLM asynchronously
    logging.info(f"Calling LLM to summarize instructions/perception for: {trajectory_path}")
    logging.info(f"Instructions/perception summary prompt:\n{prompt}")
    if config.client.client_name == "vllm":
        model_name = f"hosted_vllm/{config.client.model_id}"
    else:
        model_name = f"{config.client.client_name}/{config.client.model_id}"
    response = await asyncio.to_thread(
        litellm.responses,
        model=model_name,
        input=input_data,
        num_retries=5,
    )

    # Extract response text
    response_text = extract_llm_response_text(response)
    logging.info(f"Instructions/perception summary LLM response for {trajectory_path}:\n{response_text}")

    cost = _get_response_cost(response, config.client.model_id)

    # Extract summary from XML
    response_dict = extract_xml_kv(response_text, ["summary"])

    if not validate_response_fields(response_dict, response_text, ["summary"]):
        return "", cost
    else:
        return response_dict["summary"], cost


async def _get_experiment_summary_async(
    config: DictConfig,
    experiment: str,
    outcome_header: str,
    traj_text: str,
    trajectory_path: str,
) -> str:
    """Get a summary focused on experiment evaluation (async).

    Args:
        config: Configuration containing model information
        experiment: The experiment that was being tested in this episode
        outcome_header: Formatted episode outcome header
        traj_text: Trajectory text content
        trajectory_path: Path to the trajectory file (for logging)

    Returns:
        Summary text focused on experiment evaluation
    """
    prompt = f"""We are playing a game and testing a specific experiment about how it works.

We were testing the following experiment in this episode:
=== EXPERIMENT ===
{experiment}
=== END EXPERIMENT ===

Here is the episode outcome and trajectory:
=== EPISODE OUTCOME ===
{outcome_header}
=== END OF EPISODE OUTCOME ===
=== TRAJECTORY ===
{traj_text}
=== END OF TRAJECTORY ===

Analyse whether the above trajectory provides enough evidence to validate or invalidate the given experiment.

Format your response in XML style as -
<think>
Analyse whether we can validate or invalidate the experiment from the given trajectory.
</think>
<summary>
A summary of the reasons behind whether the given experiment is correct or not, or an explanation of insufficient evidence for a conclusion.
</summary>
"""

    # Build input for LLM
    input_data = build_llm_input(prompt)

    # Call LLM asynchronously
    logging.info(f"Calling LLM to summarize experiment for: {trajectory_path}")
    logging.info(f"Experiment summary prompt:\n{prompt}")
    if config.client.client_name == "vllm":
        model_name = f"hosted_vllm/{config.client.model_id}"
    else:
        model_name = f"{config.client.client_name}/{config.client.model_id}"
    response = await asyncio.to_thread(
        litellm.responses,
        model=model_name,
        input=input_data,
        num_retries=5,
    )

    # Extract response text
    response_text = extract_llm_response_text(response)
    logging.info(f"Experiment summary LLM response for {trajectory_path}:\n{response_text}")

    cost = _get_response_cost(response, config.client.model_id)

    # Extract summary from XML
    response_dict = extract_xml_kv(response_text, ["summary"])

    if not validate_response_fields(response_dict, response_text, ["summary"]):
        return "", cost
    else:
        return response_dict["summary"], cost


async def get_episode_summary_async(
    config: DictConfig,
    beliefs: str,
    perception: str,
    trajectory_path: str,
    default_knowledge: str,
    experiment: str | None = None,
) -> tuple[str, str]:
    """Get a summary of an episode trajectory using LLM (async).

    This function makes separate LLM calls for:
    1. Beliefs and perception evaluation
    2. Experiment evaluation (if experiment is provided)

    Args:
        config: Configuration containing model information
        beliefs: Current beliefs about the game
        perception: Current perception module code
        trajectory_path: Path to the trajectory file
        experiment: Optional experiment that was being tested in this episode

    Returns:
        Summary text extracted from LLM response(s)
    """
    # Get episode outcome header from JSON metadata
    outcome_header = get_episode_outcome_header(trajectory_path)

    traj_text = Path(trajectory_path).read_text()
    traj_text = trim_to_model_context_lim(traj_text, config.client.model_id)

    # Call for instructions and perception summary
    beliefs_perception_summary, ip_cost = await _get_beliefs_perception_summary_async(
        config, beliefs, perception, outcome_header, traj_text, trajectory_path, default_knowledge
    )

    total_cost = ip_cost
    include_experiment_summary = config.eval.evolve.get("include_experiment_summary", True)

    # If experiment is provided and enabled in config, make a separate call for experiment summary
    if experiment and include_experiment_summary:
        experiment_summary, h_cost = await _get_experiment_summary_async(
            config, experiment, outcome_header, traj_text, trajectory_path
        )
        total_cost += h_cost
        # Combine both summaries
        combined_summary = f"""=== INSTRUCTIONS AND PERCEPTION ANALYSIS ===
{beliefs_perception_summary}

=== EXPERIMENT ANALYSIS ===
{experiment_summary}"""
        return combined_summary, total_cost
    else:
        return beliefs_perception_summary, total_cost


def validate_perception_code(code: str) -> tuple[bool, str | None]:
    """Validate perception code by attempting to compile and execute it.

    Args:
        code: Python code string containing the perceive function

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is None.
    """
    if not code or not code.strip():
        return True, None  # Empty code is valid (no perception)

    # First, try to compile the code to catch syntax errors
    try:
        compile(code, "<perception_module>", "exec")
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"

    # Try to execute and verify the perceive function exists and is callable
    try:
        namespace = {}
        exec(code, namespace)
        if "perceive" not in namespace:
            return False, "No 'perceive' function found in the code"
        if not callable(namespace["perceive"]):
            return False, "'perceive' is not a callable function"

        # Test with a sample input to catch runtime errors in basic execution
        test_input = (
            "message: You see here a +0 dagger.\n\n"
            "cursor:\nYourself a rogue\n(x=10, y=5)\n\n"
            "map:\n"
            "         -----          \n"
            "         |...|          \n"
            "         |.@d.|         \n"
            "         |..F.|         \n"
            "         -----          \n\n"
            "Agent the Footpad              St:15 Dx:17 Co:13 In:13 Wi:11 Ch:6 Chaotic S:0\n"
            "Dlvl:1 $:0 HP:12(12) Pw:2(2) AC:7 Xp:1/0\n"
        )
        try:
            result = namespace["perceive"](test_input)
            if not isinstance(result, str):
                return False, f"'perceive' function must return a string, got {type(result).__name__}"
        except Exception as e:
            return False, f"Runtime error when testing perceive function: {e}"

    except Exception as e:
        return False, f"Failed to execute perception code: {e}"

    return True, None


def improve_step(
    config: DictConfig,
    base_beliefs: str,
    perception: str,
    output_dir: str,
    previous_experiments: list[str],
    default_knowledge: str,
    num_experiments: int = 0,
    rollout_results: dict[str, dict] | None = None,
) -> tuple[str, str, list[str], float]:
    """Improve step: Evaluate, Update Beliefs/Perception, optionally Generate New Experiments.

    When num_experiments > 0, the LLM is asked to generate new experiments to test.
    When num_experiments == 0, only beliefs and perception are updated (no experiment generation).

    Args:
        config: Configuration containing model information
        base_beliefs: Current beliefs/instructions
        perception: Current perception module
        output_dir: Directory containing rollout results
        previous_experiments: List of experiments tested in the last step
        default_knowledge: Default knowledge string to include in prompt
        num_experiments: Number of new experiments to generate (0 = no generation)
        rollout_results: Results from run_explore_rollouts, including any errors

    Returns:
        Tuple of (updated_beliefs, updated_perception, new_experiments, total_improve_cost).
        new_experiments is [] when num_experiments == 0.
    """
    generate_experiments = num_experiments > 0
    improve_mode = config.eval.evolve.get("improve_mode", "both")
    improve_beliefs = improve_mode in ("both", "beliefs")
    improve_perception = improve_mode in ("both", "perception")

    async def get_all_summaries():
        """Get all episode summaries in parallel."""
        episode_tasks = []

        for episode_path in Path(output_dir).rglob("*.csv"):
            try:
                rel_path = episode_path.relative_to(output_dir)
                run_name = rel_path.parts[0]  # e.g., baseline_0, experiment_1

                experiment_text = None
                if run_name.startswith("experiment_"):
                    try:
                        idx = int(run_name.split("_")[1])
                        if 0 <= idx < len(previous_experiments):
                            experiment_text = previous_experiments[idx]
                    except (ValueError, IndexError):
                        logging.warning(f"Could not map run {run_name} to an experiment")

                episode_tasks.append(
                    get_episode_summary_async(
                        config,
                        base_beliefs,
                        perception,
                        str(episode_path),
                        default_knowledge,
                        experiment=experiment_text,
                    )
                )
            except ValueError:
                logging.warning(f"Could not determine run name for {episode_path}")
                continue

        logging.info(f"Generating summaries for {len(episode_tasks)} episodes in parallel")
        return await asyncio.gather(*episode_tasks)

    # Get summaries for all episodes in parallel
    ep_results = asyncio.run(get_all_summaries())

    # Combine all summaries and track costs
    episode_summaries = "<episode_summaries>\nThese are summaries of different episodes of playing the game."
    summary_cost = 0.0
    for i, (summary, cost) in enumerate(ep_results):
        episode_summaries += f"<episode_summary episode_idx={i+1}>\n{summary.strip()}\n</episode_summary>\n"
        summary_cost += cost

    # Include any rollout errors in evidence section
    if rollout_results:
        for run_name, result in rollout_results.items():
            if "error" in result:
                episode_summaries += f"<rollout_error run={run_name}>\n{result['error']}\n<rollout_error>\n"
    episode_summaries += "</episode_summaries>"
  
    # Extract concrete (observation → perception output) pairs for grounded perception improvement
    perc_example_max_trajs = config.eval.evolve.get("perc_example_max_trajs", 3)
    perc_example_max_steps = config.eval.evolve.get("perc_example_max_steps", 1)
    perception_examples = extract_obs_perc_examples(
        output_dir, perception,
        max_trajs=perc_example_max_trajs,
        max_steps_per_traj=perc_example_max_steps,
    )

    # Build prompt — experiment generation section is conditional
    if generate_experiments:
        _steps = ["1. Analyze the results. If experiments were tested, determine if they were confirmed or refuted."]
        _step_num = 2
        if improve_beliefs:
            _steps.append(f"{_step_num}. Update our beliefs about the game based on confirmed knowledge.")
            _step_num += 1
        if improve_perception:
            _steps.append(
                f"{_step_num}. Update the perception module to make sure it is correct and that it extracts better features from the direct game observation."
            )
            _step_num += 1
        _steps.append(
            f"{_step_num}. Generate {num_experiments} NEW experiments to test in the next step.\n"
            "   - Experiments should be specific, actionable strategies or mechanics to test.\n"
            "   - They should help us achieve the main goal."
        )
        task_section = "Your task is to:\n" + "\n".join(_steps)
        experiments_xml = f"""<new_experiments>
EXPERIMENT 1: [First experiment to test]

EXPERIMENT 2: [Second experiment to test]
...
(Generate exactly {num_experiments} experiments)
</new_experiments>"""
    else:
        _steps = ["1. Analyze the results."]
        _step_num = 2
        if improve_beliefs:
            _steps.append(f"{_step_num}. Update our beliefs about the game based on confirmed knowledge.")
            _step_num += 1
        if improve_perception:
            _steps.append(
                f"{_step_num}. Update the perception module to make sure it is correct and that it extracts useful information from the direct game observation and presents in a clear and descriptive way."
            )
            _step_num += 1
        task_section = "Your task is to:\n" + "\n".join(_steps)
        experiments_xml = ""

    # Build think_tag conditioned on which components are active
    _think_actions = ["Analyze results"]
    if generate_experiments:
        _think_actions.append("evaluate experiments")
    if improve_beliefs:
        _think_actions.append("determine belief updates")
    if improve_perception:
        _think_actions.append("design perception improvements")
    if generate_experiments:
        _think_actions.append("brainstorm new experiments")
    if len(_think_actions) == 2:
        _think_inner = " and ".join(_think_actions)
    elif len(_think_actions) > 2:
        _think_inner = ", ".join(_think_actions[:-1]) + f", and {_think_actions[-1]}"
    else:
        _think_inner = _think_actions[0]
    think_tag = f"<think>\n{_think_inner}.\n</think>"

    # Build conservatism_note conditioned on which components are active
    if generate_experiments:
        conservatism_note = ""
    else:
        if improve_beliefs and improve_perception:
            _update_target = "the beliefs or the perception module"
        elif improve_beliefs:
            _update_target = "the beliefs"
        else:
            _update_target = "the perception module"
        conservatism_note = (
            f"\nSince we are only evaluating experiments, only update {_update_target} if we have learned "
            "anything new from the collected experience. Do not update if we have not learned anything new."
        )

    # Build conditional instruction blocks
    beliefs_instructions = ""
    if improve_beliefs:
        beliefs_instructions = """For beliefs:
- They should describe information about how the game works that helps us complete the objective.
- They should be brief with each point being a few sentences.
- Correct any wrong or misleading beliefs
- They should be grounded in the evidence present from the trajectories.
- They should be as simple as possible."""

    perception_instructions = ""
    if improve_perception:
        perception_instructions = """For the perception module:
- It should be a Python function `perceive(observation_text: str) -> str`.
- Input `observation_text` contains everything from the direct game observation as a string.
- The code must be valid Python.
- Ensure that the perception module is working correctly in that it is correctly extracting the intended information from the direct game state and presenting it in the features from perception module section.
- Output should be a textual description of the game state that is useful for progressing in the game.
- Output should contain all information that is necessary for progressing in the game and should be presented in a clear and description way."""

    # Build conditional XML format blocks
    beliefs_xml_fmt = ""
    if improve_beliefs:
        beliefs_xml_fmt = """<updated_beliefs>
- [belief 1]
- [belief 2]
...
</updated_beliefs>"""

    perception_xml_fmt = ""
    if improve_perception:
        perception_xml_fmt = """<updated_perception>
```python
def perceive(observation_text: str) -> str:
    # Your implementation here
    pass
```
</updated_perception>"""

    base_prompt = f"""We are playing a game and trying to figure out how it works.

The agent receives the following default instructions/knowledge by default:
=== DEFAULT KNOWLEDGE ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

We maintain the following current beliefs about the game:
=== CURRENT BELIEFS ===
{base_beliefs if base_beliefs else "(empty - no beliefs yet)"}
=== END OF CURRENT BELIEFS ===

The following code is used to extract useful features from the raw game observations:
=== PERCEPTION MODULE ===
{perception if perception else "(empty - no perception module yet)"}
=== END OF PERCEPTION MODULE ===

We have collected the following experience by playing the game:
=== COLLECTED EXPERIENCE ===
{episode_summaries}
{perception_examples}
=== END OF COLLECTED EXPERIENCES ===

{task_section}

{beliefs_instructions}

{perception_instructions}

{conservatism_note}

Format your response in XML style as:
{think_tag}
{beliefs_xml_fmt}
{perception_xml_fmt}
{experiments_xml}
"""

    # Setup model name
    if config.client.client_name == "vllm":
        model_name = f"hosted_vllm/{config.client.model_id}"
    else:
        model_name = f"{config.client.client_name}/{config.client.model_id}"

    # Determine which XML fields to extract
    xml_fields = []
    if improve_beliefs:
        xml_fields.append("updated_beliefs")
    if improve_perception:
        xml_fields.append("updated_perception")
    if generate_experiments:
        xml_fields.append("new_experiments")

    # Retry loop for perception validation
    max_retries = 3
    perception_error = None
    updated_beliefs = base_beliefs
    updated_perception = perception
    new_experiments = []
    improve_call_cost = 0.0

    for attempt in range(max_retries):
        # Build prompt with error feedback if this is a retry
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

        logging.info(f"Improve step prompt (attempt {attempt + 1}/{max_retries}):\n{prompt}")

        # Build input for LLM
        input_data = build_llm_input(prompt)

        # Call LLM
        logging.info(f"Calling LLM for improve step (attempt {attempt + 1}/{max_retries})")
        response = litellm.responses(
            model=model_name,
            input=input_data,
            num_retries=5,
        )
        improve_call_cost += _get_response_cost(response, config.client.model_id)

        # Extract response text
        response_text = extract_llm_response_text(response)
        logging.info(f"Improve step LLM response (attempt {attempt + 1}/{max_retries}):\n{response_text}")

        # Extract fields
        response_dict = extract_xml_kv(response_text, xml_fields)
        validate_response_fields(response_dict, response_text, xml_fields)

        # Process beliefs
        if improve_beliefs and "updated_beliefs" in response_dict:
            updated_beliefs = response_dict["updated_beliefs"].strip()

        # If perception updating is disabled, no validation loop needed
        if not improve_perception:
            break

        # Process perception
        candidate_perception = perception
        if "updated_perception" in response_dict:
            candidate_perception = response_dict["updated_perception"].strip()
            # Strip markdown markers
            if candidate_perception.startswith("```python"):
                candidate_perception = candidate_perception[len("```python"):].strip()
            elif candidate_perception.startswith("```"):
                candidate_perception = candidate_perception[len("```"):].strip()
            if candidate_perception.endswith("```"):
                candidate_perception = candidate_perception[:-len("```")].strip()

        # Validate perception code
        is_valid, error_msg = validate_perception_code(candidate_perception)
        if is_valid:
            updated_perception = candidate_perception
            perception_error = None
            logging.info(f"Perception code validated successfully on attempt {attempt + 1}")
            break
        else:
            perception_error = error_msg
            logging.warning(f"Perception code validation failed (attempt {attempt + 1}/{max_retries}): {error_msg}")
            if attempt == max_retries - 1:
                # On final attempt, keep the previous working perception
                logging.error(f"All {max_retries} attempts to generate valid perception code failed. Keeping previous perception.")
                updated_perception = perception

        # Process experiments (do this on every attempt so we have them even if perception fails)
        if generate_experiments and "new_experiments" in response_dict:
            experiments_text = response_dict["new_experiments"].strip()
            experiment_pattern = r'EXPERIMENT\s*\d+\s*:\s*(.+?)(?=EXPERIMENT\s*\d+\s*:|$)'
            matches = re.findall(experiment_pattern, experiments_text, re.DOTALL | re.IGNORECASE)
            new_experiments = [m.strip() for m in matches if m.strip()]

            # Fallback split
            if not new_experiments:
                new_experiments = [b.strip() for b in experiments_text.split('\n\n') if b.strip()]

    # Process experiments from final response if not already processed
    if generate_experiments and not new_experiments and "new_experiments" in response_dict:
        experiments_text = response_dict["new_experiments"].strip()
        experiment_pattern = r'EXPERIMENT\s*\d+\s*:\s*(.+?)(?=EXPERIMENT\s*\d+\s*:|$)'
        matches = re.findall(experiment_pattern, experiments_text, re.DOTALL | re.IGNORECASE)
        new_experiments = [m.strip() for m in matches if m.strip()]

        # Fallback split
        if not new_experiments:
            new_experiments = [b.strip() for b in experiments_text.split('\n\n') if b.strip()]

    # Limit to requested num
    if generate_experiments:
        new_experiments = new_experiments[:num_experiments]

    logging.info(f"Updated beliefs:\n{updated_beliefs}")
    logging.info(f"Updated perception:\n{updated_perception}")
    logging.info(f"Generated {len(new_experiments)} new experiments")

    total_improve_cost = summary_cost + improve_call_cost
    return updated_beliefs, updated_perception, new_experiments, total_improve_cost
