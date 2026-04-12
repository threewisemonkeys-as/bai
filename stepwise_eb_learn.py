"""Stepwise EB-learning: experiment-driven per-step learning.

Like stepwise B-learning but more experiment-driven: the agent generates
questions about the environment, designs experiments to answer them, and
updates Q from trajectory evidence.  No critical moments — improvement is
via beliefs/perception (Tracks 1a, 1b) and QA (Track 2) only.
"""

import asyncio
import csv
import json
import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from balrog.agents import AgentFactory
from balrog.environments import make_env
from balrog.utils import get_unique_seed

from explore import get_default_actions, get_default_knowledge, override_temperature, evolve_logger
from mixed_improve import (
    QAPair,
    _run_perception_on_observation,
)
from b_learn_improve import (
    qa_forward_pass,
    qa_get_feedback,
    serialize_qa_feedback_results,
    _improve_with_perception_validation_conversational,
    _improve_beliefs_only_conversational,
)
from stepwise_b_learn_improve import (
    parse_submit_signal,
    build_perception_followup_message,
    build_perception_with_analysis_prompt,
    build_qa_followup_message,
    _build_obs_section,
    _build_execution_report_section,
    PERCEPTION_INSTRUCTIONS,
    RESPONSE_FORMAT,
    BELIEFS_ONLY_RESPONSE_FORMAT,
)
from stepwise_eb_learn_improve import (
    EBQAPair,
    serialize_eb_qa_pairs,
    deserialize_eb_qa_pairs,
    eb_qa_to_qa,
    generate_questions_from_steps,
    formulate_experiment_from_question,
    update_qa_from_trajectory,
    trim_qa_pairs,
)
from llm_utils import extract_xml_key
from stepwise_b_learn import (
    format_steps_context,
    _compose_obs_text,
    _refresh_buffer_with_perception,
    _sample_observations_from_buffer,
    _inject_beliefs,
    _flush_improve_progress,
)
from stepwise_explore import (
    load_perception_fn,
    apply_perception,
)
from run_utils import setup_run, improve_logging, _update_summary_json


@dataclass
class StepwiseEBLearnConfig:
    n_environment_steps: int
    max_perception_iterations: "int | list[list[int]]"  # Track 1b turns (int or schedule)
    max_qa_iterations: "int | list[list[int]]"          # Track 2 turns (int or schedule)
    max_qa_per_forward: int
    max_total_qa_pairs: int
    num_questions: int                 # Questions per generation step
    num_sample_obs: int
    explore_temp: float
    artifact_update_interval: int
    improve_interval: int
    experiment_interval: int
    max_steps_context_chars: int


def _resolve_schedule(value: "int | list[list[int]]", global_step: int) -> int:
    """Resolve a schedule value based on global_step.

    If value is an int, return it directly.
    If value is a list of [step_threshold, count] pairs, return the count
    for the first range that contains global_step. The last entry acts as
    the default (its threshold is ignored).

    Example: [[10, 10], [20, 5], [0, 3]]
      - steps 0-9:  10
      - steps 10-19: 5
      - steps 20+:   3
    """
    if isinstance(value, int):
        return value
    cumulative = 0
    for i, entry in enumerate(value):
        threshold, count = entry[0], entry[1]
        if i == len(value) - 1:
            return count
        if global_step < cumulative + threshold:
            return count
        cumulative += threshold
    return value[-1][1]


# ---------------------------------------------------------------------------
# Artifact saving helpers
# ---------------------------------------------------------------------------


def _save_step_artifacts_eb(
    step_dir: Path,
    beliefs: str,
    perception: str,
    qa_pairs: list[EBQAPair],
    feedback_history: list[dict],
    extraction_log: dict | None = None,
    experiment_log: dict | None = None,
    trim_log: dict | None = None,
):
    """Save all artifacts for a completed step."""
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / "beliefs.txt").write_text(beliefs)
    (step_dir / "perception.py").write_text(perception)
    with open(step_dir / "qa_pairs.json", "w") as f:
        json.dump(serialize_eb_qa_pairs(qa_pairs), f, indent=4)
    if feedback_history:
        with open(step_dir / "feedback_history.json", "w") as f:
            json.dump(feedback_history, f, indent=4, default=str)
    if extraction_log:
        with open(step_dir / "extraction_log.json", "w") as f:
            json.dump(extraction_log, f, indent=4, default=str)
    if experiment_log:
        with open(step_dir / "experiment_log.json", "w") as f:
            json.dump(experiment_log, f, indent=4, default=str)
    if trim_log:
        with open(step_dir / "trim_log.json", "w") as f:
            json.dump(trim_log, f, indent=4, default=str)


def _save_step_log_eb(
    step_dir: Path,
    step: int,
    global_step: int,
    action: str | None,
    reward: float,
    done: bool,
    episode_return: float,
    agent_cost: float,
    extract_cost: float,
    improve_cost: float,
    experiment_cost: float,
    trim_cost: float,
    num_qa: int,
    num_unanswered: int,
    did_gen_questions: bool = False,
    did_formulate_experiment: bool = False,
    did_trim: bool = False,
    active_experiment: str | None = None,
    phase: str = "complete",
):
    """Write a per-step JSON log with action, costs, and artifact counts."""
    step_log = {
        "step": step,
        "global_step": global_step,
        "phase": phase,
        "action": action,
        "reward": reward,
        "done": done,
        "episode_return_so_far": episode_return,
        "agent_step_cost": agent_cost,
        "extract_cost": extract_cost,
        "improve_cost": improve_cost,
        "experiment_cost": experiment_cost,
        "trim_cost": trim_cost,
        "step_total_cost": agent_cost + extract_cost + improve_cost + experiment_cost + trim_cost,
        "num_qa_pairs": num_qa,
        "num_unanswered_questions": num_unanswered,
        "did_gen_questions": did_gen_questions,
        "did_formulate_experiment": did_formulate_experiment,
        "did_trim": did_trim,
        "active_experiment": active_experiment,
    }
    with open(step_dir / "step_log.json", "w") as f:
        json.dump(step_log, f, indent=4)


def _save_episode_artifacts_eb(
    episode_dir: Path,
    beliefs: str,
    perception: str,
    qa_pairs: list[EBQAPair],
    trajectory_buffer: list[dict] | None = None,
    past_experiments: list[str] | None = None,
):
    """Save all artifacts for a completed episode."""
    episode_dir.mkdir(parents=True, exist_ok=True)
    (episode_dir / "beliefs.txt").write_text(beliefs)
    (episode_dir / "perception.py").write_text(perception)
    with open(episode_dir / "qa_pairs.json", "w") as f:
        json.dump(serialize_eb_qa_pairs(qa_pairs), f, indent=4)
    if trajectory_buffer is not None:
        with open(episode_dir / "trajectory_buffer.json", "w") as f:
            json.dump(trajectory_buffer, f, indent=2, default=str)
    if past_experiments is not None:
        with open(episode_dir / "past_experiments.json", "w") as f:
            json.dump(past_experiments, f, indent=4)


def _find_last_completed_episode_eb(
    output_dir: str,
) -> tuple[int, str, str, list[EBQAPair]]:
    """Find the last completed episode directory and restore EB state.

    Returns: (last_episode, beliefs, perception, qa_pairs)
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return -1, "", "", []

    episode_dirs = []
    for item in output_path.iterdir():
        if item.is_dir() and item.name.startswith("episode_"):
            try:
                ep_num = int(item.name.split("_")[1])
                if (item / "beliefs.txt").exists():
                    episode_dirs.append((ep_num, item))
            except (ValueError, IndexError):
                continue

    if not episode_dirs:
        return -1, "", "", []

    episode_dirs.sort(key=lambda x: x[0])
    last_ep, last_dir = episode_dirs[-1]

    beliefs = (last_dir / "beliefs.txt").read_text()

    perception = ""
    perc_file = last_dir / "perception.py"
    if perc_file.exists():
        perception = perc_file.read_text()

    qa_pairs: list[EBQAPair] = []
    qa_file = last_dir / "qa_pairs.json"
    if qa_file.exists():
        try:
            qa_pairs = deserialize_eb_qa_pairs(json.loads(qa_file.read_text()))
        except (json.JSONDecodeError, TypeError):
            pass

    evolve_logger.info(f"Resuming from episode {last_ep} in {last_dir}")
    return last_ep, beliefs, perception, qa_pairs


# ---------------------------------------------------------------------------
# Inner improve loop (conversational only, no moments)
# ---------------------------------------------------------------------------


def _run_improve_loop_eb(
    config: DictConfig,
    eb_config: StepwiseEBLearnConfig,
    beliefs: str,
    perception: str,
    qa_pairs: list[EBQAPair],
    trajectory_buffer: list[dict],
    default_knowledge: str,
    step: int,
    global_step: int = 0,
    step_dir: Path | None = None,
) -> tuple[str, str, list[EBQAPair], float, list[dict]]:
    """Run the 2-track adaptive improve loop (beliefs/perception + QA).

    Track 1a: Steps-based beliefs improvement (1 turn)
    Track 1b: Perception improvement from analysis (multi-turn)
    Track 2:  QA-based improvement (multi-turn, answered questions only)

    Returns: (beliefs, perception, qa_pairs, total_cost, feedback_history)
    """
    total_cost = 0.0
    feedback_history: list[dict] = []
    tag = f"[g{global_step}]"

    max_perception_iters = _resolve_schedule(eb_config.max_perception_iterations, global_step)
    max_qa_iters = _resolve_schedule(eb_config.max_qa_iterations, global_step)

    steps_context = format_steps_context(
        trajectory_buffer, perception, eb_config.max_steps_context_chars,
    )
    sample_obs = _sample_observations_from_buffer(
        trajectory_buffer, eb_config.num_sample_obs,
    )

    num_answered = sum(1 for q in qa_pairs if q.answer is not None)
    evolve_logger.info(
        f"{tag} Improve loop: perception={max_perception_iters}, "
        f"qa={max_qa_iters} iters, "
        f"{len(qa_pairs)} QA ({num_answered} answered), "
        f"{len(steps_context)} chars context"
    )

    try:

        # ========================================
        # Track 1a: Steps-based beliefs improvement
        # ========================================
        if steps_context:
            track1a_record = {"track": "steps_beliefs", "step": step, "global_step": global_step, "turns": []}

            steps_beliefs_prompt = f"""We are interacting with an environment and trying to figure out how it works. We maintain beliefs about the environment to guide the agent's decisions.

The agent receives the following default instructions/knowledge:
=== DEFAULT KNOWLEDGE ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

We maintain the following current beliefs about the environment:
=== CURRENT BELIEFS ===
{beliefs if beliefs else "(empty - no beliefs yet)"}
=== END CURRENT BELIEFS ===

Below is the actual sequence of the agent's recent interactions with the environment.
Each step shows: the raw state observation, the perception module's output on that state, the agent's reasoning, and the action taken.

=== SEQUENCE OF STEPS ===
{steps_context}
=== END SEQUENCE OF STEPS ===

Your task is to:
1. Analyze the step sequence.
2. Update our beliefs about the environment based on confirmed knowledge from the steps.

Provide analysis highlighting:
- Belief learning: What beliefs can we infer from the observations.
- Belief update: How should we update our current beliefs so that they more accurately reflect how the world works.
- Perception analysis: What information was presented in output of the perception module. What part of that information was helpful, what information was misleading / incorrect and what additional information would have helped if extracted by the perception module?

For beliefs:
- Overall they beliefs should be split into two sections:
  * <world_knowledge>: Facts about how the environment works — mechanics, properties, cause-and-effect relationships.
  * <policy>: Tactical approaches — what to do in specific situations, priorities, strategies for completing the objective.
- Correct any wrong or misleading beliefs in either section.
- Both sections should be consise, made up of a few brief points, merging any redundant or stale information.
- They should be grounded in the evidence present in the step sequence, only containing inferences from what we have observed so far.

{PERCEPTION_INSTRUCTIONS}

{BELIEFS_ONLY_RESPONSE_FORMAT}"""

            beliefs, turn_cost, _, response_text = asyncio.run(
                _improve_beliefs_only_conversational(
                    config=config,
                    beliefs=beliefs,
                    conversation_history=[],
                    user_message=steps_beliefs_prompt,
                )
            )
            total_cost += turn_cost

            perception_analysis = extract_xml_key(response_text, "perception_analysis") or ""

            turn_record = {"turn": 1, "cost": turn_cost, "prompt": steps_beliefs_prompt, "response": response_text}
            track1a_record["turns"].append(turn_record)

            evolve_logger.info(f"{tag}     Track 1a done (cost: ${turn_cost:.6f}, perception_analysis: {len(perception_analysis)} chars)")

            feedback_history.append(track1a_record)
            if step_dir is not None:
                _flush_improve_progress(step_dir, feedback_history, beliefs, perception)
        else:
            evolve_logger.info(f"{tag}     Track 1a: No steps context, skipping")
            perception_analysis = ""

        # ========================================
        # Track 1b: Perception improvement guided by beliefs analysis
        # ========================================
        if sample_obs:
            track1b_record = {"track": "perception_from_analysis", "step": step, "global_step": global_step, "turns": []}
            pre_perception_track1b = perception
            obs_section_1b = _build_obs_section(perception, sample_obs)

            perception_from_analysis_prompt = build_perception_with_analysis_prompt(
                beliefs=beliefs,
                perception=perception,
                default_knowledge=default_knowledge,
                obs_section=obs_section_1b,
                perception_analysis=perception_analysis,
                max_iterations=max_perception_iters,
            )

            perception_conv_1b: list[dict] = []
            prev_obs_section_1b = obs_section_1b

            for turn in range(max_perception_iters):
                evolve_logger.info(f"{tag}     Track 1b (perception from analysis) turn {turn + 1}/{max_perception_iters}")

                message = perception_from_analysis_prompt if turn == 0 else build_perception_followup_message(
                    perception, sample_obs, prev_obs_section_1b,
                    current_turn=turn + 1, max_turns=max_perception_iters,
                )

                _beliefs_unused, perception, turn_cost, perception_conv_1b, response_text = asyncio.run(
                    _improve_with_perception_validation_conversational(
                        config=config,
                        beliefs=beliefs,
                        perception=perception,
                        conversation_history=perception_conv_1b,
                        user_message=message,
                        sample_observations=sample_obs,
                    )
                )
                total_cost += turn_cost

                turn_record = {"turn": turn + 1, "cost": turn_cost, "prompt": message, "response": response_text}
                submitted = parse_submit_signal(response_text)
                turn_record["submitted"] = submitted
                track1b_record["turns"].append(turn_record)

                evolve_logger.info(
                    f"{tag}     Track 1b turn {turn + 1} done "
                    f"(cost: ${turn_cost:.6f}, submit: {submitted})"
                )

                if submitted:
                    break

                prev_obs_section_1b = _build_obs_section(perception, sample_obs)

            feedback_history.append(track1b_record)
            if step_dir is not None:
                _flush_improve_progress(step_dir, feedback_history, beliefs, perception)

            # Rebuild steps_context if perception changed during Track 1b
            if perception != pre_perception_track1b:
                steps_context = format_steps_context(
                    trajectory_buffer, perception, eb_config.max_steps_context_chars,
                )
        else:
            evolve_logger.info(f"{tag}     Track 1b: No sample observations, skipping")

        # ========================================
        # Track 2: QA-based conversational improvement
        # ========================================
        # Filter to answered questions only for forward/feedback evaluation
        answered_qa = [q for q in qa_pairs if q.answer is not None]
        if not answered_qa:
            evolve_logger.info(f"{tag}     Track 2: No answered questions, skipping")
        else:
            track2_record = {"track": "qa", "step": step, "global_step": global_step, "turns": []}

            # Convert to QAPair for existing forward/feedback functions
            qa_for_eval = [eb_qa_to_qa(q) for q in answered_qa]

            # Initial QA evaluation
            evolve_logger.info(f"{tag}     Track 2: Initial QA forward pass on {len(qa_for_eval)} answered questions...")
            qa_fwd_results, qa_fwd_cost, qa_fwd_prompts, qa_fwd_responses = asyncio.run(
                qa_forward_pass(
                    config=config,
                    beliefs=beliefs,
                    qa_pairs=qa_for_eval,
                    max_per_batch=eb_config.max_qa_per_forward,
                )
            )
            total_cost += qa_fwd_cost

            qa_fb_results, qa_fb_cost, qa_fb_prompts, qa_fb_responses = asyncio.run(
                qa_get_feedback(
                    config=config,
                    qa_forward_results=qa_fwd_results,
                    max_per_batch=eb_config.max_qa_per_forward,
                )
            )
            total_cost += qa_fb_cost

            qa_correct = [fr for fr in qa_fb_results if fr.verdict == "CORRECT"]
            qa_incorrect = [fr for fr in qa_fb_results if fr.verdict == "INCORRECT"]
            qa_inconclusive = [fr for fr in qa_fb_results if fr.verdict == "INCONCLUSIVE"]
            qa_actionable = [fr for fr in qa_fb_results if fr.verdict != "INCONCLUSIVE"]

            evolve_logger.info(
                f"{tag}     Track 2: Initial eval: {len(qa_correct)} correct, "
                f"{len(qa_incorrect)} incorrect, {len(qa_inconclusive)} inconclusive"
            )

            track2_record["initial_correct"] = len(qa_correct)
            track2_record["initial_incorrect"] = len(qa_incorrect)
            track2_record["qa_forward_cost"] = qa_fwd_cost
            track2_record["qa_feedback_cost"] = qa_fb_cost
            track2_record["qa_forward_prompt"] = "\n---\n".join(qa_fwd_prompts)
            track2_record["qa_forward_response"] = "\n---\n".join(qa_fwd_responses)
            track2_record["qa_feedback_prompt"] = "\n---\n".join(qa_fb_prompts)
            track2_record["qa_feedback_response"] = "\n---\n".join(qa_fb_responses)
            track2_record["qa_feedback_details"] = serialize_qa_feedback_results(qa_fb_results)

            pre_track_perception = perception
            if qa_actionable and qa_incorrect:
                # Build initial QA improvement prompt
                qa_blocks = []
                for i, fr in enumerate(qa_actionable, 1):
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

                execution_report_section = _build_execution_report_section(perception, sample_obs)

                initial_qa_prompt = f"""You are improving an agent's knowledge and perception based on testing its understanding of the environment via question-answering.

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

Results: {len(qa_correct)} correct, {len(qa_incorrect)} incorrect out of {len(qa_fb_results)} evaluated.

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

This is a multi-turn conversation. After each response, the QA pairs will be re-evaluated with your updated beliefs/perception. You can iterate up to {max_qa_iters} turns.

{PERCEPTION_INSTRUCTIONS}

{RESPONSE_FORMAT}"""

                qa_conversation: list[dict] = []
                prev_qa_correct = len(qa_correct)
                prev_qa_incorrect = len(qa_incorrect)

                for turn in range(max_qa_iters):
                    evolve_logger.info(f"{tag}     Track 2 turn {turn + 1}/{max_qa_iters}")

                    message = initial_qa_prompt if turn == 0 else build_qa_followup_message(
                        qa_fb_results, prev_qa_correct, prev_qa_incorrect,
                    )

                    beliefs, perception, turn_cost, qa_conversation, response_text = asyncio.run(
                        _improve_with_perception_validation_conversational(
                            config=config,
                            beliefs=beliefs,
                            perception=perception,
                            conversation_history=qa_conversation,
                            user_message=message,
                            sample_observations=sample_obs if sample_obs else None,
                        )
                    )
                    total_cost += turn_cost

                    turn_record = {"turn": turn + 1, "cost": turn_cost, "prompt": message, "response": response_text}
                    submitted = parse_submit_signal(response_text)
                    turn_record["submitted"] = submitted
                    track2_record["turns"].append(turn_record)

                    evolve_logger.info(
                        f"{tag}     Track 2 turn {turn + 1} done "
                        f"(cost: ${turn_cost:.6f}, submit: {submitted})"
                    )

                    if submitted:
                        evolve_logger.info(f"{tag}     Track 2: LLM submitted after {turn + 1} turn(s)")
                        break

                    # Re-evaluate QA for next turn (unless this is the last turn)
                    if turn + 1 < max_qa_iters:
                        prev_qa_correct = sum(1 for fr in qa_fb_results if fr.verdict == "CORRECT")
                        prev_qa_incorrect = sum(1 for fr in qa_fb_results if fr.verdict == "INCORRECT")

                        qa_fwd_results, qa_fwd_cost, _, _ = asyncio.run(
                            qa_forward_pass(
                                config=config,
                                beliefs=beliefs,
                                qa_pairs=qa_for_eval,
                                max_per_batch=eb_config.max_qa_per_forward,
                            )
                        )
                        total_cost += qa_fwd_cost

                        qa_fb_results, qa_fb_cost, _, _ = asyncio.run(
                            qa_get_feedback(
                                config=config,
                                qa_forward_results=qa_fwd_results,
                                max_per_batch=eb_config.max_qa_per_forward,
                            )
                        )
                        total_cost += qa_fb_cost

                        new_correct = sum(1 for fr in qa_fb_results if fr.verdict == "CORRECT")
                        new_incorrect = sum(1 for fr in qa_fb_results if fr.verdict == "INCORRECT")
                        evolve_logger.info(
                            f"{tag}     Track 2 re-eval: {new_correct} correct "
                            f"({new_correct - prev_qa_correct:+d}), "
                            f"{new_incorrect} incorrect ({new_incorrect - prev_qa_incorrect:+d})"
                        )
            else:
                evolve_logger.info(f"{tag}     Track 2: No incorrect QA, skipping improvement")

            feedback_history.append(track2_record)
            if step_dir is not None:
                _flush_improve_progress(step_dir, feedback_history, beliefs, perception)

            # Rebuild steps_context if perception changed during Track 2
            if perception != pre_track_perception:
                steps_context = format_steps_context(
                    trajectory_buffer, perception, eb_config.max_steps_context_chars,
                )

    except Exception as e:
        evolve_logger.error(f"{tag}     Improve loop failed: {e}")
        logging.exception("Improve loop failed")
        feedback_history.append({"error": str(e), "step": step, "global_step": global_step})
        if step_dir is not None:
            _flush_improve_progress(step_dir, feedback_history, beliefs, perception)

    return beliefs, perception, qa_pairs, total_cost, feedback_history


# ---------------------------------------------------------------------------
# Core per-step episode loop
# ---------------------------------------------------------------------------


def run_stepwise_eb_learn_episode(
    config: DictConfig,
    eb_config: StepwiseEBLearnConfig,
    beliefs: str,
    perception: str,
    qa_pairs: list[EBQAPair],
    current_experiment: str | None,
    default_knowledge: str,
    default_actions: str,
    original_cwd: str,
    output_dir: str,
    episode_idx: int = 0,
    global_step_start: int = 0,
    max_episode_steps: int | None = None,
    trajectory_buffer: list[dict] | None = None,
    past_experiments: list[str] | None = None,
    cumulative_cost_offset: float = 0.0,
) -> tuple[str, str, list[EBQAPair], str | None, dict, int, list[dict], list[str]]:
    """Run a single episode with per-step EB-learning.

    Returns:
        (beliefs, perception, qa_pairs, current_experiment, episode_stats, steps_taken,
         trajectory_buffer, past_experiments)
    """
    # --- Setup environment and agent ---
    env_name = config.envs.names.split("-")[0]
    tasks = config.tasks[f"{env_name}_tasks"]
    task = tasks[0]

    env = make_env(env_name, task, config)
    agent_factory = AgentFactory(config)
    agent = agent_factory.create_agent()
    agent.reset()

    seed = config.envs.env_kwargs.seed
    if seed is None:
        seed = get_unique_seed(process_num=0, episode_idx=episode_idx)
    random.seed(seed)
    np.random.seed(seed)
    obs, info = env.reset(seed=seed)

    # Inform agent of death/respawn if this is not the first episode
    if episode_idx > 0:
        obs["text"]["short_term_context"] = (
            "The previous episode was terminated and you have respawned.\n\n"
            + obs["text"]["short_term_context"]
        )

    # Setup instruction prompt with beliefs
    _inject_beliefs(config, agent, env, env_name, task, beliefs)
    agent.experiment_goal = current_experiment

    # Save raw initial obs before apply_perception modifies long_term_context in-place
    _pre_action_raw_long = obs["text"]["long_term_context"]
    _pre_action_raw_short = obs["text"].get("short_term_context", "")

    # Setup perception
    perception_fn = load_perception_fn(perception)
    if perception_fn is not None:
        apply_perception(obs, perception_fn)

    # Build initial obs_text (with perception applied) for the first buffer entry
    _pre_action_obs_text = _compose_obs_text(
        obs["text"]["short_term_context"],
        obs["text"]["long_term_context"],
    )

    # Episode tracking
    max_steps = env.max_steps if config.eval.get("max_steps_per_episode") is None else config.eval.max_steps_per_episode
    if max_episode_steps is not None:
        max_steps = min(max_steps, max_episode_steps)
    episode_log: dict = {
        "task": task,
        "action_frequency": defaultdict(int),
        "input_tokens": 0,
        "output_tokens": 0,
        "total_cost": 0.0,
    }

    trajectory_buffer = trajectory_buffer if trajectory_buffer is not None else []
    # Insert episode boundary marker
    if trajectory_buffer:
        trajectory_buffer.append({
            "step": None,
            "episode_boundary": True,
            "episode_idx": episode_idx,
            "obs_text": "",
            "raw_long_term_context": "",
            "action": None,
            "reward": 0.0,
            "reasoning": "",
            "done": True,
        })
    total_learn_cost = 0.0
    episode_return = 0.0
    cumulative_step_cost = 0.0
    step_extraction_log: dict | None = None
    step_experiment_log: dict | None = None
    past_experiments = past_experiments if past_experiments is not None else []
    if current_experiment and current_experiment not in past_experiments:
        past_experiments.append(current_experiment)

    # CSV logging
    ep_dir = Path(output_dir)
    ep_dir.mkdir(parents=True, exist_ok=True)
    csv_filename = ep_dir / "trajectory.csv"

    pbar = tqdm(total=max_steps, desc=f"Stepwise EB-learn ep {episode_idx}", leave=False, dynamic_ncols=True)
    feedback_history: list[dict] = []

    with open(csv_filename, mode="w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file, escapechar="\u02d8", quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["Step", "Action", "Reasoning", "Observation", "Auxiliary_Observation", "Reward", "Done"])

        action = None
        step = 0
        result_obs_text: str | None = None
        new_raw_short: str = ""
        done = False

        for step in range(max_steps):
            global_step = global_step_start + step

            # Per-step directory
            step_dir = ep_dir / f"step_{step:03d}"
            step_dir.mkdir(parents=True, exist_ok=True)

            step_extract_cost = 0.0
            step_improve_cost = 0.0
            step_experiment_cost = 0.0
            step_trim_cost = 0.0
            step_extraction_log = None
            step_experiment_log = None
            step_trim_log: dict | None = None
            did_trim_step = False
            step_feedback_records: list[dict] = []
            num_unanswered = sum(1 for q in qa_pairs if q.answer is None)

            # Write preliminary step_log
            _save_step_log_eb(
                step_dir=step_dir, step=step, global_step=global_step,
                action=None, reward=0.0, done=False, episode_return=episode_return,
                agent_cost=0.0, extract_cost=0.0, improve_cost=0.0, experiment_cost=0.0,
                trim_cost=0.0,
                num_qa=len(qa_pairs), num_unanswered=num_unanswered,
                did_gen_questions=False, did_formulate_experiment=False,
                active_experiment=current_experiment, phase="started",
            )

            # --- Question generation + experiment formulation ---
            should_gen_experiments = step % eb_config.experiment_interval == 0
            did_gen_questions = False
            did_formulate_experiment = False

            if should_gen_experiments:
                evolve_logger.info(f"[g{global_step}] Generating questions...")
                exp_steps_context = format_steps_context(
                    trajectory_buffer, perception, eb_config.max_steps_context_chars,
                )
                with improve_logging(step_dir):
                    # Step 1: Generate questions
                    new_questions, q_cost, q_prompt, q_response = asyncio.run(
                        generate_questions_from_steps(
                            config=config,
                            beliefs=beliefs,
                            perception_code=perception,
                            steps_context=exp_steps_context,
                            current_qa=qa_pairs,
                            current_observation=_pre_action_raw_long,
                            current_aux_observation=_pre_action_raw_short,
                            default_knowledge=default_knowledge,
                            num_questions=eb_config.num_questions,
                            current_step=global_step,
                        )
                    )
                    qa_pairs.extend(new_questions)
                    step_experiment_cost += q_cost
                    total_learn_cost += q_cost
                    did_gen_questions = True

                    evolve_logger.info(
                        f"[g{global_step}] Generated {len(new_questions)} questions — cost: ${q_cost:.6f}"
                    )

                    # Step 2: Formulate experiment from question
                    evolve_logger.info(f"[g{global_step}] Formulating experiment from questions...")
                    experiment_plan, q_idx, e_cost, e_prompt, e_response = asyncio.run(
                        formulate_experiment_from_question(
                            config=config,
                            beliefs=beliefs,
                            perception_code=perception,
                            steps_context=exp_steps_context,
                            current_qa=qa_pairs,
                            current_experiment=current_experiment,
                            current_observation=_pre_action_raw_long,
                            current_aux_observation=_pre_action_raw_short,
                            default_knowledge=default_knowledge,
                        )
                    )
                    step_experiment_cost += e_cost
                    total_learn_cost += e_cost

                    if experiment_plan is not None:
                        # Move old active experiment to past if it exists
                        if current_experiment and current_experiment not in past_experiments:
                            past_experiments.append(current_experiment)
                        current_experiment = experiment_plan
                        did_formulate_experiment = True

                    step_experiment_log = {
                        "question_gen_prompt": q_prompt,
                        "question_gen_response": q_response,
                        "new_questions": [q.question for q in new_questions],
                        "experiment_prompt": e_prompt,
                        "experiment_response": e_response,
                        "experiment_plan": experiment_plan,
                        "selected_question_index": q_idx,
                    }

                # Update current experiment and inject into agent
                agent.experiment_goal = current_experiment

                # Write experiment artifacts immediately
                with open(step_dir / "experiment_log.json", "w") as f:
                    json.dump(step_experiment_log, f, indent=4, default=str)
                with open(step_dir / "qa_pairs.json", "w") as f:
                    json.dump(serialize_eb_qa_pairs(qa_pairs), f, indent=4)

                evolve_logger.info(
                    f"[g{global_step}] Experiment: {'new' if did_formulate_experiment else 'kept'} — "
                    f"cost: ${step_experiment_cost:.6f}"
                )

            num_unanswered = sum(1 for q in qa_pairs if q.answer is None)

            # Update step_log: experiment phase done, agent about to act
            _save_step_log_eb(
                step_dir=step_dir, step=step, global_step=global_step,
                action=None, reward=0.0, done=False, episode_return=episode_return,
                agent_cost=0.0, extract_cost=0.0, improve_cost=0.0, experiment_cost=step_experiment_cost,
                trim_cost=0.0,
                num_qa=len(qa_pairs), num_unanswered=num_unanswered,
                did_gen_questions=did_gen_questions, did_formulate_experiment=did_formulate_experiment,
                active_experiment=current_experiment, phase="acting",
            )

            # --- Agent acts ---
            response = agent.act(obs, prev_action=action)
            action = response.completion
            reasoning = response.reasoning if hasattr(response, "reasoning") else ""

            episode_log["action_frequency"][action] += 1
            episode_log["input_tokens"] += response.input_tokens
            episode_log["output_tokens"] += response.output_tokens
            episode_log["total_cost"] += response.cost
            agent_step_cost = response.cost

            # --- Environment step ---
            try:
                obs, reward, terminated, truncated, info = env.step(action)
            except ValueError as e:
                logging.warning(f"[g{global_step}] Invalid action: {action} — {e}")
                if config.eval.feedback_on_invalid_action:
                    obs["text"]["long_term_context"] = (
                        f"\n\nYour previous output did not contain a valid action. Retry\n\n"
                        f"Observation:\n{obs['text']['long_term_context']}"
                    )
                terminated = False
                truncated = False
                reward = 0.0

            done = terminated or truncated
            episode_return += reward

            # Save raw new obs BEFORE applying perception
            new_raw_long = obs["text"]["long_term_context"]
            new_raw_short = obs["text"].get("short_term_context", "")

            # Apply perception to new obs
            if perception_fn is not None:
                apply_perception(obs, perception_fn)

            result_obs_text = _compose_obs_text(
                obs["text"]["short_term_context"],
                obs["text"]["long_term_context"],
            )

            # Capture agent messages
            try:
                agent_messages = [
                    {"role": m.role, "content": m.content}
                    for m in agent.last_messages
                ]
            except Exception:
                agent_messages = []
            agent_messages.append({
                "role": "assistant",
                "content": reasoning,
                "action": action,
            })

            with open(step_dir / "agent_messages.json", "w") as amf:
                json.dump(agent_messages, amf, indent=2, default=str)

            # Write CSV row immediately
            csv_writer.writerow([step, action, reasoning, _pre_action_obs_text, _pre_action_raw_short, reward, done])
            csv_file.flush()

            # Update step_log with action/reward/done
            _save_step_log_eb(
                step_dir=step_dir, step=step, global_step=global_step,
                action=action, reward=reward, done=done, episode_return=episode_return,
                agent_cost=agent_step_cost, extract_cost=0.0, improve_cost=0.0, experiment_cost=step_experiment_cost,
                trim_cost=0.0,
                num_qa=len(qa_pairs), num_unanswered=num_unanswered,
                did_gen_questions=did_gen_questions, did_formulate_experiment=did_formulate_experiment,
                active_experiment=current_experiment, phase="extracting",
            )

            # Append buffer entry
            trajectory_buffer.append({
                "step": global_step,
                "obs_text": _pre_action_obs_text,
                "raw_long_term_context": _pre_action_raw_long,
                "raw_short_term_context": _pre_action_raw_short,
                "result_raw_long_term_context": new_raw_long,
                "result_raw_short_term_context": new_raw_short,
                "action": action,
                "reward": reward,
                "reasoning": reasoning,
                "done": False,
            })

            if done:
                trajectory_buffer.append({
                    "step": global_step + 1,
                    "obs_text": result_obs_text,
                    "raw_long_term_context": new_raw_long,
                    "raw_short_term_context": new_raw_short,
                    "action": None,
                    "reward": 0.0,
                    "reasoning": "",
                    "done": True,
                })

            pbar.update(1)

            evolve_logger.info(
                f"[g{global_step}|ep{episode_idx}|s{step}] "
                f"action={action!r}  reward={reward:.2f}  return={episode_return:.2f}  "
                f"done={done}  agent_cost=${agent_step_cost:.6f}"
            )

            # --- Determine what to do this step ---
            steps_in = step + 1
            should_update_artifacts = (
                (steps_in % eb_config.artifact_update_interval == 0)
                or done
            )
            should_improve = (
                (steps_in % eb_config.improve_interval == 0)
                or done
            )

            # --- Artifact update (update Q from trajectory) ---
            if should_update_artifacts and len(trajectory_buffer) > 0:
                evolve_logger.info(f"[g{global_step}] Updating Q from {len(trajectory_buffer)} buffered steps...")
                steps_context = format_steps_context(
                    trajectory_buffer, perception, eb_config.max_steps_context_chars,
                )
                with improve_logging(step_dir):
                    qa_pairs, extract_cost, step_extraction_log = asyncio.run(
                        update_qa_from_trajectory(
                            config=config,
                            current_qa=qa_pairs,
                            steps_context=steps_context,
                            max_total_qa_pairs=eb_config.max_total_qa_pairs,
                            current_step=global_step,
                        )
                    )
                    step_extract_cost = extract_cost
                    total_learn_cost += extract_cost

                # Write extraction artifacts immediately
                with open(step_dir / "extraction_log.json", "w") as f:
                    json.dump(step_extraction_log, f, indent=4, default=str)
                with open(step_dir / "qa_pairs.json", "w") as f:
                    json.dump(serialize_eb_qa_pairs(qa_pairs), f, indent=4)

                num_unanswered = sum(1 for q in qa_pairs if q.answer is None)
                evolve_logger.info(
                    f"[g{global_step}] Q update done — "
                    f"QA: {len(qa_pairs)} ({num_unanswered} unanswered), cost: ${extract_cost:.6f}"
                )

                # --- Trim Q if over limit ---
                if len(qa_pairs) > eb_config.max_total_qa_pairs:
                    evolve_logger.info(
                        f"[g{global_step}] Trimming Q: {len(qa_pairs)} > {eb_config.max_total_qa_pairs}..."
                    )
                    with improve_logging(step_dir):
                        qa_pairs, trim_cost_val, step_trim_log = asyncio.run(
                            trim_qa_pairs(
                                config=config,
                                current_qa=qa_pairs,
                                max_total_qa_pairs=eb_config.max_total_qa_pairs,
                                current_step=global_step,
                            )
                        )
                        step_trim_cost = trim_cost_val
                        total_learn_cost += trim_cost_val
                        did_trim_step = True

                    with open(step_dir / "trim_log.json", "w") as f:
                        json.dump(step_trim_log, f, indent=4, default=str)
                    with open(step_dir / "qa_pairs.json", "w") as f:
                        json.dump(serialize_eb_qa_pairs(qa_pairs), f, indent=4)

                    num_unanswered = sum(1 for q in qa_pairs if q.answer is None)
                    evolve_logger.info(
                        f"[g{global_step}] Trim done — "
                        f"QA: {len(qa_pairs)} ({num_unanswered} unanswered), cost: ${trim_cost_val:.6f}"
                    )

                # Update step_log: extraction done, improve starting
                _save_step_log_eb(
                    step_dir=step_dir, step=step, global_step=global_step,
                    action=action, reward=reward, done=done, episode_return=episode_return,
                    agent_cost=agent_step_cost, extract_cost=step_extract_cost, improve_cost=0.0,
                    experiment_cost=step_experiment_cost, trim_cost=step_trim_cost,
                    num_qa=len(qa_pairs), num_unanswered=num_unanswered,
                    did_gen_questions=did_gen_questions, did_formulate_experiment=did_formulate_experiment,
                    did_trim=did_trim_step,
                    active_experiment=current_experiment, phase="improving",
                )

            # --- Improve loop (beliefs/perception + QA) ---
            if should_improve:
                _perc_iters = _resolve_schedule(eb_config.max_perception_iterations, global_step)
                _qa_iters = _resolve_schedule(eb_config.max_qa_iterations, global_step)
                evolve_logger.info(
                    f"[g{global_step}] Running improve loop (perception={_perc_iters}, "
                    f"qa={_qa_iters} iters)..."
                )
                pre_improve_perception = perception
                with improve_logging(step_dir):
                    beliefs, perception, qa_pairs, improve_cost, iter_records = _run_improve_loop_eb(
                        config=config,
                        eb_config=eb_config,
                        beliefs=beliefs,
                        perception=perception,
                        qa_pairs=qa_pairs,
                        trajectory_buffer=trajectory_buffer,
                        default_knowledge=default_knowledge,
                        step=step,
                        global_step=global_step,
                        step_dir=step_dir,
                    )
                    step_improve_cost = improve_cost
                    total_learn_cost += improve_cost
                    step_feedback_records = iter_records
                    feedback_history.extend(iter_records)

                perception_changed = perception != pre_improve_perception

                # Reload perception after improvement
                perception_fn = load_perception_fn(perception)

                # Re-apply updated perception to current obs for the agent's next step
                if not done:
                    obs["text"]["long_term_context"] = new_raw_long
                    obs["text"]["short_term_context"] = new_raw_short
                    if perception_fn is not None:
                        apply_perception(obs, perception_fn)

                # Rebuild all buffered observations with the latest perception
                if perception_changed:
                    _refresh_buffer_with_perception(trajectory_buffer, perception_fn)

                # Inject updated beliefs for next step
                if not done:
                    _inject_beliefs(config, agent, env, env_name, task, beliefs)

                evolve_logger.info(
                    f"[g{global_step}] Improve done — cost: ${improve_cost:.6f}"
                )

                # Write qa immediately after improve
                with open(step_dir / "qa_pairs.json", "w") as f:
                    json.dump(serialize_eb_qa_pairs(qa_pairs), f, indent=4)

            # --- Carry forward pre-action vars ---
            if not done:
                _pre_action_raw_long = new_raw_long
                _pre_action_raw_short = new_raw_short
                _pre_action_obs_text = _compose_obs_text(
                    obs["text"]["short_term_context"],
                    obs["text"]["long_term_context"],
                )

            # --- Per-step artifact save ---
            step_total_cost = agent_step_cost + step_extract_cost + step_improve_cost + step_experiment_cost + step_trim_cost
            cumulative_step_cost += step_total_cost

            did_learn = should_update_artifacts or should_improve or should_gen_experiments
            if did_learn:
                _save_step_artifacts_eb(
                    step_dir, beliefs, perception, qa_pairs,
                    step_feedback_records,
                    extraction_log=step_extraction_log,
                    experiment_log=step_experiment_log,
                    trim_log=step_trim_log,
                )

            num_unanswered = sum(1 for q in qa_pairs if q.answer is None)
            _save_step_log_eb(
                step_dir=step_dir, step=step, global_step=global_step,
                action=action, reward=reward, done=done, episode_return=episode_return,
                agent_cost=agent_step_cost, extract_cost=step_extract_cost,
                improve_cost=step_improve_cost, experiment_cost=step_experiment_cost,
                trim_cost=step_trim_cost,
                num_qa=len(qa_pairs), num_unanswered=num_unanswered,
                did_gen_questions=did_gen_questions, did_formulate_experiment=did_formulate_experiment,
                did_trim=did_trim_step,
                active_experiment=current_experiment, phase="complete",
            )

            # Per-step summary update
            _update_summary_json(
                output_dir=os.path.dirname(output_dir),
                step=global_step,
                step_cost=step_total_cost,
                cumulative_cost=cumulative_cost_offset + cumulative_step_cost,
                rollout_stats={
                    "episode_idx": episode_idx,
                    "episode_step": step,
                    "action": action,
                    "reward": reward,
                    "episode_return": episode_return,
                    "done": done,
                    "num_qa_pairs": len(qa_pairs),
                    "num_unanswered_questions": num_unanswered,
                    "did_extract": should_update_artifacts,
                    "did_improve": should_improve,
                    "did_gen_questions": did_gen_questions,
                    "did_formulate_experiment": did_formulate_experiment,
                    "did_trim": did_trim_step,
                },
            )

            if done:
                evolve_logger.info(
                    f"[g{global_step}] Episode {episode_idx} DONE — "
                    f"return={episode_return:.2f}, steps={step + 1}"
                )
                if pbar.n < pbar.total:
                    pbar.update(pbar.total - pbar.n)
                pbar.set_postfix_str("DONE")
                break

        # Write terminal row with the post-action state from the last completed step,
        # regardless of whether the episode ended via `done` or by hitting max_steps.
        # This lets the viewer show the state *after* the final action.
        if result_obs_text is not None:
            csv_writer.writerow([step + 1, "", "", result_obs_text, new_raw_short, 0.0, done])
            csv_file.flush()

    if pbar.n < pbar.total:
        pbar.update(pbar.total - pbar.n)
    pbar.close()

    # Finalize episode stats
    episode_log["episode_return"] = episode_return
    episode_log["num_steps"] = step + 1
    episode_log["failed_candidates"] = env.failed_candidates
    episode_log.update(env.get_stats())
    episode_log["seed"] = seed
    episode_log["total_learn_cost"] = total_learn_cost
    episode_log["cumulative_step_cost"] = cumulative_step_cost
    episode_log["num_qa_pairs"] = len(qa_pairs)
    episode_log["num_unanswered_questions"] = sum(1 for q in qa_pairs if q.answer is None)


    json_filename = ep_dir / "episode_log.json"
    with open(json_filename, "w") as f:
        json.dump(episode_log, f, indent=4, default=str)

    env.close()

    evolve_logger.info(
        f"Episode {episode_idx} complete — return: {episode_return:.2f}, "
        f"steps: {step + 1}, "
        f"learn cost: ${total_learn_cost:.4f}, agent cost: ${episode_log['total_cost']:.4f}"
    )

    return beliefs, perception, qa_pairs, current_experiment, episode_log, step + 1, trajectory_buffer, past_experiments


# ---------------------------------------------------------------------------
# Outer orchestrator
# ---------------------------------------------------------------------------


def stepwise_eb_learn(
    eb_config: StepwiseEBLearnConfig,
    config: DictConfig,
    original_cwd: str,
    output_dir: str,
):
    """Run stepwise EB-learning: experiment-driven per-step improvement across episodes."""
    evolve_logger.info("Starting stepwise EB-learning")

    # Check for resume
    last_ep, beliefs, perception, qa_pairs = _find_last_completed_episode_eb(output_dir)
    start_episode = last_ep + 1

    current_experiment: str | None = None
    trajectory_buffer: list[dict] = []
    past_experiments: list[str] = []
    global_steps_used = 0
    if start_episode > 0:
        num_unanswered = sum(1 for q in qa_pairs if q.answer is None)
        evolve_logger.info(f"Resuming from episode {start_episode} ({len(qa_pairs)} QA, {num_unanswered} unanswered)")
        # Recover active experiment from last episode's step logs
        last_ep_dir = Path(output_dir) / f"episode_{last_ep}"
        for step_dir in sorted(last_ep_dir.glob("step_*"), reverse=True):
            sl_file = step_dir / "step_log.json"
            if sl_file.exists():
                try:
                    sl = json.loads(sl_file.read_text())
                    current_experiment = sl.get("active_experiment")
                    break
                except (json.JSONDecodeError, TypeError):
                    pass
        # Recover global step count from episode logs
        for ep_idx in range(start_episode):
            ep_log_file = Path(output_dir) / f"episode_{ep_idx}" / "episode_log.json"
            if ep_log_file.exists():
                try:
                    ep_data = json.loads(ep_log_file.read_text())
                    global_steps_used += ep_data.get("num_steps", 0)
                except (json.JSONDecodeError, TypeError):
                    pass
        # Recover trajectory buffer and past experiments from last episode
        traj_file = Path(output_dir) / f"episode_{last_ep}" / "trajectory_buffer.json"
        if traj_file.exists():
            try:
                trajectory_buffer = json.loads(traj_file.read_text())
            except (json.JSONDecodeError, TypeError):
                trajectory_buffer = []
        past_exp_file = Path(output_dir) / f"episode_{last_ep}" / "past_experiments.json"
        if past_exp_file.exists():
            try:
                past_experiments = json.loads(past_exp_file.read_text())
            except (json.JSONDecodeError, TypeError):
                past_experiments = []
    else:
        # Load initial state
        if (beliefs_path := config.eval.get("beliefs_path", None)) is not None:
            beliefs = Path(beliefs_path).read_text()
            evolve_logger.info(f"Loaded initial beliefs from: {beliefs_path}")
        else:
            beliefs = ""
        if (perception_path := config.eval.get("perception_path", None)) is not None:
            perception = Path(perception_path).read_text()
            evolve_logger.info(f"Loaded initial perception from: {perception_path}")
        else:
            perception = ""
        qa_pairs = []

    default_knowledge = get_default_knowledge(config)
    evolve_logger.info(f"Default knowledge: {len(default_knowledge)} chars")
    default_actions = get_default_actions(config)
    evolve_logger.info(f"Default actions: {len(default_actions)} chars")

    evolve_logger.info(f"Stepwise EB-learn config:")
    evolve_logger.info(f"  Total env steps: {eb_config.n_environment_steps}")
    evolve_logger.info(f"  Improve iterations: perception={eb_config.max_perception_iterations}, qa={eb_config.max_qa_iterations} (schedule or fixed)")
    evolve_logger.info(f"  Artifact update interval: {eb_config.artifact_update_interval}")
    evolve_logger.info(f"  Improve interval: {eb_config.improve_interval}")
    evolve_logger.info(f"  Experiment interval: {eb_config.experiment_interval}")
    evolve_logger.info(f"  Num questions per gen: {eb_config.num_questions}")
    evolve_logger.info(f"  Max steps context chars: {eb_config.max_steps_context_chars}")
    evolve_logger.info(f"  Explore temp: {eb_config.explore_temp}")

    cumulative_cost = 0.0
    episode_idx = start_episode

    while global_steps_used < eb_config.n_environment_steps:
        remaining_steps = eb_config.n_environment_steps - global_steps_used

        num_unanswered = sum(1 for q in qa_pairs if q.answer is None)
        evolve_logger.info(f"\n{'='*80}")
        evolve_logger.info(
            f"STEPWISE EB-LEARN EPISODE {episode_idx} "
            f"(global steps: {global_steps_used}/{eb_config.n_environment_steps}, "
            f"remaining: {remaining_steps})"
        )
        evolve_logger.info(f"QA pairs: {len(qa_pairs)} ({num_unanswered} unanswered), Experiment: {current_experiment or 'none'}")
        evolve_logger.info(f"{'='*80}")

        episode_dir = Path(output_dir) / f"episode_{episode_idx}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Save input state
        (episode_dir / "input_beliefs.txt").write_text(beliefs)
        (episode_dir / "input_perception.py").write_text(perception)
        with override_temperature(config, eb_config.explore_temp):
            beliefs, perception, qa_pairs, current_experiment, episode_log, steps_taken, trajectory_buffer, past_experiments = (
                run_stepwise_eb_learn_episode(
                    config=config,
                    eb_config=eb_config,
                    beliefs=beliefs,
                    perception=perception,
                    qa_pairs=qa_pairs,
                    current_experiment=current_experiment,
                    default_knowledge=default_knowledge,
                    default_actions=default_actions,
                    original_cwd=original_cwd,
                    output_dir=str(episode_dir),
                    episode_idx=episode_idx,
                    global_step_start=global_steps_used,
                    max_episode_steps=remaining_steps,
                    trajectory_buffer=trajectory_buffer,
                    past_experiments=past_experiments,
                    cumulative_cost_offset=cumulative_cost,
                )
            )

        global_steps_used += steps_taken

        # Save episode artifacts
        _save_episode_artifacts_eb(
            episode_dir, beliefs, perception, qa_pairs,
            trajectory_buffer=trajectory_buffer,
            past_experiments=past_experiments,
        )

        episode_cost = episode_log.get("total_cost", 0.0) + episode_log.get("total_learn_cost", 0.0)
        cumulative_cost += episode_cost
        evolve_logger.info(
            f"[g{global_steps_used}] Episode {episode_idx} done — "
            f"cost: ${episode_cost:.4f}, cumulative: ${cumulative_cost:.4f}, "
            f"steps: {global_steps_used}/{eb_config.n_environment_steps}"
        )

        episode_idx += 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@hydra.main(config_path="BALROG/balrog/config", config_name="config", version_base="1.1")
def main(config: DictConfig):
    run_name_suffix = f"{config.agent.type}_{config.client.model_id.replace('/', '_')}_stepwise_eb_learn"

    original_cwd, output_dir = setup_run(
        config,
        run_name_suffix=run_name_suffix,
        resume_from=config.eval.resume_from,
        output_dir_base=config.eval.output_dir,
        logger_name="evolve",
    )

    evolve_cfg = config.eval.evolve

    eb_config = StepwiseEBLearnConfig(
        n_environment_steps=evolve_cfg.get("n_environment_steps", 100),
        max_perception_iterations=evolve_cfg.get(
            "max_perception_iterations",
            evolve_cfg.get("max_improve_iterations", evolve_cfg.get("num_improve_iterations", 5)),
        ),
        max_qa_iterations=evolve_cfg.get(
            "max_qa_iterations",
            evolve_cfg.get("max_improve_iterations", evolve_cfg.get("num_improve_iterations", 5)),
        ),
        max_qa_per_forward=evolve_cfg.get("max_qa_per_forward", 10),
        max_total_qa_pairs=evolve_cfg.get("max_total_qa_pairs", 50),
        num_questions=evolve_cfg.get("num_questions", 5),
        num_sample_obs=evolve_cfg.get("num_sample_obs", 3),
        explore_temp=evolve_cfg.get("explore_temp", 1.0),
        artifact_update_interval=evolve_cfg.get("artifact_update_interval", 1),
        improve_interval=evolve_cfg.get("improve_interval", 5),
        experiment_interval=evolve_cfg.get("experiment_interval", 10),
        max_steps_context_chars=evolve_cfg.get("max_steps_context_chars", 50000),
    )

    stepwise_eb_learn(
        eb_config=eb_config,
        config=config,
        original_cwd=original_cwd,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
