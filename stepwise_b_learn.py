"""Stepwise B-learning: learn after each environment step using 3-track improvement.

Combines b_learn.py's 3-track improve loop (steps-based, QA, critical moments)
with stepwise_explore.py's per-step environment interaction. Instead of collecting
full episodes then improving, this runs improvement after each environment step.
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
    CriticalMoment,
    extract_qa_and_moments_from_trajectory,
    consolidate_qa_and_moments,
    serialize_qa_pairs,
    deserialize_qa_pairs,
    serialize_moments,
    deserialize_moments,
    extract_perception_input,
    _run_perception_on_observation,
)
from b_learn_improve import (
    forward_pass,
    get_feedback,
    qa_forward_pass,
    qa_get_feedback,
    serialize_feedback_results,
    serialize_qa_feedback_results,
)
from stepwise_b_learn_improve import (
    improve_from_steps,
    improve_from_feedback_with_steps,
    improve_from_qa_feedback_with_steps,
    generate_experiments_from_steps,
    parse_submit_signal,
    build_steps_followup_message,
    build_perception_followup_message,
    build_perception_with_analysis_prompt,
    build_qa_followup_message,
    build_moments_followup_message,
    _build_obs_section,
    _build_execution_report_section,
    _extract_obs_message,
    PERCEPTION_INSTRUCTIONS,
    RESPONSE_FORMAT,
    BELIEFS_ONLY_RESPONSE_FORMAT,
)
from b_learn_improve import (
    _improve_with_perception_validation_conversational,
    _improve_beliefs_only_conversational,
)
from llm_utils import extract_xml_key
from stepwise_explore import (
    load_perception_fn,
    apply_perception,
    format_segment_text,
    find_last_completed_episode,
    _save_episode_artifacts,
)
from run_utils import setup_run, improve_logging, _update_summary_json


@dataclass
class StepwiseBLearnConfig:
    n_environment_steps: int
    max_steps_iterations: int
    max_perception_iterations: int
    use_qa: bool
    use_moments: bool
    max_qa_iterations: int
    max_moments_iterations: int
    max_moments_per_forward: int
    max_qa_per_forward: int
    max_total_moments: int
    max_total_qa_pairs: int
    num_experiments: int
    num_sample_obs: int
    explore_temp: float
    artifact_update_interval: int
    improve_interval: int
    experiment_interval: int
    max_steps_context_chars: int
    conversational_improve: bool


# ---------------------------------------------------------------------------
# Step sequence formatting
# ---------------------------------------------------------------------------


def format_steps_context(
    trajectory_buffer: list[dict],
    perception_code: str,
    max_chars: int,
) -> str:
    """Format the trajectory buffer as a step sequence for improvement prompts.

    Produces: s0, a0, s1, a1, ..., s_{t-1}, a_{t-1}, s_t — where each intermediate
    s_{n+1} is shown as the next step's State, and s_t (the current state after the
    last action) is appended as "Resulting state" on the final step.
    Trimmed to longest suffix that fits max_chars.
    """
    if not trajectory_buffer:
        return ""

    # Build formatted blocks for each step (newest last)
    # Find the last non-terminal entry index so we can append its result state
    last_action_idx = max(
        (i for i, e in enumerate(trajectory_buffer) if e.get("action") is not None),
        default=-1,
    )

    blocks = []
    for i, entry in enumerate(trajectory_buffer):
        if entry.get("episode_boundary"):
            blocks.append(f"<episode_boundary episode=\"{entry.get('episode_idx', '?')}\" />\n")
            continue

        raw_obs = entry.get("raw_long_term_context", "")
        raw_aux = entry.get("raw_short_term_context", "")
        action = entry.get("action")
        is_terminal = action is None
        reward = entry.get("reward", 0)

        if is_terminal:
            # Terminal entry is redundant: the resulting state is already shown on the
            # last action step with full raw obs + perception output.
            continue

        perc_out = _run_perception_on_observation(perception_code, raw_obs) if perception_code else ""
        reasoning = entry.get("reasoning", "")
        block = f"<step n=\"{entry['step']}\">\n"
        block += (
            f"<state>\n{raw_obs}\n</state>\n\n"
            f"<auxiliary_observation>\n{raw_aux}\n</auxiliary_observation>\n\n"
            f"<perception_output>\n{perc_out if perc_out else '(no perception module)'}\n</perception_output>\n\n"
            f"<agent_reasoning>\n{reasoning}\n</agent_reasoning>\n"
            f"<action>{action}</action>\n"
        )
        block += f"<reward>{reward}</reward>\n"
        # For the last action step, append the resulting state (s_t) so the
        # trajectory is complete: s0, a0, s1, a1, ..., s_{t-1}, a_{t-1}, s_t
        if i == last_action_idx:
            result_raw = entry.get("result_raw_long_term_context", "")
            result_raw_aux = entry.get("result_raw_short_term_context", "")
            if result_raw:
                result_perc = _run_perception_on_observation(perception_code, result_raw) if perception_code else ""
                block += (
                    f"\n<resulting_state>\n{result_raw}\n</resulting_state>\n\n"
                    f"<auxiliary_observation>\n{result_raw_aux}\n</auxiliary_observation>\n\n"
                    f"<perception_output>\n{result_perc if result_perc else '(no perception module)'}\n</perception_output>\n"
                )
        block += "</step>"
        blocks.append(block)

    # Trim from the front to fit within max_chars (keep latest steps)
    result = "\n".join(blocks)
    if len(result) <= max_chars:
        return result

    # Include as many recent steps as possible
    trimmed = []
    total_len = 0
    for block in reversed(blocks):
        if total_len + len(block) + 1 > max_chars:
            break
        trimmed.append(block)
        total_len += len(block) + 1

    trimmed.reverse()
    if not trimmed:
        # At least include the latest step, truncated
        return blocks[-1][:max_chars]
    return "\n".join(trimmed)


def _compose_obs_text(short_term_context: str, long_term_context: str) -> str:
    """Build the canonical observation text block used in trajectory buffers."""
    return (
        f"{short_term_context}\n\n"
        f"{'='*10} Start of Direct Observation {'='*10}\n"
        f"{long_term_context}\n\n"
        f"{'='*10} End of Direct Observation {'='*10}"
    )


def _refresh_buffer_with_perception(
    trajectory_buffer: list[dict],
    perception_fn,
) -> None:
    """Rebuild all buffered obs_text entries using the latest perception function."""
    for entry in trajectory_buffer:
        if entry.get("episode_boundary"):
            continue
        raw_long = entry.get("raw_long_term_context", "")
        raw_short = entry.get("raw_short_term_context", "")
        obs_like = {
            "text": {
                "long_term_context": raw_long,
                "short_term_context": raw_short,
            }
        }
        if perception_fn is not None:
            apply_perception(obs_like, perception_fn)
        entry["obs_text"] = _compose_obs_text(
            obs_like["text"]["short_term_context"],
            obs_like["text"]["long_term_context"],
        )


# ---------------------------------------------------------------------------
# Knowledge extraction from buffer
# ---------------------------------------------------------------------------


def _sample_observations_from_buffer(
    trajectory_buffer: list[dict],
    num_samples: int,
) -> list[tuple[str, int]]:
    """Sample observations from the trajectory buffer for perception validation.

    Picks start, middle, and recent steps (avoiding the very last).
    Returns list of (raw_observation, step_number).
    """
    if not trajectory_buffer:
        return []

    # Get raw observations, skip entries without raw obs
    valid = [
        (e["raw_long_term_context"], e["step"])
        for e in trajectory_buffer
        if e.get("raw_long_term_context", "").strip()
    ]
    if not valid:
        return []

    # Exclude last step (often end/death screen)
    if len(valid) > 1:
        valid = valid[:-1]

    if len(valid) <= num_samples:
        return valid

    # Pick evenly spaced indices
    picks = []
    for j in range(num_samples):
        idx = int(j * (len(valid) - 1) / (num_samples - 1))
        picks.append(valid[idx])
    return picks


async def extract_knowledge_from_buffer(
    config: DictConfig,
    trajectory_buffer: list[dict],
    beliefs: str,
    perception: str,
    existing_qa: list[QAPair],
    existing_moments: list[CriticalMoment],
    bl_config: StepwiseBLearnConfig,
    step: int,
    episode_idx: int,
    is_done: bool,
) -> tuple[list[QAPair], list[CriticalMoment], float, dict]:
    """Extract Q&A pairs and critical moments from the trajectory buffer.

    Returns: (updated_qa_pairs, updated_moments, cost, extraction_log)
    """
    total_cost = 0.0
    extraction_log = {}

    segment_text = format_steps_context(
        trajectory_buffer,
        perception,
        bl_config.max_steps_context_chars,
    )

    # Extract new Q&A and moments
    new_qa, new_moments, extract_cost, extract_prompt, extract_response = await extract_qa_and_moments_from_trajectory(
        config=config,
        traj_text=segment_text,
        trajectory_path="",
        step=step,
        use_qa=bl_config.use_qa,
        use_moments=bl_config.use_moments,
        existing_qa=existing_qa,
    )
    total_cost += extract_cost
    extraction_log["extract_prompt"] = extract_prompt
    extraction_log["extract_response"] = extract_response
    extraction_log["new_qa_count"] = len(new_qa)
    extraction_log["new_moments_count"] = len(new_moments)

    # Attach raw observations from buffer to new moments
    step_to_raw = {e["step"]: e.get("raw_long_term_context", "") for e in trajectory_buffer if not e.get("episode_boundary")}
    for moment in new_moments:
        if moment.traj_step_number >= 0 and moment.traj_step_number in step_to_raw:
            moment.raw_observation = step_to_raw[moment.traj_step_number]

    # Consolidate with existing knowledge
    if new_qa or new_moments:
        (
            kept_qa,
            kept_moments,
            remove_qa_indices,
            remove_moment_indices,
            consolidate_cost,
        ) = await consolidate_qa_and_moments(
            config=config,
            new_per_traj_qa=[new_qa],
            new_per_traj_moments=[new_moments],
            existing_qa=existing_qa,
            existing_moments=existing_moments,
        )
        total_cost += consolidate_cost

        # Apply removals
        if remove_qa_indices:
            existing_qa = [
                qa for i, qa in enumerate(existing_qa) if i not in set(remove_qa_indices)
            ]
        if remove_moment_indices:
            existing_moments = [
                m for i, m in enumerate(existing_moments) if i not in set(remove_moment_indices)
            ]

        existing_qa = existing_qa + kept_qa
        existing_moments = existing_moments + kept_moments

        evolve_logger.info(
            f"  Knowledge extraction (step {step}): "
            f"+{len(kept_qa)} QA (total {len(existing_qa)}), "
            f"+{len(kept_moments)} moments (total {len(existing_moments)})"
        )

    # Cap totals
    if len(existing_qa) > bl_config.max_total_qa_pairs:
        existing_qa.sort(key=lambda q: q.source_step)
        dropped = len(existing_qa) - bl_config.max_total_qa_pairs
        existing_qa = existing_qa[dropped:]
        evolve_logger.info(f"  Capped QA pairs: dropped {dropped} oldest")

    if len(existing_moments) > bl_config.max_total_moments:
        existing_moments.sort(key=lambda m: m.source_step)
        dropped = len(existing_moments) - bl_config.max_total_moments
        existing_moments = existing_moments[dropped:]
        evolve_logger.info(f"  Capped moments: dropped {dropped} oldest")

    return existing_qa, existing_moments, total_cost, extraction_log


# ---------------------------------------------------------------------------
# Core per-step episode loop
# ---------------------------------------------------------------------------


def _flush_improve_progress(step_dir: Path, feedback_history: list[dict], beliefs: str, perception: str) -> None:
    """Write in-progress improve state to disk so the visualizer can show live updates."""
    (step_dir / "beliefs.txt").write_text(beliefs)
    (step_dir / "perception.py").write_text(perception)
    with open(step_dir / "feedback_history.json", "w") as f:
        json.dump(feedback_history, f, indent=4, default=str)


def _save_step_artifacts(
    step_dir: Path,
    beliefs: str,
    perception: str,
    qa_pairs: list[QAPair],
    moments: list[CriticalMoment],
    experiments: list[str],
    feedback_history: list[dict],
    extraction_log: dict | None = None,
    experiment_log: dict | None = None,
):
    """Save all artifacts for a completed step."""
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / "beliefs.txt").write_text(beliefs)
    (step_dir / "perception.py").write_text(perception)
    with open(step_dir / "qa_pairs.json", "w") as f:
        json.dump(serialize_qa_pairs(qa_pairs), f, indent=4)
    with open(step_dir / "critical_moments.json", "w") as f:
        json.dump(serialize_moments(moments), f, indent=4)
    with open(step_dir / "experiments.json", "w") as f:
        json.dump(experiments, f, indent=4)
    if feedback_history:
        with open(step_dir / "feedback_history.json", "w") as f:
            json.dump(feedback_history, f, indent=4, default=str)
    if extraction_log:
        with open(step_dir / "extraction_log.json", "w") as f:
            json.dump(extraction_log, f, indent=4, default=str)
    if experiment_log:
        with open(step_dir / "experiment_log.json", "w") as f:
            json.dump(experiment_log, f, indent=4, default=str)


def _save_step_log(
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
    num_qa: int,
    num_moments: int,
    num_experiments: int,
    did_gen_experiments: bool = False,
    active_experiment: str | None = None,
    phase: str = "complete",
):
    """Write a per-step JSON log with action, costs, and artifact counts.

    Called multiple times per step so the visualizer can show live progress:
      phase="started"    — step directory created, before any work
      phase="acting"     — experiment generation done, agent is about to act
      phase="extracting" — agent acted and env stepped; extraction starting
      phase="improving"  — extraction done; improve loop starting
      phase="complete"   — all phases done, costs are final
    """
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
        "step_total_cost": agent_cost + extract_cost + improve_cost + experiment_cost,
        "num_qa_pairs": num_qa,
        "num_moments": num_moments,
        "num_experiments": num_experiments,
        # Experiment generation now happens at the START of each step (before agent acts).
        # did_gen_experiments=True means experiments were (re-)generated at the top of this step.
        # active_experiment is the experiment the agent was given when it acted this step.
        "did_gen_experiments": did_gen_experiments,
        "active_experiment": active_experiment,
    }
    with open(step_dir / "step_log.json", "w") as f:
        json.dump(step_log, f, indent=4)


def run_stepwise_b_learn_episode(
    config: DictConfig,
    bl_config: StepwiseBLearnConfig,
    beliefs: str,
    perception: str,
    qa_pairs: list[QAPair],
    moments: list[CriticalMoment],
    experiments: list[str],
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
) -> tuple[str, str, list[QAPair], list[CriticalMoment], list[str], dict, int, list[dict], list[str]]:
    """Run a single episode with per-step B-learning.

    Directory structure:
        output_dir/
            step_000/  step_001/  ...  (per-step artifacts and logs)
            trajectory.csv               (full episode trajectory)
            episode_log.json             (episode-level summary)

    Returns:
        (beliefs, perception, qa_pairs, moments, experiments, episode_stats, steps_taken,
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
            "You died in your previous attempt and have respawned as a new character. "
            "Use what you learned from your previous life to do better this time.\n\n"
            + obs["text"]["short_term_context"]
        )

    # Setup instruction prompt with beliefs
    current_experiment = experiments[0] if experiments else None
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
    episode_log = {
        "task": task,
        "action_frequency": defaultdict(int),
        "input_tokens": 0,
        "output_tokens": 0,
        "total_cost": 0.0,
    }

    trajectory_buffer = trajectory_buffer if trajectory_buffer is not None else []
    # Insert episode boundary marker so cross-episode history is clearly delineated
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
    step_extraction_log = None
    step_experiment_log = None
    past_experiments = past_experiments if past_experiments is not None else []
    if current_experiment and current_experiment not in past_experiments:
        past_experiments.append(current_experiment)

    # CSV logging
    ep_dir = Path(output_dir)
    ep_dir.mkdir(parents=True, exist_ok=True)
    csv_filename = ep_dir / "trajectory.csv"

    pbar = tqdm(total=max_steps, desc=f"Stepwise B-learn ep {episode_idx}", leave=False, dynamic_ncols=True)
    feedback_history = []

    with open(csv_filename, mode="w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file, escapechar="\u02d8", quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["Step", "Action", "Reasoning", "Observation", "Auxiliary_Observation", "Reward", "Done"])

        action = None
        step = 0

        for step in range(max_steps):
            global_step = global_step_start + step

            # Per-step directory
            step_dir = ep_dir / f"step_{step:03d}"
            step_dir.mkdir(parents=True, exist_ok=True)

            step_extract_cost = 0.0
            step_improve_cost = 0.0
            step_experiment_cost = 0.0
            step_extraction_log = None
            step_experiment_log = None
            step_feedback_records = []

            # Write preliminary step_log immediately so the step appears in the visualizer sidebar
            _save_step_log(
                step_dir=step_dir, step=step, global_step=global_step,
                action=None, reward=0.0, done=False, episode_return=episode_return,
                agent_cost=0.0, extract_cost=0.0, improve_cost=0.0, experiment_cost=0.0,
                num_qa=len(qa_pairs), num_moments=len(moments), num_experiments=len(experiments),
                did_gen_experiments=False, active_experiment=current_experiment,
                phase="started",
            )

            # --- Experiment generation (beginning of step, before agent acts) ---
            should_gen_experiments = step % bl_config.experiment_interval == 0
            if should_gen_experiments:
                evolve_logger.info(f"[g{global_step}] Generating experiments...")
                old_experiments = list(experiments)
                with improve_logging(step_dir):
                    forward_moments = [m for m in moments if m.raw_observation and m.raw_observation.strip()]
                    experiments, exp_cost, exp_prompt, exp_response = asyncio.run(
                        generate_experiments_from_steps(
                            config=config,
                            default_actions=default_actions,
                            beliefs=beliefs,
                            qa_pairs=qa_pairs,
                            critical_moments=forward_moments,
                            trajectory_buffer=trajectory_buffer,
                            perception_code=perception,
                            current_experiment=current_experiment,
                            num_experiments=bl_config.num_experiments,
                            max_text_history=config.agent.max_text_history,
                            max_cot_history=config.agent.max_cot_history,
                            current_observation=_pre_action_raw_long,
                            current_aux_observation=_pre_action_raw_short,
                            default_knowledge=default_knowledge,
                            past_experiments=past_experiments,
                        )
                    )
                    # Track only the experiment that will actually be tried
                    if experiments and experiments[0] not in past_experiments:
                        past_experiments.append(experiments[0])
                    step_experiment_cost = exp_cost
                    total_learn_cost += exp_cost
                    step_experiment_log = {
                        "old_experiments": old_experiments,
                        "new_experiments": experiments,
                        "prompt": exp_prompt,
                        "response": exp_response,
                    }

                # Update current experiment and inject into agent
                if experiments:
                    current_experiment = experiments[0]
                    agent.experiment_goal = current_experiment

                # Write experiment artifacts immediately so visualizer can see them
                with open(step_dir / "experiment_log.json", "w") as f:
                    json.dump(step_experiment_log, f, indent=4, default=str)
                with open(step_dir / "experiments.json", "w") as f:
                    json.dump(experiments, f, indent=4)

                evolve_logger.info(
                    f"[g{global_step}] Generated {len(experiments)} experiments — cost: ${exp_cost:.6f}"
                )

            # Update step_log to reflect experiment phase done, agent about to act
            _save_step_log(
                step_dir=step_dir, step=step, global_step=global_step,
                action=None, reward=0.0, done=False, episode_return=episode_return,
                agent_cost=0.0, extract_cost=0.0, improve_cost=0.0, experiment_cost=step_experiment_cost,
                num_qa=len(qa_pairs), num_moments=len(moments), num_experiments=len(experiments),
                did_gen_experiments=should_gen_experiments, active_experiment=current_experiment,
                phase="acting",
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

            # Save raw new obs BEFORE applying perception (which modifies long_term_context in-place)
            new_raw_long = obs["text"]["long_term_context"]
            new_raw_short = obs["text"].get("short_term_context", "")

            # Apply perception to new obs
            if perception_fn is not None:
                apply_perception(obs, perception_fn)

            # Post-action obs_text: used for CSV logging and the terminal buffer entry
            result_obs_text = _compose_obs_text(
                obs["text"]["short_term_context"],
                obs["text"]["long_term_context"],
            )

            # Capture agent messages (the actual prompt sent to the LLM) + response
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

            # Save agent messages for this step immediately
            with open(step_dir / "agent_messages.json", "w") as amf:
                json.dump(agent_messages, amf, indent=2, default=str)

            # Write CSV row immediately after env step so trajectory is live
            csv_writer.writerow([step, action, reasoning, _pre_action_obs_text, _pre_action_raw_short, reward, done])
            csv_file.flush()

            # Update step_log with action/reward/done so sidebar shows live info
            _save_step_log(
                step_dir=step_dir, step=step, global_step=global_step,
                action=action, reward=reward, done=done, episode_return=episode_return,
                agent_cost=agent_step_cost, extract_cost=0.0, improve_cost=0.0, experiment_cost=step_experiment_cost,
                num_qa=len(qa_pairs), num_moments=len(moments), num_experiments=len(experiments),
                did_gen_experiments=should_gen_experiments, active_experiment=current_experiment,
                phase="extracting",
            )

            # Append buffer entry: pre-action state s_step paired with action a_step and its reward.
            # Sequence: s0, a0, s1, a1, ..., s_{t-1}, a_{t-1}, s_t
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

            # Append the terminal state s_{step+1} as a separate no-action entry
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
                (steps_in % bl_config.artifact_update_interval == 0)
                or done
            )
            should_improve = (
                (steps_in % bl_config.improve_interval == 0)
                or done
            )

            # --- Artifact update (extract Q&A + moments) ---
            if should_update_artifacts and len(trajectory_buffer) > 0:
                evolve_logger.info(f"[g{global_step}] Extracting knowledge from {len(trajectory_buffer)} buffered steps...")
                with improve_logging(step_dir):
                    qa_pairs, moments, extract_cost, step_extraction_log = asyncio.run(
                        extract_knowledge_from_buffer(
                            config=config,
                            trajectory_buffer=trajectory_buffer,
                            beliefs=beliefs,
                            perception=perception,
                            existing_qa=qa_pairs,
                            existing_moments=moments,
                            bl_config=bl_config,
                            step=step,
                            episode_idx=episode_idx,
                            is_done=done,
                        )
                    )
                    step_extract_cost = extract_cost
                    total_learn_cost += extract_cost
                # Write extraction artifacts immediately
                with open(step_dir / "extraction_log.json", "w") as f:
                    json.dump(step_extraction_log, f, indent=4, default=str)
                with open(step_dir / "qa_pairs.json", "w") as f:
                    json.dump(serialize_qa_pairs(qa_pairs), f, indent=4)
                with open(step_dir / "critical_moments.json", "w") as f:
                    json.dump(serialize_moments(moments), f, indent=4)

                evolve_logger.info(
                    f"[g{global_step}] Extraction done — "
                    f"QA: {len(qa_pairs)}, moments: {len(moments)}, cost: ${extract_cost:.6f}"
                )

                # Update step_log: extraction done, improve starting
                _save_step_log(
                    step_dir=step_dir, step=step, global_step=global_step,
                    action=action, reward=reward, done=done, episode_return=episode_return,
                    agent_cost=agent_step_cost, extract_cost=step_extract_cost, improve_cost=0.0, experiment_cost=step_experiment_cost,
                    num_qa=len(qa_pairs), num_moments=len(moments), num_experiments=len(experiments),
                    did_gen_experiments=should_gen_experiments, active_experiment=current_experiment,
                    phase="improving",
                )

            # --- 3-track improve loop ---
            if should_improve:
                evolve_logger.info(
                    f"[g{global_step}] Running improve loop (steps={bl_config.max_steps_iterations}, qa={bl_config.max_qa_iterations}, moments={bl_config.max_moments_iterations} iters)..."
                )
                pre_improve_perception = perception
                with improve_logging(step_dir):
                    beliefs, perception, qa_pairs, moments, improve_cost, iter_records = _run_improve_loop(
                        config=config,
                        bl_config=bl_config,
                        beliefs=beliefs,
                        perception=perception,
                        qa_pairs=qa_pairs,
                        moments=moments,
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

                # Rebuild all buffered observations with the latest perception.
                # This ensures extraction prompts always reflect current perception.
                if perception_changed:
                    _refresh_buffer_with_perception(trajectory_buffer, perception_fn)

                # Inject updated beliefs for next step (if not done)
                if not done:
                    _inject_beliefs(config, agent, env, env_name, task, beliefs)

                evolve_logger.info(
                    f"[g{global_step}] Improve done — cost: ${improve_cost:.6f}"
                )

                # Write qa/moments immediately after improve (they may have changed)
                with open(step_dir / "qa_pairs.json", "w") as f:
                    json.dump(serialize_qa_pairs(qa_pairs), f, indent=4)
                with open(step_dir / "critical_moments.json", "w") as f:
                    json.dump(serialize_moments(moments), f, indent=4)

            # --- Carry forward pre-action vars for the next iteration ---
            if not done:
                _pre_action_raw_long = new_raw_long
                _pre_action_raw_short = new_raw_short
                _pre_action_obs_text = _compose_obs_text(
                    obs["text"]["short_term_context"],
                    obs["text"]["long_term_context"],
                )

            # --- Per-step artifact save ---
            step_total_cost = agent_step_cost + step_extract_cost + step_improve_cost + step_experiment_cost
            cumulative_step_cost += step_total_cost

            did_learn = should_update_artifacts or should_improve or should_gen_experiments
            if did_learn:
                _save_step_artifacts(
                    step_dir, beliefs, perception, qa_pairs, moments,
                    experiments, step_feedback_records,
                    extraction_log=step_extraction_log,
                    experiment_log=step_experiment_log,
                )

            _save_step_log(
                step_dir=step_dir,
                step=step,
                global_step=global_step,
                action=action,
                reward=reward,
                done=done,
                episode_return=episode_return,
                agent_cost=agent_step_cost,
                extract_cost=step_extract_cost,
                improve_cost=step_improve_cost,
                experiment_cost=step_experiment_cost,
                num_qa=len(qa_pairs),
                num_moments=len(moments),
                num_experiments=len(experiments),
                did_gen_experiments=should_gen_experiments,
                active_experiment=current_experiment,
                phase="complete",
            )

            # Per-step summary update (global level)
            _update_summary_json(
                output_dir=os.path.dirname(output_dir),  # root run dir
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
                    "num_moments": len(moments),
                    "num_experiments": len(experiments),
                    "did_extract": should_update_artifacts,
                    "did_improve": should_improve,
                    "did_gen_experiments": should_gen_experiments,
                },
            )

            if done:
                # Write terminal state s_T as a no-action row so the full
                # trajectory s0,a0,s1,...,s_{T-1},a_{T-1},s_T is in the CSV.
                csv_writer.writerow([step + 1, "", "", result_obs_text, new_raw_short, 0.0, True])
                csv_file.flush()
                evolve_logger.info(
                    f"[g{global_step}] Episode {episode_idx} DONE — "
                    f"return={episode_return:.2f}, steps={step + 1}"
                )
                if pbar.n < pbar.total:
                    pbar.update(pbar.total - pbar.n)
                pbar.set_postfix_str("DONE")
                break

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
    episode_log["num_moments"] = len(moments)
    episode_log["num_experiments"] = len(experiments)

    # Save episode-level log
    json_filename = ep_dir / "episode_log.json"
    with open(json_filename, "w") as f:
        json.dump(episode_log, f, indent=4, default=str)

    env.close()

    evolve_logger.info(
        f"Episode {episode_idx} complete — return: {episode_return:.2f}, "
        f"steps: {step + 1}, "
        f"learn cost: ${total_learn_cost:.4f}, agent cost: ${episode_log['total_cost']:.4f}"
    )

    return beliefs, perception, qa_pairs, moments, experiments, episode_log, step + 1, trajectory_buffer, past_experiments


# ---------------------------------------------------------------------------
# Belief injection helper
# ---------------------------------------------------------------------------


def _inject_beliefs(
    config: DictConfig,
    agent,
    env,
    env_name: str,
    task: str,
    beliefs: str,
):
    """Inject beliefs into the agent's instruction prompt."""
    if config.eval.get("beliefs_in_system_prompt", False):
        if env_name == "minihack":
            from balrog.environments.minihack import get_loaded_instruction_prompt
            instruction_prompt = get_loaded_instruction_prompt(
                env=env, load=beliefs, task=task,
            )
        else:
            from balrog.environments.nle import get_loaded_instruction_prompt
            instruction_prompt = get_loaded_instruction_prompt(
                env=env, load=beliefs, task=task,
            )
        agent.prompt_builder.update_instruction_prompt(instruction_prompt)
    else:
        instructions = None
        if env_name == "babyai":
            from balrog.environments import make_env as _  # noqa: F401
            instructions = getattr(env, "mission", None)
        base_instruction = env.get_instruction_prompt(instructions=instructions)
        if beliefs:
            base_instruction += f"\n\nTips -\n{beliefs}"
        agent.prompt_builder.update_instruction_prompt(base_instruction)


# ---------------------------------------------------------------------------
# Inner improve loop
# ---------------------------------------------------------------------------


def _run_improve_loop(
    config: DictConfig,
    bl_config: StepwiseBLearnConfig,
    beliefs: str,
    perception: str,
    qa_pairs: list[QAPair],
    moments: list[CriticalMoment],
    trajectory_buffer: list[dict],
    default_knowledge: str,
    step: int,
    global_step: int = 0,
    step_dir: Path | None = None,
) -> tuple[str, str, list[QAPair], list[CriticalMoment], float, list[dict]]:
    """Dispatch to conversational or independent improve loop based on config."""
    if bl_config.conversational_improve:
        return _run_improve_loop_conversational(
            config, bl_config, beliefs, perception, qa_pairs, moments,
            trajectory_buffer, default_knowledge, step, global_step, step_dir=step_dir,
        )
    return _run_improve_loop_independent(
        config, bl_config, beliefs, perception, qa_pairs, moments,
        trajectory_buffer, default_knowledge, step, global_step, step_dir=step_dir,
    )


def _run_improve_loop_independent(
    config: DictConfig,
    bl_config: StepwiseBLearnConfig,
    beliefs: str,
    perception: str,
    qa_pairs: list[QAPair],
    moments: list[CriticalMoment],
    trajectory_buffer: list[dict],
    default_knowledge: str,
    step: int,
    global_step: int = 0,
    step_dir: Path | None = None,
) -> tuple[str, str, list[QAPair], list[CriticalMoment], float, list[dict]]:
    """Run the 3-track inner improve loop with independent LLM calls (legacy mode).

    Returns: (beliefs, perception, qa_pairs, moments, total_cost, feedback_history)
    """
    total_cost = 0.0
    feedback_history = []
    tag = f"[g{global_step}]"

    steps_context = format_steps_context(
        trajectory_buffer, perception, bl_config.max_steps_context_chars,
    )
    sample_obs = _sample_observations_from_buffer(
        trajectory_buffer, bl_config.num_sample_obs,
    )
    forward_moments = [m for m in moments if m.raw_observation and m.raw_observation.strip()]

    evolve_logger.info(
        f"{tag} Improve loop (independent): steps={bl_config.max_steps_iterations}, qa={bl_config.max_qa_iterations}, moments={bl_config.max_moments_iterations} iters, "
        f"{len(qa_pairs)} QA, {len(forward_moments)} fwd moments, "
        f"{len(steps_context)} chars context"
    )

    for inner_iter in range(bl_config.max_steps_iterations):
        evolve_logger.info(f"{tag}   iter {inner_iter + 1}/{bl_config.max_steps_iterations}")
        iter_record = {"iteration": inner_iter + 1, "step": step, "global_step": global_step}

        try:
            # Track 1: Steps-based improvement
            if steps_context:
                evolve_logger.info(f"{tag}     Track 1: Steps-based improvement...")
                prev_perception = perception
                beliefs, perception, steps_cost, steps_prompt, steps_response = asyncio.run(
                    improve_from_steps(
                        config=config, beliefs=beliefs, perception=perception,
                        steps_context=steps_context,
                        sample_observations=sample_obs if sample_obs else None,
                        default_knowledge=default_knowledge,
                    )
                )
                total_cost += steps_cost
                evolve_logger.info(f"{tag}     Track 1 done (cost: ${steps_cost:.6f})")
                iter_record["steps_cost"] = steps_cost
                iter_record["steps_prompt"] = steps_prompt
                iter_record["steps_response"] = steps_response

                # Rebuild steps_context if perception changed
                if perception != prev_perception:
                    steps_context = format_steps_context(
                        trajectory_buffer, perception, bl_config.max_steps_context_chars,
                    )
            else:
                evolve_logger.info(f"{tag}     Track 1: No steps context, skipping")

            # Track 2: QA-based improvement
            if not bl_config.use_qa:
                evolve_logger.info(f"{tag}     Track 2: Disabled (use_qa=False), skipping")
            elif qa_pairs:
                evolve_logger.info(f"{tag}     Track 2a: QA forward pass on {len(qa_pairs)} questions...")
                qa_fwd_results, qa_fwd_cost, qa_fwd_prompts, qa_fwd_responses = asyncio.run(
                    qa_forward_pass(config=config, beliefs=beliefs, qa_pairs=qa_pairs,
                                    max_per_batch=bl_config.max_qa_per_forward)
                )
                total_cost += qa_fwd_cost

                qa_fb_results, qa_fb_cost, qa_fb_prompts, qa_fb_responses = asyncio.run(
                    qa_get_feedback(config=config, qa_forward_results=qa_fwd_results,
                                    max_per_batch=bl_config.max_qa_per_forward)
                )
                total_cost += qa_fb_cost

                qa_correct = [fr for fr in qa_fb_results if fr.verdict == "CORRECT"]
                qa_incorrect = [fr for fr in qa_fb_results if fr.verdict == "INCORRECT"]
                qa_inconclusive = [fr for fr in qa_fb_results if fr.verdict == "INCONCLUSIVE"]
                qa_actionable = [fr for fr in qa_fb_results if fr.verdict != "INCONCLUSIVE"]

                evolve_logger.info(
                    f"      Track 2b: {len(qa_correct)} correct, {len(qa_incorrect)} incorrect, "
                    f"{len(qa_inconclusive)} inconclusive"
                )
                iter_record["qa_num_pairs"] = len(qa_pairs)
                iter_record["qa_num_correct"] = len(qa_correct)
                iter_record["qa_num_incorrect"] = len(qa_incorrect)
                iter_record["qa_num_inconclusive"] = len(qa_inconclusive)
                iter_record["qa_forward_cost"] = qa_fwd_cost
                iter_record["qa_feedback_cost"] = qa_fb_cost
                iter_record["qa_forward_prompt"] = "\n---\n".join(qa_fwd_prompts)
                iter_record["qa_forward_response"] = "\n---\n".join(qa_fwd_responses)
                iter_record["qa_feedback_prompt"] = "\n---\n".join(qa_fb_prompts)
                iter_record["qa_feedback_response"] = "\n---\n".join(qa_fb_responses)
                iter_record["qa_feedback_details"] = serialize_qa_feedback_results(qa_fb_results)

                if qa_actionable and qa_incorrect:
                    evolve_logger.info(f"{tag}     Track 2c: Improving from {len(qa_actionable)} QA feedback items...")
                    prev_perception = perception
                    beliefs, perception, qa_improve_cost, qa_improve_prompt, qa_improve_response = asyncio.run(
                        improve_from_qa_feedback_with_steps(
                            config=config, beliefs=beliefs, perception=perception,
                            qa_feedback_results=qa_actionable, default_knowledge=default_knowledge,
                            steps_context=steps_context,
                            sample_observations=sample_obs if sample_obs else None,
                        )
                    )
                    total_cost += qa_improve_cost
                    evolve_logger.info(f"{tag}     Track 2c done (cost: ${qa_improve_cost:.6f})")
                    iter_record["qa_improve_cost"] = qa_improve_cost
                    iter_record["qa_improve_prompt"] = qa_improve_prompt
                    iter_record["qa_improve_response"] = qa_improve_response

                    # Rebuild steps_context if perception changed
                    if perception != prev_perception:
                        steps_context = format_steps_context(
                            trajectory_buffer, perception, bl_config.max_steps_context_chars,
                        )
                else:
                    evolve_logger.info(f"{tag}     Track 2c: No incorrect QA, skipping")
            else:
                evolve_logger.info(f"{tag}     Track 2: No QA pairs, skipping")

            # Track 3: Critical moment improvement
            if not bl_config.use_moments:
                evolve_logger.info(f"{tag}     Track 3: Disabled (use_moments=False), skipping")
            elif forward_moments:
                evolve_logger.info(f"{tag}     Track 3a: Forward pass on {len(forward_moments)} moments...")
                fwd_results, fwd_cost = asyncio.run(
                    forward_pass(config, beliefs, perception, forward_moments,
                                 max_per_batch=bl_config.max_moments_per_forward)
                )
                total_cost += fwd_cost

                fb_results, fb_cost, _, _ = asyncio.run(
                    get_feedback(config, fwd_results,
                                 max_per_batch=bl_config.max_moments_per_forward)
                )
                total_cost += fb_cost

                correct = [fr for fr in fb_results if fr.verdict == "CORRECT"]
                incorrect = [fr for fr in fb_results if fr.verdict == "INCORRECT"]
                inconclusive = [fr for fr in fb_results if fr.verdict == "INCONCLUSIVE"]
                actionable = [fr for fr in fb_results if fr.verdict != "INCONCLUSIVE"]

                evolve_logger.info(
                    f"      Track 3b: {len(correct)} correct, {len(incorrect)} incorrect, "
                    f"{len(inconclusive)} inconclusive"
                )
                iter_record["moment_num_moments"] = len(forward_moments)
                iter_record["moment_num_correct"] = len(correct)
                iter_record["moment_num_incorrect"] = len(incorrect)
                iter_record["moment_num_inconclusive"] = len(inconclusive)
                iter_record["moment_forward_cost"] = fwd_cost
                iter_record["moment_feedback_cost"] = fb_cost
                iter_record["moment_feedback_details"] = serialize_feedback_results(fb_results)

                if actionable and incorrect:
                    evolve_logger.info(f"{tag}     Track 3c: Improving from {len(actionable)} moment feedback items...")
                    prev_perception = perception
                    beliefs, perception, improve_cost, improve_prompt, improve_response = asyncio.run(
                        improve_from_feedback_with_steps(
                            config=config, beliefs=beliefs, perception=perception,
                            feedback_results=actionable, default_knowledge=default_knowledge,
                            steps_context=steps_context,
                            sample_observations=sample_obs if sample_obs else None,
                        )
                    )
                    total_cost += improve_cost
                    evolve_logger.info(f"{tag}     Track 3c done (cost: ${improve_cost:.6f})")
                    iter_record["moment_improve_cost"] = improve_cost
                    iter_record["moment_improve_prompt"] = improve_prompt
                    iter_record["moment_improve_response"] = improve_response

                    # Rebuild steps_context if perception changed
                    if perception != prev_perception:
                        steps_context = format_steps_context(
                            trajectory_buffer, perception, bl_config.max_steps_context_chars,
                        )
                else:
                    evolve_logger.info(f"{tag}     Track 3c: No incorrect moments, skipping")
            else:
                evolve_logger.info(f"{tag}     Track 3: No forward moments, skipping")

            feedback_history.append(iter_record)
            if step_dir is not None:
                _flush_improve_progress(step_dir, feedback_history, beliefs, perception)

        except Exception as e:
            evolve_logger.error(f"{tag}     Improve iter {inner_iter + 1} failed: {e}")
            logging.exception("Improve iteration failed")
            iter_record["error"] = str(e)
            feedback_history.append(iter_record)
            if step_dir is not None:
                _flush_improve_progress(step_dir, feedback_history, beliefs, perception)

    return beliefs, perception, qa_pairs, moments, total_cost, feedback_history


def _run_improve_loop_conversational(
    config: DictConfig,
    bl_config: StepwiseBLearnConfig,
    beliefs: str,
    perception: str,
    qa_pairs: list[QAPair],
    moments: list[CriticalMoment],
    trajectory_buffer: list[dict],
    default_knowledge: str,
    step: int,
    global_step: int = 0,
    step_dir: Path | None = None,
) -> tuple[str, str, list[QAPair], list[CriticalMoment], float, list[dict]]:
    """Run the 3-track adaptive improve loop with per-track conversations.

    Each track maintains its own conversation history. The LLM can exit early
    by returning <status>SUBMIT</status>. Beliefs/perception flow sequentially
    from Track 1 -> Track 2 -> Track 3.

    Returns: (beliefs, perception, qa_pairs, moments, total_cost, feedback_history)
    """
    total_cost = 0.0
    feedback_history = []
    tag = f"[g{global_step}]"

    # Build steps context (rebuilt when perception changes)
    steps_context = format_steps_context(
        trajectory_buffer, perception, bl_config.max_steps_context_chars,
    )

    # Build sample observations for perception validation
    sample_obs = _sample_observations_from_buffer(
        trajectory_buffer, bl_config.num_sample_obs,
    )

    # Filter moments with raw observations for forward pass
    forward_moments = [m for m in moments if m.raw_observation and m.raw_observation.strip()]

    evolve_logger.info(
        f"{tag} Improve loop: steps={bl_config.max_steps_iterations}, perception_from_analysis={bl_config.max_perception_iterations}, qa={bl_config.max_qa_iterations}, moments={bl_config.max_moments_iterations} iters, "
        f"{len(qa_pairs)} QA, {len(forward_moments)} fwd moments, "
        f"{len(steps_context)} chars context"
    )

    try:

        # ========================================
        # Track 1b: Steps-based beliefs improvement
        # ========================================
        if steps_context:
            track1b_record = {"track": "steps_beliefs", "step": step, "global_step": global_step, "turns": []}

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
- Perception analysis: What information was presented in the explicit features from perception module section. What part of that information was helpful, what information was misleading / incorrect and what additional information would have helped if extracted by the perception module?

For beliefs:
- Overall they beliefs should be split into two sections:
  * <world_knowledge>: Facts about how the environment works — mechanics, properties, cause-and-effect relationships.
  * <policy>: Tactical approaches — what to do in specific situations, priorities, strategies for completing the objective.
- Correct any wrong or misleading beliefs in either section.
- Both sections should be consise, made up of at most 10 brief points, merging any redundant or stale information.
- They should be grounded in the evidence present in the step sequence, only containing inferences from what we have observed so far.

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
            track1b_record["turns"].append(turn_record)

            evolve_logger.info(f"{tag}     Track 1b done (cost: ${turn_cost:.6f}, perception_analysis: {len(perception_analysis)} chars)")

            feedback_history.append(track1b_record)
            if step_dir is not None:
                _flush_improve_progress(step_dir, feedback_history, beliefs, perception)
        else:
            evolve_logger.info(f"{tag}     Track 1b: No steps context, skipping")
            perception_analysis = ""

        # ========================================
        # Track 1c: Perception improvement guided by beliefs analysis
        # ========================================
        if sample_obs:
            track1c_record = {"track": "perception_from_analysis", "step": step, "global_step": global_step, "turns": []}
            pre_perception_track1c = perception
            obs_section_1c = _build_obs_section(perception, sample_obs)

            perception_from_analysis_prompt = build_perception_with_analysis_prompt(
                beliefs=beliefs,
                perception=perception,
                default_knowledge=default_knowledge,
                obs_section=obs_section_1c,
                perception_analysis=perception_analysis,
                max_iterations=bl_config.max_perception_iterations,
            )

            perception_conv_1c: list[dict] = []
            prev_obs_section_1c = obs_section_1c

            for turn in range(bl_config.max_perception_iterations):
                evolve_logger.info(f"{tag}     Track 1c (perception from analysis) turn {turn + 1}/{bl_config.max_perception_iterations}")

                message = perception_from_analysis_prompt if turn == 0 else build_perception_followup_message(
                    perception, sample_obs, prev_obs_section_1c,
                    current_turn=turn + 1, max_turns=bl_config.max_perception_iterations,
                )

                _beliefs_unused, perception, turn_cost, perception_conv_1c, response_text = asyncio.run(
                    _improve_with_perception_validation_conversational(
                        config=config,
                        beliefs=beliefs,
                        perception=perception,
                        conversation_history=perception_conv_1c,
                        user_message=message,
                        sample_observations=sample_obs,
                    )
                )
                total_cost += turn_cost

                turn_record = {"turn": turn + 1, "cost": turn_cost, "prompt": message, "response": response_text}
                submitted = parse_submit_signal(response_text)
                turn_record["submitted"] = submitted
                track1c_record["turns"].append(turn_record)

                evolve_logger.info(
                    f"{tag}     Track 1c turn {turn + 1} done "
                    f"(cost: ${turn_cost:.6f}, submit: {submitted})"
                )

                if submitted:
                    break

                prev_obs_section_1c = _build_obs_section(perception, sample_obs)

            feedback_history.append(track1c_record)
            if step_dir is not None:
                _flush_improve_progress(step_dir, feedback_history, beliefs, perception)

            # Rebuild steps_context if perception changed during Track 1c
            if perception != pre_perception_track1c:
                steps_context = format_steps_context(
                    trajectory_buffer, perception, bl_config.max_steps_context_chars,
                )
        else:
            evolve_logger.info(f"{tag}     Track 1c: No sample observations, skipping")

        # ========================================
        # Track 2: QA-based conversational improvement
        # ========================================
        if not bl_config.use_qa:
            evolve_logger.info(f"{tag}     Track 2: Disabled (use_qa=False), skipping")
        elif qa_pairs:
            track2_record = {"track": "qa", "step": step, "global_step": global_step, "turns": []}

            # Initial QA evaluation
            evolve_logger.info(f"{tag}     Track 2: Initial QA forward pass on {len(qa_pairs)} questions...")
            qa_fwd_results, qa_fwd_cost, qa_fwd_prompts, qa_fwd_responses = asyncio.run(
                qa_forward_pass(
                    config=config,
                    beliefs=beliefs,
                    qa_pairs=qa_pairs,
                    max_per_batch=bl_config.max_qa_per_forward,
                )
            )
            total_cost += qa_fwd_cost

            qa_fb_results, qa_fb_cost, qa_fb_prompts, qa_fb_responses = asyncio.run(
                qa_get_feedback(
                    config=config,
                    qa_forward_results=qa_fwd_results,
                    max_per_batch=bl_config.max_qa_per_forward,
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

This is a multi-turn conversation. After each response, the QA pairs will be re-evaluated with your updated beliefs/perception. You can iterate up to {bl_config.max_qa_iterations} turns.

{PERCEPTION_INSTRUCTIONS}

{RESPONSE_FORMAT}"""

                qa_conversation: list[dict] = []
                prev_qa_correct = len(qa_correct)
                prev_qa_incorrect = len(qa_incorrect)

                for turn in range(bl_config.max_qa_iterations):
                    evolve_logger.info(f"{tag}     Track 2 turn {turn + 1}/{bl_config.max_qa_iterations}")

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
                    if turn + 1 < bl_config.max_qa_iterations:
                        prev_qa_correct = sum(1 for fr in qa_fb_results if fr.verdict == "CORRECT")
                        prev_qa_incorrect = sum(1 for fr in qa_fb_results if fr.verdict == "INCORRECT")

                        qa_fwd_results, qa_fwd_cost, _, _ = asyncio.run(
                            qa_forward_pass(
                                config=config,
                                beliefs=beliefs,
                                qa_pairs=qa_pairs,
                                max_per_batch=bl_config.max_qa_per_forward,
                            )
                        )
                        total_cost += qa_fwd_cost

                        qa_fb_results, qa_fb_cost, _, _ = asyncio.run(
                            qa_get_feedback(
                                config=config,
                                qa_forward_results=qa_fwd_results,
                                max_per_batch=bl_config.max_qa_per_forward,
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
                    trajectory_buffer, perception, bl_config.max_steps_context_chars,
                )
        else:
            evolve_logger.info(f"{tag}     Track 2: No QA pairs, skipping")

        # ========================================
        # Track 3: Critical moment conversational improvement
        # ========================================
        if not bl_config.use_moments:
            evolve_logger.info(f"{tag}     Track 3: Disabled (use_moments=False), skipping")
        elif forward_moments:
            track3_record = {"track": "moments", "step": step, "global_step": global_step, "turns": []}

            # Initial moment evaluation
            evolve_logger.info(f"{tag}     Track 3: Initial forward pass on {len(forward_moments)} moments...")
            fwd_results, fwd_cost = asyncio.run(
                forward_pass(
                    config, beliefs, perception, forward_moments,
                    max_per_batch=bl_config.max_moments_per_forward,
                )
            )
            total_cost += fwd_cost

            fb_results, fb_cost, _, _ = asyncio.run(
                get_feedback(
                    config, fwd_results,
                    max_per_batch=bl_config.max_moments_per_forward,
                )
            )
            total_cost += fb_cost

            correct = [fr for fr in fb_results if fr.verdict == "CORRECT"]
            incorrect = [fr for fr in fb_results if fr.verdict == "INCORRECT"]
            inconclusive = [fr for fr in fb_results if fr.verdict == "INCONCLUSIVE"]
            actionable = [fr for fr in fb_results if fr.verdict != "INCONCLUSIVE"]

            evolve_logger.info(
                f"{tag}     Track 3: Initial eval: {len(correct)} correct, "
                f"{len(incorrect)} incorrect, {len(inconclusive)} inconclusive"
            )

            track3_record["initial_correct"] = len(correct)
            track3_record["initial_incorrect"] = len(incorrect)
            track3_record["moment_forward_cost"] = fwd_cost
            track3_record["moment_feedback_cost"] = fb_cost
            track3_record["moment_feedback_details"] = serialize_feedback_results(fb_results)

            if actionable and incorrect:
                # Build initial moments improvement prompt
                feedback_blocks = []
                for i, fr in enumerate(actionable, 1):
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

                execution_report_section = _build_execution_report_section(perception, sample_obs)

                initial_moment_prompt = f"""You are improving an agent's knowledge and perception based on feedback from testing its decision-making.

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

Results: {len(correct)} correct, {len(incorrect)} incorrect out of {len(fb_results)} evaluated.

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
- Keep each beliefs section to at most 10 brief points. Merge redundant points.
- Balance safety with progress toward the objective.
- Remove beliefs that are contradicted by the feedback evidence.

This is a multi-turn conversation. After each response, the critical moments will be re-evaluated with your updated beliefs/perception. You can iterate up to {bl_config.max_moments_iterations} turns.

{PERCEPTION_INSTRUCTIONS}

{RESPONSE_FORMAT}"""

                moment_conversation: list[dict] = []
                prev_correct = len(correct)
                prev_incorrect = len(incorrect)

                for turn in range(bl_config.max_moments_iterations):
                    evolve_logger.info(f"{tag}     Track 3 turn {turn + 1}/{bl_config.max_moments_iterations}")

                    message = initial_moment_prompt if turn == 0 else build_moments_followup_message(
                        fb_results, prev_correct, prev_incorrect,
                    )

                    beliefs, perception, turn_cost, moment_conversation, response_text = asyncio.run(
                        _improve_with_perception_validation_conversational(
                            config=config,
                            beliefs=beliefs,
                            perception=perception,
                            conversation_history=moment_conversation,
                            user_message=message,
                            sample_observations=sample_obs if sample_obs else None,
                        )
                    )
                    total_cost += turn_cost

                    turn_record = {"turn": turn + 1, "cost": turn_cost, "prompt": message, "response": response_text}
                    submitted = parse_submit_signal(response_text)
                    turn_record["submitted"] = submitted
                    track3_record["turns"].append(turn_record)

                    evolve_logger.info(
                        f"{tag}     Track 3 turn {turn + 1} done "
                        f"(cost: ${turn_cost:.6f}, submit: {submitted})"
                    )

                    if submitted:
                        evolve_logger.info(f"{tag}     Track 3: LLM submitted after {turn + 1} turn(s)")
                        break

                    # Re-evaluate moments for next turn (unless this is the last turn)
                    if turn + 1 < bl_config.max_moments_iterations:
                        prev_correct = sum(1 for fr in fb_results if fr.verdict == "CORRECT")
                        prev_incorrect = sum(1 for fr in fb_results if fr.verdict == "INCORRECT")

                        fwd_results, fwd_cost = asyncio.run(
                            forward_pass(
                                config, beliefs, perception, forward_moments,
                                max_per_batch=bl_config.max_moments_per_forward,
                            )
                        )
                        total_cost += fwd_cost

                        fb_results, fb_cost, _, _ = asyncio.run(
                            get_feedback(
                                config, fwd_results,
                                max_per_batch=bl_config.max_moments_per_forward,
                            )
                        )
                        total_cost += fb_cost

                        new_correct = sum(1 for fr in fb_results if fr.verdict == "CORRECT")
                        new_incorrect = sum(1 for fr in fb_results if fr.verdict == "INCORRECT")
                        evolve_logger.info(
                            f"{tag}     Track 3 re-eval: {new_correct} correct "
                            f"({new_correct - prev_correct:+d}), "
                            f"{new_incorrect} incorrect ({new_incorrect - prev_incorrect:+d})"
                        )
            else:
                evolve_logger.info(f"{tag}     Track 3: No incorrect moments, skipping improvement")

            feedback_history.append(track3_record)
            if step_dir is not None:
                _flush_improve_progress(step_dir, feedback_history, beliefs, perception)
        else:
            evolve_logger.info(f"{tag}     Track 3: No forward moments, skipping")

    except Exception as e:
        evolve_logger.error(f"{tag}     Improve loop failed: {e}")
        logging.exception("Improve loop failed")
        feedback_history.append({"error": str(e), "step": step, "global_step": global_step})
        if step_dir is not None:
            _flush_improve_progress(step_dir, feedback_history, beliefs, perception)

    return beliefs, perception, qa_pairs, moments, total_cost, feedback_history


# ---------------------------------------------------------------------------
# Outer orchestrator
# ---------------------------------------------------------------------------


def stepwise_b_learn(
    bl_config: StepwiseBLearnConfig,
    config: DictConfig,
    original_cwd: str,
    output_dir: str,
):
    """Run stepwise B-learning: per-step improvement across episodes."""
    evolve_logger.info("Starting stepwise B-learning")

    # Check for resume
    last_ep, beliefs, perception, qa_pairs, moments = find_last_completed_episode(output_dir)
    start_episode = last_ep + 1

    # Also restore experiments and global step count
    experiments = []
    global_steps_used = 0
    if start_episode > 0:
        evolve_logger.info(f"Resuming from episode {start_episode} ({len(qa_pairs)} QA, {len(moments)} moments)")
        # Recover experiments from last episode dir
        exp_file = Path(output_dir) / f"episode_{last_ep}" / "experiments.json"
        if exp_file.exists():
            try:
                experiments = json.loads(exp_file.read_text())
            except (json.JSONDecodeError, TypeError):
                pass
        # Recover global step count from episode logs
        for ep_idx in range(start_episode):
            ep_json = list((Path(output_dir) / f"episode_{ep_idx}").glob("*_run_*.json"))
            for jf in ep_json:
                try:
                    ep_data = json.loads(jf.read_text())
                    global_steps_used += ep_data.get("num_steps", 0)
                except (json.JSONDecodeError, TypeError):
                    pass
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
        moments = []

    default_knowledge = get_default_knowledge(config)
    evolve_logger.info(f"Default knowledge: {len(default_knowledge)} chars")
    default_actions = get_default_actions(config)
    evolve_logger.info(f"Default actions: {len(default_actions)} chars")

    evolve_logger.info(f"Stepwise B-learn config:")
    evolve_logger.info(f"  Total env steps: {bl_config.n_environment_steps}")
    evolve_logger.info(f"  Improve iterations: perception={bl_config.max_perception_iterations}, steps={bl_config.max_steps_iterations}, qa={bl_config.max_qa_iterations}, moments={bl_config.max_moments_iterations} ({'conversational' if bl_config.conversational_improve else 'independent'})")
    evolve_logger.info(f"  Artifact update interval: {bl_config.artifact_update_interval}")
    evolve_logger.info(f"  Improve interval: {bl_config.improve_interval}")
    evolve_logger.info(f"  Experiment interval: {bl_config.experiment_interval}")
    evolve_logger.info(f"  Max steps context chars: {bl_config.max_steps_context_chars}")
    evolve_logger.info(f"  Explore temp: {bl_config.explore_temp}")

    cumulative_cost = 0.0
    episode_idx = start_episode
    trajectory_buffer: list[dict] = []
    past_experiments: list[str] = []

    while global_steps_used < bl_config.n_environment_steps:
        remaining_steps = bl_config.n_environment_steps - global_steps_used

        evolve_logger.info(f"\n{'='*80}")
        evolve_logger.info(
            f"STEPWISE B-LEARN EPISODE {episode_idx} "
            f"(global steps: {global_steps_used}/{bl_config.n_environment_steps}, "
            f"remaining: {remaining_steps})"
        )
        evolve_logger.info(f"QA pairs: {len(qa_pairs)}, Moments: {len(moments)}, Experiments: {len(experiments)}")
        evolve_logger.info(f"{'='*80}")

        episode_dir = Path(output_dir) / f"episode_{episode_idx}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Save input state
        (episode_dir / "input_beliefs.txt").write_text(beliefs)
        (episode_dir / "input_perception.py").write_text(perception)
        with open(episode_dir / "input_experiments.json", "w") as f:
            json.dump(experiments, f, indent=4)

        with override_temperature(config, bl_config.explore_temp):
            beliefs, perception, qa_pairs, moments, experiments, episode_log, steps_taken, trajectory_buffer, past_experiments = (
                run_stepwise_b_learn_episode(
                    config=config,
                    bl_config=bl_config,
                    beliefs=beliefs,
                    perception=perception,
                    qa_pairs=qa_pairs,
                    moments=moments,
                    experiments=experiments,
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
        _save_episode_artifacts(episode_dir, beliefs, perception, qa_pairs, moments)
        with open(episode_dir / "experiments.json", "w") as f:
            json.dump(experiments, f, indent=4)

        episode_cost = episode_log.get("total_cost", 0.0) + episode_log.get("total_learn_cost", 0.0)
        cumulative_cost += episode_cost
        evolve_logger.info(
            f"[g{global_steps_used}] Episode {episode_idx} done — "
            f"cost: ${episode_cost:.4f}, cumulative: ${cumulative_cost:.4f}, "
            f"steps: {global_steps_used}/{bl_config.n_environment_steps}"
        )

        episode_idx += 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@hydra.main(config_path="BALROG/balrog/config", config_name="config", version_base="1.1")
def main(config: DictConfig):
    run_name_suffix = f"{config.agent.type}_{config.client.model_id.replace('/', '_')}_stepwise_b_learn"

    original_cwd, output_dir = setup_run(
        config,
        run_name_suffix=run_name_suffix,
        resume_from=config.eval.resume_from,
        output_dir_base=config.eval.output_dir,
        logger_name="evolve",
    )

    evolve_cfg = config.eval.evolve

    bl_config = StepwiseBLearnConfig(
        n_environment_steps=evolve_cfg.get("n_environment_steps", 100),
        max_steps_iterations=evolve_cfg.get(
            "max_steps_iterations",
            evolve_cfg.get("max_improve_iterations", evolve_cfg.get("num_improve_iterations", 1)),
        ),
        max_perception_iterations=evolve_cfg.get(
            "max_perception_iterations",
            evolve_cfg.get("max_steps_iterations",
                           evolve_cfg.get("max_improve_iterations", evolve_cfg.get("num_improve_iterations", 5))),
        ),
        use_qa=evolve_cfg.get("use_qa_improve", True),
        use_moments=evolve_cfg.get("use_moments_improve", True),
        max_qa_iterations=evolve_cfg.get(
            "max_qa_iterations",
            evolve_cfg.get("max_improve_iterations", evolve_cfg.get("num_improve_iterations", 1)),
        ),
        max_moments_iterations=evolve_cfg.get(
            "max_moments_iterations",
            evolve_cfg.get("max_improve_iterations", evolve_cfg.get("num_improve_iterations", 1)),
        ),
        max_moments_per_forward=evolve_cfg.get("max_moments_per_forward", 10),
        max_qa_per_forward=evolve_cfg.get("max_qa_per_forward", 10),
        max_total_moments=evolve_cfg.get("max_total_moments", 50),
        max_total_qa_pairs=evolve_cfg.get("max_total_qa_pairs", 50),
        num_experiments=evolve_cfg.get("num_experiments", 5),
        num_sample_obs=evolve_cfg.get("num_sample_obs", 3),
        explore_temp=evolve_cfg.get("explore_temp", 1.0),
        artifact_update_interval=evolve_cfg.get("artifact_update_interval", 1),
        improve_interval=evolve_cfg.get("improve_interval", 5),
        experiment_interval=evolve_cfg.get("experiment_interval", 10),
        max_steps_context_chars=evolve_cfg.get("max_steps_context_chars", 50000),
        conversational_improve=evolve_cfg.get("conversational_improve", True),
    )

    stepwise_b_learn(
        bl_config=bl_config,
        config=config,
        original_cwd=original_cwd,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
