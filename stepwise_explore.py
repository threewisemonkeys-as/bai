"""Stepwise exploration: learn mid-episode when the agent signals subgoal completion."""

import asyncio
import csv
import json
import logging
import os
import re
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from balrog.agents import AgentFactory
from balrog.environments import make_env
from balrog.utils import get_unique_seed

from explore import get_default_knowledge, evolve_logger
from mixed_improve import (
    QAPair,
    CriticalMoment,
    extract_qa_and_moments_from_trajectory,
    consolidate_qa_and_moments,
    score_beliefs,
    improve_beliefs_from_scores,
    improve_beliefs_simple,
    serialize_qa_pairs,
    deserialize_qa_pairs,
    serialize_moments,
    deserialize_moments,
)
from run_utils import setup_run, improve_logging, _update_summary_json


# ---------------------------------------------------------------------------
# Subgoal signal parsing helpers
# ---------------------------------------------------------------------------


def parse_subgoal_complete(reasoning: str) -> bool:
    """Check if agent signaled subgoal completion in its reasoning output."""
    if not reasoning:
        return False
    match = re.search(
        r"<subgoal_complete>\s*(YES)\s*</subgoal_complete>", reasoning, re.IGNORECASE
    )
    return match is not None


def parse_subgoal(reasoning: str) -> str:
    """Extract the current subgoal text from agent reasoning."""
    if not reasoning:
        return ""
    match = re.search(r"<subgoal>(.*?)</subgoal>", reasoning, re.DOTALL)
    return match.group(1).strip() if match else ""


# ---------------------------------------------------------------------------
# Trajectory segment formatting
# ---------------------------------------------------------------------------


def format_segment_text(segment: list[dict]) -> str:
    """Format a trajectory buffer segment into CSV-like text.

    Matches the format that extract_qa_and_moments_from_trajectory expects.
    """
    lines = ["Step,Action,Reasoning,Observation,Reward,Done"]
    for entry in segment:
        if entry.get("episode_boundary"):
            continue
        # Escape commas/newlines minimally for readability
        obs = entry.get("obs_text", "").replace("\n", " | ")
        reasoning = entry.get("reasoning", "").replace("\n", " | ")
        action = entry.get("action") or ""
        lines.append(
            f"{entry['step']},{action},{reasoning},{obs},{entry.get('reward', 0)},{entry.get('done', False)}"
        )
    return "\n".join(lines)


def format_segment_outcome_header(
    segment: list[dict], episode_idx: int, is_done: bool
) -> str:
    """Create an outcome header for a trajectory segment (no JSON file available mid-episode)."""
    real_entries = [e for e in segment if not e.get("episode_boundary")]
    total_reward = sum(e.get("reward", 0) for e in real_entries)
    num_steps = len(real_entries)
    start_step = real_entries[0]["step"] if real_entries else 0
    end_step = real_entries[-1]["step"] if real_entries else 0

    status = "COMPLETED" if is_done else "IN PROGRESS (subgoal segment)"
    return (
        f"Episode {episode_idx} — steps {start_step}–{end_step} ({num_steps} steps)\n"
        f"Status: {status}\n"
        f"Segment reward: {total_reward:.2f}\n"
    )


# ---------------------------------------------------------------------------
# Perception helpers
# ---------------------------------------------------------------------------


def load_perception_fn(perception_code: str):
    """Compile perception code and return the perceive function, or None."""
    if not perception_code or not perception_code.strip():
        return None
    try:
        compile(perception_code, "<perception_module>", "exec")
    except SyntaxError as e:
        logging.error(f"Perception syntax error at line {e.lineno}: {e.msg}")
        return None
    try:
        ns = {}
        exec(perception_code, ns)
        if "perceive" in ns:
            return ns["perceive"]
        logging.warning("Perception code has no 'perceive' function")
    except Exception as e:
        logging.error(f"Failed to load perception module: {e}")
    return None


def apply_perception(obs: dict, perception_fn) -> None:
    """Apply perception function to observation in-place (mirrors evaluator logic)."""
    if perception_fn is None:
        return
    long_term_text = obs["text"]["long_term_context"]
    try:
        perception_output = perception_fn(long_term_text)
    except Exception as e:
        perception_output = f"Perception code failed with error -\n{e}"
        logging.warning(f"Perception module failed: {e}")
    obs["text"]["short_term_context"] = (
        f"\n{'='*10} Start of features from Perception Module {'='*10}\n"
        f"{perception_output}\n\n"
        f"{'='*10} End of features from Perception Module {'='*10}\n\n"
        f"{'='*10} Start of Auxilliary Observation {'='*10}\n"
        f"{obs['text']['short_term_context']}\n\n"
        f"{'='*10} End of Auxilliary Observation {'='*10}"
    )


def format_beliefs_prompt(beliefs: str) -> str:
    """Wrap beliefs text for injection into the agent's instruction prompt."""
    if not beliefs or not beliefs.strip():
        return ""
    return beliefs


# ---------------------------------------------------------------------------
# Mid-episode and end-of-episode learning
# ---------------------------------------------------------------------------


async def learn_from_segment(
    config: DictConfig,
    beliefs: str,
    perception: str,
    segment: list[dict],
    qa_pairs: list[QAPair],
    moments: list[CriticalMoment],
    default_knowledge: str,
    episode_idx: int,
    step: int,
    is_episode_end: bool,
) -> tuple[str, list[QAPair], list[CriticalMoment], float]:
    """Learn from a trajectory segment. Light mid-episode, full at episode end.

    Returns:
        (updated_beliefs, updated_qa_pairs, updated_moments, cost)
    """
    total_cost = 0.0

    # Format segment as trajectory text
    segment_text = format_segment_text(segment)

    # 1. Extract Q&A pairs and moments from this segment
    new_qa, new_moments, extract_cost = await extract_qa_and_moments_from_trajectory(
        config=config,
        traj_text=segment_text,
        trajectory_path="",  # no CSV file for mid-episode segments
        step=step,
        use_qa=True,
        use_moments=True,
    )
    total_cost += extract_cost

    # 2. Consolidate with existing knowledge
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
            existing_qa=qa_pairs,
            existing_moments=moments,
        )
        total_cost += consolidate_cost

        # Apply removals
        if remove_qa_indices:
            qa_pairs = [
                qa for i, qa in enumerate(qa_pairs) if i not in set(remove_qa_indices)
            ]
        if remove_moment_indices:
            moments = [
                m for i, m in enumerate(moments) if i not in set(remove_moment_indices)
            ]

        qa_pairs = qa_pairs + kept_qa
        moments = moments + kept_moments

        evolve_logger.info(
            f"  Segment learning (step {step}): +{len(kept_qa)} QA, +{len(kept_moments)} moments"
        )

    # 3. Improve beliefs
    if is_episode_end and (qa_pairs or moments):
        # Full end-of-episode learning: score then improve from scores
        try:
            qa_score, moment_score, score_details, score_cost = await score_beliefs(
                config, beliefs, qa_pairs, moments, default_knowledge,
            )
            total_cost += score_cost

            evolve_logger.info(
                f"  Episode end scores: QA={qa_score:.2%}, Moments={moment_score:.2%}"
            )

            beliefs, improve_cost = await improve_beliefs_from_scores(
                config,
                beliefs,
                qa_pairs,
                moments,
                score_details,
                default_knowledge,
                outcome_header,
            )
            total_cost += improve_cost
        except Exception as e:
            evolve_logger.error(f"  End-of-episode belief improvement failed: {e}")
            logging.exception("End-of-episode belief improvement failed")
    else:
        # Mid-episode: simple improvement from the segment summary
        try:
            beliefs, improve_cost = await improve_beliefs_simple(
                config, beliefs, default_knowledge, outcome_header,
            )
            total_cost += improve_cost
        except Exception as e:
            evolve_logger.error(f"  Mid-episode belief improvement failed: {e}")
            logging.exception("Mid-episode belief improvement failed")

    return beliefs, qa_pairs, moments, total_cost


# ---------------------------------------------------------------------------
# Stepwise episode runner
# ---------------------------------------------------------------------------


def run_stepwise_episode(
    config: DictConfig,
    beliefs: str,
    perception: str,
    qa_pairs: list[QAPair],
    moments: list[CriticalMoment],
    default_knowledge: str,
    original_cwd: str,
    output_dir: str,
    episode_idx: int = 0,
    periodic_fallback_steps: int = 30,
) -> tuple[str, str, list[QAPair], list[CriticalMoment], dict]:
    """Run a single episode with mid-episode learning triggered by subgoal signals.

    Returns:
        (beliefs, perception, qa_pairs, moments, episode_stats)
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

    # Setup instruction prompt
    if config.eval.get("beliefs_in_system_prompt", False):
        if env_name == "minihack":
            from balrog.environments.minihack import get_loaded_instruction_prompt
        else:
            from balrog.environments.nle import get_loaded_instruction_prompt
        instruction_prompt = get_loaded_instruction_prompt(env=env, load=beliefs, task=task)
        agent.prompt_builder.update_instruction_prompt(instruction_prompt)
    else:
        instructions = None
        if env_name == "babyai":
            instructions = obs["mission"]
        base_instruction = env.get_instruction_prompt(instructions=instructions)
        if beliefs:
            base_instruction += f"\n\nTips -\n{beliefs}"
        agent.prompt_builder.update_instruction_prompt(base_instruction)

    # Setup perception
    perception_fn = load_perception_fn(perception)
    if perception_fn is not None:
        apply_perception(obs, perception_fn)

    # Episode tracking
    max_steps = env.max_steps if config.eval.get("max_steps_per_episode") is None else config.eval.max_steps_per_episode
    episode_log = {
        "task": task,
        "action_frequency": defaultdict(int),
        "input_tokens": 0,
        "output_tokens": 0,
        "total_cost": 0.0,
    }

    trajectory_buffer = []
    last_trigger_idx = 0
    steps_since_trigger = 0
    total_learn_cost = 0.0
    learn_triggers = []
    episode_return = 0.0

    # CSV logging
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    csv_filename = os.path.join(output_dir, f"{task}_run_{episode_idx:02d}.csv")
    Path(csv_filename).parent.mkdir(exist_ok=True, parents=True)

    pbar = tqdm(total=max_steps, desc=f"Stepwise ep {episode_idx}", leave=False, dynamic_ncols=True)

    with open(csv_filename, mode="w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file, escapechar="\u02d8", quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["Step", "Action", "Reasoning", "Observation", "Reward", "Done"])

        action = None
        step = 0

        for step in range(max_steps):
            # --- Agent acts ---
            response = agent.act(obs, prev_action=action)
            action = response.completion
            reasoning = response.reasoning if hasattr(response, "reasoning") else ""

            episode_log["action_frequency"][action] += 1
            episode_log["input_tokens"] += response.input_tokens
            episode_log["output_tokens"] += response.output_tokens
            episode_log["total_cost"] += response.cost

            # Parse subgoal signals from reasoning
            subgoal_complete = parse_subgoal_complete(reasoning)
            current_subgoal = parse_subgoal(reasoning)

            # --- Environment step ---
            try:
                obs, reward, terminated, truncated, info = env.step(action)
            except ValueError as e:
                logging.warning(f"Invalid action: {action} led to error-\n{e}")
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

            # Apply perception
            if perception_fn is not None:
                apply_perception(obs, perception_fn)

            # Build observation text for logging
            obs_text = (
                f"{obs['text']['short_term_context']}\n\n"
                f"{'='*10} Start of Direct Observation {'='*10}\n"
                f"{obs['text']['long_term_context']}\n\n"
                f"{'='*10} End of Direct Observation {'='*10}"
            )

            # Record step in buffer
            trajectory_buffer.append({
                "step": step,
                "obs_text": obs_text,
                "action": action,
                "reward": reward,
                "reasoning": reasoning,
                "done": done,
                "subgoal": current_subgoal,
                "subgoal_complete": subgoal_complete,
            })

            # CSV logging
            csv_writer.writerow([step, action, reasoning, obs_text, reward, done])
            csv_file.flush()

            pbar.update(1)

            # --- Check learning trigger ---
            steps_since_trigger += 1
            should_learn = (
                subgoal_complete
                or done
                or steps_since_trigger >= periodic_fallback_steps
            )

            if should_learn and len(trajectory_buffer) > last_trigger_idx:
                segment = trajectory_buffer[last_trigger_idx:]
                trigger_reason = (
                    "subgoal_complete" if subgoal_complete
                    else "episode_end" if done
                    else "periodic_fallback"
                )
                evolve_logger.info(
                    f"Learning trigger at step {step}: {trigger_reason} "
                    f"(segment: {len(segment)} steps, subgoal: {current_subgoal!r})"
                )

                last_trigger_idx = len(trajectory_buffer)
                steps_since_trigger = 0

                # --- Learn from segment ---
                with improve_logging(Path(output_dir)):
                    beliefs, qa_pairs, moments, learn_cost = asyncio.run(
                        learn_from_segment(
                            config=config,
                            beliefs=beliefs,
                            perception=perception,
                            segment=segment,
                            qa_pairs=qa_pairs,
                            moments=moments,
                            default_knowledge=default_knowledge,
                            episode_idx=episode_idx,
                            step=step,
                            is_episode_end=done,
                        )
                    )
                total_learn_cost += learn_cost

                learn_triggers.append({
                    "step": step,
                    "reason": trigger_reason,
                    "segment_length": len(segment),
                    "subgoal": current_subgoal,
                    "learn_cost": learn_cost,
                    "num_qa": len(qa_pairs),
                    "num_moments": len(moments),
                })

                # Inject updated beliefs for next step (if not done)
                if not done:
                    if hasattr(agent, "instruction_text"):
                        agent.instruction_text = beliefs
                    # Also update via prompt_builder for agents that use it
                    if config.eval.get("beliefs_in_system_prompt", False):
                        if env_name == "minihack":
                            from balrog.environments.minihack import get_loaded_instruction_prompt
                        else:
                            from balrog.environments.nle import get_loaded_instruction_prompt
                        instruction_prompt = get_loaded_instruction_prompt(
                            env=env, load=beliefs, task=task
                        )
                        agent.prompt_builder.update_instruction_prompt(instruction_prompt)

            if done:
                episode_log["done"] = True
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
    episode_log["learn_triggers"] = learn_triggers
    episode_log["total_learn_cost"] = total_learn_cost
    episode_log["num_qa_pairs"] = len(qa_pairs)
    episode_log["num_moments"] = len(moments)

    # Save episode log
    json_filename = os.path.join(output_dir, f"{task}_run_{episode_idx:02d}.json")
    with open(json_filename, "w") as f:
        json.dump(episode_log, f, indent=4, default=str)

    env.close()

    evolve_logger.info(
        f"Episode {episode_idx} complete — return: {episode_return:.2f}, "
        f"steps: {step + 1}, triggers: {len(learn_triggers)}, "
        f"learn cost: ${total_learn_cost:.4f}, agent cost: ${episode_log['total_cost']:.4f}"
    )

    return beliefs, perception, qa_pairs, moments, episode_log


# ---------------------------------------------------------------------------
# Episode state persistence helpers
# ---------------------------------------------------------------------------


def _save_episode_artifacts(
    episode_dir: Path,
    beliefs: str,
    perception: str,
    qa_pairs: list[QAPair],
    moments: list[CriticalMoment],
):
    """Save all artifacts for a completed episode."""
    episode_dir.mkdir(parents=True, exist_ok=True)
    (episode_dir / "beliefs.txt").write_text(beliefs)
    (episode_dir / "perception.py").write_text(perception)
    with open(episode_dir / "qa_pairs.json", "w") as f:
        json.dump(serialize_qa_pairs(qa_pairs), f, indent=4)
    with open(episode_dir / "critical_moments.json", "w") as f:
        json.dump(serialize_moments(moments), f, indent=4)


def find_last_completed_episode(
    output_dir: str,
) -> tuple[int, str, str, list[QAPair], list[CriticalMoment]]:
    """Find the last completed episode directory (has beliefs.txt).

    Returns:
        (last_episode, beliefs, perception, qa_pairs, moments)
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return -1, "", "", [], []

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
        return -1, "", "", [], []

    episode_dirs.sort(key=lambda x: x[0])
    last_ep, last_dir = episode_dirs[-1]

    beliefs = (last_dir / "beliefs.txt").read_text()

    perception = ""
    perc_file = last_dir / "perception.py"
    if perc_file.exists():
        perception = perc_file.read_text()

    qa_pairs = []
    qa_file = last_dir / "qa_pairs.json"
    if qa_file.exists():
        try:
            qa_pairs = deserialize_qa_pairs(json.loads(qa_file.read_text()))
        except (json.JSONDecodeError, TypeError):
            pass

    moments = []
    moments_file = last_dir / "critical_moments.json"
    if moments_file.exists():
        try:
            moments = deserialize_moments(json.loads(moments_file.read_text()))
        except (json.JSONDecodeError, TypeError):
            pass

    evolve_logger.info(f"Resuming from episode {last_ep} in {last_dir}")
    return last_ep, beliefs, perception, qa_pairs, moments


# ---------------------------------------------------------------------------
# Outer loop
# ---------------------------------------------------------------------------


def stepwise_explore(config: DictConfig, original_cwd: str, output_dir: str):
    """Run stepwise exploration: episodes with mid-episode learning."""
    evolve_cfg = config.eval.evolve
    num_episodes = evolve_cfg.get("num_episodes", 5)
    periodic_fallback_steps = evolve_cfg.get("periodic_fallback_steps", 30)

    evolve_logger.info("Starting stepwise exploration")
    evolve_logger.info(f"Episodes: {num_episodes}, periodic fallback: {periodic_fallback_steps} steps")

    # Check for resume
    last_ep, beliefs, perception, qa_pairs, moments = find_last_completed_episode(output_dir)
    start_episode = last_ep + 1

    if start_episode > 0:
        evolve_logger.info(f"Resuming from episode {start_episode} ({len(qa_pairs)} QA, {len(moments)} moments)")
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

    cumulative_cost = 0.0

    for episode_idx in range(start_episode, num_episodes):
        evolve_logger.info(f"\n{'='*80}")
        evolve_logger.info(f"STEPWISE EPISODE {episode_idx}/{num_episodes - 1}")
        evolve_logger.info(f"QA pairs: {len(qa_pairs)}, Moments: {len(moments)}")
        evolve_logger.info(f"{'='*80}")

        episode_dir = Path(output_dir) / f"episode_{episode_idx}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Save input state
        (episode_dir / "input_beliefs.txt").write_text(beliefs)
        (episode_dir / "input_perception.py").write_text(perception)

        beliefs, perception, qa_pairs, moments, episode_log = run_stepwise_episode(
            config=config,
            beliefs=beliefs,
            perception=perception,
            qa_pairs=qa_pairs,
            moments=moments,
            default_knowledge=default_knowledge,
            original_cwd=original_cwd,
            output_dir=str(episode_dir),
            episode_idx=episode_idx,
            periodic_fallback_steps=periodic_fallback_steps,
        )

        # Save episode artifacts
        _save_episode_artifacts(episode_dir, beliefs, perception, qa_pairs, moments)

        episode_cost = episode_log.get("total_cost", 0.0) + episode_log.get("total_learn_cost", 0.0)
        cumulative_cost += episode_cost
        evolve_logger.info(f"Episode {episode_idx} cost: ${episode_cost:.4f}")
        evolve_logger.info(f"Cumulative cost: ${cumulative_cost:.4f}")

        _update_summary_json(
            output_dir=output_dir,
            step=episode_idx,
            step_cost=episode_cost,
            cumulative_cost=cumulative_cost,
            rollout_stats={
                "episode_return": episode_log.get("episode_return", 0.0),
                "num_steps": episode_log.get("num_steps", 0),
                "learn_triggers": len(episode_log.get("learn_triggers", [])),
            },
        )

        evolve_logger.info(f"Updated beliefs:\n{beliefs}")


@hydra.main(config_path="BALROG/balrog/config", config_name="config", version_base="1.1")
def main(config: DictConfig):
    run_name_suffix = f"{config.agent.type}_{config.client.model_id.replace('/', '_')}_stepwise"

    original_cwd, output_dir = setup_run(
        config,
        run_name_suffix=run_name_suffix,
        resume_from=config.eval.resume_from,
        output_dir_base=config.eval.output_dir,
        logger_name="evolve",
    )

    stepwise_explore(config, original_cwd, output_dir)


if __name__ == "__main__":
    main()
