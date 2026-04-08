import asyncio
import json
import logging
import os
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path

import hydra
from omegaconf import DictConfig

from explore import (
    step_logging,
    get_default_knowledge,
    override_temperature,
    _get_rollout_stats,
    _compute_rollout_cost,
    _log_rollout_stats,
    evolve_logger,
)
from mixed_improve import (
    CriticalMoment,
    QAPair,
    serialize_moments,
    deserialize_moments,
    serialize_qa_pairs,
    deserialize_qa_pairs,
    generate_experiments_from_gaps,
    _read_csv_observations,
    extract_perception_input,
)
from b_learn_improve import (
    forward_pass,
    get_feedback,
    improve_from_feedback,
    extract_and_consolidate_knowledge,
    serialize_feedback_results,
    improve_from_summaries,
    qa_forward_pass,
    qa_get_feedback,
    improve_from_qa_feedback,
    serialize_qa_feedback_results,
)
from rollout import run_explore_rollouts
from run_utils import setup_run, improve_logging, _update_summary_json


@dataclass
class BLearnConfig:
    num_steps: int
    num_trajectories: int
    num_improve_iterations: int
    max_moments_per_forward: int
    explore_temp: float
    num_experiments: int
    max_total_moments: int
    max_total_qa_pairs: int
    max_qa_per_forward: int
    num_sample_obs_per_traj: int
    experiments_per_rollout: int


def _capture_first_frame(config: DictConfig) -> str:
    """Instantiate the environment briefly to capture the first observation."""
    from balrog.environments import make_env
    env_name = config.envs.names.split("-")[0]
    tasks = config.tasks[f"{env_name}_tasks"]
    task = tasks[0] if tasks else env_name
    env = make_env(env_name, task, config)
    obs, info = env.reset()
    first_frame = obs["text"]["long_term_context"]
    short_term = obs["text"].get("short_term_context", "")
    if short_term:
        first_frame = short_term + "\n\n" + first_frame
    env.close()
    return first_frame


def find_last_completed_step(
    output_dir: str,
) -> tuple[int, str, str, list[CriticalMoment], list[QAPair], list[str]]:
    """Find the last completed step (has beliefs.txt).

    Returns:
        (last_step, beliefs, perception, critical_moments, qa_pairs, experiments)
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return 0, "", "", [], [], []

    step_dirs = []
    for item in output_path.iterdir():
        if item.is_dir() and item.name.startswith("step_"):
            try:
                step_num = int(item.name.split("_")[1])
                beliefs_file = item / "beliefs.txt"
                if beliefs_file.exists():
                    step_dirs.append((step_num, item))
            except (ValueError, IndexError):
                continue

    if not step_dirs:
        return 0, "", "", [], [], []

    step_dirs.sort(key=lambda x: x[0])
    last_step_num, last_step_dir = step_dirs[-1]

    beliefs = (last_step_dir / "beliefs.txt").read_text()

    perception = ""
    perc_file = last_step_dir / "perception.py"
    if perc_file.exists():
        perception = perc_file.read_text()

    moments = []
    moments_file = last_step_dir / "critical_moments.json"
    if moments_file.exists():
        try:
            moments = deserialize_moments(json.loads(moments_file.read_text()))
        except (json.JSONDecodeError, TypeError):
            pass

    qa_pairs = []
    qa_file = last_step_dir / "qa_pairs.json"
    if qa_file.exists():
        try:
            qa_pairs = deserialize_qa_pairs(json.loads(qa_file.read_text()))
        except (json.JSONDecodeError, TypeError):
            pass

    experiments = []
    exp_file = last_step_dir / "experiments.json"
    if exp_file.exists():
        try:
            experiments = json.loads(exp_file.read_text())
        except json.JSONDecodeError:
            pass

    evolve_logger.info(f"Found last completed step: {last_step_num}")
    evolve_logger.info(f"Resuming with beliefs from: {last_step_dir / 'beliefs.txt'}")
    if perception:
        evolve_logger.info(f"Resuming with perception from: {perc_file}")
    evolve_logger.info(f"Resuming with {len(moments)} moments, {len(qa_pairs)} QA pairs, "
                       f"{len(experiments)} experiments")

    return last_step_num, beliefs, perception, moments, qa_pairs, experiments


def _save_b_learn_artifacts(
    step_dir: Path,
    beliefs: str,
    perception: str,
    moments: list[CriticalMoment],
    qa_pairs: list[QAPair],
    experiments: list[str],
    feedback_history: list[dict] | None = None,
    per_traj_moments: list[list[CriticalMoment]] | None = None,
    new_moments: list[CriticalMoment] | None = None,
    per_traj_qa: list[list[QAPair]] | None = None,
    new_qa: list[QAPair] | None = None,
):
    """Save all step artifacts to disk."""
    (step_dir / "beliefs.txt").write_text(beliefs)
    (step_dir / "perception.py").write_text(perception)

    with open(step_dir / "critical_moments.json", "w") as f:
        json.dump(serialize_moments(moments), f, indent=4)

    with open(step_dir / "qa_pairs.json", "w") as f:
        json.dump(serialize_qa_pairs(qa_pairs), f, indent=4)

    with open(step_dir / "experiments.json", "w") as f:
        json.dump(experiments, f, indent=4)

    if feedback_history is not None:
        with open(step_dir / "feedback_history.json", "w") as f:
            json.dump(feedback_history, f, indent=4)

    if per_traj_moments is not None:
        with open(step_dir / "per_trajectory_moments.json", "w") as f:
            json.dump([serialize_moments(m_list) for m_list in per_traj_moments], f, indent=4)

    if new_moments is not None:
        with open(step_dir / "new_moments.json", "w") as f:
            json.dump(serialize_moments(new_moments), f, indent=4)

    if per_traj_qa is not None:
        with open(step_dir / "per_trajectory_qa.json", "w") as f:
            json.dump([serialize_qa_pairs(q_list) for q_list in per_traj_qa], f, indent=4)

    if new_qa is not None:
        with open(step_dir / "new_qa_pairs.json", "w") as f:
            json.dump(serialize_qa_pairs(new_qa), f, indent=4)


def b_learn(
    bl_config: BLearnConfig,
    config: DictConfig,
    original_cwd: str,
    output_dir: str,
):
    """Run B-learning loop."""
    evolve_logger.info("Running B-learning")

    # Check for existing progress and resume
    last_step, beliefs, perception, moments, qa_pairs, experiments = find_last_completed_step(output_dir)
    start_step = last_step + 1

    if last_step > 0:
        evolve_logger.info(f"Resuming from step {start_step} (found {last_step} completed steps)")
    else:
        evolve_logger.info("Starting fresh B-learning run")
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
        moments = []
        qa_pairs = []
        experiments = []

    # Fetch default knowledge once
    default_knowledge = get_default_knowledge(config)
    evolve_logger.info(f"Default knowledge extraction complete (length: {len(default_knowledge)})")

    evolve_logger.info(f"B-learning config:")
    evolve_logger.info(f"  Steps: {bl_config.num_steps}")
    evolve_logger.info(f"  Trajectories per step: {bl_config.num_trajectories}")
    evolve_logger.info(f"  Improve iterations: {bl_config.num_improve_iterations}")
    evolve_logger.info(f"  Max moments per forward batch: {bl_config.max_moments_per_forward}")
    evolve_logger.info(f"  Explore temperature: {bl_config.explore_temp}")
    evolve_logger.info(f"  Experiments: {bl_config.num_experiments}")
    evolve_logger.info(f"  Max total moments: {bl_config.max_total_moments}")
    evolve_logger.info(f"  Max total QA pairs: {bl_config.max_total_qa_pairs}")
    evolve_logger.info(f"  Max QA per forward batch: {bl_config.max_qa_per_forward}")
    evolve_logger.info(f"  Sample observations per trajectory: {bl_config.num_sample_obs_per_traj}")
    evolve_logger.info(f"  Experiments per rollout: {bl_config.experiments_per_rollout}")

    cumulative_cost = 0.0

    for step in range(start_step, bl_config.num_steps + 1):
        is_initial = (step == 1 and last_step == 0)

        evolve_logger.info(f"\n{'='*80}")
        evolve_logger.info(f"B-LEARN STEP {step}/{bl_config.num_steps}"
                           f"{' (INITIAL)' if is_initial else ''}")
        evolve_logger.info(f"Moments: {len(moments)}, QA pairs: {len(qa_pairs)}")
        evolve_logger.info(f"{'='*80}")

        step_output_dir = Path(output_dir) / f"step_{step}"
        step_output_dir.mkdir(parents=True, exist_ok=True)

        # Save input state
        (step_output_dir / "input_beliefs.txt").write_text(beliefs)
        (step_output_dir / "input_perception.py").write_text(perception)
        with open(step_output_dir / "input_experiments.json", "w") as f:
            json.dump(experiments, f, indent=4)

        rollout_dir = step_output_dir / "rollouts"
        rollout_dir.mkdir(parents=True, exist_ok=True)

        step_cost = 0.0

        # ============================================================
        # PHASE 0 (INITIAL ONLY): GENERATE EXPERIMENTS FROM FIRST FRAME
        # ============================================================
        if is_initial:
            evolve_logger.info("Phase 0: Capturing first frame and generating initial experiments")
            first_frame = _capture_first_frame(config)
            initial_obs_text = (
                "=== INITIAL ENVIRONMENT OBSERVATION (first frame) ===\n"
                f"{first_frame}\n"
                "=== END INITIAL OBSERVATION ===\n\n"
                "This is the very first interaction. No trajectories have been collected yet. "
                "Design experiments based on the default knowledge and the initial observation above."
            )
            experiments, init_exp_cost, _, _ = asyncio.run(
                generate_experiments_from_gaps(
                    config=config,
                    beliefs=beliefs,
                    qa_pairs=[],
                    critical_moments=[],
                    score_details=[],
                    episode_summaries=initial_obs_text,
                    default_knowledge=default_knowledge,
                    num_experiments=bl_config.num_experiments,
                )
            )
            step_cost += init_exp_cost
            evolve_logger.info(f"Phase 0: Generated {len(experiments)} initial experiments (cost: ${init_exp_cost:.6f})")
            for i, exp in enumerate(experiments):
                evolve_logger.info(f"  Experiment {i+1}: {exp}")

            # Save initial experiments and first frame for debugging
            with open(step_output_dir / "initial_experiments.json", "w") as f:
                json.dump({"first_frame": first_frame, "experiments": experiments}, f, indent=4)

        # ============================================================
        # PHASE 1: COLLECT TRAJECTORIES
        # ============================================================
        num_rollouts = bl_config.num_trajectories
        config.eval.num_episodes = 1
        use_experiments = experiments
        evolve_logger.info(f"Phase 1: Collecting {num_rollouts} trajectories"
                           f" ({len(use_experiments)} experiments)")

        use_explore_temp = bool(use_experiments)

        with step_logging(step_output_dir):
            logging.info(f"Step {step} rollout logs")
            logging.info(f"Current beliefs:\n{beliefs if beliefs else '(empty)'}")
            logging.info(f"Current perception:\n{perception if perception else '(empty)'}")

            temp_ctx = override_temperature(config, bl_config.explore_temp) if use_explore_temp else nullcontext()
            with temp_ctx:
                rollout_results = run_explore_rollouts(
                    base_beliefs=beliefs,
                    perception=perception,
                    experiments=use_experiments,
                    config=config,
                    original_cwd=original_cwd,
                    output_dir=str(rollout_dir),
                    num_baseline_rollouts=num_rollouts,
                    experiments_per_rollout=bl_config.experiments_per_rollout,
                )

        with open(step_output_dir / "rollout_stats.json", "w") as f:
            json.dump(rollout_results, f, indent=4, default=str)

        _log_rollout_stats(rollout_results, f"Step {step}")
        rollout_cost = _compute_rollout_cost(rollout_results)
        step_cost += rollout_cost

        # ============================================================
        # PHASE 2: EXTRACT & CONSOLIDATE KNOWLEDGE (QA + MOMENTS)
        # ============================================================
        evolve_logger.info("Phase 2: Extracting and consolidating knowledge (QA pairs + moments)")

        with improve_logging(step_output_dir):
            logging.info(f"Step {step} improve phase logs")

            (
                episode_summaries,
                per_traj_qa,
                per_traj_moments,
                new_qa,
                new_moments,
                remove_qa_indices,
                remove_moment_indices,
                extract_cost,
                extraction_records,
            ) = asyncio.run(extract_and_consolidate_knowledge(
                config=config,
                beliefs=beliefs,
                perception=perception,
                output_dir=str(rollout_dir),
                existing_qa=qa_pairs,
                existing_moments=moments,
                default_knowledge=default_knowledge,
                step=step,
            ))
            step_cost += extract_cost

            # Apply moment consolidation: remove contradicted, add new
            if remove_moment_indices:
                moments = [m for i, m in enumerate(moments) if i not in set(remove_moment_indices)]
                evolve_logger.info(f"Removed {len(remove_moment_indices)} contradicted moments")
            moments = moments + new_moments

            # Apply QA consolidation: remove contradicted, add new
            if remove_qa_indices:
                qa_pairs = [q for i, q in enumerate(qa_pairs) if i not in set(remove_qa_indices)]
                evolve_logger.info(f"Removed {len(remove_qa_indices)} contradicted QA pairs")
            qa_pairs = qa_pairs + new_qa

            evolve_logger.info(f"Phase 2 done. "
                               f"Moments: +{len(new_moments)} -{len(remove_moment_indices)} = {len(moments)}. "
                               f"QA: +{len(new_qa)} -{len(remove_qa_indices)} = {len(qa_pairs)}. "
                               f"(cost: ${extract_cost:.6f})")

            # Save extraction artifacts immediately
            with open(step_output_dir / "per_trajectory_moments.json", "w") as f:
                json.dump([serialize_moments(m_list) for m_list in per_traj_moments], f, indent=4)
            with open(step_output_dir / "new_moments.json", "w") as f:
                json.dump(serialize_moments(new_moments), f, indent=4)
            with open(step_output_dir / "per_trajectory_qa.json", "w") as f:
                json.dump([serialize_qa_pairs(q_list) for q_list in per_traj_qa], f, indent=4)
            with open(step_output_dir / "new_qa_pairs.json", "w") as f:
                json.dump(serialize_qa_pairs(new_qa), f, indent=4)
            with open(step_output_dir / "extraction_prompts.json", "w") as f:
                json.dump(extraction_records, f, indent=4)

            # Cap total moments
            if len(moments) > bl_config.max_total_moments:
                moments.sort(key=lambda m: m.source_step)
                dropped = len(moments) - bl_config.max_total_moments
                moments = moments[dropped:]
                evolve_logger.info(f"Capped moments: dropped {dropped} oldest, keeping {len(moments)}")

            # Cap total QA pairs
            if len(qa_pairs) > bl_config.max_total_qa_pairs:
                qa_pairs.sort(key=lambda q: q.source_step)
                dropped = len(qa_pairs) - bl_config.max_total_qa_pairs
                qa_pairs = qa_pairs[dropped:]
                evolve_logger.info(f"Capped QA pairs: dropped {dropped} oldest, keeping {len(qa_pairs)}")

            # Filter to moments with raw observations for forward pass
            forward_moments = [m for m in moments if m.raw_observation and m.raw_observation.strip()]
            evolve_logger.info(f"Moments with raw observations for forward pass: {len(forward_moments)}/{len(moments)}")

            # Save consolidated knowledge after capping
            with open(step_output_dir / "critical_moments.json", "w") as f:
                json.dump(serialize_moments(moments), f, indent=4)
            with open(step_output_dir / "qa_pairs.json", "w") as f:
                json.dump(serialize_qa_pairs(qa_pairs), f, indent=4)

            # ============================================================
            # PHASE 3: INNER IMPROVE LOOP
            # ============================================================
            evolve_logger.info(f"Phase 3: Inner improve loop ({bl_config.num_improve_iterations} iterations)")
            feedback_history = []

            # Build sample observations for perception validation
            # Sample directly from trajectory CSVs: start, middle, penultimate step
            # (avoiding the last step which is typically an end-game/death screen)
            sample_obs: list[tuple[str, int]] = []
            csv_files = sorted(rollout_dir.rglob("*.csv"))
            for csv_path in csv_files:
                obs_map = _read_csv_observations(str(csv_path))
                if not obs_map:
                    continue
                steps = sorted(obs_map.keys())
                # Exclude last step (end-game screen)
                if len(steps) > 1:
                    steps = steps[:-1]
                k = bl_config.num_sample_obs_per_traj
                if len(steps) <= k:
                    pick = steps
                else:
                    pick = []
                    for j in range(k):
                        pick.append(steps[int(j * (len(steps) - 1) / (k - 1))])
                for step_num in pick:
                    raw = extract_perception_input(obs_map[step_num])
                    if raw.strip():
                        sample_obs.append((raw, step_num))

            for inner_iter in range(bl_config.num_improve_iterations):
                evolve_logger.info(f"  Inner iteration {inner_iter + 1}/{bl_config.num_improve_iterations}")

                iter_record = {"iteration": inner_iter + 1}

                try:
                    # ========================================
                    # 3.3 Summary + perception improvement
                    # ========================================
                    if episode_summaries:
                        evolve_logger.info(f"    3.3 Summary-based improvement...")
                        beliefs, perception, summary_cost, summary_prompt, summary_response = asyncio.run(
                            improve_from_summaries(
                                config=config,
                                beliefs=beliefs,
                                perception=perception,
                                episode_summaries=episode_summaries,
                                sample_observations=sample_obs if sample_obs else None,
                                default_knowledge=default_knowledge,
                            )
                        )
                        step_cost += summary_cost
                        evolve_logger.info(f"    3.3 Done (cost: ${summary_cost:.6f})")
                        iter_record["summary_cost"] = summary_cost
                        iter_record["summary_improve_prompt"] = summary_prompt
                        iter_record["summary_improve_response"] = summary_response

                        # Save after summary improvement
                        (step_output_dir / "beliefs.txt").write_text(beliefs)
                        (step_output_dir / "perception.py").write_text(perception)
                    else:
                        evolve_logger.info(f"    3.3 No summaries, skipping summary improvement")

                    # ========================================
                    # 3.4 QA-based improvement
                    # ========================================
                    if qa_pairs:
                        evolve_logger.info(f"    3.4a QA forward pass on {len(qa_pairs)} questions...")
                        qa_fwd_results, qa_fwd_cost, _, _ = asyncio.run(
                            qa_forward_pass(
                                config=config,
                                beliefs=beliefs,
                                qa_pairs=qa_pairs,
                                max_per_batch=bl_config.max_qa_per_forward,
                            )
                        )
                        step_cost += qa_fwd_cost
                        evolve_logger.info(f"    3.4a Done (cost: ${qa_fwd_cost:.6f})")

                        evolve_logger.info(f"    3.4b QA feedback on {len(qa_fwd_results)} predictions...")
                        qa_fb_results, qa_fb_cost, qa_fb_prompts, qa_fb_responses = asyncio.run(
                            qa_get_feedback(
                                config=config,
                                qa_forward_results=qa_fwd_results,
                                max_per_batch=bl_config.max_qa_per_forward,
                            )
                        )
                        step_cost += qa_fb_cost

                        # Categorize QA feedback
                        qa_correct = [fr for fr in qa_fb_results if fr.verdict == "CORRECT"]
                        qa_incorrect = [fr for fr in qa_fb_results if fr.verdict == "INCORRECT"]
                        qa_inconclusive = [fr for fr in qa_fb_results if fr.verdict == "INCONCLUSIVE"]
                        qa_actionable = [fr for fr in qa_fb_results if fr.verdict != "INCONCLUSIVE"]

                        evolve_logger.info(f"    3.4b Done. {len(qa_correct)} correct, {len(qa_incorrect)} incorrect, "
                                           f"{len(qa_inconclusive)} inconclusive (cost: ${qa_fb_cost:.6f})")

                        iter_record["qa_num_pairs"] = len(qa_pairs)
                        iter_record["qa_num_correct"] = len(qa_correct)
                        iter_record["qa_num_incorrect"] = len(qa_incorrect)
                        iter_record["qa_num_inconclusive"] = len(qa_inconclusive)
                        iter_record["qa_forward_cost"] = qa_fwd_cost
                        iter_record["qa_feedback_cost"] = qa_fb_cost
                        iter_record["qa_feedback_details"] = serialize_qa_feedback_results(qa_fb_results)
                        iter_record["qa_feedback_prompts"] = qa_fb_prompts
                        iter_record["qa_feedback_responses"] = qa_fb_responses

                        if qa_actionable and qa_incorrect:
                            evolve_logger.info(f"    3.4c Improving from {len(qa_actionable)} QA feedback items...")
                            beliefs, perception, qa_improve_cost, qa_improve_prompt, qa_improve_response = asyncio.run(
                                improve_from_qa_feedback(
                                    config=config,
                                    beliefs=beliefs,
                                    perception=perception,
                                    qa_feedback_results=qa_actionable,
                                    default_knowledge=default_knowledge,
                                    episode_summaries=episode_summaries,
                                    sample_observations=sample_obs if sample_obs else None,
                                )
                            )
                            step_cost += qa_improve_cost
                            evolve_logger.info(f"    3.4c Done (cost: ${qa_improve_cost:.6f})")
                            iter_record["qa_improve_cost"] = qa_improve_cost
                            iter_record["qa_improve_prompt"] = qa_improve_prompt
                            iter_record["qa_improve_response"] = qa_improve_response

                            # Save after QA improvement
                            (step_output_dir / "beliefs.txt").write_text(beliefs)
                            (step_output_dir / "perception.py").write_text(perception)
                        else:
                            evolve_logger.info(f"    3.4c No incorrect QA predictions, skipping QA improvement")
                    else:
                        evolve_logger.info(f"    3.4 No QA pairs, skipping QA improvement")

                    # ========================================
                    # 3.5 Critical moment improvement
                    # ========================================
                    if not forward_moments:
                        evolve_logger.info("    3.5 No moments with raw observations, skipping moment improvement")
                    else:
                        # 3.5a. Forward pass
                        evolve_logger.info(f"    3.5a Forward pass on {len(forward_moments)} moments...")
                        forward_results, fwd_cost = asyncio.run(
                            forward_pass(
                                config, beliefs, perception, forward_moments,
                                max_per_batch=bl_config.max_moments_per_forward,
                            )
                        )
                        step_cost += fwd_cost
                        evolve_logger.info(f"    3.5a Done (cost: ${fwd_cost:.6f})")

                        # 3.5b. Get feedback
                        evolve_logger.info(f"    3.5b Getting feedback on {len(forward_results)} predictions...")
                        feedback_results, fb_cost, fb_prompts, fb_responses = asyncio.run(
                            get_feedback(
                                config, forward_results,
                                max_per_batch=bl_config.max_moments_per_forward,
                            )
                        )
                        step_cost += fb_cost

                        # Categorize
                        correct = [fr for fr in feedback_results if fr.verdict == "CORRECT"]
                        incorrect = [fr for fr in feedback_results if fr.verdict == "INCORRECT"]
                        inconclusive = [fr for fr in feedback_results if fr.verdict == "INCONCLUSIVE"]
                        actionable = [fr for fr in feedback_results if fr.verdict != "INCONCLUSIVE"]

                        evolve_logger.info(f"    3.5b Done. {len(correct)} correct, {len(incorrect)} incorrect, "
                                           f"{len(inconclusive)} inconclusive (cost: ${fb_cost:.6f})")

                        iter_record["moment_num_moments"] = len(forward_moments)
                        iter_record["moment_num_correct"] = len(correct)
                        iter_record["moment_num_incorrect"] = len(incorrect)
                        iter_record["moment_num_inconclusive"] = len(inconclusive)
                        iter_record["moment_forward_cost"] = fwd_cost
                        iter_record["moment_feedback_cost"] = fb_cost
                        iter_record["moment_feedback_details"] = serialize_feedback_results(feedback_results)
                        iter_record["moment_feedback_prompts"] = fb_prompts
                        iter_record["moment_feedback_responses"] = fb_responses

                        # 3.5c. Improve from moment feedback
                        if actionable and incorrect:
                            evolve_logger.info(f"    3.5c Improving from {len(actionable)} moment feedback items...")
                            beliefs, perception, improve_cost, improve_prompt, improve_response = asyncio.run(
                                improve_from_feedback(
                                    config, beliefs, perception, actionable,
                                    default_knowledge, episode_summaries,
                                    sample_observations=sample_obs if sample_obs else None,
                                )
                            )
                            step_cost += improve_cost
                            evolve_logger.info(f"    3.5c Done (cost: ${improve_cost:.6f})")
                            iter_record["moment_improve_cost"] = improve_cost
                            iter_record["moment_improve_prompt"] = improve_prompt
                            iter_record["moment_improve_response"] = improve_response

                            # Save after moment improvement
                            (step_output_dir / "beliefs.txt").write_text(beliefs)
                            (step_output_dir / "perception.py").write_text(perception)
                        else:
                            evolve_logger.info(f"    3.5c No incorrect moment predictions, skipping moment improvement")

                    feedback_history.append(iter_record)

                    # Save feedback history incrementally
                    with open(step_output_dir / "feedback_history.json", "w") as f:
                        json.dump(feedback_history, f, indent=4)

                except Exception as e:
                    evolve_logger.error(f"    Inner iteration {inner_iter + 1} failed: {e}")
                    logging.exception(f"Inner iteration {inner_iter + 1} failed")
                    break

            evolve_logger.info(f"Phase 3 done.")

            # ============================================================
            # PHASE 4: GENERATE EXPERIMENTS
            # ============================================================
            evolve_logger.info(f"Phase 4: Generating {bl_config.num_experiments} experiments")
            experiments, exp_cost, _, _ = asyncio.run(
                generate_experiments_from_gaps(
                    config=config,
                    beliefs=beliefs,
                    qa_pairs=qa_pairs,
                    critical_moments=moments,
                    score_details=[],
                    episode_summaries=episode_summaries,
                    default_knowledge=default_knowledge,
                    num_experiments=bl_config.num_experiments,
                )
            )
            step_cost += exp_cost
            evolve_logger.info(f"Phase 4 done. Generated {len(experiments)} experiments (cost: ${exp_cost:.6f})")

            # Save experiments immediately
            with open(step_output_dir / "experiments.json", "w") as f:
                json.dump(experiments, f, indent=4)

        # ============================================================
        # PHASE 5: FINAL SAVE & LOG
        # ============================================================
        # Final save ensures all artifacts are consistent at step completion
        _save_b_learn_artifacts(
            step_dir=step_output_dir,
            beliefs=beliefs,
            perception=perception,
            moments=moments,
            qa_pairs=qa_pairs,
            experiments=experiments,
            feedback_history=feedback_history if feedback_history else None,
            per_traj_moments=per_traj_moments if per_traj_moments else None,
            new_moments=new_moments if new_moments else None,
            per_traj_qa=per_traj_qa if per_traj_qa else None,
            new_qa=new_qa if new_qa else None,
        )

        cumulative_cost += step_cost
        evolve_logger.info(f"Step {step} cost: ${step_cost:.6f}")
        evolve_logger.info(f"Cumulative cost after step {step}: ${cumulative_cost:.6f}")

        _update_summary_json(
            output_dir=output_dir,
            step=step,
            step_cost=step_cost,
            cumulative_cost=cumulative_cost,
            rollout_stats=_get_rollout_stats(rollout_results),
        )

        evolve_logger.info(f"Step {step} completed.")
        evolve_logger.info(f"Updated beliefs:\n{beliefs}")
        if perception:
            evolve_logger.info(f"Updated perception ({len(perception)} chars)")
        evolve_logger.info(f"QA pairs: {len(qa_pairs)}, Moments: {len(moments)}")
        evolve_logger.info(f"Generated {len(experiments)} experiments for next step:")
        for i, exp in enumerate(experiments):
            evolve_logger.info(f"  Experiment {i+1}: {exp}")


@hydra.main(config_path="BALROG/balrog/config", config_name="config", version_base="1.1")
def main(config: DictConfig):
    run_name_suffix = f"{config.agent.type}_{config.client.model_id.replace('/', '_')}_b_learn"

    original_cwd, output_dir = setup_run(
        config,
        run_name_suffix=run_name_suffix,
        resume_from=config.eval.resume_from,
        output_dir_base=config.eval.output_dir,
        logger_name="evolve",
    )

    evolve_cfg = config.eval.evolve

    bl_config = BLearnConfig(
        num_steps=evolve_cfg.get("num_steps", 1),
        num_trajectories=evolve_cfg.get("num_trajectories", 1),
        num_improve_iterations=evolve_cfg.get("num_improve_iterations", 5),
        max_moments_per_forward=evolve_cfg.get("max_moments_per_forward", 10),
        explore_temp=evolve_cfg.get("explore_temp", 1.0),
        num_experiments=evolve_cfg.get("num_experiments", 5),
        max_total_moments=evolve_cfg.get("max_total_moments", 50),
        max_total_qa_pairs=evolve_cfg.get("max_total_qa_pairs", 50),
        max_qa_per_forward=evolve_cfg.get("max_qa_per_forward", 10),
        num_sample_obs_per_traj=evolve_cfg.get("num_sample_obs_per_traj", 3),
        experiments_per_rollout=evolve_cfg.get("experiments_per_rollout", 1),
    )

    b_learn(
        bl_config=bl_config,
        config=config,
        original_cwd=original_cwd,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
