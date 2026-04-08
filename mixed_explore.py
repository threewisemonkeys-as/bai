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
from improve import extract_obs_perc_examples
from mixed_improve import (
    QAPair,
    CriticalMoment,
    process_trajectories,
    score_beliefs,
    improve_beliefs_from_scores,
    improve_beliefs_simple,
    improve_perception,
    generate_experiments_from_gaps,
    score_perception_on_moments,
    serialize_qa_pairs,
    deserialize_qa_pairs,
    serialize_moments,
    deserialize_moments,
    _run_perception_on_observation,
)
from rollout import run_explore_rollouts
from run_utils import setup_run, improve_logging, _update_summary_json
from tqdm import tqdm


@dataclass
class MixedExploreConfig:
    num_steps: int
    num_initial_trajectories: int
    num_trajectories_per_iteration: int
    num_belief_improvement: int
    num_experiments: int
    explore_temp: float
    use_qa_pairs: bool
    use_moments: bool
    use_raw_observations: bool


def find_last_completed_step(output_dir: str) -> tuple[int, str, str, list[QAPair], list[CriticalMoment], list[str]]:
    """Find the last completed step (has beliefs.txt).

    Returns:
        (last_step, beliefs, perception, qa_pairs, critical_moments, experiments)
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

    qa_pairs = []
    qa_file = last_step_dir / "qa_pairs.json"
    if qa_file.exists():
        try:
            qa_pairs = deserialize_qa_pairs(json.loads(qa_file.read_text()))
        except (json.JSONDecodeError, TypeError):
            pass

    moments = []
    moments_file = last_step_dir / "critical_moments.json"
    if moments_file.exists():
        try:
            moments = deserialize_moments(json.loads(moments_file.read_text()))
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
    evolve_logger.info(f"Resuming with {len(qa_pairs)} Q&A pairs, {len(moments)} moments, {len(experiments)} experiments")

    return last_step_num, beliefs, perception, qa_pairs, moments, experiments


def _save_step_artifacts(
    step_dir: Path,
    beliefs: str,
    perception: str,
    qa_pairs: list[QAPair],
    moments: list[CriticalMoment],
    experiments: list[str],
    per_traj_extractions: list[dict] | None = None,
    new_qa: list[QAPair] | None = None,
    new_moments: list[CriticalMoment] | None = None,
    scoring_history: list[dict] | None = None,
    perception_history: list[dict] | None = None,
):
    """Save all step artifacts to disk."""
    (step_dir / "beliefs.txt").write_text(beliefs)
    (step_dir / "perception.py").write_text(perception)

    with open(step_dir / "qa_pairs.json", "w") as f:
        json.dump(serialize_qa_pairs(qa_pairs), f, indent=4)

    with open(step_dir / "critical_moments.json", "w") as f:
        json.dump(serialize_moments(moments), f, indent=4)

    with open(step_dir / "experiments.json", "w") as f:
        json.dump(experiments, f, indent=4)

    if per_traj_extractions is not None:
        with open(step_dir / "per_trajectory_extractions.json", "w") as f:
            json.dump(per_traj_extractions, f, indent=4)

    if new_qa is not None:
        with open(step_dir / "new_qa_pairs.json", "w") as f:
            json.dump(serialize_qa_pairs(new_qa), f, indent=4)

    if new_moments is not None:
        with open(step_dir / "new_critical_moments.json", "w") as f:
            json.dump(serialize_moments(new_moments), f, indent=4)

    if scoring_history is not None:
        with open(step_dir / "scoring_history.json", "w") as f:
            json.dump(scoring_history, f, indent=4)

    if perception_history:
        with open(step_dir / "perception_history.json", "w") as f:
            json.dump(perception_history, f, indent=4)


def mixed_explore(
    mixed_config: MixedExploreConfig,
    config: DictConfig,
    original_cwd: str,
    output_dir: str,
):
    """Run mixed exploration loop with structured knowledge extraction."""
    evolve_logger.info("Running mixed exploration with structured knowledge")

    # Determine if we should improve perception
    improve_mode = config.eval.evolve.get("improve_mode", "both")
    do_improve_perception = improve_mode in ("both", "perception")

    # Check for existing progress and resume
    last_step, beliefs, perception, qa_pairs, moments, experiments = find_last_completed_step(output_dir)
    start_step = last_step + 1

    if last_step > 0:
        evolve_logger.info(f"Resuming from step {start_step} (found {last_step} completed steps)")
    else:
        evolve_logger.info("Starting fresh mixed exploration")
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
        experiments = []

    # Fetch default knowledge once
    default_knowledge = get_default_knowledge(config)
    evolve_logger.info(f"Default knowledge extraction complete (length: {len(default_knowledge)})")

    evolve_logger.info(f"Starting mixed exploration with {mixed_config.num_steps} steps")
    evolve_logger.info(f"Initial trajectories: {mixed_config.num_initial_trajectories}")
    evolve_logger.info(f"Trajectories per iteration: {mixed_config.num_trajectories_per_iteration}")
    evolve_logger.info(f"Belief improvement iterations: {mixed_config.num_belief_improvement}")
    evolve_logger.info(f"Experiments per step: {mixed_config.num_experiments}")
    evolve_logger.info(f"Explore temperature: {mixed_config.explore_temp}")
    evolve_logger.info(f"Use Q&A pairs: {mixed_config.use_qa_pairs}")
    evolve_logger.info(f"Use critical moments: {mixed_config.use_moments}")
    evolve_logger.info(f"Improve mode: {improve_mode} (perception improvement: {do_improve_perception})")

    cumulative_cost = 0.0

    for step in range(start_step, mixed_config.num_steps + 1):
        is_initial = (step == 1 and last_step == 0)

        evolve_logger.info(f"\n{'='*80}")
        evolve_logger.info(f"MIXED EXPLORE STEP {step}/{mixed_config.num_steps}"
                           f"{' (INITIAL)' if is_initial else ''}")
        evolve_logger.info(f"Q&A pairs: {len(qa_pairs)}, Moments: {len(moments)}")
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
        # PHASE 1: COLLECT EXPERIENCE
        # ============================================================
        if is_initial:
            num_rollouts = mixed_config.num_initial_trajectories
            config.eval.num_episodes = 1
            use_experiments = []
            evolve_logger.info(f"Phase 1: Collecting {num_rollouts} initial baseline trajectories")
        else:
            num_rollouts = mixed_config.num_trajectories_per_iteration
            config.eval.num_episodes = 1
            use_experiments = experiments
            evolve_logger.info(f"Phase 1: Collecting {num_rollouts} trajectories"
                               f" ({len(use_experiments)} experiments)")

        use_explore_temp = bool(use_experiments)

        with step_logging(step_output_dir):
            logging.info(f"Step {step} rollout logs")
            logging.info(f"Current beliefs:\n{beliefs if beliefs else '(empty)'}")
            logging.info(f"Current perception:\n{perception if perception else '(empty)'}")

            temp_ctx = override_temperature(config, mixed_config.explore_temp) if use_explore_temp else nullcontext()
            with temp_ctx:
                rollout_results = run_explore_rollouts(
                    base_beliefs=beliefs,
                    perception=perception,
                    experiments=use_experiments,
                    config=config,
                    original_cwd=original_cwd,
                    output_dir=str(rollout_dir),
                    num_baseline_rollouts=num_rollouts,
                )

        with open(step_output_dir / "rollout_stats.json", "w") as f:
            json.dump(rollout_results, f, indent=4, default=str)

        _log_rollout_stats(rollout_results, f"Step {step}")
        rollout_cost = _compute_rollout_cost(rollout_results)
        step_cost += rollout_cost

        # ============================================================
        # PHASE 2: LEARN
        # ============================================================
        evolve_logger.info("Phase 2: Learning from trajectories")

        with improve_logging(step_output_dir):
            logging.info(f"Step {step} improve phase logs")

            # Process trajectories: summarize + extract Q&A/moments in parallel
            evolve_logger.info("  2a. Processing trajectories (summarize + extract)...")
            (
                episode_summaries,
                per_traj_qa,
                per_traj_moments,
                new_qa,
                new_moments,
                remove_qa_indices,
                remove_moment_indices,
                process_cost,
                _extraction_records,
            ) = asyncio.run(process_trajectories(
                config=config,
                beliefs=beliefs,
                perception=perception,
                output_dir=str(rollout_dir),
                existing_qa=qa_pairs,
                existing_moments=moments,
                default_knowledge=default_knowledge,
                step=step,
                use_qa=mixed_config.use_qa_pairs,
                use_moments=mixed_config.use_moments,
            ))
            step_cost += process_cost
            evolve_logger.info(f"  2a. Done. Extracted {sum(len(q) for q in per_traj_qa)} Q&A, "
                               f"{sum(len(m) for m in per_traj_moments)} moments from trajectories "
                               f"(cost: ${process_cost:.6f})")

            # 2b. Consolidate structured knowledge
            evolve_logger.info("  2b. Consolidating structured knowledge...")
            if remove_qa_indices:
                removed_qa = [qa_pairs[i] for i in remove_qa_indices]
                qa_pairs = [qa for i, qa in enumerate(qa_pairs) if i not in set(remove_qa_indices)]
                evolve_logger.info(f"Removed {len(remove_qa_indices)} contradicted Q&A pairs:")
                for qa in removed_qa:
                    evolve_logger.info(f"  - Q: {qa.question} | A: {'YES' if qa.answer else 'NO'}")

            if remove_moment_indices:
                removed_moments = [moments[i] for i in remove_moment_indices]
                moments = [m for i, m in enumerate(moments) if i not in set(remove_moment_indices)]
                evolve_logger.info(f"Removed {len(remove_moment_indices)} contradicted moments:")
                for m in removed_moments:
                    evolve_logger.info(f"  - State: {m.state} | Goal: {m.goal} | Good: {m.good_actions} | Bad: {m.bad_actions}")

            # Add new consolidated items
            qa_pairs = qa_pairs + new_qa
            moments = moments + new_moments

            evolve_logger.info(f"  2b. Done. +{len(new_qa)} Q&A, -{len(remove_qa_indices)} Q&A, "
                               f"+{len(new_moments)} moments, -{len(remove_moment_indices)} moments")
            evolve_logger.info(f"      Cumulative: {len(qa_pairs)} Q&A pairs, {len(moments)} moments")

            # 2c. Inner loop: score and improve beliefs
            scoring_history = []
            use_structured = mixed_config.use_qa_pairs or mixed_config.use_moments

            if use_structured and (qa_pairs or moments):
                evolve_logger.info(f"  2c. Scoring and improving beliefs ({mixed_config.num_belief_improvement} max iterations)...")
                prev_qa_score, prev_moment_score = -1.0, -1.0
                for inner_iter in range(mixed_config.num_belief_improvement):
                    evolve_logger.info(f"    Belief iteration {inner_iter + 1}/{mixed_config.num_belief_improvement}")

                    try:
                        # Score current beliefs
                        qa_score, moment_score, score_details, score_cost = asyncio.run(
                            score_beliefs(config, beliefs, qa_pairs, moments, default_knowledge,
                                          perception=perception if mixed_config.use_raw_observations else "",
                                          use_raw_observations=mixed_config.use_raw_observations)
                        )
                        step_cost += score_cost

                        scoring_history.append({
                            "iteration": inner_iter + 1,
                            "qa_score": qa_score,
                            "moment_score": moment_score,
                            "num_qa": len(qa_pairs),
                            "num_moments": len(moments),
                            "details": score_details,
                        })

                        evolve_logger.info(f"      Scores: Q&A={qa_score:.2%}, Moments={moment_score:.2%}")

                        # Early stopping: if scores unchanged from previous iteration
                        if qa_score == prev_qa_score and moment_score == prev_moment_score:
                            evolve_logger.info(f"      Scores unchanged, stopping early at iteration {inner_iter + 1}")
                            break
                        prev_qa_score, prev_moment_score = qa_score, moment_score

                        # Improve beliefs based on scores
                        evolve_logger.info(f"      Improving beliefs...")
                        beliefs, improve_cost = asyncio.run(
                            improve_beliefs_from_scores(
                                config, beliefs, qa_pairs, moments, score_details,
                                default_knowledge, episode_summaries,
                                perception=perception if mixed_config.use_raw_observations else "",
                            )
                        )
                        step_cost += improve_cost

                        evolve_logger.info(f"      Beliefs updated (cost: ${improve_cost:.6f})")
                    except Exception as e:
                        evolve_logger.error(f"      Belief iteration {inner_iter + 1} failed: {e}")
                        logging.exception(f"Inner iteration {inner_iter + 1} failed")
                        break
                evolve_logger.info(f"  2c. Done. Final scores: Q&A={prev_qa_score:.2%}, Moments={prev_moment_score:.2%}")
            else:
                # Fallback: simple improvement from summaries
                evolve_logger.info(f"  2c. Simple belief improvement ({mixed_config.num_belief_improvement} iterations, no structured knowledge)")
                for inner_iter in range(mixed_config.num_belief_improvement):
                    try:
                        evolve_logger.info(f"    Simple iteration {inner_iter + 1}/{mixed_config.num_belief_improvement}...")
                        beliefs, improve_cost = asyncio.run(
                            improve_beliefs_simple(config, beliefs, default_knowledge, episode_summaries)
                        )
                        step_cost += improve_cost
                        evolve_logger.info(f"    Done (cost: ${improve_cost:.6f})")
                        scoring_history.append({
                            "iteration": inner_iter + 1,
                            "mode": "simple",
                        })
                    except Exception as e:
                        evolve_logger.error(f"    Simple iteration {inner_iter + 1} failed: {e}")
                        logging.exception(f"Simple improvement iteration {inner_iter + 1} failed")
                        break
                evolve_logger.info(f"  2c. Done.")

            # 2d. Improve perception if enabled
            perception_history = []
            if do_improve_perception:
                num_perc_iters = config.eval.evolve.get("num_perception_improvement", 1)
                num_perc_candidates = config.eval.evolve.get("num_perception_candidates", 1)
                perc_example_max_trajs = config.eval.evolve.get("perc_example_max_trajs", 1)
                perc_example_max_steps = config.eval.evolve.get("perc_example_max_steps", 3)
                evolve_logger.info(f"  2d. Improving perception module ({num_perc_iters} iteration(s), {num_perc_candidates} candidate(s))...")
                # Capture initial perception and its examples before the loop —
                # episode summaries were generated under this perception, so it
                # provides grounding context for later iterations.
                initial_perception = perception
                initial_perception_examples = extract_obs_perc_examples(
                    str(rollout_dir), initial_perception,
                    max_trajs=perc_example_max_trajs,
                    max_steps_per_traj=perc_example_max_steps,
                )

                # Helper to get a perception I/O example from moments
                obs_moments = [m for m in moments if m.raw_observation and m.raw_observation.strip()]
                sample_moment = obs_moments[len(obs_moments) // 2] if obs_moments else None

                # Build sample observations for execution verification
                # Pick diverse samples: first, middle, last qualifying moment
                sample_obs_for_verify: list[tuple[str, int]] = []
                if obs_moments:
                    indices = {0, len(obs_moments) // 2, len(obs_moments) - 1}
                    for idx in sorted(indices):
                        m = obs_moments[idx]
                        sample_obs_for_verify.append((m.raw_observation, m.traj_step_number))

                def _make_io_example(perc_code):
                    if not sample_moment:
                        return None
                    return {
                        "step_num": sample_moment.traj_step_number,
                        "raw_observation": sample_moment.raw_observation,
                        "perception_output": _run_perception_on_observation(perc_code, sample_moment.raw_observation),
                    }

                # Async helper: run one iteration (score + improve) for a single candidate
                async def _run_one_perc_iteration(cand_perception, perc_iter, cand_idx):
                    perc_examples = extract_obs_perc_examples(
                        str(rollout_dir), cand_perception,
                        max_trajs=perc_example_max_trajs,
                        max_steps_per_traj=perc_example_max_steps,
                        randomize=(perc_iter > 0),
                    )
                    feedback, m_score, s_cost, s_debug = await score_perception_on_moments(
                        config, moments, cand_perception, beliefs=beliefs,
                    )
                    new_perc, i_cost, reasoning, prompt_used = await improve_perception(
                        config, cand_perception, beliefs, episode_summaries,
                        perc_examples, default_knowledge,
                        initial_perception=initial_perception,
                        initial_perception_examples=initial_perception_examples,
                        moment_perception_feedback=feedback,
                        sample_observations=sample_obs_for_verify,
                    )
                    return {
                        "perception": new_perc,
                        "cost": s_cost + i_cost,
                        "perc_moment_score": m_score,
                        "reasoning": reasoning,
                        "scoring_prompt": s_debug.get("scoring_prompt", ""),
                        "scoring_response": s_debug.get("scoring_response", ""),
                        "scoring_feedback": s_debug.get("feedback", ""),
                        "improve_prompt": prompt_used,
                        "io_example": _make_io_example(new_perc),
                    }

                # Initialize candidates — all start from the same initial perception
                candidates = [perception] * num_perc_candidates
                # Per-candidate history: list of lists
                candidate_histories = [[] for _ in range(num_perc_candidates)]

                for perc_iter in tqdm(range(num_perc_iters), desc="Perception improvement", leave=False):
                    try:
                        # Run all candidates in parallel
                        async def _run_all_candidates():
                            return await asyncio.gather(*[
                                _run_one_perc_iteration(cand, perc_iter, ci)
                                for ci, cand in enumerate(candidates)
                            ])
                        results = asyncio.run(_run_all_candidates())
                        iter_cost = 0.0
                        for ci, res in enumerate(results):
                            candidates[ci] = res["perception"]
                            iter_cost += res["cost"]
                            candidate_histories[ci].append({
                                "iteration": perc_iter + 1,
                                **res,
                            })
                            evolve_logger.info(
                                f"    Candidate {ci + 1} iter {perc_iter + 1}: "
                                f"moment_score={res['perc_moment_score']:.2%}, cost=${res['cost']:.6f}"
                            )
                        step_cost += iter_cost
                    except Exception as e:
                        evolve_logger.error(f"    Perception iteration {perc_iter + 1} failed: {e}")
                        logging.exception(f"Perception improvement iteration {perc_iter + 1} failed")
                        break

                # Select best candidate by final moment score
                if num_perc_candidates > 1:
                    # Score all final candidates
                    async def _score_all_final():
                        return await asyncio.gather(*[
                            score_perception_on_moments(config, moments, cand, beliefs=beliefs)
                            for cand in candidates
                        ])
                    final_scores = asyncio.run(_score_all_final())
                    best_idx = 0
                    best_score = -1.0
                    for ci, (_, fscore, fcost, _) in enumerate(final_scores):
                        step_cost += fcost
                        evolve_logger.info(f"    Final score candidate {ci + 1}: {fscore:.2%}")
                        if fscore > best_score:
                            best_score = fscore
                            best_idx = ci
                    perception = candidates[best_idx]
                    evolve_logger.info(f"  2d. Selected candidate {best_idx + 1} (score: {best_score:.2%}, {len(perception)} chars)")
                else:
                    perception = candidates[0]
                    evolve_logger.info(f"  2d. Done. Perception updated ({len(perception)} chars)")

                # Build perception_history for saving
                # For single candidate: flat list (backwards compatible)
                # For multiple candidates: list with candidate_index field
                if num_perc_candidates == 1:
                    perception_history = candidate_histories[0]
                else:
                    perception_history = []
                    for ci, hist in enumerate(candidate_histories):
                        for entry in hist:
                            entry["candidate"] = ci + 1
                            perception_history.append(entry)
                    perception_history.append({
                        "selected_candidate": best_idx + 1,
                        "final_score": best_score,
                        "num_candidates": num_perc_candidates,
                    })

            # 2e. Generate experiments for next step
            evolve_logger.info(f"  2e. Generating {mixed_config.num_experiments} experiments...")
            last_score_details = []
            if scoring_history and "details" in scoring_history[-1]:
                last_score_details = scoring_history[-1]["details"]

            experiments, exp_cost, _, _ = asyncio.run(
                generate_experiments_from_gaps(
                    config, beliefs, qa_pairs, moments, last_score_details,
                    episode_summaries, default_knowledge, mixed_config.num_experiments,
                )
            )
            step_cost += exp_cost
            evolve_logger.info(f"  2e. Done. Generated {len(experiments)} experiments (cost: ${exp_cost:.6f})")

        # Build per-trajectory extractions for saving
        per_traj_extractions = []
        for i, (qa_list, m_list) in enumerate(zip(per_traj_qa, per_traj_moments)):
            per_traj_extractions.append({
                "trajectory_index": i,
                "qa_pairs": serialize_qa_pairs(qa_list),
                "critical_moments": serialize_moments(m_list),
            })

        # Save all artifacts
        _save_step_artifacts(
            step_dir=step_output_dir,
            beliefs=beliefs,
            perception=perception,
            qa_pairs=qa_pairs,
            moments=moments,
            experiments=experiments,
            per_traj_extractions=per_traj_extractions if per_traj_extractions else None,
            new_qa=new_qa if new_qa else None,
            new_moments=new_moments if new_moments else None,
            scoring_history=scoring_history if scoring_history else None,
            perception_history=perception_history if perception_history else None,
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
        evolve_logger.info(f"Generated {len(experiments)} experiments for next step:")
        for i, exp in enumerate(experiments):
            evolve_logger.info(f"  Experiment {i+1}: {exp}")


@hydra.main(config_path="BALROG/balrog/config", config_name="config", version_base="1.1")
def main(config: DictConfig):
    run_name_suffix = f"{config.agent.type}_{config.client.model_id.replace('/', '_')}_mixed_explore"

    original_cwd, output_dir = setup_run(
        config,
        run_name_suffix=run_name_suffix,
        resume_from=config.eval.resume_from,
        output_dir_base=config.eval.output_dir,
        logger_name="evolve",
    )

    evolve_cfg = config.eval.evolve

    mc = MixedExploreConfig(
        num_steps=evolve_cfg.get("num_steps", 10),
        num_initial_trajectories=evolve_cfg.get("num_initial_trajectories", 3),
        num_trajectories_per_iteration=evolve_cfg.get("num_trajectories_per_iteration", 3),
        num_belief_improvement=evolve_cfg.get("num_belief_improvement", 3),
        num_experiments=evolve_cfg.get("num_experiments", 5),
        explore_temp=evolve_cfg.get("explore_temp", 1.0),
        use_qa_pairs=evolve_cfg.get("use_qa_pairs", True),
        use_moments=evolve_cfg.get("use_moments", True),
        use_raw_observations=evolve_cfg.get("use_raw_observations", True),
    )

    mixed_explore(
        mixed_config=mc,
        config=config,
        original_cwd=original_cwd,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
