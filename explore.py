import json
import logging
import os
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from balrog.utils import setup_environment
from balrog.environments import make_env
from balrog.environments.nle import get_loaded_instruction_prompt
from improve import (
    improve_step,
    prepare_improve_context,
    generate_candidate_beliefs,
    generate_experiments_from_baseline,
    analyze_all_experiment_trajectories,
    score_candidate_beliefs,
)
from rollout import run_explore_rollouts


@contextmanager
def override_temperature(config: DictConfig, temperature: float):
    """Temporarily override client.generate_kwargs.temperature."""
    original = config.client.generate_kwargs.get("temperature")
    config.client.generate_kwargs.temperature = temperature
    try:
        yield
    finally:
        config.client.generate_kwargs.temperature = original


# Top-level logger for high-level evolution logs (eval.log)
evolve_logger = logging.getLogger("evolve")


@contextmanager
def step_logging(step_output_dir: Path):
    """Context manager to redirect all logging to a step-specific log file.

    During the context, all log messages go to step_output_dir/step.log.
    After exiting, logging returns to normal (eval.log).
    """
    step_log_file = step_output_dir / "step.log"

    # Create a handler for the step log
    step_handler = logging.FileHandler(step_log_file)
    step_handler.setLevel(logging.INFO)
    step_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # Get the root logger and save its current handlers
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers.copy()

    # Replace handlers with step handler
    root_logger.handlers = [step_handler]

    try:
        yield step_log_file
    finally:
        # Restore original handlers
        step_handler.close()
        root_logger.handlers = original_handlers


@contextmanager
def improve_logging(step_output_dir: Path):
    """Context manager to redirect all logging to an improve-specific log file.

    During the context, all log messages go to step_output_dir/improve.log.
    After exiting, logging returns to normal (eval.log).
    This is used for logging LLM prompts and outputs during the improve phase.
    """
    improve_log_file = step_output_dir / "improve.log"

    # Create a handler for the improve log
    improve_handler = logging.FileHandler(improve_log_file)
    improve_handler.setLevel(logging.INFO)
    improve_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # Get the root logger and save its current handlers
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers.copy()

    # Replace handlers with improve handler
    root_logger.handlers = [improve_handler]

    try:
        yield improve_log_file
    finally:
        # Restore original handlers
        improve_handler.close()
        root_logger.handlers = original_handlers


@dataclass
class ExploreConfig:
    num_steps: int
    rollouts_per_step: int
    num_experiments: int
    num_baseline_rollouts: int
    num_candidates: int = 3
    num_score_experiments: int = 5
    min_experiments_for_scoring: int = 3


def _get_rollout_stats(rollout_results: dict[str, dict]) -> dict:
    """Calculate rollout success statistics and return as a dict."""
    total_rollouts = len(rollout_results)
    successful_rollouts = 0
    partial_rollouts = 0
    failed_rollouts = 0
    error_rollouts = 0

    for run_name, result in rollout_results.items():
        if "error" in result:
            error_rollouts += 1
            continue
        summary = result.get("summary", {})
        for env_name, env_stats in summary.items():
            if env_stats.get("num_perfect", 0) > 0:
                successful_rollouts += 1
            elif env_stats.get("avg_prog", 0) > 0:
                partial_rollouts += 1
            else:
                failed_rollouts += 1

    return {
        "total": total_rollouts,
        "successful": successful_rollouts,
        "partial": partial_rollouts,
        "failed": failed_rollouts,
        "errors": error_rollouts,
    }


def _log_rollout_stats(rollout_results: dict[str, dict], phase_name: str):
    """Calculate and log rollout success statistics."""
    stats = _get_rollout_stats(rollout_results)
    evolve_logger.info(
        f"[{phase_name}] Rollout results: {stats['total']} total, {stats['successful']} successful (100%), "
        f"{stats['partial']} partial progress, {stats['failed']} failed (0%), {stats['errors']} errors"
    )


def _compute_rollout_cost(rollout_results: dict[str, dict]) -> float:
    """Sum up rollout costs from results."""
    cost = 0.0
    for _, result in rollout_results.items():
        if "summary" in result:
            for env_stats in result["summary"].values():
                cost += env_stats.get("total_cost", 0)
    return cost


def _update_summary_json(output_dir: str, step: int, step_cost: float, cumulative_cost: float,
                         rollout_stats: dict, phase_stats: dict | None = None):
    """Update the summary.json file with data from the completed step.

    Args:
        output_dir: Root output directory containing summary.json
        step: Current step number
        step_cost: Total cost for this step
        cumulative_cost: Cumulative cost up to and including this step
        rollout_stats: Dict with keys: total, successful, partial, failed, errors
        phase_stats: Optional dict of per-phase stats (e.g. {"baseline": {...}, "explore": {...}})
    """
    summary_path = Path(output_dir) / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = {"steps": []}

    step_data = {
        "step": step,
        "step_cost": step_cost,
        "cumulative_cost": cumulative_cost,
        "rollout_stats": rollout_stats,
    }
    if phase_stats:
        step_data["phase_stats"] = phase_stats

    summary["steps"].append(step_data)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)


def get_default_knowledge(config: DictConfig) -> str:
    """Get the default instructions/knowledge for the environment.

    Args:
        config: BALROG configuration

    Returns:
        String containing default instructions (actions, goal, etc.)
    """
    env_name = config.envs.names.split("-")[0]
    # Get the first task for this environment
    tasks = config.tasks[f"{env_name}_tasks"]
    if not tasks:
        return ""
    
    task = tasks[0]
    logging.info(f"Extracting default knowledge for env: {env_name}, task: {task}")
    
    try:
        env = make_env(env_name, task, config)
        instruction_prompt = get_loaded_instruction_prompt(env, load="", task=task)
        env.close()
        return instruction_prompt
    except Exception as e:
        logging.warning(f"Failed to extract default knowledge: {e}")
        return ""


def find_last_completed_step(
    output_dir: str, two_step: bool = False, experiment_guided: bool = False,
) -> tuple[int, str, str, list[str], dict[str, list[dict]]]:
    """Find the last completed step with beliefs.txt, perception.py, and experiments.json files.

    Args:
        output_dir: Output directory containing step folders
        two_step: If True, look for completed explore phase in step_N/explore/ subdirectory
                  (two-step mode). If False, look directly in step_N/ (single-step mode).
        experiment_guided: If True, also load experiments.json and experiment_results.json
                           from the last completed step.

    Returns:
        Tuple of (last_step_number, beliefs_content, perception_content, experiments_list, experiment_results).
        In two_step mode, experiments_list is always empty (Phase A generates them fresh).
        experiment_results is only populated when experiment_guided=True.
        Returns (0, "", "", [], {}) if no steps found.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return 0, "", "", [], {}

    # Find all step directories
    step_dirs = []
    for item in output_path.iterdir():
        if item.is_dir() and item.name.startswith("step_"):
            try:
                step_num = int(item.name.split("_")[1])
                if two_step:
                    # Two-step mode: look in step_N/explore/ for final phase output
                    explore_dir = item / "explore"
                    beliefs_file = explore_dir / "beliefs.txt"
                    perception_file = explore_dir / "perception.py"
                else:
                    # Single-step mode: look directly in step_N/
                    beliefs_file = item / "beliefs.txt"
                    perception_file = item / "perception.py"

                experiments_file = item / "experiments.json"
                if beliefs_file.exists() and perception_file.exists():
                    step_dirs.append((step_num, item, beliefs_file, perception_file, experiments_file))
            except (ValueError, IndexError):
                continue

    if not step_dirs:
        return 0, "", "", [], {}

    # Sort by step number and get the last one
    step_dirs.sort(key=lambda x: x[0])
    last_step_num, last_step_dir, last_beliefs_file, last_perception_file, last_experiments_file = step_dirs[-1]
    beliefs_content = last_beliefs_file.read_text()
    perception_content = last_perception_file.read_text()

    experiments = []
    if not two_step and last_experiments_file.exists():
        # Only load experiments in single-step mode; two-step generates them fresh each step
        try:
            experiments = json.loads(last_experiments_file.read_text())
        except json.JSONDecodeError:
            experiments = []

    experiment_results = {}
    if experiment_guided:
        exp_results_file = last_step_dir / "experiment_results.json"
        if exp_results_file.exists():
            try:
                experiment_results = json.loads(exp_results_file.read_text())
            except json.JSONDecodeError:
                experiment_results = {}
        # In experiment-guided mode, always load experiments from the step dir
        exp_file = last_step_dir / "experiments.json"
        if exp_file.exists():
            try:
                experiments = json.loads(exp_file.read_text())
            except json.JSONDecodeError:
                pass

    evolve_logger.info(f"Found last completed step: {last_step_num}")
    evolve_logger.info(f"Resuming with beliefs from: {last_beliefs_file}")
    evolve_logger.info(f"Resuming with perception from: {last_perception_file}")

    return last_step_num, beliefs_content, perception_content, experiments, experiment_results


def online_explore(
    explore_config: ExploreConfig,
    config: DictConfig,
    original_cwd: str,
    output_dir: str,
):
    """Run online exploration loop with unified Explore -> Improve steps.

    Args:
        explore_config: Exploration configuration
        config: BALROG configuration
        original_cwd: Original working directory
        output_dir: Output directory for results
    """
    evolve_logger.info("Running exploration with experiment generation")

    # Check for existing progress and resume if available
    last_step, b, p, experiments, _ = find_last_completed_step(output_dir)
    start_step = last_step + 1

    if last_step > 0:
        evolve_logger.info(f"Resuming from step {start_step} (found {last_step} completed steps)")
        evolve_logger.info(f"Resuming with {len(experiments)} experiments from previous step")
    else:
        evolve_logger.info("Starting fresh exploration (no existing steps found)")
        # Load initial beliefs from file if specified
        if (beliefs_path := config.eval.get("beliefs_path", None)) is not None:
            b = Path(beliefs_path).read_text()
            evolve_logger.info(f"Loaded initial beliefs from: {beliefs_path}")
        else:
            b = ""

        # Load initial perception from file if specified
        if (perception_path := config.eval.get("perception_path", None)) is not None:
            p = Path(perception_path).read_text()
            evolve_logger.info(f"Loaded initial perception from: {perception_path}")
        else:
            p = ""
        experiments = []  # Start with no experiments -> will trigger baseline rollouts

    config.eval.num_episodes = explore_config.rollouts_per_step
    explore_temp = config.eval.evolve.get("explore_temp", 1.0)

    # Fetch default knowledge once
    default_knowledge = get_default_knowledge(config)
    evolve_logger.info(f"Default knowledge extraction complete (length: {len(default_knowledge)})")

    evolve_logger.info(f"Starting exploration with {explore_config.num_steps} steps")
    evolve_logger.info(f"Rollouts per step: {explore_config.rollouts_per_step}")
    evolve_logger.info(f"Experiments per step: {explore_config.num_experiments}")
    evolve_logger.info(f"Baseline rollouts: {explore_config.num_baseline_rollouts}")
    evolve_logger.info(f"Explore temperature: {explore_temp}")

    cumulative_cost = 0.0

    for step in range(start_step, explore_config.num_steps + 1):
        evolve_logger.info(f"\n{'='*80}")
        evolve_logger.info(f"EXPLORATION STEP {step}/{explore_config.num_steps}")
        evolve_logger.info(f"Experiments to test: {len(experiments)}")
        if experiments:
            for i, exp in enumerate(experiments):
                evolve_logger.info(f"  Experiment {i+1}: {exp}")
        else:
            evolve_logger.info("  (no experiments - running baseline rollouts)")
        evolve_logger.info(f"{'='*80}")

        step_output_dir = Path(output_dir) / f"step_{step}"
        step_output_dir.mkdir(parents=True, exist_ok=True)

        # Save current state inputs (outside logging context)
        (step_output_dir / "input_beliefs.txt").write_text(b)
        (step_output_dir / "input_perception.txt").write_text(p)
        with open(step_output_dir / "input_experiments.json", "w") as f:
            json.dump(experiments, f, indent=4)

        rollout_dir = step_output_dir / "rollouts"
        rollout_dir.mkdir(parents=True, exist_ok=True)

        # Use explore_temp for experiment rollouts, default temp for baseline
        use_explore_temp = bool(experiments)

        # Phase 1: Explore (Rollouts) - with step-specific logging for agent steps
        with step_logging(step_output_dir):
            logging.info(f"Step {step} agent rollout logs")
            logging.info(f"Current beliefs:\n{b if b else '(empty)'}")
            logging.info(f"Current perception:\n{p if p else '(empty)'}")

            logging.info("=== Phase 1: Explore (Rollouts) ===")

            temp_ctx = override_temperature(config, explore_temp) if use_explore_temp else nullcontext()
            with temp_ctx:
                rollout_results = run_explore_rollouts(
                    base_beliefs=b,
                    perception=p,
                    experiments=experiments,
                    config=config,
                    original_cwd=original_cwd,
                    output_dir=str(rollout_dir),
                    num_baseline_rollouts=explore_config.num_baseline_rollouts,
                )

        # Save detailed rollout stats (outside logging context)
        with open(step_output_dir / "rollout_stats.json", "w") as f:
            json.dump(rollout_results, f, indent=4, default=str)

        _log_rollout_stats(rollout_results, "Explore")

        # Phase 2: Improve (Analyze -> Update -> Generate Experiments) - with improve-specific logging
        with improve_logging(step_output_dir):
            logging.info(f"Step {step} improve phase logs")
            logging.info("=== Phase 2: Improve (Update & Generate) ===")
            logging.info(f"Current beliefs:\n{b if b else '(empty)'}")
            logging.info(f"Current perception:\n{p if p else '(empty)'}")

            new_b, new_p, new_experiments, improve_cost = improve_step(
                config=config,
                base_beliefs=b,
                perception=p,
                output_dir=str(rollout_dir),
                previous_experiments=experiments,
                default_knowledge=default_knowledge,
                num_experiments=explore_config.num_experiments,
                rollout_results=rollout_results,
            )

        # Update state for next step
        b = new_b
        p = new_p
        experiments = new_experiments

        # Save updated state (outside logging context)
        (step_output_dir / "beliefs.txt").write_text(b)
        (step_output_dir / "perception.py").write_text(p)
        with open(step_output_dir / "experiments.json", "w") as f:
            json.dump(experiments, f, indent=4)

        # Calculate and log costs
        rollout_cost = _compute_rollout_cost(rollout_results)
        total_step_cost = rollout_cost + improve_cost
        cumulative_cost += total_step_cost
        evolve_logger.info(f"Step {step} costs: rollout=${rollout_cost:.6f}, improve=${improve_cost:.6f}, total=${total_step_cost:.6f}")
        evolve_logger.info(f"Cumulative cost after step {step}: ${cumulative_cost:.6f}")

        _update_summary_json(
            output_dir=output_dir,
            step=step,
            step_cost=total_step_cost,
            cumulative_cost=cumulative_cost,
            rollout_stats=_get_rollout_stats(rollout_results),
        )

        # Log summary to eval.log
        evolve_logger.info(f"Step {step} completed.")
        evolve_logger.info(f"Updated beliefs:\n{b}")
        evolve_logger.info(f"Generated {len(experiments)} new experiments for next step:")
        for i, exp in enumerate(experiments):
            evolve_logger.info(f"  Experiment {i+1}: {exp}")


def online_explore_2step(
    explore_config: ExploreConfig,
    config: DictConfig,
    original_cwd: str,
    output_dir: str,
):
    """Run 2-step online exploration loop.

    Each iteration consists of:
      Phase A (Baseline): Rollouts without experiments -> Improve (updates beliefs/perception, generates experiments)
      Phase B (Explore):  Rollouts with experiments   -> Improve (evaluates experiments, updates beliefs/perception, no new experiments)

    The baseline phase focuses on pure gameplay and generates experiments to test.
    The explore phase tests those experiments and incorporates findings.

    Args:
        explore_config: Exploration configuration
        config: BALROG configuration
        original_cwd: Original working directory
        output_dir: Output directory for results
    """
    evolve_logger.info("Running 2-step exploration (baseline + explore per iteration)")

    # Check for existing progress and resume if available
    last_step, b, p, _, _ = find_last_completed_step(output_dir, two_step=True)
    start_step = last_step + 1

    if last_step > 0:
        evolve_logger.info(f"Resuming from step {start_step} (found {last_step} completed steps)")
    else:
        evolve_logger.info("Starting fresh 2-step exploration (no existing steps found)")
        # Load initial beliefs from file if specified
        if (beliefs_path := config.eval.get("beliefs_path", None)) is not None:
            b = Path(beliefs_path).read_text()
            evolve_logger.info(f"Loaded initial beliefs from: {beliefs_path}")
        else:
            b = ""

        # Load initial perception from file if specified
        if (perception_path := config.eval.get("perception_path", None)) is not None:
            p = Path(perception_path).read_text()
            evolve_logger.info(f"Loaded initial perception from: {perception_path}")
        else:
            p = ""
    # Experiments start empty each step; Phase A.2 generates them fresh
    experiments = []

    config.eval.num_episodes = explore_config.rollouts_per_step
    explore_temp = config.eval.evolve.get("explore_temp", 1.0)

    # Fetch default knowledge once
    default_knowledge = get_default_knowledge(config)
    evolve_logger.info(f"Default knowledge extraction complete (length: {len(default_knowledge)})")

    evolve_logger.info(f"Starting 2-step exploration with {explore_config.num_steps} steps")
    evolve_logger.info(f"Rollouts per step: {explore_config.rollouts_per_step}")
    evolve_logger.info(f"Experiments per step: {explore_config.num_experiments}")
    evolve_logger.info(f"Baseline rollouts: {explore_config.num_baseline_rollouts}")
    evolve_logger.info(f"Explore temperature: {explore_temp}")

    cumulative_cost = 0.0

    for step in range(start_step, explore_config.num_steps + 1):
        evolve_logger.info(f"\n{'='*80}")
        evolve_logger.info(f"2-STEP EXPLORATION STEP {step}/{explore_config.num_steps}")
        evolve_logger.info(f"{'='*80}")

        step_output_dir = Path(output_dir) / f"step_{step}"
        step_output_dir.mkdir(parents=True, exist_ok=True)

        # Save current state inputs
        (step_output_dir / "input_beliefs.txt").write_text(b)
        (step_output_dir / "input_perception.txt").write_text(p)

        # ================================================================
        # PHASE A: Baseline (no experiments)
        # ================================================================
        evolve_logger.info("--- Phase A: Baseline (no experiments) ---")

        baseline_dir = step_output_dir / "baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        baseline_rollout_dir = baseline_dir / "rollouts"
        baseline_rollout_dir.mkdir(parents=True, exist_ok=True)

        # Phase A.1: Baseline rollouts (no experiments)
        with step_logging(baseline_dir):
            logging.info(f"Step {step} Phase A: Baseline rollout logs")
            logging.info(f"Current beliefs:\n{b if b else '(empty)'}")
            logging.info(f"Current perception:\n{p if p else '(empty)'}")
            logging.info("=== Phase A.1: Baseline Rollouts (no experiments) ===")

            baseline_rollout_results = run_explore_rollouts(
                base_beliefs=b,
                perception=p,
                experiments=[],  # No experiments - just play the game
                config=config,
                original_cwd=original_cwd,
                output_dir=str(baseline_rollout_dir),
                num_baseline_rollouts=explore_config.num_baseline_rollouts,
            )

        # Save baseline rollout stats
        with open(baseline_dir / "rollout_stats.json", "w") as f:
            json.dump(baseline_rollout_results, f, indent=4, default=str)

        _log_rollout_stats(baseline_rollout_results, "Baseline")

        # Phase A.2: Baseline improve (generates experiments for Phase B)
        with improve_logging(baseline_dir):
            logging.info(f"Step {step} Phase A: Baseline improve logs")
            logging.info("=== Phase A.2: Baseline Improve (generates experiments) ===")
            logging.info(f"Current beliefs:\n{b if b else '(empty)'}")
            logging.info(f"Current perception:\n{p if p else '(empty)'}")

            new_b, new_p, new_experiments, baseline_improve_cost = improve_step(
                config=config,
                base_beliefs=b,
                perception=p,
                output_dir=str(baseline_rollout_dir),
                previous_experiments=[],  # No experiments were tested in baseline
                default_knowledge=default_knowledge,
                num_experiments=explore_config.num_experiments,
                rollout_results=baseline_rollout_results,
            )

        # Update state from baseline phase
        b = new_b
        p = new_p
        experiments = new_experiments

        # Save baseline phase outputs
        (baseline_dir / "beliefs.txt").write_text(b)
        (baseline_dir / "perception.py").write_text(p)
        with open(baseline_dir / "experiments.json", "w") as f:
            json.dump(experiments, f, indent=4)

        baseline_rollout_cost = _compute_rollout_cost(baseline_rollout_results)
        baseline_total_cost = baseline_rollout_cost + baseline_improve_cost
        evolve_logger.info(
            f"Phase A costs: rollout=${baseline_rollout_cost:.6f}, "
            f"improve=${baseline_improve_cost:.6f}, total=${baseline_total_cost:.6f}"
        )
        evolve_logger.info(f"Phase A updated beliefs:\n{b}")
        evolve_logger.info(f"Phase A generated {len(experiments)} experiments for Phase B:")
        for i, exp in enumerate(experiments):
            evolve_logger.info(f"  Experiment {i+1}: {exp}")

        # ================================================================
        # PHASE B: Explore (with experiments from Phase A)
        # ================================================================
        evolve_logger.info("--- Phase B: Explore (with experiments from Phase A) ---")
        evolve_logger.info(f"Experiments to test: {len(experiments)}")
        if experiments:
            for i, exp in enumerate(experiments):
                evolve_logger.info(f"  Experiment {i+1}: {exp}")
        else:
            evolve_logger.info("  (no experiments generated - running baseline rollouts for explore phase)")

        explore_dir = step_output_dir / "explore"
        explore_dir.mkdir(parents=True, exist_ok=True)
        explore_rollout_dir = explore_dir / "rollouts"
        explore_rollout_dir.mkdir(parents=True, exist_ok=True)

        # Phase B.1: Explore rollouts (with experiments) — use explore_temp
        with step_logging(explore_dir):
            logging.info(f"Step {step} Phase B: Explore rollout logs")
            logging.info(f"Current beliefs:\n{b if b else '(empty)'}")
            logging.info(f"Current perception:\n{p if p else '(empty)'}")
            logging.info("=== Phase B.1: Explore Rollouts (with experiments) ===")

            with override_temperature(config, explore_temp):
                explore_rollout_results = run_explore_rollouts(
                    base_beliefs=b,
                    perception=p,
                    experiments=experiments,
                    config=config,
                    original_cwd=original_cwd,
                    output_dir=str(explore_rollout_dir),
                    num_baseline_rollouts=explore_config.num_baseline_rollouts,
                )

        # Save explore rollout stats
        with open(explore_dir / "rollout_stats.json", "w") as f:
            json.dump(explore_rollout_results, f, indent=4, default=str)

        _log_rollout_stats(explore_rollout_results, "Explore")

        # Phase B.2: Explore improve (evaluates experiments, does NOT generate new ones)
        with improve_logging(explore_dir):
            logging.info(f"Step {step} Phase B: Explore improve logs")
            logging.info("=== Phase B.2: Explore Improve (evaluate experiments, no generation) ===")
            logging.info(f"Current beliefs:\n{b if b else '(empty)'}")
            logging.info(f"Current perception:\n{p if p else '(empty)'}")

            new_b, new_p, _, explore_improve_cost = improve_step(
                config=config,
                base_beliefs=b,
                perception=p,
                output_dir=str(explore_rollout_dir),
                previous_experiments=experiments,
                default_knowledge=default_knowledge,
                num_experiments=0,
                rollout_results=explore_rollout_results,
            )

        # Update state from explore phase (no new experiments - Phase A generates them)
        b = new_b
        p = new_p

        # Save explore phase outputs
        (explore_dir / "beliefs.txt").write_text(b)
        (explore_dir / "perception.py").write_text(p)

        explore_rollout_cost = _compute_rollout_cost(explore_rollout_results)
        explore_total_cost = explore_rollout_cost + explore_improve_cost
        evolve_logger.info(
            f"Phase B costs: rollout=${explore_rollout_cost:.6f}, "
            f"improve=${explore_improve_cost:.6f}, total=${explore_total_cost:.6f}"
        )

        total_step_cost = baseline_total_cost + explore_total_cost
        cumulative_cost += total_step_cost
        evolve_logger.info(f"Step {step} total cost: ${total_step_cost:.6f}")
        evolve_logger.info(f"Cumulative cost after step {step}: ${cumulative_cost:.6f}")

        _update_summary_json(
            output_dir=output_dir,
            step=step,
            step_cost=total_step_cost,
            cumulative_cost=cumulative_cost,
            rollout_stats=_get_rollout_stats(explore_rollout_results),
            phase_stats={
                "baseline": {
                    "cost": baseline_total_cost,
                    "rollout_stats": _get_rollout_stats(baseline_rollout_results),
                },
                "explore": {
                    "cost": explore_total_cost,
                    "rollout_stats": _get_rollout_stats(explore_rollout_results),
                },
            },
        )

        # Log summary
        evolve_logger.info(f"Step {step} completed.")
        evolve_logger.info(f"Updated beliefs:\n{b}")


def online_explore_experiment_guided(
    explore_config: ExploreConfig,
    config: DictConfig,
    original_cwd: str,
    output_dir: str,
):
    """Run experiment-guided exploration loop.

    Each step:
    1. Run baseline rollouts with current beliefs/perception
    2. Analyze baseline -> generate new experiments, select which to run
    3. Run experiment rollouts with selected experiments
    4. Analyze each experiment trajectory for conclusiveness -> update results
    5. Generate num_candidates candidate beliefs (perception from first candidate)
    6. Score candidates against experiment knowledge base, keep highest score

    Args:
        explore_config: Exploration configuration
        config: BALROG configuration
        original_cwd: Original working directory
        output_dir: Output directory for results
    """
    evolve_logger.info("Running experiment-guided exploration")

    # Check for existing progress and resume if available
    last_step, b, p, all_experiments, experiment_results = find_last_completed_step(
        output_dir, experiment_guided=True
    )
    start_step = last_step + 1

    if last_step > 0:
        evolve_logger.info(f"Resuming from step {start_step} (found {last_step} completed steps)")
        evolve_logger.info(f"Resuming with {len(all_experiments)} experiments in pool")
        num_answered = sum(1 for e in all_experiments if experiment_results.get(e))
        evolve_logger.info(f"  {num_answered} answered, {len(all_experiments) - num_answered} unanswered")
    else:
        evolve_logger.info("Starting fresh experiment-guided exploration")
        if (beliefs_path := config.eval.get("beliefs_path", None)) is not None:
            b = Path(beliefs_path).read_text()
            evolve_logger.info(f"Loaded initial beliefs from: {beliefs_path}")
        else:
            b = ""

        if (perception_path := config.eval.get("perception_path", None)) is not None:
            p = Path(perception_path).read_text()
            evolve_logger.info(f"Loaded initial perception from: {perception_path}")
        else:
            p = ""
        all_experiments = []
        experiment_results = {}

    config.eval.num_episodes = explore_config.rollouts_per_step
    explore_temp = config.eval.evolve.get("explore_temp", 1.0)

    # Fetch default knowledge once
    default_knowledge = get_default_knowledge(config)
    evolve_logger.info(f"Default knowledge extraction complete (length: {len(default_knowledge)})")

    evolve_logger.info(f"Starting experiment-guided exploration with {explore_config.num_steps} steps")
    evolve_logger.info(f"Rollouts per step: {explore_config.rollouts_per_step}")
    evolve_logger.info(f"Experiments per step: {explore_config.num_experiments}")
    evolve_logger.info(f"Baseline rollouts: {explore_config.num_baseline_rollouts}")
    evolve_logger.info(f"Candidates per step: {explore_config.num_candidates}")
    evolve_logger.info(f"Score experiments: {explore_config.num_score_experiments}")
    evolve_logger.info(f"Min experiments for scoring: {explore_config.min_experiments_for_scoring}")
    evolve_logger.info(f"Explore temperature: {explore_temp}")

    cumulative_cost = 0.0

    for step in range(start_step, explore_config.num_steps + 1):
        evolve_logger.info(f"\n{'='*80}")
        evolve_logger.info(f"EXPERIMENT-GUIDED STEP {step}/{explore_config.num_steps}")
        evolve_logger.info(f"Experiment pool size: {len(all_experiments)}")
        evolve_logger.info(f"{'='*80}")

        step_output_dir = Path(output_dir) / f"step_{step}"
        step_output_dir.mkdir(parents=True, exist_ok=True)

        # Save current state inputs
        (step_output_dir / "input_beliefs.txt").write_text(b)
        (step_output_dir / "input_perception.txt").write_text(p)
        with open(step_output_dir / "input_experiments.json", "w") as f:
            json.dump(all_experiments, f, indent=4)
        with open(step_output_dir / "input_experiment_results.json", "w") as f:
            json.dump(experiment_results, f, indent=4, default=str)

        step_cost = 0.0

        # ================================================================
        # STEP 1: Run baseline rollouts
        # ================================================================
        evolve_logger.info("--- Step 1: Baseline Rollouts ---")
        baseline_dir = step_output_dir / "baseline"
        baseline_rollout_dir = baseline_dir / "rollouts"
        baseline_rollout_dir.mkdir(parents=True, exist_ok=True)

        with step_logging(baseline_dir) as _:
            logging.info(f"Step {step}: Baseline rollout logs")
            logging.info(f"Current beliefs:\n{b if b else '(empty)'}")
            logging.info(f"Current perception:\n{p if p else '(empty)'}")

            baseline_rollout_results = run_explore_rollouts(
                base_beliefs=b,
                perception=p,
                experiments=[],
                config=config,
                original_cwd=original_cwd,
                output_dir=str(baseline_rollout_dir),
                num_baseline_rollouts=explore_config.num_baseline_rollouts,
            )

        with open(baseline_dir / "rollout_stats.json", "w") as f:
            json.dump(baseline_rollout_results, f, indent=4, default=str)
        _log_rollout_stats(baseline_rollout_results, "Baseline")
        baseline_rollout_cost = _compute_rollout_cost(baseline_rollout_results)
        step_cost += baseline_rollout_cost

        # ================================================================
        # STEP 2: Analyze baseline -> generate/select experiments
        # ================================================================
        evolve_logger.info("--- Step 2: Generate & Select Experiments ---")

        with improve_logging(step_output_dir) as _:
            logging.info(f"Step {step}: Experiment generation logs")

            # Compute baseline summaries
            baseline_summaries, _, baseline_summary_cost = prepare_improve_context(
                config=config,
                base_beliefs=b,
                perception=p,
                output_dir=str(baseline_rollout_dir),
                previous_experiments=[],
                default_knowledge=default_knowledge,
                rollout_results=baseline_rollout_results,
            )
            step_cost += baseline_summary_cost

            all_experiments, selected_experiments, refinements, gen_cost = generate_experiments_from_baseline(
                config=config,
                base_beliefs=b,
                episode_summaries=baseline_summaries,
                all_experiments=all_experiments,
                experiment_results=experiment_results,
                default_knowledge=default_knowledge,
                num_to_generate=explore_config.num_experiments,
                num_to_select=explore_config.num_experiments,
                refine_experiments=config.eval.evolve.get("refine_experiments", True),
            )
            step_cost += gen_cost

        with open(step_output_dir / "experiment_generation.json", "w") as f:
            json.dump({
                "all_experiments": all_experiments,
                "selected_experiments": selected_experiments,
                "refinements": refinements,
                "cost": gen_cost,
            }, f, indent=4)

        if refinements:
            evolve_logger.info(f"Refined {len(refinements)} experiments:")
            for r in refinements:
                evolve_logger.info(f"  '{r['old_wording']}' -> '{r['new_wording']}' (keep_results={r['keep_results']})")

        evolve_logger.info(f"Selected {len(selected_experiments)} experiments for testing")
        for i, exp in enumerate(selected_experiments):
            evolve_logger.info(f"  Experiment {i+1}: {exp}")

        # ================================================================
        # STEP 3: Run experiment rollouts
        # ================================================================
        evolve_logger.info("--- Step 3: Experiment Rollouts ---")
        explore_dir = step_output_dir / "explore"
        explore_rollout_dir = explore_dir / "rollouts"
        explore_rollout_dir.mkdir(parents=True, exist_ok=True)

        with step_logging(explore_dir) as _:
            logging.info(f"Step {step}: Experiment rollout logs")
            logging.info(f"Current beliefs:\n{b if b else '(empty)'}")
            logging.info(f"Current perception:\n{p if p else '(empty)'}")

            with override_temperature(config, explore_temp):
                experiment_rollout_results = run_explore_rollouts(
                    base_beliefs=b,
                    perception=p,
                    experiments=selected_experiments,
                    config=config,
                    original_cwd=original_cwd,
                    output_dir=str(explore_rollout_dir),
                    num_baseline_rollouts=explore_config.num_baseline_rollouts,
                )

        with open(explore_dir / "rollout_stats.json", "w") as f:
            json.dump(experiment_rollout_results, f, indent=4, default=str)
        _log_rollout_stats(experiment_rollout_results, "Experiment")
        experiment_rollout_cost = _compute_rollout_cost(experiment_rollout_results)
        step_cost += experiment_rollout_cost

        # ================================================================
        # STEP 4: Analyze experiment trajectories for conclusiveness
        # ================================================================
        evolve_logger.info("--- Step 4: Analyze Experiment Conclusiveness ---")

        with improve_logging(step_output_dir) as _:
            logging.info(f"Step {step}: Experiment analysis logs")

            new_conclusive_results, analysis_cost = analyze_all_experiment_trajectories(
                config=config,
                experiment_rollout_results=experiment_rollout_results,
                selected_experiments=selected_experiments,
                output_dir=str(explore_rollout_dir),
            )
            step_cost += analysis_cost

        # Merge new conclusive results into experiment_results
        for exp, results in new_conclusive_results.items():
            if exp not in experiment_results:
                experiment_results[exp] = []
            experiment_results[exp].extend(results)

        with open(step_output_dir / "experiment_analysis.json", "w") as f:
            json.dump({
                "new_conclusive_results": new_conclusive_results,
                "cost": analysis_cost,
            }, f, indent=4, default=str)

        num_answered = sum(1 for e in all_experiments if experiment_results.get(e))
        evolve_logger.info(f"Experiment pool: {len(all_experiments)} total, {num_answered} answered")

        # ================================================================
        # STEP 5: Generate candidate beliefs using ALL trajectories
        # ================================================================
        evolve_logger.info(f"--- Step 5: Generate {explore_config.num_candidates} Candidate Beliefs ---")

        # Combine all rollout dirs for summaries
        all_rollout_results = {}
        all_rollout_results.update(baseline_rollout_results)
        all_rollout_results.update(experiment_rollout_results)

        candidates_dir = step_output_dir / "candidates"
        candidates_dir.mkdir(parents=True, exist_ok=True)

        with improve_logging(step_output_dir) as _:
            logging.info(f"Step {step}: Candidate generation logs")

            # Compute summaries from all trajectories (baseline + experiment)
            # We need to combine CSVs from both dirs. Create a temp parent referencing both.
            # Actually, prepare_improve_context rglobs for *.csv in output_dir.
            # We'll call it twice and concatenate, or we can just point to the step dir.
            # The rollout dirs are under baseline/rollouts and explore/rollouts.
            # Let's compute summaries from each dir and concatenate.

            baseline_ep_summaries, baseline_perc_examples, bs_cost = prepare_improve_context(
                config=config,
                base_beliefs=b,
                perception=p,
                output_dir=str(baseline_rollout_dir),
                previous_experiments=[],
                default_knowledge=default_knowledge,
                rollout_results=baseline_rollout_results,
            )
            step_cost += bs_cost

            experiment_ep_summaries, experiment_perc_examples, es_cost = prepare_improve_context(
                config=config,
                base_beliefs=b,
                perception=p,
                output_dir=str(explore_rollout_dir),
                previous_experiments=selected_experiments,
                default_knowledge=default_knowledge,
                rollout_results=experiment_rollout_results,
            )
            step_cost += es_cost

            # Combine summaries
            combined_summaries = (
                baseline_ep_summaries + "\n" + experiment_ep_summaries
            )
            combined_perc_examples = baseline_perc_examples
            if experiment_perc_examples:
                combined_perc_examples += "\n" + experiment_perc_examples

            # Generate candidates
            candidate_beliefs_list = []
            candidate_perception = p  # Will use perception from first candidate

            for cand_idx in range(explore_config.num_candidates):
                cand_beliefs, cand_perception, _, cand_cost = generate_candidate_beliefs(
                    config=config,
                    base_beliefs=b,
                    perception=p,
                    episode_summaries=combined_summaries,
                    perception_examples=combined_perc_examples,
                    previous_experiments=selected_experiments,
                    default_knowledge=default_knowledge,
                    num_experiments=0,
                    candidate_index=cand_idx,
                )
                step_cost += cand_cost
                candidate_beliefs_list.append(cand_beliefs)

                # Use perception from first candidate only
                if cand_idx == 0:
                    candidate_perception = cand_perception

                # Save candidate
                (candidates_dir / f"candidate_{cand_idx}.txt").write_text(cand_beliefs)
                evolve_logger.info(f"Candidate {cand_idx} beliefs:\n{cand_beliefs}")

        # ================================================================
        # STEP 6: Score candidates and select best
        # ================================================================
        evolve_logger.info("--- Step 6: Score Candidates ---")

        # Count answered experiments with clear majority vote
        num_answered_clear = 0
        for exp in all_experiments:
            results = experiment_results.get(exp, [])
            if results:
                yes_count = sum(1 for r in results if r["answer"])
                no_count = sum(1 for r in results if not r["answer"])
                if yes_count != no_count:
                    num_answered_clear += 1

        scoring_data = {}
        if num_answered_clear < explore_config.min_experiments_for_scoring:
            evolve_logger.info(
                f"Too few answered experiments for scoring ({num_answered_clear} < "
                f"{explore_config.min_experiments_for_scoring}). Using first candidate."
            )
            best_idx = 0
            scoring_data = {
                "skipped": True,
                "reason": f"Too few answered experiments ({num_answered_clear} < {explore_config.min_experiments_for_scoring})",
                "best_candidate": best_idx,
            }
        elif len(candidate_beliefs_list) == 1:
            evolve_logger.info("Only one candidate. Skipping scoring.")
            best_idx = 0
            scoring_data = {
                "skipped": True,
                "reason": "Only one candidate",
                "best_candidate": best_idx,
            }
        else:
            with improve_logging(step_output_dir) as _:
                logging.info(f"Step {step}: Scoring logs")

                scores, details, scoring_cost = score_candidate_beliefs(
                    config=config,
                    candidates=candidate_beliefs_list,
                    experiment_results=experiment_results,
                    default_knowledge=default_knowledge,
                    num_score_experiments=explore_config.num_score_experiments,
                )
                step_cost += scoring_cost

            best_idx = scores.index(max(scores))
            evolve_logger.info(f"Candidate scores: {scores}")
            evolve_logger.info(f"Best candidate: {best_idx} (score={scores[best_idx]})")

            scoring_data = {
                "skipped": False,
                "scores": scores,
                "best_candidate": best_idx,
                "details": details,
                "cost": scoring_cost,
            }

        with open(step_output_dir / "scoring.json", "w") as f:
            json.dump(scoring_data, f, indent=4, default=str)

        # Update state with best candidate
        b = candidate_beliefs_list[best_idx]
        p = candidate_perception

        # Save final outputs
        (step_output_dir / "beliefs.txt").write_text(b)
        (step_output_dir / "perception.py").write_text(p)
        with open(step_output_dir / "experiments.json", "w") as f:
            json.dump(all_experiments, f, indent=4)
        with open(step_output_dir / "experiment_results.json", "w") as f:
            json.dump(experiment_results, f, indent=4, default=str)

        cumulative_cost += step_cost
        evolve_logger.info(f"Step {step} cost: ${step_cost:.6f}")
        evolve_logger.info(f"Cumulative cost after step {step}: ${cumulative_cost:.6f}")

        _update_summary_json(
            output_dir=output_dir,
            step=step,
            step_cost=step_cost,
            cumulative_cost=cumulative_cost,
            rollout_stats=_get_rollout_stats(experiment_rollout_results),
            phase_stats={
                "baseline": {
                    "cost": baseline_rollout_cost,
                    "rollout_stats": _get_rollout_stats(baseline_rollout_results),
                },
                "experiment": {
                    "cost": experiment_rollout_cost,
                    "rollout_stats": _get_rollout_stats(experiment_rollout_results),
                },
            },
        )

        evolve_logger.info(f"Step {step} completed.")
        evolve_logger.info(f"Updated beliefs:\n{b}")
        evolve_logger.info(f"Experiment pool: {len(all_experiments)} experiments")


@contextmanager
def redirect_to_file(filepath):
    original = sys.stdout
    with open(filepath, "w") as file:
        sys.stdout = file
        try:
            yield
        finally:
            sys.stdout = original


@hydra.main(config_path="BALROG/balrog/config", config_name="config", version_base="1.1")
def main(config: DictConfig):
    original_cwd = get_original_cwd()
    setup_environment(original_cwd=original_cwd)

    two_step = config.eval.evolve.get("two_step", False)
    experiment_guided = config.eval.evolve.get("experiment_guided", False)

    # Determine output directory
    if config.eval.resume_from is not None:
        output_dir: str = config.eval.resume_from
    else:
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        if experiment_guided:
            suffix = "_explore_expguided"
        elif two_step:
            suffix = "_explore2step"
        else:
            suffix = "_explore"
        run_name = f"{timestamp}_{config.agent.type}_{config.client.model_id.replace('/', '_')}{suffix}"
        output_dir = os.path.join(config.eval.output_dir, run_name)

        # Create the directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Setup loggers
    log_filename = os.path.join(output_dir, "eval.log")
    log_format = "%(asctime)s - %(levelname)s - %(message)s"

    # Configure root logger (used for detailed step logs)
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[logging.FileHandler(log_filename)],
        force=True,
    )

    # Configure evolve_logger for high-level logs (always writes to eval.log)
    evolve_handler = logging.FileHandler(log_filename)
    evolve_handler.setLevel(logging.INFO)
    evolve_handler.setFormatter(logging.Formatter(log_format))
    evolve_logger.addHandler(evolve_handler)
    evolve_logger.setLevel(logging.INFO)
    evolve_logger.propagate = False  # Don't propagate to root logger

    # Print output location to terminal
    print(f"Output directory: {output_dir}")
    print(f"Log file: {log_filename}")

    # Save config to output directory
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w") as f:
        OmegaConf.save(config=config, f=f)
    evolve_logger.info(f"Saved config to {config_path}")

    # Get exploration config with defaults
    num_experiments = config.eval.evolve.get("num_experiments", 1)
    num_baseline_rollouts = config.eval.evolve.get("num_baseline_rollouts", 1)

    match config.eval.mode:
        case "eval":
            # Simple eval mode - just run one step
            from rollout import one_step_wrap
            one_step_wrap(config=config, original_cwd=original_cwd, output_dir=output_dir)

        case "explore":
            ec = ExploreConfig(
                num_steps=config.eval.evolve.num_steps,
                rollouts_per_step=config.eval.evolve.rollouts_per_step,
                num_experiments=num_experiments,
                num_baseline_rollouts=num_baseline_rollouts,
                num_candidates=config.eval.evolve.get("num_candidates", 3),
                num_score_experiments=config.eval.evolve.get("num_score_experiments", 5),
                min_experiments_for_scoring=config.eval.evolve.get("min_experiments_for_scoring", 3),
            )
            if experiment_guided:
                online_explore_experiment_guided(
                    explore_config=ec,
                    config=config,
                    original_cwd=original_cwd,
                    output_dir=output_dir,
                )
            elif two_step:
                online_explore_2step(
                    explore_config=ec,
                    config=config,
                    original_cwd=original_cwd,
                    output_dir=output_dir,
                )
            else:
                online_explore(
                    explore_config=ec,
                    config=config,
                    original_cwd=original_cwd,
                    output_dir=output_dir,
                )

        case _:
            evolve_logger.error(f"Unsupported mode: {config.eval.mode}. explore_eval.py supports 'eval' and 'explore' modes.")
            raise ValueError(f"Unsupported mode: {config.eval.mode}")


if __name__ == "__main__":
    main()
