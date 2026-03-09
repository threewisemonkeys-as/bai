import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from balrog.utils import setup_environment


def setup_run(
    config: DictConfig,
    run_name_suffix: str,
    resume_from: str | None = None,
    output_dir_base: str | None = None,
    logger_name: str | None = None,
) -> tuple[str, str]:
    """Generic setup for a Hydra-based run.

    Handles environment setup, output directory resolution (resume or
    timestamp-based), logging configuration, and config saving.

    Args:
        config: Hydra DictConfig.
        run_name_suffix: Suffix appended to the timestamped run directory name.
        resume_from: If set, reuse this directory instead of creating a new one.
        output_dir_base: Base directory under which new run dirs are created.
        logger_name: If provided, configure a named logger (in addition to root)
                     that writes to the same log file without propagating.

    Returns:
        (original_cwd, output_dir)
    """
    original_cwd = get_original_cwd()
    setup_environment(original_cwd=original_cwd)

    # Resolve output directory
    if resume_from is not None:
        output_dir: str = resume_from
    else:
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{timestamp}_{run_name_suffix}"
        if output_dir_base is None:
            output_dir_base = config.eval.output_dir
        output_dir = os.path.join(output_dir_base, run_name)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Setup loggers
    log_filename = os.path.join(output_dir, "eval.log")
    log_format = "%(asctime)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[logging.FileHandler(log_filename)],
        force=True,
    )

    if logger_name is not None:
        named_logger = logging.getLogger(logger_name)
        handler = logging.FileHandler(log_filename)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(log_format))
        named_logger.addHandler(handler)
        named_logger.setLevel(logging.INFO)
        named_logger.propagate = False

    # Print output location to terminal
    print(f"Output directory: {output_dir}")
    print(f"Log file: {log_filename}")

    # Save config
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w") as f:
        OmegaConf.save(config=config, f=f)
    logging.info(f"Saved config to {config_path}")

    return original_cwd, output_dir


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
