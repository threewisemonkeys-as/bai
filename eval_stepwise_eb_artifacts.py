"""Evaluate learned stepwise EB artifacts from an existing run.

This script intentionally reuses the frozen post-learning evaluation functions
from stepwise_eb_learn.py so artifact sweeps use the same rollout setup as the
in-training post-learning eval path.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, ListConfig, OmegaConf

from balrog.client import set_mock_mode as set_client_mock_mode
from mixed_improve import set_mock_mode
from run_utils import setup_run
from stepwise_eb_learn import (
    build_stepwise_eb_config,
    run_frozen_autumn_evaluation,
    run_frozen_environment_evaluation,
)


logger = logging.getLogger("evolve")


def _select(config: DictConfig, key: str, default=None):
    return OmegaConf.select(config, key, default=default)


def _resolve_path(path: str | Path, original_cwd: str) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return Path(original_cwd) / value


def _safe_label(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def _episode_index(path: Path) -> int:
    return int(path.name.split("_", 1)[1])


def _step_index(path: Path) -> int:
    return int(path.name.split("_", 1)[1])


def _artifact_from_dir(
    *,
    source_run: Path,
    artifact_dir: Path,
    kind: str,
    label: str,
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    beliefs_path = artifact_dir / "beliefs.txt"
    perception_path = artifact_dir / "perception.py"
    if not beliefs_path.exists() or not perception_path.exists():
        return None

    return {
        "kind": kind,
        "label": label,
        "artifact_dir": str(artifact_dir),
        "beliefs_path": str(beliefs_path),
        "perception_path": str(perception_path),
        "source_run": str(source_run),
        "metadata": metadata,
    }


def _empty_baseline_artifact() -> dict[str, Any]:
    return {
        "kind": "empty_baseline",
        "label": "g0000_empty_baseline",
        "artifact_dir": "",
        "beliefs_path": "",
        "perception_path": "",
        "source_run": "",
        "empty": True,
        "metadata": {"global_step": 0},
    }


def discover_artifacts(source_run: Path) -> list[dict[str, Any]]:
    """Return step and episode-final artifacts from a stepwise EB run."""
    artifacts: list[dict[str, Any]] = []
    cumulative_steps = 0

    episode_dirs = sorted(
        [p for p in source_run.glob("episode_*") if p.is_dir()],
        key=_episode_index,
    )
    for episode_dir in episode_dirs:
        episode_idx = _episode_index(episode_dir)

        step_dirs = sorted(
            [p for p in episode_dir.glob("step_*") if p.is_dir()],
            key=_step_index,
        )
        for step_dir in step_dirs:
            step_idx = _step_index(step_dir)
            step_log = _read_json(step_dir / "step_log.json")
            global_step = step_log.get("global_step")
            if global_step is None:
                global_step = cumulative_steps + step_idx
            label = f"g{int(global_step):04d}_episode_{episode_idx}_step_{step_idx:03d}"
            artifact = _artifact_from_dir(
                source_run=source_run,
                artifact_dir=step_dir,
                kind="step",
                label=label,
                metadata={
                    "episode_idx": episode_idx,
                    "episode_step": step_idx,
                    "global_step": int(global_step),
                    "step_log": step_log,
                },
            )
            if artifact is not None:
                artifacts.append(artifact)

        episode_log = _read_json(episode_dir / "episode_log.json")
        num_steps = int(episode_log.get("num_steps", len(step_dirs)))
        cumulative_steps += num_steps
        label = f"g{cumulative_steps:04d}_episode_{episode_idx}_final"
        artifact = _artifact_from_dir(
            source_run=source_run,
            artifact_dir=episode_dir,
            kind="episode_final",
            label=label,
            metadata={
                "episode_idx": episode_idx,
                "global_step_after_episode": cumulative_steps,
                "episode_log": episode_log,
            },
        )
        if artifact is not None:
            artifacts.append(artifact)

    return artifacts


def _as_requested_list(raw_steps) -> list[Any]:
    if raw_steps is None:
        return ["final"]
    if isinstance(raw_steps, ListConfig):
        return list(raw_steps)
    if isinstance(raw_steps, (list, tuple)):
        return list(raw_steps)
    return [raw_steps]


def select_artifacts(
    artifacts: list[dict[str, Any]],
    raw_steps,
    every_n: int | None = None,
) -> list[dict[str, Any]]:
    if not artifacts:
        raise ValueError("No artifacts with beliefs.txt and perception.py were found.")

    by_label = {artifact["label"]: artifact for artifact in artifacts}
    step_artifacts = [a for a in artifacts if a["kind"] == "step"]
    episode_artifacts = [a for a in artifacts if a["kind"] == "episode_final"]

    selected: list[dict[str, Any]] = []

    def add(artifact: dict[str, Any]):
        if artifact["label"] not in {a["label"] for a in selected}:
            selected.append(artifact)

    requested = _as_requested_list(raw_steps)
    if len(requested) == 1 and str(requested[0]).lower() == "all":
        for artifact in step_artifacts:
            add(artifact)
    else:
        for item in requested:
            text = str(item)
            lower = text.lower()

            if lower in {"final", "latest", "last"}:
                add(episode_artifacts[-1] if episode_artifacts else artifacts[-1])
                continue

            if lower.startswith("episode:"):
                episode_idx = int(text.split(":", 1)[1])
                matches = [
                    a for a in episode_artifacts
                    if a["metadata"].get("episode_idx") == episode_idx
                ]
                if not matches:
                    raise ValueError(f"No completed episode artifact found for {text!r}.")
                add(matches[-1])
                continue

            if lower.startswith("label:"):
                label = text.split(":", 1)[1]
                if label not in by_label:
                    raise ValueError(f"No artifact label found for {label!r}.")
                add(by_label[label])
                continue

            global_step = int(item)
            matches = [
                a for a in step_artifacts
                if a["metadata"].get("global_step") == global_step
            ]
            if not matches:
                matches = [
                    a for a in episode_artifacts
                    if a["metadata"].get("global_step_after_episode") == global_step
                ]
            if not matches:
                raise ValueError(f"No artifact found for global step {global_step}.")
            add(matches[-1])

    if every_n is not None and every_n > 0:
        for artifact in step_artifacts:
            global_step = artifact["metadata"].get("global_step")
            if global_step is not None and int(global_step) % every_n == 0:
                add(artifact)

    return selected


def _load_run_config(config: DictConfig, source_run: Path) -> DictConfig:
    use_source_config = bool(_select(config, "artifact_eval.use_source_config", True))
    if not use_source_config:
        OmegaConf.set_struct(config, False)
        config.eval.resume_from = None
        _apply_artifact_eval_overrides(config)
        return config

    source_config_path = source_run / "config.yaml"
    if not source_config_path.exists():
        raise FileNotFoundError(f"Source run config not found: {source_config_path}")

    run_config = OmegaConf.load(source_config_path)
    OmegaConf.set_struct(run_config, False)
    run_config.eval.output_dir = config.eval.output_dir
    run_config.eval.resume_from = None
    run_config.artifact_eval = _select(config, "artifact_eval", {})
    _apply_artifact_eval_overrides(run_config)
    return run_config


def _apply_artifact_eval_overrides(config: DictConfig) -> None:
    artifact_eval = _select(config, "artifact_eval", {})
    is_autumn = str(_select(config, "envs.names", "")).split("-")[0] == "autumn"

    if is_autumn:
        if _select(config, "artifact_eval.autumn_eval_task_types", None) is None:
            OmegaConf.update(
                config,
                "eval.evolve.autumn_eval_task_types",
                ["planning"],
                merge=False,
            )
        if _select(config, "artifact_eval.autumn_eval_render_mode", None) is None:
            OmegaConf.update(
                config,
                "eval.evolve.autumn_eval_render_mode",
                "image",
                merge=False,
            )
        if _select(config, "artifact_eval.autumn_eval_start_in_test_phase", None) is None:
            OmegaConf.update(
                config,
                "artifact_eval.autumn_eval_start_in_test_phase",
                True,
                merge=False,
            )

    override_map = {
        "agent_goal_mode": "eval.agent_goal_mode",
        "agent_goal_text": "eval.agent_goal_text",
        "mock_mode": "eval.evolve.mock_mode",
        "frozen_eval_max_steps": "eval.evolve.frozen_eval_max_steps",
        "frozen_eval_envs": "eval.evolve.frozen_eval_envs",
        "autumn_eval_max_steps": "eval.evolve.autumn_eval_max_steps",
        "autumn_eval_task_types": "eval.evolve.autumn_eval_task_types",
        "autumn_eval_render_mode": "eval.evolve.autumn_eval_render_mode",
    }
    for source_key, target_key in override_map.items():
        value = _select(config, f"artifact_eval.{source_key}", None)
        if value is not None:
            OmegaConf.update(config, target_key, value, merge=False)

    if (
        _select(config, "artifact_eval.agent_goal_mode", None) is not None
        or _select(config, "artifact_eval.agent_goal_text", None) is not None
    ):
        for legacy_key in (
            "eval.evolve.frozen_eval_minihack_goal",
            "eval.evolve.frozen_eval_arc_agi_goal",
            "eval.evolve.frozen_eval_autumn_planning_goal",
        ):
            OmegaConf.update(config, legacy_key, None, merge=False)


def _write_artifact_inputs(output_dir: Path, artifact: dict[str, Any]) -> tuple[str, str]:
    if artifact.get("empty"):
        beliefs, perception = "", ""
    else:
        beliefs = Path(artifact["beliefs_path"]).read_text()
        perception = Path(artifact["perception_path"]).read_text()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "beliefs.txt").write_text(beliefs)
    (output_dir / "perception.py").write_text(perception)
    with open(output_dir / "source_artifact.json", "w") as f:
        json.dump(artifact, f, indent=4, default=str)
    return beliefs, perception


def _configure_worker_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_filename = output_dir / "eval.log"
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[logging.FileHandler(log_filename)],
        force=True,
    )
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)


def _seed_for_repeat(config: DictConfig, repeat_idx: int) -> int | None:
    configured_seed = _select(config, "artifact_eval.seed_base", None)
    if configured_seed is None:
        configured_seed = _select(config, "envs.env_kwargs.seed", None)
    if configured_seed is None:
        return None
    return int(configured_seed) + int(repeat_idx)


def _aggregate_repeat_records(repeat_records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [record for record in repeat_records if record.get("error") is None]
    aggregate_reward = 0.0
    aggregate_num_tasks = 0
    total_cost = 0.0
    for record in successful:
        summary = record.get("summary")
        if isinstance(summary, dict):
            aggregate_reward += float(summary.get("aggregate_reward", 0.0))
            aggregate_num_tasks += int(summary.get("num_tasks", 0))
            total_cost += float(summary.get("total_cost", 0.0))
    return {
        "num_repeats": len(repeat_records),
        "num_successful_repeats": len(successful),
        "num_failed_repeats": len(repeat_records) - len(successful),
        "aggregate_reward_sum": aggregate_reward,
        "aggregate_reward_mean": (
            aggregate_reward / len(successful) if successful else None
        ),
        "aggregate_num_tasks_sum": aggregate_num_tasks,
        "total_cost": total_cost,
        "cost_per_repeat_mean": total_cost / len(successful) if successful else None,
    }


def _make_repeat_record(
    artifact: dict[str, Any],
    repeat_idx: int,
    repeat_output_dir: Path,
    seed: int | None,
    summary: dict[str, Any],
    error: str | None,
) -> dict[str, Any]:
    return {
        "artifact_label": artifact["label"],
        "repeat_idx": repeat_idx,
        "seed": seed,
        "source": artifact,
        "output_dir": str(repeat_output_dir),
        "summary": summary,
        "error": error,
    }


def _run_frozen_eval_for_artifact(
    *,
    config: DictConfig,
    eb_config,
    artifact: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    beliefs, perception = _write_artifact_inputs(output_dir, artifact)
    env_name = config.envs.names.split("-")[0]

    if env_name == "autumn":
        return run_frozen_autumn_evaluation(
            config=config,
            eb_config=eb_config,
            beliefs=beliefs,
            perception=perception,
            output_dir=str(output_dir),
        )

    return run_frozen_environment_evaluation(
        config=config,
        eb_config=eb_config,
        beliefs=beliefs,
        perception=perception,
        output_dir=str(output_dir),
    )


def _evaluate_artifact_repeat(
    *,
    run_config_container: dict[str, Any],
    artifact: dict[str, Any],
    repeat_idx: int,
    repeat_output_dir: str,
    seed: int | None,
) -> dict[str, Any]:
    output_dir = Path(repeat_output_dir)
    _configure_worker_logging(output_dir)

    worker_config = OmegaConf.create(run_config_container)
    OmegaConf.set_struct(worker_config, False)
    worker_config.eval.output_dir = str(output_dir.parent)
    worker_config.eval.resume_from = None
    if seed is not None:
        OmegaConf.update(worker_config, "envs.env_kwargs.seed", int(seed), merge=False)

    with open(output_dir / "config.yaml", "w") as f:
        OmegaConf.save(config=worker_config, f=f)

    eb_config = build_stepwise_eb_config(worker_config)
    set_mock_mode(bool(eb_config.mock_mode))
    set_client_mock_mode(bool(eb_config.mock_mode))

    try:
        eval_summary = _run_frozen_eval_for_artifact(
            config=worker_config,
            eb_config=eb_config,
            artifact=artifact,
            output_dir=output_dir,
        )
        error = None
    except Exception as exc:
        logging.exception("Artifact frozen evaluation failed")
        eval_summary = {}
        error = str(exc)

    return _make_repeat_record(
        artifact=artifact,
        repeat_idx=repeat_idx,
        repeat_output_dir=output_dir,
        seed=seed,
        summary=eval_summary,
        error=error,
    )


@hydra.main(config_path="BALROG/balrog/config", config_name="config", version_base="1.1")
def main(config: DictConfig):
    original_cwd = get_original_cwd()
    source_run_raw = _select(config, "artifact_eval.source_run")

    if source_run_raw is None:
        run_config = config
        OmegaConf.set_struct(run_config, False)
        run_config.eval.resume_from = None
        _apply_artifact_eval_overrides(run_config)
        source_run = None
        selected = [_empty_baseline_artifact()]
        num_discovered = 0
    else:
        source_run = _resolve_path(source_run_raw, original_cwd)
        run_config = _load_run_config(config, source_run)
        artifacts = discover_artifacts(source_run)
        selected = select_artifacts(
            artifacts,
            _select(run_config, "artifact_eval.steps", ["final"]),
            every_n=_select(run_config, "artifact_eval.every_n", None),
        )
        num_discovered = len(artifacts)
        logger.info(f"Source run: {source_run}")

    run_name_suffix = (
        f"{run_config.agent.type}_"
        f"{run_config.client.model_id.replace('/', '_')}_"
        "stepwise_eb_artifact_eval"
    )
    _, output_dir = setup_run(
        run_config,
        run_name_suffix=run_name_suffix,
        resume_from=None,
        output_dir_base=run_config.eval.output_dir,
        logger_name="evolve",
    )
    output_path = Path(output_dir)

    logger.info(f"Discovered {num_discovered} artifacts; evaluating {len(selected)}")

    repeats = int(_select(run_config, "artifact_eval.repeats", 1) or 1)
    parallel_workers = int(_select(run_config, "artifact_eval.parallel_workers", 1) or 1)
    run_config_container = OmegaConf.to_container(run_config, resolve=True)

    summary: dict[str, Any] = {
        "source_run": str(source_run) if source_run else None,
        "num_discovered_artifacts": num_discovered,
        "num_evaluated_artifacts": len(selected),
        "repeats": repeats,
        "parallel_workers": parallel_workers,
        "artifacts": {},
    }

    jobs: list[dict[str, Any]] = []
    for artifact in selected:
        label = _safe_label(artifact["label"])
        artifact_output_dir = output_path / label
        summary["artifacts"][artifact["label"]] = {
            "source": artifact,
            "output_dir": str(artifact_output_dir),
            "repeat_records": [],
        }
        for repeat_idx in range(repeats):
            repeat_output_dir = (
                artifact_output_dir
                if repeats == 1
                else artifact_output_dir / f"repeat_{repeat_idx:03d}"
            )
            jobs.append(
                {
                    "artifact": artifact,
                    "repeat_idx": repeat_idx,
                    "repeat_output_dir": repeat_output_dir,
                    "seed": _seed_for_repeat(run_config, repeat_idx),
                }
            )

    def record_result(result: dict[str, Any]) -> None:
        artifact_label = result["artifact_label"]
        artifact_summary = summary["artifacts"][artifact_label]
        artifact_summary["repeat_records"].append(result)
        artifact_summary["repeat_records"].sort(key=lambda rec: rec["repeat_idx"])
        artifact_summary["repeat_aggregate"] = _aggregate_repeat_records(
            artifact_summary["repeat_records"]
        )
        if repeats == 1:
            artifact_summary["summary"] = result["summary"]
            artifact_summary["error"] = result["error"]
        with open(output_path / "artifact_eval_summary.json", "w") as f:
            json.dump(summary, f, indent=4, default=str)

    if parallel_workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            logger.info(
                "Evaluating artifact %s repeat %03d from %s",
                job["artifact"]["label"],
                job["repeat_idx"],
                job["artifact"]["artifact_dir"],
            )
            result = _evaluate_artifact_repeat(
                run_config_container=run_config_container,
                artifact=job["artifact"],
                repeat_idx=job["repeat_idx"],
                repeat_output_dir=str(job["repeat_output_dir"]),
                seed=job["seed"],
            )
            record_result(result)
    else:
        with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
            futures = []
            for job in jobs:
                logger.info(
                    "Queueing artifact %s repeat %03d from %s",
                    job["artifact"]["label"],
                    job["repeat_idx"],
                    job["artifact"]["artifact_dir"],
                )
                futures.append(
                    executor.submit(
                        _evaluate_artifact_repeat,
                        run_config_container=run_config_container,
                        artifact=job["artifact"],
                        repeat_idx=job["repeat_idx"],
                        repeat_output_dir=str(job["repeat_output_dir"]),
                        seed=job["seed"],
                    )
                )
            for future in as_completed(futures):
                record_result(future.result())

    logger.info(f"Artifact evaluation complete: {output_path / 'artifact_eval_summary.json'}")


if __name__ == "__main__":
    main()
