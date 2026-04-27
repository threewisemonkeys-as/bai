"""Evaluate simple-stepwise artifacts (saved message histories) from a phase-1 run.

Phase-2 entry for the simple baseline in the ExploreEval (EE) scheme. Discovers
``dynamics_history.json`` checkpoints under a phase-1 source run, then re-runs
``simple_stepwise.stepwise_simple`` with each selected history seeded into the
agent and an env-specific eval goal forced via Hydra overrides.

Mirrors ``eval_stepwise_eb_artifacts.py`` in shape so the per-method eval scripts
behave the same way under launch_ee.py.
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

from run_utils import setup_run
from simple_stepwise import SimpleStepwiseConfig, stepwise_simple


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


def discover_artifacts(source_run: Path) -> list[dict[str, Any]]:
    """Return step / episode-final / final artifacts produced by simple_stepwise."""
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
            history_path = step_dir / "dynamics_history.json"
            if not history_path.exists():
                continue
            step_idx = _step_index(step_dir)
            step_log = _read_json(step_dir / "step_log.json")
            global_step = step_log.get("global_step")
            if global_step is None:
                global_step = cumulative_steps + step_idx
            label = f"g{int(global_step):04d}_episode_{episode_idx}_step_{step_idx:03d}"
            artifacts.append({
                "kind": "step",
                "label": label,
                "history_path": str(history_path),
                "source_run": str(source_run),
                "metadata": {
                    "episode_idx": episode_idx,
                    "episode_step": step_idx,
                    "global_step": int(global_step),
                },
            })

        episode_log = _read_json(episode_dir / "episode_log.json")
        num_steps = int(episode_log.get("num_steps", len(step_dirs)))
        cumulative_steps += num_steps
        episode_history = episode_dir / "dynamics_history.json"
        if episode_history.exists():
            label = f"g{cumulative_steps:04d}_episode_{episode_idx}_final"
            artifacts.append({
                "kind": "episode_final",
                "label": label,
                "history_path": str(episode_history),
                "source_run": str(source_run),
                "metadata": {
                    "episode_idx": episode_idx,
                    "global_step_after_episode": cumulative_steps,
                },
            })

    final_history = source_run / "dynamics_history.json"
    if final_history.exists():
        artifacts.append({
            "kind": "final",
            "label": f"g{cumulative_steps:04d}_final",
            "history_path": str(final_history),
            "source_run": str(source_run),
            "metadata": {"global_step_after_run": cumulative_steps},
        })

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
        raise ValueError("No simple_stepwise artifacts (dynamics_history.json) found.")

    by_label = {a["label"]: a for a in artifacts}
    step_artifacts = [a for a in artifacts if a["kind"] == "step"]
    episode_artifacts = [a for a in artifacts if a["kind"] == "episode_final"]
    final_artifacts = [a for a in artifacts if a["kind"] == "final"]

    selected: list[dict[str, Any]] = []

    def add(artifact: dict[str, Any]):
        if artifact["label"] not in {a["label"] for a in selected}:
            selected.append(artifact)

    requested = _as_requested_list(raw_steps)
    if len(requested) == 1 and str(requested[0]).lower() == "all":
        for a in step_artifacts + episode_artifacts + final_artifacts:
            add(a)
    else:
        for item in requested:
            text = str(item)
            lower = text.lower()

            if lower in {"final", "latest", "last"}:
                if final_artifacts:
                    add(final_artifacts[-1])
                elif episode_artifacts:
                    add(episode_artifacts[-1])
                else:
                    add(artifacts[-1])
                continue

            if lower.startswith("episode:"):
                episode_idx = int(text.split(":", 1)[1])
                matches = [
                    a for a in episode_artifacts
                    if a["metadata"].get("episode_idx") == episode_idx
                ]
                if not matches:
                    raise ValueError(f"No completed episode artifact for {text!r}.")
                add(matches[-1])
                continue

            if lower.startswith("label:"):
                label = text.split(":", 1)[1]
                if label not in by_label:
                    raise ValueError(f"No artifact label {label!r}.")
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
                raise ValueError(f"No artifact for global step {global_step}.")
            add(matches[-1])

    if every_n is not None and every_n > 0:
        for a in step_artifacts:
            global_step = a["metadata"].get("global_step")
            if global_step is not None and int(global_step) % every_n == 0:
                add(a)

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
    override_map = {
        "agent_goal_mode": "eval.agent_goal_mode",
        "agent_goal_text": "eval.agent_goal_text",
        "mock_mode": "eval.simple.mock_mode",
        "n_environment_steps": "eval.simple.n_environment_steps",
        "history_window": "eval.simple.history_window",
        "max_image_history": "eval.simple.max_image_history",
        "hide_obs_when_image": "eval.simple.hide_obs_when_image",
    }
    for source_key, target_key in override_map.items():
        value = _select(config, f"artifact_eval.{source_key}", None)
        if value is not None:
            OmegaConf.update(config, target_key, value, merge=False)


def _configure_worker_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_filename = output_dir / "eval.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
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
    successful = [r for r in repeat_records if r.get("error") is None]
    aggregate_reward = 0.0
    for record in successful:
        summary = record.get("summary")
        if isinstance(summary, dict):
            aggregate_reward += float(summary.get("episode_return", 0.0))
    return {
        "num_repeats": len(repeat_records),
        "num_successful_repeats": len(successful),
        "num_failed_repeats": len(repeat_records) - len(successful),
        "episode_return_sum": aggregate_reward,
        "episode_return_mean": (
            aggregate_reward / len(successful) if successful else None
        ),
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


def _build_simple_config(config: DictConfig) -> SimpleStepwiseConfig:
    scfg = config.eval.get("simple", {}) or {}
    return SimpleStepwiseConfig(
        n_environment_steps=int(scfg.get("n_environment_steps", 50)),
        history_window=int(scfg.get("history_window", 20)),
        max_image_history=int(scfg.get("max_image_history", 1)),
        model=str(scfg.get("model", "anthropic/claude-sonnet-4-5-20250929")),
        api_key_env=str(scfg.get("api_key_env", "ANTHROPIC_API_KEY")),
        base_url=scfg.get("base_url", None),
        temperature=scfg.get("temperature", None),
        max_output_tokens=scfg.get("max_output_tokens", None),
        hide_obs_when_image=bool(scfg.get("hide_obs_when_image", False)),
        mock_mode=bool(scfg.get("mock_mode", False)),
        dynamics_history_path=scfg.get("dynamics_history_path", None),
        history_checkpoint_interval=int(scfg.get("history_checkpoint_interval", 0)),
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
    worker_config.eval.resume_from = None
    OmegaConf.update(
        worker_config,
        "eval.simple.dynamics_history_path",
        str(artifact["history_path"]),
        merge=False,
    )
    if seed is not None:
        OmegaConf.update(worker_config, "envs.env_kwargs.seed", int(seed), merge=False)
    worker_config.eval.output_dir = str(output_dir.parent)

    with open(output_dir / "config.yaml", "w") as f:
        OmegaConf.save(config=worker_config, f=f)

    sc = _build_simple_config(worker_config)

    summary: dict[str, Any] = {}
    error: str | None = None
    try:
        stepwise_simple(
            sc=sc,
            config=worker_config,
            original_cwd=str(output_dir),
            output_dir=str(output_dir),
        )
        episode_logs = sorted(output_dir.glob("episode_*/episode_log.json"))
        episode_return = 0.0
        num_steps = 0
        for log_path in episode_logs:
            data = _read_json(log_path)
            episode_return += float(data.get("episode_return", 0.0))
            num_steps += int(data.get("num_steps", 0))
        summary = {
            "episode_return": episode_return,
            "num_episodes": len(episode_logs),
            "num_steps": num_steps,
        }
    except Exception as exc:
        logging.exception("Simple artifact evaluation failed")
        error = str(exc)

    return _make_repeat_record(
        artifact=artifact,
        repeat_idx=repeat_idx,
        repeat_output_dir=output_dir,
        seed=seed,
        summary=summary,
        error=error,
    )


@hydra.main(config_path="BALROG/balrog/config", config_name="config", version_base="1.1")
def main(config: DictConfig):
    original_cwd = get_original_cwd()
    source_run_raw = _select(config, "artifact_eval.source_run")
    if source_run_raw is None:
        raise ValueError("artifact_eval.source_run is required for simple eval.")

    source_run = _resolve_path(source_run_raw, original_cwd)
    run_config = _load_run_config(config, source_run)
    artifacts = discover_artifacts(source_run)
    selected = select_artifacts(
        artifacts,
        _select(run_config, "artifact_eval.steps", ["final"]),
        every_n=_select(run_config, "artifact_eval.every_n", None),
    )

    run_name_suffix = (
        f"simple_{run_config.envs.names}_artifact_eval"
    )
    _, output_dir = setup_run(
        run_config,
        run_name_suffix=run_name_suffix,
        resume_from=None,
        output_dir_base=run_config.eval.output_dir,
        logger_name="evolve",
    )
    output_path = Path(output_dir)
    logger.info(f"Source run: {source_run}")
    logger.info(f"Discovered {len(artifacts)} artifacts; evaluating {len(selected)}")

    repeats = int(_select(run_config, "artifact_eval.repeats", 1) or 1)
    parallel_workers = int(_select(run_config, "artifact_eval.parallel_workers", 1) or 1)
    run_config_container = OmegaConf.to_container(run_config, resolve=True)

    summary: dict[str, Any] = {
        "source_run": str(source_run),
        "num_discovered_artifacts": len(artifacts),
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
            jobs.append({
                "artifact": artifact,
                "repeat_idx": repeat_idx,
                "repeat_output_dir": repeat_output_dir,
                "seed": _seed_for_repeat(run_config, repeat_idx),
            })

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
                job["artifact"]["label"], job["repeat_idx"], job["artifact"]["history_path"],
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
                    job["artifact"]["label"], job["repeat_idx"], job["artifact"]["history_path"],
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

    logger.info(f"Simple artifact evaluation complete: {output_path / 'artifact_eval_summary.json'}")


if __name__ == "__main__":
    main()
