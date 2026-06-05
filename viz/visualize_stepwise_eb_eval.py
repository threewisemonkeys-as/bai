#!/usr/bin/env python3
"""Visualize stepwise eval logs with the stepwise EB viewer."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

from stepwise_eb_viewer_data import (
    is_run_dir as is_stepwise_run_dir,
    is_safe_step_image_path,
    load_combined_trajectory as load_stepwise_combined_trajectory,
    load_experiment_timeline,
    load_log_dir as load_stepwise_log_dir,
    load_qa_timeline,
    load_step_detail as load_stepwise_step_detail,
    load_trajectory as load_stepwise_trajectory,
)


ASSET_DIR = Path(__file__).resolve().parent / "stepwise_eb_learn"
INDEX_HTML_PATH = ASSET_DIR / "index.html"
APP_JS_PATH = ASSET_DIR / "app.js"


def read_file(path: Path) -> str:
    try:
        return path.read_text()
    except (FileNotFoundError, IsADirectoryError, OSError):
        return ""


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def load_asset(path: Path) -> str:
    return path.read_text()


def render_index_html() -> str:
    config = {
        "mode": "dynamic",
        "allowDynamicInput": True,
        "dataIndexPath": None,
    }
    html = load_asset(INDEX_HTML_PATH)
    injection = (
        "<script>"
        f"window.STEPWISE_EB_VIEWER_CONFIG = {json.dumps(config)};"
        "</script>"
    )
    return html.replace("<!-- CONFIG_INJECTION -->", injection)


def is_eval_run_dir(path: Path) -> bool:
    return path.is_dir() and (path / "artifact_eval_summary.json").is_file()


def is_repeat_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "source_artifact.json").is_file()
        and ((path / "frozen_eval").is_dir() or (path / "autumn_frozen_eval").is_dir())
    )


def is_loadable_run_dir(path: Path) -> bool:
    return is_eval_run_dir(path) or is_repeat_dir(path) or is_stepwise_run_dir(str(path))


def _safe_rel_id(root: Path, path: Path) -> str:
    rel = path.resolve().relative_to(root.resolve())
    return "" if str(rel) == "." else rel.as_posix()


def _episode_dirs(run_dir: Path) -> list[tuple[int, Path]]:
    episodes = []
    if not run_dir.is_dir():
        return episodes
    for episode_dir in run_dir.glob("episode_*"):
        if not episode_dir.is_dir():
            continue
        try:
            idx = int(episode_dir.name.replace("episode_", ""))
        except ValueError:
            continue
        episodes.append((idx, episode_dir))
    return sorted(episodes)


def _step_dirs(episode_dir: Path) -> list[tuple[int, Path]]:
    steps = []
    if not episode_dir.is_dir():
        return steps
    for step_dir in episode_dir.glob("step_*"):
        if not step_dir.is_dir():
            continue
        try:
            idx = int(step_dir.name.replace("step_", ""))
        except ValueError:
            continue
        steps.append((idx, step_dir))
    return sorted(steps)


def _count_stepwise_run(run_path: Path) -> tuple[int, int]:
    episodes = _episode_dirs(run_path)
    return len(episodes), sum(len(_step_dirs(episode_dir)) for _idx, episode_dir in episodes)


def _task_run_dirs(eval_root: Path):
    if not eval_root.is_dir():
        return []
    return sorted(
        [p for p in eval_root.iterdir() if p.is_dir() and (p / "trajectory.json").is_file()],
        key=lambda p: p.name,
    )


def _task_run_step_count(task_run: dict) -> int:
    if task_run.get("kind") == "stepwise_episode":
        return len(_step_dirs(Path(task_run["episode_dir"])))
    return len(read_json(task_run["trajectory_path"]) or [])


def _run_entry(run_path: Path, run_id: str, name: str) -> dict:
    try:
        if is_stepwise_run_dir(str(run_path)) and not is_eval_run_dir(run_path):
            episode_count, step_count = _count_stepwise_run(run_path)
            task_runs = [None] * episode_count
        else:
            task_runs = discover_eval_task_runs(run_path)
            step_count = sum(_task_run_step_count(task_run) for task_run in task_runs)
    except Exception:
        task_runs = []
        step_count = 0
    return {
        "id": run_id,
        "name": name,
        "path": str(run_path),
        "episodes": len(task_runs),
        "steps": step_count,
    }


def _discover_loadable_runs(root: Path) -> list[Path]:
    if is_loadable_run_dir(root):
        return [root]

    runs: list[Path] = []

    def walk(path: Path, depth: int) -> None:
        if depth > 6:
            return
        try:
            children = sorted([p for p in path.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
        except OSError:
            return
        for child in children:
            if child.name == "phase1_learn":
                continue
            if is_loadable_run_dir(child):
                runs.append(child)
            else:
                walk(child, depth + 1)

    walk(root, 0)
    return runs


def list_runs(log_dir: str) -> list[dict]:
    root = Path(log_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")
    if is_loadable_run_dir(root):
        return [_run_entry(root, "", root.name)]

    runs = []
    for run_path in _discover_loadable_runs(root):
        runs.append(_run_entry(run_path, _safe_rel_id(root, run_path), run_path.name))
    return runs


def resolve_run_dir(raw_log_dir: str, run_id: str | None = None) -> Path:
    log_dir = Path(raw_log_dir).resolve()
    if not log_dir.is_dir():
        raise ValueError(f"Not a directory: {log_dir}")
    if run_id:
        parts = [p for p in run_id.split("/") if p]
        if not parts or any(p in (".", "..") for p in parts):
            raise ValueError(f"Invalid run id: {run_id}")
        candidate = (log_dir / Path(*parts)).resolve()
        if os.path.commonpath([str(candidate), str(log_dir)]) != str(log_dir):
            raise ValueError(f"Run id must resolve inside log dir: {run_id}")
        if not is_loadable_run_dir(candidate):
            raise ValueError(f"Not a visualizable eval run: {candidate}")
        return candidate
    if is_loadable_run_dir(log_dir):
        return log_dir
    raise ValueError(f"{log_dir} contains multiple runs; specify one via the run parameter")


def _get_run_dir(params) -> Path:
    raw = params.get("log_dir", [None])[0]
    if not raw:
        raise ValueError("Missing log_dir parameter")
    return resolve_run_dir(raw, params.get("run", [None])[0] or None)


def _get_log_dir_only(params) -> Path:
    raw = params.get("log_dir", [None])[0]
    if not raw:
        raise ValueError("Missing log_dir parameter")
    log_dir = Path(raw).resolve()
    if not log_dir.is_dir():
        raise ValueError(f"Not a directory: {log_dir}")
    return log_dir


def _task_runs_from_repeat_dir(artifact_label: str, repeat_dir: Path) -> list[dict]:
    source_artifact = read_json(repeat_dir / "source_artifact.json") or {}
    task_runs = []
    for eval_type, eval_root_name in (("frozen_eval", "frozen_eval"), ("autumn_frozen_eval", "autumn_frozen_eval")):
        eval_root = repeat_dir / eval_root_name
        for task_dir in _task_run_dirs(eval_root):
            task_runs.append({
                "kind": "eb_eval_task",
                "artifact_label": artifact_label,
                "artifact_dir": repeat_dir,
                "source_artifact": source_artifact,
                "eval_type": eval_type,
                "run_key": task_dir.name,
                "task_dir": task_dir,
                "episode_log_path": task_dir / "episode_log.json",
                "trajectory_path": task_dir / "trajectory.json",
            })
    if not task_runs and is_stepwise_run_dir(str(repeat_dir)):
        task_runs.extend(_task_runs_from_stepwise_run(artifact_label, repeat_dir, source_artifact))
    return task_runs


def _task_runs_from_stepwise_run(artifact_label: str, run_dir: Path, source_artifact: dict) -> list[dict]:
    task_runs = []
    for episode_idx, episode_dir in _episode_dirs(run_dir):
        task_runs.append({
            "kind": "stepwise_episode",
            "artifact_label": artifact_label,
            "artifact_dir": run_dir,
            "source_artifact": source_artifact,
            "eval_type": "artifact_eval",
            "run_key": f"{run_dir.name}/episode_{episode_idx}",
            "run_dir": run_dir,
            "episode_idx": episode_idx,
            "episode_dir": episode_dir,
            "episode_log_path": episode_dir / "episode_log.json",
        })
    return task_runs


def _task_runs_from_artifact_dir(artifact_label: str, artifact_output_dir: Path, source_fallback: dict) -> list[dict]:
    # Check for repeat_NNN subdirs (parallel runner structure)
    repeat_dirs = sorted(
        [p for p in artifact_output_dir.iterdir() if p.is_dir() and (is_repeat_dir(p) or is_stepwise_run_dir(str(p)))],
        key=lambda p: p.name,
    ) if artifact_output_dir.is_dir() else []
    if repeat_dirs and not is_stepwise_run_dir(str(artifact_output_dir)):
        task_runs = []
        for repeat_dir in repeat_dirs:
            task_runs.extend(_task_runs_from_repeat_dir(f"{artifact_label}/{repeat_dir.name}", repeat_dir))
        return task_runs

    source_artifact = read_json(artifact_output_dir / "source_artifact.json") or source_fallback
    task_runs = []
    for eval_type, eval_root_name in (("frozen_eval", "frozen_eval"), ("autumn_frozen_eval", "autumn_frozen_eval")):
        eval_root = artifact_output_dir / eval_root_name
        for task_dir in _task_run_dirs(eval_root):
            task_runs.append({
                "kind": "eb_eval_task",
                "artifact_label": artifact_label,
                "artifact_dir": artifact_output_dir,
                "source_artifact": source_artifact,
                "eval_type": eval_type,
                "run_key": task_dir.name,
                "task_dir": task_dir,
                "episode_log_path": task_dir / "episode_log.json",
                "trajectory_path": task_dir / "trajectory.json",
            })
    if not task_runs and is_stepwise_run_dir(str(artifact_output_dir)):
        task_runs.extend(_task_runs_from_stepwise_run(artifact_label, artifact_output_dir, source_artifact))
    return task_runs


def discover_eval_task_runs(run_dir: Path) -> list[dict]:
    # Handle a single repeat_NNN dir passed directly
    if is_repeat_dir(run_dir):
        return _task_runs_from_repeat_dir(run_dir.name, run_dir)
    if is_stepwise_run_dir(str(run_dir)) and not is_eval_run_dir(run_dir):
        return _task_runs_from_stepwise_run(run_dir.name, run_dir, {})

    summary = read_json(run_dir / "artifact_eval_summary.json") or {}
    artifacts = summary.get("artifacts") if isinstance(summary, dict) else None
    task_runs: list[dict] = []

    if isinstance(artifacts, dict):
        for artifact_label, artifact_record in artifacts.items():
            artifact_output_dir = Path(artifact_record.get("output_dir") or run_dir / artifact_label)
            if not artifact_output_dir.is_absolute():
                artifact_output_dir = (run_dir / artifact_output_dir).resolve()
                if not artifact_output_dir.exists():
                    artifact_output_dir = Path(artifact_record.get("output_dir", "")).resolve()
            task_runs.extend(_task_runs_from_artifact_dir(
                artifact_label, artifact_output_dir, artifact_record.get("source") or {}
            ))

    if task_runs:
        return task_runs

    # Fallback: allow opening a directory that contains artifact subdirs even if
    # artifact_eval_summary.json is incomplete.
    for artifact_dir in sorted([p for p in run_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
        task_runs.extend(_task_runs_from_artifact_dir(artifact_dir.name, artifact_dir, {}))
    return task_runs


def _episode_title(task_run: dict, episode_log: dict) -> str:
    task = episode_log.get("task") or episode_log.get("program") or task_run["run_key"]
    env = episode_log.get("env_name") or episode_log.get("program") or ""
    label = task_run["artifact_label"]
    return f"{label} | {env} {task}".strip()


def _bool_done(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _compose_observation(row: dict) -> str:
    pre_aux = row.get("pre_auxiliary_observation") or ""
    pre_obs = row.get("pre_observation") or ""
    post_aux = row.get("post_auxiliary_observation") or ""
    post_obs = row.get("post_observation") or ""
    parts = []
    if pre_aux:
        parts.append(f"Pre-action auxiliary observation:\n{pre_aux}")
    if pre_obs:
        parts.append(f"Pre-action raw observation:\n{pre_obs}")
    if post_aux or post_obs:
        parts.append("Post-action state:")
        if post_aux:
            parts.append(f"Auxiliary:\n{post_aux}")
        if post_obs:
            parts.append(f"Raw:\n{post_obs}")
    return "\n\n".join(parts)


def _is_safe_step_image_path(path: str) -> bool:
    if not isinstance(path, str) or not path:
        return False
    normalized = path.replace("\\", "/")
    if normalized.startswith("/") or normalized.startswith("../") or "/../" in normalized:
        return False
    parts = [part for part in normalized.split("/") if part]
    if any(part in (".", "..") for part in parts):
        return False
    return normalized.lower().endswith(".png")


def _step_dir(task_run: dict, step_idx: int) -> Path:
    if task_run.get("kind") == "stepwise_episode":
        return Path(task_run["episode_dir"]) / f"step_{step_idx:03d}"
    return Path(task_run["task_dir"]) / f"step_{step_idx:03d}"


def _logged_agent_messages(task_run: dict, step_idx: int, row: dict, episode_log: dict) -> list[dict]:
    messages = read_json(_step_dir(task_run, step_idx) / "agent_messages.json")
    if isinstance(messages, list):
        return messages
    return []


def _rows_for_task_run(task_run: dict) -> list[dict]:
    if task_run.get("kind") == "stepwise_episode":
        rows = []
        for step_idx, step_dir in _step_dirs(Path(task_run["episode_dir"])):
            step_log = read_json(step_dir / "step_log.json") or {}
            row = {**step_log}
            row.setdefault("step", step_idx)
            rows.append(row)
        return rows
    trajectory = read_json(task_run["trajectory_path"]) or []
    return trajectory if isinstance(trajectory, list) else []


def load_eval_run(log_dir: Path) -> dict:
    if is_stepwise_run_dir(str(log_dir)) and not is_eval_run_dir(log_dir):
        return load_stepwise_log_dir(str(log_dir))

    summary = read_json(log_dir / "artifact_eval_summary.json") or {}
    config = read_file(log_dir / "config.yaml")
    task_runs = discover_eval_task_runs(log_dir)

    episodes = []
    steps = []
    global_step = 0
    for ep_idx, task_run in enumerate(task_runs):
        episode_log = read_json(task_run["episode_log_path"]) or {}
        trajectory = _rows_for_task_run(task_run)
        episodes.append({
            "index": ep_idx,
            "log": {
                **episode_log,
                "artifact_label": task_run["artifact_label"],
                "run_key": task_run["run_key"],
                "eval_type": task_run["eval_type"],
                "title": _episode_title(task_run, episode_log),
            },
            "input_beliefs": read_file(task_run["artifact_dir"] / "beliefs.txt"),
            "input_perception": read_file(task_run["artifact_dir"] / "perception.py"),
        })
        episode_return_so_far = 0.0
        for local_idx, row in enumerate(trajectory):
            step_idx = int(row.get("step", local_idx) or local_idx)
            step_dir = _step_dir(task_run, step_idx)
            reward = float(row.get("reward", 0.0) or 0.0)
            episode_return_so_far = float(row.get("episode_return_so_far", episode_return_so_far + reward) or 0.0)
            info = row.get("info") if isinstance(row.get("info"), dict) else {}
            env_info = row.get("env_info") or info or episode_log.get("env_stats") or episode_log.get("final_info") or {}
            steps.append({
                "episode_idx": ep_idx,
                "step": step_idx,
                "global_step": row.get("global_step", global_step),
                "action": row.get("action", ""),
                "reward": reward,
                "done": _bool_done(row.get("done", False)),
                "episode_return_so_far": episode_return_so_far,
                "agent_step_cost": row.get("agent_step_cost", 0),
                "extract_cost": row.get("extract_cost", 0),
                "improve_cost": row.get("improve_cost", 0),
                "experiment_cost": row.get("experiment_cost", 0),
                "trim_cost": row.get("trim_cost", 0),
                "step_total_cost": row.get("step_total_cost", row.get("step_cost", 0)),
                "num_qa_pairs": row.get("num_qa_pairs", 0),
                "num_unanswered_questions": row.get("num_unanswered_questions", 0),
                "has_beliefs": (step_dir / "beliefs.txt").is_file(),
                "has_feedback": (step_dir / "feedback_history.json").is_file(),
                "has_improve_log": (step_dir / "improve.log").is_file(),
                "has_extraction_log": (step_dir / "extraction_log.json").is_file(),
                "has_trim_log": (step_dir / "trim_log.json").is_file(),
                "has_question_selection_log": (step_dir / "question_selection_log.json").is_file(),
                "has_experiment_log": (step_dir / "experiment_log.json").is_file(),
                "has_agent_messages": (step_dir / "agent_messages.json").is_file(),
                "has_obs_before": bool(row.get("has_obs_before")) or (step_dir / "obs_before.png").is_file(),
                "has_obs_after": bool(row.get("has_obs_after")) or (step_dir / "obs_after.png").is_file(),
                "env_info": env_info,
                "did_gen_questions": row.get("did_gen_questions", False),
                "did_formulate_experiment": row.get("did_formulate_experiment", False),
                "did_trim": row.get("did_trim", False),
                "active_experiment": f"{task_run['artifact_label']} / {task_run['run_key']}",
                "selected_question": row.get("selected_question"),
                "phase": "complete",
                "path": str(step_dir),
            })
            global_step += 1

    return {
        "config": config,
        "summary": summary,
        "episodes": episodes,
        "steps": steps,
        "total_cost": sum(float((ep["log"] or {}).get("total_cost", 0.0) or 0.0) for ep in episodes),
        "log_dir_name": log_dir.name,
    }


def _task_run_for_episode(log_dir: Path, episode_idx: int) -> dict | None:
    task_runs = discover_eval_task_runs(log_dir)
    if 0 <= episode_idx < len(task_runs):
        return task_runs[episode_idx]
    return None


def load_eval_step_detail(log_dir: Path, episode_idx: int, step_idx: int) -> dict:
    if is_stepwise_run_dir(str(log_dir)) and not is_eval_run_dir(log_dir):
        return load_stepwise_step_detail(str(log_dir), episode_idx, step_idx)

    task_run = _task_run_for_episode(log_dir, episode_idx)
    if task_run is None:
        return {"error": f"Episode not found: {episode_idx}"}
    trajectory = _rows_for_task_run(task_run)
    if not (0 <= step_idx < len(trajectory)):
        return {"error": f"Step not found: episode={episode_idx} step={step_idx}"}
    row = trajectory[step_idx]
    logged_step_idx = int(row.get("step", step_idx) or step_idx)
    episode_log = read_json(task_run["episode_log_path"]) or {}
    step_dir = _step_dir(task_run, logged_step_idx)
    qa_pairs = read_json(step_dir / "qa_pairs.json") or []
    experiment_log = read_json(step_dir / "experiment_log.json") or {}
    experiment_scoring_log = (
        read_json(step_dir / "experiment_scoring_log.json")
        or experiment_log.get("experiment_scoring")
        or {}
    )
    return {
        "beliefs": read_file(step_dir / "beliefs.txt") or read_file(task_run["artifact_dir"] / "beliefs.txt"),
        "perception": read_file(step_dir / "perception.py") or read_file(task_run["artifact_dir"] / "perception.py"),
        "qa_pairs": qa_pairs,
        "qa_pairs_ordered": qa_pairs,
        "feedback_history": read_json(step_dir / "feedback_history.json") or [],
        "extraction_log": read_json(step_dir / "extraction_log.json") or {},
        "trim_log": read_json(step_dir / "trim_log.json") or {},
        "question_selection_log": read_json(step_dir / "question_selection_log.json") or {},
        "experiment_log": {
            **experiment_log,
            "artifact_label": task_run["artifact_label"],
            "run_key": task_run["run_key"],
            "eval_type": task_run["eval_type"],
            "source_artifact": task_run["source_artifact"],
            "episode_log": episode_log,
            "trajectory_row": row,
            "agent_messages_path": str(step_dir / "agent_messages.json"),
        },
        "experiment_scoring_log": experiment_scoring_log,
        "agent_messages": _logged_agent_messages(task_run, logged_step_idx, row, episode_log),
        "improve_log": read_file(step_dir / "improve.log"),
        "step_log": {**row, **(read_json(step_dir / "step_log.json") or {}), "path": str(step_dir)},
        "question_scoring": {},
        "has_obs_before": bool(row.get("has_obs_before")) or (step_dir / "obs_before.png").is_file(),
        "has_obs_after": bool(row.get("has_obs_after")) or (step_dir / "obs_after.png").is_file(),
    }


def load_eval_trajectory(log_dir: Path, episode_idx: int) -> list[dict]:
    if is_stepwise_run_dir(str(log_dir)) and not is_eval_run_dir(log_dir):
        return load_stepwise_trajectory(str(log_dir), episode_idx)

    task_run = _task_run_for_episode(log_dir, episode_idx)
    if task_run is None:
        return []
    if task_run.get("kind") == "stepwise_episode":
        csv_file = Path(task_run["episode_dir"]) / "trajectory.csv"
        text = read_file(csv_file)
        if text:
            return [
                {
                    "step": row.get("Step", ""),
                    "action": row.get("Action", ""),
                    "reasoning": row.get("Reasoning", ""),
                    "observation": row.get("Observation", ""),
                    "auxiliary_observation": row.get("Auxiliary_Observation", ""),
                    "reward": row.get("Reward", ""),
                    "done": row.get("Done", ""),
                }
                for row in csv.DictReader(io.StringIO(text))
            ]
    trajectory = _rows_for_task_run(task_run)
    rows = []
    for idx, row in enumerate(trajectory):
        rows.append({
            "step": str(row.get("step", idx)),
            "action": row.get("action", ""),
            "reasoning": row.get("reasoning", ""),
            "observation": _compose_observation(row),
            "auxiliary_observation": row.get("pre_auxiliary_observation", ""),
            "reward": str(row.get("reward", "")),
            "done": "True" if _bool_done(row.get("done", False)) else "False",
        })
    return rows


def load_eval_combined_trajectory(log_dir: Path) -> list[dict]:
    if is_stepwise_run_dir(str(log_dir)) and not is_eval_run_dir(log_dir):
        return load_stepwise_combined_trajectory(str(log_dir))

    combined = []
    global_step = 0
    for ep_idx, _task_run in enumerate(discover_eval_task_runs(log_dir)):
        if combined:
            combined.append({
                "episode_boundary": True,
                "episode_idx": ep_idx,
                "global_step": global_step,
            })
        for row in load_eval_trajectory(log_dir, ep_idx):
            row["episode_idx"] = ep_idx
            row["global_step"] = global_step
            combined.append(row)
            global_step += 1
    return combined


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            params = parse_qs(parsed.query)

            if path in ("/", "/index.html"):
                self._html_response(render_index_html())
            elif path == "/app.js":
                self._text_response(load_asset(APP_JS_PATH), "application/javascript; charset=utf-8")
            elif path == "/api/runs":
                log_dir = _get_log_dir_only(params)
                self._json_response({"log_dir": str(log_dir), "runs": list_runs(str(log_dir))})
            elif path == "/api/data":
                log_dir = _get_run_dir(params)
                self._json_response(load_eval_run(log_dir))
            elif path == "/api/step_detail":
                log_dir = _get_run_dir(params)
                episode = int(params.get("episode", [0])[0])
                step = int(params.get("step", [0])[0])
                self._json_response(load_eval_step_detail(log_dir, episode, step))
            elif path == "/api/trajectory":
                log_dir = _get_run_dir(params)
                episode = int(params.get("episode", [0])[0])
                self._json_response(load_eval_trajectory(log_dir, episode))
            elif path == "/api/combined_trajectory":
                log_dir = _get_run_dir(params)
                self._json_response(load_eval_combined_trajectory(log_dir))
            elif path == "/api/step_image":
                log_dir = _get_run_dir(params)
                episode = int(params.get("episode", [0])[0])
                step = int(params.get("step", [0])[0])
                name = params.get("name", [""])[0]
                self._file_response(resolve_step_image(log_dir, episode, step, name))
            elif path == "/api/experiment_timeline":
                log_dir = _get_run_dir(params)
                if is_stepwise_run_dir(str(log_dir)) and not is_eval_run_dir(log_dir):
                    self._json_response(load_experiment_timeline(str(log_dir)))
                else:
                    self._json_response([])
            elif path == "/api/qa_timeline":
                log_dir = _get_run_dir(params)
                if is_stepwise_run_dir(str(log_dir)) and not is_eval_run_dir(log_dir):
                    self._json_response(load_qa_timeline(str(log_dir)))
                else:
                    self._json_response([])
            else:
                self.send_error(404)
        except Exception as exc:
            err = f"[visualize eval] request failed for {self.path}: {exc}\n{traceback.format_exc()}"
            print(err, file=sys.stderr, flush=True)
            try:
                self._json_response({"error": str(exc)})
            except Exception:
                pass

    def _json_response(self, obj):
        data = json.dumps(obj, default=str).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def _html_response(self, html: str):
        data = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def _text_response(self, text: str, content_type: str):
        data = text.encode()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def _file_response(self, path: Path):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)


def resolve_step_image(log_dir: Path, episode_idx: int, step_idx: int, name: str) -> Path:
    if not is_safe_step_image_path(name):
        raise ValueError(f"Unsafe image path: {name!r}")
    if is_stepwise_run_dir(str(log_dir)) and not is_eval_run_dir(log_dir):
        step_dir = (log_dir / f"episode_{episode_idx}" / f"step_{step_idx:03d}").resolve()
    else:
        task_run = _task_run_for_episode(log_dir, episode_idx)
        if task_run is None:
            raise ValueError(f"Episode not found: {episode_idx}")
        step_dir = _step_dir(task_run, step_idx).resolve()
    image_path = (step_dir / name).resolve()
    if os.path.commonpath([str(image_path), str(step_dir)]) != str(step_dir):
        raise ValueError(f"Image path escapes step dir: {name!r}")
    if not image_path.is_file():
        raise FileNotFoundError(str(image_path))
    return image_path


def main():
    parser = argparse.ArgumentParser(description="Visualize stepwise eval logs")
    parser.add_argument(
        "log_dir",
        nargs="?",
        default=None,
        help="Path to an eval run, greedy run, or parent containing launch_ee/launch_g runs",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8767, help="HTTP port (default: 8767)")
    parser.add_argument("--open-browser", action="store_true", help="Open browser automatically")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), Handler)
    display_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    url = f"http://{display_host}:{args.port}"

    if args.log_dir:
        abs_path = str(Path(args.log_dir).resolve())
        try:
            runs = list_runs(abs_path)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        if len(runs) == 1:
            print(f"Log dir resolved to: {runs[0]['path']}")
            data = load_eval_run(Path(runs[0]["path"]))
            print(f"Found {len(data['episodes'])} eval task runs, {len(data['steps'])} steps")
        elif len(runs) > 1:
            print(f"Found {len(runs)} eval runs under {abs_path}; pick one in the browser")
        else:
            print(f"No artifact eval runs found under {abs_path}", file=sys.stderr)
            sys.exit(1)
        url += f"?log_dir={quote(abs_path, safe='')}"

    print(f"Serving at {url}")

    if args.open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
