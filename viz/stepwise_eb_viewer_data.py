#!/usr/bin/env python3
"""Shared data loading and static export helpers for the stepwise EB viewer."""

from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
from datetime import datetime, timezone


STATIC_INDEX_VERSION = 1
PROMPT_IMAGE_PATH_KEYS = {
    "prompt_image_paths",
    "question_gen_image_paths",
    "experiment_image_paths",
}


def read_file(path):
    try:
        with open(path, "r") as f:
            return f.read()
    except (FileNotFoundError, IsADirectoryError):
        return ""


def read_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def is_safe_step_image_path(path):
    """Return True for step-local PNG paths that are safe to serve/copy."""
    if not isinstance(path, str) or not path:
        return False
    normalized = path.replace("\\", "/")
    if normalized.startswith("/") or normalized.startswith("../") or "/../" in normalized:
        return False
    parts = [part for part in normalized.split("/") if part]
    if any(part in (".", "..") for part in parts):
        return False
    return normalized.lower().endswith(".png")


def collect_prompt_image_paths(obj):
    """Collect explicit prompt-image relative paths from nested log objects."""
    paths = []

    def walk(value, key=None):
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                walk(child_value, child_key)
        elif isinstance(value, list):
            if key in PROMPT_IMAGE_PATH_KEYS:
                for item in value:
                    if is_safe_step_image_path(item) and item not in paths:
                        paths.append(item)
            else:
                for item in value:
                    walk(item, key)

    walk(obj)
    return paths


def resolve_log_dir(raw_path):
    """Resolve and validate a log directory path."""
    log_dir = os.path.abspath(raw_path)
    if not os.path.isdir(log_dir):
        raise ValueError(f"Not a directory: {log_dir}")

    has_episodes = any(
        name.startswith("episode_") and os.path.isdir(os.path.join(log_dir, name))
        for name in os.listdir(log_dir)
    )
    if not has_episodes:
        subdirs = sorted(
            name for name in os.listdir(log_dir)
            if os.path.isdir(os.path.join(log_dir, name))
        )
        if not subdirs:
            raise ValueError(f"No subdirectories in: {log_dir}")
        log_dir = os.path.join(log_dir, subdirs[-1])

    return log_dir


def load_log_dir(log_dir):
    """Load the full step-centric log structure."""
    summary = read_json(os.path.join(log_dir, "summary.json"))
    config = read_file(os.path.join(log_dir, "config.yaml"))

    all_steps = []
    episodes = []

    episode_dirs = []
    for name in os.listdir(log_dir):
        if name.startswith("episode_") and os.path.isdir(os.path.join(log_dir, name)):
            try:
                idx = int(name.replace("episode_", ""))
                episode_dirs.append((idx, name))
            except ValueError:
                pass
    episode_dirs.sort()

    for ep_idx, ep_name in episode_dirs:
        ep_path = os.path.join(log_dir, ep_name)
        ep_log = read_json(os.path.join(ep_path, "episode_log.json"))

        if not ep_log:
            for filename in os.listdir(ep_path):
                if filename.endswith(".json") and "_run_" in filename:
                    ep_log = read_json(os.path.join(ep_path, filename))
                    if ep_log and "episode_return" in ep_log:
                        break

        ep_info = {
            "index": ep_idx,
            "path": ep_path,
            "log": ep_log or {},
            "input_beliefs": read_file(os.path.join(ep_path, "input_beliefs.txt")),
            "input_perception": read_file(os.path.join(ep_path, "input_perception.py")),
        }
        episodes.append(ep_info)

        step_dirs = []
        for sname in os.listdir(ep_path):
            if sname.startswith("step_") and os.path.isdir(os.path.join(ep_path, sname)):
                try:
                    s_idx = int(sname.replace("step_", ""))
                    step_dirs.append((s_idx, sname))
                except ValueError:
                    pass
        step_dirs.sort()

        for s_idx, s_name in step_dirs:
            s_path = os.path.join(ep_path, s_name)
            step_log = read_json(os.path.join(s_path, "step_log.json")) or {}
            all_steps.append({
                "episode_idx": ep_idx,
                "step": s_idx,
                "global_step": step_log.get("global_step", len(all_steps)),
                "action": step_log.get("action", ""),
                "reward": step_log.get("reward", 0),
                "done": step_log.get("done", False),
                "episode_return_so_far": step_log.get("episode_return_so_far", 0),
                "agent_step_cost": step_log.get("agent_step_cost", 0),
                "extract_cost": step_log.get("extract_cost", 0),
                "improve_cost": step_log.get("improve_cost", 0),
                "experiment_cost": step_log.get("experiment_cost", 0),
                "step_total_cost": step_log.get("step_total_cost", 0),
                "num_qa_pairs": step_log.get("num_qa_pairs", 0),
                "num_unanswered_questions": step_log.get("num_unanswered_questions", 0),
                "has_beliefs": os.path.exists(os.path.join(s_path, "beliefs.txt")),
                "has_feedback": os.path.exists(os.path.join(s_path, "feedback_history.json")),
                "has_improve_log": os.path.exists(os.path.join(s_path, "improve.log")),
                "has_extraction_log": os.path.exists(os.path.join(s_path, "extraction_log.json")),
                "has_trim_log": os.path.exists(os.path.join(s_path, "trim_log.json")),
                "has_experiment_log": os.path.exists(os.path.join(s_path, "experiment_log.json")),
                "has_agent_messages": os.path.exists(os.path.join(s_path, "agent_messages.json")),
                "has_obs_before": os.path.exists(os.path.join(s_path, "obs_before.png")),
                "has_obs_after": os.path.exists(os.path.join(s_path, "obs_after.png")),
                "env_info": step_log.get("env_info"),
                "trim_cost": step_log.get("trim_cost", 0),
                "did_gen_questions": step_log.get("did_gen_questions", False),
                "did_formulate_experiment": step_log.get("did_formulate_experiment", False),
                "did_trim": step_log.get("did_trim", False),
                "active_experiment": step_log.get("active_experiment"),
                "selected_question": step_log.get("selected_question"),
                "phase": step_log.get("phase", "complete"),
                "path": s_path,
            })

    total_cost = 0
    if summary and "steps" in summary and summary["steps"]:
        total_cost = sum(step.get("step_cost", 0) for step in summary["steps"])

    return {
        "config": config,
        "summary": summary,
        "episodes": [{
            "index": episode["index"],
            "log": episode["log"],
            "input_beliefs": episode["input_beliefs"],
            "input_perception": episode["input_perception"],
        } for episode in episodes],
        "steps": all_steps,
        "total_cost": total_cost,
        "log_dir_name": os.path.basename(log_dir),
    }


def load_step_detail(log_dir, episode_idx, step_idx):
    """Load full detail for a specific step."""
    step_path = os.path.join(log_dir, f"episode_{episode_idx}", f"step_{step_idx:03d}")
    if not os.path.isdir(step_path):
        return {"error": f"Step dir not found: {step_path}"}
    detail = {
        "beliefs": read_file(os.path.join(step_path, "beliefs.txt")),
        "perception": read_file(os.path.join(step_path, "perception.py")),
        "qa_pairs": read_json(os.path.join(step_path, "qa_pairs.json")) or [],
        "feedback_history": read_json(os.path.join(step_path, "feedback_history.json")) or [],
        "extraction_log": read_json(os.path.join(step_path, "extraction_log.json")) or {},
        "trim_log": read_json(os.path.join(step_path, "trim_log.json")) or {},
        "experiment_log": read_json(os.path.join(step_path, "experiment_log.json")) or {},
        "agent_messages": read_json(os.path.join(step_path, "agent_messages.json")) or [],
        "improve_log": read_file(os.path.join(step_path, "improve.log")),
        "step_log": read_json(os.path.join(step_path, "step_log.json")) or {},
    }
    # Include image availability flags
    detail["has_obs_before"] = os.path.exists(os.path.join(step_path, "obs_before.png"))
    detail["has_obs_after"] = os.path.exists(os.path.join(step_path, "obs_after.png"))
    return detail


def load_trajectory(log_dir, episode_idx):
    """Load trajectory CSV for an episode."""
    episode_dir = os.path.join(log_dir, f"episode_{episode_idx}")
    csv_file = os.path.join(episode_dir, "trajectory.csv")
    if not os.path.isfile(csv_file):
        for filename in os.listdir(episode_dir):
            if filename.endswith(".csv"):
                csv_file = os.path.join(episode_dir, filename)
                break
        else:
            return []

    text = read_file(csv_file)
    if not text:
        return []

    steps = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        steps.append({
            "step": row.get("Step", ""),
            "action": row.get("Action", ""),
            "reasoning": row.get("Reasoning", ""),
            "observation": row.get("Observation", ""),
            "auxiliary_observation": row.get("Auxiliary_Observation", ""),
            "reward": row.get("Reward", ""),
            "done": row.get("Done", ""),
        })

    # Fallback for older runs whose CSV does not contain a terminal post-action
    # row (only written when done=True). If trajectory_buffer.json exists, use
    # the last buffer entry's `result_raw_*_term_context` to synthesize a
    # terminal row showing the state *after* the final action.
    def _has_terminal_row(rows):
        if not rows:
            return False
        last = rows[-1]
        return (not last.get("action")) and str(last.get("done", "")).strip().lower() == "true"

    if not _has_terminal_row(steps):
        buffer_path = os.path.join(episode_dir, "trajectory_buffer.json")
        buffer = read_json(buffer_path)
        if isinstance(buffer, list):
            # Find the last non-boundary entry that belongs to this episode and
            # has a captured post-action observation.
            last_entry = None
            for entry in reversed(buffer):
                if not isinstance(entry, dict):
                    continue
                if entry.get("episode_boundary"):
                    break  # earlier entries belong to a previous episode
                if entry.get("result_raw_long_term_context") is not None:
                    last_entry = entry
                    break

            if last_entry is not None:
                result_long = last_entry.get("result_raw_long_term_context", "") or ""
                result_short = last_entry.get("result_raw_short_term_context", "") or ""
                # Format matches stepwise_eb_learn._compose_obs_text. Note: this
                # is the *raw* post-action observation (pre-perception), since
                # only the raw texts are persisted in trajectory_buffer.json.
                sep = "=" * 10
                obs_text = (
                    f"{result_short}\n\n"
                    f"{sep} Start of Direct Observation (post-action, raw) {sep}\n"
                    f"{result_long}\n\n"
                    f"{sep} End of Direct Observation {sep}"
                )
                try:
                    next_step_num = int(steps[-1].get("step", len(steps) - 1)) + 1 if steps else 0
                except (TypeError, ValueError):
                    next_step_num = len(steps)
                steps.append({
                    "step": str(next_step_num),
                    "action": "",
                    "reasoning": "",
                    "observation": obs_text,
                    "auxiliary_observation": result_short,
                    "reward": "0.0",
                    "done": "True",
                })
    return steps


def load_combined_trajectory(log_dir):
    """Load trajectories from all episodes, with episode boundary markers."""
    episode_dirs = []
    for name in os.listdir(log_dir):
        if name.startswith("episode_") and os.path.isdir(os.path.join(log_dir, name)):
            try:
                idx = int(name.replace("episode_", ""))
                episode_dirs.append((idx, name))
            except ValueError:
                pass
    episode_dirs.sort()

    combined = []
    global_step = 0
    for ep_idx, _ in episode_dirs:
        if combined:
            combined.append({
                "episode_boundary": True,
                "episode_idx": ep_idx,
                "global_step": global_step,
            })
        episode_traj = load_trajectory(log_dir, ep_idx)
        for row in episode_traj:
            row["episode_idx"] = ep_idx
            is_terminal = (not row.get("action")) and str(row.get("done", "")).strip().lower() == "true"
            row["global_step"] = global_step
            if not is_terminal:
                global_step += 1
            combined.append(row)
    return combined


def load_experiment_timeline(log_dir):
    """Collect experiment + question history across all episodes and steps."""
    episode_dirs = []
    for name in os.listdir(log_dir):
        if name.startswith("episode_") and os.path.isdir(os.path.join(log_dir, name)):
            try:
                idx = int(name.replace("episode_", ""))
                episode_dirs.append((idx, name))
            except ValueError:
                pass
    episode_dirs.sort()

    timeline = []
    total_questions_generated = 0
    total_experiments_formulated = 0
    for ep_idx, ep_name in episode_dirs:
        ep_path = os.path.join(log_dir, ep_name)
        step_dirs = []
        for sname in os.listdir(ep_path):
            if sname.startswith("step_") and os.path.isdir(os.path.join(ep_path, sname)):
                try:
                    s_idx = int(sname.replace("step_", ""))
                    step_dirs.append((s_idx, sname))
                except ValueError:
                    pass
        step_dirs.sort()

        for s_idx, s_name in step_dirs:
            s_path = os.path.join(ep_path, s_name)
            exp_log = read_json(os.path.join(s_path, "experiment_log.json"))
            step_log = read_json(os.path.join(s_path, "step_log.json")) or {}
            if not exp_log:
                continue

            new_questions = exp_log.get("new_questions", [])
            experiment_plan = exp_log.get("experiment_plan")
            selected_q_idx = exp_log.get("selected_question_index")

            # Try to resolve the selected question text from qa_pairs
            selected_question_text = None
            if selected_q_idx is not None:
                qa_pairs = read_json(os.path.join(s_path, "qa_pairs.json")) or []
                if 0 <= selected_q_idx < len(qa_pairs):
                    selected_question_text = qa_pairs[selected_q_idx].get("question", "")

            total_questions_generated += len(new_questions)
            if experiment_plan:
                total_experiments_formulated += 1

            timeline.append({
                "episode_idx": ep_idx,
                "step": s_idx,
                "global_step": step_log.get("global_step", s_idx),
                "new_questions": new_questions,
                "experiment_plan": experiment_plan,
                "selected_question_index": selected_q_idx,
                "selected_question_text": selected_question_text,
                "cumulative_questions": total_questions_generated,
                "cumulative_experiments": total_experiments_formulated,
            })
    return timeline


def load_qa_timeline(log_dir):
    """Collect QA pair evolution across all episodes and steps."""
    episode_dirs = []
    for name in os.listdir(log_dir):
        if name.startswith("episode_") and os.path.isdir(os.path.join(log_dir, name)):
            try:
                idx = int(name.replace("episode_", ""))
                episode_dirs.append((idx, name))
            except ValueError:
                pass
    episode_dirs.sort()

    timeline = []
    for ep_idx, ep_name in episode_dirs:
        ep_path = os.path.join(log_dir, ep_name)
        step_dirs = []
        for sname in os.listdir(ep_path):
            if sname.startswith("step_") and os.path.isdir(os.path.join(ep_path, sname)):
                try:
                    s_idx = int(sname.replace("step_", ""))
                    step_dirs.append((s_idx, sname))
                except ValueError:
                    pass
        step_dirs.sort()

        for s_idx, s_name in step_dirs:
            s_path = os.path.join(ep_path, s_name)
            qa_file = os.path.join(s_path, "qa_pairs.json")
            step_log = read_json(os.path.join(s_path, "step_log.json")) or {}
            if not os.path.exists(qa_file):
                continue
            qa_pairs = read_json(qa_file) or []
            answered = sum(1 for qa in qa_pairs if qa.get("answer") is not None)
            unanswered = len(qa_pairs) - answered

            # Include new questions from experiment_log if available
            exp_log = read_json(os.path.join(s_path, "experiment_log.json"))
            new_questions = exp_log.get("new_questions", []) if exp_log else []

            # Include all questions with their status
            all_questions = []
            for qa in qa_pairs:
                all_questions.append({
                    "question": qa.get("question", ""),
                    "answer": qa.get("answer"),
                    "source_step": qa.get("source_step"),
                })

            timeline.append({
                "episode_idx": ep_idx,
                "step": s_idx,
                "global_step": step_log.get("global_step", s_idx),
                "total": len(qa_pairs),
                "answered": answered,
                "unanswered": unanswered,
                "new_questions": new_questions,
                "all_questions": all_questions,
            })
    return timeline


def default_run_id(log_dir):
    name = os.path.basename(log_dir.rstrip(os.sep))
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-._")
    return slug or "stepwise-eb-run"


def export_static_report(log_dir, export_root, run_id=None, title=None, description=None):
    """Export one log directory to static JSON files and update the run index."""
    resolved = resolve_log_dir(log_dir)
    run_id = run_id or default_run_id(resolved)
    run_dir = os.path.join(os.path.abspath(export_root), run_id)
    os.makedirs(run_dir, exist_ok=True)

    report = load_log_dir(resolved)
    write_json(os.path.join(run_dir, "report.json"), report)
    write_json(os.path.join(run_dir, "combined_trajectory.json"), load_combined_trajectory(resolved))
    write_json(os.path.join(run_dir, "experiment_timeline.json"), load_experiment_timeline(resolved))
    write_json(os.path.join(run_dir, "qa_timeline.json"), load_qa_timeline(resolved))

    for episode in report["episodes"]:
        ep_idx = episode["index"]
        trajectory_name = f"episode_{ep_idx:03d}.json"
        write_json(os.path.join(run_dir, "trajectories", trajectory_name), load_trajectory(resolved, ep_idx))

    for step in report["steps"]:
        detail_name = f"ep_{step['episode_idx']:03d}_step_{step['step']:03d}.json"
        step_detail = load_step_detail(resolved, step["episode_idx"], step["step"])
        write_json(
            os.path.join(run_dir, "step_details", detail_name),
            step_detail,
        )
        # Copy observation images for static export
        src_step_dir = os.path.join(resolved, f"episode_{step['episode_idx']}", f"step_{step['step']:03d}")
        for img_name in ("obs_before.png", "obs_after.png"):
            src_img = os.path.join(src_step_dir, img_name)
            if os.path.isfile(src_img):
                dst_img = os.path.join(
                    run_dir, "images",
                    f"ep_{step['episode_idx']:03d}_step_{step['step']:03d}_{img_name}",
                )
                os.makedirs(os.path.dirname(dst_img), exist_ok=True)
                shutil.copy2(src_img, dst_img)

        for rel_img in collect_prompt_image_paths(step_detail):
            src_img = os.path.normpath(os.path.join(src_step_dir, rel_img))
            src_root = os.path.abspath(src_step_dir)
            src_abs = os.path.abspath(src_img)
            if not src_abs.startswith(src_root + os.sep) or not os.path.isfile(src_abs):
                continue
            dst_img = os.path.join(
                run_dir,
                "images",
                f"ep_{step['episode_idx']:03d}_step_{step['step']:03d}",
                rel_img.replace("\\", "/"),
            )
            os.makedirs(os.path.dirname(dst_img), exist_ok=True)
            shutil.copy2(src_abs, dst_img)

    index_path = os.path.join(os.path.abspath(export_root), "index.json")
    index_data = read_json(index_path) or {"version": STATIC_INDEX_VERSION, "runs": []}
    index_data["version"] = STATIC_INDEX_VERSION

    run_record = {
        "id": run_id,
        "title": title or report["log_dir_name"],
        "description": description or "",
        "path": f"{run_id}/",
        "log_dir_name": report["log_dir_name"],
        "source_log_dir": resolved,
        "episodes": len(report["episodes"]),
        "steps": len(report["steps"]),
        "total_cost": report["total_cost"],
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }

    updated_runs = []
    replaced = False
    for existing in index_data.get("runs", []):
        if existing.get("id") == run_id:
            updated_runs.append(run_record)
            replaced = True
        else:
            updated_runs.append(existing)
    if not replaced:
        updated_runs.append(run_record)

    index_data["runs"] = updated_runs
    write_json(index_path, index_data)

    return {
        "resolved_log_dir": resolved,
        "run_dir": run_dir,
        "index_path": index_path,
        "run_record": run_record,
    }
