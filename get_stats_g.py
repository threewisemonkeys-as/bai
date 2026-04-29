"""Summarize greedy launcher runs.

Usage:
    uv run get_stats_g.py logs/dev/apr28/20260428-170726
    uv run get_stats_g.py logs/dev/apr28/20260428-170726 --json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


EPISODE_RE = re.compile(r"episode_(\d+)$")


@dataclass
class RunStats:
    name: str
    path: str
    complete: bool
    successful: bool
    first_success_steps: int | None
    first_success_episode: int | None
    episodes: int
    total_steps: int
    source: str
    note: str | None = None


def _load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_success(stats: dict[str, Any]) -> bool:
    """Use explicit progress when present, otherwise fall back to reward signals."""
    env_stats = stats.get("env_stats")
    if isinstance(env_stats, dict):
        progression = _as_float(env_stats.get("progression"))
        if progression is not None and progression > 0:
            return True

        env_return = _as_float(env_stats.get("episode_return"))
        if env_return is not None and env_return > 0:
            return True

    reward = _as_float(stats.get("reward"))
    if reward is not None and reward > 0:
        return True

    episode_return = _as_float(stats.get("episode_return"))
    if episode_return is not None and episode_return > 0:
        return True

    return False


def _episode_idx(path: Path, data: dict[str, Any]) -> int:
    idx = data.get("episode_idx")
    if isinstance(idx, int):
        return idx
    match = EPISODE_RE.match(path.parent.name)
    if match:
        return int(match.group(1))
    return 0


def _summary_files(run_dir: Path) -> list[Path]:
    return sorted(
        p for p in run_dir.rglob("summary.json")
        if "__pycache__" not in p.parts
    )


def _episode_logs(run_dir: Path) -> list[Path]:
    return sorted(
        p for p in run_dir.rglob("episode_log.json")
        if "__pycache__" not in p.parts
    )


def _from_episode_logs(run_dir: Path) -> RunStats | None:
    episode_logs = _episode_logs(run_dir)
    if not episode_logs:
        return None

    episodes: list[tuple[int, Path, dict[str, Any]]] = []
    for path in episode_logs:
        try:
            data = _load_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            return RunStats(
                name=run_dir.name,
                path=str(run_dir),
                complete=False,
                successful=False,
                first_success_steps=None,
                first_success_episode=None,
                episodes=0,
                total_steps=0,
                source="episode_log.json",
                note=f"failed to read {path}: {exc}",
            )
        if isinstance(data, dict):
            episodes.append((_episode_idx(path, data), path, data))

    episodes.sort(key=lambda item: (item[0], str(item[1])))
    cumulative_steps = 0
    first_success_steps: int | None = None
    first_success_episode: int | None = None

    for episode_idx, _path, data in episodes:
        num_steps = int(_as_float(data.get("num_steps")) or 0)
        cumulative_steps += num_steps
        if first_success_steps is None and _is_success(data):
            first_success_steps = cumulative_steps
            first_success_episode = episode_idx

    return RunStats(
        name=run_dir.name,
        path=str(run_dir),
        complete=True,
        successful=first_success_steps is not None,
        first_success_steps=first_success_steps,
        first_success_episode=first_success_episode,
        episodes=len(episodes),
        total_steps=cumulative_steps,
        source="episode_log.json",
    )


def _from_summary(run_dir: Path) -> RunStats | None:
    summaries = _summary_files(run_dir)
    if not summaries:
        return None

    summary_path = summaries[0]
    try:
        data = _load_json(summary_path)
    except (OSError, json.JSONDecodeError) as exc:
        return RunStats(
            name=run_dir.name,
            path=str(run_dir),
            complete=False,
            successful=False,
            first_success_steps=None,
            first_success_episode=None,
            episodes=0,
            total_steps=0,
            source="summary.json",
            note=f"failed to read {summary_path}: {exc}",
        )

    steps = data.get("steps") if isinstance(data, dict) else None
    if not isinstance(steps, list):
        return RunStats(
            name=run_dir.name,
            path=str(run_dir),
            complete=False,
            successful=False,
            first_success_steps=None,
            first_success_episode=None,
            episodes=0,
            total_steps=0,
            source="summary.json",
            note=f"{summary_path} has no steps list",
        )

    first_success_steps: int | None = None
    first_success_episode: int | None = None
    completed_episode_ids: set[int] = set()
    max_step = -1

    for item in steps:
        if not isinstance(item, dict):
            continue
        step_num = int(_as_float(item.get("step")) or 0)
        max_step = max(max_step, step_num)
        rollout = item.get("rollout_stats")
        if not isinstance(rollout, dict):
            continue

        episode_idx = int(_as_float(rollout.get("episode_idx")) or 0)
        if rollout.get("done") is True:
            completed_episode_ids.add(episode_idx)
        if first_success_steps is None and _is_success(rollout):
            first_success_steps = step_num + 1
            first_success_episode = episode_idx

    return RunStats(
        name=run_dir.name,
        path=str(run_dir),
        complete=True,
        successful=first_success_steps is not None,
        first_success_steps=first_success_steps,
        first_success_episode=first_success_episode,
        episodes=len(completed_episode_ids),
        total_steps=max_step + 1 if max_step >= 0 else 0,
        source="summary.json",
    )


def summarize_run(run_dir: Path) -> RunStats:
    stats = _from_episode_logs(run_dir)
    if stats is not None:
        return stats

    stats = _from_summary(run_dir)
    if stats is not None:
        return stats

    return RunStats(
        name=run_dir.name,
        path=str(run_dir),
        complete=False,
        successful=False,
        first_success_steps=None,
        first_success_episode=None,
        episodes=0,
        total_steps=0,
        source="none",
        note="no summary.json or episode_log.json found",
    )


def find_run_dirs(root: Path) -> list[Path]:
    runs = [p for p in sorted(root.iterdir()) if p.is_dir() and (p / "cmd.txt").exists()]
    if runs:
        return runs
    if (root / "cmd.txt").exists():
        return [root]
    return [p for p in sorted(root.iterdir()) if p.is_dir()]


def _fmt_avg(values: list[int]) -> str:
    if not values:
        return "n/a"
    return f"{statistics.mean(values):.2f}"


def print_text(root: Path, runs: list[RunStats]) -> None:
    complete_runs = [run for run in runs if run.complete]
    successful_runs = [run for run in complete_runs if run.successful]
    first_success_steps = [
        run.first_success_steps
        for run in successful_runs
        if run.first_success_steps is not None
    ]

    print(f"Run directory: {root}")
    print(f"Runs: {len(runs)}")
    print(f"Completed runs with stats: {len(complete_runs)}")
    print(f"Runs with a successful episode: {len(successful_runs)}")
    print(f"Average steps to first success: {_fmt_avg(first_success_steps)}")

    incomplete = [run for run in runs if not run.complete]
    if incomplete:
        print(f"Incomplete/missing stats: {len(incomplete)}")

    print()
    print("Per-run:")
    for run in runs:
        success = "yes" if run.successful else "no"
        first = "n/a" if run.first_success_steps is None else str(run.first_success_steps)
        episode = "n/a" if run.first_success_episode is None else str(run.first_success_episode)
        line = (
            f"- {run.name}: success={success}, first_success_steps={first}, "
            f"first_success_episode={episode}, episodes={run.episodes}, "
            f"total_steps={run.total_steps}, source={run.source}"
        )
        if run.note:
            line += f", note={run.note}"
        print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Timestamp directory produced by launch_g.py")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    root = args.run_dir
    if not root.exists():
        print(f"error: directory does not exist: {root}", file=sys.stderr)
        return 2
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 2

    run_dirs = find_run_dirs(root)
    runs = [summarize_run(run_dir) for run_dir in run_dirs]

    if args.json:
        successful_steps = [
            run.first_success_steps
            for run in runs
            if run.successful and run.first_success_steps is not None
        ]
        payload = {
            "run_dir": str(root),
            "runs": len(runs),
            "completed_runs_with_stats": sum(run.complete for run in runs),
            "successful_runs": sum(run.successful for run in runs),
            "average_steps_to_first_success": (
                statistics.mean(successful_steps) if successful_steps else None
            ),
            "per_run": [asdict(run) for run in runs],
        }
        print(json.dumps(payload, indent=2))
    else:
        print_text(root, runs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
