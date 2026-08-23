#!/usr/bin/env python3
"""Plot average human interactive-phase actions for every BASIS game."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


DATA_MEMBER = (
    "Human and AI baselines data/cleaned_gameplay_data_filtered.json"
)
TASK_SUFFIXES = (
    "_next_frame_prediction",
    "_defect_detection",
    "_planning",
)
ACTION_ORDER = (
    "click",
    "left",
    "right",
    "up",
    "down",
    "reset",
    "start_task",
    "button_click",
    "noop",
)
COLORS = {
    "click": "#F59E0B",
    "left": "#2563EB",
    "right": "#38BDF8",
    "up": "#22C55E",
    "down": "#0F766E",
    "reset": "#EF4444",
    "start_task": "#8B5CF6",
    "button_click": "#EC4899",
    "noop": "#CBD5E1",
}


def game_from_task_id(task_id: str) -> str:
    for suffix in TASK_SUFFIXES:
        if task_id.endswith(suffix):
            return task_id[: -len(suffix)]
    raise ValueError(f"Unrecognized task ID: {task_id}")


def load_averages(archive_path: Path) -> tuple[dict[str, dict[str, float]], Counter]:
    totals: dict[str, Counter] = defaultdict(Counter)
    trajectory_counts: Counter = Counter()

    with ZipFile(archive_path) as archive:
        with archive.open(DATA_MEMBER) as raw_data:
            data = json.load(raw_data)

    for user in data["users"]:
        for task_id, trajectory in user["tasks"].items():
            game = game_from_task_id(task_id)
            trajectory_counts[game] += 1
            totals[game].update(
                event["actionType"] for event in trajectory["interactive_phase"]
            )

    discovered_actions = set().union(*(counts.keys() for counts in totals.values()))
    unexpected_actions = discovered_actions - set(ACTION_ORDER)
    if unexpected_actions:
        raise ValueError(f"Unexpected interactive actions: {sorted(unexpected_actions)}")

    averages = {
        game: {
            action: totals[game][action] / trajectory_counts[game]
            for action in ACTION_ORDER
        }
        for game in totals
    }
    return averages, trajectory_counts


def write_csv(
    output_path: Path,
    averages: dict[str, dict[str, float]],
    trajectory_counts: Counter,
) -> None:
    rows = sorted(
        averages,
        key=lambda game: sum(averages[game].values()),
        reverse=True,
    )
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            ["game", "trajectories", "average_total_steps", *ACTION_ORDER]
        )
        for game in rows:
            values = [averages[game][action] for action in ACTION_ORDER]
            writer.writerow(
                [
                    game,
                    trajectory_counts[game],
                    f"{sum(values):.6f}",
                    *(f"{value:.6f}" for value in values),
                ]
            )


def plot(
    output_stem: Path,
    averages: dict[str, dict[str, float]],
    trajectory_counts: Counter,
) -> None:
    games = sorted(
        averages,
        key=lambda game: sum(averages[game].values()),
        reverse=True,
    )
    totals = [sum(averages[game].values()) for game in games]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(16, 20))
    left = [0.0] * len(games)

    for action in ACTION_ORDER:
        values = [averages[game][action] for game in games]
        ax.barh(
            games,
            values,
            left=left,
            height=0.72,
            color=COLORS[action],
            edgecolor="white",
            linewidth=0.25,
            label=action,
        )
        left = [current + value for current, value in zip(left, values)]

    ax.invert_yaxis()
    ax.set_xlabel(
        "Average logged events per trajectory (interactive phase)",
        fontsize=13,
        labelpad=12,
    )
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=10)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.8)
    ax.grid(axis="y", visible=False)

    label_offset = max(totals) * 0.009
    ax.set_xlim(0, max(totals) * 1.085)
    for index, total in enumerate(totals):
        ax.text(
            total + label_offset,
            index,
            f"{total:.1f}",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#334155",
        )

    fig.suptitle(
        "Human interaction steps by game",
        x=0.115,
        y=0.986,
        ha="left",
        fontsize=22,
        fontweight="bold",
        color="#0F172A",
    )
    fig.text(
        0.115,
        0.963,
        "Average action counts across each game's interactive-phase trajectories",
        ha="left",
        fontsize=12.5,
        color="#475569",
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.11, 0.948),
        ncol=9,
        frameon=False,
        fontsize=10,
        handlelength=1.5,
        columnspacing=1.2,
    )
    n_per_game = sorted(set(trajectory_counts.values()))
    sample_note = (
        str(n_per_game[0]) if len(n_per_game) == 1 else "/".join(map(str, n_per_game))
    )
    fig.text(
        0.115,
        0.012,
        f"Each game has {sample_note} human trajectories (20 per task variant). "
        "noop denotes logged idle/frame ticks; totals are shown at bar ends.",
        ha="left",
        fontsize=10,
        color="#64748B",
    )

    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.19, right=0.965, top=0.925, bottom=0.045)
    fig.savefig(output_stem.with_suffix(".png"), dpi=200, facecolor="white")
    fig.savefig(output_stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("basis_data.zip"),
        help="Path to basis_data.zip",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path("viz/human_interaction_steps_by_game"),
        help="Output path without an extension",
    )
    args = parser.parse_args()

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    averages, trajectory_counts = load_averages(args.archive)
    write_csv(
        args.output_stem.with_suffix(".csv"), averages, trajectory_counts
    )
    plot(args.output_stem, averages, trajectory_counts)


if __name__ == "__main__":
    main()
