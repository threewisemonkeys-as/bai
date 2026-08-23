#!/usr/bin/env python3
"""Create a non-pooled summary dashboard for a coverage-plan evaluation.

    uv run python offline_learning/scripts/plot_coverage_plan_summary.py \
        --input logs/coverage_plan_eval_claude.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO / "logs/coverage_plan_eval_claude.json"
DEFAULT_OUTPUT = REPO / "logs/coverage_plan_eval_claude_summary"

BUCKETS = ("act", "wait", "maintain")
HORIZONS = (1, 2, 4, 8, 12)
GAMES = ("bt3gb", "dq8gc", "n2ntd", "s2kt7", "83wkq")
GAME_LABELS = {
    "bt3gb": "ice",
    "dq8gc": "disease",
    "n2ntd": "mario",
    "s2kt7": "ants",
    "83wkq": "particles",
}
METRICS = ("raw", "lmwm", "wc", "noop_success", "random_success")
ARM_METRICS = ("raw", "lmwm", "wc")
LABELS = {
    "raw": "Raw Sonnet",
    "lmwm": "LMWM + Sonnet",
    "wc": "WorldCoder",
    "noop_success": "Always noop",
    "random_success": "Random",
}
COLORS = {
    "raw": "#6f7782",
    "lmwm": "#1479b8",
    "wc": "#d44a3a",
    "noop_success": "#a17b31",
    "random_success": "#b8bcc2",
}


def load_rows(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text())
    rows = [row for result in payload.get("results", []) for row in result.get("rows", [])]
    if not rows:
        raise ValueError(f"no evaluation rows found in {path}")
    return payload, rows


def success_rate(rows: list[dict], metric: str) -> float:
    if not rows:
        return float("nan")
    if metric in ARM_METRICS:
        values = [float(row[metric]["success"]) for row in rows]
    else:
        values = [float(row[metric]) for row in rows]
    return sum(values) / len(values)


def style_axis(ax: plt.Axes, *, grid: str = "y") -> None:
    ax.grid(axis=grid, color="#dfe3e8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#a8adb4")
    ax.spines["bottom"].set_color("#a8adb4")


def add_percentage(ax: plt.Axes, x: float, value: float, *, size: int = 7) -> None:
    if value != value:
        return
    ax.text(x, value + 0.025, f"{value:.0%}", ha="center", va="bottom", fontsize=size,
            color="#333333")


def plot_summary(payload: dict, rows: list[dict], output: Path) -> None:
    fig = plt.figure(figsize=(15.5, 10.0), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.12), hspace=0.38, wspace=0.25)
    bucket_ax = fig.add_subplot(grid[0, :])
    horizon_ax = fig.add_subplot(grid[1, 0])
    game_ax = fig.add_subplot(grid[1, 1])

    # Bucket-separated headline rates; no act/wait/maintain pooling.
    bucket_x = np.arange(len(BUCKETS), dtype=float)
    width = 0.15
    offsets = np.linspace(-2, 2, len(METRICS)) * width
    for metric, offset in zip(METRICS, offsets):
        values = [success_rate([r for r in rows if r["bucket"] == bucket], metric)
                  for bucket in BUCKETS]
        bars = bucket_ax.bar(bucket_x + offset, values, width=width * 0.92,
                             color=COLORS[metric], label=LABELS[metric])
        for bar, value in zip(bars, values):
            if value >= 0.075:
                add_percentage(bucket_ax, bar.get_x() + bar.get_width() / 2, value)
    bucket_counts = [sum(r["bucket"] == bucket for r in rows) for bucket in BUCKETS]
    bucket_ax.set_xticks(bucket_x, [f"{bucket}\n(n={count})"
                                   for bucket, count in zip(BUCKETS, bucket_counts)])
    bucket_ax.set_ylim(0, 1.12)
    bucket_ax.set_ylabel("Success rate")
    bucket_ax.set_title("A. Success by problem bucket", loc="left", fontweight="bold")
    bucket_ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    style_axis(bucket_ax)

    # Act-only horizon curves.
    act_rows = [row for row in rows if row["bucket"] == "act"]
    horizon_x = np.arange(len(HORIZONS), dtype=float)
    markers = {"raw": "^", "lmwm": "o", "wc": "s"}
    for metric in ARM_METRICS:
        values = [success_rate([r for r in act_rows if r["h"] == horizon], metric)
                  for horizon in HORIZONS]
        horizon_ax.plot(horizon_x, values, color=COLORS[metric], marker=markers[metric],
                        linewidth=2.5 if metric != "raw" else 1.8,
                        markersize=6, label=LABELS[metric])
        for x, value in zip(horizon_x, values):
            if value >= 0.045:
                horizon_ax.text(x, value + 0.025, f"{value:.0%}", color=COLORS[metric],
                                ha="center", va="bottom", fontsize=7)
    horizon_counts = [sum(r["h"] == horizon for r in act_rows) for horizon in HORIZONS]
    horizon_ax.set_xticks(horizon_x, [f"{h}\n(n={n})"
                                     for h, n in zip(HORIZONS, horizon_counts)])
    horizon_ax.set_ylim(0, 0.55)
    horizon_ax.set_xlabel("Horizon (steps)")
    horizon_ax.set_ylabel("Success rate")
    horizon_ax.set_title("B. Act problems by horizon", loc="left", fontweight="bold")
    horizon_ax.legend(frameon=False, loc="upper right")
    style_axis(horizon_ax)

    # Act-only per-game rates.
    game_x = np.arange(len(GAMES), dtype=float)
    game_width = 0.24
    for index, metric in enumerate(ARM_METRICS):
        values = [success_rate([r for r in act_rows if r["game"] == game], metric)
                  for game in GAMES]
        bars = game_ax.bar(game_x + (index - 1) * game_width, values,
                           width=game_width * 0.9, color=COLORS[metric], label=LABELS[metric])
        for bar, value in zip(bars, values):
            if value >= 0.035:
                add_percentage(game_ax, bar.get_x() + bar.get_width() / 2, value)
    game_counts = [sum(r["game"] == game for r in act_rows) for game in GAMES]
    game_ax.set_xticks(game_x, [f"{GAME_LABELS[g]}\n(n={n})"
                               for g, n in zip(GAMES, game_counts)])
    game_ax.set_ylim(0, 0.65)
    game_ax.set_ylabel("Success rate")
    game_ax.set_title("C. Act problems by game", loc="left", fontweight="bold")
    game_ax.legend(frameon=False, loc="upper right")
    style_axis(game_ax)

    config = payload.get("config", {})
    model = config.get("model", "unknown")
    plan_cap = config.get("plan_cap", "?")
    fig.suptitle("Coverage-anchored planning evaluation", fontsize=18,
                 fontweight="bold", y=0.985)
    fig.text(0.5, 0.95,
             f"Claude {model} planner · {len(rows)} curated problems · plan cap {plan_cap} · "
             "act/wait/maintain never pooled",
             ha="center", color="#555b63", fontsize=10.5)
    fig.text(0.5, 0.018,
             "Raw and LMWM use the same Sonnet planner; WorldCoder is deterministic program search.",
             ha="center", color="#666666", fontsize=9)
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.07, right=0.98)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output.with_suffix('.png')}")
    print(f"wrote {output.with_suffix('.svg')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="output stem; .png and .svg are written")
    args = parser.parse_args()
    payload, rows = load_rows(args.input)
    plot_summary(payload, rows, args.output)


if __name__ == "__main__":
    main()
