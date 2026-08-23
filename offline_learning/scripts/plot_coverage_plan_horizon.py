#!/usr/bin/env python3
"""Plot WC and NLWM planning performance against horizon.

The coverage-anchored evaluation deliberately separates act, wait, and maintain
problems.  This script preserves that separation in the headline figure and uses
only the act bucket for the per-game figure.

    uv run python offline_learning/scripts/plot_coverage_plan_horizon.py
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO / "logs/coverage_plan_eval.json"
DEFAULT_OUT_DIR = REPO / "logs"

HORIZONS = [1, 2, 4, 8, 12]
BUCKETS = ["act", "wait", "maintain"]
GAMES = ["bt3gb", "dq8gc", "n2ntd", "s2kt7", "83wkq"]
GAME_LABELS = {
    "bt3gb": "bt3gb (ice)",
    "dq8gc": "dq8gc (disease)",
    "n2ntd": "n2ntd (mario)",
    "s2kt7": "s2kt7 (ants)",
    "83wkq": "83wkq (particles)",
}

COLORS = {
    "lmwm": "#1677b8",
    "wc": "#d1493f",
    "raw": "#686868",
    "noop_success": "#9c7a32",
    "random_success": "#b9b9b9",
}
LABELS = {
    "lmwm": "NLWM",
    "wc": "WC",
    "raw": "Raw LLM",
    "noop_success": "Always noop",
    "random_success": "Random",
}


def load_rows(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text())
    if "results" in payload:  # offline eval: {results: [{rows: [...]}]}
        rows = [row for result in payload["results"] for row in result.get("rows", [])]
    else:                     # online eval: flat {rows: [...]}
        rows = payload.get("rows", [])
    if not rows:
        raise ValueError(f"no evaluation rows found in {path}")
    missing_arms = {arm for arm in ("lmwm", "wc")
                    if any(row.get(arm) is None for row in rows)}
    if missing_arms:
        raise ValueError(f"rows in {path} are missing arms: {sorted(missing_arms)}")
    return payload, rows


def mean(rows: list[dict], metric: str) -> float | None:
    if not rows:
        return None
    if metric in ("lmwm", "wc", "raw"):
        values = [float(row[metric]["success"]) for row in rows]
    else:
        values = [float(row[metric]) for row in rows]
    return sum(values) / len(values)


def series(rows: list[dict], metric: str) -> tuple[list[float], list[int]]:
    values, counts = [], []
    for horizon in HORIZONS:
        cell = [row for row in rows if row["h"] == horizon]
        value = mean(cell, metric)
        values.append(float("nan") if value is None else value)
        counts.append(len(cell))
    return values, counts


def style_axis(ax: plt.Axes) -> None:
    ax.set_xticks(range(len(HORIZONS)), [str(h) for h in HORIZONS])
    ax.set_ylim(-0.03, 1.04)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", color="#dedede", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#a0a0a0")
    ax.spines["bottom"].set_color("#a0a0a0")


def save_both(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    print(f"wrote {stem.with_suffix('.png')}")
    print(f"wrote {stem.with_suffix('.svg')}")


def plot_by_bucket(rows: list[dict], payload: dict, out_dir: Path,
                   prefix: str = "coverage_plan", note: str = "") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), sharey=True)
    xpos = list(range(len(HORIZONS)))
    line_specs = [
        ("lmwm", "-", "o", 2.8, 7),
        ("wc", "-", "s", 2.8, 6),
        ("raw", "--", "^", 1.8, 5),
        ("noop_success", ":", None, 1.7, 0),
        ("random_success", ":", None, 1.7, 0),
    ]

    for ax, bucket in zip(axes, BUCKETS):
        bucket_rows = [row for row in rows if row["bucket"] == bucket]
        for metric, linestyle, marker, width, marker_size in line_specs:
            values, _ = series(bucket_rows, metric)
            ax.plot(
                xpos,
                values,
                color=COLORS[metric],
                linestyle=linestyle,
                marker=marker,
                linewidth=width,
                markersize=marker_size,
                label=LABELS[metric],
                zorder=4 if metric in ("lmwm", "wc") else 2,
            )
            if metric in ("lmwm", "wc"):
                for x, value in zip(xpos, values):
                    if value == value:  # NaN-safe
                        offset = 0.035 if metric == "lmwm" else -0.055
                        ax.text(
                            x,
                            min(1.02, max(0.01, value + offset)),
                            f"{value:.0%}",
                            color=COLORS[metric],
                            fontsize=8,
                            ha="center",
                            va="center",
                            fontweight="semibold",
                        )

        _, counts = series(bucket_rows, "lmwm")
        for x, count in zip(xpos, counts):
            if count:
                ax.text(x, -0.015, f"n={count}", color="#777777", fontsize=7,
                        ha="center", va="bottom")
        ax.set_title(f"{bucket.capitalize()} problems (n={len(bucket_rows)})",
                     fontsize=12, fontweight="semibold")
        ax.set_xlabel("Horizon (steps)")
        style_axis(ax)

    axes[0].set_ylabel("Planning success rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.965))
    config = payload.get("config", {})
    model = config.get("model", "unknown planner")
    fig.suptitle(
        "Coverage-anchored planning performance vs horizon",
        fontsize=15,
        fontweight="bold",
        y=1.07,
    )
    fig.text(
        0.5,
        0.995,
        f"WC program search vs NLWM ({model}); buckets shown separately",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=2.0)
    save_both(fig, out_dir / f"{prefix}_performance_vs_horizon")
    plt.close(fig)


def plot_act_by_game(rows: list[dict], out_dir: Path,
                     prefix: str = "coverage_plan", note: str = "") -> None:
    act_rows = [row for row in rows if row["bucket"] == "act"]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0), sharex=True, sharey=True)
    xpos = list(range(len(HORIZONS)))

    for ax, game in zip(axes.flat, GAMES):
        game_rows = [row for row in act_rows if row["game"] == game]
        for metric, linestyle, marker, width in (
            ("lmwm", "-", "o", 2.6),
            ("wc", "-", "s", 2.6),
            ("raw", "--", "^", 1.5),
            ("random_success", ":", None, 1.5),
        ):
            values, _ = series(game_rows, metric)
            ax.plot(xpos, values, color=COLORS[metric], linestyle=linestyle,
                    marker=marker, linewidth=width, markersize=5,
                    label=LABELS[metric])
        ax.set_title(f"{GAME_LABELS[game]}  ·  n={len(game_rows)}",
                     fontsize=11, fontweight="semibold")
        style_axis(ax)

    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="center", frameon=False, fontsize=11)
    legend_ax.text(
        0.5,
        0.23,
        "Act bucket only\nCell sizes vary by game and horizon",
        ha="center",
        va="center",
        color="#666666",
        fontsize=10,
        transform=legend_ax.transAxes,
    )
    for ax in axes[-1, :2]:
        ax.set_xlabel("Horizon (steps)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Planning success rate")
    title = "Act-problem planning performance vs horizon, by game"
    if note:
        title += f"  ({note})"
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.0, w_pad=1.4)
    save_both(fig, out_dir / f"{prefix}_act_per_game_vs_horizon")
    plt.close(fig)


def plot_combined(rows: list[dict], payload: dict, out_dir: Path,
                  prefix: str = "coverage_plan", note: str = "") -> None:
    """Combine act/wait/maintain into one per-problem overall average."""
    fig, ax = plt.subplots(figsize=(7.6, 5.1))
    xpos = list(range(len(HORIZONS)))
    line_specs = [
        ("lmwm", "-", "o", 2.8, 7),
        ("wc", "-", "s", 2.8, 6),
        ("raw", "--", "^", 1.8, 5),
        ("noop_success", ":", None, 1.7, 0),
        ("random_success", ":", None, 1.7, 0),
    ]

    for metric, linestyle, marker, width, marker_size in line_specs:
        values, _ = series(rows, metric)
        ax.plot(
            xpos,
            values,
            color=COLORS[metric],
            linestyle=linestyle,
            marker=marker,
            linewidth=width,
            markersize=marker_size,
            label=LABELS[metric],
            zorder=4 if metric in ("lmwm", "wc") else 2,
        )
        if metric in ("lmwm", "wc"):
            for x, value in zip(xpos, values):
                offset = 0.035 if metric == "lmwm" else -0.055
                ax.text(
                    x,
                    min(1.02, max(0.01, value + offset)),
                    f"{value:.0%}",
                    color=COLORS[metric],
                    fontsize=8,
                    ha="center",
                    va="center",
                    fontweight="semibold",
                )

    _, counts = series(rows, "lmwm")
    for x, count in zip(xpos, counts):
        ax.text(x, -0.015, f"n={count}", color="#777777", fontsize=7,
                ha="center", va="bottom")
    ax.set_xlabel("Horizon (steps)")
    ax.set_ylabel("Planning success rate")
    style_axis(ax)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.955))
    model = payload.get("config", {}).get("model", "unknown planner")
    fig.suptitle("Combined planning performance vs horizon",
                 fontsize=15, fontweight="bold", y=1.07)
    subtitle = f"Act + wait + maintain · WC program search vs NLWM ({model})"
    if note:
        subtitle += f" · {note}"
    fig.text(0.5, 0.99, subtitle, ha="center", fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_both(fig, out_dir / f"{prefix}_combined_performance_vs_horizon")
    plt.close(fig)


def plot_combined_by_game(rows: list[dict], out_dir: Path,
                          prefix: str = "coverage_plan", note: str = "") -> None:
    """Plot the pooled act/wait/maintain average separately for each game."""
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0), sharex=True, sharey=True)
    xpos = list(range(len(HORIZONS)))
    line_specs = [
        ("lmwm", "-", "o", 2.6, 6),
        ("wc", "-", "s", 2.6, 5),
        ("raw", "--", "^", 1.5, 4),
        ("noop_success", ":", None, 1.4, 0),
        ("random_success", ":", None, 1.4, 0),
    ]

    for ax, game in zip(axes.flat, GAMES):
        game_rows = [row for row in rows if row["game"] == game]
        for metric, linestyle, marker, width, marker_size in line_specs:
            values, _ = series(game_rows, metric)
            ax.plot(
                xpos,
                values,
                color=COLORS[metric],
                linestyle=linestyle,
                marker=marker,
                linewidth=width,
                markersize=marker_size,
                label=LABELS[metric],
                zorder=4 if metric in ("lmwm", "wc") else 2,
            )
        _, counts = series(game_rows, "lmwm")
        for x, count in zip(xpos, counts):
            if count:
                ax.text(x, -0.015, f"n={count}", color="#777777", fontsize=7,
                        ha="center", va="bottom")
        ax.set_title(f"{GAME_LABELS[game]}  ·  n={len(game_rows)}",
                     fontsize=11, fontweight="semibold")
        style_axis(ax)

    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="center", frameon=False, fontsize=11)
    legend_ax.text(
        0.5,
        0.18,
        "Overall average within each game\nAct + wait + maintain",
        ha="center",
        va="center",
        color="#666666",
        fontsize=10,
        transform=legend_ax.transAxes,
    )
    for ax in axes[-1, :2]:
        ax.set_xlabel("Horizon (steps)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Planning success rate")
    title = "Combined planning performance vs horizon, by game"
    if note:
        title += f"  ({note})"
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.0, w_pad=1.4)
    save_both(fig, out_dir / f"{prefix}_combined_per_game_vs_horizon")
    plt.close(fig)


def print_summary(rows: list[dict]) -> None:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["bucket"]].append(row)
    for bucket in BUCKETS:
        print(f"{bucket}:")
        for metric in ("lmwm", "wc"):
            values, counts = series(grouped[bucket], metric)
            cells = ["--" if value != value else f"{value:.3f}" for value in values]
            print(f"  {metric:4s} values={cells} n={counts}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--stem-prefix", default="coverage_plan",
                        help="output filename prefix (use 'coverage_online' for the MPC run)")
    parser.add_argument("--title-note", default="",
                        help="short note appended to titles, e.g. 'online (receding-horizon)'")
    args = parser.parse_args()

    payload, rows = load_rows(args.input)
    prefix, note = args.stem_prefix, args.title_note
    plot_by_bucket(rows, payload, args.out_dir, prefix, note)
    plot_act_by_game(rows, args.out_dir, prefix, note)
    plot_combined(rows, payload, args.out_dir, prefix, note)
    plot_combined_by_game(rows, args.out_dir, prefix, note)
    print_summary(rows)


if __name__ == "__main__":
    main()
