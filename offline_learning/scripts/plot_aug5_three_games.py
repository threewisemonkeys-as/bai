#!/usr/bin/env python3
"""Latest-and-best multistep results: the aug5_rexpure ships for bt3gb, n2ntd,
s2kt7 (data: logs/multistep_shards_aug5_rexpure.json, identical seed-1 windows
per game). Three metric panels; color = game, line style = arm (learned solid,
raw-frame dashed, random-plan floor dotted).

    uv run python offline_learning/scripts/plot_aug5_three_games.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
LOGS = REPO / "logs"
HS = [1, 2, 4, 8]

COLORS = {"bt3gb": "#2a78d6", "n2ntd": "#e34948", "s2kt7": "#4a3aa7"}
# bt3gb: aug4_mixed run — best multistep FD/planning arm (test 0.780); the
# aug5_rexpure ship wins single-step ID (0.823) but plans far worse.
SOURCES = {"bt3gb": "multistep_shards_aug4mixed.json",
           "n2ntd": "multistep_shards_aug5_rexpure.json",
           "s2kt7": "multistep_shards_aug5_rexpure.json"}
LABELS = {"bt3gb": "bt3gb aug4_mixed (test set-ID 0.780)",
          "n2ntd": "n2ntd aug5_rexpure (0.682)",
          "s2kt7": "s2kt7 aug5_rexpure (0.632)"}

PANELS = [("fd_exact", "Multi-step FD exact"),
          ("fd_partial", "Multi-step FD partial (delta-F1)"),
          ("plan", "Goal-conditioned planning success (in-engine)")]


def value(summary: dict, h: int, panel: str, mode: str):
    s = summary[str(h)]
    if panel == "plan":
        return s["plan"][mode]["success"]
    return s["fd"][mode]["exact" if panel == "fd_exact" else "partial"]


def main():
    sums = {}
    for game, fname in SOURCES.items():
        data = json.loads((LOGS / fname).read_text())
        (res,) = [r for r in data["results"] if r["game"] == game]
        sums[game] = res["summary"]

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2), dpi=110)
    for ax, (panel, title) in zip(axes, PANELS):
        for game, color in COLORS.items():
            s = sums[game]
            ax.plot(HS, [value(s, h, panel, "learned") for h in HS],
                    color=color, marker="o", lw=2.4, markersize=6, zorder=5)
            ax.plot(HS, [value(s, h, panel, "raw") for h in HS],
                    color=color, ls="--", lw=1.3, alpha=0.75)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("horizon h")
        ax.set_xticks(HS)
        ax.set_ylim(-0.03, 1.05)
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("score")

    game_handles = [Line2D([], [], color=c, marker="o", lw=2.4,
                           label=LABELS[g]) for g, c in COLORS.items()]
    style_handles = [
        Line2D([], [], color="#444", marker="o", lw=2.4, label="learned P + beliefs"),
        Line2D([], [], color="#444", ls="--", lw=1.3, label="raw-frame baseline"),
    ]
    axes[0].legend(handles=game_handles, fontsize=9, loc="upper right",
                   framealpha=0.9)
    axes[2].legend(handles=style_handles, fontsize=9, loc="upper right",
                   framealpha=0.9)
    fig.suptitle("Best runs per game (bt3gb: aug4_mixed, best planner; "
                 "n2ntd/s2kt7: aug5_rexpure ships) — multistep FD + planning on "
                 "test50 drives, gpt-oss decode; 10 windows per horizon, "
                 "baselines on identical windows", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = LOGS / "multistep_fd_plan_aug5_bestruns.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
