#!/usr/bin/env python3
"""Budget-decoupled (cap-10) planning results for s2kt7/bt3gb/n2ntd vs the
h-capped runs, dsflash decode, ever-hit criterion on identical seed-0 windows.

Everything is read from the paired ONLINE eval jsons (they carry both the
closed-loop success and the offline ever-hit rescore of the same windows):

  cap-10 (fixed 10-step budget at every horizon; h only sets how far ahead the
  goal was recorded):
    s2kt7  logs/online_plan_s2kt7_honest24_dsflash_cap10.json  (honest prog24)
    bt3gb  logs/online_plan_aug4mixed_dsflash_bt3gb_cap10.json (aug4_mixed)
    n2ntd  logs/online_plan_aug5_rexpure_dsflash_n2ntd_cap10.json (aug5_rexpure)
  h-capped (budget == h, the historical protocol):
    s2kt7  logs/online_plan_s2kt7_honest24_dsflash.json
    bt3gb  logs/online_plan_aug4mixed_dsflash_bt3gb.json
    n2ntd  logs/online_plan_aug5_dsflash_n2ntd_s2kt7.json

Figure 1  cap10_planning_3games.png   panel per game, ever-hit success vs h;
          color = arm (learned blue / raw gray), style = loop (closed solid,
          open dashed).
Figure 2  cap10_budget_effect.png     panel per game, grouped bars of the
          per-cell delta (cap-10 minus h-capped) for the same four series.

    uv run python offline_learning/scripts/plot_cap10_budget_effect.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
LOGS = REPO / "logs"
OUT = REPO / "offline_learning" / "results"
HS = [1, 2, 4, 8]

BLUE, GRAY, INK = "#2a78d6", "#8a8a86", "#0b0b0b"

GAMES = ["s2kt7", "bt3gb", "n2ntd"]
SOURCES = {
    "cap10": {"s2kt7": "online_plan_s2kt7_honest24_dsflash_cap10.json",
              "bt3gb": "online_plan_aug4mixed_dsflash_bt3gb_cap10.json",
              "n2ntd": "online_plan_aug5_rexpure_dsflash_n2ntd_cap10.json"},
    "hcap": {"s2kt7": "online_plan_s2kt7_honest24_dsflash.json",
             "bt3gb": "online_plan_aug4mixed_dsflash_bt3gb.json",
             "n2ntd": "online_plan_aug5_dsflash_n2ntd_s2kt7.json"},
}
TITLES = {"s2kt7": "s2kt7 — honest prog24 beliefs",
          "bt3gb": "bt3gb — aug4_mixed ship",
          "n2ntd": "n2ntd — aug5_rexpure ship"}
# (mode, loop-key) -> label / color / linestyle; loop "success" = closed-loop
SERIES = [("learned", "success", "learned, closed-loop", BLUE, "-", 2.4),
          ("learned", "offline_everhit", "learned, open-loop", BLUE, "--", 1.4),
          ("raw", "success", "raw, closed-loop", GRAY, "-", 2.4),
          ("raw", "offline_everhit", "raw, open-loop", GRAY, "--", 1.4)]


def load(kind: str, game: str) -> dict:
    data = json.loads((LOGS / SOURCES[kind][game]).read_text())
    (res,) = [r for r in data["results"] if r["game"] == game]
    return {(m, k): [res["summary"][str(h)]["plan"][m][k] for h in HS]
            for m, k, *_ in SERIES}


def curves_figure(vals: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=110, sharey=True)
    for ax, game in zip(axes, GAMES):
        # raw first, learned on top with a white marker ring, so coinciding
        # lines (s2kt7 closed-loop: both arms at 1.00) stay distinguishable
        for m, k, label, color, ls, lw in reversed(SERIES):
            ax.plot(HS, vals[game][(m, k)], color=color, ls=ls, lw=lw,
                    marker="o", markersize=6 if ls == "-" else 4.5,
                    markeredgecolor="white", markeredgewidth=0.9,
                    zorder=(6 if color == BLUE else 5) if ls == "-" else 4)
        ax.set_title(TITLES[game], fontsize=11.5)
        ax.set_xlabel("horizon h (goal recorded h steps ahead)")
        ax.set_xticks(HS)
        ax.set_ylim(-0.03, 1.05)
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("planning success (ever-hit)")
    axes[0].annotate("closed-loop: BOTH arms at 1.00", xy=(4, 1.0),
                     xytext=(2.05, 0.86), fontsize=8.5, color=INK,
                     arrowprops=dict(arrowstyle="-", color=INK, lw=0.7))
    handles = [Line2D([], [], color=c, ls=ls, lw=lw, marker="o", label=lab)
               for _, _, lab, c, ls, lw in SERIES]
    axes[2].legend(handles=handles, fontsize=9, loc="upper right", framealpha=0.9)
    fig.suptitle("Planning with a fixed 10-step budget at every horizon "
                 "(h only sets how far ahead the goal was recorded) — "
                 "dsflash decode, test50 seed-0 windows, ever-hit criterion",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = OUT / "cap10_planning_3games.png"
    fig.savefig(out)
    print(f"wrote {out}")


def delta_figure(vals: dict, prev: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=110, sharey=True)
    width, x = 0.19, np.arange(len(HS))
    for ax, game in zip(axes, GAMES):
        for i, (m, k, label, color, ls, _lw) in enumerate(SERIES):
            delta = [a - b for a, b in zip(vals[game][(m, k)], prev[game][(m, k)])]
            ax.bar(x + (i - 1.5) * width, delta, width, color=color,
                   alpha=1.0 if ls == "-" else 0.45,
                   edgecolor="white", linewidth=0.8)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_title(TITLES[game], fontsize=11.5)
        ax.set_xlabel("horizon h")
        ax.set_xticks(x, [str(h) for h in HS])
        ax.grid(alpha=0.25, axis="y")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Δ ever-hit (cap-10 − h-capped)")
    handles = [Patch(facecolor=c, alpha=1.0 if ls == "-" else 0.45, label=lab)
               for _, _, lab, c, ls, _ in SERIES]
    fig.legend(handles=handles, fontsize=9, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 0.93), frameon=False)
    fig.suptitle("Effect of decoupling the budget from the horizon "
                 "(cap-10 − h-capped, identical windows)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    out = OUT / "cap10_budget_effect.png"
    fig.savefig(out)
    print(f"wrote {out}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    vals = {g: load("cap10", g) for g in GAMES}
    prev = {g: load("hcap", g) for g in GAMES}
    curves_figure(vals)
    delta_figure(vals, prev)


if __name__ == "__main__":
    main()
