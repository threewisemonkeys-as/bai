#!/usr/bin/env python3
"""The per-game panel grid: one metric, one small-multiple figure, 15 games plus a key.

Shared by the two families that use it -- `fig_perception_compression_per_game.py` (the
abstraction program P) and `fig_dynamics_size_per_game.py` (the dynamics model K) -- so that
a change to the layout reaches both. The metric-specific parts (which column, its colour,
title, axis label, unit scale, reference line and the paragraph in the key) are arguments;
everything about the grid is here.

Each panel autoscales to its own range, so the curves fill the frame and the axis is
generally not zero-based: read the shape from the line and the level from the ticks. Games
therefore are NOT comparable panel-to-panel by eye -- for that, use a figure whose axes are
shared.

Each line is the INCUMBENT: at iteration i, the value for whichever node has the best
train_score so far -- what the run would have shipped had it stopped there. It steps only
when a node beats the running best, so a flat stretch is a node that left this parameter
alone and changed the other one. The ringed marker is where the run actually stopped, i.e.
the artifact the paper's NLWM (Plain) column plans with.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fig_perception_metrics import (  # noqa: E402
    ARM_LABEL, INK, INK2, INK3, SURFACE, style, trajectories,
)
from report_planning_v2_online import display_name  # noqa: E402

OUT = REPO / "analysis/wm_quant"
# True of the layout rather than of any one metric, so it is appended to every key.
LAYOUT_NOTE = ("Every panel has its own linear scale and is generally not\n"
               "zero-based: read the shape from the line, the level from\n"
               "the ticks. Panels are not comparable to each other by eye.")


def ship_point(rows, game, field):
    """(iteration, value) of the node the run shipped, or None if it has no logged iteration."""
    r = next((r for r in rows if r["game"] == game and r["is_ship"]), None)
    return (r["iteration"], r[field]) if r and r["iteration"] is not None else None


def draw_grid(rows, games, field, split, *, color, stem, subtitle, ylab, note,
              scale=1.0, ref=None, arm="Plain"):
    """One figure: `field` over the search, a panel per game. Returns the curve table.

    `ref` is the label for a dashed line at 1.0, drawn only where the axis comes near it --
    off-axis it either disappears or squashes the curve to fit a line nobody needs to see.
    """
    curves = trajectories(rows, field, False)

    ncol = 4
    nrow = -(-(len(games) + 1) // ncol)          # +1 leaves the last cell for the key
    fig, axes = plt.subplots(nrow, ncol, figsize=(11.6, 2.3 * nrow), facecolor=SURFACE,
                             sharex=True)
    flat = [ax for row in axes for ax in row]

    for ax, game in zip(flat, games):
        its, inc, _mean = curves[(arm, game)]
        xs = [i for i, v in zip(its, inc) if v is not None]
        ys = [v * scale for v in inc if v is not None]
        ax.plot(xs, ys, color=color, lw=1.9, zorder=3, solid_capstyle="round")
        sp = ship_point(rows, game, field)
        if sp and sp[1] is not None:
            ax.scatter([sp[0]], [sp[1] * scale], s=42, facecolor=color, edgecolor=SURFACE,
                       linewidths=2, zorder=5)
        style(ax, display_name(game), "", "")
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        if ref:
            lo, hi = ax.get_ylim()
            if hi > 0.85 and lo < 1.04:
                ax.axhline(1.0, color=INK3, lw=0.8, ls=(0, (2, 2)), zorder=1)
                ax.set_ylim(lo, max(hi, 1.04))

    for ax in flat[len(games):]:
        ax.axis("off")
    key = flat[len(games)] if len(games) < len(flat) else None

    for ax in axes[-1]:
        if ax.axison:
            ax.set_xlabel("search iteration", color=INK2, fontsize=8)
    for row in axes:
        row[0].set_ylabel(ylab, color=INK2, fontsize=8)
    if key is not None and len(games) >= ncol:      # the panel above the key has no neighbour
        flat[len(games) - ncol].set_xlabel("search iteration", color=INK2, fontsize=8)
        flat[len(games) - ncol].tick_params(labelbottom=True)

    if key is not None:
        handles = [Line2D([], [], color=color, lw=2.2, label=f"{ylab} — incumbent"),
                   Line2D([], [], color=color, lw=0, marker="o", markersize=7,
                          markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.6,
                          label="the node that shipped")]
        if ref:
            handles.append(Line2D([], [], color=INK3, lw=0.9, ls=(0, (2, 2)), label=ref))
        key.legend(handles=handles, loc="upper left", frameon=False, fontsize=8,
                   labelcolor=INK2, handlelength=2.3, labelspacing=0.85, borderaxespad=0.4)
        key.text(0.0, 0.55, f"{note}\n\n{LAYOUT_NOTE}", transform=key.transAxes, va="top",
                 ha="left", color=INK2, fontsize=7.2, linespacing=1.5)

    fig.suptitle(f"{subtitle}, per game  ·  {ARM_LABEL[arm]}  ·  {split} split",
                 color=INK, fontsize=11, x=0.008, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.972))
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, dpi=200, facecolor=SURFACE)
        print(f"wrote {p}")
    plt.close(fig)
    return curves


def summarise(curves, rows, games, fields, spec, split):
    """The table behind each figure: seed, ship, range and span of the incumbent per game.

    `span` is max over the smallest POSITIVE value, not over the minimum: the seed scores 0
    on several of these measures -- an empty K, a P that emits nothing -- and a ratio to 0
    says only that, where the span among the values that exist is worth reading.
    """
    for f in fields:
        print(f"\n{spec[f]['ylab']}   ({f}, {split} split)")
        print(f"{'game':16s} {'seed':>8s} {'ship':>8s} {'min':>8s} {'max':>8s} {'span':>7s}")
        for game in games:
            k = spec[f].get("scale", 1.0)
            ys = [v * k for v in curves[f][("Plain", game)][1] if v is not None]
            ship = ship_point(rows, game, f)
            pos = [v for v in ys if v > 0]
            span = f"{max(ys) / min(pos):5.1f}x" if pos else "     --"
            print(f"{display_name(game):16s} {ys[0]:8.4f} {ship[1] * k:8.4f} "
                  f"{min(ys):8.4f} {max(ys):8.4f} {span:>7s}")
