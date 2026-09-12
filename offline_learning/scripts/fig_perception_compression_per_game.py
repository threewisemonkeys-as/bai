#!/usr/bin/env python3
"""The compression ratios over the search, one panel per game, one figure per ratio.

The summary figure (`fig_perception_metrics.py`, panels B and C) medians these curves across
the 15 games, which is the right way to state the trend and the wrong way to see that the
games disagree. This draws each game on its own axes.

    diversity_bytes        gz([P(X) for all X]) -- output diversity, in bytes
    norm_diversity         gz([P(X) for all X]) / gz([X for all X])
    info_extraction_ratio  mean over frames of gz(P(X)) / gz(X)

and, by name rather than by default, the two uncompressed-numerator companions `dl_ratio`
(bytes P emits per byte of observation) and `pf_dl_gz_ratio` (per COMPRESSED byte, frame by
frame).

The first two are corpus-level, which is not a fair fight: gzipping a game's frames as one
stream lets each side reuse its predecessors, and P's outputs -- nearly the same line every
frame -- dedupe ~7x against the grids' ~3x. Much of a sub-1.0 `norm_diversity` is therefore
P being repetitive rather than P being compact. `info_extraction_ratio` prices each frame on
its own, that discount is gone, and most games land ABOVE 1.0.

Each score gets its **own figure on its own linear scale**, rather than sharing a log axis
or a panel with two y-scales. Both alternatives were tried and are worse here:

  * a shared log axis fits several curves but flattens the shape of each, which is the thing
    worth reading;
  * two y-scales in one panel puts the crossings and the gap between the curves at the mercy
    of an arbitrary alignment -- and one of these is a byte count while the others are
    ratios, which a dual axis would blur rather than show.

Each panel autoscales to its own range, so the curves fill the frame and the axis is
generally not zero-based: read the shape from the line and the level from the ticks. Games
therefore are NOT comparable panel-to-panel by eye -- for that, use the summary figure, whose
axes are shared.

    uv run python offline_learning/scripts/perception_metrics.py          # writes metrics.csv
    uv run python offline_learning/scripts/fig_perception_compression_per_game.py

Writes one figure per score under `analysis/wm_quant/`: by default
`perception_diversity_per_game`, `perception_norm_diversity_per_game` and
`perception_info_extraction_per_game` (`.pdf` and `.png` each). `--metric all` adds
`perception_dl_per_game` and `perception_pf_dl_gz_per_game`; `--metric <name>` draws one.

Each line is the INCUMBENT: at iteration i, the value for whichever node has the best
train_score so far -- the P the run would have shipped had it stopped there. It steps only
when a node beats the running best, so a flat stretch is a node that revised only the world
knowledge K and inherited its parent's P. The ringed marker is where the run actually
stopped, i.e. the artifact the paper's NLWM (Plain) column plans with.
"""
from __future__ import annotations

import argparse
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
    ARM_LABEL, INK, INK2, INK3, SURFACE, load, style, trajectories,
)
from report_planning_v2_online import display_name  # noqa: E402

OUT = REPO / "analysis/wm_quant"
# Categorical slots 1-4 of the validated reference palette (dataviz skill), taken in order.
# Each figure carries a SINGLE series, so no two of these ever share a panel and adjacent-pair
# CVD separation is not the binding constraint; the order is kept anyway so the family reads
# as one, and slots 1 and 2 stay consistent with the summary figure's panels B and C.
SERIES = {
    # the three named scores, in the order they are meant to be read
    "diversity_bytes": ("#2a78d6", "perception_diversity_per_game",
                        "Output diversity — the gzipped size of everything P emits over the "
                        "corpus",
                        "gz(all P(X))   (kB)", "diversity"),
    "norm_diversity": ("#eb6834", "perception_norm_diversity_per_game",
                       "Normalised diversity — the same against what the observations "
                       "themselves cost",
                       "gz(all P(X)) / gz(all X)", "norm_div"),
    "info_extraction_ratio": ("#1baf7a", "perception_info_extraction_per_game",
                              "Information extraction — compressed feature bytes per "
                              "compressed observation byte, frame by frame",
                              "mean gz(P(X)) / gz(X)", "info_extr"),
    # kept, and drawable by name, but not part of the default set
    "dl_ratio": ("#eda100", "perception_dl_per_game",
                 "Bytes P emits per byte of observation",
                 "dl_ratio", "dl"),
    "pf_dl_gz_ratio": ("#e87ba4", "perception_pf_dl_gz_per_game",
                       "Bytes P emits per compressed byte of observation, frame by frame",
                       "mean |P(X)| / |gzip(X)|", "pf_dl"),
}
NAMED = ["diversity_bytes", "norm_diversity", "info_extraction_ratio"]
# What crossing 1.0 means, per ratio. `diversity_bytes` is an absolute count with no such
# line, and `dl_ratio`'s denominator is the padded JSON dump, which nothing beats.
REF_LABEL = {
    "norm_diversity": "1.0 — as costly as the frames, compressed",
    "info_extraction_ratio": "1.0 — as costly as the frame, compressed",
    "pf_dl_gz_ratio": "1.0 — as long as the frame's compressed size",
}
# `diversity_bytes` is drawn in kB; everything else is a ratio and drawn as it is.
SCALE = {"diversity_bytes": 1 / 1000}


def ship_point(rows, game, field):
    """(iteration, value) of the node the run shipped, or None if it has no logged iteration."""
    r = next((r for r in rows if r["game"] == game and r["is_ship"]), None)
    return (r["iteration"], r[field]) if r and r["iteration"] is not None else None


def draw(rows, games, field, split):
    color, stem, subtitle, ylab, _short = SERIES[field]
    curves = trajectories(rows, field, False)

    ncol = 4
    nrow = -(-(len(games) + 1) // ncol)          # +1 leaves the last cell for the key
    fig, axes = plt.subplots(nrow, ncol, figsize=(11.6, 2.3 * nrow), facecolor=SURFACE,
                             sharex=True)
    flat = [ax for row in axes for ax in row]

    for ax, game in zip(flat, games):
        its, inc, _mean = curves[("Plain", game)]
        k = SCALE.get(field, 1.0)
        xs = [i for i, v in zip(its, inc) if v is not None]
        ys = [v * k for v in inc if v is not None]
        ax.plot(xs, ys, color=color, lw=1.9, zorder=3, solid_capstyle="round")
        sp = ship_point(rows, game, field)
        if sp and sp[1] is not None:
            ax.scatter([sp[0]], [sp[1] * k], s=42, facecolor=color, edgecolor=SURFACE,
                       linewidths=2, zorder=5)
        style(ax, display_name(game), "", "")
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        # the 1.0 line only where it is in reach -- otherwise it is either off the axis or
        # squashes the curve to fit a line nobody needs to see
        if field in REF_LABEL:
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
        if field in REF_LABEL:
            handles.append(Line2D([], [], color=INK3, lw=0.9, ls=(0, (2, 2)),
                                  label=REF_LABEL[field]))
        key.legend(handles=handles, loc="upper left", frameon=False, fontsize=8,
                   labelcolor=INK2, handlelength=2.3, labelspacing=0.85, borderaxespad=0.4)
        key.text(0.0, 0.55, "The incumbent is the best node so far, so a flat stretch\n"
                 "is a node that revised only the world knowledge K and\n"
                 "inherited its parent's P.\n\n"
                 "Every panel has its own linear scale and is generally not\n"
                 "zero-based: read the shape from the line, the level from\n"
                 "the ticks. Panels are not comparable to each other by eye.",
                 transform=key.transAxes, va="top", ha="left", color=INK2, fontsize=7.2,
                 linespacing=1.5)

    fig.suptitle(f"{subtitle}, per game  ·  {ARM_LABEL['Plain']}  ·  {split} split",
                 color=INK, fontsize=11, x=0.008, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.972))
    for ext in ("pdf", "png"):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, dpi=200, facecolor=SURFACE)
        print(f"wrote {p}")
    plt.close(fig)
    return curves


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train", choices=("train", "test"))
    ap.add_argument("--metric", default="named",
                    choices=("named", "all", *SERIES),
                    help="'named' (default) draws the three named scores; 'all' adds the "
                         "uncompressed-numerator pair")
    a = ap.parse_args()
    rows = load(a.split)
    games = sorted({r["game"] for r in rows}, key=display_name)
    fields = {"named": NAMED, "all": list(SERIES)}.get(a.metric, [a.metric])
    curves = {f: draw(rows, games, f, a.split) for f in fields}

    for f in fields:
        print(f"\n{SERIES[f][3]}   ({f}, {a.split} split)")
        print(f"{'game':16s} {'seed':>8s} {'ship':>8s} {'min':>8s} {'max':>8s} {'x':>7s}")
        for game in games:
            ys = [v for v in curves[f][("Plain", game)][1] if v is not None]
            ship = ship_point(rows, game, f)
            k = SCALE.get(f, 1.0)
            ys = [v * k for v in ys]
            lo, hi = min(ys), max(ys)
            print(f"{display_name(game):16s} {ys[0]:8.4f} {ship[1] * k:8.4f} "
                  f"{lo:8.4f} {hi:8.4f} {hi / lo if lo else float('inf'):6.1f}x")


if __name__ == "__main__":
    main()
