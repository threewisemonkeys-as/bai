#!/usr/bin/env python3
"""How big P is and how much it emits, over the search, one panel per game, one figure each.

The summary figure (`fig_perception_metrics.py`, panels A to C) medians these curves across
the 15 games, which is the right way to state the trend and the wrong way to see that the
games disagree. This draws each game on its own axes.

    ast_nodes              the size of the program itself
    mean_out_chars         mean |P(X)| -- how much text P emits about one frame
    diversity_bytes        gz([P(X) for all X]) -- output diversity, in bytes
    norm_diversity         gz([P(X) for all X]) / gz([X for all X])
    info_extraction_ratio  mean over frames of gz(P(X)) / gz(X)

and, by name rather than by default, the two uncompressed-numerator companions `dl_ratio`
(bytes P emits per byte of observation) and `pf_dl_gz_ratio` (per COMPRESSED byte, frame by
frame).

`ast_nodes` is the size of the module P, counted over its parsed tree rather than its text,
so the comments and blank lines the reflector varies freely do not move it; a node whose
code never parsed has no program size at all and is absent from the curve (`code_chars` and
`sloc` are the by-name companions that do count text). `mean_out_chars` is the plain size of
what P writes, in characters, before any compressor
or denominator gets involved -- and within a game it is rank-identical to `pf_dl_gz_ratio`,
whose denominator does not depend on the node. Read them as one curve in two units.

The two corpus scores are, which is not a fair fight: gzipping a game's frames as one
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

The panel grid itself, and what the incumbent line and the ringed marker mean, are in
`wm_panel_grid.py`; the companion family for the dynamics model K is
`fig_dynamics_size_per_game.py`.

    uv run python offline_learning/scripts/perception_metrics.py          # writes metrics.csv
    uv run python offline_learning/scripts/fig_perception_compression_per_game.py

Writes one figure per score under `analysis/wm_quant/`: by default
`perception_program_size_per_game`, `perception_output_chars_per_game`,
`perception_diversity_per_game`, `perception_norm_diversity_per_game` and
`perception_info_extraction_per_game` (`.pdf` and `.png` each). `--metric all` adds the
`dl`, `pf_dl_gz`, `code_chars` and `sloc` companions; `--metric <name>` draws one.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fig_perception_metrics import load  # noqa: E402
from report_planning_v2_online import display_name  # noqa: E402
from wm_panel_grid import draw_grid, summarise  # noqa: E402

# Categorical slots of the validated reference palette (dataviz skill), taken in order
# (green is slot 6; violet, slot 7, belongs to the dynamics-model family; red is slot 8, and
# the three code-text measures of the same thing share it).
# Each figure carries a SINGLE series, so no two of these ever share a panel and adjacent-pair
# CVD separation is not the binding constraint; the order is kept anyway so the family reads
# as one, and slots 1 and 2 stay consistent with the summary figure's panels B and C.
# `ref` is the meaning of the 1.0 line: `diversity_bytes` is an absolute count with no such
# line, and `dl_ratio`'s denominator is the padded JSON dump, which nothing beats.
SERIES = {
    # the named scores, in the order they are meant to be read
    "ast_nodes": dict(
        color="#e34948", stem="perception_program_size_per_game",
        subtitle="Program size — the parsed tree of P, not its text",
        ylab="AST nodes"),
    "mean_out_chars": dict(
        color="#008300", stem="perception_output_chars_per_game",
        subtitle="Output size — the characters P emits about one frame",
        ylab="mean |P(X)|   (chars)"),
    "diversity_bytes": dict(
        color="#2a78d6", stem="perception_diversity_per_game",
        subtitle="Output diversity — the gzipped size of everything P emits over the corpus",
        ylab="gz(all P(X))   (kB)", scale=1 / 1000),
    "norm_diversity": dict(
        color="#eb6834", stem="perception_norm_diversity_per_game",
        subtitle="Normalised diversity — the same against what the observations themselves "
                 "cost",
        ylab="gz(all P(X)) / gz(all X)", ref="1.0 — as costly as the frames, compressed"),
    "info_extraction_ratio": dict(
        color="#1baf7a", stem="perception_info_extraction_per_game",
        subtitle="Information extraction — compressed feature bytes per compressed "
                 "observation byte, frame by frame",
        ylab="mean gz(P(X)) / gz(X)", ref="1.0 — as costly as the frame, compressed"),
    # kept, and drawable by name, but not part of the default set
    "dl_ratio": dict(
        color="#eda100", stem="perception_dl_per_game",
        subtitle="Bytes P emits per byte of observation", ylab="dl_ratio"),
    "code_chars": dict(
        color="#e34948", stem="perception_code_chars_per_game",
        subtitle="Characters of P, comments and all", ylab="code characters"),
    "sloc": dict(
        color="#e34948", stem="perception_sloc_per_game",
        subtitle="Non-comment, non-blank lines of P", ylab="SLOC"),
    "pf_dl_gz_ratio": dict(
        color="#e87ba4", stem="perception_pf_dl_gz_per_game",
        subtitle="Bytes P emits per compressed byte of observation, frame by frame",
        ylab="mean |P(X)| / |gzip(X)|", ref="1.0 — as long as the frame's compressed size"),
}
NAMED = ["ast_nodes", "mean_out_chars", "diversity_bytes", "norm_diversity",
         "info_extraction_ratio"]
NOTE = ("The incumbent is the best node so far, so a flat stretch\n"
        "is a node that revised only the world knowledge K and\n"
        "inherited its parent's P.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train", choices=("train", "test"))
    ap.add_argument("--metric", default="named",
                    choices=("named", "all", *SERIES),
                    help="'named' (default) draws the five named scores; 'all' adds the "
                         "ratio and code-text companions")
    a = ap.parse_args()
    rows = load(a.split)
    games = sorted({r["game"] for r in rows}, key=display_name)
    fields = {"named": NAMED, "all": list(SERIES)}.get(a.metric, [a.metric])
    curves = {f: draw_grid(rows, games, f, a.split, note=NOTE, **SERIES[f]) for f in fields}
    summarise(curves, rows, games, fields, SERIES, a.split)


if __name__ == "__main__":
    main()
