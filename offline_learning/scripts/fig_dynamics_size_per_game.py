#!/usr/bin/env python3
"""Size of the dynamics model K over the search, one panel per game, one figure per measure.

The companion of `fig_perception_compression_per_game.py`. A run learns two things: the
abstraction program P, which turns a frame into features, and the world knowledge K -- the
paper's **dynamics model** -- the English rules the planner reads alongside those features.
The perception figures follow P; these follow K.

The figure is `k_chars`, characters of world knowledge. Six companions are measured into
`metrics.csv` and drawable by name -- `k_sentences` (roughly how many things K asserts),
`k_gzip_bytes` (gz(K)), `k_words`, `k_claims` (lines opening with a bullet or a number),
`k_lines` (non-blank lines) and `k_norm_bytes` (gz(K) over gz of all the observations, the
one measure here comparable across games) -- but they are companions, not the figure: within
a game they mostly trace the same staircase in different units.

K is prose, so there is no AST to fall back on as there is for P: `k_chars` is the headline
and the structural counts are proxies for how many rules the text states. Sentences are the
most even of them -- ~80-160 characters each across these 15 texts, against 84-544 per
bullet, because some games' K is written as paragraphs and others as lists -- but they are
still a writing-style measure as much as a content one, so read the trajectory within a game
and not the level across games.

The seed K is the EMPTY STRING, so every curve starts at 0 and the first proposal that
writes anything is a jump from nothing, not a growth step.

    uv run python offline_learning/scripts/perception_metrics.py          # writes metrics.csv
    uv run python offline_learning/scripts/fig_dynamics_size_per_game.py

Writes `analysis/wm_quant/dynamics_chars_per_game.{pdf,png}`. `--metric <name>` draws one
of the companions instead, `--metric all` draws every measure. The panel grid, the incumbent
line and the ringed marker are `wm_panel_grid.py`.
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

# Violet, slot 7 of the validated reference palette (dataviz skill), throughout. In the
# perception family the colour distinguishes the SCORE; here it distinguishes the PARAMETER,
# which is the identity worth carrying: every figure in this family measures the dynamics
# model, so all of them are violet and none collides with a perception figure's blue, orange
# or aqua. Each figure carries a single series, so no two hues ever share a panel.
K = "#4a3aa7"
SERIES = {
    # the one that is drawn
    "k_chars": dict(
        color=K, stem="dynamics_chars_per_game",
        subtitle="Size of the dynamics model — characters of world knowledge",
        ylab="K characters"),
    # measured and kept, drawable by name, but not part of the default figure set
    "k_sentences": dict(
        color=K, stem="dynamics_sentences_per_game",
        subtitle="Assertions in the dynamics model — sentences of world knowledge",
        ylab="K sentences"),
    "k_gzip_bytes": dict(
        color=K, stem="dynamics_gzip_per_game",
        subtitle="Compressed size of the dynamics model — how much of it is irreducible",
        ylab="gz(K)   (bytes)"),
    "k_words": dict(
        color=K, stem="dynamics_words_per_game",
        subtitle="Words of world knowledge", ylab="K words"),
    "k_claims": dict(
        color=K, stem="dynamics_claims_per_game",
        subtitle="Bullet rules in the dynamics model", ylab="K bullet lines"),
    "k_lines": dict(
        color=K, stem="dynamics_lines_per_game",
        subtitle="Non-blank lines of world knowledge", ylab="K lines"),
    "k_norm_bytes": dict(
        color=K, stem="dynamics_norm_bytes_per_game",
        subtitle="The dynamics model against the observations it describes",
        ylab="gz(K) / gz(all X)", ref="1.0 — as costly as the frames, compressed"),
}
NAMED = ["k_chars"]
NOTE = ("The incumbent is the best node so far, so a flat stretch\n"
        "is a node that revised only the abstraction program P and\n"
        "inherited its parent's dynamics model. It starts at 0:\n"
        "the seed K is empty.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train", choices=("train", "test"))
    ap.add_argument("--metric", default="named", choices=("named", "all", *SERIES),
                    help="'named' (default) draws k_chars alone; 'all' draws every "
                         "measure; or name one")
    a = ap.parse_args()
    rows = load(a.split)
    games = sorted({r["game"] for r in rows}, key=display_name)
    fields = {"named": NAMED, "all": list(SERIES)}.get(a.metric, [a.metric])
    curves = {f: draw_grid(rows, games, f, a.split, note=NOTE, **SERIES[f]) for f in fields}
    summarise(curves, rows, games, fields, SERIES, a.split)


if __name__ == "__main__":
    main()
