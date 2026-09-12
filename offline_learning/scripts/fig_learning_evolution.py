#!/usr/bin/env python3
"""What the learner knows, four times over, on one Autumn game.

The paper's tables say the learned world model plans better than the raw frame; they do
not show what was learned. This draws that: three transitions from a training drive
across the top, and underneath, the state of the two parameters -- the abstraction
program `P` and the world knowledge `K` -- at four points along the REx lineage that
produced the shipped artifact.

    uv run python offline_learning/scripts/fig_learning_evolution.py                # dq8gc
    uv run python offline_learning/scripts/fig_learning_evolution.py -c mario_coin
    uv run python offline_learning/scripts/fig_learning_evolution.py -c all

Writes `analysis/learning_example/learning_evolution[_<candidate>].pdf` (and `.png`), authored at the
ICLR text width so the font sizes here are the font sizes on the page. `dq8gc` keeps the
unsuffixed name because `main.tex` already includes it.

Everything printed in the figure is read out of the run, never retyped:

  * the frames are rows of `<drive>/episode_0/trajectory.csv`, the same file the
    training pool was sliced from, put through `strip_autumn_obs_metadata` so they are
    the strings the learner actually saw;
  * every `P(X)` line is produced by `run_perceive`-ing that node's own perception code
    over that frame here, at draw time -- so a wrong node index shows up as a wrong
    string rather than as a plausible-looking caption;
  * the code and belief excerpts are quoted from `candidates.jsonl`, and `verify()`
    asserts each fragment occurs verbatim in the node it is attributed to (modulo
    markdown emphasis and whitespace). An excerpt that drifts out of the artifact stops
    the script instead of printing. `...` is this script's elision marker.

A belief bullet may carry a `warn`: a note saying the frames above contradict it. Those
are written by hand against the game's `.sexp` and are the one thing here the script
cannot check for you -- each is justified in a comment beside it.

The candidates, and why each exists:

  dq8gc  (Disease)  the click that only moves a latent variable; K goes from "clicks have
                    no visible effect" to "click sets the hidden active cell".
  mario_jump        Mario jumping through a platform, then falling on a noop, then
                    vanishing under a coin. One agent action beside two autonomous ones.
  mario_coin        the same fall carried one step further, to where the coin is eaten --
                    which is exactly what node 3's "gold cells ... are stationary and
                    never change" forbids, and node 16 repairs.
  mario_ledge       walking off the end of a platform and falling two rows: the cleanest
                    reading of gravity, with the enemy block marching throughout.
  ice_click         Ice, whose P is the only learned abstraction that names objects
                    (`fixed:`/`movable:`/`other:`) -- and whose K, revised at node 6,
                    acquires a click qualifier that these very frames refute.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import FancyArrow, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from offline_learning.validate import (  # noqa: E402
    run_perceive,
    strip_autumn_obs_metadata,
)

csv.field_size_limit(10**7)

TREE = REPO / "logs/2026-08-24/human_curated/rexpure"
HUMAN = REPO / "offline_learning/human_data"

# --- palette: the paper's, imported rather than restated (see analyze_planning_difficulty).
sys.path.insert(0, str(HERE))
from analyze_planning_difficulty import COLOR, GRID, INK, INK2, INK3, SURFACE  # noqa: E402

BLUE, WARN = COLOR["NLWM (Plain)"], COLOR["ICL"]
MONO, SANS = "DejaVu Sans Mono", "DejaVu Sans"

# ---------------------------------------------------------------------------
# Per-game rows: the state of (P, K) at four nodes of the shipped artifact's lineage.
# `code` and `bullets` are quoted from the node named in `node`; [] collapses the P box
# to its tag line, which is what to do when that node changed only K.
# ---------------------------------------------------------------------------
SEED = dict(node=0, when="node 0", sub="seed", tag_p="blank slate", tag_k=None,
            code=["grid = _parse_grid(obs)", "...", "return \"\""],
            gloss="parses the grid, emits nothing", bullets=[])

ROWS = {
    # lineage 0 -> 1 -> 14 -> 16 -> 29; node 14 differs from 16 in P only.
    "dq8gc": [
        SEED,
        dict(node=1, when="node 1", sub="iter 2 · P", tag_p="updated", tag_k=None,
             code=["background = colour_counts.most_common(1)[0][0]",
                   "...",
                   "for r, row in enumerate(grid):",
                   "    for c, colour in enumerate(row):",
                   "        if colour != background:",
                   "            features.append(f\"{r},{c}:{colour}\")",
                   "result = \"; \".join(features)"],
             gloss="every non-background cell, as row,col:colour", bullets=[]),
        dict(node=16, when="node 16", sub="iter 24 · K",
             tag_p="inherited from node 14", tag_k="new",
             code=["parts = [f\"bg:{background}\", f\"hash:{h}\",",
                   "         f\"dim:{rows}x{cols}\"]",
                   "...",
                   "parts.append(\"; \".join(objects))",
                   "result = \" | \".join(parts)"],
             gloss="plus a background, a grid hash and the dimensions",
             bullets=[
                 ["Clicks have no visible effect",
                  "the set of colored cells is unchanged"],
                 ["Every timestep, after the action effect, an automatic process occurs:",
                  "any gray cell that is orthogonally adjacent",
                  "to any darkgreen cell becomes darkgreen"],
                 ["Directional actions", "move exactly one darkgreen cell one step",
                  "The identity of the moving cell is not deducible from the current set "
                  "of darkgreen cells alone; it depends on a hidden", "active", "cell"],
             ]),
        dict(node=29, when="node 29", sub="iter 40 · K",
             tag_p="unchanged from node 16 — same code, same features", tag_k="updated",
             code=[], gloss="",
             bullets=[
                 ["Each timestep proceeds in two phases:", "Adjacency phase",
                  "every gray cell that is orthogonally adjacent",
                  "becomes darkgreen. This uses the state", "before", "the action"],
                 ["click ROW COL", "Sets the hidden", "active cell",
                  "to the cell at (ROW, COL)",
                  "Suppresses the adjacency phase for this timestep",
                  "No other visible change"],
                 ["Directional actions",
                  "Move the hidden active cell one step in the given direction"],
             ]),
    ],
    # lineage 0 -> 2 -> 3 -> 6 -> 15 -> 16; 6 and 15 are P-only, and 16 carries 15's P.
    "n2ntd": [
        SEED,
        dict(node=2, when="node 2", sub="iter 3 · P", tag_p="updated", tag_k=None,
             code=["parts_alt = [f\"grid {rows}x{cols}; bg:{bg}; cells:\"]",
                   "for r, c, clr in cells:",
                   "    parts_alt.append(f\"({r},{c},{clr});\")",
                   "summary = \"\".join(parts_alt)"],
             gloss="one flat list of non-background cells", bullets=[]),
        dict(node=3, when="node 3", sub="iter 4 · K",
             tag_p="unchanged from node 2", tag_k="new", code=[], gloss="",
             bullets=[
                 ["The world contains a red cell that is moved by directional actions:",
                  "up", "moves it 4 cells upward", "left", "moves it 1 cell left",
                  "right", "moves it 1 cell right"],
                 ["The blue 2×3 block moves autonomously: each time step it shifts either "
                  "left or right by 1 column. This movement is independent of the action "
                  "taken"],
                 # mario.sexp: `moveDownNoCollision` runs inside mario's own `next`, and
                 # the coin is removed on contact -- so both halves of this are false,
                 # and every drawn window refutes them.
                 dict(frags=["All other objects (gold cells, dark orange L-shaped groups) "
                             "are stationary and never change"],
                      warn="and no gravity rule at all — refuted by the transitions above"),
             ]),
        dict(node=16, when="node 16", sub="iter 20 · K",
             tag_p="inherited from node 15 — grouped by colour", tag_k="updated",
             code=["cells_by_colour.setdefault(colour, []).append((r, c))",
                   "...",
                   "coords_str = \" \".join(f\"{r},{c}\" for r, c in positions_sorted)",
                   "summary_parts.append(f\"{colour}: {coords_str}\")"],
             gloss="",
             bullets=[
                 ["The red cell moves downward by 1 cell on each noop step, unless it is "
                  "at row 11 or the cell directly below is occupied by any other object"],
                 ["When a red cell moves", "onto a cell containing a gold cell, the gold "
                  "cell disappears immediately. The red cell does not appear in that cell "
                  "until the next time step"],
                 ["It moves right until its rightmost column reaches 11, then moves left "
                  "until its leftmost column reaches 0", "oscillating continuously"],
             ]),
    ],
    # lineage 0 -> 3 -> 4 -> 6 -> 8 -> 12; K is revised at 6 and then frozen, P at 8/12.
    "bt3gb": [
        SEED,
        dict(node=3, when="node 3", sub="iter 3 · P", tag_p="updated", tag_k=None,
             code=["cells.append((r, c, colour))",
                   "...",
                   "cell_strs = [f\"{colour}({r},{c})\" for r, c, colour in cells]",
                   "parts.append(\", \".join(cell_strs))"],
             gloss="every non-background cell, one by one", bullets=[]),
        dict(node=4, when="node 4", sub="iter 4 · K",
             tag_p="unchanged from node 3", tag_k="new", code=[], gloss="",
             bullets=[
                 ["There is a fixed 2x2 block at (0,0),(0,1),(1,0),(1,1) that toggles "
                  "between gold and gray"],
                 ["There is a contiguous block of three gray cells on row 0 (the",
                  "movable block", ") that moves left/right with the corresponding "
                  "direction actions"],
                 ["On click (any ROW COL): all cells in the top-left 2x2 block toggle "
                  "gold↔gray; all blue/lightblue cells anywhere on the grid toggle "
                  "blue↔lightblue"],
             ]),
        dict(node=12, when="node 12", sub="iter 22 · P",
             tag_p="rewritten at node 8, refined here — it emits the objects K names",
             tag_k="revised at node 6, then frozen",
             code=["gray_row0_cols = sorted([c for r,c,col in cells if r==0",
                   "        and col==\"gray\" and c not in fixed_cols_row0])",
                   "...",
                   "parts.append(\"fixed:\" + \",\".join(fixed_entries))",
                   "parts.append(f\"movable:gray_row0:{movable_run[0]}-{movable_run[1]}\")"],
             gloss="",
             bullets=[
                 ["These cells are either gold or gray. Their color persists even when "
                  "covered by the movable block; they are visible only when not covered"],
                 ["On noop: every blue/lightblue cell not on row 15 attemps to move down "
                  "one row. If the cell directly below is empty", "it moves"],
                 # ice.sexp: `(on clicked ...)` toggles celestialBody.day AND every water
                 # drop, with no position test. Node 4 had this right; the node-6
                 # revision added a qualifier these frames disprove.
                 dict(frags=["If ROW in {0,1} and COL in {0,1} (i.e. clicked inside the "
                             "top-left 2x2 block), then the fixed 2x2 block toggles",
                             "Else (click outside the top-left 2x2 block), then both of "
                             "the following occur"],
                      warn="X₀→X₁ is a click at (1,1) and the water toggles too — "
                           "node 4 had this rule right"),
             ]),
    ],
    # lineage 0 -> 2 -> 3 -> 7 -> 10 -> 11 -> 14 -> 19 -> 23. Nodes 11/14/19 reformat
    # P (ranges, labels) without adding anything; node 23 is the one that computes.
    "7xf97": [
        SEED,
        dict(node=2, when="node 2", sub="iter 3 · P", tag_p="updated", tag_k=None,
             code=["objects.append((r, c, colour))",
                   "...",
                   "parts = [f\"{r},{c}:{clr}\" for r, c, clr in objects]",
                   "summary = \"|\".join(parts)"],
             gloss="every non-background cell, as row,col:colour", bullets=[]),
        dict(node=10, when="node 10", sub="iter 12 · K",
             tag_p="unchanged from node 2", tag_k="new", code=[], gloss="",
             bullets=[
                 ["Gold and gray cells appear only in rows 0-2. They form two contiguous "
                  "blocks", "The two blocks are always separated by at least one empty "
                  "column at the start of an episode, but actions can bring them "
                  "together or change their widths and positions"],
                 dict(frags=["If a gray block exists, it shifts left by one column",
                             "The column vacated on the right becomes empty unless it "
                             "is immediately left of a gold block, in which case that "
                             "column becomes gold (the gold block expands leftwards "
                             "into it)"],
                      warn="X₂→X₃ vacates column 2 with no gold block anywhere on the "
                           "grid, and it comes back gold"),
                 ["The column vacated on the left becomes empty unless it is immediately "
                  "right of a gold block, in which case that column becomes gold (the "
                  "gold block expands rightwards into it)"],
             ]),
        dict(node=23, when="node 23", sub="iter 27 · P",
             tag_p="reformatted at 11/14/19, then rewritten here", tag_k=None,
             k_from=10,
             k_note="unchanged from node 10 — the same text, byte for byte. What this "
                    "iteration changed is P: it now measures the relation those rules "
                    "branch on, so the condition is in the state description and not "
                    "left to the reader.",
             code=["gold_right = max(c for _, c in gold_cells)",
                   "gray_left = min(c for _, c in gray_cells)",
                   "...",
                   "if gold_exists and gray_exists and gold_right + 1 == gray_left:",
                   "    gold_adj_right_to_gray = 1",
                   "...",
                   "flag_str = \"f:\" + \",\".join(flag_pairs)"],
             gloss="a flag block: which blocks exist, which touch an edge, and whether "
                   "gold and gray are adjacent with no gap",
             bullets=[]),
    ],
}

# ---------------------------------------------------------------------------
# The candidates: a window of one drive, plus the words for the three transitions.
# ---------------------------------------------------------------------------
CANDIDATES = {
    "dq8gc": dict(
        game="dq8gc", drive="train_d0", steps=(21, 22, 23, 24), ring=BLUE, plain=True,
        title="Disease (DQ8GC) — three transitions from one training drive",
        notes=[("right", "the steered particle moves one cell right, into the row above "
                         "a gray one"),
               ("noop", "no action, and the state changes anyway: (3,4) is orthogonally "
                        "adjacent to (2,4), so it turns dark green"),
               ("right", "only (2,4) moves again — being dark green is not what makes a "
                         "cell the one the arrows steer")],
        foot=""),
    "mario_jump": dict(
        game="n2ntd", drive="train_d0", steps=(14, 15, 16, 17), ring=INK,
        title="Mario (N2NTD) — three transitions from one training drive",
        notes=[("up", "the red player jumps 4 rows, (11,1) → (7,1), straight through the "
                      "platform above it"),
               ("noop", "no action, and it falls back one row"),
               ("noop", "it falls onto the coin at (9,1) and disappears — the player is "
                        "drawn underneath the gold")],
        foot="on every one of these steps the blue 2×3 block also marches one column "
             "left, whatever the agent does"),
    "mario_coin": dict(
        game="n2ntd", drive="train_d0", steps=(15, 16, 17, 18), ring=INK,
        title="Mario (N2NTD) — three transitions from one training drive",
        notes=[("noop", "no action, and the player falls one row, (7,1) → (8,1)"),
               ("noop", "it falls onto the coin at (9,1) and vanishes — the player is "
                        "drawn underneath the gold"),
               ("noop", "the coin is gone and the player stands in its place: gold cells "
                        "are not stationary after all")],
        foot="on every one of these steps the blue 2×3 block also marches one column "
             "left, whatever the agent does"),
    "mario_ledge": dict(
        game="n2ntd", drive="train_d0", steps=(22, 23, 24, 25), ring=None,
        title="Mario (N2NTD) — three transitions from one training drive",
        notes=[("right", "the player steps right, (9,2) → (9,3), off the end of the "
                         "platform"),
               ("noop", "no action, and it falls one row"),
               ("noop", "and falls again, to the floor at row 11")],
        foot="on every one of these steps the blue 2×3 block also advances one column, "
             "whatever the agent does"),
    "ice_click": dict(
        game="bt3gb", drive="train_d0", steps=(62, 63, 64, 65), ring=SURFACE,
        title="Ice (BT3GB) — three transitions from one training drive",
        notes=[("click 1 1", "one click toggles two things at once: the 2×2 latch "
                             "gray→gold, and both water cells lightblue→blue"),
               ("noop", "no action, and each water cell falls one row"),
               ("noop", "and again — the lower one reaches the floor at row 15")],
        foot=""),
    "grow_cover": dict(
        game="7xf97", drive="train_d2", steps=(43, 44, 45, 46), ring=None, diff=2,
        title="Grow (7XF97) — three transitions from one training drive",
        notes=[("left", "the gray block slides one column left, onto the last gold "
                        "column still showing, and gold leaves the state altogether"),
               ("left", "it slides again, off the left edge of the grid — three of its "
                        "four columns are still drawn"),
               ("left", "and gold is back: at column 2, the column the gray block has "
                        "just vacated")],
        foot="no gold cell in this drive is ever drawn outside columns 0-2"),
}


# ---------------------------------------------------------------------------
# One short line per top-level bullet of that node's world knowledge, in order.
# These are the only hand-written strings in the dump, so each is bound to the
# text it summarises by a hash of that text: change K and the guard fires rather
# than letting a stale precis go on describing a rule that is no longer there.
# ---------------------------------------------------------------------------
K_SHORT = {
    ("dq8gc", 16): ("e9ac356b", [
        "16x16, black background; only darkgreen and gray ever appear",
        "click: no visible effect on the set of coloured cells",
        "after each action, any gray orthogonally adjacent to a darkgreen turns "
        "darkgreen, visible in the next state",
        "arrows move exactly one darkgreen cell one step; which one is latent — a "
        "hidden active cell set by past actions — and the rest stay put",
        "gray cells never move; they leave only by turning darkgreen",
        "a darkgreen landing on a darkgreen coalesces into one",
        "moving off-grid: assumed no change, never observed",
    ]),
    ("dq8gc", 29): ("e82f2e53", [
        "16x16, black background; only darkgreen and gray ever appear",
        "each step is two phases: adjacency on the pre-action state, then the action",
        "click ROW COL: sets the hidden active cell to (ROW,COL) and suppresses "
        "adjacency for that step; nothing else visible",
        "noop: adjacency only",
        "arrows: move the active cell one step, coalescing if a darkgreen is already "
        "there; off-grid assumed inert; the active cell follows the move",
        "active cell: the hidden state picking what the arrows move, set by a click "
        "or by a successful move; adjacency never changes it",
        "collisions: only darkgreen-onto-darkgreen merging is confirmed; gray "
        "collisions unobserved",
        "ordering: adjacency strictly before the action effect",
    ]),
    ("n2ntd", 3): ("c0f6a043", [
        "arrows move the red cell: up 4 rows, left/right 1 column (down assumed 4 "
        "rows, unobserved)",
        "click anywhere: red becomes mediumpurple and drifts up a row per step; a "
        "new red respawns at the old location on the next step",
        "the blue 2x3 block moves on its own, one column per step, independent of "
        "the action",
        "everything else — gold cells, darkorange L-shapes — is stationary and "
        "never changes",
    ]),
    ("n2ntd", 16): ("4c41d097", [
        "red falls one row per noop unless it is at row 11 or the cell below is "
        "occupied; a direction key overrides the fall on its own step",
        "arrows: up 4 rows, left/right 1 column (down assumed 4 rows, unobserved)",
        "click anywhere: red becomes mediumpurple and drifts up a row per step; a "
        "new red respawns at the old location on the next step",
        "red moving onto gold eats it at once; red is invisible for one step, then "
        "appears there and the gold is gone for good",
        "the blue 2x3 block oscillates on its own: right until column 11, left "
        "until column 0, repeating, independent of the action",
        "the darkorange L-shapes are stationary, but block the fall from directly "
        "below",
    ]),
    ("bt3gb", 4): ("52337a0b", [
        "16x16, rows and columns 0-15, black background",
        "four cell types: gold, gray, blue, lightblue",
        "a fixed 2x2 at (0,0)-(1,1) toggling gold/gray",
        "a movable block of three gray cells on row 0, shifted by left/right when "
        "the target cells are empty; clicks do not affect it",
        "a bottom block of contiguous blue/lightblue on row 15, colour toggled by "
        "clicks, width varying",
        "noop: every blue/lightblue off row 15 falls one row if the cell below is "
        "empty; row 15 never moves",
        "left: the movable block shifts one column left if possible",
        "right: the movable block shifts one column right if possible",
        "down: spawns a blue/lightblue at (1,c) under the movable block's middle "
        "column if that cell is empty, coloured like the bottom block",
        "click, any ROW COL: the 2x2 toggles gold/gray and every blue/lightblue "
        "toggles; the movable block is untouched",
        "up: unobserved, assumed to do nothing",
    ]),
    ("bt3gb", 12): ("55462e84", [
        "16x16, rows and columns 0-15, black background",
        "four cell types: gold, gray, blue, lightblue",
        "a fixed 2x2 at (0,0)-(1,1), gold or gray, whose colour persists while "
        "covered and reappears when uncovered",
        "the movable block moves only within bounds and only if no blue/lightblue "
        "blocks it; it overwrites the fixed 2x2, which resurfaces when it leaves",
        "a bottom block of contiguous blue/lightblue on row 15, colour toggled by "
        "clicks outside the 2x2, growing as cells arrive",
        "noop: every blue/lightblue off row 15 falls one row if the cell below is "
        "empty; row 15 never moves",
        "left: the movable block shifts one column left if possible",
        "right: the movable block shifts one column right if possible",
        "down: spawns a blue/lightblue at (1,c) under the movable block's middle "
        "column if that cell is empty, coloured like the bottom block",
        "click ROW COL: inside the top-left 2x2, only the fixed block toggles "
        "(stored colour, possibly hidden); outside, the 2x2 and every "
        "blue/lightblue toggle together",
        "up: no visible effect",
    ]),
    ("7xf97", 10): ("983de101", [
        "left: gray shifts one column left; the column it vacates on the right turns "
        "gold when it is immediately left of a gold block, else empty; with no gray "
        "block, nothing happens",
        "right: the mirror image — gray shifts right, and the column vacated on the "
        "left turns gold when immediately right of a gold block",
        "up: no effect on gold or gray; blue drifts as usual",
        "down: no effect on gold or gray, and the blue drift is cancelled for that step",
        "click ROW COL: on gold, the block shifts left if its rightmost column was "
        "clicked and right otherwise, vanishing or truncating at an edge; on gray, an "
        "existing gold block disappears and an absent one appears just left of the "
        "gray; on anything else, nothing",
        "noop: no effect on gold or gray; and every action is followed by the blue "
        "drift, down excepted",
    ]),
}

# ---------------------------------------------------------------------------
# Load + verify
# ---------------------------------------------------------------------------
def load_candidates(game: str) -> dict[int, dict]:
    p = TREE / f"{game}_s1/rexpure_run_seed1/candidates.jsonl"
    if not p.exists():
        raise SystemExit(f"no candidates.jsonl at {p}")
    return {c["idx"]: c for c in map(json.loads, p.read_text().splitlines())}


def load_frames(game: str, drive: str, steps) -> list[dict]:
    p = HUMAN / f"{game}/informative_curated/drives/{drive}/episode_0/trajectory.csv"
    rows = {int(r["Step"]): r for r in csv.DictReader(p.open())}
    out = []
    for s in steps:
        obs = strip_autumn_obs_metadata(rows[s]["Observation"])
        m = re.search(r"\[\[.*\]\]", obs, re.S)
        out.append(dict(step=s, action=rows[s]["Action"], obs=obs,
                        grid=json.loads(m.group(0))))
    return out


def frags_of(bullet):
    return bullet["frags"] if isinstance(bullet, dict) else bullet


def _norm(s: str) -> str:
    """Fold what is formatting rather than content: markdown emphasis, curly quotes,
    dashes, runs of whitespace."""
    s = s.replace("**", "").replace("`", "").replace("*", "")
    s = (s.replace("“", '"').replace("”", '"').replace("’", "'")
          .replace("–", "-").replace("—", "-"))
    return re.sub(r"\s+", " ", s).strip()


def verify(rows, cands: dict[int, dict]) -> None:
    """Every quoted fragment must occur in the node it is attributed to."""
    for row in rows:
        c = cands[row["node"]]
        for line in row["code"] or []:
            if line.strip() in ("", "..."):
                continue
            if _norm(line) not in _norm(c["perception"]):
                raise SystemExit(f"node {row['node']}: code not in artifact: {line!r}")
        for bullet in row["bullets"]:
            for frag in frags_of(bullet):
                if _norm(frag) not in _norm(c["world_knowledge"] or ""):
                    raise SystemExit(
                        f"node {row['node']}: belief fragment not in artifact: {frag!r}")
        if not row["bullets"] and (c["world_knowledge"] or "").strip():
            src = row.get("k_from")
            if src is None:
                raise SystemExit(
                    f"node {row['node']}: drawn as empty K but artifact has one")
            if (c["world_knowledge"] or "").strip() != (
                    cands[src]["world_knowledge"] or "").strip():
                raise SystemExit(f"node {row['node']}: k_from={src}, but the two "
                                 f"world knowledge texts differ")


# ---------------------------------------------------------------------------
# Text measurement. The renderer's advance widths, not ink extents -- ink under-
# measures by ~10% on long strings, which is exactly enough to run off the page.
# ---------------------------------------------------------------------------
DPI = 220
_MEAS = plt.figure(dpi=DPI)
_MEAS.canvas.draw()
_R = _MEAS.canvas.get_renderer()


def width_in(s: str, size: float, mono: bool) -> float:
    w, _h, _d = _R.get_text_width_height_descent(
        s, FontProperties(family=MONO if mono else SANS, size=size), False)
    return w / DPI


def _hard_split(w: str, box_in: float, size: float, mono: bool) -> list[str]:
    """Break a single word too wide for the box. A feature dump is one unbroken token --
    `cells:(0,6,blue);(0,7,blue);...` -- and without this it runs off the page."""
    out = []
    while width_in(w, size, mono) > box_in and len(w) > 1:
        lo, hi = 1, len(w)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if width_in(w[:mid], size, mono) <= box_in:
                lo = mid
            else:
                hi = mid - 1
        out.append(w[:lo])
        w = w[lo:]
    out.append(w)
    return out


def wrap(text: str, box_in: float, size: float, mono: bool) -> list[str]:
    """Greedy wrap to a box width in inches, measured. Leading whitespace is part of the
    line, not a word: code excerpts carry their nesting in it."""
    lead = text[: len(text) - len(text.lstrip(" "))]
    words = text.strip(" ").split(" ")
    words[:1] = [lead + w for w in words[:1]]      # the indent rides on the first word
    lines, cur = [], ""
    for w in words:
        trial = w if not cur else cur + " " + w
        if width_in(trial, size, mono) <= box_in:
            cur = trial
        elif width_in(w, size, mono) > box_in:
            if cur:
                lines.append(cur)
            chunks = _hard_split(w, box_in, size, mono)
            lines.extend(chunks[:-1])
            cur = chunks[-1]
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


# ---------------------------------------------------------------------------
# Geometry (inches; origin bottom-left)
# ---------------------------------------------------------------------------
W = 5.5                      # ICLR \textwidth
PAD = 0.03                   # keeps strokes off the trim edge
GSZ = 0.92                   # frame side; the gap then fills the text width
GAP = (W - 2 * PAD - 4 * GSZ) / 3
LEFT = 0.60                  # gutter holding the row labels
COL_GAP = 0.12
P_W = 2.06
K_W = W - PAD - LEFT - P_W - COL_GAP
FS_CODE, FS_OUT, FS_K, FS_NOTE, FS_LBL, FS_TAG = 4.8, 4.7, 5.3, 5.5, 5.8, 5.0
LH_CODE, LH_OUT, LH_K = FS_CODE * 1.42 / 72, FS_OUT * 1.42 / 72, FS_K * 1.34 / 72
IND = 0.09                   # hanging indent for wrapped code / feature lines


def background_of(grid) -> str:
    """The frame's own background: its most common colour. Autumn worlds are not all
    black -- mario's is white -- and a hardcoded background silently inverts them."""
    flat = [c for row in grid for c in row]
    return max(set(flat), key=flat.count)


def draw_grid(ax, x, y, size, grid, bg, rings=()):
    n = len(grid)
    c = size / n
    dark = sum(to_rgb(bg)) / 3 < 0.5
    ax.add_patch(Rectangle((x, y), size, size, facecolor=bg,
                           edgecolor=GRID if dark else INK3, linewidth=0.5, zorder=2))
    for r, row in enumerate(grid):
        for j, name in enumerate(row):
            if name == bg:
                continue
            ax.add_patch(Rectangle((x + j * c, y + size - (r + 1) * c), c, c,
                                   facecolor=name, edgecolor="none", zorder=3))
    for (r, j), colour in rings:
        ax.add_patch(Rectangle((x + j * c - 0.013, y + size - (r + 1) * c - 0.013),
                               c + 0.026, c + 0.026, facecolor="none", edgecolor=colour,
                               linewidth=0.8, zorder=4))


def changed_cells(a, b):
    """Every cell whose colour differs. All of them get a ring: in a world where the
    scenery moves by itself, ringing only one would say the others did not change.
    A candidate with ring=None draws no rings at all and lets the frames speak."""
    return [(r, j) for r, row in enumerate(b) for j, v in enumerate(row) if a[r][j] != v]


def feature_diff(a: str, b: str, lab: str = "₂") -> str:
    """What P(X_lab) says that the frame before it did not, token for token. The abstraction is a string,
    so the diff is over its tokens -- `|`- and `;`-separated fields -- computed here
    rather than asserted, so it cannot describe a change that is not there."""
    def items(val):
        """Split a value into its items: on whitespace, and on commas that separate
        items rather than the halves of a coordinate pair -- never inside brackets, so
        `(0,6,blue)` survives whole while `gold(0,0),gold(0,1)` comes apart."""
        out, cur, depth = [], "", 0
        for i, ch in enumerate(val):
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth = max(0, depth - 1)
            nxt = val[i + 1] if i + 1 < len(val) else ""
            if depth == 0 and (ch.isspace() or (ch == "," and not nxt.isdigit())):
                if cur:
                    out.append(cur)
                cur = ""
            else:
                cur += ch
        if cur:
            out.append(cur)
        return out

    def toks(s):
        # A field can carry several labelled groups -- `blue: 0,6 0,7 red: 8,1` -- and
        # each item under a label is its own token. Without this a P that groups cells
        # by colour reports every cell as changed the moment one of them moves.
        out = []
        for field in (t.strip() for t in re.split(r"[|;]", s) if t.strip()):
            parts = re.split(r"\b([A-Za-z_]\w*):", field)
            if len(parts) < 3:
                out.extend(items(field))
                continue
            if parts[0].strip():
                out.extend(items(parts[0]))
            for lab, val in zip(parts[1::2], parts[2::2]):
                vals = items(val)
                out.extend(f"{lab}:{i}" for i in vals) if vals else out.append(f"{lab}:")
        return out

    ta, tb = toks(a), toks(b)
    gone, new = [t for t in ta if t not in tb], [t for t in tb if t not in ta]
    if not gone and not new:
        return f"P(X{lab}): identical — the transition is invisible to P"
    hashed = any(t.startswith("hash:") for t in gone + new)
    gone = [t for t in gone if not t.startswith("hash:")]
    new = [t for t in new if not t.startswith("hash:")]
    parts = [f"P(X{lab}):"] + [f"−{t}" for t in gone] + [f"+{t}" for t in new]
    return " ".join(parts) + ("  (and the hash)" if hashed else "")


SUB = "₀₁₂₃"


def prepare(rows, cands, frames, d: int = 1):
    """Wrap every block once, so a row's height is known before anything is drawn.
    `d` picks the transition the P column reports: it prints P(X_d) and the diff into
    X_(d+1), so a candidate can point it at the step where its features do the work."""
    for row in rows:
        code = cands[row["node"]]["perception"]
        row["pout"] = []
        for f in frames:
            out, err = run_perceive(code, f["obs"])
            if err:
                raise SystemExit(f"node {row['node']} perceive() raised: {err}")
            row["pout"].append(out)
        row["score"] = cands[row["node"]]["train_score"]

        row["_code"] = []
        for line in row["code"] or []:
            for j, seg in enumerate(wrap(line, P_W - 0.02, FS_CODE, mono=True) or [""]):
                row["_code"].append((seg, IND if j else 0.0))
        row["_tag_p"] = wrap(row["tag_p"] + (" — " + row["gloss"] if row["gloss"] else ""),
                             P_W - 0.02, FS_TAG, mono=False)
        first = row["pout"][d]
        row["_out"] = [] if not row["_code"] else [
            wrap(f"P(X{SUB[d]}) = {first}" if first
                 else f"P(X{SUB[d]}) = (empty string)", P_W - 0.02, FS_OUT, mono=True),
            wrap(feature_diff(row["pout"][d], row["pout"][d + 1], SUB[d + 1]),
                 P_W - 0.02, FS_OUT, True)]
        row["_bul"] = []
        for b in row["bullets"]:
            text = " … ".join(_norm(f) for f in frags_of(b))
            warn = b.get("warn") if isinstance(b, dict) else None
            row["_bul"].append((wrap(text, K_W - 0.10, FS_K, False),
                                wrap("↑ " + warn, K_W - 0.10, FS_K - 0.4, False)
                                if warn else []))

        p_h = (0.14 + LH_CODE * 1.15 * len(row["_tag_p"])
               + LH_CODE * len(row["_code"])
               + (0.05 + LH_OUT * sum(len(b) for b in row["_out"]) if row["_out"] else 0))
        row["_knote"] = wrap(row.get("k_note") or "(empty — no world knowledge yet)",
                             K_W - 0.10, FS_K, False)
        k_h = (0.14 + (LH_CODE * 1.15 if row["tag_k"] else 0)
               + (sum(LH_K * (len(b) + len(w)) + LH_K * 0.45 for b, w in row["_bul"])
                  if row["_bul"] else LH_K * len(row["_knote"])))
        row["_h"] = max(p_h, k_h) + 0.05


def k_bullets(text: str) -> list[str]:
    """Top-level bullets of a world-knowledge text. Indented lines are continuations
    of the bullet above them -- sub-bullets belong to their parent rule, not beside
    it -- and any preamble line before the first bullet is dropped. A trailing
    <changes> block is the reflector's edit log rather than world knowledge, so the
    rules stop there; `changelog_lines` counts what was left out."""
    out = []
    for line in (text or "").splitlines():
        if line.strip().lower() == "<changes>":
            break
        if line.startswith("- "):
            out.append(line[2:].strip())
        elif out and line.strip():
            out[-1] += " " + line.strip()
    return out


def changelog_lines(text: str) -> int:
    """How many lines of the <changes> block ship inside K without being rules."""
    lines = (text or "").splitlines()
    for i, line in enumerate(lines):
        if line.strip().lower() == "<changes>":
            return sum(1 for l in lines[i + 1:] if l.strip()
                       and l.strip().lower() != "</changes>")
    return 0


def _kwords(b: str) -> list[str]:
    return _norm(b).lower().split()


def pair_bullets(old: list[str], new: list[str]) -> tuple[list, list]:
    """Match each new bullet to the old bullet it most plausibly descends from, by
    token overlap. Returns (per-new-bullet (tag, old index or None), dropped indices).
    Greedy on the best scores first, so a near-identical pair claims its partner
    before a weaker candidate can."""
    import difflib
    cand = sorted(((difflib.SequenceMatcher(None, _kwords(o), _kwords(n)).ratio(), i, j)
                   for j, n in enumerate(new) for i, o in enumerate(old)),
                  key=lambda t: -t[0])
    taken_o, taken_n, link = set(), set(), {}
    for r, i, j in cand:
        if r < 0.30 or i in taken_o or j in taken_n:
            continue
        taken_o.add(i)
        taken_n.add(j)
        link[j] = (i, r)
    tags = []
    for j in range(len(new)):
        if j not in link:
            tags.append(("new", None))
        else:
            i, r = link[j]
            tags.append(("kept" if _norm(old[i]) == _norm(new[j]) else "revised", i))
    return tags, [i for i in range(len(old)) if i not in taken_o]


FRAMING = set("""a an the this that these those it its they them their there
is are was were be been being do does did has have had can could may might will would
of to by in on at for from with as and or but not no than then thus also other another
each such same only just any all both either every when where which who whose while if
world contains contain consists consist called named known appears appear seen observed
occurs occur happens happen means note assume assumed assumption
move moves moved moving movement motion action actions directional direction
cell cells object objects grid state step steps time timestep""".split())


def phrase_diff(old: str, new: str, cap: int = 3) -> list[str]:
    """What a revision did to a rule, in its own words. A word-level diff alone is
    mostly noise -- a sentence resplit reads as a wholesale rewrite -- so a run of
    edited words is only reported when it carries a term the other text does not have
    anywhere. That leaves the edits that change what the rule says, and drops the ones
    that only change how it is phrased."""
    import difflib
    import textwrap
    a, b = _norm(old).split(), _norm(new).split()
    # Words whose coming or going is a change of phrasing, not of rule: if a run of
    # edited text is novel only in these, it is a rewrite of the same claim.
    have = {t: set(re.findall(r"[a-z0-9]+", " ".join(t).lower())) for t in (tuple(a), tuple(b))}
    aset, bset = have[tuple(a)], have[tuple(b)]
    add, rem = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if op in ("insert", "replace") and j2 - j1 >= 3:
            add.append(" ".join(b[j1:j2]))
        if op in ("delete", "replace") and i2 - i1 >= 3:
            rem.append(" ".join(a[i1:i2]))

    def novel(phrase, other):
        return bool(set(re.findall(r"[a-z0-9]+", phrase.lower())) - other - FRAMING)

    add = [q for q in add if novel(q, aset)]
    rem = [q for q in rem if novel(q, bset)]
    out = []
    for sign, phrases in (("+", add), ("\u2212", rem)):
        phrases.sort(key=len, reverse=True)
        for q in phrases[:cap]:
            body = textwrap.wrap(f"{sign} \u201c{q}\u201d", 60)
            out.append(body[0])
            out += ["   " + ln for ln in body[1:]]
        if len(phrases) > cap:
            n = len(phrases) - cap
            out.append(f"{sign} (and {n} shorter edit{'s' if n > 1 else ''})")
    return out or ["(reworded; no term added or dropped)"]


def k_condensed(game, node, ktext, prev) -> list[str]:
    """One short line per rule, tagged with what the revision did to it. The tags and
    the quoted edits are computed from the two texts; only the short lines themselves
    are written by hand, and the hash beside each set is what keeps them honest."""
    import hashlib
    import textwrap
    key = (game, node)
    if key not in K_SHORT:
        raise SystemExit(f"no K_SHORT entry for {game} node {node}; add one "
                         f"(hash {hashlib.sha1(ktext.encode()).hexdigest()[:8]})")
    want, short = K_SHORT[key]
    got = hashlib.sha1(ktext.encode()).hexdigest()[:8]
    if got != want:
        raise SystemExit(f"K_SHORT[{game!r}, {node}] was written against K {want}, "
                         f"but node {node}'s K now hashes {got} -- reread and rewrite it")
    bullets = k_bullets(ktext)
    if len(short) != len(bullets):
        raise SystemExit(f"K_SHORT[{game!r}, {node}] has {len(short)} lines for "
                         f"{len(bullets)} bullets")

    head = (f"K [node {node}], condensed"
            + (f" -- one line per rule, against node {prev[0]}" if prev
               else " -- one line per rule (the first K on this lineage)"))
    out = ["", "-" * 74, head, "-" * 74]
    old_bullets = k_bullets(prev[1]) if prev else []
    if prev:
        tags, dropped = pair_bullets(old_bullets, bullets)
        prev_short = K_SHORT[(game, prev[0])][1]
    else:
        tags, dropped, prev_short = [("new", None)] * len(bullets), [], []
    pw = 13
    for j, (line, (tag, i)) in enumerate(zip(short, tags), 1):
        body = textwrap.wrap(line, 74 - pw) or [""]
        out.append(f"{j:>2} [{tag}]".ljust(pw) + body[0])
        out += [" " * pw + b for b in body[1:]]
        if tag == "revised":
            out += [" " * (pw + 1) + d
                    for d in phrase_diff(old_bullets[i], bullets[j - 1])]
    for i in dropped:
        body = textwrap.wrap(prev_short[i], 74 - pw) or [""]
        out.append(" - [dropped]".ljust(pw) + body[0])
        out += [" " * pw + b for b in body[1:]]
    lines = ktext.splitlines()
    first = next((i for i, l in enumerate(lines) if l.startswith("- ")), len(lines))
    pre = [l.strip() for l in lines[:first] if l.strip()]
    if sum(len(l) for l in pre) > 60:  # a one-line header is not a preamble
        out += ["", f"   ({len(pre)} preamble line(s) above the first rule are not "
                    "condensed here;", "   they are in the verbatim K above)"]
    n = changelog_lines(ktext)
    if n:
        out += ["", f"   ...and a <changes> block of {n} lines -- the reflector's own "
                    "edit log,", "   which is not world knowledge but ships inside K "
                    "all the same."]
    return out


def dump_features(key, spec, rows, cands, frames, stem) -> Path:
    """The full P(X) and world knowledge K for every node in the figure, unabridged. The
    figure has room for P(X\u2081), a diff, and quoted excerpts of K; this is the same
    material with nothing dropped, written from the same run_perceive() calls and the
    same candidate records, so the two cannot drift apart."""
    src = f"logs/2026-08-24/human_curated/rexpure/{spec['game']}_s1/rexpure_run_seed1"
    out = [f"P(X) and K for every node in {stem.name}.pdf",
           "=" * 74, "",
           f"game       {spec['game']}   drive {spec['drive']}   "
           f"steps {spec['steps'][0]}-{spec['steps'][-1]}",
           f"artifacts  {src}/candidates.jsonl",
           f"frames     {spec['drive']}/episode_0/trajectory.csv",
           "",
           "Every P(X) below is that node's own perception() run over that step's",
           "observation, and every K is that node's world_knowledge field verbatim --",
           "the same material the figure draws from, nothing transcribed. Each K is",
           "followed by a condensed form, one line per rule, tagged [new] / [revised] /",
           "[kept] / [dropped] against the previous K on the lineage. The tags and the",
           "quoted edits under a [revised] line are computed from the two texts; the",
           "short lines are hand-written and hash-bound to the rule they summarise.",
           ""]
    for i, f in enumerate(frames):
        arrow = f"  --{f['action']}-->" if i + 1 < len(frames) else ""
        out.append(f"  X{i} = step {f['step']}{arrow}")
    prev = None
    for row in rows:
        node = row["node"]
        same = [r["node"] for r in rows
                if r is not row and cands[r["node"]]["perception"]
                == cands[node]["perception"] and r["node"] < node]
        out += ["", "=" * 74,
                f"{row['when']}  ({row['sub']})   min(ID, cFD) = {row['score']:.3f}"
                + (f"   [P identical to node {same[-1]}]" if same else ""),
                "-" * 74]
        for i, f in enumerate(frames):
            val = row["pout"][i]
            out.append(f"P(X{i}) [step {f['step']}] = "
                       + (val if val else "(empty string)"))
        ktext = (cands[node]["world_knowledge"] or "").strip()
        kin = [r["node"] for r in rows
               if r is not row and r["node"] < node
               and (cands[r["node"]]["world_knowledge"] or "").strip() == ktext]
        out += ["", "-" * 74,
                f"K [node {node}]"
                + (f"   [identical to node {kin[-1]}]" if kin else ""),
                "-" * 74,
                ktext or "(empty -- no world knowledge yet)"]
        if ktext and not kin:
            out += k_condensed(spec["game"], node, ktext, prev)
            prev = (node, ktext)
    path = stem.with_name(stem.name + "_features.txt")
    path.write_text("\n".join(out) + "\n")
    print(f"wrote {path.relative_to(REPO)}")
    return path


def build(key: str, spec: dict) -> Path:
    cands = load_candidates(spec["game"])
    rows = [dict(r) for r in ROWS[spec["game"]]]
    verify(rows, cands)
    frames = load_frames(spec["game"], spec["drive"], spec["steps"])
    prepare(rows, cands, frames, spec.get("diff", 1))
    bg = background_of(frames[0]["grid"])

    notes = spec["notes"]
    note_lines = [wrap(n, W - PAD - 0.92, FS_NOTE, mono=False) for _a, n in notes]
    foot_lines = wrap(spec["foot"], W - PAD - 0.92, FS_NOTE, False) if spec["foot"] else []
    strip_h = (0.20 + GSZ + 0.14 + 0.10
               + (sum(len(nl) for nl in note_lines) + len(foot_lines))
               * FS_NOTE * 1.5 / 72 + 0.22)
    head_h = 0.20
    H = strip_h + head_h + sum(r["_h"] for r in rows) + 0.10

    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.set_facecolor(SURFACE)

    def txt(x, y, s, size, color=INK, mono=False, weight="normal", ha="left", va="top"):
        ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, zorder=6,
                family=MONO if mono else SANS, weight=weight)

    # ---- A: the frame strip
    y = H - 0.13
    txt(PAD, y, spec["title"], 6.4, INK, weight="bold")
    gy = y - 0.20 - GSZ
    for i, f in enumerate(frames):
        x = PAD + i * (GSZ + GAP)
        rings = [(cell, spec["ring"])
                 for cell in (changed_cells(frames[i - 1]["grid"], f["grid"])
                              if i and spec["ring"] else [])]
        draw_grid(ax, x, gy, GSZ, f["grid"], bg, rings)
        txt(x + GSZ / 2, gy - 0.03, ["X₀", "X₁", "X₂", "X₃"][i],
            FS_LBL, INK2, ha="center")
        if i < len(frames) - 1:
            ax.add_patch(FancyArrow(x + GSZ + 0.09, gy + GSZ / 2, GAP - 0.18, 0,
                                    width=0.004, head_width=0.032, head_length=0.055,
                                    length_includes_head=True, color=INK3, zorder=5))
            txt(x + GSZ + GAP / 2, gy + GSZ / 2 + 0.04, f["action"],
                FS_LBL if len(f["action"]) < 8 else FS_LBL - 1.2, INK, mono=True,
                ha="center", va="bottom")

    ny = gy - 0.24
    for i, ((act, _n), lines) in enumerate(zip(notes, note_lines)):
        txt(PAD, ny, f"X{'₀₁₂'[i]}→X{'₁₂₃'[i]}", FS_NOTE, INK3, mono=True)
        txt(PAD + 0.44, ny, act, FS_NOTE if len(act) < 8 else FS_NOTE - 1.2, INK,
            mono=True, weight="bold")
        for line in lines:
            txt(PAD + 0.89, ny, line, FS_NOTE, INK2)
            ny -= FS_NOTE * 1.5 / 72
    for line in foot_lines:
        txt(PAD + 0.89, ny, line, FS_NOTE, INK3)
        ny -= FS_NOTE * 1.5 / 72

    # ---- B: the four checkpoints
    y = ny - 0.14
    ax.plot([PAD, W - PAD], [y, y], color=INK3, linewidth=0.7, zorder=1)
    y -= 0.04
    txt(LEFT, y, "abstraction program   P", FS_LBL, INK3, weight="bold")
    txt(LEFT + P_W + COL_GAP, y, "world knowledge   K", FS_LBL, INK3, weight="bold")
    y -= head_h - 0.04

    for n, row in enumerate(rows):
        top = y
        if n:
            ax.plot([PAD, W - PAD], [top, top], color=GRID, linewidth=0.6, zorder=1)
        ty = top - 0.12

        # gutter: which node, what it changed, what it scored on the training objective
        txt(PAD, ty, row["when"], FS_LBL, INK, weight="bold")
        txt(PAD, ty - 0.085, row["sub"], FS_TAG, INK3)
        txt(PAD, ty - 0.20, f"{row['score']:.2f}", 7.2, BLUE, weight="bold")
        ax.add_patch(Rectangle((PAD, ty - 0.295), 0.46, 0.02, facecolor=GRID,
                               edgecolor="none", zorder=2))
        if row["score"] > 0:
            ax.add_patch(Rectangle((PAD, ty - 0.295), 0.46 * row["score"], 0.02,
                                   facecolor=BLUE, edgecolor="none", zorder=3))
        txt(PAD, ty - 0.335, "min(ID, cFD)", FS_TAG - 0.6, INK3)

        # P column
        px, k = LEFT, ty
        for line in row["_tag_p"]:
            txt(px, k, line, FS_TAG, INK3)
            k -= LH_CODE * 1.15
        for line, ind in row["_code"]:
            txt(px + ind, k, line, FS_CODE, INK3 if line == "..." else INK, mono=True)
            k -= LH_CODE
        k -= 0.05
        for block in row["_out"]:
            for j, line in enumerate(block):
                txt(px + (IND if j else 0), k, line, FS_OUT, INK2, mono=True)
                k -= LH_OUT

        # K column
        kx, k = LEFT + P_W + COL_GAP, ty
        if row["tag_k"]:
            txt(kx, k, row["tag_k"], FS_TAG, INK3)
            k -= LH_CODE * 1.15
        for line in (row["_knote"] if not row["_bul"] else []):
            txt(kx, k, line, FS_K, INK3)
            k -= LH_K
        for lines, warn in row["_bul"]:
            ax.plot([kx + 0.018], [k - 0.030], marker="o", markersize=1.2,
                    color=WARN if warn else BLUE, zorder=6)
            for line in lines:
                txt(kx + 0.075, k, line, FS_K, INK)
                k -= LH_K
            for line in warn:
                txt(kx + 0.075, k, line, FS_K - 0.4, WARN)
                k -= LH_K
            k -= LH_K * 0.45

        y = top - row["_h"]

    ax.plot([PAD, W - PAD], [y + 0.02, y + 0.02], color=INK3, linewidth=0.7, zorder=1)

    stem = REPO / ("analysis/learning_example/learning_evolution"
                   + ("" if spec.get("plain") else f"_{key}"))
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(stem.with_suffix("." + ext), dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    dump_features(key, spec, rows, cands, frames, stem)
    print(f"wrote {stem.relative_to(REPO)}.pdf/.png  ({W:.2f} x {H:.2f} in)  "
          f"{spec['game']} {spec['drive']} steps {spec['steps'][0]}-{spec['steps'][-1]}")
    return stem


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--candidate", default="dq8gc",
                    help=f"one of {', '.join(CANDIDATES)}, or 'all'")
    a = ap.parse_args()
    keys = list(CANDIDATES) if a.candidate == "all" else [a.candidate]
    for k in keys:
        if k not in CANDIDATES:
            raise SystemExit(f"unknown candidate {k!r}; "
                             f"choose from {', '.join(CANDIDATES)}")
        build(k, CANDIDATES[k])


if __name__ == "__main__":
    main()
