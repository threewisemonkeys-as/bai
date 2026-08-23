"""Generate FRESH 83wkq trajectories under non-zero seeds (real uniformChoice random
walk), to replace the seed-0-degenerate clean_data3/83wkq.

83WKQ.sexp: a click spawns ONE blue Particle at the click cell; on every non-click
tick each existing particle moves to a uniformly-random in-bounds orthogonal neighbor
(uniformChoice over adjPositions). Under seed=0 this random walk COLLAPSES: the walk
becomes a deterministic 1-D straight line + wall-bounce (a particle at row 8 slides
col 8->0 then oscillates 0<->1 forever; row never changes over 40 steps). Under any
non-zero seed the particle wanders on BOTH axes (verified: row deltas {-1,0,1} AND
col deltas {-1,0,1}). See ../s2kt7_seed1 for the sibling fix of the same bug class.

The seed-0 episodes cannot be 1:1 re-seeded (mid-game segments that diverge across
seeds), so we generate new playthroughs whose observations come from the env's own
render (render_grid, validated in gen_s2kt7_seed1.py to reproduce seed-0 data byte-for
-byte). Each episode uses a distinct non-zero seed; the action policy guarantees both
noop and click transitions and exercises spawn -> (D2) click-frame freeze -> long
random walk. Click coords are varied per episode (interior + boundary spawns) so the
walk exercises the boundary-adjacency constraint too.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

MARA = Path("/home/ays57/bai/MARAProtocol/python_examples")
sys.path.insert(0, str(MARA))
from autumnbench.interpreter_module import Interpreter  # noqa: E402
from autumnbench.autumnstdlib import autumnstdlib  # noqa: E402
from autumnbench.env_utils import (  # noqa: E402
    interpreter_action_to_text,
    render_grid,
    load_yaml_to_dict,
)

BENCH = MARA / "autumnbench/example_benchmark"
PROG = (BENCH / "programs/83WKQ.sexp").read_text()
COLOR_DICT = load_yaml_to_dict(str(BENCH / "color_dict.yaml"))
COLOR_STR_TO_INT = {v: k for k, v in COLOR_DICT.items()}

HEADER = (
    "Task: interactive\n"
    "Step: {step}\n"
    "Phase: Interactive\n"
    "Available actions now:\n"
    "- left\n- right\n- up\n- down\n"
    "- click ROW COL  (ROW first, then COL, both in 0..15; matches the (row, col)"
    " order the perception reports)\n"
    "- noop\n- quit\n- reset\n\n"
    "========== Start of Direct Observation ==========\n{grid}"
)

# Per-episode click LOCATIONS (the "click a b" args). The interpreter maps arg1->x(col),
# arg2->y(row), so "click a b" spawns the particle at render (row=b, col=a). Locations are
# spread across the grid so the click LABEL varies.
# SPAWN1 = the first (step-1) spawn. EVEN episodes get an INTERIOR spawn1 and NO second
# click -> a single particle random-walks ~12 uninterrupted noop steps = a long, clean,
# unconstrained 2-D walk (the primary non-degeneracy proof). ODD episodes get a BOUNDARY
# spawn1 + a second click (SPAWN2) mid-episode -> a boundary walk (D4 reduced adjacency)
# plus D2 click-frame suppression of the existing particle and a 2nd walker.
SPAWN1 = [(7, 6), (0, 0), (9, 10), (15, 15), (4, 11), (0, 8), (11, 4), (15, 3)]
SPAWN2 = [(3, 3), (12, 2), (2, 13), (13, 13), (5, 5), (8, 1), (1, 7), (6, 6)]
# Spawn cells for the randomized (de-leaked) draw: the whole grid MINUS the curated
# train pools, so a single-spawn test50 script (= just its one spawn cell) can never
# equal a single-spawn TRAIN script. Random cells still mix interior + boundary, so
# the D4 boundary-adjacency coverage the curated pools gave is preserved statistically.
_RESERVED = set(SPAWN1) | set(SPAWN2)
SPAWN_POOL = [(a, b) for a in range(16) for b in range(16) if (a, b) not in _RESERVED]


def render_state(interp) -> str:
    render_dict = json.loads(interp.render_all())
    matrix = render_grid(
        render_dict, background_color=interp.get_background(), color_dict=COLOR_STR_TO_INT
    )
    return json.dumps(matrix)


def episode_actions(seed: int, length: int) -> list[str]:
    """Per-episode plan keyed off the split-DISJOINT seed (not the episode index) so
    train/test/test50 get DIFFERENT action scripts. Keying off idx made episode_i share
    one script across every split -- a train->test50 action-sequence leak (and split
    seed-offsets are all congruent mod small n, so `seed % n` would alias too; a seeded
    PRNG on the absolute seed does not). Step 0 observes the empty grid; step 1 clicks to
    spawn a particle that random-walks (exercises the uniformChoice RNG on both axes).
    ~half the episodes stop there (one particle, ~12 uninterrupted noop-walk steps = clean
    2-D proof); the other ~half add a mid-episode click: a 2nd particle appears and the
    existing particle FREEZES on that click frame (D2), then both resume. Rest are noops."""
    rng = random.Random(seed)
    c1 = rng.choice(SPAWN_POOL)
    acts = ["noop"] * length
    acts[1] = f"click {c1[0]} {c1[1]}"        # early spawn -> long random walk
    if rng.random() < 0.5:                    # ~half add the D2 / 2nd-walker click
        c2_step = rng.randint(8, 10)          # step 8-10 -> ~7-9 noop walk steps first
        c2 = rng.choice(SPAWN_POOL)
        acts[c2_step] = f"click {c2[0]} {c2[1]}"
    return acts


def gen_episode(idx: int, seed: int, length: int) -> list[dict]:
    interp = Interpreter()
    interp.run_script(PROG, autumnstdlib, "", seed)
    acts = episode_actions(seed, length)
    rows = []
    for s in range(length):
        grid = render_state(interp)
        rows.append({
            "Step": s,
            "Action": acts[s],
            "Reasoning": "regen_seed",
            "Observation": HEADER.format(step=s, grid=grid),
            "Auxiliary_Observation": "",
            "Reward": 0.0,
            "Done": False,
        })
        if s < length - 1:  # apply all but the last (its result is the next state)
            interpreter_action_to_text(interp, acts[s])
            interp.step()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/ays57/bai/offline_learning/clean_data3/83wkq_seed1")
    ap.add_argument("--base-seed", type=int, default=1)
    ap.add_argument("--length", type=int, default=14)
    # split -> (n_episodes, seed_offset) so train/test/test50 use DISJOINT seeds
    ap.add_argument("--train-n", type=int, default=8)
    ap.add_argument("--test-n", type=int, default=2)
    ap.add_argument("--test50-n", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    plan = [("train", args.train_n, 100), ("test", args.test_n, 300), ("test50", args.test50_n, 500)]
    for split, n, off in plan:
        for i in range(n):
            seed = args.base_seed + off + i  # all non-zero, disjoint across splits
            rows = gen_episode(i, seed, args.length)
            ep = out / split / f"episode_{i}"
            ep.mkdir(parents=True, exist_ok=True)
            with (ep / "trajectory.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        print(f"[{split}] {n} episodes (seeds {args.base_seed+off}..{args.base_seed+off+n-1}) -> {out/split}")
    print(f"DONE -> {out}")


if __name__ == "__main__":
    main()
