"""Generate FRESH f5w3n trajectories under non-zero seeds (space-invaders shooter),
to replace the seed-0-degenerate clean_data3/f5w3n.

f5w3n's enemies auto-fire one bullet every 15 ticks (rule `time % 15 == 3`) at a
target chosen by `uniformChoice` over the combined enemy pool (enemies1 row y=1 U
enemies2 row y=3, 10 enemies total). Under seed 0 this PARTIALLY collapses:
uniformChoice always returns the SAME index-0 enemy, so every fire spawns at the
enemies1 col-1 enemy, whose only motion is the ±1 march -> a fixed 2-value
alternation (2,1)/(1,1) confined to the SINGLE y=1 row; the y=3 row is NEVER hit
and only 2/10 pool candidates ever fire. Under non-zero seeds the draw is genuine:
targets span BOTH rows with real variety (~5-6 distinct cells across the fires).

The seed-0 episodes cannot be 1:1 re-seeded (they are driven playthroughs and the
game diverges completely across seeds), so we generate new playthroughs whose
observations come from the env's own `render_grid` (validated to reproduce the
initial frame exactly: enemies1 @ y=1 x in {1,4,7,10,13}, enemies2 @ y=3 x in
{2,5,8,11,14}, hero @ (15,8)). Each episode uses a distinct non-zero seed for
dataset variety; the action policy exercises left/right/up(fire)/noop and each
20-step episode spans two enemy-fire events (obs-steps 4 and 19), so the enemy-fire
randomness is exercised properly (both rows, varied columns).

Actions are MOVEMENT (left,right,up,noop) -- NO click, NO down (both are no-ops in
this game).
"""

from __future__ import annotations

import argparse
import csv
import json
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
PROG = (BENCH / "programs/F5W3N.sexp").read_text()
COLOR_DICT = load_yaml_to_dict(str(BENCH / "color_dict.yaml"))
COLOR_STR_TO_INT = {v: k for k, v in COLOR_DICT.items()}

# Same observation framing as the existing clean_data3/f5w3n and gen_s2kt7_seed1.
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

# 20-step base plan: 5 up (fire), 4 left, 4 right, 7 noop. Rotated by episode idx so
# each episode's action-to-clock alignment (and which action coincides with an enemy
# fire) differs, while every episode keeps the same balanced action multiset.
BASE_PLAN = [
    "noop", "up", "left", "left", "noop", "up", "right", "noop",
    "left", "up", "noop", "right", "right", "noop", "up", "left",
    "noop", "right", "up", "noop",
]


def render_state(interp) -> str:
    render_dict = json.loads(interp.render_all())
    matrix = render_grid(
        render_dict, background_color=interp.get_background(), color_dict=COLOR_STR_TO_INT
    )
    return json.dumps(matrix)


def episode_actions(idx: int, length: int) -> list[str]:
    """Deterministic per-episode plan: a rotation of BASE_PLAN by idx. Every episode
    has the same balanced multiset (up/left/right/noop) but a different phase, so the
    same enemy-fire schedule lands under different actions/hero positions."""
    k = idx % len(BASE_PLAN)
    rot = BASE_PLAN[k:] + BASE_PLAN[:k]
    # tile in case length > len(BASE_PLAN)
    while len(rot) < length:
        rot += BASE_PLAN
    return rot[:length]


def gen_episode(idx: int, seed: int, length: int) -> list[dict]:
    interp = Interpreter()
    interp.run_script(PROG, autumnstdlib, "", seed)
    acts = episode_actions(idx, length)
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
    ap.add_argument("--out", default="/home/ays57/bai/offline_learning/clean_data3/f5w3n_seed1")
    ap.add_argument("--base-seed", type=int, default=1)
    ap.add_argument("--length", type=int, default=20)
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
