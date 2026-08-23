"""Generate FRESH s2kt7 trajectories under non-zero seeds (real random-foraging
game), to replace the seed-0-degenerate clean_data3/s2kt7 (food pinned to (0,0)).

The seed-0 episodes cannot be 1:1 re-seeded: they are mid-game segments and the
game diverges completely across seeds. So we generate new playthroughs whose
observations come from the env's own render (render_state validated to reproduce
seed-0 data byte-for-byte in regen_s2kt7_seed.py). Each episode uses a distinct
non-zero seed for dataset variety; the action policy guarantees both noop and
click transitions and exercises spawn(random) -> ant-forage -> eat.
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
PROG = (BENCH / "programs/S2KT7.sexp").read_text()
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


def render_state(interp) -> str:
    render_dict = json.loads(interp.render_all())
    matrix = render_grid(
        render_dict, background_color=interp.get_background(), color_dict=COLOR_STR_TO_INT
    )
    return json.dumps(matrix)


def episode_actions(seed: int, length: int) -> list[str]:
    """Per-episode action plan keyed off the split-DISJOINT seed (not the episode
    index) so train/test/test50 get DIFFERENT action scripts. Keying off idx made
    episode_i share one script across every split -- a train->test50 action-sequence
    leak (and split seed-offsets are all congruent mod small n, so `seed % n` would
    alias too; a seeded PRNG on the absolute seed does not). Clicks trigger spawns;
    noops let the ants forage/eat. Click LOCATION is irrelevant to dynamics but kept
    in the label. 2-3 clicks in [1, length-4] so >=3 trailing noop-forage steps remain."""
    rng = random.Random(seed)
    n_clicks = rng.choice([2, 3])
    click_steps = set(rng.sample(range(1, length - 3), n_clicks))
    acts = []
    for s in range(length):
        if s in click_steps:
            acts.append(f"click {rng.randint(2, 14)} {rng.randint(1, 14)}")
        else:
            acts.append("noop")
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
    ap.add_argument("--out", default="/home/ays57/bai/offline_learning/clean_data3/s2kt7_seed1")
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
