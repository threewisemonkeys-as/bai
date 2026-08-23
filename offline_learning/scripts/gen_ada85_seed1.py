"""Generate FRESH ada85 trajectories under non-zero seeds, to replace the
seed-0-DEGENERATE clean_data3/ada85.

The bug: the whole clean_data corpus was generated at seed 0. ada85 spawns Rocks on
click, each carrying a HIDDEN boolean `breaksBottle = (uniformChoice (list true false))`.
Under seed 0 this collapses -- EVERY rock spawned gets breaksBottle=true (verified 8/8),
so the true/false randomness is never exercised. Under non-zero seeds you get a genuine
true/false MIX. (The old dynamics.txt understated this as "only the FIRST rock is true
under seed0"; it is actually every rock.)

WHAT breaksBottle DOES / observability
--------------------------------------
A Rock is ALWAYS a single gray cell -- breaksBottle is NOT visible on the rock and is NOT
visible while it travels. Its ONLY observable consequence is at the END of the rock's
journey: when a rock reaches the BottleSpot (5,10), a breaksBottle=TRUE rock turns the
Bottle broken (a GOLD cell appears at (5,10), the col-10 palette flips), and it STAYS
broken; a breaksBottle=FALSE rock reaches the same spot and produces NO change. So the
randomness is observable ONLY as "did the bottle break when the rock arrived?" -- and a
rock spawned at a corner takes ~16-17 steps to arrive (break appears at spawn_frame+17).

=> A short (~14-step) episode NEVER reaches the bottle, so within it a true vs false rock
   look IDENTICAL (a gray cell travelling), and the observable data is seed-independent.
   To make regeneration actually change the USABLE data we therefore (a) spawn the
   determining rock EARLY (step 0/1) and (b) run episodes long enough (length 20) for the
   break to land, so break-vs-no-break -- the exercised randomness -- appears in the grid.

CLICK LABELS (replay-consistent with the standard engine path)
--------------------------------------------------------------
Actions are applied EXACTLY like the corpus / gen_s2kt7_seed1.py: via
env_utils.interpreter_action_to_text(interp, label) + interp.step(), with NO manual
coordinate swap. That function runs interp.click(int(arg1), int(arg2)), and the engine's
click(x, y) reads (x=col, y=row). So a label "click A B" applies at (col=A, row=B), and a
button at (row R, col C) is hit by the label "click C R". We re-determined the working
labels EMPIRICALLY (spawn confirmed via evaluate_to_string('rocks')):
  - Suzie (row 0,  col 0)  ->  "click 0 0"   (spawns a rock at origin col0,row0)
  - Billy (row 10, col 0)  ->  "click 0 10"  (spawns a rock at origin col0,row10)
  ("click 10 0" spawns NOTHING under this path -- that was the old swapped-label bug.)
This makes the recorded observations reproduce under standard replay
(interpreter_action_to_text from the episode seed) at every frame -- the check that a
manual interp.click(COL, ROW) swap FAILED (test50_sim / verify_drive_in_sim reject it).
The label-arg (row,col vs col,row) convention mismatch is a KNOWN corpus-wide issue handled
uniformly elsewhere; we only require self-consistency with the standard replay path here.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
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
PROG = (BENCH / "programs/ADA85.sexp").read_text()
COLOR_DICT = load_yaml_to_dict(str(BENCH / "color_dict.yaml"))
COLOR_STR_TO_INT = {v: k for k, v in COLOR_DICT.items()}

_RE_BB = re.compile(r"Rock\{breaksBottle:\s*\((true|false):")

HEADER = (
    "Task: interactive\n"
    "Step: {step}\n"
    "Phase: Interactive\n"
    "Available actions now:\n"
    "- left\n- right\n- up\n- down\n"
    "- click ROW COL  (ROW first, then COL, both in 0..10; matches the (row, col)"
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


def breaks_bottle_list(interp) -> list[str]:
    """Read the HIDDEN breaksBottle flags of the live rocks from internal Autumn state
    (NOT rendered in the grid). Robust to the verbose evaluate_to_string dump."""
    s = interp.evaluate_to_string("rocks")
    return _RE_BB.findall(s)


def episode_actions(idx: int, length: int) -> list[str]:
    """Deterministic per-episode plan.

    IMPORTANT engine quirk: clicking a spawner FREEZES the passive per-step rock movement
    on that frame (the `on clicked` handler overrides the movement `next`). So every click
    before a rock arrives delays its arrival by one frame. To keep the PRIMARY rock's
    break OBSERVABLE we spawn it at f0 and leave the run to the bottle click-free; a single
    SECONDARY spawn (the OTHER button) at f8 adds action + spawn-location variety and costs
    just one freeze, so the primary break still lands (~f18) inside a length-20 episode.
    The observed break therefore tracks the seed's FIRST uniformChoice draw (the primary
    rock's breaksBottle). Spawn LOCATION alternates Suzie<->Billy across episodes.

    Labels (confirmed to spawn under interpreter_action_to_text): Suzie="click 0 0",
    Billy="click 0 10"."""
    suzie, billy = "click 0 0", "click 0 10"
    primary = suzie if idx % 2 == 0 else billy
    other = billy if idx % 2 == 0 else suzie
    acts = ["noop"] * length
    acts[0] = primary          # primary spawn -> travels to bottle, break observable
    if 8 < length:
        acts[8] = other        # one secondary spawn (other location), visible aftermath
    return acts


def gen_episode(idx: int, seed: int, length: int):
    interp = Interpreter()
    interp.run_script(PROG, autumnstdlib, "", seed)
    acts = episode_actions(idx, length)
    rows = []
    primary_bb = None   # breaksBottle of the FIRST rock spawned (determines observed break)
    gold_seen = False
    for s in range(length):
        grid = render_state(interp)
        if primary_bb is None:
            bbs = breaks_bottle_list(interp)
            if bbs:
                primary_bb = bbs[0]
        if '"gold"' in grid:
            gold_seen = True
        rows.append({
            "Step": s,
            "Action": acts[s],
            "Reasoning": "regen_seed",
            "Observation": HEADER.format(step=s, grid=grid),
            "Auxiliary_Observation": "",
            "Reward": 0.0,
            "Done": False,
        })
        if s < length - 1:  # apply all but the last (its result would be the next state)
            interpreter_action_to_text(interp, acts[s])  # NO swap: standard replay path
            interp.step()
    return rows, primary_bb, gold_seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/ays57/bai/offline_learning/clean_data3/ada85_seed1")
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
        summ = []
        for i in range(n):
            seed = args.base_seed + off + i  # all non-zero, disjoint across splits
            rows, primary_bb, gold = gen_episode(i, seed, args.length)
            ep = out / split / f"episode_{i}"
            ep.mkdir(parents=True, exist_ok=True)
            with (ep / "trajectory.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            summ.append((i, seed, primary_bb, gold))
        print(f"[{split}] {n} eps (seeds {args.base_seed+off}..{args.base_seed+off+n-1}) -> {out/split}")
        for i, seed, bb, gold in summ:
            print(f"    ep{i} seed={seed} primary_breaksBottle={bb} bottle_broke_observed={gold}")
    print(f"DONE -> {out}")


if __name__ == "__main__":
    main()
