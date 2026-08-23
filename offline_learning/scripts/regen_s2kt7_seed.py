"""Regenerate s2kt7 clean_data3 trajectories under a NON-ZERO seed.

Background: the whole clean_data corpus was generated at seed=0, which is
PATHOLOGICAL for s2kt7 -- (randomPositions GRID_SIZE 2) collapses to (0,0) for
every draw under seed 0, so clicks deterministically drop food at (0,0) instead
of scattering it randomly (the real game). See memory seed0-degenerate-data-gen.

Approach: REPLAY each episode's recorded action sequence through the real
compiled interpreter under a new seed, re-rendering every observation grid with
the env's own render_grid(). Only the [[...]] grid inside each Observation is
swapped; the header/Action/Reward/Done columns are preserved verbatim, so the
output is format-identical to the original and consumable by validate.py.

Self-validating: with --seed 0 it must reproduce the original grids byte-for-byte
(asserted) before we trust the --seed 1 output.
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
PROG = (BENCH / "programs/S2KT7.sexp").read_text()
COLOR_DICT = load_yaml_to_dict(str(BENCH / "color_dict.yaml"))
COLOR_STR_TO_INT = {v: k for k, v in COLOR_DICT.items()}

csv.field_size_limit(10**9)


def render_state(interp) -> str:
    """Reproduce the env's text grid exactly (concrete_envs.get_observation path)."""
    render_dict = json.loads(interp.render_all())
    matrix = render_grid(
        render_dict,
        background_color=interp.get_background(),
        color_dict=COLOR_STR_TO_INT,
    )
    return json.dumps(matrix)


def replay(actions: list[str], seed: int, n_states: int) -> list[str]:
    """Return the grid string for each of the first n_states states, replaying
    actions[0..n_states-2] from a fresh seeded interpreter."""
    interp = Interpreter()
    interp.run_script(PROG, autumnstdlib, "", seed)
    grids = [render_state(interp)]  # state_0 = post-run_script (matches reset->get_observation)
    for i in range(n_states - 1):
        interpreter_action_to_text(interp, actions[i])
        interp.step()
        grids.append(render_state(interp))
    return grids


def swap_grid(observation: str, new_grid: str) -> str:
    s = observation.find("[[")
    e = observation.rfind("]]")
    if s == -1 or e == -1:
        raise ValueError("no grid in observation")
    return observation[:s] + new_grid + observation[e + 2 :]


def process_episode(ep_dir: Path, seed: int, out_dir: Path, validate: bool):
    rows = list(csv.DictReader((ep_dir / "trajectory.csv").open()))
    actions = [r["Action"].strip() for r in rows]
    grids = replay(actions, seed, len(rows))
    if validate:
        for i, r in enumerate(rows):
            orig = r["Observation"]
            og = orig[orig.find("[[") : orig.rfind("]]") + 2]
            if og != grids[i]:
                raise AssertionError(
                    f"seed-0 mismatch in {ep_dir} row {i}:\n  orig={og[:120]}\n  new ={grids[i][:120]}"
                )
        return None  # validation-only
    out_ep = out_dir / ep_dir.name
    out_ep.mkdir(parents=True, exist_ok=True)
    with (out_ep / "trajectory.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        for i, r in enumerate(rows):
            r = dict(r)
            r["Observation"] = swap_grid(r["Observation"], grids[i])
            w.writerow(r)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/ays57/bai/offline_learning/clean_data3/s2kt7")
    ap.add_argument("--out", default="/home/ays57/bai/offline_learning/clean_data3/s2kt7_seed1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--splits", default="train,test,test50,train_regen")
    ap.add_argument("--validate-seed0", action="store_true",
                    help="assert replay under seed 0 reproduces the original grids, then exit")
    args = ap.parse_args()
    src = Path(args.src)
    seed = 0 if args.validate_seed0 else args.seed
    out = Path(args.out)
    total_eps = total_rows = 0
    for split in args.splits.split(","):
        sp = src / split
        if not sp.is_dir():
            print(f"[skip] {split} (no dir)")
            continue
        eps = sorted(sp.glob("episode_*"))
        for ep in eps:
            n = process_episode(ep, seed, out / split, validate=args.validate_seed0)
            total_eps += 1
            if n:
                total_rows += n
        print(f"[{split}] {len(eps)} episodes"
              + ("" if args.validate_seed0 else f" -> {out/split}"))
    if args.validate_seed0:
        print(f"VALIDATION OK: seed-0 replay reproduced all {total_eps} episodes byte-for-byte.")
    else:
        print(f"WROTE {total_eps} episodes / {total_rows} rows under seed {seed} -> {out}")


if __name__ == "__main__":
    main()
