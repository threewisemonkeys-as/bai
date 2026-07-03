"""Generate clean DQ8GC transitions that expose the click->move dynamics.

Why: in DQ8GC the click action has NO single-step observable effect -- it only
reassigns which particle the arrow keys control (the "active" particle). Its
fingerprint shows up one action later: after clicking a gray particle, the next
MOVE relocates that gray (instead of the darkgreen active). The gray->darkgreen
conversion in this game is ambient (adjacency to the active darkgreen), not a
click effect. (Verified directly against the ground-truth engine.)

This driver runs the ground-truth interpreter on DQ8GC.sexp and emits scripted
episodes in the SAME trajectory.csv schema the rest of the pipeline consumes
(Step,Action,Reasoning,Observation,Auxiliary_Observation,Reward,Done). Each
episode interleaves:
  - baseline moves of the darkgreen active (normal, single-step identifiable),
  - noops,
  - a click on a gray particle kept AWAY from darkgreen (single-step invisible),
  - moves of that now-active gray -> the click->move fingerprint.

Run:
  uv run python prototypes/perc_invdyn/gen_clickmove.py --out clean_data/dq8gc_clickmove --episodes 6
"""

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "Autumn.cpp" / "build"))
sys.path.insert(0, str(REPO / "MARAProtocol" / "python_examples"))

import interpreter_module  # noqa: E402
from autumnbench.autumnstdlib import autumnstdlib  # noqa: E402

GS = 16
FRAME_RATE = 8
SEXP = (
    REPO
    / "MARAProtocol/python_examples/autumnbench/example_benchmark/programs/DQ8GC.sexp"
).read_text()

HEADER = (
    "Task: interactive\nStep: {step}\nPhase: Interactive\nAvailable actions now:\n"
    "- left\n- right\n- up\n- down\n"
    "- click ROW COL  (ROW first, then COL, both in 0..15; matches the (row, col) order the perception reports)\n"
    "- noop\n- quit\n- reset\n\n"
    "========== Start of Direct Observation ==========\n"
)


def new_interp():
    itp = interpreter_module.Interpreter()
    itp.run_script(SEXP, autumnstdlib, "", 42)
    return itp


def grid(itp):
    out = json.loads(itp.render_all())
    out.pop("GRID_SIZE", None)
    g = [["black"] * GS for _ in range(GS)]
    for _typ, elems in out.items():
        for e in elems:
            x, y = e["position"]["x"], e["position"]["y"]  # x=col, y=row
            if 0 <= y < GS and 0 <= x < GS:
                g[y][x] = e["color"].lower()
    return g


def cells(g, col):
    return sorted((r, c) for r in range(GS) for c in range(GS) if g[r][c] == col)


def observation(g, step):
    return HEADER.format(step=step) + json.dumps(g)


def apply(itp, action):
    """One faithful env transition: dispatch the action, then FRAME_RATE ticks."""
    if action == "noop":
        pass
    elif action.startswith("click"):
        # Action strings are row-major ('click ROW COL'); native click is
        # column-first (click(x=col, y=row)), so transpose on dispatch.
        _, row, col = action.split()
        itp.click(int(col), int(row))
    else:
        getattr(itp, action)()
    for _ in range(FRAME_RATE):
        itp.step()


# Each episode is a list of action strings. Coordinates in click are "row col".
# Grays start at (row,col) (3,4),(5,3),(5,7),(6,6); darkgreen active at (2,2).
# We click a gray far from darkgreen, then move it -> it relocates (the fingerprint).
# Baseline: move the darkgreen active toward the (0,0) corner (up=row-1, left=col-1)
# so it never becomes adjacent to a gray (which would trigger an ambient conversion
# and contaminate the click->move signal). Then click a gray in the lower-right
# region and walk it through empty space -> clean relocations.
EPISODES = [
    ["up", "left", "noop", "click 5 3", "down", "right", "right", "noop"],
    ["left", "up", "noop", "click 5 7", "down", "down", "right", "noop"],
    ["up", "up", "noop", "click 6 6", "right", "right", "down", "noop"],
    ["left", "left", "noop", "click 5 3", "right", "down", "down", "noop"],
    ["up", "noop", "left", "click 5 7", "right", "right", "down", "noop"],
    ["left", "up", "noop", "click 6 6", "down", "right", "right", "noop"],
]


def gen_episode(actions, ep_dir):
    itp = new_interp()
    rows = []
    for i, action in enumerate(actions):
        g = grid(itp)
        rows.append(
            {
                "Step": i,
                "Action": action,
                "Reasoning": "scripted_clickmove",
                "Observation": observation(g, i),
                "Auxiliary_Observation": "",
                "Reward": "0.0",
                "Done": "False",
            }
        )
        apply(itp, action)
    # terminal row: final observation, no further transition
    g = grid(itp)
    rows.append(
        {
            "Step": len(actions),
            "Action": "noop",
            "Reasoning": "scripted_clickmove",
            "Observation": observation(g, len(actions)),
            "Auxiliary_Observation": "",
            "Reward": "0.0",
            "Done": "True",
        }
    )
    ep_dir.mkdir(parents=True, exist_ok=True)
    with (ep_dir / "trajectory.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "Step",
                "Action",
                "Reasoning",
                "Observation",
                "Auxiliary_Observation",
                "Reward",
                "Done",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="clean_data/dq8gc_clickmove")
    ap.add_argument("--episodes", type=int, default=len(EPISODES))
    args = ap.parse_args()
    out = (HERE / args.out) if not Path(args.out).is_absolute() else Path(args.out)

    n = min(args.episodes, len(EPISODES))
    total = 0
    for k in range(n):
        rows = gen_episode(EPISODES[k], out / f"episode_{k}")
        total += len(rows) - 1
        # sanity: classify each gray change as a RELOCATION (click fingerprint:
        # gray vacates one cell, appears in another) vs a CONVERSION contaminant
        # (gray -> darkgreen, i.e. vacated cell becomes darkgreen).
        relocations, conversions = [], []
        for i in range(len(rows) - 1):
            g0 = json.loads(rows[i]["Observation"][rows[i]["Observation"].find("[[") :])
            g1 = json.loads(
                rows[i + 1]["Observation"][rows[i + 1]["Observation"].find("[[") :]
            )
            gy0, gy1 = set(cells(g0, "gray")), set(cells(g1, "gray"))
            gone, appeared = sorted(gy0 - gy1), sorted(gy1 - gy0)
            for (r, c) in gone:
                if g1[r][c] == "darkgreen":
                    conversions.append((i, rows[i]["Action"], (r, c)))
            if appeared:
                relocations.append((i, rows[i]["Action"], gone, appeared))
        flag = "  <-- CONVERSION CONTAMINATION" if conversions else ""
        print(
            f"episode_{k}: {len(rows)-1} transitions | "
            f"click->move relocations: {relocations} | conversions: {conversions}{flag}"
        )
    print(f"\nwrote {n} episodes ({total} transitions) -> {out}")


if __name__ == "__main__":
    main()
