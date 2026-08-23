"""Manual ARC-AGI-3 trajectory driver (live, via arc_agi.Arcade).

Drives ArcAgiEnvWrapper through a chosen action sequence, prints a compact
grid view + available actions + state per step (to inspect dynamics and refine
moves), and writes a load_transitions-compatible trajectory.csv.

Actions on the CLI:
    ACTION1..ACTION5      -> passed through
    click_X_Y             -> "ACTION6 x=X y=Y"   (complex/click action)

The Action on row i is taken FROM row i's Observation, producing row i+1's.

Usage (from bai root):
    uv run python offline_learning/arc_drive.py ls20 /tmp/clean/ls20 \
        --actions ACTION1,ACTION2,click_10_20,ACTION3
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

_BAI_ROOT = Path(__file__).resolve().parents[1]
if str(_BAI_ROOT) not in sys.path:
    sys.path.insert(0, str(_BAI_ROOT))

from arc_agi_env import ArcAgiEnvWrapper

# 16 ARC color indices -> single chars (0 background = '.')
CH = ".123456789ABCDEF"


def _grids_from_obs(obs_text: str):
    """Extract list of grids (each a list[list[int]]) from <grid_N> blocks."""
    grids = []
    for block in re.findall(r"<grid_\d+>(.*?)</grid_\d+>", obs_text, re.DOTALL):
        rows = []
        for line in block.strip().splitlines():
            nums = re.findall(r"-?\d+", line)
            if nums:
                rows.append([int(n) for n in nums])
        if rows:
            grids.append(rows)
    return grids


def _ascii(grid) -> str:
    if not grid:
        return "(no grid)"
    n = len(grid[0])
    head = "    " + "".join(str(c % 10) for c in range(n))
    lines = [head]
    for r, row in enumerate(grid):
        lines.append(f"{r:>3} " + "".join(CH[c] if 0 <= c < 16 else "?" for c in row))
    return "\n".join(lines)


def _legend(grid) -> str:
    seen = {}
    for row in grid or []:
        for c in row:
            if c != 0:
                seen[c] = seen.get(c, 0) + 1
    return "  ".join(f"{CH[k]}={k}({v})" for k, v in sorted(seen.items()))


def _observation_cell(obs: dict) -> str:
    st = obs["text"]["short_term_context"]
    lt = obs["text"]["long_term_context"]
    return f"{st}\n\n========== Start of Direct Observation ==========\n{lt}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game_id")
    ap.add_argument("outdir")
    ap.add_argument("--actions", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=400)
    args = ap.parse_args()

    actions = []
    for a in args.actions.split(","):
        a = a.strip()
        if not a:
            continue
        if a.startswith("click_"):
            _, x, y = a.split("_")
            actions.append(f"ACTION6 x={x} y={y}")
        else:
            actions.append(a.upper())

    env = ArcAgiEnvWrapper(game_id=args.game_id, max_episode_steps=args.max_steps,
                           seed=args.seed)
    obs, info = env.reset(seed=args.seed)
    print(f"=== {args.game_id} reset === win_levels={info.get('win_levels')}")
    print("avail actions:", env.language_action_space)
    g = _grids_from_obs(_observation_cell(obs))
    print(_ascii(g[-1] if g else None)); print("legend:", _legend(g[-1] if g else None)); print()

    rows = []
    step = 0
    cur = obs
    for a in actions:
        rows.append({"Step": step, "Action": a, "Reasoning": "manual",
                     "Observation": _observation_cell(cur), "Auxiliary_Observation": "",
                     "Reward": 0.0, "Done": False})
        nxt, reward, term, trunc, inf = env.step(a)
        step += 1
        g = _grids_from_obs(_observation_cell(nxt))
        gg = g[-1] if g else None
        print(f"--- step {step}: action={a!r} reward={reward} term={term} "
              f"avail={env.language_action_space} ---")
        print(_ascii(gg)); print("legend:", _legend(gg)); print()
        cur = nxt
        if term:
            break
    rows.append({"Step": step, "Action": "", "Reasoning": "manual",
                 "Observation": _observation_cell(cur), "Auxiliary_Observation": "",
                 "Reward": 0.0, "Done": True})

    outdir = Path(args.outdir) / "episode_0"
    outdir.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(10_000_000)
    with (outdir / "trajectory.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Step", "Action", "Reasoning",
                                          "Observation", "Auxiliary_Observation",
                                          "Reward", "Done"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows ({len(rows)-1} transitions) -> {outdir/'trajectory.csv'}")


if __name__ == "__main__":
    main()
