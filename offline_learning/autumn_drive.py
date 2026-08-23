"""Manual AutumnBench trajectory driver.

Drives AutumnBenchEnvWrapper(task_type='interactive') through a chosen action
sequence, prints a compact grid view per step (so the dynamics can be inspected
and the moves refined), and writes a load_transitions-compatible trajectory.csv:

    Step,Action,Reasoning,Observation,Auxiliary_Observation,Reward,Done

The Action on row i is the action taken FROM row i's Observation, producing
row i+1's Observation (matching offline_learning/validate.load_transitions).

Usage (from bai root):
    uv run python offline_learning/autumn_drive.py DQ8GC /tmp/clean/dq8gc \
        --actions noop,right,right,down,click_5_4
Click actions are ROW-MAJOR: `click_ROW_COL` on the CLI is emitted to the env as
`click ROW COL`. The env (AutumnBenchEnvWrapper) transposes to the native
column-first interpreter, so the recorded label and the observed transition agree
under a (row, col) reading.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_BAI_ROOT = Path(__file__).resolve().parents[1]
if str(_BAI_ROOT) not in sys.path:
    sys.path.insert(0, str(_BAI_ROOT))

from autumn_env import AutumnBenchEnvWrapper

COLORS = {
    "black": ".", "darkgreen": "G", "gray": "y", "grey": "y", "blue": "B",
    "red": "R", "orange": "O", "green": "g", "white": "W", "yellow": "Y",
    "purple": "P", "darkblue": "b", "lightblue": "l",
    # zip-sourced games (balloon / tetris / logic_gates / rink)
    "brown": "n", "mediumpurple": "M", "tan": "T", "limegreen": "L", "pink": "K",
}


def _grid_from_obs(obs_text: str):
    i = obs_text.find("[[")
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(obs_text)):
        if obs_text[j] == "[":
            depth += 1
        elif obs_text[j] == "]":
            depth -= 1
            if depth == 0:
                return json.loads(obs_text[i:j + 1])
    return None


_GLYPH_POOL = "abcdefhijkmopqrstuvwxzACDEFHIJNQSUVXZ0123456789"


def _background(grid) -> str:
    """The modal colour. Backgrounds are not all black (mario white, balloon skyblue), and the
    harness renders empty cells in the program's declared background colour."""
    counts: dict[str, int] = {}
    for row in grid or []:
        for c in row:
            counts[c] = counts.get(c, 0) + 1
    return max(counts, key=counts.get) if counts else "black"


def _glyphs(grid) -> dict[str, str]:
    """COLORS plus a stable auto-assigned glyph for any colour the table lacks; the
    background always prints as '.'."""
    g = dict(COLORS)
    bg = _background(grid)
    if bg != "black":
        g["black"] = "k"          # '.' is reserved for the background
    g[bg] = "."
    used = set(g.values())
    for row in grid or []:
        for c in row:
            if c not in g:
                nxt = next((ch for ch in _GLYPH_POOL if ch not in used), "?")
                g[c] = nxt
                used.add(nxt)
    return g


def _ascii(grid) -> str:
    if not grid:
        return "(no grid)"
    g = _glyphs(grid)
    lines = []
    n = len(grid[0])
    lines.append("    " + "".join(str(c % 10) for c in range(n)))
    for r, row in enumerate(grid):
        lines.append(f"{r:>3} " + "".join(g.get(c, "?") for c in row))
    return "\n".join(lines)


def _legend(grid) -> str:
    g = _glyphs(grid)
    bg = _background(grid)
    seen = {}
    for row in grid or []:
        for c in row:
            if c != bg:
                seen[c] = seen.get(c, 0) + 1
    return "  ".join(f"{g.get(k, '?')}={k}({v})" for k, v in sorted(seen.items())) + f"  [background={bg}]"


def _observation_cell(obs: dict) -> str:
    st = obs["text"]["short_term_context"]
    lt = obs["text"]["long_term_context"]
    return f"{st}\n\n========== Start of Direct Observation ==========\n{lt}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env_name")
    ap.add_argument("outdir")
    ap.add_argument("--actions", required=True,
                    help="comma-separated; row-major click_ROW_COL -> 'click ROW COL'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--noop-pad", type=int, default=0, metavar="N",
                    help="insert N 'noop' steps after every non-noop action (and N leading "
                         "noops before the first decided action), so passive/clock dynamics "
                         "unfold in isolation around each action. Pick N >= the context_k used "
                         "downstream so the inverse-dynamics window sees a clean noop baseline "
                         "on both sides of each action transition.")
    args = ap.parse_args()

    actions = []
    for a in args.actions.split(","):
        a = a.strip()
        if not a:
            continue
        if a.startswith("click_"):
            # row-major: click_ROW_COL -> 'click ROW COL' (the env transposes to
            # the native column-first interpreter).
            _, row, col = a.split("_")
            actions.append(f"click {row} {col}")
        else:
            actions.append(a)

    # noop-padding: surround each decided action with N passive (noop) steps. The env
    # is stepped through every noop, so per-tick clock dynamics (drain, contagion,
    # gravity, bullet flight, cooking) are recorded as standalone (s, noop, s')
    # transitions AND become the clean baseline frames in each action's ctx window.
    # Already-noop actions are not re-padded (avoids runaway noop runs).
    if args.noop_pad > 0:
        pad = ["noop"] * args.noop_pad
        expanded = list(pad)  # leading baseline so the FIRST action also has a window
        for a in actions:
            expanded.append(a)
            if a != "noop":
                expanded.extend(pad)
        actions = expanded

    env = AutumnBenchEnvWrapper(env_name=args.env_name, task_type="interactive",
                                max_episode_steps=args.max_steps, seed=args.seed,
                                render_mode="text")
    obs, _ = env.reset(seed=args.seed)

    rows = []
    step = 0
    cur = obs
    print(f"=== {args.env_name} reset (seed {args.seed}) ===")
    g = _grid_from_obs(_observation_cell(cur))
    print(_ascii(g)); print("legend:", _legend(g)); print()

    for a in actions:
        cell = _observation_cell(cur)
        rows.append({"Step": step, "Action": a, "Reasoning": "manual",
                     "Observation": cell, "Auxiliary_Observation": "",
                     "Reward": 0.0, "Done": False})
        nxt, reward, term, trunc, info = env.step(a)
        step += 1
        g = _grid_from_obs(_observation_cell(nxt))
        print(f"--- step {step}: action={a!r} reward={reward} term={term} ---")
        print(_ascii(g)); print("legend:", _legend(g)); print()
        cur = nxt
        if term:
            break
    # final row (terminal observation, no action)
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
    env.close()


if __name__ == "__main__":
    main()
