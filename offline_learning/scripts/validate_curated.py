#!/usr/bin/env python3
"""Independent audit of the curated planning set.

Deliberately re-runs every plan through AutumnBenchEnvWrapper -- the OTHER engine driver --
rather than the raw-interpreter `Sim` the builder used, so a disagreement between the two
shows up as a failed check instead of a silently wrong dataset.

  V1  the reference plan reproduces the stored goal frame exactly, and the stored start
  V2  the goal frame occurs at NO earlier step (so it cannot be reached by accident or
      already hold at t=0)
  V3  noop^h from the start misses the goal
  V4f the plan is FRAME-incompressible: no single action can be deleted and still hit the
      goal frame.  This is the statement an agent could exploit.
  V4t the plan is TASK-incompressible: no single action can be deleted and still satisfy the
      objective predicate.  This is the anti-padding statement, and it is the only one with
      teeth on n2ntd -- there the enemy's patrol clock-stamps the frame, so V4f passes
      vacuously whether or not the plan is padded.
  V5  the recorded `quiescent` flag is true: absorbing goals survive an extra noop
  V6  random plans of the same length essentially never hit the goal.  Only enforced for
      h >= 2: a one-action problem IS guessable from a six-verb alphabet, and the recorded
      `random_success` rate reports that honestly rather than pretending otherwise.

    uv run python offline_learning/scripts/validate_curated.py logs/.../problems.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(REPO), str(REPO / "offline_learning")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from autumn_env import AutumnBenchEnvWrapper  # noqa: E402
from offline_learning.human_replay import GAMES, _grid, _obs_cell  # noqa: E402
from offline_learning.curated_plan import _problems  # noqa: E402
from offline_learning.mechanics_rules import SIZE  # noqa: E402

CHECKS = ["V1", "V2", "V3", "V4f", "V4t", "V5", "V6"]


def wrapper_frames(program: str, seed: int, plan: list[str]) -> list[list[list[str]]]:
    """Every rendered frame from a reset at `seed`, index 0 = start."""
    env = AutumnBenchEnvWrapper(env_name=program, task_type="interactive",
                                max_episode_steps=len(plan) + 8, seed=seed,
                                render_mode="text")
    obs, _ = env.reset(seed=seed)
    out = [json.loads(_grid(_obs_cell(obs)))]
    for a in plan:
        obs, _r, _t, _tr, _i = env.step(a)
        out.append(json.loads(_grid(_obs_cell(obs))))
    env.close()
    return out


def random_plan(game: str, h: int, rng: random.Random) -> list[str]:
    verbs = list(GAMES[game][2])
    n = SIZE[game]
    draw = [rng.choice(verbs) for _ in range(h)]
    return [f"click {rng.randrange(n)} {rng.randrange(n)}" if v == "click" else v
            for v in draw]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("problems")
    ap.add_argument("--random-trials", type=int, default=40)
    a = ap.parse_args()

    rows = json.loads(Path(a.problems).read_text())
    preds = {(p.game, p.pid): p.goal for p in _problems()}
    from offline_learning.curated_plan import trace as sim_trace

    tally: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    failures: list[str] = []
    rates: dict[str, float] = {}
    rng = random.Random(0)

    for row in rows:
        game, pid, seed, plan, h = (row["game"], row["id"], row["seed"],
                                    row["plan"], row["h"])
        prog, goal, start = row["program"], row["goal"], row["start"]
        tag = f"{game}/{pid}"
        tally[game]["n"] += 1
        pred = preds[(game, pid)]

        frames = wrapper_frames(prog, seed, plan + ["noop"])
        ok = {}
        ok["V1"] = frames[0] == start and frames[h] == goal
        ok["V2"] = all(frames[i] != goal for i in range(h))
        ok["V3"] = all(f != goal for f in wrapper_frames(prog, seed, ["noop"] * h)[1:])

        def hits_frame(p: list[str]) -> bool:
            return any(f == goal for f in wrapper_frames(prog, seed, p)[1:])

        def hits_task(p: list[str]) -> bool:
            return any(pred(s) for s in sim_trace(game, seed, p)[1:])

        ok["V4f"] = not any(hits_frame(plan[:j] + plan[j + 1:]) for j in range(h))
        ok["V4t"] = not any(hits_task(plan[:j] + plan[j + 1:]) for j in range(h))
        ok["V5"] = (frames[h + 1] == goal) == row["quiescent"]
        rate = sum(hits_frame(random_plan(game, h, rng))
                   for _ in range(a.random_trials)) / a.random_trials
        ok["V6"] = h < 2 or rate <= 0.05
        rates[tag] = rate

        for k in CHECKS:
            tally[game][k] += int(ok[k])
        bad = [k for k in CHECKS if not ok[k]]
        if bad:
            failures.append(f"{tag}: {', '.join(bad)}")
        print(f"  {tag:<26} rand={rate:.2f}  "
              f"{'ok' if not bad else 'FAIL ' + ','.join(bad)}", flush=True)

    print()
    print(f"{'game':<8}{'n':>4}" + "".join(f"{k:>6}" for k in CHECKS))
    for game, t in tally.items():
        print(f"{game:<8}{t['n']:>4}" + "".join(f"{t[k]:>6}" for k in CHECKS))
    guessable = [k for k, v in rates.items() if v > 0.05]
    if guessable:
        print(f"\n  guessable by a random plan (all h=1 by design): "
              f"{', '.join(guessable)}")
    total = sum(t["n"] for t in tally.values())
    clean = total - len(failures)
    print(f"\n{clean}/{total} problems pass every check")
    for f in failures:
        print(f"  FAILED  {f}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
