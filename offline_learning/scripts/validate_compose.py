"""Independently re-verify a compositional problem set from its JSON alone.

The builder screens as it goes, but that check shares the builder's own code and its
serialisation. This re-derives every guarantee from the stored (seed, prefix, plan)
against a fresh engine, so a problem that only LOOKS valid because of how it was written
out is caught:

  V1  the stored gt plan actually reaches the stored goal      (solvable as shipped)
      (goals are FULL exact frames -- no masking anywhere in this pipeline)
  V2  start != goal                                            (something must change)
  V3  noop^h misses the goal                                   (waiting is not enough)
  V4  no single action can be deleted and still reach the goal (h_min == h, no padding)
  V5  every mechanic the chain claims is re-detected on replay (labels are honest)

    uv run python offline_learning/scripts/validate_compose.py logs/.../problems.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_BAI_ROOT = Path(__file__).resolve().parents[2]
if str(_BAI_ROOT) not in sys.path:
    sys.path.insert(0, str(_BAI_ROOT))

from offline_learning.compose_plan import exec_from, trace  # noqa: E402
from offline_learning.human_replay import GAMES  # noqa: E402


def check(p: dict) -> dict:
    game, seed = p["game"], p["seed"]
    prog = GAMES[game][0]
    prefix, plan = p["prefix"], p["gt_actions"]
    goal = p["goal_grid"]
    r = {"game": game, "h": p["h"]}

    r["v1_gt_reaches"] = (exec_from(prog, seed, prefix, plan) == goal)
    r["v2_changed"] = (p["start_grid"] != goal)
    r["v3_noop_fails"] = (exec_from(prog, seed, prefix, ["noop"] * len(plan)) != goal)

    shorter = False
    for j in range(len(plan)):
        cand = plan[:j] + plan[j + 1:]
        if cand and exec_from(prog, seed, prefix, cand) == goal:
            shorter = True
            break
    r["v4_incompressible"] = not shorter

    t = trace(game, seed, prefix, plan)
    seen = {m for f in t["fired"] for m in f.all}
    missing = [m for m in p["chain"] if m not in seen]
    r["v5_labels_honest"] = not missing
    r["missing"] = missing
    r["ok"] = all(r[k] for k in ("v1_gt_reaches", "v2_changed", "v3_noop_fails",
                                 "v4_incompressible", "v5_labels_honest"))
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("problems")
    ap.add_argument("--limit", type=int, default=0, help="check only the first N per game")
    args = ap.parse_args()
    ps = json.loads(Path(args.problems).read_text())["problems"]
    if args.limit:
        per, sel = Counter(), []
        for p in ps:
            if per[p["game"]] < args.limit:
                sel.append(p)
                per[p["game"]] += 1
        ps = sel

    fails, per_game = [], Counter()
    checks = ["v1_gt_reaches", "v2_changed", "v3_noop_fails",
              "v4_incompressible", "v5_labels_honest"]
    tally = {c: Counter() for c in checks}
    for i, p in enumerate(ps):
        r = check(p)
        per_game[p["game"]] += 1
        for c in checks:
            tally[c][(p["game"], r[c])] += 1
        if not r["ok"]:
            fails.append((i, p["game"], p["chain"], r))
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(ps)} checked", flush=True)

    print(f"\n{'game':<8} {'n':>4} " + " ".join(f"{c.split('_')[0]:>6}" for c in checks))
    for g in sorted(per_game):
        row = [f"{tally[c][(g, True)]:>6}" for c in checks]
        print(f"{g:<8} {per_game[g]:>4} " + " ".join(row))
    print(f"\n{len(ps) - len(fails)}/{len(ps)} problems pass every check")
    for i, g, ch, r in fails[:15]:
        bad = [c for c in checks if not r[c]]
        print(f"  FAIL #{i} {g} {ch} -> {bad} missing={r['missing']}")


if __name__ == "__main__":
    main()
