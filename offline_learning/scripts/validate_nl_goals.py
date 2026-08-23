#!/usr/bin/env python3
"""Audit of the NL-goal pilot (`offline_learning/nl_goals.py`).

The curated set's validator (`validate_curated.py`) proves things about ONE FRAME reached by
one plan.  An NL goal accepts a SET of trajectories, so the questions change: the checker can
now be too loose (accepting something the sentence does not say) as easily as too tight, and
ANY-STEP scoring means anything that is briefly true anywhere in a 50-action rollout counts.

  N1  the curated reference plan satisfies the checker.  Padded with `hold - 1` noops first:
      the curated plans were trimmed to the first frame that satisfied the OLD predicate, so
      a checker that asks for two consecutive frames can never see the second one otherwise.
      The reference is a witness that the goal is reachable, not a submission.
  N2  the checker is false on the start state, and first satisfied strictly after it
  N3  noop^CAP does not satisfy it -- doing nothing is not a solution
  N4  INFORMATIONAL, not a gate: greedily delete actions from the reference while it still
      satisfies the checker, and report what is left as `nl_h` -- the horizon the SENTENCE
      demands, as opposed to `h`, the horizon the exact frame demanded.  The two differ
      legitimately (n2ntd's reference carries six trailing noops that exist only to bring
      mario to rest for the frame), so a shrink is not evidence of a loose checker here the
      way it was for the curated set.  Flagged as `suspicious` only if it more than halves
  N5  random floor, ANY-STEP, at the reference length h AND at the eval budget (50).  The
      second number is the one that matters and the one the frame-goal set never had to ask:
      50 actions of flailing gets many more chances at a set-valued goal than at one frame
  N6  authored positives accepted, negatives rejected -- and, for each negative, whether the
      checker WITHOUT its guard clause would have been fooled (`naive` column)
  N7  transient acceptance: over the N5 drives, how often an accepting step stops accepting
      one noop later.  High rates mean the goal is a knife-edge and `hold` may be too low
  N8  cheap-shortcut search: BFS to a bounded depth over the game's verbs plus the clicks the
      reference uses.  Reports the shortest accepting sequence found, or the budget it hit --
      never claims exhaustiveness
  X   cross-driver: the reference replayed through AutumnBenchEnvWrapper (the other engine
      driver) must give frame-identical results and the same checker verdict

    uv run python offline_learning/scripts/validate_nl_goals.py \
        --problems logs/2026-08-18/curated/problems.json \
        --out logs/2026-08-19/nl_pilot/validation.json
"""
from __future__ import annotations

import argparse
import json
import random
import zlib
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(REPO), str(REPO / "offline_learning")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from autumn_env import AutumnBenchEnvWrapper  # noqa: E402
from offline_learning.curated_plan import trace  # noqa: E402
from offline_learning.human_replay import GAMES, _grid, _obs_cell  # noqa: E402
from offline_learning import nl_goals as NG  # noqa: E402
from offline_learning.nl_goals import GOALS, NLGoal, first_satisfied  # noqa: E402

CAP = 50            # eval budget (eval_curated_plan.PLAN_CAP)
TRIALS = 400        # random-floor trials per length
N8_DEPTH = 5        # cheap-shortcut BFS depth ceiling
N8_NODES = 12_000


# --------------------------------------------------------------------------- rollouts
def frames(goal: NLGoal, plan: list[str]) -> list:
    return [s.grid for s in trace(goal.game, goal.seed, plan)]


def sat(goal: NLGoal, plan: list[str]) -> int | None:
    return first_satisfied(goal, frames(goal, plan), plan)


def naive_check(goal: NLGoal, plan: list[str]) -> int | None:
    """The same checker with its anti-exploit clause removed, when the goal defines one.
    Only used by N6 to report what each guard buys -- never to score.  Goals without a guard
    (or whose guard IS the whole checker) carry no naive variant and report None."""
    if goal.naive is None:
        return None
    fr = frames(goal, plan)
    return next((k for k in range(1, len(fr)) if goal.naive(fr[:k + 1], plan[:k])), None)


def random_plan(game: str, n: int, rng: random.Random) -> list[str]:
    verbs = list(GAMES[game][2])
    size = 12 if game == "n2ntd" else 16
    return [f"click {rng.randrange(size)} {rng.randrange(size)}" if v == "click" else v
            for v in (rng.choice(verbs) for _ in range(n))]


# ----------------------------------------------------------------------------- checks
def n4_compress(goal: NLGoal, plan: list[str]) -> list[str]:
    cur = list(plan)
    changed = True
    while changed and len(cur) > 1:
        changed = False
        for j in range(len(cur)):
            cand = cur[:j] + cur[j + 1:]
            if sat(goal, cand) is not None:
                cur, changed = cand, True
                break
    return cur


_DRIVES: dict[tuple[str, int, int], list[tuple[list[str], list]]] = {}


def drives_for(goal: NLGoal, n: int, trials: int) -> list[tuple[list[str], list]]:
    """Random drives of length n, TRACED ONCE per (game, seed, n) and reused by every goal of
    that game.  The drives do not depend on the goal, and re-tracing them per goal is what
    made the pilot's validator O(40 min) at 30 goals.

    crc32, not hash(): str hashing is salted per process, and a floor that moves between runs
    of the same validator is not a dataset field.  Keyed on the GAME, not the pid, now that
    the drives are shared -- the pilot's per-pid seeding is not reproducible here anyway."""
    key = (goal.game, goal.seed, n)
    if key not in _DRIVES:
        rng = random.Random(zlib.crc32(f"{goal.game}|{goal.seed}|{n}".encode()))
        out = []
        for _ in range(trials):
            p = random_plan(goal.game, n, rng)
            out.append((p, frames(goal, p)))
        _DRIVES[key] = out
    return _DRIVES[key]


def n5_floor(goal: NLGoal, n: int, trials: int) -> tuple[float, list[list[str]]]:
    hits = 0
    for p, fr in drives_for(goal, n, trials):
        if first_satisfied(goal, fr, p) is not None:
            hits += 1
    return hits / trials, [p for p, _ in drives_for(goal, n, trials)]


def n7_transient(goal: NLGoal, drives: list[list[str]]) -> tuple[int, int]:
    """Accepting steps that stop accepting one noop later, over the given drives."""
    n_acc = n_lost = 0
    for p in drives:
        fr = frames(goal, p)
        for k in range(1, len(fr)):
            if goal.check(fr[:k + 1], p[:k]):
                n_acc += 1
                nxt = frames(goal, p[:k] + ["noop"])
                if not goal.check(nxt, p[:k] + ["noop"]):
                    n_lost += 1
    return n_acc, n_lost


def n8_shortcut(goal: NLGoal, ref: list[str], depth: int, budget: int) -> dict:
    """BFS over the game's verbs plus the reference's clicks.  Dedups on the frame sequence
    tail rather than the frame alone -- these goals are trajectory properties, so two nodes
    with the same frame are not interchangeable -- which in practice means dedup on (frame,
    depth), the same conservative key `curated_plan.bfs` uses."""
    verbs = [v for v in GAMES[goal.game][2] if v != "click"]
    clicks = sorted({a for a in ref if a.startswith("click")})
    alpha = verbs + clicks
    frontier: list[list[str]] = [[]]
    seen: set = set()
    nodes = 0
    for d in range(1, depth + 1):
        nxt = []
        for pre in frontier:
            for a in alpha:
                nodes += 1
                if nodes > budget:
                    return {"found": None, "depth_reached": d - 1, "nodes": nodes,
                            "exhausted": False, "alphabet": alpha}
                plan = pre + [a]
                fr = frames(goal, plan)
                if goal.check(fr, plan):
                    return {"found": plan, "depth_reached": d, "nodes": nodes,
                            "exhausted": False, "alphabet": alpha}
                key = (fr[-1], d)
                if key in seen:
                    continue
                seen.add(key)
                nxt.append(plan)
        frontier = nxt
        if not frontier:
            break
    return {"found": None, "depth_reached": depth, "nodes": nodes,
            "exhausted": True, "alphabet": alpha}


def x_cross_driver(goal: NLGoal, plan: list[str]) -> dict:
    env = AutumnBenchEnvWrapper(env_name=GAMES[goal.game][0], task_type="interactive",
                                max_episode_steps=len(plan) + 8, seed=goal.seed,
                                render_mode="text")
    obs, _ = env.reset(seed=goal.seed)
    out = [json.loads(_grid(_obs_cell(obs)))]
    for a in plan:
        obs, _r, _t, _tr, _i = env.step(a)
        out.append(json.loads(_grid(_obs_cell(obs))))
    env.close()
    wrap = [tuple(tuple(r) for r in g) for g in out]
    return {"frames_match": wrap == frames(goal, plan),
            "verdict": first_satisfied(goal, wrap, plan)}


# ------------------------------------------------------------------------------ main
def audit(goal: NLGoal, row: dict, fast: bool) -> dict:
    base = goal.ref if goal.ref is not None else row["plan"]
    h = len(base) if goal.ref is not None else row["h"]
    ref = base + ["noop"] * (goal.hold - 1)
    r: dict = {"game": goal.game, "pid": goal.pid, "tier": goal.tier, "seed": goal.seed,
               "nl": goal.nl, "hold": goal.hold, "h": h, "ref_padded": len(ref),
               "own_ref": goal.ref is not None}

    k = sat(goal, ref)
    r["N1"] = {"pass": k is not None, "first_sat": k}

    fr0 = frames(goal, [])
    r["N2"] = {"pass": (not goal.check(fr0, [])) and (k or 0) > 0,
               "start_satisfied": goal.check(fr0, [])}

    r["N3"] = {"pass": sat(goal, ["noop"] * CAP) is None}

    comp = n4_compress(goal, ref)
    r["N4"] = {"nl_h": len(comp), "deleted": len(ref) - len(comp),
               "suspicious": len(comp) * 2 < h,
               "plan": comp if len(comp) < len(ref) else None}

    # One set of length-CAP drives answers every budget: a length-CAP random drive truncated
    # to L is a length-L random drive, and "accepted within L" is exactly first-sat <= L.
    # The pilot traced a second 400-drive set per goal just to get the floor at h.
    ks = [first_satisfied(goal, fr, p) for p, fr in drives_for(goal, CAP, TRIALS)]

    def _floor(lim: int) -> float:
        return sum(1 for k in ks if k is not None and k <= lim) / len(ks)

    floor_cap = _floor(CAP)
    drives = [p for p, _ in drives_for(goal, CAP, TRIALS)]
    r["N5"] = {"pass": floor_cap <= 0.05, "floor_at_h": _floor(h),
               "floor_at_2h": _floor(2 * h), "floor_at_cap": floor_cap, "trials": TRIALS}

    pos = [{"plan_len": len(p), "sat": sat(goal, p), "naive": naive_check(goal, p)}
           for p in goal.positives]
    neg = [{"plan_len": len(p), "sat": sat(goal, p), "naive": naive_check(goal, p)}
           for p in goal.negatives]
    r["N6"] = {"pass": all(x["sat"] is not None for x in pos)
                       and all(x["sat"] is None for x in neg),
               "positives": pos, "negatives": neg,
               "guard_caught": sum(1 for x in neg if x["naive"] is not None)}

    n_acc, n_lost = n7_transient(goal, drives[:40])
    settled = ref + ["noop"] * 8
    r["N7"] = {"accepting_steps": n_acc, "lost_after_noop": n_lost,
               "transient_rate": (n_lost / n_acc) if n_acc else None,
               "ref_absorbing": goal.check(frames(goal, settled), settled)}

    r["N8"] = ({"skipped": "fast"} if fast else
               n8_shortcut(goal, base, min(N8_DEPTH, h), N8_NODES))
    if not fast and r["N8"].get("found") is not None:
        r["N8"]["shorter_than_ref"] = len(r["N8"]["found"]) < h

    r["X"] = x_cross_driver(goal, ref)
    # Three verdicts, not two.  A goal whose ONLY failing check is the random floor is not
    # broken -- it is easy, and the eval reports its floor in the `rand@50` column so every
    # score is read against it.  Conflating that with a checker that accepts a wrong state
    # would hide the second kind behind the first.
    sound = all(r[c]["pass"] for c in ("N1", "N2", "N3", "N6")) \
        and r["X"]["frames_match"] and r["X"]["verdict"] == k
    r["sound"] = sound
    r["pass"] = sound and r["N5"]["pass"]
    r["verdict"] = "PASS" if r["pass"] else ("FLOOR" if sound else "FAIL")
    return r


def report(rows: list[dict]) -> None:
    print("\n" + "=" * 96)
    print(f"NL-GOAL VALIDATION  ({len(rows)} goals)")
    print("=" * 96)
    for r in rows:
        print(f"\n[{r['verdict']:5s}] {r['game']}/{r['pid']}  {r['tier']}  seed={r['seed']}  "
              f"h={r['h']}  hold={r['hold']}")
        print(f'       "{r["nl"]}"')
        print(f"  N1 reference satisfied .......... {r['N1']['pass']!s:5s} "
              f"first_sat={r['N1']['first_sat']} (padded len {r['ref_padded']})")
        print(f"  N2 not true at the start ........ {r['N2']['pass']!s:5s}")
        print(f"  N3 noop^{CAP} misses ............... {r['N3']['pass']!s:5s}")
        print(f"  N4 horizon the sentence needs ... {'-':5s} "
              f"nl_h={r['N4']['nl_h']} vs frame h={r['h']}"
              + ("  SUSPICIOUS" if r["N4"]["suspicious"] else ""))
        print(f"  N5 random floor ................. {r['N5']['pass']!s:5s} "
              f"@h={r['N5']['floor_at_h']:.3f}  @2h={r['N5']['floor_at_2h']:.3f}  "
              f"@cap={r['N5']['floor_at_cap']:.3f}")
        print(f"  N6 fixtures ..................... {r['N6']['pass']!s:5s} "
              f"{len(r['N6']['positives'])} pos / {len(r['N6']['negatives'])} neg, "
              f"{r['N6']['guard_caught']} neg would fool the unguarded checker")
        tr = r["N7"]["transient_rate"]
        print(f"  N7 transient acceptance ......... {'-':5s} "
              f"{r['N7']['lost_after_noop']}/{r['N7']['accepting_steps']} accepting steps "
              f"lost after one noop" + (f" ({tr:.2f})" if tr is not None else ""))
        n8 = r["N8"]
        if "skipped" in n8:
            print(f"  N8 shortcut search .............. skipped")
        elif n8["found"]:
            print(f"  N8 shortcut search .............. found len={len(n8['found'])} "
                  f"(ref h={r['h']}): {n8['found']}")
        else:
            print(f"  N8 shortcut search .............. none to depth "
                  f"{n8['depth_reached']} ({n8['nodes']} nodes, "
                  f"{'exhaustive over ' + str(len(n8['alphabet'])) + ' actions' if n8['exhausted'] else 'budget hit'})")
        print(f"  X  cross-driver ................. {r['X']['frames_match']!s:5s} "
              f"wrapper verdict={r['X']['verdict']}")
    n_ok = sum(1 for r in rows if r["verdict"] == "PASS")
    n_fl = sum(1 for r in rows if r["verdict"] == "FLOOR")
    n_bad = sum(1 for r in rows if r["verdict"] == "FAIL")
    print(f"\n{n_ok} PASS / {n_fl} FLOOR (sound, but random play finds it) / "
          f"{n_bad} FAIL, of {len(rows)}\n")
    if n_bad:
        print("FAIL: " + ", ".join(f"{r['game']}/{r['pid']}" for r in rows
                                   if r["verdict"] == "FAIL") + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="logs/2026-08-18/curated/problems.json")
    ap.add_argument("--out", default="logs/2026-08-19/nl_pilot/validation.json")
    ap.add_argument("--pid", action="append", help="restrict to these problem ids")
    ap.add_argument("--fast", action="store_true", help="skip N8 (the slow one)")
    a = ap.parse_args()

    curated = {r["id"]: r for r in json.loads(Path(a.problems).read_text())}
    goals = [g for g in GOALS if not a.pid or g.pid in a.pid]

    rows = []
    for g in goals:
        t0 = time.time()
        print(f"[{g.game}/{g.pid}] auditing...", flush=True)
        rows.append(audit(g, curated[g.pid], a.fast))
        print(f"  {rows[-1]['verdict']} ({time.time() - t0:.0f}s)",
              flush=True)

    report(rows)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1, default=str))
    print(f"wrote {out}")
    sys.exit(0 if all(r["pass"] for r in rows) else 1)


if __name__ == "__main__":
    main()
