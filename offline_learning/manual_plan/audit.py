"""Screens a manually curated planning problem must pass. EXACT frames throughout --
success is `rendered_grid == goal_grid`, no masking anywhere in this pipeline.

Every screen re-derives its answer from the stored (seed, prefix, plan) against a fresh
engine, so a problem that only LOOKS valid because of how the curator wrote it out is
caught. Same discipline as scripts/validate_compose.py, which this deliberately mirrors.

  M1  the curated plan actually reaches the goal                  (solvable as shipped)
  M2  start != goal                                               (something must change)
  M3  noop^h misses the goal                                      (waiting is not enough)
  M4  no action can be replaced by noop and still reach the goal  (every action matters)
  M5  no action can be deleted and still reach the goal           (no shorter plan exists)
  M6  random length-h plans reach the goal rarely                 (not trivially hittable)
  M7  the stored start frame replays, and a second independent replay agrees (determinism)

M4 vs M5 under exact frames. M5 is the classic anti-padding screen, but on a game with an
object that evolves on a fixed cycle the agent cannot influence (n2ntd's enemy, among these
five) the goal frame doubles as a clock stamp: deleting ANY action shifts that object and
misses the goal, so M5 passes without evidence. M4 preserves plan length, so the clock still
lines up and only the action's own contribution is tested -- it stays sharp exactly where M5
goes vacuous. On tick-locked games M5 is reported with `vacuous: true` and M4 is the screen
that carries the anti-padding guarantee.
"""
from __future__ import annotations

import random

from offline_learning.human_replay import GAMES
from offline_learning.manual_plan.session import (
    TICK_LOCKED, canon, final_grid, replay,
)
from offline_learning.mechanics_rules import SIZE

SCREEN_IDS = ["m1_reaches", "m2_changed", "m3_noop_fails", "m4_actions_matter",
              "m5_no_shorter", "m6_random_rare", "m7_replay_stable"]
SCREEN_LABEL = {
    "m1_reaches": "plan reaches goal",
    "m2_changed": "start != goal",
    "m3_noop_fails": "noop^h misses",
    "m4_actions_matter": "every action matters",
    "m5_no_shorter": "no shorter plan",
    "m6_random_rare": "random plans rare",
    "m7_replay_stable": "replay deterministic",
}
RANDOM_MAX = 0.05          # a random plan may hit the goal at most this often


def random_plan(game: str, h: int, rng: random.Random) -> list[str]:
    game = canon(game)
    verbs, n = GAMES[game][2], SIZE[game]
    out = []
    for _ in range(h):
        v = rng.choice(verbs)
        out.append(f"click {rng.randrange(n)} {rng.randrange(n)}" if v == "click" else v)
    return out


def audit(problem: dict, n_random: int = 12, rng_seed: int = 0,
          screens: list[str] | None = None) -> dict:
    """Run the screens over one problem record. Cost is ~(2h + n_random) replays of
    (len(prefix) + h) steps; at ~0.6 ms/step a 100-step prefix with h=8 is well under a
    second, which is why the curator can call this on every edit."""
    want = set(screens or SCREEN_IDS)
    game = canon(problem["game"])
    seed = int(problem["seed"])
    prefix = list(problem.get("prefix", []))
    plan = list(problem.get("gt_actions", []))
    goal = problem.get("goal_grid")
    start = problem.get("start_grid")
    h = len(plan)
    res: dict[str, dict] = {}
    warnings: list[str] = []

    def add(sid, ok, detail="", **extra):
        if sid in want:
            res[sid] = {"id": sid, "label": SCREEN_LABEL[sid], "ok": bool(ok),
                        "detail": detail, **extra}

    if h == 0:
        add("m1_reaches", False, "empty plan")
        return {"ok": False, "screens": res, "warnings": ["empty plan"],
                "stats": {"h": 0, "prefix_len": len(prefix), "noops": 0}}

    reached = final_grid(game, seed, prefix, plan)
    add("m1_reaches", reached == goal,
        "reaches the stored goal" if reached == goal else "final frame != goal")

    add("m2_changed", start != goal,
        "start and goal differ" if start != goal else "goal IS the start frame")

    noop_end = final_grid(game, seed, prefix, ["noop"] * h)
    add("m3_noop_fails", noop_end != goal,
        "noop^h misses" if noop_end != goal else f"noop^{h} already reaches the goal")

    if "m4_actions_matter" in want:
        dead = [i for i in range(h) if plan[i] != "noop"
                and final_grid(game, seed, prefix, plan[:i] + ["noop"] + plan[i + 1:]) == goal]
        add("m4_actions_matter", not dead,
            "every action is load-bearing" if not dead
            else f"noop-substitutable at step(s) {dead} -> {[plan[i] for i in dead]}",
            dead_indices=dead)

    if "m5_no_shorter" in want:
        short = [i for i in range(h)
                 if h > 1 and final_grid(game, seed, prefix, plan[:i] + plan[i + 1:]) == goal]
        vacuous = game in TICK_LOCKED
        add("m5_no_shorter", not short,
            "no single deletion reaches the goal" if not short
            else f"deletable at step(s) {short} -> plan is padded",
            deletable_indices=short, vacuous=vacuous)
        if vacuous and not short:
            warnings.append(
                f"m5 is vacuous on {game}: autonomous scenery makes the goal frame a clock "
                "stamp, so no deletion can ever reach it. m4 carries the anti-padding guarantee.")

    # Does the world evolve on its own from the start state? If it does, a deletion shifts
    # every autonomous object and M5 passes for reasons unrelated to padding -- the same
    # vacuity as TICK_LOCKED, arrived at per-problem rather than per-game (bt3gb's falling
    # water is agent-created but tick-locked once it exists).
    world_autonomous = final_grid(game, seed, prefix, ["noop"]) != start

    if "m6_random_rare" in want:
        rng = random.Random(rng_seed)
        hits = sum(1 for _ in range(n_random)
                   if final_grid(game, seed, prefix, random_plan(game, h, rng)) == goal)
        rate = hits / max(1, n_random)
        add("m6_random_rare", rate <= RANDOM_MAX,
            f"{hits}/{n_random} random length-{h} plans reach the goal", rate=rate)

    if "m7_replay_stable" in want:
        # Whether the plan lands on the goal is M1's job. This screen only asks whether the
        # record still ADDRESSES the state it claims (stored start frame) and whether the
        # engine is deterministic at all -- the second replay bypasses the cache so it is a
        # genuinely independent run, not a dictionary lookup.
        g1 = replay(game, seed, prefix + plan)
        g2 = replay(game, seed, prefix + plan, use_cache=False)
        start_ok = g1[len(prefix)] == start
        det_ok = g1 == g2
        add("m7_replay_stable", start_ok and det_ok,
            "cold replay reproduces the start frame and is deterministic" if start_ok and det_ok
            else ("stored start_grid != cold replay at t" if not start_ok
                  else "two cold replays of the same actions disagree"))

    if world_autonomous and "m5_no_shorter" in res and res["m5_no_shorter"]["ok"]:
        res["m5_no_shorter"]["weak"] = True
        warnings.append(
            "the world evolves under noop from this start, so m5 is weakly informative here "
            "(any deletion shifts the autonomous objects). m4 is the screen to trust.")

    return {"ok": all(s["ok"] for s in res.values()), "screens": res, "warnings": warnings,
            "stats": {"h": h, "prefix_len": len(prefix), "world_autonomous": world_autonomous,
                      "noops": sum(1 for a in plan if a == "noop"),
                      "clicks": sum(1 for a in plan if a.startswith("click"))}}


def repair(problem: dict, max_extra: int = 4, limit: int = 8) -> list[dict]:
    """Small single edits to a plan that MISSES its goal, and which of them land on it.

    Under exact frames the common failure after trimming is being one or two ticks out of
    phase, so the search is deliberately tiny and local: append noops, delete one action,
    noop one action, insert one noop. It proposes; the curator still has to look at what the
    edit means. Nothing here reaches the audit -- a repaired plan is re-screened from
    scratch like any other.
    """
    game, seed = canon(problem["game"]), int(problem["seed"])
    prefix, plan = list(problem["prefix"]), list(problem["gt_actions"])
    goal = problem["goal_grid"]
    cands: list[tuple[str, int, list[str]]] = []
    for k in range(1, max_extra + 1):
        cands.append(("append noops", k, plan + ["noop"] * k))
    for i in range(len(plan)):
        cands.append(("delete", i, plan[:i] + plan[i + 1:]))
        if plan[i] != "noop":
            cands.append(("noop out", i, plan[:i] + ["noop"] + plan[i + 1:]))
        cands.append(("insert noop", i, plan[:i] + ["noop"] + plan[i:]))
    out = []
    for kind, i, cand in cands:
        if not cand:
            continue
        if final_grid(game, seed, prefix, cand) == goal:
            out.append({"kind": kind, "index": i, "plan": cand, "h": len(cand)})
            if len(out) >= limit:
                break
    return out
