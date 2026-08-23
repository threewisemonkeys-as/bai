"""COMPOSITIONAL planning problems: chains of mechanics, grown forward.

The human-derived set (`coverage_plan.py`) builds a window as
    start = grid[i-(h-1)]        goal = grid[i+1]
so the goal is pinned to ONE mechanic firing at step i and h only slides the start
backward. Two measured consequences: horizon carries no difficulty (h=8 and h=12 both
average 2.23 non-noop actions, differing only in leading noops), and every problem tests
a single mechanic (dq8gc `click + contagion-spread` co-occurs in 0% of shipped windows).

This module builds problems the other way round. A problem targets a CHAIN of mechanics
M = [m1..mk]; the window is grown FORWARD through their firings and the goal sits after
the last one, so horizon grows because there is more to do. Every problem must pass:

  S1  goal != start
  S2  noop^h from start misses the goal
  S3  random plans rarely reach the goal
  S4  each m in M is NECESSARY: ablate the step where it fires -> goal missed
  S5  the plan is INCOMPRESSIBLE: no single action can be deleted and still reach the
      goal.  This is the anti-padding screen -- it is what makes h mean difficulty.

GOALS ARE FULL EXACT FRAMES. Every screen compares the complete rendered grid -- no
object class is excluded. Caveat this buys: where a game has an object that evolves on a
fixed cycle the agent cannot influence (only n2ntd's enemy, among these four), its
position in the goal frame encodes the elapsed tick count, so S5 cannot fail -- deleting
any action shifts the enemy and misses the goal regardless of whether the action mattered.
For n2ntd the anti-padding guarantee therefore rests on the generation-time liveness rule
(every step must fire a non-scenery rule), not on S5. bt3gb/dq8gc/s2kt7 have no such
object and S5 is fully load-bearing there.

    uv run python -m offline_learning.compose_plan census --game dq8gc
    uv run python -m offline_learning.compose_plan build --all --out logs/compose/problems.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_BAI_ROOT = Path(__file__).resolve().parents[1]
if str(_BAI_ROOT) not in sys.path:
    sys.path.insert(0, str(_BAI_ROOT))

from autumn_env import AutumnBenchEnvWrapper  # noqa: E402
from offline_learning.human_replay import GAMES, _grid, _obs_cell  # noqa: E402
from offline_learning.mechanics_rules import (  # noqa: E402
    BACKGROUND, BG, CLASSES, SIZE, fired,
)

csv.field_size_limit(10_000_000)
DATA = _BAI_ROOT / "offline_learning/human_data"

# ------------------------------------------------------------------ mechanic -> classes
# Which object classes a mechanic's outcome is expressed in. Goals are full frames, so
# this no longer gates comparison -- it is used only by `is_scenery` to decide whether a
# mechanic is agent-driven (a chain link) or autonomous backdrop.
MECH_CLASSES: dict[str, dict[str, set[str]]] = {
    "bt3gb": {"cloud-move-left": {"cloud"}, "cloud-move-right": {"cloud"},
              "cloud-clamp": {"cloud"}, "rain-day": {"water"}, "rain-night": {"water"},
              "click-toggle": {"celestial"}, "click-flip-droplets": {"water"},
              "liquid-fall": {"water"}, "liquid-slide": {"water"},
              "solid-fall": {"water"}, "liquid-rest": {"water"}, "solid-rest": {"water"}},
    "dq8gc": {m: {"particle"} for m in
              ("move-left", "move-right", "move-up", "move-down", "move-overlap",
               "move-offgrid", "move-return", "move-invisible", "move-blocked",
               "click-select", "click-empty", "contagion-spread")},
    "n2ntd": {"move-left": {"mario"}, "move-right": {"mario"}, "move-blocked": {"mario"},
              "jump": {"mario"}, "jump-blocked": {"mario"}, "gravity-fall": {"mario"},
              "coin-collect": {"mario", "coin"},
              "shoot": {"mario", "bullet"}, "shoot-no-ammo": {"mario"},
              "bullet-move": {"bullet"}, "bullet-despawn": {"bullet"},
              "enemy-patrol": {"enemy"}, "enemy-bounce": {"enemy"},
              "enemy-hit": {"enemy", "bullet"}},
    "s2kt7": {"click-spawn-food": {"food"}, "food-eaten": {"food", "ant"},
              "ant-move": {"ant", "food"}, "ant-idle": {"ant"}},
    "83wkq": {"click-spawn-particle": {"particle"}, "particle-diffuse": {"particle"}},
}

COMPOSABLE_GAMES = ["bt3gb", "dq8gc", "n2ntd", "s2kt7", "83wkq"]

# CAVEAT on 83wkq: its only object class random-walks every tick
# (`uniformChoice (adjPositions ...)`), which makes BOTH anti-padding guards vacuous there.
# S5 cannot fail (deleting a tick reshuffles every particle, so any shorter plan misses the
# goal) and the liveness rule cannot fail either (`particle-diffuse` fires on every tick a
# particle exists). Goals stay reproducible -- the engine RNG is seeded -- so the problems
# are solvable; what they test is "recover the click times and places", with unpredictable
# diffusion as filler. Read 83wkq horizons as click-count, not as difficulty.
# n2ntd is a milder version of the same thing: S5 vacuous, liveness still binding.


# ------------------------------------------------------------------------- engine
def exec_from(prog: str, seed: int, prefix: list[str], plan: list[str],
              all_frames: bool = False):
    """Replay `prefix` from a reset at `seed`, then run `plan`. Returns the final
    rendered grid (or every grid along the plan when all_frames)."""
    env = AutumnBenchEnvWrapper(env_name=prog, task_type="interactive",
                                max_episode_steps=len(prefix) + len(plan) + 8,
                                seed=seed, render_mode="text")
    env.reset(seed=seed)
    term = False
    for a in prefix:
        _o, _r, term, _t, _i = env.step(a)
        if term:
            break
    out = []
    for a in plan:
        if term:
            out.append(None)
            continue
        o, _r, term, _t, _i = env.step(a)
        out.append(_grid(_obs_cell(o)))
    env.close()
    if all_frames:
        return out
    return out[-1] if out else None


def trace(game: str, seed: int, prefix: list[str], plan: list[str]) -> dict:
    """Grids + per-step `fired` for a plan run from the post-prefix state."""
    prog = GAMES[game][0]
    g0 = (exec_from(prog, seed, prefix[:-1], prefix[-1:]) if prefix
          else _start_grid(prog, seed))
    g = [g0] + exec_from(prog, seed, prefix, plan, all_frames=True)
    fs = []
    for i in range(len(plan)):
        cf = exec_from(prog, seed, prefix + plan[:i], ["noop"])
        fs.append(fired(game, g[i], plan[i], cf, g[i + 1]))
    return {"grids": g, "fired": fs}


def _start_grid(prog: str, seed: int) -> str:
    env = AutumnBenchEnvWrapper(env_name=prog, task_type="interactive",
                                max_episode_steps=4, seed=seed, render_mode="text")
    o, _ = env.reset(seed=seed)
    g = _grid(_obs_cell(o))
    env.close()
    return g


# ----------------------------------------------------------------------- proposer
def _active_cell(game: str, prev: str | None, grid: str) -> tuple[int, int] | None:
    """Locate dq8gc's ACTIVE particle by diffing consecutive frames: the one cell that
    moved is the one under control. Generator-side only -- which particle is active is
    deliberately not observable to a solver, and nothing here reaches the problem."""
    if game != "dq8gc" or prev is None:
        return None
    a, b = set(_occupied(game, prev)), set(_occupied(game, grid))
    gained = b - a
    return next(iter(gained)) if len(gained) == 1 else None


def _occupied(game: str, grid: str) -> list[tuple[int, int]]:
    return [(r, c) for r, row in enumerate(json.loads(grid))
            for c, v in enumerate(row) if v not in BG[game]]


def propose(game: str, grid: str, rng: random.Random, n: int = 6,
            prev: str | None = None) -> list[str]:
    """Candidate actions that are LIKELY to fire a rule in this state. This is the only
    per-game knowledge in the generator -- the analogue of the old `synth_hint`s, but
    used to bias a search rather than to script one fixed sequence."""
    n_grid = SIZE[game]
    cells = json.loads(grid)
    occupied = [(r, c) for r, row in enumerate(cells)
                for c, v in enumerate(row) if v not in BG[game]]
    if game == "bt3gb":
        out = ["left", "right", "down"]
        out.append(f"click {rng.randrange(n_grid)} {rng.randrange(n_grid)}")
    elif game == "dq8gc":
        out = ["left", "right", "up", "down"]
        for (r, c) in rng.sample(occupied, min(2, len(occupied))):
            out.append(f"click {r} {c}")            # click a PARTICLE -> transfers control
        # contagion only fires when the controlled particle reaches an INFECTED one, which
        # an unbiased walk almost never does (1/10 windows in a first run). Steer toward
        # the nearest darkgreen so the click x infection composition is actually reachable.
        act = _active_cell(game, prev, grid)
        if act is not None:
            sick = [(r, c) for r, row in enumerate(cells)
                    for c, v in enumerate(row) if v == "darkgreen" and (r, c) != act]
            if sick:
                tr, tc = min(sick, key=lambda q: max(abs(q[0]-act[0]), abs(q[1]-act[1])))
                if tr < act[0]:
                    out += ["up"] * 2
                elif tr > act[0]:
                    out += ["down"] * 2
                if tc < act[1]:
                    out += ["left"] * 2
                elif tc > act[1]:
                    out += ["right"] * 2
    elif game == "n2ntd":
        out = ["left", "right", "up"]
        out.append(f"click {rng.randrange(n_grid)} {rng.randrange(n_grid)}")
    elif game == "s2kt7":
        out = [f"click {rng.randrange(n_grid)} {rng.randrange(n_grid)}" for _ in range(2)]
    elif game == "83wkq":
        # clicks are the only input; mix free cells with cells ADJACENT to an existing
        # particle so spawns can land beside a diffusing one rather than always in
        # isolation (the old exam's particles bucket was all opening-click-on-empty-grid).
        out = [f"click {rng.randrange(n_grid)} {rng.randrange(n_grid)}" for _ in range(2)]
        for (r, c) in rng.sample(occupied, min(2, len(occupied))):
            dr, dc = rng.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
            if 0 <= r + dr < n_grid and 0 <= c + dc < n_grid:
                out.append(f"click {r + dr} {c + dc}")
    else:
        out = ["noop"]
    # A click must survive truncation: the dq8gc contagion bias below adds several move
    # entries, and without this guard it crowds clicks out entirely (a 20-problem run went
    # from click-select in 9/10 problems to 0/20). Composition needs BOTH.
    clicks = [a for a in out if a.startswith("click")]
    rng.shuffle(out)
    out = out[:n]
    if clicks and not any(a.startswith("click") for a in out):
        out[rng.randrange(len(out))] = rng.choice(clicks)
    return out


def random_plan(game: str, h: int, rng: random.Random) -> list[str]:
    verbs = list(GAMES[game][2])
    n = SIZE[game]
    return [f"click {rng.randrange(n)} {rng.randrange(n)}" if rng.choice(verbs) == "click"
            else rng.choice(verbs) for _ in range(h)]


# ------------------------------------------------------------------------- screens
def compress(game: str, seed: int, prefix: list[str], plan: list[str],
             goal: str) -> list[str]:
    """Greedily delete any single action whose removal still reaches the goal,
    to fixpoint. The returned plan's length is h_min -- an upper bound on the shortest
    solution. h_min < h means the horizon was padding, not difficulty."""
    prog = GAMES[game][0]
    cur = list(plan)
    changed = True
    while changed and len(cur) > 1:
        changed = False
        for j in range(len(cur)):
            cand = cur[:j] + cur[j + 1:]
            if not cand:
                continue
            if exec_from(prog, seed, prefix, cand) == goal:
                cur, changed = cand, True
                break
    return cur


def screen(game: str, seed: int, prefix: list[str], plan: list[str], chain: list[str],
           fire_at: dict[str, tuple[int, bool]], rng: random.Random,
           n_random: int = 6) -> dict:
    """Run S1-S5. `fire_at[m] = (step, is_action)`. Returns a verdict; `ok` iff all pass.

    S4 ablates only ACTION-triggered mechanics: substituting `noop` at a step that is
    already `noop` is a no-op, so a passive rule can never be tested that way. Passive
    mechanics are instead carried by S5 -- a tick whose passive effect is not needed is
    exactly a tick compression can delete."""
    prog = GAMES[game][0]
    h = len(plan)
    start = (exec_from(prog, seed, prefix[:-1], prefix[-1:]) if prefix
             else _start_grid(prog, seed))
    goal = exec_from(prog, seed, prefix, plan)
    v = {"h": h, "chain": chain}

    v["s1_changed"] = (goal is not None and goal != start)
    v["s2_noop_fails"] = (exec_from(prog, seed, prefix, ["noop"] * h) != goal)
    hits = sum(exec_from(prog, seed, prefix, random_plan(game, h, rng)) == goal
               for _ in range(n_random))
    v["random_success"] = hits / n_random
    v["s3_not_trivial"] = (hits == 0)

    nec = {}
    for m in chain:
        j, is_action = fire_at[m]
        if not is_action:
            continue
        abl = plan[:j] + ["noop"] + plan[j + 1:]
        nec[m] = (exec_from(prog, seed, prefix, abl) != goal)
    v["necessity"] = nec
    v["s4_all_necessary"] = all(nec.values()) if nec else True

    comp = compress(game, seed, prefix, plan, goal)
    v["h_min"] = len(comp)
    v["n_dec"] = sum(a.split()[0] != "noop" for a in comp)
    v["s5_incompressible"] = (len(comp) == h)

    v["ok"] = bool(v["s1_changed"] and v["s2_noop_fails"] and v["s3_not_trivial"]
                   and v["s4_all_necessary"] and v["s5_incompressible"])
    v["start_grid"], v["goal_grid"] = start, goal
    return v


# -------------------------------------------------------------------------- search
def rollout(game: str, seed: int, prefix: list[str], length: int, rng: random.Random,
            noop_p: float = 0.35) -> dict:
    """One proposer-driven random rollout from the post-prefix state, with its per-step
    `fired` trace. Noops are mixed in so passive-consequence chains (a bullet in flight,
    a droplet falling) have room to play out."""
    prog = GAMES[game][0]
    prev = None
    plan, g = [], (exec_from(prog, seed, prefix[:-1], prefix[-1:]) if prefix
                   else _start_grid(prog, seed))
    for _ in range(length):
        if g is None:
            break
        a = ("noop" if rng.random() < noop_p
             else rng.choice(propose(game, g, rng, prev=prev)))
        plan.append(a)
        prev, g = g, exec_from(prog, seed, prefix + plan, [])
        if g is None:
            g = exec_from(prog, seed, prefix, plan)
    t = trace(game, seed, prefix, plan)
    return {"plan": plan, **t}


def windows(game: str, roll: dict, horizons: list[int], min_mech: int = 2):
    """Every window of the rollout in which EVERY step fires a consequential rule and at
    least `min_mech` distinct mechanics appear. Requiring no dead steps is the cheap
    generation-time analogue of S5 -- it stops padded windows being screened at all."""
    fs, plan = roll["fired"], roll["plan"]
    live = [any(not is_scenery(game, m) for m in f.consequential()) for f in fs]
    for h in horizons:
        for t in range(0, len(plan) - h + 1):
            if not all(live[t:t + h]):
                continue
            fire_at, order = {}, []
            for j in range(t, t + h):
                for m, is_action in ([(x, True) for x in fs[j].action]
                                     + [(x, False) for x in fs[j].passive]):
                    if (m in ("static-noop", "ant-idle", "particle-idle")
                            or m in fire_at or is_scenery(game, m)):
                        continue
                    fire_at[m] = (j - t, is_action)
                    order.append(m)
            if len(order) >= min_mech:
                yield {"t": t, "h": h, "plan": plan[t:t + h], "chain": order,
                       "fire_at": fire_at}



# ------------------------------------------------------------------------- recipes
# Directed constructions for mechanic combinations a random walk almost never reaches.
# Each reads the CURRENT board and emits an action sequence; the generator may use
# engine-side facts (which particle is active, where the infected ones are) because none
# of it reaches the solver -- the shipped problem is only (start frame, goal frame).
# A recipe returns None when the board cannot support it from this state.

def _rec_dq8gc_click_infect(game, grid, rng):
    """click a HEALTHY particle (transfers control), walk it next to an INFECTED one,
    then wait a tick for contagion. Composes click-select x move x contagion-spread --
    the pair that co-occurs in 0% of the human-derived windows."""
    P_ = json.loads(grid)
    healthy = [(r, c) for r, row in enumerate(P_) for c, v in enumerate(row) if v == "gray"]
    sick = [(r, c) for r, row in enumerate(P_) for c, v in enumerate(row) if v == "darkgreen"]
    if not healthy or not sick:
        return None
    a = rng.choice(healthy)
    t = min(sick, key=lambda q: abs(q[0] - a[0]) + abs(q[1] - a[1]))
    plan = [f"click {a[0]} {a[1]}"]
    r, c = a
    # `adj obj ... 1` is ORTHOGONAL-only: stopping at Chebyshev distance 1 leaves the
    # particle diagonally placed and contagion never fires. Walk to Manhattan distance 1.
    while abs(r - t[0]) + abs(c - t[1]) > 1 and len(plan) < 7:
        if r < t[0]:
            plan.append("down"); r += 1
        elif r > t[0]:
            plan.append("up"); r -= 1
        elif c < t[1]:
            plan.append("right"); c += 1
        elif c > t[1]:
            plan.append("left"); c -= 1
        else:
            break
    return plan + ["noop"] if len(plan) >= 2 else None


def _rec_bt3gb_rain_flip_fall(game, grid, rng):
    """rain a LIQUID droplet, let it fall, click to flip it to SOLID, watch it fall under
    the other rule. Composes rain x liquid-fall x click-flip x solid-fall."""
    n = SIZE[game]
    return ["down", "noop", f"click {rng.randrange(n)} {rng.randrange(n)}", "noop", "noop"]


def _rec_n2ntd_move_jump_move(game, grid, rng):
    """walk, jump, keep moving while AIRBORNE, then fall. Composes move x jump x gravity
    -- jump is only ever tested in isolation in the human-derived set."""
    d = rng.choice(["left", "right"])
    return [d, "up", d, d, "noop", "noop"]


def _rec_s2kt7_spawn_eat(game, grid, rng):
    """spawn food, let the ants walk to it and eat. Composes click-spawn x ant-move x
    food-eaten. The click LOCATION is ignored by this program -- `randomPositions` places
    the 2 food anywhere -- so the ants' walk length is not controllable and the recipe
    just allows enough ticks for a nearby spawn to be reached."""
    return [f"click {rng.randrange(SIZE[game])} {rng.randrange(SIZE[game])}"] + ["noop"] * 8


def _rec_83wkq_spawn_diffuse(game, grid, rng):
    """spawn a particle, let it random-walk, spawn a second one. Composes
    click-spawn-particle x particle-diffuse with TWO clicks, so the problem carries more
    than a single input even though the passive half is unpredictable."""
    n = SIZE[game]
    return [f"click {rng.randrange(n)} {rng.randrange(n)}", "noop",
            f"click {rng.randrange(n)} {rng.randrange(n)}", "noop"]


RECIPES = {
    "dq8gc": [_rec_dq8gc_click_infect],
    "83wkq": [_rec_83wkq_spawn_diffuse],
    "bt3gb": [_rec_bt3gb_rain_flip_fall],
    "n2ntd": [_rec_n2ntd_move_jump_move],
    "s2kt7": [_rec_s2kt7_spawn_eat],
}


def recipe_rollout(game: str, seed: int, prefix: list[str], rng: random.Random):
    """Run every recipe available for `game` from the post-prefix state."""
    prog = GAMES[game][0]
    g = (exec_from(prog, seed, prefix[:-1], prefix[-1:]) if prefix
         else _start_grid(prog, seed))
    out = []
    for fn in RECIPES.get(game, []):
        if g is None:
            break
        plan = fn(game, g, rng)
        if not plan:
            continue
        out.append({"plan": plan, **trace(game, seed, prefix, plan)})
    return out


# --------------------------------------------------------------------------- build
def load_prefixes(game: str, rng: random.Random, n: int, max_t: int) -> list[list[str]]:
    """Cut points from real human drives, so synthetic chains play out on realistic,
    populated boards rather than always from the initial state. Capped because every
    screen re-replays the prefix, so a long one multiplies the whole cost."""
    root = DATA / game / "coverage" / "drives"
    drives = []
    for ep in sorted(root.glob("*/episode_0/trajectory.csv")):
        acts = [(r.get("Action") or "").strip()
                for r in csv.DictReader(ep.open())]
        acts = [a for a in acts if a]
        if len(acts) > 4:
            drives.append(acts)
    if not drives:
        return []
    out = []
    for _ in range(n):
        d = rng.choice(drives)
        t = rng.randrange(1, min(max_t, len(d) - 1) + 1)
        out.append(d[:t])
    return out


def is_scenery(game: str, m: str) -> bool:
    """A mechanic whose outcome lives ENTIRELY in autonomous-background classes is SCENERY:
    it still has to be predicted (goals are full frames) but it is not something the agent
    brings about, so it does not count as a link in the chain.

    This is bookkeeping, not goal semantics. n2ntd's enemy patrols on a fixed cycle and
    fires `enemy-patrol` on literally every tick; admitting it would (a) inflate `n_mech`,
    the measure of how compositional a problem is, on every n2ntd problem equally, and
    (b) make every noop a "live" step, defeating the generation-time liveness rule that is
    n2ntd's only remaining anti-padding guard now that S5 cannot fail there.
    `enemy-hit` spans {enemy, bullet} -- not scenery, and a real chain link."""
    cls = MECH_CLASSES[game].get(m, set())
    return bool(cls) and cls <= BACKGROUND[game]


def _sig(game: str, w: dict) -> tuple:
    return (game, tuple(sorted(w["chain"])), w["h"])


def build_game(game: str, rng: random.Random, n_rollouts: int, horizons: list[int],
               per_chain: int, max_screen: int, prefix_frac: float,
               max_prefix_t: int, roll_len: int) -> list[dict]:
    t0 = time.time()
    seed = 1
    n_pref = int(n_rollouts * prefix_frac)
    prefixes = ([[]] * (n_rollouts - n_pref)
                + load_prefixes(game, rng, n_pref, max_prefix_t))
    rng.shuffle(prefixes)

    cands = []
    for k, pre in enumerate(prefixes):
        try:
            r = rollout(game, seed, pre, roll_len, rng)
        except Exception as e:                                   # engine/termination
            print(f"  [{game}] rollout {k} failed: {type(e).__name__}", flush=True)
            continue
        rolls = [r]
        try:
            rolls += recipe_rollout(game, seed, pre, rng)
        except Exception as e:
            print(f"  [{game}] recipe {k} failed: {type(e).__name__}", flush=True)
        for rr in rolls:
            for w in windows(game, rr, horizons):
                cands.append({"prefix": pre + rr["plan"][:w["t"]], **w})
        if (k + 1) % 20 == 0:
            print(f"  [{game}] {k+1}/{len(prefixes)} rollouts, "
                  f"{len(cands)} candidate windows ({time.time()-t0:.0f}s)", flush=True)

    # prefer richer chains, then shorter prefixes (cheaper to screen and to present)
    cands.sort(key=lambda w: (-len(w["chain"]), len(w["prefix"])))
    kept, per = [], Counter()
    screened = 0
    for w in cands:
        if screened >= max_screen:
            break
        sig = _sig(game, w)
        if per[sig] >= per_chain:
            continue
        screened += 1
        try:
            v = screen(game, seed, w["prefix"], w["plan"], w["chain"], w["fire_at"], rng)
        except Exception as e:
            print(f"  [{game}] screen failed: {type(e).__name__}", flush=True)
            continue
        if not v["ok"]:
            continue
        per[sig] += 1
        kept.append({"game": game, "seed": seed, "prefix": w["prefix"],
                     "gt_actions": w["plan"], "chain": w["chain"],
                     "fire_at": {m: list(x) for m, x in w["fire_at"].items()},
                     "h": v["h"], "h_min": v["h_min"], "n_dec": v["n_dec"],
                     "n_mech": len(w["chain"]),
                     "start_grid": v["start_grid"], "goal_grid": v["goal_grid"],
                     "random_success": v["random_success"], "necessity": v["necessity"],
                     "source": "compose"})
    print(f"  [{game}] screened {screened}, kept {len(kept)} ({time.time()-t0:.0f}s)",
          flush=True)
    return kept


def report(problems: list[dict]) -> None:
    if not problems:
        print("\n(no problems)")
        return
    print("\n" + "=" * 72)
    print(f"COMPOSITIONAL PROBLEM SET  ({len(problems)} problems)")
    print("=" * 72)
    print(f"\n  {'game':<8} {'n':>4} {'h':>18} {'mean n_mech':>12} {'mean n_dec':>11}")
    for g in sorted({p["game"] for p in problems}):
        gp = [p for p in problems if p["game"] == g]
        hs = Counter(p["h"] for p in gp)
        print(f"  {g:<8} {len(gp):>4} {str(dict(sorted(hs.items()))):>18} "
              f"{sum(p['n_mech'] for p in gp)/len(gp):>12.1f} "
              f"{sum(p['n_dec'] for p in gp)/len(gp):>11.1f}")
    pad = sum(p["h_min"] != p["h"] for p in problems)
    print(f"\n  h_min == h for {len(problems)-pad}/{len(problems)} problems "
          f"(S5 guarantees this; {pad} violations would be a bug)")
    print(f"\n  most common chains:")
    for ch, n in Counter(tuple(p["chain"]) for p in problems).most_common(12):
        print(f"    {n:>3}  {' -> '.join(ch)}")
    print(f"\n  mechanic participation:")
    for g in sorted({p["game"] for p in problems}):
        c = Counter(m for p in problems if p["game"] == g for m in p["chain"])
        print(f"    {g}: {dict(c.most_common())}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--game", choices=COMPOSABLE_GAMES)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--horizons", default="2,3,4,5,6")
    ap.add_argument("--rollouts", type=int, default=60)
    ap.add_argument("--roll-len", type=int, default=10)
    ap.add_argument("--per-chain", type=int, default=3)
    ap.add_argument("--max-screen", type=int, default=200)
    ap.add_argument("--prefix-frac", type=float, default=0.5)
    ap.add_argument("--max-prefix-t", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    games = COMPOSABLE_GAMES if args.all else ([args.game] if args.game else [])
    if not games:
        ap.error("pass --game or --all")
    horizons = [int(h) for h in args.horizons.split(",") if h]
    rng = random.Random(args.seed)

    problems = []
    for g in games:
        problems += build_game(g, rng, args.rollouts, horizons, args.per_chain,
                               args.max_screen, args.prefix_frac, args.max_prefix_t,
                               args.roll_len)
    report(problems)
    out = Path(args.out or (_BAI_ROOT / "logs/compose/problems.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"source": "compose_plan", "horizons": horizons,
                               "seed": args.seed, "n": len(problems),
                               "problems": problems}, indent=1) + "\n")
    print(f"\nwrote {out} ({len(problems)} problems)")


if __name__ == "__main__":
    main()
