"""CURATED planning problems: one small hand-designed ladder per game.

The compositional set (`compose_plan.py`) targeted CHAINS OF MECHANICS and grew a window
forward until the chain completed.  Nothing in that construction asked whether the chain
added up to anything, so it produced `move-up -> move-left -> contagion-spread` and never
`every particle is infected`; `jump -> gravity-fall` and never `the enemy is dead`.

This module goes the other way: ~5-8 problems per game, each authored around something the
game is actually ABOUT, ordered into a difficulty ladder (L1 one mechanic -> L4 the game's
objective).  AutumnBench itself ships one such problem per game; its N2NTD goal asserts
background over the enemy's two rows -- i.e. kill the enemy -- and its ice goal is night,
cloud parked left, three ice blocks on the floor.

A GOAL IS A CONCRETE END STATE: one full rendered frame, compared for exact equality.
No masks, no partial goals, no predicate evaluation at eval time.  Predicates live only in
here -- they define the subgoals the solver chains through, and they carry the
incompressibility proof (below).  Nothing predicate-shaped ships.

Why exact frames work: every one of these games has an ABSORBING end state.  Killing
n2ntd's enemy removes the only autonomously-moving object in any of the five, and the world
goes completely still; bt3gb water settles, dq8gc particles move only on input, s2kt7 ants
freeze once the food is gone.  So the hardest goal in the set is also the cleanest frame --
reach it and it holds, and "matched at step h" and "matched at any step" coincide.

PADDING CONTROL.  n2ntd's enemy patrols on an 18-tick cycle regardless of the agent, so an
exact frame taken while it lives pins the tick mod 18.  That does not pad the reference plan
(the frame is snapshotted FROM the reference) but it destroys the PROOF that there is no
padding: greedy deletion can never succeed, because dropping any action re-phases the enemy
and misses the frame whether or not the action mattered.  That is exactly the hole that let
the old set ship 12-tick problems averaging 2.23 real actions.  Fix: compress against the
authored PREDICATE, which ignores the enemy, and ship the frame reached by the compressed
plan.  Predicate in, frame out.

    uv run python -m offline_learning.curated_plan build --out logs/curated/problems.json
    uv run python -m offline_learning.curated_plan build --game n2ntd --verbose
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

_BAI_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_BAI_ROOT), str(_BAI_ROOT / "MARAProtocol"),
           str(_BAI_ROOT / "MARAProtocol/python_examples/autumnbench")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from python_examples.autumnbench.autumnstdlib import autumnstdlib  # noqa: E402
from python_examples.autumnbench.env_utils import render_grid  # noqa: E402
from python_examples.autumnbench.interpreter_module import Interpreter  # noqa: E402

from offline_learning.human_replay import GAMES  # noqa: E402

PROGRAMS = _BAI_ROOT / "MARAProtocol/python_examples/autumnbench/example_benchmark/programs"

Grid = tuple[tuple[str, ...], ...]


# ---------------------------------------------------------------------------- engine
class Sim:
    """Raw-interpreter driver.  ~880 steps/s vs ~320 through AutumnBenchEnvWrapper, which
    matters because the solver replays from reset for every node it expands
    (`restore_environment` exists but its own dump does not round-trip -- it raises
    ParseError on the string `get_environment_string` emits -- so there is no snapshot).

    Clicks are ROW-MAJOR here, matching AutumnBenchEnvWrapper's agent-facing interface.
    The interpreter itself is column-first (`click(x=col, y=row)`), so they are transposed
    on the way in.  `validate_curated.py` re-runs every plan through the wrapper to prove
    the two drivers agree frame for frame.

    Hidden state (mario's ammo, which particle dq8gc is driving, bt3gb's day flag) is not
    rendered, so it is tracked here by mirroring the .sexp rules.  It never reaches a goal;
    it exists so the solver can dedup states that look identical but are not.
    """

    def __init__(self, game: str, seed: int = 0):
        self.game = game
        self.seed = seed
        prog_name = GAMES[game][0]
        self.it = Interpreter()
        self.it.run_script((PROGRAMS / f"{prog_name}.sexp").read_text(),
                           autumnstdlib, "", seed)
        self.bg = self.it.get_background()
        self.hid = _INIT_HIDDEN[game]()
        self.n = 0

    def grid(self) -> Grid:
        d = json.loads(self.it.render_all())
        return tuple(tuple(row) for row in
                     render_grid(d, background_color=self.bg, color_dict={}))

    def step(self, action: str) -> None:
        before = self.grid()
        verb, *rest = action.split()
        if verb == "click":
            row, col = int(rest[0]), int(rest[1])
            self.it.click(col, row)          # interpreter is column-first
        elif verb in ("left", "right", "up", "down"):
            getattr(self.it, verb)()
        elif verb != "noop":
            raise ValueError(f"bad action {action!r}")
        self.it.step()
        self.n += 1
        _TRACK[self.game](self.hid, before, self.grid(), action)

    def state(self) -> "St":
        return St(self.grid(), tuple(sorted(self.hid.items())))


@dataclass(frozen=True)
class St:
    grid: Grid
    hid: tuple

    def at(self, *colors: str) -> frozenset[tuple[int, int]]:
        want = set(colors)
        return frozenset((r, c) for r, row in enumerate(self.grid)
                         for c, v in enumerate(row) if v in want)

    def one(self, color: str) -> tuple[int, int] | None:
        s = self.at(color)
        return next(iter(s)) if len(s) == 1 else None


def replay(game: str, seed: int, actions: list[str]) -> Sim:
    sim = Sim(game, seed)
    for a in actions:
        sim.step(a)
    return sim


def trace(game: str, seed: int, actions: list[str]) -> list[St]:
    """State after each action, with the start state at index 0."""
    sim = Sim(game, seed)
    out = [sim.state()]
    for a in actions:
        sim.step(a)
        out.append(sim.state())
    return out


# ------------------------------------------------------------------- hidden tracking
def _hid_n2ntd() -> dict:
    # `t` is part of the key because two things about n2ntd are real but unrendered: the
    # enemy's `movingLeft` flag (it sits on the same column going both ways -- t=12 and
    # t=14 render identically and evolve oppositely, which silently pruned the only branch
    # that kills it), and mario himself, who is invisible for the frame he stands on a coin
    # (coins render after him).  Both are functions of the tick, so carrying `t` makes the
    # search a sound BFS over the time-expanded graph.
    return {"ammo": 0, "t": 0}


def _track_n2ntd(h: dict, before: Grid, after: Grid, action: str) -> None:
    """Ammo = coins collected - shots fired.  A coin leaves the frame on exactly the tick
    the `on intersects` handler credits the bullet, so the two stay in lock-step."""
    gold_before = sum(row.count("gold") for row in before)
    gold_after = sum(row.count("gold") for row in after)
    spent = 1 if action.startswith("click") and h["ammo"] > 0 else 0
    h["ammo"] += (gold_before - gold_after) - spent
    h["t"] += 1


def _hid_dq8gc() -> dict:
    return {"active": (2, 2)}


def _track_dq8gc(h: dict, before: Grid, after: Grid, action: str) -> None:
    verb, *rest = action.split()
    if verb == "click":
        rc = (int(rest[0]), int(rest[1]))
        if before[rc[0]][rc[1]] in ("gray", "darkgreen"):
            h["active"] = rc                  # control transfers to the clicked particle
    elif verb in _DELTA:
        dr, dc = _DELTA[verb]
        h["active"] = (h["active"][0] + dr, h["active"][1] + dc)


_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def _hid_bt3gb() -> dict:
    return {"day": 1, "cloud": 4}


def _track_bt3gb(h: dict, before: Grid, after: Grid, action: str) -> None:
    verb = action.split()[0]
    if verb == "click":
        h["day"] ^= 1                         # `on clicked` has no position guard
    elif verb == "left" and h["cloud"] > 1:
        h["cloud"] -= 1
    elif verb == "right" and h["cloud"] < 14:
        h["cloud"] += 1


def _hid_clicks() -> dict:
    # s2kt7 carries `t` for the same reason n2ntd does: an ant standing on a food cell is
    # occluded (food renders after ants), so two different ant positions can render alike.
    return {"clicks": 0, "t": 0}


def _track_clicks(h: dict, before: Grid, after: Grid, action: str) -> None:
    if action.startswith("click"):
        h["clicks"] += 1
    h["t"] += 1


_INIT_HIDDEN: dict[str, Callable[[], dict]] = {
    "n2ntd": _hid_n2ntd, "dq8gc": _hid_dq8gc, "bt3gb": _hid_bt3gb,
    "s2kt7": _hid_clicks, "83wkq": _hid_clicks,
}
_TRACK: dict[str, Callable] = {
    "n2ntd": _track_n2ntd, "dq8gc": _track_dq8gc, "bt3gb": _track_bt3gb,
    "s2kt7": _track_clicks, "83wkq": _track_clicks,
}


# --------------------------------------------------------------------------- solver
@dataclass
class Seg:
    """One subgoal.  `actions` is the alphabet the search may use to reach it -- keeping it
    tight is what makes replay-based BFS affordable (a 14-tick wait under {noop} is a
    straight line; under the full alphabet it is 5^14)."""
    goal: Callable[[St], bool]
    actions: list[str]
    cap: int
    label: str = ""


@dataclass
class Problem:
    game: str
    pid: str
    tier: str
    objective: str
    segs: list[Seg]
    goal: Callable[[St], bool]
    seed: int = 0
    note: str = ""
    scripted: list[str] | None = None      # bypass search (83wkq: nothing to search over)


MOVE4 = ["up", "down", "left", "right"]
WAIT = ["noop"]


def bfs(game: str, seed: int, prefix: list[str], seg: Seg,
        max_nodes: int = 60_000) -> list[str] | None:
    """Shortest action sequence from `prefix` satisfying `seg.goal`.

    Dedup keys on the FULL state (frame + hidden), autonomous objects included.  n2ntd's
    enemy is a function of depth, so within a layer it is constant and dedup still bites;
    across layers it correctly refuses to merge states that differ in enemy phase, which is
    what a firing-window subgoal needs.
    """
    start = replay(game, seed, prefix).state()
    if seg.goal(start):
        return []
    seen = {start}
    frontier: dict[St, list[str]] = {start: []}
    nodes = 0
    for _ in range(seg.cap):
        nxt: dict[St, list[str]] = {}
        for plan in frontier.values():
            for a in seg.actions:
                nodes += 1
                if nodes > max_nodes:
                    return None
                st = replay(game, seed, prefix + plan + [a]).state()
                if seg.goal(st):
                    return plan + [a]
                if st in seen:
                    continue
                seen.add(st)
                nxt[st] = plan + [a]
        if not nxt:
            return None
        frontier = nxt
    return None


def solve(p: Problem, verbose: bool = False) -> list[str] | None:
    if p.scripted is not None:
        return list(p.scripted)
    plan: list[str] = []
    for i, seg in enumerate(p.segs):
        t0 = time.time()
        got = bfs(p.game, p.seed, plan, seg)
        if got is None:
            if verbose:
                print(f"    seg {i} ({seg.label}) UNREACHED")
            return None
        plan += got
        if verbose:
            print(f"    seg {i} ({seg.label}) +{len(got)} -> {len(plan)} "
                  f"[{time.time() - t0:.0f}s]")
    return plan


def compress(p: Problem, plan: list[str]) -> list[str]:
    """Greedily delete any single action whose removal still satisfies the OBJECTIVE
    PREDICATE, to fixpoint.  Predicate, not frame: on n2ntd a frame comparison is
    clock-stamped by the enemy and no deletion can ever succeed, so the screen would pass
    vacuously on exactly the game that needs it most."""
    cur = list(plan)
    changed = True
    while changed and len(cur) > 1:
        changed = False
        for j in range(len(cur)):
            cand = cur[:j] + cur[j + 1:]
            if any(p.goal(s) for s in trace(p.game, p.seed, cand)[1:]):
                cur, changed = cand, True
                break
    return cur


# ------------------------------------------------------------------------ predicates
def _rest_n2ntd(s: St) -> bool:
    """Mario is standing, not mid-fall: the cell under him is floor or platform.  He is
    occluded for one frame while overlapping a coin (coins render after him), so `one`
    returning None is itself a not-at-rest signal."""
    m = s.one("red")
    if m is None:
        return False
    r, c = m
    return r == 11 or s.grid[r + 1][c] == "darkorange"


def _n2ntd_coins(s: St) -> frozenset:
    return s.at("gold")


def _water(s: St, solid: bool) -> frozenset:
    return s.at("lightblue" if solid else "blue")


def _night(s: St) -> bool:
    return s.grid[1][0] == "gray"          # celestial row 1 is never covered by the cloud


def _tower(s: St, col: int, h: int) -> bool:
    return _water(s, True) == frozenset((15 - i, col) for i in range(h))


def _floor_run(s: St, solid: bool, n: int) -> bool:
    w = _water(s, solid)
    return len(w) == n and all(r == 15 for r, _ in w) and _contiguous({c for _, c in w})


def _contiguous(cols: set[int]) -> bool:
    return bool(cols) and max(cols) - min(cols) + 1 == len(cols)


def _h(s: St, key: str):
    return dict(s.hid)[key]


# ------------------------------------------------------------------------- ladders
# Coordinates are (row, col) on the rendered grid, matching the agent-facing click order.
N_COINS = {(9, 1), (4, 7), (5, 9)}          # n2ntd, in render coordinates
DQ_HEALTHY = {(3, 4), (5, 3), (5, 7), (6, 6)}


def _problems() -> list[Problem]:
    P: list[Problem] = []
    fs = frozenset

    # ---------------------------------------------------------------- n2ntd / mario
    # Coins are ammo; one bullet kills the enemy; platforms absorb bullets from below, so
    # the only firing columns from the floor are 3, 7 and 11.  `on left`/`on right` override
    # the gravity clause, so mario walks laterally in mid-air -- that is what puts the upper
    # coins in reach.
    MV = ["left", "right", "up", "noop"]
    P.append(Problem("n2ntd", "platform", "L1", "Stand on the middle platform",
                     segs=[Seg(lambda s: s.one("red") == (7, 6), MV, 4, "on platform")],
                     goal=lambda s: s.one("red") == (7, 6)))
    P.append(Problem("n2ntd", "high-ground", "L1", "Stand on the top-right platform",
                     segs=[Seg(lambda s: s.one("red") == (3, 6), ["up", "noop"], 4, "apex"),
                           Seg(lambda s: s.one("red") == (5, 8), ["right", "noop"], 6, "land")],
                     goal=lambda s: s.one("red") == (5, 8)))
    P.append(Problem("n2ntd", "coin-ground", "L2", "Collect the coin on the low platform",
                     segs=[Seg(lambda s: s.one("red") == (11, 1), ["left", "noop"], 7, "walk"),
                           Seg(lambda s: _n2ntd_coins(s) == fs({(4, 7), (5, 9)})
                               and s.one("red") == (9, 1), ["up", "noop"], 6, "jump+fall")],
                     goal=lambda s: _n2ntd_coins(s) == fs({(4, 7), (5, 9)})
                     and s.one("red") == (9, 1)))
    P.append(Problem("n2ntd", "coin-air", "L2", "Collect the mid-air coin",
                     segs=[Seg(lambda s: s.one("red") == (3, 6), ["up", "noop"], 4, "apex"),
                           Seg(lambda s: _n2ntd_coins(s) == fs({(9, 1), (5, 9)}),
                               ["right", "noop"], 5, "air-walk+drop"),
                           Seg(lambda s: s.one("red") == (11, 7), ["noop"], 10, "land")],
                     goal=lambda s: _n2ntd_coins(s) == fs({(9, 1), (5, 9)})
                     and s.one("red") == (11, 7)))
    _all_coins = [
        Seg(lambda s: s.one("red") == (11, 1), ["left", "noop"], 7, "walk"),
        Seg(lambda s: _n2ntd_coins(s) == fs({(4, 7), (5, 9)}) and s.one("red") == (9, 1),
            ["up", "noop"], 6, "coin (9,1)"),
        Seg(lambda s: _n2ntd_coins(s) == fs({(4, 7)}), ["up", "right", "noop"], 12,
            "coin (5,9)"),
        Seg(lambda s: not _n2ntd_coins(s), ["up", "left", "noop"], 10, "coin (4,7)"),
    ]
    P.append(Problem("n2ntd", "all-coins", "L3", "Collect all three coins",
                     segs=_all_coins + [Seg(_rest_n2ntd, ["noop"], 10, "settle")],
                     goal=lambda s: not _n2ntd_coins(s) and _rest_n2ntd(s)))
    P.append(Problem("n2ntd", "kill-one-coin", "L3",
                     "Collect one coin, then shoot the enemy",
                     segs=[_all_coins[0], _all_coins[1],
                           Seg(lambda s: not s.at("blue"), ["noop", "click 0 0"], 16, "kill"),
                           Seg(lambda s: not s.at("mediumpurple"), ["noop"], 14, "clear")],
                     goal=lambda s: not s.at("blue") and not s.at("mediumpurple")
                     and _n2ntd_coins(s) == fs({(4, 7), (5, 9)})))
    P.append(Problem("n2ntd", "all-coins-kill", "L4",
                     "Collect all three coins, then shoot the enemy",
                     segs=_all_coins + [
                         Seg(lambda s: not s.at("blue"), ["noop", "click 0 0"], 18, "kill"),
                         Seg(lambda s: not s.at("mediumpurple") and _rest_n2ntd(s),
                             ["noop"], 14, "clear")],
                     goal=lambda s: not s.at("blue") and not s.at("mediumpurple")
                     and not _n2ntd_coins(s) and _rest_n2ntd(s),
                     note="the goal frame is absorbing: killing the enemy removes the only "
                          "autonomously-moving object in any of these five games"))

    # ------------------------------------------------------------------ bt3gb / ice
    # `down` drops water at the cloud's column; day makes liquid, night makes solid; one
    # click flips day/night AND every drop already in the world.  Ice stacks
    # (nextSolid = moveDownNoCollision); liquid slides toward the nearest hole in the row
    # below, so it always ends as a flat run.  ice-tower and freeze-pool are the
    # discriminating pair: a stacked column can only be made by freezing BEFORE the drops,
    # a spread ice run only by freezing AFTER they settle.
    CLICK = ["click 8 8"]
    # Drops must be SPACED.  `on down` assigns water from the CURRENT list and overrides
    # the fall clause, so two downs on consecutive ticks land both drops on the same cell
    # and they travel together for ever -- three of them render as one.
    RAIN = ["down", "noop"]
    P.append(Problem("bt3gb", "nightfall", "L1", "Turn day into night",
                     segs=[Seg(_night, CLICK, 2, "click")], goal=_night))
    P.append(Problem("bt3gb", "park-cloud", "L1", "Park the cloud against the left wall",
                     segs=[Seg(lambda s: _h(s, "cloud") == 1, ["left"], 5, "walk cloud")],
                     goal=lambda s: _h(s, "cloud") == 1))
    P.append(Problem("bt3gb", "one-drop", "L2", "Land one raindrop at column 2",
                     segs=[Seg(lambda s: _h(s, "cloud") == 2, ["left"], 4, "aim"),
                           Seg(lambda s: len(_water(s, False)) == 1, ["down"], 2, "rain"),
                           Seg(lambda s: _water(s, False) == fs({(15, 2)}), WAIT, 20, "fall")],
                     goal=lambda s: _water(s, False) == fs({(15, 2)})
                     and not _water(s, True)))
    P.append(Problem("bt3gb", "one-ice", "L2", "Land one block of ice at column 9",
                     segs=[Seg(_night, CLICK, 2, "night"),
                           Seg(lambda s: _h(s, "cloud") == 9, ["right"], 8, "aim"),
                           Seg(lambda s: len(_water(s, True)) == 1, ["down"], 2, "snow"),
                           Seg(lambda s: _water(s, True) == fs({(15, 9)}), WAIT, 20, "fall")],
                     goal=lambda s: _water(s, True) == fs({(15, 9)})
                     and not _water(s, False) and _night(s)))
    P.append(Problem("bt3gb", "pool", "L2", "Spread three raindrops into a pool",
                     segs=[Seg(lambda s: len(_water(s, False)) == 3, RAIN, 8, "rain x3"),
                           Seg(lambda s: _floor_run(s, False, 3), WAIT, 24, "settle")],
                     goal=lambda s: _floor_run(s, False, 3) and not _water(s, True)
                     and not _night(s)))
    P.append(Problem("bt3gb", "ice-tower", "L3", "Stack three blocks of ice into a tower",
                     segs=[Seg(_night, CLICK, 2, "night"),
                           Seg(lambda s: len(_water(s, True)) == 3, RAIN, 8, "snow x3"),
                           Seg(lambda s: _tower(s, 4, 3), WAIT, 24, "stack")],
                     goal=lambda s: _tower(s, 4, 3) and not _water(s, False) and _night(s),
                     note="only reachable by freezing BEFORE the drops -- liquid spreads"))
    P.append(Problem("bt3gb", "freeze-pool", "L3", "Freeze a settled pool into flat ice",
                     segs=[Seg(lambda s: len(_water(s, False)) == 3, RAIN, 8, "rain x3"),
                           Seg(lambda s: _floor_run(s, False, 3), WAIT, 24, "settle"),
                           Seg(lambda s: _floor_run(s, True, 3) and _night(s), CLICK, 2,
                               "freeze")],
                     goal=lambda s: _floor_run(s, True, 3) and not _water(s, False)
                     and _night(s),
                     note="only reachable by freezing AFTER the drops settle -- ice stacks"))
    _stair = fs({(15, 8), (14, 9), (15, 9), (13, 10), (14, 10), (15, 10)})
    P.append(Problem("bt3gb", "staircase", "L4", "Build a 3-2-1 staircase of ice",
                     segs=[Seg(_night, CLICK, 2, "night"),
                           Seg(lambda s: _h(s, "cloud") == 10, ["right"], 8, "aim c10"),
                           Seg(lambda s: len(_water(s, True)) == 3, RAIN, 8, "snow x3"),
                           Seg(lambda s: _tower(s, 10, 3), WAIT, 24, "stack c10"),
                           Seg(lambda s: _h(s, "cloud") == 9, ["left"], 3, "aim c9"),
                           Seg(lambda s: len(_water(s, True)) == 5, RAIN, 6, "snow x2"),
                           Seg(lambda s: _water(s, True) == _tower_set(10, 3) | fs({(14, 9), (15, 9)}),
                               WAIT, 24, "stack c9"),
                           Seg(lambda s: _h(s, "cloud") == 8, ["left"], 3, "aim c8"),
                           Seg(lambda s: len(_water(s, True)) == 6, RAIN, 3, "snow x1"),
                           Seg(lambda s: _water(s, True) == _stair, WAIT, 24, "stack c8")],
                     goal=lambda s: _water(s, True) == _stair and not _water(s, False)
                     and _night(s)))

    # -------------------------------------------------------------- dq8gc / disease
    # A click hands control of the clicked particle over and returns the old one to the
    # pool; contagion crosses orthogonal adjacency with a one-tick delay and is permanent.
    M5 = MOVE4 + ["noop"]
    P.append(Problem("dq8gc", "walk", "L1", "Drive the infected particle to the corner",
                     segs=[Seg(lambda s: s.at("darkgreen") == fs({(0, 0)}), M5, 6, "walk")],
                     goal=lambda s: s.at("darkgreen") == fs({(0, 0)})
                     and s.at("gray") == fs(DQ_HEALTHY)))
    P.append(Problem("dq8gc", "infect-one", "L2", "Infect the nearest healthy particle",
                     segs=[Seg(lambda s: s.at("darkgreen") == fs({(2, 4), (3, 4)}), M5, 5,
                               "approach+spread")],
                     goal=lambda s: s.at("darkgreen") == fs({(2, 4), (3, 4)})
                     and s.at("gray") == fs({(5, 3), (5, 7), (6, 6)})))
    P.append(Problem("dq8gc", "swap-drive", "L2",
                     "Take control of a healthy particle and drive it to the wall",
                     segs=[Seg(lambda s: _h(s, "active") == (5, 3), ["click 5 3"], 2, "select"),
                           Seg(lambda s: s.at("gray") == fs({(5, 0), (3, 4), (5, 7), (6, 6)}),
                               ["left", "noop"], 5, "drive")],
                     goal=lambda s: s.at("gray") == fs({(5, 0), (3, 4), (5, 7), (6, 6)})
                     and s.at("darkgreen") == fs({(2, 2)}),
                     note="a click is invisible at t+1, so the goal is the MOVED particle -- "
                          "which makes the select instrumental and observable"))
    P.append(Problem("dq8gc", "chain", "L3",
                     "Hand off the infection: drive a newly-infected particle to a third one",
                     segs=[Seg(lambda s: s.at("darkgreen") == fs({(2, 4), (3, 4)}), M5, 5,
                               "infect (3,4)"),
                           Seg(lambda s: _h(s, "active") == (3, 4), ["click 3 4"], 2, "select"),
                           Seg(lambda s: s.at("darkgreen") == fs({(2, 4), (4, 3), (5, 3)}),
                               M5, 6, "carry+spread")],
                     goal=lambda s: s.at("darkgreen") == fs({(2, 4), (4, 3), (5, 3)})
                     and s.at("gray") == fs({(5, 7), (6, 6)})))
    P.append(Problem("dq8gc", "infect-all", "L3", "Infect every particle",
                     segs=[Seg(lambda s: s.at("gray") == fs({(5, 3), (5, 7), (6, 6)}), M5, 5,
                               "infect (3,4)"),
                           Seg(lambda s: s.at("gray") == fs({(5, 7), (6, 6)}), M5, 6,
                               "infect (5,3)"),
                           Seg(lambda s: not s.at("gray"), M5, 8, "infect the rest")],
                     goal=lambda s: not s.at("gray") and len(s.at("darkgreen")) == 5))
    _block = fs({(5, 5), (5, 6), (6, 5), (6, 6)})
    _gather = [
        Seg(lambda s: _h(s, "active") == (3, 4), ["click 3 4"], 2, "select (3,4)"),
        Seg(lambda s: s.at("gray") == fs({(5, 5), (5, 3), (5, 7), (6, 6)}), M5, 6, "drive"),
        Seg(lambda s: _h(s, "active") == (5, 3), ["click 5 3"], 2, "select (5,3)"),
        Seg(lambda s: s.at("gray") == fs({(5, 5), (6, 5), (5, 7), (6, 6)}), M5, 6, "drive"),
        Seg(lambda s: _h(s, "active") == (5, 7), ["click 5 7"], 2, "select (5,7)"),
        Seg(lambda s: s.at("gray") == _block, M5, 4, "drive"),
    ]
    P.append(Problem("dq8gc", "gather", "L4",
                     "Herd the four healthy particles into a 2x2 block",
                     segs=_gather,
                     goal=lambda s: s.at("gray") == _block
                     and s.at("darkgreen") == fs({(2, 2)}),
                     note="unsolvable without repeated click-select: this is click composed "
                          "with movement, which the compositional set never produced"))
    P.append(Problem("dq8gc", "infect-all-gather", "L4",
                     "Herd them into a block, then infect the block",
                     segs=_gather + [
                         Seg(lambda s: _h(s, "active") == (2, 2), ["click 2 2"], 2, "select"),
                         Seg(lambda s: _h(s, "active") == (4, 5), MOVE4, 7, "carry"),
                         Seg(lambda s: not s.at("gray"), WAIT, 6, "spread")],
                     goal=lambda s: not s.at("gray")
                     and s.at("darkgreen") == _block | fs({(4, 5)})))

    # ----------------------------------------------------------------- s2kt7 / ants
    # A click spawns 2 food at RANDOM positions (not the clicked cell); ants walk one cell
    # per tick toward the nearest, horizontal-first; food vanishes on contact; ants freeze
    # when no food exists.  The RNG advances only inside the `on clicked` handler, so spawn
    # positions are a function of the CLICK INDEX alone -- the world is fully deterministic
    # given the plan, and the spawn table is learnable from exploration.
    # Seed 1: seed 0 is degenerate (randomPositions collapses both foods onto (0,0)).
    CL = ["click 0 0"]
    P.append(Problem("s2kt7", "spawn", "L1", "Put food on the board",
                     segs=[Seg(lambda s: len(s.at("red")) == 2, CL, 2, "click")],
                     goal=lambda s: len(s.at("red")) == 2, seed=1))
    P.append(Problem("s2kt7", "intercept", "L3", "Stop the clock mid-forage",
                     segs=[Seg(lambda s: len(s.at("red")) == 2, CL, 2, "click"),
                           Seg(lambda s: s.at("gray") == fs({(5, 14), (14, 10)}), WAIT, 14,
                               "march")],
                     goal=lambda s: s.at("gray") == fs({(5, 14), (14, 10)})
                     and len(s.at("red")) == 2, seed=1,
                     note="the only NON-absorbing goal in the set: it names a frame in "
                          "mid-forage, so it tests predicting ant speed rather than waiting"))
    P.append(Problem("s2kt7", "one-eaten", "L2", "Let the ants eat one food",
                     segs=[Seg(lambda s: len(s.at("red")) == 2, CL, 2, "click"),
                           Seg(lambda s: len(s.at("red")) == 1, WAIT, 30, "forage")],
                     goal=lambda s: len(s.at("red")) == 1, seed=1))
    P.append(Problem("s2kt7", "all-eaten", "L3", "Let the ants clear the board",
                     segs=[Seg(lambda s: len(s.at("red")) == 2, CL, 2, "click"),
                           Seg(lambda s: not s.at("red"), WAIT, 30, "forage")],
                     goal=lambda s: not s.at("red") and _h(s, "clicks") == 1, seed=1))
    P.append(Problem("s2kt7", "two-rounds", "L3", "Feed the ants twice",
                     segs=[Seg(lambda s: _h(s, "clicks") == 1, CL, 2, "click 1"),
                           Seg(lambda s: not s.at("red"), WAIT, 30, "forage"),
                           Seg(lambda s: _h(s, "clicks") == 2, CL, 2, "click 2"),
                           Seg(lambda s: not s.at("red"), WAIT, 30, "forage")],
                     goal=lambda s: not s.at("red") and _h(s, "clicks") == 2, seed=1))
    P.append(Problem("s2kt7", "three-rounds", "L4", "Feed the ants three times",
                     segs=[Seg(lambda s: _h(s, "clicks") == 1, CL, 2, "click 1"),
                           Seg(lambda s: not s.at("red"), WAIT, 30, "forage"),
                           Seg(lambda s: _h(s, "clicks") == 2, CL, 2, "click 2"),
                           Seg(lambda s: not s.at("red"), WAIT, 30, "forage"),
                           Seg(lambda s: _h(s, "clicks") == 3, CL, 2, "click 3"),
                           Seg(lambda s: not s.at("red"), WAIT, 30, "forage")],
                     goal=lambda s: not s.at("red") and _h(s, "clicks") == 3, seed=1))

    # ------------------------------------------------------------ 83wkq / particles
    # Scripted, not searched: a click is the only input and every particle random-walks
    # every tick with no bounds check, so there is nothing for a search to choose.
    P.append(Problem("83wkq", "spawn-one", "L1", "Spawn one particle at (8,8)",
                     segs=[], goal=lambda s: s.at("blue") == fs({(8, 8)}),
                     scripted=["click 8 8"],
                     note="SOLVABLE: the goal is the frame one tick after the only click, "
                          "before any diffusion has happened"))
    P.append(Problem("83wkq", "spawn-two", "L2", "Spawn two particles a tick apart",
                     segs=[],
                     # the `(4,4) not in` clause is what keeps the noop alive: with only
                     # `len == 2` the compressor drops it, back-to-back clicks freeze
                     # diffusion, and the problem collapses back to naming two clicked cells
                     goal=lambda s: (len(s.at("blue")) == 2
                                     and (4, 4) not in s.at("blue")),
                     scripted=["click 4 4", "noop", "click 10 10"],
                     note="DELIBERATELY MUCH HARDER, and honestly close to unsolvable: the "
                          "second particle sits where it was clicked, but the first has "
                          "taken one uniform random step, so the exact frame cannot be "
                          "derived from the rules -- only memorised from the RNG. The noop "
                          "is LOAD-BEARING: `on clicked` assigns `particles` and so "
                          "suppresses that variable's `next` clause for the tick, meaning "
                          "back-to-back clicks freeze diffusion and the goal collapses to "
                          "the two clicked cells (which all three eval arms then solved "
                          "5/5). An action tick freezes the dynamics it writes to."))
    return P


def _tower_set(col: int, h: int) -> frozenset:
    return frozenset((15 - i, col) for i in range(h))


# --------------------------------------------------------------------------- build
def _label(p: Problem, plan: list[str]) -> list[str]:
    """Which .sexp rules the reference plan actually fires, via the noop counterfactual.
    Documentation only -- nothing here gates a problem."""
    from offline_learning.mechanics_rules import fired
    grids = [json.dumps([list(r) for r in s.grid])
             for s in trace(p.game, p.seed, plan)]
    seen: list[str] = []
    for i, a in enumerate(plan):
        cf = json.dumps([list(r) for r in
                         replay(p.game, p.seed, plan[:i] + ["noop"]).grid()])
        for m in fired(p.game, grids[i], a, cf, grids[i + 1]).consequential():
            if m not in seen:
                seen.append(m)
    return seen


def build_one(p: Problem, verbose: bool = False) -> dict | None:
    plan = solve(p, verbose=verbose)
    if plan is None:
        return None
    plan = compress(p, plan)
    states = trace(p.game, p.seed, plan)
    hit = [i for i, s in enumerate(states) if p.goal(s)]
    if not hit or hit[0] == 0:
        return None
    plan = plan[:hit[0]]                     # trim anything past the first satisfaction
    states = states[:hit[0] + 1]
    goal = states[-1]
    after = replay(p.game, p.seed, plan + ["noop"]).grid()
    _REF[p.pid] = plan
    return {
        "game": p.game, "id": p.pid, "tier": p.tier, "objective": p.objective,
        "program": GAMES[p.game][0], "seed": p.seed,
        "start": [list(r) for r in states[0].grid],
        "goal": [list(r) for r in goal.grid],
        "plan": plan,
        "h": len(plan),
        "n_decisions": sum(1 for a in plan if a != "noop"),
        "quiescent": after == goal.grid,
        "random_success": random_success(p, len(plan)),
        "mechanics": _label(p, plan),
        "note": p.note,
    }


def random_success(p: Problem, h: int, trials: int = 40) -> float:
    """How often a random plan of the same length lands on the goal frame.  This is the
    floor any result has to beat, and at h=1 it is genuinely high -- an L1 problem with a
    six-verb alphabet is guessable, and the number says so instead of hiding it."""
    rng = random.Random(hash((p.game, p.pid)) & 0xffff)
    verbs = list(GAMES[p.game][2])
    n = {"n2ntd": 12}.get(p.game, 16)
    goal = trace(p.game, p.seed, _REF[p.pid])[-1].grid
    hits = 0
    for _ in range(trials):
        draw = [rng.choice(verbs) for _ in range(h)]
        plan = [f"click {rng.randrange(n)} {rng.randrange(n)}" if v == "click" else v
                for v in draw]
        if any(s.grid == goal for s in trace(p.game, p.seed, plan)[1:]):
            hits += 1
    return hits / trials


_REF: dict[str, list[str]] = {}


def report(rows: list[dict]) -> None:
    print("\n" + "=" * 78)
    print(f"CURATED PLANNING SET  ({len(rows)} problems)")
    print("=" * 78 + "\n")
    for game in dict.fromkeys(r["game"] for r in rows):
        sub = [r for r in rows if r["game"] == game]
        print(f"  {game} / {GAMES[game][1]}")
        for r in sub:
            q = "  " if r["quiescent"] else " *"
            print(f"    {r['tier']} {r['id']:<18} h={r['h']:<3} dec={r['n_decisions']:<3}"
                  f"rand={r['random_success']:.2f}{q} {r['objective']}")
        print()
    n_q = sum(1 for r in rows if not r["quiescent"])
    print(f"  * = non-absorbing goal ({n_q} of {len(rows)}): the frame names one tick, so a "
          f"solution must land on it exactly")
    print(f"  mechanics touched: {len(set(m for r in rows for m in r['mechanics']))}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--game", action="append")
    ap.add_argument("--id", action="append")
    ap.add_argument("--out", default="logs/curated/problems.json")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    probs = _problems()
    if a.game:
        probs = [p for p in probs if p.game in a.game]
    if a.id:
        probs = [p for p in probs if p.pid in a.id]

    rows, failed = [], []
    for p in probs:
        t0 = time.time()
        print(f"[{p.game}/{p.pid}] solving...", flush=True)
        row = build_one(p, verbose=a.verbose)
        if row is None:
            failed.append(f"{p.game}/{p.pid}")
            print(f"  FAILED ({time.time() - t0:.0f}s)", flush=True)
            continue
        rows.append(row)
        print(f"  h={row['h']} dec={row['n_decisions']} quiescent={row['quiescent']} "
              f"random={row['random_success']:.2f} ({time.time() - t0:.0f}s)", flush=True)

    report(rows)
    if failed:
        print(f"\n  UNSOLVED: {', '.join(failed)}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out} ({len(rows)} problems)")


if __name__ == "__main__":
    main()
