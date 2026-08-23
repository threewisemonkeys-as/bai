"""NATURAL-LANGUAGE goals for the curated planning set — pilot, one problem per game.

The curated set ships a goal as ONE EXACT FRAME (`curated_plan.py`): unambiguous, but it
hands the agent the answer's shape.  Here the goal is an English sentence and success is
decided by a CHECKER over the executed trajectory, scored ANY-STEP: the run succeeds if the
checker holds after any prefix of the agent's actions.

Two rules govern every entry, and everything below follows from them.

  1. THE CHECKER IS THE DENOTATION OF THE SENTENCE.  Relax every coordinate the sentence
     does not mention; tighten the sentence wherever the checker must stay exact.  The
     curated `Problem.goal` predicates are NOT reusable here -- they were BFS targets, so
     they pin things the English never says (`ice-tower` pins column 4, `infect-all` pins
     nothing but its siblings pin whole particle layouts).  `_bt3gb_ice_tower` below accepts
     a tower at ANY column, because "stack three blocks of ice into a tower" does.

  2. CHECKERS READ RENDERED FRAMES AND ACTIONS, NOTHING ELSE.  `curated_plan`'s hidden-state
     trackers dead-reckon (`cloud`, `active`) from the action string; that is sound along
     the reference plan the solver searched and can desync on an agent's arbitrary sequence,
     which would silently score the wrong thing.  No pilot checker needs them.

ANY-STEP SCORING IS WHAT MAKES OCCLUSION DANGEROUS.  An object that disappears for one frame
because something rendered on top of it is indistinguishable, in that frame, from an object
that is gone.  Measured in the engine, not assumed:

  * n2ntd: a bullet renders ABOVE a coin.  Collect two coins, fire along the third's column,
    and for exactly one frame there are zero gold cells -- "collect all three coins" would
    pass having collected two.  This is not hypothetical: the 42-action witness is in
    `negatives` below, inside the eval's 50-action budget.  A bullet moves one cell per tick,
    so a two-frame `hold` would close it -- but so does refusing to score a frame that has a
    bullet in it, and that version does not also reject a plan ending on the collecting tick.
    The checker names the occluder instead: no gold cells AND no bullet on the board.
  * dq8gc: particles can share a cell (`move-overlap`), so `no gray` alone would pass with a
    healthy particle parked under the infected one -- and unlike the n2ntd bullet this one is
    PERMANENT: contagion crosses orthogonal adjacency, which a particle at distance 0 is not,
    so the survivor never gets infected and the board reads fully-infected for ever (the
    12-action witness is in `negatives`, the same length as the reference solution).
    `len(darkgreen) == 5` closes it: five distinct green cells means five separate infected
    particles.  This does make the checker demand a VISIBLE demonstration -- five infected
    particles stacked on four cells is rejected until the agent separates them, which it can
    always do, since it is the thing doing the stacking.
  * s2kt7: food renders above ants, so food is never occluded -- but the board starts EMPTY,
    so "no food left" is true at t=0 and any-step scoring would hand out a free pass.  The
    checker is a trajectory property: food was present earlier and is gone now.  That alone
    is not enough.  The ants clear the board about 15 ticks after a click almost regardless
    of how much food is out, so 75% of random 50-action drives satisfy "put food out and let
    them eat it" somewhere along the way (measured, N5).  The sentence has to name the
    quantity -- ONE round -- and the checker counts clicks; that drops the floor to zero.
  * 83wkq: two particles on one cell render as one, and back-to-back clicks freeze diffusion
    (`on clicked` assigns `particles`, suppressing that variable's `next` clause), so
    clicking (8,8) twice looks exactly like clicking it once.  The action trace settles it.
    Two particles cannot drift into each other -- every particle takes the SAME random step
    each tick, so they move in lockstep and never collide -- and within the pilot's horizons
    none walks off the rendered grid either, so no fixture exercises this guard; it is kept
    because it costs nothing and the 50-action eval budget is longer than anything probed.

    uv run python -m offline_learning.nl_goals              # summary
    uv run python -m offline_learning.nl_goals --show n2ntd/all-coins
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

Grid = tuple[tuple[str, ...], ...]
Check = Callable[[list[Grid], list[str]], bool]


# --------------------------------------------------------------------------- helpers
def cells(g: Grid, *colors: str) -> frozenset[tuple[int, int]]:
    want = set(colors)
    return frozenset((r, c) for r, row in enumerate(g)
                     for c, v in enumerate(row) if v in want)


def n_clicks(actions: list[str]) -> int:
    return sum(1 for a in actions if a.split()[0] == "click")


def held(pred: Callable[[Grid], bool], grids: list[Grid], n: int) -> bool:
    """`pred` true on each of the last `n` frames.  n=2 defeats one-frame occlusion."""
    return len(grids) >= n and all(pred(g) for g in grids[-n:])


def nverb(actions: list[str], verb: str) -> int:
    return sum(1 for a in actions if a.split()[0] == verb)


def click_cells(actions: list[str]) -> list[tuple[int, int, int]]:
    """(step index, row, col) for every click, in order.  The index is into `actions`, so
    `grids[i]` is the frame the agent was looking at when it issued click `i`."""
    out = []
    for i, a in enumerate(actions):
        p = a.split()
        if p[0] == "click":
            out.append((i, int(p[1]), int(p[2])))
    return out


def flat_run(g: Grid, color: str, n: int) -> bool:
    """Exactly n cells of `color`, all on the floor row, in contiguous columns."""
    w = cells(g, color)
    if len(w) != n or any(r != len(g) - 1 for r, _ in w):
        return False
    cols = {c for _, c in w}
    return max(cols) - min(cols) + 1 == len(cols)


def staircase(g: Grid, color: str) -> bool:
    """Six cells of `color` in three adjacent columns with heights 3-2-1 or 1-2-3, every
    column resting on the floor.  Location and chirality are free -- the sentence says
    "a staircase", not "that staircase"."""
    w = cells(g, color)
    if len(w) != 6:
        return False
    by: dict[int, set[int]] = {}
    for r, c in w:
        by.setdefault(c, set()).add(r)
    cols = sorted(by)
    if len(cols) != 3 or cols[2] - cols[0] != 2:
        return False
    bottom = len(g) - 1
    if any(rs != {bottom - i for i in range(len(rs))} for rs in by.values()):
        return False
    return [len(by[c]) for c in cols] in ([3, 2, 1], [1, 2, 3])


def block2x2(pts: frozenset[tuple[int, int]]) -> bool:
    if len(pts) != 4:
        return False
    rs = {r for r, _ in pts}
    cs = {c for _, c in pts}
    return (len(rs) == 2 and len(cs) == 2
            and max(rs) - min(rs) == 1 and max(cs) - min(cs) == 1)


# --------------------------------------------------------------------- per-game readers
N2NTD_COINS = frozenset({(9, 1), (4, 7), (5, 9)})   # static: `coins` has no next clause


def mario(g: Grid) -> tuple[int, int] | None:
    """None while mario is occluded -- coins, the enemy and bullets all render over him."""
    m = cells(g, "red")
    return next(iter(m)) if len(m) == 1 else None


def standing_on(g: Grid, platform_row: int) -> bool:
    """Mario at rest on the platform whose cells are in `platform_row`.  Measured: the three
    `Step` objects never move (only the enemy patrols), so a platform's ROW names it for
    ever, which is what lets these checkers drop the column the curated predicates pinned."""
    m = mario(g)
    if m is None:
        return False
    r, c = m
    return r == platform_row - 1 and g[platform_row][c] == "darkorange"


def no_bullet_on(g: Grid, where: frozenset[tuple[int, int]]) -> bool:
    """A bullet renders over gold, so a claim that a coin is GONE has to exclude a bullet
    parked on its cell.  The guard names those cells rather than the whole board because a
    bullet that reaches a platform does not despawn -- it STALLS under it for ever (measured:
    a shot from column 6 sits at (9,6) indefinitely).  "No bullet anywhere" would therefore
    permanently reject any plan that wastes a shot, which the two kill problems invite."""
    return not (cells(g, "mediumpurple") & where)


def five_visible(g: Grid) -> bool:
    """All five dq8gc particles rendered on distinct cells.  The driven particle renders OVER
    anything it stands on, and it can also walk off the grid and stop rendering at all."""
    return len(cells(g, "gray")) + len(cells(g, "darkgreen")) == 5


def peak_food(grids: list[Grid]) -> int:
    """Most food ever on the board.  A click spawns two pieces at random positions and they
    sometimes coincide (seed 1, click 5, renders as one cell), so "one of the two" has to be
    counted against what was actually observed."""
    return max(len(cells(g, "red")) for g in grids)


def night(g: Grid) -> bool:
    """The celestial body is a 2x2 block at rows 0-1; the cloud is row 0 only, so row 1
    column 0 reads the day/night flag whatever the cloud is doing."""
    return g[1][0] == "gray"


# -------------------------------------------------------------------------- checkers
# Each takes the frames of the trajectory SO FAR (grids[0] = start, grids[-1] = now) and the
# actions that produced them (len(actions) == len(grids) - 1), and answers: does the
# trajectory ending here satisfy the sentence?

def _n2ntd_all_coins(grids: list[Grid], actions: list[str]) -> bool:
    """"Collect all three coins."

    Coins leave the frame only when mario touches one, so "no gold" IS "all collected" --
    except while a bullet is sitting on a coin's cell.  The clause names the occluder rather
    than asking the empty board to persist (`hold=2`), which would have been the general
    guard but would also fail the most natural correct plan there is: one that ENDS on the
    action collecting the third coin.  A bullet is the only thing in this game that renders
    over a coin.  The clause names the coin CELLS, not the whole board: a bullet that reaches
    a platform stalls under it for ever instead of despawning, so "no bullet anywhere" would
    permanently reject a plan that wasted one shot and then collected everything."""
    g = grids[-1]
    return not cells(g, "gold") and no_bullet_on(g, N2NTD_COINS)


def _bt3gb_ice_tower(grids: list[Grid], actions: list[str]) -> bool:
    """"Stack three blocks of ice into a tower three cells tall, standing on the ground."

    ANY column -- the curated predicate's column 4 is an artifact of where the cloud starts.
    Exactly three ice cells, so a fourth block lying elsewhere is not a tower of three.
    hold=1, unlike n2ntd: a falling stack cannot fake this.  Two `down`s on consecutive ticks
    land on the SAME cell, so drops in flight are >= 2 rows apart, and a run at rows 13/14/15
    means the bottom one is on the ground with the others already resting on it.  Verified by
    the transient-acceptance scan in `validate_nl_goals.py`."""
    ice = cells(grids[-1], "lightblue")
    return (len(ice) == 3 and len({c for _, c in ice}) == 1
            and {r for r, _ in ice} == {13, 14, 15})


def _dq8gc_infect_all(grids: list[Grid], actions: list[str]) -> bool:
    """"Infect every particle, leaving no healthy ones."

    Five particles, all green, no two sharing a cell -- see the overlap note above."""
    g = grids[-1]
    return not cells(g, "gray") and len(cells(g, "darkgreen")) == 5


def _s2kt7_one_round(grids: list[Grid], actions: list[str]) -> bool:
    """"Put out a single round of food and let the ants eat all of it, without putting out
    any more."

    Trajectory property, not a state property: the board is empty at t=0 too.  Food appears
    only from a click and leaves only into an ant, so `was present, now absent` is exactly
    "the ants ate it".  `a single round` is one click, and it is the clause that makes the
    problem non-trivial -- see the s2kt7 note in the module docstring."""
    return (n_clicks(actions) == 1 and not cells(grids[-1], "red")
            and any(cells(g, "red") for g in grids[:-1]))


def _83wkq_spawn_one(grids: list[Grid], actions: list[str]) -> bool:
    """"Create exactly one particle, at row 8, column 8."

    `exactly one` is checked against the ACTION trace because two coincident particles render
    as one cell.  Every click spawns a particle (the handler has no position guard), so the
    click count is the particle count.  hold=1: the particle random-walks every tick, so
    demanding it stay is demanding luck."""
    return n_clicks(actions) == 1 and cells(grids[-1], "blue") == frozenset({(8, 8)})


# ------------------------------------------------------------------ checkers, phase 2
def _n2ntd_platform(grids: list[Grid], actions: list[str]) -> bool:
    """"Stand on the middle platform."  Any cell of it: the curated predicate's column 6 is
    only where a single jump from the start happens to land."""
    return standing_on(grids[-1], 8)


def _n2ntd_high_ground(grids: list[Grid], actions: list[str]) -> bool:
    """"Stand on the highest platform." """
    return standing_on(grids[-1], 6)


def _n2ntd_coin_low(grids: list[Grid], actions: list[str]) -> bool:
    """"Collect the coin resting on the lowest platform, and leave the other two."

    Coins never move, so naming one by where it sits is stable.  The curated predicate also
    demanded mario be standing at (9,1); the sentence says nothing about where he ends up."""
    return cells(grids[-1], "gold") == frozenset({(4, 7), (5, 9)})


def _n2ntd_coin_air(grids: list[Grid], actions: list[str]) -> bool:
    """"Collect the coin floating in mid-air, and leave the other two."  The other two rest
    on platforms; (4,7) is the only one with nothing under it.  No bullet guard: naming the
    survivors as a SET already excludes occlusion -- a bullet parked on a coin changes the
    visible set to one that is not the target set.  Same for the other two coin problems that
    name survivors; only the "no gold at all" forms need the guard."""
    return cells(grids[-1], "gold") == frozenset({(9, 1), (5, 9)})


def _n2ntd_kill_one(grids: list[Grid], actions: list[str]) -> bool:
    """"Collect a single coin and use it to shoot the enemy dead."

    WHICH coin is not in the sentence, so the checker counts instead of naming.  Coins are
    ammo -- clicking with none does nothing -- so a kill implies a collection.  The enemy is
    six cells and only a one-cell bullet renders over it, so `no blue` cannot be faked."""
    g = grids[-1]
    return not cells(g, "blue") and len(cells(g, "gold")) == 2


def _n2ntd_all_kill(grids: list[Grid], actions: list[str]) -> bool:
    """"Collect all three coins and shoot the enemy dead."

    Deliberately NOT "then": the end state cannot show the order, and collect-one / shoot /
    collect-two satisfies the English just as well.  Where order really is the content
    (`infect-all-gather`) the checker reads the prefix instead."""
    g = grids[-1]
    return (not cells(g, "blue") and not cells(g, "gold")
            and no_bullet_on(g, N2NTD_COINS))


def _bt3gb_nightfall(grids: list[Grid], actions: list[str]) -> bool:
    """"Make it night." """
    return night(grids[-1])


def _bt3gb_park_cloud(grids: list[Grid], actions: list[str]) -> bool:
    """"Push the cloud all the way to the left wall without making it rain or changing the
    time of day."

    The cloud is three gray cells in row 0 and clamps with its left edge at column 0.  At
    night the celestial body is gray too, but it occupies columns 0-1 of rows 0 AND 1, so
    "row 0 is gray at column 2 and not at column 3" identifies the parked cloud in both
    phases.  The two abstinence clauses are not decoration: without them a random 50-action
    drive parks the cloud somewhere along the way 45% of the time (measured); with them,
    2%."""
    g = grids[-1]
    return (g[0][2] == "gray" and g[0][3] != "gray"
            and nverb(actions, "down") == 0 and nverb(actions, "click") == 0)


def _bt3gb_one_drop(grids: list[Grid], actions: list[str]) -> bool:
    """"Release a single drop of rain and let it land on the floor at column 2."

    `a single drop` is counted on the ACTION trace, not on the board: two `down`s on
    consecutive ticks put both drops on the same cell and they fall as one for ever."""
    return (nverb(actions, "down") == 1
            and cells(grids[-1], "blue") == frozenset({(15, 2)}))


def _bt3gb_one_ice(grids: list[Grid], actions: list[str]) -> bool:
    """"Make it night so the rain freezes, then land a single block of ice on the floor at
    column 9."  Colour carries the phase, so no separate night clause is needed."""
    return (nverb(actions, "down") == 1
            and cells(grids[-1], "lightblue") == frozenset({(15, 9)}))


def _bt3gb_pool(grids: list[Grid], actions: list[str]) -> bool:
    """"Rain three drops of water and let them spread into a flat pool three cells wide."

    Liquid slides toward the nearest hole in the row below, so three drops always end as a
    contiguous run -- the discriminating half of the pair with `ice-tower`, which can only be
    built by freezing FIRST."""
    return nverb(actions, "down") == 3 and flat_run(grids[-1], "blue", 3)


def _bt3gb_freeze_pool(grids: list[Grid], actions: list[str]) -> bool:
    """"Let three drops of water settle into a flat pool, then freeze the pool solid."

    A flat ice run can only be made by freezing AFTER the drops settle; freezing first gives
    a tower.  The shapes are what separate them, so no ordering clause is needed."""
    return nverb(actions, "down") == 3 and flat_run(grids[-1], "lightblue", 3)


def _bt3gb_staircase(grids: list[Grid], actions: list[str]) -> bool:
    """"Build a staircase out of ice: three blocks tall in one column, two in the next, one
    in the next."  Any location, either direction -- the curated predicate pinned columns
    8/9/10 and one chirality."""
    return nverb(actions, "down") == 6 and staircase(grids[-1], "lightblue")


def _dq8gc_walk(grids: list[Grid], actions: list[str]) -> bool:
    """"Drive the green particle into the top-left corner without infecting anyone." """
    g = grids[-1]
    return (cells(g, "darkgreen") == frozenset({(0, 0)})
            and len(cells(g, "gray")) == 4)


def _dq8gc_infect_one(grids: list[Grid], actions: list[str]) -> bool:
    """"Infect exactly one of the four healthy particles and leave the other three healthy,
    without ever taking control of another particle."

    WHICH one is not in the sentence.  The abstinence clause is the floor fix: random drives
    click constantly, and without it a random 50-action drive passes through "exactly one
    infected" 61% of the time (measured); with it, 18%.  Still the highest floor in the set,
    and reported as such."""
    g = grids[-1]
    return (len(cells(g, "darkgreen")) == 2 and len(cells(g, "gray")) == 3
            and nverb(actions, "click") == 0)


def _dq8gc_swap_drive(grids: list[Grid], actions: list[str]) -> bool:
    """"Take control of one of the healthy particles and drive it to the left wall, keeping
    it healthy."

    Colour is HEALTH, not control: after a click you are driving a GRAY particle, and the one
    you let go of stays green.  A click is invisible at t+1, so what the sentence can name is
    the particle that MOVED -- which makes the select instrumental and observable."""
    g = grids[-1]
    return five_visible(g) and any(c == 0 for _, c in cells(g, "gray"))


def _dq8gc_chain(grids: list[Grid], actions: list[str]) -> bool:
    """"Infect a healthy particle, take control of that one, and use it to infect a third."

    The handoff leaves no trace in the end state, so the checker reads the action trace
    against the frame the agent was looking at: a click on a cell that was green at the
    moment it was clicked."""
    g = grids[-1]
    if len(cells(g, "darkgreen")) < 3 or not five_visible(g):
        return False
    return any((r, c) in cells(grids[i], "darkgreen")
               for i, r, c in click_cells(actions))


def _dq8gc_gather(grids: list[Grid], actions: list[str]) -> bool:
    """"Herd the four healthy particles into a 2x2 block, without infecting any of them."

    Any location -- the curated predicate pinned rows 5-6, columns 5-6.  Contagion crosses
    orthogonal adjacency only (measured: diagonal contact does not infect), so a block is
    reachable as long as the green one is kept a step away."""
    g = grids[-1]
    return block2x2(cells(g, "gray")) and len(cells(g, "darkgreen")) == 1


def _dq8gc_infect_all_gather(grids: list[Grid], actions: list[str]) -> bool:
    """"Herd the four healthy particles into a 2x2 block, then infect the whole block."

    Here the order IS the content, and the prefix is where it is visible: the block has to
    have existed while they were all still healthy."""
    g = grids[-1]
    if cells(g, "gray") or len(cells(g, "darkgreen")) != 5:
        return False
    return any(block2x2(cells(x, "gray")) and len(cells(x, "darkgreen")) == 1
               for x in grids[:-1])


def _s2kt7_spawn(grids: list[Grid], actions: list[str]) -> bool:
    """"Put out one round of food." """
    return nverb(actions, "click") == 1 and bool(cells(grids[-1], "red"))


def _s2kt7_one_eaten(grids: list[Grid], actions: list[str]) -> bool:
    """"Put out a single round of food and let the ants eat exactly one piece of it."

    Counted against the round's OBSERVED peak rather than against two: the two pieces spawn
    at random positions and sometimes land on the same cell."""
    if nverb(actions, "click") != 1:
        return False
    peak = peak_food(grids)
    return peak >= 2 and len(cells(grids[-1], "red")) == peak - 1


def _rounds(grids: list[Grid], actions: list[str], k: int) -> bool:
    """"Put out k rounds of food and let the ants eat every piece."

    Rounds are allowed to overlap, because they do: the curated three-rounds reference clicks
    again two ticks after the previous round while food is still on the board.  So what the
    sentence can require is that there were k rounds, that each one actually put food out,
    and that none of it is left."""
    idx = [i for i, _, _ in click_cells(actions)]
    if len(idx) != k or cells(grids[-1], "red"):
        return False
    return all(any(cells(g, "red") for g in grids[i + 1:]) for i in idx)


def _s2kt7_two_rounds(grids: list[Grid], actions: list[str]) -> bool:
    return _rounds(grids, actions, 2)


def _s2kt7_three_rounds(grids: list[Grid], actions: list[str]) -> bool:
    return _rounds(grids, actions, 3)


def _s2kt7_intercept(grids: list[Grid], actions: list[str]) -> bool:
    """"Put out a single round of food and stop the moment an ant is right next to a piece,
    before anything has been eaten."

    Food renders OVER ants, so an ant standing ON a piece is invisible -- which is why the
    relation is adjacency, which is visible, rather than contact, which is not.  This is the
    one goal in the set whose content is timing, and any-step scoring cannot enforce
    stopping: a run that clicks once and waits passes through the accepting frame on its way
    to anywhere.  Kept because the frame it replaces was a pure coordinate dump, and reported
    with that caveat."""
    g = grids[-1]
    if nverb(actions, "click") != 1:
        return False
    food = cells(g, "red")
    if not food or len(food) != peak_food(grids):
        return False
    ants = cells(g, "gray")
    return any(abs(a[0] - f[0]) + abs(a[1] - f[1]) == 1 for a in ants for f in food)


def _83wkq_spawn_two(grids: list[Grid], actions: list[str]) -> bool:
    """"Create a particle at row 4 column 4; wait a tick; then create a second at row 10
    column 10."

    The curated goal was one exact frame and could not be DERIVED -- the first particle takes
    a uniform random step, so the frame could only be memorised from the RNG.  Naming the two
    spawns instead makes the goal a consequence of the rules, and the load-bearing noop
    survives: `on clicked` assigns `particles` and so suppresses that variable's next clause
    for the tick, so back-to-back clicks freeze diffusion and (4,4) is still occupied."""
    cl = click_cells(actions)
    if len(cl) != 2 or cl[1][0] - cl[0][0] < 2:
        return False
    if (cl[0][1], cl[0][2]) != (4, 4) or (cl[1][1], cl[1][2]) != (10, 10):
        return False
    b = cells(grids[-1], "blue")
    return len(b) == 2 and (4, 4) not in b and (10, 10) in b


# ----------------------------------------------------------------------------- goals
@dataclass
class NLGoal:
    game: str
    pid: str                      # matches `id` in the curated problems.json
    tier: str
    seed: int
    nl: str                       # the goal, as stated to the agent (intentional register)
    check: Check
    hold: int = 1                 # frames the state condition must persist (documentation)
    positives: list[list[str]] = field(default_factory=list)   # MUST be accepted
    negatives: list[list[str]] = field(default_factory=list)   # MUST be rejected
    note: str = ""
    ref: list[str] | None = None  # reference plan, when the CURATED one does not satisfy the
                                  # sentence -- `intercept` names a relation the curated plan
                                  # stops one tick short of.  None = use the curated plan.
    naive: Check | None = None    # the same checker with its anti-exploit clause removed.
                                  # Not scored; N6 reports it so each guard's cost is on the
                                  # record instead of being asserted.


# `positives` are alternate routes the sentence plainly allows and the exact-frame goal would
# have rejected; `negatives` are near misses that separate the sentence from a looser reading
# -- most of them are accepted by the checker with its guard clause removed, which is the
# point.  Both are literal action sequences replayed from a bare reset, not slices of the
# curated reference plan: a fixture that reads from another artifact stops testing anything
# the day that artifact is rebuilt.  Every one is verified by `validate_nl_goals.py` (N6).
_WALK_TO_COIN_1 = ["left"] * 5 + ["up", "up", "noop", "up"]          # -> coin (9,1) collected
_ALSO_COIN_2 = _WALK_TO_COIN_1 + ["up"] + ["right"] * 8 + ["up"]     # -> coin (5,9) too

_PILOT: list[NLGoal] = [
    NLGoal(
        game="n2ntd", pid="all-coins", tier="L3", seed=0,
        nl="Collect all three coins.",
        check=_n2ntd_all_coins,
        naive=lambda gr, ac: not cells(gr[-1], "gold"),
        positives=[
            # Reference route, cut the instant the third coin goes: mario is in mid-air at
            # (6,7) rather than landed at (11,7).  The curated frame goal rejects this; the
            # sentence says nothing about where he ends up, so the checker accepts it.
            _ALSO_COIN_2 + ["up", "up", "up", "up", "left", "left", "up", "noop"],
        ],
        negatives=[
            # Two of three, then wait.
            _ALSO_COIN_2 + ["noop"] * 6,
            # Two of three, then fire along the third coin's column: the bullet renders over
            # the coin at (4,7) and for exactly one frame there is no gold on the board.
            # Accepted by the checker without its no-bullet clause, rejected with it.
            _ALSO_COIN_2 + ["left", "left"] + ["noop"] * 8 + ["click 0 0"] + ["noop"] * 12,
        ],
        note="curated predicate additionally demanded mario be at rest -- an artifact of "
             "needing an absorbing frame, dropped here: the sentence says nothing about "
             "where he ends up.",
    ),
    NLGoal(
        game="bt3gb", pid="ice-tower", tier="L3", seed=0,
        nl="Stack three blocks of ice into a tower three cells tall, standing on the ground.",
        check=_bt3gb_ice_tower,
        positives=[
            # Same tower, three columns to the right: the relaxation from the curated
            # predicate's column 4, and the whole reason this problem needed re-authoring.
            ["click 8 8", "right", "right", "right", "down", "noop", "down", "noop", "noop",
             "down"] + ["noop"] * 14,
        ],
        negatives=[
            # Three blocks of ice lying FLAT on the floor (the curated freeze-pool route):
            # same material, same count, not a tower.  This is the discriminating pair, and
            # it survives the column relaxation.
            ["down", "noop", "down", "noop", "noop", "down"] + ["noop"] * 12 + ["click 8 8"],
            # A tower of two.
            ["click 8 8", "down", "noop", "down"] + ["noop"] * 16,
        ],
        note="relaxed from column 4 to any column.",
    ),
    NLGoal(
        game="dq8gc", pid="infect-all", tier="L3", seed=0,
        nl="Infect every particle, leaving no healthy ones.",
        check=_dq8gc_infect_all,
        naive=lambda gr, ac: not cells(gr[-1], "gray"),
        positives=[
            # Reference route, then keep driving.  Infection is permanent, so the sentence
            # stays true; the curated frame goal, which pinned every particle's cell, does not.
            ["down", "right", "up", "down", "down", "up", "down", "down", "right", "right",
             "right", "up", "left", "left", "up", "noop"],
        ],
        negatives=[
            # Four of five.
            ["down", "right", "up", "down", "down", "up", "down", "down"] + ["noop"] * 4,
            # Four infected, and the fifth parked UNDER the one being driven: no gray cell
            # anywhere, permanently, with a healthy particle alive at (5,7).  Twelve actions,
            # exactly as long as the real solution, and accepted for ever by the checker
            # without its `len(darkgreen) == 5` clause.
            ["down", "right", "up", "down", "down", "up", "down", "down",
             "right", "right", "right", "right"] + ["noop"] * 8,
        ],
        note="the curated predicate was already faithful; kept as-is.",
    ),
    NLGoal(
        game="s2kt7", pid="all-eaten", tier="L3", seed=1,
        nl="Put out a single round of food and let the ants eat all of it, without putting "
           "out any more.",
        check=_s2kt7_one_round,
        naive=lambda gr, ac: not cells(gr[-1], "red"),
        positives=[
            # Click anywhere: spawn positions come from the RNG, indexed by click ORDER, not
            # by where the click landed.
            ["click 7 7"] + ["noop"] * 14,
            # Dither first, then feed them.
            ["noop", "noop", "click 0 0"] + ["noop"] * 14,
        ],
        negatives=[
            # Never put food out.  The board is empty at t=0, so "no food left" is true from
            # the first frame -- under any-step scoring this is the free pass the trajectory
            # clause exists to refuse.
            ["noop"] * 20,
            # One of the two foods eaten.
            ["click 0 0"] + ["noop"] * 8,
            # Two rounds put out, both eventually eaten.  This is what 75% of random drives
            # stumble into, and the reading of the sentence that has to be excluded.
            ["click 0 0", "click 5 5"] + ["noop"] * 20,
        ],
        note="the curated predicate read the hidden click counter (== 1) and its objective "
             "line did not mention it; here the count is in the sentence, because without it "
             "the goal is reachable by flailing.",
    ),
    NLGoal(
        game="83wkq", pid="spawn-one", tier="L1", seed=0,
        nl="Create exactly one particle, at row 8, column 8.",
        check=_83wkq_spawn_one,
        naive=lambda gr, ac: cells(gr[-1], "blue") == frozenset({(8, 8)}),
        positives=[
            ["click 8 8"],
            ["noop", "noop", "click 8 8"],
            # Clicking (8,8) twice renders as ONE cell (coincident particles, and the second
            # click freezes diffusion for that tick) -- but the prefix after the first click
            # already satisfied the sentence, and any-step scoring credits that.  Listed as a
            # positive so the semantics are on the record rather than discovered later.
            ["click 8 8", "click 8 8"],
        ],
        negatives=[
            # Two particles, one of them at (8,8): the frame alone cannot tell this from one
            # particle at (8,8) plus one elsewhere, so the click count does the work.
            ["click 3 3", "click 8 8"],
            ["click 3 3"],
        ],
        note="coordinate-laden by necessity: a click is the only input and every particle "
             "immediately random-walks, so the only thing a sentence can pin is the spawn.",
    ),
]

# Fixtures for the 25 below follow the same rule as the pilot's: literal action sequences
# replayed from a bare reset, never slices of another artifact.  Every one was replayed
# through the engine while it was written -- the comments record what the replay showed, not
# what the rules suggested it should show.
_GET_LOW_COIN = ["left"] * 5 + ["up", "up", "noop", "up"]      # -> coin (9,1), on the platform
_ALL_THREE = _GET_LOW_COIN + [
    "up", "right", "right", "right", "right", "right", "right", "right", "right",
    "up", "up", "up", "up", "up", "left", "left", "up"]        # -> all three coins

_PHASE2: list[NLGoal] = [
    # ------------------------------------------------------------------ n2ntd / mario
    NLGoal(
        game="n2ntd", pid="platform", tier="L1", seed=0,
        nl="Stand on the middle platform.",
        check=_n2ntd_platform,
        positives=[
            ["left", "up"],          # column 5, not the curated column 6
            ["noop", "noop", "up"],
        ],
        negatives=[
            ["noop"] * 6,                                  # never jumps
            ["right", "right", "up"] + ["noop"] * 4,       # jumps where there is no platform
            ["left", "left", "left", "up"] + ["noop"] * 4,
        ],
        note="floor-dominated: a random 50-action drive passes through this 84% of the time. "
             "One jump is the whole task and no clause can change that -- reported, not "
             "headlined.",
    ),
    NLGoal(
        game="n2ntd", pid="high-ground", tier="L1", seed=0,
        nl="Stand on the highest platform.",
        check=_n2ntd_high_ground,
        positives=[["up", "up", "right", "right", "noop", "right", "noop", "noop"]],
        negatives=[
            ["up"] + ["noop"] * 6,           # the MIDDLE platform: right shape, wrong one
            ["up", "up"] + ["noop"] * 6,     # jumped twice, never walked right
            ["noop"] * 8,
        ],
        note="relaxed from column 8 to any cell of the platform.",
    ),
    NLGoal(
        game="n2ntd", pid="coin-ground", tier="L2", seed=0,
        nl="Collect the coin resting on the lowest platform, and leave the other two alone.",
        check=_n2ntd_coin_low,
        positives=[["left", "noop", "left", "left", "left", "left", "up", "up", "noop", "up"]],
        negatives=[
            ["left"] * 5 + ["noop"] * 4,     # walked there, never jumped
            ["noop"] * 10,
            # The mid-air coin instead: gold never equals the target pair.
            ["up", "up", "right", "noop", "noop", "right"] + ["noop"] * 8,
        ],
        note="curated predicate also demanded mario be standing at (9,1); the sentence says "
             "nothing about where he ends up.  `leave the other two alone` is enforceable "
             "only at the accepting instant: a run that collects this coin FIRST and then "
             "goes on to take the others passes through the accepting state, and any-step "
             "scoring credits that prefix.  So collect-all-three is not a negative here (it "
             "is for `coin-air`, whose coin is not the one collected first).",
    ),
    NLGoal(
        game="n2ntd", pid="coin-air", tier="L2", seed=0,
        nl="Collect the coin floating in mid-air, and leave the other two alone.",
        check=_n2ntd_coin_air,
        positives=[["up", "up", "right", "noop", "noop", "right"] + ["noop"] * 8],
        negatives=[
            _GET_LOW_COIN + ["noop"] * 4,    # the wrong coin
            ["noop"] * 10,
            _ALL_THREE,
        ],
        note="the other two coins rest on platforms; (4,7) is the only one with nothing "
             "under it, so 'floating in mid-air' picks it out without coordinates.",
    ),
    NLGoal(
        game="n2ntd", pid="kill-one-coin", tier="L3", seed=0,
        nl="Collect a single coin and use it to shoot the enemy dead.",
        check=_n2ntd_kill_one,
        positives=[
            # Same coin, fired three ticks later in the window -- the enemy patrols, so WHEN
            # you fire is the whole difficulty, and the curated frame pinned one tick of it.
            _GET_LOW_COIN + ["noop"] * 5 + ["click 0 0"] + ["noop"] * 16,
        ],
        negatives=[
            _GET_LOW_COIN + ["noop"] * 15,                       # coin, never fired
            ["click 0 0"] * 3 + ["noop"] * 10,                   # fired with no ammo: nothing
            ["left", "up", "noop", "noop", "up"] + ["noop"] * 20,
        ],
        note="WHICH coin is not in the sentence, so the checker counts survivors instead of "
             "naming them.",
    ),
    NLGoal(
        game="n2ntd", pid="all-coins-kill", tier="L4", seed=0,
        nl="Collect all three coins and shoot the enemy dead.",
        check=_n2ntd_all_kill,
        naive=lambda gr, ac: (not cells(gr[-1], "blue") and not cells(gr[-1], "gold")),
        positives=[_ALL_THREE + ["noop", "noop", "click 0 0"] + ["noop"] * 16],
        negatives=[
            _ALL_THREE + ["noop"] * 8,                                   # coins, no kill
            _GET_LOW_COIN + ["noop"] * 3 + ["click 0 0"] + ["noop"] * 12,  # kill, one coin
            ["noop"] * 20,
        ],
        note="not 'then': the end state cannot show the order, and collect-one / shoot / "
             "collect-two satisfies the English just as well.",
    ),

    # -------------------------------------------------------------------- bt3gb / ice
    NLGoal(
        game="bt3gb", pid="nightfall", tier="L1", seed=0,
        nl="Make it night.",
        check=_bt3gb_nightfall,
        positives=[["noop", "noop", "click 0 0"]],   # the click position is ignored
        negatives=[["noop"] * 5, ["down", "noop", "down", "noop"], ["left", "right", "left"]],
        note="floor-dominated: a toggle reached by one click, and a random 50-action drive "
             "clicks. Measured floor 1.000; the tightest sentence found still floors at "
             "0.515. Reported, not headlined.",
    ),
    NLGoal(
        game="bt3gb", pid="park-cloud", tier="L1", seed=0,
        nl="Push the cloud all the way to the left wall, without making it rain and without "
           "changing the time of day.",
        check=_bt3gb_park_cloud,
        naive=lambda gr, ac: gr[-1][0][2] == "gray" and gr[-1][0][3] != "gray",
        positives=[
            ["left", "noop", "left", "noop", "left"],
            ["left"] * 5,                       # the cloud clamps; overshooting is allowed
        ],
        negatives=[
            ["left", "left"] + ["noop"] * 3,                    # one column short
            ["down", "left", "left", "left"] + ["noop"] * 3,    # rained on the way
            ["click 8 8", "left", "left", "left"] + ["noop"] * 3,
            ["right", "left", "left", "left"] + ["noop"] * 3,
        ],
        note="the two abstinence clauses are the floor fix: 0.450 -> 0.020 measured. They "
             "also read as ordinary English, which is the test a tightening has to pass.",
    ),
    NLGoal(
        game="bt3gb", pid="one-drop", tier="L2", seed=0,
        nl="Release a single drop of rain and let it land on the floor at column 2.",
        check=_bt3gb_one_drop,
        naive=lambda gr, ac: cells(gr[-1], "blue") == frozenset({(15, 2)}),
        positives=[["left", "noop", "left", "noop", "down"] + ["noop"] * 20],
        negatives=[
            ["left", "down"] + ["noop"] * 20,                       # column 3
            ["left", "left", "down", "noop", "down"] + ["noop"] * 20,   # two drops
            ["click 8 8", "left", "left", "down"] + ["noop"] * 20,      # froze it
        ],
        note="'a single drop' is counted on the ACTION trace: two downs close together put "
             "both drops on one cell and they fall as one for ever.",
    ),
    NLGoal(
        game="bt3gb", pid="one-ice", tier="L2", seed=0,
        nl="Make it night so the rain freezes, then land a single block of ice on the floor "
           "at column 9.",
        check=_bt3gb_one_ice,
        naive=lambda gr, ac: cells(gr[-1], "lightblue") == frozenset({(15, 9)}),
        positives=[["right"] * 5 + ["click 8 8", "down"] + ["noop"] * 20],
        negatives=[
            ["right"] * 5 + ["down"] + ["noop"] * 20,                        # still liquid
            ["click 8 8"] + ["right"] * 5 + ["down", "noop", "down"] + ["noop"] * 20,
            ["click 8 8"] + ["right"] * 4 + ["down"] + ["noop"] * 20,        # column 8
        ],
        note="no separate night clause: the colour carries the phase.",
    ),
    NLGoal(
        game="bt3gb", pid="pool", tier="L2", seed=0,
        nl="Rain exactly three drops of water and let them spread into a flat pool three "
           "cells wide on the floor.",
        check=_bt3gb_pool,
        naive=lambda gr, ac: flat_run(gr[-1], "blue", 3),
        positives=[["down", "noop", "noop", "down", "noop", "noop", "down"] + ["noop"] * 20],
        negatives=[
            ["down", "noop", "down"] + ["noop"] * 20,                        # two
            ["down", "noop", "down", "noop", "down", "noop", "down"] + ["noop"] * 20,
            ["click 8 8", "down", "noop", "down", "noop", "down"] + ["noop"] * 20,  # ice
        ],
        note="freezing AFTER the pool forms is not a negative and should not be -- that run "
             "passes through a flat blue pool, which is what the sentence asks for.",
    ),
    NLGoal(
        game="bt3gb", pid="freeze-pool", tier="L3", seed=0,
        nl="Let three drops of water settle into a flat pool on the floor, then freeze the "
           "pool solid.",
        check=_bt3gb_freeze_pool,
        naive=lambda gr, ac: flat_run(gr[-1], "lightblue", 3),
        positives=[["down", "noop", "down", "noop", "down"] + ["noop"] * 18 + ["click 8 8"]],
        negatives=[
            # Freeze FIRST and you get a tower, not a flat run: the discriminating pair with
            # ice-tower survives the relaxation, because the shapes cannot be confused.
            ["click 8 8", "down", "noop", "down", "noop", "down"] + ["noop"] * 20,
            ["down", "noop", "down", "noop", "down"] + ["noop"] * 20,        # never froze
            ["down", "noop", "down"] + ["noop"] * 15 + ["click 8 8"] + ["noop"] * 3,
        ],
    ),
    NLGoal(
        game="bt3gb", pid="staircase", tier="L4", seed=0,
        nl="Build a staircase out of ice: three blocks tall in one column, two in the next, "
           "one in the next.",
        check=_bt3gb_staircase,
        naive=lambda gr, ac: staircase(gr[-1], "lightblue"),
        positives=[
            # The MIRROR of the curated route -- 3-2-1 left to right instead of 1-2-3, built
            # at columns 4/5/6 instead of 8/9/10.  Both relaxations in one fixture.
            ["click 8 8", "down", "noop", "down", "noop", "noop", "down",
             "right", "down", "noop", "down", "right", "down"] + ["noop"] * 22,
        ],
        negatives=[
            ["click 8 8", "down", "noop", "down", "noop", "noop", "down"] + ["noop"] * 20,
            ["down", "noop", "down", "noop", "down", "noop", "down", "noop", "down", "noop",
             "down"] + ["noop"] * 15 + ["click 8 8"],                        # six, but flat
            ["click 8 8", "down", "noop", "down", "noop", "noop", "down",
             "right", "down", "noop", "down", "noop", "noop", "down"] + ["noop"] * 22,  # 3+3
        ],
        note="drops must be SPACED, and the spacing needed is longer than it looks: a new "
             "drop appears at row 1 and the previous one cannot move on a `down` tick, so "
             "down/noop/down/noop/down merges the last two into one cell.",
    ),

    # ---------------------------------------------------------------- dq8gc / disease
    NLGoal(
        game="dq8gc", pid="walk", tier="L1", seed=0,
        nl="Drive the green particle into the top-left corner, without infecting anyone.",
        check=_dq8gc_walk,
        positives=[["left", "left", "up", "up"]],          # same corner, other order
        negatives=[
            ["left"] * 4 + ["up", "up"],                   # drove it off the grid: nothing renders
            ["up", "up"] + ["noop"] * 4,
            ["noop"] * 6,
        ],
        note="'without infecting anyone' is free here -- the corner is nowhere near a healthy "
             "particle -- but it is what makes the four-gray clause part of the sentence "
             "rather than a hidden extra.",
    ),
    NLGoal(
        game="dq8gc", pid="infect-one", tier="L2", seed=0,
        nl="Infect exactly one of the four healthy particles and leave the other three "
           "healthy, without ever taking control of another particle.",
        check=_dq8gc_infect_one,
        naive=lambda gr, ac: (len(cells(gr[-1], "darkgreen")) == 2
                              and len(cells(gr[-1], "gray")) == 3),
        positives=[["right", "right", "noop", "noop"]],
        negatives=[
            # Two at once: (5,6) is orthogonally adjacent to BOTH (5,7) and (6,6), so the
            # count goes 1 -> 3 and never passes through two.  Routed along row 1 to avoid
            # brushing (3,4) on the way.
            ["up", "right", "right", "right", "right", "down", "down", "down", "down",
             "noop", "noop"],
            ["up", "up", "up", "noop", "noop"],                       # infected nobody
            ["click 3 4", "down", "down", "noop", "noop", "noop"],    # took control
        ],
        note="the highest floor in the set even after tightening (0.175): random walking in "
             "a cluster of four infects exactly one of them fairly often.",
    ),
    NLGoal(
        game="dq8gc", pid="swap-drive", tier="L2", seed=0,
        nl="Take control of one of the healthy particles and drive it to the left wall, "
           "keeping it healthy.",
        check=_dq8gc_swap_drive,
        positives=[["click 3 4", "left", "left", "left", "left"]],   # a different particle
        negatives=[
            ["left", "left", "left", "left"],       # drove the GREEN one: not a healthy one
            ["click 5 3"] + ["noop"] * 4,           # selected, never drove
            ["noop"] * 6,
        ],
        note="colour is HEALTH, not control: after the click you are driving a gray particle "
             "and the one you let go of stays green, so no sentence here may name the "
             "controlled particle by colour.",
    ),
    NLGoal(
        game="dq8gc", pid="chain", tier="L3", seed=0,
        nl="Infect a healthy particle, take control of that one, and use it to infect a "
           "third.",
        check=_dq8gc_chain,
        positives=[["right", "right", "noop", "click 3 4", "down", "left", "noop", "noop"]],
        negatives=[
            # Three infected by driving the ORIGINAL particle around: no handoff.
            ["down", "right", "up", "down", "down", "up", "down", "down"] + ["noop"] * 4,
            ["right", "right", "noop", "click 3 4"] + ["noop"] * 4,
            ["noop"] * 8,
        ],
        note="the handoff leaves no trace in the end state, so the checker reads the click "
             "against the frame the agent was looking at when it clicked.",
    ),
    NLGoal(
        game="dq8gc", pid="gather", tier="L4", seed=0,
        nl="Herd the four healthy particles into a 2x2 block, without infecting any of them.",
        check=_dq8gc_gather,
        positives=[["click 3 4", "down", "down", "right", "click 5 3", "down", "right",
                    "right", "click 5 7", "left", "noop"]],
        negatives=[
            ["click 3 4", "down", "down", "right", "click 5 3", "down", "right"] + ["noop"] * 4,
            ["noop"] * 10,
            ["click 3 4", "down", "down"] + ["noop"] * 6,
        ],
        note="any location -- the curated predicate pinned rows 5-6, columns 5-6.",
    ),
    NLGoal(
        game="dq8gc", pid="infect-all-gather", tier="L4", seed=0,
        nl="Herd the four healthy particles into a 2x2 block, then infect the whole block.",
        check=_dq8gc_infect_all_gather,
        naive=lambda gr, ac: (not cells(gr[-1], "gray")
                              and len(cells(gr[-1], "darkgreen")) == 5),
        positives=[["click 3 4", "down", "down", "right", "click 5 3", "down", "right",
                    "right", "click 5 7", "left", "click 2 2", "down", "down", "right",
                    "right", "right", "noop", "noop", "noop", "noop"]],
        negatives=[
            # The block, never infected.
            ["click 3 4", "down", "down", "right", "click 5 3", "down", "right", "right",
             "click 5 7", "left"] + ["noop"] * 6,
            # Everyone infected where they stand -- no block was ever formed.
            ["down", "right", "up", "down", "down", "up", "down", "down", "right", "right",
             "right", "up"] + ["noop"] * 4,
            ["noop"] * 12,
        ],
        note="the only goal in the set where ORDER is the content, and the prefix is where "
             "it is visible: the block has to have existed while they were all healthy.",
    ),

    # ------------------------------------------------------------------ s2kt7 / ants
    NLGoal(
        game="s2kt7", pid="spawn", tier="L1", seed=1,
        nl="Put out one round of food.",
        check=_s2kt7_spawn,
        naive=lambda gr, ac: bool(cells(gr[-1], "red")),
        positives=[["noop", "noop", "click 9 9"]],   # spawn positions ignore the click cell
        negatives=[["noop"] * 20],
        note="one negative, not three: the only way to fail this sentence is never to click. "
             "Floor-dominated (1.000) and irreducible -- 'and let none of it be eaten' does "
             "not move it, because the accepting frame is the tick after the click.",
    ),
    NLGoal(
        game="s2kt7", pid="one-eaten", tier="L2", seed=1,
        nl="Put out a single round of food and let the ants eat exactly one piece of it.",
        check=_s2kt7_one_eaten,
        naive=lambda gr, ac: len(cells(gr[-1], "red")) == 1,
        positives=[
            ["click 9 9"] + ["noop"] * 12,
            ["noop", "noop", "click 0 0"] + ["noop"] * 12,
        ],
        negatives=[
            ["noop"] * 20,
            ["click 0 0"] + ["noop"] * 5,                 # stopped before they reached it
            ["click 0 0", "click 0 0"] + ["noop"] * 20,   # two rounds
        ],
        note="counted against the round's OBSERVED peak, not against two: the pieces spawn "
             "at random positions and sometimes land on the same cell.",
    ),
    NLGoal(
        game="s2kt7", pid="two-rounds", tier="L3", seed=1,
        nl="Put out two rounds of food and let the ants eat every piece.",
        check=_s2kt7_two_rounds,
        naive=lambda gr, ac: not cells(gr[-1], "red"),
        positives=[["click 0 0"] + ["noop"] * 15 + ["click 0 0"] + ["noop"] * 8],
        negatives=[
            ["click 0 0"] + ["noop"] * 18,                                   # one round
            ["click 0 0"] + ["noop"] * 12 + ["click 0 0", "noop", "click 0 0"] + ["noop"] * 5,
            ["click 0 0"] + ["noop"] * 12 + ["click 0 0", "noop"],           # food still out
        ],
        note="rounds are allowed to overlap because they do -- the curated three-rounds "
             "reference clicks again two ticks after the previous round, with food still on "
             "the board. So the sentence says 'two rounds' and 'every piece', not 'wait "
             "until it is all eaten', which the reference would violate.",
    ),
    NLGoal(
        game="s2kt7", pid="three-rounds", tier="L4", seed=1,
        nl="Put out three rounds of food and let the ants eat every piece.",
        check=_s2kt7_three_rounds,
        naive=lambda gr, ac: not cells(gr[-1], "red"),
        positives=[["click 0 0"] + ["noop"] * 13 + ["click 0 0"] + ["noop"] * 5
                   + ["click 0 0"] + ["noop"] * 10],
        negatives=[
            ["click 0 0"] + ["noop"] * 12 + ["click 0 0"] + ["noop"] * 8,    # two rounds
            ["click 0 0"] + ["noop"] * 12 + ["click 0 0", "noop", "click 0 0", "noop",
                                             "click 0 0"] + ["noop"] * 10,   # four
            ["click 0 0"] + ["noop"] * 12 + ["click 0 0", "noop", "click 0 0", "noop"],
        ],
    ),
    NLGoal(
        game="s2kt7", pid="intercept", tier="L3", seed=1,
        nl="Put out a single round of food and stop the moment an ant is right next to a "
           "piece of it, before anything has been eaten.",
        check=_s2kt7_intercept,
        naive=lambda gr, ac: any(
            abs(a[0] - f[0]) + abs(a[1] - f[1]) == 1
            for a in cells(gr[-1], "gray") for f in cells(gr[-1], "red")),
        ref=["click 0 0"] + ["noop"] * 10,
        positives=[["click 5 5"] + ["noop"] * 10],
        negatives=[
            ["noop"] * 15,
            ["click 0 0"] + ["noop"] * 5,                      # stopped too early
            ["click 0 0", "click 0 0"] + ["noop"] * 15,        # two rounds
        ],
        note="the curated plan is one tick SHORT of this -- its goal was a named mid-forage "
             "frame, and adjacency happens at t=11, not t=10 -- so this goal carries its own "
             "reference. Any-step scoring cannot enforce the 'stop': a run that clicks once "
             "and waits passes through the accepting frame on its way anywhere. Kept because "
             "the frame it replaces was a pure coordinate dump; reported with that caveat.",
    ),

    # ------------------------------------------------------------- 83wkq / particles
    NLGoal(
        game="83wkq", pid="spawn-two", tier="L2", seed=0,
        nl="Create a particle at row 4, column 4; wait a tick; then create a second at row "
           "10, column 10.",
        check=_83wkq_spawn_two,
        naive=lambda gr, ac: (len(cells(gr[-1], "blue")) == 2
                              and (4, 4) not in cells(gr[-1], "blue")),
        positives=[["click 4 4", "noop", "noop", "click 10 10"]],
        negatives=[
            ["click 4 4", "click 10 10"],            # no gap: diffusion frozen, (4,4) occupied
            ["click 3 3", "noop", "click 10 10"],    # wrong first cell
            ["click 4 4", "noop", "noop"],           # one particle
            ["click 10 10", "noop", "click 4 4"],    # reversed
        ],
        note="the curated goal was an exact frame that could only be MEMORISED -- the first "
             "particle takes a uniform random step. Naming the two spawns instead makes the "
             "goal a consequence of the rules, and takes the random floor from 0.510 (for "
             "'two particles, one of them drifted') to 0.000.",
    ),
]


_ORDER = ["n2ntd", "bt3gb", "dq8gc", "s2kt7", "83wkq"]
_TIER = {"L1": 0, "L2": 1, "L3": 2, "L4": 3}
GOALS: list[NLGoal] = sorted(
    _PILOT + _PHASE2,
    key=lambda g: (_ORDER.index(g.game), _TIER[g.tier], g.pid))

BY_PID: dict[str, NLGoal] = {g.pid: g for g in GOALS}


# ------------------------------------------------------------------------- evaluation
def satisfied_steps(goal: NLGoal, grids: list[Grid], actions: list[str]) -> list[int]:
    """Every step index k >= 1 whose prefix satisfies the goal.  ANY-STEP scoring is
    `bool(satisfied_steps(...))`; k is reported so a run can say how fast it got there and
    whether it held."""
    return [k for k in range(1, len(grids))
            if goal.check(grids[:k + 1], actions[:k])]


def first_satisfied(goal: NLGoal, grids: list[Grid], actions: list[str]) -> int | None:
    for k in range(1, len(grids)):
        if goal.check(grids[:k + 1], actions[:k]):
            return k
    return None


def rollout(goal: NLGoal, actions: list[str]) -> tuple[list[Grid], list[str]]:
    """Replay `actions` from a bare reset and return the frames.  Imported lazily: loading
    the interpreter costs ~1 s and callers that only want the sentences should not pay it."""
    from offline_learning.curated_plan import trace
    return [s.grid for s in trace(goal.game, goal.seed, actions)], list(actions)


def check_actions(goal: NLGoal, actions: list[str]) -> int | None:
    grids, acts = rollout(goal, actions)
    return first_satisfied(goal, grids, acts)


def load_curated(path: str | Path) -> dict[str, dict]:
    rows = json.loads(Path(path).read_text())
    return {r["id"]: r for r in rows if r["id"] in BY_PID}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", help="game/pid to print in full")
    a = ap.parse_args()
    if a.show:
        pid = a.show.split("/")[-1]
        g = BY_PID[pid]
        print(f"{g.game}/{g.pid}  {g.tier}  seed={g.seed}  hold={g.hold}")
        print(f'  "{g.nl}"')
        print(f"  {g.check.__doc__}")
        print(f"  note: {g.note}")
        return
    print(f"{'game':7s} {'pid':12s} {'tier':5s} hold  goal")
    for g in GOALS:
        print(f"{g.game:7s} {g.pid:12s} {g.tier:5s} {g.hold:<5d} {g.nl}")


if __name__ == "__main__":
    main()
