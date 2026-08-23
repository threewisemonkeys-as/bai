"""Rule-faithful, MULTI-LABEL mechanic detection for the 5 human-origin Autumn games.

Why a second module (vs `mechanics.py`): `mechanics.py` returns ONE bucket per
transition, chosen by a fixed priority chain, and infers which rule fired from colour
COUNTS. That is wrong in two ways that block compositional problem construction:

  * a transition where several rules fire (click + contagion, jump + gravity) is
    reported as one of them, so composition is invisible to any selector; and
  * count-based inference misattributes: a dq8gc move that walks the active particle
    OFF-GRID loses a cell and is reported as `move-overlap`; an 83wkq click onto an
    occupied cell adds no cell and is reported as `static-noop`.

This module decomposes each transition with the noop counterfactual instead:

    g0  --clock only-->  cf        PASSIVE mechanics  = what the clock did   (g0 -> cf)
    g0  --clock+input--> g1        ACTION  mechanics  = what the input did   (cf vs g1)

and reports which RULES fired (faithful to the .sexp handlers), separately from whether
their effect was OBSERVABLE. `(on clicked ...)` fires on every click whether or not the
frame changes, so `click-toggle` is emitted for every bt3gb click and the aliasing is
recorded as `visible=False` rather than silently relabelled. That single test --
`visible = (g1 != cf)` -- subsumes the per-game special cases the old detector got wrong
(cloud-clamp, rain-onto-occupied, move-blocked, spawn-onto-occupied, shoot-no-ammo).

    fired(game, g0, action, cf, g1) -> Fired(action=[...], passive=[...], visible=bool)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

# ---------------------------------------------------------------- per-game constants
BG = {"bt3gb": {"black"}, "dq8gc": {"black"}, "n2ntd": {"white"},
      "s2kt7": {"black"}, "83wkq": {"black"}}
SIZE = {"bt3gb": 16, "dq8gc": 16, "n2ntd": 12, "s2kt7": 16, "83wkq": 16}

# Object classes per game: colour -> class. Used for GOAL MASKING (compare only the
# classes a tested mechanic touches) and for background identification.
CLASSES = {
    "bt3gb": {"gold": "celestial", "gray": "cloud_or_celestial",
              "blue": "water", "lightblue": "water"},
    "dq8gc": {"gray": "particle", "darkgreen": "particle"},
    "n2ntd": {"red": "mario", "darkorange": "step", "gold": "coin",
              "blue": "enemy", "mediumpurple": "bullet"},
    "s2kt7": {"gray": "ant", "red": "food"},
    "83wkq": {"blue": "particle"},
}

# Classes that evolve AUTONOMOUSLY -- on a fixed cycle the agent cannot influence. Their
# position in a goal frame encodes elapsed ticks, so the goal doubles as a clock stamp and
# noop padding becomes un-compressible. Masked out unless a mechanic under test declares
# them (shoot/enemy-hit legitimately need the enemy).
#   n2ntd enemy  patrols on a fixed cycle until shot                   -> background
#   s2kt7 ants   walk toward the CLOSEST FOOD, which the agent spawns  -> NOT background
#                (their position is a consequence of the click history, not a clock)
#   83wkq        particles random-walk and are the only object class; nothing is left to
#                compare if masked, so 83wkq cannot support long-horizon planning at all
BACKGROUND = {"bt3gb": set(), "dq8gc": set(), "n2ntd": {"enemy"},
              "s2kt7": set(), "83wkq": set()}


@dataclass
class Fired:
    action: list[str] = field(default_factory=list)
    passive: list[str] = field(default_factory=list)
    visible: bool = True          # did the INPUT change the frame (g1 != cf)?

    @property
    def all(self) -> list[str]:
        return self.action + self.passive

    def consequential(self) -> list[str]:
        return [m for m in self.all if not m.endswith("-idle") and m != "static-noop"]


# --------------------------------------------------------------------- grid helpers
def P(g: str) -> list[list[str]]:
    return json.loads(g)


def cells(g: str, bg: set[str]) -> dict[tuple[int, int], str]:
    return {(r, c): v for r, row in enumerate(P(g)) for c, v in enumerate(row) if v not in bg}


def pos(g: str, *colors: str) -> list[tuple[int, int]]:
    return sorted((r, c) for r, row in enumerate(P(g)) for c, v in enumerate(row) if v in colors)


def _verb(a: str) -> str:
    return (a or "noop").split()[0]


def _clicked(a: str) -> bool:
    """A bare `click` (no coordinates) never reaches an `(on clicked ...)` handler --
    verified against the noop counterfactual in all 5 programs. Only a parameterised
    `click <col> <row>` fires the rule."""
    return _verb(a) == "click" and _click_xy(a) is not None


def _click_xy(a: str) -> tuple[int, int] | None:
    """`click A B` targets RENDER cell (row=A, col=B). Verified directly against the
    engine on 83wkq, whose click spawns a particle at the clicked cell: `click 2 9`
    spawns at (row 2, col 9) and `click 9 2` at (row 9, col 2); cross-checked on dq8gc,
    where only `click <row> <col>` transfers control. Returns (row, col)."""
    parts = (a or "").split()
    if len(parts) < 3:
        return None
    try:
        return int(parts[1]), int(parts[2])
    except ValueError:
        return None


# =============================================================== bt3gb / ice
_DROP = ("blue", "lightblue")


def _day(g: str) -> bool | None:
    """Read day/night off the celestial 2x2 (rows 0-1, cols 0-1). `gold` only ever comes
    from the celestial body, so any gold there means day. Row 1 can never hold the cloud
    (cloud is row 0), so a gray there means night. Water can still occlude both."""
    p = P(g)
    if any(p[r][c] == "gold" for r in (0, 1) for c in (0, 1)):
        return True
    if p[1][0] == "gray" or p[1][1] == "gray":
        return False
    return None


def _act_bt3gb(g0, a, cf, g1, bg):
    v, out = _verb(a), []
    if v in ("left", "right"):
        # the cloud rule always fires; `nextCloud` keeps it put when the move leaves bounds
        out.append(f"cloud-move-{v}" if (cf is not None and g1 != cf) else "cloud-clamp")
    elif v == "down":
        d0, d1 = set(pos(cf, *_DROP)) if cf else set(), set(pos(g1, *_DROP))
        gained = [p for p in d1 - d0]
        phase = None
        if gained:
            pg = P(g1)
            phase = "day" if any(pg[r][c] == "blue" for r, c in gained) else "night"
        else:                                    # spawn cell already occupied -> occluded
            day = _day(g0)
            phase = "day" if day is not False else "night"
        out.append(f"rain-{phase}")
    elif _clicked(a):
        out.append("click-toggle")               # (on clicked ...) always flips day
        if pos(g0, *_DROP):
            out.append("click-flip-droplets")    # ... and always flips existing droplets
    return out


def _pas_bt3gb(g0, cf, bg):
    """nextWater per drop: liquid falls, else slides diagonally, else rests; solid falls
    or rests. Matched nearest-above so a drop is tracked across the step."""
    if cf is None:
        return []
    d0 = {p: v for p, v in cells(g0, bg).items() if v in _DROP}
    d1 = {p: v for p, v in cells(cf, bg).items() if v in _DROP}
    if not d0:
        return []
    used, out = set(), []
    for p1 in sorted(d1):
        cand = [p0 for p0 in d0 if p0 not in used
                and 0 <= p1[0] - p0[0] <= 1 and abs(p1[1] - p0[1]) <= 1]
        if not cand:
            continue
        p0 = min(cand, key=lambda q: (p1[0] - q[0], abs(p1[1] - q[1])))
        used.add(p0)
        liquid = d0[p0] == "blue"
        dr, dc = p1[0] - p0[0], p1[1] - p0[1]
        if dc != 0:
            out.append("liquid-slide")
        elif dr == 1:
            out.append("liquid-fall" if liquid else "solid-fall")
        else:
            out.append("liquid-rest" if liquid else "solid-rest")
    return out


# =============================================================== dq8gc / disease
_DELTA = {"left": (0, -1), "right": (0, 1), "up": (-1, 0), "down": (1, 0)}


def _act_dq8gc(g0, a, cf, g1, bg):
    v = _verb(a)
    if v == "click":
        if not _clicked(a):
            return []
        xy = _click_xy(a)
        hit = P(g0)[xy[0]][xy[1]] not in bg
        # both branches leave the frame identical; only FUTURE control differs
        return ["click-select" if hit else "click-empty"]
    if v not in _DELTA or cf is None:
        return []
    dr, dc = _DELTA[v]
    c0, c1 = set(cells(cf, bg)), set(cells(g1, bg))
    lost, gained = sorted(c0 - c1), sorted(c1 - c0)
    n = SIZE["dq8gc"]
    if len(lost) == 1 and len(gained) == 1:
        return [f"move-{v}"]
    if len(lost) == 1 and not gained:
        r, c = lost[0]
        tr, tc = r + dr, c + dc
        if not (0 <= tr < n and 0 <= tc < n):
            return ["move-offgrid"]              # bare `move`, no bounds check in the sexp
        return ["move-overlap"] if (tr, tc) in c0 else [f"move-{v}"]
    if len(gained) == 1 and not lost:
        return ["move-return"]                   # an off-grid active walks back into view
    return [f"move-{v}"] if lost or gained else ["move-invisible"]


def _pas_dq8gc(g0, cf, bg):
    if cf is None:
        return []
    n0 = len(pos(g0, "darkgreen"))
    n1 = len(pos(cf, "darkgreen"))
    return ["contagion-spread"] if n1 > n0 else []


# =============================================================== n2ntd / mario
def _act_n2ntd(g0, a, cf, g1, bg):
    v = _verb(a)
    if cf is None:
        return []
    if v in ("left", "right"):
        return [f"move-{v}" if pos(g1, "red") != pos(cf, "red") else "move-blocked"]
    if v == "up":
        r1, rc = pos(g1, "red"), pos(cf, "red")
        rose = r1 and rc and min(r for r, _ in r1) < min(r for r, _ in rc)
        return ["jump" if rose else "jump-blocked"]
    if v == "click":
        if not _clicked(a):
            return []
        return ["shoot" if len(pos(g1, "mediumpurple")) > len(pos(cf, "mediumpurple"))
                else "shoot-no-ammo"]
    return []


def _pas_n2ntd(g0, cf, bg):
    if cf is None:
        return []
    out = []
    if len(pos(cf, "gold")) < len(pos(g0, "gold")):
        out.append("coin-collect")               # (on (intersects mario coins)) -- COLLISION
    b0, b1 = pos(g0, "blue"), pos(cf, "blue")
    if b0 and not b1:
        out.append("enemy-hit")
    elif b0 and b1 and b0 != b1:
        # bounce rules fire at origin.x==1 (cols 0-2) and origin.x==10 (cols 9-11);
        # only the REVERSAL frame is a bounce, not every frame spent near the wall.
        c0 = {c for _, c in b0}
        dx = min(c for _, c in b1) - min(c for _, c in b0)
        left_wall, right_wall = min(c0) == 0, max(c0) == SIZE["n2ntd"] - 1
        out.append("enemy-bounce" if (left_wall and dx > 0) or (right_wall and dx < 0)
                   else "enemy-patrol")
    p0, p1 = pos(g0, "mediumpurple"), pos(cf, "mediumpurple")
    if p0 and len(p1) < len(p0):
        out.append("bullet-despawn")             # bullet reaching a step is removed
    elif p0 != p1 and (p0 or p1):
        out.append("bullet-move")
    r0, r1 = pos(g0, "red"), pos(cf, "red")
    if r0 and r1 and min(r for r, _ in r1) > min(r for r, _ in r0):
        out.append("gravity-fall")
    return out


# =============================================================== s2kt7 / ants
def _act_s2kt7(g0, a, cf, g1, bg):
    return ["click-spawn-food"] if _clicked(a) else []


def _pas_s2kt7(g0, cf, bg):
    if cf is None:
        return []
    out = []
    if len(pos(cf, "red")) < len(pos(g0, "red")):
        out.append("food-eaten")
    out.append("ant-move" if pos(g0, "gray") != pos(cf, "gray") else "ant-idle")
    return out


# =============================================================== 83wkq / particles
def _act_83wkq(g0, a, cf, g1, bg):
    return ["click-spawn-particle"] if _clicked(a) else []


def _pas_83wkq(g0, cf, bg):
    if cf is None:
        return []
    p0, p1 = pos(g0, "blue"), pos(cf, "blue")
    if not p0:
        return []
    return ["particle-diffuse"] if p0 != p1 else ["particle-idle"]


_ACT = {"bt3gb": _act_bt3gb, "dq8gc": _act_dq8gc, "n2ntd": _act_n2ntd,
        "s2kt7": _act_s2kt7, "83wkq": _act_83wkq}
_PAS = {"bt3gb": _pas_bt3gb, "dq8gc": _pas_dq8gc, "n2ntd": _pas_n2ntd,
        "s2kt7": _pas_s2kt7, "83wkq": _pas_83wkq}


def fired(game: str, g0: str, action: str, cf: str | None, g1: str) -> Fired:
    """Which RULES fired across one transition, split by trigger.

    `cf` is the noop counterfactual from g0 (== g1 for a noop step). Passive labels are
    read off g0 -> cf, i.e. what the clock would do alone; on the rare tick where an
    input SUPPRESSES a passive update (a bt3gb click freezes the water update for that
    tick) the passive label reports the clock's intent, not the realised frame. Chain
    construction verifies necessity by engine ablation, so a label of that kind cannot
    silently survive into a shipped problem."""
    if g0 is None or g1 is None:
        return Fired(visible=False)
    bg = BG[game]
    if _verb(action) == "noop":
        cf = g1
    return Fired(action=_ACT[game](g0, action, cf, g1, bg),
                 passive=_PAS[game](g0, cf, bg),
                 visible=(cf is not None and g1 != cf))
