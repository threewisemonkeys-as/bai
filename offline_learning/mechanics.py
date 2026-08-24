"""Core-dynamics taxonomy + behavioral detectors for the 5 human-origin games.

Each Autumn program (`Autumn.cpp/tests/<name>.sexp`) decomposes into a small set of
RULES: the `(on <event> ...)` input handlers plus the passive `next`-update rules that
fire every clock tick. Those rules ARE the "core dynamics". At OUTCOME-LEVEL granularity
a single rule can split into several mechanics when it produces observably distinct
results (a cloud that MOVES vs one that CLAMPS at the wall; a `down` that rains a LIQUID
vs a SOLID droplet; a `click` that only toggles day/night vs one that also flips existing
droplets). This module pins that taxonomy as data (`MECHANICS`) and classifies a single
replayed transition into exactly one bucket (`classify`).

A detector sees the three grids the pipeline already produces per step:
    g0  = grid BEFORE the action              (x_t)
    g1  = grid AFTER the human's action       (x_{t+1})
    cf  = grid after a single `noop` from g0  (passive-only counterfactual; may be None)
`cf` is what `human_replay.noop_counterfactual` computes, and it is what separates what
the ACTION did (g1 vs cf) from what the CLOCK did (cf vs g0). Grids are the JSON
`[[...]]` colour matrices `_grid()` extracts; here they are parsed with json into
`list[list[str]]` indexed [row][col].

Caveat for readers of the scores: this feeds an INVERSE-dynamics exam. Action-triggered
mechanics (rain-day/night, click-toggle, click-swap, coin-collect, shoot, jump,
cloud-move/clamp) are directly discriminable -- the action label differs. Purely PASSIVE
outcome splits (liquid-slide vs liquid-fall, contagion-spread, enemy-patrol,
particle-diffuse) all carry the same action label (`noop`), so the ID exam tests them
only as "recognise this was a passive step, not an input"; telling the sub-outcomes apart
is a FORWARD question. The buckets are kept fine anyway so the test set demonstrably
CONTAINS a transition exercising each core aspect (and so a later forward pass can reuse
them).
"""
from __future__ import annotations

import json

from offline_learning.program_meta import DerivedTable, background as _program_background

# per-game background colour(s) that are not objects -- derived from the program's
# `(= background ...)` declaration (default black) so any installed game resolves and a
# non-black world (n2ntd white, balloon skyblue) is never mistaken for one made of objects
_BG = DerivedTable(lambda game: {_program_background(game)})

# canonical bucket list per game. kind: "action" (input handler) | "passive" (clock rule)
# | "null" (nothing observably happened -- kept so a static noop still has a home).
# `synth_hint` documents how to drive the engine to the bucket if human play never does.
MECHANICS: dict[str, list[dict]] = {
    "bt3gb": [
        {"id": "cloud-move-left", "kind": "action", "desc": "left moves the cloud one cell left (in bounds)"},
        {"id": "cloud-move-right", "kind": "action", "desc": "right moves the cloud one cell right (in bounds)"},
        {"id": "cloud-clamp", "kind": "action", "desc": "left/right at the wall: cloud stays put (bounds clamp)",
         "synth_hint": "spam left (or right) until the cloud is at the edge, then press it again"},
        {"id": "rain-day", "kind": "action", "desc": "down rains a LIQUID (blue) droplet while day=true"},
        {"id": "rain-night", "kind": "action", "desc": "down rains a SOLID (lightblue) droplet while day=false",
         "synth_hint": "click once (-> night) then press down"},
        {"id": "click-toggle", "kind": "action", "desc": "click flips celestial gold<->gray (no droplets present)"},
        {"id": "click-flip-droplets", "kind": "action", "desc": "click also flips every existing droplet blue<->lightblue",
         "synth_hint": "press down to spawn a droplet, then click"},
        {"id": "liquid-fall", "kind": "passive", "desc": "a liquid (blue) droplet falls straight down one cell"},
        {"id": "liquid-slide", "kind": "passive", "desc": "a blocked liquid droplet slides sideways toward a hole",
         "synth_hint": "rain several day droplets into one column so later ones must spread"},
        {"id": "solid-fall", "kind": "passive", "desc": "a solid (lightblue) droplet falls straight down / stacks",
         "synth_hint": "click to night, press down, then noop to watch it fall"},
        {"id": "static-noop", "kind": "null", "desc": "up/noop with no visible change"},
    ],
    "dq8gc": [
        {"id": "move-left", "kind": "action", "desc": "left moves the active particle col-1"},
        {"id": "move-right", "kind": "action", "desc": "right moves the active particle col+1"},
        {"id": "move-up", "kind": "action", "desc": "up moves the active particle row-1"},
        {"id": "move-down", "kind": "action", "desc": "down moves the active particle row+1"},
        {"id": "move-overlap", "kind": "action", "desc": "a move lands the active onto an occupied cell (they overlap)",
         "synth_hint": "walk the active particle onto an inactive particle's cell"},
        {"id": "click", "kind": "action", "desc": "click on a particle SWAPS control of it (or on empty does nothing) -- either way the NEXT FRAME IS UNCHANGED (positions/colours identical), so a disease click is aliased with noop at t+1; only future control differs"},
        {"id": "contagion-spread", "kind": "passive", "desc": "a particle adjacent to an infected one turns darkgreen (covers both the inactive-contagion and active-infection rules -- which cell is 'active' is not observable from the grid, so they share one outcome bucket)"},
        {"id": "static-noop", "kind": "null", "desc": "noop with no visible change"},
    ],
    "n2ntd": [
        {"id": "move-left", "kind": "action", "desc": "left moves mario left"},
        {"id": "move-right", "kind": "action", "desc": "right moves mario right"},
        {"id": "jump", "kind": "action", "desc": "up jumps when grounded (mario rises)"},
        {"id": "jump-blocked", "kind": "action", "desc": "up while airborne: no jump",
         "synth_hint": "press up twice in a row (2nd is mid-air)"},
        {"id": "coin-collect", "kind": "action", "desc": "landing on a coin removes it and grants a bullet"},
        {"id": "shoot", "kind": "action", "desc": "click fires a bullet when mario has ammo",
         "synth_hint": "collect a coin (get a bullet), then click"},
        {"id": "shoot-no-ammo", "kind": "action", "desc": "click with no ammo: nothing fires"},
        {"id": "gravity-fall", "kind": "passive", "desc": "mario falls one cell when unsupported"},
        {"id": "enemy-patrol", "kind": "passive", "desc": "the enemy strides left/right each tick"},
        {"id": "enemy-bounce", "kind": "passive", "desc": "the enemy reverses direction at a wall (x=1 / x=14)",
         "synth_hint": "noop until the enemy reaches a side wall"},
        {"id": "bullet-move", "kind": "passive", "desc": "a fired bullet travels upward / despawns on a step",
         "synth_hint": "shoot, then noop"},
        {"id": "enemy-hit", "kind": "passive", "desc": "a bullet reaching the enemy kills it",
         "synth_hint": "line mario under the enemy, shoot, noop until impact"},
        {"id": "static-noop", "kind": "null", "desc": "noop with no visible change"},
    ],
    "s2kt7": [
        {"id": "click-spawn-food", "kind": "action", "desc": "click spawns 2 food at random positions"},
        {"id": "food-eaten", "kind": "passive", "desc": "an ant reaching food removes it"},
        {"id": "ant-move", "kind": "passive", "desc": "ants step toward the closest food"},
        {"id": "static-noop", "kind": "null", "desc": "noop with no visible change (no food on board)"},
    ],
    "83wkq": [
        {"id": "click-spawn-particle", "kind": "action", "desc": "click spawns a particle at the clicked cell"},
        {"id": "particle-diffuse", "kind": "passive", "desc": "every particle random-walks to an adjacent cell"},
        {"id": "static-noop", "kind": "null", "desc": "noop with no particles on board"},
    ],
}


def mechanic_ids(game: str) -> list[str]:
    return [m["id"] for m in MECHANICS[game]]


# ------------------------------------------------------------------ grid helpers
def _p(g: str) -> list[list[str]]:
    return json.loads(g)


def _cells(P: list[list[str]], bg: set[str]) -> dict[tuple[int, int], str]:
    return {(r, c): v for r, row in enumerate(P) for c, v in enumerate(row) if v not in bg}


def _count(C: dict, *colors: str) -> int:
    return sum(1 for v in C.values() if v in colors)


def _pos(C: dict, *colors: str) -> list[tuple[int, int]]:
    return sorted(p for p, v in C.items() if v in colors)


# ----------------------------------------------------------------- per game rules
def _ice(g0, a, cf, g1, bg):
    verb = a.split()[0]
    P0, P1 = _p(g0), _p(g1)
    C0, C1 = _cells(P0, bg), _cells(P1, bg)
    drops0 = {p: v for p, v in C0.items() if v in ("blue", "lightblue")}
    drops1 = {p: v for p, v in C1.items() if v in ("blue", "lightblue")}

    if verb in ("left", "right"):
        if cf is not None and g1 == cf:          # action changed nothing -> clamped
            return "cloud-clamp"
        return f"cloud-move-{verb}"
    if verb == "down":
        base = _cells(_p(cf), bg) if cf is not None else C0
        old = {p for p, v in base.items() if v in ("blue", "lightblue")}
        new = {p: v for p, v in drops1.items() if p not in old}
        if any(v == "blue" for v in new.values()):
            return "rain-day"
        if any(v == "lightblue" for v in new.values()):
            return "rain-night"
        return None
    if verb == "click":
        # (1,0)/(1,1) are celestial-only cells (the cloud sits on row 0), so they read
        # the gold<->gray toggle without cloud occlusion.
        toggled = (P0[1][0], P0[1][1]) != (P1[1][0], P1[1][1])
        flipped = drops0 and any(drops0.get(p) != drops1.get(p) for p in set(drops0) | set(drops1))
        if flipped:
            return "click-flip-droplets"
        if toggled:
            return "click-toggle"
        return None
    # up / noop -> passive water
    if drops0 == drops1:
        return "static-noop"
    used, tags = set(), []
    for p1, v1 in sorted(drops1.items()):
        cand = [p0 for p0 in drops0 if p0 not in used
                and 0 <= p1[0] - p0[0] <= 1 and abs(p1[1] - p0[1]) <= 1]
        if not cand:
            continue
        p0 = min(cand, key=lambda q: (p1[0] - q[0], abs(p1[1] - q[1])))
        used.add(p0)
        v0 = drops0[p0]
        dr, dc = p1[0] - p0[0], p1[1] - p0[1]
        if dc != 0:
            tags.append("liquid-slide")
        elif dr == 1:
            tags.append("liquid-fall" if v0 == "blue" else "solid-fall")
    for t in ("liquid-slide", "solid-fall", "liquid-fall"):
        if t in tags:
            return t
    return "static-noop"


def _disease(g0, a, cf, g1, bg):
    verb = a.split()[0]
    C0, C1 = _cells(_p(g0), bg), _cells(_p(g1), bg)
    if verb in ("left", "right", "up", "down"):
        if cf is not None and g1 == cf:
            return None                          # move with no visible effect (off-grid)
        if len(C1) < len(C0):
            return "move-overlap"
        return f"move-{verb}"
    if verb == "click":
        return "click"          # swap or empty -- both leave the next frame unchanged
    # noop -> passive
    if _count(C1, "darkgreen") > _count(C0, "darkgreen"):
        return "contagion-spread"
    return "static-noop"


def _mario(g0, a, cf, g1, bg):
    verb = a.split()[0]
    C0, C1 = _cells(_p(g0), bg), _cells(_p(g1), bg)
    red0, red1 = _pos(C0, "red"), _pos(C1, "red")
    if _count(C1, "gold") < _count(C0, "gold"):
        return "coin-collect"                    # coin despawn dominates the frame
    if verb == "click":
        return "shoot" if _count(C1, "mediumpurple") > _count(C0, "mediumpurple") else "shoot-no-ammo"
    if verb == "left":
        return "move-left"
    if verb == "right":
        return "move-right"
    if verb == "up":
        rose = red0 and red1 and min(r for r, _ in red1) < min(r for r, _ in red0)
        return "jump" if rose else "jump-blocked"
    # noop -> passive: prefer the salient change. a kill removes the enemy ENTIRELY
    # (enemyLives 1->0 -> removeObj), so require blue to vanish -- the enemy never clips
    # at a wall (it bounces with all cells in bounds), so a partial drop is not a kill.
    if _count(C0, "blue") > 0 and _count(C1, "blue") == 0:
        return "enemy-hit"
    if _count(C1, "mediumpurple") != _count(C0, "mediumpurple") or _pos(C1, "mediumpurple") != _pos(C0, "mediumpurple"):
        return "bullet-move"
    if red0 and red1 and min(r for r, _ in red1) > min(r for r, _ in red0):
        return "gravity-fall"
    blue0, blue1 = _pos(C0, "blue"), _pos(C1, "blue")
    if blue0 != blue1:
        cols0 = {c for _, c in blue0}
        # a bounce shows up as the enemy touching an extreme column
        if 1 in cols0 or 14 in cols0 or 0 in cols0 or 15 in cols0:
            return "enemy-bounce"
        return "enemy-patrol"
    return "static-noop"


def _ants(g0, a, cf, g1, bg):
    verb = a.split()[0]
    C0, C1 = _cells(_p(g0), bg), _cells(_p(g1), bg)
    food0, food1 = _pos(C0, "red"), _pos(C1, "red")
    if verb == "click":
        return "click-spawn-food" if len(food1) > len(food0) else "static-noop"
    if len(food1) < len(food0):
        return "food-eaten"
    if _pos(C0, "gray") != _pos(C1, "gray"):
        return "ant-move"
    return "static-noop"


def _particles(g0, a, cf, g1, bg):
    verb = a.split()[0]
    C0, C1 = _cells(_p(g0), bg), _cells(_p(g1), bg)
    p0, p1 = _pos(C0, "blue"), _pos(C1, "blue")
    if verb == "click":
        return "click-spawn-particle" if len(p1) > len(p0) else "static-noop"
    if set(p0) != set(p1):
        return "particle-diffuse"
    return "static-noop"


_DETECT = {"bt3gb": _ice, "dq8gc": _disease, "n2ntd": _mario,
           "s2kt7": _ants, "83wkq": _particles}


def classify(game: str, g0: str, action: str, cf: str | None, g1: str) -> str | None:
    """Bucket one transition. Returns a mechanic id, or None if unattributable."""
    if g0 is None or g1 is None:
        return None
    return _DETECT[game](g0, action, cf, g1, _BG[game])
