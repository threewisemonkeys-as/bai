"""Scripted action sequences that drive the Autumn engine into each mechanic bucket.

For a mechanic that human play never reaches, `coverage_exam.build` replays each
candidate `(seed, actions)` here, runs the SAME `mechanics.classify` detector over every
step, and keeps the first step whose label equals the target bucket. So a synthesized
transition is confirmed to exercise the bucket it fills by the exact detector used on the
human data -- no hand-asserted labels. Buckets no candidate reaches are reported as
unreachable rather than faked.

Clicks are in the env-facing `click ROW COL` form (`human_replay.replay` consumes that
directly). Sequences are best-effort and intentionally over-provided; the confirm step
filters them.
"""
from __future__ import annotations


def _clk(row: int, col: int) -> str:
    return f"click {row} {col}"


# mario: reach column 1, jump, fall onto the row-10 step -> land on coin (9,1) = +1 bullet
_COIN = ["left"] * 5 + ["up", "noop", "noop", "noop"]


# game -> mechanic -> list[(seed, actions)]
_SYNTH: dict[str, dict[str, list]] = {
    "bt3gb": {
        "cloud-move-left": [(1, ["left"])],
        "cloud-move-right": [(1, ["right"])],
        "cloud-clamp": [(1, ["left"] * 7), (1, ["right"] * 13)],
        "rain-day": [(1, ["down"])],
        "rain-night": [(1, [_clk(0, 0), "down"])],
        "click-toggle": [(1, [_clk(0, 0)])],
        "click-flip-droplets": [(1, ["down", "noop", "noop", _clk(0, 0)])],
        "liquid-fall": [(1, ["down", "noop"])],
        "liquid-slide": [(1, ["down", "noop"] * 4 + ["noop"] * 16)],
        "solid-fall": [(1, [_clk(0, 0), "down"] + ["noop"] * 5)],
    },
    "dq8gc": {
        "move-left": [(1, ["left"])],
        "move-right": [(1, ["right"])],
        "move-up": [(1, ["up"])],
        "move-down": [(1, ["down"])],
        # walk the active particle (col2,row2) onto an inactive (col4,row3 / col3,row5)
        "move-overlap": [(1, ["right", "right", "down"]),
                         (1, ["right", "down", "down", "down"]),
                         (1, ["down", "down", "down", "right"])],
        # disease click leaves the next frame unchanged (swap is invisible at t+1);
        # populated from human play, kept here only as a fallback
        "click": [(1, [_clk(3, 4)]), (1, [_clk(0, 0)])],
        # move adjacent to an inactive, then noop lets the infection step across
        "contagion-spread": [(1, ["right", "down", "noop"]),
                             (1, ["right", "right", "down", "noop"]),
                             (1, ["noop"] * 4)],
    },
    "n2ntd": {
        "move-left": [(1, ["left"])],
        "move-right": [(1, ["right"])],
        "jump": [(1, ["up"])],
        "jump-blocked": [(1, ["up", "up"])],
        "shoot-no-ammo": [(1, [_clk(0, 0)])],
        "gravity-fall": [(1, ["up", "noop"]), (1, ["up", "up", "noop"])],
        "enemy-patrol": [(1, ["noop"])],
        "enemy-bounce": [(1, ["noop"] * 8), (1, ["noop"] * 16)],
        # go to column 1, jump, and FALL onto the row-10 step -- mario lands at (9,1),
        # which is a coin, so `on intersects mario coins` collects it (verified)
        "coin-collect": [(1, ["left"] * 5 + ["up", "noop", "noop", "noop"])],
        # a bullet exists only after a coin; collect one, then click fires it up column 1
        "shoot": [(1, _COIN + [_clk(0, 0)])],
        "bullet-move": [(1, _COIN + [_clk(0, 0), "noop"])],
        # enemy-hit: after ammo, delay w ticks so the bullet reaches the top as the enemy
        # patrols over column 1, then the ascending bullet removes it (blue vanishes)
        "enemy-hit": [(1, _COIN + ["noop"] * w + [_clk(0, 0)] + ["noop"] * 14)
                      for w in range(0, 8)],
    },
    "s2kt7": {
        "click-spawn-food": [(1, [_clk(8, 8)])],
        "ant-move": [(1, [_clk(5, 6), "noop"]), (1, [_clk(8, 8), "noop", "noop"])],
        "food-eaten": [(1, [_clk(5, 6)] + ["noop"] * 30),
                       (1, [_clk(5, 5), _clk(1, 14)] + ["noop"] * 30)],
        "static-noop": [(1, ["noop"])],
    },
    "83wkq": {
        "click-spawn-particle": [(1, [_clk(8, 8)])],
        "particle-diffuse": [(1, [_clk(8, 8), "noop"]), (1, [_clk(8, 8), "noop", "noop"])],
        "static-noop": [(1, ["noop"])],
    },
}


def synth_candidates(game: str, mechanic: str) -> list:
    """Candidate (seed, actions) sequences to try for a bucket. Empty if none scripted."""
    return _SYNTH.get(game, {}).get(mechanic, [])
