"""Authored v2 planning ladders for the 11 games added after the original four."""
from __future__ import annotations

import itertools
from typing import Any

from offline_learning.human_replay import GAMES
from offline_learning.planning_v2 import raw_trace
from offline_learning.planning_nl_goals import get_python_goal

SEED = 101
MULTI_SEEDS = (101, 211, 307)


def exact(game: str, pid: str, tier: str, objective: str, plan: list[str],
          mechanics: list[str], *, prefix: list[str] | None = None,
          seed: int = SEED, note: str = "",
          must_keep: list[str] | None = None) -> dict[str, Any]:
    checker_id = f"{game}/{pid}"
    checker = get_python_goal(checker_id)
    if checker.nl != objective:
        raise ValueError(f"{checker_id}: objective disagrees with Python goal registry")
    return {"game": game, "id": pid, "tier": tier, "objective": objective,
            "nl_goal": objective, "seed": seed, "prefix": prefix or [], "plan": plan,
            "frame_success_mode": "final",
            "nl_checker": checker_id, "must_keep": must_keep or [],
            "mechanics": mechanics, "note": note}


def semantic(game: str, pid: str, template: str, tier: str, objective: str,
             seed: int, prefix: list[str], plan: list[str],
             mechanics: list[str], *, note: str = "") -> dict[str, Any]:
    checker_id = f"{game}/{template}"
    checker = get_python_goal(checker_id)
    if checker.nl != objective:
        raise ValueError(f"{checker_id}: objective disagrees with Python goal registry")
    return {"game": game, "id": pid, "template_id": template, "tier": tier,
            "objective": objective, "nl_goal": objective, "seed": seed,
            "prefix": prefix, "plan": plan, "frame_success_mode": "final",
            "nl_checker": checker_id,
            "mechanics": mechanics,
            "note": note}


def paint_specs() -> list[dict[str, Any]]:
    return [
        exact("eahcw", "red-mark", "L1", "Paint a red mark at row 4, column 4.",
              ["click 4 4"], ["paint-default-red"]),
        exact("eahcw", "gold-blue-pair", "L2",
              "Paint a gold mark at (3,3) and a blue mark at (4,4).",
              ["up", "click 3 3", "right", "click 4 4"],
              ["select-gold", "paint", "select-blue"]),
        exact("eahcw", "three-colour-diagonal", "L3",
              "Paint purple, green, and gold marks along the diagonal.",
              ["down", "click 3 3", "left", "click 4 4", "up", "click 5 5"],
              ["select-colour", "paint", "switch-colour"]),
        semantic(
            "eahcw", "five-colour-palette", "five-colour-palette", "L4",
            "Create five distinct marks in this order: red, gold, purple, green, blue.",
            SEED, [],
            ["click 2 2", "up", "click 3 3", "down", "down", "click 4 4",
             "left", "click 5 5", "right", "right", "click 6 6"],
            ["paint-default-red", "select-colour", "cancel-opposites", "paint"],
            note="Python-primary (review-check F2): the ordering is part of the goal, "
                 "and an exact final frame cannot see creation order."),
    ]


def egg_specs() -> list[dict[str, Any]]:
    return [
        exact("egg", "carry-left", "L1", "Move the intact egg three cells left.",
              ["left"] * 3, ["carry-egg"]),
        exact("egg", "controlled-gravity-drop", "L2",
              "From the raised position, turn gravity on, land the egg, then turn gravity off.",
              ["click 0 0"] + ["noop"] * 4 + ["click 0 0", "noop"],
              ["toggle-gravity", "gravity-fall"], prefix=["up", "up"]),
        exact("egg", "shatter", "L3", "Raise the egg above the safe height and shatter it.",
              ["up"] * 4 + ["click 0 0"],
              ["carry-egg", "height-latch", "shatter"]),
        semantic(
            "egg", "boundary-drop", "boundary-drop", "L3",
            "Raise the egg to just below the shatter height, then turn gravity on "
            "so it falls and lands intact.",
            SEED, [], ["up", "up", "up", "click 0 0"] + ["noop"] * 6,
            ["carry-egg", "height-latch", "toggle-gravity", "gravity-fall"],
            note="Engine boundary: height is latched at the button click; origin y=10 "
                 "(egg top row 8) is the last safe raise, y=9 shatters."),
        exact("egg", "left-fragment-bed", "L4",
              "Carry the egg left, shatter it, and let the fragments settle along the floor.",
              ["left"] * 3 + ["up"] * 4 + ["click 0 0"] + ["noop"] * 30,
              ["carry-egg", "shatter", "fragment-flow"]),
    ]


def grow_specs() -> list[dict[str, Any]]:
    return [
        exact("7xf97", "move-cloud", "L1", "Move the cloud three cells left.",
              ["left"] * 3, ["move-cloud"]),
        exact("7xf97", "move-sun", "L1", "Move the sun one cell to the right.",
              ["click 1 1"], ["click-move-sun"]),
        exact("7xf97", "water-left-plant", "L3",
              "Move the cloud over the plant at column 11 and grow it by one cell.",
              ["left", "left", "down"] + ["noop"] * 18,
              ["move-cloud", "drop-water", "water-fall", "grow-leaf"]),
        semantic(
            "7xf97", "shower-drain", "shower-drain", "L3",
            "Shower a plant with water and let the water drain completely without "
            "causing any growth.",
            SEED, [], ["left"] * 10 + ["down"] + ["noop"] * 16,
            ["move-cloud", "drop-water", "water-fall", "sun-suppresses-growth"],
            note="Growth requires the sun NOT overlapping the cloud; ten lefts park the "
                 "cloud on the sun, so the shower is absorbed with zero growth."),
        exact("7xf97", "bloom-purple", "L4",
              "Water the plant at column 13 until it produces a purple bloom.",
              (["down"] + ["noop"] * 16) * 3,
              ["drop-water", "water-fall", "grow-leaf", "bloom"]),
    ]


def sand_specs() -> list[dict[str, Any]]:
    return [
        exact("va6fq", "settle-sand", "L1", "Add one grain of sand and let it settle.",
              ["click 2 3"] + ["noop"] * 12,
              ["sand-brush", "sand-settle"]),
        exact("va6fq", "pour-water", "L1",
              "Pour one block of water into the sand and let it settle.",
              ["click 0 7", "click 4 7"] + ["noop"] * 20,
              ["select-water", "water-spawn", "water-flow"]),
        exact("va6fq", "water-channel", "L2",
              "Pour water into the hole in the left side of the sand.",
              ["click 0 7", "click 5 3"] + ["noop"] * 20,
              ["select-water", "water-spawn", "water-flow", "wet-sand"],
              note="Review 2026-08-30: retargeted from the old top-of-pile pour; water "
                   "dropped into the left hole (column 3) drains to the bottom-left."),
        exact("va6fq", "sand-rectangle", "L3",
              "Make the block of sand look like a complete rectangle without holes.",
              ["click 6 3", "click 5 3", "click 6 6", "click 5 6"] + ["noop"] * 4,
              ["sand-brush", "sand-settle"],
              note="Bottom of each hole first: a grain dropped at row 5 falls to row 6 "
                   "and blocks the second click."),
        exact("va6fq", "wet-sand", "L3",
              "Pour water into the hole in the right side of the sand and wet at "
              "least four grains.",
              ["click 0 7", "click 4 6", "click 5 6"] + ["noop"] * 24,
              ["select-water", "water-flow", "wet-sand", "liquid-sand"],
              note="Retargeted to the RIGHT hole (review 2026-08-30) so water-channel "
                   "(left hole) and wet-sand stay distinct trajectories; one block wets "
                   "only 3 grains, two are needed."),
        exact("va6fq", "mixed-cascade", "L4",
              "Add sand, then two streams of water, and let the mixed cascade settle.",
              ["click 2 3"] + ["noop"] * 12
              + ["click 0 7", "click 2 5", "click 2 6"] + ["noop"] * 35,
              ["sand-settle", "select-water", "water-flow", "wet-sand"],
              must_keep=["click 2 3"],
              note="must_keep pins the sand phase: the pre-fix checker was satisfied by "
                   "the pre-existing pile and compression deleted the sand click."),
        exact("va6fq", "both-holes", "L4",
              "Add one block of sand in each hole, then pour one block of water into "
              "each hole and let it settle.",
              ["click 6 3", "click 6 6", "click 0 7", "click 5 3", "click 5 6"]
              + ["noop"] * 20,
              ["sand-brush", "select-water", "water-flow", "wet-sand"],
              must_keep=["click 6 3", "click 6 6"]),
    ]


def logic_specs() -> list[dict[str, Any]]:
    settled = ["noop"]
    return [
        exact("logic_gates", "switch-one", "L1", "Turn on the left switch and settle the circuit.",
              ["click 12 4", "noop", "noop"],
              ["toggle-switch", "wire-delay", "or", "not", "xor"], prefix=settled),
        exact("logic_gates", "switch-two", "L2", "Turn on the right switch and settle the circuit.",
              ["click 12 19", "noop", "noop"],
              ["toggle-switch", "wire-delay", "or", "xor"], prefix=settled),
        exact("logic_gates", "first-off-rest-on", "L3",
              "Set the switches so that the first output light from the top is off "
              "and the other three are on.",
              ["click 12 4", "click 12 19", "noop", "noop"],
              ["toggle-switch", "wire-delay", "or", "not", "xor"],
              prefix=["noop", "click 12 4", "noop", "noop"],
              note="From the left-switch-on prefix: turn the left switch OFF and the "
                   "right one ON (AND off, OR/NOT/XOR on). Replaces transfer-input "
                   "(review 2026-08-30)."),
        exact("logic_gates", "both-inputs", "L4",
              "Set the switches so that the first and second output lights from the "
              "top are on and the other two are off.",
              ["click 12 4", "click 12 19", "noop", "noop"],
              ["toggle-switch", "wire-delay", "and", "or", "not", "xor"], prefix=settled),
    ]


def magnet_specs() -> list[dict[str, Any]]:
    return [
        exact("7www9", "move-away", "L1", "Move the blue magnet three cells left.",
              ["left"] * 3, ["move-magnet"]),
        exact("7www9", "snap-above", "L2",
              "Approach from above and let opposite poles snap the magnets together.",
              ["up", "right", "noop"],
              ["move-magnet", "opposite-pole-attraction"]),
        exact("7www9", "snap-below", "L3",
              "Approach from below and let opposite poles snap the magnets together.",
              ["down", "right", "noop"],
              ["move-magnet", "opposite-pole-attraction"]),
        exact("7www9", "route-around", "L4",
              "Route the blue magnet around the fixed magnet to its far side.",
              ["right", "up", "up", "right", "right", "right", "noop"],
              ["like-pole-repulsion", "opposite-pole-attraction", "route-around"]),
    ]


def colour_lines_specs() -> list[dict[str, Any]]:
    out = []
    for seed in MULTI_SEEDS:
        out.append(semantic(
            "colour_lines", f"move-blue-s{seed}", "move-blue", "L3",
            "Select the original blue ball and move it from (5,4) to (2,4).",
            seed, [], ["click 5 4", "click 2 4"] + ["noop"] * 5,
            ["select-ball", "set-destination", "walk-to-destination", "random-spawn"],
            note="The random extra ball is deliberately ignored by the semantic goal."))
    return out


def _set_reference(seed: int) -> tuple[list[str], list[str]]:
    prefix = ["noop"] * 9
    grid = raw_trace(GAMES["SET"][0], seed, prefix)[-1]
    mapping = {"coral": 1, "blue": 2, "seagreen": 3}
    cards = [((r, c), mapping[grid[r][c]]) for r in (3, 9, 15) for c in (5, 9, 13)]
    for triple in itertools.combinations(cards, 3):
        if sum(card[1] for card in triple) % 3 == 0:
            return prefix, [f"click {r} {c}" for (r, c), _value in triple] + ["noop"]
    raise RuntimeError(f"seed {seed} produced no valid SET triple")


def set_specs() -> list[dict[str, Any]]:
    out = []
    for seed in MULTI_SEEDS:
        prefix, plan = _set_reference(seed)
        out.append(semantic(
            "SET", f"remove-valid-set-s{seed}", "remove-valid-set", "L4",
            "Remove any valid set of three cards from the dealt board.",
            seed, prefix, plan,
            ["deal-cards", "select-card", "validate-set", "remove-set"],
            note="The checker accepts any valid triple; the stored plan is only one witness."))
    return out


def diffusion_specs() -> list[dict[str, Any]]:
    out = [
        semantic("diffusion", "open-membrane", "open-membrane", "L1",
                 "Open the membrane to its sparsest state.", 101, [], ["up"],
                 ["cycle-density"]),
        semantic("diffusion", "half-membrane", "half-membrane", "L2",
                 "Set the membrane to its middle density.", 211, [], ["up", "up"],
                 ["cycle-density"]),
    ]
    for seed in MULTI_SEEDS:
        out.append(semantic(
            "diffusion", f"two-colour-half-s{seed}", "two-colour-half", "L3",
            "Set middle membrane density and add one red and one blue cell.",
            seed, [], ["up", "up", "click 4 1", "click 4 7"],
            ["cycle-density", "spawn-red", "spawn-blue", "random-walk"],
            note="Random cell positions are outside the goal; controlled counts and density are scored."))
    # Harder replacements (2026-08-30): the >= count goals above saturate under the
    # 50-action any-step evaluator (random floor ~1.0). EXACT upper-bounded counts keep
    # the floor down: random clicking overshoots them, wrong-side clicks break them.
    # (red-pair-mid and balanced-brood-mid were removed in the 2026-08-30 review;
    # balanced-brood-crossed below is the redesigned L4.)
    for seed in MULTI_SEEDS:
        out.append(semantic(
            "diffusion", f"crimson-trio-mid-s{seed}", "crimson-trio-mid", "L3",
            "Set middle membrane density with exactly four red cells and one blue cell.",
            seed, [], ["up", "up", "click 7 0", "click 5 2", "click 7 2"],
            ["spawn-red", "cycle-density", "random-walk"],
            note="Exactly three spawned reds and no blues; walk positions are outside the goal."))
    # per-seed noop tails: the crossing is MOMENTARY (walkers wander back), so each
    # reference must END on its seed's first crossing step (engine-probed 37/40/23) --
    # a longer tail fails final-mode scoring outright and compression cannot rescue it.
    for seed, tail in ((101, 32), (211, 35), (307, 18)):
        out.append(semantic(
            "diffusion", f"balanced-brood-crossed-s{seed}", "balanced-brood-crossed", "L4",
            "Set the sparsest membrane density with exactly three red cells and "
            "three blue cells and let at least one of each diffuse into the "
            "opposite section.",
            seed, [],
            ["up", "click 7 1", "click 5 0", "click 1 7", "click 3 8"] + ["noop"] * tail,
            ["spawn-red", "spawn-blue", "cycle-density", "random-walk", "membrane-crossing"],
            note="Sparsest density opens the wall; the noop tail waits for the seeded "
                 "walk to carry one red right of column 4 and one blue left of it "
                 "(compression trims it to the first crossing)."))
    return out


def dino_specs() -> list[dict[str, Any]]:
    out = []
    for seed in MULTI_SEEDS:
        out.append(semantic(
            "dino", f"survive-two-cacti-s{seed}", "survive-two-cacti", "L4",
            "Remain alive through two cactus passes.",
            seed, ["noop"] * 10,
            ["noop"] * 3 + ["up"] + ["noop"] * 19 + ["up"] + ["noop"] * 6,
            ["cactus-drift", "jump", "gravity", "collision"],
            note="This is a deadline/survival predicate; an exact bird frame is not scored."))
    return out


_EXACTLY_THREE_PLANS = {
    101: ["right", "up", "noop", "noop", "right", "up", "noop", "right", "noop", "left",
          "up", "left", "up", "noop", "up", "left", "left", "up"] + ["noop"] * 18,
    211: ["up", "right", "left", "up", "left", "up", "noop", "up", "left", "left",
          "left", "up", "left", "right", "noop", "noop", "noop", "left", "up"]
         + ["noop"] * 17,
    307: ["up", "up", "up", "right", "noop", "right", "left", "right", "up"]
         + ["noop"] * 16,
}

_DOWN_FIVE_PLANS = {
        101: ['right', 'up', 'noop', 'noop', 'right', 'up', 'noop', 'right', 'noop', 'left', 'up', 'left', 'up', 'noop', 'up', 'left', 'left', 'up', 'noop', 'left', 'left', 'noop', 'up', 'left', 'noop', 'noop', 'noop', 'up', 'left', 'left', 'up', 'noop', 'right', 'noop', 'right', 'noop', 'up', 'left', 'noop', 'left', 'up', 'left', 'up', 'noop', 'right'],
        211: ['up', 'right', 'left', 'up', 'left', 'up', 'noop', 'up', 'left', 'left', 'left', 'up', 'left', 'right', 'noop', 'noop', 'noop', 'left', 'up', 'left', 'up', 'right', 'up', 'noop', 'noop', 'right', 'left', 'left', 'left', 'right', 'left', 'up', 'right', 'up', 'right', 'right', 'up', 'right', 'up', 'left', 'left', 'right', 'left', 'noop', 'left', 'left', 'left', 'left', 'left', 'right'],
        307: ['up', 'up', 'up', 'right', 'noop', 'right', 'left', 'right', 'up', 'left', 'left', 'right', 'left', 'right', 'right', 'right', 'right', 'up', 'right', 'noop', 'noop', 'left', 'right', 'left', 'up', 'noop', 'up', 'up', 'right', 'up', 'right', 'noop', 'right', 'up', 'left', 'noop', 'left', 'up', 'left', 'right', 'up', 'right', 'right', 'noop'],
}


def invader_specs() -> list[dict[str, Any]]:
    out = []
    for seed in MULTI_SEEDS:
        out.append(semantic(
            "f5w3n", f"down-five-s{seed}", "down-five", "L4",
            "Shoot down at least five invaders and keep the hero alive.",
            seed, [], list(_DOWN_FIVE_PLANS[seed]),
            ["fire", "bullet-travel", "enemy-hit", "enemy-volley", "survive"],
            note="Reference found by seeded random search and checker-compressed; "
                 "review 2026-08-30 calibration: >=3 kills is random-trivial "
                 "(floor ~0.5), all 10 unreachable within the 50-action cap (max "
                 "observed 6), >=5 floors at 0.02-0.03."))
    for seed in MULTI_SEEDS:
        out.append(semantic(
            "f5w3n", f"exactly-one-s{seed}", "exactly-one", "L2",
            "Shoot exactly one invader, let the shot resolve, and keep the hero alive.",
            seed, [], ["up"] + ["noop"] * 16,
            ["fire", "bullet-travel", "enemy-hit", "enemy-volley", "survive"],
            note="Settled-count rule (exact deficit + no bullet in flight, 2-frame guard) "
                 "replaces the 2026-09-01-retired shoot-enemy rows: the at-least-one goal "
                 "was random-trivial at cap 50 (floors 0.90-0.96) once its occlusion "
                 "false-positive was fixed. Preview floors 0.015-0.025."))
    for seed in MULTI_SEEDS:
        out.append(semantic(
            "f5w3n", f"exactly-three-s{seed}", "exactly-three", "L3",
            "Shoot exactly three invaders, let the shots resolve, and keep the hero alive.",
            seed, [], list(_EXACTLY_THREE_PLANS[seed]),
            ["fire", "bullet-travel", "enemy-hit", "enemy-volley", "survive", "aim-multiple"],
            note="Reference = the down-five kill chain truncated after the third resolved "
                 "kill + settling tail; stopping on an exact count is the control skill "
                 "the at-least form could not test. Preview floors 0.005-0.025."))
    return out


def all_specs() -> list[dict[str, Any]]:
    return (paint_specs() + egg_specs() + grow_specs() + sand_specs() + logic_specs()
            + magnet_specs() + colour_lines_specs() + set_specs() + diffusion_specs()
            + dino_specs() + invader_specs())
