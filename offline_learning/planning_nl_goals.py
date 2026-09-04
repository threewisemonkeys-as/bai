"""Executable Python goal programs for NL evaluation of the 15-game planning set.

Every checker has the legacy interface check(grids, actions) -> bool. grids[0] is the
planning state after replaying the problem prefix, and the remaining grids are the states
produced by actions. The English sentence and executable checker live together in
PythonGoal; problem rows refer to them by stable nl_checker identifiers.

The registry deliberately contains no declarative predicate interpreter. Helpers such as
cells and color_count are ordinary Python utilities that checker programs may use.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

from offline_learning.nl_goals import GOALS as LEGACY_GOALS

CHECKER_VERSION = "planning-python-goals-v1"

FrozenGrid = tuple[tuple[str, ...], ...]
GridLike = Sequence[Sequence[str]]
Check = Callable[[list[FrozenGrid], list[str]], bool]


def freeze_grid(grid: GridLike) -> FrozenGrid:
    return tuple(tuple(row) for row in grid)


def cells(grid: GridLike, *colors: str) -> frozenset[tuple[int, int]]:
    wanted = set(colors)
    return frozenset(
        (row, col)
        for row, values in enumerate(grid)
        for col, value in enumerate(values)
        if value in wanted
    )


def color_count(grid: GridLike, color: str) -> int:
    return sum(row.count(color) for row in grid)


def _coordinate(grid: GridLike, color: str, axis: int, choose, empty: int = -1) -> int:
    points = cells(grid, color)
    return choose(point[axis] for point in points) if points else empty


def min_row(grid: GridLike, color: str, empty: int = -1) -> int:
    return _coordinate(grid, color, 0, min, empty)


def max_row(grid: GridLike, color: str, empty: int = -1) -> int:
    return _coordinate(grid, color, 0, max, empty)


def min_col(grid: GridLike, color: str, empty: int = -1) -> int:
    return _coordinate(grid, color, 1, min, empty)


def max_col(grid: GridLike, color: str, empty: int = -1) -> int:
    return _coordinate(grid, color, 1, max, empty)


_SET_ROWS = (3, 9, 15)
_SET_COLS = (5, 9, 13)


def set_card_count(grid: GridLike) -> int:
    return sum(grid[row][col] != "black" for row in _SET_ROWS for col in _SET_COLS)


def set_selected_count(grid: GridLike) -> int:
    return sum(grid[row - 2][col] == "gold" for row in _SET_ROWS for col in _SET_COLS)


@dataclass(frozen=True)
class PythonGoal:
    checker_id: str
    nl: str
    check: Check
    success_mode: Literal["any", "final"]
    require_quiescent: bool = False
    seed: int | None = None
    reference_plan: tuple[str, ...] | None = None
    positives: tuple[tuple[str, ...], ...] = field(default_factory=tuple)
    negatives: tuple[tuple[str, ...], ...] = field(default_factory=tuple)
    note: str = ""


# ------------------------------------------------------------------ paint / eahcw
def _eahcw_red_mark(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    return color_count(grids[-1], "red") == 1


def _eahcw_gold_blue_pair(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    grid = grids[-1]
    return grid[3][3] == "gold" and grid[4][4] == "blue"


def _eahcw_three_colour_diagonal(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    grid = grids[-1]
    return (grid[3][3] == "purple" and grid[4][4] == "green"
            and grid[5][5] == "gold")


def _eahcw_five_colour_palette(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    # ORDERED: each colour's first appearance must come strictly after the previous
    # one's (review 2026-08-30: the order of creation is part of the task).
    firsts = []
    for colour in ("red", "gold", "purple", "green", "blue"):
        first = next((i for i, g in enumerate(grids) if color_count(g, colour) > 0), None)
        if first is None:
            return False
        firsts.append(first)
    return all(a < b for a, b in zip(firsts, firsts[1:]))


# --------------------------------------------------------------------- egg / egg
def _egg_carry_left(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    grid = grids[-1]
    return max_col(grid, "tan") == 6 and color_count(grid, "tan") == 21


def _egg_controlled_gravity_drop(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    grid = grids[-1]
    return max_row(grid, "tan") == 15 and grid[0][0] == "red"


def _egg_shatter(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    grid = grids[-1]
    return color_count(grid, "tan") == 0 and color_count(grid, "gold") >= 14


def _egg_left_fragment_bed(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    grid = grids[-1]
    return (color_count(grid, "tan") == 0 and max_row(grid, "gold") == 15
            and min_col(grid, "gold") == 0)


# --------------------------------------------------------------- growing / 7xf97
def _grow_move_cloud(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    return min_col(grids[-1], "gray") == 9


def _grow_move_sun(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    return min_col(grids[-1], "gold") == 1


def _grow_water_left_plant(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    return grids[-1][14][11] == "green"


def _grow_bloom_purple(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    return grids[-1][12][13] == "mediumpurple"


# ------------------------------------------------------------------ sand / va6fq
def _sand_settle_one(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    return color_count(grids[-1], "tan") - color_count(grids[0], "tan") >= 1


def _sand_water_channel(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    # Water poured into the LEFT hole (column 3) drains through the pile and settles at
    # the bottom-left corner, wetting the left flank on the way (engine-probed). The
    # in-hole witness is required: water clicked straight onto the left flank reaches
    # the same corner without ever entering the hole (review-check F4 cheat).
    entered = any((r, 3) in cells(g, "skyblue") for g in grids for r in (4, 5, 6))
    grid = grids[-1]
    water = cells(grid, "skyblue")
    return (entered and len(water) >= 1 and all(r >= 8 and c <= 2 for r, c in water)
            and color_count(grid, "sandybrown") >= 2)


def _sand_wet_sand(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    grid = grids[-1]
    return color_count(grid, "sandybrown") >= 4 and color_count(grid, "skyblue") >= 1


def _sand_total(grid: FrozenGrid) -> int:
    return color_count(grid, "tan") + color_count(grid, "sandybrown")


def _sand_mixed_cascade(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    # The added grain must show up as a TRANSIENT peak above the starting pile count --
    # final counts are unreliable because liquified sand merges/overlaps during the
    # cascade, and a final-only count let compression delete the authored sand phase.
    peak = max(_sand_total(g) for g in grids)
    grid = grids[-1]
    return (peak > _sand_total(grids[0]) and color_count(grid, "skyblue") >= 2
            and color_count(grid, "sandybrown") >= 3)


def _sand_pour_water(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    grid = grids[-1]
    return color_count(grid, "skyblue") >= 1 and color_count(grid, "sandybrown") >= 1


def _sand_rectangle(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    sand = cells(grids[-1], "tan", "sandybrown")
    rows = [r for r, _ in sand]
    cols = [c for _, c in sand]
    full = all((r, c) in sand
               for r in range(min(rows), max(rows) + 1)
               for c in range(min(cols), max(cols) + 1))
    return full and len(sand) > len(cells(grids[0], "tan", "sandybrown"))


def _sand_both_holes(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    # Placement witnesses: dropped DRY sand renders tan at the hole bottoms (sliding
    # liquified pile sand is sandybrown), and the poured water must pass through each
    # hole (review-check F3: sand-anywhere + corner water previously passed).
    sand_left = any((6, 3) in cells(g, "tan") for g in grids)
    sand_right = any((6, 6) in cells(g, "tan") for g in grids)
    water_left = any((r, 3) in cells(g, "skyblue") for g in grids for r in (4, 5))
    water_right = any((r, 6) in cells(g, "skyblue") for g in grids for r in (4, 5))
    peak = max(_sand_total(g) for g in grids)
    grid = grids[-1]
    water = cells(grid, "skyblue")
    left = any(r >= 8 and c <= 2 for r, c in water)
    right = any(r >= 8 and c >= 7 for r, c in water)
    return (peak >= _sand_total(grids[0]) + 2 and sand_left and sand_right
            and water_left and water_right and left and right)


# ----------------------------------------------------------- logic / logic_gates
def _egg_boundary_drop(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    # Raised exactly to the boundary (egg top row 8 == origin y 10; one cell higher
    # shatters when gravity is toggled), then landed intact at the floor (top row 11).
    tops = [min((r for r, _ in cells(g, "tan")), default=99) for g in grids]
    grid = grids[-1]
    return (min(tops) == 8 and tops[-1] == 11
            and color_count(grid, "tan") == 21 and color_count(grid, "gold") == 0)


def _water_above_leaf(grid: FrozenGrid) -> bool:
    leaves = cells(grid, "green", "mediumpurple")
    return any((r + 1, c) in leaves for r, c in cells(grid, "blue"))


def _grow_shower_drain(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    # Growth only fires when the sun does NOT overlap the cloud; showering from under
    # the sun lets the water be absorbed by the plant with zero growth. The
    # water-above-a-leaf witness is required: without it, water dropped over an empty
    # column drains off-grid and "showers" nothing (review-check F1 cheat).
    leaves0 = color_count(grids[0], "green") + color_count(grids[0], "mediumpurple")
    grid = grids[-1]
    return (any(_water_above_leaf(g) for g in grids)
            and color_count(grid, "blue") == 0
            and color_count(grid, "green") + color_count(grid, "mediumpurple") == leaves0)


def _logic_light(grid: FrozenGrid, row0: int) -> bool:
    """Output blocks are 2x2 at column 12: AND row 4, OR row 8, NOT row 16, XOR row 20."""
    return (row0, 12) in cells(grid, "orange")


def _logic_first_off_rest_on(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    grid = grids[-1]
    return (not _logic_light(grid, 4) and _logic_light(grid, 8)
            and _logic_light(grid, 16) and _logic_light(grid, 20))


def _diffusion_balanced_brood_crossed(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    grid = grids[-1]
    reds, blues = cells(grid, "red"), cells(grid, "blue")
    return (color_count(grid, "white") == 3 and len(reds) == 3 and len(blues) == 3
            and any(c > 4 for _, c in reds) and any(c < 4 for _, c in blues))


def _invader_down_five(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    # Two consecutive frames guard against one-tick rendering occlusions (a spawning
    # orange bullet over an enemy, a just-fired red bullet over the hero).
    if len(grids) < 2:
        return False
    return all(color_count(g, "blue") <= 5 and color_count(g, "gray") >= 1
               for g in grids[-2:])


def _logic_switch_one(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    grid = grids[-1]
    return grid[12][4] == "red" and grid[12][19] == "pink"


def _logic_switch_two(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    grid = grids[-1]
    return grid[12][4] == "pink" and grid[12][19] == "red"


def _logic_both_inputs(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    grid = grids[-1]
    return (_logic_light(grid, 4) and _logic_light(grid, 8)
            and not _logic_light(grid, 16) and not _logic_light(grid, 20))


# --------------------------------------------------------------- magnets / 7www9
def _magnet_move_away(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    return min_col(grids[-1], "blue") == 1


def _magnet_snap_above(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    grid = grids[-1]
    return min_row(grid, "blue") == 6 and min_col(grid, "blue") == 6


def _magnet_snap_below(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    grid = grids[-1]
    return min_row(grid, "blue") == 8 and min_col(grid, "blue") == 6


def _magnet_route_around(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    grid = grids[-1]
    return min_row(grid, "blue") == 5 and min_col(grid, "blue") == 9


# ---------------------------------------------------------- native stochastic NL
def _colour_lines_move_blue(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    grid = grids[-1]
    return grid[2][4] == "blue" and grid[5][4] != "blue"


def _set_remove_valid_set(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    grid = grids[-1]
    return set_card_count(grid) <= 6 and set_selected_count(grid) == 0


def _diffusion_open_membrane(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    return color_count(grids[-1], "white") == 3


def _diffusion_half_membrane(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    return color_count(grids[-1], "white") == 6


def _diffusion_two_colour_half(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    grid = grids[-1]
    return (color_count(grid, "white") == 6 and color_count(grid, "red") >= 2
            and color_count(grid, "blue") >= 2)


def _diffusion_crimson_trio_mid(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    grid = grids[-1]
    return (color_count(grid, "white") == 6 and color_count(grid, "red") == 4
            and color_count(grid, "blue") == 1)


def _dino_survive_two_cacti(grids: list[FrozenGrid], _actions: list[str]) -> bool:
    grid = grids[-1]
    return color_count(grid, "red") >= 1 and min_col(grid, "green", empty=-1) >= 15


def _invader_exactly_n(n: int):
    """Settled-count kill goal: the deficit reads EXACTLY n with no hero bullet still
    in flight and the hero alive, held on two consecutive frames. The two-frame window
    guards one-tick rendering occlusions (a spawning orange bullet over an enemy reads
    as a phantom kill for a single frame); red == 0 here is the "shots resolved" clause
    -- without it, any-step scoring would award an exact count while a further kill was
    already in flight, and "exactly n" would collapse to "at least n"."""
    def check(grids: list[FrozenGrid], _actions: list[str]) -> bool:
        if len(grids) < 2:
            return False
        start = grids[0]
        return all(color_count(start, "blue") - color_count(g, "blue") == n
                   and color_count(g, "red") == 0 and color_count(g, "gray") >= 1
                   for g in grids[-2:])
    return check


_invader_exactly_one = _invader_exactly_n(1)
_invader_exactly_three = _invader_exactly_n(3)


_NEW_GOALS = [
    PythonGoal("eahcw/red-mark", "Paint a red mark at row 4, column 4.",
               _eahcw_red_mark, "final", True),
    PythonGoal("eahcw/gold-blue-pair",
               "Paint a gold mark at (3,3) and a blue mark at (4,4).",
               _eahcw_gold_blue_pair, "final", True),
    PythonGoal("eahcw/three-colour-diagonal",
               "Paint purple, green, and gold marks along the diagonal.",
               _eahcw_three_colour_diagonal, "final", True),
    PythonGoal("eahcw/five-colour-palette",
               "Create five distinct marks in this order: red, gold, purple, green, blue.",
               _eahcw_five_colour_palette, "final", True),
    PythonGoal("egg/carry-left", "Move the intact egg three cells left.",
               _egg_carry_left, "final", True),
    PythonGoal("egg/controlled-gravity-drop",
               "From the raised position, turn gravity on, land the egg, then turn gravity off.",
               _egg_controlled_gravity_drop, "final", True),
    PythonGoal("egg/boundary-drop",
               "Raise the egg to just below the shatter height, then turn gravity on "
               "so it falls and lands intact.",
               _egg_boundary_drop, "final", True),
    PythonGoal("egg/shatter", "Raise the egg above the safe height and shatter it.",
               _egg_shatter, "final"),
    PythonGoal("egg/left-fragment-bed",
               "Carry the egg left, shatter it, and let the fragments settle along the floor.",
               _egg_left_fragment_bed, "final", True),
    PythonGoal("7xf97/move-cloud", "Move the cloud three cells left.",
               _grow_move_cloud, "final", True),
    PythonGoal("7xf97/move-sun", "Move the sun one cell to the right.",
               _grow_move_sun, "final", True),
    PythonGoal("7xf97/water-left-plant",
               "Move the cloud over the plant at column 11 and grow it by one cell.",
               _grow_water_left_plant, "final", True),
    PythonGoal("7xf97/shower-drain",
               "Shower a plant with water and let the water drain completely without "
               "causing any growth.",
               _grow_shower_drain, "final", True),
    PythonGoal("7xf97/bloom-purple",
               "Water the plant at column 13 until it produces a purple bloom.",
               _grow_bloom_purple, "final", True),
    PythonGoal("va6fq/settle-sand", "Add one grain of sand and let it settle.",
               _sand_settle_one, "final", True),
    PythonGoal("va6fq/water-channel",
               "Pour water into the hole in the left side of the sand.",
               _sand_water_channel, "final", True),
    PythonGoal("va6fq/pour-water",
               "Pour one block of water into the sand and let it settle.",
               _sand_pour_water, "final", True),
    PythonGoal("va6fq/sand-rectangle",
               "Make the block of sand look like a complete rectangle without holes.",
               _sand_rectangle, "final", True),
    PythonGoal("va6fq/both-holes",
               "Add one block of sand in each hole, then pour one block of water into "
               "each hole and let it settle.",
               _sand_both_holes, "final", True),
    PythonGoal("va6fq/wet-sand",
               "Pour water into the hole in the right side of the sand and wet at "
               "least four grains.",
               _sand_wet_sand, "final", True),
    PythonGoal("va6fq/mixed-cascade",
               "Add sand, then two streams of water, and let the mixed cascade settle.",
               _sand_mixed_cascade, "final", True),
    PythonGoal("logic_gates/switch-one",
               "Turn on the left switch and settle the circuit.",
               _logic_switch_one, "final", True),
    PythonGoal("logic_gates/switch-two",
               "Turn on the right switch and settle the circuit.",
               _logic_switch_two, "final", True),
    PythonGoal("logic_gates/both-inputs",
               "Set the switches so that the first and second output lights from the "
               "top are on and the other two are off.",
               _logic_both_inputs, "final", True),
    PythonGoal("logic_gates/first-off-rest-on",
               "Set the switches so that the first output light from the top is off "
               "and the other three are on.",
               _logic_first_off_rest_on, "final", True),
    PythonGoal("7www9/move-away", "Move the blue magnet three cells left.",
               _magnet_move_away, "final", True),
    PythonGoal("7www9/snap-above",
               "Approach from above and let opposite poles snap the magnets together.",
               _magnet_snap_above, "final", True),
    PythonGoal("7www9/snap-below",
               "Approach from below and let opposite poles snap the magnets together.",
               _magnet_snap_below, "final", True),
    PythonGoal("7www9/route-around",
               "Route the blue magnet around the fixed magnet to its far side.",
               _magnet_route_around, "final", True),
    PythonGoal("colour_lines/move-blue",
               "Select the original blue ball and move it from (5,4) to (2,4).",
               _colour_lines_move_blue, "final"),
    PythonGoal("SET/remove-valid-set",
               "Remove any valid set of three cards from the dealt board.",
               _set_remove_valid_set, "final"),
    PythonGoal("diffusion/open-membrane", "Open the membrane to its sparsest state.",
               _diffusion_open_membrane, "final"),
    PythonGoal("diffusion/half-membrane", "Set the membrane to its middle density.",
               _diffusion_half_membrane, "final"),
    PythonGoal("diffusion/two-colour-half",
               "Set middle membrane density and add one red and one blue cell.",
               _diffusion_two_colour_half, "final"),
    PythonGoal("diffusion/crimson-trio-mid",
               "Set middle membrane density with exactly four red cells and one blue cell.",
               _diffusion_crimson_trio_mid, "final"),
    PythonGoal("diffusion/balanced-brood-crossed",
               "Set the sparsest membrane density with exactly three red cells and "
               "three blue cells and let at least one of each diffuse into the "
               "opposite section.",
               _diffusion_balanced_brood_crossed, "final"),
    PythonGoal("dino/survive-two-cacti",
               "Remain alive through two cactus passes.",
               _dino_survive_two_cacti, "final"),
    PythonGoal("f5w3n/exactly-one",
               "Shoot exactly one invader, let the shot resolve, and keep the hero alive.",
               _invader_exactly_one, "final"),
    PythonGoal("f5w3n/exactly-three",
               "Shoot exactly three invaders, let the shots resolve, and keep the hero alive.",
               _invader_exactly_three, "final"),
    PythonGoal("f5w3n/down-five",
               "Shoot down at least five invaders and keep the hero alive.",
               _invader_down_five, "final"),
]


def legacy_checker_id(game: str, pid: str) -> str:
    return f"{game}/{pid}"


_LEGACY_PYTHON_GOALS = [
    PythonGoal(
        checker_id=legacy_checker_id(goal.game, goal.pid),
        nl=goal.nl,
        check=goal.check,
        success_mode="any",
        seed=goal.seed,
        reference_plan=tuple(goal.ref) if goal.ref is not None else None,
        positives=tuple(tuple(plan) for plan in goal.positives),
        negatives=tuple(tuple(plan) for plan in goal.negatives),
        note=goal.note,
    )
    for goal in LEGACY_GOALS
]


GOALS_BY_ID: dict[str, PythonGoal] = {}
for _goal in _LEGACY_PYTHON_GOALS + _NEW_GOALS:
    if _goal.checker_id in GOALS_BY_ID:
        raise ValueError(f"duplicate Python goal checker id {_goal.checker_id!r}")
    if _goal.require_quiescent and _goal.success_mode != "final":
        raise ValueError(
            f"{_goal.checker_id}: quiescence is only defined for final-step goals"
        )
    GOALS_BY_ID[_goal.checker_id] = _goal


def get_python_goal(checker_id: str) -> PythonGoal:
    try:
        return GOALS_BY_ID[checker_id]
    except KeyError as exc:
        raise KeyError(f"unknown Python NL checker {checker_id!r}") from exc


def validate_problem_goal(problem: dict) -> PythonGoal:
    checker_id = problem.get("nl_checker")
    if not checker_id:
        raise ValueError(f"{problem.get('task_uid', '<unknown>')}: missing nl_checker")
    goal = get_python_goal(checker_id)
    if problem.get("nl_goal") != goal.nl:
        raise ValueError(
            f"{problem.get('task_uid', '<unknown>')}: nl_goal disagrees with "
            f"{checker_id!r}"
        )
    uid = problem.get("task_uid", "<unknown>")
    if problem.get("nl_checker_version") not in {None, CHECKER_VERSION}:
        raise ValueError(f"{uid}: unsupported nl_checker_version")
    if problem.get("nl_success_mode") not in {None, goal.success_mode}:
        raise ValueError(f"{uid}: nl_success_mode disagrees with {checker_id!r}")
    if problem.get("nl_require_quiescent") not in {None, goal.require_quiescent}:
        raise ValueError(f"{uid}: nl_require_quiescent disagrees with {checker_id!r}")
    if goal.seed is not None and problem.get("seed") != goal.seed:
        raise ValueError(f"{uid}: seed disagrees with {checker_id!r}")
    expected_reference = list(goal.reference_plan or problem.get("plan", []))
    stored_reference = problem.get("nl_reference_plan")
    if stored_reference is not None and stored_reference != expected_reference:
        raise ValueError(f"{uid}: nl_reference_plan disagrees with {checker_id!r}")
    return goal


def checker_holds_at_start(goal: PythonGoal, start: GridLike) -> bool:
    """Test the state condition at the planning state, independent of temporal scheduling."""
    return bool(goal.check([freeze_grid(start)], []))


def score_python_goal(
    goal: PythonGoal,
    start: GridLike,
    frames: list[GridLike | None],
    actions: list[str],
    *,
    stable_after_final: bool | None = None,
) -> tuple[bool, int | None]:
    """Score an executed candidate with legacy-compatible trajectory semantics."""
    if len(frames) != len(actions):
        raise ValueError(
            f"{goal.checker_id}: {len(frames)} frames for {len(actions)} actions"
        )

    frozen_start = freeze_grid(start)
    if goal.success_mode == "final":
        if not frames or frames[-1] is None or any(frame is None for frame in frames):
            return False, None
        if goal.require_quiescent and stable_after_final is not True:
            return False, None
        grids = [frozen_start] + [freeze_grid(frame) for frame in frames if frame is not None]
        hit = bool(goal.check(grids, list(actions)))
        return hit, (len(actions) if hit else None)

    if goal.require_quiescent:
        raise ValueError(f"{goal.checker_id}: any-step quiescence is undefined")
    grids = [frozen_start]
    executed: list[str] = []
    for step, (action, frame) in enumerate(zip(actions, frames), 1):
        if frame is None:
            break
        executed.append(action)
        grids.append(freeze_grid(frame))
        if goal.check(grids, executed):
            return True, step
    return False, None
