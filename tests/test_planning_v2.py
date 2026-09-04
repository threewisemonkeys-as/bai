"""Focused schema and engine tests for the 15-game planning set."""
from collections import Counter, defaultdict

from offline_learning.nl_goals import GOALS as LEGACY_GOALS
from offline_learning.planning_nl_goals import (
    GOALS_BY_ID, get_python_goal, legacy_checker_id, score_python_goal,
)
from offline_learning.planning_v2 import (
    SELECTED_GAMES, STOCHASTIC_GAMES, materialize, raw_trace, stable_seed,
    wrapper_trace,
)
from offline_learning.planning_v2_specs import all_specs, paint_specs
from offline_learning.scripts import build_planning_v2


def test_python_goal_program_and_quiescence_are_scored_directly():
    start = [["black"] * 8 for _ in range(8)]
    final = [row[:] for row in start]
    final[4][4] = "red"
    goal = get_python_goal("eahcw/red-mark")

    assert score_python_goal(
        goal, start, [final], ["click 4 4"], stable_after_final=False,
    ) == (False, None)
    assert score_python_goal(
        goal, start, [final], ["click 4 4"], stable_after_final=True,
    ) == (True, 1)


def test_every_legacy_program_is_reused_by_the_unified_registry():
    # 68 of these back a row of the shipped 86-problem set; the other two belong to
    # 83wkq, which is not one of the selected 15 games.
    assert len(GOALS_BY_ID) == 70
    for legacy in LEGACY_GOALS:
        goal = get_python_goal(legacy_checker_id(legacy.game, legacy.pid))
        assert goal.check is legacy.check
        assert goal.nl == legacy.nl
        assert goal.seed == legacy.seed
        assert goal.success_mode == "any"


def test_specs_cover_every_new_game_and_stochastic_games_have_multiseed_template():
    specs = all_specs()
    counts = Counter(s["game"] for s in specs)
    assert set(counts) == set(SELECTED_GAMES) - {"bt3gb", "dq8gc", "n2ntd", "s2kt7"}
    by_template = defaultdict(set)
    for spec in specs:
        by_template[(spec["game"], spec.get("template_id", spec["id"]))].add(spec["seed"])
        assert "goal_mode" not in spec
        assert spec["nl_checker"]
        assert not any(key in spec for key in ("goal_spec", "author_goal_spec"))
    for game in STOCHASTIC_GAMES:
        assert max(len(seeds) for (g, _), seeds in by_template.items() if g == game) >= 3


def test_raw_and_wrapper_drivers_agree_with_a_prefix():
    actions = ["up", "up", "click 0 0", "noop", "noop"]
    assert raw_trace("egg", 101, actions) == wrapper_trace("egg", 101, actions)


def test_materialization_compresses_without_shipping_a_quiescence_noop():
    row = materialize(paint_specs()[0], random_trials=0)
    assert row["plan"] == ["click 4 4"]
    assert row["h"] == 1 and row["frame_reference_quiescent"] is True
    assert row["nl_require_quiescent"] is True
    assert row["nl_checker"] == "eahcw/red-mark"
    assert not any(key in row for key in ("goal_spec", "author_goal_spec"))


def test_task_floor_seed_is_process_stable():
    assert stable_seed("eahcw:red-mark:s101") == stable_seed("eahcw:red-mark:s101")
    assert stable_seed("eahcw:red-mark:s101") != stable_seed("egg:shatter:s101")


def test_legacy_normalization_restarts_to_a_fixed_point(monkeypatch):
    allowed = {
        ("left", "noop", "right"),
        ("noop", "noop", "right"),
    }

    def fake_trace(_program, _seed, actions):
        return [list(actions), list(actions)]

    def fake_success(_probe, presentation, start, _frames):
        assert presentation == "frame"
        return tuple(start) in allowed, None

    monkeypatch.setattr(build_planning_v2, "raw_trace", fake_trace)
    monkeypatch.setattr(build_planning_v2, "success", fake_success)
    plan, repairs = build_planning_v2.normalize_legacy_plan({
        "program": "fake", "seed": 0, "goal": [],
        "plan": ["left", "up", "right"],
    })

    assert plan == ["noop", "noop", "right"]
    assert [repair["index"] for repair in repairs] == [1, 0]
