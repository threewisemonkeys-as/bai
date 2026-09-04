"""Regression tests for v1/v2 curated-plan evaluation protocols."""
import asyncio
import pytest
from collections import Counter

from offline_learning.scripts import eval_curated_plan as evaluator


def _problems():
    _meta, problems = evaluator.load_eval_problems(evaluator.DEFAULT_PROBLEMS)
    return problems


def test_declared_presentation_is_rejected():
    with pytest.raises(ValueError, match="unknown goal presentation"):
        evaluator.select_goal_presentation(_problems(), "declared")


def test_every_problem_supports_explicit_frame_and_nl_presentations():
    problems = _problems()
    frame = evaluator.select_goal_presentation(problems, "frame")
    nl = evaluator.select_goal_presentation(problems, "nl")

    assert len(frame) == len(problems) == 86
    assert {p["_eval_presentation"] for p in frame} == {"frame"}
    assert Counter(p["_eval_checker_source"] for p in frame) == {"exact-frame": 86}
    assert len(nl) == len(problems)
    assert {p["_eval_presentation"] for p in nl} == {"nl"}
    assert Counter(p["_eval_checker_source"] for p in nl) == {"python-registry": 86}


def test_nl_prompt_uses_canonical_sentence_and_never_reference_frame():
    nl = evaluator.select_goal_presentation(_problems(), "nl")
    legacy = next(p for p in nl if p["source"] == "curated-v1-accepted-migration")
    assert legacy["_eval_nl_goal"] == legacy["_eval_python_goal"].nl
    assert legacy["_eval_nl_goal"] == legacy["nl_goal"]
    assert legacy["nl_checker"] == legacy["_eval_python_goal"].checker_id

    reference = evaluator.gstr(legacy["goal"])
    prompt = evaluator.build_prompt(
        legacy, "raw", evaluator.gstr(legacy["start"]),
        goal_features="SHOULD_NOT_APPEAR",
    )
    assert legacy["_eval_nl_goal"] in prompt
    assert reference not in prompt
    assert "SHOULD_NOT_APPEAR" not in prompt


def test_exact_prompt_respects_any_step_and_final_step_scoring():
    frame = evaluator.select_goal_presentation(_problems(), "frame", "reference")
    any_step = next(p for p in frame if p["_eval_success_mode"] == "any")
    final_step = next(p for p in frame if p["_eval_success_mode"] == "final")
    any_prompt = evaluator.build_prompt(any_step, "raw", evaluator.gstr(any_step["start"]))
    final_prompt = evaluator.build_prompt(
        final_step, "raw", evaluator.gstr(final_step["start"])
    )
    assert "at some point during your plan" in any_prompt
    assert "after your FINAL action" in final_prompt


def test_v1_list_is_normalized_to_replayable_exact_any_step_rows():
    path = evaluator.REPO / "logs/2026-08-18/curated/problems.json"
    meta, problems = evaluator.load_eval_problems(path)
    assert meta == {}
    assert problems
    assert all(p["prefix"] == [] for p in problems)
    assert all(p["frame_success_mode"] == "any" for p in problems)
    assert all("goal_mode" not in p for p in problems)
    assert all(p["nl_checker_version"] == evaluator.CHECKER_VERSION for p in problems)
    assert len({p["task_uid"] for p in problems}) == len(problems)


def test_v2_artifact_contains_no_declarative_goal_payloads():
    problems = _problems()
    forbidden = {"goal_spec", "author_goal_spec", "author_require_quiescent"}
    assert all(not forbidden.intersection(problem) for problem in problems)
    assert len({problem["nl_checker"] for problem in problems}) == 68


def test_raw_arm_replays_prefix_without_artifacts_and_wc_marks_nl_na(
        monkeypatch, tmp_path):
    problem = next(
        p for p in evaluator.select_goal_presentation(_problems(), "nl")
        if p["game"] == "SET" and p["prefix"]
    )
    seen = {}

    async def fake_llm(prompt, _sem, _llm):
        seen["prompt"] = prompt
        body = "\n".join(problem["plan"])
        return f"<reasoning>fixture</reasoning>\n<plan>\n{body}\n</plan>", "", 0.0, []

    monkeypatch.setattr(evaluator, "llm_call", fake_llm)
    missing_artifacts = tmp_path / "missing"
    raw_result = asyncio.run(evaluator.eval_game(
        problem["game"], [problem], asyncio.Semaphore(1), object(),
        missing_artifacts, ["raw"], 1,
    ))
    row = raw_result["rows"][0]
    assert row["raw"]["pass_rate"] == 1.0
    assert row["prefix_len"] == len(problem["prefix"])
    assert row["goal_grid"] is None
    assert evaluator.gstr(problem["goal"]) not in seen["prompt"]

    nl_problem = evaluator.configure_evaluation_goal(problem, "nl")
    wc_result = asyncio.run(evaluator.eval_game(
        problem["game"], [nl_problem], None, None, missing_artifacts, ["wc"], 1,
    ))
    assert wc_result["rows"][0]["wc"]["status"] == "not-applicable"
    assert wc_result["rows"][0]["wc"]["pass_rate"] is None


# ---------------------------------------------------- reference-scaled action caps
def test_action_cap_doubles_short_problems_and_scales_long_ones_by_half():
    assert [evaluator.action_cap(n) for n in (1, 4, 10)] == [2, 8, 20]
    assert [evaluator.action_cap(n) for n in (11, 17, 40)] == [17, 26, 60]
    # the rule is discontinuous at the boundary by design: 10 -> 20, 11 -> 17
    assert evaluator.action_cap(11) < evaluator.action_cap(10)
    for bad in (0, -3, 2.5, True, None):
        with pytest.raises(ValueError):
            evaluator.action_cap(bad)


def test_reference_reach_reads_the_anystep_field_not_the_success_mode_one():
    """dino stores reference_reached_at=30 under `final` mode but first holds at 10."""
    dino = [p for p in _problems() if p["game"] == "dino"]
    assert dino, "fixture set no longer contains dino"
    for p in dino:
        assert p["nl_reference_reached_at"] == 30
        assert evaluator.reference_reach(p, "nl") == 10
        assert evaluator.action_cap(evaluator.reference_reach(p, "nl")) == 20


def test_fixed_mode_gives_every_row_the_flat_budget():
    problems = evaluator.select_goal_presentation(_problems(), "nl")
    caps = evaluator.resolve_action_caps(problems, "fixed")
    assert set(caps.values()) == {evaluator.PLAN_CAP}


def test_per_game_cap_is_the_max_over_the_games_rows():
    problems = evaluator.select_goal_presentation(_problems(), "nl")
    per_problem = evaluator.resolve_action_caps(problems, "per-problem")
    per_game = evaluator.resolve_action_caps(problems, "per-game")
    for p in problems:
        assert per_game[p["task_uid"]] >= per_problem[p["task_uid"]]
    by_game = {}
    for p in problems:
        by_game.setdefault(p["game"], []).append(per_problem[p["task_uid"]])
    for p in problems:
        assert per_game[p["task_uid"]] == max(by_game[p["game"]])


def test_scaled_caps_need_a_measured_reach():
    problems = evaluator.select_goal_presentation(_problems()[:2], "nl")
    for p in problems:
        p["nl_anystep_reached_at"] = None
    with pytest.raises(ValueError, match="any-step reference reach"):
        evaluator.resolve_action_caps(problems, "per-game")


def test_apply_action_caps_stamps_the_budget_on_every_row():
    problems = evaluator.select_goal_presentation(_problems(), "nl")
    caps = evaluator.apply_action_caps(problems, "per-problem")
    assert all(p["_eval_action_cap"] == caps[p["task_uid"]] for p in problems)


def test_floor_refuses_a_cap50_floor_under_a_scaled_budget():
    """A floor and a score only compare when both were measured at the same budget."""
    problems = evaluator.select_goal_presentation(_problems(), "nl")
    row = next(p for p in problems if p["nl_random_success_cap50"] is not None
               and evaluator.action_cap(evaluator.reference_reach(p, "nl")) != 50)
    assert evaluator._floor(row) == row["nl_random_success_cap50"]  # flat default

    evaluator.apply_action_caps([row], "per-problem")
    assert evaluator._floor(row) is None                            # mismatched, refused

    row["nl_random_floors"] = {str(row["_eval_action_cap"]): {
        "success": 0.25, "trials": 200, "cap_mode": "per-problem"}}
    assert evaluator._floor(row) == 0.25                            # matched, used
