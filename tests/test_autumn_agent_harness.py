"""The Autumn agent harness: the contract it must not drift from, and the study round.

Everything the arm's comparability rests on is a property of code borrowed from the
published evaluators, so what is worth testing is that the borrowing still holds -- the
same 86 rows, the same budgets, the goal frame actually absent, the start reproduced --
and the one mechanism that is genuinely new here, the empty-list study round.

The scripted agent means all of this runs with no API key and no cost.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RG = REPO / "RGB-Agent"
for _p in (RG, RG / "research/arc-agi-3"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

pytestmark = pytest.mark.skipif(
    not (RG / "research/autumn/rig.py").is_file()
    or not (REPO / "logs/2026-08-29/planning_v2/problems.json").is_file(),
    reason="needs the prolong-autumn submodule branch and the planning_v2 problem set",
)

rig = pytest.importorskip("research.autumn.rig")
from research.autumn.actions import parse_actions_json  # noqa: E402
from research.autumn.env import AutumnPlanningEnv  # noqa: E402
from research.autumn.runner import AutumnRunner  # noqa: E402

logging.disable(logging.INFO)


class Scripted:
    """An agent that replies from a fixed script. No LLM, no cost.

    Once the script runs out it falls back to `default`, which burns the budget so the
    run ends. The fallback must NOT be an empty list: that is a study round, so the
    session would keep asking for more turns and the counts under test would drift with
    the study budget rather than with the script.
    """

    def __init__(self, *replies, default=None):
        self.replies = list(replies)
        self.default = default if default is not None else {"actions": ["noop"]}
        self.prompts: list[str] = []

    def analyze(self, log_path, prompt, is_first=False):
        self.prompts.append(prompt)
        if not self.replies:
            return self.default, {"output_tokens": 1}
        return self.replies.pop(0), {"output_tokens": 1}


@pytest.fixture(scope="module")
def problems():
    return rig.load_problems()


@pytest.fixture(scope="module")
def problem(problems):
    return next(p for p in problems if p["game"] == "n2ntd")


def _run(problem, agent, **kw):
    env = AutumnPlanningEnv()
    try:
        runner = AutumnRunner(env, agent, problem, log_path=kw.pop("log_path"),
                              alphabet=rig.actions_for(problem["game"]), **kw)
        return runner.run()
    finally:
        env.close()


# ------------------------------------------------------------------- the contract
def test_the_battery_is_the_published_one(problems):
    caps = [p["_eval_action_cap"] for p in problems]
    assert len(problems) == 86
    assert len({p["game"] for p in problems}) == 15
    assert (min(caps), max(caps), sum(caps)) == (2, 60, 1725)


def test_the_goal_frame_is_dropped_not_unrendered(problems):
    """Under `nl` the agent gets a sentence. A blank-but-present goal frame would be a
    different arm than the one `raw`/`icl`/`lmwm` ran."""
    for p in problems:
        assert p["_eval_presentation"] == "nl"
        assert p["goal_grid"] == ""
        assert p["_z_goal"] == ""
        assert p["nl_goal"]


def test_reset_reproduces_the_recorded_start(problem):
    env = AutumnPlanningEnv()
    try:
        obs = env.reset(task={"problem": problem})
        assert obs["grid"] == problem["start_grid"]
        assert obs["remaining"] == problem["_eval_action_cap"]
    finally:
        env.close()


def test_the_reference_plan_scores_under_the_online_rule(problem):
    plan = list(problem.get("_eval_oracle_plan") or problem["plan"])
    ok, at = rig.replay_and_score(problem, plan[:problem["_eval_action_cap"]])
    assert ok and at is not None


# ---------------------------------------------------------------- the study round
def test_an_empty_list_costs_no_actions(problem, tmp_path):
    """The mechanism the single phase needs: turn one is where megabytes of recorded
    transitions get read, and the smallest budget in the battery is two actions."""
    cap = problem["_eval_action_cap"]
    agent = Scripted({"actions": []}, {"actions": []})
    out = _run(problem, agent, log_path=tmp_path / "logs.txt", study_rounds=5)
    assert out["study_rounds_used"] == 2
    # the two study rounds cost nothing: the whole budget was still there to spend
    assert out["actions_used"] == cap
    assert [r["kind"] for r in out["rounds"]][:2] == ["study", "study"]


def test_study_rounds_are_bounded(problem, tmp_path):
    agent = Scripted(*[{"actions": []} for _ in range(10)], default={"actions": []})
    out = _run(problem, agent, log_path=tmp_path / "logs.txt", study_rounds=2)
    assert out["study_rounds_used"] == 2
    assert out["failed_reason"] == "study-rounds-exhausted"
    assert out["actions_used"] == 0


def test_a_missing_actions_file_is_a_retry_not_a_study_round(problem, tmp_path):
    """The distinction the whole mechanism turns on: `None` is a malformed response,
    `{"actions": []}` is a decision."""
    agent = Scripted(None)
    out = _run(problem, agent, log_path=tmp_path / "logs.txt", study_rounds=5)
    assert out["study_rounds_used"] == 0
    assert out["actions_used"] == problem["_eval_action_cap"]
    assert any("did not produce a usable" in x for x in agent.prompts)


def test_retries_are_bounded(problem, tmp_path):
    agent = Scripted(*[None] * 10, default=None)
    out = _run(problem, agent, log_path=tmp_path / "logs.txt", agent_retries=3)
    assert out["failed_reason"] == "no-plan"
    assert out["usage"]["calls"] == 3


# -------------------------------------------------------------------- the actions
def test_click_is_row_first_and_bounds_checked():
    dims, alphabet = (12, 12), {"noop", "click", "up"}
    actions, rejected = parse_actions_json(
        {"actions": ["click 3 4", "click 0 11", "click 12 0", "click 0 12"]},
        dims, alphabet, cap=10)
    assert actions == ["click 3 4", "click 0 11"]
    assert len(rejected) == 2


def test_actions_outside_this_worlds_alphabet_are_dropped():
    actions, rejected = parse_actions_json(
        {"actions": ["noop", "left", "up"]}, (12, 12), {"noop", "up"}, cap=10)
    assert actions == ["noop", "up"]
    assert any("left" in r for r in rejected)


def test_a_plan_cannot_exceed_the_remaining_budget():
    actions, rejected = parse_actions_json(
        {"actions": ["noop"] * 8}, (12, 12), {"noop"}, cap=3)
    assert actions == ["noop"] * 3
    assert len(rejected) == 5


def test_the_dict_action_form_is_accepted():
    """A model emitting structured JSON is different, not wrong; rejecting it would
    score prompt-following rather than planning."""
    actions, _ = parse_actions_json(
        {"actions": [{"action": "click", "row": 2, "col": 5}, {"action": "noop"}]},
        (12, 12), {"click", "noop"}, cap=10)
    assert actions == ["click 2 5", "noop"]


def test_an_over_budget_plan_stops_at_the_cap(problems, tmp_path):
    """The budget is the arm's, not the agent's to choose."""
    p = next(x for x in problems if x["_eval_action_cap"] == 2)
    agent = Scripted({"actions": ["noop"] * 20})
    out = _run(p, agent, log_path=tmp_path / "logs.txt")
    assert out["actions_used"] == 2

# ------------------------------------------------ parity with the published arms
#
# The agent arm is only worth reporting if it was measured the same way `raw`, `icl` and
# `lmwm` were. Three of the four things that has to mean are cheap to assert -- the same
# rows, the same budgets, the same eval configuration -- and the fourth, that our scorer
# is THEIR scorer, is provable rather than arguable: replay the baselines' own executed
# actions through `rig.replay_and_score` and require the verdicts to match.

REFERENCE_RUN = REPO / "logs/2026-09-05/planning_v2_online_icl_full"
PUBLISHED_ARMS = ("raw", "icl", "lmwm")

needs_reference = pytest.mark.skipif(
    not REFERENCE_RUN.is_dir(),
    reason="needs the published online run to compare against")


def _reference_rows():
    import glob
    rows = {}
    for f in sorted(glob.glob(str(REFERENCE_RUN / "*/online.json"))):
        for row in (json.loads(Path(f).read_text()).get("rows") or []):
            rows[row["task_uid"]] = row
    return rows


@needs_reference
def test_the_problem_set_and_budgets_are_the_published_ones(problems):
    ref = _reference_rows()
    ours = {p["task_uid"]: p["_eval_action_cap"] for p in problems}
    assert set(ours) == set(ref), "the agent arm would score a different problem set"
    assert [u for u in ours if ours[u] != ref[u]["action_cap"]] == []


@needs_reference
def test_the_eval_configuration_is_the_published_one():
    """Every knob that changes what a pass rate MEANS, read off the runs themselves."""
    import glob
    for f in glob.glob(str(REFERENCE_RUN / "*/online.json")):
        cfg = json.loads(Path(f).read_text())["config"]
        assert cfg["goal_presentation"] == rig.GOAL_PRESENTATION == "nl"
        assert cfg["cap_mode"] == rig.CAP_MODE == "per-problem"
        assert cfg["max_floor"] == rig.MAX_FLOOR == 0.95
        assert cfg["success_mode"] == "online-any-step"
        # one attempt per problem, so the baselines' pass_rate == pass_any and the
        # agent's single attempt is the same statistic, not a best-of-N against a mean
        assert cfg["attempts_planned"] == 1
        assert cfg["model"] == "deepseek/deepseek-v4-flash"


@needs_reference
def test_our_scorer_is_the_scorer_the_published_arms_were_scored_with(problems):
    """The one that cannot be argued from source: 258 real rollouts, three arms, and the
    verdict AND the step it was reached at must both agree.

    `replay_and_score` is a second implementation of a rule that already has a published
    answer, which is the one way this arm can be wrong without anyone noticing.
    """
    probs = {p["task_uid"]: p for p in problems}
    checked = 0
    mismatches = []
    for uid, row in _reference_rows().items():
        problem = probs.get(uid)
        if problem is None:
            continue
        for arm in PUBLISHED_ARMS:
            for attempt in ((row.get(arm) or {}).get("attempts") or []):
                executed = [r["executed"] for r in (attempt.get("rounds") or [])
                            if r.get("executed")]
                if not executed:
                    continue
                ok, at = rig.replay_and_score(problem, executed)
                checked += 1
                if ok != bool(attempt["success"]) or (ok and at != attempt.get("reached_at")):
                    mismatches.append(
                        f"{uid}/{arm}: published {attempt['success']}@"
                        f"{attempt.get('reached_at')} vs replay {ok}@{at}")
    assert checked > 200, f"only {checked} rollouts replayed; the comparison is thin"
    assert not mismatches, "\n".join(mismatches[:10])


@needs_reference
def test_the_agents_env_plays_the_same_branch_the_scorer_replays():
    """`max_episode_steps` is sized off the 4th `Branch` argument. The live env and the
    published rollout both pass the problem's cap; a scorer that passed anything else
    would be replaying in a differently-sized episode than the one being scored."""
    import inspect
    src = inspect.getsource(rig.replay_and_score)
    assert "_eval_action_cap" in src
