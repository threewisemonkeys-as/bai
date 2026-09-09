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
from research.autumn import launch  # noqa: E402

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


# ------------------------------------------------- the world-model condition (F22)
#
# The `agent+wm` arm hands each session the artifacts the `lmwm` arm plans with. Two
# things have to hold for that to mean anything: the session must actually receive the
# bytes the published arm was scored with, and the arm WITHOUT the flag must be the run
# already on disk -- an accidental prompt change would silently make the comparison a
# comparison of two different agents.

NLWM_ROOT = REPO / "logs/2026-08-24/human_curated"

needs_artifacts = pytest.mark.skipif(
    not (NLWM_ROOT / "rexpure").is_dir(), reason="needs the seed-1 rexpure artifacts")


@needs_artifacts
def test_every_world_has_the_artifacts_the_lmwm_arm_used(problems):
    launch.check_world_model(NLWM_ROOT, problems)      # raises SystemExit if not


@needs_artifacts
def test_the_world_model_lands_in_the_workspace(problem, tmp_path):
    ws = launch.build_workspace(problem, REPO / "corpora", tmp_path,
                                study_rounds=5, nlwm_root=NLWM_ROOT)
    src = launch.nlwm_dir(NLWM_ROOT, problem["game"])
    for dst, source in launch.NLWM_FILES.items():
        assert (ws / "nlwm" / dst).read_bytes() == (src / source).read_bytes()
    assert (ws / "nlwm" / "README.md").is_file()
    assert "nlwm/" in (ws / "AGENTS.md").read_text()
    assert (ws / "drives").is_dir(), "the world model does not replace the evidence"


def test_without_the_flag_the_prompt_is_the_one_the_run_on_disk_used(problem, tmp_path):
    ws = launch.build_workspace(problem, REPO / "corpora", tmp_path, study_rounds=5)
    assert not (ws / "nlwm").exists()
    text = (ws / "AGENTS.md").read_text()
    assert "nlwm" not in text
    shipped = REPO / ("logs/2026-09-06/agent_full/N2NTD/"
                      + problem["task_uid"].replace(":", "_") + "/AGENTS.md")
    if shipped.is_file():                       # the completed arm, byte for byte
        assert text == shipped.read_text()


@needs_artifacts
def test_every_perception_module_runs_on_its_own_start_grids(problems):
    """`compile_perceive` swallows exceptions and returns ("", err); an agent calling
    the module directly sees the traceback instead. So the modules are checked here
    against every start state they will be handed."""
    by_game: dict[str, list] = {}
    for p in problems:
        by_game.setdefault(p["game"], []).append(p)
    for game, rows in by_game.items():
        code = (launch.nlwm_dir(NLWM_ROOT, game)
                / launch.NLWM_FILES["perception.py"]).read_text()
        ns: dict = {}
        exec(code, ns)                                              # noqa: S102
        perceive = ns.get("perceive")
        assert callable(perceive), f"{game}: no perceive()"
        for p in rows:
            out = perceive([p["start_grid"]])
            assert isinstance(out, str) and out.strip(), f"{game}/{p['task_uid']}: empty"


# ------------------------------------------------------------- parallel scheduling
#
# Parallelism is a scheduling change only -- the sessions were independent already. The
# one thing it adds that can corrupt a result is two workers playing the same problem
# and both writing a row, so the claim is tested rather than assumed.

def test_two_workers_cannot_take_the_same_problem(tmp_path):
    claims = tmp_path / "claims"
    claims.mkdir()
    assert launch.claim(claims, "n2ntd:platform:s0") is True
    assert launch.claim(claims, "n2ntd:platform:s0") is False
    assert launch.claim(claims, "n2ntd:high-ground:s0") is True


def test_merging_the_workers_rows_keeps_one_row_per_problem(tmp_path):
    (tmp_path / "rows.w0.jsonl").write_text(
        '{"task_uid": "a"}\n{"task_uid": "b"}\n')
    (tmp_path / "rows.w1.jsonl").write_text(
        '{"task_uid": "b"}\n{"task_uid": "c"}\nnot json\n')
    assert launch.merge_rows(tmp_path) == 3
    uids = [json.loads(l)["task_uid"]
            for l in (tmp_path / "rows.jsonl").read_text().splitlines()]
    assert sorted(uids) == ["a", "b", "c"]
    assert launch.recorded(tmp_path) == {"a", "b", "c"}


def test_the_long_problems_are_handed_out_first(problems, tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    slow, fast = problems[-1], problems[0]
    for p, wall in ((slow, 9000.0), (fast, 10.0)):
        (traces / (p["task_uid"].replace(":", "_") + ".json")).write_text(
            json.dumps({"task_uid": p["task_uid"], "wall_s": wall}))
    order = launch.order_longest_first(list(problems), str(tmp_path))
    assert order[0]["task_uid"] == slow["task_uid"]
    assert order[-1]["task_uid"] == fast["task_uid"]
    assert len(order) == len(problems), "scheduling must not drop a problem"


def test_each_worker_gets_its_own_proxy_and_reasoning_log():
    """A turn's reasoning is read back by byte offset into the proxy transcript, so two
    sessions sharing one transcript cross-contaminate every turn. The supervisor must
    therefore hand each worker a proxy of its own."""
    class A:
        proxy_port_base = 8790
        proxy_log_root = "logs/parity_proxy"
        base_url = "http://127.0.0.1:8788/v1"
        transcript = "logs/parity_proxy/reasoning.jsonl"

    routes = [launch.worker_routing(A, i) for i in range(6)]
    urls = [r[r.index("--base-url") + 1] for r in routes]
    logs = [r[r.index("--transcript") + 1] for r in routes]
    assert len(set(urls)) == 6 and len(set(logs)) == 6
    assert urls[0].endswith(":8790/v1") and urls[5].endswith(":8795/v1")
    assert logs[3] == "logs/parity_proxy/w3/reasoning.jsonl"

    A.proxy_port_base = 0                       # unset: the single-session default
    assert launch.worker_routing(A, 2) == [
        "--base-url", A.base_url, "--transcript", A.transcript]
