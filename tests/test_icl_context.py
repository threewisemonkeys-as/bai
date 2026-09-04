"""The `icl` planning arm: the raw planner handed the world model's own training data.

The arm only means anything if the block is (a) exactly the transitions the world model
was fit on and (b) spliced into the raw prompt without changing anything else. Both are
asserted here, plus the losslessness of the `diff` rendering -- a diff that dropped a
changed cell would quietly hand the baseline a wrong dynamics dataset.
"""
from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "offline_learning", REPO / "offline_learning/scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import icl_context  # noqa: E402
import eval_curated_plan as E  # noqa: E402

DATA_ROOT = icl_context.DEFAULT_DATA_ROOT
ARTIFACTS = REPO / "logs/2026-08-24/human_curated"
PROBLEMS = REPO / "logs/2026-08-29/planning_v2/problems.json"

pytestmark = pytest.mark.skipif(
    not (DATA_ROOT / "diffusion" / icl_context.DEFAULT_POOL).is_dir(),
    reason="human_data pools are a local 221M blob, regenerated from basis_data.zip",
)


def _games() -> list[str]:
    return sorted(p.name for p in DATA_ROOT.iterdir()
                  if (p / icl_context.DEFAULT_POOL).is_dir())


def test_icl_is_off_by_default_but_available():
    assert "icl" in E.ARMS and "icl" in E.LLM_ARMS
    assert "icl" not in E.DEFAULT_ARMS, "the arm adds 20-160k tokens/call; keep it opt-in"


def test_block_is_exactly_the_learners_training_batch():
    """Every pool must equal the --train-n the paired rexpure run recorded, or the arm
    would see data the world model never did."""
    for game in _games():
        trs = icl_context.load_pool_transitions(game)
        check = icl_context.assert_matches_launch(game, trs, ARTIFACTS)
        assert check["checked"], f"{game}: {check.get('reason')}"
        assert check["pool_n"] == check["train_n"] == len(trs)


def test_assert_matches_launch_rejects_a_superset(tmp_path):
    (tmp_path / "rexpure" / "x_s1").mkdir(parents=True)
    (tmp_path / "rexpure" / "x_s1" / "launch.json").write_text(
        json.dumps({"cmd": ["--train-n", "40"]}))
    with pytest.raises(RuntimeError, match="superset"):
        icl_context.assert_matches_launch("x", [None] * 60, tmp_path)


@pytest.mark.parametrize("game", ["diffusion", "n2ntd", "va6fq"])
def test_diff_rendering_is_lossless(game):
    """`(r, c): a -> b` lines must reconstruct the next state from the current one."""
    trs = icl_context.load_pool_transitions(game)
    for t in trs:
        lines = icl_context._diff_lines(t.x_t, t.x_t1)
        assert lines is not None, "grids parsed as JSON in the fixtures"
        grid = json.loads(t.x_t)
        for line in lines:
            cell, change = line.strip().split(": ", 1)
            r, c = (int(x) for x in cell.strip("()").split(", "))
            before, after = change.split(" -> ")
            assert grid[r][c] == before
            grid[r][c] = after
        assert grid == json.loads(t.x_t1)


def test_every_transition_appears_in_the_block():
    trs = icl_context.load_pool_transitions("diffusion")
    block = icl_context.render_block(trs, render="full")
    assert f"({len(trs)} recorded transitions" in block
    for t in trs:
        assert t.x_t in block and t.x_t1 in block and f"ACTION: {t.action}" in block


@pytest.mark.parametrize("presentation", ["frame", "nl"])
def test_icl_prompt_is_the_raw_prompt_plus_the_block(presentation):
    _meta, problems = E.load_eval_problems(PROBLEMS)
    rows = E.select_goal_presentation(
        [dict(p) for p in problems if p["game"] == "diffusion"][:2], presentation, "any")
    E.apply_action_caps(rows, "per-problem", E.PLAN_CAP)
    block, meta = E.load_icl_block("diffusion", ARTIFACTS,
                                   E.icl_config(Namespace(icl_render="diff")))
    assert meta["n_transitions"] == 60
    for p in rows:
        raw = E.build_prompt(p, "raw", E.gstr(p["start"]), cap=9)
        icl = E.build_prompt(p, "icl", E.gstr(p["start"]), cap=9, icl_block=block)
        # nothing but the block differs: same goal, same transcript, same budget
        assert icl.replace("\n" + block.strip("\n") + "\n", "", 1) == raw
        # and it sits ahead of the per-round transcript, so it is a cacheable prefix
        assert icl.index(block.strip("\n")) < icl.index("STATE[t]")


def test_icl_without_a_block_fails_loudly():
    _meta, problems = E.load_eval_problems(PROBLEMS)
    rows = E.select_goal_presentation(
        [dict(p) for p in problems if p["game"] == "diffusion"][:1], "nl", "any")
    E.apply_action_caps(rows, "per-problem", E.PLAN_CAP)
    with pytest.raises(ValueError, match="non-empty"):
        E.build_prompt(rows[0], "icl", E.gstr(rows[0]["start"]), cap=9)
