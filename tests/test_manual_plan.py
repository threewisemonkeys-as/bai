"""Screens for manually curated Autumn planning problems.

Real engine calls (~ms each at 0.6 ms/step), no mocking: the point of the audit is that it
re-derives everything from a cold interpreter, so mocking it would test nothing.
"""

import pytest

from offline_learning.manual_plan import audit as A
from offline_learning.manual_plan import problems as P
from offline_learning.manual_plan import session as S


def _problem(game, seed, prefix, plan):
    grids = S.replay(game, seed, prefix + plan)
    return {"game": game, "seed": seed, "prefix": prefix, "gt_actions": plan,
            "start_grid": grids[len(prefix)], "goal_grid": grids[-1]}


def test_aliases_and_registry():
    assert S.canon("ice") == "bt3gb"
    assert S.canon("ANTS") == "s2kt7"
    assert S.canon("SET") == "SET"
    assert S.canon("paint") == "eahcw"
    assert {g["game"] for g in S.games()} == {
        "eahcw", "egg", "bt3gb", "dq8gc", "7xf97", "n2ntd", "va6fq", "s2kt7",
        "colour_lines", "SET", "diffusion", "dino", "f5w3n", "logic_gates", "7www9",
    }
    assert S.info("mario")["tick_locked"] is True       # n2ntd's enemy patrol
    with pytest.raises(KeyError):
        S.canon("not_a_game")


def test_replay_is_deterministic_and_indexed():
    a = ["right", "right", "down", "noop"]
    g1 = S.replay("bt3gb", 3, a)
    g2 = S.replay("bt3gb", 3, a, use_cache=False)
    assert g1 == g2
    assert len(g1) == len(a) + 1                        # index 0 is the post-reset frame
    assert S.exec_plan("bt3gb", 3, a[:2], a[2:]) == g1[3:]


def test_wellformed_problem_passes_every_screen():
    r = A.audit(_problem("bt3gb", 3, ["right"] * 3, ["down", "noop", "noop", "click 5 5"]))
    assert r["ok"], {k: v["detail"] for k, v in r["screens"].items() if not v["ok"]}


def test_goal_that_is_the_start_fails_m2():
    p = _problem("bt3gb", 3, ["right"] * 3, ["noop"])
    p["goal_grid"] = p["start_grid"]
    r = A.audit(p, screens=["m2_changed"])
    assert not r["screens"]["m2_changed"]["ok"]


def test_all_noop_plan_fails_m3():
    r = A.audit(_problem("bt3gb", 3, ["right"] * 3, ["noop", "noop"]),
                screens=["m3_noop_fails"])
    assert not r["screens"]["m3_noop_fails"]["ok"]


def test_padding_action_is_caught_by_m4():
    """bt3gb has no `on up` rule, so an `up` in the plan is a noop wearing a verb -- the
    exact kind of stray action the curator is trimming. Substituting noop must reach the
    same frame, and M4 must name the index."""
    r = A.audit(_problem("bt3gb", 3, [], ["up", "down"]), screens=["m4_actions_matter"])
    assert not r["screens"]["m4_actions_matter"]["ok"]
    assert r["screens"]["m4_actions_matter"]["dead_indices"] == [0]


def test_edited_plan_that_misses_the_pinned_goal_fails_m1_only():
    """Stripping noops is exactly the edit the curator makes; under exact frames it breaks
    the goal whenever the world ticks on, and M1 -- not M7 -- is what must say so."""
    p = _problem("bt3gb", 3, ["right"] * 3, ["down", "noop", "noop", "click 5 5"])
    p["gt_actions"] = [a for a in p["gt_actions"] if a != "noop"]
    r = A.audit(p)
    failed = [k for k, v in r["screens"].items() if not v["ok"]]
    assert failed == ["m1_reaches"]


def test_tick_locked_game_flags_m5_as_vacuous():
    r = A.audit(_problem("n2ntd", 1, ["right"] * 3, ["up", "right", "right"]),
                screens=["m5_no_shorter"])
    assert r["screens"]["m5_no_shorter"]["vacuous"] is True
    assert any("vacuous" in w for w in r["warnings"])


def test_problem_record_is_self_contained(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "DATA", tmp_path)
    monkeypatch.setattr(P, "DATA", tmp_path)
    p = P.build("ice", 3, ["right"] * 3, ["down", "click 5 5"], note="unit")
    p["audit"] = A.audit(p)
    ps, pid = P.upsert("ice", p)
    assert pid == "bt3gb-001" and len(ps) == 1
    # the cached frames must be reproducible from (seed, prefix, plan) alone
    grids = S.replay(p["game"], p["seed"], p["prefix"] + p["gt_actions"])
    assert p["start_grid"] == grids[len(p["prefix"])] and p["goal_grid"] == grids[-1]
    out = tmp_path / "export.json"
    assert P.export(["ice"], out)["n"] == (1 if p["audit"]["ok"] else 0)
    assert P.delete("ice", pid) == []
