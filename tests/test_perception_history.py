"""--perception-history: perceive() gets recent observations, not only the current one.

perceive() has always taken a list, but training called it with a one-element list, so
code in a learned module that reads earlier frames never ran. These tests pin the three
things that make the flag safe to use: N = 1 is exactly the old behaviour, every history
is causal (a state is never perceived with a later one -- and a contrastive option never
with the true next frame), and the planning loop calls the shipped module with the
history length it was trained with.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "offline_learning"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import invdyn_core as C  # noqa: E402
import rexpure_optimize as R  # noqa: E402
from offline_learning.scripts import eval_coverage_online as eco  # noqa: E402
from offline_learning.scripts import eval_curated_online as eco_v2  # noqa: E402

# perceive() that reports exactly what it was given
ECHO = "def perceive(h):\n    return '|'.join(h)\n"


def _window_tr(k=9):
    """A transition with a full K window: frames f0..f(2K) in time order, X_t = f(K)."""
    prev = [(f"f{i}", f"a{i}") for i in range(k)]
    nxt = [(f"a{i}", f"f{i}") for i in range(k + 2, 2 * k + 1)]
    return C.Transition(x_t=f"f{k}", x_t1=f"f{k + 1}", action=f"a{k}",
                        ctx_prev=prev, ctx_next=nxt)


def _ids(z):
    return [int(f[1:]) for f in z.split("|")]


# ------------------------------------------------------------------ run_perceive
def test_run_perceive_wraps_a_single_observation():
    assert C.run_perceive(ECHO, "obs") == ("obs", None)


def test_run_perceive_passes_a_history_oldest_first_and_copies_it():
    frames = ["a", "b", "c"]
    mutate = "def perceive(h):\n    h.clear()\n    return 'x'\n"
    assert C.run_perceive(ECHO, frames) == ("a|b|c", None)
    C.run_perceive(mutate, frames)
    assert frames == ["a", "b", "c"]


# ------------------------------------------------------------------ build_window
def test_history_one_is_the_single_frame_window():
    win, err = C.build_window(ECHO, _window_tr())
    assert err is None
    assert [z for z, _ in win["prev"]] == [f"f{i}" for i in range(9)]
    assert (win["z_t"], win["z_t1"]) == ("f9", "f10")
    assert [z for _, z in win["nxt"]] == [f"f{i}" for i in range(11, 19)]


@pytest.mark.parametrize("n", [2, 10, 19, 50])
def test_every_window_state_gets_a_causal_capped_history(n):
    win, _ = C.build_window(ECHO, _window_tr(), history=n)
    feats = ([z for z, _ in win["prev"]] + [win["z_t"], win["z_t1"]]
             + [z for _, z in win["nxt"]])
    for i, z in enumerate(feats):
        assert _ids(z) == list(range(max(0, i - n + 1), i + 1))


def test_the_actions_stay_aligned_with_their_states():
    tr = _window_tr()
    win, _ = C.build_window(ECHO, tr, history=10)
    assert [a for _, a in win["prev"]] == [a for _, a in tr.ctx_prev]
    assert [a for a, _ in win["nxt"]] == [a for a, _ in tr.ctx_next]
    assert win["z_t"].endswith("f9") and win["z_t1"].endswith("f10")


def test_a_cut_window_still_perceives_every_state():
    """At an episode edge ctx_prev is short; the history is simply shorter."""
    tr = C.Transition(x_t="f0", x_t1="f1", action="a0", ctx_prev=[],
                      ctx_next=[("a1", "f2")])
    win, _ = C.build_window(ECHO, tr, history=10)
    assert (win["z_t"], win["z_t1"], win["nxt"][0][1]) == ("f0", "f0|f1", "f0|f1|f2")


def test_center_histories_match_what_the_window_perceives():
    tr = _window_tr()
    win, _ = C.build_window(ECHO, tr, history=4)
    h_t, h_t1 = C.center_histories(tr, 4)
    assert "|".join(h_t) == win["z_t"] and "|".join(h_t1) == win["z_t1"]


# --------------------------------------------------------------- contrastive options
@pytest.mark.parametrize("n", [1, 2, 10])
def test_option_history_ends_at_x_t_then_the_option(n):
    tr = _window_tr()
    hist = C.option_history(tr, "decoy", n)
    assert hist[-1] == "decoy" and len(hist) == min(n, 10)
    assert tr.x_t1 not in hist
    if n > 1:
        assert hist[-2] == tr.x_t


def test_contrastive_option_renderings_are_cached_per_history():
    """Two transitions offering the same decoy frame after different pasts must not
    share a rendering once P can see the past."""
    tr_a = C.Transition(x_t="A", x_t1="B", action="a", ctx_prev=[("P", "a")])
    tr_b = C.Transition(x_t="Q", x_t1="R", action="a", ctx_prev=[("P", "a")])
    assert C.option_history(tr_a, "D", 3) != C.option_history(tr_b, "D", 3)
    assert C.option_history(tr_a, "D", 1) == C.option_history(tr_b, "D", 1) == ["D"]


# ------------------------------------------------------------------ guards
def test_history_needs_a_window():
    with pytest.raises(ValueError, match="context_k"):
        C.InvDynAdapter(cfg=None, action_pool=["a"], context_k=0, perception_history=2)
    with pytest.raises(ValueError, match=">= 1"):
        C.InvDynAdapter(cfg=None, action_pool=["a"], context_k=3, perception_history=0)
    C.InvDynAdapter(cfg=None, action_pool=["a"], context_k=3, perception_history=4)


@pytest.mark.parametrize("argv,msg", [
    (["--perception-history", "2", "--context-k", "0"], "--context-k >= 1"),
    (["--perception-history", "0"], "must be >= 1"),
])
def test_rejected_history_flags(argv, msg, tmp_path, capsys):
    parser = R.build_parser()
    args = parser.parse_args(["--run", str(tmp_path)] + argv)
    with pytest.raises(SystemExit):
        R.validate_args(parser, args)
    assert msg in capsys.readouterr().err


def test_gate_inputs_are_the_old_frames_at_history_one():
    train = [{"tr": _window_tr()}]
    assert R._gate_inputs(train, 1) == ["f9", "f10"]
    assert R._gate_inputs(train, 3) == [("f7", "f8", "f9"), ("f8", "f9", "f10")]


# ------------------------------------------------------------------ proposer prompt
def test_history_one_leaves_the_perception_template_untouched():
    assert (C.build_reflection_templates("autumn", perception_history=1)
            == C.build_reflection_templates("autumn"))


@pytest.mark.parametrize("no_id", [False, True])
def test_history_is_described_to_the_proposer(no_id):
    text = C.build_reflection_templates("autumn", no_id=no_id,
                                        perception_history=10)["perception"]
    assert C._P_CONTRACT not in text
    assert "up to 10 of them" in text and "OLDEST FIRST" in text


# ------------------------------------------------------------------ planning
def test_planning_perceive_keeps_the_trained_history_length():
    seen = eco.compile_perceive(ECHO, history=3)
    assert seen("g") == ("g", None)
    assert seen(["g0", "g1", "g2", "g3", "g4"]) == ("g2|g3|g4", None)
    assert eco.compile_perceive(ECHO)(["g0", "g1"]) == ("g1", None)


def test_trained_history_is_read_from_the_run_summary(tmp_path):
    assert eco_v2.trained_perception_history(tmp_path) == 1
    s = tmp_path / "test_summary_rexpure_seed1.json"
    s.write_text(json.dumps({"inverse_accuracy": 0.5}))
    assert eco_v2.trained_perception_history(tmp_path) == 1   # predates the flag
    s.write_text(json.dumps({"perception_history": 10}))
    assert eco_v2.trained_perception_history(tmp_path) == 10
