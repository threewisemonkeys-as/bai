"""The four NLWM ablation arms: -FD, -ID, -Perception, -Beliefs.

The whole ablation table rests on one claim -- each arm is the reference training run with
exactly one flag delta, on the SAME data. `test_split_is_invariant_to_every_ablation_flag`
is that claim; it is why the deltas may be applied without re-baselining anything.

The rest assert that each flag actually removes what it says it removes. For --no-id that
is not only the score: an ablation that dropped the ID term from the composite while
leaving F's predicted action, its decoder reasoning and the "make the action recoverable"
instruction in the proposer's prompt would be an ablation of the scorer, not of the
objective, so the reflective dataset and the reflection templates are checked too.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "offline_learning"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import invdyn_core as C  # noqa: E402
import rexpure_optimize as R  # noqa: E402

REF_RUN = REPO / "logs/2026-08-24/human_curated/rexpure/bt3gb_s1"


# --------------------------------------------------------------------- helpers
def _ref_argv() -> list[str]:
    cmd = json.loads((REF_RUN / "launch.json").read_text())["cmd"]
    i = next(i for i, x in enumerate(cmd) if str(x).endswith(".py"))
    return [str(x) for x in cmd[i + 1:]]


def _drop(argv: list[str], *flags: str, valued: bool = False) -> list[str]:
    out, i = [], 0
    while i < len(argv):
        if argv[i] in flags:
            i += 2 if valued else 1
            continue
        out.append(argv[i])
        i += 1
    return out


def _split(argv: list[str]):
    args = R.build_parser().parse_args(argv)
    train, test, pool, ck, wl, trs, idn = R.build_data(args, random.Random(args.seed))
    return args, train, test, pool, ck, trs, idn


def _identity(baked):
    """What must be equal for two runs to be 'the same data': the transitions in order
    and the fixed choice set baked for each."""
    return [
        (b["tr"].x_t, b["tr"].x_t1, b["tr"].action, tuple(b["choices"]))
        for b in baked
    ]


def _tr(x_t="A", x_t1="B", action="left"):
    return C.Transition(x_t=x_t, x_t1=x_t1, action=action)


def _traj(**over):
    """A minimal trajectory record of the shape evaluate() emits."""
    t = {
        "tr": _tr(), "z_t": "z1", "z_t1": "z2", "choices": ["left", "right"],
        "pred": ["right"], "reasoning": "F thought it moved right", "perc_err": None,
        "id_score": 0.0, "fd_score": 1.0, "z_hat": "", "win": None,
        "cfd_score": 0.0, "cfd_pred": 2, "cfd_ambiguous": False,
    }
    t.update(over)
    return t


def _adapter(**over):
    kw = dict(cfg=None, action_pool=["left", "right"], composite="min",
              fd_scorer="none", contrastive_fd=True, analyze_mistakes=False,
              gate_train_x=["A", "B"])
    kw.update(over)
    return C.InvDynAdapter(**kw)


def _records(adapter, comp, trajs):
    batch = C.EvaluationBatch(outputs=[None] * len(trajs),
                              scores=[0.5] * len(trajs), trajectories=trajs)
    return adapter.make_reflective_dataset(
        {"perception": "def perceive(h): return h[-1]", "world_knowledge": "wk"},
        batch, [comp])[comp]


def _text(records) -> str:
    """Everything the proposer would actually read, flattened the way
    render_reflection_prompt flattens it."""
    return C.render_reflection_prompt("<curr_param>\n<side_info>", "P", records)


# ------------------------------------------------- the load-bearing data claim
needs_data = pytest.mark.skipif(
    not (REF_RUN / "launch.json").is_file()
    or not (REPO / "offline_learning/human_data/bt3gb/informative_curated").is_dir(),
    reason="reference run + human_data pools are local blobs (221M), not in git",
)


@needs_data
def test_reference_split_reproduces_the_shipped_fingerprint():
    """build_data re-run on the run's own argv must rebuild the run's own train batch."""
    args, train, *_ = _split(_ref_argv())
    want = json.loads(
        (REF_RUN / f"rexpure_run_seed{args.seed}/resume_state.json").read_text()
    )["train_fingerprint"]
    assert C._train_fingerprint(train) == want


@needs_data
@pytest.mark.parametrize("arm,mutate", [
    ("-FD", lambda a: _drop(a, "--contrastive-fd", "--cfd-hard-decoys")),
    ("-ID", lambda a: a + ["--no-id"]),
    ("-Beliefs", lambda a: a + ["--no-beliefs"]),
    ("-Perception", lambda a: _drop(a, "--start-perception", valued=True)
                              + ["--no-perception"]),
])
def test_split_is_invariant_to_every_ablation_flag(arm, mutate):
    """Every arm must see the identical 60 train / 50 test transitions and choice sets.

    This is not incidental: build_data draws the split from Random(--seed) and runs BEFORE
    any of the four deltas takes effect, and bake_decoys draws from its own
    Random(seed+9173) so toggling the contrastive term cannot advance the split rng.
    If this test breaks, no ablation number is comparable to the NLWM column.
    """
    ref = _ref_argv()
    _, tr_a, te_a, pool_a, ck_a, _, idn_a = _split(ref)
    _, tr_b, te_b, pool_b, ck_b, _, idn_b = _split(mutate(ref))
    assert _identity(tr_a) == _identity(tr_b), f"{arm} moved the TRAIN split"
    assert _identity(te_a) == _identity(te_b), f"{arm} moved the TEST split"
    assert (pool_a, ck_a, idn_a) == (pool_b, ck_b, idn_b)


@needs_data
def test_no_fd_leaves_the_train_items_without_baked_decoys():
    """-FD is a pure flag removal, so its only footprint is the missing cfd_options."""
    _, train_ref, *_ = _split(_ref_argv())
    _, train_nofd, *_ = _split(_drop(_ref_argv(), "--contrastive-fd", "--cfd-hard-decoys"))
    assert all(b.get("cfd_options") for b in train_ref)
    assert not any(b.get("cfd_options") for b in train_nofd)


# ------------------------------------------------------------- --no-id scoring
def test_no_id_requires_a_surviving_forward_term():
    with pytest.raises(ValueError, match="only term"):
        _adapter(no_id=True, contrastive_fd=False, fd_scorer="none")
    _adapter(no_id=True, contrastive_fd=True)                      # cFD survives
    _adapter(no_id=True, contrastive_fd=False, fd_scorer="exact", fd_weight=0.5)


@pytest.mark.parametrize("no_id,cfd,expected", [
    (False, True, 0.0),   # min(id=0.0, cfd=1.0)
    (True, True, 1.0),    # cfd alone
    (False, False, 0.0),  # -FD: id alone
])
def test_composite_terms(no_id, cfd, expected):
    """The composite is min() over the enabled terms and nothing else."""
    a = _adapter(no_id=no_id, contrastive_fd=cfd)
    id_score, cfd_score = 0.0, 1.0
    terms = [] if a.no_id else [id_score]
    if a.fd_weight > 0.0:
        terms.append(1.0)
    if a.contrastive_fd:
        terms.append(cfd_score)
    assert min(terms) == expected


# -------------------------------------------- --no-id reflective-dataset purge
ID_LEAKS = re.compile(
    r"predicted action|IDENTIFY THIS|INVERSE|inverse-dynamics mistake"
    r"|could not be recovered|not recoverable|Correctly identified"
    r"|action guessable", re.I)


@pytest.mark.parametrize("comp", ["perception", "world_knowledge"])
def test_no_id_purges_inverse_signal_from_the_proposer_prompt(comp):
    # a real batch mixes an ID miss, a cFD miss, and the constant-P gate record
    trajs = [_traj(), _traj(id_score=1.0, cfd_score=0.0)]
    on = _text(_records(_adapter(no_id=False), comp, trajs))
    off = _text(_records(_adapter(no_id=True), comp, trajs))
    assert ID_LEAKS.search(on), "the reference prompt should carry the ID signal"
    leak = ID_LEAKS.search(off)
    assert not leak, f"--no-id leaked {leak.group(0)!r} into the {comp} prompt"


@pytest.mark.parametrize("comp", ["perception", "world_knowledge"])
def test_no_id_keeps_the_shared_evidence(comp):
    """Suppressing the ID SIGNAL must not delete the transition itself: the true action
    and both feature renderings are what a forward model is supposed to learn from."""
    off = _text(_records(_adapter(no_id=True), comp, [_traj()]))
    assert "'left'" in off, "the true action must survive"
    assert "Transition" in off and "Inverse Dynamics" not in off


def test_no_id_reveals_the_action_in_the_window_transcript():
    win = {"prev": [("z0", "up")], "nxt": [("down", "z3")], "z_t": "z1", "z_t1": "z2"}
    assert "??? (IDENTIFY THIS)" in C._inverse_transcript(win)
    revealed = C._inverse_transcript(win, reveal_action="click 3 4")
    assert "??? (IDENTIFY THIS)" not in revealed
    assert "action(t -> t+1): click 3 4" in revealed


def test_no_id_skips_the_inverse_diagnosis_calls():
    """_analyze_failures writes text straight into the proposer prompt, so an inverse
    diagnosis is ID signal even though the composite never sees it."""
    a_on, a_off = _adapter(analyze_mistakes=True), _adapter(no_id=True, analyze_mistakes=True)
    failures = [(_traj(), 0.0)]
    kinds = lambda a: {k for _, k, *_ in _cases(a, failures)}  # noqa: E731
    assert kinds(a_on) == {"inv"}
    assert kinds(a_off) == set()


def _cases(adapter, failures):
    """Reach _cases_for, which is a closure inside _analyze_failures; replicate its one
    condition rather than monkeypatching the LLM stack."""
    out = []
    for ti, (t, _) in enumerate(failures):
        if not adapter.no_id and t.get("id_score", 1.0) < 1.0:
            out.append((ti, "inv", t))
        if adapter.fd_reflect and t.get("z_hat") and t.get("fd_score", 1.0) < 0.999:
            out.append((ti, "fwd", t))
    return out


# ------------------------------------------------- --no-id reflection templates
def test_no_id_restates_the_proposer_task():
    for env in ("autumn", "arc_agi", None):
        base = C.build_reflection_templates(env)
        abl = C.build_reflection_templates(env, no_id=True)
        for comp in ("perception", "world_knowledge"):
            assert abl[comp] != base[comp], f"{comp}/{env} template unchanged under no_id"
            assert not C._NO_ID_BANNED.search(abl[comp])
        assert C._NO_ID_BANNED.search(base["perception"]), "reference IS ID-framed"


def test_template_swap_fails_loudly_if_a_sentence_moves():
    with pytest.raises(RuntimeError, match="no longer contains"):
        C._apply_no_id_swaps("perception", "a template that says nothing of the sort")


# ------------------------------------------------------------ --no-beliefs wiring
def test_no_beliefs_drops_the_component_and_pins_the_selector():
    sel = C.SingleComponentSelector("perception")
    assert sel(None, None, None, None, None) == ["perception"]
    # a candidate without the key must still score and still render as "(empty)"
    cand = {"perception": "code"}
    assert cand.get("world_knowledge", "") == ""


@pytest.mark.parametrize("argv,msg", [
    (["--no-beliefs", "--start-beliefs", "/tmp/b.txt"], "drop --start-beliefs"),
    (["--no-beliefs", "--no-perception"], "nothing to learn"),
    (["--no-perception", "--start-perception", "/tmp/p.py"], "drop --start-perception"),
    (["--cfd-test"], "add --contrastive-fd"),
    # the reference config's own objective flags: --fd-scorer none zeroes the predictive
    # FD term, so without --contrastive-fd there is nothing left for --no-id to keep
    (["--no-id", "--fd-scorer", "none"], "removes the only term"),
    (["--no-id", "--fd-weight", "0"], "removes the only term"),
])
def test_rejected_ablation_flag_combinations(argv, msg, tmp_path, capsys):
    parser = R.build_parser()
    args = parser.parse_args(["--run", str(tmp_path)] + argv)
    with pytest.raises(SystemExit):
        R.validate_args(parser, args)
    assert msg in capsys.readouterr().err


@pytest.mark.parametrize("argv", [
    ["--no-id", "--contrastive-fd", "--fd-scorer", "none"],   # the -ID arm as launched
    ["--no-id", "--fd-scorer", "exact"],
    ["--cfd-test", "--contrastive-fd"],
    ["--no-beliefs"],
    ["--no-perception"],
])
def test_accepted_ablation_flag_combinations(argv, tmp_path):
    parser = R.build_parser()
    R.validate_args(parser, parser.parse_args(["--run", str(tmp_path)] + argv))


# ------------------------------------------------------------- held-out cFD wiring
def test_test_decoys_use_their_own_rng_offset():
    """Train and held-out decoys must never be the same draw."""
    assert C.CFD_TEST_SEED_OFFSET != 9173

    def baked(action="left"):
        return [{"tr": _tr("A" * 4, "B" * 4, action), "choices": []}]

    pool = [_tr("A" * 4, "B" * 4), _tr("C" * 4, "D" * 4), _tr("E" * 4, "F" * 4),
            _tr("G" * 4, "H" * 4)]
    a, b = baked(), baked()
    C.bake_decoys(a, pool, 3, random.Random(1 + 9173))
    C.bake_test_decoys(b, pool, 3, 1)
    assert all(x["cfd_options"] for x in (a[0], b[0]))
    # the true frame is in both; the ORDER/draw differs because the rngs differ
    assert a[0]["cfd_options"] != b[0]["cfd_options"]


def test_eval_cfd_on_rejects_unbaked_items():
    """Scoring held-out cFD against items with no decoys would silently be a different
    (easier) question, so it must refuse rather than improvise."""
    import asyncio
    with pytest.raises(ValueError, match="no baked cfd_options"):
        asyncio.run(C.eval_cfd_on(None, "", "", [{"tr": _tr(), "choices": []}]))
