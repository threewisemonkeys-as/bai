"""The agent arm's corpus: the same data as `icl`, with none of the answer key.

Two properties make the `agent` column mean something, and neither is self-evident from
reading the exporter:

* it is the SAME data the `icl` arm reads and the world model was fit on -- otherwise
  `agent - icl` measures the corpus, not the shell;
* it carries none of the metadata that names the world or summarises its dynamics --
  otherwise the agent is answering an easier question than the arms it is compared to.

The first is checked by identity against `icl_context`, not by re-deriving the pool.
The second is checked by planting each leak and requiring the export to fail: a guard
that has never rejected anything is not known to reject anything.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "offline_learning", REPO / "offline_learning/scripts",
           REPO / "cc_autumn/autumn-code/rig"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import icl_context  # noqa: E402
import export_agent_corpus as X  # noqa: E402

DATA_ROOT = icl_context.DEFAULT_DATA_ROOT
GAME = "bt3gb"

pytestmark = pytest.mark.skipif(
    not (DATA_ROOT / GAME / icl_context.DEFAULT_POOL).is_dir(),
    reason="human_data pools are a local blob, regenerated from basis_data.zip",
)


@pytest.fixture(scope="module")
def exported(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("corpus")
    X.export_game(GAME, out)
    return out / X.LABELS[GAME]


def _records(root: Path) -> list[dict]:
    return [json.loads(p.read_text())
            for p in sorted((root / "drives").glob("t*.json"))]


def test_export_is_exactly_the_icl_arms_transitions(exported):
    """Byte-level identity with the pool `icl_context` hands the ICL arm."""
    want = icl_context.load_pool_transitions(GAME)
    got = _records(exported)
    assert len(got) == len(want)
    for record, transition in zip(got, want):
        assert record["state"] == transition.x_t
        assert record["next_state"] == transition.x_t1
        assert record["action"] == transition.action
        assert record["context"] == [[s, a] for s, a in transition.ctx_prev]


def test_the_export_is_the_train_split_and_nothing_else(exported):
    """Leak 5: `test_d*` is the learner's held-out set and must not be exported.

    Stated as set equality against `train_d*`, not as disjointness from `test_d*`.
    Disjointness is the wrong assertion and fails on this data: boards recur, and one
    of bt3gb's 50 held-out triples is byte-identical to a train triple (4 of its 45
    distinct held-out states appear in train too). That overlap is a property of the
    pool as the learner was fit on it -- `lmwm` is scored against the same one -- so
    the exporter's job is to reproduce the train split exactly, not to improve it.
    """
    from validate import load_transitions, strip_autumn_obs_metadata

    root = DATA_ROOT / GAME / icl_context.DEFAULT_POOL

    def triples(dirs):
        out = set()
        for d in dirs:
            for t in load_transitions([d], None, context_k=0):
                out.add((strip_autumn_obs_metadata(t.x_t), t.action,
                         strip_autumn_obs_metadata(t.x_t1)))
        return out

    train = triples(sorted(root.glob("train_d*")))
    held_out = triples(sorted(root.glob("test_d*")))
    assert held_out, "expected a held-out split to exist for this check to mean anything"

    got = {(r["state"], r["action"], r["next_state"]) for r in _records(exported)}
    assert got == train
    # nothing held out sneaks in beyond what the train split itself already contains
    assert got & held_out == train & held_out


def test_the_workspace_directory_is_an_opaque_code(tmp_path):
    """The directory name must not be a hint.

    Checked on an English-named world: nine of the fifteen are already opaque codes
    whose label is just the uppercase name (`bt3gb` -> `BT3GB`), so asserting on one of
    those tests nothing. `egg` -> `K3QP2` is where the mapping does work.
    """
    english = "egg"
    if not (DATA_ROOT / english / icl_context.DEFAULT_POOL).is_dir():
        pytest.skip(f"no pool for {english}")
    X.export_game(english, tmp_path)
    label = X.LABELS[english]
    assert label == "K3QP2"
    assert english not in label.lower()
    assert (tmp_path / label).is_dir()


def test_the_obs_header_is_gone(exported):
    for record in _records(exported):
        assert not X._obs_header_present(record["state"])
        assert not X._obs_header_present(record["next_state"])


def test_index_covers_every_transition(exported):
    import csv
    with open(exported / "drives" / "index.csv") as handle:
        rows = list(csv.DictReader(handle))
    records = _records(exported)
    assert len(rows) == len(records)
    for row, record in zip(rows, records):
        assert row["id"] == record["id"]
        assert row["action"] == record["action"]
        assert (exported / row["file"]).is_file()


@pytest.mark.parametrize("planted", [
    '{"human_game": "ice"}',
    '{"selection_note": "night spawns+solid stacking"}',
    '{"task_id": "ice_defect_detection"}',
    "Task: interactive\nStep: 14\nPhase: Interactive\n",
    "the dynamics of this world",
])
def test_the_guard_rejects_each_leak(tmp_path, planted):
    out = tmp_path / "BT3GB"
    (out / "drives").mkdir(parents=True)
    (out / "drives" / "t000.json").write_text(planted)
    with pytest.raises(AssertionError):
        X._assert_clean(out, GAME, {"n_transitions": 1}, readme="")


@pytest.mark.parametrize("name", ["ice", "SET", "logic_gates", "bt3gb"])
def test_the_guard_rejects_a_world_name(tmp_path, name):
    """Naming any of the fifteen narrows the field, not just naming this one."""
    if name == "ice":
        pytest.skip("'ice' is the human name, caught via human_game rather than LABELS")
    out = tmp_path / "BT3GB"
    (out / "drives").mkdir(parents=True)
    (out / "drives" / "t000.json").write_text(f'{{"note": "this is {name}"}}')
    with pytest.raises(AssertionError):
        X._assert_clean(out, GAME, {"n_transitions": 1}, readme="")


def test_the_readme_is_pinned(exported):
    """The README is authored, so it is pinned rather than scanned -- four worlds are
    spelled with ordinary English words and prose trips a name scan."""
    readme = (exported / "drives" / "README.md")
    original = readme.read_text()
    readme.write_text(original + "\nthe world is ice\n")
    with pytest.raises(AssertionError, match="pinned template"):
        X._assert_clean(exported, GAME, {"n_transitions": 60}, readme=original)
    readme.write_text(original)


def test_an_empty_export_is_an_error(tmp_path):
    out = tmp_path / "BT3GB"
    (out / "drives").mkdir(parents=True)
    with pytest.raises(AssertionError, match="empty corpus"):
        X._assert_clean(out, GAME, {"n_transitions": 0}, readme="")
