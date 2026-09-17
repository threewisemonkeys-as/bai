"""Scientific correctness checks for the probe protocol, all without model calls."""
from __future__ import annotations

import asyncio
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "offline_learning"))

import probing_common as C  # noqa: E402
import probing_eval as E  # noqa: E402
import probing_manifest as M  # noqa: E402
import probing_report as R  # noqa: E402


def grid(colour="black", pos=0):
    row = ["black"] * 4
    row[pos] = colour
    return C.json_text([row])


def test_grid_scoring_is_structural_and_position_sensitive():
    start, target = grid(), grid("red", 1)
    equivalent = '<next_state> [ [ "black", "red", "black", "black" ] ] </next_state>'
    assert C.grid_scores(equivalent, target, start=start)["exact"] == 1
    for wrong in (grid("red", 2), grid("blue", 1)):
        score = C.grid_scores(wrong, target, start=start)
        assert score["change_f1"] == 0
        assert score["foreground_f1"] == 0
    assert C.grid_scores(start, target, start=start)["cell_accuracy"] == 0.75
    assert C.grid_scores(start, target, start=start)["change_f1"] == 0
    assert C.grid_scores(start, start, start=start)["change_f1"] == 1


@pytest.mark.parametrize("answer", ["", "not a grid", '[["black"]]',
    '[["black", "red"], ["black"]]', '<next_state>[["black"]]',
    '<next_state>[["red"]]</next_state><next_state>[["blue"]]</next_state>',
    'explanation [["black", "red", "black", "black"]]'])
def test_invalid_answers_are_failures(answer):
    scores = C.grid_scores(answer, grid("red", 1), start=grid())
    assert scores["exact"] == scores["cell_accuracy"] == scores["change_f1"] == 0
    assert scores["parse_error"]


def test_observation_metadata_only_allowed_on_input():
    text = 'Task: interactive\nStep: 99\n========== Start of Direct Observation ==========\n' + grid("red")
    original, canonical = C.observation_grid(text)
    assert "Step" not in original and canonical == grid("red")
    with pytest.raises(ValueError):
        C.parse_prediction(text)


def test_generic_lossless_baseline_and_collision_bound():
    examples = [grid(), grid("red", 1), C.json_text([["red", "red"], ["blue", "red"]])]
    for x in examples:
        assert C.decode_lossless(C.lossless_encoding(x)) == x
    stats = C.collision_summary(["a", "a", "b"], ["x", "y", "z"])
    assert stats["empirical_reconstruction_bound"] == pytest.approx(2 / 3)
    assert stats["ambiguous_outputs"] == 1


def test_horizon_windows_preserve_actions_and_terminal_target():
    rows = [{"Step": str(i), "Action": "right", "Done": "False", "Observation": grid("red", i % 4)}
            for i in range(8)]
    rows[-1].update(Action="", Done="True")
    starts, errors = M.eligible_starts(rows, 2, 3, {"right"})
    assert starts == [2, 3, 4] and not errors
    broken = copy.deepcopy(rows)
    broken[4]["Done"] = "True"
    assert M.eligible_starts(broken, 2, 3, {"right"})[0] == []
    broken = copy.deepcopy(rows)
    broken[4]["Step"] = "0"
    assert M.eligible_starts(broken, 2, 3, {"right"})[0] == []
    rows[3]["Action"] = "click 0 3"
    assert M.edge_error(rows, 3, {"right", "click"}) is None
    rows[3]["Action"] = "click 3 0"
    assert M.edge_error(rows, 3, {"right", "click"}) == "out_of_bounds_click"


def test_sampling_balances_users_with_unequal_drive_counts():
    items = [{"user_id": user, "drive_id": f"{user}/{drive}", "row_index": i}
             for user, drives in (("a", 3), ("b", 1)) for drive in range(drives) for i in range(20)]
    a = M.balanced_sample(copy.deepcopy(items), 10, M.random.Random(8), separation=4)
    b = M.balanced_sample(copy.deepcopy(items), 10, M.random.Random(8), separation=4)
    assert a == b
    assert sum(x["user_id"] == "a" for x in a) == 5
    for i, x in enumerate(a):
        for y in a[i + 1:]:
            if x["drive_id"] == y["drive_id"]:
                assert abs(x["row_index"] - y["row_index"]) >= 4


def test_incumbents_use_iteration_ids_and_original_tie_break():
    candidates = [{"idx": 0, "train_score": 0}, {"idx": 1, "train_score": 0.5},
                  {"idx": 2, "train_score": 0.5}, {"idx": 3, "train_score": 0.4}]
    process = [{"i": 3, "new_idx": 3, "new_score": 0.4},
               {"i": 2, "new_idx": 2, "new_score": 0.5},
               {"i": 1, "new_idx": 1, "new_score": 0.5}]
    cps, _ = C.incumbent_checkpoints(candidates, process, 4, lambda c: {"working": c["idx"] != 0})
    assert cps[0]["candidate_idx"] == cps[-1]["candidate_idx"] == 1
    assert cps[0]["iteration"] == 1 and cps[-1]["iteration"] == 4
    with pytest.raises(ValueError, match="mapping"):
        C.incumbent_checkpoints(candidates, process[1:], 4, lambda c: {"working": True})


def test_perception_resets_history_and_detects_nondeterminism(tmp_path):
    code = "counter = 0\ndef perceive(history):\n global counter\n counter += 1\n return str((counter, len(history)))"
    values = C.perceive(code, {"a": grid(), "b": grid("red")}, cache_dir=tmp_path)
    assert values["a"] == values["b"] == {"output": "(1, 1)", "error": None}
    changed = C.perceive("def perceive(h): return 'different'", {"a": grid()}, cache_dir=tmp_path)
    assert changed["a"]["output"] == "different"
    random_code = "import uuid\ndef perceive(h): return str(uuid.uuid4())"
    assert "nondeterministic" in C.perceive(random_code, {"a": grid()})["a"]["error"]
    loop = "def perceive(h):\n while True: pass"
    assert "timeout" in C.perceive(loop, {"a": grid()}, timeout=0.05)["a"]["error"]


@pytest.fixture
def manifest():
    xs = [grid("train"), grid("past"), grid("current"), grid("future_one"), grid("future_two")]
    frames = {str(i): {"observation": x, "grid": x, "grid_sha256": C.digest(x)} for i, x in enumerate(xs)}
    code = "def perceive(history): return 'ENCODED=' + history[0]"
    artifact = {"perception": code, "knowledge": "KNOWN_RULES",
                "perception_sha256": C.digest(code), "knowledge_sha256": C.digest("KNOWN_RULES")}
    checkpoints = [{"label": "first_working", "iteration": 1, "candidate_idx": 1},
                   {"label": "final", "iteration": 10, "candidate_idx": 2}]
    game = {"frames": frames, "program": {"background": "black"}, "example_frame_ids": ["0"],
            "training_target_positions": {"train": [{"frame_ids": ["0", "0"]}]},
            "runs": [{"arm": "Plain", "seed": 1, "artifacts": {"1": artifact, "2": artifact},
                      "checkpoints": checkpoints}],
            "audit": {"replay_status": "verified"},
            "windows": [{"id": "w", "drive_id": "test_d0", "user_id": "u", "row_index": 1,
                         "context_frame_ids": ["1", "2"], "past_actions": ["left"],
                         "future_frame_ids": ["3", "4"], "future_actions": ["click 0 1", "noop"],
                         "changed": {"1": True, "2": True}, "target_seen_in_train": {"1": False, "2": False},
                         "context_seen_in_train": False}],
            "reconstruction_queries": [{"id": "q1", "frame_id": "3", "user_id": "u", "drive_id": "test_d0",
                                        "row_index": 2, "seen_in_train": False},
                                       {"id": "q2", "frame_id": "4", "user_id": "v", "drive_id": "test_d1",
                                        "row_index": 3, "seen_in_train": False}]}
    body = {"schema_version": C.SCHEMA_VERSION, "config": {"horizons": [1, 2], "split": "test", "sample_seed": 0,
            "perception_runtime": C.perception_runtime(2.0)}, "reference_arm": "Plain", "games": {"toy": game}, "sources": {}}
    return {**body, "manifest_sha256": C.digest(body)}


@pytest.fixture
def config():
    return {"forward_modes": ["learned_raw", "learned_native", "raw", "copy"],
            "reconstruction_modes": ["learned", "raw", "constant", "shuffled", "lossless_inverse"],
            "checkpoints": ["first_working", "final"], "examples": 1, "arms": [], "max_prompt_chars": 10000,
            "model": {"id": "fixture", "provider": "fixture", "reasoning": {}, "temperature": 0,
                      "max_output_tokens": 256, "timeout_s": 1}}


def test_prompts_hide_future_frames_and_reconstruction_has_no_knowledge(manifest, tmp_path):
    jobs = list(E.study_jobs(manifest, cache_dir=tmp_path,
        forward_modes={"learned_raw", "learned_native"}, reconstruction_modes={"learned", "shuffled"},
        checkpoint_labels={"first_working", "final"}))
    forecasts = [j for j in jobs if j["kind"] == "forward"]
    assert len(forecasts) == 4  # shared P/K checkpoint pair evaluated once, with both labels
    for j in forecasts:
        assert "future_one" not in j["prompt"] and "future_two" not in j["prompt"]
        assert "past" in j["prompt"] and "current" in j["prompt"]
        assert "click 0 1" in j["prompt"]
        assert j["target"] != j["start"]
    for j in jobs:
        if j["kind"] == "reconstruction":
            assert "KNOWN_RULES" not in j["prompt"]
            assert '"past"' not in j["prompt"] and '"current"' not in j["prompt"]
            assert j["knowledge_sha256"] is None
            if j["mode"] == "shuffled":
                assert j["query_id"] != j["control_donor_query_id"]
                assert j["control_changes_representation"]


def test_source_hashes_detect_observation_changes_with_same_actions(manifest, tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("left,observation_a")
    manifest["sources"] = {str(source): {"sha256": C.file_digest(source)}}
    manifest["manifest_sha256"] = C.digest({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    M.verify_manifest(manifest)
    source.write_text("left,observation_b")
    with pytest.raises(ValueError, match="source changed"):
        M.verify_manifest(manifest)


def test_prepare_execute_resume_and_incomplete_reporting(manifest, config, tmp_path):
    summary = E.prepare(manifest, tmp_path, config)
    assert summary["uncached_requests"] > 0
    coverage = E.score_jobs(tmp_path)
    assert coverage["pending"] > 0
    early = R.report(tmp_path, plots=False, draws=20)
    assert any(not r["complete"] and r["metrics"] is None for r in early["macro"])
    calls = []

    async def fake(prompt):
        calls.append(prompt)
        return {"response": "malformed", "usage": {"cost": 0.01}, "finish_reason": "stop"}

    with pytest.raises(ValueError, match="exceed"):
        asyncio.run(E.execute(tmp_path, fake, max_calls=0))
    assert not calls
    asyncio.run(E.execute(tmp_path, fake, max_calls=100, attempts=1, concurrency=2))
    assert len(calls) == summary["unique_model_requests"]
    count = len(calls)
    asyncio.run(E.execute(tmp_path, fake, max_calls=0))
    assert len(calls) == count
    assert E.score_jobs(tmp_path).get("pending", 0) == 0
    result = R.report(tmp_path, plots=False, draws=20)
    learned = [r for r in result["macro"] if r["arm"] == "Plain"]
    assert all(r["complete"] and r["metrics"]["exact"] == 0 for r in learned)
    assert result["reported_cost"] == pytest.approx(count * 0.01)
    changed = copy.deepcopy(config)
    changed["model"]["temperature"] = 0.5
    with pytest.raises(ValueError, match="different protocol"):
        E.prepare(manifest, tmp_path, changed)


def test_provider_errors_stay_visible_and_are_not_silent_zero_scores(manifest, config, tmp_path):
    E.prepare(manifest, tmp_path, config)

    async def failing(_prompt):
        raise TimeoutError("fixture timeout")

    asyncio.run(E.execute(tmp_path, failing, attempts=1, max_calls=100))
    E.score_jobs(tmp_path)
    rows = list(C.read_jsonl(tmp_path / "results.jsonl"))
    assert all(r["metrics"] is None for r in rows if r["status"] == "provider_error")
    result = R.report(tmp_path, plots=False, draws=10)
    assert all(r["metrics"] is None for r in result["macro"] if r["arm"] == "Plain")


def test_cache_cannot_be_reused_with_a_different_prompt(tmp_path):
    p = tmp_path / "call.json"
    C.write_json(p, {"request_key": "same", "prompt_sha256": C.digest("before"), "status": "completed"})
    with pytest.raises(ValueError, match="identity"):
        E.load_call(p, "same", "after")


def test_modified_protocol_cannot_execute_or_score_cached_requests(manifest, config, tmp_path):
    E.prepare(manifest, tmp_path, config)
    path = tmp_path / "protocol.json"
    protocol = C.read_json(path)
    protocol["config"]["model"]["id"] = "different-model"
    C.write_json(path, protocol)

    async def must_not_call(_prompt):
        pytest.fail("modified protocols must fail before calling a model")

    with pytest.raises(ValueError, match="protocol identity"):
        asyncio.run(E.execute(tmp_path, must_not_call))
    with pytest.raises(ValueError, match="protocol identity"):
        E.score_jobs(tmp_path)


def test_saved_batch_audit_checks_context_actions_not_only_grids(tmp_path):
    folder = tmp_path / "rexpure_run_seed1"
    folder.mkdir()
    tr = SimpleNamespace(x_t="x", x_t1="y", ctx_prev=[("p", "left")], ctx_next=[("right", "n")])
    win = {"x_t": "x", "x_t1": "y", "prev_raw": ["p"], "nxt_raw": ["n"],
           "prev": [["feature-p", "left"]], "nxt": [["right", "feature-n"]]}
    record = {"cand_idx": 0, "trajectories": [{"win": win}]}
    path = folder / "resume_batches.jsonl"
    path.write_text(C.json_text(record) + "\n")
    builder = M.ManifestBuilder(repo=tmp_path)
    assert builder.check_batches(tmp_path, 1, [{"tr": tr}], [{}])["raw_contexts_match_saved_batches"]
    win["prev"][0][1] = "right"
    path.write_text(C.json_text(record) + "\n")
    with pytest.raises(ValueError, match="context actions"):
        builder.check_batches(tmp_path, 1, [{"tr": tr}], [{}])


def test_audit_restores_training_functions_after_loader_failure(tmp_path, monkeypatch):
    def original_loader(*_args, **_kwargs):
        return []

    def original_similarity(a, b):
        return float(a == b)

    def fail(*_args):
        raise RuntimeError("fixture loader failure")

    args = SimpleNamespace(keep_obs_metadata=False, collapse_action_params=False, seed=1)
    optimizer = SimpleNamespace(load_transitions=original_loader, build_data=fail,
        build_parser=lambda: SimpleNamespace(parse_args=lambda _argv: args))
    core = SimpleNamespace(_frame_sim=original_similarity)
    monkeypatch.setitem(sys.modules, "rexpure_optimize", optimizer)
    monkeypatch.setitem(sys.modules, "invdyn_core", core)
    C.write_json(tmp_path / "launch.json", {"cmd": ["runner.py"]})
    with pytest.raises(RuntimeError, match="fixture loader failure"):
        M.ManifestBuilder(repo=tmp_path).rebuild(tmp_path)
    assert optimizer.load_transitions is original_loader
    assert core._frame_sim is original_similarity


def test_unique_state_metric_averages_occurrences_within_each_state():
    rows = []
    for state, value in (("a", 1), ("a", 0), ("b", 0)):
        rows.append({"kind": "reconstruction", "arm": "Plain", "mode": "learned", "horizon": 0,
                     "game": "g", "checkpoints": [{"label": "final"}], "status": "completed",
                     "user_id": "u", "seen_in_train": False, "target_grid_sha256": state,
                     "target": state, "representation": "same", "representation_errors": [],
                     "metrics": {"exact": value}})
    result = next(r for r in R.summarize(rows) if r["stratum"] == "all")
    assert result["unique_state_exact"] == 0.25
    assert result["representation_audit_queries"] == 3
    assert result["empirical_reconstruction_bound"] == pytest.approx(2 / 3)
    assert result["ambiguous_representations"] == 1


def test_unknown_filters_and_mutated_jobs_fail_before_calls(manifest, config, tmp_path):
    bad = copy.deepcopy(config)
    bad["checkpoints"] = ["fianl"]
    with pytest.raises(ValueError, match="filter"):
        E.prepare(manifest, tmp_path, bad)
    E.prepare(manifest, tmp_path, config)
    with (tmp_path / "jobs.jsonl").open("a") as fh:
        fh.write("{}\n")
    with pytest.raises(ValueError, match="changed"):
        E.score_jobs(tmp_path)


def test_api_adapter_sends_only_the_prompt_and_enforces_output_cap(config, monkeypatch):
    import httpx
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")
    captured = []

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "fixture", "model": "fixture", "provider": "fixture",
            "choices": [{"message": {"content": grid()}, "finish_reason": "stop"}],
            "usage": {"cost": 0, "prompt_tokens": 8, "completion_tokens": 4}})

    async def run():
        client = E.OpenRouterClient(config["model"])
        await client.http.aclose()
        client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            result = await client("only the permitted context")
            assert result["usage"]["prompt_tokens"] == 8
        finally:
            await client.close()

    asyncio.run(run())
    assert captured[0]["messages"] == [{"role": "user", "content": "only the permitted context"}]
    assert captured[0]["max_tokens"] == 256
    assert captured[0]["provider"]["allow_fallbacks"] is False


def test_paired_analysis_never_treats_checkpoints_as_independent_samples():
    rows = []
    for game in ("a", "b"):
        for user, delta in (("shared", 0.4), (f"only-{game}", 0.2)):
            for checkpoint, value in (("first_working", 0.1), ("final", 0.1 + delta)):
                rows.append({"game": game, "user_id": user, "query_id": user,
                    "mode": "learned_raw", "horizon": 4, "arm": "Plain",
                    "checkpoints": [{"label": checkpoint}], "target_grid_sha256": "x",
                    "metrics": {"change_f1": value}})
    result = R.paired_comparison(rows, "Plain", "learned_raw", 4, "change_f1", ["a", "b"], draws=100)
    assert result["delta"] == pytest.approx(0.3)
    assert result["user_clusters"] == 3
    assert result["interval_95"][0] <= result["delta"] <= result["interval_95"][1]
    rows[-1]["metrics"] = None
    assert not R.paired_comparison(rows, "Plain", "learned_raw", 4, "change_f1", ["a", "b"])["complete"]
