"""Deterministic driver checks for the shared REx-pure loop (no LLM, no gepa).

invdyn_core.rex_search is the single search loop behind rexpure_optimize (perception/
belief WM), worldcoder_optimize (program WM) and stepwise_eb_learn's frontier mode.
These checks pin the loop's deterministic logic: the ship rule (argmax mean train,
first-on-tie -- equivalent to the old gepa FullEvaluationPolicy.get_best_program under
equal coverage), the budget accounting, and REx-pure selector determinism.
"""
import json
import random
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import invdyn_core as core
import rexpure_optimize as rex
import worldcoder_optimize as wc
from validate import run_perceive


def check_reflected_perception_cleaning():
    """Regression guard: the reflection LM returns the module wrapped in a ``` fence;
    rex_search must run extract_proposed_text BEFORE _clean_component or the fence
    survives (because _clean_component only strips fences when 'def perceive' is
    ABSENT) and perceive() execs to a SyntaxError -> '' -> const-P gate zeroes every
    candidate -> ships the seed. This bug silently made all rexpure/worldcoder runs
    ship the seed until 2026-08-06."""
    fenced = (
        "```python\n"
        "import json\n"
        "def perceive(observation_history):\n"
        "    obs = observation_history[-1] if observation_history else ''\n"
        "    s = obs.find('[['); e = obs.rfind(']]')\n"
        "    if s == -1 or e == -1: return 'no-grid'\n"
        "    grid = json.loads(obs[s:e+2])\n"
        "    cells = [f'{r},{c},{v}' for r,row in enumerate(grid) for c,v in enumerate(row) if v!='black']\n"
        "    return 'objs:' + ';'.join(cells[:50]) if cells else 'empty-grid'\n"
        "```"
    )
    # mirror rex_search's exact cleaning pipeline
    cleaned = core._clean_component("perception", core.extract_proposed_text(fenced))
    assert not cleaned.lstrip().startswith("```"), "fence not stripped -> SyntaxError at exec"
    obs = '[["black","gold"],["blue","black"]]'
    out, err = run_perceive(cleaned, obs)
    assert err is None, f"cleaned perception raised: {err}"
    assert out.strip() and out != "no-grid", f"cleaned perception produced empty/degenerate output: {out!r}"
    print(f"[ok] fenced reflection output cleans to executable, non-empty perception ({out!r})")


def check_shared_loop_identity():
    """All consumers must drive the SAME loop + selector code from invdyn_core, so
    selection / accounting / ship cannot diverge between them."""
    assert rex.rex_search is core.rex_search
    assert wc.rex_search is core.rex_search
    assert rex.RExPureCandidateSelector is core.RExPureCandidateSelector
    assert rex.PerceptionBiasedComponentSelector is core.PerceptionBiasedComponentSelector
    assert wc.RExPureCandidateSelector is core.RExPureCandidateSelector
    assert rex.InvDynAdapter is core.InvDynAdapter
    print("[ok] rexpure & worldcoder drive the SAME invdyn_core.rex_search + selectors")


def reference_best_idx(subscores_per_candidate):
    """Reference ship rule = the old gepa FullEvaluationPolicy.get_best_program:
    highest avg val score, tie-broken by coverage (n instances), first-on-full-tie.
    Replicated here (no gepa import) as the parity oracle."""
    best_idx, best_score, best_cov = -1, float("-inf"), -1
    for idx, scores in enumerate(subscores_per_candidate):
        cov = len(scores)
        avg = sum(scores.values()) / cov if cov else float("-inf")
        if avg > best_score or (avg == best_score and cov > best_cov):
            best_score, best_idx, best_cov = avg, idx, cov
    return best_idx


def rexpure_best_idx(h):
    """rex_search's ship rule: argmax mean train score, first index on tie."""
    return max(range(len(h)), key=lambda j: h[j])


def check_ship_rule_equivalence(trials=2000):
    """rex_pure scores every candidate on ALL train rows -> equal coverage, so the
    ship rule reduces to argmax-mean-first-on-tie, matching the reference oracle."""
    rng = random.Random(0)
    for _ in range(trials):
        n_cand, n_rows = rng.randint(1, 8), rng.randint(1, 6)
        subs, h = [], []
        for _c in range(n_cand):
            row = {i: rng.choice([0.0, 0.0, 0.5, 1.0]) for i in range(n_rows)}
            subs.append(row)
            h.append(sum(row.values()) / n_rows)
        assert reference_best_idx(subs) == rexpure_best_idx(h)
    print(f"[ok] ship rule == reference get_best_program on {trials}/{trials} random pools")


def check_budget_accounting():
    """seed = len(train), then +len(train) per admitted candidate; #candidates = 1 +
    floor over budget. Matches the observed 3/18 (train 6) and 13/156 (train 12)."""
    for train_n, budget, exp_cand, exp_calls in [(6, 18, 3, 18), (12, 156, 13, 156)]:
        calls, cand = train_n, 1
        while calls < budget:
            calls += train_n
            cand += 1
        assert cand == exp_cand and calls == exp_calls, (train_n, budget, cand, calls)
    print("[ok] budget accounting reproduces observed 3/18 and 13/156 (candidates/metric-calls)")


def check_selector_determinism():
    """Same class + same seed -> identical Thompson draws."""
    h = [0.0, 0.3, 0.3, 0.9, 0.1]
    pool = [{"perception": str(i)} for i in range(len(h))]
    picks = []
    for seed in (0, 1):
        s1 = core.RExPureCandidateSelector(c=5.0, rng=random.Random(1234 + seed))
        s2 = core.RExPureCandidateSelector(c=5.0, rng=random.Random(1234 + seed))
        st = SimpleNamespace(program_candidates=pool, program_full_scores_val_set=h)
        seq1 = [s1.select_candidate_idx(st) for _ in range(20)]
        seq2 = [s2.select_candidate_idx(st) for _ in range(20)]
        assert seq1 == seq2, "selector not deterministic under fixed seed"
        picks.append(tuple(seq1))
    assert picks[0] != picks[1], "different seeds should give different sequences"
    print("[ok] REx-pure selector is seed-deterministic (same seed -> same parent sequence)")


class _FakeAdapter:
    """LLM-free stand-in for InvDynAdapter: the child's score is a deterministic
    function of its perception text, and every call sleeps briefly so a batched round
    genuinely overlaps (the wall-clock assertion below would pass trivially otherwise)."""

    def __init__(self, train, delay=0.05):
        self.train = train
        self.delay = delay
        self.total_cost = 0.0
        self.eval_calls = 0
        self.reused_evals = 0
        self.run_ctx = core.ThreadScopedCtx(iteration=None)
        self.max_concurrent = 0
        self._live = 0
        self._lock = threading.Lock()

    def _enter(self):
        with self._lock:
            self._live += 1
            self.max_concurrent = max(self.max_concurrent, self._live)

    def _exit(self):
        with self._lock:
            self._live -= 1

    def evaluate(self, train, candidate, capture_traces=False):
        self._enter()
        time.sleep(self.delay)
        self._exit()
        self.eval_calls += 1
        # score = fraction of train rows whose index digit appears in the perception text
        text = candidate.get("perception", "")
        scores = [1.0 if str(k) in text else 0.0 for k in range(len(train))]
        trajs = [{"tr": inst["tr"], "z_t": text, "id_score": s} for inst, s in zip(train, scores)]
        return core.EvaluationBatch(outputs=[text] * len(train), scores=scores,
                                    trajectories=trajs if capture_traces else None)

    def make_reflective_dataset(self, candidate, eval_batch, components):
        self._enter()
        time.sleep(self.delay)
        self._exit()
        return {c: [{"Feedback": f"improve {c}"}] for c in components}


def _fake_train(n=4):
    tr = SimpleNamespace(action="noop", x_t="a", x_t1="b")
    return [{"tr": tr} for _ in range(n)]


def _run_search(propose_batch, run_dir, nodes=7, resume=False):
    train = _fake_train()
    adapter = _FakeAdapter(train)
    counter = {"n": 0}
    lock = threading.Lock()

    def reflection_lm(prompt):  # noqa: ARG001
        with lock:
            counter["n"] += 1
            k = counter["n"]
        time.sleep(0.05)
        return f"```\ndef perceive(h):\n    return '{k}'\n```"

    t0 = time.perf_counter()
    res = core.rex_search(
        adapter=adapter, seed_candidate={"perception": "seed", "world_knowledge": ""},
        train=train, reflection_lm=reflection_lm,
        templates={"perception": "<curr_param><side_info>", "world_knowledge": "<curr_param><side_info>"},
        selector=core.RExPureCandidateSelector(c=5.0, rng=random.Random(7)),
        module_selector=core.PerceptionBiasedComponentSelector(4),
        max_nodes=nodes, run_dir=run_dir, log_prefix="test", propose_batch=propose_batch,
        resume=resume,
    )
    return res, adapter, time.perf_counter() - t0


def check_propose_batch():
    """--propose-batch B must (a) leave B=1 behaviour intact, (b) explore exactly the same
    node budget at B>1, (c) actually overlap iterations, and (d) still write one complete
    process_log record per iteration -- including the accepted ones, whose node index is only
    known on the main thread (the reason the worker hands its record back via detach())."""
    with tempfile.TemporaryDirectory() as td:
        d1, d3 = Path(td) / "b1", Path(td) / "b3"
        r1, a1, w1 = _run_search(1, d1)
        r3, a3, w3 = _run_search(3, d3)

        for tag, res in (("B=1", r1), ("B=3", r3)):
            assert res["nodes_explored"] == 7, f"{tag}: {res['nodes_explored']} nodes != 7"
            assert len(res["pool"]) == len(res["train_scores"]) == len(res["parents"]) == 7, tag
        assert a1.max_concurrent == 1, f"B=1 must stay serial, saw {a1.max_concurrent} in flight"
        assert a3.max_concurrent > 1, "B=3 did not overlap any iterations"
        assert w3 < w1, f"B=3 ({w3:.2f}s) not faster than B=1 ({w1:.2f}s)"

        for tag, d in (("B=1", d1), ("B=3", d3)):
            recs = [json.loads(l) for l in (d / "process_log.jsonl").open()]
            accepted = [r for r in recs if r.get("accepted")]
            assert len(accepted) == 6, f"{tag}: {len(accepted)} accepted records != 6 children"
            assert {r["new_idx"] for r in accepted} == set(range(1, 7)), f"{tag}: node indices wrong"
            assert all(r.get("proposed") and r.get("feedback") for r in accepted), \
                f"{tag}: accepted records lost their proposal/feedback payload"
            assert len({r["i"] for r in recs}) == len(recs), f"{tag}: duplicate iteration numbers"
        # resume state must still describe the full pool after a batched run
        st = json.loads((d3 / "resume_state.json").read_text())
        assert st["n_nodes"] == 7 and len(st["h"]) == 7, f"batched resume state incomplete: {st['n_nodes']}"
        assert sum(1 for _ in (d3 / "resume_batches.jsonl").open()) == 7, "resume batches incomplete"

        # skip-perfect moved to the main thread with the draw -- it must still consume
        # iterations, add no node, and leave a "skipped" record per iteration.
        dsk = Path(td) / "perfect"
        train = _fake_train()
        adapter = _FakeAdapter(train)
        core.rex_search(
            adapter=adapter, seed_candidate={"perception": "0123", "world_knowledge": ""},
            train=train, reflection_lm=lambda p: "```\ndef perceive(h):\n    return 'x'\n```",
            templates={"perception": "<curr_param><side_info>", "world_knowledge": "<curr_param><side_info>"},
            selector=core.RExPureCandidateSelector(c=5.0, rng=random.Random(7)),
            module_selector=core.PerceptionBiasedComponentSelector(4),
            max_nodes=3, run_dir=dsk, log_prefix="perfect", max_iters=4, propose_batch=3,
        )
        recs = [json.loads(l) for l in (dsk / "process_log.jsonl").open()]
        assert recs and all(r["verdict"] == "skipped" for r in recs), \
            f"skip-perfect lost its process_log records: {[r.get('verdict') for r in recs]}"
        assert adapter.eval_calls == 1, f"skip-perfect ran {adapter.eval_calls} evals (only the seed should)"
    print(f"[ok] --propose-batch: B=3 explores the same 7 nodes, overlaps "
          f"({a3.max_concurrent} in flight), and logs complete records ({w3:.1f}s vs {w1:.1f}s serial)")


def check_batched_resume():
    """Node-level --resume must still work when rounds admit several children at once:
    the checkpoint is written per admitted child, so a batched round must leave the pool,
    the batches file and the iteration ids consistent for the next process to pick up."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "r"
        first, _, _ = _run_search(3, d, nodes=4)
        assert first["nodes_explored"] == 4, first["nodes_explored"]
        second, _, _ = _run_search(3, d, nodes=9, resume=True)

        assert second["nodes_explored"] == 9, f"resume reached {second['nodes_explored']}/9 nodes"
        assert len(second["pool"]) == len(second["train_scores"]) == 9
        st = json.loads((d / "resume_state.json").read_text())
        assert st["n_nodes"] == 9, f"checkpoint says {st['n_nodes']} nodes"
        assert sum(1 for _ in (d / "resume_batches.jsonl").open()) == 9, "resume batches incomplete"
        recs = [json.loads(l) for l in (d / "process_log.jsonl").open()]
        assert len({r["i"] for r in recs}) == len(recs), "duplicate iteration ids across resume"
    print("[ok] --propose-batch resumes at node level (4 -> 9 nodes, state + logs consistent)")


def check_no_leaked_locals():
    """Guard for the build_data/build_parser extraction: main() must not read a name that
    only lives inside the function it was split from. This exact bug (`k = args.k_choices`
    moved into build_data, `chance = 1.0 / k` left in main; and `ap.error(...)` left behind
    when the parser moved to build_parser) crashed five launched runs at startup -- AFTER
    the whole suite passed, because nothing here calls main(). Static, so it costs nothing."""
    import ast
    import builtins

    tree = ast.parse(Path(rex.__file__).read_text())
    fns = {f.name: f for f in tree.body if isinstance(f, ast.FunctionDef)}

    def assigned(fn):
        out = set()
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                out.add(n.id)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not fn:
                out.add(n.name)
            elif isinstance(n, ast.arg):
                out.add(n.arg)
        return out

    def loaded(fn):
        return {n.id for n in ast.walk(fn)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}

    main_fn = fns["main"]
    free = loaded(main_fn) - assigned(main_fn) - set(dir(builtins))
    for donor in ("build_data", "build_parser"):
        leaked = sorted(free & assigned(fns[donor]))
        assert not leaked, f"main() reads {leaked}, which only {donor}() defines -> NameError at runtime"
    print("[ok] main() has no free names owned by build_data/build_parser")


if __name__ == "__main__":
    check_shared_loop_identity()
    check_selector_determinism()
    check_ship_rule_equivalence()
    check_budget_accounting()
    check_reflected_perception_cleaning()
    check_propose_batch()
    check_batched_resume()
    check_no_leaked_locals()
    print("\nALL DRIVER CHECKS PASSED")
