"""WorldCoder-style program world-model learner (arm B vs the language-WM arm).

Learns a Python `transition(prev, grid, action) -> next_grid` program (contract in
program_runtime.CONTRACT) by LLM refinement against the train buffer -- WorldCoder
(arXiv 2402.12275) instantiated on the perc_invdyn data/tooling:

- ϕ1 objective: exact canonical-grid fit of logged transitions. Scoring is pure
  EXECUTION (program_runtime.ProgramRuntime) -- there is NO task LLM; the only LLM
  budget is proposer (reflection) calls, capped by --max-proposals.
- Aggregate h = 0.5*fit_changed + 0.5*fit_static (per-item scores are reweighted so
  the mean reproduces this). The identity program sits at exactly 0.5 and an
  identity-on-all-changed candidate has its whole batch zeroed (the analog of
  the constant-P gate), so `return grid` is never a Thompson attractor.
- Search loop: invdyn_core.rex_search -- faithful REx (Tang et al. 2024): gate-free
  pool, parent Thompson-sampled from full-train h, every child admitted as an arm,
  full-train reflection each step. Counterexample backprompting via
  make_reflective_dataset: failing transitions with the program's wrong prediction /
  traceback and NL cell diffs (paper F.1/F.3).
- Ship criterion: REx selects on train h (paper semantics); afterwards EVERY pool
  candidate is re-scored (free) on a disjoint stratified val carve and the shipped
  program is argmax val balanced score (ties -> shorter code).

The action buffer is NEVER collapsed: the program always receives the fully
parameterized canonical action ('click ROW COL'). --collapse-action-params only
changes the end-of-run test ID protocol (bare-'click' choices are checked with
click_enum semantics: consistent iff ANY cell's click reproduces the next grid).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import program_runtime as prt  # noqa: E402
from validate import (  # noqa: E402
    _extract_code,
    backfill_context_from_source,
    load_transitions,
    make_choices,
    make_config,
    strip_transitions_obs_metadata,
)
from invdyn_core import (  # noqa: E402
    EvaluationBatch,
    ProcessLogger,
    RExPureCandidateSelector,
    SingleComponentSelector,
    _REFLECTION,
    make_reflection_lm,
    rex_search,
    stratified_split,
)

COMPONENT = "transition_program"

WC_REFINE_TMPL = f"""You are learning a WORLD MODEL of a deterministic grid game as a Python program, by fitting logged (state, action, next_state) transitions.

{prt.CONTRACT}

Current program (empty means: write it from scratch following the contract):
```
<curr_param>
```

Execution feedback -- logged transitions the program must model, with its current predictions/errors:
```
<side_info>
```

Grids are JSON arrays of color-name strings, row-major: 'cell (r,c)' is row r, column c. Diff lines list only the cells that differ.

Work out the single consistent RULE SET that explains ALL shown transitions: what each action verb does, what clicking a cell does (its effect may be remote or conditional), how objects (connected color regions) move or interact, and what changes on its own every step regardless of the action (passive drift, timers, growth). Static transitions matter as much as changes: a rule that fires when it should not is as wrong as a missing rule.

CRITICAL -- GENERALIZE, do NOT memorize: never special-case a specific full grid, step index, or memorized (state -> next state) pair; such lookups cannot transfer to held-out states and score zero there. Every rule must be a general function of the visible configuration (and, if needed, the recent history window `prev`).

Rewrite the COMPLETE module. Reply with exactly ONE fenced code block containing the full module and nothing else in fences -- the text between the first and last ``` of your reply is extracted verbatim as the program."""


def _clean_program(text: str) -> str:
    """extract_proposed_text returns first-fence..last-fence content; if prose or a
    stray fence slipped in, re-extract the code block."""
    if "def transition" in text:
        return text
    return _extract_code(text)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------
class ProgramWMAdapter:
    """Adapter over PreparedTransition instances (evaluate + make_reflective_dataset,
    the interface rex_search drives). evaluate() executes the candidate program --
    no LLM. Keeps the cost-print conventions (total_cost/eval_calls/reuse_traces/
    reused_evals)."""

    def __init__(self, train_items: list, timeout_s: float = 1.0, max_runtimes: int = 6,
                 init_samples: int = 7):
        self.train_items = train_items
        self.init_samples = init_samples
        n = len(train_items)
        nc = sum(1 for it in train_items if it.changed)
        ns = n - nc
        # per-item weights so mean(scores) == 0.5*fit_changed + 0.5*fit_static
        self.w_changed = (n / (2.0 * nc)) if nc else 0.0
        self.w_static = (n / (2.0 * ns)) if ns else 0.0
        self.timeout_s = timeout_s
        self.max_runtimes = max_runtimes
        self._runtimes: OrderedDict[str, prt.ProgramRuntime] = OrderedDict()
        self._trace_cache: dict = {}  # (code_hash, item.idx) -> ItemResult
        self.eval_calls = 0
        self.reused_evals = 0
        self.total_cost = 0.0  # no task LLM
        self.reuse_traces = True

    @staticmethod
    def _hash(code: str) -> str:
        return hashlib.sha1(code.encode()).hexdigest()[:16]

    def runtime(self, code: str) -> prt.ProgramRuntime:
        h = self._hash(code)
        rt = self._runtimes.get(h)
        if rt is None:
            rt = prt.ProgramRuntime(code, timeout_s=self.timeout_s)
            self._runtimes[h] = rt
            while len(self._runtimes) > self.max_runtimes:
                _, old = self._runtimes.popitem(last=False)
                old.close()
        else:
            self._runtimes.move_to_end(h)
        return rt

    def evaluate(self, batch, candidate, capture_traces=False):
        # tolerate fence/prose leakage from the proposer: extract the module so a
        # good program wrapped in stray text is not unfairly zeroed.
        code = _clean_program(candidate.get(COMPONENT, ""))
        h = self._hash(code)
        results = []
        rt = None
        for it in batch:
            key = (h, it.idx)
            r = self._trace_cache.get(key)
            if r is None:
                if rt is None:
                    rt = self.runtime(code)
                r = rt.score_buffer([it])[0]
                self._trace_cache[key] = r
                self.eval_calls += 1
            else:
                self.reused_evals += 1
            results.append(r)
        # identity gate (analog of the constant-P gate): a program that acts as
        # the identity map on EVERY changed transition of the batch is degenerate
        # -- zero the whole batch so it never becomes a bandit attractor.
        changed = [r for r in results if r.changed]
        gate = bool(changed) and all(r.identity_pred or r.error is not None for r in changed) \
            and any(r.identity_pred for r in changed)
        scores = []
        for it, r in zip(batch, results):
            if gate:
                scores.append(0.0)
            else:
                w = self.w_changed if it.changed else self.w_static
                scores.append(w if r.exact else 0.0)
        trajectories = None
        if capture_traces:
            trajectories = [
                {"item": it, "res": r, "gate": gate} for it, r in zip(batch, results)
            ]
        return EvaluationBatch(
            outputs=[r.pred_canon for r in results], scores=scores,
            trajectories=trajectories,
        )

    # ---- reflective dataset (counterexample backprompt) -------------------
    def _tr_record(self, it, res=None) -> dict:
        rec = {}
        if it.prev:
            lines = [f"earliest grid in window: {prt.canon_grid(it.prev[0][0])}"]
            for j, (g, a) in enumerate(it.prev):
                nxt = it.prev[j + 1][0] if j + 1 < len(it.prev) else it.grid
                d = prt.render_diff(prt.grid_diff_cells(g, nxt), labels=("before", "after"))
                lines.append(f"then action '{prt.unparse_action(a)}' -> {d}")
            rec["History window (before the current grid)"] = "\n".join(lines)
        rec["Current grid"] = prt.canon_grid(it.grid)
        rec["Action taken"] = it.action_str
        rec["True next grid"] = prt.canon_grid(it.next_grid)
        rec["True change (current -> next)"] = prt.render_diff(
            prt.grid_diff_cells(it.grid, it.next_grid), labels=("current", "next"))
        if res is not None:
            if res.error is not None:
                rec["Your program"] = f"FAILED: {res.error}"
            elif res.exact:
                rec["Your program"] = "predicted this transition CORRECTLY"
            else:
                rec["Your program predicted"] = res.pred_canon
                try:
                    pred = json.loads(res.pred_canon)
                    rec["Prediction error (predicted vs true)"] = prt.render_diff(
                        prt.grid_diff_cells(pred, it.next_grid))
                except Exception:  # noqa: BLE001
                    pass
        return rec

    def _init_samples_pick(self, k: int, rng: random.Random) -> list:
        """Action-stratified sample of train transitions for the from-scratch prompt."""
        by_verb: dict = {}
        for it in self.train_items:
            by_verb.setdefault(it.action[0], []).append(it)
        for v in by_verb.values():
            rng.shuffle(v)
        picks, verbs = [], list(by_verb)
        i = 0
        while len(picks) < min(k, len(self.train_items)) and any(by_verb.values()):
            v = verbs[i % len(verbs)]
            i += 1
            if by_verb[v]:
                picks.append(by_verb[v].pop())
        return picks

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        code = candidate.get(COMPONENT, "")
        trajs = eval_batch.trajectories or []
        records = []
        if not code.strip():
            rng = random.Random(1234 + len(trajs))
            records.append({
                "Task": "The current program is EMPTY. Write the complete world-model "
                        "module from scratch, following the contract, so that it "
                        "reproduces the example transitions below (and generalizes).",
                "Example transitions": [self._tr_record(it) for it in
                                        self._init_samples_pick(self.init_samples, rng)],
            })
            return {COMPONENT: records}
        stats = prt.fit_stats([t["res"] for t in trajs])
        if trajs and trajs[0]["gate"]:
            records.append({
                "Degenerate program": (
                    "Your program currently acts as the IDENTITY map (or fails) on "
                    "every transition where the grid actually changed, so its score "
                    "was zeroed. Returning the input grid unchanged is not a world "
                    "model. Model the changes shown below."),
            })
        failures = [t for t in trajs if not t["res"].exact]
        failures.sort(key=lambda t: (not t["item"].changed, t["res"].cell_f1))
        records.extend(self._tr_record(t["item"], t["res"]) for t in failures[:6])
        correct = [t for t in trajs if t["res"].exact]
        shown = []
        for want_changed in (True, False):
            for t in correct:
                if t["item"].changed == want_changed:
                    shown.append(t)
                    break
        records.extend(self._tr_record(t["item"], t["res"]) for t in shown)
        records.append({
            "Overall fit": (
                f"{stats['fit_changed']:.0%} of changed transitions and "
                f"{stats['fit_static']:.0%} of static transitions predicted exactly "
                f"({stats['n_changed']} changed / {stats['n_static']} static in batch; "
                f"crash rate {stats['crash_rate']:.0%}, timeouts {stats['timeout_rate']:.0%})."),
            "Feedback": (
                "Fix the program so it exactly reproduces the failing transitions "
                "above WITHOUT breaking the ones it already predicts correctly. "
                "Prefer revising the general rules over adding special cases."),
        })
        return {COMPONENT: records}


# ---------------------------------------------------------------------------
# Free scoring / ID helpers (also used by the end-of-run test eval)
# ---------------------------------------------------------------------------
def score_program(adapter_or_timeout, code: str, items: list) -> dict:
    """Fit stats of `code` on prepared items (fresh runtime, no cache)."""
    timeout = adapter_or_timeout if isinstance(adapter_or_timeout, float) else 1.0
    rt = prt.ProgramRuntime(code, timeout_s=timeout)
    try:
        results = rt.score_buffer(items)
    finally:
        rt.close()
    st = prt.fit_stats(results)
    st["balanced"] = prt.balanced_score(st)
    st["cell_f1_all"] = (sum(r.cell_f1 for r in results) / len(results)) if results else 0.0
    return st


def choice_consistent(rt: prt.ProgramRuntime, it, choice: str, memo: dict) -> bool:
    """Is `choice` consistent with the observed transition under T-hat?
    Bare 'click' (collapsed protocol) uses click_enum semantics: consistent iff
    ANY cell's click reproduces the recorded next grid."""
    key = (it.idx, choice)
    if key in memo:
        return memo[key]
    a = prt.parse_action(choice)
    if a[0] == "click" and a[1] is None:
        ok = False
        for r in range(len(it.grid)):
            for c in range(len(it.grid[0]) if it.grid else 0):
                pred, err = rt.transition(it.prev, it.grid, ("click", r, c))
                if err is None and prt.canon_grid(pred) == it.next_c:
                    ok = True
                    break
            if ok:
                break
    else:
        pred, err = rt.transition(it.prev, it.grid, a)
        ok = err is None and prt.canon_grid(pred) == it.next_c
    memo[key] = ok
    return ok


def program_id_eval(code: str, items: list, action_pool: list, k: int,
                    collapse: bool, rng: random.Random, timeout_s: float = 1.0) -> dict:
    """Forward-simulation inverse dynamics on baked choice sets.
    strict = 1 iff the consistent set is exactly {truth};
    set_credit = 1/|S| if truth in S else 0 (id_set_metrics semantics)."""
    rt = prt.ProgramRuntime(code, timeout_s=timeout_s)
    memo: dict = {}
    rows = []
    try:
        for it in items:
            truth = it.action_str.split()[0] if collapse else it.action_str
            choices = make_choices(truth, action_pool, k, rng)
            consistent = [c for c in choices if choice_consistent(rt, it, c, memo)]
            strict = 1.0 if consistent == [truth] else 0.0
            credit = (1.0 / len(consistent)) if truth in consistent else 0.0
            rows.append({"truth": truth, "choices": choices,
                         "consistent": consistent, "strict": strict,
                         "set_credit": credit})
    finally:
        rt.close()
    n = max(1, len(rows))
    return {
        "n": len(rows),
        "strict": sum(r["strict"] for r in rows) / n,
        "set_credit": sum(r["set_credit"] for r in rows) / n,
        "mean_set_size": sum(len(r["consistent"]) for r in rows) / n,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Data pipeline (load -> backfill -> strip; NO collapse)
# ---------------------------------------------------------------------------
def load_split(run_arg: str, source_arg: str | None, whitelist, context_k: int) -> list:
    run_dirs = [Path(p) for p in run_arg.split(",") if p.strip()]
    if source_arg:
        source_dirs = [Path(p) for p in source_arg.split(",") if p.strip()]
        if len(source_dirs) == len(run_dirs):
            transitions = []
            for target_dir, source_dir in zip(run_dirs, source_dirs):
                t = load_transitions([target_dir], whitelist, context_k=context_k)
                backfill_context_from_source(t, [source_dir], whitelist, context_k=context_k)
                transitions.extend(t)
        else:
            transitions = load_transitions(run_dirs, whitelist, context_k=context_k)
            backfill_context_from_source(transitions, source_dirs, whitelist,
                                         context_k=context_k)
    else:
        transitions = load_transitions(run_dirs, whitelist, context_k=context_k)
    strip_transitions_obs_metadata(transitions)
    return transitions


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="comma-separated train dirs")
    ap.add_argument("--test-run", default=None, help="comma-separated held-out test dirs")
    ap.add_argument("--context-source-run", default=None)
    ap.add_argument("--test-context-source-run", default=None)
    ap.add_argument("--actions", required=True, help="comma-separated verb whitelist")
    ap.add_argument("--collapse-action-params", action="store_true",
                    help="collapse 'click R C'->'click' in the TEST ID choice protocol "
                         "only (the program buffer always keeps full actions)")
    ap.add_argument("--context-k", type=int, default=9)
    ap.add_argument("--train-n", type=int, default=-1,
                    help="-1 = full train buffer (scoring is free)")
    ap.add_argument("--val-n", type=int, default=30,
                    help="disjoint stratified val carve (ship gate); 0 disables")
    ap.add_argument("--test-n", type=int, default=50)
    ap.add_argument("--k-choices", type=int, default=5)
    ap.add_argument("--max-proposals", type=int, default=60,
                    help="LLM budget in reflection calls (the parity currency)")
    ap.add_argument("--selector", choices=["rex_pure"], default="rex_pure",
                    help="faithful REx (no gate); the only supported selector.")
    ap.add_argument("--rex-c", type=float, default=5.0)
    ap.add_argument("--program-timeout", type=float, default=1.0)
    ap.add_argument("--init-samples", type=int, default=7)
    ap.add_argument("--start-program", default=None,
                    help="warm-start from a previously LEARNED program file")
    ap.add_argument("--reflection-model", default="openai/gpt-oss-120b")
    ap.add_argument("--client", default="openrouter")
    ap.add_argument("--reflection-client", default=None,
                    help="litellm provider prefix for the reflection calls (default: --client); "
                         "'vllm' = OpenAI-compatible endpoint via HOSTED_VLLM_API_BASE, e.g. "
                         "the local Claude CLI proxy (scripts/claude_cli_proxy.py)")
    ap.add_argument("--reflection-provider-order", default=None)
    ap.add_argument("--reflection-hedge-delay", type=float, default=60.0)
    ap.add_argument("--reflection-timeout", type=float, default=240.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="re-attach to <out-dir>/wc_run_seed<seed>/resume_state.json (rex_search "
                         "node-level checkpoint) instead of starting the search from scratch")
    args = ap.parse_args()

    os.environ.setdefault("LLM_TIMEOUT_S", "240")
    os.environ.setdefault("LLM_HEDGE_DELAY_S", "60")

    rng = random.Random(args.seed)
    refl_cfg = make_config(
        args.reflection_model, args.reflection_client or args.client,
        provider_order=args.reflection_provider_order,
        hedge_delay_s=args.reflection_hedge_delay,
        timeout_s=args.reflection_timeout,
    )
    whitelist = set(filter(None, args.actions.split(","))) or None
    context_k = args.context_k

    transitions = load_split(args.run, args.context_source_run, whitelist, context_k)
    rng.shuffle(transitions)
    n_total = len(transitions)
    # train keeps priority on small pools: val is capped at a third of the pool
    # (val_n = min(30, n-5) handed val 25/30 of a 30-transition game and left
    # train=5 -- backwards; the ship gate needs less data than the learner).
    val_n = min(args.val_n, n_total // 3)
    train_target = (n_total - val_n) if args.train_n < 0 else min(args.train_n, n_total - val_n)
    if val_n > 0:
        train_tr, val_tr = stratified_split(transitions, train_target, val_n, rng)
    else:
        train_tr, val_tr = transitions[:train_target], []
    train_items = prt.prepare_transitions(train_tr, context_k)
    for i, it in enumerate(train_items):
        it.idx = i
    val_items = prt.prepare_transitions(val_tr, context_k)
    for i, it in enumerate(val_items):
        it.idx = 10_000_000 + i

    outd = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parents[1] / "logs" / "perc_invdyn_wc"
        / f"{time.strftime('%Y%m%d-%H%M%S')}_seed{args.seed}")
    outd.mkdir(parents=True, exist_ok=True)
    run_dir = outd / f"wc_run_seed{args.seed}"
    print(f"[out] artifacts + rex run_dir -> {outd}")

    avg_prev = sum(len(it.prev) for it in train_items) / max(1, len(train_items))
    print(f"transitions: total={n_total} train={len(train_items)} val={len(val_items)} "
          f"| context_k={context_k} avg_prev={avg_prev:.1f}")
    print(f"train action balance: "
          f"{dict(sorted(Counter(it.action[0] for it in train_items).items()))}")

    # ---- floors / ceilings ------------------------------------------------
    ident = score_program(args.program_timeout, prt.IDENTITY_PROGRAM, train_items)
    ceil = {k: prt.determinism_ceiling(train_items, k) for k in (0, 1, context_k)}
    print(f"[floors] identity: balanced={ident['balanced']:.3f} "
          f"(fit_changed={ident['fit_changed']:.2f} fit_static={ident['fit_static']:.2f}) "
          f"| determinism ceiling k0/k1/k{context_k} = "
          f"{ceil[0]:.3f}/{ceil[1]:.3f}/{ceil[context_k]:.3f}")

    # ---- REx-pure optimization -------------------------------------------
    seed_code = Path(args.start_program).read_text() if args.start_program else ""
    if args.start_program:
        print(f"[warm-start] seed program <- {args.start_program}")
    seed_candidate = {COMPONENT: seed_code}

    adapter = ProgramWMAdapter(train_items, timeout_s=args.program_timeout,
                               init_samples=args.init_samples)
    selector = RExPureCandidateSelector(c=args.rex_c, rng=random.Random(args.seed + 7331))
    sel_desc = f"REx-pure(C={args.rex_c:g}, no gate, h=full-train)"

    # budget = nodes explored (evaluated candidates). Under rex_pure/always-accept each
    # proposal admits one child = one node, plus the seed; +2 headroom for the seed and
    # any skip-perfect slack (worldcoder programs can hit fit=1.0).
    max_nodes = args.max_proposals + 2
    refl_before = dict(_REFLECTION)
    print(f"[wc] optimizing: selector={sel_desc} | max_proposals={args.max_proposals} "
          f"(max_nodes={max_nodes}) | reflection_lm={args.reflection_model} "
          f"| h = 0.5*fit_changed + 0.5*fit_static + identity gate")
    t0 = time.perf_counter()
    # selection h = full-train (REx paper semantics); the disjoint val carve is used
    # AFTER the run as the (free) ship gate over the whole pool.
    result = rex_search(
        adapter=adapter,
        seed_candidate=seed_candidate,
        train=train_items,
        reflection_lm=make_reflection_lm(refl_cfg, log_path=str(run_dir / "reflection_calls.jsonl")),
        templates={COMPONENT: WC_REFINE_TMPL},
        selector=selector,
        module_selector=SingleComponentSelector(COMPONENT),
        max_nodes=max_nodes,
        run_dir=run_dir,
        log_prefix="wc",
        resume=args.resume,
        # cap iterations so an all-perfect pool (programs can fit every transition
        # -> perfect -> skipped, costing no budget) can't spin forever.
        max_iters=20 * args.max_proposals + 100,
    )
    wall_s = time.perf_counter() - t0
    refl_calls = _REFLECTION["calls"] - refl_before["calls"]
    refl_cost = _REFLECTION["cost"] - refl_before["cost"]

    # ---- ship gate: rescore the WHOLE pool on the held-out val (free) ------
    pool = [{COMPONENT: _clean_program(c.get(COMPONENT, ""))} for c in result["pool"]]
    pool_rows = []
    for i, cand in enumerate(pool):
        code = cand[COMPONENT]
        train_st = score_program(args.program_timeout, code, train_items) if code.strip() else None
        val_st = (score_program(args.program_timeout, code, val_items)
                  if code.strip() and val_items else None)
        pool_rows.append({
            "idx": i, "chars": len(code),
            "train_balanced": train_st["balanced"] if train_st else 0.0,
            "train_fit_changed": train_st["fit_changed"] if train_st else 0.0,
            "val_balanced": val_st["balanced"] if val_st else None,
            "val_fit_changed": val_st["fit_changed"] if val_st else None,
        })
    if val_items:
        ship_key = lambda r: (r["val_balanced"] or 0.0, -r["chars"])  # noqa: E731
    else:
        ship_key = lambda r: (r["train_balanced"], -r["chars"])  # noqa: E731
    ship = max(pool_rows, key=ship_key)
    best_code = pool[ship["idx"]][COMPONENT]
    print(f"[wc] pool={len(pool)} candidates | shipped #{ship['idx']} "
          f"(val_balanced={ship['val_balanced']}, train_balanced={ship['train_balanced']:.3f}, "
          f"{ship['chars']} chars) | rex best_idx={result['best_idx']}")
    (run_dir / "pool_val_scores.json").write_text(json.dumps(pool_rows, indent=2))

    # ---- clean test eval ---------------------------------------------------
    test_summary = None
    if args.test_run:
        test_tr = load_split(args.test_run, args.test_context_source_run, whitelist, context_k)
        rng2 = random.Random(args.seed + 17)
        rng2.shuffle(test_tr)
        test_tr = test_tr[: args.test_n if args.test_n >= 0 else len(test_tr)]
        test_items = prt.prepare_transitions(test_tr, context_k)
        for i, it in enumerate(test_items):
            it.idx = 20_000_000 + i
        fd = score_program(args.program_timeout, best_code, test_items)
        fd_stale = score_program(args.program_timeout, prt.IDENTITY_PROGRAM, test_items)
        if args.collapse_action_params:
            action_pool = sorted({t.action.split()[0] for t in train_tr}
                                 | {t.action.split()[0] for t in test_tr})
        else:
            action_pool = sorted({t.action for t in train_tr} | {t.action for t in test_tr})
        idr = program_id_eval(best_code, test_items, action_pool, args.k_choices,
                              args.collapse_action_params, random.Random(args.seed + 29),
                              timeout_s=args.program_timeout)
        print(f"[wc] CLEAN test program-FD exact={fd['fit_all']:.3f} "
              f"(changed={fd['fit_changed']:.2f} static={fd['fit_static']:.2f} "
              f"cell_f1={fd['cell_f1_all']:.3f}) | stale-floor exact={fd_stale['fit_all']:.3f}")
        print(f"[wc] CLEAN test program-ID strict={idr['strict']:.3f} "
              f"set_credit={idr['set_credit']:.3f} mean_set_size={idr['mean_set_size']:.2f} "
              f"(k={args.k_choices}, pool={len(action_pool)}, "
              f"collapse={args.collapse_action_params})")
        test_summary = {
            "n_test": idr["n"],
            "program_fd": {k: fd[k] for k in
                           ("fit_all", "fit_changed", "fit_static", "cell_f1_all")},
            "stale_fd_exact": fd_stale["fit_all"],
            "program_id": {k: idr[k] for k in ("strict", "set_credit", "mean_set_size")},
            "id_protocol": {"k": args.k_choices, "collapse": args.collapse_action_params,
                            "pool_size": len(action_pool)},
        }
        (outd / f"test_id_rows_wc_seed{args.seed}.json").write_text(
            json.dumps(idr["rows"], indent=2))

    # ---- artifacts ---------------------------------------------------------
    tag = f"wc_seed{args.seed}"
    (outd / f"best_transition_{tag}.py").write_text(best_code)
    summary = {
        "arm": "worldcoder_program_wm",
        "train": {"n": len(train_items), "identity_floor": ident["balanced"],
                  "determinism_ceiling": ceil,
                  "shipped_train_balanced": ship["train_balanced"],
                  "shipped_train_fit_changed": ship["train_fit_changed"]},
        "val": {"n": len(val_items), "shipped_val_balanced": ship["val_balanced"]},
        "test": test_summary,
        "pool": {"n_candidates": len(pool), "shipped_idx": ship["idx"],
                 "rex_best_idx": result["best_idx"],
                 "nodes_explored": result["nodes_explored"]},
        "budget": {"max_proposals": args.max_proposals, "reflection_calls": refl_calls,
                   "reflection_cost_usd": round(refl_cost, 4),
                   "task_llm_cost_usd": 0.0, "wall_s": round(wall_s, 1)},
        "config": {"context_k": context_k, "selector": args.selector,
                   "rex_c": args.rex_c, "program_timeout": args.program_timeout,
                   "reflection_model": args.reflection_model, "seed": args.seed,
                   "actions": args.actions,
                   "collapse_action_params": args.collapse_action_params},
    }
    (outd / f"test_summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(f"[wc] reflection cost=${refl_cost:.4f} ({refl_calls} calls) | "
          f"task-LLM cost=$0 | wall={wall_s:.0f}s")
    print(f"[wc] shipped program -> best_transition_{tag}.py | summary -> "
          f"test_summary_{tag}.json")


if __name__ == "__main__":
    main()
