"""Standalone REx-pure optimizer for perception P (+ world knowledge B).

A standalone REx-pure perception/belief optimizer. It shares ALL the heavy lifting
with the other optimizers via invdyn_core.py (the InvDynAdapter scorer, the REx-pure /
component selectors, the reflection LM + templates, eval_on / eval_fd_on, and data
baking); the loop itself lives in invdyn_core.rex_search.

REx-pure (Tang et al. 2024) deliberately keeps only the essentials of the search
dependency -- the Pareto frontier (parent selection is Thompson sampling), the
acceptance gate (every child is admitted), the val set (val is aliased to train),
and minibatch subsampling (each candidate is one full-train eval). What remains of
engine -- a ~40-line driver loop:

  1. Score the seed on the full train batch (h = mean train composite). This is node 1.
  2. Loop until `max_nodes` candidates have been evaluated:
       a. select a parent by REx Thompson sampling over {h_i} (progressive widening);
       b. if the parent already scores perfectly on every train row, skip (matches
          skip_perfect_score) -- the expansion still counts but adds no node;
       c. pick the component to mutate (perception, or beliefs every Nth selection);
       d. build the reflective dataset from the parent's cached traces, render the
          reflection prompt, call the reflection LM, extract the new component text;
       e. score the child on the full train batch (one fresh eval) and admit it
          unconditionally as a new arm -- one more node explored.
  3. Ship argmax_i h_i (best full-train score).

Budget = nodes explored (evaluated candidates): the seed plus one per admitted child.
The parent re-eval for reflection is served from the adapter trace cache (kept on
batches[i]) and costs no fresh decode, so it does not add a node.

Usage (rex_pure-relevant flags):
  uv run offline_learning/rexpure_optimize.py \
      --run "logs/.../BT3GB/..." --task-model google/gemini-2.5-flash \
      --max-nodes 15 --rex-c 5
"""

import argparse
import asyncio
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

from invdyn_core import *  # noqa: F401,F403
from invdyn_core import _REFLECTION, _clean_component, Image


# ---------------------------------------------------------------------------
# The data pipeline is a fixed action-balanced full-train batch (same rng stream,
# same balanced_split / bake_choices calls) so a same-seed run sees an identical
# train/test set and identical baselines -- the basis for parity.
# ---------------------------------------------------------------------------
def build_data(args, rng):
    """Build the EXACT train/test batches (+ action pool, decoys, id universe) a run uses.

    Extracted from main() so an offline analysis can rebuild a completed run's batch
    byte-identically instead of re-deriving the RNG-order-sensitive split (shuffle ->
    test split -> train split -> bake_choices -> bake_decoys). Correctness is checkable:
    invdyn_core._train_fingerprint(train) must match the `train_fingerprint` the run
    checkpointed in resume_state.json.

    Returns (train, test, action_pool, context_k, whitelist, transitions, id_n)."""
    whitelist = set(filter(None, args.actions.split(","))) or None
    context_k = args.context_k

    def _collapse(trs):
        if not args.keep_obs_metadata:
            strip_transitions_obs_metadata(trs)
        if args.collapse_action_params:
            for t in trs:
                t.action = t.action.split()[0]
                t.ctx_prev = [(s, a.split()[0]) for s, a in t.ctx_prev]
                t.ctx_next = [(a.split()[0], s) for a, s in t.ctx_next]
        return trs

    # ---- load transitions (+ optional context backfill) ----
    run_dirs = [Path(p) for p in args.run.split(",") if p.strip()]
    if args.context_source_run:
        csrc = [Path(p) for p in args.context_source_run.split(",") if p.strip()]
        if len(csrc) == len(run_dirs):
            transitions = []
            for tdir, sdir in zip(run_dirs, csrc):
                tt = load_transitions([tdir], whitelist, context_k=context_k)
                backfill_context_from_source(tt, [sdir], whitelist, context_k=context_k)
                transitions.extend(tt)
        else:
            transitions = load_transitions(run_dirs, whitelist, context_k=context_k)
            backfill_context_from_source(transitions, csrc, whitelist, context_k=context_k)
    else:
        transitions = load_transitions(run_dirs, whitelist, context_k=context_k)
    transitions = _collapse(transitions)

    # ---- train / test split (rex_pure branch: no val; train == scoring set) ----
    if args.test_run:
        test_dirs = [Path(p) for p in args.test_run.split(",") if p.strip()]
        if args.test_context_source_run:
            tcs = [Path(p) for p in args.test_context_source_run.split(",") if p.strip()]
            if len(tcs) == len(test_dirs):
                test_transitions = []
                for tdir, sdir in zip(test_dirs, tcs):
                    tt = load_transitions([tdir], whitelist, context_k=context_k)
                    backfill_context_from_source(tt, [sdir], whitelist, context_k=context_k)
                    test_transitions.extend(tt)
            else:
                test_transitions = load_transitions(test_dirs, whitelist, context_k=context_k)
                backfill_context_from_source(test_transitions, tcs, whitelist, context_k=context_k)
        else:
            test_transitions = load_transitions(test_dirs, whitelist, context_k=context_k)
        test_transitions = _collapse(test_transitions)
        action_pool = sorted({t.action for t in transitions} | {t.action for t in test_transitions})
        train_pool = list(transitions)
        rng.shuffle(train_pool)
        test_pool = list(test_transitions)
        rng.shuffle(test_pool)
        test_n = args.test_n if args.test_n >= 0 else 10**9
        _, test_tr = balanced_split(test_pool, test_n, 10**9, rng)
        _, train_tr = balanced_split(train_pool, args.train_n, 10**9, rng)
        print(f"[cross-traj] train <- {args.run} | test <- {args.test_run}")
    else:
        action_pool = sorted({t.action for t in transitions})
        pin_idx = sorted({int(x) for x in args.pin_train_idx.split(",") if x.strip()})
        pinned = [transitions[i] for i in pin_idx if 0 <= i < len(transitions)]
        pinned_ids = {id(t) for t in pinned}
        pool = [t for t in transitions if id(t) not in pinned_ids]
        rng.shuffle(pool)
        rest, test_tr = balanced_split(pool, args.test_n, 10**9, rng)
        _, train_core = balanced_split(rest, args.train_n, 10**9, rng)
        train_tr = pinned + train_core
        if pinned:
            print(f"[pin-train] forced {len(pinned)} key transition(s) into train (orig idx {pin_idx})")

    k = args.k_choices
    train = bake_choices(train_tr, action_pool, k, rng)
    test = bake_choices(test_tr, action_pool, k, rng)

    if args.contrastive_fd:
        cfd_rng = random.Random(args.seed + 9173)
        short = bake_decoys(train, transitions, args.cfd_decoys, cfd_rng, hard=args.cfd_hard_decoys)
        n_frames = len({(x or "").strip() for t2 in transitions for x in (t2.x_t, t2.x_t1)})
        print(f"[contrastive-fd] {args.cfd_decoys} decoys/item from {n_frames} pool frames"
              + (" | hard" if args.cfd_hard_decoys else "")
              + (f" | {short} short" if short else ""))

    id_n = None
    if args.id_set_loss:
        id_n, id_grid = compute_action_universe(
            whitelist, args.collapse_action_params, transitions, override=args.id_n_actions)
        print(f"[id-set] eps={args.id_eps} N={id_n}" + (f" (grid {id_grid}x{id_grid})" if id_grid else ""))

    print(f"transitions: {len(transitions)} | train={len(train)} test={len(test)} "
          "(rex_pure: train==scoring set, no val)")
    if context_k > 0:
        avg_prev = sum(len(t.ctx_prev) for t in transitions) / len(transitions)
        avg_next = sum(len(t.ctx_next) for t in transitions) / len(transitions)
        print(f"temporal window K={context_k} | avg ctx_prev={avg_prev:.1f} avg ctx_next={avg_next:.1f}")
    print(f"train action balance: {dict(Counter(t['tr'].action.split()[0] for t in train))}")
    print(f"test action balance: {dict(Counter(t['tr'].action for t in test))}")
    print(f"action pool ({len(action_pool)}): {action_pool}")
    print(f"task_lm (F) = {args.task_model} | reflection_lm = {args.reflection_model}\n")

    return train, test, action_pool, context_k, whitelist, transitions, id_n


# ---------------------------------------------------------------------------
# The CLI lives in its own function so an offline analysis can re-parse a completed
# run's saved argv through the SAME parser (inheriting every default) instead of
# hand-copying flags that then drift. See scripts/replay_proposal_ab.py.
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="comma-separated run dirs (pooled)")
    ap.add_argument("--keep-obs-metadata", action="store_true",
                    help="keep the recorded Task:/Step:/Phase: harness header (default: strip)")
    ap.add_argument("--test-run", default=None,
                    help="comma-separated run dirs for a cross-trajectory held-out TEST set")
    ap.add_argument("--context-source-run", default=None,
                    help="original full-traj dirs to backfill ctx for --run targets")
    ap.add_argument("--test-context-source-run", default=None,
                    help="original full-traj dirs to backfill ctx for --test-run targets")
    ap.add_argument("--train-n", type=int, default=20)
    ap.add_argument("--test-n", type=int, default=10)
    ap.add_argument("--k-choices", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--llm-timeout", type=float, default=None)
    ap.add_argument("--hedge-delay", type=float, default=None)
    ap.add_argument("--task-model", default="google/gemini-2.5-flash", help="frozen decoder F")
    ap.add_argument("--reflection-model", default=None, help="proposer LM (default: --task-model)")
    ap.add_argument("--task-provider-order", default=None)
    ap.add_argument("--reflection-provider-order", default=None)
    ap.add_argument("--task-reasoning-json", default=None,
                    help="OpenRouter reasoning override for F calls, e.g. "
                    "'{\"effort\": \"low\"}' (gpt-oss) or '{\"effort\": \"none\"}' "
                    "(disable qwen3.7-flash hidden thinking)")
    ap.add_argument("--reflection-reasoning-json", default=None,
                    help="same, for reflection/analysis calls only")
    ap.add_argument("--analysis-reasoning-json", default=None,
                    help="override the reasoning setting for the --analyze-mistakes "
                         "DIAGNOSIS calls only (default: follow --reflection-reasoning-json). "
                         "Lets a run keep thinking ON for diagnosis while turning it off for "
                         "the proposer, isolating which of the two actually needs it")
    ap.add_argument("--reflection-hedge-delay", type=float, default=None)
    ap.add_argument("--reflection-timeout", type=float, default=None)
    ap.add_argument("--client", default="openrouter")
    ap.add_argument("--reflection-client", default=None,
                    help="litellm provider prefix for the reflection/analysis calls only "
                         "(default: --client). 'vllm' = any OpenAI-compatible endpoint via "
                         "litellm's hosted_vllm provider, e.g. the local Claude CLI proxy "
                         "(scripts/claude_cli_proxy.py) with HOSTED_VLLM_API_BASE set")
    ap.add_argument("--max-nodes", type=int, default=67,
                    help="search budget = number of evaluated candidates (nodes): the seed "
                    "plus admitted children. Default 67 = the canonical rex_pure production "
                    "budget (aug5/aug7 runs = 2000 old metric-calls / train_n 30); covers the "
                    "slow-climbing games (n2ntd/s2kt7 reach 95%% of final train ~node 54).")
    ap.add_argument("--actions", default="left,right,up,down,noop,click")
    ap.add_argument("--collapse-action-params", action="store_true")
    ap.add_argument("--good-baseline", action="store_true",
                    help="also report the DQ8GC-specific single-cell GOOD_P ceiling")
    ap.add_argument("--pin-train-idx", default="",
                    help="comma-separated ORIGINAL transition indices forced into train")
    ap.add_argument("--image-mode", action="store_true")
    ap.add_argument("--cell-px", type=int, default=8)
    ap.add_argument("--f-image", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--start-perception", default="",
                    help="warm-start P (learned artifacts only)")
    ap.add_argument("--start-beliefs", default="",
                    help="warm-start B (learned artifacts only)")
    ap.add_argument("--belief-update-period", type=int, default=4,
                    help="update beliefs once every N component selections (else perception)")
    ap.add_argument("--no-perception", action="store_true",
                    help="ABLATION: do not learn a perception. P is frozen to the identity "
                         "module (offline_learning/identity_perception.py), so every prompt "
                         "shows the RAW GRID where it would otherwise show P's features, and "
                         "the search mutates ONLY the world-knowledge block. Mutually "
                         "exclusive with --start-perception (and ignores "
                         "--belief-update-period, which has no other component to cede to)")
    ap.add_argument("--fd-scorer", choices=["none", "textdiff", "judge", "exact"], default="exact")
    ap.add_argument("--fd-weight", type=float, default=0.5)
    ap.add_argument("--id-set-loss", action="store_true")
    ap.add_argument("--id-eps", type=float, default=0.1)
    ap.add_argument("--id-n-actions", type=int, default=None)
    ap.add_argument("--credited-scoring", action="store_true")
    ap.add_argument("--analysis-memo", action=argparse.BooleanOptionalAction, default=True,
                    help="memoize --analyze-mistakes diagnosis calls on the exact prompt, so a "
                         "re-selected parent does not re-run an identical diagnosis (14-26%% of "
                         "them in the aug8 runs). --no-analysis-memo restores always-fresh "
                         "diagnoses (more proposal diversity per repeat visit, slower)")
    ap.add_argument("--reflect-max-failures", type=int, default=8,
                    help="failing transitions shown to the proposer/analysis per iteration "
                         "(default 8 = historical). Lower = smaller reflection prompt = less "
                         "prefill latency and cost, at the cost of evidence breadth")
    ap.add_argument("--reflect-raw-prefix", type=int, default=1500,
                    help="chars of each raw observation pasted into the reflection prompt as "
                         "an orientation hint (default 1500 = historical; 0 = omit). Two per "
                         "shown mistake, so this is ~31%% of the aug8 proposer prompt")
    ap.add_argument("--propose-batch", type=int, default=1,
                    help="iterations (analysis+proposer+eval) run concurrently per round. "
                         "1 = the serial loop. B>1 draws B parents against the same "
                         "one-round-stale pool (batched Thompson) and cuts wall-clock ~B x "
                         "at the same node budget; each in-flight eval opens its own "
                         "--concurrency fan-out, so keep B*concurrency within provider limits")
    ap.add_argument("--rex-c", type=float, default=5.0,
                    help="REx prior strength C: low=explore, high=exploit. [5,10] validated.")
    ap.add_argument("--composite", choices=["additive", "min", "softmin"], default="additive")
    ap.add_argument("--softmin-tau", type=float, default=0.25,
                    help="Boltzmann temperature for --composite softmin (tau->0 = hard min)")
    ap.add_argument("--contrastive-fd", action="store_true")
    ap.add_argument("--cfd-decoys", type=int, default=3)
    ap.add_argument("--cfd-hard-decoys", action="store_true")
    ap.add_argument("--cfd-raw-targets", action="store_true")
    ap.add_argument("--fd-reflect", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--analyze-mistakes", action="store_true")
    ap.add_argument("--context-k", type=int, default=3)
    ap.add_argument("--analyze-mode", choices=["combined", "per-mistake"], default="combined")
    ap.add_argument("--log-reflection", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run in --out-dir from its last checkpointed "
                    "node (rex_search resume_state.json / resume_batches.jsonl) instead of "
                    "restarting from the seed; test baselines are reloaded, not recomputed")
    ap.add_argument("--out-dir", default=None)
    return ap


# ---------------------------------------------------------------------------
# main(): argparse + data build + baselines + loop + test eval + summary.
# ---------------------------------------------------------------------------
def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.reflection_model is None:
        args.reflection_model = args.task_model

    if args.llm_timeout is not None:
        os.environ["LLM_TIMEOUT_S"] = str(args.llm_timeout)
    else:
        os.environ.setdefault("LLM_TIMEOUT_S", "90")
    if args.hedge_delay is not None:
        os.environ["LLM_HEDGE_DELAY_S"] = str(args.hedge_delay)
    else:
        os.environ.setdefault("LLM_HEDGE_DELAY_S", "30")
    if args.id_set_loss and (args.image_mode or args.f_image):
        parser.error("--id-set-loss does not support --image-mode/--f-image")
    if args.credited_scoring and args.f_image:
        parser.error("--credited-scoring does not support --f-image")
    if args.composite in ("min", "softmin") and args.credited_scoring:
        parser.error(f"--composite {args.composite} and --credited-scoring are mutually exclusive")
    if args.contrastive_fd and args.f_image:
        parser.error("--contrastive-fd does not support --f-image")

    rng = random.Random(args.seed)
    task_cfg = make_config(args.task_model, args.client,
                           provider_order=args.task_provider_order,
                           reasoning_json=args.task_reasoning_json)
    refl_client = args.reflection_client or args.client
    refl_cfg = make_config(
        args.reflection_model, refl_client,
        provider_order=args.reflection_provider_order,
        hedge_delay_s=args.reflection_hedge_delay, timeout_s=args.reflection_timeout,
        reasoning_json=args.reflection_reasoning_json,
    )
    # The DIAGNOSIS call is the reasoning-heaviest of the two (aug8 dsv4flash: ~10.6k
    # reasoning tokens for a ~320-token answer) AND the one whose quality the proposer
    # merely transcribes, so it carries the most risk when thinking is disabled. Keep its
    # reasoning setting separately switchable to isolate that risk; default = the
    # proposer's, i.e. one flag still moves both.
    analysis_cfg = refl_cfg if args.analysis_reasoning_json is None else make_config(
        args.reflection_model, refl_client,
        provider_order=args.reflection_provider_order,
        hedge_delay_s=args.reflection_hedge_delay, timeout_s=args.reflection_timeout,
        reasoning_json=args.analysis_reasoning_json,
    )
    (train, test, action_pool, context_k, whitelist, transitions,
     id_n) = build_data(args, rng)

    # Scope the proposer's OBSERVATION SCHEMA to the format of the loaded frames (autumn vs
    # ARC) so the other env's palette never leaks into the proposed perception code.
    env_name = infer_env_name(transitions)

    if args.no_perception:
        if args.start_perception:
            parser.error("--no-perception freezes P to the identity module; "
                         "drop --start-perception")
        args.start_perception = str(Path(__file__).resolve().parent / "identity_perception.py")
        print(f"[no-perception] ABLATION: P frozen to identity ({args.start_perception}); "
              "prompts show the raw grid and only world_knowledge is learned")
    seed_code = Path(args.start_perception).read_text() if args.start_perception else ""
    if args.start_perception and not args.no_perception:
        print(f"[warm-start] seed perception <- {args.start_perception}")
    seed_beliefs = Path(args.start_beliefs).read_text() if args.start_beliefs else ""
    if args.start_beliefs:
        print(f"[warm-start] seed beliefs <- {args.start_beliefs}")
    seed_candidate = {"perception": seed_code, "world_knowledge": seed_beliefs}

    if args.out_dir:
        outd = Path(args.out_dir)
    else:
        repo_root = Path(__file__).resolve().parents[1]
        outd = repo_root / "logs" / "perc_invdyn" / f"{time.strftime('%Y%m%d-%H%M%S')}_rexpure_seed{args.seed}"
    outd.mkdir(parents=True, exist_ok=True)
    print(f"[out] artifacts + traces + run_dir -> {outd}")

    # ---- baselines on the CLEAN test split (not charged to the budget) ----
    # On --resume they are reloaded from resume_baselines.json rather than recomputed
    # (they cost test-set F calls but never change across a resume of the same split).
    chance = 1.0 / args.k_choices
    baselines_path = outd / "resume_baselines.json"
    if args.resume and baselines_path.exists():
        b = json.loads(baselines_path.read_text())
        raw_acc, start_acc, good_acc = b["raw_acc"], b["start_acc"], b.get("good_acc")
        print("[resume] reloaded test baselines (skipped recompute)")
    else:
        raw_acc, _ = run_async(eval_on(
            task_cfg, "", "", test, raw_mode=True, image_mode=args.image_mode, cell=args.cell_px,
            log_path=outd / f"test_trace_raw_seed{args.seed}.json",
            id_set_loss=args.id_set_loss, id_eps=args.id_eps, id_n_actions=id_n,
            credited_scoring=args.credited_scoring))
        start_acc, _ = run_async(eval_on(
            task_cfg, seed_code, "", test, context_k=context_k,
            id_set_loss=args.id_set_loss, id_eps=args.id_eps, id_n_actions=id_n,
            credited_scoring=args.credited_scoring))
        good_acc = None
        if args.good_baseline:
            good_acc, _ = run_async(eval_on(
                task_cfg, GOOD_P, "", test, context_k=context_k,
                id_set_loss=args.id_set_loss, id_eps=args.id_eps, id_n_actions=id_n,
                credited_scoring=args.credited_scoring))
        baselines_path.write_text(json.dumps(
            {"raw_acc": raw_acc, "start_acc": start_acc, "good_acc": good_acc}))
    print(f"[test baselines] random={chance:.2f} | start-P={start_acc:.2f} | raw-frame={raw_acc:.2f}"
          + (f" | single-cell GOOD_P={good_acc:.2f}" if good_acc is not None else "") + "\n")

    # ---- REx-pure optimization ----
    run_ctx = ThreadScopedCtx(iteration=None)
    run_dir = outd / f"rexpure_run_seed{args.seed}"
    adapter = InvDynAdapter(
        task_cfg, action_pool,
        analysis_cfg=analysis_cfg, concurrency=args.concurrency,
        image_mode=args.image_mode, cell=args.cell_px,
        fd_scorer=args.fd_scorer, fd_weight=args.fd_weight, fd_reflect=args.fd_reflect,
        analyze_mistakes=args.analyze_mistakes, analyze_mode=args.analyze_mode,
        pred_log_path=run_dir / "predictions.jsonl",
        analysis_log_path=(run_dir / "analysis_calls.jsonl" if args.analyze_mistakes else None),
        run_ctx=run_ctx, context_k=context_k, reuse_traces=True, f_image=args.f_image,
        gate_train_x=[inst["tr"].x_t for inst in train] + [inst["tr"].x_t1 for inst in train],
        id_set_loss=args.id_set_loss, id_eps=args.id_eps, id_n_actions=id_n,
        credited_scoring=args.credited_scoring, composite=args.composite,
        softmin_tau=args.softmin_tau,
        contrastive_fd=args.contrastive_fd, cfd_raw_targets=args.cfd_raw_targets,
        analysis_memo=args.analysis_memo,
        reflect_max_failures=args.reflect_max_failures,
        reflect_raw_prefix=args.reflect_raw_prefix,
        image_cls=Image,
    )
    templates = build_reflection_templates(env_name)
    selector = RExPureCandidateSelector(c=args.rex_c, rng=random.Random(args.seed + 7331))
    module_selector = (SingleComponentSelector("world_knowledge") if args.no_perception
                       else PerceptionBiasedComponentSelector(args.belief_update_period))
    refl_log = str(run_dir / "reflection_calls.jsonl") if args.log_reflection else None
    reflection_lm = make_reflection_lm(refl_cfg, log_path=refl_log)

    print(f"[rexpure] optimizing (max_nodes={args.max_nodes}, REx-pure C={args.rex_c:g}, "
          f"no gate, h=full-train, mb=full | context_k={context_k})...")
    t0 = time.perf_counter()
    result = rex_search(
        adapter=adapter, seed_candidate=seed_candidate, train=train,
        reflection_lm=reflection_lm, templates=templates,
        selector=selector, module_selector=module_selector,
        max_nodes=args.max_nodes, run_dir=run_dir,
        log_prefix="rexpure", resume=args.resume,
        propose_batch=args.propose_batch,
    )
    secs = time.perf_counter() - t0

    best = dict(result["best_candidate"])
    best_code = _clean_component("perception", best.get("perception", ""))
    best_beliefs = best.get("world_knowledge", "")
    print(f"\n[rexpure] best full-train score = {result['best_train_score']:.4f} | "
          f"candidates explored = {result['num_candidates']} | "
          f"nodes explored = {result['nodes_explored']} | iterations = {result['iterations']}")

    test_acc, _ = run_async(eval_on(
        task_cfg, best_code, best_beliefs, test,
        log_path=outd / f"test_trace_rexpure_seed{args.seed}.json", context_k=context_k,
        id_set_loss=args.id_set_loss, id_eps=args.id_eps, id_n_actions=id_n,
        credited_scoring=args.credited_scoring))
    cost = adapter.total_cost + _REFLECTION["cost"]
    print(f"[rexpure] CLEAN test acc (inverse) = {test_acc:.2f}")

    fd_test = None
    if args.fd_scorer != "none":
        fd_test, fd_cost = run_async(eval_fd_on(
            task_cfg, best_code, best_beliefs, test, args.fd_scorer, args.concurrency,
            context_k=context_k, log_path=outd / f"test_trace_fd_rexpure_seed{args.seed}.json",
            credited_scoring=args.credited_scoring))
        cost += fd_cost
        print(f"[rexpure] CLEAN test FD[{args.fd_scorer}] = {fd_test:.2f}")

    summary = {
        "inverse_accuracy": test_acc,
        "forward_score": fd_test,
        "forward_scorer": args.fd_scorer if args.fd_scorer != "none" else None,
        "n_test": len(test),
        "context_k": context_k,
        "best_train_score": result["best_train_score"],
        "num_candidates": result["num_candidates"],
        "nodes_explored": result["nodes_explored"],
        "inverse_trace": f"test_trace_rexpure_seed{args.seed}.json",
        "forward_trace": (f"test_trace_fd_rexpure_seed{args.seed}.json" if fd_test is not None else None),
        "id_metric": "set" if args.id_set_loss else "exact-match",
    }
    if args.id_set_loss:
        trace = json.loads((outd / f"test_trace_rexpure_seed{args.seed}.json").read_text())
        summary["inverse_set"] = {
            "eps": args.id_eps, "n_actions": id_n, "hit_rate": trace["hit_rate"],
            "mean_set_size": trace["mean_set_size"], "mean_loss": trace["mean_loss"],
            "strict_singleton_accuracy": trace["strict_singleton_accuracy"],
        }
        print(f"[rexpure] ID-set: hit_rate={trace['hit_rate']:.2f} "
              f"mean_set_size={trace['mean_set_size']:.2f} mean_loss={trace['mean_loss']:.3f} "
              f"strict={trace['strict_singleton_accuracy']:.2f}")
    if args.credited_scoring:
        inv_trace = json.loads((outd / f"test_trace_rexpure_seed{args.seed}.json").read_text())
        summary["credited"] = {
            "credited_id_acc": inv_trace.get("credited_id_acc"),
            "blind_id_floor_mean": inv_trace.get("blind_id_floor_mean"),
        }
        if fd_test is not None:
            fd_trace = json.loads((outd / f"test_trace_fd_rexpure_seed{args.seed}.json").read_text())
            summary["credited"]["credited_fd_acc"] = fd_trace.get("credited_fd_acc")
            summary["credited"]["blind_fd_floor_mean"] = fd_trace.get("blind_fd_floor_mean")
    (outd / f"test_summary_rexpure_seed{args.seed}.json").write_text(json.dumps(summary, indent=2))

    print(f"[rexpure] F (task_lm) cost=${adapter.total_cost:.4f} ({adapter.eval_calls} fresh F evals, "
          f"{adapter.reused_evals} reused) | reflection cost=${_REFLECTION['cost']:.4f} "
          f"({_REFLECTION['calls']} calls)")
    acs = analysis_cache_stats()
    if acs["calls"]:
        print(f"[rexpure] diagnosis calls={acs['calls']} served from memo={acs['hits']} "
              f"({100 * acs['hits'] / acs['calls']:.0f}% of re-selected parents)")
    hs = llm_hedge_stats()
    print(f"[rexpure] LLM requests={hs['calls']} hedged={hs['hedged']} hedge_wins={hs['hedge_wins']} "
          f"retries={hs['retries']} (timeout={os.environ.get('LLM_TIMEOUT_S')}s, "
          f"hedge_delay={os.environ.get('LLM_HEDGE_DELAY_S')}s)")

    tag = f"rexpure_seed{args.seed}"
    (outd / f"best_perception_{tag}.py").write_text(best_code)
    (outd / f"best_beliefs_{tag}.txt").write_text(best_beliefs)
    print(f"[rexpure] best P -> best_perception_{tag}.py | best B -> best_beliefs_{tag}.txt")

    print("\n=== REx-pure (clean test) ===")
    print(f"task_lm(F)={args.task_model}  reflection_lm={args.reflection_model}  seed={args.seed}")
    print(f"{'method':<34} {'test_acc':>8} {'cost($)':>9} {'time(s)':>8}")
    print(f"{'raw-frame ref':<34} {raw_acc:>8.2f} {'-':>9} {'-':>8}")
    print(f"{'REx-pure (standalone)':<34} {test_acc:>8.2f} {cost:>9.4f} {secs:>8.0f}")


if __name__ == "__main__":
    main()
