"""Population (beam) search optimizer for perception P + world knowledge B, built
from scratch as a simpler alternative to GEPA's pareto engine -- but reusing GEPA's
reflective proposal and our inverse+forward objective verbatim.

Algorithm (cf. one GEPA run in <run>/optim_viz.html):

  seed node = {P: "", B: ""} (both empty) -> eval on the WHOLE train set,
                                             capturing the mistakes it makes.
  each generation g, alternating the ACTIVE component P, B, P, B, ...:
    for EVERY node in the current population:
      take the mistakes it made when it was last evaluated (as a child / the seed),
      feed them to the reflection LM to get feedback, and propose N children that
      rewrite ONLY the active component (the other is copied from the parent).
    evaluate ALL children (from all nodes) on the WHOLE train set.
    keep the top-K children -> next population.
  report + save the best node ever seen on a CLEAN test split it never touched.

What is reused from gepa_optimize.py (same prompts / parsing / eval / objective):
  - InvDynAdapter.evaluate            : inverse-dynamics (+ optional forward) scoring
  - InvDynAdapter.make_reflective_dataset : per-mistake feedback (the "gradient")
  - build_reflection_templates / make_reflection_lm : the proposer
  - GEPA's InstructionProposalSignature : renders the reflective dataset into the
                                          proposal prompt and parses the new component
  - bake_choices / eval_on / load_transitions / balanced_split / make_config / ...

The ONLY new code is the generational beam loop below. The key difference from GEPA
is selection: GEPA samples ONE parent off a pareto front and proposes ONE child per
iteration; here every node spawns N children each generation and we keep the global
top-K on the full train set.

Usage (matches gepa_optimize flags for the objective; defaults to empty seeds):
  uv run prototypes/perc_invdyn/pop_optimize.py \
      --run "logs/.../DQ8GC/...,logs/seed_autumn/DQ8GC/..." \
      --task-model google/gemini-2.5-flash \
      --fd-scorer exact --context-k 3 \
      --generations 6 --children-per-node 3 --keep-top-k 4
"""

import argparse
import asyncio
import json
import random
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from gepa.strategies.instruction_proposal import InstructionProposalSignature

# Reuse gepa_optimize verbatim: importing it sets up sys.path (repo root) + .env via
# its own transitive imports, and re-exports all the data/split/eval/objective plumbing.
from gepa_optimize import (  # noqa: E402
    GOOD_P,
    InvDynAdapter,
    _REFLECTION,
    _clean_component,
    bake_choices,
    balanced_split,
    swap_click_into_train,
    build_reflection_templates,
    eval_fd_on,
    eval_on,
    load_transitions,
    make_config,
    make_reflection_lm,
)
from mixed_improve import set_meta_temperature  # noqa: E402


def _prompt_text(prompt) -> str:
    """Flatten a proposer prompt to text for logging. prompt_renderer returns a str
    normally, or an OpenAI-style multimodal messages list in --image-mode."""
    if isinstance(prompt, str):
        return prompt
    return json.dumps(prompt, default=str)


# ---------------------------------------------------------------------------
# A population node: a candidate plus the evaluation it earned (its mistakes).
# ---------------------------------------------------------------------------
@dataclass
class Node:
    candidate: dict  # {"perception": <P code>, "world_knowledge": <B text>}
    eval_batch: object  # gepa EvaluationBatch from the full-train eval (carries trajectories)
    score: float  # mean train score
    nid: int
    gen: int
    parent: int = -1
    proposed_component: str = ""  # which component this node's expansion rewrote ("" for seed)
    kept: bool = True  # survived into the next population (always True for the seed)
    prompt: str = ""  # exact proposer prompt that produced this node (the mistakes are embedded in it)
    response: str = ""  # raw proposer response
    feedback: object = None  # the reflective dataset (parent's mistakes) shown to the proposer


class PopulationOptimizer:
    """Generational beam search over (P, B) candidates.

    Parallelism: within a generation the three LLM-bound phases each fan out as much as
    they can. (1) reflective-dataset build (one make_reflective_dataset per node, which may
    itself issue analyze-mistakes calls), (2) proposals (N children per node), and (3) child
    evaluations on the full train set, all run concurrently via asyncio.to_thread + gather,
    bounded by --propose-parallel / --eval-parallel. Each child evaluation is ALSO internally
    concurrent over transitions (the adapter's own --concurrency), so effective parallelism is
    eval_parallel x concurrency. To keep predictions.jsonl writes and cost accounting race-free
    under threads, every parallel unit of work uses its OWN adapter (built by make_adapter),
    writing its predictions to a per-node shard that is merged at the end."""

    def __init__(
        self,
        make_adapter,  # (pred_log_path | None) -> fresh InvDynAdapter
        reflect_lm,
        templates: dict,
        train: list,
        n_children: int,
        keep_top_k: int,
        out_dir: Path,
        eval_parallel: int,
        propose_parallel: int,
        log_path: Path | None = None,
    ):
        self.make_adapter = make_adapter
        self.reflect_lm = reflect_lm
        self.templates = templates
        self.train = train
        self.n_children = n_children
        self.keep_top_k = keep_top_k
        self.eval_parallel = eval_parallel
        self.propose_parallel = propose_parallel
        self.log_path = log_path  # nodes.jsonl: one record per node (seed + every child)
        self.shard_dir = out_dir / "_pred_shards"  # per-node prediction shards (merged later)
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self._next_id = 0
        self.best: Node | None = None  # best node EVER seen (elitism for the final artifact)
        # aggregated accounting across every (short-lived, per-work-unit) adapter
        self.total_cost = 0.0
        self.eval_calls = 0
        self.reused_evals = 0
        self._lock = threading.Lock()

    def _account(self, ad):
        with self._lock:
            self.total_cost += ad.total_cost
            self.eval_calls += ad.eval_calls
            self.reused_evals += ad.reused_evals

    def _track_best(self, node):
        with self._lock:
            if self.best is None or node.score > self.best.score:
                self.best = node

    # ---- evaluate one candidate on the WHOLE train set (its own adapter + shard) ----
    def _evaluate(self, candidate: dict, nid: int, gen: int, parent: int, proposed: str) -> Node:
        ad = self.make_adapter(self.shard_dir / f"pred_{nid}.jsonl")
        eb = ad.evaluate(self.train, candidate, capture_traces=True)
        self._account(ad)
        score = sum(eb.scores) / max(1, len(eb.scores))
        node = Node(candidate, eb, score, nid, gen, parent, proposed)
        self._track_best(node)
        return node

    def seed(self, candidate: dict) -> list[Node]:
        node = self._evaluate(candidate, nid=self._next_id, gen=0, parent=-1, proposed="")
        self._next_id += 1
        self._write_node(node)
        return [node]

    def resume(self, nodes_path: Path):
        """Continue a finished run: reload the surviving top-K population from the last
        generation and RE-EVALUATE their candidates to recover each node's mistakes
        (eval_batch), which nodes.jsonl doesn't persist. Returns (population, next_gen).
        Existing node ids / predictions are preserved (re-eval writes no shards)."""
        recs = [json.loads(line) for line in nodes_path.open() if line.strip()]
        if not recs:
            raise FileNotFoundError(f"no nodes to resume from in {nodes_path}")
        self._next_id = max(r["nid"] for r in recs) + 1
        # carry the global best forward (elitism); eval_batch unused since best is never expanded
        b = max(recs, key=lambda r: r["score"])
        self.best = Node(b["candidate"], None, b["score"], b["nid"], b["gen"], b["parent"], b.get("active", ""))
        last_gen = max(r["gen"] for r in recs)
        survivors = [r for r in recs if r["gen"] == last_gen and r.get("kept")]

        def reval(r):
            ad = self.make_adapter(None)  # no shard: predictions for these nodes already exist
            eb = ad.evaluate(self.train, r["candidate"], capture_traces=True)
            self._account(ad)
            score = sum(eb.scores) / max(1, len(eb.scores))
            node = Node(r["candidate"], eb, score, r["nid"], r["gen"], r["parent"], r.get("active", ""))
            self._track_best(node)
            return node

        async def run():
            sem = asyncio.Semaphore(self.eval_parallel)

            async def one(r):
                async with sem:
                    return await asyncio.to_thread(reval, r)

            return await asyncio.gather(*(one(r) for r in survivors))

        return asyncio.run(run()), last_gen + 1

    # ---- build each node's reflective dataset (mistakes) concurrently ----
    def _build_feedbacks(self, population: list[Node], active: str) -> dict:
        async def run():
            sem = asyncio.Semaphore(self.propose_parallel)

            async def one(node):
                async with sem:
                    ad = self.make_adapter(None)  # reflective build writes no predictions
                    rd = await asyncio.to_thread(
                        ad.make_reflective_dataset, node.candidate, node.eval_batch, [active]
                    )
                self._account(ad)
                return node.nid, rd.get(active)

            return dict(await asyncio.gather(*(one(n) for n in population)))

        return asyncio.run(run())

    # ---- propose all children (N per node) concurrently ----
    def _propose(self, population: list[Node], active: str, feedbacks: dict) -> list[dict]:
        jobs = [(node, feedbacks[node.nid]) for node in population for _ in range(self.n_children)]

        async def run():
            sem = asyncio.Semaphore(self.propose_parallel)

            async def one(node, feedback):
                prompt = InstructionProposalSignature.prompt_renderer(
                    {
                        "current_instruction_doc": node.candidate[active],
                        "dataset_with_feedback": feedback,
                        "prompt_template": self.templates[active],
                    }
                )
                async with sem:
                    response = await asyncio.to_thread(self.reflect_lm, prompt)
                text = _clean_component(
                    active,
                    InstructionProposalSignature.output_extractor((response or "").strip())["new_instruction"],
                )
                child = dict(node.candidate)
                child[active] = text
                return {
                    "parent": node, "cand": child, "active": active,
                    "prompt": _prompt_text(prompt), "response": response, "feedback": feedback,
                }

            return await asyncio.gather(*(one(n, f) for n, f in jobs))

        return asyncio.run(run())

    # ---- evaluate all children concurrently (each internally concurrent too) ----
    def _eval_children(self, specs: list[dict], gen: int, active: str) -> list[Node]:
        async def run():
            sem = asyncio.Semaphore(self.eval_parallel)

            async def one(spec):
                async with sem:
                    ad = self.make_adapter(self.shard_dir / f"pred_{spec['nid']}.jsonl")
                    eb = await asyncio.to_thread(ad.evaluate, self.train, spec["cand"], True)
                self._account(ad)
                score = sum(eb.scores) / max(1, len(eb.scores))
                node = Node(spec["cand"], eb, score, spec["nid"], gen, spec["parent"].nid, active)
                node.prompt, node.response, node.feedback = spec["prompt"], spec["response"], spec["feedback"]
                self._track_best(node)
                return node

            return await asyncio.gather(*(one(s) for s in specs))

        return asyncio.run(run())

    def step(self, population: list[Node], gen: int, active: str) -> list[Node]:
        # 1) each node's mistakes -> reflective dataset (parallel)
        feedbacks = self._build_feedbacks(population, active)
        # 2) propose N children per node (parallel); assign node ids up front (single-threaded)
        specs = self._propose(population, active, feedbacks)
        for s in specs:
            s["nid"] = self._next_id
            self._next_id += 1
        # 3) evaluate ALL children on the WHOLE train set (parallel across children)
        child_nodes = self._eval_children(specs, gen, active)
        # 4) keep the global top-K children -> next population
        child_nodes.sort(key=lambda n: n.score, reverse=True)
        kept_ids = {n.nid for n in child_nodes[: self.keep_top_k]}
        for cn in child_nodes:
            cn.kept = cn.nid in kept_ids
        for cn in sorted(child_nodes, key=lambda n: n.nid):  # write in birth order
            self._write_node(cn)
        return [cn for cn in child_nodes if cn.kept]

    def merge_predictions(self, out_path: Path):
        """Concatenate the per-node prediction shards into one predictions.jsonl (the format
        build_pop_viz / build_optim_viz expect)."""
        with out_path.open("w") as out:
            for shard in sorted(self.shard_dir.glob("pred_*.jsonl"), key=lambda p: int(p.stem.split("_")[1])):
                out.write(shard.read_text())

    def _write_node(self, node: Node):
        if not self.log_path:
            return

        def _safe(obj):
            return json.loads(json.dumps(obj, default=str))

        rec = {
            "nid": node.nid,
            "parent": node.parent,
            "gen": node.gen,
            "active": node.proposed_component,  # which component this node rewrote ("" = seed)
            "score": node.score,
            "kept": node.kept,
            "candidate": node.candidate,
            "prompt": node.prompt,  # exact proposer prompt (mistakes embedded) -- "" for seed
            "response": node.response,
            "feedback": _safe(node.feedback) if node.feedback is not None else None,
        }
        with self.log_path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="comma-separated run dirs (pooled)")
    ap.add_argument("--train-n", type=int, default=24, help="train set = the WHOLE set children are scored on")
    ap.add_argument("--test-n", type=int, default=20, help="clean test split neither search nor selection touches")
    ap.add_argument("--k-choices", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=16,
                    help="within ONE candidate eval: max concurrent transitions (each fires inv+fwd)")
    ap.add_argument("--eval-parallel", type=int, default=4,
                    help="max child evaluations run at once; effective parallelism = eval-parallel x concurrency")
    ap.add_argument("--propose-parallel", type=int, default=8,
                    help="max proposal / reflective-build LLM calls run at once")
    ap.add_argument("--task-model", default="google/gemini-2.5-flash", help="frozen decoder F (the agent)")
    ap.add_argument("--reflection-model", default=None, help="proposer LM (defaults to --task-model)")
    ap.add_argument("--client", default="openrouter")
    ap.add_argument("--actions", default="left,right,up,down,noop")
    ap.add_argument("--swap-click-train", action=argparse.BooleanOptionalAction, default=False,
        help="swap ONE click from test INTO train (and one 'up' to test) so the scored train "
        "set holds a click-identification item; a click remains in the clean test set")
    ap.add_argument("--collapse-action-params", action="store_true",
                    help="collapse parameterized actions to their verb (e.g. 'click 3 5' -> 'click')")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="sampling temperature for proposals (>0 so the N children differ) and F")
    # ---- population-search knobs (the new part) ----
    ap.add_argument("--generations", type=int, default=6, help="number of beam generations")
    ap.add_argument("--children-per-node", type=int, default=3, help="N children proposed per node per generation")
    ap.add_argument("--keep-top-k", type=int, default=4, help="K children kept as the next population")
    # ---- objective knobs (identical to gepa_optimize) ----
    ap.add_argument("--fd-scorer", choices=["none", "textdiff", "judge", "exact"], default="none",
                    help="forward-dynamics term: none=pure inverse dynamics; exact=1.0 iff predicted "
                    "next features exactly equal P(X_t+1); textdiff/judge as in gepa_optimize")
    ap.add_argument("--fd-weight", type=float, default=0.5, help="composite weight w: score=(1-w)*ID + w*FD")
    ap.add_argument("--fd-reflect", action=argparse.BooleanOptionalAction, default=True,
                    help="feed forward mistakes to the proposer too (all-objective feedback); default ON, --no-fd-reflect for inverse-only")
    ap.add_argument("--analyze-mistakes", action="store_true", help="extra LLM diagnosis of each shown mistake")
    ap.add_argument("--analyze-mode", choices=["combined", "per-mistake"], default="combined")
    ap.add_argument("--context-k", type=int, default=3,
                    help="temporal window K shown to F (0 = two-state). Matches gepa_optimize.")
    ap.add_argument("--image-mode", action="store_true", help="show rendered IMAGE (not raw text) to the P-writer")
    ap.add_argument("--cell-px", type=int, default=8)
    ap.add_argument("--good-baseline", action="store_true", help="also report the DQ8GC single-cell GOOD_P ceiling")
    ap.add_argument("--out-dir", default=None,
                    help="durable artifacts dir. Default: <repo>/logs/perc_invdyn/pop_<timestamp>_seed<seed>")
    ap.add_argument("--resume", action="store_true",
                    help="continue an existing --out-dir: reload its surviving top-K population and run "
                    "--generations MORE generations (node ids + predictions are appended, not reset). "
                    "Pass the SAME data/split/objective flags the original run used.")
    args = ap.parse_args()
    if args.resume and not args.out_dir:
        ap.error("--resume requires --out-dir pointing at the run to continue")
    if args.reflection_model is None:
        args.reflection_model = args.task_model

    rng = random.Random(args.seed)
    set_meta_temperature(args.temperature)  # diverse children + (lightly) stochastic F
    task_cfg = make_config(args.task_model, args.client)
    refl_cfg = make_config(args.reflection_model, args.client)
    whitelist = set(filter(None, args.actions.split(","))) or None

    context_k = args.context_k
    if context_k > 0 and args.image_mode:
        print("[warn] --image-mode incompatible with --context-k>0; forcing context_k=0.")
        context_k = 0

    run_dirs = [Path(p) for p in args.run.split(",") if p.strip()]
    transitions = load_transitions(run_dirs, whitelist, context_k=context_k)
    if args.collapse_action_params:
        for t in transitions:
            t.action = t.action.split()[0]
            t.ctx_prev = [(s, a.split()[0]) for s, a in t.ctx_prev]
            t.ctx_next = [(a.split()[0], s) for a, s in t.ctx_next]
    rng.shuffle(transitions)
    action_pool = sorted({t.action for t in transitions})

    # carve the clean test split first so it is identical regardless of search settings
    rest, test_tr = balanced_split(transitions, args.test_n, 10**9, rng)
    _, train_tr = balanced_split(rest, args.train_n, 10**9, rng)
    if args.swap_click_train:  # put a click-ID item into the SCORED train set
        moved = swap_click_into_train(train_tr, test_tr)
        print(f"[swap-click-train] moved click into train: {moved} (an 'up' moved to test)")
    k = args.k_choices
    train = bake_choices(train_tr, action_pool, k, rng)
    test = bake_choices(test_tr, action_pool, k, rng)

    print(f"transitions: {len(transitions)} | train={len(train)} test={len(test)}")
    print(f"test action balance: {dict(Counter(t['tr'].action for t in test))}")
    print(f"action pool ({len(action_pool)}): {action_pool}")
    print(f"task_lm (F) = {args.task_model} | reflection_lm = {args.reflection_model} | temp={args.temperature}")
    print(f"population: generations={args.generations} children/node={args.children_per_node} keep_top_k={args.keep_top_k}\n")

    if args.out_dir:
        outd = Path(args.out_dir)
    else:
        repo_root = Path(__file__).resolve().parents[2]
        outd = repo_root / "logs" / "perc_invdyn" / f"pop_{time.strftime('%Y%m%d-%H%M%S')}_seed{args.seed}"
    outd.mkdir(parents=True, exist_ok=True)
    print(f"[out] artifacts + process log -> {outd}")

    seed_code = ""  # learning always starts from an empty perception module
    seed_candidate = {"perception": seed_code, "world_knowledge": ""}

    # ---- baselines on the CLEAN test split (same as gepa_optimize) -------
    chance = 1.0 / k
    raw_acc, _ = asyncio.run(eval_on(task_cfg, "", "", test, raw_mode=True,
                                     image_mode=args.image_mode, cell=args.cell_px))
    start_acc, _ = asyncio.run(eval_on(task_cfg, seed_code, "", test, context_k=context_k))
    good_acc = None
    if args.good_baseline:
        good_acc, _ = asyncio.run(eval_on(task_cfg, GOOD_P, "", test, context_k=context_k))
    print(f"[test baselines] random={chance:.2f} | start-P={start_acc:.2f} | raw-frame={raw_acc:.2f}"
          + (f" | single-cell GOOD_P={good_acc:.2f}" if good_acc is not None else "") + "\n")

    # ---- population search -----------------------------------------------
    # Every parallel unit of work gets its OWN adapter (race-free predictions + cost), all
    # sharing the same config/flags. pred_log_path=None for reflective-only adapters.
    def make_adapter(pred_log_path):
        return InvDynAdapter(
            task_cfg, action_pool, concurrency=args.concurrency,
            image_mode=args.image_mode, cell=args.cell_px,
            fd_scorer=args.fd_scorer, fd_weight=args.fd_weight, fd_reflect=args.fd_reflect,
            analyze_mistakes=args.analyze_mistakes, analyze_mode=args.analyze_mode,
            pred_log_path=pred_log_path, context_k=context_k, reuse_traces=False,
        )

    templates = build_reflection_templates(None)
    reflect_lm = make_reflection_lm(refl_cfg)
    opt = PopulationOptimizer(
        make_adapter, reflect_lm, templates, train,
        n_children=args.children_per_node, keep_top_k=args.keep_top_k,
        out_dir=outd, eval_parallel=args.eval_parallel, propose_parallel=args.propose_parallel,
        log_path=outd / "nodes.jsonl",
    )
    print(f"[parallel] eval_parallel={args.eval_parallel} x concurrency={args.concurrency} "
          f"| propose_parallel={args.propose_parallel}\n")

    t0 = time.perf_counter()
    if args.resume:
        population, start_gen = opt.resume(outd / "nodes.jsonl")
        scores = ", ".join(f"{n.score:.3f}" for n in population)
        print(f"[resume] reloaded {len(population)} survivors from gen {start_gen - 1} "
              f"(re-eval train scores: [{scores}], best-ever={opt.best.score:.3f}); "
              f"running gens {start_gen}..{start_gen + args.generations - 1}")
    else:
        population = opt.seed(seed_candidate)
        start_gen = 1
        print(f"[gen 0] seed train score = {population[0].score:.3f}")
    for gen in range(start_gen, start_gen + args.generations):
        active = "perception" if gen % 2 == 1 else "world_knowledge"  # alternate P, B, P, B, ...
        population = opt.step(population, gen, active)
        scores = ", ".join(f"{n.score:.3f}" for n in population)
        print(f"[gen {gen}] active={active:<15} top-{len(population)} train scores: [{scores}] "
              f"| best-ever={opt.best.score:.3f} | F evals={opt.eval_calls}")
    secs = time.perf_counter() - t0
    opt.merge_predictions(outd / "predictions.jsonl")  # shards -> one predictions.jsonl for the viz

    best = opt.best
    best_code = _clean_component("perception", best.candidate.get("perception", ""))
    best_beliefs = best.candidate.get("world_knowledge", "")

    # ---- clean test eval of the best-ever node ---------------------------
    test_acc, _ = asyncio.run(eval_on(task_cfg, best_code, best_beliefs, test,
                                      log_path=outd / f"test_trace_pop_seed{args.seed}.json",
                                      context_k=context_k))
    cost = opt.total_cost + _REFLECTION["cost"]
    print(f"\n[pop] best node nid={best.nid} (gen {best.gen}) train={best.score:.3f}")
    print(f"[pop] CLEAN test acc (inverse) = {test_acc:.2f}")
    if args.fd_scorer != "none":
        fd_test, fd_cost = asyncio.run(eval_fd_on(task_cfg, best_code, best_beliefs, test,
                                                  args.fd_scorer, args.concurrency, context_k=context_k))
        cost += fd_cost
        print(f"[pop] CLEAN test FD[{args.fd_scorer}] = {fd_test:.2f}")
    print(f"[pop] F cost=${opt.total_cost:.4f} ({opt.eval_calls} fresh F evals) | "
          f"reflection cost=${_REFLECTION['cost']:.4f} ({_REFLECTION['calls']} calls) | time={secs:.0f}s")

    (outd / f"best_perception_pop_seed{args.seed}.py").write_text(best_code)
    (outd / f"best_beliefs_pop_seed{args.seed}.txt").write_text(best_beliefs)
    print(f"[pop] best P -> best_perception_pop_seed{args.seed}.py | best B -> best_beliefs_pop_seed{args.seed}.txt")

    print("\n=== RESULT (clean test) ===")
    print(f"{'method':<34} {'test_acc':>8} {'cost($)':>9} {'time(s)':>8}")
    print(f"{'raw-frame ref':<34} {raw_acc:>8.2f} {'-':>9} {'-':>8}")
    print(f"{'population beam (P/B)':<34} {test_acc:>8.2f} {cost:>9.4f} {secs:>8.0f}")


if __name__ == "__main__":
    main()
