"""legacy_pop: population-based legacy optimizer for P/B, vs the legacy baseline.

Idea (the "missing middle"): keep a population + pareto-by-example selection,
but swap reflective free-form mutation for legacy's G1 failure-directed update.
Each candidate is mutated by the same compute_g1 -> update_perception/update_beliefs
operator the legacy loop uses; selection keeps a frontier that covers DIFFERENT
val examples (diversity), so the result is a SET of distinct {P,B} -- exactly what
`select_discriminating_action` needs, and what plain legacy (singleton) cannot give.

Run (DQ8GC, same split as sweep_20260616 so legacy matches its known numbers):
  uv run offline_learning/legacy_pop.py --seeds 1,2,3
"""

import argparse
import asyncio
import random
import time
from pathlib import Path

# Reuse the exact baseline plumbing (same dir; run via uv from repo root).
from invdyn_core import (  # noqa: E402
    InvDynAdapter,
    _clean_component,
    bake_choices,
    build_reflection_templates,
    eval_on,
    make_reflection_lm,
    predict_action_img,
    run_legacy_loop,
    _REFLECTION,
)
from validate_beliefs import (  # noqa: E402
    Transition,
    balanced_split,
    compute_g1,
    forward_eval,
    load_transitions,
    make_config,
    perception_runs,
    predict_action,
    run_perceive,
    update_beliefs,
    update_perception,
    _llm_call,
    _parse_tag,
)


class ImgTransition:
    """Inverse-dynamics transition whose states are IMAGES (for image-native
    environments like ARC where the obs has no text grid to parse)."""

    __slots__ = ("x_t", "x_t1", "action", "img_t", "img_t1")

    def __init__(self, action, img_t, img_t1, x_t="", x_t1=""):
        self.action = action
        self.img_t, self.img_t1 = img_t, img_t1
        self.x_t, self.x_t1 = x_t, x_t1

REPO = Path(__file__).resolve().parents[1]
# DQ8GC source transitions + action set, identical to run_sweep.GAMES["DQ8GC"].
DQ8GC_RUNS = [REPO / "logs/seed_autumn/DQ8GC/2026-06-05_14-07-07_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn"]
DQ8GC_ACTIONS = {"left", "right", "up", "down", "noop", "click"}


# ---------------------------------------------------------------------------
# val scoring with FIXED (baked) choice sets -> every candidate faces the same
# choices, so the per-example correctness vectors are directly comparable.
# ---------------------------------------------------------------------------
async def eval_vec(cfg, code, beliefs, baked, sem):
    async def one(inst):
        tr, choices = inst["tr"], inst["choices"]
        z_t = run_perceive(code, tr.x_t)[0]
        z_t1 = run_perceive(code, tr.x_t1)[0]
        pred, _, c = await predict_action(cfg, z_t, z_t1, beliefs, choices, sem)
        return (1.0 if pred == tr.action else 0.0), c
    res = await asyncio.gather(*(one(i) for i in baked))
    vec = [r[0] for r in res]
    cost = sum(r[1] for r in res)
    return (sum(vec) / max(1, len(vec))), vec, cost


def pareto_select(vecs, pop_size):
    """Keep candidates covering DIFFERENT val examples, then fill by aggregate.
    vecs[i] = per-example correctness (1/0) of candidate i. Returns selected idxs."""
    n = len(vecs)
    n_ex = len(vecs[0]) if n else 0
    frontier = set()
    for j in range(n_ex):
        best = max(vecs[i][j] for i in range(n))
        if best <= 0:
            continue  # nobody solves example j -> contributes no diversity signal
        for i in range(n):
            if vecs[i][j] == best:
                frontier.add(i)
    agg = [sum(vecs[i]) for i in range(n)]
    ranked = sorted(range(n), key=lambda i: -agg[i])
    sel = [i for i in ranked if i in frontier][:pop_size]
    for i in ranked:  # fill remaining slots by aggregate
        if len(sel) >= pop_size:
            break
        if i not in sel:
            sel.append(i)
    return sel


async def run_legacy_pop(cfg, train_tr, baked_val, action_pool, k, sem, seed,
                         rounds, pop_size, start_code):
    """Population of {code, beliefs, acc, vec}. Mutate each with legacy's G1 update;
    select pop_size by pareto-by-example over the (fixed-choice) val set."""
    cost = 0.0

    async def scored(cand):
        acc, vec, c = await eval_vec(cfg, cand["code"], cand["beliefs"], baked_val, sem)
        cand["acc"], cand["vec"] = acc, vec
        return c

    pop = [{"code": start_code, "beliefs": ""} for _ in range(pop_size)]
    cost += sum(await asyncio.gather(*(scored(c) for c in pop)))

    for rnd in range(1, rounds + 1):
        active = "P" if rnd % 2 == 1 else "B"  # alternate, like the legacy loop

        async def mutate(ci, cand):
            nonlocal_cost = 0.0
            # per-candidate minibatch -> identical seeds diverge into a real population
            crng = random.Random((seed * 1000 + rnd * 31 + ci) & 0x7FFFFFFF)
            mb = train_tr[:]
            crng.shuffle(mb)
            mb = mb[: max(4, len(train_tr))]  # small data: use all, just reshuffled
            _, recs, c = await forward_eval(cfg, cand["code"], cand["beliefs"], mb,
                                            action_pool, k, sem, crng)
            nonlocal_cost += c
            failures = [r for r in recs if not r.correct]
            g1_p, g1_b, c = await compute_g1(cfg, failures, cand["beliefs"], sem)
            nonlocal_cost += c
            if active == "P":
                exs = [r.tr for r in failures] or [r.tr for r in recs]
                newcode, c = await update_perception(cfg, cand["code"], g1_p,
                                                     cand["beliefs"], exs, sem)
                nonlocal_cost += c
                newcode = _clean_component("perception", newcode)
                ok, _ = perception_runs(newcode, [t.x_t for t in train_tr[:4]])
                child = {"code": newcode if ok else cand["code"], "beliefs": cand["beliefs"]}
            else:
                newbel, c = await update_beliefs(cfg, cand["beliefs"], g1_b,
                                                 failures or recs, sem)
                nonlocal_cost += c
                child = {"code": cand["code"], "beliefs": newbel}
            return child, nonlocal_cost

        results = await asyncio.gather(*(mutate(ci, c) for ci, c in enumerate(pop)))
        children = [r[0] for r in results]
        cost += sum(r[1] for r in results)
        cost += sum(await asyncio.gather(*(scored(c) for c in children)))  # only children need eval

        pool = pop + children
        sel = pareto_select([c["vec"] for c in pool], pop_size)
        pop = [pool[i] for i in sel]

    pop.sort(key=lambda c: -c["acc"])
    frontier = [{"perception": c["code"], "world_knowledge": c["beliefs"],
                 "val_acc": c["acc"]} for c in pop]
    return frontier, cost


# ---------------------------------------------------------------------------
# Image-native variant: for ARC-style envs the observation is an IMAGE with no
# text grid, so a text->text perception module has nothing to parse. We learn
# WORLD KNOWLEDGE (B) only, via image inverse-dynamics: the predictor sees the
# BEFORE/AFTER frames directly (predict_action_img) and B is the only learnable
# component. Exploits val==train (tied) so failures come free from the eval vec.
# ---------------------------------------------------------------------------
UPDATE_B_IMG_TMPL = """You maintain a WORLD KNOWLEDGE block: concise GENERAL facts about how this grid environment works (what the agent/entities are, coordinate conventions, what each action does). It is shared across ALL states, so it must contain only general truths -- NEVER facts about one specific state, and NEVER action frequencies/priors.

=== CURRENT WORLD KNOWLEDGE ===
{beliefs}
=== END CURRENT WORLD KNOWLEDGE ===

A predictor must identify which action was taken between two consecutive states by LOOKING at the BEFORE and AFTER images. It errs due to missing GENERAL knowledge. Below are IMAGE pairs (BEFORE then AFTER, in order) with the TRUE action taken between them:
{labels}

Infer the general rule(s) that explain how each action changes the world -- especially how the visible change (player/object movement, colour change, etc.) maps to each action name. Rewrite the FULL world knowledge block. Keep it concise and GENERAL: what the agent is, the coordinate convention, and how each action changes the state -- inferred ONLY from these examples. Respond as:
<world_knowledge>
- ...
</world_knowledge>"""


async def update_beliefs_img(cfg, beliefs, fail_records, sem, n_examples=6):
    imgs, labels = [], []
    for i, tr in enumerate(fail_records[:n_examples], 1):
        imgs.append(tr.img_t)
        imgs.append(tr.img_t1)
        labels.append(f"- pair {i}: action = {tr.action}")
    prompt = UPDATE_B_IMG_TMPL.format(
        beliefs=beliefs.strip() or "(empty)", labels="\n".join(labels),
    )
    async with sem:
        text, cost = await _llm_call(cfg, prompt, images=imgs or None)
    return _parse_tag(text or "", "world_knowledge"), cost


async def run_legacy_pop_img(cfg, baked_val, sem, seed, rounds, pop_size):
    """B-only population learner over IMAGE transitions (val==train tied).
    baked_val: [{tr, choices}] where tr is an ImgTransition (has img_t/img_t1)."""
    cost = 0.0

    async def eval_cand(cand):
        async def one(inst):
            tr, ch = inst["tr"], inst["choices"]
            pred, _, c = await predict_action_img(
                cfg, tr.img_t, tr.img_t1, cand["world_knowledge"], ch, sem)
            return (1.0 if pred == tr.action else 0.0), c
        res = await asyncio.gather(*(one(i) for i in baked_val))
        cand["vec"] = [r[0] for r in res]
        cand["acc"] = sum(cand["vec"]) / max(1, len(cand["vec"]))
        return sum(r[1] for r in res)

    pop = [{"perception": "", "world_knowledge": ""} for _ in range(pop_size)]
    cost += sum(await asyncio.gather(*(eval_cand(c) for c in pop)))

    for _rnd in range(1, rounds + 1):
        async def mutate(ci, cand):
            # train == val: failures are the val items this candidate got wrong.
            fails = [baked_val[i]["tr"] for i, ok in enumerate(cand["vec"]) if not ok]
            exs = fails or [b["tr"] for b in baked_val]
            newB, c = await update_beliefs_img(cfg, cand["world_knowledge"], exs, sem)
            return {"perception": "", "world_knowledge": newB}, c

        results = await asyncio.gather(*(mutate(ci, c) for ci, c in enumerate(pop)))
        children = [r[0] for r in results]
        cost += sum(r[1] for r in results)
        cost += sum(await asyncio.gather(*(eval_cand(c) for c in children)))
        pool = pop + children
        sel = pareto_select([c["vec"] for c in pool], pop_size)
        pop = [pool[i] for i in sel]

    pop.sort(key=lambda c: -c["acc"])
    return [{"perception": "", "world_knowledge": c["world_knowledge"],
             "val_acc": c["acc"]} for c in pop], cost


def run_seed(seed, args):
    rng = random.Random(seed)
    cfg = make_config(args.task_model, args.client)
    transitions = load_transitions(DQ8GC_RUNS, DQ8GC_ACTIONS)
    for t in transitions:  # collapse parametric actions (keep_params=False for DQ8GC)
        t.action = t.action.split()[0]
    rng.shuffle(transitions)
    action_pool = sorted({t.action for t in transitions})

    rest, test_tr = balanced_split(transitions, args.test_n, 10**9, rng)
    _, train_tr = balanced_split(rest, args.train_n, 10**9, rng)  # tied: val == train
    val_tr = train_tr
    k = args.k_choices
    train = bake_choices(train_tr, action_pool, k, rng)
    val = train  # tie-train-val
    test = bake_choices(test_tr, action_pool, k, rng)

    print(f"\n========== SEED {seed} ==========")
    print(f"transitions={len(transitions)} train=val={len(train)} test={len(test)} "
          f"pool({len(action_pool)})={action_pool}")

    seed_code = ""  # start=empty, matching the known legacy 0.75 baseline
    raw_acc, _ = asyncio.run(eval_on(cfg, "", "", test, raw_mode=True))
    print(f"[baseline] raw-frame test acc = {raw_acc:.2f}  (chance={1.0/k:.2f})")

    methods = {m.strip() for m in args.methods.split(",") if m.strip()}
    out = {"seed": seed, "raw": raw_acc,
           "legacy": None, "legacy_cost": None, "pop": None, "pop_cost": None,
           "distinctB": None, "frontier": None, "top": None}


    # ---- legacy (singleton) ----------------------------------------------
    if "legacy" in methods:
        async def _legacy():
            sem = asyncio.Semaphore(args.concurrency)
            lrng = random.Random(seed)
            return await run_legacy_loop(cfg, train_tr, val_tr, action_pool, k, sem, lrng,
                                         args.legacy_rounds, seed_code)
        t0 = time.perf_counter()
        lcode, lbel, legacy_cost = asyncio.run(_legacy())
        legacy_secs = time.perf_counter() - t0
        out["legacy"], _ = asyncio.run(eval_on(cfg, lcode, lbel, test))
        out["legacy_cost"] = legacy_cost
        print(f"[legacy]     test={out['legacy']:.2f}  cost=${legacy_cost:.3f}  {legacy_secs:.0f}s  "
              f"P_empty={not lcode.strip()}")

    # ---- legacy_pop (population) -----------------------------------------
    if "pop" in methods:
        async def _pop():
            sem = asyncio.Semaphore(args.concurrency)
            return await run_legacy_pop(cfg, train_tr, val, action_pool, k, sem, seed,
                                        args.pop_rounds, args.pop_size, seed_code)
        t0 = time.perf_counter()
        frontier, pop_cost = asyncio.run(_pop())
        pop_secs = time.perf_counter() - t0
        # report TOP candidate on test, plus the frontier's distinctness
        top = frontier[0]
        out["pop"], _ = asyncio.run(eval_on(cfg, top["perception"], top["world_knowledge"], test))
        out["pop_cost"] = pop_cost
        out["distinctB"] = len({c["world_knowledge"].strip() for c in frontier})
        out["frontier"], out["top"] = frontier, top
        distinct_P = len({c["perception"].strip() for c in frontier})
        # per-candidate test acc -> does the FRONTIER itself spread, not just top?
        per_test = [round(asyncio.run(eval_on(cfg, c["perception"], c["world_knowledge"], test))[0], 2)
                    for c in frontier]
        print(f"[legacy_pop] test={out['pop']:.2f}  cost=${pop_cost:.3f}  {pop_secs:.0f}s  "
              f"frontier={len(frontier)} distinctB={out['distinctB']} distinctP={distinct_P} "
              f"top_P_empty={not top['perception'].strip()}")
        print(f"[legacy_pop]   val_accs={[round(c['val_acc'],2) for c in frontier]}  "
              f"test_accs={per_test}")
        for i, c in enumerate(frontier):
            b = c["world_knowledge"].strip().replace("\n", " ")
            print(f"[legacy_pop]   cand{i}: P_empty={not c['perception'].strip()} "
                  f"B[:160]={b[:160]!r}")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--task-model", default="google/gemini-2.5-flash")
    ap.add_argument("--client", default="openrouter")
    ap.add_argument("--train-n", type=int, default=5)   # matches sweep (low data)
    ap.add_argument("--test-n", type=int, default=20)
    ap.add_argument("--k-choices", type=int, default=5)
    ap.add_argument("--max-metric-calls", type=int, default=120)  # metric budget (sweep)
    ap.add_argument("--legacy-rounds", type=int, default=6)
    ap.add_argument("--pop-size", type=int, default=4)
    ap.add_argument("--pop-rounds", type=int, default=6)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--methods", default="legacy,pop",
                    help="comma subset of: legacy,pop")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rows = [run_seed(s, args) for s in seeds]

    def f(v, w=6):  # format a possibly-None cell
        return (f"{v:>{w}.2f}" if isinstance(v, float) else f"{'-':>{w}}")

    def mean(key):
        vals = [x[key] for x in rows if isinstance(x[key], float)]
        return f(sum(vals) / len(vals)) if vals else f(None)

    print("\n\n=== HEAD-TO-HEAD (DQ8GC, clean test, start=empty, val==train) ===")
    print(f"{'seed':>4} {'raw':>6} {'legacy':>7} {'pop':>6} "
          f"{'leg$':>6} {'pop$':>6} {'distB':>6}")
    for r in rows:
        distb = f"{r['distinctB']:>6}" if r['distinctB'] is not None else f"{'-':>6}"
        print(f"{r['seed']:>4} {f(r['raw'])} {f(r['legacy'],7)} "
              f"{f(r['pop'])} {f(r['legacy_cost'])} "
              f"{f(r['pop_cost'])} {distb}")
    if rows:
        print(f"{'mean':>4} {mean('raw')} {mean('legacy',)} "
              f"{mean('pop')} {mean('legacy_cost')} {mean('pop_cost')}")


if __name__ == "__main__":
    main()
