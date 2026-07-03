"""Greedy (joint B+P) optimizer with a COMPOSITE inverse-dynamics + forward-dynamics
objective, 5 seeds, same 15 DQ8GC transitions. Mirrors the joint greedy from
compare_alt_joint.py but the accept/reject gate uses (1-W)*ID + W*FD[judge], so it
is the greedy analog of GEPA --fd-scorer judge --fd-weight 0.5.

  uv run prototypes/perc_invdyn/greedy_fd.py
"""
import asyncio
import csv
import json
import random
from pathlib import Path

from validate import (  # noqa: E402
    load_transitions,
    make_config,
    perception_runs,
    run_perceive,
)
from validate_beliefs import (  # noqa: E402
    balanced_split,
    compute_g1,
    forward_eval,
)
from compare_alt_joint import joint_update  # reuse the joint rewrite
from gepa_optimize import predict_next_state  # forward predictor
from forward_objective import judge_score      # FD scorer (LLM judge)

DATA = Path("/tmp/dq8gc_first15")
ACTIONS = {"left", "right", "up", "down", "noop", "click"}
ROUNDS = 6
K = 5
W = 0.5  # composite weight on FD
SEEDS = [1, 2, 3, 4, 5]
MODEL = "google/gemini-2.5-flash"


async def fd_score(cfg, code, beliefs, transitions, sem):
    async def one(tr):
        zt = run_perceive(code, tr.x_t)[0]
        zt1 = run_perceive(code, tr.x_t1)[0]
        zhat, c1 = await predict_next_state(cfg, zt, tr.action, beliefs, sem)
        s, c2 = await judge_score(cfg, zt, zhat, zt1, sem)
        return s, c1 + c2
    res = await asyncio.gather(*(one(t) for t in transitions))
    return sum(s for s, _ in res) / max(1, len(res)), sum(c for _, c in res)


async def composite(cfg, code, beliefs, evalset, action_pool, sem, rng):
    acc, _, c1 = await forward_eval(cfg, code, beliefs, evalset, action_pool, K, sem, rng)
    fd, c2 = await fd_score(cfg, code, beliefs, evalset, sem)
    return (1 - W) * acc + W * fd, acc, fd, c1 + c2


async def run_loop(cfg, train, holdout, action_pool, seed, sem, log):
    rng = random.Random(seed * 7 + 1)
    code, beliefs = "", ""
    cost = 0.0
    best, _, _, c = await composite(cfg, code, beliefs, holdout, action_pool, sem, rng); cost += c
    best_code, best_bel = code, beliefs
    for rnd in range(1, ROUNDS + 1):
        _, recs, c = await forward_eval(cfg, code, beliefs, train, action_pool, K, sem, rng); cost += c
        failures = [r for r in recs if not r.correct]
        g1_p, g1_b, c = await compute_g1(cfg, failures, beliefs, sem); cost += c
        ncode, nbel, c = await joint_update(cfg, code, beliefs, g1_p, g1_b, failures, train, sem); cost += c
        ok, _ = perception_runs(ncode, [t.x_t for t in train[:4]])
        ncode = ncode if ok else code
        new, _, _, c = await composite(cfg, ncode, nbel, holdout, action_pool, sem, rng); cost += c
        if new >= best:
            best, best_code, best_bel, code, beliefs = new, ncode, nbel, ncode, nbel
    comp, idacc, fd, c = await composite(cfg, best_code, best_bel, train, action_pool, sem, rng); cost += c
    log(f"[greedyFD seed{seed}] DONE  composite={best:.2f}  train_id={idacc:.2f} train_fd={fd:.2f}  cost=${cost:.4f}")
    return {"seed": seed, "composite": best, "train_id": idacc, "train_fd": fd,
            "cost": cost, "perception": best_code, "beliefs": best_bel}


async def main():
    cfg = make_config(MODEL, "openrouter")
    transitions = load_transitions([DATA], ACTIONS)
    for t in transitions:
        t.action = t.action.split()[0]
    action_pool = sorted({t.action for t in transitions})
    print(f"transitions={len(transitions)} action_pool={action_pool}", flush=True)

    sem = asyncio.Semaphore(48)

    def log(m):
        print(m, flush=True)

    tasks = []
    for seed in SEEDS:
        rng = random.Random(seed)
        train, holdout = balanced_split(transitions, 3, 12, rng)
        tasks.append(run_loop(cfg, train, holdout, action_pool, seed, sem, log))
    results = await asyncio.gather(*tasks)

    outdir = Path("logs/greedy_fd")
    outdir.mkdir(parents=True, exist_ok=True)
    json.dump([{k: v for k, v in r.items() if k != "perception"} for r in results],
              (outdir / "results.json").open("w"), indent=2, default=str)
    for r in results:
        (outdir / f"P_greedyfd_seed{r['seed']}.py").write_text(r["perception"] or "")
        (outdir / f"B_greedyfd_seed{r['seed']}.txt").write_text(r["beliefs"] or "")
    print(f"\ntotal cost ${sum(r['cost'] for r in results):.4f}  mean ${sum(r['cost'] for r in results)/len(results):.4f}")
    print(f"artifacts -> {outdir}")


if __name__ == "__main__":
    asyncio.run(main())
