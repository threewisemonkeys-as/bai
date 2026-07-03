"""Greedy (joint B+P) optimizer on the MATCHED split (train = val = all 15, tied;
test = 0 -- identical to the GEPA --tie-train-val setup). Runs three objective modes
x 5 seeds:  none (ID only) | judge (ID+FD via LLM judge) | textdiff (ID+FD via
deterministic textdiff_delta_f1).

  uv run prototypes/perc_invdyn/greedy_matched.py
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
    compute_g1,
    forward_eval,
)
from compare_alt_joint import joint_update
from gepa_optimize import predict_next_state
from forward_objective import judge_score, textdiff_delta_f1

DATA = Path("/tmp/dq8gc_first15")
ACTIONS = {"left", "right", "up", "down", "noop", "click"}
ROUNDS = 6
K = 5
W = 0.5
SEEDS = [1, 2, 3, 4, 5]
MODES = ["none", "judge", "textdiff"]
MODEL = "google/gemini-2.5-flash"


async def fd_score(cfg, code, beliefs, transitions, scorer, sem):
    """Mean forward-dynamics score: predict next-state from P(X_t)+action+B, grade vs
    P(X_t+1). scorer 'judge' = LLM; 'textdiff' = free deterministic delta-F1."""
    async def one(tr):
        zt = run_perceive(code, tr.x_t)[0]
        zt1 = run_perceive(code, tr.x_t1)[0]
        zhat, c1 = await predict_next_state(cfg, zt, tr.action, beliefs, sem)
        if scorer == "judge":
            s, c2 = await judge_score(cfg, zt, zhat, zt1, sem)
        else:  # textdiff (free)
            s, c2 = textdiff_delta_f1(zt, zhat, zt1), 0.0
        return s, c1 + c2
    res = await asyncio.gather(*(one(t) for t in transitions))
    return sum(s for s, _ in res) / max(1, len(res)), sum(c for _, c in res)


async def gate_score(cfg, code, beliefs, evalset, action_pool, scorer, sem, rng):
    acc, _, c1 = await forward_eval(cfg, code, beliefs, evalset, action_pool, K, sem, rng)
    if scorer == "none":
        return acc, acc, 0.0, c1
    fd, c2 = await fd_score(cfg, code, beliefs, evalset, scorer, sem)
    return (1 - W) * acc + W * fd, acc, fd, c1 + c2


async def run_loop(cfg, mode, train, holdout, action_pool, seed, sem, log):
    rng = random.Random(seed * 7 + 1)
    code, beliefs = "", ""
    cost = 0.0
    best, _, _, c = await gate_score(cfg, code, beliefs, holdout, action_pool, mode, sem, rng); cost += c
    best_code, best_bel = code, beliefs
    for rnd in range(1, ROUNDS + 1):
        _, recs, c = await forward_eval(cfg, code, beliefs, train, action_pool, K, sem, rng); cost += c
        failures = [r for r in recs if not r.correct]
        g1_p, g1_b, c = await compute_g1(cfg, failures, beliefs, sem); cost += c
        ncode, nbel, c = await joint_update(cfg, code, beliefs, g1_p, g1_b, failures, train, sem); cost += c
        ok, _ = perception_runs(ncode, [t.x_t for t in train[:4]])
        ncode = ncode if ok else code
        new, _, _, c = await gate_score(cfg, ncode, nbel, holdout, action_pool, mode, sem, rng); cost += c
        if new >= best:
            best, best_code, best_bel, code, beliefs = new, ncode, nbel, ncode, nbel
    _, idacc, fd, c = await gate_score(cfg, best_code, best_bel, train, action_pool, mode, sem, rng); cost += c
    log(f"[greedy-{mode} seed{seed}] DONE  gate={best:.2f} train_id={idacc:.2f} cost=${cost:.4f}")
    return {"mode": mode, "seed": seed, "gate": best, "train_id": idacc,
            "cost": cost, "perception": best_code, "beliefs": best_bel}


async def main():
    cfg = make_config(MODEL, "openrouter")
    transitions = load_transitions([DATA], ACTIONS)
    for t in transitions:
        t.action = t.action.split()[0]
    action_pool = sorted({t.action for t in transitions})
    print(f"transitions={len(transitions)} | MATCHED split: train={len(transitions)} "
          f"val={len(transitions)} (TIED) test=0 | action_pool={action_pool}", flush=True)

    sem = asyncio.Semaphore(64)

    def log(m):
        print(m, flush=True)

    tasks = []
    for mode in MODES:
        for seed in SEEDS:
            train = list(transitions)          # all 15
            holdout = list(transitions)        # tied: val == train == all 15
            tasks.append(run_loop(cfg, mode, train, holdout, action_pool, seed, sem, log))
    results = await asyncio.gather(*tasks)

    outdir = Path("logs/greedy_matched")
    outdir.mkdir(parents=True, exist_ok=True)
    json.dump([{k: v for k, v in r.items() if k != "perception"} for r in results],
              (outdir / "results.json").open("w"), indent=2, default=str)
    for r in results:
        tag = f"{r['mode']}_seed{r['seed']}"
        (outdir / f"P_{tag}.py").write_text(r["perception"] or "")
        (outdir / f"B_{tag}.txt").write_text(r["beliefs"] or "")
    for mode in MODES:
        rs = [x for x in results if x["mode"] == mode]
        print(f"greedy-{mode:<9} total_cost=${sum(x['cost'] for x in rs):.4f} "
              f"mean=${sum(x['cost'] for x in rs)/len(rs):.4f}")
    print(f"artifacts -> {outdir}")


if __name__ == "__main__":
    asyncio.run(main())
