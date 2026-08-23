"""Diagnostic: run the REAL forward predictor on learned P(X) across ALL envs and see
how the two FD metrics (textdiff_delta_f1, llm judge) behave on actual LLM predictions.

For each env we load its learned P (+B) from the sweep, perceive sampled transitions into
P-space, run Fwd(P(X_t), A, B) -> Z_hat (the SAME predict_next_state wired into GEPA),
and score Z_hat against the true P(X_t+1) with BOTH metrics. We also score the STALE
baseline (predict "no change" = z_t) under both, so a real score is interpretable:
the useful quantity is the LIFT = real - stale (did the LLM's prediction beat copying the
current state?). Transitions are grouped by whether P actually MOVED (z_t != z_t1 in
P-space) -- forward prediction is only nontrivial there.

No optimization loop runs; this is purely to diagnose the metrics on real predictions
before we spend budget on the composite. Everything is parallelised across (env, item)
with one shared semaphore.

Usage:
  uv run offline_learning/scripts/diag_forward.py --moved 20 --static 8 --concurrency 24
"""

import os as _bos, sys as _bsys  # offline_learning/ on sys.path (flat-import the kept libs)
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
import argparse
import asyncio
import json
import random
import time
from pathlib import Path

from run_sweep import GAMES                          # (image, actions_csv, run_dirs) per env
from forward_objective import judge_score, textdiff_delta_f1, spearman
from invdyn_core import predict_next_state
from validate import load_transitions, make_config, run_perceive

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SWEEP = REPO / "logs" / "perc_invdyn" / "sweep_20260615-110303"


def load_pb(env):
    """First available seed's learned P (+B) for this env from the sweep."""
    pf = sorted(SWEEP.glob(f"{env}/seed*/best_perception_gepa_seed*.py"))[0]
    seed = pf.parent.name
    bf = pf.parent / pf.name.replace("best_perception", "best_beliefs").replace(".py", ".txt")
    B = bf.read_text() if bf.exists() else ""
    return pf.read_text(), B, seed


def sample(env, rng, n_moved, n_static):
    """Perceive all transitions, split by movement in P-space, sample caps from each."""
    image, actions_csv, run_dirs = GAMES[env]
    whitelist = set(filter(None, actions_csv.split(","))) or None
    trs = load_transitions([REPO / d for d in run_dirs], whitelist)
    for t in trs:
        t.action = t.action.split()[0]
    P, B, seed = load_pb(env)
    rng.shuffle(trs)
    moved, static = [], []
    for t in trs:
        z0 = run_perceive(P, t.x_t)[0]
        z1 = run_perceive(P, t.x_t1)[0]
        (moved if z0.strip() != z1.strip() else static).append((t, z0, z1))
    return P, B, seed, moved[:n_moved], static[:n_static], len(moved), len(trs)


async def score_item(cfg, env, B, t, z_t, z_t1, moved, sem):
    z_hat, c1 = await predict_next_state(cfg, z_t, t.action, B, sem)
    (jr, c2), (js, c3) = await asyncio.gather(
        judge_score(cfg, z_t, z_hat, z_t1, sem),     # judge: real prediction
        judge_score(cfg, z_t, z_t, z_t1, sem))       # judge: stale baseline (copy z_t)
    return {
        "env": env, "action": t.action, "moved": moved,
        "z_t": z_t, "z_t1": z_t1, "z_hat": z_hat,
        "td_real": textdiff_delta_f1(z_t, z_hat, z_t1),
        "td_stale": textdiff_delta_f1(z_t, z_t, z_t1),
        "judge_real": jr, "judge_stale": js,
        "cost": c1 + c2 + c3,
    }


async def main_async(args):
    cfg = make_config(args.task_model, args.client)
    sem = asyncio.Semaphore(args.concurrency)
    rng = random.Random(args.seed)
    envs = [e for e in GAMES if e in args.envs.split(",")] if args.envs else list(GAMES)

    tasks, meta = [], {}
    for env in envs:
        P, B, seed, moved, static, n_moved, n_all = sample(env, rng, args.moved, args.static)
        meta[env] = {"seed": seed, "n_moved_total": n_moved, "n_all": n_all,
                     "sampled_moved": len(moved), "sampled_static": len(static)}
        for (t, z0, z1) in moved:
            tasks.append(score_item(cfg, env, B, t, z0, z1, True, sem))
        for (t, z0, z1) in static:
            tasks.append(score_item(cfg, env, B, t, z0, z1, False, sem))
        print(f"[{env}] seed={seed} P-moved {n_moved}/{n_all} "
              f"({n_moved/max(1,n_all)*100:.0f}%) | sampling {len(moved)} moved + {len(static)} static")

    print(f"\nrunning {len(tasks)} forward predictions x (textdiff + judge x2) "
          f"@ concurrency {args.concurrency} ...", flush=True)
    t0 = time.time()
    rows = await asyncio.gather(*tasks)
    print(f"done in {time.time()-t0:.0f}s | total cost ${sum(r['cost'] for r in rows):.3f}\n")
    return rows, meta, envs


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def report(rows, meta, envs):
    out = ["# Forward-prediction metric diagnostic (real LLM predictions, all envs)\n",
           "Per env: learned P from the sweep, real Fwd(P(X_t),A,B)->Ẑ, scored vs true",
           "P(X_t+1). `real`=score of the LLM prediction, `stale`=score of copying z_t",
           "(predict no change), `lift`=real-stale (did the prediction beat copying?).",
           "Grouped by whether P moved in P-space.\n",
           "| env | grp | n | textdiff real | td stale | td lift | judge real | judge stale | judge lift |",
           "|-----|-----|--:|--------------:|---------:|--------:|-----------:|------------:|-----------:|"]
    for env in envs:
        for grp, flag in (("moved", True), ("static", False)):
            g = [r for r in rows if r["env"] == env and r["moved"] == flag]
            if not g:
                continue
            tdr, tds = mean([r["td_real"] for r in g]), mean([r["td_stale"] for r in g])
            jr, js = mean([r["judge_real"] for r in g]), mean([r["judge_stale"] for r in g])
            out.append(f"| {env} | {grp} | {len(g)} | {tdr:.3f} | {tds:.3f} | {tdr-tds:+.3f} "
                       f"| {jr:.3f} | {js:.3f} | {jr-js:+.3f} |")
    # cross-metric agreement on the REAL predictions (the diagnostic question)
    real = [r for r in rows if r["moved"]]
    if len(real) >= 2:
        sp = spearman([r["td_real"] for r in real], [r["judge_real"] for r in real])
        out += ["", f"Spearman(textdiff_real, judge_real) over {len(real)} MOVED items = {sp:.3f}"]
    out += ["", "## sampling", "| env | seed | P-moved / total |", "|-----|------|-----------------|"]
    for env in envs:
        m = meta[env]
        out.append(f"| {env} | {m['seed']} | {m['n_moved_total']}/{m['n_all']} |")
    text = "\n".join(out)
    print(text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moved", type=int, default=20, help="max MOVED transitions per env")
    ap.add_argument("--static", type=int, default=8, help="max STATIC transitions per env")
    ap.add_argument("--envs", default="", help="comma subset (default: all GAMES)")
    ap.add_argument("--task-model", default="google/gemini-2.5-flash")
    ap.add_argument("--client", default="openrouter")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(HERE / "diag_forward_results"))
    args = ap.parse_args()

    rows, meta, envs = asyncio.run(main_async(args))
    text = report(rows, meta, envs)
    Path(args.out + ".json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2, default=str))
    Path(args.out + ".md").write_text(text)
    print(f"\nwrote {args.out}.md and {args.out}.json")


if __name__ == "__main__":
    main()
