"""Replay the 30-run low-data sweep (6 games x 5 seeds) from saved P/B artifacts and
dump per-failure materials for hand attribution. Applies deterministic pre-labels that
encode the rules validated by the seed-1 hand read-through, and FLAGS the only cases that
need eyeballing (P faithfully differs between frames AND a non-empty belief is present)."""
import asyncio
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from analyze_seeds import (load_games, rebuild_test, _safe_perceive, OUT_DIR,
                           TASK_MODEL, CLIENT, SEEDS)
from gepa_optimize import make_config, predict_action

OUT = Path(__file__).resolve().parent / "seeds5_failures_dump.json"
ITEM_TIMEOUT = 60


def prelabel(perr, zt, zt1, b_empty):
    """Deterministic judgement (internal observables only); None => read by hand."""
    if perr:
        return "PERCEPTION"            # P empty / raised -> F gets nothing
    if zt.strip() == zt1.strip():
        return "PERCEPTION"            # P output identical for both states -> no signal for F
    if b_empty:
        return "BELIEF"                # P differs but no convention for F
    return None                        # P differs + B present -> ambiguous, eyeball it


async def replay(cfg, game, seed, sem):
    env = game["env"]
    d = OUT_DIR / env / f"seed{seed}"
    pf = d / f"best_perception_gepa_seed{seed}.py"
    bf = d / f"best_beliefs_gepa_seed{seed}.txt"
    if not pf.exists():
        return {"env": env, "seed": seed, "status": "MISSING"}
    P = pf.read_text()
    B = bf.read_text() if bf.exists() else ""
    b_empty = not B.strip()
    test = rebuild_test(game, seed)

    async def run_item(it):
        tr, choices = it["tr"], it["choices"]
        z0, e0 = _safe_perceive(P, tr.x_t)
        z1, e1 = _safe_perceive(P, tr.x_t1)
        zt, zt1 = str(z0 or ""), str(z1 or "")
        try:
            pred, reasoning, _ = await asyncio.wait_for(
                predict_action(cfg, zt, zt1, B, choices, sem), timeout=ITEM_TIMEOUT)
        except asyncio.TimeoutError:
            pred, reasoning = "<TIMEOUT>", "<TIMEOUT>"
        return {"truth": tr.action, "pred": pred, "correct": pred == tr.action,
                "perr": e0 or e1,
                "z_t": zt, "z_t1": zt1, "reasoning": reasoning}

    rows = await asyncio.gather(*(run_item(it) for it in test))
    acc = sum(r["correct"] for r in rows) / len(rows)
    fails = []
    for r in rows:
        if r["correct"]:
            continue
        r["prelabel"] = prelabel(r["perr"], r["z_t"], r["z_t1"], b_empty)
        fails.append(r)
    return {"env": env, "seed": seed, "status": "OK", "acc": acc,
            "P_empty": not P.strip(), "B_empty": b_empty, "fails": fails}


async def main():
    cfg = make_config(TASK_MODEL, CLIENT)
    sem = asyncio.Semaphore(8)
    games = load_games()
    results = []
    for game in games:
        for seed in SEEDS:
            t = time.time()
            r = await replay(cfg, game, seed, sem)
            results.append(r)
            OUT.write_text(json.dumps(results, indent=2, default=str))
            if r["status"] == "OK":
                amb = sum(1 for f in r["fails"] if f["prelabel"] is None)
                print(f"{r['env']:7} s{seed} acc={r['acc']:.2f} P_empty={r['P_empty']} "
                      f"B_empty={r['B_empty']} fails={len(r['fails'])} ambiguous={amb} "
                      f"[{time.time()-t:.0f}s]", flush=True)
            else:
                print(f"{r['env']:7} s{seed} {r['status']}", flush=True)
    # aggregate of deterministic prelabels (ambiguous left as None)
    agg = Counter()
    for r in results:
        if r.get("status") != "OK":
            continue
        for f in r["fails"]:
            agg[f["prelabel"] or "AMBIGUOUS"] += 1
    print("\nPRELABEL TOTALS:", dict(agg), flush=True)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
