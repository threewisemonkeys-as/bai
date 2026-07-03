"""Robust per-failure materials dump (NO automated judge) for the previous gemini
low-data runs. Reconstructs each run's GEPA best P+B from its log, replays the same
clean 20-item test split, and for every failure records the learned belief, P's output
on both frames, the deterministic raw change, and F's reasoning.

Runs game-by-game, writes partial results after each game, and bounds every F call
with a timeout so one throttled call can't stall the whole job.
Output: prototypes/perc_invdyn/prev_failures_dump.json (for human one-by-one reading)."""
import asyncio
import json
import time
from pathlib import Path

from analyze_prev_logs import GAMES, reconstruct_best, rebuild_test, reported_acc, HERE
from analyze_seeds import _safe_perceive
from gepa_optimize import make_config, predict_action
from analyze_prev_logs import TASK_MODEL, CLIENT

OUT = HERE / "prev_failures_dump.json"
ITEM_TIMEOUT = 60


async def dump_game(cfg, env, sem):
    logfile, image, actions, run_dirs = GAMES[env]
    logtext = Path(logfile).read_text()
    P, B, best_idx = reconstruct_best(logtext)
    test = rebuild_test(actions, run_dirs)

    async def run_item(item):
        tr, choices = item["tr"], item["choices"]
        zt = str(_safe_perceive(P, tr.x_t)[0] or "")
        zt1 = str(_safe_perceive(P, tr.x_t1)[0] or "")
        try:
            # NB: predict_action acquires `sem` itself -- do NOT wrap in `async with sem`
            # here or the same semaphore is acquired twice and deadlocks until timeout.
            pred, reasoning, _ = await asyncio.wait_for(
                predict_action(cfg, zt, zt1, B, choices, sem), timeout=ITEM_TIMEOUT)
        except asyncio.TimeoutError:
            pred, reasoning = "<TIMEOUT>", "<TIMEOUT>"
        return {"truth": tr.action, "pred": pred, "correct": pred == tr.action,
                "z_t": zt, "z_t1": zt1,
                "reasoning": reasoning}

    rows = await asyncio.gather(*(run_item(it) for it in test))
    acc = sum(r["correct"] for r in rows) / len(rows)
    return {"env": env, "image": image, "best_idx": best_idx, "beliefs": B,
            "perception_empty": not P.strip(), "acc": acc,
            "reported_acc": reported_acc(logtext),
            "fails": [r for r in rows if not r["correct"]]}


async def main():
    cfg = make_config(TASK_MODEL, CLIENT)
    sem = asyncio.Semaphore(6)
    results = []
    for env in GAMES:
        t = time.time()
        r = await dump_game(cfg, env, sem)
        results.append(r)
        OUT.write_text(json.dumps(results, indent=2, default=str))   # incremental
        print(f"{r['env']:7} acc={r['acc']:.2f} (logged {r['reported_acc']}) "
              f"P_empty={r['perception_empty']} B_empty={not r['beliefs'].strip()} "
              f"fails={len(r['fails'])}  [{time.time()-t:.0f}s]", flush=True)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
