"""Failure attribution for the PREVIOUS completed gemini low-data runs (seed 1).

These runs (the matrix + arc-image sweep) saved their best artifacts to a shared
filename that got clobbered by parallelism, so here we instead RECONSTRUCT the exact
best candidate (perception P + belief B) from the GEPA log's candidate lineage:
  - "Iteration i: Selected program P"          -> parent of the mutation
  - "Iteration i: Proposed new text for <comp>" -> the mutated component + its text
  - "Iteration i: New program candidate index: K" -> accepted child index
  - "Best program as per aggregate score on valset: N" (last) -> best_idx
Program 0 is the seed (perception="" since --start empty, world_knowledge="").

Then it replays the same clean 20-item test split (seed 1) and classifies every
failure with the same scheme as analyze_seeds.py (PERCEPTION / BELIEF / F_REASONING
/ NO_SIGNAL). Constraint: gemini-2.5-flash, low-data, image-mode for ARC games.
"""
import asyncio
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from analyze_seeds import (HERE, TASK_MODEL, CLIENT, judge, _safe_perceive)
from gepa_optimize import (bake_choices, balanced_split, load_transitions,
                           make_config, predict_action)
from validate_beliefs import _extract_code

# game -> (logfile, image_mode, actions, run_dirs)   [matches /tmp/seeds5_envs.tsv]
GAMES = {
    "DQ8GC": ("/tmp/mat_gemini_DQ8GC.log", False, "left,right,up,down,noop",
              ["logs/dynamics_full/DQ8GC/2026-06-08_12-48-57_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn",
               "logs/seed_autumn/DQ8GC/2026-06-05_14-07-07_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn"]),
    "7WWW9": ("/tmp/mat_gemini_7WWW9.log", False, "left,right,up,down,noop,click",
              ["logs/dynamics_full/7WWW9/2026-06-08_12-48-59_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn",
               "logs/seed_autumn/7WWW9/2026-06-05_14-07-07_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn"]),
    "ice":   ("/tmp/mat_gemini_ice.log", False, "left,right,up,down,noop,click",
              ["logs/seed_autumn/ice/2026-06-05_14-07-07_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn"]),
    "ft09":  ("/tmp/arcimg_ft09.log", True, "ACTION0,ACTION1,ACTION2,ACTION3,ACTION4",
              ["logs/arc_parallel/20260528-184557/eb_learn__arc_agi__gemini-2p5-flash__ft09/2026-05-28_18-46-01_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn"]),
    "sp80":  ("/tmp/arcimg_sp80.log", True, "ACTION0,ACTION1,ACTION2,ACTION3,ACTION4",
              ["logs/arc_parallel/20260528-184557/eb_learn__arc_agi__gemini-2p5-flash__sp80/2026-05-28_18-46-01_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn"]),
    "ls20":  ("/tmp/arcimg_ls20.log", True, "ACTION0,ACTION1,ACTION2,ACTION3,ACTION4",
              ["logs/arc_parallel/20260528-184557/eb_learn__arc_agi__gemini-2p5-flash__ls20/2026-05-28_18-46-01_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn"]),
}
SEED, TRAIN_N, TEST_N, K = 1, 5, 20, 5

_SPLIT = re.compile(r"\n(?=Iteration \d+:|\[gepa\]|\[legacy\]|=== HEAD)")


def reconstruct_best(logtext):
    """Return (best_perception, best_beliefs, best_idx) from a GEPA log."""
    blocks = _SPLIT.split(logtext)
    sel, prop, newidx = {}, {}, {}
    for b in blocks:
        m = re.match(r"Iteration (\d+): Selected program (\d+)", b)
        if m:
            sel[int(m.group(1))] = int(m.group(2))
        m = re.match(r"Iteration (\d+): Proposed new text for (perception|world_knowledge):(.*)",
                     b, re.S)
        if m:
            prop[int(m.group(1))] = (m.group(2), m.group(3).strip())
        m = re.match(r"Iteration (\d+): New program candidate index: (\d+)", b)
        if m:
            newidx[int(m.group(1))] = int(m.group(2))
    best_idx = 0
    for m in re.finditer(r"Best program as per aggregate score on valset: (\d+)", logtext):
        best_idx = int(m.group(1))
    # build program table by lineage
    prog = {0: {"perception": "", "world_knowledge": ""}}
    for it in sorted(newidx):
        k = newidx[it]
        parent = sel.get(it, 0)
        comp, text = prop.get(it, ("perception", ""))
        child = dict(prog.get(parent, {"perception": "", "world_knowledge": ""}))
        child[comp] = text
        prog[k] = child
    best = prog.get(best_idx, prog[0])
    P = best["perception"]
    if "```" in P:
        P = _extract_code(P) or P
    return P, best["world_knowledge"], best_idx


def rebuild_test(actions, run_dirs):
    import random
    rng = random.Random(SEED)
    wl = set(filter(None, actions.split(",")))
    transitions = load_transitions([Path(p) for p in run_dirs], wl)
    for t in transitions:
        t.action = t.action.split()[0]
    rng.shuffle(transitions)
    pool = sorted({t.action for t in transitions})
    rest, test_tr = balanced_split(transitions, TEST_N, 10**9, rng)
    _, train_tr = balanced_split(rest, TRAIN_N, 10**9, rng)
    bake_choices(train_tr, pool, K, rng)            # consume rng for train (val==train)
    return bake_choices(test_tr, pool, K, rng)


def reported_acc(logtext):
    m = re.search(r"GEPA \(pareto.*?\)\s+([\d.]+)", logtext)
    return float(m.group(1)) if m else None


async def analyze_game(cfg, env, sem):
    logfile, image, actions, run_dirs = GAMES[env]
    logtext = Path(logfile).read_text()
    P, B, best_idx = reconstruct_best(logtext)
    test = rebuild_test(actions, run_dirs)

    async def run_item(item):
        tr, choices = item["tr"], item["choices"]
        z_t, err_t = _safe_perceive(P, tr.x_t)
        z_t1, err_t1 = _safe_perceive(P, tr.x_t1)
        zt = "" if z_t is None else str(z_t)
        zt1 = "" if z_t1 is None else str(z_t1)
        pred, reasoning, _ = await predict_action(cfg, zt, zt1, B, choices, sem)
        return tr, zt, zt1, (err_t or err_t1), pred, reasoning

    rows = await asyncio.gather(*(run_item(it) for it in test))
    acc = sum(1 for r in rows if r[4] == r[0].action) / len(rows)
    fails = []
    for tr, zt, zt1, perr, pred, reasoning in rows:
        if pred == tr.action:
            continue
        rec = {"env": env, "truth": tr.action, "pred": pred,
               "z_t": zt, "z_t1": zt1, "beliefs": B, "choices": [tr.action, pred],
               "reasoning": reasoning}
        if perr:
            rec["cause"], rec["why"] = "PERCEPTION", f"P failed to run/produce output {perr}"
        elif zt.strip() == zt1.strip():
            rec["cause"], rec["why"] = "PERCEPTION", "P outputs identical for both states (no discriminative signal)"
        else:
            rec["cause"], rec["why"] = await judge(cfg, rec)
        fails.append(rec)
    return {"env": env, "image": image, "best_idx": best_idx, "B_empty": not B.strip(),
            "acc": acc, "reported_acc": reported_acc(logtext), "fails": fails}


async def main():
    cfg = make_config(TASK_MODEL, CLIENT)
    sem = asyncio.Semaphore(8)
    results = []
    for env in GAMES:
        r = await analyze_game(cfg, env, sem)
        cc = Counter(f["cause"] for f in r["fails"])
        print(f"{env:7} {'img' if r['image'] else 'txt'}  acc={r['acc']:.2f} "
              f"(logged {r['reported_acc']})  B_empty={r['B_empty']}  "
              f"fails={len(r['fails'])}  {dict(cc)}")
        results.append(r)

    overall = Counter()
    L = ["# Previous gemini low-data runs (seed 1) — failure attribution\n",
         "Reconstructed each run's GEPA best candidate (P + B) from its log lineage, replayed the",
         "same clean 20-item test split, and attributed every failure. Constraint: gemini-2.5-flash,",
         "low-data (5 train==val / 20 test), image-mode for ARC (ft09/sp80/ls20), text for AutumnBench.\n",
         "Buckets: PERCEPTION (learned P incorrect/incomplete), BELIEF (learned B incorrect/incomplete),",
         "F_REASONING (P+B fine, decoder erred), NO_SIGNAL (no observable change -> action unidentifiable).\n",
         "| game | mode | acc | logged | B empty? | fails | PERCEPTION | BELIEF | F_REASONING | NO_SIGNAL |",
         "|------|------|----:|-------:|:--------:|------:|-----------:|-------:|------------:|----------:|"]
    for r in results:
        c = Counter(f["cause"] for f in r["fails"])
        for k, v in c.items():
            overall[k] += v
        L.append(f"| {r['env']} | {'image' if r['image'] else 'text'} | {r['acc']:.2f} | "
                 f"{r['reported_acc']} | {'yes' if r['B_empty'] else 'no'} | {len(r['fails'])} | "
                 f"{c['PERCEPTION']} | {c['BELIEF']} | {c['F_REASONING']} | {c['NO_SIGNAL']} |")
    tot = sum(overall.values())
    L += ["", "## Overall failure attribution\n", f"Total failures: {tot}\n",
          "| cause | count | share |", "|-------|------:|------:|"]
    for cause in ("PERCEPTION", "BELIEF", "F_REASONING", "NO_SIGNAL"):
        n = overall[cause]
        L.append(f"| {cause} | {n} | {(n/tot*100 if tot else 0):.0f}% |")
    L += ["", "## Per-failure detail\n"]
    for r in results:
        if not r["fails"]:
            continue
        L.append(f"### {r['env']} ({'image' if r['image'] else 'text'}, acc={r['acc']:.2f}, "
                 f"B_empty={r['B_empty']})")
        for f in r["fails"]:
            L.append(f"- **{f['cause']}** true=`{f['truth']}` pred=`{f['pred']}` — {f['why']}")
        L.append("")
    (HERE / "prev_logs_analysis.md").write_text("\n".join(L))
    (HERE / "prev_logs_analysis.json").write_text(json.dumps(results, indent=2, default=str))
    print("\nOverall:", dict(overall))
    print(f"wrote {HERE/'prev_logs_analysis.md'}")


if __name__ == "__main__":
    asyncio.run(main())
