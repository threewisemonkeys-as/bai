#!/usr/bin/env python3
"""Offline re-rank of the aug7_softmin pools by hard min, + stage the three ship
arms for the multistep planning eval.

For each game we have a rex_pure pool searched under softmin. Every candidate's
per-(transition) terms are persisted in predictions.jsonl (id_score, cfd_score,
and `score` = the softmin composite the search used). This recomputes, per
candidate, mean(min(id,cfd)) over the SAME train rows -- a pure re-aggregation,
no LLM calls -- and takes the argmax as the "hardmin-select" ship. It compares
that to the softmin ship (argmax of the persisted softmin mean).

Three arms are then staged as gepa-named artifacts for eval_multistep_fd_plan.py:
  softmin        -> the aug7_softmin shipped P+B (softmin learn + softmin select)
  hardmin        -> the aug5_rexpure control shipped P+B (hard-min learn + select)
  hardmin_select -> the aug7 pool re-ranked by hard min (softmin learn + hardmin select)
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

REPO = Path("/home/ays57/bai")
SOFT = REPO / "logs/aug7_softmin"
CTRL = REPO / "logs/aug5_rexpure"
OUT = REPO / "logs/aug7_softmin_planeval"

GAMES = {  # game -> (softmin run subdir, control run subdir)
    "bt3gb": ("bt3gb_strat30_seed1", "bt3gb_strat30_seed1"),
    "n2ntd": ("n2ntd_seed1", "n2ntd_seed1"),
    "s2kt7": ("s2kt7_seed5data", "s2kt7_seed5data"),
}


def cand_hash(perc: str, wk: str) -> str:
    h = hashlib.md5()
    for p in (perc, wk):
        h.update((p or "").encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def load_pool(run_dir: Path):
    cands = [json.loads(l) for l in (run_dir / "candidates.jsonl").read_text().splitlines() if l.strip()]
    by_hash_rows = defaultdict(list)
    for l in (run_dir / "predictions.jsonl").read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        by_hash_rows[r["cand_hash"]].append(r)
    return cands, by_hash_rows


def rerank(game, soft_sub, ctrl_sub):
    run_dir = SOFT / soft_sub / "rexpure_run_seed1"
    cands, rows_by_hash = load_pool(run_dir)

    recs = []
    for c in cands:
        h = cand_hash(c.get("perception", ""), c.get("world_knowledge", ""))
        rows = rows_by_hash.get(h, [])
        if not rows:
            continue
        gate_zeroed = any(r.get("gate_zeroed") for r in rows)
        soft_scores, hard_scores = [], []
        for r in rows:
            terms = [r["id_score"]]
            if r.get("cfd_score") is not None:
                terms.append(r["cfd_score"])
            soft_scores.append(r["score"])          # persisted softmin composite
            hard_scores.append(min(terms))          # recomputed hard min
        soft_mean = 0.0 if gate_zeroed else mean(soft_scores)
        hard_mean = 0.0 if gate_zeroed else mean(hard_scores)
        recs.append({
            "idx": c["idx"], "hash": h, "n_rows": len(rows),
            "soft_mean": soft_mean, "hard_mean": hard_mean,
            "stored_train_score": c["train_score"],
            "perception": c.get("perception", ""), "world_knowledge": c.get("world_knowledge", ""),
        })

    soft_ship = max(recs, key=lambda r: r["soft_mean"])
    hard_ship = max(recs, key=lambda r: r["hard_mean"])
    # faithfulness check: persisted softmin mean should match candidates.jsonl train_score
    max_drift = max(abs(r["soft_mean"] - r["stored_train_score"]) for r in recs)
    return recs, soft_ship, hard_ship, max_drift


def stage(arm, game, perc: str, wk: str):
    d = OUT / arm / game
    d.mkdir(parents=True, exist_ok=True)
    (d / "best_perception_gepa_seed1.py").write_text(perc)
    (d / "best_beliefs_gepa_seed1.txt").write_text(wk or "")
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for game, (soft_sub, ctrl_sub) in GAMES.items():
        recs, soft_ship, hard_ship, drift = rerank(game, soft_sub, ctrl_sub)
        changed = soft_ship["idx"] != hard_ship["idx"]
        print(f"\n=== {game} === pool={len(recs)} cands | faithfulness drift={drift:.4f}")
        print(f"  softmin ship        : idx {soft_ship['idx']:>2}  soft={soft_ship['soft_mean']:.3f}  hard={soft_ship['hard_mean']:.3f}")
        print(f"  hardmin-select ship : idx {hard_ship['idx']:>2}  soft={hard_ship['soft_mean']:.3f}  hard={hard_ship['hard_mean']:.3f}"
              f"   {'<-- DIFFERENT candidate' if changed else '(same candidate as softmin)'}")

        # stage arms. softmin + hardmin come from the actual shipped artifacts on disk
        # (identical content, but read them so the eval uses exactly what shipped).
        soft_run = SOFT / soft_sub
        stage("softmin", game,
              (soft_run / "best_perception_rexpure_seed1.py").read_text(),
              (soft_run / "best_beliefs_rexpure_seed1.txt").read_text())
        ctrl_run = CTRL / ctrl_sub
        stage("hardmin", game,
              (ctrl_run / "best_perception_gepa_seed1.py").read_text(),
              (ctrl_run / "best_beliefs_gepa_seed1.txt").read_text())
        stage("hardmin_select", game, hard_ship["perception"], hard_ship["world_knowledge"])

        report[game] = {
            "pool_size": len(recs), "faithfulness_drift": drift,
            "softmin_ship_idx": soft_ship["idx"],
            "softmin_ship_soft": soft_ship["soft_mean"], "softmin_ship_hard": soft_ship["hard_mean"],
            "hardmin_select_idx": hard_ship["idx"],
            "hardmin_select_soft": hard_ship["soft_mean"], "hardmin_select_hard": hard_ship["hard_mean"],
            "changed": changed,
        }
    (OUT / "rerank_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT/'rerank_report.json'}; staged arms under {OUT}/(softmin|hardmin|hardmin_select)/<game>/")


if __name__ == "__main__":
    main()
