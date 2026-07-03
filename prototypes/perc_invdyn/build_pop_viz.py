#!/usr/bin/env python3
"""Build a self-contained HTML visualisation of a POPULATION (beam) search run
(pop_optimize.py), in the same style as build_optim_viz.py does for GEPA.

It joins:
  - nodes.jsonl       -> one record per node (seed + every child): nid, parent, gen,
                         active component, train score, kept?, the candidate (P+B),
                         and the exact proposer prompt + response that produced it.
  - predictions.jsonl -> per-(candidate, transition) inverse+forward detail, keyed by
                         the SAME content hashes gepa_optimize writes (so cells join 1:1).
  - the raw trajectory.csv (--run) -> ground-truth action + on-grid movement, reconstructed
                         into the SAME train/test split the run used (seeded, deterministic).

Output: <out-dir>/optim_viz.html  (open in any browser; no server needed).

The view mirrors the GEPA one: a TRANSITION x NODE matrix (every node ever born, in birth
order; cells green/red by inverse-dynamics correctness, click for the inverse+forward chain),
a per-generation timeline, the search-process TREE (seed -> children; kept = survived top-K,
dropped = pruned; click a node for the full proposer call incl. the mistakes shown to it),
candidate cards, and the held-out test rows for the best node.

Usage (matches the run that produced the dir):
  uv run python prototypes/perc_invdyn/build_pop_viz.py \
      --out-dir logs/perc_invdyn/pop_<ts>_seed1 \
      --run prototypes/perc_invdyn/clean_data/dq8gc \
      --seed 1 --train-n 20 --test-n 10 --tie-train-val --context-k 3 --actions ""
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

# Reuse build_optim_viz verbatim: identical data loading / split / grid / hashing, and the
# HTML+JS template. We only re-assemble the model dict from the population artifacts and
# relabel a few GEPA-specific strings for the beam-search semantics.
from build_optim_viz import (  # noqa: E402
    HTML,
    content_hash,
    grid_summary,
    load_predictions,
    movement,
    reconstruct_split,
    verb,
    window_payload,
)


def load_nodes(path: Path) -> list[dict]:
    out = []
    for line in path.open():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def feedback_blocks(entries) -> dict:
    """reflective dataset (list of records for the active component) -> {comp: [{feedback,out}]}
    in the same shape build_optim_viz uses (substance only)."""
    rows = []
    for e in (entries or []):
        if not isinstance(e, dict):
            rows.append({"feedback": str(e)})
            continue
        go = e.get("Generated Outputs") or {}
        rows.append({
            "feedback": e.get("Feedback", ""),
            "out": "  |  ".join(f"{k}: {v}" for k, v in go.items()) if isinstance(go, dict) else str(go),
        })
    return rows


def build_model(args):
    outd = Path(args.out_dir)
    train, test, action_pool, vec_actions = reconstruct_split(args)

    # global aliasing (same as build_optim_viz.build_model)
    def alias_partners(sig, own):
        return sorted(a for a in vec_actions.get(sig, {}) if a != own)
    sig_total = {s: sum(c.values()) for s, c in vec_actions.items()}

    nodes = load_nodes(outd / "nodes.jsonl")
    nodes_by_id = {n["nid"]: n for n in nodes}
    preds = load_predictions(outd)  # reads <dir>/predictions.jsonl -> (cand_hash, tr_hash) -> detail
    cand_hash_by_nid = {
        n["nid"]: content_hash(
            n["candidate"].get("perception", ""), n["candidate"].get("world_knowledge", "")
        )
        for n in nodes
    }

    # ---- train transitions (matrix rows) with movement + aliasing ----
    vinfo = []
    for i, item in enumerate(train):
        tr = item["tr"]
        mv = movement(tr.x_t, tr.x_t1)
        vinfo.append({
            "task": i, "action": verb(tr.action), "move": mv,
            "tr_hash": content_hash(tr.x_t, tr.x_t1, tr.action),
            "grid_t": grid_summary(tr.x_t), "grid_t1": grid_summary(tr.x_t1),
            "window": window_payload(tr),
        })
    for v in vinfo:
        sig = v["move"]["sig"]
        v["aliased_with"] = alias_partners(sig, v["action"])
        v["oneshot"] = sig_total.get(sig, 0) <= 1

    # ---- per-cell matrix from predictions.jsonl (complete: every node x train transition) ----
    cand_cols = sorted(nodes_by_id)
    cells = {}
    for task, v in enumerate(vinfo):
        truth = v["action"]
        for nid in cand_cols:
            det = preds.get((cand_hash_by_nid.get(nid), v["tr_hash"]))
            if not det:
                continue
            pred = det.get("pred")
            ok = (verb(pred) == truth) if pred is not None else (det.get("id_score", 0) >= 1.0)
            cell = {
                "ok": bool(ok), "label": verb(pred) if pred is not None else f"{det.get('score', 0):.2f}",
                "pred": verb(pred) if pred is not None else None,
                "score": det.get("score"), "src": "pred",
                "det": {
                    "pred": det.get("pred"), "id_score": det.get("id_score"),
                    "reasoning": det.get("reasoning", ""),
                    "z_t": det.get("z_t", ""), "z_t1": det.get("z_t1", ""),
                    "z_hat": det.get("z_hat", ""), "fd_score": det.get("fd_score"),
                    "fd_reasoning": det.get("fd_reasoning", ""), "score": det.get("score"),
                    "inv_prompt": det.get("inv_prompt", ""), "inv_response": det.get("inv_response", ""),
                    "fwd_prompt": det.get("fwd_prompt", ""), "fwd_response": det.get("fwd_response", ""),
                },
            }
            cells[f"{task},{nid}"] = cell

    # ---- candidate cards + role (best / kept / dropped) ----
    best_nid = max(nodes_by_id, key=lambda n: nodes_by_id[n]["score"])
    cand_meta = []
    for n in nodes:
        role = "Best" if n["nid"] == best_nid else ("" if n.get("kept", True) else "dropped")
        cand_meta.append({
            "idx": n["nid"], "score": round(n["score"], 3),
            "parents": "seed" if n["parent"] < 0 else f"node {n['parent']}",
            "role": role,
            "perception": n["candidate"].get("perception", ""),
            "world_knowledge": n["candidate"].get("world_knowledge", ""),
        })

    # ---- generation timeline: one row per child (parent -> child, train score) ----
    timeline = [
        {"i": n["gen"], "selected": n["parent"], "born": n["nid"], "val_mean": round(n["score"], 3)}
        for n in nodes if n["parent"] >= 0
    ]

    # ---- search process tree: seed -> children, kept(=accepted) / dropped(=rejected) ----
    tree_nodes = []
    for n in nodes:
        nid, parent = n["nid"], n["parent"]
        active = n.get("active") or ""
        if parent < 0:  # seed
            tree_nodes.append({
                "id": "c0", "kind": "seed", "i": 0, "parent": None, "cand": 0,
                "comp": [], "proposed": {}, "feedback": {}, "proposer": {},
                "label": f"node {nid}", "sub": "seed",
            })
            continue
        kind = "accepted" if n.get("kept") else "rejected"  # reuse gepa CSS: green=kept, red=dropped
        parent_score = nodes_by_id.get(parent, {}).get("score")
        tree_nodes.append({
            "id": f"c{nid}", "parent": f"c{parent}", "kind": kind,
            "i": n["gen"], "selected": parent, "selected_score": parent_score, "cand": nid,
            "comp": [active] if active else [],
            "proposed": {active: n["candidate"].get(active, "")} if active else {},
            "parent_text": nodes_by_id.get(parent, {}).get("candidate", {}),
            "feedback": {active: feedback_blocks(n.get("feedback"))} if active else {},
            "proposer": {active: {"prompt": n.get("prompt", ""), "response": n.get("response", "")}} if active else {},
            "old_score": parent_score, "new_score": n["score"], "reason": "",
            "label": f"node {nid}", "sub": f"{active} · {n['score']:.2f}",
        })

    # ---- held-out test rows (best node) ----
    trace_path = next(outd.glob("test_trace_pop_seed*.json"), None)
    verify = {"ok": None, "detail": ""}
    test_records = []
    if trace_path:
        tr = json.load(trace_path.open())
        recs = tr.get("records", [])
        recon = [verb(t["tr"].action) for t in test]
        saved = [verb(r["truth"]) for r in recs]
        verify["ok"] = recon == saved
        verify["detail"] = f"reconstructed test truths {'==' if verify['ok'] else '!='} test_trace truths"
        for i, r in enumerate(recs):
            tr_i = test[i]["tr"] if i < len(test) else None
            test_records.append({
                "idx": r["idx"], "truth": r["truth"], "pred": r["pred"], "correct": r["correct"],
                "z_t": r.get("z_t", ""), "z_t1": r.get("z_t1", ""), "reasoning": r.get("reasoning", ""),
                "grid_t": grid_summary(tr_i.x_t) if tr_i else None,
                "grid_t1": grid_summary(tr_i.x_t1) if tr_i else None,
                "move": movement(tr_i.x_t, tr_i.x_t1) if tr_i else None,
                "window": window_payload(tr_i) if tr_i else None,
            })

    return {
        "title": outd.name, "verify": verify, "action_pool": action_pool,
        "val": vinfo, "cand_cols": cand_cols, "cells": cells, "candidates": cand_meta,
        "timeline": timeline, "tree": tree_nodes, "test": test_records,
    }


# Relabel the GEPA template strings for beam-search semantics (the structure is identical).
RELABEL = [
    ("GEPA optimisation viewer — ", "Population search viewer — "),
    ("<h2>Iteration timeline</h2>", "<h2>Generation timeline</h2>"),
    ("<th>iter</th><th>selected parent</th><th>candidate born</th><th>full-val mean</th>",
     "<th>generation</th><th>parent node</th><th>child born</th><th>train score</th>"),
    ("read from <code>run_log.json</code>'s per-instance scores",
     "read from <code>predictions.jsonl</code>"),
    # process-tree legend
    ("Every proposal over the whole run, hung off the parent candidate it was mutated from.",
     "Every node born over the whole run, hung off the parent it was expanded from."),
    ("= accepted (passed the subsample gate)", "= kept (survived to the top-K population)"),
    ("= <b>rejected dead end</b> (proposed but not better on the subsample)",
     "= <b>dropped</b> (evaluated but pruned by the top-K cut)"),
    ("= skipped (empty/failed proposal)", "= (unused)"),
    # openProc panel
    ("<h4>Subsample gate</h4>", "<h4>Train score (parent → child)</h4>"),
    ("parent full-val score:", "parent train score:"),
    ("subsample before→after:", "child vs parent:"),
    (" — passed, added to pool", " — kept (top-K)"),
    (" — not better, rejected", " — dropped"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="the pop_optimize run dir (holds nodes.jsonl + predictions.jsonl)")
    ap.add_argument("--run", required=True, help="comma-separated data run dirs (same as the run used)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--train-n", type=int, default=20)
    ap.add_argument("--val-n", type=int, default=20)
    ap.add_argument("--test-n", type=int, default=10)
    ap.add_argument("--k-choices", type=int, default=5)
    ap.add_argument("--actions", default="")
    ap.add_argument("--tie-train-val", action="store_true",
                    help="match pop_optimize's split (it always uses train==val); pass this")
    ap.add_argument("--context-k", type=int, default=3)
    ap.add_argument("--swap-click-train", action="store_true",
        help="match a run launched with --swap-click-train so the reconstructed split aligns")
    ap.add_argument("--collapse-action-params", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # build_model needs out_dir + the attrs reconstruct_split reads; pop_optimize always
    # runs in the tied regime (train==val), so force tie_train_val=True.
    margs = SimpleNamespace(
        out_dir=args.out_dir, seed=args.seed, actions=args.actions, run=args.run,
        collapse_action_params=args.collapse_action_params, context_k=args.context_k,
        test_n=args.test_n, train_n=args.train_n, val_n=args.val_n,
        tie_train_val=True, k_choices=args.k_choices,
    )
    model = build_model(margs)

    out = Path(args.out) if args.out else Path(args.out_dir) / "optim_viz.html"
    html = HTML.replace("__TITLE__", model["title"]).replace("__DATA__", json.dumps(model))
    for a, b in RELABEL:
        html = html.replace(a, b)
    out.write_text(html)
    v = model["verify"]
    print(f"[viz] wrote {out}")
    print(f"[viz] alignment check: ok={v['ok']} ({v['detail']})")
    print(f"[viz] {len(model['val'])} train transitions | {len(model['cand_cols'])} node columns | "
          f"{sum(1 for x in model['val'] if x['aliased_with'])} aliased rows")


if __name__ == "__main__":
    main()
