#!/usr/bin/env python3
"""Per-drive review sheet for MANUAL drive curation (2026-08-24 decision).

For every replayable segment of a game: steps, per-verb informative counts (noop-
counterfactual at the recipe horizon), distinct frames, and the outcome SIGNATURES it
contains -- per-colour sign of the cell-count change over a transition, e.g. blue:-
= blue cells vanished (mario: enemy death), gold:- = coin pickup. Signatures are game-
agnostic; rare ones (low corpus count) are exactly the events activity ranking misses.

    uv run python offline_learning/scripts/drive_review_sheet.py --game 7xf97 \
        --out logs/2026-08-24/human_unified3_build/drive_sheets
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import Counter
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "offline_learning")); sys.path.insert(0, str(REPO))
import human_replay as H

def signature(g0, g1):
    c0, c1 = Counter(v for r in g0 for v in r), Counter(v for r in g1 for v in r)
    sig = []
    for col in sorted(set(c0) | set(c1)):
        d = c1[col] - c0[col]
        if d: sig.append(f"{col}{'+' if d > 0 else '-'}")
    return ",".join(sig) or "static"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--oov", default="noop")
    ap.add_argument("--variant", default="informative_unified3",
                    help="variant whose MANIFEST marks the currently-chosen drives")
    ap.add_argument("--out", default=str(REPO / "logs/2026-08-24/human_unified3_build/drive_sheets"))
    a = ap.parse_args()
    game = a.game; prog, human, wl = H.GAMES[game]
    sessions = H.load_sessions(game, H.DEFAULT_ZIP, REPO / "offline_learning/human_data/_cache")
    segs = [s for x in sessions for s in H.segment(x, set(wl), a.oov)]
    chosen = {}
    mpath = REPO / f"offline_learning/human_data/{game}/{a.variant}/MANIFEST.json"
    if mpath.exists():
        man = json.loads(mpath.read_text())
        chosen = {(d["user_id"], d["seed"], d["seg_idx"]): f"{sp}_d{i}"
                  for sp in ("train", "test") for i, d in enumerate(man["drives"][sp])}
    rows, corpus_sig = [], Counter()
    for s in segs:
        t0 = time.time()
        rep = H.replay(prog, s["seed"], s["actions"])
        gs = [json.loads(g) for g in rep["grids"]]
        idx = [i for i, x in enumerate(rep["actions"]) if x.split()[0] != "noop"]
        cf = H.noop_counterfactual(prog, s["seed"], rep["actions"], idx, horizon=a.horizon)
        cands = H.candidates(rep, cf)
        byv = Counter(c["verb"] for c in cands if c["informative"])
        sigs = Counter(signature(x, y) for x, y in zip(gs, gs[1:]))
        corpus_sig += sigs
        rows.append({
            "user": s["user_id"], "seed": s["seed"], "seg": s["seg_idx"],
            "steps": len(rep["actions"]), "nonnoop": H._nonnoop_count(s),
            "informative": sum(byv.values()), "by_verb": dict(byv),
            "distinct_frames": len({json.dumps(g) for g in gs}),
            "signatures": dict(sigs),
            "chosen": chosen.get((s["user_id"], s["seed"], s["seg_idx"]), ""),
        })
        print(f"[{game}] {s['user_id'][:8]}/{s['seed']}/{s['seg_idx']} "
              f"inf={rows[-1]['informative']} ({time.time()-t0:.0f}s)", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"{game}.json").write_text(json.dumps(
        {"game": game, "human": human, "whitelist": wl, "horizon": a.horizon,
         "corpus_signatures": dict(corpus_sig.most_common()), "segments": rows}, indent=1))
    rare = [s for s, n in corpus_sig.items() if n <= max(5, 0.002 * sum(corpus_sig.values()))]
    lines = [f"# {game} ({human}) — drive review sheet", "",
             f"{len(rows)} segments; whitelist {wl}; horizon {a.horizon}; oov {a.oov}.",
             f"RARE signatures (<=5 or 0.2% of corpus): {', '.join(sorted(rare)) or 'none'}", "",
             "| drive (user/seed/seg) | chosen | steps | informative (by verb) | rare events |",
             "|---|---|--:|---|---|"]
    for r in sorted(rows, key=lambda r: -r["informative"]):
        rr = {k: v for k, v in r["signatures"].items() if k in rare}
        lines.append(f"| {r['user'][:8]}/{r['seed']}/{r['seg']} | {r['chosen']} | {r['steps']} | "
                     f"{r['informative']} {r['by_verb']} | {rr if rr else ''} |")
    (out / f"{game}.md").write_text("\n".join(lines) + "\n")
    print(f"[{game}] sheet -> {out/f'{game}.md'}", flush=True)

if __name__ == "__main__":
    main()
