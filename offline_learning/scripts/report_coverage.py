"""Format the per-mechanic coverage scores (from score_coverage.py --out) into clean
lines-free tables, both raw and ceiling-normalised, without any further model calls.

    uv run python offline_learning/scripts/report_coverage.py \
        logs/2026-08-11/human_unified/coverage_scores.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from offline_learning.human_replay import GAMES  # noqa: E402
from offline_learning.mechanics import MECHANICS  # noqa: E402

NAME = {c: GAMES[c][1] for c in GAMES}
ARMS = ["raw", "wc", "lmwm"]


def agg(rows, mids, arms, norm):
    sel = [r for r in rows if r["mechanic"] in mids]
    if not sel:
        return None
    vr = [r for r in sel if r["max"]]
    ceil = sum(r["max"] for r in vr) / len(vr) if vr else None
    out = {"n": len(sel), "synth": sum(1 for r in sel if r["synthetic"]), "ceiling": ceil}
    for a in arms:
        if norm:
            out[a] = (sum(r[a] for r in vr) / len(vr) / ceil) if ceil else None
        else:
            out[a] = sum(r[a] for r in sel) / len(sel)
    return out


def fmt(v):
    return f"{v:5.2f}" if v is not None else "  -  "


def table(game, rows, norm):
    tag = "normalised (score / engine-ceiling)" if norm else "raw ID (set-credit, chance 0.20)"
    print(f"\n{game} / {NAME[game]}   {tag}")
    print(f"  {'mechanic':<20} {'src':<5} {'n':>2} {'ceil':>5} "
          + " ".join(f"{a:>5}" for a in ARMS))
    for m in MECHANICS[game]:
        a = agg(rows, {m["id"]}, ARMS, norm)
        if not a:
            continue
        src = "synth" if a["synth"] == a["n"] else "human"
        print(f"  {m['id']:<20} {src:<5} {a['n']:>2} {fmt(a['ceiling'])} "
              + " ".join(fmt(a[x]) for x in ARMS))
    ov = agg(rows, {m["id"] for m in MECHANICS[game]}, ARMS, norm)
    print(f"  {'OVERALL':<20} {'':<5} {ov['n']:>2} {fmt(ov['ceiling'])} "
          + " ".join(fmt(ov[x]) for x in ARMS))


def main():
    data = json.loads(Path(sys.argv[1]).read_text())
    for norm in (False, True):
        print("=" * 60)
        print("RAW" if not norm else "NORMALISED")
        for g in GAMES:
            if g in data:
                table(g, data[g]["rows"], norm)
    # grand mean over all items
    allrows = [r for g in data for r in data[g]["rows"]]
    print("\n" + "=" * 60)
    for norm in (False, True):
        ov = agg(allrows, {r["mechanic"] for r in allrows}, ARMS, norm)
        lab = "normalised" if norm else "raw"
        print(f"ALL GAMES ({ov['n']} items) {lab:11s}: "
              + "  ".join(f"{a}={fmt(ov[a])}" for a in ARMS) + f"   ceiling={fmt(ov['ceiling'])}")


if __name__ == "__main__":
    main()
