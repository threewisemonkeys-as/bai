#!/usr/bin/env python3
"""Side-by-side report: the no-perception ablation vs its learned-P reference.

Two tables, because the ablation has two distinct readouts:

  LEARNING   end-of-run inverse-dynamics accuracy on the held-out 50, plus each run's own
             no-belief control (`start-P`, printed by rexpure_optimize as a test baseline).
             For the ablation start-P is identity-P with EMPTY beliefs -- the raw-grid
             window with nothing learned -- so `final - start-P` is exactly what the belief
             block bought. For the reference start-P is the seed perception, so the two
             `final` columns are the comparison and the two deltas are NOT commensurable.

  PLANNING   the `lmwm` arm of the curated planning evals (OFFLINE open-loop and ONLINE
             receding-horizon), which is the arm that reads the artifacts. `raw` is carried
             along as the shared control -- it reads no artifact, so it should agree between
             the two runs up to sampling noise (identically, where the ONLINE checkpoint was
             seeded from the reference).

    uv run python offline_learning/scripts/compare_noperc_ablation.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GAMES = ["bt3gb", "dq8gc", "n2ntd", "s2kt7", "83wkq"]
BASE_RE = re.compile(r"random=([\d.]+) \| start-P=([\d.]+) \| raw-frame=([\d.]+)")


def learning_row(root: Path, game: str) -> dict:
    d = root / "rexpure" / f"{game}_s1"
    out: dict = {}
    s = d / "test_summary_rexpure_seed1.json"
    if s.exists():
        j = json.loads(s.read_text())
        out["final"] = j.get("inverse_accuracy")
        out["train"] = j.get("best_train_score")
        out["strict"] = (j.get("inverse_set") or {}).get("strict_singleton_accuracy")
    b = d / "resume_baselines.json"
    if b.exists():
        jb = json.loads(b.read_text())
        out["start"], out["raw"] = jb.get("start_acc"), jb.get("raw_acc")
    else:                                     # fall back to the console line
        log = d / "stdout.txt"
        if log.exists():
            m = BASE_RE.search(log.read_text())
            if m:
                out["start"], out["raw"] = float(m.group(2)), float(m.group(3))
    return out


def f(v, w=6):
    return f"{v:>{w}.2f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"


def plan_index(path: Path) -> dict:
    if not path.exists():
        return {}
    return {(r["game"], r["id"]): r for r in json.loads(path.read_text())["rows"]}


def plan_table(title: str, ref: dict, abl: dict) -> list[str]:
    if not ref and not abl:
        return [f"## {title}", "", "_(not run)_", ""]
    games = [g for g in GAMES if any(k[0] == g for k in (abl or ref))]
    L = [f"## {title}", "",
         "| game | n | lmwm ref | lmwm ABL | delta | raw ref | raw ABL |",
         "|---|--:|--:|--:|--:|--:|--:|"]
    for g in games:
        keys = [k for k in (abl or ref) if k[0] == g and k in ref and k in abl]
        if not keys:
            continue

        def m(idx, arm):
            v = [idx[k][arm]["pass_rate"] for k in keys if arm in idx[k]]
            return sum(v) / len(v) if v else None

        lr, la = m(ref, "lmwm"), m(abl, "lmwm")
        dl = (la - lr) if (lr is not None and la is not None) else None
        L.append(f"| {g} | {len(keys)} | {f(lr)} | {f(la)} | "
                 f"{('%+.2f' % dl) if dl is not None else '-':>6} | "
                 f"{f(m(ref, 'raw'))} | {f(m(abl, 'raw'))} |")
    both = [k for k in (abl or {}) if k in ref]
    if both:
        def pooled(idx, arm):
            v = [idx[k][arm]["pass_rate"] for k in both if arm in idx[k]]
            return sum(v) / len(v) if v else None
        lr, la = pooled(ref, "lmwm"), pooled(abl, "lmwm")
        dl = (la - lr) if (lr is not None and la is not None) else None
        L.append(f"| **all** | {len(both)} | {f(lr)} | {f(la)} | "
                 f"{('%+.2f' % dl) if dl is not None else '-':>6} | "
                 f"{f(pooled(ref, 'raw'))} | {f(pooled(abl, 'raw'))} |")
    return L + ["", "_Pooled row is a problem-count mean across games; per-game rows are the "
                "scored unit (games differ in difficulty and problem count)._", ""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", default=str(REPO / "logs/2026-08-19/noperc_ablation"))
    ap.add_argument("--reference", default=str(REPO / "logs/2026-08-11/human_unified"))
    ap.add_argument("--ablation-eval", default=str(REPO / "logs/2026-08-19/noperc_ablation/curated_eval"))
    ap.add_argument("--reference-eval", default=str(REPO / "logs/2026-08-18/curated/eval"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    abl_root, ref_root = Path(a.ablation), Path(a.reference)
    abl_ev, ref_ev = Path(a.ablation_eval), Path(a.reference_eval)

    L = ["# No-perception ablation vs learned-P reference", "",
         f"ablation  `{abl_root}`", f"reference `{ref_root}`", "",
         "## Learning: held-out inverse-dynamics accuracy (n=50)", "",
         "| game | ref start-P | ref final | ABL start-P | ABL final | final delta |",
         "|---|--:|--:|--:|--:|--:|"]
    for g in GAMES:
        r, b = learning_row(ref_root, g), learning_row(abl_root, g)
        d = ((b.get("final") - r.get("final"))
             if isinstance(b.get("final"), float) and isinstance(r.get("final"), float) else None)
        L.append(f"| {g} | {f(r.get('start'))} | {f(r.get('final'))} | "
                 f"{f(b.get('start'))} | {f(b.get('final'))} | "
                 f"{('%+.2f' % d) if d is not None else '-':>6} |")
    L += ["", "`start-P` is each run's OWN no-belief control (reference: seed perception; "
          "ablation: identity P = raw grid). The `final` columns are the comparison; the "
          "two start-P columns are different controls and must not be differenced against "
          "each other.", ""]

    L += plan_table("Planning: curated OFFLINE (open-loop) pass@1",
                    plan_index(ref_ev / "offline.json"), plan_index(abl_ev / "offline.json"))
    L += plan_table("Planning: curated ONLINE (receding horizon) pass@1",
                    plan_index(ref_ev / "online.json"), plan_index(abl_ev / "online.json"))

    md = "\n".join(L) + "\n"
    print(md)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(md)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
