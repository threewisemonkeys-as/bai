"""Result table for a human-origin training tree (launch_human_origin.py output).

    uv run python offline_learning/scripts/summarize_human_runs.py \
        --root logs/2026-08-24/human_curated --ref logs/2026-08-11/human_unified

Per game: rexpure credited test ID (`inverse_accuracy`, set-credit ~1/|S|), strict
singleton accuracy, train composite, nodes; worldcoder credited test ID (`set_credit`,
the comparable number -- NEVER headline wc `strict`), program FD fit on changed
transitions, budget. `--ref` adds the same numbers from a reference tree (delta in
parentheses) for games present in both. Rows for unfinished runs are marked `..`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GAMES = ["eahcw", "egg", "bt3gb", "dq8gc", "7xf97", "n2ntd", "va6fq", "s2kt7",
         "colour_lines", "SET", "diffusion", "dino", "f5w3n", "logic_gates", "7www9"]


def rex(root: Path, g: str) -> dict | None:
    p = root / "rexpure" / f"{g}_s1" / "test_summary_rexpure_seed1.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    raw = root / "rexpure" / f"{g}_s1" / "test_trace_raw_seed1.json"
    raw_acc = json.loads(raw.read_text()).get("acc") if raw.exists() else None
    cost = None
    so = root / "rexpure" / f"{g}_s1" / "stdout.txt"
    if so.exists():
        for line in so.read_text().splitlines()[::-1]:
            if "F (task_lm) cost=" in line:
                try:
                    f = float(line.split("cost=$")[1].split()[0])
                    r = float(line.split("reflection cost=$")[1].split()[0])
                    cost = f + r
                except (IndexError, ValueError):
                    pass
                break
    # z-blind check: every test target has a changed frame, so z_t == z_t1 means the
    # shipped P did not see the change (the stale-feature artifact behind fake FD=1.00).
    stale = err = None
    tr = root / "rexpure" / f"{g}_s1" / d.get("inverse_trace", "test_trace_rexpure_seed1.json")
    if tr.exists():
        recs = json.loads(tr.read_text()).get("records") or []
        if recs:
            stale = sum(r.get("z_t") == r.get("z_t1") for r in recs) / len(recs)
            err = sum("error" in str(r.get("z_t", "")).lower() for r in recs) / len(recs)
    return {"id": d.get("inverse_accuracy"), "strict": (d.get("inverse_set") or {}).get("strict_singleton_accuracy"),
            "raw": raw_acc, "train": d.get("best_train_score"), "nodes": d.get("nodes_explored"), "cost": cost,
            "stale": stale, "err": err}


def wc(root: Path, g: str) -> dict | None:
    p = root / "worldcoder" / f"{g}_s1" / "test_summary_wc_seed1.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    t = d["test"]
    return {"id": t["program_id"]["set_credit"], "strict": t["program_id"]["strict"],
            "fd": t["program_fd"]["fit_changed"], "stale": t.get("stale_fd_exact"),
            "collapse": t["id_protocol"]["collapse"],
            "train": d["train"].get("shipped_train_fit_changed"),
            "cost": d["budget"]["reflection_cost_usd"], "wall": d["budget"]["wall_s"]}


def f(x, ref=None, nd=2):
    if x is None:
        return ".."
    s = f"{x:.{nd}f}" if isinstance(x, float) else str(x)
    if ref is not None and isinstance(x, (int, float)) and isinstance(ref, (int, float)):
        s += f" ({x - ref:+.{nd}f})"
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "logs/2026-08-24/human_curated"))
    ap.add_argument("--ref", default=None, help="reference tree for deltas")
    ap.add_argument("--games", default=",".join(GAMES))
    ap.add_argument("--out", default=None, help="write markdown here too")
    a = ap.parse_args()
    root, ref = Path(a.root), (Path(a.ref) if a.ref else None)
    games = a.games.split(",")
    lines = [f"# Human-origin runs: `{root.relative_to(REPO) if root.is_relative_to(REPO) else root}`"
             + (f" vs `{a.ref}`" if ref else ""), "",
             "Credited test ID = rexpure `inverse_accuracy` / worldcoder `set_credit` (same ~1/|S| rule). "
             "Deltas in parentheses are against the reference tree.", "",
             "`rex z-blind` = share of test targets whose shipped-P features are identical at t and t+1 "
             "(every target's frame changed, so >0 means P misses it); `P err` = share with an error string in "
             "the features; `wc stale` = wc's `stale_fd_exact` (program output == input frame).", "",
             "| game | rex raw ID | rex ID | rex strict | rex z-blind | P err | rex train | nodes | rex $ "
             "| wc ID | wc strict | wc FD(changed) | wc stale | wc train FD | wc $ | wc wall |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    agg = {"rex": [], "wc": [], "rex_ref": [], "wc_ref": []}
    for g in games:
        r, w = rex(root, g), wc(root, g)
        rr = rex(ref, g) if ref else None
        wr = wc(ref, g) if ref else None
        row = [g]
        if r:
            row += [f(r["raw"], rr and rr["raw"]), f(r["id"], rr and rr["id"]), f(r["strict"], rr and rr["strict"]),
                    f(r["stale"]), f(r["err"]), f(r["train"], rr and rr["train"]), f(r["nodes"]), f(r["cost"])]
            agg["rex"].append(r["id"])
            if rr and rr["id"] is not None:
                agg["rex_ref"].append((r["id"], rr["id"]))
        else:
            row += [".."] * 8
        if w:
            row += [f(w["id"], wr and wr["id"]), f(w["strict"], wr and wr["strict"]), f(w["fd"], wr and wr["fd"]),
                    f(w["stale"]), f(w["train"], wr and wr["train"]), f(w["cost"], nd=3), f"{w['wall'] / 60:.0f}m"]
            agg["wc"].append(w["id"])
            if wr:
                agg["wc_ref"].append((w["id"], wr["id"]))
        else:
            row += [".."] * 7
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    for k, lab in (("rex", "rexpure"), ("wc", "worldcoder")):
        v = [x for x in agg[k] if x is not None]
        if v:
            s = f"- {lab}: mean credited test ID {sum(v) / len(v):.3f} over {len(v)} games"
            pr = agg[k + "_ref"]
            if pr:
                s += (f"; paired vs reference over {len(pr)}: {sum(x for x, _ in pr) / len(pr):.3f} vs "
                      f"{sum(y for _, y in pr) / len(pr):.3f} ({sum(x - y for x, y in pr) / len(pr):+.3f}), "
                      f"wins {sum(x > y for x, y in pr)}/{len(pr)}")
            lines.append(s)
    txt = "\n".join(lines) + "\n"
    print(txt)
    if a.out:
        Path(a.out).write_text(txt)


if __name__ == "__main__":
    main()
