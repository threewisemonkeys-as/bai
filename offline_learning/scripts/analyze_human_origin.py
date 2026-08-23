"""Compare the human-origin runs against their artificial-data references.

Both arms use byte-identical learner configs (see launch_human_origin.py); the only
difference is where the (X_t, a_t, X_t+1) transitions came from. So a per-game delta is
attributable to data provenance -- with one caveat the report makes explicit: the two
arms have DIFFERENT test sets (each dataset carries its own held-out pool), so absolute
accuracies are not on a common scale. Lift over that test set's own raw-frame baseline
is, which is why it is the headline column for rexpure.

  rexpure     inverse_accuracy from test_summary_rexpure_seed1.json,
              baselines parsed from the run's stdout ("[test baselines] ...")
  worldcoder  program_fd.fit_all / program_id.strict / stale_fd_exact from
              test_summary_wc_seed1.json

Usage:
    uv run python offline_learning/scripts/analyze_human_origin.py \
        --out logs/aug10_human_origin
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GAMES = ["bt3gb", "dq8gc", "n2ntd", "s2kt7", "83wkq"]
REF = {
    "rexpure": ROOT / "logs/batch3_consolidated/{game}_s1_batch3",
    "worldcoder": ROOT / "logs/wc_seed1_consolidated/{game}_s1_wc",
}


def _f(x, n=3):
    return None if x is None else round(float(x), n)


def read_rexpure(d: Path) -> dict | None:
    s = d / "test_summary_rexpure_seed1.json"
    if not s.exists():
        return None
    j = json.loads(s.read_text())
    out = {"test_id": _f(j.get("inverse_accuracy")),
           "train_score": _f(j.get("best_train_score")),
           "n_test": j.get("n_test"), "nodes": j.get("nodes_explored")}
    log = d / "stdout.txt"
    if log.exists():
        t = log.read_text(errors="ignore")
        m = re.search(r"\[test baselines\] random=([\d.]+) \| start-P=([\d.]+) \| "
                      r"raw-frame=([\d.]+)", t)
        if m:
            out |= {"random": float(m.group(1)), "start_p": float(m.group(2)),
                    "raw_frame": float(m.group(3))}
        m = re.search(r"transitions: (\d+) \| train=(\d+) test=(\d+)", t)
        if m:
            out |= {"pool": int(m.group(1)), "n_train": int(m.group(2))}
        m = re.search(r"train action balance: (\{.*?\})", t)
        if m:
            out["train_balance"] = m.group(1)
    if out.get("raw_frame") is not None and out["test_id"] is not None:
        out["lift_over_raw"] = _f(out["test_id"] - out["raw_frame"])
    return out


def read_worldcoder(d: Path) -> dict | None:
    s = d / "test_summary_wc_seed1.json"
    if not s.exists():
        return None
    j = json.loads(s.read_text())
    t, b = j.get("test", {}), j.get("budget", {})
    return {"fd_exact": _f(t.get("program_fd", {}).get("fit_all")),
            "fd_changed": _f(t.get("program_fd", {}).get("fit_changed")),
            "fd_static": _f(t.get("program_fd", {}).get("fit_static")),
            "stale_fd": _f(t.get("stale_fd_exact")),
            "id_strict": _f(t.get("program_id", {}).get("strict")),
            "id_set_credit": _f(t.get("program_id", {}).get("set_credit")),
            "n_test": t.get("n_test"),
            "train_fit": _f(j.get("train", {}).get("shipped_train_balanced")),
            "val_fit": _f(j.get("val", {}).get("shipped_val_balanced")),
            "n_train": j.get("train", {}).get("n"), "n_val": j.get("val", {}).get("n"),
            "cost_usd": _f(b.get("reflection_cost_usd"), 4),
            "wall_s": _f(b.get("wall_s"), 0)}


READERS = {"rexpure": read_rexpure, "worldcoder": read_worldcoder}


def dataset_facts(game: str, variant: str) -> dict:
    p = ROOT / "offline_learning/human_data" / game / variant
    if not (p / "MANIFEST.json").exists():
        return {}
    m = json.loads((p / "MANIFEST.json").read_text())
    v = json.loads((ROOT / "offline_learning/human_data/VALIDATION.json").read_text()) \
        if (ROOT / "offline_learning/human_data/VALIDATION.json").exists() else []
    val = next((r for r in v if r["game"] == game and r["variant"] == variant), {})
    if not (val.get("oracle_test") or {}).get("ceiling"):
        # every variant of a game shares one test set (verified byte-identical), so the
        # ceiling measured on `informative` applies to all of them
        val = next((r for r in v if r["game"] == game
                    and (r.get("oracle_test") or {}).get("ceiling") is not None), val)
    return {"train_pool": m["stats"]["train"]["n_targets"],
            "test_pool": m["stats"]["test"]["n_targets"],
            "train_verbs": m["stats"]["train"]["verbs"],
            "test_verbs": m["stats"]["test"]["verbs"],
            "train_users": [d["user_id"] for d in m["drives"]["train"]],
            "test_users": [d["user_id"] for d in m["drives"]["test"]],
            "oracle_ceiling": (val.get("oracle_test") or {}).get("ceiling")}


def ref_ceilings(out: Path) -> dict:
    """Oracle ID ceilings of the ARTIFICIAL test sets (scripts/oracle_ceiling_ref.py)."""
    merged = {}
    for p in sorted(out.glob("ref_ceiling*.json")):
        for g, r in json.loads(p.read_text()).items():
            if isinstance(r, dict) and r.get("ceiling") is not None:
                merged[g] = r["ceiling"]
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "logs/aug10_human_origin"))
    ap.add_argument("--games", default=",".join(GAMES))
    args = ap.parse_args()
    out = Path(args.out)
    refc = ref_ceilings(out)
    games = args.games.split(",")
    variant = {"rexpure": "informative", "worldcoder": "informative_wc"}

    results = {"learners": {}}
    for learner, reader in READERS.items():
        rows = []
        for g in games:
            human = reader(out / learner / f"{g}_s1")
            art = reader(Path(str(REF[learner]).format(game=g)))
            ds = dataset_facts(g, variant[learner])
            ds["artificial_oracle_ceiling"] = refc.get(g)
            row = {"game": g, "human": human, "artificial": art, "dataset": ds}
            if learner == "rexpure":                       # unfiltered-human control arm
                row["human_raw"] = reader(out / f"{learner}_raw" / f"{g}_s1")
            rows.append(row)
        results["learners"][learner] = rows

    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    lines = ["# Human-origin vs artificial training data\n"]
    for learner, rows in results["learners"].items():
        lines.append(f"\n## {learner}\n")
        if learner == "rexpure":
            lines.append("Lift = test ID − that test set's own raw-frame baseline. The three "
                         "arms have different test sets, so lift is the comparable column, "
                         "not absolute ID.\n")
            lines.append("| game | human-filtered ID | human-raw ID | artificial ID "
                         "| filtered lift | raw lift | artificial lift "
                         "| human oracle | art oracle |")
            lines.append("|---|---|---|---|---|---|---|---|---|")
            acc = {"filtered": [], "raw": [], "artificial": []}
            for r in rows:
                h, a = r["human"] or {}, r["artificial"] or {}
                w = r.get("human_raw") or {}
                for key, src in (("filtered", h), ("raw", w), ("artificial", a)):
                    if src.get("lift_over_raw") is not None:
                        acc[key].append(src["lift_over_raw"])
                d = r["dataset"]
                lines.append(
                    f"| {r['game']} | {h.get('test_id')} | {w.get('test_id')} "
                    f"| {a.get('test_id')} | {h.get('lift_over_raw')} "
                    f"| {w.get('lift_over_raw')} | {a.get('lift_over_raw')} "
                    f"| {d.get('oracle_ceiling')} | {d.get('artificial_oracle_ceiling')} |")
            lines.append("")
            for key, v in acc.items():
                if v:
                    lines.append(f"- mean {key} lift: **{statistics.mean(v):+.3f}** "
                                 f"({len(v)} games)")
        else:
            lines.append("FD-exact is not comparable across arms: the human test sets are "
                         "filtered to observable transitions, so their stale (predict-no-"
                         "change) floor is near zero while the artificial sets score 0.14-"
                         "0.34 for free. FD lift over that floor is the comparable column.\n")
            lines.append("| game | human FD | art FD | human stale | art stale "
                         "| human FD lift | art FD lift | human ID | art ID "
                         "| human val fit | art val fit |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
            hl, al = [], []
            for r in rows:
                h, a = r["human"] or {}, r["artificial"] or {}
                hlift = (None if h.get("fd_exact") is None or h.get("stale_fd") is None
                         else round(h["fd_exact"] - h["stale_fd"], 3))
                alift = (None if a.get("fd_exact") is None or a.get("stale_fd") is None
                         else round(a["fd_exact"] - a["stale_fd"], 3))
                if hlift is not None:
                    hl.append(hlift)
                if alift is not None:
                    al.append(alift)
                lines.append(
                    f"| {r['game']} | {h.get('fd_exact')} | {a.get('fd_exact')} "
                    f"| {h.get('stale_fd')} | {a.get('stale_fd')} | {hlift} | {alift} "
                    f"| {h.get('id_strict')} | {a.get('id_strict')} "
                    f"| {h.get('val_fit')} | {a.get('val_fit')} |")
            if hl and al:
                lines.append(f"\n- mean FD lift: human **{statistics.mean(hl):+.3f}** vs "
                             f"artificial **{statistics.mean(al):+.3f}**")
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {out/'results.json'}  {out/'RESULTS.md'}")


if __name__ == "__main__":
    main()
