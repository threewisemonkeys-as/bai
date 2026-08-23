#!/usr/bin/env python3
"""WorldCoder program-WM vs GEPA language-WM: final A/B tables + plots.

Aggregates every eval-of-record into one report + one overview figure:

  - test50 inverse dynamics, exact AND sim-grounded, per game x arm
    (raw LLM / learned LLM (P+B) / program) -- the sim rescore is the fair
    aliasing-aware comparison, all arms scored on identical rows;
  - test50 forward dynamics: program grid-exact + cell-F1 vs stale floor
    (LLM learned-mode FD is feature-space -- shown for reference only);
  - planning success vs horizon (offline one-shot + online MPC), per game,
    per arm incl. program-search + hybrid, with the random floor;
  - learn-time cost per game (GEPA task+reflection vs WC reflection-only).

Tolerates missing inputs (renders what exists). Re-run any time:

    uv run python offline_learning/scripts/compare_wc_viz.py
"""
from __future__ import annotations

import os as _bos, sys as _bsys  # offline_learning/ on sys.path
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

AB_GAMES = ["dq8gc", "bt3gb", "n2ntd", "s2kt7", "83wkq"]
ARM_COLORS = {"rand": "#bbbbbb", "raw": "#7f7f7f", "learned": "#1f77b4",
              "program": "#d62728", "hybrid": "#9467bd"}


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def by_game(payload, key="results"):
    return {r["game"]: r for r in (payload or {}).get(key, [])}


def gepa_cost(artifact_dir: str):
    """(task_usd, refl_usd, refl_calls) parsed from the paired run's stdout."""
    st = Path(artifact_dir) / "stdout.txt"
    if not st.exists():
        return None
    text = st.read_text()
    task = re.findall(r"task_lm\) cost=\$([0-9.]+)", text) or \
        re.findall(r"F \(task_lm\) cost=\$([0-9.]+)", text)
    refl = re.findall(r"reflection cost=\$([0-9.]+) \((\d+) calls\)", text)
    if not refl:
        return None
    return (float(task[-1]) if task else None, float(refl[-1][0]), int(refl[-1][1]))


def collect(args):
    d = {
        "id_llm": by_game(load(args.id_json)),
        "sim_llm": by_game(load(args.sim_json)),
        "id_prog": by_game(load(args.program_json)),
        "sim_prog": by_game(load(args.program_sim_json)),
        "wc_summaries": {}, "plan": {}, "online": {},
    }
    for g in AB_GAMES:
        s = load(args.sweep_root / f"{g}_seed1" / "test_summary_wc_seed1.json")
        if s:
            d["wc_summaries"][g] = s
    for p in args.plan_jsons:
        for g, r in by_game(load(Path(p))).items():
            d["plan"][g] = r
    for p in args.online_jsons:
        for g, r in by_game(load(Path(p))).items():
            d["online"][g] = r
    return d


def fmt(v, n=3):
    return "—" if v is None else f"{v:.{n}f}"


def build_report(d) -> str:
    lines = ["# WorldCoder program WM vs GEPA language WM — A/B report", ""]

    lines += ["## test50 inverse dynamics (exact / sim-grounded)", "",
              "| game | raw exact | raw sim | learned exact | learned sim | "
              "program strict | program sim | program set-credit |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for g in AB_GAMES:
        idl = d["id_llm"].get(g)
        sml = (d["sim_llm"].get(g) or {}).get("summary", {})
        idp = d["id_prog"].get(g)
        smp = (d["sim_prog"].get(g) or {}).get("summary", {})
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            g,
            fmt(idl and idl["summary"]["raw"]["exact"]),
            fmt((sml.get("raw") or {}).get("sim")),
            fmt(idl and idl["summary"]["learned"]["exact"]),
            fmt((sml.get("learned") or {}).get("sim")),
            fmt(idp and idp["summary"]["program"]["exact"]),
            fmt((smp.get("program") or {}).get("sim")),
            fmt(idp and idp["summary"]["program"]["set_credit"]),
        ))

    lines += ["", "## test50 forward dynamics (program arm, raw-grid space)", "",
              "| game | program exact | program cell-F1 | stale floor | "
              "program errors | learned-LLM FD (feature-space, ref only) |",
              "|---|---:|---:|---:|---:|---:|"]
    for g in AB_GAMES:
        idp = d["id_prog"].get(g)
        f = idp and idp["fd_summary"]["program"]
        wc = d["wc_summaries"].get(g)
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            g, fmt(f and f["exact"]), fmt(f and f["cell_f1"]),
            fmt(f and f["stale_exact"]), f and f["program_errors"] or 0,
            "(see gepa test_summary)" if not wc else "—"))

    if d["plan"]:
        lines += ["", "## Planning success by horizon (offline one-shot, engine-executed)", ""]
        for g, r in d["plan"].items():
            hs = sorted(int(h) for h in r["summary"])
            modes = sorted({m for h in r["summary"].values() for m in h["plan"]})
            lines += [f"**{g}** (env_seed {r.get('env_seed')})", "",
                      "| arm | " + " | ".join(f"h={h}" for h in hs) + " |",
                      "|---|" + "---:|" * len(hs)]
            lines.append("| random | " + " | ".join(
                fmt(r["summary"][str(h)]["plan"]["raw"]["random_success"], 2)
                for h in hs) + " |")
            for m in modes:
                lines.append(f"| {m} | " + " | ".join(
                    fmt((r["summary"][str(h)]["plan"].get(m) or {}).get("success"), 2)
                    for h in hs) + " |")
            lines.append("")

    if d["online"]:
        lines += ["## Online (MPC) planning success by horizon", ""]
        for g, r in d["online"].items():
            hs = sorted(int(h) for h in r["summary"])
            modes = sorted({m for h in r["summary"].values() for m in h["plan"]})
            lines += [f"**{g}**", "",
                      "| arm | " + " | ".join(f"h={h}" for h in hs) + " |",
                      "|---|" + "---:|" * len(hs)]
            for m in modes:
                lines.append(f"| {m} | " + " | ".join(
                    fmt((r["summary"][str(h)]["plan"].get(m) or {}).get("success"), 2)
                    for h in hs) + " |")
            lines.append("")

    lines += ["## Learn-time budget per game", "",
              "| game | GEPA task $ | GEPA refl $ (calls) | WC refl $ (calls) | "
              "WC task $ | WC wall (s) | WC shipped val | WC fit_changed (train) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for g in AB_GAMES:
        idl = d["id_llm"].get(g)
        gc = gepa_cost(idl["artifact_dir"]) if idl else None
        wc = d["wc_summaries"].get(g)
        b = wc and wc["budget"]
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            g,
            fmt(gc and gc[0], 2), f"{fmt(gc and gc[1], 2)} ({gc[2]})" if gc else "—",
            f"{fmt(b and b['reflection_cost_usd'], 2)} ({b['reflection_calls']})" if b else "—",
            "0.00" if b else "—", fmt(b and b["wall_s"], 0),
            fmt(wc and wc["val"]["shipped_val_balanced"], 2),
            fmt(wc and wc["train"]["shipped_train_fit_changed"], 2)))
    return "\n".join(lines) + "\n"


def build_figure(d, out_png: Path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("WorldCoder program WM vs GEPA language WM (5-game A/B)", fontsize=14)

    # (0,0) sim-grounded test50 ID bars
    ax = axes[0][0]
    arms = [("raw", "sim_llm", "raw"), ("learned", "sim_llm", "learned"),
            ("program", "sim_prog", "program")]
    width = 0.25
    xs = range(len(AB_GAMES))
    for i, (label, src, key) in enumerate(arms):
        vals = [((d[src].get(g) or {}).get("summary", {}).get(key) or {}).get("sim")
                for g in AB_GAMES]
        ax.bar([x + (i - 1) * width for x in xs],
               [v if v is not None else 0 for v in vals],
               width, label=label, color=ARM_COLORS[label],
               alpha=[1.0 if v is not None else 0.15 for v in vals][0])
        for x, v in zip(xs, vals):
            if v is not None:
                ax.text(x + (i - 1) * width, v + 0.01, f"{v:.2f}",
                        ha="center", fontsize=7)
    ax.set_xticks(list(xs)); ax.set_xticklabels(AB_GAMES)
    ax.set_ylim(0, 1.12); ax.legend(fontsize=8)
    ax.set_title("test50 ID, sim-grounded (identical rows)")

    # (0,1) program FD vs stale
    ax = axes[0][1]
    fd = [(d["id_prog"].get(g) or {}).get("fd_summary", {}).get("program") for g in AB_GAMES]
    ax.bar([x - 0.15 for x in xs], [(f or {}).get("exact") or 0 for f in fd], 0.3,
           label="program FD exact", color=ARM_COLORS["program"])
    ax.bar([x + 0.15 for x in xs], [(f or {}).get("stale_exact") or 0 for f in fd], 0.3,
           label="stale (identity) floor", color="#cccccc")
    ax.plot(list(xs), [(f or {}).get("cell_f1") or 0 for f in fd], "k.--",
            label="program cell-F1")
    ax.set_xticks(list(xs)); ax.set_xticklabels(AB_GAMES)
    ax.set_ylim(0, 1.12); ax.legend(fontsize=8)
    ax.set_title("test50 FD, raw-grid space (program arm)")

    # (1,0) planning curves (offline solid, online dashed) for games present
    ax = axes[1][0]
    for g, r in d["plan"].items():
        hs = sorted(int(h) for h in r["summary"])
        for m in ("raw", "learned", "program", "hybrid"):
            vals = [(r["summary"][str(h)]["plan"].get(m) or {}).get("success") for h in hs]
            if any(v is not None for v in vals):
                ax.plot(hs, vals, "-o", color=ARM_COLORS[m], label=f"{m} offline", ms=4)
        ax.plot(hs, [r["summary"][str(h)]["plan"]["raw"]["random_success"] for h in hs],
                ":", color=ARM_COLORS["rand"], label="random")
        onr = d["online"].get(g)
        if onr:
            for m in ("raw", "learned", "program"):
                vals = [(onr["summary"].get(str(h), {}).get("plan", {}).get(m) or {}).get("success")
                        for h in hs]
                if any(v is not None for v in vals):
                    ax.plot(hs, vals, "--s", color=ARM_COLORS[m],
                            label=f"{m} online", ms=4, alpha=0.6)
        break  # one game per panel; extra games go to the per-game figure
    ax.set_xlabel("horizon h"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title(f"planning success ({next(iter(d['plan']), '—')})")

    # (1,1) learn cost bars (log scale)
    ax = axes[1][1]
    gep_t, gep_r, wc_r = [], [], []
    for g in AB_GAMES:
        idl = d["id_llm"].get(g)
        gc = gepa_cost(idl["artifact_dir"]) if idl else None
        wc = d["wc_summaries"].get(g)
        gep_t.append(gc[0] if gc and gc[0] else 0)
        gep_r.append(gc[1] if gc else 0)
        wc_r.append(wc["budget"]["reflection_cost_usd"] if wc else 0)
    ax.bar([x - 0.15 for x in xs], gep_t, 0.3, label="GEPA task $", color="#1f77b4")
    ax.bar([x - 0.15 for x in xs], gep_r, 0.3, bottom=gep_t,
           label="GEPA reflection $", color="#aec7e8")
    ax.bar([x + 0.15 for x in xs], wc_r, 0.3, label="WC reflection $ (task=$0)",
           color=ARM_COLORS["program"])
    ax.set_xticks(list(xs)); ax.set_xticklabels(AB_GAMES)
    ax.set_ylabel("USD"); ax.legend(fontsize=8)
    ax.set_title("learn-time cost per game")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def build_per_game_planning(d, out_png: Path):
    games = [g for g in d["plan"]]
    if not games:
        return False
    fig, axes = plt.subplots(1, len(games), figsize=(4.2 * len(games), 3.8),
                             squeeze=False)
    for ax, g in zip(axes[0], games):
        r = d["plan"][g]
        hs = sorted(int(h) for h in r["summary"])
        ax.plot(hs, [r["summary"][str(h)]["plan"]["raw"]["random_success"] for h in hs],
                ":", color=ARM_COLORS["rand"], label="random")
        for m in ("raw", "learned", "program", "hybrid"):
            vals = [(r["summary"][str(h)]["plan"].get(m) or {}).get("success") for h in hs]
            if any(v is not None for v in vals):
                ax.plot(hs, vals, "-o", color=ARM_COLORS[m], label=m, ms=4)
        onr = d["online"].get(g)
        if onr:
            for m in ("learned", "program"):
                vals = [(onr["summary"].get(str(h), {}).get("plan", {}).get(m) or {}).get("success")
                        for h in hs]
                if any(v is not None for v in vals):
                    ax.plot(hs, vals, "--s", color=ARM_COLORS[m],
                            label=f"{m} MPC", ms=4, alpha=0.6)
        ax.set_title(g); ax.set_xlabel("h"); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=6)
    fig.suptitle("Planning success by horizon (engine-executed)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id-json", type=Path,
                    default=REPO / "logs/id_eval_test50_raw_vs_learned.json")
    ap.add_argument("--sim-json", type=Path,
                    default=REPO / "logs/id_eval_test50_sim_rescore.json")
    ap.add_argument("--program-json", type=Path,
                    default=REPO / "logs/id_eval_test50_program.json")
    ap.add_argument("--program-sim-json", type=Path,
                    default=REPO / "logs/id_eval_test50_program_sim_rescore.json")
    ap.add_argument("--sweep-root", type=Path, default=REPO / "logs/aug7_wc_sweep")
    ap.add_argument("--plan-jsons", nargs="*",
                    default=[REPO / "logs/msplan_wc_dq8gc.json",
                             REPO / "logs/msplan_wc_5game.json"])
    ap.add_argument("--online-jsons", nargs="*",
                    default=[REPO / "logs/online_plan_wc_dq8gc.json",
                             REPO / "logs/online_plan_wc_5game.json"])
    ap.add_argument("--out-dir", type=Path, default=REPO / "logs/wc_ab_report")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    d = collect(args)
    report = build_report(d)
    (args.out_dir / "report.md").write_text(report)
    build_figure(d, args.out_dir / "wc_ab_overview.png")
    per_game = build_per_game_planning(d, args.out_dir / "wc_ab_planning_per_game.png")
    print(report)
    print(f"wrote {args.out_dir}/report.md, wc_ab_overview.png"
          + (", wc_ab_planning_per_game.png" if per_game else ""), flush=True)


if __name__ == "__main__":
    main()
