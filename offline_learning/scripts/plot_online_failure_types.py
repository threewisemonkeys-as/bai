#!/usr/bin/env python3
"""Break down ONLINE planning FAILURES by type, per game and per method.

The eval's own `failed_reason` is near-degenerate — the LLM arms are ~100%
`budget-exhausted` and wc is 100% `invalid-plan` — which says nothing about why a
method missed. This script decomposes each failure by HOW FAR it actually got:
`dmin` = the fewest differing cells between any EXECUTED frame and the goal, against
`d0` = the cells already differing at the start. The four resulting types form a
severity ladder (never acted -> acted but never improved -> got closer -> within 2
cells), so they are coloured with an ordinal one-hue ramp, not categorical hues.

Reads the FULL online eval json (needs per-round `grid_after`, so the *_slim.json
will not do) plus the problem set it was run against. Emits png + svg + a markdown
table (the accessibility fallback for the colour encoding).

    uv run python offline_learning/scripts/plot_online_failure_types.py \
        --input    logs/2026-08-14/canonfix_v2/eval/coverage_online_eval_v2.json \
        --problems logs/2026-08-14/canonfix_v2/data/coverage_plan_problems_v2.json \
        --out-dir  logs/2026-08-14/canonfix_v2/plots
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO / "logs/coverage_online_eval.json"
DEFAULT_PROBLEMS = REPO / "logs/coverage_plan_problems.json"
DEFAULT_OUT_DIR = REPO / "logs"

PKEY = ("game", "seed", "t", "bucket", "mechanic", "h")
GAMES = [("bt3gb", "ice"), ("dq8gc", "disease"), ("n2ntd", "mario"),
         ("s2kt7", "ants"), ("83wkq", "particles")]
ARMS = [("raw", "Raw LLM"), ("lmwm", "NLWM"), ("wc", "WC")]

# dataviz reference palette: light surface + text tokens, and the blue ORDINAL ramp
# (steps 250/400/550/700). Validated: validate_palette.js --ordinal --mode light.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
TYPES = ["near miss", "partial progress", "no progress", "no plan found"]
LEGEND = ["Near miss", "Partial progress", "No progress", "No plan found"]
DEFS = ("Mutually exclusive, assigned top-down:\n"
        "  No plan found  never acted at all\n"
        "  No progress    never beat standing still\n"
        "  Partial        closer, still >2 cells off\n"
        "  Near miss      within 2 cells, never exact\n\n"
        "Bar height = failure rate;\n"
        "headroom to 1.0 = solved.\n"
        "Success = EXACT grid match, <=20 actions.")


def cell_dist(a: str, b: str) -> int:
    """Cells differing between two serialised grids (the goal test is exact equality)."""
    fa = [c for row in json.loads(a) for c in row]
    fb = [c for row in json.loads(b) for c in row]
    return sum(1 for x, y in zip(fa, fb) if x != y)


def classify(rec: dict) -> str:
    """Most fundamental first, so every failure lands in exactly one bucket."""
    if rec["failed"] == "terminated":
        return "env terminated"
    if rec["nexec"] == 0:                 # search/model never yielded a usable plan
        return "no plan found"
    if rec["dmin"] >= rec["d0"]:          # acting never beat standing still
        return "no progress"
    return "near miss" if rec["dmin"] <= 2 else "partial progress"


def load_records(input_path: Path, problems_path: Path) -> list[dict]:
    problems = {tuple(p[f] for f in PKEY): p
                for p in json.loads(problems_path.read_text())["problems"]}
    rows = json.loads(input_path.read_text())["rows"]
    recs = []
    for r in rows:
        pr = problems.get(tuple(r[f] for f in PKEY))
        if pr is None:
            continue
        d0 = cell_dist(pr["start_grid"], pr["goal_grid"])
        for arm, _ in ARMS:
            d = r.get(arm) or {}
            ds = [cell_dist(q["grid_after"], pr["goal_grid"])
                  for q in (d.get("rounds") or []) if q.get("grid_after") is not None]
            recs.append({"game": r["game"], "arm": arm, "success": bool(d.get("success")),
                         "failed": d.get("failed_reason"), "d0": d0, "nexec": len(ds),
                         "dmin": min(ds) if ds else None, "key": tuple(r[f] for f in PKEY)})
    return recs


def plot(recs: list[dict], out_dir: Path, stem: str, note: str) -> list[dict]:
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9.5, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.6), facecolor=SURFACE)
    fig.subplots_adjust(left=.055, right=.985, top=.845, bottom=.075, hspace=.42, wspace=.20)

    table = []
    for ax, (game, human) in zip(axes.flat, GAMES):
        ax.set_facecolor(SURFACE)
        n_g = len({r["key"] for r in recs if r["game"] == game and r["arm"] == "raw"})
        for xi, (arm, label) in enumerate(ARMS):
            sel = [r for r in recs if r["game"] == game and r["arm"] == arm]
            counts = Counter(classify(r) for r in sel if not r["success"])
            bottom = 0.0
            for ti, t in enumerate(TYPES):
                share = counts[t] / len(sel)
                if share:
                    # surface-coloured edge IS the 2px gap between stacked fills
                    ax.bar(xi, share, .62, bottom=bottom, color=RAMP[ti],
                           edgecolor=SURFACE, linewidth=1.4, zorder=3)
                    if share >= .085:            # selective labels only — never clipped
                        ax.text(xi, bottom + share / 2, str(counts[t]), ha="center",
                                va="center", color="#ffffff" if ti >= 2 else INK,
                                fontsize=8.5, fontweight="600", zorder=4)
                bottom += share
                table.append({"game": game, "human": human, "arm": label, "type": t,
                              "n_fail": counts[t], "n": len(sel)})
            if bottom:
                ax.text(xi, bottom + .022, f"{bottom:.2f}", ha="center", va="bottom",
                        color=INK, fontsize=9, fontweight="700", zorder=4)

        ax.set_title(f"{game} ({human})  ·  n={n_g}", fontsize=11, fontweight="600",
                     color=INK, pad=9)
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels([lbl for _, lbl in ARMS])
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, .25, .5, .75, 1.0])
        ax.set_yticklabels(["0", ".25", ".50", ".75", "1"])
        ax.grid(axis="y", color=GRID, linewidth=.8, zorder=0)      # solid hairline
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(length=0)
        if ax in (axes[0][0], axes[1][0]):
            ax.set_ylabel("share of problems", fontsize=9.5)

    legend_ax = axes[1][2]
    legend_ax.axis("off")
    # legend order mirrors the stack top-down, so it reads in bar order
    legend_ax.legend(handles=[Patch(facecolor=RAMP[i], label=LEGEND[i]) for i in (3, 2, 1, 0)],
                     loc="upper left", frameon=False, fontsize=9.6, handlelength=1.2,
                     handleheight=1.2, labelspacing=.62, borderaxespad=0,
                     bbox_to_anchor=(-.08, .98))
    legend_ax.text(-.08, .56, DEFS, transform=legend_ax.transAxes, fontsize=8.4,
                   color=INK2, va="top", linespacing=1.5, family="monospace")

    fig.suptitle("Why online planning fails, by game and method", x=.055, y=.955,
                 ha="left", fontsize=16, fontweight="700", color=INK)
    fig.text(.055, .887, note, ha="left", fontsize=9.8, color=INK2)

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=170, facecolor=SURFACE)
    return table


def write_table(table: list[dict], out_dir: Path, stem: str) -> None:
    lines = ["| game | method | problems | failures | " + " | ".join(TYPES) + " |",
             "|---|---|---|---|" + "---|" * len(TYPES)]
    for game, human in GAMES:
        for _, label in ARMS:
            rows = {r["type"]: r for r in table if r["game"] == game and r["arm"] == label}
            tot = sum(rows[t]["n_fail"] for t in TYPES)
            n = rows[TYPES[0]]["n"]
            lines.append(f"| {game} ({human}) | {label} | {n} | {tot} ({tot / n:.0%}) | "
                         + " | ".join(str(rows[t]["n_fail"]) for t in TYPES) + " |")
    (out_dir / f"{stem}.md").write_text(
        "# Online planning failures by type\n\n"
        "Mutually exclusive, assigned most-fundamental first. `dmin` = fewest differing "
        "cells between any executed frame and the goal; `d0` = differing cells at the start.\n\n"
        "- **no plan found** — never produced a usable plan, so never acted\n"
        "- **no progress** — acted, but `dmin >= d0`: never beat standing still\n"
        "- **near miss** — `dmin <= 2` (and better than `d0`)\n"
        "- **partial progress** — got closer than the start, but stayed >2 cells off\n"
        "- *env terminated* — a fifth type the classifier supports; zero in this run\n\n"
        + "\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                    help="FULL online eval json (slim lacks per-round grids)")
    ap.add_argument("--problems", type=Path, default=DEFAULT_PROBLEMS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--stem", default="coverage_v2_online_failures_per_game")
    ap.add_argument("--title-note", default="")
    args = ap.parse_args()

    recs = load_records(args.input, args.problems)
    if not recs:
        raise SystemExit("no records — check that --input and --problems line up on PKEY")
    stray = [r for r in recs if not r["success"] and classify(r) == "env terminated"]
    if stray:
        print(f"note: {len(stray)} env-termination failures are omitted from the chart "
              f"(add a 5th ramp step to include them)")
    table = plot(recs, args.out_dir, args.stem, args.title_note)
    write_table(table, args.out_dir, args.stem)
    fails = [r for r in recs if not r["success"]]
    print(f"{len(recs)} runs, {len(fails)} failures -> {args.out_dir / args.stem}.{{png,svg,md}}")
    for t, c in Counter(classify(r) for r in fails).most_common():
        print(f"  {t:18s} {c}")


if __name__ == "__main__":
    main()
