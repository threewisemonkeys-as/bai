#!/usr/bin/env python3
"""Is a planning problem hard in proportion to the length of its reference plan?

The 86 curated planning problems each ship a reference plan -- the action sequence a
solver used to build the problem -- so `h = len(plan)` is a difficulty label that costs
nothing to compute and is independent of any method's score. This script joins that
label to the per-problem outcome of every arm the paper's results table reports, and
asks whether performance falls off with it.

Everything is read from the same run dirs `report_planning_v2_online.py` reads, so a
number here is a number in an `online.json` (or, for the agent, in its `rows.jsonl`):

    Raw / Plain  logs/2026-09-03/planning_v2_online_ds_percap_nl   arms `raw`, `lmwm`
    ICL          logs/2026-09-05/planning_v2_online_icl_full       arm  `icl`
    NLWM (SL)    logs/2026-09-02/planning_v2_online_opus5_nl       arm  `lmwm`
    Agent        logs/2026-09-06/agent_full                        rows.jsonl
    Agentic      logs/2026-09-08/agent_wm_full                     rows.jsonl

The arm names are the paper's: `NLWM (Plain)` plans with the learned world model,
`NLWM (Agentic)` is the `Agent` harness whose workspace also holds that world model. So
`Agent` vs `NLWM (Agentic)` is what the world model is worth to an agent, and
`NLWM (Plain)` vs `NLWM (Agentic)` is what the agent loop is worth given the same model.

All six score the identical 86 problems under identical per-problem action budgets --
asserted on load, because a difficulty curve computed over different problem sets or
different budgets would compare nothing.

    uv run python offline_learning/scripts/analyze_planning_difficulty.py

Writes `analysis/planning_difficulty/`: the per-problem join (`per_problem.csv`), the
report (`REPORT.md`), and the figure (`difficulty.png` / `.pdf`).

Three difficulty labels are compared, because "length of the reference plan" has three
readings and they disagree on seven problems:
  * `h`            -- the reference plan's length. The headline axis.
  * `nl_reached`   -- the step at which the NL goal FIRST holds along that plan. These
                      runs score any-step, so this is the length actually demanded; a
                      dino "survive two cactus passes" plan is 30 long but the goal is
                      met at 10.
  * `n_decisions`  -- distinct decisions in the plan (a run of `right` counts once).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import optimize, stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# label -> (run dir, arm). The order is the order everything prints in.
SOURCES = [
    ("Raw",             "logs/2026-09-03/planning_v2_online_ds_percap_nl", "raw"),
    ("ICL",             "logs/2026-09-05/planning_v2_online_icl_full",     "icl"),
    ("NLWM (Plain)",   "logs/2026-09-03/planning_v2_online_ds_percap_nl", "lmwm"),
    ("NLWM (SL)",       "logs/2026-09-02/planning_v2_online_opus5_nl",     "lmwm"),
    ("Agent",           "logs/2026-09-06/agent_full",                      "agent"),
    ("NLWM (Agentic)", "logs/2026-09-08/agent_wm_full",                   "agent"),
]
PROBLEMS = "logs/2026-09-03/planning_v2_online_ds_percap_nl/problems.per-problem-floors.json"
# the figure plots the methods the question is about over the Raw baseline; NLWM (SL)
# is Plain under a stronger reflector, so it stays in the tables
# rather than adding a sixth line that duplicates a hue's worth of story
PLOTTED = ["Raw", "ICL", "NLWM (Plain)", "Agent", "NLWM (Agentic)"]
# log-spaced, each bin roughly doubling the last, sized to keep n >= 11 everywhere
BINS = [(1, 2), (3, 5), (6, 11), (12, 23), (24, 40)]

# --- palette: reference categorical slots 1-3 plus slot 7 (violet), and a neutral for
# the baseline, which is a reference series rather than a category. The theme's fourth
# slot is yellow; it is passed over here because a 2px yellow line on the light surface
# is too weak for a print figure, and because yellow beside orange is the one pair the
# palette flags. Identity never rests on hue alone: every series also carries its own
# marker, a direct label at the line end, and a legend entry. Text wears ink, not the
# series colour -- the end marker is what sits beside the label.
COLOR = {"NLWM (Plain)": "#2a78d6", "ICL": "#eb6834", "Agent": "#1baf7a",
         "NLWM (Agentic)": "#4a3aa7", "Raw": "#6f6d67"}
MARKER = {"NLWM (Plain)": "o", "ICL": "s", "Agent": "^", "NLWM (Agentic)": "v",
          "Raw": "D"}
SURFACE, INK, INK2, INK3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8880", "#e6e5e1"


# ------------------------------------------------------------------ loading
def load_evaluator_run(root: Path) -> dict[str, dict]:
    """`task_uid -> row` over a run's per-game `online.json` files."""
    rows = {}
    for f in sorted(root.glob("*/online.json")):
        ev = json.loads(f.read_text())
        cap = ev.get("config", {}).get("max_actions")
        for r in ev["rows"]:
            r["_cap"] = r.get("action_cap") or cap
            rows[r["task_uid"]] = r
    return rows


def load_agent_run(root: Path) -> dict[str, dict]:
    """The agent writes one `rows.jsonl` for the whole run (or one per worker)."""
    rows = {}
    for f in [root / "rows.jsonl", *sorted(root.glob("rows.w*.jsonl"))]:
        if not f.is_file():
            continue
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                r["_cap"] = r.get("action_cap")
                rows.setdefault(r["task_uid"], r)
    return rows


def build_table(repo: Path) -> list[dict]:
    """One record per problem: its difficulty labels and every arm's pass@1."""
    problems = json.loads((repo / PROBLEMS).read_text())["problems"]
    runs: dict[str, dict[str, dict]] = {}
    for label, path, _arm in SOURCES:
        if path not in runs:
            root = repo / path
            runs[path] = (load_agent_run(root) if (root / "rows.jsonl").exists()
                          else load_evaluator_run(root))

    # every arm must have scored the same problems under the same budgets, or the
    # curves are not on one axis
    ref = {p["task_uid"]: None for p in problems}
    base = runs[SOURCES[0][1]]
    for label, path, _ in SOURCES:
        got = runs[path]
        missing = [u for u in ref if u not in got]
        bad_cap = [u for u in ref if u in got and got[u]["_cap"] != base[u]["_cap"]]
        if missing or bad_cap:
            raise SystemExit(f"{label} ({path}): {len(missing)} problems missing, "
                             f"{len(bad_cap)} budget mismatches -- not comparable")

    recs = []
    for p in problems:
        uid = p["task_uid"]
        rec = {
            "task_uid": uid, "game": p["game"], "id": p["id"], "tier": p["tier"],
            "h": p["h"], "n_decisions": p["n_decisions"],
            "nl_reached": p.get("nl_anystep_reached_at"),
            "cap": base[uid]["_cap"], "floor": base[uid].get("random_floor"),
            "stochastic": p.get("stochastic"), "n_mechanics": len(p.get("mechanics") or []),
        }
        for label, path, arm in SOURCES:
            cell = runs[path][uid].get(arm)
            if not isinstance(cell, dict) or cell.get("pass_rate") is None:
                raise SystemExit(f"{label} has no score for {uid}")
            rec[label] = cell["pass_rate"]
        recs.append(rec)
    return recs


# ------------------------------------------------------------------ statistics
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Score interval -- honest at n ~ 11 and at p = 0 or 1, where Wald is not."""
    if n == 0:
        return 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def logistic_fit(y: np.ndarray, h: np.ndarray) -> tuple[float, float]:
    """P(solve) = sigma(a + b * log2 h). `b` is the logit cost of doubling the plan."""
    x = np.log2(h)

    def nll(th):
        z = th[0] + th[1] * x
        return float(np.sum(np.logaddexp(0, z) - y * z))

    return tuple(optimize.minimize(nll, [0.0, 0.0], method="BFGS").x)


def crossing(a: float, b: float, p: float = 0.5) -> float:
    """The plan length at which the fit passes `p` -- the arm's solvable horizon."""
    return float(2 ** ((math.log(p / (1 - p)) - a) / b)) if b else float("nan")


def cluster_bootstrap(y, h, games, n=4000, seed=0):
    """Resample GAMES, not problems: problems within a game share a world model."""
    rng = np.random.default_rng(seed)
    uniq = sorted(set(games))
    idx_of = {g: np.where(games == g)[0] for g in uniq}
    slopes, cross = [], []
    for _ in range(n):
        pick = np.concatenate([idx_of[g] for g in rng.choice(uniq, len(uniq))])
        yy = y[pick]
        if yy.min() == yy.max():          # a degenerate resample has no fit
            continue
        a, b = logistic_fit(yy, h[pick])
        if b < 0:                         # a positive slope has no 50% crossing to report
            slopes.append(b)
            cross.append(crossing(a, b))
        else:
            slopes.append(b)
    return np.array(slopes), np.array(cross)


def demeaned_corr(y, x, games):
    """Within-game correlation: does length still bite once the game is held fixed?"""
    def demean(v):
        out = np.array(v, dtype=float)
        for g in set(games):
            m = games == g
            out[m] -= out[m].mean()
        return out
    r = stats.pearsonr(demean(np.log2(x)), demean(y))
    return r.statistic, r.pvalue


# ------------------------------------------------------------------ report
def report(recs: list[dict], out: Path) -> list[str]:
    labels = [lbl for lbl, _, _ in SOURCES]
    games = np.array([r["game"] for r in recs])
    h = np.array([r["h"] for r in recs], float)
    axes = {"reference plan length h": h,
            "NL any-step length": np.array([r["nl_reached"] for r in recs], float),
            "distinct decisions": np.array([r["n_decisions"] for r in recs], float)}
    Y = {lbl: np.array([r[lbl] for r in recs], float) for lbl in labels}
    floor = np.array([r["floor"] for r in recs], float)

    L = [f"# Difficulty vs reference plan length ({len(recs)} curated planning problems)",
         "",
         "Every arm scores the same 86 problems under the same per-problem action caps.",
         "`h` is the length of the problem's reference plan; the caps were set from it",
         f"(Spearman(h, cap) = {stats.spearmanr(h, np.array([r['cap'] for r in recs])).statistic:.2f}),",
         "so a longer plan is given proportionally more budget and the fall-off below is",
         "*not* a budget artefact.", ""]

    L += ["## Does plan length predict failure?", "",
          "Spearman rho between the difficulty label and per-problem success (n = 86).", "",
          "| Difficulty label | " + " | ".join(labels) + " |",
          "|---|" + "---|" * len(labels)]
    for name, x in axes.items():
        cells = []
        for lbl in labels:
            r = stats.spearmanr(x, Y[lbl])
            star = "***" if r.pvalue < 1e-3 else "**" if r.pvalue < 1e-2 else "*" if r.pvalue < .05 else ""
            cells.append(f"{r.statistic:+.2f}{star}")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L += ["", "Within game (both variables game-demeaned, Pearson on log2 length):", "",
          "| | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels),
          "| log2 h, game-demeaned | " + " | ".join(
              f"{demeaned_corr(Y[l], h, games)[0]:+.2f}" for l in labels) + " |", ""]

    L += ["## pass@1 by reference-plan-length bin", "",
          "95% Wilson intervals in brackets; `rand` is the cap-matched any-step random floor.", "",
          "| plan length | n | rand | " + " | ".join(labels) + " |",
          "|---|---|---|" + "---|" * len(labels)]
    for lo, hi in BINS:
        m = (h >= lo) & (h <= hi)
        n = int(m.sum())
        cells = []
        for lbl in labels:
            v = Y[lbl][m]
            a, b = wilson(int(v.sum()), n)
            cells.append(f"{v.mean():.2f} [{a:.2f},{b:.2f}]")
        L.append(f"| {lo}–{hi} | {n} | {floor[m].mean():.2f} | " + " | ".join(cells) + " |")
    L += [f"| **all** | {len(recs)} | {floor.mean():.2f} | "
          + " | ".join(f"**{Y[l].mean():.2f}**" for l in labels) + " |", ""]

    # The top h-bin is the one place the axis lies: three dino "survive two cactus
    # passes" problems carry a 30-action reference plan whose goal already holds at
    # step 10. Re-binning on the NL any-step length moves them down and the top bin
    # falls the way the trend says it should.
    nl_len = np.array([r["nl_reached"] for r in recs], float)
    L += ["Same bins on the NL any-step length (the length the any-step scorer actually", "demands):", "",
          "| NL length | n | " + " | ".join(labels) + " |", "|---|---|" + "---|" * len(labels)]
    for lo, hi in BINS:
        m = (nl_len >= lo) & (nl_len <= hi)
        L.append(f"| {lo}–{hi} | {int(m.sum())} | "
                 + " | ".join(f"{Y[l][m].mean():.2f}" for l in labels) + " |")
    L.append("")

    # The two comparisons the Agentic arm exists to make. Both arms saw the same
    # problem, so the test is paired: only the problems where they disagree carry
    # information, and an aggregate gap of a few points over 86 problems need not
    # survive one (exact McNemar, two-sided).
    L += ["## Paired comparisons, split at the median plan length", "",
          "`gained`/`lost` count the problems the second arm flips to solved / unsolved.", "",
          "| comparison | subset | n | gained | lost | p |", "|---|---|---|---|---|---|"]
    for lo_lbl, hi_lbl in [("Agent", "NLWM (Agentic)"),
                           ("NLWM (Plain)", "NLWM (Agentic)"),
                           ("NLWM (Plain)", "Agent")]:
        a, b = Y[lo_lbl], Y[hi_lbl]
        for name, m in [("all", np.ones(len(h), bool)), ("h <= 11", h <= 11),
                        ("h >= 12", h >= 12)]:
            g = int(((a[m] == 0) & (b[m] == 1)).sum())
            ls = int(((a[m] == 1) & (b[m] == 0)).sum())
            p = stats.binomtest(g, g + ls, 0.5).pvalue if g + ls else 1.0
            L.append(f"| {lo_lbl} -> {hi_lbl} | {name} | {int(m.sum())} | {g} | {ls} "
                     f"| {p:.3f} |")
    L.append("")

    L += ["## Fitted fall-off", "",
          "P(solve) = sigma(a + b·log2 h). `b` is the logit cost of doubling the plan;",
          "`h@50%` is where the fit crosses one-in-two -- the arm's solvable horizon.",
          "Intervals are 4000 bootstrap resamples **of games**, not of problems.", "",
          "| Method | slope per doubling | h@50% | h@25% |", "|---|---|---|---|"]
    fits = {}
    for lbl in labels:
        a, b = logistic_fit(Y[lbl], h)
        fits[lbl] = (a, b)
        sl, cr = cluster_bootstrap(Y[lbl], h, games)
        slo, shi = np.percentile(sl, [2.5, 97.5])
        clo, chi = (np.percentile(cr, [2.5, 97.5]) if len(cr) > 100 else (np.nan, np.nan))
        L.append(f"| {lbl} | {b:+.2f} [{slo:+.2f}, {shi:+.2f}] | "
                 f"{crossing(a, b):.1f} [{clo:.1f}, {chi:.1f}] | {crossing(a, b, 0.25):.1f} |")
    L += ["", "## Where the long problems are", "",
          "| task | h | cap | tier | " + " | ".join(labels) + " |",
          "|---|---|---|---|" + "---|" * len(labels)]
    for r in sorted([r for r in recs if r["h"] >= 24], key=lambda r: -r["h"]):
        L.append(f"| {r['task_uid']} | {r['h']} | {r['cap']} | {r['tier']} | "
                 + " | ".join("Y" if r[l] else "." for l in labels) + " |")
    L.append("")
    (out / "REPORT.md").write_text("\n".join(L) + "\n")
    return L, fits


# ------------------------------------------------------------------ figure
# The two panels are drawn by separate functions so each can stand alone in the paper
# (`--fig-dir`, one PDF each, no titles -- the LaTeX caption says what the panel says)
# while the combined PNG kept beside the report stays titled for reading in a terminal.
def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_ylim(-0.03, 1.06)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=9)


def panel_binned(ax, recs: list[dict], title: bool) -> None:
    """Measured pass@1 per plan-length bin, with Wilson intervals.

    Chance is a filled region rather than a sixth line, so it can never be mistaken for
    a series. No end labels: three arms land within 0.06 of each other in the last bin,
    so a label displaced far enough to be legible would point at the wrong line -- the
    legend plus five distinct markers carry identity here."""
    h = np.array([r["h"] for r in recs], float)
    floor = np.array([r["floor"] for r in recs], float)
    Y = {lbl: np.array([r[lbl] for r in recs], float) for lbl in PLOTTED}

    xs = np.arange(len(BINS))
    ns = [int(((h >= lo) & (h <= hi)).sum()) for lo, hi in BINS]
    fl_bin = [floor[(h >= lo) & (h <= hi)].mean() for lo, hi in BINS]
    ax.fill_between(xs, 0, fl_bin, color="#f0efec", linewidth=0, zorder=1)
    ax.annotate("random floor", (xs[0], fl_bin[0]), xytext=(7, 5),
                textcoords="offset points", color=INK3, fontsize=8.5, zorder=2)
    for lbl in PLOTTED:
        pts, lo_e, hi_e = [], [], []
        for lo, hi in BINS:
            m = (h >= lo) & (h <= hi)
            p = Y[lbl][m].mean()
            a, b = wilson(int(Y[lbl][m].sum()), int(m.sum()))
            pts.append(p); lo_e.append(p - a); hi_e.append(b - p)
        ax.errorbar(xs, pts, yerr=[lo_e, hi_e], fmt="none", ecolor=COLOR[lbl],
                    elinewidth=1, capsize=2.5, alpha=0.45, zorder=3)
        ax.plot(xs, pts, color=COLOR[lbl], linewidth=2, marker=MARKER[lbl],
                markersize=7, markeredgecolor=SURFACE, markeredgewidth=2,
                solid_capstyle="round", zorder=4, label=lbl)
    ax.set_xticks(xs, [f"{lo}\u2013{hi}\n$n$={n}" for (lo, hi), n in zip(BINS, ns)])
    ax.set_xlim(-0.25, len(BINS) - 0.75)
    ax.set_xlabel("reference plan length", color=INK2, fontsize=10, labelpad=6)
    ax.set_ylabel("pass@1", color=INK2, fontsize=10)
    if title:
        ax.set_title("Measured: success falls with plan length", color=INK,
                     fontsize=11, loc="left", pad=10)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=9, handlelength=1.6,
                    labelspacing=0.35, borderpad=0.2)
    for t in leg.get_texts():
        t.set_color(INK2)


def panel_fitted(ax, fits: dict, title: bool) -> None:
    """The fitted fall-off, and where each arm crosses one-in-two.

    The crossings sit close together on a log axis and the arm names are long, so each
    marker is annotated with its horizon alone -- the number is what the legend cannot
    say -- and the legend beside it carries identity. What the bare numbers mean is left
    to the caption; every in-plot home for that sentence collides with a curve."""
    place = {"Raw": (0, 11, "center"), "ICL": (0, -19, "center"),
             "NLWM (Plain)": (0, 11, "center"), "Agent": (0, -19, "center"),
             "NLWM (Agentic)": (0, 11, "center")}
    grid = np.logspace(0, math.log2(40), 200, base=2)
    for lbl in PLOTTED:
        a, b = fits[lbl]
        ax.plot(grid, 1 / (1 + np.exp(-(a + b * np.log2(grid)))), color=COLOR[lbl],
                linewidth=2, solid_capstyle="round", zorder=3, label=lbl)
        xc = crossing(a, b)
        if 1 <= xc <= 40:
            ax.plot([xc], [0.5], marker=MARKER[lbl], markersize=8, color=COLOR[lbl],
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
            dx, dy, ha = place.get(lbl, (0, 12, "center"))
            ax.annotate(f"{xc:.0f}", (xc, 0.5), xytext=(dx, dy),
                        textcoords="offset points", ha=ha, color=INK, fontsize=9,
                        zorder=6)
    ax.axhline(0.5, color=GRID, linewidth=1, zorder=1)
    ax.set_xscale("log", base=2)
    ax.set_xlim(1, 40)
    ax.set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
    ax.set_xlabel("reference plan length (log scale)", color=INK2, fontsize=10,
                  labelpad=6)
    ax.set_ylabel("fitted P(solve)", color=INK2, fontsize=10)
    if title:
        ax.set_title("Fitted: the horizon moves, the slope does not", color=INK,
                     fontsize=11, loc="left", pad=10)
    leg = ax.legend(loc="lower left", frameon=False, fontsize=9, handlelength=1.6,
                    labelspacing=0.35, borderpad=0.2)
    for t in leg.get_texts():
        t.set_color(INK2)


def figures(recs: list[dict], fits: dict, out: Path, fig_dir: Path | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(fig, *paths):
        fig.tight_layout()
        for p in paths:
            fig.savefig(p, dpi=200, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)

    # combined and titled -- the one to look at while working
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.4, 4.4), facecolor=SURFACE,
                                   gridspec_kw={"width_ratios": [1.05, 1]})
    style_axes(axL); style_axes(axR)
    panel_binned(axL, recs, title=True)
    panel_fitted(axR, fits, title=True)
    save(fig, out / "difficulty.png", out / "difficulty.pdf")

    # standalone and untitled -- the pair the paper includes. The PDF goes straight to
    # the paper's figures dir; the PNG stays here so the panel can be eyeballed.
    for stem, draw in [("planning_difficulty_binned",
                        lambda ax: panel_binned(ax, recs, title=False)),
                       ("planning_difficulty_fitted",
                        lambda ax: panel_fitted(ax, fits, title=False))]:
        fig, ax = plt.subplots(figsize=(5.6, 4.0), facecolor=SURFACE)
        style_axes(ax)
        draw(ax)
        save(fig, out / f"{stem}.png", (fig_dir or out) / f"{stem}.pdf")


# ------------------------------------------------------------------ LaTeX
def tex_binned(recs: list[dict]) -> str:
    """The binned pass@1 table, as the body between this table's AUTO markers.

    Written the way `report_planning_v2_online.py` writes the results tables, so the
    appendix number has the same provenance as the main one: regenerated from the run
    dirs, never hand-copied."""
    labels = [lbl for lbl, _, _ in SOURCES]
    h = np.array([r["h"] for r in recs], float)
    floor = np.array([r["floor"] for r in recs], float)
    Y = {lbl: np.array([r[lbl] for r in recs], float) for lbl in labels}
    L = [r"\toprule",
         "Plan length & $n$ & Rand & " + " & ".join(labels) + r" \\",
         r"\midrule"]
    for lo, hi in BINS:
        m = (h >= lo) & (h <= hi)
        L.append(f"{lo}--{hi} & {int(m.sum())} & {floor[m].mean():.2f} & "
                 + " & ".join(f"{Y[l][m].mean():.2f}" for l in labels) + r" \\")
    L += [r"\midrule",
          rf"\textbf{{All}} & {len(recs)} & {floor.mean():.2f} & "
          + " & ".join(rf"\textbf{{{Y[l].mean():.2f}}}" for l in labels) + r" \\",
          r"\bottomrule"]
    return "\n".join("    " + line for line in L)


def write_tex(path: Path, recs: list[dict]) -> bool:
    """Replace the `% BEGIN AUTO tab:planning-difficulty` .. `% END AUTO` block in place."""
    label = "tab:planning-difficulty"
    begin, end = f"% BEGIN AUTO {label}", f"% END AUTO {label}"
    lines = path.read_text().split("\n")
    bi = next((i for i, l in enumerate(lines) if l.rstrip().endswith(begin)), None)
    if bi is None:
        raise SystemExit(f"{path} has no {begin}")
    ei = next((i for i, l in enumerate(lines[bi:], bi) if l.rstrip().endswith(end)), None)
    if ei is None:
        raise SystemExit(f"{begin} in {path} has no matching {end}")
    block = [f"    {begin}", *tex_binned(recs).split("\n"), f"    {end}"]
    changed = block != lines[bi:ei + 1]
    lines[bi:ei + 1] = block
    path.write_text("\n".join(lines))
    return changed


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="analysis/planning_difficulty",
                    help="directory for per_problem.csv, REPORT.md and the figures")
    ap.add_argument("--fig-dir", default="paper/figures",
                    help="where the two untitled panel PDFs the paper includes go")
    ap.add_argument("--write-tex", nargs="?", const="paper/main.tex", metavar="PATH",
                    help="also regenerate the tab:planning-difficulty AUTO block")
    a = ap.parse_args()

    def resolve(p):
        return Path(p) if Path(p).is_absolute() else REPO / p

    out, fig_dir = resolve(a.out), resolve(a.fig_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    recs = build_table(REPO)
    with (out / "per_problem.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)

    lines, fits = report(recs, out)
    figures(recs, fits, out, fig_dir)
    print("\n".join(lines))
    print(f"\nwrote {out}/per_problem.csv, REPORT.md, difficulty.png/pdf, "
          f"planning_difficulty_{{binned,fitted}}.png")
    print(f"wrote {fig_dir}/planning_difficulty_{{binned,fitted}}.pdf")
    if a.write_tex:
        tex = resolve(a.write_tex)
        print(f"{'updated' if write_tex(tex, recs) else 'unchanged'}: "
              f"{tex} tab:planning-difficulty")


if __name__ == "__main__":
    main()
