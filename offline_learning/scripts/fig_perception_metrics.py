#!/usr/bin/env python3
"""Draw what `perception_metrics.py` measured: P's size and compression over the search.

    uv run python offline_learning/scripts/perception_metrics.py      # writes metrics.csv
    uv run python offline_learning/scripts/fig_perception_metrics.py

Writes `analysis/wm_quant/perception_metrics.pdf` (+ `.png`) and prints the numbers the panels
show, so a caption can be written from the terminal rather than from the picture.

Six cells, and the question each answers:

  A  Does P grow?  `ast_nodes` against search iteration, normalised to the seed module all
     15 runs start from (autumn_seed_perception.py), so 1.0 means the same thing on every
     line. Two series per arm: the INCUMBENT (best-so-far by train_score -- the P the run
     would have shipped had it stopped there, solid) and the POOL MEAN (every node accepted
     so far -- what the reflector is writing rather than what selection keeps, dashed). The
     gap between them is selection pressure.

  B  Does P compress?  `dl_ratio`, bytes of features per byte of observation, same layout.

  C  Does it compress anything that was not already redundant?  `norm_diversity`, compressed
     feature bytes per compressed observation byte. The raw frames are mostly background --
     a game's ~200 KB of observations gzip to ~2 KB -- so B and C do NOT agree: nodes tied
     on byte length come 1.7x apart here, and a ratio above 1 means P's output costs more
     compressed bytes than the observations it summarises. Rank correlation with train_score
     is stronger here than in B on 5 of 6 arm-game cells probed, but its SIGN is not stable
     across games, which is the same conclusion the scatter draws.

  D  How much of the search is dead code?  Every node by status. `collapsed` is the
     |{P(X)}|/|{X}| = 1/|X| case -- a module that renders every observation identically,
     which is what that ratio detects and the only thing it detects (every working node
     scores exactly 1.000).

  E  Does either buy anything?  Each working node's train_score against its compression,
     ship nodes ringed. If the cloud is flat, compression is a free parameter the objective does
     not constrain -- which is the interesting reading, not a null result.

Per-game lines are drawn in one recessive ink rather than 15 hues: the distribution is the
message and no individual game is named, so there is nothing for a 15-entry legend to say.
Arm identity is carried by the two categorical slots, and by direct labels on the medians.
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from math import comb
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SRC = REPO / "analysis/perception_metrics/metrics.csv"
OUT = REPO / "analysis/wm_quant"

# Categorical slots 1 and 7 of the validated reference palette (dataviz skill); the pair
# passes the six checks on the light surface (CVD dE 13.0 deutan, normal 16.3).
ARM_COLOR = {"Plain": "#2a78d6"}
ARM_LABEL = {"Plain": "NLWM (Plain)"}
STATUS = [("ok", "#0ca30c"), ("collapsed", "#fab219"),
          ("runtime", "#ec835a"), ("syntax", "#d03b3b")]
SURFACE, INK, INK2, INK3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8880", "#e6e5e1"


def load(split: str = "train") -> list[dict]:
    if not SRC.is_file():
        raise SystemExit(f"{SRC} missing -- run perception_metrics.py first")
    rows = []
    for r in csv.DictReader(SRC.open()):
        if r["split"] != split:
            continue
        for k in ("idx", "iteration", "depth", "ast_nodes", "is_ship", "on_ship_lineage",
                  "n_unique_outputs", "out_vocab", "diversity_bytes", "raw_gzip_bytes",
                  "k_chars", "k_lines", "k_sentences", "k_claims", "k_words",
                  "k_gzip_bytes", "code_chars", "code_lines", "sloc", "code_gzip_bytes"):
            r[k] = int(r[k]) if r[k] not in ("", "None") else None
        if r["ast_nodes"] == -1:
            r["ast_nodes"] = None       # never parsed: no program size, not a size of -1
        for k in ("train_score", "set_ratio", "dl_ratio", "norm_diversity", "lzma_ratio",
                  "twopart_ratio", "pf_dl_gz_ratio", "info_extraction_ratio", "static_rate",
                  "change_ratio", "decoy_collapse", "err_rate", "k_norm_bytes",
                  "mean_out_chars", "mean_raw_chars"):
            r[k] = float(r[k]) if r[k] not in ("", "None") else None
        rows.append(r)
    return rows


def trajectories(rows: list[dict], field: str, norm_to_seed: bool):
    """(arm, game) -> (iterations, incumbent, pool_mean) over the search.

    A node with no iteration is the seed (it precedes the log); it starts both series.
    `incumbent` steps only when a node beats the running best train_score, so its flat
    stretches are real -- a node that revised only the world knowledge K inherits its
    parent's P and moves nothing here.
    """
    out = {}
    for (arm, game), rs in group(rows).items():
        seed = next((r for r in rs if r["iteration"] is None or r["idx"] == 0), None)
        base = seed[field] if (norm_to_seed and seed and seed[field]) else 1.0
        rs = sorted([r for r in rs if r["iteration"] is not None], key=lambda r: r["iteration"])
        its, inc, mean = [], [], []
        best_score, best_val, seen = -1.0, (seed[field] if seed else None), []
        for r in rs:
            if r[field] is None:
                continue
            seen.append(r[field])
            if r["train_score"] > best_score:
                best_score, best_val = r["train_score"], r[field]
            its.append(r["iteration"])
            inc.append(best_val / base if best_val is not None else None)
            mean.append(st.fmean(seen) / base)
        out[(arm, game)] = (its, inc, mean)
    return out


def group(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["arm"], r["game"])].append(r)
    return g


def median_curve(curves, arm, which: int, grid: list[int]):
    """Median across games at each iteration, stepping each game's curve forward."""
    out = []
    for t in grid:
        vals = []
        for (a, _g), (its, inc, mean) in curves.items():
            if a != arm:
                continue
            series = inc if which == 0 else mean
            v = [s for i, s in zip(its, series) if i <= t and s is not None]
            if v:
                vals.append(v[-1])
        out.append(st.median(vals) if vals else None)
    return out


def panel_curve(ax, rows, field, *, norm, title, ylabel, logy=False):
    curves = trajectories(rows, field, norm)
    grid = list(range(1, max(max(i) for i, _, _ in curves.values()) + 1))
    for (arm, _game), (its, inc, _mean) in curves.items():
        ax.plot(its, inc, color=INK3, lw=0.7, alpha=0.35, zorder=1,
                solid_capstyle="round")
    for arm, c in ARM_COLOR.items():
        if not any(a == arm for a, _ in curves):
            continue
        ax.plot(grid, median_curve(curves, arm, 1, grid), color=c, lw=1.6, ls=(0, (4, 2)),
                zorder=3, alpha=0.85)
        ax.plot(grid, median_curve(curves, arm, 0, grid), color=c, lw=2.4, zorder=4,
                solid_capstyle="round")
    style(ax, title, "search iteration", ylabel)
    if logy:
        ax.set_yscale("log")


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=9.5, loc="left", pad=7)
    ax.set_xlabel(xlabel, color=INK2, fontsize=8)
    ax.set_ylabel(ylabel, color=INK2, fontsize=8)
    ax.tick_params(colors=INK2, labelsize=7.5, length=3, width=0.6)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
        ax.spines[s].set_linewidth(0.8)


def panel_status(ax, rows):
    """The dead tail only. `ok` dominates, so it is stated rather than drawn -- a 90%-full
    bar hides the thing worth seeing, which is how the rest failed."""
    arms = [a for a in ARM_COLOR if any(r["arm"] == a for r in rows)]
    dead = [s for s, _ in STATUS if s != "ok"]
    w = 0.36
    for k, a in enumerate(arms):
        rs = [r for r in rows if r["arm"] == a]
        xs = [i + (k - (len(arms) - 1) / 2) * (w + 0.02) for i in range(len(dead))]
        hs = [sum(1 for r in rs if r["status"] == s) for s in dead]
        ax.bar(xs, hs, width=w, color=ARM_COLOR[a], zorder=3, edgecolor=SURFACE,
               linewidth=2)                                  # 2px surface gap between bars
        for x, h in zip(xs, hs):
            ax.text(x, h + 0.6, str(h), ha="center", va="bottom", color=INK2, fontsize=7.5,
                    zorder=4)
        ok = sum(1 for r in rs if r["status"] == "ok")
        ax.text(0.005, 0.97 - 0.11 * k, f"{ARM_LABEL[a].split(' · ')[0]}  ran: {ok}/{len(rs)}",
                transform=ax.transAxes, ha="left", va="top", color=ARM_COLOR[a], fontsize=7.5)
    ax.set_xticks(range(len(dead)),
                  ["collapsed\n(constant output)", "runtime\n(raises)", "syntax\n(never parsed)"],
                  color=INK2)
    style(ax, "D  Nodes whose P never produced features", "", "pool nodes")
    ax.grid(True, axis="x", lw=0)
    ax.margins(y=0.22)


def panel_scatter(ax, rows):
    for arm, c in ARM_COLOR.items():
        rs = [r for r in rows if r["arm"] == arm and r["status"] == "ok"]
        if not rs:
            continue
        ax.scatter([r["norm_diversity"] for r in rs if not r["is_ship"]],
                   [r["train_score"] for r in rs if not r["is_ship"]],
                   s=11, color=c, alpha=0.42, linewidths=0, zorder=2)
        sh = [r for r in rs if r["is_ship"]]
        ax.scatter([r["norm_diversity"] for r in sh], [r["train_score"] for r in sh],
                   s=46, facecolor=c, edgecolor=SURFACE, linewidths=2, zorder=4)
    style(ax, "E  Compression against training score  (nodes that ran)",
          "norm_diversity  (gz of all P(X) / gz of all X)", "train score")
    ax.set_xscale("log")


def panel_legend(ax):
    """The key, given the spare cell rather than a strip under the figure -- at this width a
    bottom legend sits a long way from the curves it names."""
    ax.set_facecolor(SURFACE)
    ax.axis("off")
    handles = ([Line2D([], [], color=c, lw=2.4, label=ARM_LABEL[a])
                for a, c in ARM_COLOR.items()] if len(ARM_COLOR) > 1 else [])
    arm = next(iter(ARM_COLOR.values()))
    handles += [Line2D([], [], color=arm, lw=2.2, label="incumbent — best node so far"),
                Line2D([], [], color=arm, lw=1.6, ls=(0, (4, 2)),
                       label="pool mean — every proposal so far"),
                Line2D([], [], color=INK3, lw=0.9, alpha=0.6,
                       label="one game's incumbent (A–C)")]
    leg = ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8.5,
                    labelcolor=INK2, handlelength=2.4, borderaxespad=0.6, labelspacing=0.9)
    leg.set_title("", prop={"size": 8})
    ax.text(0.0, 0.30, "A–C: x is the search iteration; a flat stretch is a node that\n"
            "revised only the world knowledge K and inherited its parent's P.\n\n"
            "B vs C: the raw frames are mostly background, so ~200 KB of\n"
            "observations gzip to ~2 KB — byte length and compressed size\n"
            "are not the same measurement, and they rank nodes differently.",
            transform=ax.transAxes, va="top", ha="left", color=INK2, fontsize=7.3,
            linespacing=1.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train", choices=("train", "test"))
    a = ap.parse_args()
    rows = load(a.split)
    OUT.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.6), facecolor=SURFACE)
    panel_curve(axes[0][0], rows, "ast_nodes", norm=True,
                title="A  Size of P over the search", ylabel="AST nodes  (seed = 1)")
    panel_curve(axes[0][1], rows, "dl_ratio", norm=False,
                title="B  Bytes emitted per byte of observation", ylabel="dl_ratio", logy=True)
    panel_curve(axes[0][2], rows, "norm_diversity", norm=False,
                title="C  The same after compression", ylabel="norm_diversity", logy=True)
    axes[0][2].axhline(1.0, color=INK3, lw=0.9, ls=(0, (2, 2)), zorder=2)
    axes[0][2].text(0.99, 1.0, " costs more compressed than the raw frames", transform=
                    axes[0][2].get_yaxis_transform(), ha="right", va="bottom", color=INK3,
                    fontsize=6.5)
    panel_status(axes[1][0], rows)
    panel_scatter(axes[1][1], rows)
    panel_legend(axes[1][2])

    arms = " + ".join(ARM_LABEL[k] for k in ARM_COLOR)
    fig.suptitle(f"The abstraction program P over the learning run  ·  {arms}  ·  "
                 f"15 Autumn games, {a.split} split",
                 color=INK, fontsize=11, x=0.009, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    for ext in ("pdf", "png"):
        p = OUT / f"perception_metrics.{ext}"
        fig.savefig(p, dpi=200, facecolor=SURFACE)
        print(f"wrote {p}")
    report(rows)


def spearman(xs, ys):
    """Rank correlation, ties averaged. Small n and one seed per game, so this is a
    direction indicator, not an inference."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    a, b = ranks(xs), ranks(ys)
    ma, mb = st.fmean(a), st.fmean(b)
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
    den = (sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b)) ** 0.5
    return num / den if den else float("nan")


def dynamics_section(rows) -> list[str]:
    """K -- the world knowledge, the paper's dynamics model -- over the same runs.

    Drawn by `fig_dynamics_size_per_game.py`. The seed K is the empty string, so the first
    proposal that writes anything is a jump from nothing; everything here separates that
    jump from the growth that follows it, because they are not the same event and only the
    first one is clearly worth score.
    """
    out = ["", "## The dynamics model K", "",
           "K is the run's other learned parameter: the English rules the planner reads "
           "alongside P's features. The seed is EMPTY, so `first` below is the first "
           "incumbent that had any dynamics model at all and `growth` is what happened to "
           "it afterwards.", "",
           "| game | first written at | first chars | ship chars | growth | ship sentences "
           "| ship gz(K) B | gz(K)/gz(all X) | distinct sizes | shrinks |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    agg = []
    for (arm, game), rs in sorted(group(rows).items()):
        seq, best = [], -1.0
        for r in sorted((r for r in rs if r["iteration"] is not None),
                        key=lambda r: r["iteration"]):
            if r["train_score"] > best:
                best, _ = r["train_score"], seq.append(r)
        ks = [r["k_chars"] for r in seq]
        nz = next((r for r in seq if r["k_chars"] > 0), None)
        ship = next(r for r in rs if r["is_ship"])
        if nz is None:
            continue
        grow = ship["k_chars"] / nz["k_chars"]
        sizes = len({k for k in ks if k > 0})
        dips = sum(1 for a, b in zip(ks, ks[1:]) if b < a)
        agg.append((nz["iteration"], nz["k_chars"], ship["k_chars"], grow, sizes, dips))
        out.append(f"| {game} | it {nz['iteration']} | {nz['k_chars']} | {ship['k_chars']} | "
                   f"{grow:.2f}x | {ship['k_sentences']} | {ship['k_gzip_bytes']} | "
                   f"{ship['k_norm_bytes']:.3f} | {sizes} | {dips} |")
    m = [st.median(c) for c in zip(*agg)]
    out.append(f"| **median** | it {m[0]:.0f} | {m[1]:.0f} | {m[2]:.0f} | {m[3]:.2f}x | -- | "
               f"-- | -- | {m[4]:.0f} | {sum(a[5] for a in agg)} total |")

    empty = [r for r in rows if r["status"] == "ok" and r["k_chars"] == 0]
    ok = [r for r in rows if r["status"] == "ok"]
    ce, cn = [], []
    for _key, rs in group(rows).items():
        a = [r for r in rs if r["status"] == "ok"]
        b = [r for r in a if r["k_chars"] > 0]
        ce.append(spearman([r["k_chars"] for r in a], [r["train_score"] for r in a]))
        if len(b) >= 4:
            cn.append(spearman([r["k_chars"] for r in b], [r["train_score"] for r in b]))
    med_e = st.median(r["train_score"] for r in empty)
    med_f = st.median(r["train_score"] for r in ok if r["k_chars"] > 0)
    out += ["", "**Does a longer dynamics model score better?** Only in the sense that "
            f"having one does. Over all nodes that ran, `k_chars` correlates with "
            f"`train_score` at median rho {st.median(ce):+.3f}, positive on "
            f"{sum(c > 0 for c in ce)}/{len(ce)} games -- the strongest relation anywhere in "
            "this analysis. But "
            f"{len(empty)}/{len(ok)} of those nodes have an EMPTY K, and they sit at median "
            f"train score {med_e:.3f} against {med_f:.3f} for the ones with a dynamics "
            "model. Drop them and the correlation falls to median rho "
            f"{st.median(cn):+.3f}, positive on {sum(c > 0 for c in cn)}/{len(cn)} games. "
            "Writing down the rules is worth a great deal; writing MORE of them, past that, "
            "is not something these runs grade."]
    return out


def report(rows):
    """The numbers the panels show, printed AND written to REPORT.md, so a caption is
    quoted from the data rather than read off the picture."""
    lines = ["# P over the learning run", "",
             "Generated by `offline_learning/scripts/fig_perception_metrics.py` from",
             "`analysis/perception_metrics/metrics.csv`. Scope and provenance:",
             "`notes/perception-metrics-plan.md`.", "",
             "## Per arm (medians across the 15 games)", "",
             "| arm | games | nodes | ran | dead | ship AST | pool max AST | ship out chars "
             "| ship dl_ratio "
             "| ship gz(P) B | ship norm_diversity | ship lzma_ratio | ship two-part | "
             "ship pf_dl_gz | ship info_extr | ship K chars | ship gz(K) B | "
             "ship train score |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ARM_COLOR:
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        ships = [r for r in rs if r["is_ship"]]
        pool_max = [max((x["ast_nodes"] for x in rs if x["game"] == r["game"]
                         and x["ast_nodes"] is not None), default=0) for r in ships]
        lines.append(
            f"| {ARM_LABEL[arm]} | {len({r['game'] for r in rs})} | {len(rs)} | "
            f"{sum(r['status'] == 'ok' for r in rs)} | {sum(r['status'] != 'ok' for r in rs)} | "
            f"{st.median(r['ast_nodes'] for r in ships):.0f} | {st.median(pool_max):.0f} | "
            f"{st.median(r['mean_out_chars'] for r in ships):.0f} | "
            f"{st.median(r['dl_ratio'] for r in ships):.4f} | "
            f"{st.median(r['diversity_bytes'] for r in ships):.0f} | "
            f"{st.median(r['norm_diversity'] for r in ships):.3f} | "
            f"{st.median(r['lzma_ratio'] for r in ships):.3f} | "
            f"{st.median(r['twopart_ratio'] for r in ships):.3f} | "
            f"{st.median(r['pf_dl_gz_ratio'] for r in ships):.3f} | "
            f"{st.median(r['info_extraction_ratio'] for r in ships):.3f} | "
            f"{st.median(r['k_chars'] for r in ships):.0f} | "
            f"{st.median(r['k_gzip_bytes'] for r in ships):.0f} | "
            f"{st.median(r['train_score'] for r in ships):.3f} |")

    lines += ["", "`dead` = collapsed + runtime + syntax. `set_ratio` = |{P(X)}|/|{X}|; every "
              "node that ran scores exactly 1.000 on every game, which is why it is reported "
              "as a status in panel D rather than drawn as a curve. A `norm_diversity` above 1 "
              "means P's features cost more compressed bytes than the observations they "
              "summarise.", "",
              "`diversity_bytes` is gz of every P(X) concatenated -- output diversity in "
              "bytes -- and `norm_diversity` is that over gz of every X. That denominator "
              "is a constant per game and split, so WITHIN a game the two are the same "
              "curve in different units (Spearman exactly +1.000 on all 15); the "
              "normalisation buys comparability ACROSS games, nothing else. Both gzip each "
              "side as ONE stream, which is not a fair fight: P's outputs are nearly the "
              "same line every frame and dedupe ~7x against the grids' ~3x, so part of a "
              "sub-1.0 `norm_diversity` is P repeating itself. `info_extraction_ratio`, the "
              "mean over frames of gz(P(X))/gz(X), prices each observation on its own, and "
              "`pf_dl_gz` is its uncompressed-numerator twin. Both per-frame scores invert "
              "the verdict: the median ship goes from 0.60 corpus-wide to above 1.0 per "
              "frame.",
              "",
              "## Does compression track the objective?", "",
              "Spearman rank correlation between each compression score and `train_score`, "
              "over the nodes that ran, per arm. A score that mattered would be signed "
              "consistently across games.", "",
              "| arm | game | n | dl | norm_div | lzma | two-part | pf_dl_gz | info_extr |",
              "|---|---|---|---|---|---|---|---|---|"]
    corr_fields = ("dl_ratio", "norm_diversity", "lzma_ratio", "twopart_ratio",
                   "pf_dl_gz_ratio", "info_extraction_ratio")
    by_field = {f: [] for f in corr_fields}
    for arm in ARM_COLOR:
        for g in sorted({r["game"] for r in rows if r["arm"] == arm}):
            rs = [r for r in rows if r["arm"] == arm and r["game"] == g and r["status"] == "ok"]
            if len(rs) < 4:
                continue
            cs = [spearman([r[f] for r in rs], [r["train_score"] for r in rs])
                  for f in corr_fields]
            for f, c in zip(corr_fields, cs):
                by_field[f].append(c)
            lines.append(f"| {arm} | {g} | {len(rs)} | " + " | ".join(f"{c:+.2f}" for c in cs) + " |")

    lines += ["", "Median and sign test over those per-game correlations -- a score that "
              "tracked the objective would be signed the same way on most games:", "",
              "| score | median rho | positive | two-sided sign test |",
              "|---|---|---|---|"]
    for f, cs in by_field.items():
        n, pos = len(cs), sum(c > 0 for c in cs)
        pv = sum(comb(n, k) for k in range(max(pos, n - pos), n + 1)) / 2 ** n * 2
        lines.append(f"| {f} | {st.median(cs):+.3f} | {pos}/{n} | {min(pv, 1.0):.3f} |")
    lines += ["", "`pf_dl_gz` is rank-identical to `mean_out_chars` WITHIN a game (median "
              "Spearman +1.000): its denominator does not depend on the node, so as a "
              "learning curve it is a verbosity curve with a per-game normaliser, and its "
              "value is in making the games comparable. `info_extraction_ratio` is not (median +0.80), "
              "and it is one of only two scores here whose sign holds up across games -- "
              "the other being `twopart_ratio`, which likewise grows with how much P emits. "
              "Both lean POSITIVE: the nodes the objective prefers cost more compressed "
              "bytes per frame, not fewer. Six scores were tested against one outcome with "
              "no correction, so read p=0.035 as a hint and not a result; what it is NOT is "
              "evidence that the search compresses."]
    lines += dynamics_section(rows)
    lines += ["",
              "## Shipped node per game", "",
              "| game | arm | node | iteration | AST | out chars | dl_ratio | gz(P) B | "
              "norm_diversity | pf_dl_gz | info_extr | K chars | K sent | gz(K) B | "
              "static_rate | change_ratio | train score |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted((r for r in rows if r["is_ship"]), key=lambda r: (r["game"], r["arm"])):
        cr = f"{r['change_ratio']:.2f}" if r["change_ratio"] is not None else "--"
        lines.append(
            f"| {r['game']} | {r['arm']} | #{r['idx']} | {r['iteration']} | {r['ast_nodes']} | "
            f"{r['mean_out_chars']:.0f} | {r['dl_ratio']:.4f} | {r['diversity_bytes']} | "
            f"{r['norm_diversity']:.4f} | {r['pf_dl_gz_ratio']:.3f} | "
            f"{r['info_extraction_ratio']:.3f} | {r['k_chars']} | {r['k_sentences']} | "
            f"{r['k_gzip_bytes']} | {r['static_rate']:.3f} | {cr} | "
            f"{r['train_score']:.3f} |")

    lines += ["", "## Selection against size", ""]
    for arm in ARM_COLOR:
        ships = [r for r in rows if r["arm"] == arm and r["is_ship"]]
        if not ships:
            continue
        smaller = sum(r["ast_nodes"] < max((x["ast_nodes"] for x in rows
                                            if x["arm"] == arm and x["game"] == r["game"]
                                            and x["ast_nodes"] is not None), default=0)
                      for r in ships)
        lines.append(f"* **{ARM_LABEL[arm]}** — the shipped P is smaller than its own pool's "
                     f"largest node in {smaller}/{len(ships)} games; median static_rate "
                     f"{st.median(r['static_rate'] for r in ships):.3f}.")

    out = SRC.parent / "REPORT.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[6:13]))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
