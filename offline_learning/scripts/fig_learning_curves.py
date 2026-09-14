#!/usr/bin/env python3
"""Learning curves of the runs behind the planning evaluation: the objective over the search.

Three figures, one panel per game in each, over the same runs:
  objective  -- train_score, what the search maximises: the mean over the 60 training
      transitions of min(ID_set, cFD_hard)
  id         -- the inverse-dynamics term alone (set-credited `id_score`)
  fd         -- the forward-dynamics term alone (contrastive FD, P-rendered options)

Each dot is a node the search evaluated, at its position in the pool (node 0 is the blank
seed every run starts from; the budget is 30 nodes). The line is the best node so far BY THE
OBJECTIVE -- what the run would have shipped had it stopped there -- and in the id/fd figures
it plots that same node's score on the term, so it can fall as well as rise. Following the
running best of the term instead would trace nodes the search never selected and leave the
shipped model off its own line. The ringed marker is the node that shipped, i.e. the artifact
NLWM (Plain) and NLWM (Agentic) plan with.

The per-term train scores come from `resume_batches.jsonl`, which holds every node's 60
per-transition records; the script checks they rebuild each node's train_score exactly. A node
whose P emits the same text for every frame is zeroed by the constant-output gate: its
objective is 0 while its raw term scores are not. The term figures zero it too, so a dot
means the same thing in all three (this is the seed in every run, plus one node in SET).

Each panel also carries two numbers about the shipped model:
  held-out  -- the figure's quantity on the 50-transition test split, drawn as a diamond at
      the shipped node's x so its gap to the filled marker is the generalisation gap. rex_pure
      aliases val to train, so the test split is the only held-out set these runs have. ID is
      the in-run test trace (`test_trace_rexpure_seed1.json`); FD is the held-out cFD trace in
      the rendering training uses (`test_trace_cfd_perceived_rexpure_seed1.json`, hard decoys,
      written by `eval_heldout_cfd.py`); the objective is their per-transition min, averaged.
      Both traces are checked to hold the shipped P and K and the same transitions in order.
  planning  -- NLWM (Agentic) pass@1 for the game, read through `report_planning_v2_online`'s
      own loaders so it is the number in the paper's per-environment table. Printed, not
      plotted: it is a different quantity and has no place on the score axis.

The x-axis is the node index, not rex_search's iteration counter `i`: iterations include
reflection calls that failed (mostly OpenRouter 429s) and added no node, so `i` stretches the
curve by an amount that is infrastructure noise. The perception appendix figures
(`wm_panel_grid.py`) use `i`; the node index is the budget unit.

All panels share one [0, 1] y-axis because each score means the same thing in every game.

The runs are resolved from the eval's own manifest rather than hard-coded, and every shipped
file is checked against the sha256 the eval recorded, so the figure cannot drift onto a run
the planner did not use.

    uv run python offline_learning/scripts/fig_learning_curves.py            # all three
    uv run python offline_learning/scripts/fig_learning_curves.py --term id

Writes `analysis/wm_quant/learning_curves{,_id,_fd}_per_game.{pdf,png}` and prints the numbers
behind each panel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fig_perception_metrics import INK, INK2, SURFACE, style  # noqa: E402
from report_planning_v2_online import (  # noqa: E402
    AGENT_ARM, DEFAULT_AGENT_WM, DEFAULT_RUNS, display_name, load_agent_run, load_run, scored,
)

MANIFEST = REPO / "logs/2026-09-08/agent_wm_full/nlwm_manifest.json"
OUT = REPO / "analysis/wm_quant"
COLOR = "#2a78d6"                       # categorical slot 1 of the reference palette
HELDOUT = "#eb6834"                     # slot 2: the adjacent pair the palette validates
TERMS = {
    "objective": dict(stem="learning_curves_per_game", ylab="objective",
                      line="best proposal so far", held="its objective on held-out data"),
    "id": dict(stem="learning_curves_id_per_game", ylab="inverse dynamics (ID)",
               line="best proposal so far (by the objective)",
               held="its ID score on held-out data"),
    "fd": dict(stem="learning_curves_fd_per_game", ylab="forward dynamics (FD)",
               line="best proposal so far (by the objective)",
               held="its FD score on held-out data"),
}


def load_runs(manifest: Path) -> dict[str, dict]:
    """game -> {pool, ship, train, heldout} for the learning run behind each shipped artifact.

    `train[term]` and `heldout[term]` are keyed by the TERMS names; `train` is per node."""
    m = json.loads(manifest.read_text())
    runs = {}
    for game, files in m["games"].items():
        for f in files.values():
            got = hashlib.sha256(Path(f["source"]).read_bytes()).hexdigest()
            if got != f["sha256"]:
                raise SystemExit(f"{game}: {f['source']} changed since the eval ran")
        d = Path(files["perception.py"]["source"]).parent
        pool = [json.loads(l) for l in (d / "rexpure_run_seed1/candidates.jsonl").open()]
        if [c["idx"] for c in pool] != list(range(len(pool))):
            raise SystemExit(f"{game}: candidates.jsonl is not in node order")
        best_p = Path(files["perception.py"]["source"]).read_text().strip()
        best_k = Path(files["beliefs.txt"]["source"]).read_text().strip()
        ship = [c["idx"] for c in pool if c["perception"].strip() == best_p
                and c["world_knowledge"].strip() == best_k]
        if len(ship) != 1:
            raise SystemExit(f"{game}: shipped (P, K) matches {len(ship)} pool nodes, want 1")
        top = max(c["train_score"] for c in pool)
        if pool[ship[0]]["train_score"] != top:
            raise SystemExit(f"{game}: shipped node {ship[0]} is not argmax train_score")
        runs[game] = {"dir": d, "pool": pool, "ship": ship[0],
                      "train": train_terms(d, pool),
                      "heldout": heldout_terms(d, best_p, best_k)}
    return runs


def train_terms(d: Path, pool: list[dict]) -> dict[str, list[float]]:
    """Per-node train scores for every term, rebuilt from the per-transition batch records."""
    ids, fds = [None] * len(pool), [None] * len(pool)
    for line in (d / "rexpure_run_seed1/resume_batches.jsonl").open():
        b = json.loads(line)
        i, trs, scores = int(b["cand_idx"]), b["trajectories"], b["scores"]
        mins = [min(t["id_score"], t["cfd_score"]) for t in trs]
        # the constant-output gate zeroes every transition of a node whose P says one thing
        gated = all(s == 0 for s in scores) and any(mins)
        if not gated and any(abs(m - s) > 1e-9 for m, s in zip(mins, scores)):
            raise SystemExit(f"{d.name}/{i}: per-transition scores are not min(ID, cFD)")
        if abs(sum(scores) / len(scores) - pool[i]["train_score"]) > 1e-9:
            raise SystemExit(f"{d.name}/{i}: batch does not rebuild train_score")
        ids[i] = 0.0 if gated else sum(t["id_score"] for t in trs) / len(trs)
        fds[i] = 0.0 if gated else sum(t["cfd_score"] for t in trs) / len(trs)
    if None in ids:
        raise SystemExit(f"{d.name}: nodes with no batch record: "
                         f"{[i for i, v in enumerate(ids) if v is None]}")
    return {"objective": [c["train_score"] for c in pool], "id": ids, "fd": fds}


def heldout_terms(d: Path, best_p: str, best_k: str) -> dict[str, float]:
    """The shipped model's test-split score on each term, and on their per-transition min."""
    id_tr = json.loads((d / "test_trace_rexpure_seed1.json").read_text())
    cfd_tr = json.loads((d / "test_trace_cfd_perceived_rexpure_seed1.json").read_text())
    for name, tr in (("ID", id_tr), ("cFD", cfd_tr)):
        if tr["perception"].strip() != best_p or tr["beliefs"].strip() != best_k:
            raise SystemExit(f"{d.name}: test {name} trace is not the shipped model")
    ids = {int(r["idx"]): r for r in id_tr["records"]}
    cfds = {int(r["idx"]): r for r in cfd_tr["records"]}
    if set(ids) != set(cfds) or any(ids[i]["truth"] != cfds[i]["action"] for i in ids):
        raise SystemExit(f"{d.name}: test ID and cFD traces cover different transitions")
    summary = json.loads((d / "heldout_cfd_seed1.json").read_text())
    if abs(summary["perceived"] - cfd_tr["cfd_score"]) > 1e-9:
        raise SystemExit(f"{d.name}: cFD trace disagrees with heldout_cfd_seed1.json")
    id_mean = sum(float(r["id_score"]) for r in ids.values()) / len(ids)
    if abs(id_mean - id_tr["acc"]) > 1e-9:
        raise SystemExit(f"{d.name}: test ID records do not rebuild the trace's acc")
    return {"objective": sum(min(float(ids[i]["id_score"]), float(cfds[i]["score"]))
                             for i in ids) / len(ids),
            "id": id_mean, "fd": cfd_tr["cfd_score"]}


def planning_scores() -> dict[str, float]:
    """game -> NLWM (Agentic) pass@1, exactly as the paper's per-environment table reads it."""
    label, path = DEFAULT_RUNS[0]
    reference = load_run(label, REPO / path)
    agent = load_agent_run(DEFAULT_AGENT_WM[0], REPO / DEFAULT_AGENT_WM[1], reference)
    if agent["only_mine"] or agent["only_ref"] or agent["cap_diff"]:
        raise SystemExit("NLWM (Agentic) problem set or budgets differ from the reference")
    out = {}
    for game in agent["games"]:
        g = scored(agent, game)
        if g is None:
            raise SystemExit(f"NLWM (Agentic) has not finished {game}")
        out[game] = g["arms"][AGENT_ARM]["pass1"]
    return out


def incumbents(objective: list[float]) -> list[int]:
    """Index of the best node so far by the objective, at every node (first of any tie)."""
    out, best = [], 0
    for i, s in enumerate(objective):
        if s > objective[best]:
            best = i
        out.append(best)
    return out


def draw(term: str, runs: dict, planning: dict, games: list[str]):
    spec = TERMS[term]
    n_nodes = max(len(r["pool"]) for r in runs.values())
    ncol = 5
    nrow = -(-len(games) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(11.6, 2.3 * nrow + 0.4),
                             facecolor=SURFACE, sharex=True, sharey=True)
    flat = [ax for row in axes for ax in row]

    print(f"\n{spec['ylab']}")
    print(f"{'game':16s} {'ship':>4s} {'train':>6s} {'heldout':>7s} {'plan':>5s} {'@95%':>5s}  run")
    for ax, game in zip(flat, games):
        run, ship = runs[game], runs[game]["ship"]
        ys = run["train"][term]
        xs = list(range(len(ys)))
        inc = [ys[j] for j in incumbents(run["train"]["objective"])]
        ax.scatter(xs, ys, s=11, color=COLOR, alpha=0.3, linewidths=0, zorder=2)
        ax.step(xs, inc, where="post", color=COLOR, lw=1.9, zorder=3)
        ax.scatter([ship], [ys[ship]], s=44, facecolor=COLOR, edgecolor=SURFACE,
                   linewidths=1.1, zorder=5)
        held = run["heldout"][term]
        # hollow, and larger than the ship dot: where held-out ~ train the two coincide and
        # a filled diamond would erase the marker it is being compared against
        ax.scatter([ship], [held], s=78, marker="D", facecolor="none", edgecolor=HELDOUT,
                   linewidths=1.7, zorder=6)
        style(ax, "", "", "")
        ax.set_title(display_name(game), color=INK, fontsize=10, loc="left", pad=15)
        ax.text(0.0, 1.035, f"held-out {held:.2f}  ·  planning {planning[game]:.2f}",
                transform=ax.transAxes, ha="left", va="bottom", color=INK2, fontsize=7.8)
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_xlim(-1, n_nodes)
        ax.set_xticks([0, 10, 20, 30])
        ax.tick_params(labelsize=7.5)

        at95 = next(i for i, v in zip(xs, inc) if v >= 0.95 * inc[-1]) if inc[-1] > 0 else 0
        print(f"{display_name(game):16s} {ship:4d} {ys[ship]:6.3f} {held:7.3f} "
              f"{planning[game]:5.2f} {at95:5d}  {run['dir'].relative_to(REPO)}")

    for ax in flat[len(games):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("proposal", color=INK2, fontsize=8)
    for row in axes:
        row[0].set_ylabel(spec["ylab"], color=INK2, fontsize=8)

    handles = [Line2D([], [], color=COLOR, lw=2.2, label=spec["line"]),
               Line2D([], [], color=COLOR, lw=0, marker="o", markersize=4.5, alpha=0.3,
                      markerfacecolor=COLOR, markeredgewidth=0, label="every proposal"),
               Line2D([], [], color=COLOR, lw=0, marker="o", markersize=7,
                      markerfacecolor=COLOR, markeredgecolor=SURFACE, markeredgewidth=1.0,
                      label="world model used for planning"),
               Line2D([], [], color=HELDOUT, lw=0, marker="D", markersize=7.5,
                      markerfacecolor="none", markeredgecolor=HELDOUT, markeredgewidth=1.7,
                      label=spec["held"])]
    fig.legend(handles=handles, loc="upper left", ncol=4, frameon=False, fontsize=9,
               labelcolor=INK2, bbox_to_anchor=(0.005, 1.0), handlelength=2.3,
               columnspacing=2.2)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = OUT / f"{spec['stem']}.{ext}"
        fig.savefig(p, dpi=200, facecolor=SURFACE)
        print(f"wrote {p}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--term", choices=("all", *TERMS), default="all")
    a = ap.parse_args()

    runs = load_runs(a.manifest)
    planning = planning_scores()
    if set(planning) != set(runs):
        raise SystemExit(f"planning games {sorted(planning)} != learning runs {sorted(runs)}")
    games = sorted(runs, key=display_name)
    for term in (TERMS if a.term == "all" else [a.term]):
        draw(term, runs, planning, games)


if __name__ == "__main__":
    main()
