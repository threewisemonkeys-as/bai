#!/usr/bin/env python3
"""Cross-game average of the multistep FD + planning eval, best run per game.

For each of the 21 autumn clean_data3 games, takes the best-performing run so
far (highest single-step test50 set-ID among runs that have a gpt-oss-decoded
multistep eval; games with a single evaluated arm use that arm) and averages
the learned curves over games. Baselines (raw-frame prediction, random-plan
floor) come from the SAME eval file as each game's chosen arm, so every
per-game comparison shares identical windows and decode model.

Outputs logs/multistep_fd_plan_bestpergame_avg.png and a provenance .md.

    uv run python offline_learning/scripts/plot_multistep_bestpergame.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
LOGS = REPO / "logs"
HS = [1, 2, 4, 8]

CD3 = LOGS / "multistep_fd_plan_eval_gptoss.json"
CD3_SH1 = LOGS / "multistep_fd_plan_eval_gptoss_shard1.json"
JUL28U = LOGS / "multistep_fd_plan_eval_jul28u.json"
JUL30MC = LOGS / "multistep_fd_plan_eval_jul30mc.json"
AUG5 = LOGS / "multistep_shards_aug5_rexpure.json"

# game -> (eval json with its best arm, test50 set-ID or None if only one arm)
BEST = {
    "27vwc": (CD3, None), "83wkq": (CD3, None), "ada85": (CD3, None),
    "aw9wd": (CD3, None), "dgg2c": (CD3, None), "eahcw": (CD3_SH1, None),
    "ice": (CD3, None), "nrdf6": (CD3, None), "ntq4y": (CD3, None),
    "qqm74": (CD3, None), "va6fq": (CD3, None),
    "7www9": (JUL28U, 0.707), "7xf97": (JUL28U, 0.557),
    "e3v6m": (JUL28U, 0.821), "f5w3n": (JUL28U, 0.875),
    "qfsvc": (JUL28U, 0.826), "vqjh6": (JUL28U, 0.365),
    "dq8gc": (JUL30MC, 0.880),
    "bt3gb": (AUG5, 0.823), "n2ntd": (AUG5, 0.682), "s2kt7": (AUG5, 0.632),
}

PANELS = [("fd_exact", "Multi-step FD exact"),
          ("fd_partial", "Multi-step FD partial (delta-F1)"),
          ("plan", "Goal-conditioned planning success (in-engine)")]

GREEN, INK, GRAY = "#008300", "#0b0b0b", "#8a8a86"


def value(summary: dict, h: int, panel: str, mode: str):
    s = summary[str(h)]
    if panel == "plan":
        return s["plan"][mode]["success"]
    return s["fd"][mode]["exact" if panel == "fd_exact" else "partial"]


def main():
    cache: dict[Path, dict] = {}
    per_game: dict[str, dict] = {}
    for game, (path, score) in sorted(BEST.items()):
        if path not in cache:
            cache[path] = json.loads(path.read_text())
        (res,) = [r for r in cache[path]["results"] if r["game"] == game]
        per_game[game] = {
            "summary": res["summary"],
            "artifact": res["artifact_dir"].replace(str(LOGS) + "/", ""),
            "file": path.name, "score": score,
            "rand": [res["summary"][str(h)]["plan"]["raw"]["random_success"]
                     for h in HS],
        }

    # per-horizon columns: some games have no valid windows at h=1
    # (aw9wd/dgg2c: every candidate window static or noop-solvable)
    def collect(panel: str, mode: str) -> tuple[list[float], list[list[float]], list[int]]:
        by_h = [[v for g, d in per_game.items()
                 if (v := value(d["summary"], h, panel, mode)) is not None]
                for h in HS]
        mean = [sum(v) / len(v) for v in by_h]
        return mean, by_h, [len(v) for v in by_h]

    def iqr(by_h: list[list[float]]) -> tuple[list[float], list[float]]:
        lo, hi = [], []
        for vals in by_h:
            v = sorted(vals)
            lo.append(v[int(0.25 * (len(v) - 1))])
            hi.append(v[int(round(0.75 * (len(v) - 1)))])
        return lo, hi

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2), dpi=110)
    for ax, (panel, title) in zip(axes, PANELS):
        lmean, lcols, ln = collect(panel, "learned")
        rmean, rcols, _ = collect(panel, "raw")
        llo, lhi = iqr(lcols)
        rlo, rhi = iqr(rcols)
        ax.fill_between(HS, llo, lhi, color=GREEN, alpha=0.12, lw=0)
        ax.fill_between(HS, rlo, rhi, color=GRAY, alpha=0.15, lw=0)
        nlab = f"n={min(ln)}" if min(ln) == max(ln) else f"n={min(ln)}-{max(ln)}"
        ax.plot(HS, lmean, color=GREEN, marker="o", lw=2.6, markersize=6,
                zorder=5, label=f"best learned per game ({nlab} games)")
        ax.plot(HS, rmean, color=INK, ls=":", marker="x", lw=1.8, markersize=7,
                label="raw-frame baseline (same windows)")
        if panel == "plan":
            rvals = [[d["rand"][i] for d in per_game.values()
                      if d["rand"][i] is not None] for i in range(len(HS))]
            ax.plot(HS, [sum(v) / len(v) for v in rvals], color=GRAY, ls=":",
                    lw=1.6, marker=".", label="random-plan floor")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("horizon h")
        ax.set_xticks(HS)
        ax.set_ylim(-0.03, 1.05)
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("mean score over games")
    axes[2].legend(fontsize=9, loc="upper right", framealpha=0.9)
    fig.suptitle(f"Best run per game, averaged over {len(per_game)} games — "
                 "multistep FD + planning on test50 drives (gpt-oss decode; "
                 "bands = interquartile range across games)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = LOGS / "multistep_fd_plan_bestpergame_avg.png"
    fig.savefig(out)
    print(f"wrote {out}")

    lines = ["# Best-run-per-game multistep average — roster\n",
             "| game | arm (artifact) | test50 set-ID | eval file |",
             "|---|---|---:|---|"]
    for g, d in sorted(per_game.items()):
        sc = f"{d['score']:.3f}" if d["score"] is not None else "only arm"
        lines.append(f"| {g} | {d['artifact']} | {sc} | {d['file']} |")
    md = LOGS / "multistep_fd_plan_bestpergame_avg.md"
    md.write_text("\n".join(lines) + "\n")
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
