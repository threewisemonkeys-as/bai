#!/usr/bin/env python3
"""Updated 6-game macro of the multistep FD + planning eval.

Refreshes the jul30 "new objective vs prior rosters" macro curve
(multistep_fd_plan_jul30mc_vs_all.png) with the August best runs: the macro now
covers 6 games (bt3gb, dq8gc, e3v6m, n2ntd, qfsvc, s2kt7) and compares, over
the SAME game set:

  * pre-August roster — jul30_minclick arms for the 5 original games + the cd3
    sweep arm for s2kt7 (its pre-August state, incl. the fake-FD collapse)
  * current best roster — aug5_rexpure for bt3gb / n2ntd / s2kt7, existing best
    data for dq8gc (jul30_minclick), e3v6m and qfsvc (jul28_unified)

Baselines (raw-frame, random-plan floor) come from each game's current-roster
file; within a game every file shares identical windows (seed-1 protocol).

    uv run python offline_learning/scripts/plot_multistep_6game_update.py
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
JUL28U = LOGS / "multistep_fd_plan_eval_jul28u.json"
JUL30MC = LOGS / "multistep_fd_plan_eval_jul30mc.json"
AUG5 = LOGS / "multistep_shards_aug5_rexpure.json"

GAMES = ["bt3gb", "dq8gc", "e3v6m", "n2ntd", "qfsvc", "s2kt7"]
PRE = {"bt3gb": JUL30MC, "dq8gc": JUL30MC, "e3v6m": JUL30MC,
       "n2ntd": JUL30MC, "qfsvc": JUL30MC, "s2kt7": CD3}
CUR = {"bt3gb": AUG5, "dq8gc": JUL30MC, "e3v6m": JUL28U,
       "n2ntd": AUG5, "qfsvc": JUL28U, "s2kt7": AUG5}

GREEN, INK, GRAY = "#008300", "#0b0b0b", "#8a8a86"


def summaries(roster: dict[str, Path]) -> dict[str, dict]:
    cache: dict[Path, dict] = {}
    out = {}
    for g, path in roster.items():
        if path not in cache:
            cache[path] = json.loads(path.read_text())
        (res,) = [r for r in cache[path]["results"] if r["game"] == g]
        out[g] = res["summary"]
    return out


def macro(sums: dict[str, dict], panel: str, mode: str) -> list[float]:
    out = []
    for h in HS:
        vals = []
        for s in sums.values():
            v = (s[str(h)]["plan"][mode]["success"] if panel == "plan"
                 else s[str(h)]["fd"][mode]["exact"])
            vals.append(v)
        assert all(v is not None for v in vals)
        out.append(sum(vals) / len(vals))
    return out


def main():
    pre, cur = summaries(PRE), summaries(CUR)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=110)
    for ax, panel, title in [(axes[0], "fd", "Multi-step FD exact"),
                             (axes[1], "plan", "Goal-conditioned planning success (in-engine)")]:
        ax.plot(HS, macro(pre, panel, "learned"), color=GRAY, ls="--",
                marker="o", lw=1.8, markersize=6,
                label="pre-August roster (jul30 minclick ×5 + cd3 s2kt7)")
        ax.plot(HS, macro(cur, panel, "learned"), color=GREEN, marker="o",
                lw=2.6, markersize=6, zorder=5,
                label="current best roster (aug5 ×3 + prior best ×3)")
        ax.plot(HS, macro(cur, panel, "raw"), color=INK, ls=":", marker="x",
                lw=1.8, markersize=7, label="raw-frame baseline (same windows)")
        if panel == "plan":
            rand = [sum(cur[g][str(h)]["plan"]["raw"]["random_success"]
                        for g in GAMES) / len(GAMES) for h in HS]
            ax.plot(HS, rand, color=GRAY, ls=":", lw=1.6, marker=".",
                    label="random-plan floor")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("horizon h")
        ax.set_xticks(HS)
        ax.set_ylim(-0.03, 1.05)
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("macro mean over 6 games")
    axes[0].legend(fontsize=9, loc="upper right", framealpha=0.9)
    fig.suptitle("Macro over bt3gb, dq8gc, e3v6m, n2ntd, qfsvc, s2kt7 — "
                 "test50 drives, gpt-oss decode; identical windows per game "
                 "across arms", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = LOGS / "multistep_fd_plan_6game_update.png"
    fig.savefig(out)
    print(f"wrote {out}")
    for panel in ("fd", "plan"):
        print(panel, "pre:", [round(v, 3) for v in macro(pre, panel, "learned")],
              "cur:", [round(v, 3) for v in macro(cur, panel, "learned")],
              "raw:", [round(v, 3) for v in macro(cur, panel, "raw")])


if __name__ == "__main__":
    main()
