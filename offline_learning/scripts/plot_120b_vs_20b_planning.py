#!/usr/bin/env python3
"""Compare learned plan success of the 120b-trained vs 20b-trained hard-min ships
across horizons, on the SHARED planning windows (env-seed 777, 40 win/horizon).

"120b/20b" = the model that scored candidates during rex_pure TRAINING; the
planner LLM in the eval was the SAME for both arms (gpt-oss-120b@cerebras,
effort=low), so the plot isolates ship quality. Only bt3gb/n2ntd/s2kt7 have a
120b-trained ship on these windows (dq8gc/83wkq were 20b-only), so those 3 games.

Sources:
  120b-trained (hard min, aug5): logs/aug7_softmin_planeval/plan_hardmin_w40.json
  20b-trained  (hard min, aug8): logs/aug8_hardmin_planeval/plan_aug8.json
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/home/ays57/bai")
GAMES = ["bt3gb", "n2ntd", "s2kt7"]
H = ["1", "2", "4", "8"]
XPOS = [0, 1, 2, 3]  # even categorical spacing (horizons double: 1,2,4,8)

# Okabe-Ito CVD-safe pair; color follows the entity (training-evaluator size).
C_120 = "#0072B2"   # blue
C_20 = "#D55E00"    # vermillion
INK, MUTED, GRID = "#1a1a1a", "#5a5a5a", "#d9d9d9"


def learned(path):
    res = {r["game"]: r["summary"] for r in json.loads(Path(path).read_text())["results"]}
    return {g: [res[g][h]["plan"]["learned"]["success"] for h in H] for g in GAMES}


d120 = learned(REPO / "logs/aug7_softmin_planeval/plan_hardmin_w40.json")
d20 = learned(REPO / "logs/aug8_hardmin_planeval/plan_aug8.json")

fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
for ax, g in zip(axes, GAMES):
    ax.plot(XPOS, d120[g], "-o", color=C_120, lw=2, ms=7, label="120b-trained", zorder=3)
    ax.plot(XPOS, d20[g], "-o", color=C_20, lw=2, ms=7, label="20b-trained", zorder=3)
    ax.set_title(g, fontsize=12, color=INK, pad=8)
    ax.set_xticks(XPOS)
    ax.set_xticklabels(H)
    ax.set_xlabel("horizon (steps ahead)", fontsize=10, color=MUTED)
    ax.set_ylim(-0.02, 1.0)
    ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)

axes[0].set_ylabel("learned plan success", fontsize=10, color=INK)
axes[0].legend(frameon=False, fontsize=10, loc="upper right")
fig.suptitle("Learned plan success by horizon — 120b- vs 20b-trained hard-min ships\n"
             "(shared windows, env-seed 777; planner = gpt-oss-120b @ cerebras, effort=low)",
             fontsize=11.5, color=INK, y=1.06)
fig.tight_layout()
out = REPO / "logs/aug8_hardmin_planeval/plot_120b_vs_20b_planning.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
