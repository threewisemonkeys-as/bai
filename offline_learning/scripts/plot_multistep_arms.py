#!/usr/bin/env python3
"""Multi-arm comparison plots for the multistep FD + planning eval.

Reads several eval_multistep_fd_plan.py output JSONs (one per learned arm) that
were computed on IDENTICAL windows (same seed/drives — verified at load time)
and renders, per game, a 3-panel figure:

  1. Multi-step FD exact (learned P features)
  2. Multi-step FD partial (textdiff delta-F1, learned P)
  3. Goal-conditioned planning success (executed in-engine)

The raw-frame baseline is averaged over the arm files that used the same task
model (same windows, independent decode reruns); the random-plan floor is taken
from the reference arm (deterministic given the seed).

    uv run python offline_learning/scripts/plot_multistep_arms.py --game bt3gb
    uv run python offline_learning/scripts/plot_multistep_arms.py --game n2ntd
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
LOGS = REPO / "logs"

# Categorical slots from the validated reference palette (dataviz skill):
BLUE, RED, GREEN, VIOLET = "#2a78d6", "#e34948", "#008300", "#4a3aa7"
GRAY, INK = "#8a8a86", "#0b0b0b"

# label, path, style kwargs, include-in-raw-mean (same decode model as ref)
ARMS = {
    "bt3gb": [
        ("old prompts, idcfd (test 0.579)",
         LOGS / "multistep_shards_jul30idcfd/bt3gb.json",
         dict(color=GRAY, ls="--", marker="o", lw=1.8), True),
        ("new prompts + gpt-oss refl (0.667)",
         LOGS / "multistep_shards_aug3newprompts.json",
         dict(color=BLUE, ls="-", marker="o", lw=1.8), True),
        ("mixed: + deepseek refl (0.780)",
         LOGS / "multistep_shards_aug4mixed.json",
         dict(color=RED, ls="-", marker="o", lw=1.8), True),
        ("aug5 rex-pure ship (0.823)",
         LOGS / "multistep_shards_aug5_rexpure.json",
         dict(color=GREEN, ls="-", marker="o", lw=2.6, zorder=5), True),
    ],
    "n2ntd": [
        ("rex, old prompts (test 0.438)",
         LOGS / "multistep_shards_jul30rex_n2ntd.json",
         dict(color=GRAY, ls="--", marker="o", lw=1.8), True),
        ("new prompts + gpt-oss refl (0.618)",
         LOGS / "multistep_shards_aug4_n2ntd.json",
         dict(color=BLUE, ls="-", marker="o", lw=1.8), True),
        ("mixed, gpt-oss decode (0.625)",
         LOGS / "multistep_shards_aug4mixed_n2ntd.json",
         dict(color=RED, ls="-", marker="o", lw=1.8), True),
        ("mixed, deepseek decode",
         LOGS / "multistep_shards_aug4mixed_n2ntd_dsdecode.json",
         dict(color=VIOLET, ls="-.", marker="o", lw=1.8), False),
        ("aug5 rex-pure ship (0.682)",
         LOGS / "multistep_shards_aug5_rexpure.json",
         dict(color=GREEN, ls="-", marker="o", lw=2.6, zorder=5), True),
    ],
}

OUT = {"bt3gb": LOGS / "multistep_fd_plan_bt3gb_4arm.png",
       "n2ntd": LOGS / "multistep_fd_plan_n2ntd_5arm.png"}


def load_game(path: Path, game: str) -> dict:
    d = json.loads(path.read_text())
    (res,) = [r for r in d["results"] if r["game"] == game]
    res["_windows_key"] = [(w["drive"], w["t"], w["h"]) for w in res["windows"]]
    return res


def series(res: dict, hs: list[int], panel: str, mode: str) -> list[float]:
    out = []
    for h in hs:
        s = res["summary"][str(h)]
        out.append(s["fd"][mode]["exact"] if panel == "fd_exact"
                   else s["fd"][mode]["partial"] if panel == "fd_partial"
                   else s["plan"][mode]["success"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True, choices=sorted(ARMS))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    game, arms = args.game, ARMS[args.game]

    loaded = [(lab, load_game(p, game), st, in_raw) for lab, p, st, in_raw in arms]
    ref = loaded[-1][1]
    for lab, res, _, _ in loaded:
        assert res["_windows_key"] == ref["_windows_key"], f"window mismatch: {lab}"
    hs = sorted(int(h) for h in ref["summary"])

    raw_pool = [res for _, res, _, in_raw in loaded if in_raw]
    raw = {p: [sum(series(r, hs, p, "raw")[i] for r in raw_pool) / len(raw_pool)
               for i in range(len(hs))] for p in ("fd_exact", "fd_partial", "plan")}
    rand = [ref["summary"][str(h)]["plan"]["raw"]["random_success"] for h in hs]

    panels = [("fd_exact", "Multi-step FD exact, learned P"),
              ("fd_partial", "Multi-step FD partial (delta-F1), learned P"),
              ("plan", "Goal-conditioned planning success (in-engine)")]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2), dpi=110)
    for ax, (panel, title) in zip(axes, panels):
        for lab, res, style, _ in loaded:
            ax.plot(hs, series(res, hs, panel, "learned"), label=lab,
                    markersize=6, **style)
        ax.plot(hs, raw[panel], color=INK, ls=":", marker="x", markersize=7,
                lw=1.8, label=f"raw-frame baseline (mean of {len(raw_pool)} reruns)")
        if panel == "plan":
            ax.plot(hs, rand, color=GRAY, ls=":", lw=1.6, marker=".",
                    label="random-plan floor")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("horizon h")
        ax.set_xticks(hs)
        ax.set_ylim(-0.03, 1.05)
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("score")
    axes[0].legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    n_win = len(ref["_windows_key"])
    fig.suptitle(f"{game} test50 drives — identical {n_win} windows per arm; "
                 f"legend numbers = single-step test50 set-ID", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = args.out or OUT[game]
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
