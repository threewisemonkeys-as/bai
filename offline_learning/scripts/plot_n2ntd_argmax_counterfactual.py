#!/usr/bin/env python3
"""n2ntd selection counterfactual: the shipped val-argmax candidate vs the
iter-28 belief revision the accept gate rejected (it TIED the shipped parent on
the 15-row minibatch, 10.59 = 10.59, and the strict-`>` rule discarded it).

Both arms share candidate-8's perception and the gpt-oss decoder and were
evaluated on the identical 40 test50 windows; the only difference is the belief
text (iter-28 adds bullet-freeze-under-platform + bullet-kills-blue, bundled
with one false rule). Raw-frame baseline = mean of the two files' raw reruns.

    uv run python offline_learning/scripts/plot_n2ntd_argmax_counterfactual.py
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

SHIPPED = LOGS / "multistep_shards_aug4mixed_n2ntd.json"
REJECTED = LOGS / "multistep_shards_aug4_n2ntd_counterfactual.json"

BLUE, GREEN, INK = "#2a78d6", "#008300", "#0b0b0b"

PANELS = [("fd_exact", "Multi-step FD exact"),
          ("fd_partial", "Multi-step FD partial (delta-F1)"),
          ("plan", "Goal-conditioned planning success (in-engine)")]


def load(path: Path) -> dict:
    d = json.loads(path.read_text())
    (res,) = [r for r in d["results"] if r["game"] == "n2ntd"]
    return res


def series(res: dict, panel: str, mode: str) -> list[float]:
    out = []
    for h in HS:
        s = res["summary"][str(h)]
        out.append(s["plan"][mode]["success"] if panel == "plan"
                   else s["fd"][mode]["exact" if panel == "fd_exact" else "partial"])
    return out


def main():
    ship, rej = load(SHIPPED), load(REJECTED)
    kship = [(w["drive"], w["t"], w["h"]) for w in ship["windows"]]
    krej = [(w["drive"], w["t"], w["h"]) for w in rej["windows"]]
    assert kship == krej, "window mismatch"

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2), dpi=110)
    for ax, (panel, title) in zip(axes, PANELS):
        ax.plot(HS, series(ship, panel, "learned"), color=BLUE, marker="o",
                lw=2.0, markersize=6,
                label="SHIPPED: val-argmax cand-8 (test set-ID 0.625)")
        ax.plot(HS, series(rej, panel, "learned"), color=GREEN, marker="o",
                lw=2.6, markersize=6, zorder=5,
                label="REJECTED: iter-28 revision of cand-8 (0.679)")
        raw = [(a + b) / 2 for a, b in zip(series(ship, panel, "raw"),
                                           series(rej, panel, "raw"))]
        ax.plot(HS, raw, color=INK, ls=":", marker="x", lw=1.8, markersize=7,
                label="raw-frame baseline (mean of 2 reruns)")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("horizon h")
        ax.set_xticks(HS)
        ax.set_ylim(-0.03, 1.05)
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("score")
    axes[0].legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    fig.suptitle("n2ntd — the accept gate discarded the better candidate: "
                 "iter-28 beliefs tied the minibatch (10.59 = 10.59) and the "
                 "strict-> rule rejected them; same perception + decoder, "
                 "identical 40 windows", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = LOGS / "multistep_fd_plan_n2ntd_argmax_counterfactual.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
