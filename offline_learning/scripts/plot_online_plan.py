#!/usr/bin/env python3
"""Plots for the online (receding-horizon) planning eval.

    uv run python offline_learning/scripts/plot_online_plan.py

Writes logs/online_plan_curve.png (average success vs horizon: online vs
offline, raw vs learned, random floor; z-blind games excluded) and
logs/online_plan_pergame.png (per-game online-learned lines + bold average).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]


def series(results, horizons, mode, key, keep):
    """Mean over games (with data at that h) of summary[h].plan[mode][key]."""
    out = []
    for h in horizons:
        vals = []
        for res in results:
            if not keep(res["game"]):
                continue
            s = res["summary"].get(str(h))
            v = s["plan"][mode].get(key) if s and mode in s["plan"] else None
            if v is not None:
                vals.append(v)
        out.append(sum(vals) / len(vals) if vals else None)
    return out


def game_series(res, horizons, mode, key):
    out = []
    for h in horizons:
        s = res["summary"].get(str(h))
        out.append(s["plan"][mode].get(key) if s and mode in s["plan"] else None)
    return out


def plot_masked(ax, xs, ys, **kw):
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--online-json", type=Path, default=REPO / "logs/online_plan_eval.json")
    ap.add_argument("--out-curve", type=Path, default=REPO / "logs/online_plan_curve.png")
    ap.add_argument("--out-pergame", type=Path, default=REPO / "logs/online_plan_pergame.png")
    ap.add_argument("--out-online-only", type=Path,
                    default=REPO / "logs/online_plan_curve_onlineonly.png")
    args = ap.parse_args()

    payload = json.loads(args.online_json.read_text())
    results = payload["results"]
    horizons = payload["config"]["horizons"]
    zblind = set(payload["config"]["zblind"])
    keep = lambda g: g not in zblind  # noqa: E731
    n_kept = sum(keep(r["game"]) for r in results)

    # ---- average curve: online vs offline, raw vs learned -------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    for mode, color in (("raw", "tab:blue"), ("learned", "tab:red")):
        plot_masked(ax, horizons, series(results, horizons, mode, "success", keep),
                    color=color, marker="o", lw=2.2, label=f"{mode} online")
        plot_masked(ax, horizons, series(results, horizons, mode, "offline_everhit", keep),
                    color=color, marker="s", lw=1.6, ls="--", alpha=0.65,
                    label=f"{mode} offline")
    plot_masked(ax, horizons, series(results, horizons, "raw", "random_success", keep),
                color="gray", marker=".", lw=1.2, ls=":", label="random plans")
    ax.set_xscale("log", base=2)
    ax.set_xticks(horizons)
    ax.set_xticklabels([str(h) for h in horizons])
    ax.set_xlabel("horizon h (steps)")
    ax.set_ylabel("plan success (executed in engine)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Goal-conditioned planning: online (replan each step) vs offline\n"
                 f"mean over {n_kept} games (z-blind excluded)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_curve, dpi=150)
    print(f"wrote {args.out_curve}")

    # ---- online-only curve: raw vs learned ---------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    def annotate(ys, color, dy):
        for x, y in zip(horizons, ys):
            if y is not None:
                ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                            xytext=(0, dy), ha="center", fontsize=9,
                            color=color, fontweight="bold")

    for mode, color, dy in (("raw", "tab:blue", -16), ("learned", "tab:red", 9)):
        ys = series(results, horizons, mode, "success", keep)
        plot_masked(ax, horizons, ys, color=color, marker="o", lw=2.4, label=f"{mode}")
        annotate(ys, color, dy)
    ys = series(results, horizons, "raw", "random_success", keep)
    plot_masked(ax, horizons, ys, color="gray", marker=".", lw=1.2, ls=":",
                label="random plans")
    annotate(ys, "gray", 7)
    ax.set_xscale("log", base=2)
    ax.set_xticks(horizons)
    ax.set_xticklabels([str(h) for h in horizons])
    ax.set_xlabel("horizon h (steps)")
    ax.set_ylabel("plan success (executed in engine)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Online planning (replan each step, warm start): raw vs learned\n"
                 f"mean over {n_kept} games (z-blind excluded)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_online_only, dpi=150)
    print(f"wrote {args.out_online_only}")

    # ---- per-game online-learned spaghetti ---------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 6))
    cmap = plt.get_cmap("tab20")
    kept = [r for r in results if keep(r["game"])]
    for i, res in enumerate(kept):
        ys = game_series(res, horizons, "learned", "success")
        plot_masked(ax, horizons, ys, color=cmap(i % 20), marker="o", lw=1.1,
                    alpha=0.55, ms=4, label=res["game"])
    plot_masked(ax, horizons, series(results, horizons, "learned", "success", keep),
                color="black", marker="o", lw=3.2, ms=7, label="average", zorder=5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(horizons)
    ax.set_xticklabels([str(h) for h in horizons])
    ax.set_xlabel("horizon h (steps)")
    ax.set_ylabel("plan success (executed in engine)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Online planning success per game — learned (P+B), "
                 f"{len(kept)} games (z-blind excluded)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(args.out_pergame, dpi=150)
    print(f"wrote {args.out_pergame}")


if __name__ == "__main__":
    main()
