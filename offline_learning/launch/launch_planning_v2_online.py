#!/usr/bin/env python3
"""Game-at-a-time driver for the ONLINE planning-v2 eval.

Runs eval_curated_online.py once per game, in an order that puts the stochastic and
partially-observable games first, so per-game results + viz land incrementally instead
of at the end of a monolithic run. Each game gets its own out dir (own checkpoint, own
online.json/.md, own viz.html); re-running the driver resumes for free — a finished
game's evaluator invocation finds every rollout in its checkpoint and just re-emits.

    uv run python offline_learning/launch/launch_planning_v2_online.py \
        --goal-presentation frame \
        --out-root logs/2026-08-30/planning_v2_online_ds
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# stochastic first (diffusion/f5w3n/dino/SET/colour_lines carry stochastic rows; s2kt7's
# random food spawns predate the flag), then latent-state (paint mode, click mode, ammo,
# sun direction, latched shatter flag), then fully observable deterministic
GAME_ORDER = [
    "f5w3n", "diffusion", "s2kt7", "dino", "SET", "colour_lines",
    "eahcw", "va6fq", "n2ntd", "7xf97", "egg",
    "dq8gc", "bt3gb", "logic_gates", "7www9",
]


def summarize(root: Path, order: list[str], status: dict[str, str],
              arms: list[str]) -> None:
    cols = ["game", "status", "rows", "cap"] + [f"{a} pass" for a in arms] \
        + [f"{a} adj" for a in arms] + ["cost"]
    lines = ["# Planning v2 ONLINE — deepseek planner, game-at-a-time", "",
             "| " + " | ".join(cols) + " |",
             "|" + "|".join("------" for _ in cols) + "|"]
    for g in order:
        f = root / g / "online.json"
        if not f.exists():
            lines.append(f"| {g} | {status.get(g, 'pending')} |"
                         + "|" * (len(cols) - 2) + "|")
            continue
        ev = json.loads(f.read_text())
        rows = ev["rows"]
        cells = {}
        for arm in arms:
            pr, adj = [], []
            for r in rows:
                cell = r.get(arm)
                if not isinstance(cell, dict) or cell.get("pass_rate") is None:
                    continue
                # `random_floor` is the floor measured at the budget the rollouts
                # actually ran under; the cap50 fields only describe a flat-50 run
                fl = r.get("random_floor")
                if fl is None:
                    fl = r.get("random_success_cap50")
                    fl = r.get("random_success") if fl is None else fl
                pr.append(cell["pass_rate"])
                # a row with no measured floor under this presentation contributes to the
                # raw pass column only
                if fl is not None:
                    adj.append(max(0.0, (cell["pass_rate"] - fl) / (1 - fl)) if fl < 1 else 0.0)
            cells[arm] = (f"{sum(pr)/len(pr):.2f} (n={len(pr)})" if pr else "--",
                          f"{sum(adj)/len(adj):.2f}" if adj else "--")
        caps = {r.get("action_cap") for r in rows if r.get("action_cap") is not None}
        cap = (str(caps.pop()) if len(caps) == 1
               else (f"{min(caps)}-{max(caps)}" if caps else "?"))
        lines.append(f"| {g} | {status.get(g, 'done')} | {len(rows)} | {cap} | "
                     + " | ".join([cells[a][0] for a in arms]
                                  + [cells[a][1] for a in arms])
                     + f" | ${ev.get('cost', 0.0):.2f} |")
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="logs/2026-08-30/planning_v2_online_ds")
    ap.add_argument("--goal-presentation", choices=("frame", "nl"), required=True)
    ap.add_argument("--games", default=",".join(GAME_ORDER))
    ap.add_argument("--arms", default="raw,lmwm")
    ap.add_argument("--attempts", type=int, default=1)
    ap.add_argument("--problems", default="logs/2026-08-29/planning_v2/problems.json")
    ap.add_argument("--artifact-root", default="logs/2026-08-24/human_curated")
    ap.add_argument("--model", default="deepseek/deepseek-v4-flash")
    ap.add_argument("--provider-only", default="parasail/fp8,novita/fp8,alibaba/fp8")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--seed-from", default="",
                    help="an earlier out-root whose per-game checkpoints are copied in "
                         "before the first game runs. Use it to ADD an arm to a finished "
                         "run without re-rolling the arms it already has: rollout keys "
                         "are (task, arm, attempt, cap), so the old arms resume for free "
                         "and stay bit-identical to the published columns instead of "
                         "being resampled, and only the new arm is paid for. Requires the "
                         "same --problems, --goal-presentation and --cap-mode.")
    ap.add_argument("--icl-render", choices=("full", "diff"), default="full",
                    help="only used when --arms includes icl")
    ap.add_argument("--icl-context-k", type=int, default=0)
    ap.add_argument("--icl-pool", default="informative_curated")
    ap.add_argument("--cap-mode", default="fixed",
                    help="rollout action budget rule, passed through to the evaluator: "
                    "fixed | per-game | per-problem")
    a = ap.parse_args()

    root = REPO / a.out_root
    root.mkdir(parents=True, exist_ok=True)
    llm_arms = [x for x in (y.strip() for y in a.arms.split(",")) if x and x != "wc"]
    if a.seed_from:
        src_root = REPO / a.seed_from
        ck = f"online.{a.goal_presentation}.ckpt.jsonl"
        seeded = []
        for src in sorted(src_root.glob(f"*/{ck}")):
            dst = root / src.parent.name / ck
            if dst.exists():
                continue                      # never clobber rollouts this root already has
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            seeded.append(src.parent.name)
        print(f"seeded checkpoints from {src_root} for {len(seeded)} game(s): "
              f"{', '.join(seeded) or 'none'}", flush=True)
    games = [g for g in a.games.split(",") if g]
    unknown = [g for g in games if g not in GAME_ORDER]
    if unknown:
        sys.exit(f"unknown games: {unknown}")
    status: dict[str, str] = {g: "done (prior run)" for g in GAME_ORDER
              if (root / g / "online.json").exists() and g not in games}
    failures = []
    for g in games:
        t0 = time.time()
        status[g] = "RUNNING"
        summarize(root, GAME_ORDER, status, llm_arms)
        out = root / g / "online"
        cmd = [sys.executable, str(REPO / "offline_learning/scripts/eval_curated_online.py"),
               "--problems", a.problems, "--artifact-root", a.artifact_root,
               "--games", g, "--goal-presentation", a.goal_presentation,
               "--arms", a.arms, "--attempts", str(a.attempts),
               "--out", str(out), "--concurrency", str(a.concurrency),
               "--llm-backend", "openrouter", "--model", a.model,
               "--cap-mode", a.cap_mode,
               "--provider-only", a.provider_only]
        if "icl" in llm_arms:
            cmd += ["--icl-render", a.icl_render, "--icl-pool", a.icl_pool,
                    "--icl-context-k", str(a.icl_context_k)]
        print(f"\n=== {g}: {' '.join(cmd)}", flush=True)
        log = (root / g / "driver.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as lf:
            rc = subprocess.run(cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            print(f"=== {g} FAILED (rc={rc}), see {log}", flush=True)
            status[g] = f"FAILED rc={rc}"
            failures.append(g)
            summarize(root, GAME_ORDER, status, llm_arms)
            continue
        vz = subprocess.run(
            [sys.executable, str(REPO / "offline_learning/scripts/viz_v2_online.py"),
             "--eval", str(out) + ".json", "--problems", a.problems,
             "--out", str(root / g / "viz.html")], cwd=REPO)
        # the replay page: the same rollouts stepped one action at a time, with the
        # prompt and response at each round
        subprocess.run(
            [sys.executable, str(REPO / "offline_learning/scripts/viz_plan_replay.py"),
             "--eval", str(out) + ".json", "--problems", a.problems,
             "--out", str(root / g / "replay.html")], cwd=REPO)
        status[g] = f"done {int((time.time()-t0)/60)}min" + ("" if vz.returncode == 0
                                                             else " (viz FAILED)")
        summarize(root, GAME_ORDER, status, llm_arms)
        print(f"=== {g} complete in {(time.time()-t0)/60:.0f} min -> "
              f"{root / g}/online.md + viz.html", flush=True)
    summarize(root, GAME_ORDER, status, llm_arms)
    print(f"\nall done; failures: {failures or 'none'} -> {root}/SUMMARY.md", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
