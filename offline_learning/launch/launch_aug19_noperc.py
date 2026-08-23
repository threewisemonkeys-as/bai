"""NO-PERCEPTION ablation of the canonfix_v2 nlwm (rexpure) learning run.

One variable changes against `logs/2026-08-11/human_unified/rexpure/<game>_s1` (the run
whose console logs are archived at logs/2026-08-14/canonfix_v2/learning/): the learner no
longer learns a perception module. P is frozen to the identity (raw grid passthrough) and
the REx search mutates ONLY the world-knowledge block, so every prompt that used to carry
P's features carries the RAW GRID instead. Data, split, seed, models, provider pins,
objective (min(ID-set, contrastive-FD) with hard decoys), node budget, propose-batch,
rex-C and the test protocol are all byte-identical to the reference.

The reference command is reconstructed here rather than read from the reference run's
launch.json, because `logs/batch3_consolidated/` (the ancestor launch_human_origin.py
rebuilt from) no longer exists. It is a verbatim copy of the surviving
logs/2026-08-11/human_unified/rexpure/{bt3gb,n2ntd}_s1/launch.json flag list, which the
console-log headers of the three re-learned games (83wkq/dq8gc/s2kt7) match on every
reported setting; only `--actions` differs per game (s2kt7/83wkq are click-only).

    uv run python offline_learning/launch/launch_aug19_noperc.py
    uv run python offline_learning/launch/launch_aug19_noperc.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "offline_learning/human_data"
OUT_ROOT = ROOT / "logs/2026-08-19/noperc_ablation/rexpure"
SCRIPT = ROOT / "offline_learning/rexpure_optimize.py"
VARIANT = "informative_unified"
GAMES = ["bt3gb", "dq8gc", "n2ntd", "s2kt7", "83wkq"]

# Verbatim from the reference launch.json, minus the six data/out flags (supplied per
# game below) and minus --start-perception (replaced by --no-perception). --concurrency
# appears twice in the reference exactly as recorded; argparse keeps the last (4).
REF_FLAGS = [
    "--test-n", "50",
    "--k-choices", "5",
    "--context-k", "9",
    "--concurrency", "12",
    "--seed", "1",
    "--max-nodes", "30",
    "--fd-scorer", "none",
    "--task-model", "openai/gpt-oss-20b",
    "--task-provider-order", "groq",
    "--task-reasoning-json", '{"effort": "low"}',
    "--reflection-model", "deepseek/deepseek-v4-flash",
    "--reflection-provider-order", "deepseek,baidu,fireworks",
    "--reflection-hedge-delay", "120",
    "--reflection-timeout", "300",
    "--analyze-mistakes",
    "--no-perception",
    "--id-set-loss",
    "--id-eps", "0.1",
    "--composite", "min",
    "--contrastive-fd",
    "--cfd-hard-decoys",
    "--rex-c", "5",
    "--propose-batch", "3",
    "--no-analysis-memo",
    "--concurrency", "4",
    "--train-n", "60",
]

# The ONE deviation from the reference command, and it is I/O-only: with P frozen to the
# identity, every F prompt carries 19 raw grids (~44k chars) instead of a compact feature
# transcript, so calls are ~10x larger. Under the reference's 30 s hedge every call
# out-ran the delay and raced a duplicate, which doubled request volume into groq's shared
# pool for gpt-oss-20b and killed 4 of 5 games with unhandled 429s during the seed eval.
# These knobs change only how long we wait and how often we re-ask -- same model, same
# provider pin, same prompts, so nothing the search sees changes.
IO_FLAGS = ["--hedge-delay", "120", "--llm-timeout", "180"]
IO_ENV = {"LLM_RETRIES": "10"}


def build(game: str, outd: Path) -> list[str]:
    paths = json.loads((DATA / game / VARIANT / "dataset_paths.json").read_text())
    return [sys.executable, "-u", str(SCRIPT),
            "--run", paths["run"],
            "--context-source-run", paths["context_source_run"],
            "--test-run", paths["test_run"],
            "--test-context-source-run", paths["test_context_source_run"],
            "--actions", paths["actions"],
            "--out-dir", str(outd)] + REF_FLAGS + IO_FLAGS


def _spawn(game: str, cmd: list[str], outd: Path, env: dict, attempt: int = 1):
    """Launch one game, appending to its stdout so a retry keeps the earlier console.

    A checkpointed run always resumes: rex_search re-derives the train fingerprint and
    falls back to a fresh start by itself if the split does not match, so passing
    --resume can only save work, never silently continue the wrong search."""
    if (outd / "rexpure_run_seed1/resume_state.json").exists():
        cmd = cmd + ["--resume"]
    with (outd / "stdout.txt").open("a") as f:
        f.write(f"\n===== attempt {attempt} =====\n")
        f.flush()
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                             stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    (outd / "launch.json").write_text(json.dumps(
        {"game": game, "learner": "rexpure", "ablation": "no-perception",
         "variant": VARIANT, "pid": p.pid, "attempt": attempt, "cmd": cmd,
         "reference_run": f"logs/2026-08-11/human_unified/rexpure/{game}_s1"}, indent=2) + "\n")
    print(f"start noperc/{game} (attempt {attempt}): pid={p.pid} -> {outd}", flush=True)
    return p


def _drain(live: list, pending: dict, env: dict, max_attempts: int) -> None:
    """Wait for every launched game, relaunching one that exits without a test summary."""
    while live:
        time.sleep(20)
        for p in list(live):
            if p.poll() is None:
                continue
            live.remove(p)
            game, cmd, outd, attempt = pending.pop(p.pid)
            if (outd / "test_summary_rexpure_seed1.json").exists():
                print(f"done  {game} (attempt {attempt})", flush=True)
                continue
            if attempt >= max_attempts:
                print(f"FAIL  {game}: exited rc={p.returncode} on attempt {attempt}/"
                      f"{max_attempts}, giving up", flush=True)
                continue
            print(f"retry {game}: exited rc={p.returncode} without shipping "
                  f"(attempt {attempt}/{max_attempts})", flush=True)
            q = _spawn(game, cmd, outd, env, attempt + 1)
            live.append(q)
            pending[q.pid] = (game, cmd, outd, attempt + 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default=",".join(GAMES))
    ap.add_argument("--max-parallel", type=int, default=3)
    ap.add_argument("--attempts", type=int, default=4,
                    help="relaunch a game that exits without shipping, each retry passing "
                         "--resume so it continues from its last checkpointed node. Provider "
                         "429 storms kill a process outright, so one crash must not cost the run")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    env = dict(os.environ, OPENROUTER_PROVIDER_ORDER="groq", **IO_ENV)  # F is gpt-oss-20b@groq
    live: list[subprocess.Popen] = []
    pending: dict[int, tuple] = {}   # pid -> (game, cmd, out-dir, attempt no.)
    for game in args.games.split(","):
        outd = OUT_ROOT / f"{game}_s1"
        if (outd / "test_summary_rexpure_seed1.json").exists():
            print(f"skip  {game}: already complete")
            continue
        cmd = build(game, outd)
        if args.dry_run:
            print(f"[dry-run] {game}:\n  {' '.join(cmd)}\n")
            continue
        while len([p for p in live if p.poll() is None]) >= args.max_parallel:
            time.sleep(15)
        outd.mkdir(parents=True, exist_ok=True)
        p = _spawn(game, cmd, outd, env)
        live.append(p)
        pending[p.pid] = (game, cmd, outd, 1)
    _drain(live, pending, env, args.attempts)
    print("all done")


if __name__ == "__main__":
    main()
