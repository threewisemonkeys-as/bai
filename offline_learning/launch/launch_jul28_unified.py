"""Relaunch the 9 phase-1 Autumn GEPA runs with --credited-scoring (credited FD/ID +
min composite -- see gepa_optimize.py's CREDITED SCORING section) on gpt-oss-120b,
using the UNTIED 30-train/30-val split (distinct samples, new train2 pools).

Per-game settings are inherited from the phase-1 launch.json with the SAME split edits
as launch_jul28_untied_ab.py (so results are split-comparable with the jul28 A/B):
  - --tie-train-val -> --no-tie-train-val
  - --run += ,clean_data3/<game>/train2
  - --context-source-run += ,clean_data3/<game>/train_regen2
plus this sweep's own edits:
  - --out-dir -> logs/jul28_unified/<game>_seed1
  - --task-model/--reflection-model -> openai/gpt-oss-120b (verified live on OpenRouter)
  - + --id-set-loss --id-eps 0.1 --credited-scoring
Queue semantics match launch_jul21_phase1_setloss.py: existing gepa processes count
toward the cap, pending jobs start as slots free up.
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
SRC = ROOT / "logs/clean_sweep_gepa_cd3_fulltraj_aug30_mb15_test50_fulltestctx_phase1"
OUT = ROOT / "logs/jul28_unified"
QUEUE_LOG = OUT / "jul28_queue.log"
QUEUE_STATE = OUT / "jul28_queue_state.json"

GAMES = ["7www9", "7xf97", "bt3gb", "dq8gc", "e3v6m", "f5w3n", "n2ntd", "qfsvc", "vqjh6"]

MODEL = "openai/gpt-oss-120b"
NEW_FLAGS = ["--id-set-loss", "--id-eps", "0.1", "--credited-scoring"]


def active_gepa_pids() -> list[int]:
    active = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"offline_learning/gepa_optimize.py" in cmdline:
            active.append(int(entry.name))
    return sorted(active)


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with QUEUE_LOG.open("a") as handle:
        handle.write(f"{timestamp} {message}\n")


def write_state(pending: list[str], launched: dict[str, int]) -> None:
    state = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "queue_pid": os.getpid(),
        "pending": pending,
        "launched": launched,
        "active_gepa_pids": active_gepa_pids(),
    }
    QUEUE_STATE.write_text(json.dumps(state, indent=2) + "\n")


def build_cmd(game: str, out_dir: Path) -> list[str]:
    source = SRC / f"{game}_seed1/launch.json"
    cmd = list(json.loads(source.read_text())["cmd"])
    cmd[0] = sys.executable  # phase-1 files mix .venv/bin/python and python3
    cmd[cmd.index("--out-dir") + 1] = str(out_dir)
    cmd[cmd.index("--task-model") + 1] = MODEL
    cmd[cmd.index("--reflection-model") + 1] = MODEL
    # untied 30/30 split on the new train2 pools (same edits as launch_jul28_untied_ab.py)
    cmd[cmd.index("--tie-train-val")] = "--no-tie-train-val"
    i = cmd.index("--run")
    cmd[i + 1] += f",{ROOT}/offline_learning/clean_data3/{game}/train2"
    i = cmd.index("--context-source-run")
    cmd[i + 1] += f",{ROOT}/offline_learning/clean_data3/{game}/train_regen2"
    return cmd + NEW_FLAGS


def launch(game: str) -> int:
    out_dir = OUT / f"{game}_seed1"
    if out_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing run: {out_dir}")
    command = build_cmd(game, out_dir)
    out_dir.mkdir(parents=True)

    with (out_dir / "stdout.txt").open("w") as stdout:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    launch_record = {
        "game": game,
        "pid": process.pid,
        "cmd": command,
        "source_launch": str(SRC / f"{game}_seed1/launch.json"),
    }
    (out_dir / "launch.json").write_text(json.dumps(launch_record, indent=2) + "\n")
    (out_dir / "pid").write_text(f"{process.pid}\n")
    return process.pid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("games", nargs="*", default=None)
    parser.add_argument("--max-gepa-processes", type=int, default=5)
    parser.add_argument("--poll-seconds", type=int, default=20)
    args = parser.parse_args()
    games = args.games or list(GAMES)
    unknown = [g for g in games if g not in GAMES]
    if unknown:
        parser.error(f"unknown game(s) {unknown}; choose from {GAMES}")

    OUT.mkdir(parents=True, exist_ok=True)
    pending = list(games)
    launched: dict[str, int] = {}
    log(f"queue started pid={os.getpid()} model={MODEL} pending={pending}")
    while pending:
        active = active_gepa_pids()
        while pending and len(active) < args.max_gepa_processes:
            game = pending.pop(0)
            pid = launch(game)
            launched[game] = pid
            log(f"launched game={game} pid={pid}")
            active = active_gepa_pids()
        write_state(pending, launched)
        if pending:
            time.sleep(args.poll_seconds)
    write_state(pending, launched)
    log(f"queue complete launched={launched}")


if __name__ == "__main__":
    main()
