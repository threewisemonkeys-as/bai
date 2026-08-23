"""Relaunch the 9 phase-1 Autumn GEPA runs with the set-based ID loss (--id-set-loss).

Every per-game setting (data paths, whitelist, collapse, models, budget, seed) is
inherited VERBATIM from the command saved in the phase-1 run's launch.json; only the
out dir changes (logs/jul21) and the new-metric flags are appended. Queue semantics
match launch_remaining_autumn_phase1.py: existing gepa processes count toward the
cap, pending jobs start as slots free up.
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
OUT = ROOT / "logs/jul21"
QUEUE_LOG = OUT / "jul21_queue.log"
QUEUE_STATE = OUT / "jul21_queue_state.json"

GAMES = ["7www9", "7xf97", "bt3gb", "dq8gc", "e3v6m", "f5w3n", "n2ntd", "qfsvc", "vqjh6"]

ID_SET_FLAGS = ["--id-set-loss", "--id-eps", "0.1"]


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
    i = cmd.index("--out-dir")
    cmd[i + 1] = str(out_dir)
    return cmd + ID_SET_FLAGS


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
    log(f"queue started pid={os.getpid()} pending={pending}")
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
