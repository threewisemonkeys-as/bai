"""Launch the jul28 untied-split A/B: 9 phase-1 Autumn games x 2 arms (exact-match vs
set-loss ID), both on the SAME current binary and the SAME untied 30/30 train/val split.

Per-game settings are inherited VERBATIM from the phase-1 launch.json, with exactly these
edits per arm:
  - --out-dir -> logs/jul28/<game>_<arm>_seed1
  - --tie-train-val -> --no-tie-train-val (untied split; --train-n 30 --val-n 30 already set)
  - --run  += ,clean_data3/<game>/train2          (the new ~30-transition pools)
  - --context-source-run += ,clean_data3/<game>/train_regen2  (pairwise-zipped backfill src)
  - setloss arm only: + --id-set-loss --id-eps 0.1
Queue semantics match launch_jul21_phase1_setloss.py (cap-N /proc scan).
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
OUT = ROOT / "logs/jul28"
QUEUE_LOG = OUT / "jul28_queue.log"
QUEUE_STATE = OUT / "jul28_queue_state.json"

GAMES = ["7www9", "7xf97", "bt3gb", "dq8gc", "e3v6m", "f5w3n", "n2ntd", "qfsvc", "vqjh6"]
ARMS = {"exact": [], "setloss": ["--id-set-loss", "--id-eps", "0.1"]}


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


def build_cmd(game: str, arm: str, out_dir: Path) -> list[str]:
    source = SRC / f"{game}_seed1/launch.json"
    cmd = list(json.loads(source.read_text())["cmd"])
    cmd[0] = sys.executable
    cmd[cmd.index("--out-dir") + 1] = str(out_dir)
    cmd[cmd.index("--tie-train-val")] = "--no-tie-train-val"
    i = cmd.index("--run")
    cmd[i + 1] += f",{ROOT}/offline_learning/clean_data3/{game}/train2"
    i = cmd.index("--context-source-run")
    cmd[i + 1] += f",{ROOT}/offline_learning/clean_data3/{game}/train_regen2"
    return cmd + ARMS[arm]


def launch(job: str) -> int:
    game, arm = job.rsplit(":", 1)
    out_dir = OUT / f"{game}_{arm}_seed1"
    if out_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing run: {out_dir}")
    command = build_cmd(game, arm, out_dir)
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
        "arm": arm,
        "pid": process.pid,
        "cmd": command,
        "source_launch": str(SRC / f"{game}_seed1/launch.json"),
    }
    (out_dir / "launch.json").write_text(json.dumps(launch_record, indent=2) + "\n")
    (out_dir / "pid").write_text(f"{process.pid}\n")
    return process.pid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "jobs",
        nargs="*",
        default=None,
        help="jobs as <game>:<arm> (default: all 9 games x both arms)",
    )
    parser.add_argument("--max-gepa-processes", type=int, default=5)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--print-cmds", action="store_true")
    args = parser.parse_args()
    jobs = args.jobs or [f"{g}:{a}" for g in GAMES for a in ARMS]
    for job in jobs:
        game, _, arm = job.rpartition(":")
        if game not in GAMES or arm not in ARMS:
            parser.error(f"bad job {job!r}; use <game>:<arm> from {GAMES} x {list(ARMS)}")

    if args.print_cmds:
        for job in jobs:
            game, arm = job.rsplit(":", 1)
            print(f"# {job}")
            print(" ".join(build_cmd(game, arm, OUT / f"{game}_{arm}_seed1")))
        return

    OUT.mkdir(parents=True, exist_ok=True)
    pending = list(jobs)
    launched: dict[str, int] = {}
    log(f"queue started pid={os.getpid()} pending={pending}")
    while pending:
        active = active_gepa_pids()
        while pending and len(active) < args.max_gepa_processes:
            job = pending.pop(0)
            pid = launch(job)
            launched[job] = pid
            log(f"launched job={job} pid={pid}")
            active = active_gepa_pids()
        write_state(pending, launched)
        if pending:
            time.sleep(args.poll_seconds)
    write_state(pending, launched)
    log(f"queue complete launched={launched}")


if __name__ == "__main__":
    main()
