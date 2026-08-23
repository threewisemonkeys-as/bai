"""Queue remaining Autumn phase-2 runs after all phase-1 jobs are launched."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "offline_learning"
INPUT = ROOT / "logs/fulltraj_context_remaining_autumn_phase2_inputs"
OUT = ROOT / "logs/clean_sweep_gepa_cd3_fulltraj_aug30_mb15_test50_fulltestctx_phase2"
PHASE1_STATE = (
    ROOT
    / "logs/clean_sweep_gepa_cd3_fulltraj_aug30_mb15_test50_fulltestctx_phase1"
    / "phase1_queue_state.json"
)
QUEUE_LOG = OUT / "phase2_queue.log"
QUEUE_STATE = OUT / "phase2_queue_state.json"

CONFIG = {
    "83wkq": {
        "actions": "noop,click",
        "train_target": HERE / "clean_data3/83wkq/train",
        "train_source": HERE / "clean_data2/83wkq/train",
        "collapse": False,
    },
    "27vwc": {
        "actions": "noop,click",
        "train_target": HERE / "clean_data3/27vwc/train",
        "train_source": HERE / "clean_data2/27vwc/train",
        "collapse": False,
    },
    "qqm74": {
        "actions": "left,right,up,down,noop,click",
        "train_target": HERE / "clean_data3/qqm74/train",
        "train_source": HERE / "clean_data2/qqm74/train",
        "collapse": True,
    },
    "eahcw": {
        "actions": "left,right,up,down,noop,click",
        "train_target": INPUT / "eahcw/train20",
        "train_source": HERE / "clean_data2/eahcw/train",
        "collapse": True,
    },
    "s2kt7": {
        "actions": "noop,click",
        "train_target": HERE / "clean_data3/s2kt7/train",
        "train_source": HERE / "clean_data3/s2kt7/train_regen",
        "collapse": False,
    },
    "ice": {
        "actions": "left,right,up,down,noop,click",
        "train_target": HERE / "clean_data3/ice/train",
        "train_source": HERE / "clean_data2/ice/train",
        "collapse": True,
    },
}


def active_gepa_pids() -> list[int]:
    active = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(b"offline_learning/gepa_optimize.py" in arg for arg in argv):
            active.append(int(entry.name))
    return sorted(active)


def phase1_pending() -> list[str]:
    if not PHASE1_STATE.exists():
        return ["phase1 state unavailable"]
    return list(json.loads(PHASE1_STATE.read_text()).get("pending", []))


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with QUEUE_LOG.open("a") as handle:
        handle.write(f"{timestamp} {message}\n")


def write_state(pending: list[str], launched: dict[str, int]) -> None:
    state = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "queue_pid": os.getpid(),
        "waiting_for_phase1": phase1_pending(),
        "pending": pending,
        "launched": launched,
        "active_gepa_pids": active_gepa_pids(),
    }
    QUEUE_STATE.write_text(json.dumps(state, indent=2) + "\n")


def launch(game: str) -> int:
    config = CONFIG[game]
    manifest = json.loads((INPUT / game / "test50_context_manifest.json").read_text())
    test_runs = ",".join(str(ROOT / row["target_root"]) for row in manifest)
    test_sources = ",".join(str(ROOT / row["source_root"]) for row in manifest)
    out_dir = OUT / f"{game}_seed1"
    if out_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing run: {out_dir}")
    out_dir.mkdir(parents=True)
    command = [
        sys.executable,
        str(HERE / "rexpure_optimize.py"),
        "--run",
        f"{config['train_target']},{INPUT}/{game}/old_test10",
        "--test-run",
        test_runs,
        "--context-source-run",
        f"{config['train_source']},{HERE}/clean_data2/{game}/test",
        "--test-context-source-run",
        test_sources,
        "--train-n",
        "30",
        "--test-n",
        "50",
        "--actions",
        str(config["actions"]),
        "--fd-scorer",
        "exact",
        "--max-nodes",
        "67",
        "--task-model",
        "deepseek/deepseek-v4-flash",
        "--reflection-model",
        "deepseek/deepseek-v4-flash",
        "--context-k",
        "9",
        "--concurrency",
        "16",
        "--seed",
        "1",
        "--out-dir",
        str(out_dir),
        "--analyze-mistakes",
        "--start-perception",
        str(HERE / "autumn_seed_perception.py"),
    ]
    if config["collapse"]:
        command.append("--collapse-action-params")
    with (out_dir / "stdout.txt").open("w") as stdout:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    launch_record = {"game": game, "pid": process.pid, "cmd": command}
    (out_dir / "launch.json").write_text(json.dumps(launch_record, indent=2) + "\n")
    (out_dir / "pid").write_text(f"{process.pid}\n")
    return process.pid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("games", nargs="*", choices=CONFIG, default=list(CONFIG))
    parser.add_argument("--max-gepa-processes", type=int, default=5)
    parser.add_argument("--poll-seconds", type=int, default=20)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pending = list(args.games)
    launched: dict[str, int] = {}
    log(f"queue started pid={os.getpid()} pending={pending}")
    while phase1_pending():
        write_state(pending, launched)
        time.sleep(args.poll_seconds)
    log("phase1 launch queue drained; phase2 may fill free evaluator slots")
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
