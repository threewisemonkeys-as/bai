"""Relaunch the jul28_unified sweep with OpenRouter provider pinning + full-wave
concurrency after a graceful gepa.stop of the in-flight runs.

Per game in logs/jul28_unified/<game>_seed1:
  - existing dir  -> resume: reuse the exact cmd from its launch.json (GEPA
    reloads gepa_state.bin incl. the metric-call budget), preserve the at-stop
    test summary/traces the graceful stop produced, remove gepa.stop
  - missing dir   -> fresh launch via launch_jul28_unified.build_cmd
All 9 launch immediately (no queue cap -- the A/B deepseek sweep keeps its 5
slots untouched; its cap-5 queue stays dormant while >=5 gepa procs run) with:
  - OPENROUTER_PROVIDER_ORDER=cerebras,groq,sambanova (env-gated pin in
    mixed_improve._llm_call; deepseek A/B processes never see this env var)
  - --concurrency 48 (a full reflection-minibatch-15 wave in one pass)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from launch_jul28_unified import GAMES, OUT, build_cmd, log

PIN = "cerebras,groq,sambanova"
CONCURRENCY = "48"
AT_STOP_KEEP = [
    "test_summary_gepa_seed1.json",
    "test_trace_gepa_seed1.json",
    "test_trace_fd_gepa_seed1.json",
]


def old_pid(out_dir: Path) -> int | None:
    try:
        return int((out_dir / "pid").read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def wait_for_exits(timeout_s: int = 1800) -> None:
    deadline = time.time() + timeout_s
    while True:
        alive = [
            (g, p)
            for g in GAMES
            if (p := old_pid(OUT / f"{g}_seed1")) and Path(f"/proc/{p}").exists()
        ]
        if not alive:
            return
        if time.time() > deadline:
            raise RuntimeError(f"old runs still alive after {timeout_s}s: {alive}")
        print(f"waiting on {alive}", flush=True)
        time.sleep(15)


def relaunch(game: str) -> int:
    out_dir = OUT / f"{game}_seed1"
    resume = out_dir.exists()
    if resume:
        cmd = list(json.loads((out_dir / "launch.json").read_text())["cmd"])
        for name in AT_STOP_KEEP:
            src = out_dir / name
            if src.exists():
                shutil.copy2(src, out_dir / name.replace(".json", ".at_stop.json"))
        (out_dir / "gepa_run_seed1" / "gepa.stop").unlink(missing_ok=True)
    else:
        cmd = build_cmd(game, out_dir)
        out_dir.mkdir(parents=True)
    cmd[0] = sys.executable
    cmd[cmd.index("--concurrency") + 1] = CONCURRENCY

    env = dict(os.environ, OPENROUTER_PROVIDER_ORDER=PIN)
    with (out_dir / "stdout.txt").open("a") as stdout:
        process = subprocess.Popen(
            cmd,
            cwd=OUT.parents[1],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (out_dir / "relaunch.json").write_text(
        json.dumps(
            {"game": game, "pid": process.pid, "resume": resume, "pin": PIN,
             "concurrency": CONCURRENCY, "cmd": cmd},
            indent=2,
        )
        + "\n"
    )
    (out_dir / "pid").write_text(f"{process.pid}\n")
    return process.pid


def main() -> None:
    wait_for_exits()
    launched = {}
    for game in GAMES:
        launched[game] = relaunch(game)
        log(f"pinned-relaunch game={game} pid={launched[game]}")
        print(f"launched {game} pid={launched[game]}", flush=True)
    (OUT / "jul28_queue_state.json").write_text(
        json.dumps(
            {
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "note": "pinned relaunch, no queue: all 9 launched at once",
                "pending": [],
                "launched": launched,
            },
            indent=2,
        )
        + "\n"
    )
    log(f"pinned relaunch complete launched={launched}")


if __name__ == "__main__":
    main()
