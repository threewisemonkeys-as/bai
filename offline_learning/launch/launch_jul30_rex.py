"""Launch the REx-selector validation A/B: bt3gb (lineage lock-in test) + n2ntd
(control) with EXACTLY the jul30_minclick config plus `--selector rex --rex-c 5`.

Isolation: same objective (min(ID, FD[exact], contrastiveFD)), same uncollapsed
click actions, same data/split/budget/model/seed -- the ONLY change is parent
selection (pareto -> REx Thompson sampling, arXiv:2405.17503). Any difference in
pool composition or outcome is attributable to the selector.

Validation targets (vs logs/jul30_minclick):
  bt3gb: seed re-selection count > 0 (pareto: 0/48 after the forced first pick);
         first hash-free non-seed candidate exists (pareto pool: 36/36 hash-carrying);
         representation-bucket count in final pool.
  n2ntd: no regression in test ID / pool quality (REx should exploit ~as well
         since its pool had genuinely higher-h arms).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from _launch_util import sanitize_rexpure_cmd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "logs/jul30_minclick"
OUT = ROOT / "logs/jul30_rex"

GAMES = ["bt3gb", "n2ntd"]
# gated rex was removed with gepa; this now re-runs the minclick config as REx-pure
# (--rex-c 5 is REx-pure's default, so no extra flags are appended).
ADD = ["--rex-c", "5"]
ENV_PIN = "cerebras,groq,sambanova"


def build_cmd(game: str, out_dir: Path) -> list[str]:
    cmd = sanitize_rexpure_cmd(
        list(json.loads((SRC / f"{game}_seed1/launch.json").read_text())["cmd"]))
    cmd[0] = sys.executable
    cmd[cmd.index("--out-dir") + 1] = str(out_dir)
    return cmd + ADD


def launch(game: str) -> int:
    out_dir = OUT / f"{game}_seed1"
    if out_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing run: {out_dir}")
    command = build_cmd(game, out_dir)
    out_dir.mkdir(parents=True)
    env = dict(os.environ, OPENROUTER_PROVIDER_ORDER=ENV_PIN)
    with (out_dir / "stdout.txt").open("w") as stdout:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (out_dir / "launch.json").write_text(
        json.dumps(
            {
                "game": game,
                "pid": process.pid,
                "cmd": command,
                "env_pin": ENV_PIN,
                "source_launch": str(SRC / f"{game}_seed1/launch.json"),
                "ab_note": "jul30_minclick config re-run as REx-pure (gated rex removed with gepa)",
            },
            indent=2,
        )
        + "\n"
    )
    (out_dir / "pid").write_text(f"{process.pid}\n")
    return process.pid


def main() -> None:
    games = sys.argv[1:] or list(GAMES)
    unknown = [g for g in games if g not in GAMES]
    if unknown:
        raise SystemExit(f"unknown game(s) {unknown}; choose from {GAMES}")
    OUT.mkdir(parents=True, exist_ok=True)
    for game in games:
        pid = launch(game)
        print(f"launched {game} pid={pid} -> {OUT / f'{game}_seed1'}")


if __name__ == "__main__":
    main()
