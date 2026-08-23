"""Launch the 5-game A/B of the raw min-composite + contrastive-FD objective with
click semantics enabled (no action-param collapsing).

Per-game commands inherit logs/jul28_unified/<game>_seed1/launch.json (gpt-oss-120b,
untied 30/30 split, train2 pools, --id-set-loss) with exactly these edits:
  - REMOVE --credited-scoring        (root cause of the jul28_unified regression:
    the candidate-dependent blind-FD floor taxed competence and vetoed belief edits)
  - REMOVE --collapse-action-params  (full 'click R C' strings become ID targets;
    the symmetric context window makes click coordinates inferable from the
    post-click transitions, so click semantics are now learnable)
  - ADD --composite min --contrastive-fd  (score = min(ID, FD[exact], contrastive
    FD); contrastive target space is candidate-independent raw frames, closing the
    information-destruction exploit that min(ID, FD) alone provably cannot see)
  - --concurrency 48 (validated with OPENROUTER_PROVIDER_ORDER=cerebras,groq,sambanova)

Game subset rationale (vs old sweeps + jul28_unified + jul28_unified_reselect):
  e3v6m  belief-veto recovery (pool contained the true-mechanics candidate)
  n2ntd  extreme belief-veto case (24/26 pool candidates had empty beliefs)
  bt3gb  degenerate empty-P lineage acceptance test
  dq8gc  the color-collapse + click=select game (contrastive + uncollapse both bite)
  qfsvc  no-regression control (top scorer under credited)

All 5 launch immediately (240 peak concurrent calls, stress-validated at 405).
Existing gepa processes (e.g. the logs/jul28 deepseek A/B tail) are left alone.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "logs/jul28_unified"
OUT = ROOT / "logs/jul30_minclick"

GAMES = ["bt3gb", "dq8gc", "e3v6m", "n2ntd", "qfsvc"]
DROP = {"--credited-scoring", "--collapse-action-params"}
ADD = ["--composite", "min", "--contrastive-fd"]
ENV_PIN = "cerebras,groq,sambanova"


def build_cmd(game: str, out_dir: Path) -> list[str]:
    # runs started fresh by the pinned relaunch have relaunch.json, not launch.json
    src = SRC / f"{game}_seed1/launch.json"
    if not src.exists():
        src = SRC / f"{game}_seed1/relaunch.json"
    cmd = list(json.loads(src.read_text())["cmd"])
    cmd[0] = sys.executable
    cmd = [c for c in cmd if c not in DROP]
    cmd[cmd.index("--out-dir") + 1] = str(out_dir)
    cmd[cmd.index("--concurrency") + 1] = "48"
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
