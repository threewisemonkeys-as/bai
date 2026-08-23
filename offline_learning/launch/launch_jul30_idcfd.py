"""Launch the FD-term-fix run: bt3gb with the jul30_rex config minus FD-exact,
plus hardened contrastive decoys and the conditional latent-state prompt line.

Derived from logs/jul30_rex/bt3gb_seed1/launch.json (REx selector + min composite +
uncollapsed clicks) with exactly these changes:
  - --fd-scorer exact -> none   (drops FD-exact from the min: score = min(ID, cFD);
    FD-exact pinned 30/45 val rows at 0 -- near-misses on color tokens no candidate
    could pay off without token-exact beliefs)
  - + --cfd-hard-decoys         (near-miss decoys: current frame + most-similar pool
    frames; uniform decoys saturated cFD at 0.83-0.93 = no gradient)
  - the latent-state sentence in the wk reflection template is now conditional
    (code change in gepa_optimize.py; the old declarative form primed 'click is
    invisible' on a game where click flips every color on the board)

Comparison baselines: logs/jul30_rex/bt3gb_seed1 (min incl. FD-exact, uniform
decoys) and logs/jul30_minclick/bt3gb_seed1 (same + pareto selector).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "logs/jul30_rex"
OUT = ROOT / "logs/jul30_idcfd"

GAMES = ["bt3gb"]
ADD = ["--cfd-hard-decoys"]
ENV_PIN = "cerebras,groq,sambanova"


def build_cmd(game: str, out_dir: Path) -> list[str]:
    cmd = list(json.loads((SRC / f"{game}_seed1/launch.json").read_text())["cmd"])
    cmd[0] = sys.executable
    cmd[cmd.index("--out-dir") + 1] = str(out_dir)
    cmd[cmd.index("--fd-scorer") + 1] = "none"
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
                "ab_note": "jul30_rex config with --fd-scorer none + --cfd-hard-decoys "
                "+ conditional latent-state wk prompt line",
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
