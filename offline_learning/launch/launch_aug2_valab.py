"""Launch the val-design A/B on bt3gb: what should the selection/val signal be?

Both arms inherit the jul30_idcfd config (min(ID, cFD-hard), fd_scorer none, REx,
conditional latent line) plus the val-id cache fix, and change ONLY the split:

  strat30: --stratified-split           -- distinct 30/30, but each action's rows
           dealt alternately so BOTH sets contain every action (up/click included);
           val stays held-out. Fixes "val cannot arbitrate action semantics"
           without giving up the generalization check.
  tied60:  train==val = the whole ~62-transition pool (drop --no-tie-train-val,
           --train-n 60) -- maximum evidence + coverage, but candidates are ranked
           and shipped on the same rows the gate reflects on (overfitting risk;
           test50 is the arbiter).

Baseline for comparison: logs/jul30_idcfd/bt3gb_seed1 (distinct 30/30, val = the
degenerate leftover carve: 0 up, 0 click, 2 down).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "logs/jul30_idcfd/bt3gb_seed1/launch.json"
OUT = ROOT / "logs/aug2_valab"

ENV_PIN = "cerebras,groq,sambanova"
ARMS = ["strat30", "tied60"]


def build_cmd(arm: str, out_dir: Path) -> list[str]:
    cmd = list(json.loads(SRC.read_text())["cmd"])
    cmd[0] = sys.executable
    cmd[cmd.index("--out-dir") + 1] = str(out_dir)
    if arm == "strat30":
        cmd.append("--stratified-split")
    elif arm == "tied60":
        cmd.remove("--no-tie-train-val")  # tie-train-val defaults ON
        cmd[cmd.index("--train-n") + 1] = "60"
    else:
        raise ValueError(arm)
    return cmd


def launch(arm: str) -> int:
    out_dir = OUT / f"bt3gb_{arm}_seed1"
    if out_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing run: {out_dir}")
    command = build_cmd(arm, out_dir)
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
                "game": "bt3gb",
                "arm": arm,
                "pid": process.pid,
                "cmd": command,
                "env_pin": ENV_PIN,
                "source_launch": str(SRC),
                "ab_note": "val-design A/B vs jul30_idcfd: only the train/val split "
                "changed (strat30 = stratified distinct 30/30; tied60 = train==val=60)",
            },
            indent=2,
        )
        + "\n"
    )
    (out_dir / "pid").write_text(f"{process.pid}\n")
    return process.pid


def main() -> None:
    arms = sys.argv[1:] or list(ARMS)
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; choose from {ARMS}")
    OUT.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        pid = launch(arm)
        print(f"launched {arm} pid={pid} -> {OUT / f'bt3gb_{arm}_seed1'}")


if __name__ == "__main__":
    main()
