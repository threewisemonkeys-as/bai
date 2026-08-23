"""Re-run s2kt7 with the aug4_mixed config + --selector rex_pure (B arm), exactly
as the aug5_rexpure s2kt7 run, but on the NEW seed5 data (clean_data3/s2kt7_seed5)
instead of the seed-0-degenerate clean_data3/s2kt7.

Config = n2ntd aug4_mixed cmd verbatim (gpt-oss-120b F / deepseek reflection /
context-k 9 / min(ID, cFD-hard) / id-set-loss / rex_pure) with only the data
flags swapped:
  --run       -> clean_data3/s2kt7_seed5/train   (8 full-trajectory episodes)
  --test-run  -> clean_data3/s2kt7_seed5/test50  (8 held-out episodes)
  --actions   -> noop,click
The seed5 episodes are full trajectories with inline K-step context, so the
--context-source-run / --test-context-source-run backfill flags are DROPPED
(the phase2 seed-0 run needed them because its targets were isolated slices).

Data seed5 = non-degenerate s2kt7 (clicks scatter 2 food at random cells; the
seed-0 set collapsed every click to (0,0)). See memory seed0-degenerate-data-gen.
Metadata strip is on by default (Task:/Step: header removed at load).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from _launch_util import sanitize_rexpure_cmd

ROOT = Path(__file__).resolve().parents[2]
TASK_PIN = "cerebras,groq,sambanova"

MIXED_N2NTD = ROOT / "logs/aug4_mixed/n2ntd_seed1/launch.json"
DATA = ROOT / "offline_learning/clean_data3/s2kt7_seed5"
OUT = ROOT / "logs/aug5_rexpure/s2kt7_seed5data"


def load_cmd(path: Path) -> list[str]:
    return list(json.loads(path.read_text())["cmd"])


def setval(cmd: list[str], flag: str, value: str) -> None:
    cmd[cmd.index(flag) + 1] = value


def drop_flag(cmd: list[str], flag: str) -> None:
    """Remove a `--flag value` pair if present."""
    if flag in cmd:
        i = cmd.index(flag)
        del cmd[i : i + 2]


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite existing run: {OUT}")

    cmd = sanitize_rexpure_cmd(load_cmd(MIXED_N2NTD))
    cmd[0] = sys.executable
    setval(cmd, "--run", str(DATA / "train"))
    setval(cmd, "--test-run", str(DATA / "test50"))
    setval(cmd, "--actions", "noop,click")
    drop_flag(cmd, "--context-source-run")
    drop_flag(cmd, "--test-context-source-run")
    setval(cmd, "--out-dir", str(OUT))

    env = dict(os.environ, OPENROUTER_PROVIDER_ORDER=TASK_PIN)
    OUT.mkdir(parents=True)
    with (OUT / "stdout.txt").open("w") as stdout:
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (OUT / "launch.json").write_text(
        json.dumps(
            {
                "game": "s2kt7",
                "data": "clean_data3/s2kt7_seed5 (non-degenerate, base seed 5)",
                "arm": "rexpure_mixed_gptoss_F_deepseek_refl",
                "pid": process.pid,
                "cmd": cmd,
                "env_pin": TASK_PIN,
                "control": "logs/aug5_rexpure/s2kt7_seed1 (same config, seed-0 degenerate data)",
            },
            indent=2,
        )
        + "\n"
    )
    (OUT / "pid").write_text(f"{process.pid}\n")
    print(f"launched s2kt7 (seed5 data) pid={process.pid} -> {OUT}")


if __name__ == "__main__":
    main()
