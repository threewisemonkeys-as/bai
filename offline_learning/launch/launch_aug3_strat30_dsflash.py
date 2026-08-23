"""Re-run the strat30 arm with deepseek-v4-flash as task+reflection model.

Inherits the full bt3gb strat30 config verbatim (min(ID, cFD-hard), fd_scorer
none, REx, conditional latent line, val-id cache fix, --stratified-split) from
logs/aug2_valab/bt3gb_strat30_seed1/launch.json and changes ONLY:
  --task-model / --reflection-model  ->  deepseek/deepseek-v4-flash
  --out-dir                          ->  logs/aug3_dsflash/bt3gb_strat30_seed1

No OPENROUTER_PROVIDER_ORDER pin: the cerebras/groq/sambanova chain is
gpt-oss-120b-specific; the jul28 deepseek runs used the default routing.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "logs/aug2_valab/bt3gb_strat30_seed1/launch.json"
OUT = ROOT / "logs/aug3_dsflash/bt3gb_strat30_seed1"

MODEL = "deepseek/deepseek-v4-flash"


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite existing run: {OUT}")
    cmd = list(json.loads(SRC.read_text())["cmd"])
    cmd[0] = sys.executable
    cmd[cmd.index("--out-dir") + 1] = str(OUT)
    cmd[cmd.index("--task-model") + 1] = MODEL
    cmd[cmd.index("--reflection-model") + 1] = MODEL
    OUT.mkdir(parents=True)
    with (OUT / "stdout.txt").open("w") as stdout:
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (OUT / "launch.json").write_text(
        json.dumps(
            {
                "game": "bt3gb",
                "arm": "strat30_dsflash",
                "pid": process.pid,
                "cmd": cmd,
                "source_launch": str(SRC),
                "note": "strat30 config verbatim, task+reflection model swapped "
                "to deepseek/deepseek-v4-flash; no provider pin",
            },
            indent=2,
        )
        + "\n"
    )
    (OUT / "pid").write_text(f"{process.pid}\n")
    print(f"launched pid={process.pid} -> {OUT}")


if __name__ == "__main__":
    main()
