"""n2ntd under the full bt3gb aug3_newprompts recipe: derive from the jul30_rex n2ntd
launch (min(ID,FD-exact,cFD) + REx) and apply the same deltas that produced the
bt3gb arm -- --fd-scorer none, --cfd-hard-decoys, --stratified-split -- plus the
2026-08-03 template changes (falsification-first analyzer, audit/tabulate/rewrite
reflection, cFD feedback fix) which are picked up from gepa_optimize.py at import.

Comparison baselines: jul30_rex n2ntd 0.438 (best prior), jul30_minclick 0.295.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "logs/jul30_rex/n2ntd_seed1/launch.json"
OUT = ROOT / "logs/aug4_newprompts/n2ntd_seed1"

ENV_PIN = "cerebras,groq,sambanova"


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite existing run: {OUT}")
    cmd = list(json.loads(SRC.read_text())["cmd"])
    cmd[0] = sys.executable
    cmd[cmd.index("--out-dir") + 1] = str(OUT)
    cmd[cmd.index("--fd-scorer") + 1] = "none"
    cmd += ["--cfd-hard-decoys", "--stratified-split"]
    OUT.mkdir(parents=True)
    env = dict(os.environ, OPENROUTER_PROVIDER_ORDER=ENV_PIN)
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
                "game": "n2ntd",
                "arm": "newprompts",
                "pid": process.pid,
                "cmd": cmd,
                "env_pin": ENV_PIN,
                "source_launch": str(SRC),
                "ab_note": "n2ntd under the bt3gb aug3_newprompts recipe: jul30_rex config "
                "+ fd-scorer none + cfd-hard-decoys + stratified-split + new templates",
            },
            indent=2,
        )
        + "\n"
    )
    (OUT / "pid").write_text(f"{process.pid}\n")
    print(f"launched pid={process.pid} -> {OUT}")


if __name__ == "__main__":
    main()
