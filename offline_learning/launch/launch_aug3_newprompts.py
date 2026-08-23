"""Rerun the strat30 arm with the NEW analysis/reflection prompts (2026-08-03 edits:
falsification-first analyzer w/ REVISE:/DELETE:/ADD: prefixes + truth-label supremacy,
AUDIT->TABULATE->REWRITE wk reflection protocol w/ <changes> list, cFD-invisibility
feedback fix, correct-row reword). Config otherwise inherited VERBATIM from
logs/aug2_valab/bt3gb_strat30_seed1/launch.json (min(ID,cFD-hard), REx, stratified
30/30 split, gpt-oss-120b, seed 1) -- only --out-dir changes, so the A/B isolates
the prompt changes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "logs/aug2_valab/bt3gb_strat30_seed1/launch.json"
OUT = ROOT / "logs/aug3_newprompts/bt3gb_strat30_seed1"

ENV_PIN = "cerebras,groq,sambanova"


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite existing run: {OUT}")
    cmd = list(json.loads(SRC.read_text())["cmd"])
    cmd[0] = sys.executable
    cmd[cmd.index("--out-dir") + 1] = str(OUT)
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
                "game": "bt3gb",
                "arm": "strat30_newprompts",
                "pid": process.pid,
                "cmd": cmd,
                "env_pin": ENV_PIN,
                "source_launch": str(SRC),
                "ab_note": "prompt-change A/B vs aug2 strat30: identical config; only "
                "gepa_optimize.py templates changed (falsification-first analyzer, "
                "audit/tabulate/rewrite wk reflection, cFD feedback fix)",
            },
            indent=2,
        )
        + "\n"
    )
    (OUT / "pid").write_text(f"{process.pid}\n")
    print(f"launched pid={process.pid} -> {OUT}")


if __name__ == "__main__":
    main()
