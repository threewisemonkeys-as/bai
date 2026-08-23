"""Decoder-confound check for the aug3_dsflash 0.750 result: re-evaluate the
deepseek run's SHIPPED beliefs+perception on the same test50 with gpt-oss-120b
(cerebras pin) as the decoder F. Warm-start from the shipped artifacts with
--max-nodes 1 so the optimizer evaluates only the seed, proposes nothing, and ships
the seed straight into the test eval.

Isolates: is the dsflash lift (0.750 vs 0.667 new-prompts / 0.551 old-prompts,
both gpt-oss-decoded) in the belief content, or in deepseek being a stronger
test-time decoder?
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_RUN = ROOT / "logs/aug3_dsflash/bt3gb_strat30_seed1"
OUT = ROOT / "logs/aug4_dsflash_gptoss_decode/bt3gb_seed1"

ENV_PIN = "cerebras,groq,sambanova"


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite existing run: {OUT}")
    cmd = list(json.loads((SRC_RUN / "launch.json").read_text())["cmd"])
    cmd[0] = sys.executable
    cmd[cmd.index("--out-dir") + 1] = str(OUT)
    cmd[cmd.index("--task-model") + 1] = "openai/gpt-oss-120b"
    cmd[cmd.index("--reflection-model") + 1] = "openai/gpt-oss-120b"
    # seed-only warm-start: budget = 1 node. Handle both the historical
    # (--max-metric-calls) and current (--max-nodes) budget flag names.
    _bi = cmd.index("--max-metric-calls") if "--max-metric-calls" in cmd else cmd.index("--max-nodes")
    cmd[_bi], cmd[_bi + 1] = "--max-nodes", "1"
    cmd[cmd.index("--start-perception") + 1] = str(
        SRC_RUN / "best_perception_gepa_seed1.py"
    )
    cmd += ["--start-beliefs", str(SRC_RUN / "best_beliefs_gepa_seed1.txt")]
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
                "arm": "dsflash_beliefs_gptoss_decode",
                "pid": process.pid,
                "cmd": cmd,
                "env_pin": ENV_PIN,
                "source_launch": str(SRC_RUN / "launch.json"),
                "ab_note": "eval-only: aug3_dsflash shipped artifacts decoded by "
                "gpt-oss-120b on the same test50 -- separates belief quality from "
                "decoder strength in the 0.750 result",
            },
            indent=2,
        )
        + "\n"
    )
    (OUT / "pid").write_text(f"{process.pid}\n")
    print(f"launched pid={process.pid} -> {OUT}")


if __name__ == "__main__":
    main()
