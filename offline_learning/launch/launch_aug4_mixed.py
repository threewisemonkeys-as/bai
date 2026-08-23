"""Mixed-model config: gpt-oss-120b decoder (F, cerebras-pinned) + deepseek-v4-flash
reflection/analysis (pinned to its fastest providers), NEW prompts. Motivated by the
2026-08-04 decomposition of the aug3_dsflash 10h wall: deepseek-as-decoder unpinned
cost ~5.7h of eval and the 30s hedge fired on 80% of requests, while the 0.750 lift
lives ~entirely in the belief content (gpt-oss re-decode of dsflash artifacts: 0.743).
Deepseek is therefore needed only in the reflection role.

Derives each game's cmd verbatim from its NEW-prompts gpt-oss run and swaps only the
reflection role + per-role routing (flags added to gepa_optimize 2026-08-04):
  bt3gb : logs/aug3_newprompts/bt3gb_strat30_seed1  (test 0.667)
  n2ntd : logs/aug4_newprompts/n2ntd_seed1          (test 0.618)
Comparisons: same-decoder A/B isolates the reflection model under new prompts;
vs aug3_dsflash (0.750/0.743, old prompts) stacks prompts x reflection model.

Reflection hedge 120s ~ p75-p90 of a deepseek reflection call (7-8k ch at ~70 tps)
so hedges race the tail, not the median; timeout 300s clears the global 90s default.
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
REFL_PIN = "deepseek,baidu,fireworks"

RUNS = [
    ("bt3gb", "logs/aug3_newprompts/bt3gb_strat30_seed1", "logs/aug4_mixed/bt3gb_strat30_seed1"),
    ("n2ntd", "logs/aug4_newprompts/n2ntd_seed1", "logs/aug4_mixed/n2ntd_seed1"),
]


def main() -> None:
    for game, src, out in RUNS:
        src_launch = ROOT / src / "launch.json"
        outd = ROOT / out
        if outd.exists():
            raise SystemExit(f"refusing to overwrite existing run: {outd}")
        cmd = sanitize_rexpure_cmd(json.loads(src_launch.read_text())["cmd"])
        cmd[0] = sys.executable
        cmd[cmd.index("--out-dir") + 1] = str(outd)
        cmd[cmd.index("--reflection-model") + 1] = "deepseek/deepseek-v4-flash"
        cmd += [
            "--task-provider-order", TASK_PIN,
            "--reflection-provider-order", REFL_PIN,
            "--reflection-hedge-delay", "120",
            "--reflection-timeout", "300",
        ]
        outd.mkdir(parents=True)
        env = dict(os.environ, OPENROUTER_PROVIDER_ORDER=TASK_PIN)
        with (outd / "stdout.txt").open("w") as stdout:
            process = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        (outd / "launch.json").write_text(
            json.dumps(
                {
                    "game": game,
                    "arm": "mixed_gptoss_F_deepseek_refl",
                    "pid": process.pid,
                    "cmd": cmd,
                    "task_pin": TASK_PIN,
                    "refl_pin": REFL_PIN,
                    "source_launch": str(src_launch),
                    "ab_note": "same cmd as the new-prompts gpt-oss run; only the "
                    "reflection/analysis role swaps to deepseek-v4-flash with its own "
                    "provider pin + hedge/timeout (per-role routing, 2026-08-04)",
                },
                indent=2,
            )
            + "\n"
        )
        (outd / "pid").write_text(f"{process.pid}\n")
        print(f"launched {game} pid={process.pid} -> {outd}")


if __name__ == "__main__":
    main()
