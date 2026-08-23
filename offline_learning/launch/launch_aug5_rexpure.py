"""Launch --selector rex_pure (B arm) with the mixed model config on bt3gb, n2ntd,
and s2kt7. Controls are the gated aug4_mixed runs (bt3gb 0.780 / n2ntd 0.625);
s2kt7 has no mixed-config control and is included as a stress test of the
no-gate mode on the FD-collapse-attractor game.

Config = aug4_mixed verbatim except --selector rex_pure (no acceptance gate,
h = full-train, full-train reflection minibatch, traced-eval budget refund).
bt3gb/n2ntd reuse their own aug4_mixed cmds; s2kt7 transplants its data args
(--run/--test-run/--context-source-run/--test-context-source-run/--actions)
into the n2ntd mixed cmd so the optimizer config is identical across games.

NOTE: these runs use today's default obs-metadata STRIP (Task:/Step: header
removed at load). The aug4_mixed controls predate the strip, so cross-arm
comparisons carry that caveat (the header was a score-gaming side channel;
B faces the cleaner objective).
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
MIXED_BT3GB = ROOT / "logs/aug4_mixed/bt3gb_strat30_seed1/launch.json"
S2KT7_DATA_SRC = (
    ROOT / "logs/clean_sweep_gepa_cd3_fulltraj_aug30_mb15_test50_fulltestctx_phase2/s2kt7_seed1/launch.json"
)

DATA_FLAGS = [
    "--run",
    "--test-run",
    "--context-source-run",
    "--test-context-source-run",
    "--actions",
]


def load_cmd(path: Path) -> list[str]:
    return list(json.loads(path.read_text())["cmd"])


def setval(cmd: list[str], flag: str, value: str) -> None:
    cmd[cmd.index(flag) + 1] = value


def main() -> None:
    n2ntd_cmd = load_cmd(MIXED_N2NTD)
    s2kt7_src = load_cmd(S2KT7_DATA_SRC)

    s2kt7_cmd = list(n2ntd_cmd)
    for flag in DATA_FLAGS:
        setval(s2kt7_cmd, flag, s2kt7_src[s2kt7_src.index(flag) + 1])

    runs = [
        ("bt3gb", load_cmd(MIXED_BT3GB), "bt3gb_strat30_seed1"),
        ("n2ntd", n2ntd_cmd, "n2ntd_seed1"),
        ("s2kt7", s2kt7_cmd, "s2kt7_seed1"),
    ]

    for game, base_cmd, out_name in runs:
        out = ROOT / "logs/aug5_rexpure" / out_name
        if out.exists():
            raise SystemExit(f"refusing to overwrite existing run: {out}")

    env = dict(os.environ, OPENROUTER_PROVIDER_ORDER=TASK_PIN)
    for game, base_cmd, out_name in runs:
        out = ROOT / "logs/aug5_rexpure" / out_name
        cmd = sanitize_rexpure_cmd(base_cmd)
        cmd[0] = sys.executable
        setval(cmd, "--out-dir", str(out))
        out.mkdir(parents=True)
        with (out / "stdout.txt").open("w") as stdout:
            process = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        (out / "launch.json").write_text(
            json.dumps(
                {
                    "game": game,
                    "arm": "rexpure_mixed_gptoss_F_deepseek_refl",
                    "pid": process.pid,
                    "cmd": cmd,
                    "env_pin": TASK_PIN,
                    "control": (
                        f"logs/aug4_mixed/{out_name} (gated rex, pre-strip headers)"
                        if game != "s2kt7"
                        else "none (stress test; nearest = clean_sweep phase2 runs)"
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        (out / "pid").write_text(f"{process.pid}\n")
        print(f"launched {game} pid={process.pid} -> {out}")


if __name__ == "__main__":
    main()
