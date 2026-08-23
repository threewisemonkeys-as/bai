"""Softmin A/B vs the aug5_rexpure hard-min runs: same three games, same data,
same model mix (gpt-oss-120b F cerebras-pinned / deepseek-v4-flash reflection),
with the per-transition composite swapped min -> Boltzmann softmin (tau 0.25).
Softmin restores selection visibility of progress on the non-binding term (the
hard min's "min-veto" killed breakthrough beliefs in the bt3gb lineage
forensics) while capping the one-term-gaming payoff at ~sigmoid(-1/tau) ~ 0.018
per transition.

Derivation: each game's cmd comes verbatim from its aug5_rexpure launch.json
(s2kt7 uses the seed5data run -- non-degenerate data, see memory
seed0-degenerate-data-gen), sanitized for the standalone rexpure CLI, with only:
  - dead prototypes/perc_invdyn/ paths rewritten to offline_learning/ (aug7 rename)
  - --max-metric-calls 2000 (old gepa budget) -> --max-nodes 25/45/45
  - --composite min -> softmin, + --softmin-tau 0.25
  - --task-reasoning-json '{"effort": "low"}' for the gpt-oss F calls
    (aug7 bench: effort low beats prod on wall/cost at equal quality)
Budget caveat for the A/B: the aug5 controls ran under the old metric-call
accounting, so train-score-vs-nodes curves, not endpoints, are the comparison.
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

RUNS = [
    ("bt3gb", "logs/aug5_rexpure/bt3gb_strat30_seed1", "bt3gb_strat30_seed1", 25),
    ("n2ntd", "logs/aug5_rexpure/n2ntd_seed1", "n2ntd_seed1", 45),
    ("s2kt7", "logs/aug5_rexpure/s2kt7_seed5data", "s2kt7_seed5data", 45),
]


def repoint(tok: str) -> str:
    return tok.replace(str(ROOT / "prototypes/perc_invdyn"), str(ROOT / "offline_learning"))


def setval(cmd: list[str], flag: str, value: str) -> None:
    cmd[cmd.index(flag) + 1] = value


def drop_flag(cmd: list[str], flag: str) -> None:
    if flag in cmd:
        i = cmd.index(flag)
        del cmd[i : i + 2]


def main() -> None:
    for _, src, out_name, _ in RUNS:
        out = ROOT / "logs/aug7_softmin" / out_name
        if out.exists():
            raise SystemExit(f"refusing to overwrite existing run: {out}")

    env = dict(os.environ, OPENROUTER_PROVIDER_ORDER=TASK_PIN)
    for game, src, out_name, max_nodes in RUNS:
        out = ROOT / "logs/aug7_softmin" / out_name
        cmd = sanitize_rexpure_cmd(json.loads((ROOT / src / "launch.json").read_text())["cmd"])
        cmd = [repoint(tok) for tok in cmd]
        cmd[0] = sys.executable
        drop_flag(cmd, "--max-metric-calls")
        cmd += ["--max-nodes", str(max_nodes)]
        setval(cmd, "--composite", "softmin")
        cmd += ["--softmin-tau", "0.25"]
        cmd += ["--task-reasoning-json", '{"effort": "low"}']
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
                    "arm": "rexpure_softmin_tau0.25",
                    "pid": process.pid,
                    "cmd": cmd,
                    "env_pin": TASK_PIN,
                    "max_nodes": max_nodes,
                    "control": f"{src} (hard min, old 2000-metric-call budget)",
                },
                indent=2,
            )
            + "\n"
        )
        (out / "pid").write_text(f"{process.pid}\n")
        print(f"launched {game} pid={process.pid} max_nodes={max_nodes} -> {out}")


if __name__ == "__main__":
    main()
