"""Launch the WorldCoder program-WM arm on dq8gc, paired against the
jul30_minclick GEPA run (the current best language-WM baseline for dq8gc:
test ID 0.88, FD[exact] 0.70, $7.28 task + $0.31 reflection, 57 reflection calls).

Pairing protocol (same as launch_aug5_rexpure): transplant the DATA flags
verbatim from the paired run's launch.json (--run/--test-run/
--context-source-run/--test-context-source-run/--actions) so both arms learn
from and are tested on the identical transitions, and match the LLM budget in
the WC arm's native currency: --max-proposals = the paired run's reflection-call
count (57). Reflection model matched too (gpt-oss-120b, cerebras/groq pin).

Usage: uv run python offline_learning/launch/launch_wc_dq8gc.py [--proposals N] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parents[1]
PAIRED = ROOT / "logs/jul30_minclick/dq8gc_seed1/launch.json"
PIN = "cerebras,groq,sambanova"

DATA_FLAGS = ["--run", "--test-run", "--context-source-run",
              "--test-context-source-run", "--actions"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", type=int, default=57,
                    help="reflection-call budget (default: paired run's 57)")
    ap.add_argument("--out-name", default="aug6_wc")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    paired_cmd = list(json.loads(PAIRED.read_text())["cmd"])

    def paired_val(flag: str) -> str:
        return paired_cmd[paired_cmd.index(flag) + 1]

    out = ROOT / "logs" / args.out_name / f"dq8gc_seed{args.seed}"
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing run: {out}")

    cmd = [str(ROOT / ".venv/bin/python"), str(HERE / "worldcoder_optimize.py")]
    for flag in DATA_FLAGS:
        cmd += [flag, paired_val(flag)]
    if "--collapse-action-params" in paired_cmd:
        cmd += ["--collapse-action-params"]
    cmd += [
        "--context-k", paired_val("--context-k"),
        "--test-n", paired_val("--test-n"),
        "--seed", str(args.seed),
        "--max-proposals", str(args.proposals),
        "--reflection-model", "openai/gpt-oss-120b",
        "--reflection-provider-order", PIN,
        "--out-dir", str(out),
    ]

    print(" \\\n  ".join(cmd))
    if args.dry_run:
        return
    out.mkdir(parents=True)
    (out / "launch.json").write_text(json.dumps(
        {"cmd": cmd, "paired": str(PAIRED), "ts": time.strftime("%F %T")}, indent=2))
    with (out / "stdout.txt").open("w") as stdout:
        process = subprocess.Popen(
            cmd, cwd=ROOT, stdout=stdout, stderr=subprocess.STDOUT,
            env=dict(os.environ), start_new_session=True)
    (out / "pid").write_text(str(process.pid))
    print(f"[launched] pid={process.pid} -> {out}/stdout.txt")


if __name__ == "__main__":
    main()
