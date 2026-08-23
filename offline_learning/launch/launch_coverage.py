"""Drive the whole coverage-exam pipeline end to end.

Three stages, each idempotent:
  1. build     coverage_exam.py build  -> human_data/<game>/coverage/   (replays human
               drives, classifies every transition into a core-mechanic bucket, selects
               K per populated bucket + synthesizes empty ones)
  2. protocol  id_protocol.py --spec coverage -> <proto-dir>/<game>_coverage.json
               (engine-verifies s_true + ceiling, attaches the mechanic label per item)
  3. score     score_coverage.py -> per-mechanic ID for raw / wc / lmwm (LLM arms)

Stages 1-2 are CPU-bound engine replay (no API calls); stage 3 makes the model calls.
Pass --stage to run a subset.

    uv run python offline_learning/launch/launch_coverage.py --scan-drives 20
    uv run python offline_learning/launch/launch_coverage.py --stage score --normalise
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GAMES = ["bt3gb", "dq8gc", "n2ntd", "s2kt7", "83wkq"]
PROTO_DIR = ROOT / "logs/2026-08-11/human_unified/coverage_protocols"
PY = sys.executable


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default=",".join(GAMES))
    ap.add_argument("--stage", choices=["build", "protocol", "score", "all"], default="all")
    ap.add_argument("--scan-drives", type=int, default=20)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--normalise", action="store_true")
    ap.add_argument("--arms", default="raw,wc,lmwm")
    args = ap.parse_args()
    games = args.games.split(",")

    if args.stage in ("build", "all"):
        for g in games:
            run([PY, "offline_learning/coverage_exam.py", "build", "--game", g,
                 "--scan-drives", str(args.scan_drives), "--k", str(args.k)])
    if args.stage in ("protocol", "all"):
        for g in games:
            run([PY, "offline_learning/id_protocol.py", "--spec", "coverage",
                 "--game", g, "--out", str(PROTO_DIR)])
    if args.stage in ("score", "all"):
        cmd = [PY, "offline_learning/scripts/score_coverage.py", "--all",
               "--protocol-dir", str(PROTO_DIR), "--arms", args.arms,
               "--out", str(ROOT / "logs/2026-08-11/human_unified/coverage_scores.json")]
        if args.normalise:
            cmd.append("--normalise")
        run(cmd)


if __name__ == "__main__":
    main()
