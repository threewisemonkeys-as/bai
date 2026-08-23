"""Phase-3 WorldCoder sweep: the 5-game A/B set at per-game budget parity.

Pairing = the artifact-of-record behind logs/id_eval_test50_raw_vs_learned.json:

  dq8gc  fulltraj_phase1 (launch.json transplant)   55 reflection calls
  n2ntd  fulltraj_phase1 (launch.json transplant)   52
  bt3gb  clean_sweep_gepa_cd3_phase1 (plain)        50
  s2kt7  clean_sweep_gepa_cd3_phase2 (plain)        53
  83wkq  clean_sweep_gepa_cd3_phase2 (plain)        51

Transplanted runs reuse the paired run's exact --run/--test-run/--context-source
args (stale prototypes/perc_invdyn prefixes remapped to offline_learning/);
plain runs use clean_data3/<game>/{train,test50} with clean_sweep.GAMES action
flags -- the same data those artifacts trained/tested on. Reflection model =
deepseek/deepseek-v4-flash (all five paired runs), --max-proposals = the paired
run's reflection-call count. All runs seed 1, context_k 9, test_n 50.

Usage: uv run python offline_learning/launch/launch_wc_sweep.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE))
from clean_sweep import GAMES  # noqa: E402

OUT_ROOT = ROOT / "logs/aug7_wc_sweep"
DATA_FLAGS = ["--run", "--test-run", "--context-source-run",
              "--test-context-source-run", "--actions"]
STALE = "/prototypes/perc_invdyn/"
FRESH = "/offline_learning/"

SWEEP = {
    "dq8gc": {"paired": ROOT / "logs/clean_sweep_gepa_cd3_fulltraj_aug30_mb15_test50_fulltestctx_phase1/dq8gc_seed1/launch.json",
              "proposals": 55},
    "n2ntd": {"paired": ROOT / "logs/clean_sweep_gepa_cd3_fulltraj_aug30_mb15_test50_fulltestctx_phase1/n2ntd_seed1/launch.json",
              "proposals": 52},
    "bt3gb": {"paired": None, "proposals": 50},
    "s2kt7": {"paired": None, "proposals": 53},
    "83wkq": {"paired": None, "proposals": 51},
}


def remap(p: str) -> str:
    return p.replace(STALE, FRESH)


def build_cmd(game: str, spec: dict, out: Path) -> list[str]:
    cmd = [str(ROOT / ".venv/bin/python"), str(CORE / "worldcoder_optimize.py")]
    if spec["paired"]:
        paired_cmd = list(json.loads(spec["paired"].read_text())["cmd"])
        for flag in DATA_FLAGS:
            cmd += [flag, remap(paired_cmd[paired_cmd.index(flag) + 1])]
        if "--collapse-action-params" in paired_cmd:
            cmd += ["--collapse-action-params"]
    else:
        actions, keep, _mmc, _r = GAMES[game]
        cmd += ["--run", str(CORE / f"clean_data3/{game}/train"),
                "--test-run", str(CORE / f"clean_data3/{game}/test50"),
                "--actions", actions]
        if not keep:
            cmd += ["--collapse-action-params"]
    cmd += ["--context-k", "9", "--test-n", "50", "--seed", "1",
            "--max-proposals", str(spec["proposals"]),
            "--reflection-model", "deepseek/deepseek-v4-flash",
            "--out-dir", str(out)]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default=",".join(SWEEP))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    games = [g for g in args.games.split(",") if g]

    for game in games:
        out = OUT_ROOT / f"{game}_seed1"
        if out.exists():
            raise SystemExit(f"refusing to overwrite existing run: {out}")

    procs = []
    for game in games:
        spec = SWEEP[game]
        out = OUT_ROOT / f"{game}_seed1"
        cmd = build_cmd(game, spec, out)
        print(f"[{game}] proposals={spec['proposals']} "
              f"paired={'transplant' if spec['paired'] else 'plain clean_data3'}")
        if args.dry_run:
            print("  " + " ".join(cmd[:6]) + " ...")
            continue
        out.mkdir(parents=True)
        (out / "launch.json").write_text(json.dumps(
            {"cmd": cmd, "paired": str(spec["paired"]), "ts": time.strftime("%F %T")},
            indent=2))
        with (out / "stdout.txt").open("w") as stdout:
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=stdout,
                                 stderr=subprocess.STDOUT,
                                 env=dict(os.environ), start_new_session=True)
        (out / "pid").write_text(str(p.pid))
        procs.append((game, p.pid))
    if procs:
        print(f"[launched] {procs} -> {OUT_ROOT}/<game>_seed1/stdout.txt")


if __name__ == "__main__":
    main()
