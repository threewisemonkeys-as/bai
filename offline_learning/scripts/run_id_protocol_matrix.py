"""Score every artifact set against every protocol, then tabulate.

The 2x2 that the within-pool baselines could never settle: each learner's artifact,
trained on one data source, sat against BOTH exams.

  lmwm-human   REx-pure trained on human-origin data
  lmwm-synth   REx-pure trained on clean_data3
  wc-human     WorldCoder trained on human-origin data
  wc-synth     WorldCoder trained on clean_data3

plus three references that implement the same interface: `raw` (no perception, no
beliefs), `blind` (return all five -> exactly 1/k), `oracle` (the engine-verified set
-> the attainable ceiling).

    uv run python offline_learning/scripts/run_id_protocol_matrix.py
    uv run python offline_learning/scripts/run_id_protocol_matrix.py --report-only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTO = ROOT / "logs/aug10_human_origin/protocols"
OUT = ROOT / "logs/aug10_human_origin/id_matrix"
GAMES = ["bt3gb", "dq8gc", "n2ntd", "s2kt7", "83wkq"]
POOLS = ["human", "synthetic"]

# label -> (arm, artifact-dir template or None)
ARTIFACTS = {
    "lmwm-human": ("lmwm", "logs/aug10_human_origin/rexpure/{game}_s1"),
    "lmwm-synth": ("lmwm", "logs/batch3_consolidated/{game}_s1_batch3"),
    "wc-human":   ("wc",   "logs/aug10_human_origin/worldcoder/{game}_s1"),
    "wc-synth":   ("wc",   "logs/wc_seed1_consolidated/{game}_s1_wc"),
    "raw":        ("raw",  None),
    "blind":      ("blind", None),
    "oracle":     ("oracle", None),
}


def run_cell(game: str, pool: str, label: str, force: bool) -> Path | None:
    arm, tmpl = ARTIFACTS[label]
    proto = PROTO / f"{game}_{pool}.json"
    if not proto.exists():
        return None
    out = OUT / f"{game}_{pool}_{label}.json"
    if out.exists() and not force:
        return out
    cmd = [sys.executable, str(ROOT / "offline_learning/scripts/score_id_protocol.py"),
           "--protocol", str(proto), "--arm", arm, "--label", label, "--out", str(out)]
    if tmpl:
        d = ROOT / tmpl.format(game=game)
        if not d.exists():
            return None
        cmd += ["--artifact-dir", str(d)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAIL {game}/{pool}/{label}: {r.stderr.strip().splitlines()[-1:]}")
        return None
    print("  " + r.stdout.strip())
    return out


def report() -> None:
    res = {}
    for p in sorted(OUT.glob("*.json")):
        if p.name == "matrix.json":          # our own summary, not a scored cell
            continue
        d = json.loads(p.read_text())
        res[(d["game"], d["pool"], d["arm"])] = d
    order = ["oracle", "lmwm-human", "lmwm-synth", "wc-human", "wc-synth", "raw", "blind"]
    lines = []
    for pool in POOLS:
        lines.append(f"\n=== {pool.upper()} protocol "
                     f"(normalised = raw / that pool's engine-verified ceiling) ===")
        lines.append(f"{'arm':12s}" + "".join(f"{g:>10s}" for g in GAMES) + f"{'mean':>10s}")
        for arm in order:
            row, vals = f"{arm:12s}", []
            for g in GAMES:
                d = res.get((g, pool, arm))
                v = d["normalised"] if d else None
                row += f"{v:10.3f}" if v is not None else f"{'—':>10s}"
                if v is not None:
                    vals.append(v)
            row += f"{sum(vals)/len(vals):10.3f}" if vals else f"{'—':>10s}"
            lines.append(row)
        lines.append(f"{'ceiling':12s}" + "".join(
            f"{res[(g,pool,'oracle')]['ceiling']:10.3f}" if (g, pool, 'oracle') in res
            else f"{'—':>10s}" for g in GAMES))
    print("\n".join(lines))
    (OUT / "MATRIX.txt").write_text("\n".join(lines) + "\n")
    (OUT / "matrix.json").write_text(json.dumps(
        {f"{k[0]}|{k[1]}|{k[2]}": {x: v[x] for x in
         ("n", "raw", "ceiling", "normalised", "strict", "mean_set_size", "empty_sets")}
         for k, v in res.items()}, indent=1) + "\n")
    print(f"\n-> {OUT/'MATRIX.txt'}  {OUT/'matrix.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default=",".join(GAMES))
    ap.add_argument("--pools", default=",".join(POOLS))
    ap.add_argument("--arms", default=",".join(ARTIFACTS))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        for pool in args.pools.split(","):
            for game in args.games.split(","):
                print(f"[{game}/{pool}]")
                for label in args.arms.split(","):
                    run_cell(game, pool, label, args.force)
    report()


if __name__ == "__main__":
    main()
