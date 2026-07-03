"""Generate optim_viz.html for FINISHED games in the clean_data3 phase-1 sweep
(logs/clean_sweep_gepa_cd3_phase1), reading split args from clean_sweep.GAMES and using the
clean_data3 train/test dirs. 7xf97 ran at train-n 24 (its enriched pool); all others at 20.

Idempotent: only builds for games whose stdout.txt has the final HEAD-TO-HEAD table and that
lack an optim_viz.html. Safe to call repeatedly (e.g. from a watcher poll).

    uv run python prototypes/perc_invdyn/gen_optim_viz_cd3.py [--force] [--games a,b]
"""
from __future__ import annotations
import argparse, importlib.util, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA3 = HERE / "clean_data3"
ROOT = HERE.parent.parent / "logs" / "clean_sweep_gepa_cd3_phase1"
spec = importlib.util.spec_from_file_location("clean_sweep", HERE / "clean_sweep.py")
cs = importlib.util.module_from_spec(spec); spec.loader.exec_module(cs)

PHASE1 = "qfsvc 7www9 bt3gb 7xf97 dgg2c dq8gc e3v6m f5w3n n2ntd nrdf6 vqjh6".split()
TRAIN_N = {"7xf97": "24"}  # per-game overrides; default 20


def finished(rd: Path) -> bool:
    s = rd / "stdout.txt"
    return s.exists() and "HEAD-TO-HEAD" in s.read_text(errors="replace")


def build_one(g: str, force: bool) -> str:
    rd = ROOT / f"{g}_seed1"
    gepa = rd / "gepa_run_seed1"
    out = rd / "optim_viz.html"
    if not finished(rd):
        return f"PENDING   {g}"
    if not gepa.exists():
        return f"NOGEPA    {g}"
    if out.exists() and not force:
        return f"skip      {g} (exists)"
    wl, keep, *_ = cs.GAMES[g]
    cmd = [sys.executable, str(HERE / "build_optim_viz.py"), "--gepa-dir", str(gepa),
           "--run", str(DATA3 / g / "train"), "--test-run", str(DATA3 / g / "test"),
           "--seed", "1", "--train-n", TRAIN_N.get(g, "20"), "--test-n", "10",
           "--tie-train-val", "--context-k", "9", "--actions", wl, "--out", str(out)]
    if not keep:
        cmd.append("--collapse-action-params")
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        tail = (p.stderr.strip().splitlines() or ["rc!=0"])[-1]
        return f"ERROR     {g}: {tail}"
    ok = any("alignment check" in l and "ok=True" in l for l in p.stdout.splitlines())
    return f"GENERATED {g} ({'aligned' if ok else 'ALIGN-MISMATCH'})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default=",".join(PHASE1))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for g in (x for x in args.games.split(",") if x.strip()):
        print(build_one(g, args.force), flush=True)


if __name__ == "__main__":
    main()
