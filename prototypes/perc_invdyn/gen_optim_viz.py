"""Regenerate optim_viz.html for every FINISHED game in the padded ctxk9 cross-traj sweep.

Idempotent: only builds a viz for games whose run is complete (stdout.txt contains the final
HEAD-TO-HEAD table) and that don't already have an up-to-date optim_viz.html. Per-game split
args (whitelist + collapse) are read straight from clean_sweep.GAMES so they always match the
sweep. Safe to call repeatedly (e.g. from a watcher) -- prints GENERATED <game> for each new one.

    uv run python prototypes/perc_invdyn/gen_optim_viz.py            # all finished games
    uv run python prototypes/perc_invdyn/gen_optim_viz.py --force    # rebuild even if present
    uv run python prototypes/perc_invdyn/gen_optim_viz.py --games dq8gc,ice
"""
from __future__ import annotations
import argparse, importlib.util, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA2 = HERE / "clean_data2"
ROOT = HERE.parent.parent / "logs" / "clean_sweep_gepa_padded_ctxk9"
spec = importlib.util.spec_from_file_location("clean_sweep", HERE / "clean_sweep.py")
cs = importlib.util.module_from_spec(spec); spec.loader.exec_module(cs)


def finished(rundir: Path) -> bool:
    """A game is fully done once clean_sweep has flushed its captured stdout with the final
    HEAD-TO-HEAD table (written only after the gepa subprocess exits)."""
    s = rundir / "stdout.txt"
    return s.exists() and "HEAD-TO-HEAD" in s.read_text(errors="replace")


def build_one(game: str, force: bool) -> str:
    wl, keep, *_ = cs.GAMES[game]
    rundir = ROOT / f"{game}_seed1"
    gepa = rundir / "gepa_run_seed1"
    out = rundir / "optim_viz.html"
    if not finished(rundir):
        return f"PENDING  {game}"
    if not gepa.exists():
        return f"NOGEPA   {game}"
    if out.exists() and not force:
        return f"skip     {game} (exists)"
    cmd = [sys.executable, str(HERE / "build_optim_viz.py"),
           "--gepa-dir", str(gepa),
           "--run", str(DATA2 / game / "train"), "--test-run", str(DATA2 / game / "test"),
           "--seed", "1", "--train-n", "20", "--test-n", "10", "--tie-train-val",
           "--context-k", "9", "--actions", wl, "--out", str(out)]
    if not keep:
        cmd.append("--collapse-action-params")
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return f"ERROR    {game}: {p.stderr.strip().splitlines()[-1] if p.stderr.strip() else 'rc!=0'}"
    okline = next((l for l in p.stdout.splitlines() if "alignment check" in l), "")
    ok = "ok=True" in okline
    return f"GENERATED {game} ({'aligned' if ok else 'ALIGN-MISMATCH'})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default=cs.DEFAULT_GAMES)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for g in (x for x in args.games.split(",") if x.strip()):
        print(build_one(g, args.force), flush=True)


if __name__ == "__main__":
    main()
