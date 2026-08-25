"""Install Autumn programs into the harness directory the env wrappers read.

`autumn_env.py`, `offline_learning/curated_plan.py` (PROGRAMS), `game_profile.py` and
`autumn_oracle.py` all load `<name>.sexp` from
`MARAProtocol/python_examples/autumnbench/example_benchmark/programs/`.  That directory is
gitignored inside the MARAProtocol submodule (only the benchmark download and the
force-added `ice` files live there), so any program copied in by hand disappears on a
dataset re-download.  The tracked copies live in `autumn_programs/` at the repo root
(the 55 sources originally distributed as autumn_programs_55.zip); this script copies the ones we use.

    uv run tools/install_autumn_programs.py              # the SELECTED zip-sourced games
    uv run tools/install_autumn_programs.py rink dino    # named programs
    uv run tools/install_autumn_programs.py --all        # all 55 (adds duplicates of the
                                                         #   benchmark worlds under their names)
    uv run tools/install_autumn_programs.py --check      # report only; exit 1 on drift

An existing file with DIFFERENT content is never overwritten without --force: the
benchmark ids (BT3GB, DQ8GC, ...) and `ice` are the scored copies and `run.py` guards
`ice` by content hash.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "autumn_programs"
DST = ROOT / "MARAProtocol/python_examples/autumnbench/example_benchmark/programs"

# Games in experimental_plan.md that come from the zip rather than the benchmark download.
# the zip-sourced games of the 2026-08-23 selection in experimental_plan.md (rink, balloon,
# tetris were dropped from the selection; still installable by name or --all)
SELECTED = ["logic_gates", "colour_lines", "diffusion", "dino", "SET", "egg"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="program names (default: SELECTED)")
    ap.add_argument("--all", action="store_true", help="install every source in autumn_programs/")
    ap.add_argument("--check", action="store_true", help="report only, do not write")
    ap.add_argument("--force", action="store_true", help="overwrite files whose content differs")
    a = ap.parse_args()

    if a.all:
        names = sorted(p.stem for p in SRC.glob("*.sexp"))
    else:
        names = a.names or SELECTED
    if not DST.is_dir():
        print(f"missing harness dir {DST} (is the MARAProtocol submodule checked out?)")
        return 1

    drift = 0
    for n in names:
        src = SRC / f"{n}.sexp"
        dst = DST / f"{n}.sexp"
        if not src.is_file():
            print(f"  ??  {n}: no source in {SRC}")
            drift += 1
            continue
        body = src.read_bytes()
        if dst.is_file():
            if dst.read_bytes() == body:
                print(f"  ok  {n}")
                continue
            if not a.force:
                print(f"  !!  {n}: installed copy differs from autumn_programs/ (use --force to overwrite)")
                drift += 1
                continue
            state = "upd"
        else:
            state = "new"
        if a.check:
            print(f"  --  {n}: would install ({state})")
            drift += 1
            continue
        dst.write_bytes(body)
        print(f"  {state} {n}")
    return 1 if (a.check and drift) else 0


if __name__ == "__main__":
    sys.exit(main())
