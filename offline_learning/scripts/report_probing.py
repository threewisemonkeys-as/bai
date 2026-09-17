#!/usr/bin/env python3
"""Report completed, failed and pending probe results without hiding missing coverage."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing_report import report  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--bootstrap-draws", type=int, default=2000)
    a = p.parse_args()
    if a.bootstrap_draws < 0:
        p.error("bootstrap draws must be nonnegative")
    result = report(a.run, plots=a.plots, draws=a.bootstrap_draws)
    print(f"Wrote {a.run / 'REPORT.md'}; coverage={result['coverage']}")


if __name__ == "__main__":
    main()
