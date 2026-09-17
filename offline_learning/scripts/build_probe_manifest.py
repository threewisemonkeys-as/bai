#!/usr/bin/env python3
"""Freeze checkpoint identities, human-drive windows, and training-only examples."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing_common import REPO, read_json, write_json  # noqa: E402
from probing_manifest import ManifestBuilder, verify_manifest  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-root", action="append", metavar="LABEL=PATH",
                   help="repeat for additional arms; first arm gets learning checkpoints, others final only")
    p.add_argument("--include-ablations", action="store_true")
    p.add_argument("--games", default="", help="comma-separated; defaults to all reference games")
    p.add_argument("--train-seed", type=int, default=1)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--context-k", type=int, default=9)
    p.add_argument("--horizons", default="1,2,4,8")
    p.add_argument("--windows", type=int, default=50)
    p.add_argument("--reconstruction", type=int, default=100)
    p.add_argument("--examples", type=int, default=8)
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--nonoverlapping", action="store_true")
    p.add_argument("--replay", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--perception-timeout", type=float, default=2.0)
    p.add_argument("--out", type=Path, default=REPO / "analysis/probing/manifest.json.gz")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verify", type=Path, help="verify an existing manifest and all its source hashes")
    a = p.parse_args()
    if a.verify:
        verify_manifest(read_json(a.verify))
        print(f"Verified {a.verify}")
        return
    specs = a.artifact_root or ["Plain=logs/2026-08-24/human_curated"]
    if a.include_ablations:
        specs += [f"{arm}=logs/2026-09-09/ablations/{arm}"
                  for arm in ("nofd", "noid", "noperc", "nobeliefs")]
    roots = []
    for spec in specs:
        label, sep, value = spec.partition("=")
        if not sep or not label or not value:
            p.error("artifact roots must be LABEL=PATH")
        path = Path(value)
        roots.append((label, path if path.is_absolute() else REPO / path))
    suffix = f"_s{a.train_seed}"
    available = sorted(d.name[:-len(suffix)] for d in (roots[0][1] / "rexpure").glob(f"*{suffix}")
                       if d.is_dir())
    games = a.games.split(",") if a.games else available
    if not games or set(games) - set(available):
        p.error("games must exist in the reference artifact root")
    builder = ManifestBuilder(cache_dir=a.out.parent / "perception_cache", replay=a.replay,
        context_k=a.context_k, horizons=[int(h) for h in a.horizons.split(",")],
        windows=a.windows, reconstruction=a.reconstruction, examples=a.examples,
        sample_seed=a.sample_seed, split=a.split, nonoverlapping=a.nonoverlapping,
        perception_timeout=a.perception_timeout)
    manifest = builder.build(roots, games, a.train_seed)
    if a.out.exists() and not a.overwrite:
        old = read_json(a.out)
        if old.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise SystemExit("output contains a different manifest; use a new path or --overwrite")
    write_json(a.out, manifest)
    print(f"Wrote {a.out}: {len(games)} games, {len(roots)} arms, "
          f"sha256={manifest['manifest_sha256']}")


if __name__ == "__main__":
    main()
