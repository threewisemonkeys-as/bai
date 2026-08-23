"""Verify reconstructed phase-2 drives and write per-episode context manifests."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "offline_learning/clean_data3"
OUTPUT = ROOT / "logs/fulltraj_context_remaining_autumn_phase2_inputs"


def rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def pairs(path: Path) -> set[tuple[str, str, str]]:
    data = rows(path)
    return {
        (
            row.get("Observation") or "",
            (row.get("Action") or "").strip(),
            next_row.get("Observation") or "",
        )
        for row, next_row in zip(data, data[1:])
        if (row.get("Action") or "").strip()
    }


def write_rows(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def merge_at_shared_boundary(game: str, first: int, second: int, drive: str) -> None:
    first_rows = rows(DATA / game / f"test50/episode_{first}/trajectory.csv")
    second_rows = rows(DATA / game / f"test50/episode_{second}/trajectory.csv")
    if first_rows[-1] != second_rows[0]:
        raise RuntimeError(f"{game} episodes {first}/{second} do not share an exact boundary")
    destination = OUTPUT / game / f"test50_sources/drive{drive}/episode_0/trajectory.csv"
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite {destination}")
    write_rows(destination, first_rows + second_rows[1:])


def build_manifest(game: str, mapping: list[str]) -> dict:
    entries = []
    source_pairs = {
        drive: pairs(
            OUTPUT / game / f"test50_sources/drive{drive}/episode_0/trajectory.csv"
        )
        for drive in sorted(set(mapping))
    }
    matched = 0
    for episode, drive in enumerate(mapping):
        source_csv = DATA / game / f"test50/episode_{episode}/trajectory.csv"
        target_pairs = pairs(source_csv)
        overlap = target_pairs & source_pairs[drive]
        if overlap != target_pairs:
            raise RuntimeError(
                f"{game} episode_{episode} -> drive{drive}: "
                f"matched {len(overlap)}/{len(target_pairs)} exact pairs"
            )
        matched += len(target_pairs)
        target_root = OUTPUT / game / f"test50_targets_by_episode/episode_{episode}"
        target_csv = target_root / "episode_0/trajectory.csv"
        target_csv.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_csv, target_csv)
        entries.append(
            {
                "curated_episode": episode,
                "drive": drive,
                "target_root": str(target_root.relative_to(ROOT)),
                "source_root": str(
                    (OUTPUT / game / f"test50_sources/drive{drive}").relative_to(ROOT)
                ),
            }
        )
    manifest = OUTPUT / game / "test50_context_manifest.json"
    if manifest.exists():
        raise RuntimeError(f"refusing to overwrite {manifest}")
    manifest.write_text(json.dumps(entries, indent=2) + "\n")
    return {"game": game, "episodes": len(entries), "exact_target_pairs": matched}


def main() -> None:
    merge_at_shared_boundary("s2kt7", 0, 1, "T1")
    merge_at_shared_boundary("s2kt7", 2, 3, "T2")
    results = [
        build_manifest("s2kt7", ["T1", "T1", "T2", "T2", "T3", "T3", "T4", "T4"]),
        build_manifest("ice", ["A", "A", "A", "A", "B", "B", "B", "C", "D", "E", "E", "F"]),
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
