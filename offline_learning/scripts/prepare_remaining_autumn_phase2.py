"""Prepare old-test augmentation and intact-drive test manifests for Autumn phase 2."""

from __future__ import annotations

import os as _bos, sys as _bsys  # offline_learning/ on sys.path (flat-import the kept libs)
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from validate import load_transitions
from validate_beliefs import balanced_split


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "offline_learning"
DATA = HERE / "clean_data3"
OLD_RUNS = ROOT / "logs/clean_sweep_gepa_cd3_phase2"
OUTPUT = ROOT / "logs/fulltraj_context_remaining_autumn_phase2_inputs"

CONFIG = {
    "83wkq": {"whitelist": {"noop", "click"}, "collapse": False},
    "27vwc": {"whitelist": {"noop", "click"}, "collapse": False},
    "qqm74": {
        "whitelist": {"left", "right", "up", "down", "noop", "click"},
        "collapse": True,
    },
    "eahcw": {
        "whitelist": {"left", "right", "up", "down", "noop", "click"},
        "collapse": True,
    },
    "s2kt7": {"whitelist": {"noop", "click"}, "collapse": False},
    "ice": {
        "whitelist": {"left", "right", "up", "down", "noop", "click"},
        "collapse": True,
    },
}

# These test50 directories are already complete independent drives rather than
# slices from longer drives, so their own full rows are the context source.
INTACT_TEST_DRIVES = ("83wkq", "qqm74", "eahcw")


def collapsed(action: str, enabled: bool) -> str:
    action = action.strip()
    return action.split()[0] if enabled else action


def csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_pair(path: Path, row: dict, next_row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerows([row, next_row])


def prepare_old_test10(game: str) -> None:
    config = CONFIG[game]
    trace_path = OLD_RUNS / f"{game}_seed1/test_trace_raw_seed1.json"
    trace = json.loads(trace_path.read_text())
    source_files = sorted((DATA / game / "test").glob("episode_*/trajectory.csv"))
    candidates = []
    for source in source_files:
        rows = csv_rows(source)
        for row_index, (row, next_row) in enumerate(zip(rows, rows[1:])):
            action = (row.get("Action") or "").strip()
            if not action or action.split()[0] not in config["whitelist"]:
                continue
            candidates.append((source, row_index, row, next_row))

    destination = OUTPUT / game / "old_test10"
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite {destination}")
    used: set[tuple[Path, int]] = set()
    selected = []
    for trace_index, record in enumerate(trace["records"]):
        truth = record["truth"]
        matches = []
        for source, row_index, row, next_row in candidates:
            if (source, row_index) in used:
                continue
            if collapsed(row.get("Action") or "", config["collapse"]) != truth:
                continue
            if (row.get("Observation") or "") != record["z_t"]:
                continue
            if (next_row.get("Observation") or "") != record["z_t1"]:
                continue
            matches.append((source, row_index, row, next_row))
        if len(matches) != 1:
            raise RuntimeError(
                f"{game} trace item {trace_index}: expected one match, found {len(matches)}"
            )
        source, row_index, row, next_row = matches[0]
        used.add((source, row_index))
        target = destination / f"episode_{trace_index}/trajectory.csv"
        write_pair(target, row, next_row)
        selected.append(
            {
                "trace_index": trace_index,
                "action": (row.get("Action") or "").strip(),
                "source": str(source.relative_to(ROOT)),
                "row": row_index,
            }
        )
    if len(selected) != 10:
        raise RuntimeError(f"{game}: expected 10 old test items, found {len(selected)}")
    manifest = {"game": game, "trace": str(trace_path.relative_to(ROOT)), "selected": selected}
    (OUTPUT / game / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n")


def prepare_eahcw_train20() -> None:
    game = "eahcw"
    source = DATA / game / "train"
    destination = OUTPUT / game / "train20"
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite {destination}")
    transitions = load_transitions([source], CONFIG[game]["whitelist"], context_k=0)
    if len(transitions) != 30:
        raise RuntimeError(f"expected 30 eahcw train targets, found {len(transitions)}")
    for transition in transitions:
        transition.action = transition.action.split()[0]
    _, selected = balanced_split(transitions, 20, 10**9, random.Random(1))
    wanted = Counter((t.x_t, t.action, t.x_t1) for t in selected)
    matches = []
    for source_csv in sorted(source.glob("episode_*/trajectory.csv")):
        rows = csv_rows(source_csv)
        for row_index, (row, next_row) in enumerate(zip(rows, rows[1:])):
            key = (
                row.get("Observation") or "",
                collapsed(row.get("Action") or "", True),
                next_row.get("Observation") or "",
            )
            if wanted[key] > 0:
                wanted[key] -= 1
                matches.append((source_csv, row_index, row, next_row))
    if len(matches) != 20 or sum(wanted.values()):
        raise RuntimeError("failed to materialize the selected eahcw train20 set")
    records = []
    for index, (source_csv, row_index, row, next_row) in enumerate(matches):
        target = destination / f"episode_{index}/trajectory.csv"
        write_pair(target, row, next_row)
        records.append(
            {
                "target": str(target.relative_to(ROOT)),
                "source": str(source_csv.relative_to(ROOT)),
                "source_row": row_index,
                "action": collapsed(row.get("Action") or "", True),
            }
        )
    manifest = {
        "seed": 1,
        "method": "balanced_split holdout selection after action collapse",
        "source_transition_count": 30,
        "selected_transition_count": 20,
        "action_counts": dict(sorted(Counter(r["action"] for r in records).items())),
        "records": records,
    }
    (OUTPUT / game / "train20_selection.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def prepare_intact_test_drives(game: str) -> None:
    entries = []
    episodes = sorted(
        (DATA / game / "test50").glob("episode_*"),
        key=lambda path: int(path.name.split("_")[1]),
    )
    for episode_index, source_episode in enumerate(episodes):
        root = OUTPUT / game / "test50_full_drives" / f"drive_{episode_index}"
        destination = root / "episode_0/trajectory.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_episode / "trajectory.csv", destination)
        entries.append(
            {
                "curated_episode": episode_index,
                "drive": f"drive_{episode_index}",
                "target_root": str(root.relative_to(ROOT)),
                "source_root": str(root.relative_to(ROOT)),
            }
        )
    (OUTPUT / game / "test50_context_manifest.json").write_text(
        json.dumps(entries, indent=2) + "\n"
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for game in CONFIG:
        prepare_old_test10(game)
    prepare_eahcw_train20()
    for game in INTACT_TEST_DRIVES:
        prepare_intact_test_drives(game)
    print(
        json.dumps(
            {
                "old_test10": list(CONFIG),
                "train20": ["eahcw"],
                "intact_test50": list(INTACT_TEST_DRIVES),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
