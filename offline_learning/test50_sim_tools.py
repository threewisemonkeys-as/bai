"""Shared helpers for simulator-grounded test50 rescoring.

A test50 episode slice is verbatim rows of a seed-0 autumn_drive.py drive; the
Step column indexes into that drive. These helpers verify a candidate drive CSV
reproduces a slice exactly and that the drive itself replays deterministically
in the Autumn engine.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

BAI_ROOT = Path(__file__).resolve().parents[1]
if str(BAI_ROOT) not in sys.path:
    sys.path.insert(0, str(BAI_ROOT))

csv.field_size_limit(10_000_000)

ENV_NAMES = {"ice": "ice"}  # everything else is the uppercased game id


def env_name_for(game: str) -> str:
    return ENV_NAMES.get(game, game.upper())


def read_rows(csv_path: Path) -> list[dict]:
    with Path(csv_path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def grid_json(text: str) -> str:
    """Extract the first [[...]] grid from an observation cell -> canonical JSON."""
    i = text.find("[[")
    if i < 0:
        raise ValueError("no grid in observation")
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "[":
            depth += 1
        elif text[j] == "]":
            depth -= 1
            if depth == 0:
                return json.dumps(json.loads(text[i:j + 1]), separators=(",", ":"))
    raise ValueError("unbalanced grid in observation")


def match_slice_to_drive(slice_rows: list[dict], drive_rows: list[dict]) -> list[str]:
    """Return a list of mismatch descriptions (empty = the slice is verbatim rows
    of the drive at its Step indices). The final slice row's Action/Done may have
    been rewritten during slicing, so only Observation is compared there."""
    problems = []
    for k, row in enumerate(drive_rows):
        if str(row["Step"]) != str(k):
            problems.append(f"drive Step column broken at index {k}: {row['Step']}")
            return problems
    for i, srow in enumerate(slice_rows):
        t = int(srow["Step"])
        if t >= len(drive_rows):
            problems.append(f"slice step {t} beyond drive length {len(drive_rows)}")
            continue
        drow = drive_rows[t]
        if srow["Observation"] != drow["Observation"]:
            problems.append(f"observation mismatch at step {t}")
        last = i == len(slice_rows) - 1
        if not last and (srow.get("Action") or "") != (drow.get("Action") or ""):
            problems.append(
                f"action mismatch at step {t}: slice {srow.get('Action')!r} "
                f"vs drive {drow.get('Action')!r}"
            )
    return problems


def make_env(game: str, max_steps: int = 500, seed: int = 0):
    """Build an interactive Autumn env. seed=0 is the historical default but is
    PATHOLOGICAL for games using randomness (see memory seed0-degenerate-data-gen):
    randomPositions/uniformChoice collapse under seed 0. Pass a non-zero seed to run
    the real (non-degenerate) game; seed-independent (no-randomness) games are
    unaffected."""
    from autumn_env import AutumnBenchEnvWrapper
    env = AutumnBenchEnvWrapper(env_name=env_name_for(game), task_type="interactive",
                                max_episode_steps=max_steps, seed=seed,
                                render_mode="text")
    obs, _ = env.reset(seed=seed)
    return env, obs


def obs_grid(obs: dict) -> str:
    return grid_json(obs["text"]["long_term_context"])


def verify_drive_in_sim(game: str, drive_rows: list[dict]) -> list[str]:
    """Replay the drive's actions from a seed-0 reset; every recorded row's grid
    must match the engine. Returns mismatch descriptions (empty = verified)."""
    problems = []
    env, obs = make_env(game, max_steps=len(drive_rows) + 5)
    for t, row in enumerate(drive_rows):
        if obs_grid(obs) != grid_json(row["Observation"]):
            problems.append(f"engine grid diverges from recorded grid at step {t}")
            break
        action = (row.get("Action") or "").strip()
        if not action:
            break
        obs, _, term, _, _ = env.step(action)
        if term and t < len(drive_rows) - 2:
            problems.append(f"engine terminated early at step {t}")
            break
    return problems


def verify_game_sources(game: str, data_root: Path, mapping: dict[str, Path]) -> dict:
    """mapping: episode dir name -> drive trajectory.csv path. Verifies every
    episode against its drive and every drive in the simulator."""
    report = {"game": game, "episodes": {}, "drives": {}}
    for drive_csv in sorted(set(mapping.values())):
        report["drives"][str(drive_csv)] = verify_drive_in_sim(game, read_rows(drive_csv))
    for ep_name, drive_csv in sorted(mapping.items()):
        slice_rows = read_rows(data_root / game / "test50" / ep_name / "trajectory.csv")
        report["episodes"][ep_name] = match_slice_to_drive(slice_rows, read_rows(drive_csv))
    report["ok"] = not any(report["drives"].values()) and not any(report["episodes"].values())
    return report
