#!/usr/bin/env python3
"""Build one filmstrip visualization for the canonical curated offline plan eval.

The raw/NLWM plans and the WorldCoder plans live in separate evaluation artifacts but
share the same 200 windows.  This script verifies that alignment, reconstructs every
source drive, executes gold/raw/NLWM/WorldCoder plans in the Autumn engine, and embeds
the resulting grids in a self-contained filterable HTML page.

    uv run python offline_learning/scripts/build_curated_plan_viz.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for path in (REPO, REPO / "offline_learning", HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_multistep_fd_plan import (  # noqa: E402
    SeqSim,
    generate_drive,
    grids_equal,
)
from offline_learning.human_replay import GAMES as HGAMES  # noqa: E402
from viz_coverage_plan import CSS, HTML  # noqa: E402


def result_map(payload: dict) -> dict[str, dict]:
    return {result["game"]: result for result in payload["results"]}


def window_signature(window: dict) -> tuple:
    return (
        window["drive"], window["t"], window["h"], tuple(window["actions"]),
        window["start_grid"], window["goal_grid"],
    )


def reconstruct_sims(game: str, result: dict, config: dict) -> dict[str, SeqSim]:
    """Recreate the exact recorded or seeded generated drives used by the eval."""
    if result.get("drive_source") == "generated":
        base_seed = int(result.get("env_seed") or config["gen_seed"])
        rng = random.Random(f"gendrive:{base_seed}:{game}")
        sims = []
        for label in result["drives"]:
            seed = int(label.removeprefix("gen_seed"))
            rows = generate_drive(game, seed, int(config["drive_length"]), rng)
            sims.append(SeqSim(game, rows=rows, seed=seed, label=label))
    else:
        sims = [SeqSim(game, drive_csv=Path(drive)) for drive in result["drives"]]
    return {str(sim.drive_csv): sim for sim in sims}


def drive_label(drive: str) -> str:
    if drive.startswith("gen_seed"):
        return drive
    parts = Path(drive).parts
    return "/".join(parts[-3:-1]) if len(parts) >= 3 else drive


def encode(raw_payload: dict, wc_payload: dict) -> tuple[dict, list[str]]:
    raw_results, wc_results = result_map(raw_payload), result_map(wc_payload)
    if set(raw_results) != set(wc_results):
        raise ValueError(
            f"game mismatch: raw/NLWM={sorted(raw_results)}, WC={sorted(wc_results)}"
        )

    palette_index: dict[str, int] = {}

    def enc_grid(grid_json: str | None):
        if grid_json is None:
            return None
        grid = json.loads(grid_json)
        out = []
        for row in grid:
            encoded_row = []
            for color in row:
                if color not in palette_index:
                    palette_index[color] = len(palette_index)
                encoded_row.append(palette_index[color])
            out.append(encoded_row)
        return out

    problems, mismatches = [], []
    game_order = ["bt3gb", "dq8gc", "n2ntd", "83wkq", "s2kt7"]
    for game in game_order:
        raw_result, wc_result = raw_results[game], wc_results[game]
        raw_windows, wc_windows = raw_result["windows"], wc_result["windows"]
        if [window_signature(w) for w in raw_windows] != [
            window_signature(w) for w in wc_windows
        ]:
            raise ValueError(f"{game}: raw/NLWM and WorldCoder windows are not aligned")

        sims = reconstruct_sims(game, raw_result, raw_payload["config"])
        raw_rows = {
            (row["window"], row["mode"]): row for row in raw_result["plan_rows"]
        }
        wc_rows = {
            (row["window"], row["mode"]): row for row in wc_result["plan_rows"]
        }

        def execute(sim: SeqSim, window: dict, row: dict | None) -> dict:
            plan = row.get("plan") if row else None
            grids = sim.run(window["t"], plan) if plan else None
            reached = next(
                (step for step, grid in enumerate(grids or [], 1)
                 if grids_equal(grid, window["goal_grid"])),
                None,
            )
            success = bool(grids and grids_equal(grids[-1], window["goal_grid"]))
            if row is not None and success != bool(row.get("success")):
                mismatches.append(
                    f"{game} window {row['window']} {row['mode']}: "
                    f"saved={bool(row.get('success'))}, replayed={success}"
                )
            return {
                "success": success,
                "reached_at": reached,
                "plan": plan,
                "plan_error": row.get("plan_error") if row else "missing-plan-row",
                "grids": [enc_grid(grid) for grid in grids] if grids else None,
            }

        for wi, window in enumerate(raw_windows):
            sim = sims.get(window["drive"])
            if sim is None:
                raise KeyError(f"{game}: no reconstructed drive {window['drive']!r}")
            gold_plan = [action for action in window["actions"]]
            gold_grids = sim.run(window["t"], gold_plan)
            if not gold_grids or not grids_equal(gold_grids[-1], window["goal_grid"]):
                raise ValueError(f"{game} window {wi}: gold plan failed replay")
            gold_reached = next(
                step for step, grid in enumerate(gold_grids, 1)
                if grids_equal(grid, window["goal_grid"])
            )
            arms = {
                "correct": {
                    "success": True,
                    "reached_at": gold_reached,
                    "plan": gold_plan,
                    "plan_error": None,
                    "grids": [enc_grid(grid) for grid in gold_grids],
                },
                "raw": execute(sim, window, raw_rows.get((wi, "raw"))),
                "wc": execute(sim, window, wc_rows.get((wi, "program"))),
                "lmwm": execute(sim, window, raw_rows.get((wi, "learned"))),
            }
            problems.append({
                "game": game,
                "human": HGAMES[game][1],
                "bucket": raw_result.get("drive_source", "unknown"),
                "mechanic": drive_label(window["drive"]),
                "h": window["h"],
                "seed": raw_result.get("env_seed"),
                "t": window["t"],
                "gt": gold_plan,
                "noop": None,
                "rand": window.get("random_success"),
                "start": enc_grid(window["start_grid"]),
                "goal": enc_grid(window["goal_grid"]),
                "arms": arms,
            })

    palette = {i: CSS.get(name, name) for name, i in palette_index.items()}
    return {"palette": palette, "problems": problems}, mismatches


def render(payload: dict, raw_path: Path, wc_path: Path) -> str:
    html = HTML.replace(
        "Coverage planning — plan filmstrips",
        "Curated offline planning eval — plan filmstrips",
    )
    start = html.index('<p class="sub">')
    end = html.index("</p>", start) + len("</p>")
    description = (
        '<p class="sub">Canonical curated offline evaluation: 5 games × 4 horizons '
        '× 10 windows. Each plan is replayed in the Autumn engine. '
        '<span class="mono">correct</span> is the logged action sequence; '
        '<span class="mono">raw</span> sees raw grids; '
        '<span class="mono">wc</span> is WorldCoder search; '
        '<span class="mono">nlwm</span> uses learned perception and beliefs. '
        f'Sources: <span class="mono">{raw_path}</span> and '
        f'<span class="mono">{wc_path}</span>.</p>'
    )
    html = html[:start] + description + html[end:]
    html = html.replace("<label>bucket</label>", "<label>source</label>")
    html = html.replace("<label>mechanic</label>", "<label>drive</label>")
    html = html.replace(
        '["","h="+p.h],["","noop "+fmt(p.noop)],["","rand "+fmt(p.rand)]',
        '["","h="+p.h],["","t="+p.t],["","rand floor "+fmt(p.rand)]',
    )
    return html.replace("/*DATA*/{}", json.dumps(payload, separators=(",", ":")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-nlwm",
        type=Path,
        default=REPO / "logs/batch3_consolidated/multistep_batch3.json",
    )
    parser.add_argument(
        "--worldcoder",
        type=Path,
        default=REPO / "logs/wc_seed1_consolidated/multistep_wc.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "logs/curated_offline_plan_viz.html",
    )
    args = parser.parse_args()

    raw_payload = json.loads(args.raw_nlwm.read_text())
    wc_payload = json.loads(args.worldcoder.read_text())
    payload, mismatches = encode(raw_payload, wc_payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(payload, args.raw_nlwm, args.worldcoder))
    print(
        f"{len(payload['problems'])} tasks, {len(payload['palette'])} colours -> "
        f"{args.out} ({args.out.stat().st_size / 1024 / 1024:.1f} MB)"
    )
    if mismatches:
        print(f"WARNING: {len(mismatches)} saved/replayed result mismatches")
        for mismatch in mismatches[:20]:
            print(f"  {mismatch}")


if __name__ == "__main__":
    main()
