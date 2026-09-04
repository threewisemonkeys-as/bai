#!/usr/bin/env python3
"""Keep the planning-eval replay pages level with a run that is still playing.

    uv run python offline_learning/launch/watch_plan_replay.py \
        logs/2026-09-01/planning_v2_online_ds_nl

Rendering a game costs a few seconds and the combined page tens, so this runs only when
the answer would change: the fingerprint is each `<game>/online.json`'s size and mtime,
which the evaluator rewrites every `--emit-every` rollouts. A game whose file has not
moved is not re-rendered, so a long run pays for the game in flight and nothing else.

It stops when the run does -- the launcher gone from the process list and no file moving
any more -- rather than idling on a directory nothing will touch again, and a wall-clock
budget ends it regardless. Reading a live run is safe: it only reads, and the evaluator
writes `online.json` whole, so a partial read is not possible.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
VIZ = REPO / "offline_learning/scripts/viz_plan_replay.py"

sys.path.insert(0, str(HERE))
from launch_planning_v2_online import GAME_ORDER  # noqa: E402

sys.path.insert(0, str(REPO / "offline_learning/scripts"))
from viz_plan_replay import human_name  # noqa: E402


def fingerprint(root: Path) -> dict[str, tuple[int, int]]:
    """What each game's results look like right now: (size, mtime) per online.json.

    Cheap, and it moves exactly when a rollout lands -- the evaluator re-emits the whole
    file every few completions, so a changed stat is a changed page and an unchanged one
    means there is nothing to draw.
    """
    out = {}
    for f in sorted(root.glob("*/online.json")):
        try:
            st = f.stat()
        except FileNotFoundError:             # a game's directory appearing mid-scan
            continue
        out[f.parent.name] = (st.st_size, int(st.st_mtime))
    return out


def launcher_running(root: Path) -> bool:
    """Is a launcher or evaluator still working on this run root?"""
    try:
        ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True,
                            timeout=20)
    except (OSError, subprocess.SubprocessError):
        return True                           # cannot tell -> assume it is alive
    needle = str(root)
    for line in ps.stdout.splitlines():
        if needle in line and ("launch_planning_v2_online" in line
                               or "eval_curated_online" in line):
            return True
    return False


def render(games: list[str], root: Path, problems: str, combined: bool,
           expect: str = "") -> list[str]:
    """Re-render the named games, then the whole-run page. Returns what failed."""
    failed = []
    for g in games:
        cmd = [sys.executable, str(VIZ), "--eval", str(root / g / "online.json"),
               "--problems", problems, "--out", str(root / g / "replay.html"),
               "--title", f"{human_name(g)}: planning trajectories"]
        if subprocess.run(cmd, cwd=REPO).returncode != 0:
            failed.append(g)
    if combined and not failed:
        # the whole run's game list, so the scoreboard counts what has not started yet
        cmd = [sys.executable, str(VIZ), "--run-root", str(root),
               "--problems", problems, "--out", str(root / "replay.html"),
               "--expect-games", expect]
        if subprocess.run(cmd, cwd=REPO).returncode != 0:
            failed.append("(combined)")
    return failed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="the launcher out-root holding <game>/online.json")
    ap.add_argument("--problems", default="logs/2026-08-29/planning_v2/problems.json")
    ap.add_argument("--every", type=int, default=300, help="seconds between checks")
    ap.add_argument("--hours", type=float, default=48.0, help="wall-clock budget")
    ap.add_argument("--combined", action=argparse.BooleanOptionalAction, default=True,
                    help="also draw the one-page-for-the-whole-run view")
    ap.add_argument("--once", action="store_true", help="render what is there and exit")
    ap.add_argument("--expect-games", default=",".join(GAME_ORDER),
                    help="the games this run covers, in order; the scoreboard shows the "
                    "ones with no results yet as pending so the progress counter is "
                    "against the whole run")
    a = ap.parse_args()

    root = Path(a.root)
    if not root.is_dir():
        raise SystemExit(f"{root} is not a directory")
    deadline = time.time() + a.hours * 3600
    seen: dict[str, tuple[int, int]] = {}
    idle = 0

    while True:
        now = fingerprint(root)
        moved = [g for g, fp in now.items() if seen.get(g) != fp]
        if moved:
            stamp = time.strftime("%H:%M:%S")
            print(f"[{stamp}] rendering {', '.join(moved)}", flush=True)
            failed = render(moved, root, a.problems, a.combined, a.expect_games)
            # only bank the games that actually drew, so a failure is retried next tick
            for g in moved:
                if g not in failed:
                    seen[g] = now[g]
            if failed:
                print(f"  failed: {', '.join(failed)}", flush=True)
            idle = 0
        else:
            idle += 1

        if a.once:
            return
        alive = launcher_running(root)
        if not alive and not moved and idle >= 2:
            print("run is finished and nothing moved -- stopping", flush=True)
            return
        if time.time() > deadline:
            print(f"wall-clock budget of {a.hours}h reached -- stopping", flush=True)
            return
        time.sleep(a.every)


if __name__ == "__main__":
    main()
