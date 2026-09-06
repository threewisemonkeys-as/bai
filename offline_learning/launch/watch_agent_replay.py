#!/usr/bin/env python3
"""Keep the agent-arm replay pages level with a run that is still playing.

    uv run python offline_learning/launch/watch_agent_replay.py logs/2026-09-06/agent_full

The agent arm writes one `traces/<task_uid>.json` per problem, whole, at the end of that
problem -- so a page built mid-run is never half a problem, only fewer of them. That makes
watching cheap and safe: re-render a game only when one of its traces appears or changes,
and leave the rest alone.

Two pages are kept. `<root>/<game>/replay.html` is the per-game one, which is what a
15-game run needs because one heavy session renders to ~0.2 MB and 86 of them land on the
16 MB limit. `<root>/replay.html` is the whole run, which is the useful entry point early
and grows past comfortable late; it is rebuilt less often for that reason.

Liveness is a PID FILE, never a `ps` pattern. A pattern naming the module also matches the
command line of whatever shell greps for it, so `pgrep -f` reports the run alive when it
is dead and dead when it is its own caller -- measured twice on this project, once in
`proxy_ctl.sh` and once in a monitor loop that told me a finished run was still going.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
VIZ = REPO / "offline_learning/scripts/viz_agent_replay.py"


def alive(pidfile: Path) -> bool:
    """Is the run that owns this root still going?

    Absent pidfile -> assume alive: a watcher started a moment before the launcher must
    not decide the run is over and quit before it begins.
    """
    if not pidfile.is_file():
        return True
    try:
        pid = int(pidfile.read_text().strip())
    except (ValueError, OSError):
        return True
    proc = Path(f"/proc/{pid}")
    if not proc.is_dir():
        return False
    try:                                   # the pid was recycled by something unrelated
        return "autumn.launch" in (proc / "cmdline").read_bytes().decode("utf-8", "replace")
    except OSError:
        return False


GAME_RE = re.compile(r'"game"\s*:\s*"([^"]+)"')
_GAME_CACHE: dict[str, str] = {}


def game_of(path: Path) -> str | None:
    """The world a trace belongs to, read from the file rather than its name.

    The name cannot be parsed: `task_uid.replace(":", "_")` makes
    `colour_lines:row:s0` into `colour_lines_row_s0`, and two of the fifteen worlds
    (`colour_lines`, `logic_gates`) have an underscore in their own name, so splitting on
    the first `_` silently invents the worlds `colour` and `logic`. `write_trace` puts
    `game` in the first few dozen bytes, so a small read answers it without parsing a
    file that can run to hundreds of kilobytes -- and the answer is cached, because a
    trace is written once and never rewritten.
    """
    if path.name in _GAME_CACHE:
        return _GAME_CACHE[path.name]
    try:
        head = path.read_bytes()[:400].decode("utf-8", "replace")
    except OSError:
        return None
    m = GAME_RE.search(head)
    if not m:                                        # unexpected layout: pay the parse
        try:
            m2 = json.loads(path.read_text()).get("game")
        except (OSError, json.JSONDecodeError, AttributeError):
            return None
        if not m2:
            return None
        _GAME_CACHE[path.name] = m2
        return m2
    _GAME_CACHE[path.name] = m.group(1)
    return m.group(1)


def fingerprint(root: Path) -> dict[str, dict[str, tuple[int, int]]]:
    """Per game, what its traces look like right now: {game: {stem: (size, mtime)}}.

    Grouped by game because that is the unit of rendering, and stat-based because a
    trace is written once and never touched again -- a changed stat is a new problem.
    """
    out: dict[str, dict[str, tuple[int, int]]] = {}
    for f in sorted(root.glob("traces/*.json")):
        try:
            st = f.stat()
        except FileNotFoundError:
            continue
        game = game_of(f)
        if game is None:                             # still being written; next scan
            continue
        out.setdefault(game, {})[f.stem] = (st.st_size, int(st.st_mtime))
    return out


def render(root: Path, game: str | None, log) -> bool:
    out = (root / game / "replay.html") if game else (root / "replay.html")
    cmd = [sys.executable, str(VIZ), "--run-root", str(root), "--out", str(out)]
    # The run-level page is an INDEX: scoreboard plus links. Measured at 10 problems,
    # a whole-run page carrying every transcript runs ~290 KB per problem, so the full
    # 86 would be ~25 MB. The transcripts live on the per-game pages.
    cmd += ["--games", game] if game else ["--index"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=REPO)
    except subprocess.SubprocessError as exc:
        log(f"  render {game or 'run'} failed: {exc}")
        return False
    if r.returncode != 0:
        log(f"  render {game or 'run'} exited {r.returncode}: {r.stderr.strip()[-400:]}")
        return False
    log(f"  {r.stdout.strip()}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="the launch.py --out directory")
    ap.add_argument("--interval", type=int, default=180, help="seconds between scans")
    ap.add_argument("--full-every", type=int, default=1,
                    help="rebuild the run-level index every N scans that changed "
                         "anything; it is cheap, so by default every one")
    ap.add_argument("--pidfile", default="",
                    help="the run's pid file (default <root>/launch.pid)")
    ap.add_argument("--max-hours", type=float, default=30.0)
    ap.add_argument("--linger", type=int, default=2,
                    help="scans to keep going after the run exits, so the last problem "
                         "is drawn before the watcher stops")
    a = ap.parse_args()

    root = Path(a.root)
    pidfile = Path(a.pidfile) if a.pidfile else root / "launch.pid"
    deadline = time.time() + a.max_hours * 3600

    def log(msg: str) -> None:
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    log(f"watching {root} (pidfile {pidfile}, every {a.interval}s)")
    seen: dict[str, dict[str, tuple[int, int]]] = {}
    scans_changed = 0
    linger = a.linger

    while True:
        root.mkdir(parents=True, exist_ok=True)
        now = fingerprint(root)
        changed = [g for g, t in now.items() if seen.get(g) != t]
        if changed:
            done = sum(len(t) for t in now.values())
            log(f"{done} problem(s) recorded; re-rendering {', '.join(sorted(changed))}")
            for g in sorted(changed):
                if render(root, g, log):
                    seen[g] = now[g]
            scans_changed += 1
            # `scans_changed % n == 1` is never true for n == 1, which silently turned
            # "rebuild the index every scan" into "never rebuild it". Count from the
            # first changed scan instead, so every n includes n == 1.
            if (scans_changed - 1) % a.full_every == 0 or not alive(pidfile):
                render(root, None, log)     # the whole-run page, less often: it is big
        running = alive(pidfile)
        if not running:
            linger -= 1
            if linger <= 0:
                log("the run is gone; final render")
                for g in sorted(now):
                    render(root, g, log)
                render(root, None, log)
                log("done")
                return
        else:
            linger = a.linger
        if time.time() > deadline:
            log(f"wall-clock budget of {a.max_hours}h reached; stopping")
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
