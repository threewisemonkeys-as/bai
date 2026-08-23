"""Phase-1 queue entry point that counts evaluator jobs, not their uv parent."""

import os
from pathlib import Path

import launch_remaining_autumn_phase1 as launcher


def active_gepa_pids() -> list[int]:
    active = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        executable = Path(os.fsdecode(argv[0])).name if argv and argv[0] else ""
        if executable.startswith("python") and any(
            b"offline_learning/rexpure_optimize.py" in arg for arg in argv[1:]
        ):
            active.append(int(entry.name))
    return sorted(active)


launcher.active_gepa_pids = active_gepa_pids
launcher.main()
