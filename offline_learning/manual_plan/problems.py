"""Problem records: build from a curated window, persist per game, export for eval.

A record is self-contained -- (game, seed, prefix, gt_actions) regenerates every frame in
it from a cold engine, so `start_grid`/`goal_grid` are a cache that M7 re-verifies rather
than a source of truth. Field names match compose_plan / coverage_plan so the exported set
drops into scripts/eval_coverage_plan.py and eval_coverage_online.py unchanged.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from offline_learning.human_replay import GAMES
from offline_learning.manual_plan.session import DATA, canon, replay

CONTEXT_K = 9


def problems_path(game: str) -> Path:
    d = DATA / canon(game)
    d.mkdir(parents=True, exist_ok=True)
    return d / "problems.json"


def load(game: str) -> list[dict]:
    p = problems_path(game)
    if not p.exists():
        return []
    return json.loads(p.read_text()).get("problems", [])


def save(game: str, problems: list[dict]) -> Path:
    p = problems_path(game)
    p.write_text(json.dumps({"source": "manual_plan", "game": canon(game),
                             "exact_frames": True, "n": len(problems),
                             "problems": problems}, indent=1))
    return p


def next_id(game: str, existing: list[dict]) -> str:
    n = 0
    for p in existing:
        try:
            n = max(n, int(str(p.get("id", "")).rsplit("-", 1)[-1]))
        except ValueError:
            pass
    return f"{canon(game)}-{n + 1:03d}"


def build(game: str, seed: int, prefix: list[str], plan: list[str],
          *, session: str | None = None, t: int | None = None, note: str = "",
          pid: str | None = None) -> dict:
    """Materialise a problem from a window. Frames come from one cold replay."""
    game = canon(game)
    grids = replay(game, seed, list(prefix) + list(plan))
    i = len(prefix)
    ctx = [{"grid": grids[j], "action": (list(prefix) + list(plan))[j]}
           for j in range(max(0, i - CONTEXT_K), i)]
    return {
        "id": pid or "",
        "game": game, "program": GAMES[game][0], "seed": int(seed),
        "source": "manual_plan", "session": session, "t": t if t is not None else len(prefix),
        "h": len(plan),
        "prefix": list(prefix), "gt_actions": list(plan),
        "start_grid": grids[i], "goal_grid": grids[-1],
        "mask": None,                      # exact frames: no masking in this pipeline
        "context": ctx,
        "note": note,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "audit": None,
    }


def upsert(game: str, problem: dict) -> tuple[list[dict], str]:
    """Insert or replace by id; returns (problems, id)."""
    ps = load(game)
    pid = problem.get("id") or next_id(game, ps)
    problem["id"] = pid
    for k, existing in enumerate(ps):
        if existing.get("id") == pid:
            ps[k] = problem
            break
    else:
        ps.append(problem)
    save(game, ps)
    return ps, pid


def delete(game: str, pid: str) -> list[dict]:
    ps = [p for p in load(game) if p.get("id") != pid]
    save(game, ps)
    return ps


def export(games: list[str], out: Path, only_passing: bool = True) -> dict:
    """One flat problem set across games, in the shape the eval harness consumes."""
    keep, dropped = [], []
    for g in games:
        for p in load(g):
            a = p.get("audit") or {}
            if only_passing and not a.get("ok"):
                dropped.append({"id": p.get("id"), "reason": "audit not passing"})
                continue
            keep.append({k: v for k, v in p.items() if k != "audit"})
    payload = {"source": "manual_plan", "exact_frames": True, "n": len(keep),
               "games": sorted({p["game"] for p in keep}),
               "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "problems": keep}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    return {"out": str(out), "n": len(keep), "dropped": dropped}
