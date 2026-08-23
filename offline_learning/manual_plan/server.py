"""Local FastAPI app backing the play recorder and the curation/audit editor.

Every endpoint is a plain `def`, so FastAPI runs it in the threadpool, and every engine
call is serialised behind one lock: the Autumn interpreter is a C++ extension and this is a
single-operator tool, so a global lock is the honest choice over hoping it is re-entrant.

    uv run python -m offline_learning.manual_plan.cli serve
    open http://127.0.0.1:8764/
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from offline_learning.manual_plan import audit as A
from offline_learning.manual_plan import problems as P
from offline_learning.manual_plan import session as S

STATIC = Path(__file__).parent / "static"
LOCK = threading.Lock()

app = FastAPI(title="Autumn manual planning-problem authoring")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/")
def index():
    return RedirectResponse("/static/play.html")


@app.get("/curate")
def curate():
    return FileResponse(STATIC / "curate.html")


@app.get("/api/games")
def api_games():
    return {"games": S.games()}


# ------------------------------------------------------------------------------- play
@app.post("/api/session")
def api_new_session(body: dict = Body(...)):
    with LOCK:
        s = S.open_session(body["game"], int(body.get("seed", 1)))
        return {**s.state(), "info": S.info(s.game)}


def _sess(sid: str) -> S.LiveSession:
    s = S.SESSIONS.get(sid)
    if s is None:
        raise HTTPException(404, "session expired -- start a new one (only the most recent "
                                 "few stay live; saved sessions reopen in the curator)")
    return s


@app.post("/api/session/{sid}/step")
def api_step(sid: str, body: dict = Body(...)):
    with LOCK:
        return _sess(sid).step(str(body["action"]))


@app.post("/api/session/{sid}/undo")
def api_undo(sid: str):
    with LOCK:
        return _sess(sid).undo()


@app.post("/api/session/{sid}/mark")
def api_mark(sid: str, body: dict = Body(default={})):
    with LOCK:
        return _sess(sid).mark(str(body.get("note", "")))


@app.post("/api/session/{sid}/save")
def api_save(sid: str):
    with LOCK:
        s = _sess(sid)
        return {"ok": True, "path": str(S.save_session(s)), "name": s.saved_as}


# ---------------------------------------------------------------------------- curate
@app.get("/api/sessions/{game}")
def api_sessions(game: str):
    return {"sessions": S.list_sessions(game)}


@app.get("/api/session_file/{game}/{name}")
def api_session_file(game: str, name: str):
    with LOCK:
        try:
            return S.load_session(game, name)
        except FileNotFoundError:
            raise HTTPException(404, f"no session {name} for {game}")


@app.post("/api/exec")
def api_exec(body: dict = Body(...)):
    """Frames for prefix+plan, plus the cells where the end frame differs from the goal --
    the diff is what tells the curator which cells an edit broke."""
    with LOCK:
        game, seed = body["game"], int(body["seed"])
        prefix, plan = list(body.get("prefix", [])), list(body.get("plan", []))
        grids = S.replay(game, seed, prefix + plan)
        start = grids[len(prefix)]
        end = grids[-1] if plan else start
        goal = body.get("goal") or end
        out: dict[str, Any] = {
            "start_grid": start, "end_grid": end,
            "plan_grids": grids[len(prefix) + 1:],
            "reaches_goal": end == goal,
            "diff": S.changed_cells(end, goal),
            "start_goal_diff": S.changed_cells(start, goal),
        }
        if body.get("mechanics"):
            out["mechanics"] = S.mechanics(game, seed, prefix, plan)
        return out


@app.post("/api/audit")
def api_audit(body: dict = Body(...)):
    with LOCK:
        p = body["problem"]
        return A.audit(p, n_random=int(body.get("n_random", 12)),
                       screens=body.get("screens"))


@app.post("/api/repair")
def api_repair(body: dict = Body(...)):
    """Single small edits that would put a missing plan back on its goal."""
    with LOCK:
        return {"candidates": A.repair(body["problem"], int(body.get("max_extra", 4)))}


@app.get("/api/problems/{game}")
def api_problems(game: str):
    return {"problems": P.load(game)}


@app.post("/api/problems/{game}")
def api_upsert(game: str, body: dict = Body(...)):
    """Build (or rebuild) a problem from a window, audit it, and store it. The audit runs
    server-side on save so nothing lands on disk with a stale verdict attached."""
    with LOCK:
        p = P.build(game, int(body["seed"]), list(body.get("prefix", [])),
                    list(body.get("plan", [])), session=body.get("session"),
                    t=body.get("t"), note=body.get("note", ""), pid=body.get("id") or None)
        p["audit"] = A.audit(p, n_random=int(body.get("n_random", 12)))
        ps, pid = P.upsert(game, p)
        return {"ok": True, "id": pid, "problem": p, "n": len(ps)}


@app.delete("/api/problems/{game}/{pid}")
def api_delete(game: str, pid: str):
    with LOCK:
        return {"ok": True, "problems": P.delete(game, pid)}
