"""Engine access + session recording for manual planning-problem authoring.

A state is addressed ONLY as (program, seed, action prefix). The interpreter does expose
`get_environment_string()`/`restore_environment()`, but the dump is a lossy pretty-print
(every closure serialises as `<native fn: ...>`) and restore takes three strings -- it is a
debug facility, not a snapshot. Replay from reset is the one trustworthy address, and it is
cheap: ~0.6 ms/step (200 steps = 0.12 s), which is what lets the curator re-execute an
edited plan on every keystroke.

Actions are stored in the WRAPPER's canonical form -- `click ROW COL`, row-major.
`autumn_env._coerce_action` transposes that into the interpreter's column-first click, so
recording the wrapper string keeps these sessions byte-compatible with `compose_plan`,
`human_replay` and the eval harness (the click arg-order swap is a known trap).

Grids are stored as the renderer's JSON string (`"[[\"black\", ...]]"`) and compared with
string equality, exactly as `compose_plan.exec_from` does -- same renderer, same key order.
"""
from __future__ import annotations

import json
import sys
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

_BAI_ROOT = Path(__file__).resolve().parents[2]
if str(_BAI_ROOT) not in sys.path:
    sys.path.insert(0, str(_BAI_ROOT))

from autumn_env import AutumnBenchEnvWrapper  # noqa: E402
from offline_learning.human_replay import GAMES, _grid, _obs_cell  # noqa: E402
from offline_learning.mechanics_rules import BACKGROUND, BG, SIZE, fired  # noqa: E402

DATA = _BAI_ROOT / "offline_learning" / "manual_plan_data"

# The five games this pipeline is scoped to, by their study names.
ALIASES = {"ice": "bt3gb", "disease": "dq8gc", "ants": "s2kt7",
           "mario": "n2ntd", "particles": "83wkq"}
ORDER = ["bt3gb", "dq8gc", "s2kt7", "n2ntd", "83wkq"]

# Games with an object that evolves on a fixed cycle the agent cannot influence. Its
# position in an exact goal frame encodes the elapsed tick count, so DELETING an action can
# never reach the goal and the "no shorter plan" screen passes vacuously there. The
# length-preserving substitution screen stays sharp, which is why both are run.
TICK_LOCKED = {g for g, cls in BACKGROUND.items() if cls}


def canon(game: str) -> str:
    g = game.strip().lower()
    g = ALIASES.get(g, g)
    if g not in GAMES:
        raise KeyError(f"unknown game {game!r}; known: {sorted(GAMES)} + {sorted(ALIASES)}")
    return g


def info(game: str) -> dict:
    g = canon(game)
    prog, human, verbs = GAMES[g]
    return {"game": g, "program": prog, "human_name": human, "verbs": list(verbs),
            "size": SIZE[g], "background": sorted(BG[g]), "tick_locked": g in TICK_LOCKED}


def games() -> list[dict]:
    return [info(g) for g in ORDER]


# ------------------------------------------------------------------------------ engine
def _new_env(game: str, seed: int, budget: int) -> AutumnBenchEnvWrapper:
    env = AutumnBenchEnvWrapper(env_name=GAMES[canon(game)][0], task_type="interactive",
                                max_episode_steps=budget, seed=seed, render_mode="text")
    return env


def _grid_of(obs: dict) -> str:
    return _grid(_obs_cell(obs))


_CACHE: "OrderedDict[tuple, list[str]]" = OrderedDict()
_CACHE_MAX = 64


def replay(game: str, seed: int, actions: list[str], use_cache: bool = True) -> list[str]:
    """Grids for a cold replay: index 0 is the post-reset frame, index i+1 follows
    actions[i]. Length is len(actions)+1. `use_cache=False` forces a fresh engine, which is
    how the determinism screen gets a genuinely independent second opinion."""
    game = canon(game)
    key = (game, seed, tuple(actions))
    hit = _CACHE.get(key) if use_cache else None
    if hit is not None:
        _CACHE.move_to_end(key)
        return list(hit)
    env = _new_env(game, seed, len(actions) + 8)
    obs, _ = env.reset(seed=seed)
    out = [_grid_of(obs)]
    for a in actions:
        obs, _r, term, _t, _i = env.step(a)
        out.append(_grid_of(obs))
        if term:
            break
    env.close()
    while len(out) < len(actions) + 1:      # terminated early: pad so indices stay aligned
        out.append(out[-1])
    _CACHE[key] = list(out)
    _CACHE.move_to_end(key)
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return out


def exec_plan(game: str, seed: int, prefix: list[str], plan: list[str]) -> list[str]:
    """Grids after each action of `plan`, run from the state `prefix` leads to."""
    return replay(game, seed, list(prefix) + list(plan))[len(prefix) + 1:]


def final_grid(game: str, seed: int, prefix: list[str], plan: list[str]) -> str | None:
    g = exec_plan(game, seed, prefix, plan)
    return g[-1] if g else None


def mechanics(game: str, seed: int, prefix: list[str], plan: list[str]) -> list[dict]:
    """Per-plan-step fired rules. Costs one extra replay per step (the noop
    counterfactual), so the curator asks for it on demand rather than on every edit."""
    game = canon(game)
    grids = replay(game, seed, list(prefix) + list(plan))[len(prefix):]
    out = []
    for i, a in enumerate(plan):
        cf = final_grid(game, seed, list(prefix) + list(plan[:i]), ["noop"])
        f = fired(game, grids[i], a, cf, grids[i + 1])
        out.append({"action": f.action, "passive": f.passive, "visible": f.visible})
    return out


def changed_cells(a: str | None, b: str | None) -> list[list[int]]:
    """[[row, col], ...] where two rendered grids differ -- drives the diff overlay."""
    if not a or not b:
        return []
    ga, gb = json.loads(a), json.loads(b)
    return [[r, c] for r, row in enumerate(ga) for c, v in enumerate(row)
            if r < len(gb) and c < len(gb[r]) and gb[r][c] != v]


# ----------------------------------------------------------------------- live sessions
class LiveSession:
    """One in-progress play session. Holds a live env; undo rebuilds by replay."""

    def __init__(self, game: str, seed: int, budget: int = 2000):
        self.game = canon(game)
        self.seed = int(seed)
        self.budget = budget
        self.sid = uuid.uuid4().hex[:8]
        self.created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.actions: list[str] = []
        self.marks: list[dict] = []
        self.env = _new_env(self.game, self.seed, budget)
        obs, _ = self.env.reset(seed=self.seed)
        self.grids: list[str] = [_grid_of(obs)]
        self.saved_as: str | None = None

    def step(self, action: str) -> dict:
        before = len(self.env.failed_candidates)
        obs, _r, term, trunc, _i = self.env.step(action)
        rejected = len(self.env.failed_candidates) > before
        if rejected:
            return {"ok": False, "error": f"engine rejected {action!r}", **self.state()}
        self.actions.append(action)
        self.grids.append(_grid_of(obs))
        return {"ok": True, "terminated": bool(term), "truncated": bool(trunc), **self.state()}

    def undo(self) -> dict:
        if self.actions:
            self.actions.pop()
            self.marks = [m for m in self.marks if m["t"] <= len(self.actions)]
            self.rebuild()
        return {"ok": True, **self.state()}

    def rebuild(self) -> None:
        self.env.close()
        self.env = _new_env(self.game, self.seed, self.budget)
        obs, _ = self.env.reset(seed=self.seed)
        self.grids = [_grid_of(obs)]
        for a in self.actions:
            obs, _r, _term, _t, _i = self.env.step(a)
            self.grids.append(_grid_of(obs))

    def mark(self, note: str = "") -> dict:
        self.marks = [m for m in self.marks if m["t"] != len(self.actions)]
        self.marks.append({"t": len(self.actions), "note": note})
        self.marks.sort(key=lambda m: m["t"])
        return {"ok": True, **self.state()}

    def state(self) -> dict:
        return {"sid": self.sid, "game": self.game, "seed": self.seed,
                "t": len(self.actions), "grid": self.grids[-1],
                "prev_grid": self.grids[-2] if len(self.grids) > 1 else None,
                "last_action": self.actions[-1] if self.actions else None,
                "actions": self.actions, "marks": self.marks}

    def to_dict(self) -> dict:
        return {"kind": "manual_plan_session", "sid": self.sid, "game": self.game,
                "program": GAMES[self.game][0], "seed": self.seed, "created": self.created,
                "actions": self.actions, "marks": self.marks}

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass


SESSIONS: "OrderedDict[str, LiveSession]" = OrderedDict()
SESSIONS_MAX = 6


def open_session(game: str, seed: int) -> LiveSession:
    s = LiveSession(game, seed)
    SESSIONS[s.sid] = s
    while len(SESSIONS) > SESSIONS_MAX:
        _sid, old = SESSIONS.popitem(last=False)
        old.close()
    return s


# ---------------------------------------------------------------------------- files
def sessions_dir(game: str) -> Path:
    d = DATA / canon(game) / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_session(s: LiveSession) -> Path:
    name = s.saved_as or f"{s.created[:10].replace('-', '')}_{s.sid}.json"
    s.saved_as = name
    p = sessions_dir(s.game) / name
    p.write_text(json.dumps(s.to_dict(), indent=1))
    return p


def list_sessions(game: str) -> list[dict]:
    out = []
    for p in sorted(sessions_dir(game).glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        out.append({"name": p.name, "sid": d.get("sid"), "seed": d.get("seed"),
                    "created": d.get("created"), "n": len(d.get("actions", [])),
                    "marks": len(d.get("marks", []))})
    return out


def load_session(game: str, name: str, with_grids: bool = True) -> dict:
    p = sessions_dir(game) / Path(name).name
    d = json.loads(p.read_text())
    if with_grids:
        d["grids"] = replay(d["game"], d["seed"], d["actions"])
    return d
