"""Versioned planning-problem primitives for the selected 15 Autumn games.

Every problem carries both a rendered goal frame and a natural-language goal backed by an
executable Python checker. Neither representation is preferred by the data; evaluators
must explicitly select ``frame`` or ``nl``. The builder uses the raw interpreter and
validation uses AutumnBenchEnvWrapper independently.
"""
from __future__ import annotations

import json
import random
import zlib
from pathlib import Path
from typing import Any

from autumn_env import AutumnBenchEnvWrapper
from offline_learning.human_replay import GAMES, _grid, _obs_cell
from offline_learning.program_meta import grid_size, resolve
from offline_learning.planning_nl_goals import (
    CHECKER_VERSION, get_python_goal, score_python_goal, validate_problem_goal,
)

SCHEMA_VERSION = "planning-problems-v2.2"
SELECTED_GAMES = [
    "eahcw", "egg", "bt3gb", "dq8gc", "7xf97", "n2ntd", "va6fq", "s2kt7",
    "colour_lines", "SET", "diffusion", "dino", "f5w3n", "logic_gates", "7www9",
]
STOCHASTIC_GAMES = {"colour_lines", "SET", "diffusion", "dino", "f5w3n"}

Grid = list[list[str]]


def _action(it, action: str) -> None:
    verb, *args = action.split()
    if verb == "click":
        row, col = map(int, args)
        it.click(col, row)  # agent actions are row-major; interpreter clicks are x,y
    elif verb in {"left", "right", "up", "down"}:
        getattr(it, verb)()
    elif verb != "noop":
        raise ValueError(f"invalid action {action!r}")
    it.step()


def raw_trace(program: str, seed: int, actions: list[str]) -> list[Grid]:
    """Raw-interpreter frames, including the post-reset frame at index zero."""
    from python_examples.autumnbench.autumnstdlib import autumnstdlib
    from python_examples.autumnbench.env_utils import render_grid
    from python_examples.autumnbench.interpreter_module import Interpreter

    it = Interpreter()
    it.run_script(resolve(program).read_text(), autumnstdlib, "", seed)
    bg = it.get_background()

    def render() -> Grid:
        objects = json.loads(it.render_all())
        return render_grid(objects, background_color=bg, color_dict={})

    out = [render()]
    for action in actions:
        _action(it, action)
        out.append(render())  # also rebuilds occupancy before the next engine step
    return out


def wrapper_trace(program: str, seed: int, actions: list[str]) -> list[Grid | None]:
    """Independent wrapper-driver frames, including reset; pads with None after terminal."""
    env = AutumnBenchEnvWrapper(env_name=program, task_type="interactive",
                                max_episode_steps=len(actions) + 8, seed=seed,
                                render_mode="text")
    obs, _ = env.reset(seed=seed)
    out: list[Grid | None] = [json.loads(_grid(_obs_cell(obs)))]
    terminal = False
    for action in actions:
        if terminal:
            out.append(None)
            continue
        obs, _reward, term, trunc, _info = env.step(action)
        out.append(json.loads(_grid(_obs_cell(obs))))
        terminal = bool(term or trunc)
    env.close()
    return out


def rollout(program: str, seed: int, prefix: list[str], plan: list[str],
            driver: str = "raw") -> tuple[Grid, list[Grid | None], list[Grid | None]]:
    """Return (current state after prefix, post-plan frames, complete reset trace)."""
    actions = list(prefix) + list(plan)
    trace = raw_trace(program, seed, actions) if driver == "raw" else wrapper_trace(
        program, seed, actions)
    start = trace[len(prefix)]
    if start is None:
        raise ValueError("prefix terminates before the planning state")
    return start, trace[len(prefix) + 1:], trace


def quiescent_after(program: str, seed: int, prefix: list[str], plan: list[str],
                    driver: str = "raw") -> bool:
    """Whether the candidate's final frame survives one additional hidden noop."""
    trace_fn = raw_trace if driver == "raw" else wrapper_trace
    trace = trace_fn(program, seed, list(prefix) + list(plan) + ["noop"])
    return len(trace) >= 2 and trace[-1] is not None and trace[-1] == trace[-2]


def success(problem: dict[str, Any], presentation: str, start: Grid,
            frames: list[Grid | None], actions: list[str] | None = None, *,
            stable_after_final: bool | None = None) -> tuple[bool, int | None]:
    """Evaluate one explicitly selected goal representation."""
    if presentation == "nl":
        if actions is None:
            raise ValueError("NL goal scoring requires the executed actions")
        goal = validate_problem_goal(problem)
        return score_python_goal(
            goal, start, frames, actions, stable_after_final=stable_after_final,
        )
    if presentation != "frame":
        raise ValueError(f"unknown goal presentation {presentation!r}")

    mode = problem.get("frame_success_mode", "any")
    indices = (
        range(len(frames))
        if mode == "any"
        else range(max(0, len(frames) - 1), len(frames))
    )
    for index in indices:
        frame = frames[index]
        if frame is not None and frame == problem["goal"]:
            return True, index + 1
    return False, None


def _python_success(program: str, seed: int, prefix: list[str], plan: list[str],
                    checker_id: str, driver: str = "raw") -> bool:
    goal = get_python_goal(checker_id)
    start, frames, _ = rollout(program, seed, prefix, plan, driver)
    stable = None
    if goal.require_quiescent:
        stable = quiescent_after(program, seed, prefix, plan, driver)
    return score_python_goal(
        goal, start, frames, plan, stable_after_final=stable,
    )[0]


def compress_to_python(program: str, seed: int, prefix: list[str], plan: list[str],
                       checker_id: str) -> list[str]:
    """Minimize a plan to a fixed point against an executable Python goal.

    Two alternating passes: greedy DELETION (shortens h) and greedy noop
    SUBSTITUTION (keeps timing, strips redundant decisions -- deletion alone cannot
    find these because removing an action shifts every later step, which is exactly
    what the validator's A8 substitution check probes). A plan that already passes
    A8 is at the substitution fixpoint, so existing references are unchanged."""
    out = list(plan)
    changed = True
    while changed:
        changed = False
        for i in range(len(out) - 1, -1, -1):
            candidate = out[:i] + out[i + 1:]
            if candidate and _python_success(
                program, seed, prefix, candidate, checker_id, "raw",
            ):
                out = candidate
                changed = True
                break
        if changed:
            continue
        for i in range(len(out) - 1, -1, -1):
            if out[i] == "noop":
                continue
            candidate = out[:i] + ["noop"] + out[i + 1:]
            if _python_success(program, seed, prefix, candidate, checker_id, "raw"):
                out = candidate
                changed = True
                break
    return out


def stable_seed(task_uid: str) -> int:
    return zlib.crc32(task_uid.encode("utf-8")) & 0xFFFFFFFF


def random_plan(game: str, h: int, rng: random.Random) -> list[str]:
    verbs = GAMES[game][2]
    size = grid_size(game)
    out = []
    for _ in range(h):
        verb = rng.choice(verbs)
        out.append(f"click {rng.randrange(size)} {rng.randrange(size)}"
                   if verb == "click" else verb)
    return out


def materialize(spec: dict[str, Any], random_trials: int = 24) -> dict[str, Any]:
    """Compress against Python goal code, replay, snapshot, and add stable floors."""
    game = spec["game"]
    program = GAMES[game][0]
    prefix = list(spec.get("prefix", []))
    plan = list(spec["plan"])
    checker_id = spec["nl_checker"]
    python_goal = get_python_goal(checker_id)
    nl_goal = spec.get("nl_goal", spec["objective"])
    if python_goal.nl != nl_goal:
        raise ValueError(f"{game}/{spec['id']}: NL sentence disagrees with {checker_id}")
    if spec.get("compress", True):
        plan = compress_to_python(program, spec["seed"], prefix, plan, checker_id)
        # An under-specified checker lets compression silently delete an authored phase
        # (the sand mixed-cascade bug: the pre-existing pile satisfied the count, so the
        # sand click vanished). must_keep pins actions the plan must retain.
        from collections import Counter as _Counter
        missing = _Counter(spec.get("must_keep", [])) - _Counter(plan)
        if missing:
            raise ValueError(
                f"{game}/{spec['id']}: compression deleted must_keep actions "
                f"{sorted(missing.elements())} -- the checker under-specifies the task"
            )

    start, frames, _trace = rollout(program, spec["seed"], prefix, plan, "raw")
    if not frames or frames[-1] is None:
        raise ValueError(f"{game}/{spec['id']}: reference plan has no final frame")
    is_quiescent = quiescent_after(program, spec["seed"], prefix, plan, "raw")

    uid = f"{game}:{spec['id']}:s{spec['seed']}"
    row = {
        "schema_version": SCHEMA_VERSION,
        "task_uid": uid,
        "template_id": f"{game}:{spec.get('template_id', spec['id'])}",
        "game": game,
        "program": program,
        "id": spec["id"],
        "tier": spec["tier"],
        "objective": spec["objective"],
        "nl_goal": nl_goal,
        "nl_checker": checker_id,
        "nl_checker_version": CHECKER_VERSION,
        "nl_success_mode": python_goal.success_mode,
        "nl_require_quiescent": python_goal.require_quiescent,
        "nl_reference_plan": list(python_goal.reference_plan or plan),
        "seed": int(spec["seed"]),
        "prefix": prefix,
        "plan": plan,
        "h": len(plan),
        "n_decisions": sum(action != "noop" for action in plan),
        "frame_success_mode": spec.get("frame_success_mode", "final"),
        "start": start,
        "goal": frames[-1],
        "mechanics": list(spec.get("mechanics", [])),
        "source": spec.get("source", "curated-v2-python-goals"),
        "stochastic": game in STOCHASTIC_GAMES,
        "frame_reference_quiescent": is_quiescent,
        "note": spec.get("note", ""),
    }
    validate_problem_goal(row)

    frame_ok, frame_at = success(row, "frame", start, frames, plan)
    if not frame_ok:
        raise ValueError(f"{uid}: reference plan does not satisfy its frame goal")
    row["frame_reference_reached_at"] = frame_at

    nl_plan = row["nl_reference_plan"]
    nl_start, nl_frames, _ = rollout(program, row["seed"], prefix, nl_plan, "raw")
    nl_stable = (
        quiescent_after(program, row["seed"], prefix, nl_plan, "raw")
        if python_goal.require_quiescent else None
    )
    nl_ok, nl_at = success(
        row, "nl", nl_start, nl_frames, nl_plan, stable_after_final=nl_stable,
    )
    if not nl_ok:
        raise ValueError(f"{uid}: NL reference plan does not satisfy its NL goal")
    row["nl_reference_reached_at"] = nl_at
    row["nl_reference_quiescent"] = bool(nl_stable) if nl_stable is not None else None

    noop_plan = ["noop"] * row["h"]
    noop_start, noop_frames, _ = rollout(
        program, row["seed"], prefix, noop_plan, "raw",
    )
    noop_stable = (quiescent_after(program, row["seed"], prefix, noop_plan, "raw")
                   if python_goal.require_quiescent else None)
    row["frame_noop_success"] = float(success(
        row, "frame", noop_start, noop_frames, noop_plan,
    )[0])
    row["nl_noop_success"] = float(success(
        row, "nl", noop_start, noop_frames, noop_plan,
        stable_after_final=noop_stable,
    )[0])

    for presentation in ("frame", "nl"):
        rng = random.Random(stable_seed(f"{uid}:{presentation}:h{row["h"]}"))
        hits = 0
        for _ in range(random_trials):
            candidate = random_plan(game, row["h"], rng)
            rstart, rframes, _ = rollout(program, row["seed"], prefix, candidate, "raw")
            random_stable = (
                quiescent_after(program, row["seed"], prefix, candidate, "raw")
                if presentation == "nl" and python_goal.require_quiescent else None
            )
            hits += success(
                row, presentation, rstart, rframes, candidate,
                stable_after_final=random_stable,
            )[0]
        row[f"{presentation}_random_trials"] = random_trials
        row[f"{presentation}_random_success"] = (
            hits / random_trials if random_trials else None
        )
    return row


def load_problem_file(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, list):
        return {}, payload
    return payload, payload["problems"]
