#!/usr/bin/env python3
"""Profile an Autumn game for the five properties that decide how to build curated
planning problems for it.

Every design choice in `curated_plan.py` traces back to one of these, so run this BEFORE
authoring a ladder rather than discovering them the hard way:

  DRIFT      does the world evolve without input?  An autonomously-moving object stamps
             the elapsed tick into any exact goal frame, which makes a frame-based
             incompressibility screen pass vacuously -- compress against a predicate instead.
  RNG        are frames a deterministic function of (seed, actions)?  Where they are not,
             exact-frame goals past the randomised step are memorisation, not planning.
  ABSORBING  is there a state that survives a noop?  Exact-frame goals want to land on one:
             then "matched at step h" and "matched at any step" coincide.
  HIDDEN     does the rendered frame alone determine the next frame?  Where it does not,
             a search that dedups on the frame will silently prune correct branches, and a
             window-conditioned world model cannot cold-start.
  OCCLUSION  do two objects ever share a cell?  Then one is invisible, and two genuinely
             different states can render identically.

    uv run python offline_learning/scripts/game_profile.py --game dq8gc
    uv run python offline_learning/scripts/game_profile.py --all
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "MARAProtocol"),
           str(REPO / "MARAProtocol/python_examples/autumnbench")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from offline_learning.curated_plan import PROGRAMS, Sim, replay  # noqa: E402

VERBS = ["left", "right", "up", "down", "noop", "click"]


def rand_action(rnd: random.Random, g: int) -> str:
    """One random action.  Draw the verb ONCE -- drawing again inside the else branch can
    emit a bare `click`, which is not a legal action (it never reaches an `on clicked`
    handler) and blows up any parser expecting coordinates."""
    v = rnd.choice(VERBS)
    return f"click {rnd.randrange(g)} {rnd.randrange(g)}" if v == "click" else v


def _sim(prog: str, seed: int) -> Sim:
    """A bare Sim built straight from a program NAME, bypassing curated_plan's game table
    and its hidden-state tracking: the profiler must work on programs that have no ladder
    yet, and every diagnostic here is deliberately computed from rendered frames alone."""
    s = Sim.__new__(Sim)
    from python_examples.autumnbench.autumnstdlib import autumnstdlib
    from python_examples.autumnbench.interpreter_module import Interpreter
    s.game = prog
    s.seed = seed
    s.it = Interpreter()
    s.it.run_script((PROGRAMS / f"{prog}.sexp").read_text(), autumnstdlib, "", seed)
    s.bg = s.it.get_background()
    s.hid = {}
    s.n = 0
    return s


def _resolve(name: str) -> str:
    """Program file stem for a user-supplied name: benchmark ids are upper-case (DQ8GC),
    `ice` and the zip-sourced games (rink, logic_gates, ...) are lower-case."""
    for cand in (name, name.upper(), name.lower()):
        if (PROGRAMS / f"{cand}.sexp").is_file():
            return cand
    raise SystemExit(f"no program {name!r} in {PROGRAMS}")


def _step(s: Sim, action: str) -> None:
    verb, *rest = action.split()
    if verb == "click":
        s.it.click(int(rest[1]), int(rest[0]))     # row-major in, column-first out
    elif verb in ("left", "right", "up", "down"):
        getattr(s.it, verb)()
    s.it.step()
    # render_all() rebuilds the occupancy set the collision primitives (isFreePos,
    # *NoCollision, nextLiquid) read; the benchmark harness renders after every step, so a
    # replay that does not plays a different game (mario/magnets/tetris reproduce it).
    s.it.render_all()


def _size(prog: str) -> int:
    s = _sim(prog, 0)
    return len(s.grid())


def drift(prog: str, n: int = 120, seed: int = 1) -> dict:
    """Noop-only rollout from reset: does the world move on its own, and does it cycle?"""
    s = _sim(prog, seed)
    seen: dict = {}
    frames = [s.grid()]
    period = None
    for i in range(n):
        _step(s, "noop")
        g = s.grid()
        if g in seen and period is None:
            period = i + 1 - seen[g]
        seen.setdefault(g, i + 1)
        frames.append(g)
    changed = sum(1 for a, b in zip(frames, frames[1:]) if a != b)
    quiet_at_reset = frames[1] == frames[0]
    settles_after_noops = frames[-1] == frames[-2]
    return {"moves_on_its_own": changed > 0, "ticks_that_changed": changed,
            # First FRAME repeat, which is a lower bound on the true cycle: n2ntd's enemy
            # renders identically at t=3 and t=5 while moving opposite ways, so this reports
            # 2 against a real period of 18.  A short repeat here alongside a HIDDEN flag is
            # the signature of an unrendered direction bit.
            "first_frame_repeat": period,
            "absorbing_from_reset": quiet_at_reset,
            "quiet_at_reset": quiet_at_reset,
            "settles_after_noops": settles_after_noops}


def rng(prog: str, n: int = 40, seeds=(0, 1, 2, 3)) -> dict:
    """Same action sequence under different engine seeds: where do frames diverge?"""
    rnd = random.Random(0)
    g = _size(prog)
    plan = [rand_action(rnd, g) for _ in range(n)]
    runs = []
    for sd in seeds:
        s = _sim(prog, sd)
        out = []
        for a in plan:
            _step(s, a)
            out.append(s.grid())
        runs.append(out)
    first = None
    for i in range(n):
        if len({r[i] for r in runs}) > 1:
            first = i
            break
    return {"seed_dependent": first is not None, "first_divergence_step": first,
            # NOT a cause: 83wkq diverges on a random walk seeded by an earlier click, so the
            # verb standing at the divergence step is usually innocent.
            "verb_at_divergence": plan[first].split()[0] if first is not None else None}


def hidden(prog: str, drives: int = 12, steps: int = 60) -> dict:
    """Is the rendered frame a sufficient state?  Collect (frame, action) -> next frame over
    random drives; any key mapping to two different successors PROVES unrendered state."""
    g = _size(prog)
    table: dict = defaultdict(set)
    for d in range(drives):
        rnd = random.Random(d)
        # Vary non-zero seeds so uniformChoice does not hide real ambiguities.
        s = _sim(prog, 1 + d)
        cur = s.grid()
        for _ in range(steps):
            a = rand_action(rnd, g)
            _step(s, a)
            nxt = s.grid()
            table[(cur, a)].add(nxt)
            cur = nxt
    revisited = sum(1 for v in table.values() if len(v) > 1)
    return {"frame_is_markov": revisited == 0, "ambiguous_keys": revisited,
            "keys_seen": len(table)}


def occlusion(prog: str, n: int = 60) -> dict:
    """Two objects on one cell: the later-rendered one hides the earlier.  Counted straight
    off render_all, which lists elements per object BEFORE they are flattened to a grid."""
    rnd = random.Random(0)
    g = _size(prog)
    s = _sim(prog, 1)
    worst, hits = 0, 0
    for _ in range(n):
        d = json.loads(s.it.render_all())
        d.pop("GRID_SIZE", None)
        elems = [(e["position"]["x"], e["position"]["y"]) for k in d for e in d[k]]
        over = len(elems) - len(set(elems))
        worst = max(worst, over)
        hits += over > 0
        _step(s, rand_action(rnd, g))
    return {"objects_overlap": hits > 0, "frames_with_overlap": hits,
            "max_hidden_cells": worst}


def verbs(prog: str, offsets: tuple[int, ...] = (0, 3, 8)) -> dict:
    """Which verbs ever change the frame RELATIVE TO A NOOP from the same state, and does a
    click's POSITION matter?  The counterfactual is essential: on a game with passive
    dynamics (dino, tetris, diffusion) the frame changes every tick whatever you press, so
    a plain before/after comparison calls every verb live. Probes include one-action setup
    states (needed for state-dependent verbs such as egg's `down`) and scan every click cell
    until two distinct effects are found (needed for sparse switches in logic_gates)."""
    g = _size(prog)
    seed = 1

    def after(actions: list[str]) -> Grid:
        s = _sim(prog, seed)
        for a in actions:
            _step(s, a)
        return s.grid()

    prefixes = [["noop"] * k for k in offsets]
    prefixes += [[v] for v in ["left", "right", "up", "down"]]
    live = []
    for v in ["left", "right", "up", "down"]:
        for prefix in prefixes:
            if after(prefix + [v]) != after(prefix + ["noop"]):
                live.append(v)
                break

    click_live, outs = False, set()
    # Do not spend three exhaustive grids proving the absence of a handler when the
    # source itself contains no click trigger. Positive click claims still require replay.
    if "clicked" not in (PROGRAMS / f"{prog}.sexp").read_text():
        return {"live_move_verbs": live, "click_changes_state": False,
                "click_position_matters": False}
    for k in offsets:
        prefix = ["noop"] * k
        base = after(prefix + ["noop"])
        for row in range(g):
            for col in range(g):
                out = after(prefix + [f"click {row} {col}"])
                if out == base:
                    continue
                click_live = True
                outs.add(out)
                if len(outs) > 1:
                    break
            if len(outs) > 1:
                break
        if len(outs) > 1:
            break
    return {"live_move_verbs": live, "click_changes_state": click_live,
            "click_position_matters": len(outs) > 1}


def profile(prog: str) -> dict:
    return {"program": prog, "grid": _size(prog), "drift": drift(prog), "rng": rng(prog),
            "hidden": hidden(prog), "occlusion": occlusion(prog), "verbs": verbs(prog)}


def render(p: dict) -> str:
    d, r, h, o, v = p["drift"], p["rng"], p["hidden"], p["occlusion"], p["verbs"]
    flags = []
    if d["moves_on_its_own"]:
        flags.append(f"DRIFT(repeat>={d['first_frame_repeat']})")
    if r["seed_dependent"]:
        flags.append(f"RNG(step {r['first_divergence_step']})")
    if not h["frame_is_markov"]:
        flags.append(f"HIDDEN({h['ambiguous_keys']}/{h['keys_seen']})")
    if o["objects_overlap"]:
        flags.append(f"OCCLUSION(max {o['max_hidden_cells']})")
    if d["quiet_at_reset"]:
        flags.append("QUIET-AT-RESET")
    if d["settles_after_noops"] and not d["quiet_at_reset"]:
        flags.append("SETTLES-UNDER-NOOP")
    return (f"{p['program']:<9} {p['grid']:>3}  "
            f"moves={','.join(v['live_move_verbs']) or '-':<24} "
            f"click={'pos' if v['click_position_matters'] else ('yes' if v['click_changes_state'] else 'no'):<4} "
            f"{'  '.join(flags)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", action="append", help="program name, e.g. DQ8GC (repeatable)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    if a.all:
        names = sorted(p.stem for p in PROGRAMS.glob("*.sexp")
                       if "wrong_program" not in p.stem)
    else:
        names = [_resolve(g) for g in (a.game or [])]
    out = []
    print(f"{'program':<9} {'sz':>3}  {'live move verbs':<31}{'click':<6}flags")
    print("-" * 108)
    for n in names:
        try:
            p = profile(n)
        except Exception as exc:  # noqa: BLE001
            print(f"{n:<9} FAILED {type(exc).__name__}: {exc}")
            continue
        out.append(p)
        print(render(p), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
