"""Phase-0 checks for program_runtime.py (no LLM, no learner).

1. Identity program on the dq8gc train buffer: fit_static=1.0, fit_changed=0.0,
   balanced score = identity floor = 0.5, degeneracy flag set.
2. Determinism ceilings (k=0/1/9) for the 5 A/B games -- the exact-fit ceiling
   any deterministic program can reach; stochastic games (s2kt7/83wkq) < 1.0.
3. Timeout kill/restart: a program that hangs on one action times out there and
   still answers other actions afterwards (fresh worker).
4. plan_search/rollout sanity on toy T-hats: BFS on a 4-verb mover, beam on a
   click painter.

Run:  uv run python offline_learning/scripts/check_program_runtime.py
"""
from __future__ import annotations

import os as _bos, sys as _bsys  # offline_learning/ on sys.path (flat-import the kept libs)
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from validate import load_transitions
from clean_sweep import GAMES
import program_runtime as prt

DATA = HERE.parent / "clean_data3"  # offline_learning/clean_data3 (script lives in scripts/)
# (GAMES key for the action whitelist, data dir under clean_data3)
AB_GAMES = [("dq8gc", "dq8gc"), ("bt3gb", "bt3gb"), ("n2ntd", "n2ntd"),
            ("s2kt7", "s2kt7_seed1"), ("83wkq", "83wkq_seed1")]
CONTEXT_K = 9

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = ""):
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def load_buffer(game_key: str, data_dir: str):
    whitelist = set(GAMES[game_key][0].split(","))
    trs = load_transitions([DATA / data_dir / "train"], whitelist, context_k=CONTEXT_K)
    return prt.prepare_transitions(trs, CONTEXT_K)


def check_identity_on_dq8gc():
    print("== 1. identity program on dq8gc train buffer ==")
    prepared = load_buffer("dq8gc", "dq8gc")
    rt = prt.ProgramRuntime(prt.IDENTITY_PROGRAM, timeout_s=2.0)
    t0 = time.time()
    results = rt.score_buffer(prepared)
    dt = time.time() - t0
    rt.close()
    st = prt.fit_stats(results)
    print(f"  n={st['n']} (changed={st['n_changed']}, static={st['n_static']}), "
          f"scored in {dt:.2f}s ({st['n'] / dt:.0f} transitions/s)")
    check("fit_static == 1.0", st["fit_static"] == 1.0, f"{st['fit_static']:.3f}")
    check("fit_changed == 0.0", st["fit_changed"] == 0.0, f"{st['fit_changed']:.3f}")
    check("balanced == 0.5 == identity floor",
          prt.balanced_score(st) == 0.5 == st["identity_floor_balanced"],
          f"balanced={prt.balanced_score(st):.3f} floor={st['identity_floor_balanced']:.3f}")
    check("all_identity_on_changed flag", st["all_identity_on_changed"] is True)
    check("no crashes/timeouts", st["crash_rate"] == 0.0 and st["timeout_rate"] == 0.0)


def check_ceilings():
    print("== 2. determinism ceilings (k=0/1/9) ==")
    print(f"  {'game':<14} {'n':>5} {'chg%':>6} {'k=0':>6} {'k=1':>6} {'k=9':>6}")
    for game_key, data_dir in AB_GAMES:
        prepared = load_buffer(game_key, data_dir)
        cs = {k: prt.determinism_ceiling(prepared, k) for k in (0, 1, CONTEXT_K)}
        chg = sum(p.changed for p in prepared) / len(prepared)
        print(f"  {data_dir:<14} {len(prepared):>5} {chg:>6.2f} "
              f"{cs[0]:>6.3f} {cs[1]:>6.3f} {cs[CONTEXT_K]:>6.3f}")
        check(f"{data_dir}: ceilings monotone in k",
              cs[0] <= cs[1] + 1e-9 and cs[1] <= cs[CONTEXT_K] + 1e-9)


HANG_PROGRAM = '''\
def transition(prev, grid, action):
    if action[0] == "up":
        while True:
            pass
    return [row[:] for row in grid]
'''


def check_timeout_restart():
    print("== 3. timeout kill/restart ==")
    grid = [["black"] * 3 for _ in range(3)]
    rt = prt.ProgramRuntime(HANG_PROGRAM, timeout_s=0.5)
    t0 = time.time()
    pred, err = rt.transition([], grid, ("up", None, None))
    dt = time.time() - t0
    check("hang -> timeout error", pred is None and err is not None and "timeout" in err,
          f"err={err!r} in {dt:.2f}s")
    check("timeout enforced promptly", 0.4 <= dt < 2.0, f"{dt:.2f}s")
    pred, err = rt.transition([], grid, ("left", None, None))
    check("worker restarted, next call ok", err is None and pred == grid, f"err={err!r}")
    check("timeout counted", rt.n_timeouts == 1, f"n_timeouts={rt.n_timeouts}")
    rt.close()


MOVER_PROGRAM = '''\
DIRS = {"left": (0, -1), "right": (0, 1), "up": (-1, 0), "down": (1, 0)}

def transition(prev, grid, action):
    out = [row[:] for row in grid]
    pos = next(((r, c) for r, row in enumerate(grid)
                for c, v in enumerate(row) if v == "red"), None)
    if pos is None or action[0] not in DIRS:
        return out
    dr, dc = DIRS[action[0]]
    r, c = pos
    nr, nc = r + dr, c + dc
    if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
        out[r][c] = "black"
        out[nr][nc] = "red"
    return out
'''

PAINTER_PROGRAM = '''\
def transition(prev, grid, action):
    verb, r, c = action
    out = [row[:] for row in grid]
    if verb == "click" and r is not None:
        out[r][c] = "gold"
    return out
'''


def check_plan_search():
    print("== 4. plan_search / rollout on toy T-hats ==")
    # BFS branch: 4-verb mover, red (2,2) -> (2,4): shortest plan = right,right
    start = [["black"] * 5 for _ in range(5)]
    start[2][2] = "red"
    goal = [["black"] * 5 for _ in range(5)]
    goal[2][4] = "red"
    rt = prt.ProgramRuntime(MOVER_PROGRAM, timeout_s=1.0)
    universe = prt.build_action_universe(["left", "right", "up", "down"], start)
    plan = prt.plan_search(rt, [], start, goal, universe, h=4, context_k=CONTEXT_K)
    check("BFS finds shortest mover plan",
          plan == [("right", None, None)] * 2, f"plan={plan}")
    rolled = prt.rollout(rt, [], start, plan or [], context_k=CONTEXT_K)
    check("rollout endpoint reaches goal",
          bool(rolled) and rolled[-1] is not None
          and prt.canon_grid(rolled[-1]) == prt.canon_grid(goal))
    check("unreachable goal -> None",
          prt.plan_search(rt, [], start, [["red"] * 5 for _ in range(5)],
                          universe, h=2, context_k=CONTEXT_K) is None)
    # goal==start: [] (already there) must stay distinguishable from None (no plan).
    # Callers that score "the grid after the final action" pass allow_empty=False and
    # get a >=1-step holding plan instead; `if not found` conflates the two and reads
    # a solved problem as "no-plan-found".
    check("goal==start -> [] not None",
          prt.plan_search(rt, [], start, start, universe, h=2,
                          context_k=CONTEXT_K) == [])
    noop_uni = prt.build_action_universe(["noop", "left", "right"], start)
    held = prt.plan_search(rt, [], start, start, noop_uni, h=2,
                           context_k=CONTEXT_K, allow_empty=False)
    check("goal==start + allow_empty=False -> >=1-step holding plan",
          held == [("noop", None, None)], f"plan={held}")
    rt.close()

    # beam branch: click painter, paint (1,1) and (3,3) gold (26 actions > 8)
    start = [["black"] * 5 for _ in range(5)]
    goal = [row[:] for row in start]
    goal[1][1] = "gold"
    goal[3][3] = "gold"
    rt = prt.ProgramRuntime(PAINTER_PROGRAM, timeout_s=1.0)
    universe = prt.build_action_universe(["noop", "click"], start, goal)
    check("click universe expands + goal-diff cells first",
          len(universe) == 26 and universe[1] in [("click", 1, 1), ("click", 3, 3)],
          f"len={len(universe)} first_click={universe[1]}")
    plan = prt.plan_search(rt, [], start, goal, universe, h=3, context_k=CONTEXT_K)
    check("beam finds 2-click plan",
          plan is not None and len(plan) == 2
          and sorted(plan) == [("click", 1, 1), ("click", 3, 3)], f"plan={plan}")
    rt.close()


def main():
    check_identity_on_dq8gc()
    check_ceilings()
    check_timeout_restart()
    check_plan_search()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
        sys.exit(1)
    print("all phase-0 checks passed")


if __name__ == "__main__":
    main()
