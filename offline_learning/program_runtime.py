"""Execution runtime for learned *program world models* (WorldCoder-style arm).

A candidate world model is a self-contained Python module implementing

    def transition(prev, grid, action) -> next_grid

over parsed Autumn color grids (see CONTRACT below -- the single source of
truth quoted verbatim in the learner's prompts). This module owns everything
needed to run such programs safely and score/plan with them, with **no LLM
imports**:

- parse/canon helpers for grids and actions (canonical (row, col) clicks --
  stored clicks are already row-major `click ROW COL`, no swap needed);
- ProgramRuntime: a persistent worker process per program (Pipe + per-call
  deadline; kill/restart on timeout/crash). signal.alarm is unusable here:
  it is main-thread-only and eval_online_plan drives jobs from a thread pool;
- ϕ1 scoring over a prepared transition buffer (exact fit + per-cell partial
  credit as refinement feedback), identity-floor statistics, and a
  determinism ceiling (max achievable exact fit for ANY deterministic
  function of the K-window -- stochastic games sit below 1.0);
- plan_search / rollout: zero-LLM planning primitives over T-hat.
"""
from __future__ import annotations

import json
import threading
import traceback
import multiprocessing as mp
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from test50_sim_tools import grid_json  # light import: csv/json/pathlib only

Grid = list  # list[list[str]], row-major: grid[r][c] = color name
Action = tuple  # (verb, row | None, col | None); row/col ints only for click


# The contract shown verbatim to the program-writing LLM. Keep in sync with
# the enforcement in _worker_main/_validate_grid.
CONTRACT = '''\
You must implement a single Python module with this exact entry point:

    def transition(prev, grid, action):
        """Predict the next grid of the game.

        Args:
          prev:   list of up to K previous (grid, action) pairs, oldest->newest.
                  The last entry's action is the one that led INTO `grid`.
                  May be shorter (or empty) near the start of an episode.
          grid:   the current grid: list of rows, each a list of color-name
                  strings; grid[r][c] is the cell at row r, column c.
          action: a tuple (verb, row, col). row/col are ints only for
                  "click" (the clicked cell, canonical row/col order);
                  for every other verb they are None.

        Returns:
          The COMPLETE next grid: a new list of rows with the SAME dimensions
          as `grid`, every cell a color-name string.
        """

Rules:
- Deterministic pure Python. Standard library only (copy, math, itertools,
  collections are plenty). No I/O, no randomness, no global mutable state
  that persists across calls.
- Never mutate the inputs; build the next grid (e.g. deepcopy then edit).
- Model general dynamics rules (how objects/cells move, appear, change color,
  interact, and what each action verb / click does), NOT a lookup table of
  memorized (grid -> grid) pairs, and NOT anything keyed on step counts.
- You may define any helper functions/classes you like (e.g. finding objects
  as connected color regions), but `transition` is the entry point.
'''

IDENTITY_PROGRAM = '''\
def transition(prev, grid, action):
    return [row[:] for row in grid]
'''


# ---------------------------------------------------------------------------
# Parsing / canonical forms
# ---------------------------------------------------------------------------
def parse_action(s: str) -> Action:
    """Canonical action string -> (verb, row, col). Stored clicks are
    row-major `click ROW COL` (no swap needed)."""
    p = (s or "").split()
    if len(p) == 3 and p[0] == "click":
        return ("click", int(p[1]), int(p[2]))
    return (p[0] if p else "", None, None)


def unparse_action(a: Action) -> str:
    verb, r, c = a
    if verb == "click" and r is not None:
        return f"click {r} {c}"
    return verb


def parse_grid_strict(raw: str) -> Grid:
    """Extract + parse the first [[...]] color grid from an observation cell.
    Raises ValueError when no valid grid is present (autumn data always has one)."""
    g = json.loads(grid_json(raw))
    if not (isinstance(g, list) and g and all(isinstance(r, list) for r in g)):
        raise ValueError("observation grid is not a 2D array")
    return g


def canon_grid(g: Grid) -> str:
    """Compact canonical JSON -- the same form grids_equal comparisons use."""
    return json.dumps(g, separators=(",", ":"))


def _validate_grid(out, want_rows: int, want_cols: int):
    """Enforce the contract's return shape. Returns (grid, None) or (None, err)."""
    if not isinstance(out, list):
        return None, f"bad-return: transition returned {type(out).__name__}, expected list of rows"
    if len(out) != want_rows:
        return None, f"bad-return: {len(out)} rows, expected {want_rows}"
    grid = []
    for r, row in enumerate(out):
        if isinstance(row, tuple):
            row = list(row)
        if not isinstance(row, list):
            return None, f"bad-return: row {r} is {type(row).__name__}, expected list"
        if len(row) != want_cols:
            return None, f"bad-return: row {r} has {len(row)} cells, expected {want_cols}"
        for c, v in enumerate(row):
            if not isinstance(v, str):
                return None, (f"bad-return: cell ({r},{c}) is {type(v).__name__} "
                              f"({v!r}), expected a color-name string")
        grid.append(list(row))
    return grid, None


# ---------------------------------------------------------------------------
# ProgramRuntime: persistent worker process per program
# ---------------------------------------------------------------------------
def _worker_main(conn, code: str):
    """Worker loop: exec the program once, then answer (prev, grid, action)
    requests forever. All exceptions become ('err', msg) replies; the parent
    enforces deadlines and kills/restarts us on hangs."""
    load_err = None
    fn = None
    try:
        ns: dict = {}
        exec(code, ns)  # noqa: S102 -- learned artifact, same trust model as run_perceive
        fn = ns.get("transition")
        if not callable(fn):
            load_err = "program has no callable transition(prev, grid, action)"
    except Exception:
        load_err = "program failed to load:\n" + traceback.format_exc(limit=3)
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg[0] == "exit":
            return
        _, prev, grid, action = msg
        if load_err is not None:
            conn.send(("err", load_err))
            continue
        try:
            out = fn(prev, grid, action)
        except Exception:
            conn.send(("err", traceback.format_exc(limit=3)))
            continue
        grid_out, err = _validate_grid(out, len(grid), len(grid[0]) if grid else 0)
        conn.send(("ok", grid_out) if err is None else ("err", err))


class ProgramRuntime:
    """Executes one program's `transition` in a persistent child process.

    Per-call deadline via Connection.poll; on timeout/crash the worker is
    killed and lazily respawned on the next call, so a program that hangs on
    one input can still be scored on the rest of the buffer. Thread-safe
    (single in-flight request, guarded by a lock)."""

    def __init__(self, code: str, timeout_s: float = 1.0):
        self.code = code
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._ctx = mp.get_context("fork")
        self._proc = None
        self._conn = None
        self.n_calls = 0
        self.n_timeouts = 0

    def _ensure_worker(self):
        if self._proc is not None and self._proc.is_alive():
            return
        self._kill()
        self._conn, child = self._ctx.Pipe()
        self._proc = self._ctx.Process(
            target=_worker_main, args=(child, self.code), daemon=True)
        self._proc.start()
        child.close()

    def _kill(self):
        if self._proc is not None:
            try:
                self._proc.kill()
                self._proc.join(timeout=1.0)
            except Exception:
                pass
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        self._proc = None
        self._conn = None

    def transition(self, prev: list, grid: Grid, action: Action):
        """-> (next_grid, None) or (None, error). error is 'timeout' | a
        traceback | a bad-return description."""
        with self._lock:
            self.n_calls += 1
            self._ensure_worker()
            try:
                self._conn.send(("call", prev, grid, action))
            except (BrokenPipeError, OSError):
                self._kill()
                return None, "worker-dead: could not send request"
            if not self._conn.poll(self.timeout_s):
                self.n_timeouts += 1
                self._kill()
                return None, f"timeout: transition exceeded {self.timeout_s}s"
            try:
                kind, payload = self._conn.recv()
            except (EOFError, OSError):
                self._kill()
                return None, "worker-crashed: process died during call"
            return (payload, None) if kind == "ok" else (None, payload)

    def score_buffer(self, items: list) -> list:
        """Run T-hat over prepared transitions -> list[ItemResult]."""
        results = []
        for it in items:
            pred, err = self.transition(it.prev, it.grid, it.action)
            if err is not None:
                results.append(ItemResult(False, 0.0, it.changed, None, err, False))
                continue
            pred_c = canon_grid(pred)
            results.append(ItemResult(
                exact=pred_c == it.next_c,
                cell_f1=cell_f1(pred, it.next_grid),
                changed=it.changed,
                pred_canon=pred_c,
                error=None,
                identity_pred=pred_c == it.grid_c,
            ))
        return results

    def close(self):
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.send(("exit",))
                except Exception:
                    pass
            self._kill()

    def __del__(self):  # best-effort cleanup
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Prepared buffer + scoring statistics
# ---------------------------------------------------------------------------
@dataclass
class PreparedTransition:
    prev: list          # [(Grid, Action)], oldest->newest, <= context_k entries
    grid: Grid
    next_grid: Grid
    action: Action
    action_str: str
    grid_c: str
    next_c: str
    changed: bool
    src: object = None  # the originating validate.Transition (for backprompts)
    idx: int = -1       # stable id within its buffer (trace-cache key)


@dataclass
class ItemResult:
    exact: bool
    cell_f1: float
    changed: bool
    pred_canon: str | None
    error: str | None
    identity_pred: bool


def prepare_transitions(transitions: list, context_k: int) -> list:
    """Parse validate.Transition objects once. ctx_prev entries are
    (state_raw, outgoing_action) pairs oldest->newest (validate.py:98-104);
    we keep the last `context_k` of them as the program's `prev` window."""
    prepared = []
    for i, tr in enumerate(transitions):
        try:
            grid = parse_grid_strict(tr.x_t)
            next_grid = parse_grid_strict(tr.x_t1)
            prev = [(parse_grid_strict(s), parse_action(a))
                    for s, a in (tr.ctx_prev[-context_k:] if context_k > 0 else [])]
        except ValueError as e:
            raise ValueError(f"transition {i}: {e}") from e
        gc, nc = canon_grid(grid), canon_grid(next_grid)
        prepared.append(PreparedTransition(
            prev=prev, grid=grid, next_grid=next_grid,
            action=parse_action(tr.action), action_str=tr.action,
            grid_c=gc, next_c=nc, changed=gc != nc, src=tr))
    return prepared


def cell_f1(pred: Grid, true: Grid) -> float:
    """Per-cell match fraction (0.0 on dimension mismatch). Feedback-only
    partial credit -- never part of the bandit's h (ϕ1 stays exact-match)."""
    if len(pred) != len(true) or any(len(p) != len(t) for p, t in zip(pred, true)):
        return 0.0
    total = sum(len(r) for r in true)
    if total == 0:
        return 0.0
    hits = sum(1 for pr, tr_ in zip(pred, true) for a, b in zip(pr, tr_) if a == b)
    return hits / total


def grid_diff_cells(a: Grid, b: Grid) -> list:
    """[(r, c, a_val, b_val)] where the two grids differ (matching dims only)."""
    if len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b)):
        return [(-1, -1, f"{len(a)}x{len(a[0]) if a else 0}",
                 f"{len(b)}x{len(b[0]) if b else 0}")]
    return [(r, c, a[r][c], b[r][c])
            for r in range(len(a)) for c in range(len(a[r])) if a[r][c] != b[r][c]]


def render_diff(diff: list, labels=("predicted", "true"), max_cells: int = 40) -> str:
    """NL cell-diff lines for backprompts: \"cell (3,4): predicted 'black' but
    true 'gold'\". Empty diff -> 'no differences'."""
    if not diff:
        return "no differences"
    if diff[0][0] == -1:
        return f"grid dimensions differ: {labels[0]} {diff[0][2]} vs {labels[1]} {diff[0][3]}"
    lines = [f"cell ({r},{c}): {labels[0]} '{av}' but {labels[1]} '{bv}'"
             for r, c, av, bv in diff[:max_cells]]
    if len(diff) > max_cells:
        lines.append(f"... and {len(diff) - max_cells} more differing cells")
    return "; ".join(lines)


def fit_stats(results: list) -> dict:
    """Aggregate ϕ1 statistics incl. the identity floor and the degeneracy flag."""
    n = len(results)
    changed = [r for r in results if r.changed]
    static = [r for r in results if not r.changed]
    frac = lambda rs: (sum(r.exact for r in rs) / len(rs)) if rs else 0.0  # noqa: E731
    n_err = sum(1 for r in results if r.error is not None)
    n_to = sum(1 for r in results if r.error is not None and r.error.startswith("timeout"))
    return {
        "n": n, "n_changed": len(changed), "n_static": len(static),
        "fit_all": frac(results),
        "fit_changed": frac(changed),
        "fit_static": frac(static),
        # identity program fits every static transition and no changed one, so
        # under balanced_score it lands at exactly 0.5 (0.0 on all-changed buffers):
        "identity_floor_balanced": 0.5 if static else 0.0,
        "cell_f1_changed": (sum(r.cell_f1 for r in changed) / len(changed)) if changed else 0.0,
        "crash_rate": (n_err - n_to) / n if n else 0.0,
        "timeout_rate": n_to / n if n else 0.0,
        "all_identity_on_changed": bool(changed) and all(r.identity_pred or r.error is not None
                                                         for r in changed),
    }


def balanced_score(stats: dict) -> float:
    """h = 0.5*fit_changed + 0.5*fit_static (identity program lands at 0.5)."""
    return 0.5 * stats["fit_changed"] + 0.5 * stats["fit_static"]


def determinism_ceiling(prepared: list, k: int) -> float:
    """Max exact-fit any DETERMINISTIC function of the k-window can reach on
    this buffer: group transitions by (window tail, grid, action) and credit
    each group its modal outcome. <1.0 exposes stochastic games (and, at small
    k, hidden state)."""
    groups = defaultdict(Counter)
    for p in prepared:
        tail = tuple((canon_grid(g), unparse_action(a))
                     for g, a in (p.prev[-k:] if k > 0 else []))
        groups[(tail, p.grid_c, p.action_str)][p.next_c] += 1
    n = len(prepared)
    return sum(max(c.values()) for c in groups.values()) / n if n else 0.0


# ---------------------------------------------------------------------------
# Planning primitives over T-hat (zero LLM)
# ---------------------------------------------------------------------------
def build_action_universe(verbs: list, grid: Grid, goal: Grid | None = None) -> list:
    """Expand verb names into Action tuples. 'click' expands over all cells,
    ordered for beam efficiency: cells differing from the goal first, then
    non-background cells (background = most common color -- mirrors
    click_enum's non-background-first ordering), then the rest."""
    actions = [(v, None, None) for v in verbs if v != "click"]
    if "click" not in verbs:
        return actions
    rows, cols = len(grid), len(grid[0]) if grid else 0
    counts = Counter(v for row in grid for v in row)
    background = counts.most_common(1)[0][0] if counts else ""
    def rank(rc):
        r, c = rc
        if goal is not None and len(goal) == rows and goal[r][c] != grid[r][c]:
            return 0
        return 1 if grid[r][c] != background else 2
    cells = sorted(((r, c) for r in range(rows) for c in range(cols)), key=rank)
    return actions + [("click", r, c) for r, c in cells]


def _tail(seq: list, k: int) -> tuple:
    return tuple(seq[-k:]) if k > 0 else ()


def plan_search(rt: ProgramRuntime, history: list, start: Grid, goal: Grid,
                action_universe: list, h: int, *, beam: int = 64,
                node_budget: int = 4000, context_k: int = 9,
                allow_empty: bool = True):
    """Search an action sequence (<= h) whose T-hat rollout reaches `goal`.

    BFS when the branching factor is small (<= 8 actions); beam search
    (heuristic = cells matching goal) when clicks blow it up. Dedup on
    (grid canon, depth) -- depth stays in the key because passive/periodic
    dynamics make the same grid at different times non-equivalent. This makes
    the search slightly conservative for history-sensitive programs (nodes
    with identical grids but different windows collapse); acceptable for v1.
    All T-hat calls are memoized on (window, grid, action).

    allow_empty: when True (default) a start that already equals the goal
    short-circuits to the zero-length plan []. Pass False when the caller
    scores "the grid after the FINAL action" -- a zero-length plan has no
    final action and cannot be scored, so the search must instead find a
    >=1-step plan that HOLDS the goal (noop, if the passive dynamics are
    static here). Callers must distinguish [] from None: `if found is None`,
    never `if not found`.

    Returns the action list (possibly empty, see allow_empty), or None
    (unreachable within h / budget exhausted)."""
    goal_c = canon_grid(goal)
    if allow_empty and canon_grid(start) == goal_c:
        return []
    memo: dict = {}
    calls = 0

    def step(hist, hist_key, grid, grid_c, action):
        nonlocal calls
        key = (grid_c, hist_key, unparse_action(action))
        if key in memo:
            return memo[key]
        if calls >= node_budget:
            return (None, "budget")
        calls += 1
        out = rt.transition(list(hist), grid, action)
        memo[key] = out
        return out

    use_beam = len(action_universe) > 8
    hist0 = _tail(history, context_k)
    hkey0 = tuple((canon_grid(g), unparse_action(a)) for g, a in hist0)
    # node: (grid, grid_c, hist, hist_key, path)
    frontier = [(start, canon_grid(start), hist0, hkey0, ())]
    seen = {(frontier[0][1], 0)}
    for depth in range(1, h + 1):
        nxt = []
        for grid, grid_c, hist, hkey, path in frontier:
            for action in action_universe:
                pred, err = step(hist, hkey, grid, grid_c, action)
                if err == "budget":
                    break
                if err is not None or pred is None:
                    continue
                pc = canon_grid(pred)
                if pc == goal_c:
                    return list(path) + [action]
                if (pc, depth) in seen:
                    continue
                seen.add((pc, depth))
                astr = unparse_action(action)
                nxt.append((pred, pc,
                            _tail(list(hist) + [(grid, action)], context_k),
                            _tail(list(hkey) + [(grid_c, astr)], context_k),
                            path + (action,)))
            if calls >= node_budget:
                break
        if use_beam and len(nxt) > beam:
            nxt.sort(key=lambda nd: -cell_f1(nd[0], goal))
            nxt = nxt[:beam]
        frontier = nxt
        if not frontier or calls >= node_budget:
            break
    return None


def rollout(rt: ProgramRuntime, history: list, start: Grid, actions: list,
            *, context_k: int = 9) -> list:
    """Closed-loop h-step rollout of T-hat (predictions fed back in).
    Returns one Grid per action; None from the first failed step onward."""
    hist = list(history)[-context_k:] if context_k > 0 else []
    grid = start
    out = []
    failed = False
    for a in actions:
        if failed:
            out.append(None)
            continue
        pred, err = rt.transition(hist, grid, a)
        if err is not None or pred is None:
            out.append(None)
            failed = True
            continue
        hist = (hist + [(grid, a)])[-context_k:] if context_k > 0 else []
        grid = pred
        out.append(pred)
    return out
