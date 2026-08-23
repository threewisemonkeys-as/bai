"""Prototype: answer environment questions by SYNTHESISING CODE over the
structured trajectory, instead of having an LLM eyeball observations.

Hypothesis under test: an LLM is more reliable at writing a predicate that
inspects the structured (integer-grid) trajectory than at reading the same
trajectory and emitting a verdict directly. We test this offline on the saved
may29 ft09 logs by:

  1. Parsing trajectory_buffer.json into structured StepRecords (pre/post 64x64
     grids as numpy arrays, parsed action + coords, reward, counters).
  2. For each existing QA pair, asking the LLM to synthesise
        def answer(steps: list[StepRecord]) -> str   # "YES" | "NO" | "MAYBE"
  3. Executing the predicate over the real trajectory.
  4. Comparing the code's verdict to the stored LLM (log-reading) answer.

This file is the reusable substrate; run_prototype.py drives it.
"""

from __future__ import annotations

import re
import textwrap
import traceback
from dataclasses import dataclass

import numpy as np

# ARC-AGI 3 canonical palette indices -> human color name (from arc_agi_env.py).
COLORS: dict[int, str] = {
    0: "white",
    1: "off-white",
    2: "light-gray",
    3: "gray",
    4: "dark-gray",
    5: "black",
    6: "magenta",
    7: "pink",
    8: "red",
    9: "blue",
    10: "light-blue",
    11: "yellow",
    12: "orange",
    13: "maroon",
    14: "green",
    15: "purple",
}


@dataclass
class StepRecord:
    """One environment transition, fully structured (no prose, no images).

    Coordinates follow the action convention ``ACTION6 x=<col> y=<row>``, so the
    clicked cell is ``pre[y, x]`` (numpy is row-major: ``pre[row, col]``).
    """

    step: int
    action_type: str | None          # e.g. "ACTION6"
    x: int | None                     # column of the click, or None
    y: int | None                     # row of the click, or None
    pre: np.ndarray | None            # 64x64 int grid BEFORE the action
    post: np.ndarray | None           # 64x64 int grid AFTER the action
    reward: float
    state: str | None                 # "NOT_FINISHED" | "WIN" | ...
    levels_completed: int | None      # parsed from "Levels completed: a/b"
    action_count: int | None

    # ---- convenience helpers a synthesised predicate may call ----
    def clicked_cell_pre(self) -> int | None:
        if self.pre is None or self.x is None or self.y is None:
            return None
        if 0 <= self.y < self.pre.shape[0] and 0 <= self.x < self.pre.shape[1]:
            return int(self.pre[self.y, self.x])
        return None

    def clicked_cell_post(self) -> int | None:
        if self.post is None or self.x is None or self.y is None:
            return None
        if 0 <= self.y < self.post.shape[0] and 0 <= self.x < self.post.shape[1]:
            return int(self.post[self.y, self.x])
        return None

    def changed_cells(self) -> list[tuple[int, int, int, int]]:
        """Return (row, col, old_val, new_val) for every cell that changed."""
        if self.pre is None or self.post is None or self.pre.shape != self.post.shape:
            return []
        rows, cols = np.where(self.pre != self.post)
        return [
            (int(r), int(c), int(self.pre[r, c]), int(self.post[r, c]))
            for r, c in zip(rows, cols)
        ]

    def any_change(self) -> bool:
        return len(self.changed_cells()) > 0


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_GRID_RE = re.compile(r"<grid_(\d+)>(.*?)</grid_(?:\d+)>", re.DOTALL)
_ROW_RE = re.compile(r"\[([0-9,\s]+)\]")


def _parse_grid(text: str | None) -> np.ndarray | None:
    """Parse the LAST <grid_*> layer in *text* into a 2-D int numpy array."""
    if not text:
        return None
    blocks = _GRID_RE.findall(text)
    if not blocks:
        return None
    body = blocks[-1][1]  # body of the last <grid_*> layer
    rows = []
    for m in _ROW_RE.finditer(body):
        vals = [int(v) for v in m.group(1).split(",") if v.strip() != ""]
        if vals:
            rows.append(vals)
    if not rows:
        return None
    width = max(len(r) for r in rows)
    rows = [r for r in rows if len(r) == width]  # drop ragged rows defensively
    try:
        return np.array(rows, dtype=int)
    except ValueError:
        return None


_ACTION_RE = re.compile(r"(ACTION\d+)(?:\s+x=(-?\d+)\s+y=(-?\d+))?", re.IGNORECASE)
_LEVELS_RE = re.compile(r"Levels completed:\s*(\d+)\s*/\s*(\d+)")
_STATE_RE = re.compile(r"State:\s*(\S+)")
_ACTCOUNT_RE = re.compile(r"Action count:\s*(\d+)")


def _parse_action(action: str | None) -> tuple[str | None, int | None, int | None]:
    if not action:
        return None, None, None
    m = _ACTION_RE.search(action)
    if not m:
        return None, None, None
    atype = m.group(1).upper()
    x = int(m.group(2)) if m.group(2) is not None else None
    y = int(m.group(3)) if m.group(3) is not None else None
    return atype, x, y


def _parse_counters(short_ctx: str | None) -> tuple[str | None, int | None, int | None]:
    if not short_ctx:
        return None, None, None
    state = (m.group(1) if (m := _STATE_RE.search(short_ctx)) else None)
    levels = (int(m.group(1)) if (m := _LEVELS_RE.search(short_ctx)) else None)
    actc = (int(m.group(1)) if (m := _ACTCOUNT_RE.search(short_ctx)) else None)
    return state, levels, actc


def parse_trajectory(buffer: list[dict]) -> list[StepRecord]:
    """Convert a trajectory_buffer.json list into StepRecords (action steps only)."""
    records: list[StepRecord] = []
    for e in buffer:
        if e.get("episode_boundary") or e.get("action") is None:
            continue
        atype, x, y = _parse_action(e.get("action"))
        state, levels, _ = _parse_counters(e.get("result_raw_short_term_context") or e.get("raw_short_term_context"))
        _, _, actc = _parse_counters(e.get("raw_short_term_context"))
        records.append(
            StepRecord(
                step=e.get("step", -1),
                action_type=atype,
                x=x,
                y=y,
                pre=_parse_grid(e.get("raw_long_term_context")),
                post=_parse_grid(e.get("result_raw_long_term_context")),
                reward=float(e.get("reward", 0.0) or 0.0),
                state=state,
                levels_completed=levels,
                action_count=actc,
            )
        )
    return records


# ---------------------------------------------------------------------------
# Predicate execution (sandbox-ish): a throwing predicate => "MAYBE"
# ---------------------------------------------------------------------------

PREDICATE_API_DOC = '''\
You will write a Python function with this exact signature:

    def answer(steps: list[StepRecord]) -> str:
        ...
        return "YES" | "NO" | "MAYBE"

`steps` is the chronological list of environment transitions. Each StepRecord has:
  .step              int    transition index
  .action_type       str    e.g. "ACTION6"  (the only available action here)
  .x, .y             int    clicked column (x) and row (y), or None
  .pre               np.ndarray  64x64 int grid BEFORE the action  (pre[row, col] == pre[y, x])
  .post              np.ndarray  64x64 int grid AFTER the action
  .reward            float
  .state             str    "NOT_FINISHED" | "WIN" | ...
  .levels_completed  int    parsed from "Levels completed: a/b" (the `a`)
  .action_count      int

Helper methods on StepRecord:
  .clicked_cell_pre()  -> int | None    value at the clicked cell before the action
  .clicked_cell_post() -> int | None    value at the clicked cell after the action
  .changed_cells()     -> list[(row, col, old_val, new_val)]   all cells that changed
  .any_change()        -> bool          True iff the action changed any cell

Grid values are color indices. COLORS maps index -> name and is in scope:
  0 white, 2 light-gray, 4 dark-gray, 5 black(background), 8 red, 9 blue, 12 orange, ...
numpy is available as `np`.

RULES:
- Inspect the structured data only. Do NOT hard-code the verdict; compute it.
- Return "YES"/"NO" only when the trajectory contains transitions that decide it.
- Return "MAYBE" when no transition in `steps` provides evidence either way
  (honest abstention), or when the evidence is mixed/contradictory.
- Be robust: guard against None grids and out-of-range coords.

Return ONLY the function body inside a single ```python ... ``` block. No prose.
'''


def extract_code(text: str) -> str | None:
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m:
        return textwrap.dedent(m.group(1)).strip()
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if m:
        return textwrap.dedent(m.group(1)).strip()
    if "def answer" in text:
        return text.strip()
    return None


def run_predicate(code: str, steps: list[StepRecord]) -> tuple[str, str | None]:
    """Compile + run a synthesised predicate. Returns (verdict, error).

    Any exception (compile or runtime) maps to verdict "MAYBE" with the error
    captured — abstention, never a crash.
    """
    ns: dict = {"StepRecord": StepRecord, "COLORS": COLORS, "np": np}
    try:
        exec(compile(code, "<predicate>", "exec"), ns)
    except Exception:
        return "MAYBE", f"compile error:\n{traceback.format_exc()}"
    fn = ns.get("answer")
    if not callable(fn):
        return "MAYBE", "no callable `answer` defined"
    try:
        out = fn(steps)
    except Exception:
        return "MAYBE", f"runtime error:\n{traceback.format_exc()}"
    if not isinstance(out, str):
        return "MAYBE", f"non-string return: {out!r}"
    v = out.strip().upper()
    if v not in {"YES", "NO", "MAYBE"}:
        return "MAYBE", f"invalid verdict: {out!r}"
    return v, None
