"""
Contract: perceive(observation_history) -> str must never raise, never return empty.
Outputs a concise (<2000 char) summary of decision-relevant features, including
a unique state ID (hash of non‑background cells), step counter (from history length),
and correct delta (added/removed/changed) between consecutive states.
"""

import json
from collections import Counter
from hashlib import sha256
from typing import Optional, List, Tuple


def _parse_grid(obs: str) -> Optional[List[List[str]]]:
    """Extract the 2D colour grid as list[list[str]]."""
    if not obs:
        return None
    start = obs.find("[[")
    end = obs.rfind("]]")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        grid = json.loads(obs[start : end + 2])
    except Exception:
        return None
    if not grid or not isinstance(grid, list) or not isinstance(grid[0], list):
        return None
    return grid


def _determine_background(grid: List[List[str]]) -> str:
    """Most frequent colour in the grid (assumed background)."""
    flat = [cell for row in grid for cell in row]
    if not flat:
        return "black"
    count = Counter(flat)
    background, _ = count.most_common(1)[0]
    return background


def _non_bg_cells(grid: List[List[str]], background: str) -> List[Tuple[int, int, str]]:
    """Sorted list of (row, col, colour) for cells not background."""
    cells = []
    for r, row in enumerate(grid):
        for c, colour in enumerate(row):
            if colour != background:
                cells.append((r, c, colour))
    cells.sort(key=lambda x: (x[0], x[1]))
    return cells


def _format_cells(cells: List[Tuple[int, int, str]]) -> str:
    """Compact string: (r,c,colour), ..."""
    return ", ".join(f"({r},{c},{colour})" for r, c, colour in cells)


def _state_id(cur_non_bg: List[Tuple[int, int, str]], background: str, rows: int, cols: int) -> str:
    """Short hash that (almost always) differs for different grid configurations."""
    data = f"{background}:{rows}x{cols}:" + str(cur_non_bg)
    return sha256(data.encode()).hexdigest()[:8]


def perceive(observation_history: list[str]) -> str:
    # --------- - step and parsing ----------
    if not observation_history:
        return "empty"

    step = len(observation_history) - 1   # 0‑based

    cur_obs = observation_history[-1]
    cur_grid = _parse_grid(cur_obs)
    if cur_grid is None:
        return f"step:{step}; parse_error"

    background = _determine_background(cur_grid)
    rows = len(cur_grid)
    cols = len(cur_grid[0]) if rows > 0 else 0

    cur_non_bg = _non_bg_cells(cur_grid, background)
    cur_set = set(cur_non_bg)

    # --------- - previous state for delta ----------
    prev_non_bg = None
    prev_set = set()
    if step >= 1:
        prev_obs = observation_history[-2]
        prev_grid = _parse_grid(prev_obs)
        if prev_grid is not None:
            prev_bg = _determine_background(prev_grid)
            prev_non_bg = _non_bg_cells(prev_grid, prev_bg)
            prev_set = set(prev_non_bg)

    # --------- - compute deltas ----------
    added = list(cur_set - prev_set)
    removed = list(prev_set - cur_set)
    # cells that stayed at the same coordinates but changed colour
    changed: List[Tuple[int, int, str, str]] = []
    if prev_non_bg:
        prev_by_pos = {(r, c): colour for r, c, colour in prev_non_bg}
        cur_by_pos = {(r, c): colour for r, c, colour in cur_non_bg}
        for (r, c), old in prev_by_pos.items():
            new = cur_by_pos.get((r, c))
            if new is not None and new != old:
                changed.append((r, c, old, new))

    # sort consistently
    added.sort(key=lambda x: (x[0], x[1]))
    removed.sort(key=lambda x: (x[0], x[1]))
    changed.sort(key=lambda x: (x[0], x[1]))

    # --------- - build output ----------
    sid = _state_id(cur_non_bg, background, rows, cols)
    parts = [
        f"step:{step}",
        f"sid:{sid}",
        f"bg:{background}",
        f"rows:{rows}",
        f"cols:{cols}",
        "cells:" + (_format_cells(cur_non_bg) if cur_non_bg else "empty"),
        "added:" + (_format_cells(added) if added else "empty"),
        "removed:" + (_format_cells(removed) if removed else "empty"),
    ]
    if changed:
        changed_str = ", ".join(f"({r},{c},{old}->{new})" for r, c, old, new in changed)
        parts.append("changed:" + changed_str)
    else:
        parts.append("changed:empty")

    result = "; ".join(parts)

    # --------- - guarantee length <2000 ----------
    if len(result) > 1999:
        # very compact fallback
        result = (
            f"step:{step};sid:{sid};bg:{background};"
            f"rows:{rows};cols:{cols};"
            f"cells_cnt={len(cur_non_bg)}"
        )
        if added:
            result += f";add_cnt={len(added)}"
        if removed:
            result += f";rem_cnt={len(removed)}"
        if changed:
            result += f";chg_cnt={len(changed)}"

    return result if result else "empty"