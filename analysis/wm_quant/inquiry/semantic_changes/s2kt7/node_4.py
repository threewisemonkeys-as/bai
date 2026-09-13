"""
Contract: perceive(observation_history) -> str must never raise, never return empty.
Outputs a concise (<2000 char) summary of decision-relevant features, including
delta from the previous state if available, to make both inverse and forward
prediction reliable.
"""

import json
from collections import Counter
from typing import Optional, List, Tuple


def _parse_grid(obs: str) -> Optional[List[List[str]]]:
    """Extract the 2D colour grid as list[list[str]].

    Anchor on the generic '[[' / ']]' so it works regardless of the header text.
    Returns None on failure.
    """
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
    """Return the most frequent colour in the grid (assumed background)."""
    flat = [cell for row in grid for cell in row]
    if not flat:
        return "black"
    count = Counter(flat)
    background, _ = count.most_common(1)[0]
    return background


def _non_bg_cells(grid: List[List[str]], background: str) -> List[Tuple[int, int, str]]:
    """Return sorted list of (row, col, colour) for cells not background."""
    cells = []
    for r, row in enumerate(grid):
        for c, colour in enumerate(row):
            if colour != background:
                cells.append((r, c, colour))
    cells.sort(key=lambda x: (x[0], x[1]))
    return cells


def _format_cells(cells: List[Tuple[int, int, str]]) -> str:
    """Format a list of cells as a compact string: (r,c,colour), ..."""
    return ", ".join(f"({r},{c},{colour})" for r, c, colour in cells)


def perceive(observation_history: list[str]) -> str:
    # Ensure we have at least one observation
    if not observation_history:
        return "empty"

    # Parse current grid
    cur_obs = observation_history[-1]
    cur_grid = _parse_grid(cur_obs)
    if cur_grid is None:
        return "no_grid"

    # Determine background
    background = _determine_background(cur_grid)
    rows = len(cur_grid)
    cols = len(cur_grid[0]) if rows > 0 else 0

    # Get current non-background cells
    cur_non_bg = _non_bg_cells(cur_grid, background)

    # Try to get previous state for delta
    prev_non_bg = None
    if len(observation_history) >= 2:
        prev_obs = observation_history[-2]
        prev_grid = _parse_grid(prev_obs)
        if prev_grid is not None:
            # Use same background (should be same world, but recompute to be safe)
            prev_bg = _determine_background(prev_grid)
            prev_non_bg = _non_bg_cells(prev_grid, prev_bg)

    # Compute deltas if possible
    added = []
    removed = []
    changed = []  # (r, c, from_colour, to_colour)
    if prev_non_bg is not None:
        # Use sets for fast lookup
        cur_set = set((r, c, colour) for r, c, colour in cur_non_bg)
        prev_set = set((r, c, colour) for r, c, colour in prev_non_bg)

        # Cells in current but not in previous -> added
        for cell in cur_non_bg:
            r, c, colour = cell
            if (r, c, colour) not in prev_set:
                # Check if cell existed with different colour
                old_colour = None
                for (pr, pc, pc_col) in prev_non_bg:
                    if pr == r and pc == c:
                        old_colour = pc_col
                        break
                if old_colour is not None:
                    changed.append((r, c, old_colour, colour))
                else:
                    added.append((r, c, colour))

        # Cells in previous but not in current -> removed
        for cell in prev_non_bg:
            r, c, colour = cell
            if (r, c, colour) not in cur_set:
                # Check if it changed colour (already handled)
                still_there = any(pr == r and pc == c for pr, pc, _ in cur_non_bg)
                if not still_there:
                    removed.append((r, c, colour))

    # Build output parts (keep total length <2000, but we'll be generous)
    parts = []

    # Background and dimensions
    parts.append(f"bg:{background}; rows:{rows}; cols:{cols}")

    # Current cells
    if cur_non_bg:
        parts.append("cells:" + _format_cells(cur_non_bg))
    else:
        parts.append("cells:empty")

    # Deltas
    if added:
        parts.append("added:" + _format_cells(added))
    if removed:
        parts.append("removed:" + _format_cells(removed))
    if changed:
        changed_str = ", ".join(f"({r},{c},{old}->{new})" for r, c, old, new in changed)
        parts.append("changed:" + changed_str)

    # Combine
    result = "; ".join(parts)

    # Truncate if necessary (should be rare)
    if len(result) > 1999:
        # Fallback: just list counts and first/last few cells
        result = f"bg:{background}; rows:{rows}; cols:{cols}; cells_count={len(cur_non_bg)}"
        if cur_non_bg:
            first = _format_cells(cur_non_bg[:5])
            last = _format_cells(cur_non_bg[-5:])
            result += f"; first5:{first}; last5:{last}"
        if added:
            result += f"; added_count={len(added)}"
        if removed:
            result += f"; removed_count={len(removed)}"
        if changed:
            result += f"; changed_count={len(changed)}"

    # Never return empty
    return result if result else "empty"