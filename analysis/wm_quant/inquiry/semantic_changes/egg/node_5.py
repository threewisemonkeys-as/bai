"""
Contract: perceive(observation_history) -> str must never raise, never return empty,
and produce a text summary that changes whenever the grid changes so that the action
between two consecutive states can be recovered from the feature trajectory.
"""

import json
from collections import Counter


def _parse_grid(obs: str):
    """Extract the 2D colour grid as list[list[str]] (rows of colour-name strings).

    Anchor on the generic '[[' / ']]' so it works regardless of the header text that precedes
    the grid. Do NOT gate on the observation *starting* with '[[': Autumn observations are
    prefixed by a "Task:/Step:/..." header, so the grid begins mid-string. Returns None on
    failure (caller degrades gracefully, never raises)."""
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


def _find_background(grid):
    """Return the most common colour in the grid (the background)."""
    all_colours = []
    for row in grid:
        all_colours.extend(row)
    if not all_colours:
        return "gray"  # fallback
    counter = Counter(all_colours)
    return counter.most_common(1)[0][0]


def _colour_sort_key(item):
    """Sort key for (r, c, colour) – use row then column then colour."""
    return item[0], item[1], item[2]


def perceive(observation_history: list[str]) -> str:
    obs = observation_history[-1] if observation_history else ""
    grid = _parse_grid(obs)
    if grid is None:
        # On parse failure, return a non‑empty fallback that is unlikely to be constant
        # (include a hash of the raw text so it changes if the raw text changes).
        return f"parse_error_{hash(obs) & 0xFFFFFF}"

    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    background = _find_background(grid)

    # Collect all non‑background cells (row, col, colour)
    objects = []
    for r, row in enumerate(grid):
        for c, colour in enumerate(row):
            if colour != background:
                objects.append((r, c, colour))

    # Sort cells for determinism
    objects.sort(key=_colour_sort_key)

    # Build the summary string
    # Format: "HxW; bg:<background>; cells: (r,c,colour) (r,c,colour) ..."
    # Always include dimensions so that absolute positions are interpretable.
    dim_part = f"{rows}x{cols}"
    bg_part = f"bg:{background}"
    parts = [dim_part, bg_part]

    if objects:
        obj_strs = [f"({r},{c},{colour})" for r, c, colour in objects]
        full_cells = "cells: " + " ".join(obj_strs)
        summary = "; ".join(parts) + "; " + full_cells

        # Keep total length under 2000 characters – truncate if needed, but this is unlikely
        # because objects are normally sparse.
        if len(summary) > 1990:
            # Truncate object list until under limit
            while len(summary) > 1990 and obj_strs:
                obj_strs.pop()
                full_cells = "cells: " + " ".join(obj_strs)
                summary = "; ".join(parts) + "; " + full_cells
            # Add a marker that list was truncated
            summary += " [truncated]"
        return summary
    else:
        # No non‑background objects
        return f"{dim_part}; {bg_part}; no objects"