"""
Contract: perceive(observation_history) -> str must never raise, never return empty.
Output a concise (<2000 chars) summary of decision-relevant features.
Summarise all non‑background cells as (row,col,colour) so that differences
between consecutive states are visible and actions can be inferred.
"""

import json
from collections import Counter


def _parse_grid(obs: str):
    """Extract the 2D colour grid as list[list[str]].

    Finds the JSON array inside the observation text (between the first '[['
    and the last ']]') and parses it.  Returns None on any failure.
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
    # basic validation: list of lists of strings
    if not isinstance(grid, list) or not grid:
        return None
    for row in grid:
        if not isinstance(row, list):
            return None
    return grid


def _get_background(grid):
    """Return the most common colour in the grid (the background)."""
    counter = Counter()
    for row in grid:
        for cell in row:
            counter[cell] += 1
    # most_common returns list of (colour, count); pick the first
    bg, _ = counter.most_common(1)[0]
    return bg


def _summarise_non_background(grid, bg):
    """Return a string listing all foreground cells.

    Format:  bg=<colour>; cells: (r,c,colour);(r,c,colour);...
    If no foreground cells:  "bg=<colour>; cells: none"
    """
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    cells = []
    for r in range(rows):
        for c in range(cols):
            colour = grid[r][c]
            if colour != bg:
                cells.append((r, c, colour))

    if not cells:
        return f"bg={bg}; cells: none"

    # Build a compact list
    parts = [f"bg={bg}; cells:"]
    for r, c, colour in cells:
        parts.append(f"({r},{c},{colour})")
    return ";".join(parts)


def perceive(observation_history: list[str]) -> str:
    """Return a text summary of the current observation.

    The summary is guaranteed to be non‑empty and to change whenever the
    underlying grid changes, enabling inverse dynamics and forward prediction.
    """
    # Fallback – never return empty string
    fallback = "parse_error: could not interpret observation"

    if not observation_history:
        return fallback

    obs = observation_history[-1]
    grid = _parse_grid(obs)
    if grid is None:
        return fallback

    try:
        bg = _get_background(grid)
        summary = _summarise_non_background(grid, bg)
    except Exception:
        # Defensive: any unexpected error → fallback
        return fallback

    # Ensure the result is never empty (in case of empty bg or weird grid)
    if not summary:
        return fallback

    # Enforce length limit – unlikely to be needed, but safe
    if len(summary) > 2000:
        summary = summary[:1997] + "..."

    return summary