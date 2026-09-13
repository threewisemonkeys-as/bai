"""
Contract: perceive(observation_history) -> str must never raise.
Output a concise (<2000 char) text summary that lists ONLY non‑background cell
positions grouped by colour, so the action taken between two consecutive states
is recoverable even when the background colour changes between worlds.
The background colour is determined as the most frequent colour in the grid.
"""

import json
from collections import defaultdict, Counter


def _parse_grid(obs: str):
    """Extract the 2D colour grid as list[list[str]].

    Anchors on the generic '[[' / ']]' to work regardless of header text.
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


def perceive(observation_history: list[str]) -> str:
    """Produce a summary of every non‑background cell in the current grid.

    The summary is robust to parse errors and never returns an empty string.
    Format: "<colour>:<r,c> <r,c> ... ; <colour>:<r,c> ..."
    The background colour is determined dynamically and excluded.
    If the grid is empty or parse fails, returns a non‑empty fallback.
    """
    obs = observation_history[-1] if observation_history else ""
    grid = _parse_grid(obs)
    if grid is None:
        return "parse_error"

    # Determine the background colour (most frequent)
    all_colours = [cell for row in grid for cell in row]
    if not all_colours:
        return "empty_grid"
    bg_colour = Counter(all_colours).most_common(1)[0][0]

    # Group only non‑background cells by colour
    objects = defaultdict(list)   # colour -> [(r, c), ...]
    for r, row in enumerate(grid):
        for c, colour in enumerate(row):
            if colour != bg_colour:
                objects[colour].append((r, c))

    # Build compact representation: colour:r1,c1 r2,c2 ...
    parts = []
    for colour in sorted(objects):
        cells = objects[colour]
        cell_str = " ".join(f"{r},{c}" for r, c in cells)
        parts.append(f"{colour}:{cell_str}")

    result = "; ".join(parts)

    if not result:
        # No non‑background cells at all
        result = "empty_grid"
    elif len(result) > 2000:
        # In the unlikely event it's too long, switch to a more compressed
        # separator and drop spaces.
        parts2 = []
        for colour in sorted(objects):
            cells = objects[colour]
            cell_str = "|".join(f"{r},{c}" for r, c in cells)
            parts2.append(f"{colour}:{cell_str}")
        result = ";".join(parts2)
        # If still too long, fall back to counts only (lossy but safe)
        if len(result) > 2000:
            counts = {c: len(pts) for c, pts in objects.items()}
            result = "bg:" + bg_colour + "; " + \
                     "; ".join(f"{c}:{n}" for c, n in sorted(counts.items()))
    return result