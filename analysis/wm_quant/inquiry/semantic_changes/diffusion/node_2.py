"""
Contract: perceive(observation_history) -> str must never raise.
"""

import json
from collections import Counter

def _parse_grid(obs: str):
    """Extract the 2D colour grid as list[list[str]].

    Anchor on the generic '[[' / ']]' so it works regardless of header text.
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
    """Produce a concise summary of non-background cells in the current grid.

    The summary is robust to parse errors and never returns an empty string.
    Format: "bg:<dominant_colour>; <colour>:(r1,c1)(r2,c2)... <colour>:(...) ..."
    If no non‑background cells exist, the string ends with 'none'.
    """
    obs = observation_history[-1] if observation_history else ""
    grid = _parse_grid(obs)
    if grid is None:
        # Graceful fallback – never empty, never raises
        return "parse_error"

    # Count colours to find the background (most frequent)
    colour_counts = Counter()
    for row in grid:
        for cell in row:
            colour_counts[cell] += 1
    if not colour_counts:
        return "empty_grid"
    background = colour_counts.most_common(1)[0][0]

    # Collect non-background cells
    objects = {}        # colour -> list of (r, c)
    for r, row in enumerate(grid):
        for c, colour in enumerate(row):
            if colour != background:
                objects.setdefault(colour, []).append((r, c))

    # Build the summary string
    parts = [f"bg:{background}"]
    if not objects:
        parts.append("none")
    else:
        # Group by colour for readability
        for colour in sorted(objects):
            cells = objects[colour]
            # Compact representation: (r,c) concatenated
            cell_str = "".join(f"({r},{c})" for r, c in cells)
            parts.append(f"{colour}:{cell_str}")

    result = "; ".join(parts)
    # Ensure the output is never empty (in case of unexpected structural error)
    if not result:
        result = "empty_summary"
    return result