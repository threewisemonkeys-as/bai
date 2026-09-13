"""
Contract: perceive(observation_history) -> str must never raise.
Output a concise (<2000 char) text summary that lists the background colour,
grid dimensions, and every non‑background cell position grouped by colour,
so the action taken between two consecutive states is recoverable.
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
    Format: "bg:<colour>; dim:<rows>x<cols>; <colour>:<r,c> <r,c> ... ; ..."
    If the summary would exceed 2000 characters, a more compact representation
    is used (underscore separators, no spaces). If still too long, only counts
    are returned.
    """
    obs = observation_history[-1] if observation_history else ""
    grid = _parse_grid(obs)
    if grid is None:
        return "parse_error"

    rows = len(grid)
    if rows == 0:
        return "empty_grid"
    cols = len(grid[0]) if grid[0] else 0
    if cols == 0:
        return "empty_grid"

    # Count all colours to determine background
    all_colours = [cell for row in grid for cell in row]
    if not all_colours:
        return "empty_grid"
    bg = Counter(all_colours).most_common(1)[0][0]

    # Group non‑background cells by colour
    objects = defaultdict(list)   # colour -> [(r, c), ...]
    for r, row in enumerate(grid):
        for c, colour in enumerate(row):
            if colour != bg:
                objects[colour].append((r, c))

    # Build the summary
    parts = []
    for colour in sorted(objects):
        cells = objects[colour]
        cell_str = " ".join(f"{r},{c}" for r, c in cells)
        parts.append(f"{colour}:{cell_str}")

    body = "; ".join(parts)
    result = f"bg:{bg}; dim:{rows}x{cols}; {body}"

    # If the result is too long, compress format
    if len(result) > 2000:
        parts2 = []
        for colour in sorted(objects):
            cells = objects[colour]
            # use underscore and comma to save space
            cell_str = ",".join(f"{r}_{c}" for r, c in cells)
            parts2.append(f"{colour}:{cell_str}")
        body2 = ";".join(parts2)
        result = f"bg:{bg};dim:{rows}x{cols};{body2}"

        if len(result) > 2000:
            # fallback to counts only – lossy but safe
            counts = {c: len(pts) for c, pts in objects.items()}
            count_str = ";".join(f"{c}:{n}" for c, n in sorted(counts.items()))
            result = f"bg:{bg};dim:{rows}x{cols};counts:{count_str}"

    # Ensure never empty
    if not result:
        result = "empty_grid"
    return result