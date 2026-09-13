"""
Contract: perceive(observation_history) -> str must never raise.
Output a concise (<2000 char) text summary that lists ALL cell positions
grouped by colour, so the action taken between two consecutive states is
recoverable even when the background colour changes between worlds.
"""

import json
from collections import defaultdict


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
    """Produce a concise summary of every cell in the current grid.

    The summary is robust to parse errors and never returns an empty string.
    Format: "<colour>:<r,c> <r,c> ... ; <colour>:<r,c> ..."
    All colours present are reported (including the background).
    If the grid is empty, returns "empty_grid".
    """
    obs = observation_history[-1] if observation_history else ""
    grid = _parse_grid(obs)
    if grid is None:
        return "parse_error"

    # Group all cells by colour
    objects = defaultdict(list)   # colour -> [(r, c), ...]
    for r, row in enumerate(grid):
        for c, colour in enumerate(row):
            objects[colour].append((r, c))

    # Build compact representation: colour:r1,c1 r2,c2 ...
    parts = []
    # Sort colours for deterministic output (helps consistency, not required)
    for colour in sorted(objects):
        cells = objects[colour]
        # Use space-separated "r,c" to save characters over "(r,c)"
        cell_str = " ".join(f"{r},{c}" for r, c in cells)
        parts.append(f"{colour}:{cell_str}")

    result = "; ".join(parts)

    # Safety: never return an empty string
    if not result:
        result = "empty_grid"

    # If the result exceeds the 2000-char limit, fall back to a minimal
    # representation that still includes all positions but in a more compressed
    # way.  (In practice the grid size is moderate; this is a safety net.)
    if len(result) > 2000:
        # Rebuild using a shorter inter-cell separator and no spaces after commas
        parts2 = []
        for colour in sorted(objects):
            cells = objects[colour]
            # Use "r,c" without any separator yields "rc"?  No, we need commas.
            # Use "r,c" concatenated: e.g., "1,12,3" ambiguous.
            # Safer: use "|" as separator and no spaces: "1,2|3,4"
            cell_str = "|".join(f"{r},{c}" for r, c in cells)
            parts2.append(f"{colour}:{cell_str}")
        result = ";".join(parts2)
        # If still too long, truncate at the last whole colour block?
        # But this should rarely happen. We'll simply keep the shortened form.
        if len(result) > 2000:
            # Last resort: output a summary that still contains the background
            # colour and the number of cells per colour, but this loses positions.
            # Better than nothing – the downstream can at least see the counts.
            counts = {c: len(pts) for c, pts in objects.items()}
            result = "bg:" + max(counts, key=counts.get) + "; " + \
                     "; ".join(f"{c}:{n}" for c, n in sorted(counts.items()))
            # This is guaranteed to be short.
    return result