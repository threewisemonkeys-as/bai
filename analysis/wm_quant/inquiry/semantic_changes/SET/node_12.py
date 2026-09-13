import json
from collections import Counter

# This module produces a concise summary of all non‑background cells,
# including white (empty) tile cells, the grid shape, and the cursor
# position (when available).  The cursor is extracted from the raw
# observation string if present (e.g. "cursor: 5,10"); otherwise it is
# reported as "cursor:unknown".  By keeping every non‑background cell,
# the features are sufficiently rich to distinguish states and to allow
# the inverse‑dynamics predictor to uniquely identify the action.

def _determine_bg(grid):
    """Return the most frequent colour in the grid (the background)."""
    flat = [cell for row in grid for cell in row]
    if not flat:
        return "black"          # fallback
    return Counter(flat).most_common(1)[0][0]

def _parse_grid(obs: str):
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

def _parse_cursor(obs: str):
    """Extract cursor coordinates from the observation, if present.
    Returns (row, col) or None if no cursor info is given."""
    # Look for a pattern like "cursor: 3,14" in the raw string
    import re
    match = re.search(r"cursor:\s*(\d+)\s*,\s*(\d+)", obs)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None

def _format_cell(r, c, colour):
    """Compact representation: 'r,c:colour'."""
    return f"{r},{c}:{colour}"

def perceive(observation_history: list[str]) -> str:
    obs = observation_history[-1] if observation_history else ""
    grid = _parse_grid(obs)
    if grid is None:
        return "empty"   # never empty, never raise

    # Determine background colour
    try:
        bg = _determine_bg(grid)
    except Exception:
        bg = "black"     # safe fallback

    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    # Collect all cells that are NOT background (include white and any other colours)
    cells = []
    for r, row in enumerate(grid):
        for c, colour in enumerate(row):
            if colour != bg:
                cells.append(_format_cell(r, c, colour))

    # Sort numerically by row, then column
    cells.sort(key=lambda s: (int(s.split(',')[0]),
                              int(s.split(',')[1].split(':')[0])))

    # Get cursor position if available
    cursor = _parse_cursor(obs)
    if cursor is not None:
        cursor_str = f"cursor:{cursor[0]},{cursor[1]}"
    else:
        cursor_str = "cursor:unknown"

    # Build summary
    shape = f"{rows}x{cols}"
    if cells:
        summary = f"bg={bg}; shape={shape}; {cursor_str}; " + " ".join(cells)
    else:
        summary = f"bg={bg}; shape={shape}; {cursor_str}; none"

    # Final safety: ensure non‑empty and under limits
    if not summary:
        summary = "empty"
    if len(summary) > 1999:
        summary = summary[:1996] + "..."
    return summary