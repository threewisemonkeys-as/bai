"""
Perception module for grid environment.
Produces a concise summary of all non-background cells.
"""

import json
from collections import Counter


def _determine_bg(grid):
    """Return the most frequent colour in the grid (the background)."""
    flat = [cell for row in grid for cell in row]
    if not flat:
        return "black"          # fallback
    return Counter(flat).most_common(1)[0][0]


def _format_cell(r, c, colour):
    """Compact representation of a single cell: 'r,c:colour'."""
    return f"{r},{c}:{colour}"


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

    # Collect all non‑background cells
    cells = []
    for r, row in enumerate(grid):
        for c, colour in enumerate(row):
            if colour != bg:
                cells.append(_format_cell(r, c, colour))

    # Sort for consistency
    # _format_cell strings sort lexicographically by row, then col (since 'r,' is numeric)
    cells.sort()

    # Build summary – under 2000 chars guaranteed for typical grid sizes
    summary = f"bg={bg}; cells=" + " ".join(cells) if cells else f"bg={bg}; cells=none"
    # Final safety: ensure non‑empty and under limits
    if not summary:
        summary = "empty"
    # Truncate if absolutely necessary (should not happen)
    if len(summary) > 1999:
        summary = summary[:1996] + "..."
    return summary