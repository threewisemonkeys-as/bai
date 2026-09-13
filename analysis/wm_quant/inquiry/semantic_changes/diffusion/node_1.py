"""
Contract: perceive(observation_history) -> str must never raise.
Produces a concise (<2000 char) text summary of decision-relevant features.
The summary captures the positions and colours of every non-background cell,
allowing the action between consecutive states to be inferred from the changes.
"""

import json
from collections import Counter


def _parse_grid(obs: str):
    """Extract the 2D colour grid as list[list[str]].

    Locates the outermost [[ ... ]] that encloses the JSON array,
    regardless of surrounding header text. Returns None on failure.
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


def _determine_background(grid):
    """Return the most frequent cell colour (the background)."""
    colours = [cell for row in grid for cell in row]
    if not colours:
        return "black"  # fallback
    counter = Counter(colours)
    # Most common colour is the background
    return counter.most_common(1)[0][0]


def _summarise_grid(grid):
    """Build a compact string describing all non‑background cells.

    Format: "bg:<background>; <cell1> <cell2> ..."
    where each cell is (row,col,colour).
    Cells are sorted by (row, col) for determinism.
    """
    background = _determine_background(grid)
    cells = []
    for r, row in enumerate(grid):
        for c, colour in enumerate(row):
            if colour != background:
                cells.append((r, c, colour))
    # Sort for consistent output
    cells.sort(key=lambda x: (x[0], x[1]))
    # Build the string
    parts = [f"bg:{background};"]
    for r, c, colour in cells:
        parts.append(f"({r},{c},{colour})")
    return " ".join(parts)


def perceive(observation_history: list[str]) -> str:
    """Return a text summary of the current grid observation."""
    obs = observation_history[-1] if observation_history else ""
    grid = _parse_grid(obs)
    if grid is None:
        return "bg:unknown; parse_error"
    try:
        summary = _summarise_grid(grid)
    except Exception:
        # Safety net – never raise, never return empty
        return "bg:unknown; summary_error"
    # Ensure non-empty (even if grid has no non‑bg cells)
    if not summary:
        return "bg:unknown; empty_grid"
    return summary