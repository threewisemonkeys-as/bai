"""
Contract: perceive(observation_history) -> str must never raise.
Outputs a compact summary of the grid: background colour, dimensions,
agent position, colour counts, and sorted list of non-background cells.
This rich representation makes state transitions uniquely identifiable,
supporting both inverse dynamics and forward prediction.
"""

import json
from collections import Counter

def _parse_grid(obs: str):
    """Extract the 2D colour grid as list[list[str]] (rows of colour-name strings).

    Anchor on the generic '[[' / ']]' so it works regardless of header text.
    Returns None on failure (caller degrades gracefully)."""
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
    """Return the most frequent colour in the grid (the background)."""
    colours = [cell for row in grid for cell in row]
    if not colours:
        return "black"
    counter = Counter(colours)
    return counter.most_common(1)[0][0]

def perceive(observation_history: list[str]) -> str:
    obs = observation_history[-1] if observation_history else ""
    grid = _parse_grid(obs)
    if grid is None:
        return "parse error, could not extract grid"

    try:
        background = _determine_background(grid)
    except Exception:
        background = "black"

    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    agent_pos = None
    other_cells = []
    colour_counts = Counter()   # count of non-background cells by colour

    try:
        for r, row in enumerate(grid):
            for c, colour in enumerate(row):
                if colour == background:
                    continue
                if colour == "orange":
                    agent_pos = (r, c)
                else:
                    other_cells.append((r, c, colour))
                    colour_counts[colour] += 1
    except Exception:
        return "grid iteration error"

    # Sort other cells deterministically
    other_cells.sort(key=lambda x: (x[0], x[1]))

    # Build output parts
    parts = []
    # Background and dimensions
    parts.append(f"bg={background} rows={rows} cols={cols}")

    if agent_pos is not None:
        parts.append(f"agent row={agent_pos[0]} col={agent_pos[1]}")

    # Colour counts (compact)
    if colour_counts:
        counts_str = " ".join(f"{col}:{cnt}" for col, cnt in sorted(colour_counts.items()))
        parts.append(f"colors: {counts_str}")

    # List of other cells (truncated if needed)
    if other_cells:
        cell_strs = [f"({r},{c},{col})" for r, c, col in other_cells]
        combined = "; ".join(cell_strs)
        if len(combined) > 1800:  # leave room for other parts
            max_shown = 40
            truncated = cell_strs[:max_shown]
            truncated_str = "; ".join(truncated)
            parts.append(f"others ({len(cell_strs)} total, first {max_shown}): {truncated_str}")
        else:
            parts.append(f"others: {combined}")

    # Fallback if nothing found
    if not parts:
        return f"background only: {background}"

    result = " | ".join(parts)
    # Final length safety
    if len(result) > 1997:
        result = result[:1997] + "..."
    return result if result else "empty summary"