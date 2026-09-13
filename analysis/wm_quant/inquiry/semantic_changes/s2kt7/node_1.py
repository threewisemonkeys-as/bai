"""
Contract: perceive(observation_history) -> str must never raise.
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


def _determine_background(grid):
    """Return the most frequent colour in the grid (assumed to be the background)."""
    # Flatten all cells
    flat = [cell for row in grid for cell in row]
    if not flat:
        return "black"  # fallback, should not happen
    count = Counter(flat)
    # Most common colour
    background, _ = count.most_common(1)[0]
    return background


def perceive(observation_history: list[str]) -> str:
    obs = observation_history[-1] if observation_history else ""
    grid = _parse_grid(obs)
    if grid is None:
        return "no_grid"

    background = _determine_background(grid)

    non_bg = []  # list of (row, col, colour)
    for r, row in enumerate(grid):
        for c, colour in enumerate(row):
            if colour != background:
                non_bg.append((r, c, colour))

    if not non_bg:
        return "empty"

    # Sort to ensure deterministic order
    non_bg.sort(key=lambda x: (x[0], x[1]))

    # Build concise text (<2000 chars)
    parts = [f"({r},{c},{colour})" for r, c, colour in non_bg]
    result = "cells: " + "; ".join(parts)

    # Truncate if necessary (shouldn't happen with typical grid sizes)
    if len(result) > 1999:
        # Keep first and last important info, but better to keep all if possible;
        # as a fallback, count and summarise
        result = f"cells_count={len(non_bg)}"
        # Append first few and last few for discriminability
        first = "; ".join(parts[:5])
        last = "; ".join(parts[-5:])
        result += f" first5: {first} last5: {last}"

    return result