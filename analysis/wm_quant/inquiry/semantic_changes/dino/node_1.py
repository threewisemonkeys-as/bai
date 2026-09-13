"""
Contract: perceive(observation_history) -> str must never raise, never return empty.
Parses the raw grid observation as a JSON 2D array of colour names, determines the
background colour (most frequent cell), and returns a compact summary of all non‑background
cells (row, col, colour). This ensures that any movement of objects is visible in the
feature trajectory.
"""

import json
from collections import Counter


def _parse_grid(obs: str):
    """Extract and parse the 2D colour grid from the observation string.

    Returns the grid as list[list[str]] on success, or None on failure.
    """
    if not obs:
        return None
    start = obs.find("[[")
    end = obs.rfind("]]")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        grid = json.loads(obs[start: end + 2])
    except Exception:
        return None
    if not grid or not isinstance(grid, list) or not isinstance(grid[0], list):
        return None
    return grid


def _background_colour(grid):
    """Return the colour that appears most frequently in the grid (the background)."""
    flat = [cell for row in grid for cell in row]
    if not flat:
        return "black"
    # most_common returns the first element if tie; fine for background
    return Counter(flat).most_common(1)[0][0]


def perceive(observation_history: list[str]) -> str:
    # Get current observation
    if not observation_history:
        return "empty"   # fallback, never empty
    obs = observation_history[-1]

    grid = _parse_grid(obs)
    if grid is None:
        return "parse_error"   # safe non‑empty fallback

    bg = _background_colour(grid)

    # Collect all non‑background cells
    features = []
    for r, row in enumerate(grid):
        for c, colour in enumerate(row):
            if colour != bg:
                features.append((r, c, colour))

    # Sort for deterministic ordering
    features.sort(key=lambda x: (x[0], x[1]))

    # Build compact string
    parts = []
    for r, c, colour in features:
        parts.append(f"{r},{c}:{colour}")

    if not parts:
        return "empty"

    summary = ";".join(parts)
    # Ensure length <2000 (very unlikely to be exceeded, but safeguard)
    if len(summary) > 1995:
        summary = summary[:1992] + "..."
    return summary