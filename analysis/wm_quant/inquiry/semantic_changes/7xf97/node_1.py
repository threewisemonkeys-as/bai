"""
Contract: perceive(observation_history) -> str must never raise.
Outputs a concise text summary of decision-relevant features (<2000 chars).
Never returns empty; if state has no objects, returns "no objects".
"""

import json
from collections import Counter


def _parse_grid(obs: str):
    """Extract the 2D colour grid as list[list[str]] (rows of colour-name strings).

    Anchor on the generic '[[' / ']]' so it works regardless of the header text that precedes
    the grid. Returns None on failure (caller degrades gracefully, never raises)."""
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


def _most_common_colour(grid):
    """Return the most frequent cell colour (background)."""
    flat = [cell for row in grid for cell in row]
    if not flat:
        return "black"  # fallback
    counter = Counter(flat)
    # most_common returns list of (value, count) sorted descending
    return counter.most_common(1)[0][0]


def perceive(observation_history: list[str]) -> str:
    obs = observation_history[-1] if observation_history else ""
    try:
        grid = _parse_grid(obs)
        if grid is None:
            return "parse_failed"

        bg = _most_common_colour(grid)

        # Collect non-background cells
        objects = []
        for r, row in enumerate(grid):
            for c, colour in enumerate(row):
                if colour != bg:
                    objects.append((r, c, colour))

        if not objects:
            return "no_objects"

        # Build summary: sort by (row, col) for determinism
        objects.sort()
        parts = []
        for r, c, colour in objects:
            parts.append(f"{r},{c}:{colour}")
        # Join with spaces; if total > 2000, truncate with count indicator
        summary = " ".join(parts)
        if len(summary) > 1900:
            # Keep first ~1900 chars and indicate truncation
            summary = summary[:1900] + f" ... ({len(parts)} cells total)"
        return summary
    except Exception:
        # Fallback: never raise, never return empty
        return "error"