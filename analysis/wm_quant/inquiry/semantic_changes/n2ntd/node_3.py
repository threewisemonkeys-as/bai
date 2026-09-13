"""
Contract: perceive(observation_history) -> str must never raise.
Returns a concise text summary of decision-relevant features from the grid.
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


def _dominant_color(grid):
    """Return the most frequent colour in the grid (the background)."""
    flat = [cell for row in grid for cell in row]
    if not flat:
        return "black"  # fallback
    counter = Counter(flat)
    # most_common returns list of (color, count), first is highest
    return counter.most_common(1)[0][0]


def perceive(observation_history: list[str]) -> str:
    obs = observation_history[-1] if observation_history else ""
    grid = _parse_grid(obs)
    if grid is None:
        return "error: could not parse grid"

    try:
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        bg = _dominant_color(grid)

        # Collect all non-background cells
        cells = []
        for r in range(rows):
            for c in range(cols):
                color = grid[r][c]
                if color != bg:
                    cells.append((r, c, color))

        # Build concise summary
        # format: "grid NxN; bg:c; cells: (r,c,color); ..."
        parts = [f"grid {rows}x{cols}; bg:{bg}; cells:"]
        for r, c, clr in cells:
            parts.append(f"({r},{c},{clr})")
        summary = "".join(parts)  # no separator needed between cells, but we can use ';'
        # Actually adding ';' between cells for readability, but note char limit.
        # Let's assemble with ';' after cells
        parts_alt = [f"grid {rows}x{cols}; bg:{bg}; cells:"]
        for r, c, clr in cells:
            parts_alt.append(f"({r},{c},{clr});")
        summary = "".join(parts_alt)
        # Remove trailing ';'
        if summary.endswith(';'):
            summary = summary[:-1]

        # Enforce length limit (<2000)
        if len(summary) > 1995:
            # Truncate from cells list, keeping as many as fit
            summary = f"grid {rows}x{cols}; bg:{bg}; cells:"
            remaining = 1995 - len(summary) - 3  # reserve for "..."
            count = 0
            for r, c, clr in cells:
                entry = f"({r},{c},{clr});"
                if len(entry) > remaining:
                    break
                summary += entry
                remaining -= len(entry)
                count += 1
            if count < len(cells):
                summary += "..."

        if not summary:
            # Should never happen, but safety
            summary = f"grid {rows}x{cols}; bg:{bg}; cells:"

        return summary

    except Exception:
        # Never raise; return a non-empty fallback
        return "perception error"