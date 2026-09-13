"""
Contract: perceive(observation_history) -> str must never raise.
Returns a concise text summary of decision-relevant features from the grid,
including the exact positions of all non‑background cells to enable unique
state identification and accurate inverse dynamics prediction.
"""

import json
from collections import Counter
from typing import List, Tuple, Optional


def _parse_grid(obs: str) -> Optional[List[List[str]]]:
    """Extract the 2D colour grid as list[list[str]] (rows of colour-name strings)."""
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


def _dominant_colour(grid: List[List[str]]) -> str:
    """Return the most frequent colour in the grid (the background)."""
    flat = [cell for row in grid for cell in row]
    if not flat:
        return "black"
    return Counter(flat).most_common(1)[0][0]


def perceive(observation_history: List[str]) -> str:
    # Get current observation
    obs = observation_history[-1] if observation_history else ""
    grid = _parse_grid(obs)
    if grid is None:
        return "grid parse error; no features"

    try:
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        bg = _dominant_colour(grid)

        # Collect all non‑background cells, grouped by colour
        cells_by_colour = {}   # colour -> list of (row, col) tuples
        for r in range(rows):
            for c in range(cols):
                colour = grid[r][c]
                if colour != bg:
                    cells_by_colour.setdefault(colour, []).append((r, c))

        # Build the summary string, sorted by colour for consistency
        summary_parts = [f"grid {rows}x{cols}; bg:{bg};"]
        # Sort colours alphabetically to ensure deterministic output
        for colour in sorted(cells_by_colour.keys()):
            positions = cells_by_colour[colour]
            # Sort positions by (row, col) for readability and deterministic order
            positions_sorted = sorted(positions, key=lambda p: (p[0], p[1]))
            # Format: "colour: r1,c1 r2,c2 ..."
            coords_str = " ".join(f"{r},{c}" for r, c in positions_sorted)
            summary_parts.append(f"{colour}: {coords_str}")

        summary = " ".join(summary_parts)

        # Enforce length < 2000
        if len(summary) > 1995:
            # Truncate by dropping colours from the end, but keep as many as fit
            prefix = f"grid {rows}x{cols}; bg:{bg};"
            remaining = 1995 - len(prefix) - 4  # reserve " ..."
            colours_sorted = sorted(cells_by_colour.keys())
            re_summary = prefix
            for colour in colours_sorted:
                positions = sorted(cells_by_colour[colour], key=lambda p: (p[0], p[1]))
                coords_str = " ".join(f"{r},{c}" for r, c in positions)
                entry = f" {colour}: {coords_str}"
                if len(re_summary) + len(entry) > remaining:
                    re_summary += " ..."
                    break
                re_summary += entry
            summary = re_summary

        if not summary:
            summary = f"grid {rows}x{cols}; bg:{bg};"

        return summary

    except Exception:
        # Never raise; return a non‑empty fallback
        return "perception error"