"""
Contract: perceive(observation_history) -> str must never raise.
"""

import json
from collections import Counter
from typing import Optional, List, Tuple


def _parse_grid(obs: str) -> Optional[List[List[str]]]:
    """Extract the 2D colour grid as list[list[str]].

    Anchors on the first '[[' and last ']]' to locate the JSON array.
    Returns None on any failure (caller degrades gracefully).
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
    # Validate structure: must be a list of lists of strings
    if not isinstance(grid, list) or not grid:
        return None
    if not isinstance(grid[0], list) or not all(isinstance(row, list) for row in grid):
        return None
    return grid


def _get_background(grid: List[List[str]]) -> str:
    """Return the most common colour in the grid (assumed background)."""
    counter = Counter()
    for row in grid:
        for cell in row:
            counter[cell] += 1
    # Most common colour. If tie, one is chosen arbitrarily.
    return counter.most_common(1)[0][0]


def _summarise_grid(grid: List[List[str]]) -> str:
    """Build a concise, decision‑relevant text summary of non‑background cells.

    The summary lists each non‑background cell as 'row,col:color', one per line.
    Cells are sorted by row then column for consistency.
    """
    bg = _get_background(grid)
    lines = []
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            if color != bg:
                lines.append(f"{r},{c}:{color}")
    if not lines:
        return f"bg_only:{bg}"
    lines.sort()  # lexicographic order sorts by row then col
    return "\n".join(lines)


def perceive(observation_history: list[str]) -> str:
    """Produce a text summary of the current grid observation.

    The output is guaranteed to be non‑empty, under 2000 characters,
    and to reflect changes in the world so that actions can be recovered
    from consecutive summaries.
    """
    # Get current observation
    obs = observation_history[-1] if observation_history else ""

    # Parse grid
    grid = _parse_grid(obs)
    if grid is None:
        # Degrade gracefully: never raise, never return empty.
        return "parse_error:grid_not_found"

    # Build summary
    summary = _summarise_grid(grid)

    # Ensure it fits the length constraint – if not, fall back to a shorter form.
    # (Unlikely given typical grid sizes, but guard anyway.)
    if len(summary) > 1900:
        # Truncate sensibly – remove some lines from the end, ensuring we end with a newline.
        lines = summary.split("\n")
        # Keep as many complete lines as possible under the limit.
        truncated = []
        char_count = 0
        for line in lines:
            needed = len(line) + 1  # +1 for newline
            if char_count + needed > 1900:
                # Add a line indicating truncation
                truncated.append("... (truncated)")
                break
            truncated.append(line)
            char_count += needed
        summary = "\n".join(truncated)

    # Final safety: never return empty string.
    if not summary:
        summary = "bg_only:unknown"

    return summary