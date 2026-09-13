"""
Contract: perceive(observation_history) -> str must never raise.
Outputs a compact summary of non-background cells and the agent's row/col,
enabling action recovery from consecutive states.
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
    # Flatten grid into a single list of colour names
    colours = [cell for row in grid for cell in row]
    if not colours:
        return "black"  # fallback
    counter = Counter(colours)
    # Most common colour is the background
    return counter.most_common(1)[0][0]


def perceive(observation_history: list[str]) -> str:
    obs = observation_history[-1] if observation_history else ""
    grid = _parse_grid(obs)
    if grid is None:
        return "parse error, could not extract grid"

    try:
        background = _determine_background(grid)
    except Exception:
        # If we can't determine background, treat as 'black' as last resort
        background = "black"

    agent_pos = None          # (row, col) of the "orange" cell (assumed agent)
    other_cells = []          # list of (row, col, colour) for non-background, non-orange

    try:
        for r, row in enumerate(grid):
            for c, colour in enumerate(row):
                if colour == background:
                    continue
                if colour == "orange":
                    # There should be only one orange cell; keep last one found
                    agent_pos = (r, c)
                else:
                    other_cells.append((r, c, colour))
    except Exception:
        # defensive: if iteration fails, return a minimal summary
        return "grid error"

    # Sort other cells by row then col for deterministic output
    other_cells.sort(key=lambda x: (x[0], x[1]))

    # Build the output string, keeping it under 2000 characters
    parts = []
    if agent_pos is not None:
        parts.append(f"agent row={agent_pos[0]} col={agent_pos[1]}")
    if other_cells:
        # Format other cells compactly
        cell_strs = [f"({r},{c},{col})" for r, c, col in other_cells]
        # If too long, truncate and indicate count
        combined = "; ".join(cell_strs)
        if len(combined) > 1900:  # leave some room for the rest
            # Show first few and note the total number
            max_shown = 50
            truncated = cell_strs[:max_shown]
            truncated_str = "; ".join(truncated)
            parts.append(f"others ({len(cell_strs)} total, first {max_shown}): {truncated_str}")
        else:
            parts.append(f"others: {combined}")

    # If nothing was found (all background or only orange, etc.)
    if not parts:
        return f"background only: {background}"

    result = " | ".join(parts)
    # Final safety: ensure not too long
    if len(result) > 2000:
        result = result[:1997] + "..."
    # Never return empty – the above should always produce something
    return result if result else "no features"