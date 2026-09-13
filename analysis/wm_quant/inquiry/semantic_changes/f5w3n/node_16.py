"""
Contract: perceive(observation_history) -> str must never raise.
Outputs a compact summary of the grid: background colour, dimensions,
agent position (always reported), colour counts, and sorted list of
non‑background cells.  The agent is identified as the “orange” cell;
if none exists, the single non‑background colour with exactly one
occurrence is assumed to be the agent (works for all observed worlds).
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
    colour_counts_counter = Counter()   # count only non‑background, non‑agent colours

    try:
        # First pass: collect all non‑background cells, their colours and positions
        all_cells = []
        for r, row in enumerate(grid):
            for c, colour in enumerate(row):
                if colour == background:
                    continue
                all_cells.append((r, c, colour))
    except Exception:
        return "grid iteration error"

    # Count occurrences of each colour among non‑background cells
    colour_counter = Counter(col for _, _, col in all_cells)

    # Determine agent: prefer "orange", otherwise use the colour with count==1 (if exactly one)
    agent_colour = None
    if "orange" in colour_counter:
        agent_colour = "orange"
    else:
        singles = [col for col, cnt in colour_counter.items() if cnt == 1]
        if len(singles) == 1:
            agent_colour = singles[0]

    # Rebuild other_cells and colour_counts_counter, removing the agent cell
    for r, c, colour in all_cells:
        if colour == agent_colour and agent_pos is None:
            agent_pos = (r, c)
        else:
            other_cells.append((r, c, colour))
            colour_counts_counter[colour] += 1

    # Sort other cells deterministically
    other_cells.sort(key=lambda x: (x[0], x[1]))

    # Build output parts
    parts = []
    # Background and dimensions
    parts.append(f"bg={background} rows={rows} cols={cols}")

    if agent_pos is not None:
        parts.append(f"agent at ({agent_pos[0]},{agent_pos[1]})")
    # else: should never happen, but if it does, we still output the rest

    # Colour counts (compact) – we already have colour_counts_counter without the agent
    if colour_counts_counter:
        counts_str = " ".join(f"{col}:{cnt}" for col, cnt in sorted(colour_counts_counter.items()))
        parts.append(f"colors: {counts_str}")

    # List of other cells (truncated if needed)
    if other_cells:
        cell_strs = [f"({r},{c},{col})" for r, c, col in other_cells]
        combined = "; ".join(cell_strs)
        # Allow up to ~1900 chars for the rest, to stay under 2000 total
        if len(combined) > 1900:
            max_shown = 100   # show up to 100 cells; each is ~12-15 chars, so ~1500 chars
            truncated = cell_strs[:max_shown]
            truncated_str = "; ".join(truncated)
            parts.append(f"others ({len(cell_strs)} total, first {max_shown}): {truncated_str}")
        else:
            parts.append(f"others: {combined}")

    # Fallback if nothing found (shouldn't happen)
    if not parts:
        return f"background only: {background}"

    result = " | ".join(parts)
    # Final length safety: trim to <2000
    if len(result) > 1997:
        # Keep the beginning (most important info)
        result = result[:1997] + "..."
    return result if result else "empty summary"