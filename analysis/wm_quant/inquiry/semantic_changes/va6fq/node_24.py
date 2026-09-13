"""
Contract: perceive(observation_history) -> str must never raise.
Output a concise (<2000 char) text summary of decision-relevant features.
Parse the grid from the last observation, detect background colour, and
list all non-background cells in a compact format.  Also compute the
difference (delta) from the previous observation if available, making
the action directly recoverable.  Include a monotonically increasing
step counter to disambiguate identical grid configurations at different times.
"""

import json
from collections import Counter

# Global step counter to provide a unique id for each call within the same process.
_step_counter = 0

def _parse_grid(obs: str):
    """Extract the 2D colour grid as list[list[str]] (rows of colour-name strings).

    Anchor on the generic '[[' / ']]' so it works regardless of the header text.
    Returns None on failure (caller degrades gracefully, never raises)."""
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

def _determine_background(grid):
    """Return the colour that appears most frequently in the grid."""
    if not grid or not grid[0]:
        return None
    counter = Counter()
    for row in grid:
        counter.update(row)
    return counter.most_common(1)[0][0]

def _cells_set(grid, bg):
    """Return a set of (colour, row, col) tuples for non-background cells."""
    cells = set()
    for r, row in enumerate(grid):
        for c, colour in enumerate(row):
            if colour != bg:
                cells.add((colour, r, c))
    return cells

def _format_cells(cells):
    """Format a set of (colour, row, col) tuples into a compact string.

    Group by colour, then list coordinates.
    """
    by_colour = {}
    for colour, r, c in cells:
        by_colour.setdefault(colour, []).append((r, c))
    parts = []
    for colour in sorted(by_colour.keys()):
        coords = ";".join(f"{r},{c}" for r, c in sorted(by_colour[colour]))
        parts.append(f"{colour}:{coords}")
    return "; ".join(parts)

def perceive(observation_history: list[str]) -> str:
    global _step_counter
    _step_counter += 1
    step = _step_counter  # use the global counter as a unique step id

    # Get current observation
    if not observation_history:
        return f"step={step} error: empty history"

    current_obs = observation_history[-1]
    current_grid = _parse_grid(current_obs)
    if current_grid is None:
        return f"step={step} error: current grid parse failed"

    bg = _determine_background(current_grid)
    if bg is None:
        return f"step={step} error: empty grid"

    rows = len(current_grid)
    cols = len(current_grid[0]) if rows > 0 else 0
    dims = f"{rows}x{cols}"

    current_cells = _cells_set(current_grid, bg)

    # Compute delta if we have a previous observation
    delta_parts = []
    if len(observation_history) >= 2:
        prev_obs = observation_history[-2]
        prev_grid = _parse_grid(prev_obs)
        if prev_grid is not None:
            prev_bg = _determine_background(prev_grid)
            if prev_bg is not None:
                prev_cells = _cells_set(prev_grid, prev_bg)
                added = current_cells - prev_cells
                removed = prev_cells - current_cells
                if added or removed:
                    add_str = _format_cells(added) if added else ""
                    rem_str = _format_cells(removed) if removed else ""
                    d = "delta:"
                    if add_str:
                        d += f"added:{add_str}"
                    if add_str and rem_str:
                        d += " "
                    if rem_str:
                        d += f"removed:{rem_str}"
                    delta_parts.append(d)

    # Build current state summary
    if current_cells:
        cells_str = _format_cells(current_cells)
    else:
        cells_str = "none"

    # Combine parts
    parts = [f"step={step}", f"bg={bg}", f"dims={dims}"] + delta_parts + [f"cells:{cells_str}"]
    output = " ".join(parts)

    # Ensure it is never empty (should be impossible)
    if not output:
        output = f"step={step} bg={bg} dims={dims} cells:none"

    # Truncate if needed, preferring to keep step, bg, dims, delta and a compact cell count
    if len(output) > 1999:
        count = len(current_cells)
        short_cells = f"cells:{count}_non_bg"
        short_parts = [f"step={step}", f"bg={bg}", f"dims={dims}"] + delta_parts + [short_cells]
        output = " ".join(short_parts)
        # If still too long, further trim delta parts
        if len(output) > 1999:
            if len(delta_parts) > 1:
                delta_parts = delta_parts[:1]
            output = " ".join([f"step={step}", f"bg={bg}", f"dims={dims}"] + delta_parts + [short_cells])
            if len(output) > 1999:
                output = f"step={step} bg={bg} dims={dims} cells:{count}_non_bg"
    return output