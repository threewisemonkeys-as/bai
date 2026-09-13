import json
from collections import Counter

def _parse_grid(obs: str):
    """
    Extract the 2D colour grid as list[list[str]] (rows of colour-name strings).

    Anchor on the generic '[[' / ']]' so it works regardless of the header text that precedes
    the grid. Returns None on failure (caller degrades gracefully, never raises).
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
    if not grid or not isinstance(grid, list) or not isinstance(grid[0], list):
        return None
    return grid


def perceive(observation_history: list[str]) -> str:
    # Get current raw observation
    obs = observation_history[-1] if observation_history else ""
    grid = _parse_grid(obs)

    # Graceful degradation on any parse failure
    if grid is None:
        return "PARSE_ERROR"

    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    if rows == 0 or cols == 0:
        return f"EMPTY_GRID rows={rows} cols={cols}"

    # Determine background colour (dominant colour of the whole grid)
    all_colours = [cell for row in grid for cell in row]
    if not all_colours:
        return f"EMPTY_GRID rows={rows} cols={cols}"
    bg = Counter(all_colours).most_common(1)[0][0]

    # Collect non‑background cells
    objects = []          # list of (r, c, colour)
    for r, row in enumerate(grid):
        for c, colour in enumerate(row):
            if colour != bg:
                objects.append((r, c, colour))

    # Build a compact, deterministic textual summary
    # Include a step counter from the history length to break temporal symmetry
    step_index = len(observation_history)  # 1-based index of current observation
    parts = [f"step={step_index} {rows}x{cols} bg={bg}"]

    if not objects:
        parts.append("(empty)")
    else:
        # Group by colour, then sort by colour name and within colour by coordinates
        by_colour = {}
        for r, c, colour in objects:
            by_colour.setdefault(colour, []).append((r, c))
        for colour in sorted(by_colour.keys()):
            cells = by_colour[colour]
            cells.sort(key=lambda x: (x[0], x[1]))
            # Compact coordinate representation: "r,c" separated by spaces
            cell_str = " ".join(f"{r},{c}" for r, c in cells)
            parts.append(f"{colour}:{cell_str}")

    result = "; ".join(parts)

    # Safety truncation – <2000 chars guarantee; only needed for extremely dense grids
    if len(result) > 1999:
        result = result[:1996] + "..."

    return result