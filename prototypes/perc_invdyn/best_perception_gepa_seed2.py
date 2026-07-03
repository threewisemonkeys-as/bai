import json

def perceive(observation_history: list[str]) -> str:
    """
    Returns a concise summary of all non‑black cells in the grid.
    The output changes whenever any non‑black cell appears, disappears, or moves,
    making the action between two consecutive states recoverable.
    """
    # Get the latest raw observation
    obs = observation_history[-1]

    # Locate the grid: find the first '[' that starts a list of lists
    start = obs.find('[')
    if start == -1:
        return "no_grid"
    # Find the matching closing ']]'
    end = obs.rfind(']]')
    if end == -1:
        return "no_grid"
    end += 2  # include the two closing brackets

    try:
        grid = json.loads(obs[start:end])
    except Exception:
        # Fallback: try to find a different grid pattern (e.g., after "initial state")
        alt_start = obs.find('[["')
        if alt_start != -1:
            try:
                grid = json.loads(obs[alt_start:obs.rfind(']]') + 2])
            except Exception:
                return "parse_error"
        else:
            return "parse_error"

    # Collect all non-black cells
    cells = {}  # colour -> list of [row, col]
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell != "black":
                cells.setdefault(cell, []).append([r, c])

    # Sort for consistent output (optional but helps readability)
    for colour in cells:
        cells[colour].sort()

    # Never return an empty string
    if not cells:
        return "no_non_black_cells"

    # Produce a compact JSON representation
    return json.dumps(cells, separators=(',', ':'))