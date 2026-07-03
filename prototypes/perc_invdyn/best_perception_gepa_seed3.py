import json

def perceive(observation_history: list[str]) -> str:
    """
    Extract all non-black cells from the current raw observation and return
    a concise text summary of their positions and colors.
    This ensures that any change in the grid (e.g., movement of darkgreen cells)
    is reflected in the output, making the action recoverable from two consecutive summaries.
    """
    # Get the latest raw observation string
    obs = observation_history[-1]
    # Locate the JSON grid embedded in the observation string
    start = obs.find('[[')
    end = obs.rfind(']]')
    if start == -1 or end == -1:
        # No grid found – return a safe placeholder
        return "non_black_cells=[]"
    end += 2  # include closing brackets
    try:
        grid = json.loads(obs[start:end])
    except Exception:
        return "non_black_cells=[]"

    # Collect all cells that are not black
    cells = []
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            if color != "black":
                cells.append((r, c, color))

    # Sort for deterministic ordering
    cells.sort(key=lambda x: (x[0], x[1]))

    # Format as a string: non_black_cells=[(r,c,'color'), ...]
    # repr ensures proper quoting of color strings
    cell_strs = [f"({r},{c},{repr(clr)})" for r, c, clr in cells]
    result = "non_black_cells=[" + ",".join(cell_strs) + "]"
    return result