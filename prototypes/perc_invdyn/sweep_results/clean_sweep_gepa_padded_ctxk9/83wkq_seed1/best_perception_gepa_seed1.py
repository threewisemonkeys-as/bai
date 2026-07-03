import json

# Fixed colour palette for ARC integer grids
ARC_PALETTE = {
    0: 'black', 1: 'blue', 2: 'red', 3: 'green', 4: 'yellow',
    5: 'light-gray', 6: 'magenta', 7: 'orange', 8: 'light-blue',
    9: 'maroon', 10: 'white', 11: 'dark-gray'
}

def _parse_arc_grid(obs: str):
    """Parse an ARC-style integer grid from the observation string."""
    # Locate the first <grid_k> marker
    marker = None
    for line in obs.splitlines():
        stripped = line.strip()
        if stripped.startswith("<grid_"):
            marker = stripped
            break
    if marker is None:
        return None

    lines = obs.splitlines()
    # Find index of the marker line (exact match after stripping)
    marker_idx = None
    for idx, line in enumerate(lines):
        if line.strip() == marker:
            marker_idx = idx
            break
    if marker_idx is None:
        return None

    rows = []
    for i in range(marker_idx + 1, len(lines)):
        line = lines[i].strip()
        if not line.startswith('['):
            break
        try:
            row = json.loads(line)
            if not isinstance(row, list):
                break
            # All elements must be ints for a valid ARC grid
            if not all(isinstance(x, int) for x in row):
                break
            rows.append(row)
        except (json.JSONDecodeError, ValueError):
            break
    if not rows:
        return None
    return rows

def _parse_autumn_grid(obs: str):
    """Parse an Autumn-style JSON string grid from the observation string."""
    # Try to locate the grid JSON start by looking for '[[' after 'Start of Direct Observation'
    start = obs.find('[[', obs.find('Start of Direct Observation'))
    if start == -1:
        start = obs.find('[[')
    if start == -1:
        return None

    # Use bracket depth to find the end of the JSON array
    depth = 0
    end = -1
    for i in range(start, len(obs)):
        if obs[i] == '[':
            depth += 1
        elif obs[i] == ']':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        # Fallback: assume the last ']]' is the end
        end = obs.rfind(']]') + 2
        if end < 2:
            return None

    try:
        grid = json.loads(obs[start:end])
        if not isinstance(grid, list) or not grid:
            return None
        if not isinstance(grid[0], list):
            return None
        return grid
    except (json.JSONDecodeError, IndexError):
        return None

def _colour_name(value):
    """Convert an integer (ARC) or string (Autumn) to a standard colour name."""
    if isinstance(value, int):
        return ARC_PALETTE.get(value, f'unknown({value})')
    return value  # Already a string

def _summarise_grid(grid):
    """Return a concise description of non-black cells, sorted by (row,col)."""
    if not grid or not grid[0]:
        return "Grid: 0x0. Non-black cells: none"

    rows = len(grid)
    # Determine number of columns from the longest row (should be rectangular but be safe)
    cols = max(len(row) for row in grid) if rows > 0 else 0

    non_black = []
    background = 0 if isinstance(grid[0][0], int) else 'black'

    for r in range(rows):
        for c in range(len(grid[r])):
            cell = grid[r][c]
            is_bg = (cell == background) if isinstance(cell, int) else (cell == 'black')
            if not is_bg:
                non_black.append((r, c, _colour_name(cell)))

    non_black.sort(key=lambda x: (x[0], x[1]))

    summary = f"Grid: {rows}x{cols}."
    if not non_black:
        summary += " Non-black cells: none"
    else:
        items = [f"({r},{c}):{col}" for r, c, col in non_black]
        cell_str = ", ".join(items)
        # Ensure the total output stays under 2000 characters
        if len(cell_str) > 1900:  # leave room for header
            limit = max(50, len(items) - 100)  # show as many as possible
            cell_str = ", ".join(items[:limit])
            remaining = len(items) - limit
            if remaining > 0:
                cell_str += f", ... ({remaining} more)"
        summary += f" Non-black cells: {cell_str}"
    return summary

def perceive(observation_history: list[str]) -> str:
    """
    Parse the current observation and return a concise text summary
    of decision-relevant features. Output is never empty and never raises.
    Uses comma+space to separate cells and always includes colour suffix.
    The format (row,col):colour is fixed to enable reliable action inference.
    """
    try:
        obs = observation_history[-1]
    except IndexError:
        return "No observation"

    grid = _parse_arc_grid(obs)
    if grid is None:
        grid = _parse_autumn_grid(obs)

    if grid is None:
        # Last resort: try to extract any JSON 2D array from the whole text
        try:
            start = obs.find('[[', obs.find('Start of Direct Observation'))
            if start == -1:
                start = obs.find('[[')
            if start != -1:
                end = obs.rfind(']]') + 2
                if end > start:
                    grid = json.loads(obs[start:end])
        except (json.JSONDecodeError, IndexError, ValueError):
            pass

    if grid is None or not grid:
        return "Grid: parse failed, no grid detected."

    return _summarise_grid(grid)