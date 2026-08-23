import json
import collections
import re

def perceive(observation_history: list[str]) -> str:
    """
    Extract a concise text summary that includes:
    - step number (if available)
    - current active/selected color (if present in the observation)
    - background colour
    - grid dimensions
    - all non‑background cells (row, col, colour), sorted by row then column
    - a flag indicating whether any orange cells exist in rows 1-15 (for blue shift prediction)

    This richer representation allows the action between two consecutive states
    to be recovered even when the grid does not change.
    """
    obs = observation_history[-1]
    grid = None
    step = None
    active_color = None

    # --- Extract step number (present in interactive tasks) ---
    for line in obs.splitlines():
        stripped = line.strip()
        if stripped.startswith("Step:"):
            try:
                step = int(stripped.split(":")[1].strip())
            except Exception:
                pass
            break

    # --- Extract active / selected colour (common patterns in the observation) ---
    for line in obs.splitlines():
        stripped = line.strip().lower()
        if any(pat in stripped for pat in ["active color", "selected", "tool", "colour"]):
            parts = line.split(":")
            if len(parts) >= 2:
                candidate = parts[-1].strip().rstrip('.').lower()
                valid_colours = {
                    "black", "blue", "red", "green", "yellow", "light-gray", "magenta",
                    "orange", "light-blue", "maroon", "white", "dark-gray",
                    "purple", "gray", "gold", "lightblue", "darkgreen"
                }
                if candidate in valid_colours:
                    active_color = candidate
                    break

    if active_color is None:
        for line in obs.splitlines():
            stripped = line.strip().lower()
            if "active" in stripped and ("color" in stripped or "colour" in stripped):
                for sep in [':', '=']:
                    if sep in stripped:
                        candidate = stripped.split(sep)[-1].strip().rstrip('.').lstrip()
                        valid_colours = {
                            "black", "blue", "red", "green", "yellow", "light-gray", "magenta",
                            "orange", "light-blue", "maroon", "white", "dark-gray",
                            "purple", "gray", "gold", "lightblue", "darkgreen"
                        }
                        if candidate in valid_colours:
                            active_color = candidate
                            break
            if active_color:
                break

    # --- Try JSON encoding (autumn style) ---
    idx = obs.find("[[")
    if idx != -1:
        end = obs.rfind("]]")
        if end != -1 and end > idx:
            json_str = obs[idx:end+2]
            try:
                grid = json.loads(json_str)
                if not isinstance(grid, list) or not all(isinstance(row, list) for row in grid):
                    grid = None
                elif len(grid) == 0 or len(grid[0]) == 0:
                    grid = None
            except Exception:
                grid = None

    # --- Fallback to ARC integer grid encoding ---
    if grid is None:
        marker = "<grid_"
        marker_idx = obs.find(marker)
        if marker_idx != -1:
            lines = obs.splitlines()
            start_parsing = False
            rows = []
            for line in lines:
                if not start_parsing:
                    if line.strip().startswith(marker):
                        start_parsing = True
                    continue
                line = line.strip()
                if line.startswith('[') and line.endswith(']'):
                    try:
                        inner = line[1:-1]
                        row = [int(x.strip()) for x in inner.split(',') if x.strip()]
                        rows.append(row)
                    except Exception:
                        continue
                else:
                    break
            if rows and len(rows) > 0 and len(rows[0]) > 0:
                grid = rows

    # --- If no grid could be parsed, return a minimal summary ---
    if grid is None:
        parts = []
        if step is not None:
            parts.append(f"step={step}")
        if active_color:
            parts.append(f"active={active_color}")
        parts.append("grid=None")
        return "; ".join(parts)

    flat = [cell for row in grid for cell in row]
    if not flat:
        parts = []
        if step is not None:
            parts.append(f"step={step}")
        if active_color:
            parts.append(f"active={active_color}")
        parts.append("grid=[]")
        return "; ".join(parts)

    # --- Get grid dimensions ---
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0

    # --- Determine background (most frequent) ---
    counter = collections.Counter(flat)
    background = counter.most_common(1)[0][0]

    # --- Check for orange cells in rows 1-15 (for blue shift detection) ---
    orange_in_rows_1_15 = False
    for r in range(1, min(16, height)):
        for c in range(width):
            cell = grid[r][c]
            if cell == "orange" or cell == 7:
                orange_in_rows_1_15 = True
                break
        if orange_in_rows_1_15:
            break

    # --- Collect all non‑background cells, sorted by row then column ---
    features = []
    for r, row in enumerate(grid):
        for c, col in enumerate(row):
            if col != background:
                features.append((r, c, col))
    
    features.sort(key=lambda x: (x[0], x[1]))

    # --- Build output string ---
    parts = []
    if step is not None:
        parts.append(f"step={step}")
    if active_color:
        parts.append(f"active={active_color}")
    
    parts.append(f"size={height}x{width}")
    bg_str = str(background)
    parts.append(f"bg={bg_str}")
    parts.append(f"orange_in_rows_1_15={str(orange_in_rows_1_15).lower()}")

    if features:
        cells_str = ",".join(f"({r},{c},{col})" for r, c, col in features)
        parts.append(cells_str)
    else:
        parts.append("[]")

    return "; ".join(parts)