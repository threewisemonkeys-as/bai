import ast
import json
from typing import List, Optional, Tuple, Union

# Fixed palette mapping for integer grids
PALETTE = {
    0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
    5: "light-gray", 6: "magenta", 7: "orange", 8: "light-blue",
    9: "maroon", 10: "white", 11: "dark-gray"
}


def perceive(observation_history: list[str]) -> str:
    """
    Summarise the current grid observation.
    Output is concise (<2000 char) and always includes:
    - mode: 'agent' or 'no_agent'
    - colour of the pattern (e.g. yellow, white)
    - shape class and its geometric line descriptor
    - agent coordinates (if mode==agent)
    - sorted list of non‑background cells (row, col, colour)
    This rich representation makes the effect of actions recoverable.
    """
    obs = observation_history[-1] if observation_history else ""
    if not obs:
        return "empty_observation"

    try:
        # --- Parse the grid -------------------------------------------------
        if "<grid_" in obs:
            grid = _parse_integer_grid(obs)
        else:
            grid = _parse_json_grid(obs)

        if grid is None or len(grid) == 0:
            return "empty_grid"

        # --- Determine type and background ---------------------------------
        if isinstance(grid[0][0], int):
            background = 0
            is_int = True
        else:
            background = "black"
            is_int = False

        # --- Collect non‑background cells ----------------------------------
        non_bg = []
        for r, row in enumerate(grid):
            for c, val in enumerate(row):
                if is_int:
                    if val != background:
                        non_bg.append((r, c, val))
                else:
                    val_lower = val.lower()
                    if val_lower != background:
                        non_bg.append((r, c, val_lower))

        if not non_bg:
            return "all_black_grid"

        # sort consistently
        non_bg.sort(key=lambda x: (x[0], x[1]))

        # --- Detect mode and colour ----------------------------------------
        if is_int:
            first_col = non_bg[0][2]
            colour_name = PALETTE.get(first_col, f"code_{first_col}")
        else:
            colour_name = non_bg[0][2]

        # White cells indicate agent presence
        if is_int:
            white_cells = [(r, c) for r, c, val in non_bg if val == 10]
        else:
            white_cells = [(r, c) for r, c, val in non_bg if val == "white"]

        if white_cells:
            mode = "agent"
            # agent is the topmost (smallest row, then smallest col) white cell
            agent = min(white_cells, key=lambda x: (x[0], x[1]))
            shape_cells = non_bg   # all non‑black cells are part of the pattern
        else:
            mode = "no_agent"
            agent = None
            shape_cells = non_bg

        # --- Shape classification + geometric line descriptor --------------
        shape, line_desc = _classify_shape(shape_cells)

        # --- Build output string -------------------------------------------
        parts = []
        parts.append(f"mode:{mode}")
        parts.append(f"colour:{colour_name}")
        parts.append(f"shape:{shape}")
        if line_desc:
            parts.append(f"line:{line_desc}")

        if agent:
            parts.append(f"agent({agent[0]},{agent[1]})")

        # List cells (max 200 to stay under 2000 chars)
        max_cells = 200
        cell_strs = []
        for i, (r, c, val) in enumerate(non_bg):
            if i >= max_cells:
                cell_strs.append(f"+{len(non_bg)-max_cells}more")
                break
            col_name = PALETTE.get(val, f"code_{val}") if is_int else val
            cell_strs.append(f"({r},{c},{col_name})")

        parts.append("cells:" + ",".join(cell_strs))
        result = "; ".join(parts)
        if len(result) > 2000:
            result = result[:1997] + "..."
        return result

    except Exception as e:
        # Never raise – return a safe fallback
        return f"parse_error:{str(e)[:100]}"


# -------------------------------------------------------------------
# Shape analysis
# -------------------------------------------------------------------
def _classify_shape(cells: list) -> Tuple[str, Optional[str]]:
    """
    Return (shape_name, line_descriptor).
    line_descriptor is a string like "row=7", "col=7", "r-c=0", "r+c=14"
    that uniquely identifies the geometric line of the pattern.
    """
    if len(cells) < 2:
        return "none", None

    coords = [(r, c) for r, c, _ in cells]
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]

    # Horizontal: all rows equal
    if all(r == rows[0] for r in rows):
        return "horizontal", f"row={rows[0]}"

    # Vertical: all cols equal
    if all(c == cols[0] for c in cols):
        return "vertical", f"col={cols[0]}"

    # Diagonal increasing: r - c constant
    diff = [r - c for r, c in coords]
    if all(d == diff[0] for d in diff):
        return "diagonal_inc", f"r-c={diff[0]}"

    # Diagonal decreasing: r + c constant
    summ = [r + c for r, c in coords]
    if all(s == summ[0] for s in summ):
        return "diagonal_dec", f"r+c={summ[0]}"

    return "unknown_shape", None


# -------------------------------------------------------------------
# Parsing helpers
# -------------------------------------------------------------------

def _parse_integer_grid(obs: str) -> Optional[List[List[int]]]:
    """Extract the first grid block from an ARC‑style integer observation."""
    marker = "========== Start of Direct Observation =========="
    idx = obs.find(marker)
    if idx == -1:
        idx = obs.find("<grid_0>")
        if idx == -1:
            return None
        idx = obs.find("\n", idx) + 1
    else:
        idx = obs.find("\n", idx) + 1

    lines = obs[idx:].split("\n")
    rows = []
    in_grid = False
    for line in lines:
        s = line.strip()
        if s.startswith("<grid_"):
            in_grid = True
            continue
        if not in_grid and s == "":
            continue
        if s.startswith("[") and s.endswith("]"):
            try:
                row = ast.literal_eval(s)
                if isinstance(row, list) and len(row) > 0:
                    rows.append(row)
                else:
                    break
            except:
                break
        else:
            if rows:
                break
    return rows if rows else None


def _parse_json_grid(obs: str) -> Optional[List[List[str]]]:
    """Extract the 2D JSON array from the observation string."""
    start = obs.find("[[")
    if start == -1:
        return None
    end = obs.rfind("]]")
    if end == -1:
        return None
    end += 2
    json_str = obs[start:end]
    try:
        grid = json.loads(json_str)
        if isinstance(grid, list) and len(grid) > 0 and isinstance(grid[0], list):
            return grid
    except json.JSONDecodeError:
        pass
    return None