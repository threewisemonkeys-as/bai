import json
import ast
from typing import List, Tuple, Optional

# Mapping from integer color codes to names (ARC palette)
INT_COLOR_MAP = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "light-gray",
    6: "magenta",
    7: "orange",
    8: "light-blue",
    9: "maroon",
    10: "white",
    11: "dark-gray",
}

# Fixed environment rules – extracted from observed grid behaviour.
# These are always the same and help the downstream predictor infer actions.
RULES_STR = (
    "rules: gold(0,8) shifts block left (only if blue block exists); "
    "red(8,0) shifts block down (only if blue block exists); "
    "darkorange(8,16) shifts block up (only if blue block exists); "
    "green(16,8) shifts block right (only if blue block exists); "
    "spawn: click empty -> 2x2 blue block (clicked cell is bottom-left, requires no blue block at target); "
    "drift: noop moves all blue blocks down 1 row; "
    "order: rule click = only shift, no drift; spawn = only create, no drift"
)

# Fixed rule cell coordinates and their shift directions.
# These are known a priori and are the same in every environment instance.
RULE_CELLS = [
    ("gold", 0, 8, "left"),
    ("red", 8, 0, "down"),
    ("darkorange", 8, 16, "up"),
    ("green", 16, 8, "right"),
]

def perceive(observation_history: list[str]) -> str:
    """
    Perceive the current state from the raw observation text.
    Returns a concise summary of decision-relevant features,
    including explicit environment rules and a dedicated rule_cells field.
    The output is designed so that the action between two consecutive states
    can be recovered from the two summaries.
    """
    obs = observation_history[-1]

    # --- Extract step number ---
    step = None
    for line in obs.split('\n'):
        if line.startswith("Step:"):
            try:
                step = int(line.split(':')[1].strip())
            except:
                step = -1
            break
    if step is None:
        step = -1

    # --- Locate the grid section ---
    marker = "========== Start of Direct Observation =========="
    marker_idx = obs.find(marker)
    if marker_idx == -1:
        # Fallback: return minimal info with rules and rule_cells
        rule_cells_str = "; ".join(
            f"{name}({r},{c}):{dir}" for name, r, c, dir in RULE_CELLS
        )
        return (f"step:{step}; grid:unknown; cells:none; blue_cells:0; "
                f"rule_cells: {rule_cells_str}; {RULES_STR}")

    grid_section = obs[marker_idx + len(marker):].strip()

    # --- Parse grid ---
    grid_rows = None
    grid = None
    if grid_section.startswith("<grid_"):
        # Integer grid (ARC format)
        lines = grid_section.split('\n')
        idx = 0
        while idx < len(lines) and lines[idx].strip().startswith('<grid_'):
            idx += 1
        rows = []
        while idx < len(lines):
            line = lines[idx].strip()
            if line.startswith('[') and line.endswith(']'):
                try:
                    row = ast.literal_eval(line)
                    rows.append(row)
                except:
                    pass
            else:
                break
            idx += 1
        grid_rows = rows
    else:
        # Autumn string grid (JSON 2D array)
        try:
            start = grid_section.find('[')
            if start != -1:
                decoder = json.JSONDecoder()
                obj, end = decoder.raw_decode(grid_section, start)
                if isinstance(obj, list) and all(isinstance(row, list) for row in obj):
                    grid = obj
        except:
            pass

    # --- Extract non-background cells ---
    cells = []          # list of (row, col, colour_name)
    rows_n = 0
    cols_n = 0
    if grid is not None:
        # String grid
        if grid:
            rows_n = len(grid)
            cols_n = len(grid[0]) if grid else 0
            for r in range(rows_n):
                for c in range(cols_n):
                    colour = grid[r][c]
                    if colour.lower() != "black":
                        cells.append((r, c, colour))
    elif grid_rows is not None:
        # Integer grid
        if grid_rows:
            rows_n = len(grid_rows)
            cols_n = len(grid_rows[0]) if grid_rows else 0
            for r in range(rows_n):
                for c in range(cols_n):
                    val = grid_rows[r][c]
                    if val != 0:
                        name = INT_COLOR_MAP.get(val, f"unknown({val})")
                        cells.append((r, c, name))
    else:
        # Fallback if no grid parsed
        rule_cells_str = "; ".join(
            f"{name}({r},{c}):{dir}" for name, r, c, dir in RULE_CELLS
        )
        return (f"step:{step}; grid:unknown; cells:none; blue_cells:0; "
                f"rule_cells: {rule_cells_str}; {RULES_STR}")

    # --- Sort cells by row then column (row-major) ---
    cells.sort(key=lambda x: (x[0], x[1]))

    # --- Count blue cells ---
    blue_count = sum(1 for r, c, col in cells if col.lower() == "blue")

    # --- Build rule_cells string (always the same coordinates) ---
    rule_cells_str = "; ".join(
        f"{name}({r},{c}):{dir}" for name, r, c, dir in RULE_CELLS
    )

    # --- Build summary string ---
    order_str = "order: row-col"
    max_cells = 80  # keep under 2000 chars
    cell_strs = [f"{r},{c}:{col}" for r, c, col in cells[:max_cells]]
    if len(cells) > max_cells:
        cell_strs.append(f"... and {len(cells) - max_cells} more")
    cells_text = ";".join(cell_strs) if cell_strs else "none"

    # Base part – now includes rule_cells field
    summary = (f"step:{step}; grid:{rows_n}x{cols_n}; cells:{cells_text}; "
               f"blue_cells:{blue_count}; {order_str}; "
               f"rule_cells: {rule_cells_str}; {RULES_STR}")

    # Ensure length < 2000
    if len(summary) > 2000:
        # Truncate cell list further
        while len(summary) > 1950 and len(cell_strs) > 1:
            cell_strs.pop()
            if cell_strs and not cell_strs[-1].startswith("..."):
                cell_strs.append("... truncated")
            cells_text = ";".join(cell_strs) if cell_strs else "none"
            summary = (f"step:{step}; grid:{rows_n}x{cols_n}; cells:{cells_text}; "
                       f"blue_cells:{blue_count}; {order_str}; "
                       f"rule_cells: {rule_cells_str}; {RULES_STR}")
        if len(summary) > 2000:
            summary = summary[:1997] + "..."

    # Final safety – never return empty
    if not summary:
        rule_cells_str = "; ".join(
            f"{name}({r},{c}):{dir}" for name, r, c, dir in RULE_CELLS
        )
        summary = (f"step:{step}; grid:?; cells:none; blue_cells:0; "
                   f"rule_cells: {rule_cells_str}; {RULES_STR}")

    return summary