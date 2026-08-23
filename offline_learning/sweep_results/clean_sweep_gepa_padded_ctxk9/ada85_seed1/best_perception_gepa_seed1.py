import json
import re
import ast
from typing import List, Union, Dict, Tuple, Optional
from collections import Counter

# Fixed palette mapping for integer grids
INT_TO_COLOR = {
    0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
    5: "gray", 6: "magenta", 7: "orange", 8: "lightblue",
    9: "maroon", 10: "white", 11: "darkgray"
}

def _color_name(val: Union[int, str]) -> str:
    """Convert a cell value to a standard color name."""
    if isinstance(val, str):
        return val.lower().replace(" ", "")
    return INT_TO_COLOR.get(val, "unknown")

def _most_frequent_color(grid: List[List[str]]) -> str:
    """Return the most common colour in the grid."""
    flat = [cell for row in grid for cell in row]
    if not flat:
        return "black"
    counter = Counter(flat)
    return max(flat, key=lambda x: (counter[x], -len(x), x))

# ---------- domain specific helpers ----------
PATTERN_A = ["white", "yellow", "yellow", "yellow", "white"]   # rows 3-7
PATTERN_B = ["yellow", "white", "gold", "white", "yellow"]

def _pattern_state(col_cells: list) -> str:
    """Return 'A' or 'B' given the 5 cells of column 10 (rows 3..7)."""
    if len(col_cells) != 5:
        return "?"
    if col_cells == PATTERN_A:
        return "A"
    if col_cells == PATTERN_B:
        return "B"
    return "?"

def _next_pattern(current_pattern: str) -> str:
    """Return the pattern that follows the current one after auto-change."""
    if current_pattern == "A":
        return "B"
    elif current_pattern == "B":
        return "A"
    return "?"

def _next_auto_change(step: int) -> int:
    """Step number of the next autonomous pattern change (period = 20)."""
    period = 20
    return ((step // period) + 1) * period

def _gray_direction_and_next(r: int, c: int, rows: int, cols: int) -> Tuple[str, Optional[Tuple[int,int]]]:
    """Return (direction, next_position if no user intervention) for a gray agent."""
    # gray from (10,0) spawn: east along row 10, then north on col 10
    if r == 10 and c < 10:
        return "east", (10, c+1) if c+1 < cols else None
    if r == 10 and c == 10:
        return "north", (9, 10) if 9 >= 0 else None
    # gray from (0,0) spawn: east along row 0, then south on col 10
    if r == 0 and c < 10:
        return "east", (0, c+1) if c+1 < cols else None
    if r == 0 and c == 10:
        return "south", (1, 10) if 1 < rows else None
    # gray already on column 10 moving north (from (10,0) spawn)
    if c == 10 and r > 0 and r < 10:
        return "north", (r-1, 10) if r-1 >= 0 else None
    # gray already on column 10 moving south (from (0,0) spawn)
    if c == 10 and r > 0 and r < 10:
        # Need to distinguish direction: if r < 3 it's likely moving south, otherwise north?
        # Use history: from (0,0) gray goes east then south; from (10,0) east then north.
        # For r between 3 and 7, it could be either. We'll use step context? Not reliable.
        # Better: explicitly track both possible spawns? Not possible without step.
        # For simplicity, we compute both possibilities and rely on the non-bg list to disambiguate.
        # We'll return "north" for now, but this could be wrong. Use a heuristic based on row:
        # If r > 6, it's likely from (10,0) moving north; if r < 3, from (0,0) moving south.
        if r > 6:
            return "north", (r-1, 10) if r-1 >= 0 else None
        elif r < 3:
            return "south", (r+1, 10) if r+1 < rows else None
        else:
            return "unknown", None
    # fallback
    return "unknown", None

# ---------- main perceive function ----------
def perceive(observation_history: list[str]) -> str:
    """
    Produce a concise text summary of the current grid state, step number,
    action count (if available), and all decision-relevant features.
    Includes explicit next-state predictions (next pattern, next gray positions)
    to make forward prediction trivial.
    """
    obs = observation_history[-1]
    fallback = "Step: ?. ActionCount: ?. Grid: ?. (parse error)"

    # ---------- extract metadata ----------
    step = "?"
    step_match = re.search(r"Step:?\s*(\d+)", obs)
    if step_match:
        step = int(step_match.group(1))

    action_count = None
    ac_match = re.search(r"Action count:\s*(\d+)", obs)
    if ac_match:
        action_count = int(ac_match.group(1))

    levels_completed = None
    lc_match = re.search(r"Levels completed:\s*(\d+)/(\d+)", obs)
    if lc_match:
        levels_completed = f"{lc_match.group(1)}/{lc_match.group(2)}"

    try:
        # ---------- parse grid ----------
        grid = None
        if "<grid_" in obs:
            # integer grid
            lines = obs.split("\n")
            grid_start = -1
            for i, line in enumerate(lines):
                if re.match(r"<grid_\d+>", line.strip()):
                    grid_start = i + 1
                    break
            if grid_start >= 0:
                rows = []
                for line in lines[grid_start:]:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("["):
                        try:
                            row = ast.literal_eval(line)
                            if isinstance(row, list):
                                rows.append(row)
                        except:
                            parts = line.strip("[]").split(",")
                            row = [int(p.strip()) for p in parts if p.strip().isdigit()]
                            rows.append(row)
                if rows:
                    grid = [[_color_name(c) for c in row] for row in rows]
        else:
            # JSON string grid
            start = obs.find("[[")
            end = obs.rfind("]]")
            if start != -1 and end != -1:
                json_str = obs[start:end+2]
                parsed = json.loads(json_str)
                if isinstance(parsed, list) and all(isinstance(r, list) for r in parsed):
                    grid = [[_color_name(c) for c in row] for row in parsed]

        if grid is None:
            return f"Step {step}. ActionCount {action_count if action_count is not None else '?'}. Grid could not be parsed."

        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0

        # ---------- background ----------
        bg = _most_frequent_color(grid)

        # ---------- collect non‑background cells and gray agents ----------
        non_bg_cells = []
        grays = []                # (r, c, direction, next_pos)
        for r in range(rows):
            for c in range(cols):
                col = grid[r][c]
                if col != bg:
                    non_bg_cells.append((r, c, col))
                    if col == "gray":
                        direction, next_pos = _gray_direction_and_next(r, c, rows, cols)
                        grays.append((r, c, direction, next_pos))

        # sort for determinism
        non_bg_cells.sort(key=lambda x: (x[0], x[1]))
        grays.sort()

        # ---------- column‑10 pattern (rows 3‑7) ----------
        col10_cells = []
        for rr in range(3, 8):
            if cols > 10:
                col10_cells.append(grid[rr][10])
        pattern_state = _pattern_state(col10_cells)
        next_auto_step = _next_auto_change(step)
        next_pattern = _next_pattern(pattern_state) if pattern_state != "?" else "?"

        # ---------- spawn readiness ----------
        spawn_10_ready = True
        spawn_0_ready = True
        for r, c, *_ in grays:
            if (r == 10 and c < 10) or (c == 10 and r > 6):
                spawn_10_ready = False
            if (r == 0 and c < 10) or (c == 10 and r < 3):
                spawn_0_ready = False

        # ---------- immutable cells ----------
        immutables = []
        if rows > 0 and cols > 0 and grid[0][0] == "blue":
            immutables.append((0,0))
        if rows > 10 and cols > 0 and grid[10][0] == "red":
            immutables.append((10,0))

        # ---------- build output ----------
        parts = [f"Step {step}."]
        if action_count is not None:
            parts.append(f"ActionCount {action_count}.")
        if levels_completed:
            parts.append(f"Levels {levels_completed}.")
        parts.append(f"Grid {rows}x{cols}. Background: {bg}.")

        if immutables:
            imm_str = "; ".join(f"({r},{c})" for r,c in immutables)
            parts.append(f"Immutable: {imm_str}.")

        # spawn points with readiness
        parts.append(f"Spawn(0,0): {'ready' if spawn_0_ready else 'active'}.")
        parts.append(f"Spawn(10,0): {'ready' if spawn_10_ready else 'active'}.")

        # autonomous agents with direction and next position
        if grays:
            gray_strs = []
            for r,c,dir,nxt in grays:
                base = f"({r},{c}) dir={dir}"
                if nxt is not None:
                    base += f" next=({nxt[0]},{nxt[1]})"
                else:
                    base += " next=despawn"
                gray_strs.append(base)
            parts.append(f"Autonomous: gray at [{'; '.join(gray_strs)}].")
        else:
            parts.append("Autonomous: none.")

        # pattern phase with next auto-change info
        parts.append(f"Pattern: {pattern_state} (auto-change at step {next_auto_step} to {next_pattern}).")

        # column10 detail
        col10_desc = ", ".join(f"row{rr}={grid[rr][10]}" for rr in range(3,8) if cols>10)
        if col10_desc:
            parts.append(f"Col10: [{col10_desc}].")

        # non‑background cells (limit to 50)
        limit = 50
        if len(non_bg_cells) > limit:
            displayed = non_bg_cells[:limit]
            ellipsis = " ..."
        else:
            displayed = non_bg_cells
            ellipsis = ""
        cell_str = "; ".join(f"({r},{c})={col}" for r,c,col in displayed)
        if cell_str:
            parts.append(f"Non-bg: {cell_str}{ellipsis}.")
        else:
            parts.append("Non-bg: all background.")

        output = " ".join(parts)

        # ensure <2000 chars
        if len(output) > 1950:
            # reduce non‑bg cells further
            max_cells = 40
            displayed = non_bg_cells[:max_cells]
            cell_str = "; ".join(f"({r},{c})={col}" for r,c,col in displayed)
            parts[-1] = f"Non-bg: {cell_str}..."
            output = " ".join(parts)
            if len(output) > 1990:
                # drop col10 detail if needed
                parts = [p for p in parts if not p.startswith("Col10:")]
                output = " ".join(parts)

        return output

    except Exception:
        # never raise, never return empty
        return fallback