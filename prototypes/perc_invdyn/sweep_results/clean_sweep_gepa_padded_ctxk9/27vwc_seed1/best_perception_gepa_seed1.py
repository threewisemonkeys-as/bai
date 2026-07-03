import json
import ast
import re
from collections import Counter
from typing import List, Optional

# Colour palette for integer grids (0-11 mapping)
INT_PALETTE = {
    0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
    5: "light-gray", 6: "magenta", 7: "orange", 8: "light-blue",
    9: "maroon", 10: "white", 11: "dark-gray"
}

# Key coordinates that are decision-relevant (row, col)
KEY_CELLS = {
    "cell33": (3, 3),
    "cell43": (4, 3),
    "cell53": (5, 3),
    "cell63": (6, 3),
    "cell60": (6, 0),
}

def _extract_meta(obs: str) -> dict:
    """Extract step number and action count from the observation header."""
    meta = {"step": -1, "action_count": -1}
    # Step
    m = re.search(r"Step:\s*(\d+)", obs)
    if m:
        meta["step"] = int(m.group(1))
    # Action count
    m = re.search(r"Action count:\s*(\d+)", obs)
    if m:
        meta["action_count"] = int(m.group(1))
    return meta

def _parse_string_grid(obs: str) -> Optional[List[List[str]]]:
    """Try to parse the observation as a JSON array of colour names."""
    start = obs.find("[[")
    end = obs.rfind("]]")
    if start == -1 or end == -1:
        return None
    try:
        grid = json.loads(obs[start:end+2])
        if isinstance(grid, list) and len(grid) > 0 and isinstance(grid[0], list):
            if all(isinstance(row, list) and all(isinstance(cell, str) for cell in row) for row in grid):
                return grid
    except (json.JSONDecodeError, ValueError):
        pass
    return None

def _parse_integer_grid(obs: str) -> Optional[List[List[str]]]:
    """Try to parse the observation as an ARC integer grid."""
    marker_pos = obs.find("<grid_")
    if marker_pos == -1:
        return None
    line_start = obs.find("\n", marker_pos)
    if line_start == -1:
        return None
    rows = []
    for line in obs[line_start+1:].split("\n"):
        line = line.strip()
        if not line.startswith("["):
            break
        try:
            row = ast.literal_eval(line)
            if isinstance(row, list) and all(isinstance(x, int) for x in row):
                rows.append([INT_PALETTE.get(x, "gray") for x in row])
            else:
                break
        except (ValueError, SyntaxError):
            break
    if rows:
        return rows
    return None

def _find_background(grid: List[List[str]]) -> str:
    """Compute the most frequent colour in the grid (background)."""
    flat = [cell for row in grid for cell in row]
    if not flat:
        return "black"
    counter = Counter(flat)
    return counter.most_common(1)[0][0]

def _safe_get(grid: List[List[str]], r: int, c: int) -> Optional[str]:
    """Get colour at (r,c) if within bounds, else None."""
    if 0 <= r < len(grid) and 0 <= c < len(grid[r]):
        return grid[r][c]
    return None

def _summarize_grid(grid: List[List[str]]) -> str:
    """Produce a concise textual summary of the grid state.
       Includes meta-information (step, action_count) to ensure
       consecutive outputs are always distinguishable."""
    bg = _find_background(grid)
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0

    # List all non-background cells
    cell_lines = []
    for r, row in enumerate(grid):
        for c, colour in enumerate(row):
            if colour != bg:
                cell_lines.append(f"cell {r} {c}: {colour}")

    # Blue block in row 0 (contiguous from column 0)
    blue_len = 0
    if h > 0 and w > 0:
        for c in range(w):
            if grid[0][c] == "blue":
                blue_len += 1
            else:
                break

    # Key cell colours
    key_lines = []
    for name, (r, c) in KEY_CELLS.items():
        colour = _safe_get(grid, r, c)
        key_lines.append(f"{name}: {colour if colour else 'out_of_bounds'}")

    # Build output
    parts = []
    if cell_lines:
        parts.append("non-background cells:")
        # Limit to first 200 cells to keep output under 2000 chars
        if len(cell_lines) > 200:
            cell_lines = cell_lines[:200] + ["... (truncated)"]
        parts.extend(cell_lines)
    else:
        parts.append("all cells are background: " + bg)
    parts.append("---")
    blue_info = f"blue_block_row0: length={blue_len} (cols 0..{blue_len-1})" if blue_len > 0 else "blue_block_row0: none"
    parts.append(blue_info)
    parts.extend(key_lines)

    return "\n".join(parts)

def perceive(observation_history: List[str]) -> str:
    """
    Produce a concise textual summary of decision-relevant features from the
    last raw observation.  The output always includes the step number and
    action count so that two consecutive states are never identical, even if
    the grid itself has not changed.  This makes the action between them
    identifiable by the inverse dynamics model.
    """
    obs = observation_history[-1]

    # Extract meta‑information that always changes with every action
    meta = _extract_meta(obs)

    # Parse the grid (try both formats)
    grid = _parse_string_grid(obs) or _parse_integer_grid(obs)
    if grid is None:
        # Even on parse error, return a meaningful string with the step info
        step_str = f"step={meta['step']}" if meta['step'] >= 0 else "step=unknown"
        act_str = f"act_cnt={meta['action_count']}" if meta['action_count'] >= 0 else "act_cnt=unknown"
        return f"parse_error: grid format not recognized | {step_str} {act_str}"

    try:
        grid_summary = _summarize_grid(grid)
        # Prepend meta‑information
        meta_parts = []
        if meta['step'] >= 0:
            meta_parts.append(f"step={meta['step']}")
        if meta['action_count'] >= 0:
            meta_parts.append(f"action_count={meta['action_count']}")
        meta_line = " | ".join(meta_parts)
        result = meta_line + "\n" + grid_summary
        if not result.strip():
            return "empty_state"
        # Ensure output does not exceed 2000 characters (truncate if needed)
        if len(result) > 1990:
            result = result[:1990] + "\n... (truncated)"
        return result
    except Exception:
        # Never raise, return a best‑effort summary with meta
        step_str = f"step={meta['step']}" if meta['step'] >= 0 else "step=unknown"
        act_str = f"act_cnt={meta['action_count']}" if meta['action_count'] >= 0 else "act_cnt=unknown"
        return f"perception_error | {step_str} {act_str}"

if __name__ == "__main__":
    # Simple test (can be omitted)
    pass