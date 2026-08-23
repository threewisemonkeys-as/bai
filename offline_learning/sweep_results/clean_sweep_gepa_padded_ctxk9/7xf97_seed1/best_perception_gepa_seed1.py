import json
import re
import ast
from collections import Counter
from typing import Tuple, List, Dict

def _parse_grid(obs: str) -> List[List]:
    """Return a 2D list of cell values (integers or strings) parsed from obs."""
    if '[[' in obs:
        # Autumn JSON grid
        start = obs.find('[[')
        end = obs.rfind(']]') + 2
        if end <= start:
            raise ValueError("Cannot locate grid boundaries")
        grid_str = obs[start:end]
        return json.loads(grid_str)      # list of lists of strings
    else:
        # ARC integer grid
        lines = obs.splitlines()
        grid_lines = []
        in_grid = False
        for line in lines:
            if '<grid_' in line:
                in_grid = True
                continue
            if in_grid:
                if line.strip().startswith('<grid_'):
                    break
                if line.strip().startswith('['):
                    grid_lines.append(line.strip())
        if not grid_lines:
            raise ValueError("No grid lines found")
        return [ast.literal_eval(line) for line in grid_lines]

def _extract_step(obs: str) -> str:
    """Return step number or action count as a string, empty if not found."""
    m = re.search(r'\bStep:\s*(\d+)', obs)
    if m:
        return m.group(1)
    m = re.search(r'Action count:\s*(\d+)', obs)
    if m:
        return m.group(1)
    return "?"

def _non_bg_cells(grid):
    """Return (background, list of (r,c,value)) for non-background cells, sorted row-major."""
    flat = [cell for row in grid for cell in row]
    bg = Counter(flat).most_common(1)[0][0]
    cells = [(r, c, cell) for r, row in enumerate(grid)
             for c, cell in enumerate(row) if cell != bg]
    cells.sort(key=lambda x: (x[0], x[1]))  # ensure row-major order
    return bg, cells

def _count_blue(grid) -> int:
    """Number of cells that are blue (either integer 1 or string 'blue')."""
    if not grid:
        return 0
    sample = grid[0][0]
    if isinstance(sample, int):
        blue_val = 1
        return sum(1 for row in grid for cell in row if cell == blue_val)
    else:
        return sum(1 for row in grid for cell in row if cell == "blue")

def _compute_velocities(current_cells, prev_cells):
    """
    For each current cell (r,c,color), estimate its velocity by matching
    to the closest previous cell of the same colour. Returns dict mapping
    (r,c) -> (dr, dc) or (0,0) if no match.
    """
    prev_by_color = {}
    for pr, pc, pcolor in prev_cells:
        prev_by_color.setdefault(pcolor, []).append((pr, pc))

    used = set()
    velocities = {}
    for r, c, color in current_cells:
        best_dist = None
        best_dr = best_dc = 0
        prev_list = prev_by_color.get(color, [])
        for pi, (pr, pc) in enumerate(prev_list):
            if pi in used:
                continue
            dist = abs(r - pr) + abs(c - pc)
            if dist <= 3 and (best_dist is None or dist < best_dist):
                best_dist = dist
                best_dr = r - pr
                best_dc = c - pc
                best_pi = pi
        if best_dist is not None:
            velocities[(r, c)] = (best_dr, best_dc)
            used.add(best_pi)
        else:
            velocities[(r, c)] = (0, 0)
    return velocities

def _grid_equal(grid1, grid2) -> bool:
    """Check if two parsed grids are exactly equal."""
    if len(grid1) != len(grid2) or len(grid1[0]) != len(grid2[0]):
        return False
    for r in range(len(grid1)):
        for c in range(len(grid1[0])):
            if grid1[r][c] != grid2[r][c]:
                return False
    return True

def _compute_still_count(observation_history: List[str], current_grid) -> int:
    """
    Count consecutive steps (including current) where the grid has not changed.
    Returns 1 if no history or previous grid differs.
    """
    if len(observation_history) < 2:
        return 1
    # Start from the end and go back
    count = 1
    prev_grid = current_grid
    for i in range(len(observation_history)-2, -1, -1):
        try:
            grid = _parse_grid(observation_history[i])
            if _grid_equal(grid, prev_grid):
                count += 1
                prev_grid = grid
            else:
                break
        except Exception:
            break
    return count

def perceive(observation_history: list[str]) -> str:
    """
    Parse the raw observation (last element of observation_history) and return
    a concise text summary of decision-relevant features.

    Features:
    - step number
    - grid dimensions
    - background colour
    - all non-background cells with velocities (always reported, even if zero)
    - number of blue cells
    - consecutive identical frames (still count)
    """
    if not observation_history:
        return "perception_error: empty history"

    obs = observation_history[-1]
    try:
        step = _extract_step(obs)
        grid = _parse_grid(obs)
        if not grid or not grid[0]:
            raise ValueError("Empty grid")
        rows = len(grid)
        cols = len(grid[0])
        bg, current_cells = _non_bg_cells(grid)
        blue_count = _count_blue(grid)
        still = _compute_still_count(observation_history, grid)

        # Compute velocities if previous observation exists
        velocities = {}
        if len(observation_history) >= 2:
            prev_obs = observation_history[-2]
            try:
                prev_grid = _parse_grid(prev_obs)
                _, prev_cells = _non_bg_cells(prev_grid)
                velocities = _compute_velocities(current_cells, prev_cells)
            except Exception:
                pass

        # Build output – always include velocity even if (0,0)
        parts = [
            f"step {step}",
            f"grid {rows}x{cols}",
            f"bg {bg}",
            f"blue {blue_count}",
            f"still {still}"
        ]
        if current_cells:
            cell_strs = []
            for r, c, color in current_cells:
                dr, dc = velocities.get((r, c), (0, 0))
                cell_strs.append(f"r{r}c{c}:{color}+{dr},{dc}")
            all_cells = "; ".join(cell_strs)
            max_len = 1900  # leave some room for prefix
            if len(all_cells) > max_len:
                all_cells = all_cells[:max_len-3] + "..."
            parts.append(f"non_bg: {all_cells}")
        else:
            parts.append("non_bg: none")

        return "; ".join(parts)

    except Exception as e:
        # Fallback – never raise, never return empty
        step = _extract_step(observation_history[-1])
        return f"step {step}; perception_parse_error: {e}"