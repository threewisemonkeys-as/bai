import json
import re
from typing import List, Tuple, Union, Optional

# Colour code mapping for ARC integer grids
ARC_COLORS = {
    0: 'black', 1: 'blue', 2: 'red', 3: 'green', 4: 'yellow',
    5: 'light-gray', 6: 'magenta', 7: 'orange', 8: 'light-blue',
    9: 'dark-red', 10: 'white', 11: 'dark-gray'
}
# Inverse mapping for convenience
INT_COLORS = {v: k for k, v in ARC_COLORS.items()}

def _parse_grid(obs: str) -> Optional[List[List[Union[int, str]]]]:
    """Try to parse the observation string into a 2D grid.
    Returns None if parsing fails."""
    # Try ARC integer grid format
    marker_match = re.search(r'<grid_\d+>\s*\n', obs)
    if marker_match:
        start = marker_match.end()
        # Collect lines until a blank line or end of string
        lines = []
        for line in obs[start:].split('\n'):
            stripped = line.strip()
            if not stripped:
                break
            # Each line should be like [14, 14, 14, ... , 14]
            if stripped.startswith('[') and stripped.endswith(']'):
                try:
                    row = json.loads(stripped)
                    if isinstance(row, list) and all(isinstance(x, int) for x in row):
                        lines.append(row)
                except (json.JSONDecodeError, TypeError):
                    pass
        if lines:
            # Check rectangular consistency (optional)
            return lines

    # Try Autumn string grid format
    # Look for [[ ... ]] JSON array
    start_idx = obs.find('[[[')  # sometimes triple brackets? Actually double brackets
    # Safer: find first occurrence of "[[" followed by a string in quotes
    start_idx = obs.find('[[')
    if start_idx == -1:
        return None
    # Find matching closing "]]"
    end_idx = obs.rfind(']]')
    if end_idx == -1 or end_idx <= start_idx + 1:
        return None
    json_str = obs[start_idx:end_idx+2]
    try:
        grid = json.loads(json_str)
        if isinstance(grid, list) and all(isinstance(row, list) for row in grid):
            # Convert all cells to strings (some might be numbers in mixed format)
            for r_idx, row in enumerate(grid):
                for c_idx, cell in enumerate(row):
                    if isinstance(cell, int):
                        # Convert ARC integer to colour name if possible
                        grid[r_idx][c_idx] = ARC_COLORS.get(cell, f"int{cell}")
                    elif not isinstance(cell, str):
                        return None
            return grid
    except (json.JSONDecodeError, TypeError):
        pass
    return None

def _most_common_color(grid: List[List[Union[int, str]]]) -> Union[int, str]:
    """Return the most frequent cell value (background)."""
    flat = [cell for row in grid for cell in row]
    if not flat:
        return 0 if isinstance(flat, list) else 'black'
    from collections import Counter
    return Counter(flat).most_common(1)[0][0]

def perceive(observation_history: list[str]) -> str:
    # Get current observation (last in history)
    if not observation_history:
        return "no observation"
    obs = observation_history[-1]
    
    grid = _parse_grid(obs)
    if grid is None:
        return "parse_error"  # never empty
    
    H = len(grid)
    W = len(grid[0]) if H > 0 else 0
    
    # Determine background
    background = _most_common_color(grid)
    # For colour names, treat 'black' as default background if present
    if isinstance(background, str):
        if background == 'black':
            pass
        else:
            # Could be another dominant color; still use it as background
            pass
    else:  # integer
        if background == 0:
            pass
        else:
            pass
    
    # Identify agent: look for red (2 or "red") – this is the only moving object in examples
    agent_pos: Optional[Tuple[int, int]] = None
    agent_color = 2 if isinstance(grid[0][0], int) else 'red'
    other_cells: List[Tuple[int, int, Union[int, str]]] = []
    
    for r in range(H):
        for c in range(W):
            cell = grid[r][c]
            if cell == background:
                continue
            # Check if this is the agent (red)
            if cell == agent_color:
                if agent_pos is None:
                    agent_pos = (r, c)
                else:
                    # Multiple red cells – treat all as non-background
                    other_cells.append((r, c, cell))
            else:
                other_cells.append((r, c, cell))
    
    # Build compact summary
    parts = []
    # Dimensions
    parts.append(f"grid:{H}x{W}")
    # Agent position
    if agent_pos is not None:
        parts.append(f"A:({agent_pos[0]},{agent_pos[1]})")
    else:
        parts.append("A:none")
    # Other non-background cells (sorted for determinism)
    if other_cells:
        # Colour may be int or string; convert to short name
        def fmt_color(clr):
            if isinstance(clr, int):
                return ARC_COLORS.get(clr, f"c{clr}")
            else:
                return clr
        cell_strs = [f"({r},{c},{fmt_color(clr)})" for r, c, clr in sorted(other_cells)]
        # Join but keep under 2000 chars
        cell_block = ";".join(cell_strs)
        if len(cell_block) > 1800:
            # Truncate from the end
            cell_block = cell_block[:1797] + "..."
        parts.append(f"C:[{cell_block}]")
    else:
        parts.append("C:[]")
    
    result = " ".join(parts)
    # Final safety: never empty
    if not result.strip():
        return "empty_grid"
    return result