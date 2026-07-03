import re
import numpy as np

def parse_grid_string(grid_str: str) -> np.ndarray:
    # Remove outer brackets and split into rows
    grid_str = grid_str.strip()
    if not grid_str.startswith('[') or not grid_str.endswith(']'):
        raise ValueError("Invalid grid string format: must start and end with brackets")

    grid_str = grid_str[1:-1]
    rows_str = re.findall(r'\[(.*?)\]', grid_str) # Find content within each row's brackets

    if not rows_str: # Handle single-row grids which might not have inner brackets
        rows_str = [grid_str] if grid_str else []

    grid_list = []
    for row_str in rows_str:
        if row_str.strip(): # Ensure row is not empty
            grid_list.append(list(map(int, row_str.split(', '))))
    
    if not grid_list:
        return np.array([[]])
        
    return np.array(grid_list, dtype=int)


def perceive(observation_history: list[str]) -> str:
    current_observation_str = observation_history[-1]
    
    # Extract grid and other information
    grid_match = re.search(r'<grid_0>\n((?:\[[\d, ]+\]\n?)+)', current_observation_str)
    if not grid_match:
        # Fallback for when no grid is present
        return f"Observation: {current_observation_str}"

    grid_str = grid_match.group(1).strip()
    grid = parse_grid_string(grid_str)

    # Find the positions of non-background cells (assuming 0 is background)
    non_background_cells = np.argwhere(grid != 0)
    
    summary_parts = []
    if non_background_cells.size > 0:
        # Group cells by their value and report their coordinates
        cell_types = {}
        for r, c in non_background_cells:
            val = grid[r, c]
            if val not in cell_types:
                cell_types[val] = []
            cell_types[val].append(f"({r},{c})")
        
        for val, coords in sorted(cell_types.items()):
            summary_parts.append(f"Type {val}: {';'.join(coords)}")

    if not summary_parts:
        return "Grid: Empty or only background cells."

    return " | ".join(summary_parts)