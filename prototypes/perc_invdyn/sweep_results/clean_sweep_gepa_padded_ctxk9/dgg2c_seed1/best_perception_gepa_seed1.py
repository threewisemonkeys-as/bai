import json
import re
from typing import List, Tuple, Optional

def perceive(observation_history: list[str]) -> str:
    """
    Parse the raw observation text and produce a concise summary of decision-relevant features.
    The output must:
    - Be <2000 characters
    - Never be empty
    - Never raise exceptions
    - Capture enough state that the action between two consecutive states is recoverable
    """
    try:
        obs = observation_history[-1] if observation_history else ""
        if not obs:
            return "[]"
        
        # ----------------------------------------------------------------------
        # 1. Extract metadata (step number, action count, levels completed)
        # ----------------------------------------------------------------------
        step = 0
        action_count = 0
        levels_completed = "0/6"
        m_step = re.search(r"Step:\s*(\d+)", obs)
        if m_step:
            step = int(m_step.group(1))
        m_count = re.search(r"Action count:\s*(\d+)", obs)
        if m_count:
            action_count = int(m_count.group(1))
        m_levels = re.search(r"Levels completed:\s*(\S+)", obs)
        if m_levels:
            levels_completed = m_levels.group(1)
        
        # ----------------------------------------------------------------------
        # 2. Parse all grid blocks
        # ----------------------------------------------------------------------
        # Each block starts with <grid_k> and the grid data follows.
        # The grid may be in autumn (JSON) or ARC (line-by-line bracket lists) format.
        # We locate all blocks by searching for "<grid_" markers.
        
        all_grids = []          # list of (grid_index, int_grid)
        block_pattern = re.compile(r"<grid_(\d+)>")
        
        pos = 0
        while True:
            m = block_pattern.search(obs, pos)
            if not m:
                break
            grid_idx = int(m.group(1))
            block_start = m.end()   # after the marker
            # Find the next marker or end of string to determine block end
            next_m = block_pattern.search(obs, block_start)
            if next_m:
                block_text = obs[block_start:next_m.start()]
            else:
                block_text = obs[block_start:]
            
            # Try to parse the block as JSON (autumn)
            grid = None
            json_start = block_text.find("[[")
            if json_start != -1:
                json_end = block_text.rfind("]]")
                if json_end != -1:
                    try:
                        json_str = block_text[json_start:json_end+2]
                        parsed = json.loads(json_str)
                        # Ensure it's a 2D list of strings
                        if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], list):
                            grid = parsed
                    except (json.JSONDecodeError, ValueError):
                        pass
            
            # If JSON failed, try ARC integer grid (line-by-line bracket lists)
            if grid is None:
                lines = block_text.split('\n')
                int_rows = []
                for line in lines:
                    line_stripped = line.strip()
                    if line_stripped.startswith('[') and line_stripped.endswith(']'):
                        try:
                            nums_str = line_stripped.strip('[]').split(',')
                            row = [int(x.strip()) for x in nums_str]
                            int_rows.append(row)
                        except (ValueError, IndexError):
                            continue
                if int_rows:
                    grid = int_rows
            
            if grid is not None and len(grid) > 0 and isinstance(grid[0], list):
                all_grids.append((grid_idx, grid))
            
            pos = m.end()   # move past the marker to avoid infinite loop
        
        # If no grids found, fall back to searching the whole text for a JSON array
        if not all_grids:
            json_start = obs.find("[[")
            if json_start != -1:
                json_end = obs.rfind("]]")
                if json_end != -1:
                    try:
                        json_str = obs[json_start:json_end+2]
                        parsed = json.loads(json_str)
                        if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], list):
                            all_grids.append((0, parsed))
                    except (json.JSONDecodeError, ValueError):
                        pass
        
        # If still no grid, try the ARC fallback on the whole text
        if not all_grids:
            lines = obs.split('\n')
            int_rows = []
            for line in lines:
                line_stripped = line.strip()
                if line_stripped.startswith('[') and line_stripped.endswith(']'):
                    try:
                        nums_str = line_stripped.strip('[]').split(',')
                        row = [int(x.strip()) for x in nums_str]
                        int_rows.append(row)
                    except (ValueError, IndexError):
                        continue
            if int_rows:
                all_grids.append((0, int_rows))
        
        if not all_grids:
            # Last resort: return a minimal summary with metadata
            return f"[step:{step}, count:{action_count}, levels:{levels_completed}]"
        
        # ----------------------------------------------------------------------
        # 3. Convert colour names to integers if needed
        # ----------------------------------------------------------------------
        colour_map = {
            "black": 0, "blue": 1, "red": 2, "green": 3, "yellow": 4,
            "gray": 5, "gold": 5, "magenta": 6, "orange": 7, "lightblue": 8,
            "maroon": 9, "white": 10, "darkgray": 11, "darkgreen": 3,
            "light-gray": 5, "dark-red": 9, "light-blue": 8, "dark-gray": 11
        }
        
        def cell_value(cell):
            if isinstance(cell, str):
                return colour_map.get(cell.lower(), 5)  # default gray
            else:
                return int(cell)
        
        # ----------------------------------------------------------------------
        # 4. Extract decision‑relevant cells from each grid
        #    We keep all non‑black and non‑gray cells (i.e. colour != 0 and != 5)
        #    and also record the cell colour.
        # ----------------------------------------------------------------------
        all_features = []
        for grid_idx, grid in all_grids:
            rows = len(grid)
            if rows == 0:
                continue
            cols = len(grid[0]) if isinstance(grid[0], list) else 0
            if cols == 0:
                continue
            
            for r in range(rows):
                for c in range(cols):
                    val = cell_value(grid[r][c])
                    # Keep any cell that is not background (black=0, gray=5)
                    if val != 0 and val != 5:
                        # Format: (grid_idx, row, col, colour)
                        all_features.append((grid_idx, r, c, val))
        
        # ----------------------------------------------------------------------
        # 5. Build the output string
        # ----------------------------------------------------------------------
        parts = []
        # Metadata first (step, action count, levels) – essential for disambiguation
        parts.append(f"step:{step}")
        parts.append(f"count:{action_count}")
        parts.append(f"levels:{levels_completed}")
        
        # Then all non‑background cells, sorted for consistency
        if all_features:
            cell_strs = []
            # Sort by grid index, then row, then column
            all_features.sort(key=lambda x: (x[0], x[1], x[2]))
            for gf in all_features:
                cell_strs.append(f"(g{gf[0]}:{gf[1]},{gf[2]},{gf[3]})")
            parts.append("cells:" + ",".join(cell_strs))
        else:
            # If no special cells, report some spatial info from the first grid
            grid = all_grids[0][1]
            rows = len(grid)
            cols = len(grid[0]) if rows>0 and isinstance(grid[0],list) else 0
            # Report the first few cells of the grid to give a minimal signature
            signature = []
            for r in range(min(4, rows)):
                row_data = grid[r]
                if isinstance(row_data, list):
                    for c in range(min(4, len(row_data))):
                        val = cell_value(row_data[c])
                        signature.append(f"(g0:{r},{c},{val})")
            if signature:
                parts.append("cells:" + ",".join(signature))
            else:
                # Truly empty grid – fallback
                parts.append("empty_grid")
        
        result = " ".join(parts)
        # Ensure <=2000 chars
        if len(result) > 2000:
            result = result[:1997] + "..."
        return result
        
    except Exception:
        # Never raise, never return empty
        return "[]"