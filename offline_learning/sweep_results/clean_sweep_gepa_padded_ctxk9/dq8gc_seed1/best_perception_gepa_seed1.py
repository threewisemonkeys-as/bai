import json
import ast
import re
from typing import List, Tuple, Optional

def perceive(observation_history: list[str]) -> str:
    """
    Parse the latest raw observation and return a concise text summary of
    decision-relevant features (<2000 chars).  Never raises, never returns empty.
    """
    # ------------------------------------------------------------------
    # 0. Safely get the current observation
    # ------------------------------------------------------------------
    obs = observation_history[-1] if observation_history else ""

    # ------------------------------------------------------------------
    # 1. Try to extract metadata (step, action_count, state) from any encoding
    # ------------------------------------------------------------------
    step = ""
    action_count = ""
    state = ""
    try:
        m = re.search(r"Step:\s*(\d+)", obs)
        if m:
            step = m.group(1)
        m = re.search(r"Action count:\s*(\d+)", obs)
        if m:
            action_count = m.group(1)
        m = re.search(r"State:\s*(\S+)", obs)
        if m:
            state = m.group(1)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 2. Try to extract the grid in either encoding
    # ------------------------------------------------------------------
    grid = None
    try:
        # ---- ARC integer grid ----
        if '<grid_' in obs:
            lines = obs.splitlines()
            grid_start = None
            for i, line in enumerate(lines):
                if line.strip().startswith('<grid_'):
                    grid_start = i
                    break
            if grid_start is not None:
                rows = []
                for line in lines[grid_start + 1:]:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    if stripped.startswith('[') and stripped.endswith(']'):
                        row = ast.literal_eval(stripped)
                        rows.append(row)
                    elif stripped.startswith('<grid_'):
                        break   # stop at next grid marker
                if rows:
                    grid = rows
        # ---- Autumn string grid (JSON 2D array) ----
        else:
            start = obs.find('[[')
            end = obs.rfind(']]')
            if start != -1 and end != -1:
                json_str = obs[start:end + 2]
                grid = json.loads(json_str)
    except Exception:
        grid = None

    # ------------------------------------------------------------------
    # 3. Fallback if parsing failed
    # ------------------------------------------------------------------
    if not grid:
        meta = f"step={step};action_count={action_count};state={state}" if (step or action_count or state) else ""
        out = "agent=(0,0,none);objects=[];grid=0x0"
        if meta:
            out = meta + ";" + out
        return out

    # ------------------------------------------------------------------
    # 4. Determine grid dimensions
    # ------------------------------------------------------------------
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    # ------------------------------------------------------------------
    # 5. Colour helper and palette
    # ------------------------------------------------------------------
    INT_PALETTE = {
        0: 'black', 1: 'blue', 2: 'red', 3: 'green',
        4: 'yellow', 5: 'light-gray', 6: 'magenta',
        7: 'orange', 8: 'light-blue', 9: 'dark-red',
        10: 'white', 11: 'dark-gray'
    }
    AGENT_CANDIDATES = {'blue', 'red', 'green', 'yellow', 'magenta',
                        'orange', 'light-blue', 'white'}

    def cell_colour(cell) -> str:
        if isinstance(cell, int):
            return INT_PALETTE.get(cell, 'unknown')
        else:
            return cell.lower()

    # ------------------------------------------------------------------
    # 6. Count colours and collect all non-black cells
    # ------------------------------------------------------------------
    colour_count = {}
    all_cells = []   # list of (row, col, colour)
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            col = cell_colour(cell)
            if col == 'black':
                continue
            colour_count[col] = colour_count.get(col, 0) + 1
            all_cells.append((r, c, col))

    # ------------------------------------------------------------------
    # 7. Identify agent position and colour
    # ------------------------------------------------------------------
    agent_pos = None
    agent_colour = None

    # a) prefer a colour from AGENT_CANDIDATES with count == 1
    for colour in AGENT_CANDIDATES:
        if colour_count.get(colour) == 1:
            for r, c, col in all_cells:
                if col == colour:
                    agent_pos = (r, c)
                    agent_colour = colour
                    break
            break
    # b) fall back to any colour that appears exactly once
    if agent_pos is None:
        for col, cnt in colour_count.items():
            if cnt == 1:
                for r, c, col2 in all_cells:
                    if col2 == col:
                        agent_pos = (r, c)
                        agent_colour = col
                        break
                break
    # c) last resort: use the first non‑black cell
    if agent_pos is None and all_cells:
        agent_pos = all_cells[0][:2]
        agent_colour = all_cells[0][2]

    # If still nothing (only black cells) use (0,0) with 'none'
    if agent_pos is None:
        agent_pos = (0, 0)
        agent_colour = 'none'

    # ------------------------------------------------------------------
    # 8. Build list of non‑black cells that are NOT the agent
    # ------------------------------------------------------------------
    objects = []
    for r, c, col in all_cells:
        if (r, c) == agent_pos:
            continue
        objects.append((r, c, col))

    # ------------------------------------------------------------------
    # 9. Compute adjacency info (4 neighbours of the agent)
    # ------------------------------------------------------------------
    ar, ac = agent_pos
    directions = {
        'up': (ar - 1, ac),
        'down': (ar + 1, ac),
        'left': (ar, ac - 1),
        'right': (ar, ac + 1)
    }
    adj_strs = []
    for dir_name, (nr, nc) in directions.items():
        if 0 <= nr < rows and 0 <= nc < cols:
            colour = cell_colour(grid[nr][nc])
            adj_strs.append(f"adj_{dir_name}=({nr},{nc},{colour})")
        else:
            adj_strs.append(f"adj_{dir_name}=out")

    # ------------------------------------------------------------------
    # 10. Boundary flags
    # ------------------------------------------------------------------
    edges = {
        'up': ar == 0,
        'down': ar == rows - 1,
        'left': ac == 0,
        'right': ac == cols - 1
    }
    edge_strs = [f"edge_{k}={str(v).lower()}" for k, v in edges.items()]

    # ------------------------------------------------------------------
    # 11. Construct the output string (<2000 chars)
    # ------------------------------------------------------------------
    meta = f"step={step};action_count={action_count};state={state}" if (step or action_count or state) else ""
    grid_info = f"grid={rows}x{cols}"
    agent_str = f"agent=({agent_pos[0]},{agent_pos[1]},{agent_colour})"
    obj_items = [f"({r},{c},{col})" for r, c, col in objects]
    obj_str = ";".join(obj_items)
    obj_list = f"objects=[{obj_str}]"

    # Combine all parts
    parts = [grid_info, agent_str, obj_list] + adj_strs + edge_strs
    if meta:
        parts.insert(0, meta)

    result = ";".join(parts)

    # Truncate if over limit (should be rare)
    if len(result) > 2000:
        # Keep the essential parts (meta, grid, agent, objects) and drop adjacency/edges if needed
        essential = [meta, grid_info, agent_str, obj_list] if meta else [grid_info, agent_str, obj_list]
        result = ";".join(essential)
        if len(result) > 2000:
            # Further truncate objects
            max_obj_len = 2000 - len(essential[0]) - len(essential[1]) - len(essential[2]) - 2  # leave room
            if max_obj_len < 0:
                result = essential[0] + ";" + essential[1] + ";" + essential[2] + ";objects=[]"
            else:
                truncated = obj_str[:max_obj_len]
                result = essential[0] + ";" + essential[1] + ";" + essential[2] + f";objects=[{truncated}...]"

    # Final safety: never return empty
    if not result.strip():
        result = "grid=0x0;agent=(0,0,none);objects=[]"

    return result