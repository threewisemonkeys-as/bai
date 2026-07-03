import json
import re

def perceive(observation_history: list[str]) -> str:
    # ----- parse current observation step number and grid -----
    obs = observation_history[-1]
    step_match = re.search(r"Step:\s*(\d+)", obs)
    step = int(step_match.group(1)) if step_match else -1

    # find the grid as the outermost [[ ... ]]
    grid_start = obs.find('[')
    if grid_start == -1:
        return json.dumps({"error": "no_grid_start"})
    # find the matching end: count brackets
    # simpler: locate last "]]" that closes the grid
    grid_end = obs.rfind(']]')
    if grid_end == -1:
        return json.dumps({"error": "no_grid_end"})
    grid_str = obs[grid_start:grid_end+2]
    try:
        grid = json.loads(grid_str)
    except Exception:
        return json.dumps({"error": "grid_parse_fail"})

    # collect positions for each non-black color
    colors = {}
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell != "black":
                colors.setdefault(cell, []).append([r, c])

    # ----- if history length > 1, detect changes from previous -----
    movement = None
    if len(observation_history) >= 2:
        prev_obs = observation_history[-2]
        # parse previous grid similarly
        prev_grid_start = prev_obs.find('[')
        prev_grid_end = prev_obs.rfind(']]')
        if prev_grid_start != -1 and prev_grid_end != -1:
            prev_grid_str = prev_obs[prev_grid_start:prev_grid_end+2]
            try:
                prev_grid = json.loads(prev_grid_str)
            except Exception:
                prev_grid = None
            if prev_grid is not None:
                # find cells that changed (only considering non-black)
                current_cells = set((r, c) for r, row in enumerate(grid) for c, cell in enumerate(row) if cell != "black")
                prev_cells = set((r, c) for r, row in enumerate(prev_grid) for c, cell in enumerate(row) if cell != "black")
                added = current_cells - prev_cells
                removed = prev_cells - current_cells
                # assume one agent moving: if exactly one added and one removed, report as movement
                if len(added) == 1 and len(removed) == 1:
                    movement = {
                        "from": list(removed.pop()),
                        "to": list(added.pop())
                    }

    # ----- assemble summary -----
    result = {
        "step": step,
        "grid_size": [len(grid), len(grid[0]) if grid else 0]
    }
    # include all non-black positions
    for color, positions in colors.items():
        result[color + "_positions"] = positions
    if movement:
        result["movement"] = movement

    # return as JSON string (concise)
    return json.dumps(result)