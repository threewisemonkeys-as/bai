import json
import re

def perceive(observation_history: list[str]) -> str:
    obs = observation_history[-1]
    # Extract step number if present
    step_match = re.search(r'Step:\s*(\d+)', obs)
    step = int(step_match.group(1)) if step_match else None

    # Locate and parse the grid JSON array
    start = obs.find('[')
    end = obs.rfind(']') + 1
    if start == -1 or end <= start:
        # Fallback in case the grid is not found (should not happen)
        return "no_grid"
    grid_str = obs[start:end]
    try:
        grid = json.loads(grid_str)
    except Exception:
        return "no_grid"

    # Validate grid structure
    if not isinstance(grid, list) or any(not isinstance(row, list) for row in grid):
        return "no_grid"

    # Extract relevant features
    agent_cells = []
    gray = []
    other = []  # (color, r, c) for any non-black, non-gray, non-agent cells

    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == 'darkgreen':
                agent_cells.append((r, c))
            elif cell == 'gray':
                gray.append((r, c))
            elif cell != 'black':
                other.append((cell, r, c))

    # Build concise summary
    parts = []
    if step is not None:
        parts.append(f"step={step}")
    if agent_cells:
        parts.append(f"agent={agent_cells}")
    else:
        parts.append("agent=None")  # should not happen in normal use
    parts.append(f"gray={gray}")
    if other:
        parts.append(f"other={other}")

    result = ", ".join(parts)
    # Ensure we never return empty
    if not result:
        result = "no_grid"
    return result