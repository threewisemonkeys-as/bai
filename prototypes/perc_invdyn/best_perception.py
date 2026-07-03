import json

def _extract_grid_data(obs_string: str) -> list[list[str]] | None:
    """Extracts the grid data from the observation string."""
    start = obs_string.find('[["black"')
    end = obs_string.rfind(']]') + 2
    if start == -1 or end <= 1:
        return None
    try:
        grid = json.loads(obs_string[start:end])
        return grid
    except Exception:
        return None

def perceive(observation_history: list[str]) -> str:
    current_obs_str = observation_history[-1]
    current_grid = _extract_grid_data(current_obs_str)

    if current_grid is None:
        return "summary=no_grid_data"

    grid_height = len(current_grid)
    grid_width = len(current_grid[0]) if grid_height > 0 else 0

    agent_pos = None
    gray_cells = []
    obstacles = [] # Identify 'black' cells as obstacles (walls or boundaries)

    for r, row in enumerate(current_grid):
        for c, cell in enumerate(row):
            if cell == "darkgreen":
                agent_pos = (r, c)
            elif cell == "gray":
                gray_cells.append((r, c))
            elif cell == "black": # Assuming 'black' cells are impassable obstacles
                obstacles.append((r, c))

    parts = []

    # Grid dimensions and coordinate system (row, col)
    parts.append(f"grid_dimensions=({grid_height},{grid_width})")
    parts.append("coordinate_system=row_increases_down_col_increases_right")

    if agent_pos is not None:
        parts.append(f"agent_pos={agent_pos}")
    else:
        parts.append("agent_pos=unknown")

    if gray_cells:
        # Sort for consistent output, important for comparing across observations
        gray_cells.sort() 
        parts.append(f"gray_cell_positions={gray_cells}")
    else:
        parts.append("gray_cell_positions=[]")

    # Only include obstacles that are immediately adjacent to the agent or could block movement
    # This keeps the summary concise while still providing important info.
    # For now, let's just note if boundaries exist, as we assume "black" is a general obstacle.
    # A more sophisticated model might check agent's 1-cell neighborhood for black squares that aren't borders.
    
    # Check for immediate boundaries/walls if agent_pos is known
    possible_blocked_directions = []
    if agent_pos is not None:
        r, c = agent_pos
        # Check cells adjacent to the agent
        # Assuming black cells are obstacles that block movement
        if r == 0 or current_grid[r-1][c] == "black":
            possible_blocked_directions.append("up")
        if r == grid_height - 1 or current_grid[r+1][c] == "black":
            possible_blocked_directions.append("down")
        if c == 0 or current_grid[r][c-1] == "black":
            possible_blocked_directions.append("left")
        if c == grid_width - 1 or current_grid[r][c+1] == "black":
            possible_blocked_directions.append("right")

    if possible_blocked_directions:
        parts.append(f"blocked_movements_around_agent={sorted(possible_blocked_directions)}")
    else:
        parts.append("blocked_movements_around_agent=[]")

    # Add information about interaction rules if known (e.g., can push gray cells)
    # For now, assume gray cells are passive unless specified otherwise in the observation directly.
    # Since the example observations don't provide this, we'll note it as unknown.
    parts.append("gray_cell_interaction_rules=unknown_can_be_pushed_etc")

    return "; ".join(parts)