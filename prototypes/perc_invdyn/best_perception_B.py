import json

def perceive(observation_history: list[str]) -> str:
    current_observation_str = observation_history[-1]
    
    try:
        # Extract the grid part from the observation string
        grid_start_idx = current_observation_str.find("[[")
        grid_end_idx = current_observation_str.rfind("]]")
        
        if grid_start_idx == -1 or grid_end_idx == -1:
            return "No grid found in observation."

        grid_str = current_observation_str[grid_start_idx : grid_end_idx + 2]
        grid = json.loads(grid_str)

        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0

        features = []

        # Find agent(s) (darkgreen)
        agent_positions = []
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == "darkgreen":
                    agent_positions.append((r, c))

        if agent_positions:
            # Sort agent positions for consistent output
            agent_positions.sort()
            features.append(f"Agents at: {agent_positions}")
        else:
            features.append("No agents found.")
            
        # Find gray cells
        gray_cells = []
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == "gray":
                    gray_cells.append((r, c))
        
        if gray_cells:
            # Sort gray cell positions for consistent output
            gray_cells.sort()
            features.append(f"Gray cells at: {gray_cells}")
        
        # Check if the grid is all black (and no agents/gray cells)
        # This check is now less aggressive and only applies if no other features are found.
        if not agent_positions and not gray_cells and all(cell == "black" for row in grid for cell in row):
            return "The grid is entirely black (no agents or gray cells)."

        if not features:
            # This case should ideally not be hit if we output "No agents found."
            # but as a fallback for robustness.
            return "No features detected."

        return "; ".join(features)

    except Exception as e:
        # Ensure the summary is concise and doesn't exceed 2000 characters.
        error_message = f"Error during perception: {type(e).__name__}: {e}. "
        # Truncate the raw observation snippet if it's too long
        raw_snippet = current_observation_str[:150]
        if len(current_observation_str) > 150:
            raw_snippet += "..."
        error_message += f"Raw observation snippet: {raw_snippet}"
        return error_message[:2000] # Ensure overall message doesn't exceed limit