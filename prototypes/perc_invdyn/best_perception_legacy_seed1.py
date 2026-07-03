import re

def perceive(observation_history: list[str]) -> str:
    current_observation = observation_history[-1]
    
    state_match = re.search(r"State: (\w+)", current_observation)
    levels_completed_match = re.search(r"Levels completed: (\d+/\d+)", current_observation)
    action_count_match = re.search(r"Action count: (\d+)", current_observation)
    
    state = state_match.group(1) if state_match else "UNKNOWN"
    levels_completed = levels_completed_match.group(1) if levels_completed_match else "UNKNOWN"
    action_count = action_count_match.group(1) if action_count_match else "UNKNOWN"
    
    grid_str_match = re.search(r"<grid_0>\n((?:\[[\d,\s]+\]\n?)+)", current_observation)
    
    grid_features = ""
    if grid_str_match:
        grid_str_lines = grid_str_match.group(1).strip().split('\n')
        grid = []
        for line in grid_str_lines:
            # Extract numbers from the list string, handling potential truncated lists at the end
            nums = [int(n) for n in re.findall(r'\d+', line)]
            if nums:
                grid.append(nums)
        
        # Assume a fixed grid size for parsing the provided example (20x20 in the image, but the raw data is long)
        # For this specific environment, we need to carefully reconstruct the grid.
        # The given raw data looks like a flattened list that has been wrapped.
        # Let's assume a standard 20x20 structure based on the image, and then map the values.
        # The raw input starts with a line break and then arrays. Assuming it is flattened and then chunked.
        
        # A more robust parsing for the grid values, considering it might be a single flattened list.
        all_grid_values = []
        for line in grid_str_lines:
            all_grid_values.extend([int(n) for n in re.findall(r'\d+', line)])

        # Determine grid dimensions based on the provided image samples (20x20)
        # This is a heuristic; a real system would need to know grid dimensions.
        rows, cols = 20, 20
        
        if len(all_grid_values) >= rows * cols:
            # We will only process the first rows * cols elements
            grid_flat = all_grid_values[:rows * cols]
            
            # Reconstruct 2D grid for easier access
            grid_2d = [grid_flat[i * cols:(i + 1) * cols] for i in range(rows)]

            # Extract features from the grid_2d
            
            # Find the "player" (assuming pink or yellow block, let's identify the pink square)
            player_pos = None
            goal_pos = None
            lava_bars = [] # red bars

            # Simplified color mapping for the provided images (e.g., pink=12, yellow=13, red=14, blue=11)
            # This mapping is guessed from the numerical values and the image.
            # 12 is pink/player, 13 is yellow/goal indicator, 14 is red/obstacle, 11 is blue/water/boundary
            
            for r in range(rows):
                for c in range(cols):
                    cell_value = grid_2d[r][c]
                    if cell_value == 12:  # Assuming 12 is the pink player
                        player_pos = (r, c)
                    elif cell_value == 13: # Assuming 13 is the yellow goal indicator
                        goal_pos = (r, c)
                    elif cell_value == 14: # Assuming 14 is the red lava bar
                        lava_bars.append((r,c))


            if player_pos:
                grid_features += f"player_pos=({player_pos[0]},{player_pos[1]});"
            if goal_pos:
                grid_features += f"goal_pos=({goal_pos[0]},{goal_pos[1]});"
            
            # Summarize lava bars (e.g., count, extent)
            if lava_bars:
                min_r = min(p[0] for p in lava_bars)
                max_r = max(p[0] for p in lava_bars)
                min_c = min(p[1] for p in lava_bars)
                max_c = max(p[1] for p in lava_bars)
                grid_features += f"lava_bar_count={len(lava_bars)};" \
                                 f"lava_bar_span=({min_r},{min_c})_to_({max_r},{max_c});"
            
            # Add general environment observation from the grid periphery
            # Example: check top-left, top-right, bottom-left, bottom-right cells
            # This is a very simple feature, might need to be more complex based on specific game needs.
            top_left_cell = grid_2d[0][0]
            top_right_cell = grid_2d[0][cols-1]
            bottom_left_cell = grid_2d[rows-1][0]
            bottom_right_cell = grid_2d[rows-1][cols-1]
            
            grid_features += f"TL={top_left_cell};TR={top_right_cell};BL={bottom_left_cell};BR={bottom_right_cell};"

    summary = f"state={state};levels={levels_completed};actions={action_count};{grid_features}"
    
    # Ensure the summary is concise and never empty
    if not summary:
        return "No discernible features." # Fallback, should not be reached with current implementation
    return summary