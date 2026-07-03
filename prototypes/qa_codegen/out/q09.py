import numpy as np

def answer(steps: list[StepRecord]) -> str:
    # Define a helper to check if a 3x3 grid contains values of interest
    def check_grid_for_elements(grid):
        if grid is None or grid.shape != (64, 64):
            return False

        # Iterate through 3x3 regions
        for r_start in range(0, 64, 3):
            for c_start in range(0, 64, 3):
                # Ensure the 3x3 region is within bounds
                if r_start + 2 < 64 and c_start + 2 < 64:
                    subgrid = grid[r_start:r_start+3, c_start:c_start+3]

                    # Check for the central white/red element in the 3x3 subgrid
                    # The center is at (1,1) in the 3x3 subgrid
                    center_val = subgrid[1, 1]
                    if center_val == 0 or center_val == 8:  # 0 is white, 8 is red
                        return True
        return False

    can_be_selected_and_acted_upon = "MAYBE"

    for step in steps:
        if step.action_type == "ACTION6" and step.x is not None and step.y is not None:
            clicked_x, clicked_y = step.x, step.y

            # Determine which 3x3 grid the click falls into
            grid_x_start = (clicked_x // 3) * 3
            grid_y_start = (clicked_y // 3) * 3

            # Check if the clicked cell is the central element of a 3x3 grid
            # The central element of a 3x3 grid is at (row_offset + 1, col_offset + 1)
            is_central_element_x = (clicked_x == grid_x_start + 1)
            is_central_element_y = (clicked_y == grid_y_start + 1)

            if is_central_element_x and is_central_element_y:
                clicked_value = step.clicked_cell_pre()
                if clicked_value is not None and (clicked_value == 0 or clicked_value == 8):
                    # We found a central white/red element being clicked
                    # Now check if an action was performed (i.e., something changed)
                    if step.any_change():
                        # The central white/red element was clicked AND something changed.
                        # This implies it can be acted upon.
                        return "YES"

    return can_be_selected_and_acted_upon