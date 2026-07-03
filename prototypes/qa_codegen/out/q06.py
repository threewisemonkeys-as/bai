import numpy as np

def answer(steps: list[StepRecord]) -> str:
    # A "highlighted 3x3 grid" is typically indicated by a specific color surrounding it.
    # From ARC-AGI, this often means light-gray (2) or dark-gray (4) borders.
    # The "central element" of a 3x3 grid is the cell at (row+1, col+1) if the top-left is (row, col).
    # "Unchangeable blue perimeter squares": blue (9) squares that previously did not change upon clicking.
    # "Can now turn red upon subsequent clicks": after clicking the central element,
    # clicking a blue perimeter square makes it red (8).

    # State machine to track the sequence of events
    clicked_center_of_highlighted_grid = False
    per_grid_state = {}  # Stores state for each potential 3x3 highlighted grid: (top_row, top_col) -> {'blue_perimeters': set, 'changed_red_after_center_click': False}

    for i, step in enumerate(steps):
        if step.x is None or step.y is None:
            continue

        clicked_row, clicked_col = step.y, step.x
        pre_grid = step.pre
        post_grid = step.post

        if pre_grid is None or post_grid is None:
            continue

        # Look for 3x3 highlighted grids
        # A simple heuristic: check for 3x3 blocks where the border is a highlight color
        # and the interior is different.
        # We'll consider a 5x5 area to find a 3x3 grid with a border.
        # Top-left of the potential 3x3 inner square (not the border itself)
        for r in range(max(0, clicked_row - 2), min(pre_grid.shape[0] - 2, clicked_row + 1)):
            for c in range(max(0, clicked_col - 2), min(pre_grid.shape[1] - 2, clicked_col + 1)):
                # Check if this (r, c) is the top-left of a 3x3 inner grid
                # surrounded by a highlight color (e.g., light-gray or dark-gray)

                # Check for 5x5 bounding box
                if not (0 <= r-1 <= pre_grid.shape[0]-1 and 0 <= c-1 <= pre_grid.shape[1]-1 and
                        0 <= r+3 <= pre_grid.shape[0]-1 and 0 <= c+3 <= pre_grid.shape[1]-1):
                    continue # Not enough space for a 5x5 bounding box

                is_highlighted_grid = False
                highlight_color = -1

                # Check the outer 5x5 perimeter (excluding the 3x3 inner)
                perimeter_coords_5x5 = []
                for row_offset in range(-1, 4):
                    for col_offset in range(-1, 4):
                        if (row_offset in [-1, 3] or col_offset in [-1, 3]):
                            current_val = pre_grid[r + row_offset, c + col_offset]
                            if current_val in [2, 4]: # Light-gray or dark-gray
                                perimeter_coords_5x5.append( (r + row_offset, c + col_offset) )
                            else:
                                # Not a consistent highlight color
                                # If any outer cell is not a highlight color, it's not our grid
                                # Or if it's the center, then it's wrong too, so check later.
                                pass 

                # If there are enough perimeter cells and they are all the same highlight color
                if len(perimeter_coords_5x5) > 0 and \
                   all(pre_grid[y,x] == pre_grid[perimeter_coords_5x5[0]] for y,x in perimeter_coords_5x5):

                    highlight_color = pre_grid[perimeter_coords_5x5[0]]

                    # Now check the 3x3 inner grid (r to r+2, c to c+2)
                    # All cells in the inner 3x3 should NOT be the highlight color, and should not be white (0)
                    all_inner_valid = True
                    for inner_r_offset in range(3):
                        for inner_c_offset in range(3):
                            if pre_grid[r + inner_r_offset, c + inner_c_offset] in [highlight_color, 0]:
                                all_inner_valid = False
                                break
                        if not all_inner_valid:
                            break
                    if not all_inner_valid:
                        continue

                    is_highlighted_grid = True


                if not is_highlighted_grid:
                    continue

                grid_id = (r, c) # Top-left of the 3x3 inner grid

                # Identify initial blue perimeter squares within this 3x3 grid
                # Perimeter of the 3x3 inner grid itself
                current_blue_perimeters = set()
                for pr_offset in range(3):
                    for pc_offset in range(3):
                        if pr_offset == 1 and pc_offset == 1: # Center
                            continue
                        if pre_grid[r + pr_offset, c + pc_offset] == 9: # Blue
                            current_blue_perimeters.add((r + pr_offset, c + pc_offset))

                if grid_id not in per_grid_state:
                    per_grid_state[grid_id] = {
                        'blue_perimeters': current_blue_perimeters,
                        'clicked_center': False,
                        'changed_red_after_center_click': False,
                        'clicked_central_element_coords': None
                    }
                else:
                    # Update blue perimeters for existing grid_id, in case they change over time
                    per_grid_state[grid_id]['blue_perimeters'] = current_blue_perimeters


                # Case 1: Clicking the central element of a highlighted 3x3 grid
                if clicked_row == r + 1 and clicked_col == c + 1:
                    if pre_grid[clicked_row, clicked_col] != post_grid[clicked_row, clicked_col]: # Ensure it actually did something
                        # This means we clicked the center
                        per_grid_state[grid_id]['clicked_center'] = True
                        per_grid_state[grid_id]['clicked_central_element_coords'] = (clicked_row, clicked_col)
                        # The question asks if blue *perimeter* squares become changeable *after this click*,
                        # so we reset the expectation for this grid or mark it as 'ready'.

                # Case 2: Subsequent clicks attempting to turn blue perimeter squares red
                # Only consider actions *after* the center was clicked.
                if per_grid_state[grid_id]['clicked_center']:
                    # Ensure this click is not the central element click itself
                    if not (clicked_row == per_grid_state[grid_id]['clicked_central_element_coords'][0] and
                            clicked_col == per_grid_state[grid_id]['clicked_central_element_coords'][1]):

                        if (clicked_row, clicked_col) in per_grid_state[grid_id]['blue_perimeters']:
                            # Check if this blue perimeter square changed to red
                            if pre_grid[clicked_row, clicked_col] == 9 and post_grid[clicked_row, clicked_col] == 8:
                                per_grid_state[grid_id]['changed_red_after_center_click'] = True

    # Evaluate the results
    any_center_clicked = False
    any_blue_to_red_after_center = False

    for grid_id, state in per_grid_state.items():
        if state['clicked_center']:
            any_center_clicked = True
            if state['changed_red_after_center_click']:
                any_blue_to_red_after_center = True
                break # We found at least one instance, so we can say YES

    if any_center_clicked and any_blue_to_red_after_center:
        return "YES"
    elif any_center_clicked and not any_blue_to_red_after_center:
        # We clicked a center, but no blue perimeter turned red *after* that.
        # This implies it doesn't create the condition.
        return "NO"
    else:
        # No center of a highlighted grid was clicked in the trajectory.
        return "MAYBE"