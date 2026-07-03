import numpy as np

def answer(steps: list[StepRecord]) -> str:
    # A perimeter square is colored blue if its immediate neighbors (excluding diagonals) are all red.
    # An action on a blue perimeter square turns it red.
    # The question asks if a red perimeter square can revert to blue,
    # or if the condition for blue perimeter squares to turn red can be reversed or expire for a grid.

    can_red_revert_to_blue = False
    condition_expiration_observed = False

    for step_idx, step in enumerate(steps):
        if step.action_type != "ACTION6" or not step.any_change():
            continue

        changed_cells = step.changed_cells()

        for r, c, old_val, new_val in changed_cells:
            # Check if a red perimeter square reverted to blue
            if old_val == 8 and new_val == 9:  # 8 is red, 9 is blue
                # Verify if it's a perimeter cell, based on its neighboring cells being red before the change
                # This requires checking the pre-state of the grid
                if step.pre is not None and r < step.pre.shape[0] and c < step.pre.shape[1]:
                    rows, cols = step.pre.shape
                    is_perimeter_candidate = False
                    if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                        is_perimeter_candidate = True

                    if is_perimeter_candidate:
                        # Check neighbors in the pre-state
                        all_neighbors_red_pre = True
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                if step.pre[nr, nc] != 8:
                                    all_neighbors_red_pre = False
                                    break
                            else:
                                # Neighbor is out of bounds, implicitly not red
                                all_neighbors_red_pre = False
                                break

                        if all_neighbors_red_pre:
                            can_red_revert_to_blue = True

            # Also consider if the *mechanism* for blue perimeter squares turning red expires
            # This implies a blue perimeter square is clicked, but it does NOT turn red.
            # Or, a blue perimeter square turns to something other than red.
            # This is harder to definitively prove without seeing a specific counter-example.
            # Instead, we look for cases where a blue square *should* have turned red but didn't,
            # or turned into something else.

            # We need to detect "blue perimeter squares" in the `pre` grid.
            if new_val != 8 and old_val == 9: # Blue square changed to non-red
                 # Check if it was a blue perimeter candidate
                if step.pre is not None and r < step.pre.shape[0] and c < step.pre.shape[1]:
                    rows, cols = step.pre.shape
                    is_perimeter_candidate = False
                    if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                        is_perimeter_candidate = True

                    if is_perimeter_candidate:
                        all_neighbors_red_pre = True
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                if step.pre[nr, nc] != 8:
                                    all_neighbors_red_pre = False
                                    break
                            else:
                                all_neighbors_red_pre = False
                                break

                        if all_neighbors_red_pre:
                            # It was a blue perimeter square (according to the rule)
                            # if it was clicked and didn't turn red, then the condition might have expired
                            if step.x == c and step.y == r:
                                # If it didn't turn red (or turned non-red), it suggests condition change
                                condition_expiration_observed = True


    if can_red_revert_to_blue or condition_expiration_observed:
        return "YES"
    else:
        return "NO"