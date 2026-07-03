import numpy as np

def answer(steps: list[StepRecord]) -> str:
    # Iterate through each step in the trajectory
    for step_record in steps:
        # Check if an action was performed and if it changed the grid or state
        if step_record.action_type == "ACTION6" and (step_record.any_change() or step_record.state != "NOT_FINISHED"):
            x, y = step_record.x, step_record.y

            # Ensure x and y are not None, as a click must have occurred
            if x is not None and y is not None:
                is_within_3x3_grid = False

                # Check if the clicked cell (x, y) is part of any 3x3 grid
                # A 3x3 grid would have its top-left corner at (i*3, j*3)
                # and span to (i*3 + 2, j*3 + 2)
                for i in range(0, 64, 3):  # Iterate through possible top-left x coordinates of 3x3 grids
                    for j in range(0, 64, 3):  # Iterate through possible top-left y coordinates of 3x3 grids
                        if i <= x <= i + 2 and j <= y <= j + 2:
                            is_within_3x3_grid = True
                            break
                    if is_within_3x3_grid:
                        break

                # If a change occurred by clicking outside any 3x3 grid, then the answer is YES
                if not is_within_3x3_grid:
                    return "YES"

    # If no such action was found after checking all steps, it's MAYBE
    return "MAYBE"