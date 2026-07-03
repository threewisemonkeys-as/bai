import numpy as np

def answer(steps: list[StepRecord]) -> str:
    # Iterate through all steps in the trajectory
    for step_record in steps:
        # Check if the step involved a click
        if step_record.x is not None and step_record.y is not None:
            # Get the grid before and after the action
            pre_grid = step_record.pre
            post_grid = step_record.post

            if pre_grid is None or post_grid is None:
                continue

            # Iterate through all changed cells in this step
            for r, c, old_val, new_val in step_record.changed_cells():
                # Check if the original cell was a blue perimeter square
                # A square (r, c) is on the perimeter if it's on any of the four edges
                # Note: The problem description implies a 64x64 grid by pre/post.
                # However, the grids in ARC-AGI can be smaller. We should check grid dimensions.
                if pre_grid.shape[0] == 0 or pre_grid.shape[1] == 0:
                    continue

                is_on_perimeter = (
                    r == 0 or r == pre_grid.shape[0] - 1 or
                    c == 0 or c == pre_grid.shape[1] - 1
                )

                if old_val == 9 and is_on_perimeter:
                    # If a blue perimeter square changed to a non-red color
                    if new_val != 8:
                        return "YES"

    # If we iterated through all steps and didn't find any instance of a blue perimeter
    # square changing to a non-red color, we return "NO".
    # This implicitly assumes that if it *can* happen, it *would* have happened in the logs.
    # Given the phrasing "Can the blue perimeter squares ... change into ...",
    # observing a single such instance confirms it. Not observing it over a non-empty
    # log suggests it cannot within the observed play.

    # The question is about *can* it, not *did* it. If the log is extensive enough,
    # and no such event occurred, then "NO" is a reasonable conclusion *based on the log*.
    # If the log is very sparse, "MAYBE" might be more accurate, but the prompt
    # implies that "YES"/"NO" should be returned based on observed evidence for or against.

    return "NO"