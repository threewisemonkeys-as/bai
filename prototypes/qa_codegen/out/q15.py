import numpy as np

def answer(steps: list[StepRecord]) -> str:
    # This question asks about hidden UI elements beyond the 3x3 grids.
    # The game likely involves a single 3x3 grid for the primary puzzle.
    # We need to look for any changes in parts of the grid that are consistently
    # unused by the primary 3x3 puzzle elements, or any changes that suggest
    # a larger grid interaction or hidden elements becoming visible.

    # Let's consider common areas for UI elements: borders, corners,
    # or specific regions outside the typical puzzle area.
    # A typical ARC-AGI 3x3 puzzle might occupy a central portion of the 64x64 grid.
    # For a 3x3 pattern in a 64x64 grid, it's very small.
    # We can assume the puzzle itself is within a smaller, well-defined bounding box.

    # Let's try to determine the typical bounds of the active 3x3 puzzle area.
    # It's unlikely to be 64x64, as that's extremely large for 3x3.
    # We can infer the active area by finding where changes typically occur.

    min_row, max_row = 64, -1
    min_col, max_col = 64, -1

    for step in steps:
        if step.any_change():
            for r, c, _, _ in step.changed_cells():
                min_row = min(min_row, r)
                max_row = max(max_row, r)
                min_col = min(min_col, c)
                max_col = max(max_col, c)

    # If no changes occurred, we can't determine the puzzle area.
    if min_row == 64:
        return "MAYBE"

    # Define a bounding box for the "known" puzzle area.
    # We'll add some padding around the detected active area to account for borders or related controls.
    # A 3x3 grid means the active changed cells will be very localized.
    # Let's assume a generous padding of, say, 5 cells around the detected min/max.
    padding = 5
    puzzle_min_row = max(0, min_row - padding)
    puzzle_max_row = min(63, max_row + padding)
    puzzle_min_col = max(0, min_col - padding)
    puzzle_max_col = min(63, max_col + padding)

    # Now, iterate through all steps and check if any changes occur *outside* this extended puzzle area.
    for step in steps:
        if step.any_change():
            for r, c, old_val, new_val in step.changed_cells():
                # If a change occurred outside our defined puzzle area, it could indicate
                # a hidden element becoming visible or interactive.
                if not (puzzle_min_row <= r <= puzzle_max_row and
                        puzzle_min_col <= c <= puzzle_max_col):
                    return "YES"

    # If we've gone through all steps and found no changes outside the padded puzzle area,
    # then based on the provided actions, there's no evidence of other visible/interactive elements.
    return "NO"