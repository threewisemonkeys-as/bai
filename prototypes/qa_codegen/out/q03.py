import numpy as np

def answer(steps: list[StepRecord]) -> str:
    # A "highlight" is assumed to be a specific color, typically brighter like 2 (light gray) or 8 (red).
    # Let's consider 2 (light gray) as a potential highlight color as it's often used for selection.
    # The question specifies "unhighlighted grid" in the context of altering colors.
    # So we're looking for changes that affect cells other than the clicked cell,
    # or highlight movement/removal due to interaction,
    # especially when the grid might already appear "unhighlighted" or the click is on an unhighlighted square.

    highlight_colors = [2, 8]  # Common colors used for highlights: light-gray, red

    for step_index, step in enumerate(steps):
        if step.x is None or step.y is None:
            continue

        clicked_x, clicked_y = step.x, step.y
        pre_grid = step.pre
        post_grid = step.post

        if pre_grid is None or post_grid is None:
            continue

        if not (0 <= clicked_y < pre_grid.shape[0] and 0 <= clicked_x < pre_grid.shape[1]):
            continue

        changed_cells = step.changed_cells()

        # Check for highlight movement.
        # This implies a cell that was a highlight color becomes a non-highlight color,
        # and another cell that was not a highlight color becomes a highlight color.
        highlight_removed_from_pre = False
        highlight_added_to_post = False
        for r, c, old_val, new_val in changed_cells:
            if old_val in highlight_colors and new_val not in highlight_colors:
                highlight_removed_from_pre = True
            if old_val not in highlight_colors and new_val in highlight_colors:
                highlight_added_to_post = True

        if highlight_removed_from_pre or highlight_added_to_post:
            if highlight_removed_from_pre and highlight_added_to_post:
                # Highlight moved
                return "YES"
            elif highlight_removed_from_pre and not highlight_added_to_post:
                # Highlight removed
                return "YES"
            elif not highlight_removed_from_pre and highlight_added_to_post:
                # Highlight added (could be considered move)
                return "YES"

        # Check for altering colors of squares in an unhighlighted grid
        # or when clicking an unhighlighted cell, affecting other cells.
        clicked_cell_pre_val = pre_grid[clicked_y, clicked_x]

        # First, let's define what an "unhighlighted grid" means for this context.
        # It could mean a grid where no cells are of a highlight color.
        # Or, more practically, the current interaction is not specifically with a highlighted element.

        # Case 1: The clicked cell itself is not a highlight color (is "unhighlighted").
        # If clicking an unhighlighted cell causes ANY change, it's relevant.
        if clicked_cell_pre_val not in highlight_colors:
            # Check if any other cell changes color (excluding the clicked cell itself, as that's an direct interaction)
            # or if the clicked cell's color changed in a way that suggests an effect.
            for r, c, old_val, new_val in changed_cells:
                if (r, c) != (clicked_y, clicked_x) and old_val != new_val:
                    # An unhighlighted clicked cell caused other cells to change color.
                    return "YES"

                # If the clicked cell itself changes to a highlight color, that also indicates an effect
                if (r,c) == (clicked_y, clicked_x) and new_val in highlight_colors:
                    return "YES"

        # Case 2: The entire pre_grid contains no highlight colors ("unhighlighted grid" in a broader sense).
        # And any interaction causes a change.
        is_grid_unhighlighted = True
        for r in range(pre_grid.shape[0]):
            for c in range(pre_grid.shape[1]):
                if pre_grid[r, c] in highlight_colors:
                    is_grid_unhighlighted = False
                    break
            if not is_grid_unhighlighted:
                break

        if is_grid_unhighlighted:
            if step.any_change():
                # Any change in a fully unhighlighted grid means colors were altered.
                return "YES"

    return "MAYBE"