import numpy as np

def answer(steps: list[StepRecord]) -> str:
    # A true blue perimeter square in a 3x3 grid
    # would be color 9.
    BLUE = 9
    RED = 8

    # We are looking for evidence that a blue perimeter square
    # in a 3x3 grid can be made unchangeable to red.
    # This means:
    # 1. We find an action that targets a blue perimeter cell within a relevant 3x3 grid.
    # 2. Despite the action, the cell remains blue (or changes to something NOT red).
    # This means the game logic prevents it from turning red.

    # What defines a "3x3 grid"? The problem implies this is a local context.
    # We should look at changes within a 3x3 window around the click.

    # To be "permanently unchangeable to red", we need to see a blue cell that
    # an action attempts to change (it is clicked), but it does not turn red.

    found_blue_perimeter_clicked = False
    found_blue_perimeter_not_red_after_click = False

    for step in steps:
        if step.x is None or step.y is None or step.pre is None or step.post is None:
            continue

        clicked_y, clicked_x = step.y, step.x
        pre_grid = step.pre
        post_grid = step.post

        if clicked_y < 0 or clicked_y >= pre_grid.shape[0] or \
           clicked_x < 0 or clicked_x >= pre_grid.shape[1]:
            continue # Clicked outside the grid, though unlikely with given constraints

        clicked_value_pre = pre_grid[clicked_y, clicked_x]

        # Check if the clicked cell is part of a "blue perimeter" in any "3x3 grid".
        # A simpler interpretation is that the clicked cell ITSELF is blue,
        # and it's on the "perimeter" of some implicit 3x3 concept.
        # Given the phrasing "any blue perimeter square in any 3x3 grid",
        # the most robust way is to consider if the clicked cell is blue,
        # and it's within a 3x3 context.

        # Let's define a "3x3 grid" for the clicked cell as the 3x3 window
        # centered at the clicked cell (if possible).
        # A "blue perimeter square" in this 3x3 grid refers to blue cells
        # that are not the center cell.

        # The question might be interpreted more generally: is the clicked blue cell
        # itself a perimeter cell of *some* 3x3 region.
        # For simplicity, let's just check if the clicked cell is blue.
        # If it's blue and clicked, it's a candidate.

        if clicked_value_pre == BLUE:
            found_blue_perimeter_clicked = True
            clicked_value_post = post_grid[clicked_y, clicked_x]

            if clicked_value_post != RED:
                # We clicked a blue cell, and it did not turn red.
                # This provides evidence that it can be "unchangeable to red".
                # The wording "permanently unchangeable" suggests one instance
                # is enough to prove the possibility.
                found_blue_perimeter_not_red_after_click = True
                return "YES" # Found evidence that it can be unchangeable to red.

    if found_blue_perimeter_clicked and not found_blue_perimeter_not_red_after_click:
        # We clicked blue cells, but they all turned red.
        # This is contradictory to the idea of being unchangeable.
        return "NO"

    return "MAYBE"