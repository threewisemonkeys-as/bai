import numpy as np

def answer(steps: list) -> str:
    # A highlighted grid is typically indicated by a specific border color or a change in color within a region.
    # Given the constraint of 64x64 grid and 3x3 highlighted region size,
    # we need to find patterns that indicate highlighting.

    # Let's assume highlighting means a specific set of border pixels or internal pixels
    # change to a specific color (e.g., light-gray or dark-gray, often used for selection/highlighting).

    # We need to find if an action on a 3x3 region (the "clicked" region) causes
    # a *different* 3x3 region to become highlighted.
    # The "currently highlighted" implies it was highlighted *before* the action.

    # Let's define a function to check if a 3x3 region is highlighted.
    # For this puzzle type, highlighting often involves a specific border or fill,
    # and a common pattern is for cells around the 3x3 block to become a specific color,
    # or the 3x3 block itself to change to a specific pattern.

    # For ARC-AGI 3, common highlighting patterns include the border cells (usually color 2 or 4) defining the region.
    # Let's try to detect if a 3x3 grid (including its immediate 1-pixel border) is "highlighted"
    # by looking for a specific border color (e.g., light-gray 2 or dark-gray 4) surrounding a 3x3 block.

    def is_region_highlighted(grid, top_left_row, top_left_col, highlight_color=2):
        # A 3x3 region is at (top_left_row, top_left_col) to (top_left_row+2, top_left_col+2)
        # Its 1-pixel border would be from (top_left_row-1, top_left_col-1)
        # to (top_left_row+3, top_left_col+3).

        # Check bounds
        if not (0 < top_left_row < 63 and 0 < top_left_col < 63 and
                top_left_row + 2 < 63 and top_left_col + 2 < 63):
            return False # Region would be out of bounds for a border

        # Check the 1-pixel border around the 3x3 block
        # Define the border coordinates
        border_coords = []
        for r in range(top_left_row - 1, top_left_row + 4):
            for c in range(top_left_col - 1, top_left_col + 4):
                if (r < top_left_row or r > top_left_row + 2 or
                    c < top_left_col or c > top_left_col + 2):
                    border_coords.append((r, c))

        if not border_coords:
            return False

        # Check if all border cells (or a majority, or specific ones) are of the highlight_color
        # For simplicity, let's check if *any* of the border cells are the highlight color.
        # A more robust check might require all border cells, or top/bottom/left/right border cells.
        # Given the logs, typically the entire border becomes 'highlight_color'.
        # Let's assume a simple heuristic: if a sufficient number of border cells are 'highlight_color'.

        # A common pattern is cells (r,c) surrounding the target cell (r_target, c_target) change color.
        # For a 3x3 region, this means the cells (r-1, c-1) to (r+3, c+3) where the internal 3x3 is (r, c) to (r+2, c+2).

        # Let's refine the border check: the 16 cells forming the 1-pixel thick border around the 3x3 block.
        # 4 cells for each row/column at the extreme (top/bottom/left/right) excluding corners.
        # Top border: (r-1, c-1) to (r-1, c+3)
        # Bottom border: (r+3, c-1) to (r+3, c+3)
        # Left border: (r, c-1) to (r+2, c-1)
        # Right border: (r, c+3) to (r+2, c+3)

        # A more direct common way to find highlighted regions in ARC is by specific border colors.
        # Let's check for color 2 (light-gray) or 4 (dark-gray) for the border.

        # This implementation defines highlighting as:
        # The 16 cells forming its 1-pixel thick boundary are all of `highlight_color`.

        # For a 3x3 region at (r,c) (top-left of 3x3), the border indices are:

        # Row r-1, cols c-1 to c+3
        for col in range(top_left_col - 1, top_left_col + 4):
            if grid[top_left_row - 1, col] != highlight_color:
                return False
        # Row r+3, cols c-1 to c+3
        for col in range(top_left_col - 1, top_left_col + 4):
            if grid[top_left_row + 3, col] != highlight_color:
                return False
        # Cols c-1, rows r to r+2
        for row in range(top_left_row, top_left_row + 3):
            if grid[row, top_left_col - 1] != highlight_color:
                return False
        # Cols c+3, rows r to r+2
        for row in range(top_left_row, top_left_col + 3):
            if grid[row, top_left_col + 3] != highlight_color:
                return False

        return True

    all_verdicts = []

    for step in steps:
        pre_grid = step.pre
        post_grid = step.post
        clicked_x, clicked_y = step.x, step.y

        if pre_grid is None or post_grid is None or clicked_x is None or clicked_y is None:
            continue

        # The clicked (x,y) is part of a 3x3 region.
        # We need to determine the top-left (r,c) of the 3x3 region the clicked cell belongs to.
        # If the click (x,y) is on (0,0) of a 3x3, then (r,c) is (0,0).
        # If click (x,y) is on (1,1) of a 3x3, then (r,c) is (0,0).
        # Assuming the clicked cell is *any* cell within the 3x3 region,
        # then the top-left of the 3x3 region is (y_clicked - y_offset, x_clicked - x_offset)
        # where 0 <= y_offset, x_offset <= 2.
        # To determine the "clicked 3x3 region", we need to infer its boundary.

        # Let's assume the action is clicked *on the border* of the highlighted 3x3 region.
        # Or, the clicked_x, clicked_y refers to a cell *within* the 3x3 region.
        # Given the problem context "highlighted 3x3 grid", it implies the entire 3x3 grid
        # plus its border is considered as a unit.

        # A reasonable heuristic for ARC 3 is that the click (x,y) is *within* the 3x3 block,
        # and the problem assumes a fixed grid of 3x3 regions.
        # Let's assume regions are aligned on a 4x4 or 5x5 grid (3 core + 2 border).
        # A common cell size is 4x4 or 5x5 for selectable regions.
        # If a 3x3 grid is highlighted, usually it means cells from (r,c) to (r+2,c+2).

        # Let's scan for highlighted 3x3 regions.
        # Iterate through all possible top-left corners of a 3x3 block (rows 0-61, cols 0-61).

        highlighted_regions_pre = []
        highlighted_regions_post = []

        # Assuming specific highlight colors based on common ARC patterns.
        # Let's try color 2 (light-gray) for highlight border.
        highlight_colors = [2, 4] # light-gray, dark-gray

        for highlight_color in highlight_colors:
            for r in range(1, 60):  # 64-1 for border, -3 for 3x3 block, -1 for border
                for c in range(1, 60):
                    # Check for 3x3 region starting at (r, c)
                    if is_region_highlighted(pre_grid, r, c, highlight_color):
                        highlighted_regions_pre.append(((r, c), highlight_color))
                    if is_region_highlighted(post_grid, r, c, highlight_color):
                        highlighted_regions_post.append(((r, c), highlight_color))

            # If we found highlight regions, let's process this step
            if highlighted_regions_pre or highlighted_regions_post:
                 break # Found highlights for this color, proceed.

        if not highlighted_regions_pre:
            continue # No highlighted regions before the action, cannot answer the question

        # Identify the "clicked 3x3 region".
        # This is the 3x3 region that contains the clicked cell (step.y, step.x).
        clicked_region_top_left = None
        # We need to find the (r,c) of the 3x3 region where the clicked point (step.y, step.x) lies.
        # This is a bit ambiguous. Assuming blocks are on a regular grid (e.g., 4x4 or 5x5 centers).
        # For simplicity, let's assume the clicked cell is *within* one of the currently highlighted 3x3 regions.
        for region_tl, h_color in highlighted_regions_pre:
            r_start, c_start = region_tl
            if (r_start <= clicked_y <= r_start + 2) and \
               (c_start <= clicked_x <= c_start + 2):
                clicked_region_top_left = region_tl
                break

        if clicked_region_top_left is None:
            # The clicked cell was not part of a highlighted region *before* the action.
            # This doesn't fit the question's premise "performing an action on the currently highlighted 3x3 grid".
            continue


        # Now we have the clicked_region_top_left and highlighted_regions_pre.
        # Check if a *different*, unhighlighted 3x3 grid became highlighted.

        # A "different" grid means not the clicked_region_top_left.
        # An "unhighlighted" grid means it was NOT in highlighted_regions_pre.
        # "became highlighted" means it IS in highlighted_regions_post.

        found_yes = False
        found_no = False

        for post_region_tl, h_color_post in highlighted_regions_post:
            if post_region_tl == clicked_region_top_left:
                # The clicked region remained highlighted or changed highlight.
                # This doesn't help with "a *different* ... grid to become highlighted".
                continue

            # This is a different region.
            # Was it unhighlighted before?
            was_unhighlighted = True
            for pre_region_tl, h_color_pre in highlighted_regions_pre:
                if post_region_tl == pre_region_tl and h_color_post == h_color_pre:
                    was_unhighlighted = False
                    break

            if was_unhighlighted:
                # Found a different, previously unhighlighted region that became highlighted.
                found_yes = True
                break

        # Also need to check if the clicked region *stopped* being highlighted.
        # That doesn't directly answer the question, but could indicate a "move" of highlight.

        if found_yes:
            all_verdicts.append("YES")

        # Is there evidence for NO?
        # If there's only one highlighted region before, and it's the clicked one,
        # and after the action, this region is still the *only* highlighted region.

        # Consider scenarios for "NO":
        # 1. No new highlight appeared, or only the clicked region changed its highlight.
        # 2. Only the original clicked region remained highlighted.
        # 3. No regions became highlighted at all.

        # If we reached this point in the loop for a particular `step`,
        # `clicked_region_top_left` is known and was highlighted `pre`.

        if not found_yes: # If we haven't already decided YES for this step
            # Check if *any* other regions became highlighted, without being highlighted before?
            # Or, did nothing else change?

            # If for all `post_region_tl` that are NOT `clicked_region_top_left`,
            # they were *already* in `highlighted_regions_pre`.

            # This is tricky because the problem asks about "a different grid".

            # Let's re-evaluate:
            # Condition for YES:
            # Pre: `clicked_region_top_left` is highlighted.
            # Post: There exists a `new_highlighted_region` such that:
            #   1. `new_highlighted_region` is different from `clicked_region_top_left`.
            #   2. `new_highlighted_region` was NOT highlighted in `pre_grid`.
            #   3. `new_highlighted_region` IS highlighted in `post_grid`.

            # If we find such a step, the answer is "YES".
            # If after checking all steps, we don't find such a step, but we find steps
            # where highlighting changes such that this scenario *could not* have happened,
            # then it's "NO". Otherwise "MAYBE".

            # Example "NO" scenario: The only change was in the clicked region, or it became unhighlighted.
            # If all regions in `highlighted_regions_post` were already in `highlighted_regions_pre`,
            # and no new regions appeared, then it's "NO" *for this step*.

            current_step_is_no = True
            for post_reg_tl, h_color_post in highlighted_regions_post:
                if post_reg_tl == clicked_region_top_left:
                    # The clicked region could have changed color or just stayed highlighted.
                    # This by itself doesn't make it a NO, as other things might have changed.
                    continue

                # This is a different region that is highlighted post-action.
                # Check if it was highlighted pre-action.
                was_pre_highlighted = False
                for pre_reg_tl, h_color_pre in highlighted_regions_pre:
                    if post_reg_tl == pre_reg_tl: # and h_color_post == h_color_pre: (color might change)
                        # We only care that the *region itself* (its location) was already highlighted.
                        was_pre_highlighted = True
                        break

                if not was_pre_highlighted:
                    # A different region that was *not* pre-highlighted *became* highlighted.
                    # This would make the answer "YES" for this step, contradicting `current_step_is_no`.
                    current_step_is_no = False
                    break

            # If clicked region was the only one highlighted, and it's the only one highlighted post.
            if current_step_is_no:
                all_verdicts.append("NO")


    if "YES" in all_verdicts:
        return "YES"
    elif "NO" in all_verdicts:
        # Only if we have *only* 'NO' verdicts that implies nothing moved.
        # If there are steps with no relevant highlight changes, they should not turn to 'NO' too quickly.
        # If all relevant steps (where a clicked cell is part of a highlighted region AND there are highlight changes)
        # lead to "NO", then the answer is "NO".
        # If there are no relevant steps, it's "MAYBE".

        # Let's count relevant actions.
        relevant_action_found = False
        for step in steps:
            pre_grid = step.pre
            clicked_x, clicked_y = step.x, step.y
            if pre_grid is None or clicked_x is None or clicked_y is None:
                continue

            for highlight_color in highlight_colors:
                found_clicked_highlighted = False
                for r in range(1, 60):
                    for c in range(1, 60):
                        if is_region_highlighted(pre_grid, r, c, highlight_color):
                            if (r <= clicked_y <= r + 2) and (c <= clicked_x <= c + 2):
                                found_clicked_highlighted = True
                                break
                    if found_clicked_highlighted:
                        break
                if found_clicked_highlighted:
                    relevant_action_found = True
                    break
            if relevant_action_found:
                break

        if not relevant_action_found:
            return "MAYBE" # No actions on highlighted regions found at all

        if len(set(all_verdicts)) == 1 and "NO" in all_verdicts:
            return "NO"
        else: # Mixed verdicts or only MAYBE were found earlier based on absence of YES or NO specific conditions
            return "MAYBE"

    return "MAYBE"