def answer(steps: list[StepRecord]) -> str:
    highlighted_color = 2  # Light gray
    highlighted_color_new = 4  # Dark gray
    background_color = 5  # Black

    for step in steps:
        if step.action_type == "ACTION6" and step.x is not None and step.y is not None:
            pre_grid = step.pre
            post_grid = step.post

            # Find all 3x3 highlighted grids
            highlighted_grids_pre = []
            for r in range(pre_grid.shape[0] - 2):
                for c in range(pre_grid.shape[1] - 2):
                    sub_grid = pre_grid[r : r + 3, c : c + 3]
                    # A grid is considered highlighted if its border (excluding corners) is 'highlighted_color'
                    # and the central element is not 'highlighted_color' (usually background or another color).
                    # For a 3x3 grid, checking the corners might also be useful for distinguishing.
                    is_border_highlighted = all(
                        sub_grid[i, j] == highlighted_color
                        for i, j in [
                            (0, 0), (0, 1), (0, 2),
                            (1, 0),         (1, 2),
                            (2, 0), (2, 1), (2, 2),
                        ]
                    )

                    if is_border_highlighted and sub_grid[1, 1] != highlighted_color:
                        highlighted_grids_pre.append(((r, c), sub_grid))

            # Check if the clicked cell is the central element of any currently highlighted 3x3 grid
            is_clicked_center_of_highlighted = False
            for (r_start, c_start), _ in highlighted_grids_pre:
                if step.x == c_start + 1 and step.y == r_start + 1:
                    is_clicked_center_of_highlighted = True
                    break

            if is_clicked_center_of_highlighted:
                # Check for effect: removing highlight or new highlight
                effect_found = False

                # 1. Check if the original highlight is removed
                for (r_start, c_start), _ in highlighted_grids_pre:
                    # Check if the border cells changed from highlighted_color
                    # Or if the entire 3x3 area became homogeneous
                    for i, j in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]:
                        if post_grid[r_start + i, c_start + j] != highlighted_color:
                            effect_found = True
                            break
                    if effect_found:
                        break

                # 2. Check for a new highlight appearing
                highlighted_grids_post = []
                for r in range(post_grid.shape[0] - 2):
                    for c in range(post_grid.shape[1] - 2):
                        sub_grid = post_grid[r : r + 3, c : c + 3]
                        is_border_highlighted = all(
                            sub_grid[i, j] == highlighted_color
                            for i, j in [
                                (0, 0), (0, 1), (0, 2),
                                (1, 0),         (1, 2),
                                (2, 0), (2, 1), (2, 2),
                            ]
                        )
                        if is_border_highlighted and sub_grid[1, 1] != highlighted_color:
                            highlighted_grids_post.append(((r, c), sub_grid))

                if len(highlighted_grids_post) > len(highlighted_grids_pre):
                    effect_found = True

                # Special case: highlight color might change
                # Check if elements of *any* grid that was highlighted with 'highlighted_color' (2)
                # became 'highlighted_color_new' (4)
                for r_start, c_start in [(r, c) for (r,c), _ in highlighted_grids_pre]:
                    for r_offset in range(3):
                        for c_offset in range(3):
                            if pre_grid[r_start + r_offset, c_start + c_offset] == highlighted_color and \
                               post_grid[r_start + r_offset, c_start + c_offset] == highlighted_color_new:
                                effect_found = True
                                break
                        if effect_found:
                            break
                    if effect_found:
                        break


                if effect_found:
                    return "YES"
                else:
                    return "NO"

    return "MAYBE"