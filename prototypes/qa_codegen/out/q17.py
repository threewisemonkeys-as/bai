def answer(steps: list[StepRecord]) -> str:
    for step in steps:
        if step.x is None or step.y is None or not step.any_change():
            continue

        # Check if any 3x3 grid changes
        for r in range(0, 64, 3):
            for c in range(0, 64, 3):
                # Define the 3x3 grid boundaries
                grid_rows = range(r, min(r + 3, 64))
                grid_cols = range(c, min(c + 3, 64))

                # Identify the center of the 3x3 grid
                center_row = r + 1
                center_col = c + 1

                # Check if the center is within bounds
                if not (0 <= center_row < 64 and 0 <= center_col < 64):
                    continue

                # Check if the central dot was red before the change
                if step.pre[center_row, center_col] == 8:  # 8 is red
                    # Check if the central dot specifically changed
                    if step.post[center_row, center_col] != 8:
                        return "YES"

                    # Alternatively, check for any change within the 3x3 around a red center
                    for changed_r, changed_c, _, _ in step.changed_cells():
                        if changed_r in grid_rows and changed_c in grid_cols:
                            return "YES"

    return "NO"