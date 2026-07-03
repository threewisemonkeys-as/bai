import numpy as np

def answer(steps: list[StepRecord]) -> str:
    # A 3x3 grid is defined as a contiguous 3x3 block of non-background cells
    # We are looking for the appearance or disappearance of such blocks.

    def count_3x3_grids(grid: np.ndarray) -> int:
        if grid is None or grid.shape != (64, 64):
            return 0

        count = 0
        for r in range(grid.shape[0] - 2):
            for c in range(grid.shape[1] - 2):
                subgrid = grid[r:r+3, c:c+3]
                # A 3x3 grid exists if all cells in the subgrid are not background (color 5)
                if np.all(subgrid != 5):
                    count += 1
        return count

    found_change = False

    for step in steps:
        pre_grid = step.pre
        post_grid = step.post

        if pre_grid is None or post_grid is None:
            continue

        pre_3x3_count = count_3x3_grids(pre_grid)
        post_3x3_count = count_3x3_grids(post_grid)

        if pre_3x3_count != post_3x3_count:
            found_change = True
            break

    if found_change:
        return "YES"
    else:
        # If there were no visible 3x3 grids in the first place, we might not be able to tell,
        # but the question implies looking for change, so if no change was observed,
        # we assume it does not happen.
        return "NO"