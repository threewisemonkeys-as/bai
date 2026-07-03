import numpy as np

def answer(steps: list[StepRecord]) -> str:
    # This question is about level progression or unlocking new actions related to a specific pattern.
    # The game described (ARC-AGI 3) almost exclusively uses "ACTION6" so unlocking new actions is unlikely
    # for the current observed game, unless the action_type changes, which is not stated to be
    # possible from the current rules.
    # So we'll focus on "level progression".

    # Define the target colors
    RED = 8
    BLUE = 9

    # The 3x3 highlighted grid is usually a smaller part of the 64x64 grid.
    # We need to detect a 3x3 grid and its perimeter.
    # We assume "highlighted" means filled with a specific color background,
    # or outlined. Since we're looking at pattern, it's more likely
    # a consistently identified 3x3 area.

    level_progression_observed = False
    pattern_observed_before_progression = False

    for i in range(len(steps) - 1):
        current_step = steps[i]
        next_step = steps[i+1]

        # Check for level progression.
        if next_step.levels_completed > current_step.levels_completed:
            level_progression_observed = True

            # If progression happened, let's look for the 3x3 pattern in the 'pre' grid of the current step
            # or 'post' grid of the current step (which is 'pre' of the next step).
            # The pattern should be on the grid *before* the action that completes the level.

            pre_grid = current_step.post # The state *after* the action that led to progression, and *before* the next level starts.
                                         # Or current_step.pre if we want to check the state *before* the last action.
                                         # Let's check current_step.pre as it's the state *before* the final action.
            if pre_grid is None:
                continue

            # Need to find a 3x3 grid. This is the trickiest part as "highlighted" isn't strictly defined.
            # A common way to denote an interactive 3x3 grid in ARC is for it to be surrounded by a border,
            # or having a consistent background that separates it from other areas.
            # Let's assume for now a simpler detection: A consistent square of distinct color pixels.

            # Iterate through possible top-left corners for a 3x3 block
            for r in range(pre_grid.shape[0] - 2):
                for c in range(pre_grid.shape[1] - 2):
                    subgrid = pre_grid[r:r+3, c:c+3]

                    # Check if this subgrid forms a perimeter of red and blue,
                    # and if it's not all red.
                    perimeter_cells = [
                        subgrid[0, 0], subgrid[0, 1], subgrid[0, 2],
                        subgrid[1, 0],              subgrid[1, 2],
                        subgrid[2, 0], subgrid[2, 1], subgrid[2, 2]
                    ]

                    # All perimeter cells must be either RED or BLUE
                    is_red_blue_perimeter = all(cell in [RED, BLUE] for cell in perimeter_cells)

                    if is_red_blue_perimeter:
                        # Check if it's not all red
                        is_all_red = all(cell == RED for cell in perimeter_cells)
                        if not is_all_red:
                            # It's a non-all-red pattern of red and blue perimeter squares
                            pattern_observed_before_progression = True
                            break # Found the pattern for this level progression
                if pattern_observed_before_progression:
                    break
        if pattern_observed_before_progression and level_progression_observed:
            return "YES"

    # If we got here, either no progression happened or no such pattern was found before progression.
    # The question also mentions "unlock new actions". Since action_type is consistently "ACTION6",
    # we don't have evidence for new actions being unlocked.

    if level_progression_observed:
        # Progression happened, but the specific pattern was not observed.
        # This means the pattern is NOT required for progression.
        return "NO"
    else:
        # No level progression observed at all with any pattern.
        # So we cannot confirm or deny the hypothesis based on the provided logs.
        return "MAYBE"