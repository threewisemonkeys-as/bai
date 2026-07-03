import numpy as np

def answer(steps: list[StepRecord]) -> str:
    # This question is about whether specific patterns *lead to* completion or unlock new actions.
    # We can observe completion, but we cannot observe new actions if only one is available.
    # Therefore, we will focus on completion.

    # We need to detect if a specific 3x3 pattern is followed by a WIN state.

    # Helper function to check for an all-red perimeter in a 3x3 grid
    def check_red_perimeter(grid_section):
        if grid_section.shape != (3, 3):
            return False
        # Check top row, bottom row, left column, right column
        for c in range(3):
            if grid_section[0, c] != 8: # Top row
                return False
            if grid_section[2, c] != 8: # Bottom row
                return False
        for r in range(1, 2): # Middle rows (left and right edges only)
            if grid_section[r, 0] != 8: # Left column
                return False
            if grid_section[r, 2] != 8: # Right column
                return False
        return True

    # Identify 3x3 grids within the 64x64 grid
    # Assuming 3x3 grids are non-overlapping and start at multiples of 3 (or 4, 5, etc.)
    # Let's try to infer the 3x3 grid boundaries from observed changes if possible,
    # but a general check means iterating through potential 3x3 sections.
    # A common ARC pattern for multiple grids is 10x10 subgrids, often with 1-pixel borders.
    # Let's assume for now that 3x3 grids are common and might be at specific offsets.
    # Given the general nature, we'll scan the entire grid for 3x3 sections.

    evidence_for = False
    evidence_against = False

    for i in range(len(steps)):
        step = steps[i]

        if step.state == "WIN":
            # If the current step is a WIN, we check if the *previous* state contained the pattern.
            # If the game just started or if it's the first step, there's no "previous" state to check.
            if i == 0:
                continue

            prev_grid = steps[i-1].post # The state *before* the WIN action

            # Iterate through all possible 3x3 subgrids in the previous grid
            for r_start in range(prev_grid.shape[0] - 2):
                for c_start in range(prev_grid.shape[1] - 2):
                    subgrid = prev_grid[r_start : r_start + 3, c_start : c_start + 3]
                    if check_red_perimeter(subgrid):
                        evidence_for = True
                        break
                if evidence_for:
                    break


        elif step.state == "NOT_FINISHED":
            # If a pattern is present but the level is not finished, it's evidence against.
            # We check the *current* grid (after the action) for the pattern.
            current_grid = step.post

            for r_start in range(current_grid.shape[0] - 2):
                for c_start in range(current_grid.shape[1] - 2):
                    subgrid = current_grid[r_start : r_start + 3, c_start : c_start + 3]
                    if check_red_perimeter(subgrid):
                        # If a red perimeter is found, but the game is NOT_FINISHED
                        # AND there's a subsequent step in which the game is still NOT_FINISHED
                        # (to avoid cases where a pattern is formed and then completed in the very next step)
                        if i + 1 < len(steps) and steps[i+1].state == "NOT_FINISHED":
                            evidence_against = True
                            break
                if evidence_against:
                    break

    if evidence_for and not evidence_against:
        return "YES"
    elif evidence_against and not evidence_for:
        return "NO"
    elif evidence_for and evidence_against:
        return "MAYBE" # Mixed evidence
    else:
        return "MAYBE" # No decisive evidence either way