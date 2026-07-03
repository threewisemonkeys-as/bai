import numpy as np

def answer(steps: list[StepRecord]) -> str:
    # A 3x3 grid is a conceptual element, not necessarily directly represented in the StepRecord
    # We are looking for changes over a "level reset"
    # A level reset is indicated by `levels_completed` increasing.

    # Find the step where the player transitions to a new level
    # We need at least two steps to observe a transition across levels.
    if len(steps) < 2:
        return "MAYBE"

    previous_level_completed = steps[0].levels_completed

    for i in range(1, len(steps)):
        current_level_completed = steps[i].levels_completed

        # Check if a new level has been completed
        if current_level_completed > previous_level_completed:
            # This 'i' step is the first step of a new level, or just after completing one.
            # The grid state at steps[i].pre represents the state *before* any action in the new level.
            # The grid state at steps[i-1].post represents the state at the *end* of the previous level.
            # We are interested in whether the grid state changed spontaneously due to the level progression.

            # Compare steps[i-1].post (end of previous level) with steps[i].pre (start of current level)
            # The state should be reset _before_ steps[i].pre is captured, which means we compare steps[i-1].post and steps[i].pre.

            pre_new_level_grid = steps[i].pre
            post_old_level_grid = steps[i-1].post

            # If either grid is None, we can't make a conclusive decision
            if pre_new_level_grid is None or post_old_level_grid is None:
                continue

            # We need to find 3x3 grids. Assuming the 3x3 grids are
            # visually distinct sub-regions, we can check for differences
            # in 3x3 blocks.
            # A simple way to check for a reset is to see if *any* cell changed
            # between the end of the previous level and the start of the next.
            # If the entire grid is reset, then many cells will change.
            # If only some conceptual 3x3 grids are reset, then those cells will change.

            # A straightforward way to check for a reset is to compare the grids directly.
            # If they are different, it implies some reset or change occurred implicitly.

            # Check if the grids are identical. If they are not, some state was reset.
            if not np.array_equal(pre_new_level_grid, post_old_level_grid):
                # If any difference is found, it means the state was reset.
                return "YES"

        previous_level_completed = current_level_completed

    # If we went through all transitions where levels_completed increased
    # and found no grid changes, then it suggests no reset.
    # However, if no level transitions occurred, we can't say.
    # We should return "NO" only if we *observed* a level transition
    # and confirmed no reset.
    # If we reached here, it means we scanned all level transitions and found no reset,
    # OR there were no level transitions.

    # To distinguish, let's keep a flag if we saw any level completion increase.
    found_level_transition = False
    for i in range(1, len(steps)):
        if steps[i].levels_completed > steps[i-1].levels_completed:
            found_level_transition = True
            break

    if found_level_transition:
        # We observed at least one level transition and found no grid difference.
        return "NO"
    else:
        # No level transition observed to make a determination.
        return "MAYBE"