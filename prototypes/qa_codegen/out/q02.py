def answer(steps: list[StepRecord]) -> str:
    if not steps:
        return "MAYBE"

    # The initial state of the grid is the 'pre' grid of the first step
    initial_grid = steps[0].pre

    for i in range(1, len(steps)):
        current_step = steps[i]
        prev_step = steps[i-1]

        # Check for a 'reset' action or an 'undo' action
        # An undo/reset action would result in the grid reverting to an earlier state,
        # specifically the initial state of the current level or a previous state in the current level.

        # Heuristic 1: If the current grid 'post' becomes identical to the initial grid,
        # it suggests a reset to the start of the level.
        if np.array_equal(current_step.post, initial_grid):
            # However, this could also happen if the game was solved in one move
            # and that move brings it back to the initial state (unlikely for most puzzles, but possible).
            # We need to distinguish it from a normal forward move.
            # If there was a change in the previous step, and then it reset, it's a stronger indication.
            if prev_step.any_change():
                return "YES"

        # Heuristic 2: If the current grid 'post' becomes identical to any *previous* 'pre' grid
        # from the current level's sequence, it suggests an undo.
        # We need to be careful not to confuse a "do nothing" action with an undo.
        # An undo would typically involve the grid changing *from* its current state *to* a prior state.
        for j in range(i):
            if np.array_equal(current_step.post, steps[j].pre):
                # If the grid changed from its 'pre' state to a prior 'pre' state, it's an undo.
                if current_step.any_change(): # This means current_step.post != current_step.pre
                    return "YES"

    return "MAYBE"