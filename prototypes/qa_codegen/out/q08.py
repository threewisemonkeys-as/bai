import numpy as np

def answer(steps: list[StepRecord]) -> str:
    # A global timer/turn limit would typically manifest as a game state change
    # or progression occurring without a explicit user action (e.g., a click).
    # We are looking for a transition where the grid changes, but no click
    # was performed by the user.

    for step in steps:
        # Check if there's any change in the grid
        if step.any_change():
            # If there's a change, check if an action (click) was performed
            if step.x is None and step.y is None:
                # A change occurred without a click action. This strongly suggests
                # an internal game mechanism, like a timer or turn limit,
                # is causing progression or state alteration.
                return "YES"

    # If no such un-triggered change was observed, we cannot definitively say "YES".
    # However, we cannot say "NO" either, as the timer might not have run out
    # or the limit might not have been reached in the observed trajectory.
    return "MAYBE"