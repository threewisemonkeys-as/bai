import numpy as np

def answer(steps: list[StepRecord]) -> str:
    """
    Checks if level progression (e.g., reaching Levels completed: 1/6) unlocks
    the simple actions (ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, ACTION7).
    """
    encountered_actions = set()
    initial_level_completed = -1
    relevant_actions = {"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION7"}

    # Track if any relevant action was used before completing the first level
    relevant_action_used_before_level_progression = False

    for i, step in enumerate(steps):
        if step.levels_completed is not None:
            if initial_level_completed == -1:
                initial_level_completed = step.levels_completed

            # If the levels_completed count has increased from its initial value
            # and it's not the very first step, then a level progression has occurred.
            if step.levels_completed > initial_level_completed:
                # After a level progression, check if any of the target actions were used.
                # If these actions are *only* available after progression, then we should
                # see them *after* step.levels_completed > initial_level_completed.
                # If they were available *before*, then relevant_action_used_before_level_progression
                # would be True and we'd return NO.

                # If no relevant actions were seen before the first progression,
                # and we see one after, it might suggest unlocking.
                # If we saw one before, it definitely means they weren't unlocked by progression.

                # This logic is about whether *progression itself* unlocks them.
                # If a relevant action was encountered *before* initial_level_completed increased,
                # then progression doesn't unlock them.
                if relevant_action_used_before_level_progression:
                    return "NO"  # They were already available.

                # If we reach here, it means no relevant actions were used *before* any level progression.
                # Now, we need to check if they are used *after* a level progression.
                # We need to look at the actions in the *current* step and subsequent steps.

                # Let's collect all action types observed after the first level completion.
                # If the action_type in the current step (or any subsequent step after a progression) is one of the relevant actions,
                # and it was never seen before, then it would support "YES".

                # We re-evaluate if we should return YES/NO based on what happens *after* the initial progression.
                # The question is whether progression *unlocks* them.
                # If we encounter such an action *at any point after* `levels_completed` goes up, and haven't seen it earlier,
                # it leans towards YES. If we saw it before, it's NO.

                # Update `encountered_actions` with actions seen *after* a level progression.
                # Then check if any `relevant_actions` appear.
                for j in range(i, len(steps)):
                    if steps[j].action_type in relevant_actions and steps[j].action_type not in encountered_actions:
                        # This action was used only after a level progression.
                        return "YES"

                # If we progressed a level, but no *new* simple actions were used after it,
                # then we can't confirm unlocking based on this data. We need to see them used.
                # If the game ends here without new actions, it's "MAYBE".

            # Track all action types observed *before* or *at the point of* any level progression
            # to determine if relevant actions were available without progression.
            if step.action_type in relevant_actions:
                encountered_actions.add(step.action_type)
                if initial_level_completed != -1 and step.levels_completed == initial_level_completed:
                    # An action was used before the levels_completed counter increased.
                    relevant_action_used_before_level_progression = True
                elif initial_level_completed == -1: # First step with levels_completed info, action used at this point
                    relevant_action_used_before_level_progression = True

        # If levels_completed is None, we still add action types to know what was seen
        # as long as we haven't confirmed initial_level_completed.
        # If initial_level_completed is -1, it means we haven't seen "Levels completed" yet.
        # Any relevant action seen before "Levels completed" implies they are not unlocked by it.
        if initial_level_completed == -1 and step.action_type in relevant_actions:
             relevant_action_used_before_level_progression = True

        # If we are in the middle of a game before any progression, `encountered_actions` should still grow.
        # This part of the logic is trying to catch if a relevant action was executed *before* any level progression.
        if (initial_level_completed == -1 or step.levels_completed == initial_level_completed) and \
           step.action_type in relevant_actions:
            relevant_action_used_before_level_progression = True

    if relevant_action_used_before_level_progression:
        return "NO"

    return "MAYBE"