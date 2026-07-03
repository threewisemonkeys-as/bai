import numpy as np

def answer(steps: list[StepRecord]) -> str:
    # A hint or tutorial system would likely involve an action that
    # either changes the grid to display instructional content, or
    # causes a UI element to appear, which might be reflected as
    # a consistently different type of grid change compared to
    # typical puzzle-solving actions.

    # Let's look for specific patterns:
    # 1. An action changing the grid to a fixed, non-game-state grid (e.g., text, arrows).
    #       - This implies a significant and possibly complex change.
    # 2. An action consistently resulting in a "null" grid (e.g., all white or all black)
    #    or a very simple, repetitive pattern, which could be a prompt.
    # 3. Actions that are frequently performed without directly solving a puzzle step,
    #    and result in transient or "informational" changes.

    all_action_types = set(s.action_type for s in steps)

    # If there's only one action type, it's less likely to be a dedicated "hint" action
    # unless that single action can also trigger hints.
    if len(all_action_types) == 1:
        # Check if this single action ever produces a non-game-related output.
        # This is tough to distinguish from game actions.
        # For ARC-AGI, actions usually modify the grid directly as part of a task.
        # A hint would stand out by displaying text or clear instructions.

        # Let's look for actions that result in a grid that is predominantly
        # a single color, or has a very high number of distinct colors appear
        # all at once, possibly forming text.

        for step in steps:
            if step.post is not None and step.any_change():

                # Heuristic 1: If the post-state looks like a text screen/menu.
                # This could be indicated by a large number of distinct colors appearing
                # that weren't there before, or a very structured, potentially
                # non-game-like pattern. Or, conversely, a very simple, almost
                # blank screen with specific UI elements.

                # Let's look for an action that results in a grid
                # that is mostly empty (e.g., white or black) but has some
                # small, distinct patterns potentially representing UI elements or text.

                # Check for a "mostly blank" grid with some specific elements
                unique_post_colors = np.unique(step.post)
                if len(unique_post_colors) > 2: # At least background, foreground, plus something else
                    # Check for a high proportion of a background color
                    background_color_count_0 = np.sum(step.post == 0) # white
                    background_color_count_5 = np.sum(step.post == 5) # black
                    total_cells = step.post.shape[0] * step.post.shape[1]

                    # If a significant portion is a background color, but there are other colors present,
                    # it might indicate a menu or text screen. For example, >80% background.
                    if (background_color_count_0 / total_cells > 0.8 or
                        background_color_count_5 / total_cells > 0.8):

                        # And if the other colors form a specific pattern
                        # This is very hard to detect generically without knowing expected patterns.
                        # However, for a tutorial/hint, the *change* would be very structured.

                        # An extremely simple heuristic: does the action replace a complex grid
                        # with a much simpler one (e.g., text-like)?
                        cells_changed = step.changed_cells()
                        if cells_changed:
                            # If many cells change to a background color, but some to other specific colors
                            # This is very speculative.

                            # More concrete: If an action repeatedly leads to the same "tutorial/hint" screen.
                            # We would need to identify two such identical post-states.

                            # Let's assume for now that if a single action type causes a clear,
                            # non-puzzle related display more than once, it could be a hint.

                            # If a single action type leads to an output that is very different from
                            # typical puzzle solutions (e.g., not just moving/coloring blocks).
                            pass # No strong evidence from this within a single action.


    # If there are multiple action types, one might be specifically for hints/tutorials.
    # How to distinguish a "hint" action from a "game modifier" action?
    # A hint action would likely:
    # 1. Not directly solve the puzzle (i.e., `state` remains "NOT_FINISHED" after it, or does not immediately lead to WIN).
    # 2. Change the grid to display information (text, arrows, highlight areas).
    # 3. Potentially be reversible by another action, or by re-clicking the same action.

    potential_hint_actions = set()

    for i, step in enumerate(steps):
        if step.any_change():
            # If an action changes the grid significantly but is not immediately followed by a WIN state,
            # and the change itself doesn't look like a direct puzzle solution.

            # Heuristic: Look for an action that results in a grid whose unique colors count
            # or distribution significantly differs from typical task grids, especially if it
            # introduces patterns common in UI/text.
            if step.post is not None:
                # Count distinct colors in the post-state
                unique_colors_post = np.unique(step.post)

                # A tutorial might involve a sudden increase in distinct colors (e.g., black text on white bg with highlights)
                # or a large area of a single color with small, distinct elements.

                # Check for "text-like" or "menu-like" patterns:
                # - Many colors that are not typically seen together in tasks (e.g., many shades of gray for text).
                # - A very stark background/foreground contrast over a large area.

                # Let's consider a threshold for distinct colors that suggests more than just game elements.
                # (This is highly speculative without example hint grids)
                if len(unique_colors_post) > 5: # Assuming game grids usually have fewer distinct colors for actual play
                    # If this action does not lead to a WIN state immediately after
                    if step.state != "WIN" and (i + 1 == len(steps) or steps[i+1].state != "WIN"):
                        # This could indicate an informational screen.
                        potential_hint_actions.add(step.action_type)

                # Also consider the opposite: an action that clears the screen to display something simple.
                # e.g., if the post grid is mostly one color (e.g., white or black) but has a few other
                # colors forming a specific, non-game pattern.
                background_color_counts = {
                    0: np.sum(step.post == 0),
                    5: np.sum(step.post == 5),
                    # Add other common background-like colors if identified
                }
                total_cells = step.post.shape[0] * step.post.shape[1]

                for bg_color, count in background_color_counts.items():
                    if count / total_cells > 0.85 and len(unique_colors_post) > 1: # Mostly bg, but not entirely.
                        # This could be a screen with some text/icons.
                        if step.state != "WIN" and (i + 1 == len(steps) or steps[i+1].state != "WIN"):
                             potential_hint_actions.add(step.action_type)

    if potential_hint_actions:
        return "YES"

    # If no strong evidence, but there are multiple action types, it's still possible.
    # If there's only one action type, and it behaves only as a puzzle-solving action, then NO.
    if len(all_action_types) > 1:
        # If there are multiple actions, it's more plausible one is for hints.
        # But we need more concrete evidence than just "multiple actions exist".
        # If we couldn't find ANY specific pattern, we should return MAYBE.
        return "MAYBE"

    # If only one action type and we didn't find any hint-like behavior
    return "NO"