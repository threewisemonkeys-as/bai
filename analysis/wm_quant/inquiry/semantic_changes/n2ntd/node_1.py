import json
from collections import Counter
from typing import Optional

def _parse_grid(obs: str) -> Optional[list[list[str]]]:
    """Extract the 2D colour grid as list[list[str]] (rows of colour-name strings).

    Anchor on the generic '[[' / ']]' so it works regardless of the header text that precedes
    the grid. Do NOT gate on the observation *starting* with '[[': Autumn observations are
    prefixed by a "Task:/Step:/..." header, so the grid begins mid-string. Returns None on
    failure (caller degrades gracefully, never raises)."""
    if not obs:
        return None
    start = obs.find("[[")
    end = obs.rfind("]]")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        grid = json.loads(obs[start : end + 2])
    except Exception:
        return None
    if not grid or not isinstance(grid, list) or not isinstance(grid[0], list):
        return None
    return grid


def _get_background(grid: list[list[str]]) -> str:
    """Determine the background colour by frequency (most common cell colour)."""
    all_colours = [cell for row in grid for cell in row]
    if not all_colours:
        return "white"  # fallback
    count = Counter(all_colours)
    # Return the most common colour (ties don't matter for grid backgrounds)
    return count.most_common(1)[0][0]


def _find_agent(grid: list[list[str]], background: str) -> Optional[tuple[int, int]]:
    """Identify the agent cell.

    Heuristic: the agent is a single cell whose colour is 'red' or 'mediumpurple'.
    If neither is present, fall back to any singleton cell that is not 'gold' and not
    part of a larger block (size==1).  Returns (row, col) or None.
    """
    # First collect all non-background cells
    non_bg = {}
    for r, row in enumerate(grid):
        for c, colour in enumerate(row):
            if colour != background:
                non_bg[(r, c)] = colour

    # Prefer red, then mediumpurple
    for colour in ("red", "mediumpurple"):
        pos = [pos for pos, col in non_bg.items() if col == colour]
        if len(pos) == 1:
            return pos[0]

    # Fallback: any singleton that is not gold
    # Count colours (excluding background) to find singletons
    colour_counts = Counter(non_bg.values())
    for (r, c), col in non_bg.items():
        if colour_counts[col] == 1 and col != "gold":
            return (r, c)
    # If still nothing, just return the first non-background cell
    if non_bg:
        return next(iter(non_bg))
    return None


def perceive(observation_history: list[str]) -> str:
    """Produce a concise text summary of decision-relevant features.

    The output includes:
      - background colour
      - agent position (row, col)
      - list of all other non-background objects (type, position)

    This ensures the summary changes whenever the raw grid changes,
    making actions recoverable from consecutive feature strings.
    Never raises, never returns empty string.
    """
    obs = observation_history[-1] if observation_history else ""
    grid = _parse_grid(obs)
    if grid is None:
        # Degrade gracefully: return a minimal non-empty string
        return "grid_parse_error"

    try:
        background = _get_background(grid)
        agent_pos = _find_agent(grid, background)
        if agent_pos is None:
            agent_str = "agent:none"
        else:
            agent_str = f"agent:{agent_pos[0]},{agent_pos[1]}"

        objects = []
        for r, row in enumerate(grid):
            for c, colour in enumerate(row):
                if colour != background:
                    if agent_pos and (r, c) == agent_pos:
                        continue   # skip agent, already reported
                    objects.append(f"{colour}({r},{c})")

        # Sort objects for determinism
        objects.sort()
        obj_str = ",".join(objects) if objects else "none"

        return f"bg:{background} {agent_str} objs:[{obj_str}]"
    except Exception:
        # Safety net: never raise, never return empty
        return "summary_error"