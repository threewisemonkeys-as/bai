"""
Contract: perceive(observation_history) -> str must never raise.
Outputs a compact summary of non‑background cells and the agent's position,
including when the agent is off‑grid (using history to recover last known location).
Preserves all non‑background cells in a compact format so that the summary is
unique for each raw grid and enables reliable action recovery from consecutive summaries.
Truncation is used only as a last resort and is accompanied by a hash of the full
set of non‑background cells to guarantee distinguishability.
"""

import json
from collections import Counter
import hashlib

# Colour name to single‑character code (compact representation)
COLOUR_SHORT = {
    "black": "k",
    "white": "w",
    "gray": "g",
    "grey": "g",
    "skyblue": "s",
    "lightblue": "l",
    "blue": "u",
    "darkblue": "d",
    "red": "r",
    "green": "n",
    "darkgreen": "D",
    "limegreen": "L",
    "gold": "G",
    "yellow": "y",
    "orange": "o",
    "brown": "b",
    "tan": "t",
    "pink": "p",
    "mediumpurple": "m",
    "purple": "P"
}

def _short(colour):
    """Return a 1‑character code for a colour name."""
    return COLOUR_SHORT.get(colour, "?")


def _parse_grid(obs: str):
    """Extract the 2D colour grid as list[list[str]] (rows of colour-name strings).

    Anchor on the generic '[[' / ']]' so it works regardless of header text.
    Returns None on failure (caller degrades gracefully)."""
    if not obs:
        return None
    start = obs.find("[[")
    end = obs.rfind("]]")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        grid = json.loads(obs[start: end + 2])
    except Exception:
        return None
    if not grid or not isinstance(grid, list) or not isinstance(grid[0], list):
        return None
    return grid


def _determine_background(grid):
    """Return the most frequent colour in the grid (the background)."""
    colours = [cell for row in grid for cell in row]
    if not colours:
        return "black"  # fallback
    counter = Counter(colours)
    return counter.most_common(1)[0][0]


def _find_agent(grid):
    """Return (row, col) of the orange cell, or None if not present."""
    for r, row in enumerate(grid):
        for c, col in enumerate(row):
            if col == "orange":
                return (r, c)
    return None


def _hash_cells(cells):
    """Return a short hex hash of a sorted list of (r,c,colour) tuples."""
    # Use a stable deterministic representation
    raw = ";".join(f"{r},{c},{col}" for r,c,col in sorted(cells))
    return hashlib.md5(raw.encode()).hexdigest()[:8]   # 8 hex chars


def perceive(observation_history: list[str]) -> str:
    # Always take the current observation
    obs = observation_history[-1] if observation_history else ""
    grid = _parse_grid(obs)
    if grid is None:
        return "parse error, could not extract grid"

    try:
        background = _determine_background(grid)
    except Exception:
        background = "black"

    agent_pos = None  # (r, c) if on grid
    # Look for orange in current grid
    agent_pos = _find_agent(grid)

    other_cells = []   # list of (r, c, colour)
    try:
        for r, row in enumerate(grid):
            for c, colour in enumerate(row):
                if colour == background:
                    continue
                if colour == "orange":
                    continue  # already handled
                other_cells.append((r, c, colour))
    except Exception:
        return "grid error"

    # If agent is off‑grid, recover its last known position from history
    if agent_pos is None:
        # Search backwards through history for the most recent orange
        for i in range(len(observation_history) - 2, -1, -1):
            prev_obs = observation_history[i]
            prev_grid = _parse_grid(prev_obs)
            if prev_grid is not None:
                prev_agent = _find_agent(prev_grid)
                if prev_agent is not None:
                    agent_pos = ("off", prev_agent[0], prev_agent[1])
                    break
        # If still not found, leave as None (unknown)
        # In practice, game always starts with agent visible, so this is safe.

    # Build the agent part
    agent_part = ""
    if agent_pos is not None:
        if isinstance(agent_pos, tuple) and agent_pos[0] != "off":
            r, c = agent_pos
            agent_part = f"a:{r},{c}"
        else:
            # agent off: stored as ("off", r, c)
            agent_part = f"a:off({agent_pos[1]},{agent_pos[2]})"
    else:
        agent_part = "a:?"

    # Build the other cells part – compact format without parentheses
    # Format: r,c,<short colour> ; r,c,<short colour> ...
    cell_strs = [f"{r},{c},{_short(col)}" for r, c, col in other_cells]
    cell_strs.sort()  # deterministic order

    # Combine all, staying under 2000 chars
    # Start with agent part
    parts = [agent_part]
    if cell_strs:
        # Reserve enough space for the hash if we need to truncate
        # (max total 2000, we target 1950 to leave room for a hash)
        MAX_MAIN = 1950
        combined_cells = ";".join(cell_strs)
        if len(combined_cells) > MAX_MAIN:
            # Truncate and add a hash of the full set
            # Keep as many full entries as possible
            truncated = []
            current_len = 0
            for entry in cell_strs:
                # +1 for the semicolon separator (first entry no separator, but we add after each)
                add_len = len(entry) + 1 if truncated else len(entry)
                if current_len + add_len > MAX_MAIN:
                    break
                if truncated:
                    truncated.append(";" + entry)
                else:
                    truncated.append(entry)
                current_len += add_len
            truncated_part = "".join(truncated)
            # Compute hash of full set
            full_hash = _hash_cells(other_cells)
            parts.append(f"cells:{truncated_part};##{full_hash}")
        else:
            parts.append(f"cells:{combined_cells}")
    else:
        parts.append("cells:empty")

    result = " | ".join(parts)
    # Guarantee non‑empty and under 2000 chars
    if not result:
        result = "grid error"
    if len(result) > 2000:
        # This should not happen with the reservation above, but be safe
        result = result[:1997] + "..."
    return result