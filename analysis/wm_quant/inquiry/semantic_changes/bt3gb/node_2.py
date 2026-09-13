import json

def _parse_grid(obs: str):
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

def _background_colour(grid):
    """Return the most frequent colour in the grid (the background colour)."""
    colour_counts = {}
    for row in grid:
        for cell in row:
            colour_counts[cell] = colour_counts.get(cell, 0) + 1
    # Choose the colour with the highest count; if tie, any is fine.
    return max(colour_counts, key=colour_counts.get)

def perceive(observation_history: list[str]) -> str:
    obs = observation_history[-1] if observation_history else ""
    grid = _parse_grid(obs)
    if grid is None:
        # Best-effort summary: indicate parse failure but never return empty.
        return "PARSE_ERROR"

    bg = _background_colour(grid)
    non_bg = []
    for r, row in enumerate(grid):
        for c, colour in enumerate(row):
            if colour != bg:
                non_bg.append((r, c, colour))

    if not non_bg:
        return "all_background"

    # Build a compact representation: rows sorted, then cells within each row.
    # Format: row,col:colour  (space separated for brevity)
    parts = []
    for r, c, colour in sorted(non_bg, key=lambda x: (x[0], x[1])):
        parts.append(f"{r},{c}:{colour}")
    summary = " ".join(parts)

    # Ensure total length < 2000 chars – truncate if necessary (very unlikely)
    if len(summary) > 1999:
        summary = summary[:1996] + "..."

    return summary