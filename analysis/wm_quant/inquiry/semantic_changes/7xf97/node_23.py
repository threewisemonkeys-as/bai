"""
Perception module for Autumn grid environment.
Produces a detailed, lossless summary of all non-background cells,
augmented with derived boolean flags that capture spatial relationships
and grid‑identity information. These flags make consecutive states
distinct even when the raw cell sets are identical, enabling the inverse
dynamics and forward prediction to uniquely identify actions and
future states.

Output format:
  rows=<R>,cols=<C>,bg=<B>;f:<flags>;<colour1>:<coords1>;...|...

Flags (comma‑separated key=value pairs):
  gx  – gray exists
  ax  – gold exists
  gle – gold block touches left edge
  gre – gold block touches right edge
  gcle – gray block touches left edge
  gcre – gray block touches right edge
  gare – gold right‑adjacent to gray (gap=0)
  gale – gold left‑adjacent to gray (gap=0)
  uc  – grid content unchanged from previous observation (if available)

Colour groups are ordered by priority (gold, gray, blue, purple, green, …)
and within each colour cells are sorted by (row, col) with consecutive
cells in the same row collapsed into a range.
The whole summary is guaranteed non‑empty and under 2000 characters.
"""

import json
from collections import Counter

def _parse_grid(obs: str):
    """Extract the 2D colour grid as list[list[str]]."""
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

def _format_cells(colour, cells):
    """Convert a sorted list of (r,c) tuples for one colour into a compact string.
    Consecutive cells in the same row are merged into a range.
    """
    if not cells:
        return ""
    out_parts = []
    i = 0
    while i < len(cells):
        r, c = cells[i]
        start_c = c
        end_c = c
        j = i + 1
        while j < len(cells) and cells[j][0] == r and cells[j][1] == end_c + 1:
            end_c = cells[j][1]
            j += 1
        if start_c == end_c:
            out_parts.append(f"{r},{start_c}")
        else:
            out_parts.append(f"{r},{start_c}-{end_c}")
        i = j
    return colour + ":" + ";".join(out_parts)

def perceive(observation_history: list[str]) -> str:
    """Return a comprehensive summary with derived flags for action recovery."""
    # get current observation
    obs = observation_history[-1] if observation_history else ""
    grid = _parse_grid(obs)
    if grid is None:
        return "parse_error:unknown"

    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    if rows == 0 or cols == 0:
        return f"rows={rows},cols={cols},bg=unknown;empty"

    # Determine background colour (most common colour in the grid)
    flat = [cell for row in grid for cell in row]
    if not flat:
        return f"rows={rows},cols={cols},bg=unknown;empty"
    background = Counter(flat).most_common(1)[0][0]

    # Collect non‑background cells grouped by colour
    cells_by_colour = {}
    for r, row in enumerate(grid):
        for c, colour in enumerate(row):
            if colour != background:
                cells_by_colour.setdefault(colour, []).append((r, c))

    # ----- derived flags -----
    # gold / gray sets for spatial analysis
    gold_cells = cells_by_colour.get("gold", [])
    gray_cells = cells_by_colour.get("gray", [])
    gold_exists = len(gold_cells) > 0
    gray_exists = len(gray_cells) > 0

    gold_left = None
    gold_right = None
    if gold_exists:
        gold_left = min(c for _, c in gold_cells)
        gold_right = max(c for _, c in gold_cells)
    gray_left = None
    gray_right = None
    if gray_exists:
        gray_left = min(c for _, c in gray_cells)
        gray_right = max(c for _, c in gray_cells)

    gold_left_edge = 1 if (gold_exists and gold_left == 0) else 0
    gold_right_edge = 1 if (gold_exists and gold_right == cols - 1) else 0
    gray_left_edge = 1 if (gray_exists and gray_left == 0) else 0
    gray_right_edge = 1 if (gray_exists and gray_right == cols - 1) else 0

    # adjacency: gold immediately left of gray (gap=0)
    gold_adj_right_to_gray = 0
    if gold_exists and gray_exists and gold_right + 1 == gray_left:
        gold_adj_right_to_gray = 1
    gold_adj_left_to_gray = 0
    if gold_exists and gray_exists and gold_left - 1 == gray_right:
        gold_adj_left_to_gray = 1

    # grid unchanged flag (if we have a previous observation)
    grid_unchanged = 0
    if len(observation_history) >= 2:
        prev_obs = observation_history[-2]
        prev_grid = _parse_grid(prev_obs)
        if prev_grid is not None and prev_grid == grid:
            grid_unchanged = 1

    # Build the flag string (short keys, comma separated)
    # We include all flags even when 0 to keep format predictable.
    flag_pairs = [
        f"gx={1 if gray_exists else 0}",
        f"ax={1 if gold_exists else 0}",
        f"gle={gold_left_edge}",
        f"gre={gold_right_edge}",
        f"gcle={gray_left_edge}",
        f"gcre={gray_right_edge}",
        f"gare={gold_adj_right_to_gray}",
        f"gale={gold_adj_left_to_gray}",
        f"uc={grid_unchanged}"
    ]
    flag_str = "f:" + ",".join(flag_pairs)

    # ----- build colour groups -----
    # Priority order (lower number = higher priority)
    priority = {
        "gold": 0,
        "gray": 1,
        "blue": 2,
        "darkblue": 2,
        "lightblue": 2,
        "skyblue": 2,
        "mediumpurple": 3,
        "purple": 3,
        "green": 4,
        "darkgreen": 4,
        "limegreen": 4,
        "red": 5,
        "orange": 5,
        "brown": 5,
        "tan": 5,
        "pink": 5,
        "white": 6,
        "yellow": 6,
    }

    colour_groups = []
    for colour, cells in cells_by_colour.items():
        cells.sort()
        prio = priority.get(colour, 10)
        colour_groups.append((prio, colour, cells))
    colour_groups.sort(key=lambda x: (x[0], x[1]))

    # Format parts: flags first, then colour groups
    header = f"rows={rows},cols={cols},bg={background};"
    parts = [flag_str]
    for _, colour, cells in colour_groups:
        parts.append(_format_cells(colour, cells))

    summary = header + "|".join(parts)

    # Ensure length < 2000 characters (with a small safety margin)
    if len(summary) > 1990:
        # Drop colour groups from the end (lowest priority) until under limit
        # but keep at least the flags and one colour group.
        while len(summary) > 1990 and len(parts) > 2:
            parts.pop()
            summary = header + "|".join(parts)
        if len(summary) > 1990 and len(parts) == 2:
            # Only flags remain – truncate flags themselves if necessary
            # But flags are small (around 100 chars), so unlikely.
            if len(summary) > 1990:
                summary = summary[:1990] + "…"
        elif len(summary) > 1990 and len(parts) > 2:
            # Still too long – truncate the last colour group
            last_part = parts[-1]
            colon_idx = last_part.find(':')
            if colon_idx != -1 and len(last_part) > colon_idx + 1:
                excess = len(summary) - 1990
                # remove cell entries from the end
                while len(last_part) > colon_idx + 1 and excess > 0:
                    last_semi = last_part.rfind(';')
                    if last_semi == -1 or last_semi <= colon_idx:
                        break
                    removed = len(last_part) - last_semi
                    last_part = last_part[:last_semi]
                    excess -= removed
                parts[-1] = last_part
                summary = header + "|".join(parts)
                if len(summary) > 1990:
                    summary = summary[:1990] + "…"
            else:
                summary = summary[:1990] + "…"

    return summary