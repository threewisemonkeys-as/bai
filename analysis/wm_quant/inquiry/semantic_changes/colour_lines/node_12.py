"""
Contract: perceive(observation_history) -> str must never raise, never return empty.
Output a concise (<2000 chars) summary of decision-relevant features.
Summarise all non‑background cells as (row,col,colour) and, when history is available,
include the changes from the previous state (moves, additions, removals) so that actions
can be reliably recovered from the text trajectory.
"""

import json
from collections import Counter


def _parse_grid(obs: str):
    """Extract the 2D colour grid as list[list[str]].
    Finds the JSON array inside the observation text (between the first '[['
    and the last ']]') and parses it.  Returns None on any failure.
    """
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
    # basic validation: list of lists of strings
    if not isinstance(grid, list) or not grid:
        return None
    for row in grid:
        if not isinstance(row, list):
            return None
    return grid


def _get_background(grid):
    """Return the most common colour in the grid (the background)."""
    counter = Counter()
    for row in grid:
        for cell in row:
            counter[cell] += 1
    bg, _ = counter.most_common(1)[0]
    return bg


def _foreground_set(grid, bg):
    """Return a set of (row, col, colour) for all non‑background cells."""
    cells = set()
    for r, row in enumerate(grid):
        for c, colour in enumerate(row):
            if colour != bg:
                cells.add((r, c, colour))
    return cells


def _match_cells(prev_set, curr_set):
    """Given sets of (r,c,colour) from previous and current state,
    return three lists:
      - moves: list of (r_prev, c_prev, r_curr, c_curr, colour)
      - added: list of (r,c,colour) appearing only in current
      - removed: list of (r,c,colour) appearing only in previous
    Matching assumes movement by at most one step (Manhattan distance ≤1).
    """
    # Copy sets so we can remove matched items
    prev_rem = set(prev_set)
    curr_rem = set(curr_set)

    moves = []
    added = []
    removed = []

    # Try to match current cells to previous cells of the same colour within distance 1
    # We'll go through current cells and find the closest previous of the same colour
    for cur in sorted(curr_set, key=lambda x: (x[2], x[0], x[1])):  # colour, then coords
        if cur not in curr_rem:
            continue
        r_cur, c_cur, col = cur
        best = None
        best_dist = 2  # only consider distance <=1
        for prev in prev_rem:
            if prev[2] != col:
                continue
            r_prev, c_prev, _ = prev
            dist = abs(r_cur - r_prev) + abs(c_cur - c_prev)
            if dist <= best_dist:
                best_dist = dist
                best = prev
        if best is not None:
            moves.append((best[0], best[1], r_cur, c_cur, col))
            curr_rem.remove(cur)
            prev_rem.remove(best)
        # else: will remain in curr_rem -> added later

    # Remaining: added and removed
    added = sorted(curr_rem, key=lambda x: (x[2], x[0], x[1]))
    removed = sorted(prev_rem, key=lambda x: (x[2], x[0], x[1]))
    return moves, added, removed


def _format_summary(bg, curr_cells, moves, added, removed):
    """Build a text summary of ≤2000 characters.
    Format:
      bg=<colour>; cells: (r,c,colour);...
      [if history available:]
      moves: (r1,c1)->(r2,c2) colour; ...
      added: (r,c,colour); ...
      removed: (r,c,colour); ...
    If no changes, just the first line.
    """
    parts = [f"bg={bg}; cells:"]
    # Sort current cells for stable output
    sorted_curr = sorted(curr_cells, key=lambda x: (x[0], x[1]))
    cell_strs = [f"({r},{c},{colour})" for (r, c, colour) in sorted_curr]
    parts.append("".join(cell_strs))   # no extra separator to save chars

    if moves or added or removed:
        if moves:
            move_strs = [f"({pr},{pc})->({cr},{cc}){colour}" for pr, pc, cr, cc, colour in moves]
            parts.append(" moves: " + ";".join(move_strs))
        if added:
            add_strs = [f"({r},{c},{colour})" for (r, c, colour) in added]
            parts.append(" added: " + ";".join(add_strs))
        if removed:
            rem_strs = [f"({r},{c},{colour})" for (r, c, colour) in removed]
            parts.append(" removed: " + ";".join(rem_strs))

    summary = " | ".join(parts)
    # Truncate if needed, but keep last few chars for "..." if truncated
    if len(summary) > 2000:
        summary = summary[:1997] + "..."
    return summary


def perceive(observation_history: list[str]) -> str:
    """Return a text summary of the current observation.

    The summary is guaranteed to be non‑empty and to change whenever the
    underlying grid changes, enabling inverse dynamics and forward prediction.
    """
    fallback = "parse_error: could not interpret observation"

    if not observation_history:
        return fallback

    # Parse current grid
    obs_cur = observation_history[-1]
    grid_cur = _parse_grid(obs_cur)
    if grid_cur is None:
        return fallback

    try:
        bg_cur = _get_background(grid_cur)
        curr_set = _foreground_set(grid_cur, bg_cur)
    except Exception:
        return fallback

    # If there is a previous state, compute changes
    if len(observation_history) >= 2:
        obs_prev = observation_history[-2]
        grid_prev = _parse_grid(obs_prev)
        if grid_prev is not None:
            try:
                bg_prev = _get_background(grid_prev)
                prev_set = _foreground_set(grid_prev, bg_prev)
                moves, added, removed = _match_cells(prev_set, curr_set)
            except Exception:
                # fallback to absolute summary
                moves, added, removed = [], [], []
        else:
            moves, added, removed = [], [], []
    else:
        moves, added, removed = [], [], []

    summary = _format_summary(bg_cur, curr_set, moves, added, removed)

    # Ensure never empty
    if not summary:
        return fallback
    return summary