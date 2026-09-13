"""
Contract: perceive(observation_history) -> str must never raise, never return empty,
and produce a text summary that changes whenever the grid changes so that the action
between two consecutive states can be recovered from the feature trajectory.

This version uses the observation history to compute a delta between successive states,
explicitly describing movements of colour blobs and changes in the (0,0) cell.
The output format:

  <dimensions>; bg:<bg>; (0,0)=<colour>;
  <blob_summaries>; cells: <list>
  delta: (0,0) unchanged|changed; <blob_deltas>

where blob_deltas describe centroid shifts and colour transformations.
"""

import json
from collections import Counter, defaultdict
import math

def _parse_grid(obs: str):
    """Extract the 2D colour grid as list[list[str]] (rows of colour-name strings)."""
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

def _find_background(grid):
    """Return the most common colour in the grid (the background)."""
    all_colours = []
    for row in grid:
        all_colours.extend(row)
    if not all_colours:
        return "gray"
    counter = Counter(all_colours)
    return counter.most_common(1)[0][0]

def _colour_sort_key(item):
    return item[0], item[1], item[2]

def _blob_summary(grid, background):
    """
    Return a dict: colour -> { 'cells': set((r,c)),
                              'box': (min_r, max_r, min_c, max_c),
                              'centroid': (avg_r, avg_c) }
    Only non‑background cells are considered.
    Cells of the same colour are grouped into a single 'blob' (if the grid
    has multiple disjoint blobs of the same colour, they are merged; this
    is sufficient for the current task).
    """
    blobs = defaultdict(set)
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    for r in range(rows):
        for c in range(cols):
            colour = grid[r][c]
            if colour != background:
                blobs[colour].add((r, c))
    result = {}
    for colour, cells in blobs.items():
        rs = [r for r,c in cells]
        cs = [c for r,c in cells]
        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)
        centroid_r = sum(rs) / len(rs)
        centroid_c = sum(cs) / len(cs)
        result[colour] = {
            'cells': cells,
            'box': (min_r, max_r, min_c, max_c),
            'centroid': (centroid_r, centroid_c)
        }
    return result

def _format_blob(blob_info):
    """Return a string like 'tan_blob: rows 0‑4, cols 0‑4'"""
    min_r, max_r, min_c, max_c = blob_info['box']
    return f"rows {min_r}-{max_r}, cols {min_c}-{max_c}"

def perceive(observation_history: list[str]) -> str:
    obs = observation_history[-1] if observation_history else ""
    grid = _parse_grid(obs)
    if grid is None:
        return f"parse_error_{hash(obs) & 0xFFFFFF}"

    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    background = _find_background(grid)

    # Current blob info
    cur_blobs = _blob_summary(grid, background)
    # (0,0) cell colour
    zero_zero = grid[0][0] if rows > 0 and cols > 0 else "unknown"

    # Build cell list
    objects = []
    for r in range(rows):
        for c in range(cols):
            colour = grid[r][c]
            if colour != background:
                objects.append((r, c, colour))
    objects.sort(key=_colour_sort_key)
    obj_strs = [f"({r},{c},{colour})" for r,c,colour in objects]

    # Start with dimension and background
    parts = [f"{rows}x{cols}", f"bg:{background}", f"(0,0)={zero_zero}"]

    # Blob summaries
    for colour in sorted(cur_blobs.keys()):
        blob = cur_blobs[colour]
        blob_str = f"{colour}_blob: {_format_blob(blob)}"
        parts.append(blob_str)

    # Cell list
    cells_str = "cells: " + " ".join(obj_strs)
    # Truncate if needed (should rarely be needed)
    while len("; ".join(parts) + "; " + cells_str) > 1950 and obj_strs:
        obj_strs.pop()
        cells_str = "cells: " + " ".join(obj_strs)
    parts.append(cells_str)

    # ----- Delta from previous state (if available) -----
    delta_part = ""
    if len(observation_history) >= 2:
        prev_obs = observation_history[-2]
        prev_grid = _parse_grid(prev_obs)
        if prev_grid is not None:
            prev_rows = len(prev_grid)
            prev_cols = len(prev_grid[0]) if prev_rows else 0
            prev_background = _find_background(prev_grid)
            prev_blobs = _blob_summary(prev_grid, prev_background)
            prev_zero = prev_grid[0][0] if prev_rows > 0 and prev_cols > 0 else "unknown"

            delta_items = []
            # (0,0) change
            if zero_zero == prev_zero:
                delta_items.append("(0,0) unchanged")
            else:
                delta_items.append(f"(0,0) changed: {prev_zero} -> {zero_zero}")

            # For each colour that exists in either state, describe changes
            all_colours = set(cur_blobs.keys()) | set(prev_blobs.keys())
            for colour in sorted(all_colours):
                cur = cur_blobs.get(colour)
                prev = prev_blobs.get(colour)
                if prev is None and cur is not None:
                    delta_items.append(f"{colour}_blob appeared")
                elif prev is not None and cur is None:
                    delta_items.append(f"{colour}_blob disappeared")
                elif prev is not None and cur is not None:
                    # Centroid shift
                    dr = cur['centroid'][0] - prev['centroid'][0]
                    dc = cur['centroid'][1] - prev['centroid'][1]
                    # Round to 1 decimal to avoid floating noise
                    dr_str = f"{dr:.1f}" if abs(dr - round(dr)) > 0.01 else str(int(round(dr)))
                    dc_str = f"{dc:.1f}" if abs(dc - round(dc)) > 0.01 else str(int(round(dc)))
                    # Check colour transformation
                    if colour in prev_blobs and colour in cur_blobs:
                        # same colour
                        delta_items.append(f"{colour}_blob centroid shift: (dr={dr_str}, dc={dc_str})")
                    # If colour changed (e.g., tan -> gold) it would appear as one disappearing and another appearing,
                    # so this case won't occur. But we could also check cross-colour via cell overlap.
            if delta_items:
                delta_part = "delta: " + "; ".join(delta_items)

    # Assemble final string
    if delta_part:
        summary = "; ".join(parts) + "; " + delta_part
    else:
        summary = "; ".join(parts)

    # Final length check (should be <2000)
    if len(summary) > 1990:
        # Last resort: truncate cell list more aggressively
        while len(summary) > 1990 and obj_strs:
            obj_strs.pop()
            cells_str = "cells: " + " ".join(obj_strs)
            # rebuild parts with new cells_str
            base = "; ".join([f"{rows}x{cols}", f"bg:{background}", f"(0,0)={zero_zero}"] +
                              [f"{colour}_blob: {_format_blob(cur_blobs[colour])}" for colour in sorted(cur_blobs.keys())])
            summary = base + "; " + cells_str
            if delta_part:
                summary += "; " + delta_part
        if len(summary) > 1990:
            # still too long, drop delta and blob summaries, keep only essential
            summary = f"{rows}x{cols}; bg:{background}; (0,0)={zero_zero}; cells: {cells_str}"
            if len(summary) > 1990:
                summary = f"{rows}x{cols}; bg:{background}; (0,0)={zero_zero}; [grid too large]"
    return summary