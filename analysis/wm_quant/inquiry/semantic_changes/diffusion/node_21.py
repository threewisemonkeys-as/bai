import json
from collections import defaultdict
import math

# Global cache for the background colour, determined from all observations seen so far.
_bg_colour = None

def _parse_grid(obs: str):
    """Extract the 2D colour grid as list[list[str]].
    Anchors on the generic '[[' / ']]' to work regardless of header text.
    Returns None on failure.
    """
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


def _base64_encode(index, num_digits):
    """Encode an integer (0 <= index < 64**num_digits) into a
    fixed-length base64 string."""
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+/"
    result = []
    for _ in range(num_digits):
        result.append(chars[index % 64])
        index //= 64
    return ''.join(reversed(result))


def _determine_bg(history):
    """Determine the background colour as the most frequent colour across all
    observations.  This ensures a consistent background across the entire
    trajectory, avoiding flips when an object colour temporarily becomes the majority.
    """
    global _bg_colour
    if _bg_colour is not None:
        return _bg_colour

    counts = defaultdict(int)
    for obs in history:
        grid = _parse_grid(obs)
        if grid is None:
            continue
        for row in grid:
            for colour in row:
                counts[colour] += 1

    if not counts:
        # No valid grid; fall back to using the last obs (will be handled later)
        _bg_colour = None
        return None

    # Most frequent; ties broken alphabetically for determinism
    _bg_colour = max(counts, key=lambda c: (counts[c], c))
    return _bg_colour


def perceive(observation_history: list[str]) -> str:
    """Produce a compact summary of the current grid, omitting background cells.
    The background colour is determined *once* from all observations seen so far
    and then used for every call, guaranteeing a stable representation.
    Grid dimensions are included so the state can be fully reconstructed.
    Falls back to base64 encoding if the row-grouped text exceeds 2000 chars.
    """
    global _bg_colour

    obs = observation_history[-1] if observation_history else ""
    grid = _parse_grid(obs)
    if grid is None:
        # Determine background from past history if possible, else safe default
        _determine_bg(observation_history)
        return "parse_error"

    rows = len(grid)
    if rows == 0:
        return "empty_grid"
    cols = len(grid[0])
    if cols == 0:
        return "empty_grid"

    # Determine background once from the whole history (cached in _bg_colour)
    bg = _determine_bg(observation_history)
    if bg is None:
        # No history with valid grids – fall back to majority of this frame
        counts = defaultdict(int)
        for row in grid:
            for colour in row:
                counts[colour] += 1
        bg = max(counts, key=lambda c: (counts[c], c))

    # Collect non-background cells
    cells_by_colour = defaultdict(lambda: defaultdict(list))  # colour -> row -> list of cols
    for r, row in enumerate(grid):
        for c, colour in enumerate(row):
            if colour != bg:
                cells_by_colour[colour][r].append(c)

    # Build header
    header = f"bg:{bg}; dim:{rows},{cols}"
    parts = [header]

    # Build row-grouped text for non-background colours
    for colour in sorted(cells_by_colour):
        rows_dict = cells_by_colour[colour]
        row_parts = []
        for r in sorted(rows_dict):
            cols_list = sorted(rows_dict[r])
            row_parts.append(f"{r}:{','.join(map(str, cols_list))}")
        colour_part = f"{colour}:{';'.join(row_parts)}"
        parts.append(colour_part)

    result = '\n'.join(parts)

    # If within size, return immediately
    if len(result) <= 2000:
        return result

    # Fallback: base64 encoding of linearised indices
    max_index = rows * cols - 1
    num_digits = 1
    while 64 ** num_digits <= max_index:
        num_digits += 1

    fallback_parts = [header]
    for colour in sorted(cells_by_colour):
        rows_dict = cells_by_colour[colour]
        indices = []
        for r, cols_list in rows_dict.items():
            for c in cols_list:
                indices.append(r * cols + c)
        indices.sort()
        encoded = ''.join(_base64_encode(idx, num_digits) for idx in indices)
        fallback_parts.append(f"{colour}:{encoded}")

    result = '\n'.join(fallback_parts)

    # Safety: if still too long (unlikely), output counts only
    if len(result) > 2000:
        counts_only = [header]
        for colour in sorted(cells_by_colour):
            counts_only.append(f"{colour}:{sum(len(v) for v in cells_by_colour[colour].values())}")
        result = '\n'.join(counts_only)

    return result if result else "empty_grid"