import json
from collections import Counter

def _parse_grid(obs: str):
    """Extract the 2D colour grid as list[list[str]] (rows of colour-name strings).

    Anchor on the generic '[[' / ']]' so it works regardless of the header.
    Returns None on failure.
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
    if not grid or not isinstance(grid, list) or not isinstance(grid[0], list):
        return None
    return grid


def _get_runs(cols: list) -> list:
    """Convert a sorted list of column indices into a list of (start, end) runs."""
    if not cols:
        return []
    runs = []
    start = cols[0]
    end = cols[0]
    for c in cols[1:]:
        if c == end + 1:
            end = c
        else:
            runs.append((start, end))
            start = c
            end = c
    runs.append((start, end))
    return runs


def _format_runs(colour: str, row: int, runs: list) -> str:
    """Format a row's colour and runs, e.g. 'blue_row15:0-5,7-9'."""
    parts = [f"{colour}_row{row}:"]
    run_strs = []
    for s, e in runs:
        if s == e:
            run_strs.append(str(s))
        else:
            run_strs.append(f"{s}-{e}")
    parts.append(",".join(run_strs))
    return "".join(parts)  # no extra spaces


def _summarize_grid(grid, step_id):
    """Produce a compact structured summary.

    The summary explicitly identifies:
    - step id (to make states unique even when grid is static)
    - background colour
    - fixed top‑left 2x2 block (cells (0,0),(0,1),(1,0),(1,1))
    - movable block (three contiguous gray cells on row 0 that are NOT part of the fixed block)
    - all other non‑background cells, grouped by colour and row with contiguous runs.
    Output is under 2000 characters, never empty.
    """
    if not grid:
        return f"step={step_id}; bg=unknown; grid_empty"

    rows = len(grid)
    # Determine background
    colour_counts = Counter()
    for row in grid:
        colour_counts.update(row)
    if not colour_counts:
        return f"step={step_id}; bg=unknown; grid_empty"
    bg = colour_counts.most_common(1)[0][0]

    # Collect all non‑background cells as (r, c, colour)
    cells = []
    for r in range(rows):
        for c in range(len(grid[r])):
            colour = grid[r][c]
            if colour != bg:
                cells.append((r, c, colour))

    # Fixed block positions (0,0), (0,1), (1,0), (1,1)
    fixed_positions = {(0,0), (0,1), (1,0), (1,1)}
    # Collect fixed block colours
    fixed_cells = {}
    for (r,c) in fixed_positions:
        if r < rows and c < len(grid[r]):
            fixed_cells[(r,c)] = grid[r][c]
        else:
            fixed_cells[(r,c)] = "bg"

    # Identify movable block: three contiguous gray cells on row 0 NOT in fixed block
    fixed_cols_row0 = {c for (r,c) in fixed_positions if r==0}  # {0,1}
    gray_row0_cols = sorted([c for r,c,col in cells if r==0 and col=="gray" and c not in fixed_cols_row0])
    movable_run = None
    if gray_row0_cols:
        runs = _get_runs(gray_row0_cols)
        for s,e in runs:
            if e - s + 1 == 3:
                movable_run = (s, e)
                break
    if movable_run is None:
        # fallback: report all gray cells on row0 (excluding fixed) as movable? but we need something
        # If there is any run, use the longest? But we keep the fallback as list.
        movable_run = (-1, -1)  # signal not found
        movable_fallback = gray_row0_cols  # list of columns

    mob_columns = set(range(movable_run[0], movable_run[1]+1)) if movable_run[0] >= 0 else set()

    # Build output parts
    parts = [f"step={step_id}", f"bg={bg}"]

    # Fixed block
    fixed_entries = []
    for (r,c) in sorted(fixed_positions):
        col = fixed_cells.get((r,c), "bg")
        if (r,c) in mob_columns:
            fixed_entries.append(f"{col}({r},{c})[mob]")
        else:
            fixed_entries.append(f"{col}({r},{c})")
    parts.append("fixed:" + ",".join(fixed_entries))

    # Movable block
    if movable_run[0] >= 0:
        parts.append(f"movable:gray_row0:{movable_run[0]}-{movable_run[1]}")
    else:
        # fallback: list all gray cells on row0 (excluding fixed) individually
        if movable_fallback:
            parts.append(f"movable:gray_row0:{','.join(map(str, movable_fallback))}")
        else:
            parts.append("movable:none")

    # Other non‑background cells (not fixed, not movable)
    other_cells = []
    for (r,c,col) in cells:
        if (r,c) in fixed_positions:
            continue
        if r==0 and (movable_run[0] >= 0 and c in mob_columns):
            continue
        other_cells.append((r,c,col))

    # Group by colour then by row
    by_colour = {}
    for r,c,col in other_cells:
        by_colour.setdefault(col, {}).setdefault(r, []).append(c)

    other_strs = []
    for colour in sorted(by_colour.keys()):
        rows_dict = by_colour[colour]
        for r in sorted(rows_dict.keys()):
            cols = sorted(rows_dict[r])
            runs = _get_runs(cols)
            other_strs.append(_format_runs(colour, r, runs))

    if other_strs:
        parts.append("other:" + "; ".join(other_strs))
    else:
        parts.append("other:none")

    # Assemble final string, enforce 2000 limit
    result = "; ".join(parts)
    if len(result) > 2000:
        # Truncate by dropping some other parts from the end
        core = parts[:3]  # step, bg, fixed, movable -> actually 3 parts? Wait: parts[0], parts[1], parts[2] are step, bg, fixed; movable is parts[3]; other is parts[4]
        # Actually parts: index 0 step, 1 bg, 2 fixed, 3 movable, 4 other...
        # Keep step, bg, fixed, movable; drop from other
        core = parts[:4]  # step, bg, fixed, movable
        remaining = parts[4:]  # other parts
        while len("; ".join(core + remaining)) > 2000 and remaining:
            remaining.pop()
        if remaining:
            result = "; ".join(core + remaining)
        else:
            # even core might be too long? unlikely but handle
            result = "; ".join(core)[:2000]

    # Never empty
    if not result:
        result = f"step={step_id}; bg=unknown; parse_failure"
    return result


def perceive(observation_history: list[str]) -> str:
    """Produce a concise text summary of the current grid observation.

    The summary captures the step number, background, fixed block, movable block,
    and all other non‑background cells (compressed as contiguous runs per colour/row).
    This ensures that any change is reflected and the action is recoverable from a window
    of consecutive summaries. Output is never empty and never raises.
    """
    step = len(observation_history) - 1 if observation_history else 0
    obs = observation_history[-1] if observation_history else ""
    grid = _parse_grid(obs)
    if grid is None:
        return f"step={step}; bg=unknown; parse_failure"
    return _summarize_grid(grid, step)