import json
import re
from typing import List, Optional, Dict, Set, Tuple

def perceive(observation_history: list[str]) -> str:
    """
    Parse the current raw observation and output a concise text summary.
    The summary is designed so that two consecutive outputs together contain
    enough information to uniquely identify the action taken between them.
    """
    try:
        obs = observation_history[-1]
        step = _extract_int(obs, r'Step:\s*(\d+)', default=0)
        action_cnt = _extract_int(obs, r'Action count:\s*(\d+)', default=0)
        task_type = _extract_task_type(obs)
        phase = _extract_phase(obs)

        grid = _parse_grid(obs)
        if grid is None:
            return f"step={step} cnt={action_cnt} task={task_type} phase={phase} (grid_parse_failed)"

        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        bg = _get_background_color(grid)
        bg_short = _color_short(bg)

        # compute current set of non‑background cells
        current_cells = []
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != bg:
                    current_cells.append((grid[r][c], r, c))

        # ----- determine object order by first appearance in history -----
        first_occurrence = {}
        for t, prev_obs in enumerate(observation_history[:-1]):
            prev_grid = _parse_grid(prev_obs)
            if prev_grid is None:
                continue
            prev_bg = _get_background_color(prev_grid)
            for r in range(len(prev_grid)):
                for c in range(len(prev_grid[0])):
                    val = prev_grid[r][c]
                    if val != prev_bg:
                        key = (val, r, c)
                        if key not in first_occurrence:
                            first_occurrence[key] = t
        curr_timestamp = len(observation_history) - 1
        cell_order = []
        for cell in current_cells:
            ts = first_occurrence.get(cell, curr_timestamp)
            cell_order.append((ts, cell[1], cell[2], cell[0]))
        cell_order.sort()
        objs_parts = []
        for _, r, c, col in cell_order:
            col_short = _color_short(col)
            objs_parts.append(f"{col_short}:{r},{c}")
        objs_str = ";".join(objs_parts) if objs_parts else "none"
        cell_count = len(objs_parts)

        # ----- detect changes from previous state -----
        diff_str = ""
        click_info = ""
        if len(observation_history) >= 2:
            prev_obs = observation_history[-2]
            prev_grid = _parse_grid(prev_obs)
            if prev_grid is not None:
                prev_bg = _get_background_color(prev_grid)
                curr_set = set(current_cells)
                prev_set = set()
                for r in range(len(prev_grid)):
                    for c in range(len(prev_grid[0])):
                        if prev_grid[r][c] != prev_bg:
                            prev_set.add((prev_grid[r][c], r, c))
                added = curr_set - prev_set
                removed = prev_set - curr_set
                diff_parts = []
                for col, r, c in sorted(added, key=lambda x: (x[1], x[2])):
                    diff_parts.append(f"+{_color_short(col)}:{r},{c}")
                for col, r, c in sorted(removed, key=lambda x: (x[1], x[2])):
                    diff_parts.append(f"-{_color_short(col)}:{r},{c}")
                if diff_parts:
                    diff_str = " diff=" + ",".join(diff_parts)
                
                # Check if this transition looks like a click (red appears, gray disappears at same position)
                if len(added) == 1 and len(removed) == 1:
                    add_col, add_r, add_c = next(iter(added))
                    rem_col, rem_r, rem_c = next(iter(removed))
                    if add_col == 2 and add_r == rem_r and add_c == rem_c:
                        # Likely a click - mark coordinates as unknown
                        click_info = " click_effect=yes click_coords=unknown"

        # ----- extract explicit action from observation text -----
        explicit_action = _extract_action(obs)
        act_str = ""
        if explicit_action:
            act_str = f" act={explicit_action}"
        elif click_info:
            # No explicit action found but a click effect was detected
            act_str = " act=click ? ?"

        # ----- build output -----
        output = (f"step={step} cnt={action_cnt} task={task_type} phase={phase} "
                  f"grid={rows}x{cols} bg={bg_short} "
                  f"objs_count={cell_count} objs={objs_str}"
                  f"{diff_str}{click_info}{act_str}")

        # Ensure length limit
        if len(output) > 2000:
            output = output[:1997] + "..."

        return output

    except Exception as e:
        return f"(perception_error: {str(e)})"


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def _extract_int(text: str, pattern: str, default: int) -> int:
    m = re.search(pattern, text)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return default
    return default

def _extract_task_type(text: str) -> str:
    m = re.search(r'Task:\s*(\w+)', text)
    return m.group(1) if m else "unknown"

def _extract_phase(text: str) -> str:
    m = re.search(r'Phase:\s*(.+?)\s*(?:\n|$)', text)
    return m.group(1).strip() if m else "unknown"

def _extract_action(text: str) -> str:
    """
    Extract the action that was taken from the observation text.
    Returns a string like "click 2 14", "noop", or "" if not found.
    """
    lines = text.split('\n')
    for line in lines:
        stripped = line.strip()
        if 'available' in stripped.lower():
            continue
        m = re.match(
            r'^(?:Action\s+(?:taken|performed)?\s*|The\s+action\s+(?:was|is)\s*)[:=]?\s*'
            r'(click\s+\d+\s+\d+|noop|quit|reset|left|right|up|down)\b',
            stripped, re.IGNORECASE
        )
        if m:
            action = m.group(1).strip()
            if action:
                action = re.sub(r'\s+', ' ', action)
                if action.lower().startswith('click'):
                    nums = re.findall(r'\d+', action)
                    if len(nums) >= 2:
                        action = f"click {nums[0]} {nums[1]}"
                return action
    # Fallback patterns
    patterns = [
        r'(?:^|\n)\s*(?:Action\s+(?:taken|performed)?\s*|The\s+action\s+(?:was|is)\s*)[:=]?\s*(click\s+\d+\s+\d+)',
        r'(?:^|\n)\s*(?:Action\s+(?:taken|performed)?\s*|The\s+action\s+(?:was|is)\s*)[:=]?\s*(noop)',
        r'(?:^|\n)\s*(?:Action\s+(?:taken|performed)?\s*|The\s+action\s+(?:was|is)\s*)[:=]?\s*(quit)',
        r'(?:^|\n)\s*(?:Action\s+(?:taken|performed)?\s*|The\s+action\s+(?:was|is)\s*)[:=]?\s*(reset)',
        r'(?:^|\n)\s*(?:Action\s+(?:taken|performed)?\s*|The\s+action\s+(?:was|is)\s*)[:=]?\s*(left)',
        r'(?:^|\n)\s*(?:Action\s+(?:taken|performed)?\s*|The\s+action\s+(?:was|is)\s*)[:=]?\s*(right)',
        r'(?:^|\n)\s*(?:Action\s+(?:taken|performed)?\s*|The\s+action\s+(?:was|is)\s*)[:=]?\s*(up)',
        r'(?:^|\n)\s*(?:Action\s+(?:taken|performed)?\s*|The\s+action\s+(?:was|is)\s*)[:=]?\s*(down)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            action = m.group(1).strip()
            if len(action) > 60:
                action = action[:60]
            return action
    return ""


_COLOR_NAME_TO_INT: Dict[str, int] = {
    "black": 0, "blue": 1, "red": 2, "green": 3, "yellow": 4,
    "light-gray": 5, "gray": 5, "lightgray": 5,
    "magenta": 6, "orange": 7, "light-blue": 8, "lightblue": 8,
    "dark-red": 9, "maroon": 9, "white": 10, "dark-gray": 11, "darkgray": 11,
    "gold": 4
}

_INT_TO_SHORT: Dict[int, str] = {
    0: "blk", 1: "blu", 2: "red", 3: "grn", 4: "ylw",
    5: "gry", 6: "mag", 7: "orn", 8: "lbl", 9: "mrn",
    10: "wht", 11: "dgy"
}

def _color_short(val: int) -> str:
    return _INT_TO_SHORT.get(val, f"c{val}")

def _parse_grid(obs: str) -> Optional[List[List[int]]]:
    """Parse the grid from the observation text. Returns list of list of ints (0-11)."""
    # Method 1: ARC integer grid with <grid_k> markers
    m = re.search(r'<grid_\d+>', obs)
    if m:
        marker_end = m.end()
        lines_after = obs[marker_end:].strip().split('\n')
        rows = []
        for line in lines_after:
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                try:
                    row_str = line[1:-1]
                    row = [int(x.strip()) for x in row_str.split(',')]
                    if row:
                        rows.append(row)
                except (ValueError, IndexError):
                    continue
            elif line == '' or line.startswith('<') or line.startswith('='):
                if not rows:
                    continue
                break
        if rows:
            return rows

    # Method 2: JSON 2D array of colour-name strings (contiguous)
    start = obs.find('[[')
    end = obs.rfind(']]')
    if start != -1 and end != -1 and end > start:
        try:
            json_str = obs[start:end+2]
            parsed = json.loads(json_str)
            if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], list):
                grid = []
                for row in parsed:
                    int_row = []
                    for cell in row:
                        if isinstance(cell, int):
                            int_row.append(cell)
                        elif isinstance(cell, str):
                            c = cell.lower().strip()
                            int_row.append(_COLOR_NAME_TO_INT.get(c, 0))
                        else:
                            int_row.append(0)
                    grid.append(int_row)
                return grid
        except (json.JSONDecodeError, IndexError, TypeError):
            pass

    # Fallback: try to find any JSON array in the text
    try:
        for pattern in [r'\[\[.*?\]\]', r'\[.*?\]']:
            match = re.search(pattern, obs, re.DOTALL)
            if match:
                candidate = match.group()
                parsed = json.loads(candidate)
                if isinstance(parsed, list) and len(parsed) > 0:
                    if isinstance(parsed[0], list):
                        grid = []
                        for row in parsed:
                            int_row = []
                            for cell in row:
                                if isinstance(cell, int):
                                    int_row.append(cell)
                                elif isinstance(cell, str):
                                    c = cell.lower().strip()
                                    int_row.append(_COLOR_NAME_TO_INT.get(c, 0))
                                else:
                                    int_row.append(0)
                            grid.append(int_row)
                        return grid
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    return None

def _get_background_color(grid: List[List[int]]) -> int:
    counts: Dict[int, int] = {}
    for row in grid:
        for cell in row:
            counts[cell] = counts.get(cell, 0) + 1
    if not counts:
        return 0
    return max(counts, key=counts.get)