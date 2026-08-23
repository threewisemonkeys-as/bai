import json
import re
from typing import List, Optional

def perceive(observation_history: list[str]) -> str:
    """
    Produce a concise text summary of decision-relevant features from the current raw observation.
    The output explicitly includes the step modulo 4 (phase) so that transition rules are unambiguous.
    """
    try:
        raw = observation_history[-1]

        # ----- parse global metadata (step, action count) -----
        step = 0
        # action_count  # (not used but kept for completeness)
        for line in raw.split('\n'):
            m = re.search(r'Step:\s*(\d+)', line)
            if m:
                step = int(m.group(1))
            # m = re.search(r'Action count:\s*(\d+)', line)
            # if m:
            #     action_count = int(m.group(1))

        # ----- detect grid format and parse -----
        if '<grid_' in raw:
            grid = _parse_arc_grid(raw)
        else:
            grid = _parse_json_grid(raw)

        if not grid or not grid[0]:
            return f"step={step} | (empty)"

        rows = len(grid)
        cols = len(grid[0])

        # ----- colour name mapping (ARC palette) -----
        integer_color_names = {
            0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
            5: "light-gray", 6: "magenta", 7: "orange", 8: "light-blue",
            9: "maroon", 10: "white", 11: "dark-gray"
        }

        # ----- determine background colour -----
        colour_counts = {}
        for r in range(rows):
            for c in range(cols):
                cell = grid[r][c]
                val = cell if isinstance(cell, int) else cell
                colour_counts[val] = colour_counts.get(val, 0) + 1
        if not colour_counts:
            return f"step={step} | (empty)"
        bg_val = max(colour_counts, key=colour_counts.get)

        # ----- collect all non‑background cells with positions -----
        non_bg = {}   # colour_name -> list of (r,c)
        for r in range(rows):
            for c in range(cols):
                cell = grid[r][c]
                if isinstance(cell, int):
                    name = integer_color_names.get(cell, str(cell))
                else:
                    name = cell
                if name != _name(bg_val, integer_color_names):
                    non_bg.setdefault(name, []).append((r, c))

        # ----- build summary parts -----
        bg_name = _name(bg_val, integer_color_names)
        phase = step % 4
        parts = [f"step={step}", f"phase={phase}", f"grid={rows}x{cols}", f"bg={bg_name}"]

        # List non‑background colours in a fixed order: black, brown, silver, then others
        colour_order = ["black", "brown", "silver"]
        # sort by order, then alphabetically for others
        sorted_colours = sorted(non_bg.keys(), key=lambda c: colour_order.index(c) if c in colour_order else 99 + ord(c[0]))

        colour_strs = []
        for colour in sorted_colours:
            positions = non_bg[colour]
            # sort positions for consistency (row first, then col)
            positions.sort()
            pos_str = ";".join([f"({r},{c})" for r, c in positions])
            colour_strs.append(f"{colour}:{pos_str}")
        if colour_strs:
            parts.append("nonbg=" + ",".join(colour_strs))

        # Always report counts for silver and brown (even if zero) to avoid omissions
        silver_count = len(non_bg.get("silver", []))
        brown_count = len(non_bg.get("brown", []))
        parts.append(f"silver={silver_count}")
        parts.append(f"brown={brown_count}")

        result = " | ".join(parts)

        # ----- ensure length < 2000 chars -----
        if len(result) > 1900:
            # Truncate by removing positions of the largest colour group (usually black)
            # but keep the counts.
            fallback_parts = [f"step={step}", f"phase={phase}", f"grid={rows}x{cols}", f"bg={bg_name}"]
            for colour in sorted_colours:
                fallback_parts.append(f"{colour}={len(non_bg[colour])}")
            fallback_parts.append(f"silver={silver_count}")
            fallback_parts.append(f"brown={brown_count}")
            result = " | ".join(fallback_parts)

        if not result:
            return f"step={step} | (empty)"

        return result

    except Exception:
        return "(empty)"


def _name(val, mapping):
    if isinstance(val, int):
        return mapping.get(val, str(val))
    return val


def _parse_arc_grid(raw: str) -> Optional[List]:
    """Parse ARC integer grid format."""
    try:
        lines = raw.split('\n')
        in_grid = False
        grid = []
        for line in lines:
            stripped = line.strip()
            if '<grid_' in stripped:
                in_grid = True
                continue
            if in_grid:
                if stripped.startswith('[') and stripped.endswith(']'):
                    row_str = stripped[1:-1].strip()
                    if row_str:
                        row = [int(x.strip()) for x in row_str.split(',')]
                        grid.append(row)
                elif not stripped or '=' in stripped or 'State:' in stripped:
                    if grid:
                        break
        return grid if grid else None
    except Exception:
        return None


def _parse_json_grid(raw: str) -> Optional[List]:
    """Parse JSON 2D array format."""
    try:
        s = raw.find('[[')
        if s == -1:
            return None
        depth = 0
        e = s
        for i, ch in enumerate(raw[s:], s):
            if ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    e = i + 1
                    break
        if e <= s:
            return None
        json_str = raw[s:e]
        grid = json.loads(json_str)
        if not isinstance(grid, list) or not grid or not isinstance(grid[0], list):
            return None
        return grid
    except Exception:
        return None