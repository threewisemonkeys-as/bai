import json
import re
from typing import List, Tuple, Dict, Optional

def perceive(observation_history: list[str]) -> str:
    """Parse raw observation and produce decision-relevant features."""
    if not observation_history:
        return "empty grid"

    raw = observation_history[-1]

    try:
        # Detect and parse grid
        grid = None
        # Try JSON first (Autumn format)
        if raw.strip().startswith('[['):
            grid = parse_json_grid(raw)
        elif '<grid_' in raw or any(c.isdigit() for c in raw[:50]):  # heuristics for ARC
            grid = parse_arc_grid(raw)
        else:
            grid = parse_json_grid(raw)  # fallback

        if grid is None or len(grid) == 0:
            return "empty grid"

        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0

        # Flatten and count
        flat = [cell for row in grid for cell in row]
        if not flat:
            return "empty grid"

        # Count colour occurrences
        colour_counts = {}
        for c in flat:
            colour_counts[c] = colour_counts.get(c, 0) + 1

        # Dominant background
        background = max(colour_counts, key=colour_counts.get)
        bg = background

        # Identify markers (red=red, green=green) – assume they are always present
        markers = []
        for r in range(rows):
            for c in range(cols):
                val = grid[r][c]
                if val == "red":
                    markers.append((r, c, "R"))
                elif val == "green":
                    markers.append((r, c, "G"))
        # Sort markers by row,col
        markers.sort(key=lambda x: (x[0], x[1]))

        # Collect non-background cells by colour
        colour_groups: Dict[str, List[Tuple[int, int]]] = {}
        for r in range(rows):
            for c in range(cols):
                val = grid[r][c]
                if val != bg:
                    colour_groups.setdefault(val, []).append((r, c))

        # Build summary parts
        parts = []

        # Markers
        if markers:
            parts.append("markers:" + ";".join(f"{t}({r},{c})" for r,c,t in markers))

        # Dimensions and background
        meta = f"grid={rows}x{cols} bg={bg}"
        parts.append(meta)

        # For each colour that is not background and not a marker (unless marker colour appears elsewhere)
        for colour, cells in sorted(colour_groups.items()):
            # Skip marker colours if they only appear as markers? But red/green may appear only at markers.
            if colour in ("red", "green") and len(cells) <= 2:  # likely markers
                continue
            # For large groups (e.g., tan blob) use bounding box
            if len(cells) > 8:
                # compute bounding box
                min_r = min(r for r,c in cells)
                max_r = max(r for r,c in cells)
                min_c = min(c for r,c in cells)
                max_c = max(c for r,c in cells)
                box = f"bbox({min_r},{min_c},{max_r},{max_c})"
                count = f"cnt={len(cells)}"
                parts.append(f"{colour}:{box} {count}")
            else:
                # list individual positions
                positions = sorted(cells)
                pos_str = ";".join(f"({r},{c})" for r,c in positions)
                parts.append(f"{colour}:{pos_str}")

        # Join with separator
        summary = " | ".join(parts)

        # Fallback to simple list if summary becomes too long? Already compact.
        if len(summary) > 1990:
            # Truncate middle
            half = 990
            summary = summary[:half] + "...[truncated]..." + summary[-half:]

        return summary

    except Exception as e:
        return f"parse_error: {str(e)[:100]}"


def parse_json_grid(text: str) -> Optional[List[List[str]]]:
    """Parse JSON 2D array of colour names."""
    try:
        start = text.find('[[')
        if start == -1:
            return None
        # Find matching closing brackets
        depth = 0
        end = -1
        for i in range(start, len(text)):
            if text[i] == '[':
                depth += 1
            elif text[i] == ']':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end == -1:
            return None
        json_str = text[start:end]
        grid = json.loads(json_str)
        # Convert all cells to lowercase strings
        result = []
        for row in grid:
            result.append([str(cell).lower() for cell in row])
        return result
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def parse_arc_grid(text: str) -> Optional[List[List[str]]]:
    """Parse ARC integer grid format."""
    colour_map = {
        0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
        5: "lightgray", 6: "magenta", 7: "orange", 8: "lightblue",
        9: "maroon", 10: "white", 11: "darkgray"
    }
    try:
        lines = text.split('\n')
        grids = []
        current_grid = None
        for line in lines:
            # grid marker
            grid_match = re.match(r'<grid_(\d+)>', line.strip())
            if grid_match:
                if current_grid is not None:
                    grids.append(current_grid)
                current_grid = []
                continue
            # row
            if current_grid is not None:
                row_match = re.match(r'\[([^\]]+)\]', line.strip())
                if row_match:
                    try:
                        row = [int(x.strip()) for x in row_match.group(1).split(',')]
                        current_grid.append(row)
                    except ValueError:
                        continue
        if current_grid is not None and len(current_grid) > 0:
            grids.append(current_grid)

        if not grids:
            return None

        # Use first grid
        grid = grids[0]
        result = []
        for row in grid:
            result.append([colour_map.get(cell, f"x{cell}") for cell in row])
        return result

    except Exception:
        return None