import json
import ast
from collections import Counter
from typing import List, Tuple, Union, Optional

def perceive(observation_history: list[str]) -> str:
    """
    Parse the current raw observation and return a concise text summary
    that distinguishes consecutive states and makes the action recoverable.

    Fixed version:
      - Always lists all moving heads (cells whose colour differs from the
        most common non‑background colour), even if they have no outward
        directions.
      - Outputs grid dimensions (rows, cols) so that any cell not listed
        in 'trail' or 'heads' is known to be background (empty). This
        makes interior trail cells distinguishable from empty ones.
      - Robustly detects and parses both ARC integer grids and Autumn
        JSON colour‑name grids.
      - Never raises, never returns empty, and stays under 2000 chars.
    """
    try:
        obs = observation_history[-1]
    except IndexError:
        return "empty_history"

    # ----- attempt to detect and parse Autumn (JSON) format -----
    grid = None
    # Look for a JSON array starting with [[" ... a colour keyword
    import re
    # Pattern: [[ followed by a lowercase colour name (or digit for ARC)
    m = re.search(r'\[\["(black|gray|blue|red|green|gold|lightblue|darkgreen|yellow|orange|white|tan|lightgray|darkgray|pink|purple|brown|magenta|cyan|olive|navy|teal|violet|indigo|maroon|beige|lime|coral|aqua|plum|wheat|salmon|khaki|lavender|crimson|firebrick|dodgerblue|slategray|seagreen|rosybrown|peru|palegoldenrod|mediumorchid|lightsteelblue|lightsalmon|lightcyan|lightcoral|lawngreen|indigo|honeydew|hotpink|goldenrod|forestgreen|darkslategray|darkseagreen|darkorange|darkmagenta|darkgreen|darkgoldenrod|darkcyan|darkblue|darksalmon|darkred|darkviolet|darkturquoise|darkolivegreen|darkkhaki|darkgray|darkgreen|darkgrey|dimgray|dimgrey|slategrey|lightslategray|lightslategrey|gainsboro|ghostwhite|snow|seashell|mintcream|ivory|floralwhite|aliceblue|azure|honeydew|oldlace|mistyrose|papayawhip|blanchedalmond|bisque|moccasin|navajowhite|peachpuff|antiquewhite|wheat|cornflowerblue|dodgerblue|deepskyblue|skyblue|lightblue|powderblue|paleturquoise|mediumturquoise|turquoise|darkturquoise|lightcyan|cyan|aquamarine|mediumspringgreen|springgreen|palegreen|lightgreen|darkseagreen|mediumseagreen|seagreen|olivedrab|darkolivegreen|yellowgreen|greenyellow|chartreuse|lawngreen|limegreen|lime|forestgreen|green|darkgreen|darkgrey|dimgray|dimgrey|gray|grey|lightgray|lightgrey|gainsboro|whitesmoke|silver|darkgray|darkgrey|slategray|lightslategray|slategrey|lightslategrey|darkred|maroon|firebrick|brown|indianred|lightcoral|salmon|darksalmon|lightsalmon|orangered|tomato|coral|darkorange|orange|gold|yellow|lightyellow|lemonchiffon|lightgoldenrodyellow|papayawhip|moccasin|peachpuff|palegoldenrod|khaki|darkkhaki|wheat|navajowhite|antiquewhite|blanchedalmond|bisque|sandybrown|peru|chocolate|sienna|saddlebrown|burlywood|tan|rosybrown|thistle|plum|violet|orchid|fuchsia|magenta|mediumorchid|mediumpurple|blueviolet|indigo|darkslateblue|slateblue|mediumslateblue|royalblue|mediumblue|blue|darkblue|navy|midnightblue|cornflowerblue|dodgerblue|deepskyblue|steelblue|lightsteelblue|skyblue|lightskyblue|paleturquoise|mediumturquoise|turquoise|darkturquoise|lightcyan|cyan|aquamarine|mediumaquamarine|aqua|teal|darkcyan|lightseagreen|mediumseagreen|seagreen|olivedrab|darkolivegreen|yellowgreen|greenyellow|chartreuse|lawngreen|limegreen|lime|forestgreen|green|darkgreen|darkgrey|dimgray|dimgrey|gray|grey|lightgray|lightgrey|silver|darkgray|darkgrey|slategray|lightslategray)"]', obs, re.DOTALL)
    if m:
        start = m.start()
        end = obs.rfind("]]")
        if end > start:
            try:
                parsed = json.loads(obs[start:end+2])
                if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], list):
                    grid = parsed
            except (json.JSONDecodeError, ValueError):
                pass

    # ----- if not JSON, try ARC integer grid -----
    if grid is None:
        rows = []
        for line in obs.splitlines():
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                try:
                    row = ast.literal_eval(line)
                    if isinstance(row, list) and all(isinstance(x, int) for x in row):
                        rows.append(row)
                except (ValueError, SyntaxError, MemoryError):
                    continue
        if rows and len(set(len(r) for r in rows)) == 1:
            grid = rows

    if grid is None:
        return "grid_not_found"

    # ----- grid dimensions -----
    H = len(grid)
    W = len(grid[0]) if H > 0 else 0
    if H == 0 or W == 0:
        return "empty_grid"

    # ----- flatten and find background (most frequent) -----
    flat = [cell for row in grid for cell in row]
    if not flat:
        return "empty_grid"
    background = Counter(flat).most_common(1)[0][0]

    # ----- collect non‑background cells -----
    non_bg = [(r, c, grid[r][c]) for r in range(H) for c in range(W) if grid[r][c] != background]

    if not non_bg:
        return f"grid {H}x{W} bg {str(background)} empty"

    # ----- determine trail colour (most common non‑background) -----
    non_bg_colours = Counter(v for _,_,v in non_bg)
    trail_colour = non_bg_colours.most_common(1)[0][0]

    # ----- separate heads (non‑trail colours) and trail cells -----
    heads = []   # (r, c, colour)
    trail = []   # (r, c)
    for r, c, val in non_bg:
        if val == trail_colour:
            trail.append((r, c))
        else:
            heads.append((r, c, val))

    # ---- helper to get neighbour directions that are background ----
    def outward_dirs(r, c):
        dirs = []
        for dr, dc, name in [(-1,0,'up'), (1,0,'down'), (0,-1,'left'), (0,1,'right')]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == background:
                dirs.append(name)
        return dirs

    # ---- build head strings ----
    head_parts = []
    for r, c, col in heads:
        dirs = outward_dirs(r, c)
        dir_str = ','.join(dirs) if dirs else 'none'
        head_parts.append(f"({r},{c}):{col}:{dir_str}")

    # ---- build clickable boundary strings (trail cells with outward dirs) ----
    clickable_parts = []
    for r, c in trail:
        dirs = outward_dirs(r, c)
        if dirs:
            dir_str = ','.join(dirs)
            clickable_parts.append(f"({r},{c}):{dir_str}")

    # ---- build trail strings ----
    trail_parts = [f"({r},{c})" for r, c in trail]

    # ---- assemble final summary ----
    meta = f"grid {H}x{W} bg {str(background)} trail {str(trail_colour)}"

    head_section = "heads: " + ("; ".join(head_parts) if head_parts else "none")
    clickable_section = "clickable: " + ("; ".join(clickable_parts) if clickable_parts else "none")
    trail_section = "trail: " + ("; ".join(trail_parts) if trail_parts else "none")

    summary = f"{meta} | {head_section} | {clickable_section} | {trail_section}"

    # ---- enforce length limit (2000 chars) ----
    if len(summary) > 1990:
        # shorten trail section first (it's usually the longest)
        max_trail_len = 1990 - (len(summary) - len(trail_section)) - 30
        if max_trail_len > 20 and len(trail_parts) > 5:
            keep = []
            total = 0
            for p in trail_parts:
                if total + len(p) + 2 > max_trail_len:
                    break
                keep.append(p)
                total += len(p) + 2
            shortened_trail = "; ".join(keep) + f" ...(+{len(trail_parts)-len(keep)} cells)"
            summary = f"{meta} | {head_section} | {clickable_section} | trail: {shortened_trail}"
        # if still too long, hard cut
        if len(summary) > 1995:
            summary = summary[:1995] + "..."

    return summary