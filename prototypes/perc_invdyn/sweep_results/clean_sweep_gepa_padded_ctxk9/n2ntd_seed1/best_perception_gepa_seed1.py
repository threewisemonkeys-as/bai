import json
import re

def perceive(observation_history: list[str]) -> str:
    """
    Parse the raw observation and return a concise but complete text summary.
    Includes all non‑white cells with exact positions, derived features like
    player, gravity, moving platforms, plus step number and action count to
    disambiguate states where the grid is identical.
    """
    try:
        obs = observation_history[-1]
        grid = None
        grid_type = None

        # --- extract metadata (step, action count, levels) ---
        step = None
        action_count = None
        levels_completed = None
        for line in obs.split('\n'):
            line = line.strip()
            m = re.match(r'Step:\s*(\d+)', line)
            if m:
                step = int(m.group(1))
            m = re.match(r'Action count:\s*(\d+)', line)
            if m:
                action_count = int(m.group(1))
            m = re.match(r'Levels completed:\s*(\d+)/', line)
            if m:
                levels_completed = int(m.group(1))

        # --- detect and parse grid ---
        # try Autumn string grid (JSON 2D array)
        if '[[' in obs and ']]' in obs:
            try:
                start = obs.find('[[')
                end = obs.rfind(']]') + 2
                json_str = obs[start:end]
                parsed = json.loads(json_str)
                if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], list):
                    grid = parsed
                    grid_type = 'string'
            except Exception:
                pass

        # try ARC integer grid
        if grid is None:
            try:
                rows = []
                in_grid = False
                for line in obs.split('\n'):
                    line_stripped = line.strip()
                    if line_stripped.startswith('<grid_'):
                        in_grid = True
                        continue
                    if in_grid and line_stripped.startswith('[') and line_stripped.endswith(']'):
                        row_str = line_stripped.strip('[]')
                        row = [int(x.strip()) for x in row_str.split(',')]
                        rows.append(row)
                    elif in_grid and not line_stripped.startswith('['):
                        if line_stripped.startswith('<') or line_stripped.startswith('='):
                            break
                if rows:
                    grid = rows
                    grid_type = 'int'
            except Exception:
                pass

        if grid is None:
            return f"grid: parse_failed | step:{step}" if step is not None else "grid: parse_failed"

        rows_count = len(grid)
        cols_count = len(grid[0]) if rows_count > 0 else 0
        if rows_count == 0 or cols_count == 0:
            return f"grid: {rows_count}x{cols_count} empty | step:{step}"

        # --- collect all non-background cells ---
        color_short = {
            'white': 'W', 'black': 'Bk', 'blue': 'Bl', 'red': 'R',
            'green': 'G', 'gold': 'Gd', 'gray': 'Gy', 'lightgray': 'LGy',
            'darkgray': 'DGy', 'lightblue': 'LBl', 'darkgreen': 'DG',
            'darkorange': 'DO', 'mediumpurple': 'MP', 'purple': 'P',
            'orange': 'O', 'yellow': 'Y', 'magenta': 'M', 'cyan': 'C',
            'brown': 'Br', 'pink': 'Pk', 'maroon': 'Mr', 'navy': 'Nv',
            'teal': 'Tl', 'olive': 'Ol', 'lime': 'Lm', 'indigo': 'In',
            'violet': 'V', 'turquoise': 'Tq', 'tan': 'Tn', 'coral': 'Cr',
            'salmon': 'Sm', 'plum': 'Pl', 'orchid': 'Or', 'lavender': 'Lv'
        }

        cells = []  # list of (r, c, short_colour)
        player_pos = None
        gold_positions = []
        platform_positions = []   # darkorange
        moving_platform_positions = []   # mediumpurple
        blue_positions = []

        for r in range(rows_count):
            for c in range(cols_count):
                cell = grid[r][c]
                if grid_type == 'string':
                    colour_name = cell.lower() if isinstance(cell, str) else str(cell)
                    if colour_name == 'white':
                        continue
                    short = color_short.get(colour_name, colour_name[:2])
                else:  # integer grid
                    val = int(cell)
                    if val == 0:
                        continue
                    short = str(val)
                    colour_name = str(val)

                cells.append((r, c, short))

                if colour_name in ('red', '2'):
                    player_pos = (r, c)
                elif colour_name in ('gold', '4'):
                    gold_positions.append((r, c))
                elif colour_name in ('darkorange', '7'):
                    platform_positions.append((r, c))
                elif colour_name in ('mediumpurple', '6'):
                    moving_platform_positions.append((r, c))
                elif colour_name in ('blue', '1'):
                    blue_positions.append((r, c))

        # --- build summary parts ---
        parts = []
        # metadata
        meta_parts = []
        if step is not None:
            meta_parts.append(f"step:{step}")
        if action_count is not None:
            meta_parts.append(f"ac:{action_count}")
        if levels_completed is not None:
            meta_parts.append(f"lvl:{levels_completed}")
        if meta_parts:
            parts.append(' '.join(meta_parts))

        parts.append(f"grid:{rows_count}x{cols_count}")

        # all non-background cells grouped by colour
        if cells:
            by_color = {}
            for r, c, short in cells:
                by_color.setdefault(short, []).append((r, c))
            colour_strings = []
            for colour in sorted(by_color.keys()):
                pos_list = by_color[colour]
                sorted_pos = sorted(pos_list)
                pos_str = ','.join(f"({r},{c})" for r, c in sorted_pos)
                colour_strings.append(f"{colour}:{pos_str}")
            parts.append("cells:" + ';'.join(colour_strings))
        else:
            parts.append("all_white")

        # derived features
        if player_pos:
            parts.append(f"player:({player_pos[0]},{player_pos[1]})")
            # gravity
            if player_pos[0] + 1 < rows_count:
                r, c = player_pos
                below = grid[r+1][c]
                if grid_type == 'string':
                    is_empty = (below.lower() if isinstance(below, str) else str(below)) == 'white'
                else:
                    is_empty = (int(below) == 0)
                parts.append("gravity:down" if is_empty else "gravity:none")
            else:
                parts.append("gravity:none")
        else:
            # player not visible (e.g. on gold) – we can't infer gravity reliably
            pass

        # explicit lists (redundant but helpful)
        if gold_positions:
            parts.append("golds:" + ','.join(f"({r},{c})" for r,c in gold_positions))
        if moving_platform_positions:
            parts.append("mp:" + ','.join(f"({r},{c})" for r,c in moving_platform_positions))

        result = " | ".join(parts)

        if not result.strip():
            result = "grid: no_features"
        if len(result) > 1900:
            result = result[:1900] + "..."

        return result

    except Exception as e:
        return f"grid: error_{str(e)[:60]}"