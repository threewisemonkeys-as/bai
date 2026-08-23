import json
import re

def perceive(observation_history: list[str]) -> str:
    """
    Produce a concise text summary of decision-relevant features from the raw observation.
    The summary is <2000 characters, never raises, never empty.
    """
    try:
        obs = observation_history[-1]

        # ---- Extract metadata: step, action count ----
        step = -1
        action_count = -1
        m = re.search(r'Step:\s*(\d+)', obs)
        if m:
            step = int(m.group(1))
        m = re.search(r'Action count:\s*(\d+)', obs)
        if m:
            action_count = int(m.group(1))

        # ---- Parse grid (try JSON first, then ARC integer) ----
        grid = None

        # JSON (Autumn) format
        try:
            start = obs.find('[[[')
            if start == -1:
                start = obs.find('[[')
            if start != -1:
                depth = 0
                end = start
                for i in range(start, len(obs)):
                    if obs[i] == '[':
                        depth += 1
                    elif obs[i] == ']':
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                json_str = obs[start:end]
                parsed = json.loads(json_str)
                if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], list):
                    grid = parsed
        except Exception:
            pass

        # ARC integer format
        if grid is None:
            try:
                blocks = re.findall(r'<grid_\d+>\n(.*?)(?:\n\n|\Z)', obs, re.DOTALL)
                if blocks:
                    raw = blocks[-1]  # last grid block
                    rows = []
                    for line in raw.strip().split('\n'):
                        line = line.strip()
                        if line.startswith('[') and line.endswith(']'):
                            nums = [int(x.strip()) for x in line[1:-1].split(',')]
                            rows.append(nums)
                    if rows:
                        grid = rows
            except Exception:
                pass

        if not grid or len(grid) == 0 or len(grid[0]) == 0:
            return f"step:{step};ac:{action_count};state:empty"

        h = len(grid)
        w = len(grid[0])

        # ---- Normalise to integer grid ----
        colour_name_to_int = {
            "black": 0, "blue": 1, "red": 2, "green": 3, "yellow": 4,
            "gray": 5, "magenta": 6, "orange": 7, "lightblue": 8,
            "maroon": 9, "darkred": 9, "white": 10, "darkgray": 11,
            "gold": 12, "light-gray": 5, "light_gray": 5,
            "darkgreen": 3, "dark-blue": 1, "dark_blue": 1,
            "sky-blue": 8, "sky_blue": 8
        }
        if isinstance(grid[0][0], str):
            int_grid = []
            for row in grid:
                int_row = [colour_name_to_int.get(c.lower(), 0) for c in row]
                int_grid.append(int_row)
            grid = int_grid

        # ---- Identify phase: look for lightblue (8) or gold (12) ----
        has_lightblue = any(v == 8 for row in grid for v in row)
        has_gold = any(v == 12 for row in grid for v in row)
        if has_lightblue:
            phase = "lightblue"
        elif has_gold:
            phase = "gold"
        else:
            phase = "unknown"

        # ---- Find agent (blue = 1) ----
        agent_positions = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == 1]
        if not agent_positions:
            # Fallback: look for an isolated coloured cell in black area
            for r in range(h):
                for c in range(w):
                    if grid[r][c] != 0:
                        isolated = True
                        for dr in (-1, 0, 1):
                            for dc in (-1, 0, 1):
                                if dr == 0 and dc == 0:
                                    continue
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < h and 0 <= nc < w:
                                    if grid[nr][nc] != 0 and grid[nr][nc] != grid[r][c]:
                                        isolated = False
                                        break
                            if not isolated:
                                break
                        if isolated:
                            agent_positions = [(r, c)]
                            break
                if agent_positions:
                    break

        if not agent_positions:
            agent = (0, 0)
        else:
            agent = agent_positions[0]

        _, agent_col = agent

        # ---- On-gold flag ----
        on_gold = (grid[agent[0]][agent[1]] == 12)

        # ---- Build list of all non-background, non-agent cells ----
        # Each entry: (movement_type, r, c) where movement_type is a string key
        cell_entries = []

        for r in range(h):
            for c in range(w):
                v = grid[r][c]
                if v == 0 or v == 1:   # skip black and agent
                    continue
                # Determine colour name for output
                # Map integer to colour name used in keys
                if v == 5:
                    colour_name = "gray"
                elif v == 8:
                    colour_name = "lightblue"
                elif v == 12:
                    colour_name = "gold"
                elif v == 2:
                    colour_name = "red"
                elif v == 3:
                    colour_name = "green"
                elif v == 4:
                    colour_name = "yellow"
                elif v == 6:
                    colour_name = "magenta"
                elif v == 7:
                    colour_name = "orange"
                elif v == 9:
                    colour_name = "maroon"
                elif v == 10:
                    colour_name = "white"
                elif v == 11:
                    colour_name = "darkgray"
                else:
                    colour_name = f"c{v}"

                # Determine whether this cell is static or moving
                if phase == "gold":
                    if colour_name == "gold":
                        mtype = "static_gold"
                    elif colour_name == "gray":
                        mtype = "moving_gray"
                    else:
                        mtype = "static_" + colour_name
                elif phase == "lightblue":
                    if colour_name == "gray":
                        mtype = "static_gray"
                    elif colour_name == "lightblue":
                        mtype = "moving_lightblue"
                    else:
                        mtype = "static_" + colour_name
                else:
                    mtype = "static_" + colour_name

                cell_entries.append((mtype, r, c))

        # Sort entries by (r, c) to enforce row-major order across all types
        cell_entries.sort(key=lambda x: (x[1], x[2]))

        # ---- Group by movement_type, preserving order ----
        # Use ordered dict to keep first occurrence order
        from collections import OrderedDict
        groups = OrderedDict()
        for mtype, r, c in cell_entries:
            groups.setdefault(mtype, []).append((r, c))

        # ---- Build output parts ----
        parts = [f"step:{step}", f"ac:{action_count}", f"phase:{phase}",
                 f"agent:{agent[0]},{agent[1]}"]
        if on_gold:
            parts.append("ongold")

        # Output groups in a fixed desired order (only those present)
        desired_order = ["static_gold", "static_gray", "moving_gray", "moving_lightblue"]
        for key in desired_order:
            if key in groups:
                # Cells are already sorted by (r,c) from the global sort
                cells_str = ";".join(f"{r},{c}" for r, c in groups[key])
                # Limit to 200 cells per group to keep output short
                if len(cells_str) > 1200:
                    cells_str = cells_str[:1200] + "..."
                parts.append(f"{key}:{cells_str}")
        # Any remaining keys (unknown static colours) appended at end
        for key, cells in groups.items():
            if key not in desired_order:
                cells_str = ";".join(f"{r},{c}" for r, c in cells)
                if len(cells_str) > 400:
                    cells_str = cells_str[:400] + "..."
                parts.append(f"{key}:{cells_str}")

        # ---- Include teleport history for forward prediction ----
        # Only if previous state exists and a teleport likely occurred
        if len(observation_history) >= 2:
            prev_obs = observation_history[-2]
            # Try to extract previous step
            prev_step = -1
            m = re.search(r'Step:\s*(\d+)', prev_obs)
            if m:
                prev_step = int(m.group(1))
            # Extract previous agent position
            prev_grid = None
            try:
                start = prev_obs.find('[[[')
                if start == -1:
                    start = prev_obs.find('[[')
                if start != -1:
                    depth = 0
                    end = start
                    for i in range(start, len(prev_obs)):
                        if prev_obs[i] == '[':
                            depth += 1
                        elif prev_obs[i] == ']':
                            depth -= 1
                            if depth == 0:
                                end = i + 1
                                break
                    json_str = prev_obs[start:end]
                    parsed = json.loads(json_str)
                    if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], list):
                        prev_grid = parsed
            except Exception:
                pass
            if prev_grid is None:
                try:
                    blocks = re.findall(r'<grid_\d+>\n(.*?)(?:\n\n|\Z)', prev_obs, re.DOTALL)
                    if blocks:
                        raw = blocks[-1]
                        rows = []
                        for line in raw.strip().split('\n'):
                            line = line.strip()
                            if line.startswith('[') and line.endswith(']'):
                                nums = [int(x.strip()) for x in line[1:-1].split(',')]
                                rows.append(nums)
                        if rows:
                            prev_grid = rows
                except Exception:
                    pass
            if prev_grid:
                if isinstance(prev_grid[0][0], str):
                    prev_int_grid = []
                    for row in prev_grid:
                        prev_int_row = [colour_name_to_int.get(c.lower(), 0) for c in row]
                        prev_int_grid.append(prev_int_row)
                    prev_grid = prev_int_grid
                prev_h = len(prev_grid)
                prev_w = len(prev_grid[0])
                prev_agents = [(r, c) for r in range(prev_h) for c in range(prev_w) if prev_grid[r][c] == 1]
                if prev_agents:
                    prev_agent = prev_agents[0]
                    # Detect teleport: agent row changed drastically
                    row_diff = agent[0] - prev_agent[0]
                    if row_diff < -1 or row_diff > 1:
                        parts.append(f"tel:{prev_agent[0]},{prev_agent[1]}->{agent[0]},{agent[1]}")
                    # Also column drift (left/right)
                    if agent[0] == prev_agent[0] and abs(agent[1] - prev_agent[1]) == 1:
                        parts.append(f"coldrift:{prev_agent[1]}->{agent[1]}")

        # ---- Assemble final string ----
        result = ";".join(parts)
        # Ensure not too long
        if len(result) > 1900:
            # Truncate but keep essential parts
            result = result[:1900] + "..."

        return result

    except Exception:
        return "state:error"