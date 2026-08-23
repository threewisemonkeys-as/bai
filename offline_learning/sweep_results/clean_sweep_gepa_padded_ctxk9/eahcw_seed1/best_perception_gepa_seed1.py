import json

def perceive(observation_history: list[str]) -> str:
    """
    Produce a concise (<2000 char) summary of decision‑relevant features.
    The output must change when the world changes and allow the action between
    two consecutive states to be recovered.
    """
    obs = observation_history[-1]

    # ---------- Helper: parse grid from raw observation ----------
    def parse_grid(text: str):
        """Try Autumn JSON format then ARC integer format. Return (grid, step) or (None, None)."""
        step = None
        # extract step number (common to both formats)
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("Step:") and "Step:" in line:
                try:
                    step = int(line.split(":")[1].strip())
                except:
                    pass
            if line.startswith("Action count:") and "Action count:" in line:
                try:
                    step = int(line.split(":")[1].strip())
                except:
                    pass
            if step is not None:
                break

        # Try Autumn JSON
        try:
            s = text.find("[[")
            e = text.rfind("]]")
            if s != -1 and e != -1 and e > s:
                json_str = text[s:e+2]
                parsed = json.loads(json_str)
                if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], list):
                    return parsed, step
        except:
            pass

        # Try ARC integer
        try:
            marker = "<grid_0>"
            idx = text.find(marker)
            if idx != -1:
                rest = text[idx + len(marker):]
                lines = rest.split('\n')
                rows = []
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith('[') and stripped.endswith(']'):
                        inner = stripped[1:-1]
                        if inner.strip() == '':
                            row = []
                        else:
                            row = [int(x.strip()) for x in inner.split(',')]
                        rows.append(row)
                if rows:
                    return rows, step
        except:
            pass
        return None, step

    # ---------- Parse current grid ----------
    grid, step = parse_grid(obs)
    if grid is None:
        # fallback
        return "cannot_parse_grid"

    n_rows = len(grid)
    n_cols = len(grid[0]) if n_rows > 0 else 0

    # ---------- Palette ----------
    palette = {
        0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
        5: "light_gray", 6: "magenta", 7: "orange", 8: "light_blue",
        9: "maroon", 10: "white", 11: "dark_gray"
    }

    # ---------- Collect non‑black cells and find agent ----------
    non_bg = []
    agent_pos = None
    for r in range(n_rows):
        row = grid[r]
        for c in range(len(row)):
            val = row[c]
            is_black = False
            if isinstance(val, int):
                if val == 0:
                    is_black = True
                colour_str = palette.get(val, f"code_{val}")
                if val == 1:
                    agent_pos = (r, c)
            else:
                if val == "black":
                    is_black = True
                colour_str = val
                if val == "blue":
                    agent_pos = (r, c)
            if not is_black:
                non_bg.append((r, c, colour_str))

    # ---------- Compute click history from observation_history ----------
    click_count = 0
    steps_since_last_click = -1   # -1 means unknown
    last_click_pos = None
    last_click_color = None

    if len(observation_history) >= 2:
        # Parse all previous observations to find clicks
        prev_grids = []
        for i in range(len(observation_history) - 1):
            g, _ = parse_grid(observation_history[i])
            if g is None:
                prev_grids.append(None)
            else:
                prev_grids.append(g)

        # Compare current with previous to count clicks and find last
        # We'll scan backwards to find the most recent change (click)
        current_nonblack_set = set((r,c,col) for r,c,col in non_bg)
        current_nonblack_positions = set((r,c) for r,c,_ in non_bg)

        for i in range(len(prev_grids)-1, -1, -1):
            prev = prev_grids[i]
            if prev is None:
                continue
            # Build previous nonblack set
            prev_nonblack = set()
            prev_nonblack_pos = set()
            for r in range(len(prev)):
                row = prev[r]
                for c in range(len(row)):
                    val = row[c]
                    is_black = False
                    if isinstance(val, int):
                        if val == 0:
                            is_black = True
                        col_str = palette.get(val, f"code_{val}")
                    else:
                        if val == "black":
                            is_black = True
                        col_str = val
                    if not is_black:
                        prev_nonblack.add((r,c,col_str))
                        prev_nonblack_pos.add((r,c))
            # Detect click: new nonblack cell appeared (not present before)
            new_cells = current_nonblack_set - prev_nonblack
            if new_cells:
                click_count = len(set((r,c) for r,c,_ in new_cells))   # number of cells added
                # take the first (usually only one)
                first_new = list(new_cells)[0]
                last_click_pos = (first_new[0], first_new[1])
                last_click_color = first_new[2]
                # steps_since_last_click = number of steps between that click and current
                steps_since_last_click = (len(observation_history) - 1) - i - 1   # steps after the click (excluding the click step itself)
                # Actually, we want steps since the click action (the step that changed the state).
                # If the click happened between state[i] and state[i+1], then current state is after click.
                # Steps since last click = (current_index - (i+1)) because click was applied to produce state[i+1].
                # current index is len(observation_history)-1. So steps = (len-1) - (i+1) = len - i - 2.
                # But let's use a simpler approach: count how many consecutive states after the click have the same grid as current.
                # Since we are scanning backwards, and we found a change, then the steps since last click is the number of times we
                # moved forward from that point (which is (len(obs_history)-1) - i). But careful: if the click changed the grid,
                # then the state at index i+1 has the new grid. So the click occurred between i and i+1.
                # The number of steps (actions) after that click up to but not including the current? We want the number of actions
                # taken since the click (including the click itself? No, steps_since_last_non_noop is the number of noop/movement
                # actions after the click). So steps = (current_index - (i+1)). Since current index is last, steps = (len-1)-(i+1) = len-i-2.
                steps_since_last_click = len(observation_history) - i - 2
                break
            # If no new cells, continue scanning
        # If never found a change, steps_since_last_click remains -1

    # ---------- Build summary ----------
    dims = f"grid={n_rows}x{n_cols}"
    if agent_pos is not None:
        agent_str = f"agent=({agent_pos[0]},{agent_pos[1]})"
    else:
        agent_str = "agent=unknown"

    # Step number (if available)
    step_str = f"step={step}" if step is not None else ""

    # Click metadata
    click_str = f"click_cnt={click_count}"
    if last_click_pos is not None:
        click_str += f" last_click=({last_click_pos[0]},{last_click_pos[1]},{last_click_color})"
    if steps_since_last_click >= 0:
        click_str += f" steps_since_click={steps_since_last_click}"

    # Non‑black objects (limit length)
    if non_bg:
        obj_list = [f"({r},{c},{col})" for r,c,col in non_bg]
        obj_str = "nonblack:[" + ",".join(obj_list) + "]"
        if len(obj_str) > 1800 - len(dims + agent_str + step_str + click_str):
            # truncate to fit under 2000 total
            allowed = 1800 - len(dims + agent_str + step_str + click_str)
            if allowed < 20:
                obj_str = "nonblack:[]"
            else:
                # estimate how many we can fit
                n = 0
                total = 0
                for item in obj_list:
                    total += len(item) + 1  # +1 for comma
                    if total > allowed - 20:
                        break
                    n += 1
                obj_str = "nonblack:[" + ",".join(obj_list[:n]) + ",...]"
    else:
        obj_str = "grid_all_black"

    # Assemble final string
    parts = [dims, agent_str, step_str, click_str, obj_str]
    final = " ".join(p for p in parts if p)
    if len(final) > 1990:
        # drastic truncation: keep only essential
        final = f"{dims} {agent_str} step={step} click_cnt={click_count} nonblack:[...]"
    return final