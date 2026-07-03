import json
import ast
from collections import defaultdict

def perceive(observation_history: list[str]) -> str:
    """
    Parse the current raw observation from the last element of observation_history.
    Output a concise (<2000 char) summary of decision-relevant features.
    Includes step count, agent, blue phase, gray, red, and explicit rules with ordering.
    Never raises, never returns empty.
    """
    if not observation_history:
        return "empty_history"

    obs = observation_history[-1]

    # ----- Helper: palette for integer grids -----
    int_to_name = {
        0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
        5: "gray", 6: "magenta", 7: "orange", 8: "lightblue",
        9: "maroon", 10: "white", 11: "gray"
    }

    # ----- Extract step number (if present) -----
    step = None
    for line in obs.splitlines():
        line = line.strip()
        if line.startswith("Step:"):
            try:
                step = int(line.split(":")[1].strip())
            except:
                pass
            break
        if line.startswith("Action count:"):
            try:
                step = int(line.split(":")[1].strip())
            except:
                pass
            break
    step_str = f"step:{step}" if step is not None else "step:unknown"

    # ----- Core state extraction -----
    def extract_state(grid):
        rows = len(grid)
        cols = len(grid[0]) if rows else 0

        def cell_name(cell):
            if isinstance(cell, int):
                return int_to_name.get(cell, f"unknown_{cell}")
            return cell.lower()

        agent_pos = None
        blue_positions = []
        gray_positions = []
        red_positions = []

        for r in range(rows):
            row = grid[r]
            if len(row) != cols:
                continue
            for c in range(cols):
                colour = cell_name(row[c])
                if colour == "orange":
                    agent_pos = (r, c)
                elif colour == "blue":
                    blue_positions.append((r, c))
                elif colour == "gray":
                    gray_positions.append((r, c))
                elif colour == "red":
                    red_positions.append((r, c))

        # Blue phase from first blue cell
        blue_phase = -1
        if blue_positions:
            sample_col = blue_positions[0][1]
            if sample_col % 3 == 1:
                blue_phase = 0
            elif sample_col % 3 == 2:
                blue_phase = 1

        at_bottom = False
        if agent_pos is not None:
            at_bottom = (agent_pos[0] == rows - 1)

        # ----- Explicit rules with ordering -----
        # Order: automatic updates (row+1, red-1, bluePhaseFlip) happen first,
        # then the action is applied.
        rules = (
            f"rules:grid{rows}x{cols};"
            "order=autoThenAction;"
            "auto:row+1,red-1-existing,blueFlip;"
            "left/right:grayCol-/+1;"
            "up:wrap+gray2red(noRedDecrement);"
            "noop:noEffect"
        )

        parts = [rules, step_str]

        # Agent
        if agent_pos is not None:
            parts.append(f"agent:({agent_pos[0]},{agent_pos[1]})")
        else:
            parts.append("agent:none")

        # Blue phase
        if blue_phase != -1:
            parts.append(f"bluePhase:{blue_phase}")
        else:
            parts.append("bluePhase:none")

        # Gray
        if len(gray_positions) == 1:
            g = gray_positions[0]
            parts.append(f"gray:({g[0]},{g[1]})")
        elif len(gray_positions) > 1:
            coords = ",".join(f"({r},{c})" for r, c in gray_positions)
            parts.append(f"gray:{coords}")
        else:
            parts.append("gray:none")

        # Red
        if len(red_positions) == 1:
            rp = red_positions[0]
            parts.append(f"red:({rp[0]},{rp[1]})")
        elif len(red_positions) > 1:
            coords = ",".join(f"({r},{c})" for r, c in red_positions)
            parts.append(f"red:{coords}")
        else:
            parts.append("red:none")

        parts.append(f"atBottom:{'true' if at_bottom else 'false'}")

        return "; ".join(parts)

    # ----- Try Autumn JSON grid (2D array of strings) -----
    try:
        start = obs.find("[[")
        end = obs.rfind("]]") + 2
        if start != -1 and end > start:
            grid_str = obs[start:end]
            grid = json.loads(grid_str)
            if grid and len(grid) > 0 and len(grid[0]) > 0:
                result = extract_state(grid)
                if result.strip():
                    return result
    except Exception:
        pass

    # ----- Try ARC integer grid -----
    try:
        lines = obs.splitlines()
        grid_lines = []
        in_grid = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("<grid_"):
                in_grid = True
                continue
            if in_grid:
                if stripped.startswith("<grid_") or stripped.startswith("====="):
                    break
                if stripped.startswith("[") and stripped.endswith("]"):
                    grid_lines.append(stripped)
        if grid_lines:
            grid = []
            for line in grid_lines:
                try:
                    row = ast.literal_eval(line)
                    if isinstance(row, list):
                        grid.append(row)
                except Exception:
                    continue
            if grid:
                result = extract_state(grid)
                if result.strip():
                    return result
    except Exception:
        pass

    # ----- Fallback (should never trigger if observation is valid) -----
    return f"parse_fallback; {step_str}; grid_not_found"