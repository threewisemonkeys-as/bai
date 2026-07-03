import json
import ast
import re
from typing import List, Tuple, Optional

def perceive(observation_history: list[str]) -> str:
    """
    Parse the current raw observation from observation_history[-1] and return a
    concise summary of decision‑relevant features as a string.
    The output always contains information that changes when the world changes,
    and includes the step number and action count to disambiguate noop from
    blocked moves. The summary is never empty and never raises an exception.
    """
    # Fallback constant
    FALLBACK_BASE = "grid unknown"

    if not observation_history:
        return FALLBACK_BASE

    obs_current = observation_history[-1]
    obs_previous = observation_history[-2] if len(observation_history) >= 2 else None

    # ------------------------------------------------------------------------
    # Helper: convert integer colour code to canonical colour name
    # ------------------------------------------------------------------------
    INT_TO_COLOR = {
        0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
        5: "light-gray", 6: "magenta", 7: "orange", 8: "light-blue",
        9: "dark-red", 10: "white", 11: "dark-gray",
    }

    def color_name(val):
        if isinstance(val, int):
            return INT_TO_COLOR.get(val, f"color{val}")
        return str(val)

    # ------------------------------------------------------------------------
    # Helper: extract non‑background cells from a parsed grid
    # ------------------------------------------------------------------------
    def extract_features(grid):
        """Return sorted list of (colour, row, col) for non‑black cells."""
        features = []
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                # treat black/0 as background
                if isinstance(cell, int):
                    is_bg = cell == 0
                else:
                    is_bg = cell == "black"
                if not is_bg:
                    features.append((color_name(cell), r, c))
        # sort by colour, then row, then column
        features.sort(key=lambda x: (x[0], x[1], x[2]))
        return features, height, width

    # ------------------------------------------------------------------------
    # Helper: parse a grid from a raw observation string (either ARC or Autumn)
    # ------------------------------------------------------------------------
    def parse_grid(obs: str):
        grid = None
        # Try ARC integer grid format (lines starting with bracketed int list)
        if "<grid_" in obs:
            lines = obs.splitlines()
            rows = []
            in_grid = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("<grid_"):
                    in_grid = True
                    continue
                if in_grid:
                    if stripped.startswith("["):
                        try:
                            row = ast.literal_eval(stripped)
                            if isinstance(row, list):
                                rows.append(row)
                        except Exception:
                            pass
                    else:
                        if rows:
                            break
            if rows:
                grid = rows
        else:
            # Try Autumn JSON format (a single [[...]] JSON array)
            start = obs.find("[[")
            end = obs.rfind("]]")
            if start != -1 and end != -1:
                json_str = obs[start:end+2]
                try:
                    parsed = json.loads(json_str)
                    if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], list):
                        grid = parsed
                except Exception:
                    pass
        return grid

    # ------------------------------------------------------------------------
    # Extract step number and action count from the observation text
    # ------------------------------------------------------------------------
    def extract_metadata(obs: str):
        step = None
        action_count = None
        # step: "Step: <number>"
        m = re.search(r"Step:\s*(\d+)", obs, re.IGNORECASE)
        if m:
            step = int(m.group(1))
        # action count: "Action count: <number>"  (also "Actions: <number>" may appear)
        m = re.search(r"Action\s*count:\s*(\d+)", obs, re.IGNORECASE)
        if m:
            action_count = int(m.group(1))
        else:
            # Alternative: "Actions: <number>"
            m = re.search(r"Actions:\s*(\d+)", obs, re.IGNORECASE)
            if m:
                action_count = int(m.group(1))
        return step, action_count

    # ------------------------------------------------------------------------
    # Main parsing and summary construction
    # ------------------------------------------------------------------------
    try:
        grid_current = parse_grid(obs_current)
        step, action_count = extract_metadata(obs_current)

        if grid_current is None:
            # Cannot parse grid – return minimal metadata if available
            parts = []
            if step is not None:
                parts.append(f"step:{step}")
            if action_count is not None:
                parts.append(f"action_count:{action_count}")
            return FALLBACK_BASE + (" | " + " | ".join(parts) if parts else "")

        features_current, height, width = extract_features(grid_current)

        # Compute changed flag
        changed = 0
        if obs_previous is not None:
            try:
                grid_prev = parse_grid(obs_previous)
                if grid_prev is not None:
                    features_prev, _, _ = extract_features(grid_prev)
                    if features_prev != features_current:
                        changed = 1
            except Exception:
                pass

        # Build base parts
        parts = []
        # grid size
        if height > 0 and width > 0:
            parts.append(f"grid {height}x{width}")
        else:
            parts.append(FALLBACK_BASE)

        if step is not None:
            parts.append(f"step:{step}")
        if action_count is not None:
            parts.append(f"action_count:{action_count}")

        parts.append(f"changed:{changed}")

        # Feature string (limit to 2000 total)
        if len(features_current) > 100:
            # Aggregate by colour counts and bounding box
            color_counts = {}
            min_r, max_r = height, 0
            min_c, max_c = width, 0
            for col, r, c in features_current:
                color_counts[col] = color_counts.get(col, 0) + 1
                if r < min_r: min_r = r
                if r > max_r: max_r = r
                if c < min_c: min_c = c
                if c > max_c: max_c = c
            count_str = "; ".join(f"{col}:{cnt}" for col, cnt in sorted(color_counts.items()))
            bbox_str = f"bbox ({min_r},{min_c})-({max_r},{max_c})"
            feat_strs = [count_str, bbox_str]
        else:
            feat_strs = [f"{col} at ({r},{c})" for col, r, c in features_current]

        # Join all parts
        result = " | ".join(parts) + " | " + str(feat_strs)

        # Truncate if needed (keep important metadata)
        if len(result) > 2000:
            # Only keep first 20 features
            feat_strs = feat_strs[:20] if isinstance(feat_strs, list) else [feat_strs]
            # Rebuild with shortened feature list
            result = " | ".join(parts) + " | " + str(feat_strs)
            # If still too long, drop features entirely
            if len(result) > 2000:
                result = " | ".join(parts)

        return result

    except Exception:
        # Ultimate fallback: never raise, never empty
        parts = []
        step, action_count = extract_metadata(obs_current)
        parts.append(FALLBACK_BASE)
        if step is not None:
            parts.append(f"step:{step}")
        if action_count is not None:
            parts.append(f"action_count:{action_count}")
        # At least something
        if not parts:
            parts.append("unknown")
        return " | ".join(parts)