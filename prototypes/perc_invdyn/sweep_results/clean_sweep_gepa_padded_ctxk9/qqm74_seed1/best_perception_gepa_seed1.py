import json
import re
from typing import List, Tuple, Any, Optional

def perceive(observation_history: list[str]) -> str:
    """
    Extract decision‑relevant features from the raw observation.
    Returns a concise (<2000 char) text summary that changes when the world
    changes, so the action taken between two consecutive states can be recovered.
    """
    raw = observation_history[-1]  # current observation

    try:
        # 1. Extract step number
        step = ""
        m = re.search(r'Step:\s*(\d+)', raw)
        if m:
            step = f"Step {m.group(1)},"

        # 2. Extract available actions (short version)
        actions_str = ""
        actions_start = raw.find("Available actions now:")
        if actions_start >= 0:
            actions_end = raw.find("========== Start of Direct Observation ==========")
            if actions_end < 0:
                actions_end = len(raw)
            actions_block = raw[actions_start:actions_end]
            # Collect short names of actions (skip noop, quit, reset)
            short_actions = []
            for line in actions_block.split('\n'):
                line = line.strip()
                if line.startswith('- '):
                    act = line[2:].strip()
                    if act == 'noop' or act == 'quit' or act == 'reset':
                        continue
                    # Extract the first word (left/right/up/down/click)
                    first_word = act.split()[0]
                    if first_word:
                        short_actions.append(first_word[:2])  # e.g. "le", "ri", "up", "do", "cl"
            if short_actions:
                actions_str = f" acts={','.join(short_actions)}"

        # 3. Parse grid – two possible encodings
        grid = None
        grid_type = None  # 'int' or 'string'

        # Try Autumn JSON format first (look for "[[" as start of 2D array)
        # We need the LAST occurrence of "[[...]]" because there might be multiple grid blocks
        start_candidates = [m.start() for m in re.finditer(r'\[\[', raw)]
        if start_candidates:
            # Find the last complete JSON array
            for start in reversed(start_candidates):
                try:
                    end = raw.rindex("]]", start) + 2
                    if end > start:
                        candidate = raw[start:end]
                        # Quick check: should start with [[ and end with ]]
                        if candidate.startswith('[[') and candidate.endswith(']]'):
                            parsed = json.loads(candidate)
                            if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], list):
                                grid = parsed
                                grid_type = 'string'
                                break
                except (ValueError, json.JSONDecodeError):
                    continue

        if grid is None:
            # Try ARC integer grid – take the LAST <grid_k> block
            marker_matches = list(re.finditer(r'<grid_\d+>', raw))
            if marker_matches:
                m_grid = marker_matches[-1]  # last marker
                after_marker = raw[m_grid.end():].strip()
                lines = after_marker.splitlines()
                grid = []
                for line in lines:
                    line = line.strip()
                    if line.startswith('[') and line.endswith(']'):
                        try:
                            row = json.loads(line)
                            if isinstance(row, list) and len(row) > 0:
                                grid.append(row)
                        except:
                            continue
                    elif line == '' or line.startswith('<'):
                        break
                if grid:
                    grid_type = 'int'

        if grid is None or len(grid) == 0 or len(grid[0]) == 0:
            return f"{step} (no grid){actions_str}"

        # 4. Determine background colour and collect non-background cells
        is_int_grid = (grid_type == 'int')
        bg_values = {0} if is_int_grid else {"black"}

        non_bg: List[Tuple[int, int, Any]] = []
        for r in range(len(grid)):
            row = grid[r]
            for c in range(len(row)):
                val = row[c]
                if is_int_grid:
                    if val not in bg_values:
                        non_bg.append((r, c, val))
                else:
                    if val not in bg_values:
                        non_bg.append((r, c, val))

        # 5. Compute drift from previous observation
        drift_str = ""
        if len(observation_history) >= 2:
            prev_raw = observation_history[-2]
            prev_grid = None
            # Parse previous grid similarly (same format)
            # Try JSON first
            start_candidates_prev = [m.start() for m in re.finditer(r'\[\[', prev_raw)]
            if start_candidates_prev:
                for start in reversed(start_candidates_prev):
                    try:
                        end = prev_raw.rindex("]]", start) + 2
                        if end > start:
                            candidate = prev_raw[start:end]
                            if candidate.startswith('[[') and candidate.endswith(']]'):
                                parsed = json.loads(candidate)
                                if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], list):
                                    prev_grid = parsed
                                    break
                    except:
                        continue

            if prev_grid is None:
                # Try integer grid from previous
                marker_matches_prev = list(re.finditer(r'<grid_\d+>', prev_raw))
                if marker_matches_prev:
                    m_prev = marker_matches_prev[-1]
                    after_marker = prev_raw[m_prev.end():].strip()
                    lines = after_marker.splitlines()
                    prev_grid = []
                    for line in lines:
                        line = line.strip()
                        if line.startswith('[') and line.endswith(']'):
                            try:
                                row = json.loads(line)
                                if isinstance(row, list) and len(row) > 0:
                                    prev_grid.append(row)
                            except:
                                continue
                        elif line == '' or line.startswith('<'):
                            break

            if prev_grid and len(prev_grid) > 0 and len(prev_grid[0]) > 0:
                # Collect previous non-bg cells (same background assumption)
                prev_non_bg = []
                for r in range(len(prev_grid)):
                    for c in range(len(prev_grid[r])):
                        val = prev_grid[r][c]
                        if is_int_grid:
                            if val not in bg_values:
                                prev_non_bg.append((r, c, val))
                        else:
                            if val not in bg_values:
                                prev_non_bg.append((r, c, val))

                # Compute displacement: for each current cell, find closest previous cell of same colour
                displacements = []
                for r, c, colour in non_bg:
                    best_dist = 999
                    best_disp = None
                    for pr, pc, pcolour in prev_non_bg:
                        if colour == pcolour:
                            dist = abs(r - pr) + abs(c - pc)
                            if dist < best_dist and dist <= 2:  # movement is small per step
                                best_dist = dist
                                best_disp = (r - pr, c - pc)
                    if best_disp is not None:
                        displacements.append(best_disp)

                # Take the most common displacement as drift
                if displacements:
                    from collections import Counter
                    disp_counts = Counter(displacements)
                    most_common = disp_counts.most_common(1)
                    if most_common:
                        dr, dc = most_common[0][0]
                        drift_str = f" drift=({dr},{dc})"

        # 6. Build cell summary and bounding box
        non_bg.sort(key=lambda x: (x[0], x[1]))
        num_cells = len(non_bg)
        bbox_str = ""
        if num_cells > 0:
            min_r = min(c[0] for c in non_bg)
            max_r = max(c[0] for c in non_bg)
            min_c = min(c[1] for c in non_bg)
            max_c = max(c[1] for c in non_bg)
            bbox_str = f" bbox=[{min_r},{min_c}]-[{max_r},{max_c}]"

        # 7. Edge cells (optional, but keep concise)
        edge_str = ""
        if num_cells > 0:
            h = len(grid)
            w = len(grid[0])
            at_edge = []
            for r, c, colour in non_bg:
                if r == 0 or r == h-1 or c == 0 or c == w-1:
                    at_edge.append(f"({r},{c})")
            if at_edge:
                edge_str = f" edge={','.join(at_edge[:3])}"  # limit to 3

        # 8. Build cell positions string (trim to save space)
        cell_parts = []
        for r, c, colour in non_bg:
            if is_int_grid:
                cell_parts.append(f"({r},{c},{colour})")
            else:
                cell_parts.append(f"({r},{c},{colour})")

        # 9. Assemble summary
        summary_parts = [step]
        summary_parts.append(f" cells={num_cells}")
        summary_parts.append(bbox_str)
        summary_parts.append(drift_str)
        summary_parts.append(edge_str)
        summary_parts.append(actions_str)

        # Add up to 15 cell positions, then indicate remainder
        if cell_parts:
            cell_str = ' '.join(cell_parts)
            if len(cell_parts) > 15:
                cell_str = ' '.join(cell_parts[:15]) + f" ...({len(cell_parts)-15} more)"
            # Ensure total <2000
            if len(cell_str) > 1500:
                cell_str = ' '.join(cell_parts[:5]) + f" ...({len(cell_parts)-5} more)"
            summary_parts.append(f" pos=[{cell_str}]")

        summary = ''.join(summary_parts)

        # Final length check: if still >1900, strip cell positions
        if len(summary) > 1900:
            # Keep only essential info
            summary = f"{step} cells={num_cells}{bbox_str}{drift_str}{edge_str}{actions_str}"
            if len(summary) > 1900:
                # Absolute minimal
                summary = f"{step} cells={num_cells}{drift_str}{actions_str}"

        return summary

    except Exception:
        # Never raise, never return empty
        return "(parse error)"