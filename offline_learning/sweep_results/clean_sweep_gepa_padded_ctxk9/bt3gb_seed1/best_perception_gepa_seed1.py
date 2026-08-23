import json
import re

def perceive(observation_history: list[str]) -> str:
    """
    Parse the raw observation text and return a concise summary of
    decision-relevant features.  The output must be non-empty, <2000 chars,
    and change when the world changes so that the action between two
    consecutive states can be recovered.
    """
    obs = observation_history[-1]
    
    try:
        # --- Detect encoding and parse grid ---
        # Check for Autumn string grid (JSON 2D array of colour names)
        if obs.lstrip().startswith('[[') or '[' in obs[:100]:
            # Try to find a JSON 2D array
            start = obs.find('[[')
            if start == -1:
                # Try alternate approach: look for pattern like [["...
                match = re.search(r'\[\[.*?\]\]', obs, re.DOTALL)
                if match:
                    start = match.start()
                else:
                    # Fallback: search for first '[' that starts a nested list
                    for i, ch in enumerate(obs):
                        if ch == '[' and i+1 < len(obs) and obs[i+1] == '[':
                            start = i
                            break
                    else:
                        start = -1
            
            if start != -1:
                # Find matching close
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
                
                grid_str = obs[start:end]
                try:
                    grid = json.loads(grid_str)
                except json.JSONDecodeError:
                    # Try to fix common issues: trailing commas, single quotes
                    grid_str_clean = grid_str.replace("'", '"')
                    grid_str_clean = re.sub(r',\s*\]', ']', grid_str_clean)
                    try:
                        grid = json.loads(grid_str_clean)
                    except json.JSONDecodeError:
                        # Last resort: manual parse
                        grid = _manual_parse_colour_grid(obs)
                        if grid is None:
                            return "grid_parse_error"
                
                if not grid or not grid[0]:
                    return "empty_grid"
                
                # Build summary: position of non-black/non-background cells
                # Find most common colour = background
                colour_counts = {}
                for row in grid:
                    for cell in row:
                        colour_counts[cell] = colour_counts.get(cell, 0) + 1
                
                bg_colour = max(colour_counts, key=colour_counts.get) if colour_counts else "black"
                
                # Collect non-background cells
                cells = []
                for r, row in enumerate(grid):
                    for c, colour in enumerate(row):
                        if colour != bg_colour:
                            cells.append((r, c, colour))
                
                # Include a few background cells at corners to allow position inference
                # (the background pattern often defines the arena)
                bg_cells = []
                if len(grid) > 1 and len(grid[0]) > 1:
                    # Sample corners and edges
                    for corner_r, corner_c in [(0,0), (0, min(3, len(grid[0])-1)), 
                                                (min(3, len(grid)-1), 0),
                                                (min(3, len(grid)-1), min(3, len(grid[0])-1))]:
                        if len(grid) > corner_r and len(grid[0]) > corner_c:
                            c = grid[corner_r][corner_c]
                            if c == bg_colour:
                                bg_cells.append((corner_r, corner_c, c))
                
                # Also include grid dimensions
                h, w = len(grid), len(grid[0])
                
                result_parts = [f"grid_{h}x{w}_bg_{bg_colour}"]
                
                if bg_cells:
                    result_parts.append("bg:" + ",".join(f"{r},{c}" for r,c,_ in bg_cells[:4]))
                
                if cells:
                    # Sort by row then column for consistency
                    cells.sort()
                    cell_strs = []
                    for r, c, colour in cells[:60]:  # limit to avoid >2000 chars
                        cell_strs.append(f"{r},{c},{colour}")
                    result_parts.append("obj:" + ";".join(cell_strs))
                    if len(cells) > 60:
                        result_parts.append(f"+{len(cells)-60}more")
                else:
                    # If all background, just report dimensions and some sample cells
                    pass  # already have dimensions
                
                return " ".join(result_parts)
        
        # Check for ARC integer grid format
        if '<grid_' in obs or 'Start of Direct Observation' in obs:
            int_grid = _parse_arc_grid(obs)
            if int_grid and len(int_grid) > 0 and len(int_grid[0]) > 0:
                # Build summary
                h, w = len(int_grid), len(int_grid[0])
                
                # Count colours to find background
                colour_counts = {}
                for row in int_grid:
                    for val in row:
                        colour_counts[val] = colour_counts.get(val, 0) + 1
                
                bg_val = max(colour_counts, key=colour_counts.get) if colour_counts else 0
                
                # Collect non-background cells
                cells = []
                for r, row in enumerate(int_grid):
                    for c, val in enumerate(row):
                        if val != bg_val:
                            cells.append((r, c, val))
                
                result_parts = [f"grid_{h}x{w}_bg_{bg_val}"]
                
                if cells:
                    cells.sort()
                    cell_strs = []
                    for r, c, val in cells[:60]:
                        cell_strs.append(f"{r},{c},{val}")
                    result_parts.append("obj:" + ";".join(cell_strs))
                    if len(cells) > 60:
                        result_parts.append(f"+{len(cells)-60}more")
                else:
                    # Sample some corner cells to provide positional reference
                    for r, c in [(0,0), (0, min(3, w-1)), (min(3, h-1), 0)]:
                        if r < h and c < w:
                            result_parts.append(f"c{r},{c}={int_grid[r][c]}")
                
                return " ".join(result_parts)
        
        # Fallback: try to find any grid-like structure in the text
        return _fallback_parse(obs)
        
    except Exception as e:
        # Never raise, never return empty
        return f"parse_error:{str(e)[:100]}"


def _manual_parse_colour_grid(text: str) -> list[list[str]] | None:
    """Manually parse a colour-name grid if JSON parsing fails."""
    try:
        # Try to find the grid between first [[ and last ]]
        start = text.find('[[')
        if start == -1:
            return None
        
        # Extract just the grid portion
        depth = 0
        end = start
        for i in range(start, len(text)):
            if text[i] == '[':
                depth += 1
            elif text[i] == ']':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        
        grid_text = text[start:end]
        
        # Replace single quotes with double quotes
        grid_text = grid_text.replace("'", '"')
        
        # Remove any newlines within the JSON
        grid_text = ' '.join(grid_text.split())
        
        return json.loads(grid_text)
    except:
        return None


def _parse_arc_grid(text: str) -> list[list[int]] | None:
    """Parse an ARC-style integer grid from text."""
    try:
        # Find the marker or start of grid rows
        lines = text.split('\n')
        grid_lines = []
        in_grid = False
        
        for line in lines:
            line = line.strip()
            if '<grid_' in line or 'Start of Direct Observation' in line:
                in_grid = True
                continue
            
            if in_grid:
                # Stop at next marker or empty line after grid
                if not line or line.startswith('<') or line.startswith('==='):
                    if len(grid_lines) > 0:
                        break
                    continue
                
                # Check if line looks like a grid row
                if line.startswith('[') and ']' in line:
                    try:
                        # Parse the list of integers
                        row_str = line.strip()
                        # Remove brackets and split
                        row_str = row_str.strip('[]')
                        # Handle potential whitespace
                        row = [int(x.strip()) for x in row_str.split(',') if x.strip()]
                        if row:
                            grid_lines.append(row)
                    except (ValueError, IndexError):
                        continue
        
        if grid_lines:
            # Normalize row lengths if needed
            max_len = max(len(r) for r in grid_lines)
            for r in grid_lines:
                while len(r) < max_len:
                    r.append(0)  # pad with black
            return grid_lines
        return None
    except:
        return None


def _fallback_parse(text: str) -> str:
    """Last resort: try to find any structured data in the text."""
    try:
        # Try to find grid dimensions from the text
        h_match = re.search(r'(\d+)\s*x\s*(\d+)', text)
        w_match = re.search(r'(\d+)\s*rows?', text, re.IGNORECASE)
        col_match = re.search(r'(\d+)\s*cols?', text, re.IGNORECASE)
        
        parts = ["fallback"]
        if h_match:
            parts.append(f"size_{h_match.group(1)}x{h_match.group(2)}")
        if w_match:
            parts.append(f"rows_{w_match.group(1)}")
        if col_match:
            parts.append(f"cols_{col_match.group(1)}")
        
        # Try to find any non-zero/non-black values
        ints = re.findall(r'\b(\d+)\b', text)
        unique_vals = set(int(x) for x in ints if x.isdigit() and int(x) > 0)
        if unique_vals:
            parts.append(f"vals_{sorted(unique_vals)[:10]}")
        
        # Look for colour names
        colours = ['black', 'blue', 'red', 'green', 'yellow', 'gray', 'magenta',
                   'orange', 'lightblue', 'maroon', 'white', 'darkgray', 'gold',
                   'lightgray', 'darkgreen']
        found_colours = [c for c in colours if c in text.lower()]
        if found_colours:
            parts.append(f"colors_{found_colours[:5]}")
        
        return " ".join(parts) if len(parts) > 1 else "grid_content_detected"
    except:
        return "grid_present"