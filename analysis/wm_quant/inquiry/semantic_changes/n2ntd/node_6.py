"""
Contract: perceive(observation_history) -> str must never raise.
Returns a concise text summary of decision-relevant features from the grid,
including object-level motion information to enable unique state identification.
"""

import json
from collections import Counter, deque
from typing import List, Tuple, Optional


def _parse_grid(obs: str):
    """Extract the 2D colour grid as list[list[str]] (rows of colour-name strings)."""
    if not obs:
        return None
    start = obs.find("[[")
    end = obs.rfind("]]")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        grid = json.loads(obs[start : end + 2])
    except Exception:
        return None
    if not grid or not isinstance(grid, list) or not isinstance(grid[0], list):
        return None
    return grid


def _dominant_color(grid):
    """Return the most frequent colour in the grid (the background)."""
    flat = [cell for row in grid for cell in row]
    if not flat:
        return "black"
    return Counter(flat).most_common(1)[0][0]


def _find_objects(grid, bg: str) -> List[dict]:
    """
    Identify connected components (4‑neighbour) of non‑background cells.
    Returns list of dicts with: colour, row0, col0, width, height, cells.
    """
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    visited = [[False] * cols for _ in range(rows)]
    objects = []

    for r in range(rows):
        for c in range(cols):
            colour = grid[r][c]
            if colour == bg or visited[r][c]:
                continue
            # BFS
            q = deque()
            q.append((r, c))
            visited[r][c] = True
            min_r, max_r = r, r
            min_c, max_c = c, c
            cells = [(r, c)]
            while q:
                cr, cc = q.popleft()
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == colour:
                        visited[nr][nc] = True
                        q.append((nr, nc))
                        cells.append((nr, nc))
                        if nr < min_r:
                            min_r = nr
                        if nr > max_r:
                            max_r = nr
                        if nc < min_c:
                            min_c = nc
                        if nc > max_c:
                            max_c = nc
            objects.append({
                "colour": colour,
                "row0": min_r,
                "col0": min_c,
                "width": max_c - min_c + 1,
                "height": max_r - min_r + 1,
                "cells": cells
            })
    return objects


def _compute_velocity(current_objs: List[dict], prev_grid, bg: str) -> dict:
    """
    For each object in current_objs, find matching object in prev_grid (by colour
    and closest centroid) and compute (dr, dc) displacement.
    Returns dict mapping object index (in current_objs) to (dr, dc) or None.
    """
    if prev_grid is None:
        return {}
    prev_objs = _find_objects(prev_grid, bg)
    # Build lookup by colour: list of centroids and index
    prev_by_colour = {}
    for i, obj in enumerate(prev_objs):
        colour = obj["colour"]
        centroid_r = obj["row0"] + (obj["height"] - 1) / 2.0
        centroid_c = obj["col0"] + (obj["width"] - 1) / 2.0
        prev_by_colour.setdefault(colour, []).append((centroid_r, centroid_c, i))

    velocity = {}
    for idx, cur_obj in enumerate(current_objs):
        colour = cur_obj["colour"]
        if colour not in prev_by_colour:
            continue
        cur_centroid_r = cur_obj["row0"] + (cur_obj["height"] - 1) / 2.0
        cur_centroid_c = cur_obj["col0"] + (cur_obj["width"] - 1) / 2.0
        best_dist = float('inf')
        best_dr = 0
        best_dc = 0
        for prev_centroid_r, prev_centroid_c, _ in prev_by_colour[colour]:
            dr = cur_centroid_r - prev_centroid_r
            dc = cur_centroid_c - prev_centroid_c
            dist = abs(dr) + abs(dc)
            # prefer small displacements (objects typically move 1 or 4 steps)
            if dist < best_dist:
                best_dist = dist
                best_dr = dr
                best_dc = dc
        # Only record if displacement is reasonable (≤ 10)
        if best_dist <= 10:
            velocity[idx] = (best_dr, best_dc)
    return velocity


def perceive(observation_history: list[str]) -> str:
    # Get current observation
    obs = observation_history[-1] if observation_history else ""
    grid = _parse_grid(obs)
    if grid is None:
        return "grid parse error; no features"

    try:
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        bg = _dominant_color(grid)

        # Find objects in current grid
        current_objs = _find_objects(grid, bg)

        # Get previous grid (if exists) for velocity computation
        prev_grid = None
        if len(observation_history) >= 2:
            prev_obs = observation_history[-2]
            prev_grid = _parse_grid(prev_obs)

        velocities = _compute_velocity(current_objs, prev_grid, bg)

        # Build summary
        summary_parts = [f"grid {rows}x{cols}; bg:{bg}; objects:"]

        for idx, obj in enumerate(current_objs):
            colour = obj["colour"]
            r0 = obj["row0"]
            c0 = obj["col0"]
            w = obj["width"]
            h = obj["height"]
            vel = velocities.get(idx, None)
            if vel is not None:
                dr, dc = vel
                # encode direction as shorthand
                if dr == 0 and dc == 1:
                    vel_str = "vR"
                elif dr == 0 and dc == -1:
                    vel_str = "vL"
                elif dr == -1 and dc == 0:
                    vel_str = "vU"
                elif dr == 1 and dc == 0:
                    vel_str = "vD"
                elif dr == -4 and dc == 0:
                    vel_str = "vU4"
                elif dr == 4 and dc == 0:
                    vel_str = "vD4"
                else:
                    vel_str = f"v({dr},{dc})"
            else:
                vel_str = ""

            obj_str = f"{colour}({r0},{c0}){w}x{h}{vel_str}"
            summary_parts.append(obj_str)

        summary = "; ".join(summary_parts)

        # Enforce length <2000
        if len(summary) > 1995:
            # Truncate by dropping objects from the end, but keep as many as fit
            prefix = "grid {}x{}; bg:{}; objects:".format(rows, cols, bg)
            remaining = 1995 - len(prefix) - 3  # reserve for "..."
            objects_list = current_objs.copy()
            re_summary = prefix
            for idx, obj in enumerate(objects_list):
                colour = obj["colour"]
                r0 = obj["row0"]
                c0 = obj["col0"]
                w = obj["width"]
                h = obj["height"]
                vel = velocities.get(idx, None)
                vel_str = "" if vel is None else (f"v({vel[0]},{vel[1]})")
                obj_str = f"{colour}({r0},{c0}){w}x{h}{vel_str}"
                entry = f"; {obj_str}"
                if len(re_summary) + len(entry) > 1995 - 3:
                    re_summary += "..."
                    break
                re_summary += entry
            summary = re_summary

        if not summary:
            summary = f"grid {rows}x{cols}; bg:{bg}; objects:"
        return summary

    except Exception:
        # Never raise; return a non-empty fallback
        return "perception error"