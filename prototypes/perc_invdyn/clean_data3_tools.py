"""Shared helpers for building clean_data3 curated train sets.

Run from prototypes/perc_invdyn (use `uv run python` from repo root for deps).

  uv run python -c "import sys; sys.path.insert(0,'prototypes/perc_invdyn'); \
      import clean_data3_tools as T; T.dump_transitions('dq8gc','left,right,up,down,noop,click')"

Functions
  grid_at(obs)            -> 2D list of color strings (last [[...]] block in the observation)
  diff(g0,g1)             -> dict: per-color added/removed cells, centroid shift, nochange
  classify(g0,g1)         -> short human tag string summarizing what changed
  dump_transitions(game, whitelist, data_root='clean_data2')
                          -> prints EVERY whitelisted transition i->i+1 with action + diff
  verify_pool(train_dir, whitelist, context_k=9)
                          -> loads via gepa's load_transitions; prints pool composition +
                             per-target window sizes (this is what GEPA will actually score)
"""
from __future__ import annotations
import ast, csv, re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def grid_at(obs: str):
    """Parse the main grid out of an observation. Handles BOTH formats:
    autumn `[[ "color", ... ]]` color-string arrays AND ARC `<grid_N>` integer blocks
    (returns the FIRST grid block; cell values are ints 0..15)."""
    obs = obs or ""
    if "<grid_" in obs:  # ARC-AGI integer grid
        m = re.search(r"<grid_\d+>(.*?)</grid_\d+>", obs, re.DOTALL)
        if not m:
            return None
        rows = []
        for line in m.group(1).strip().splitlines():
            nums = re.findall(r"-?\d+", line)
            if nums:
                rows.append([int(n) for n in nums])
        return rows or None
    m = re.search(r"(\[\[.*?\]\])", obs.replace("\n", " "))  # autumn color array
    if not m:
        return None
    try:
        return ast.literal_eval(m.group(1))
    except Exception:
        return None


def _cells(g, color):
    return {(r, c) for r, row in enumerate(g) for c, v in enumerate(row) if v == color}


def _centroid(pts):
    if not pts:
        return None
    return (round(sum(r for r, _ in pts) / len(pts), 2),
            round(sum(c for _, c in pts) / len(pts), 2))


def diff(g0, g1):
    """Per-color change summary between two grids."""
    colors = {v for row in g0 for v in row} | {v for row in g1 for v in row}
    out = {}
    for col in colors:
        a, b = _cells(g0, col), _cells(g1, col)
        if a == b:
            continue
        ca, cb = _centroid(a), _centroid(b)
        shift = None
        if ca and cb:
            shift = (round(cb[0] - ca[0], 2), round(cb[1] - ca[1], 2))
        out[col] = {
            "n": (len(a), len(b)),
            "added": sorted(b - a)[:8],
            "removed": sorted(a - b)[:8],
            "centroid_shift": shift,
        }
    return out


def classify(g0, g1):
    if g0 is None or g1 is None:
        return "??"
    d = diff(g0, g1)
    if not d:
        return "NO_CHANGE"
    parts = []
    for col, info in sorted(d.items(), key=lambda kv: str(kv[0])):
        na, nb = info["n"]
        tag = str(col)
        if nb > na:
            tag += f"+{nb-na}"          # cells appeared
        elif nb < na:
            tag += f"-{na-nb}"          # cells vanished
        elif info["centroid_shift"] and info["centroid_shift"] != (0.0, 0.0):
            tag += f"~move{info['centroid_shift']}"   # same count, moved
        else:
            tag += "~recolor"
        parts.append(tag)
    return " ".join(parts)


def _rows(game, split="train", data_root="clean_data2", ep=0):
    p = ROOT / data_root / game / split / f"episode_{ep}" / "trajectory.csv"
    return list(csv.DictReader(p.open()))


def dump_transitions(game, whitelist, data_root="clean_data2", split="train"):
    wl = {w.strip() for w in whitelist.split(",") if w.strip()} if isinstance(whitelist, str) else set(whitelist)
    rows = _rows(game, split, data_root)
    print(f"### {game}/{split}  ({len(rows)} rows)  whitelist={sorted(wl)}")
    counts = Counter()
    for i in range(len(rows) - 1):
        a = (rows[i]["Action"] or "").strip()
        if not a or a.split()[0] not in wl:
            continue
        g0, g1 = grid_at(rows[i]["Observation"]), grid_at(rows[i + 1]["Observation"])
        tag = classify(g0, g1)
        counts[tag.split()[0] if tag != "NO_CHANGE" else "NO_CHANGE"] += 1
        s0, s1 = rows[i]["Step"], rows[i + 1]["Step"]
        print(f"  {s0:>3}->{s1:<3} | {a:14s} | {tag}")
    print("  --- change-tag histogram:", dict(counts))


def verify_pool(train_dir, whitelist, context_k=9):
    """Load the curated train dir the way GEPA will, and report what gets SCORED."""
    import sys
    sys.path.insert(0, str(ROOT))
    from validate import load_transitions  # noqa
    wl = {w.strip() for w in whitelist.split(",") if w.strip()} if isinstance(whitelist, str) else set(whitelist)
    trs = load_transitions([Path(train_dir)], wl, context_k=context_k)
    print(f"### POOL {train_dir}: {len(trs)} scored target transitions")
    print("    by action:", dict(Counter(t.action for t in trs)))
    for t in trs:
        cp = len(getattr(t, "ctx_prev", []) or [])
        cn = len(getattr(t, "ctx_next", []) or [])
        g0, g1 = grid_at(t.x_t), grid_at(t.x_t1)
        print(f"    {t.action:14s} win(prev={cp},next={cn}) | {classify(g0, g1)}")
    return trs
