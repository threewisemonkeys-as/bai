"""Drive an ARC game and render each frame to PNG (real ARC palette) for visual
inspection — used to locate the player + target objects when designing clean data."""
import argparse, re, sys
from pathlib import Path
_BAI = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BAI))
import numpy as np
from PIL import Image
from arc_agi.rendering import frame_to_rgb_array
from arc_agi_env import ArcAgiEnvWrapper


def grids(o):
    txt = o["text"]["long_term_context"] + o["text"]["short_term_context"]
    out = []
    for block in re.findall(r"<grid_\d+>(.*?)</grid_\d+>", txt, re.DOTALL):
        rows = []
        for line in block.strip().splitlines():
            nums = re.findall(r"-?\d+", line)
            if nums:
                rows.append([int(n) for n in nums])
        if rows:
            out.append(rows)
    return out


def save(grid, path, scale=8):
    arr = frame_to_rgb_array(0, np.array(grid, dtype=int), scale=scale)
    Image.fromarray(arr.astype("uint8")).save(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("outdir")
    ap.add_argument("--actions", default="")
    ap.add_argument("--scale", type=int, default=8)
    args = ap.parse_args()
    od = Path(args.outdir); od.mkdir(parents=True, exist_ok=True)
    acts = []
    for a in args.actions.split(","):
        a = a.strip()
        if not a:
            continue
        if a.startswith("click_"):
            _, x, y = a.split("_"); acts.append(f"ACTION6 x={x} y={y}")
        else:
            acts.append(a)
    env = ArcAgiEnvWrapper(game_id=args.game, max_episode_steps=300, seed=0)
    obs, _ = env.reset(seed=0)
    g = grids(obs)
    print(f"reset: {len(g)} grid layer(s), shape {len(g[0])}x{len(g[0][0])}")
    save(g[0], od / "step00_reset.png", args.scale)
    for i, a in enumerate(acts, 1):
        obs, r, term, trunc, info = env.step(a)
        g = grids(obs)
        save(g[0], od / f"step{i:02d}_{a.replace(' ', '_').replace('=', '')}.png", args.scale)
        print(f"step {i}: {a} reward={r} term={term} layers={len(g)}")
        if term:
            break
    env.close()
    print(f"wrote PNGs -> {od}")
