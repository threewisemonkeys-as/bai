"""Render a clean-data trajectory.csv as a self-contained HTML filmstrip.

Each frame is the stored Observation rendered with the real ARC palette (ARC
games) or the autumn color map; the caption shows the action taken FROM that
frame (which produces the next one), plus reward/done. For eyeballing/debugging
a clean dataset.

    uv run python offline_learning/scripts/build_dataset_viz.py \
        offline_learning/clean_data/ls20 --out .../ls20/viz.html
"""
import argparse, base64, csv, io, re, sys
from pathlib import Path
_BAI = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BAI))
import numpy as np
from PIL import Image
from matplotlib import colors as mcolors
from arc_agi.rendering import frame_to_rgb_array

# autumn renders each cell with matplotlib.colors.to_rgb(color_name) (the grid
# strings are valid matplotlib color names); "mask" -> "slategrey". We use the
# SAME conversion so the viz matches the real autumn colors exactly.
_AUT_CACHE: dict[str, tuple] = {}


def aut_rgb(name: str) -> tuple:
    name = (name or "").lower()
    if name in _AUT_CACHE:
        return _AUT_CACHE[name]
    key = "slategrey" if name == "mask" else name
    try:
        rgb = tuple(int(round(v * 255)) for v in mcolors.to_rgb(key))
    except ValueError:
        rgb = (20, 20, 24)  # unknown name -> autumn background (black)
    _AUT_CACHE[name] = rgb
    return rgb


def arc_grid(obs):
    blocks = re.findall(r"<grid_\d+>(.*?)</grid_\d+>", obs, re.DOTALL)
    if not blocks:
        return None
    rows = []
    for line in blocks[0].strip().splitlines():
        nums = re.findall(r"-?\d+", line)
        if nums:
            rows.append([int(n) for n in nums])
    return rows or None


def autumn_grid(obs):
    i = obs.find("[[")
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(obs)):
        if obs[j] == "[":
            depth += 1
        elif obs[j] == "]":
            depth -= 1
            if depth == 0:
                import json
                return json.loads(obs[i:j + 1])
    return None


def render_png_b64(obs, scale=6):
    g = arc_grid(obs)
    if g is not None:
        arr = frame_to_rgb_array(0, np.array(g, dtype=int), scale=scale)
        im = Image.fromarray(arr.astype("uint8"))
    else:
        g = autumn_grid(obs)
        if g is None:
            return None, None
        h, w = len(g), len(g[0])
        arr = np.zeros((h, w, 3), dtype="uint8")
        for r in range(h):
            for c in range(w):
                arr[r, c] = aut_rgb(g[r][c])
        im = Image.fromarray(arr).resize((w * scale * 2, h * scale * 2), Image.NEAREST)
    buf = io.BytesIO(); im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode(), f"{len(g[0])}x{len(g)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="clean_data/<game> dir (renders ALL episode_*/trajectory.csv)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--scale", type=int, default=6)
    ap.add_argument("--highlight", default="",
                    help="comma-separated card indices (= load_transitions row indices) to mark "
                         "as SELECTED train examples; they get a gold border + ★TRAIN badge")
    args = ap.parse_args()
    ddir = Path(args.dataset)
    csv.field_size_limit(10 ** 7)
    eps = sorted(ddir.glob("episode_*"),
                 key=lambda p: int(re.sub(r"\D", "", p.name) or 0))
    rows = []  # (episode name, csv row)
    for ep in eps:
        f = ep / "trajectory.csv"
        if f.exists():
            rows += [(ep.name, r) for r in list(csv.reader(open(f)))[1:]]
    hl = {int(x) for x in args.highlight.split(",") if x.strip()}

    cards = []
    cur_ep = None
    for i, (ep_name, r) in enumerate(rows):
        if len(eps) > 1 and ep_name != cur_ep:
            cur_ep = ep_name
            cards.append(f'<div class=ep>{ep_name}</div>')
        step, action, reasoning, obs = r[0], r[1], r[2], r[3]
        reward, done = r[5], r[6]
        b64, shape = render_png_b64(obs, args.scale)
        lvl = re.search(r"Levels completed: (\d+)/(\d+)", obs)
        lvl = lvl.group(0) if lvl else ""
        act_html = f'<b>{action}</b>' if action else '<i>(terminal)</i>'
        rew_html = f' · <span class="r">rew {reward}</span>' if reward not in ("0.0", "0", "") else ""
        img = (f'<img src="data:image/png;base64,{b64}">' if b64
               else '<div class="noimg">no grid</div>')
        sel = i in hl
        badge = '<span class=badge>&#9733; TRAIN</span>' if sel else ''
        cards.append(
            f'<figure class="{"sel" if sel else ""}">{badge}{img}<figcaption>'
            f'<span class=s>step {step}</span> &rarr; {act_html}{rew_html}<br>'
            f'<span class=meta>{shape or ""} {lvl} {"DONE" if done=="True" else ""}</span>'
            f'</figcaption></figure>')

    html = f"""<!doctype html><meta charset=utf8><title>{ddir.name} trajectory</title>
<style>
body{{background:#0c0e12;color:#cdd3da;font:13px ui-monospace,Menlo,monospace;margin:16px}}
h2{{margin:0 0 4px}} .sub{{color:#8a93a0;margin-bottom:14px}}
.strip{{display:flex;flex-wrap:wrap;gap:10px}}
figure{{margin:0;background:#151921;border:1px solid #232a33;border-radius:6px;padding:6px;position:relative}}
figure.sel{{border:2px solid #ffd45a;background:#211d12;box-shadow:0 0 0 2px rgba(255,212,90,.25)}}
.badge{{position:absolute;top:8px;left:8px;background:#ffd45a;color:#1a1407;font-size:10px;font-weight:700;padding:1px 5px;border-radius:3px;letter-spacing:.3px}}
img{{display:block;image-rendering:pixelated;width:180px;border:1px solid #2a3340}}
.noimg{{width:180px;height:120px;display:flex;align-items:center;justify-content:center;color:#556}}
figcaption{{padding:5px 2px 1px;line-height:1.45}}
.s{{color:#7fd1ff}} b{{color:#ffd45a}} .r{{color:#4fcc66}} .meta{{color:#7a838f;font-size:11px}}
.ep{{flex-basis:100%;color:#7fd1ff;border-bottom:1px solid #232a33;padding:10px 0 3px;font-weight:700}}
</style>
<h2>{ddir.name} &mdash; clean trajectory ({len(rows)} frames, {len(rows)-len(eps)} transitions, {len(eps)} episode{'s' if len(eps)!=1 else ''})</h2>
<div class=sub>Each frame shows the action taken FROM it (producing the next frame). Source: {ddir}{f' &middot; <b style="color:#ffd45a">{len(hl)} gold-bordered frames = the seed-1 scored train batch</b>' if hl else ''}</div>
<div class=strip>{''.join(cards)}</div>"""
    out = Path(args.out) if args.out else ddir / "viz.html"
    out.write_text(html)
    print(f"wrote {out} ({len(rows)} frames)")


if __name__ == "__main__":
    main()
