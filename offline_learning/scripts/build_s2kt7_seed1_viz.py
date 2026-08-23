"""Self-contained viz for a regenerated clean_data3 game (default s2kt7_seed1).
Renders each train episode as a strip of 16x16 colour grids (CSS cells, no deps),
labelled with Step + Action, annotating per-frame changes (food spawned/eaten,
ants moved). Shows the real foraging dynamics that the seed-0 data hid.
"""

from __future__ import annotations
import argparse, csv, json, html
from pathlib import Path

csv.field_size_limit(10**9)

# Autumn colour names -> CSS. black background rendered as a dark cell.
CSS_COLOR = {
    "black": "#0c0e12", "white": "#e8ecf1", "gray": "#8a93a0", "grey": "#8a93a0",
    "red": "#ff4d4d", "gold": "#ffd45a", "green": "#54d98c", "mediumpurple": "#9b7fe0",
    "purple": "#7a5cc0", "blue": "#4d9dff", "lightblue": "#8fd0ff",
    "darkorange": "#ff9a3d", "yellow": "#ffe14d",
}

PAGE_CSS = """
body{background:#0c0e12;color:#cdd3da;font:13px ui-monospace,Menlo,monospace;margin:16px}
h1{margin:0 0 2px;font-size:18px} h2{margin:18px 0 6px;font-size:14px;color:#ffd45a}
.sub{color:#8a93a0;margin-bottom:10px;max-width:900px;line-height:1.5}
.strip{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:8px}
figure{margin:0;background:#151921;border:1px solid #232a33;border-radius:6px;padding:6px;width:150px}
.grid{display:grid;width:138px;height:138px;border:1px solid #2a3340;image-rendering:pixelated}
.grid > i{display:block}
figcaption{padding:5px 2px 1px;line-height:1.5;font-size:11px}
.s{color:#7fd1ff}.act{color:#ffd45a;font-weight:700}.actnoop{color:#6b7480}
.chg{color:#7fd99a;font-size:10px;display:block;margin-top:3px;word-break:break-word}
.leg{margin:6px 0 14px;color:#8a93a0}.leg b{display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:-2px;margin:0 3px 0 10px}
"""


def grid_of(obs: str):
    s, e = obs.find("[["), obs.rfind("]]")
    return json.loads(obs[s:e + 2])


def cells(g):
    return {(r, c): v for r, row in enumerate(g) for c, v in enumerate(row) if v not in ("black", "white")}


def render_grid_html(g) -> str:
    n = len(g)
    out = [f'<div class="grid" style="grid-template-columns:repeat({n},1fr);grid-template-rows:repeat({n},1fr)">']
    for row in g:
        for v in row:
            out.append(f'<i style="background:{CSS_COLOR.get(v, v)}"></i>')
    out.append("</div>")
    return "".join(out)


def change_note(c1, c2, action):
    reds1 = {k for k, v in c1.items() if v == "red"}
    reds2 = {k for k, v in c2.items() if v == "red"}
    grays1 = {k for k, v in c1.items() if v in ("gray", "grey")}
    grays2 = {k for k, v in c2.items() if v in ("gray", "grey")}
    parts = []
    spawned = reds2 - reds1
    eaten = reds1 - reds2
    if spawned:
        parts.append(f"+{len(spawned)} food {sorted(spawned)}")
    if eaten:
        parts.append(f"-{len(eaten)} food eaten")
    if grays1 != grays2:
        parts.append("ants moved")
    if not parts:
        parts.append("no visible change")
    return "; ".join(parts)


def build(train_dir: Path, out: Path, title: str):
    eps = sorted(train_dir.glob("episode_*"))
    body = [f"<h1>{html.escape(title)}</h1>",
            '<div class="sub">Regenerated under non-zero per-episode seeds &mdash; the REAL '
            's2kt7 foraging game (clicks scatter 2 food at random cells; ants step toward the '
            'nearest food and eat it). Contrast the seed-0 data where every click dropped food '
            'at (0,0).</div>',
            '<div class="leg">legend: <b style="background:#8a93a0"></b>ant (gray) '
            '<b style="background:#ff4d4d"></b>food (red) <b style="background:#0c0e12;border:1px solid #2a3340"></b>empty</div>']
    for ep in eps:
        rows = list(csv.DictReader((ep / "trajectory.csv").open()))
        grids = [grid_of(r["Observation"]) for r in rows]
        body.append(f"<h2>{ep.name} ({len(rows)} frames)</h2>")
        body.append('<div class="strip">')
        for i, r in enumerate(rows):
            act = r["Action"].strip()
            acls = "actnoop" if act == "noop" else "act"
            note = ""
            if i < len(rows) - 1:
                note = f'<span class="chg">&rarr; {html.escape(change_note(cells(grids[i]), cells(grids[i+1]), act))}</span>'
            body.append(
                f'<figure>{render_grid_html(grids[i])}'
                f'<figcaption><span class="s">Step {r["Step"]}</span> '
                f'<span class="{acls}">{html.escape(act)}</span>{note}</figcaption></figure>')
        body.append("</div>")
    doc = f"<!doctype html><meta charset=utf-8><title>{html.escape(title)}</title><style>{PAGE_CSS}</style>{''.join(body)}"
    out.write_text(doc)
    print(f"[viz] wrote {out} | {len(eps)} episodes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="/home/ays57/bai/offline_learning/clean_data3/s2kt7_seed1/train")
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default="s2kt7_seed1 — train (regenerated, real foraging game)")
    args = ap.parse_args()
    td = Path(args.train_dir)
    out = Path(args.out) if args.out else td / "viz.html"
    build(td, out, args.title)


if __name__ == "__main__":
    main()
