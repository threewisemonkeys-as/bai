#!/usr/bin/env python3
"""Inspect coverage-anchored PLANNING PROBLEMS (not an eval) for manual review.

Reads a coverage_plan_problems.json (built by coverage_plan.py) and, per problem, shows
START, GOAL, and the ground-truth action sequence executed in the Autumn engine as a
filmstrip (start -> after each gt action; the frame that reaches GOAL is ringed). Header
chips carry bucket / mechanic / horizon / BOARD POPULATION (# non-bg cells in start) and
the noop / random baselines. Filter by game / bucket / mechanic / horizon / population band
-- so you can jump straight to the populated-board problems the _spread fix introduces.

    uv run python offline_learning/scripts/viz_coverage_problems.py \
        --in logs/coverage_plan_problems_83wkq_v2.json --out logs/coverage_problems_viz.html
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "offline_learning"))
from human_replay import GAMES as HGAMES  # noqa: E402
from coverage_plan import exec_plan, load_coverage  # noqa: E402
from mechanics import _BG  # noqa: E402

CSS = {"black": "#0d1017", "white": "#eef1f5", "gray": "#8b929c", "blue": "#2f6fd0",
       "lightblue": "#8fc4e8", "gold": "#e0a92e", "darkgreen": "#2e8b57", "red": "#d1442f",
       "darkorange": "#e07b1a", "mediumpurple": "#9575cd", "green": "#3fae5a",
       "orange": "#e08a1a", "yellow": "#e6c34a"}


def popband(p: int) -> str:
    return "empty" if p == 0 else "sparse(1-5)" if p <= 5 else "medium(6-15)" if p <= 15 else "dense(16+)"


def encode(data: dict, games: set[str], limit: int) -> dict:
    pal: dict[str, int] = {}

    def enc(grid_json):
        if not grid_json:
            return None
        out = []
        for row in json.loads(grid_json):
            erow = []
            for name in row:
                if name not in pal:
                    pal[name] = len(pal)
                erow.append(pal[name])
            out.append(erow)
        return out

    def population(grid_json, bg):
        return sum(1 for row in json.loads(grid_json) for c in row if c not in bg) if grid_json else 0

    cov: dict[str, dict] = {}
    per_game: dict[str, int] = {}
    out = []
    for p in data["problems"]:
        g = p["game"]
        if games and g not in games:
            continue
        if limit and per_game.get(g, 0) >= limit:
            continue
        per_game[g] = per_game.get(g, 0) + 1
        if g not in cov:
            cov[g] = load_coverage(g)
        bg = _BG.get(g, set())
        gt = p.get("gt_actions") or []
        grids, err = None, None
        try:
            prefix = cov[g]["drives_by_seed"][p["seed"]]["actions"][:p["t"]]
            grids = exec_plan(cov[g]["program"], p["seed"], prefix, gt)
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
        pop = population(p["start_grid"], bg)
        reached = next((i + 1 for i, gr in enumerate(grids or []) if gr == p["goal_grid"]), None)
        out.append({
            "game": g, "human": HGAMES[g][1], "bucket": p["bucket"], "mechanic": p["mechanic"],
            "kind": p.get("kind"), "h": p["h"], "seed": p["seed"], "t": p["t"],
            "synthetic": p.get("synthetic"), "pop": pop, "band": popband(pop),
            "noop": p.get("noop_success"), "rand": p.get("random_success"),
            "reached_at": reached, "err": err, "gt": gt,
            "start": enc(p["start_grid"]), "goal": enc(p["goal_grid"]),
            "frames": [enc(gr) for gr in grids] if grids else None})
    return {"palette": {i: CSS.get(n, n) for n, i in pal.items()}, "problems": out}


HTML = r"""<!doctype html><meta charset="utf-8">
<title>Coverage planning problems — inspect</title>
<style>
:root{--paper:#f4f5f2;--ink:#1b1f27;--muted:#5c636e;--line:#d9dcd6;--card:#fff;
  --accent:#3a6ea5;--good:#2c7a68;--chip:#eceee9;--warn:#9c7a32;}
@media (prefers-color-scheme:dark){:root{--paper:#0f1217;--ink:#e6e9ef;--muted:#98a0ad;
  --line:#242a33;--card:#161b22;--accent:#6ea8dc;--good:#4bbfa3;--chip:#1d232c;--warn:#c8a24e;}}
:root[data-theme=dark]{--paper:#0f1217;--ink:#e6e9ef;--muted:#98a0ad;--line:#242a33;
  --card:#161b22;--accent:#6ea8dc;--good:#4bbfa3;--chip:#1d232c;--warn:#c8a24e;}
:root[data-theme=light]{--paper:#f4f5f2;--ink:#1b1f27;--muted:#5c636e;--line:#d9dcd6;
  --card:#fff;--accent:#3a6ea5;--good:#2c7a68;--chip:#eceee9;--warn:#9c7a32;}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.5 ui-sans-serif,system-ui,sans-serif}
.wrap{max-width:1300px;margin:0 auto;padding:24px 20px 80px}
h1{font:600 22px/1.2 Georgia,serif;margin:0 0 4px}
.sub{color:var(--muted);margin:0 0 18px;max-width:900px}
.mono{font-family:ui-monospace,Menlo,Consolas,monospace}
.bar{position:sticky;top:0;z-index:5;background:var(--paper);border-bottom:1px solid var(--line);
  display:flex;flex-wrap:wrap;gap:10px 14px;align-items:center;padding:12px 0;margin-bottom:16px}
.bar label{font-size:12px;color:var(--muted);margin-right:4px}
select,button{font:inherit;color:var(--ink);background:var(--card);border:1px solid var(--line);
  border-radius:7px;padding:5px 8px}
button{cursor:pointer}
.count{margin-left:auto;color:var(--muted);font-size:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin-bottom:16px}
.head{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:10px}
.chip{background:var(--chip);border-radius:999px;padding:2px 10px;font-size:12px}
.chip.b{color:#fff;background:var(--accent)}
.chip.pop{color:#fff;background:var(--warn)}
.pair{display:flex;gap:22px;align-items:flex-start;flex-wrap:wrap;margin-bottom:6px}
.frame{display:flex;flex-direction:column;align-items:center;gap:4px}
.frame .cap{font-size:11px;color:var(--muted)}
.strip{display:flex;gap:8px;overflow-x:auto;padding:6px 0}
.step{display:flex;flex-direction:column;align-items:center;gap:3px;flex:0 0 auto}
.act{font-size:11px;padding:1px 6px;border-radius:5px;background:var(--chip)}
canvas{border:1px solid var(--line);border-radius:3px;display:block;image-rendering:pixelated}
.hit canvas{outline:2px solid var(--good);outline-offset:1px}
.err{color:#d1442f;font-size:12px}
.arrow{align-self:center;color:var(--muted);font-size:16px}
</style>
<div class="wrap">
  <h1>Coverage planning problems — inspect</h1>
  <p class="sub">Each card is one planning problem: <b>START</b> and <b>GOAL</b>, then the
  ground-truth actions executed in the engine (frame after each action; the frame reaching
  GOAL is green-ringed). The <span class="chip pop" style="padding:1px 6px">pop</span> chip is
  the board population (# non-background cells in START). Use the <b>population</b> filter to
  jump to populated-board problems.</p>
  <div class="bar">
    <span><label>game</label><select id="fg"></select></span>
    <span><label>bucket</label><select id="fb"></select></span>
    <span><label>mechanic</label><select id="fm"></select></span>
    <span><label>horizon</label><select id="fh"></select></span>
    <span><label>population</label><select id="fp">
      <option value="">any</option><option value="empty">empty</option>
      <option value="sparse(1-5)">sparse(1-5)</option><option value="medium(6-15)">medium(6-15)</option>
      <option value="dense(16+)">dense(16+)</option></select></span>
    <button id="prev">◀</button><button id="next">▶</button>
    <span class="count" id="count"></span>
  </div>
  <div id="list"></div>
</div>
<script>
const DATA=/*DATA*/{};
const PAL=DATA.palette, P=DATA.problems;
const CELL=8;
function draw(grid){
  const cv=document.createElement("canvas");
  if(!grid){cv.width=cv.height=CELL*3;return cv;}
  const R=grid.length,C=grid[0].length;cv.width=C*CELL;cv.height=R*CELL;
  const x=cv.getContext("2d");
  for(let r=0;r<R;r++)for(let c=0;c<C;c++){x.fillStyle=PAL[grid[r][c]]||"#333";x.fillRect(c*CELL,r*CELL,CELL,CELL);}
  return cv;
}
function frame(grid,cap,ring){
  const d=document.createElement("div");d.className="frame"+(ring?" hit":"");
  d.appendChild(draw(grid));
  if(cap!=null){const s=document.createElement("div");s.className="cap";s.textContent=cap;d.appendChild(s);}
  return d;
}
function fmt(v){return v==null?"—":(+v).toFixed(2);}
function render(p){
  const c=document.createElement("div");c.className="card";
  const h=document.createElement("div");h.className="head";
  const bits=[["b",p.game+" · "+p.human],["",p.bucket],["",p.mechanic],["","h="+p.h],
    ["pop","pop "+p.pop+" ("+p.band+")"],["","noop "+fmt(p.noop)],["","rand "+fmt(p.rand)],
    ["","seed "+p.seed+" t="+p.t]];
  if(p.synthetic)bits.push(["","synthetic"]);
  bits.forEach(([cl,t])=>{const s=document.createElement("span");s.className="chip"+(cl?" "+cl:"");s.textContent=t;h.appendChild(s);});
  c.appendChild(h);
  const pair=document.createElement("div");pair.className="pair";
  pair.appendChild(frame(p.start,"START",false));
  const ar=document.createElement("div");ar.className="arrow";ar.textContent="→ goal";pair.appendChild(ar);
  pair.appendChild(frame(p.goal,"GOAL",false));
  c.appendChild(pair);
  if(p.err){const e=document.createElement("div");e.className="err";e.textContent="gt exec error: "+p.err;c.appendChild(e);}
  if(p.frames){
    const strip=document.createElement("div");strip.className="strip";
    strip.appendChild(frame(p.start,"start",false));
    const goalStr=JSON.stringify(p.goal);
    p.frames.forEach((g,i)=>{
      const sep=document.createElement("span");sep.className="arrow";sep.textContent="›";strip.appendChild(sep);
      const st=document.createElement("div");st.className="step";
      st.appendChild(frame(g,null,JSON.stringify(g)===goalStr));
      const a=document.createElement("div");a.className="act mono";
      a.textContent=(i+1)+". "+(p.gt[i]||"?");st.appendChild(a);strip.appendChild(st);});
    c.appendChild(strip);
  }
  return c;
}
function opts(sel,vals,all){sel.innerHTML="";const o=document.createElement("option");o.value="";o.textContent=all;sel.appendChild(o);
  [...new Set(vals)].sort().forEach(v=>{const e=document.createElement("option");e.value=v;e.textContent=v;sel.appendChild(e);});}
const fg=document.getElementById("fg"),fb=document.getElementById("fb"),fm=document.getElementById("fm"),
  fh=document.getElementById("fh"),fp=document.getElementById("fp");
opts(fg,P.map(p=>p.game),"all games");opts(fb,P.map(p=>p.bucket),"all buckets");opts(fh,P.map(p=>String(p.h)),"all h");
function refreshMech(){const g=fg.value;opts(fm,P.filter(p=>!g||p.game==g).map(p=>p.mechanic),"all mechanics");}
refreshMech();
let page=0;const PER=8;
function match(p){
  if(fg.value&&p.game!=fg.value)return false;
  if(fb.value&&p.bucket!=fb.value)return false;
  if(fm.value&&p.mechanic!=fm.value)return false;
  if(fh.value&&String(p.h)!=fh.value)return false;
  if(fp.value&&p.band!=fp.value)return false;
  return true;
}
function update(){
  const sel=P.filter(match);
  const pages=Math.max(1,Math.ceil(sel.length/PER));page=Math.max(0,Math.min(page,pages-1));
  const list=document.getElementById("list");list.innerHTML="";
  sel.slice(page*PER,page*PER+PER).forEach(p=>list.appendChild(render(p)));
  const bands={};sel.forEach(p=>bands[p.band]=(bands[p.band]||0)+1);
  document.getElementById("count").textContent=sel.length+" problems · page "+(page+1)+"/"+pages+" · "+JSON.stringify(bands).replace(/"/g,"");
}
[fg,fb,fm,fh,fp].forEach(el=>el.addEventListener("change",()=>{if(el==fg)refreshMech();page=0;update();}));
document.getElementById("prev").onclick=()=>{page--;update();};
document.getElementById("next").onclick=()=>{page++;update();};
update();
</script>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(REPO / "logs/coverage_plan_problems.json"))
    ap.add_argument("--out", default=str(REPO / "logs/coverage_problems_viz.html"))
    ap.add_argument("--games", default="", help="comma-separated subset")
    ap.add_argument("--limit", type=int, default=0, help="cap problems per game")
    args = ap.parse_args()
    data = json.loads(Path(args.inp).read_text())
    games = set(filter(None, args.games.split(",")))
    enc = encode(data, games, args.limit)
    html = HTML.replace("/*DATA*/{}", json.dumps(enc))
    Path(args.out).write_text(html, encoding="utf-8")
    from collections import Counter
    bands = Counter(p["band"] for p in enc["problems"])
    print(f"{len(enc['problems'])} problems, {len(enc['palette'])} colours -> {args.out} "
          f"({len(html)/1024:.0f} KB) | population bands: {dict(bands)}")


if __name__ == "__main__":
    main()
