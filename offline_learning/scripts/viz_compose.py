#!/usr/bin/env python3
"""Render a compositional problem set as start -> trajectory -> goal filmstrips.

For every problem: the START grid, then the plan replayed frame by frame in the Autumn
engine with each action labelled underneath and the mechanics that fired at that step
shown as chips, then the GOAL grid. The last frame is ringed because it IS the goal (the
ground-truth plan reaches it by construction -- V1 in validate_compose.py re-checks that).

Per-step mechanics are re-derived with a full `trace()` per problem rather than read off
the stored `fire_at` map. `fire_at` records only the FIRST step each chain member fires,
so a step whose rules had all been seen earlier would render with no chips and read as
padding -- which is exactly the thing this set exists to not have. Scenery rules
(n2ntd's enemy patrol) are drawn dimmed: they must still be predicted, but they are not
chain links.

    uv run python offline_learning/scripts/viz_compose.py \
        --in logs/2026-08-18/compose/problems_v3.json \
        --out logs/2026-08-18/compose/viz.html
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from offline_learning.compose_plan import is_scenery, trace  # noqa: E402
from offline_learning.human_replay import GAMES  # noqa: E402

CSS = {"black": "#0d1017", "white": "#eef1f5", "gray": "#8b929c", "blue": "#2f6fd0",
       "lightblue": "#8fc4e8", "gold": "#e0a92e", "darkgreen": "#2e8b57", "red": "#d1442f",
       "darkorange": "#e07b1a", "mediumpurple": "#9575cd", "green": "#3fae5a",
       "orange": "#e08a1a", "yellow": "#e6c34a"}


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

    problems, per = [], Counter()
    for p in data["problems"]:
        g = p["game"]
        if games and g not in games:
            continue
        if limit and per[g] >= limit:
            continue
        per[g] += 1
        try:
            t = trace(g, p["seed"], p["prefix"], p["gt_actions"])
        except Exception as exc:
            frames, per_step, err = [], [], f"{type(exc).__name__}: {exc}"
        else:
            err = None
            frames = t["grids"][1:]
            per_step = []
            for f in t["fired"]:
                row = [{"m": m, "a": True, "s": is_scenery(g, m)} for m in f.action]
                row += [{"m": m, "a": False, "s": is_scenery(g, m)} for m in f.passive
                        if m not in ("static-noop", "ant-idle", "particle-idle")]
                per_step.append(row)
        by_step = {i: row for i, row in enumerate(per_step)}
        problems.append({
            "game": g, "human": GAMES[g][1], "chain": p["chain"], "h": p["h"],
            "h_min": p["h_min"], "n_dec": p["n_dec"], "n_mech": p["n_mech"],
            "rand": p.get("random_success"), "seed": p["seed"],
            "prefix_len": len(p["prefix"]), "plan": p["gt_actions"],
            "start": enc(p["start_grid"]), "goal": enc(p["goal_grid"]),
            "frames": [enc(f) for f in frames],
            "reached": bool(frames) and frames[-1] == p["goal_grid"],
            "steps": [by_step.get(i, []) for i in range(len(p["gt_actions"]))],
            "prefix": p["prefix"][-6:],
            "err": err,
        })
    return {"palette": {i: CSS.get(n, n) for n, i in pal.items()}, "problems": problems}


HTML = r"""<!doctype html><meta charset="utf-8">
<title>Compositional planning problems</title>
<style>
:root{--paper:#f4f5f2;--ink:#1b1f27;--muted:#5c636e;--line:#d9dcd6;--card:#fff;
  --accent:#3a6ea5;--good:#2c7a68;--bad:#a94f38;--chip:#eceee9;--pass:#e8f0ea;--act:#dfe9f4}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--paper:#0f1217;--ink:#e6e9ef;
  --muted:#98a0ad;--line:#242a33;--card:#161b22;--accent:#6ea8dc;--good:#4bbfa3;--bad:#d98b6f;
  --chip:#1d232c;--pass:#17251f;--act:#182430}}
:root[data-theme=dark]{--paper:#0f1217;--ink:#e6e9ef;--muted:#98a0ad;--line:#242a33;
  --card:#161b22;--accent:#6ea8dc;--good:#4bbfa3;--bad:#d98b6f;--chip:#1d232c;
  --pass:#17251f;--act:#182430}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font:14px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:24px 20px 80px}
h1{font:600 22px/1.2 Georgia,"Times New Roman",serif;margin:0 0 4px}
.sub{color:var(--muted);margin:0 0 18px;max-width:70ch}
.mono{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
.bar{position:sticky;top:0;z-index:5;background:var(--paper);border-bottom:1px solid var(--line);
  display:flex;flex-wrap:wrap;gap:10px 14px;align-items:center;padding:12px 0;margin-bottom:16px}
.bar label{font-size:12px;color:var(--muted);margin-right:4px}
select,button{font:inherit;color:var(--ink);background:var(--card);border:1px solid var(--line);
  border-radius:7px;padding:5px 8px}
button{cursor:pointer}
.count{margin-left:auto;color:var(--muted);font-size:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
  padding:14px 16px;margin-bottom:16px}
.head{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:12px}
.chip{background:var(--chip);border-radius:999px;padding:2px 10px;font-size:12px}
.chip.b{color:#fff;background:var(--accent)}
.chip.m{background:var(--act)}
.chip.p{background:var(--pass)}
.run{display:flex;gap:10px;align-items:flex-start;overflow-x:auto;padding-bottom:8px}
.slot{display:flex;flex-direction:column;align-items:center;gap:4px;flex:0 0 auto;
  min-width:64px;max-width:132px}
.slot .cap{font-size:11px;color:var(--muted);text-align:center}
.slot .act{font-size:11px;padding:1px 7px;border-radius:5px;background:var(--chip);
  font-family:ui-monospace,Menlo,monospace;white-space:nowrap}
.mechs{display:flex;flex-direction:column;gap:2px;align-items:center}
.mech{font-size:10px;line-height:1.35;padding:0 6px;border-radius:4px;white-space:nowrap}
.arrow{align-self:center;color:var(--muted);flex:0 0 auto;padding-top:22px}
canvas{border:1px solid var(--line);border-radius:3px;display:block;image-rendering:pixelated}
.goalring canvas{outline:2px solid var(--good);outline-offset:1px}
.mech.scen{opacity:.45;font-style:italic}
.err{color:var(--bad);font-size:12px}
.legend{font-size:12px;color:var(--muted);margin-top:8px}
</style>
<div class="wrap">
  <h1>Compositional planning problems</h1>
  <p class="sub">Each problem replayed in the Autumn engine: the START grid, the
  ground-truth plan frame by frame, and the GOAL. Chips under a frame name the mechanics
  that first fire at that step &mdash; <span class="chip m">blue</span> = triggered by the
  input, <span class="chip p">green</span> = a clock rule. Goals are full exact frames; the
  final frame is ringed because it is the goal. Every problem is incompressible
  (<span class="mono">h_min == h</span>): no single action can be dropped and still reach
  the goal. Dimmed italic chips are <em>scenery</em> &mdash; rules that fire on their own
  every tick (n2ntd's patrolling enemy); they must still be predicted, but they are not
  chain links and do not count toward the mechanic total.</p>
  <div class="bar">
    <span><label>game</label><select id="fg"></select></span>
    <span><label>mechanic</label><select id="fm"></select></span>
    <span><label>horizon</label><select id="fh"></select></span>
    <span><label>sort</label><select id="fs">
      <option value="mech">most mechanics</option>
      <option value="h">longest horizon</option>
      <option value="dec">most decisions</option></select></span>
    <button id="prev">&#9664;</button><button id="next">&#9654;</button>
    <span class="count" id="count"></span>
  </div>
  <div id="list"></div>
</div>
<script>
const DATA = /*DATA*/{};
const PAL = DATA.palette, P = DATA.problems;
const CELL = 7, PAGE = 24;
let page = 0;
function draw(grid){
  const cv=document.createElement("canvas");
  if(!grid){cv.width=cv.height=CELL*4;return cv;}
  const R=grid.length,C=grid[0].length;cv.width=C*CELL;cv.height=R*CELL;
  const x=cv.getContext("2d");
  for(let r=0;r<R;r++)for(let c=0;c<C;c++){x.fillStyle=PAL[grid[r][c]]||"#333";
    x.fillRect(c*CELL,r*CELL,CELL,CELL);}
  return cv;
}
function slot(grid,cap,act,mechs,ring){
  const d=document.createElement("div");d.className="slot"+(ring?" goalring":"");
  if(act!=null){const a=document.createElement("div");a.className="act";a.textContent=act;
    d.appendChild(a);}
  d.appendChild(draw(grid));
  if(cap!=null){const s=document.createElement("div");s.className="cap";s.textContent=cap;
    d.appendChild(s);}
  if(mechs&&mechs.length){const m=document.createElement("div");m.className="mechs";
    mechs.forEach(x=>{const e=document.createElement("div");
      e.className="mech "+(x.a?"chip m":"chip p")+(x.s?" scen":"");
      e.textContent=x.m;m.appendChild(e);});
    d.appendChild(m);}
  return d;
}
function card(p){
  const c=document.createElement("div");c.className="card";
  const h=document.createElement("div");h.className="head";
  const bits=[["b",p.game+" / "+p.human],["","h="+p.h],["","decisions "+p.n_dec],
              ["","mechanics "+p.n_mech],["","random "+(p.rand??0).toFixed(2)]];
  if(p.prefix_len) bits.push(["","from human prefix ("+p.prefix_len+" steps)"]);
  bits.forEach(([k,t])=>{const s=document.createElement("span");
    s.className="chip "+k;s.textContent=t;h.appendChild(s);});
  c.appendChild(h);
  const ch=document.createElement("div");ch.className="head";
  ch.appendChild(Object.assign(document.createElement("span"),
    {className:"cap mono",style:"color:var(--muted);font-size:12px",
     textContent:p.chain.join("  →  ")}));
  c.appendChild(ch);
  if(p.err){const e=document.createElement("div");e.className="err";
    e.textContent="replay failed: "+p.err;c.appendChild(e);return c;}
  const run=document.createElement("div");run.className="run";
  run.appendChild(slot(p.start,"START",null,null,false));
  const ar=document.createElement("div");ar.className="arrow";ar.textContent="→";
  run.appendChild(ar);
  p.frames.forEach((f,i)=>{
    run.appendChild(slot(f,"t+"+(i+1),p.plan[i],p.steps[i],i===p.frames.length-1&&p.reached));
  });
  const ar2=document.createElement("div");ar2.className="arrow";ar2.textContent="=";
  run.appendChild(ar2);
  run.appendChild(slot(p.goal,"GOAL",null,null,true));
  c.appendChild(run);
  return c;
}
function opts(sel,vals,label){
  sel.innerHTML="";
  const a=document.createElement("option");a.value="";a.textContent="all "+label;
  sel.appendChild(a);
  vals.forEach(v=>{const o=document.createElement("option");o.value=v;o.textContent=v;
    sel.appendChild(o);});
}
const fg=document.getElementById("fg"),fm=document.getElementById("fm"),
      fh=document.getElementById("fh"),fs=document.getElementById("fs");
opts(fg,[...new Set(P.map(p=>p.game))].sort(),"games");
opts(fm,[...new Set(P.flatMap(p=>p.chain))].sort(),"mechanics");
opts(fh,[...new Set(P.map(p=>p.h))].sort((a,b)=>a-b),"horizons");
function filtered(){
  let s=P.filter(p=>(!fg.value||p.game===fg.value)
    &&(!fm.value||p.chain.includes(fm.value))
    &&(!fh.value||String(p.h)===fh.value));
  const k=fs.value;
  s=s.slice().sort((a,b)=>k==="h"?b.h-a.h:k==="dec"?b.n_dec-a.n_dec:b.n_mech-a.n_mech);
  return s;
}
function render(){
  const s=filtered();
  const pages=Math.max(1,Math.ceil(s.length/PAGE));
  page=Math.min(page,pages-1);
  const list=document.getElementById("list");list.innerHTML="";
  s.slice(page*PAGE,(page+1)*PAGE).forEach(p=>list.appendChild(card(p)));
  document.getElementById("count").textContent=
    s.length+" problems · page "+(page+1)+"/"+pages;
}
[fg,fm,fh,fs].forEach(e=>e.addEventListener("change",()=>{page=0;render();}));
document.getElementById("prev").onclick=()=>{page=Math.max(0,page-1);render();};
document.getElementById("next").onclick=()=>{page++;render();};
render();
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", default="")
    ap.add_argument("--limit", type=int, default=0, help="max problems per game (0 = all)")
    args = ap.parse_args()
    data = json.loads(Path(args.inp).read_text())
    games = {g for g in args.games.split(",") if g}
    enc = encode(data, games, args.limit)
    n_bad = sum(1 for p in enc["problems"] if p["err"] or not p["reached"])
    html = HTML.replace("/*DATA*/{}", json.dumps(enc, separators=(",", ":")))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out}  ({len(enc['problems'])} problems, "
          f"{n_bad} with replay/goal mismatch)")


if __name__ == "__main__":
    main()
