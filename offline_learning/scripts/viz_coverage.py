#!/usr/bin/env python3
"""Unified inspection viz for the coverage exam, per game, two modes:

  * "ID exam"  -- the labelled anchors (coverage/mechanics.json): each is one inverse-dynamics
    test item, shown as  [K prior context] -> X_t  --action-->  X_t+1  (both ringed), with the
    mechanic / kind / board population / synthetic flag.
  * "Planning" -- the multi-horizon planning problems (coverage_plan_problems*.json) built from
    those anchors:  [K prior context] -> START --gt--> GOAL  (goal-reaching frame ringed), with
    bucket / mechanic / horizon / population / noop+random baselines.

Both modes show the SAME K=9 prior-context strip (dimmed) the eval feeds the model. Grids are
interned into a shared pool. Filter by game / mechanic / population band (+ bucket / horizon in
planning mode).

    uv run python offline_learning/scripts/viz_coverage.py \
        --problems logs/coverage_plan_problems_v2.json --out logs/coverage_viz.html
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "offline_learning"))
from human_replay import GAMES as HGAMES  # noqa: E402
from coverage_plan import exec_plan, load_coverage  # noqa: E402
from mechanics import _BG  # noqa: E402

GAMES = ["bt3gb", "dq8gc", "n2ntd", "s2kt7", "83wkq"]
CONTEXT_K = 9
CSS = {"black": "#0d1017", "white": "#eef1f5", "gray": "#8b929c", "blue": "#2f6fd0",
       "lightblue": "#8fc4e8", "gold": "#e0a92e", "darkgreen": "#2e8b57", "red": "#d1442f",
       "darkorange": "#e07b1a", "mediumpurple": "#9575cd", "green": "#3fae5a",
       "orange": "#e08a1a", "yellow": "#e6c34a"}


def band(p: int) -> str:
    return "empty" if p == 0 else "sparse(1-5)" if p <= 5 else "medium(6-15)" if p <= 15 else "dense(16+)"


class Pool:
    def __init__(self):
        self.pal, self.pool, self.idx = {}, [], {}

    def gid(self, grid_json):
        if not grid_json:
            return -1
        if grid_json not in self.idx:
            enc = []
            for row in json.loads(grid_json):
                er = []
                for name in row:
                    if name not in self.pal:
                        self.pal[name] = len(self.pal)
                    er.append(self.pal[name])
                enc.append(er)
            self.idx[grid_json] = len(self.pool)
            self.pool.append(enc)
        return self.idx[grid_json]


def population(grid_json, bg):
    return sum(1 for row in json.loads(grid_json) for c in row if c not in bg) if grid_json else 0


def context(cov, seed, step, pool):
    """K prior (grid, action) pairs ending just before `step` -- exactly what the eval builds."""
    d = cov["drives_by_seed"].get(seed)
    if not d:
        return []
    grids, acts = d["grids"], d["actions"]
    ctx = []
    for j in range(step - 1, max(-1, step - 1 - CONTEXT_K), -1):
        if j < 0 or grids[j] is None or not acts[j]:
            break
        ctx.insert(0, [pool.gid(grids[j]), acts[j]])
    return ctx


def build_id(game, pool):
    cov = load_coverage(game)
    bg = _BG.get(game, set())
    mj = REPO / f"offline_learning/human_data/{game}/coverage/mechanics.json"
    if not mj.exists():
        return []
    out = []
    for l in json.loads(mj.read_text()):
        src = l["src"]
        seed, step = src.get("seed"), src.get("step")
        d = cov["drives_by_seed"].get(seed)
        if d is None or step is None or step + 1 >= len(d["grids"]):
            continue
        grids, acts = d["grids"], d["actions"]
        if grids[step] is None or grids[step + 1] is None:
            continue
        pop = population(grids[step], bg)
        out.append({"mechanic": l["mechanic"], "kind": l.get("kind"),
                    "synthetic": l.get("synthetic"), "action": l["action"],
                    "pop": pop, "band": band(pop), "ctx": context(cov, seed, step, pool),
                    "x_t": pool.gid(grids[step]), "x_t1": pool.gid(grids[step + 1])})
    return out


def build_plan(game, problems, pool):
    cov = load_coverage(game)
    bg = _BG.get(game, set())
    prog = cov["program"]
    out = []
    for p in problems:
        if p["game"] != game:
            continue
        seed, t = p["seed"], p["t"]
        gt = p.get("gt_actions") or []
        grids = None
        try:
            prefix = cov["drives_by_seed"][seed]["actions"][:t]
            grids = exec_plan(prog, seed, prefix, gt)
        except Exception:  # noqa: BLE001
            grids = None
        pop = population(p["start_grid"], bg)
        out.append({"bucket": p["bucket"], "mechanic": p["mechanic"], "h": p["h"],
                    "pop": pop, "band": band(pop), "noop": p.get("noop_success"),
                    "rand": p.get("random_success"), "gt": gt,
                    "ctx": context(cov, seed, t, pool),
                    "start": pool.gid(p["start_grid"]), "goal": pool.gid(p["goal_grid"]),
                    "frames": [pool.gid(g) for g in grids] if grids else None})
    return out


def encode(problems_data, games):
    pool = Pool()
    out = []
    probs = problems_data.get("problems", []) if problems_data else []
    for g in GAMES:
        if games and g not in games:
            continue
        out.append({"game": g, "human": HGAMES[g][1],
                    "id": build_id(g, pool), "plan": build_plan(g, probs, pool)})
    return {"palette": {i: CSS.get(n, n) for n, i in pool.pal.items()}, "pool": pool.pool, "games": out}


HTML = r"""<!doctype html><meta charset="utf-8">
<title>Coverage exam & planning problems</title>
<style>
:root{--paper:#f4f5f2;--ink:#1b1f27;--muted:#5c636e;--line:#d9dcd6;--card:#fff;
  --accent:#3a6ea5;--good:#2c7a68;--chip:#eceee9;--warn:#9c7a32;--tgt:#3a6ea5;}
@media (prefers-color-scheme:dark){:root{--paper:#0f1217;--ink:#e6e9ef;--muted:#98a0ad;--line:#242a33;
  --card:#161b22;--accent:#6ea8dc;--good:#4bbfa3;--chip:#1d232c;--warn:#c8a24e;--tgt:#6ea8dc;}}
:root[data-theme=dark]{--paper:#0f1217;--ink:#e6e9ef;--muted:#98a0ad;--line:#242a33;--card:#161b22;
  --accent:#6ea8dc;--good:#4bbfa3;--chip:#1d232c;--warn:#c8a24e;--tgt:#6ea8dc;}
:root[data-theme=light]{--paper:#f4f5f2;--ink:#1b1f27;--muted:#5c636e;--line:#d9dcd6;--card:#fff;
  --accent:#3a6ea5;--good:#2c7a68;--chip:#eceee9;--warn:#9c7a32;--tgt:#3a6ea5;}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.5 ui-sans-serif,system-ui,sans-serif}
.wrap{max-width:1360px;margin:0 auto;padding:24px 20px 80px}
h1{font:600 22px/1.2 Georgia,serif;margin:0 0 4px}
.sub{color:var(--muted);margin:0 0 18px;max-width:940px}
.mono{font-family:ui-monospace,Menlo,Consolas,monospace}
.bar{position:sticky;top:0;z-index:5;background:var(--paper);border-bottom:1px solid var(--line);
  display:flex;flex-wrap:wrap;gap:10px 14px;align-items:center;padding:12px 0;margin-bottom:16px}
.bar label{font-size:12px;color:var(--muted);margin-right:4px}
select,button{font:inherit;color:var(--ink);background:var(--card);border:1px solid var(--line);border-radius:7px;padding:5px 8px}
button{cursor:pointer}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:7px;overflow:hidden}
.seg button{border:0;border-right:1px solid var(--line);border-radius:0;padding:5px 12px}
.seg button:last-child{border-right:0}.seg button.on{background:var(--accent);color:#fff}
.count{margin-left:auto;color:var(--muted);font-size:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin-bottom:14px}
.head{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:8px}
.chip{background:var(--chip);border-radius:999px;padding:2px 10px;font-size:12px}
.chip.b{color:#fff;background:var(--accent)}.chip.pop{color:#fff;background:var(--warn)}
.strip{display:flex;gap:6px;overflow-x:auto;padding:4px 0;align-items:flex-start}
.f{display:flex;flex-direction:column;align-items:center;gap:2px;flex:0 0 auto}
.f canvas{border:1px solid var(--line);border-radius:2px;display:block;image-rendering:pixelated}
.f.ctx{opacity:.55}.f.tgt canvas{outline:2px solid var(--tgt);outline-offset:1px}
.f.hit canvas{outline:2px solid var(--good);outline-offset:1px}
.a{font-size:9px;color:var(--muted);max-width:70px;text-align:center;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.a.tgt{color:var(--tgt);font-weight:700}
.ctxlbl{writing-mode:vertical-rl;font-size:9px;color:var(--muted);align-self:center;letter-spacing:1px}
.sep{align-self:stretch;border-left:1px dashed var(--line);margin:0 4px}
.arrow{align-self:center;display:flex;flex-direction:column;align-items:center;font-size:11px;font-weight:700;color:var(--muted);padding:0 3px}
.err{color:#d1442f;font-size:12px}
</style>
<div class="wrap">
  <h1>Coverage exam &amp; planning problems</h1>
  <p class="sub"><b>ID exam</b>: each labelled anchor as an inverse-dynamics item —
  <span class="mono">…context → X_t →A→ X_t+1</span>. <b>Planning</b>: the multi-horizon problems
  built from those anchors — <span class="mono">…context → START →gt→ GOAL</span>. The dimmed
  left frames are the <b>K=9 prior context</b> the eval feeds the model; the blue-ringed frames are
  the item itself, the green ring is where GOAL is reached. <span class="chip pop" style="padding:1px 6px">pop</span>
  = board population of the anchor/START.</p>
  <div class="bar">
    <span><label>game</label><select id="fg"></select></span>
    <span class="seg" id="mode"><button data-m="id" class="on">ID exam</button><button data-m="plan">Planning</button></span>
    <span><label>mechanic</label><select id="fm"></select></span>
    <span class="pl"><label>bucket</label><select id="fb"></select></span>
    <span class="pl"><label>horizon</label><select id="fh"></select></span>
    <span><label>population</label><select id="fp">
      <option value="">any</option><option value="empty">empty</option><option value="sparse(1-5)">sparse</option>
      <option value="medium(6-15)">medium</option><option value="dense(16+)">dense</option></select></span>
    <button id="prev">◀</button><button id="next">▶</button>
    <span class="count" id="count"></span>
  </div>
  <div id="list"></div>
</div>
<script>
const DATA=/*DATA*/{};const PAL=DATA.palette,POOL=DATA.pool,G=DATA.games;
let MODE="id",page=0;const PER=8;
function draw(pidx,cell){
  const cv=document.createElement("canvas"),CELL=cell||7,grid=pidx>=0?POOL[pidx]:null;
  if(!grid){cv.width=cv.height=CELL*3;return cv;}
  const R=grid.length,C=grid[0].length;cv.width=C*CELL;cv.height=R*CELL;const x=cv.getContext("2d");
  for(let r=0;r<R;r++)for(let c=0;c<C;c++){x.fillStyle=PAL[grid[r][c]]||"#333";x.fillRect(c*CELL,r*CELL,CELL,CELL);}
  return cv;
}
function frame(pidx,cap,cls,cell){
  const f=document.createElement("div");f.className="f"+(cls?" "+cls:"");f.appendChild(draw(pidx,cell));
  if(cap!=null){const s=document.createElement("div");s.className="a"+(cls&&cls.includes("tgt")?" tgt":"");s.textContent=cap;s.title=cap;f.appendChild(s);}
  return f;
}
function ctxStrip(strip,ctx){
  if(!ctx.length){const n=document.createElement("span");n.className="a";n.textContent="(no prior context)";strip.appendChild(n);return;}
  const lbl=document.createElement("div");lbl.className="ctxlbl";lbl.textContent="context";strip.appendChild(lbl);
  ctx.forEach(([p,a],i)=>strip.appendChild(frame(p,a?(a.startsWith("click")?a.replace("click ","⊕"):a.split(" ")[0]):"·","ctx",5)));
  const s=document.createElement("div");s.className="sep";strip.appendChild(s);
}
function fmt(v){return v==null?"—":(+v).toFixed(2);}
function renderID(t){
  const c=document.createElement("div");c.className="card";
  const h=document.createElement("div");h.className="head";
  const bits=[["b",t.mechanic],["",t.kind],["pop","pop "+t.pop+" ("+t.band+")"],["","A = "+t.action]];
  if(t.synthetic)bits.push(["","synthetic"]);
  bits.forEach(([cl,x])=>{const s=document.createElement("span");s.className="chip"+(cl?" "+cl:"");s.textContent=x;h.appendChild(s);});
  c.appendChild(h);
  const strip=document.createElement("div");strip.className="strip";
  ctxStrip(strip,t.ctx);
  strip.appendChild(frame(t.x_t,"X_t","tgt",8));
  const ar=document.createElement("div");ar.className="arrow";ar.textContent="→"+(t.action.startsWith("click")?t.action.replace("click ","⊕"):t.action.split(" ")[0])+"→";strip.appendChild(ar);
  strip.appendChild(frame(t.x_t1,"X_t+1","tgt",8));
  c.appendChild(strip);return c;
}
function renderPlan(p){
  const c=document.createElement("div");c.className="card";
  const h=document.createElement("div");h.className="head";
  [["b",p.bucket],["",p.mechanic],["","h="+p.h],["pop","pop "+p.pop+" ("+p.band+")"],
   ["","noop "+fmt(p.noop)],["","rand "+fmt(p.rand)]].forEach(([cl,x])=>{
    const s=document.createElement("span");s.className="chip"+(cl?" "+cl:"");s.textContent=x;h.appendChild(s);});
  c.appendChild(h);
  const strip=document.createElement("div");strip.className="strip";
  ctxStrip(strip,p.ctx);
  strip.appendChild(frame(p.start,"START","tgt",8));
  if(p.frames){
    p.frames.forEach((g,i)=>{
      const ar=document.createElement("div");ar.className="arrow";ar.textContent="›";strip.appendChild(ar);
      strip.appendChild(frame(g,(i+1)+". "+(p.gt[i]||"?"),g===p.goal?"hit":"",8));
    });
  }else{const e=document.createElement("div");e.className="err";e.textContent="(gt exec failed)";strip.appendChild(e);}
  const ar2=document.createElement("div");ar2.className="arrow";ar2.textContent="= GOAL";strip.appendChild(ar2);
  strip.appendChild(frame(p.goal,"GOAL","tgt",8));
  c.appendChild(strip);return c;
}
const fg=document.getElementById("fg"),fm=document.getElementById("fm"),fb=document.getElementById("fb"),
  fh=document.getElementById("fh"),fp=document.getElementById("fp");
G.forEach(g=>{const o=document.createElement("option");o.value=g.game;o.textContent=g.game+" / "+g.human;fg.appendChild(o);});
function cur(){return G.find(x=>x.game==fg.value)||G[0];}
function items(){return cur()[MODE]||[];}
function opts(sel,vals,all){sel.innerHTML="";const o=document.createElement("option");o.value="";o.textContent=all;sel.appendChild(o);
  [...new Set(vals)].sort().forEach(v=>{const e=document.createElement("option");e.value=v;e.textContent=v;sel.appendChild(e);});}
function refresh(){
  opts(fm,items().map(x=>x.mechanic),"all mechanics");
  document.querySelectorAll(".pl").forEach(e=>e.style.display=MODE=="plan"?"":"none");
  if(MODE=="plan"){opts(fb,items().map(x=>x.bucket),"all buckets");opts(fh,items().map(x=>String(x.h)),"all h");}
}
function match(x){
  if(fm.value&&x.mechanic!=fm.value)return false;
  if(fp.value&&x.band!=fp.value)return false;
  if(MODE=="plan"){if(fb.value&&x.bucket!=fb.value)return false;if(fh.value&&String(x.h)!=fh.value)return false;}
  return true;
}
function update(){
  const sel=items().filter(match);
  const pages=Math.max(1,Math.ceil(sel.length/PER));page=Math.max(0,Math.min(page,pages-1));
  const list=document.getElementById("list");list.innerHTML="";
  sel.slice(page*PER,page*PER+PER).forEach(x=>list.appendChild(MODE=="id"?renderID(x):renderPlan(x)));
  const bands={};sel.forEach(x=>bands[x.band]=(bands[x.band]||0)+1);
  document.getElementById("count").textContent=sel.length+" "+MODE+" items · page "+(page+1)+"/"+pages+" · "+JSON.stringify(bands).replace(/"/g,"");
}
document.querySelectorAll("#mode button").forEach(b=>b.onclick=()=>{
  document.querySelectorAll("#mode button").forEach(x=>x.classList.remove("on"));b.classList.add("on");
  MODE=b.dataset.m;page=0;refresh();update();});
fg.addEventListener("change",()=>{page=0;refresh();update();});
[fm,fb,fh,fp].forEach(el=>el.addEventListener("change",()=>{page=0;update();}));
document.getElementById("prev").onclick=()=>{page--;update();};
document.getElementById("next").onclick=()=>{page++;update();};
refresh();update();
</script>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default=str(REPO / "logs/coverage_plan_problems_v2.json"))
    ap.add_argument("--out", default=str(REPO / "logs/coverage_viz.html"))
    ap.add_argument("--games", default="")
    args = ap.parse_args()
    pdata = json.loads(Path(args.problems).read_text()) if Path(args.problems).exists() else {"problems": []}
    games = set(filter(None, args.games.split(",")))
    enc = encode(pdata, games)
    html = HTML.replace("/*DATA*/{}", json.dumps(enc))
    Path(args.out).write_text(html, encoding="utf-8")
    for g in enc["games"]:
        ib = Counter(x["band"] for x in g["id"])
        pb = Counter(x["band"] for x in g["plan"])
        print(f"  {g['game']:6} ID anchors={len(g['id'])} {dict(ib)} | planning={len(g['plan'])} {dict(pb)}")
    print(f"{len(enc['pool'])} pooled grids -> {args.out} ({len(html)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
