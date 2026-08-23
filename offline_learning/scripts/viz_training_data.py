#!/usr/bin/env python3
"""Filmstrip visualisation of the human TRAINING data (informative_unified) per game.

Two views, toggled per game:
  * "Full drives"  -- each recorded human session (drives/{train,test}_d*) as a
    horizontally-scrollable strip of frames, action-labelled, click frames ringed.
  * "Train/Test transitions" -- the individual (X_t, A_t, X_t+1) transitions the
    learner actually trains/tests on (load_transitions on {train,test}_d* with
    context backfilled from the drives, context_k=9), each shown with its K-step
    PRIOR context (ctx_prev) and NEXT context (ctx_next); the target transition
    X_t -> A_t -> X_t+1 is emphasised.

Grids are interned into a shared pool (referenced by index) so the repeated context
frames don't blow up the file.

    uv run python offline_learning/scripts/viz_training_data.py --out logs/training_data_viz.html
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "offline_learning"))
import human_replay as HR  # noqa: E402
from human_replay import GAMES as HGAMES, _grid  # noqa: E402
from validate import load_transitions, backfill_context_from_source  # noqa: E402

CACHE = REPO / "offline_learning/human_data/_cache"

csv.field_size_limit(10_000_000)
GAMES = ["bt3gb", "dq8gc", "n2ntd", "s2kt7", "83wkq"]
NAME = {"bt3gb": "ice", "dq8gc": "disease", "n2ntd": "mario", "s2kt7": "ants", "83wkq": "particles"}
CSS = {"black": "#0d1017", "white": "#eef1f5", "gray": "#8b929c", "blue": "#2f6fd0",
       "lightblue": "#8fc4e8", "gold": "#e0a92e", "darkgreen": "#2e8b57", "red": "#d1442f",
       "darkorange": "#e07b1a", "mediumpurple": "#9575cd", "green": "#3fae5a",
       "orange": "#e08a1a", "yellow": "#e6c34a"}


class Pool:
    """Intern unique grids -> index; palette built alongside."""
    def __init__(self):
        self.pal: dict[str, int] = {}
        self.pool: list = []
        self.idx: dict[str, int] = {}

    def _enc(self, grid_json: str):
        out = []
        for row in json.loads(grid_json):
            erow = []
            for name in row:
                if name not in self.pal:
                    self.pal[name] = len(self.pal)
                erow.append(self.pal[name])
            out.append(erow)
        return out

    def gid(self, obs_or_grid: str | None) -> int:
        if not obs_or_grid:
            return -1
        grid = _grid(obs_or_grid)
        if grid is None:
            return -1
        if grid not in self.idx:
            self.idx[grid] = len(self.pool)
            self.pool.append(self._enc(grid))
        return self.idx[grid]


def build_drives(game: str, pool: Pool, max_steps: int, cap: int) -> tuple[list[dict], list]:
    """Every human-play segment for the game (via human_replay), replayed to frames and
    marked train / test / unselected. The 6 train/test drives are the ones informative_unified
    selected; the rest are the unselected pool. Ordered train, test, then unselected."""
    root = REPO / f"offline_learning/human_data/{game}/informative_unified"
    man = json.loads((root / "MANIFEST.json").read_text())
    wl = set(man["whitelist"])
    prog = HGAMES[game][0]
    selmap = {}  # (user_id, seed, seg_idx) -> (split, dname)
    for split in ("train", "test"):
        for i, d in enumerate(man["drives"][split]):
            selmap[(d["user_id"], d["seed"], d["seg_idx"])] = (split, f"{split}_d{i}")

    sessions = HR.load_sessions(game, HR.DEFAULT_ZIP, CACHE)
    segs = [seg for s in sessions for seg in HR.segment(s, wl)]
    # train + test first, then unselected (optionally capped)
    sel = [s for s in segs if (s["user_id"], s["seed"], s["seg_idx"]) in selmap]
    uns = [s for s in segs if (s["user_id"], s["seed"], s["seg_idx"]) not in selmap]
    if cap:
        uns = uns[:cap]
    order = sorted(sel, key=lambda s: selmap[(s["user_id"], s["seed"], s["seg_idx"])][1]) + uns

    drives = []
    for seg in order:
        split, dname = selmap.get((seg["user_id"], seg["seed"], seg["seg_idx"]), ("unselected", None))
        rep = HR.replay(prog, seg["seed"], seg["actions"])
        grids, acts = rep["grids"], rep["actions"]
        frames, actions = [], []
        for i, g in enumerate(grids):
            gi = pool.gid(g)
            if gi < 0:
                break  # terminated / unparseable -> drive ends here
            frames.append(gi)
            actions.append(acts[i] if i < len(acts) else None)
            if max_steps and len(frames) >= max_steps:
                break
        if len(frames) < 2:
            continue
        nclk = sum(1 for a in actions if a and a.startswith("click"))
        drives.append({"split": split, "seed": seg["seed"], "task": seg["task_id"],
                       "user": seg["user_id"][:8], "seg_idx": seg["seg_idx"],
                       "drive": dname or f"seg{seg['seg_idx']}", "n_clicks": nclk,
                       "frames": frames, "actions": actions})
    print(f"  [{game}] {len(drives)} drives "
          f"({sum(d['split']=='train' for d in drives)} train, "
          f"{sum(d['split']=='test' for d in drives)} test, "
          f"{sum(d['split']=='unselected' for d in drives)} unselected)", flush=True)
    return drives, man["whitelist"]


def build_transitions(game: str, split: str, whitelist, pool: Pool) -> list[dict]:
    root = REPO / f"offline_learning/human_data/{game}/informative_unified"
    wl = set(whitelist)
    out = []
    for i in range(9):
        idd, drv = root / f"{split}_d{i}", root / "drives" / f"{split}_d{i}"
        if not idd.exists():
            continue
        trs = load_transitions([idd], wl, context_k=9)
        backfill_context_from_source(trs, [drv], wl, context_k=9)
        for tr in trs:
            out.append({
                "action": tr.action, "verb": tr.action.split()[0], "drive": f"{split}_d{i}",
                "ctx_prev": [[pool.gid(s), a] for s, a in tr.ctx_prev],
                "x_t": pool.gid(tr.x_t), "x_t1": pool.gid(tr.x_t1),
                "ctx_next": [[a, pool.gid(s)] for a, s in tr.ctx_next]})
    return out


def encode(games: set[str], max_steps: int, cap: int) -> dict:
    pool = Pool()
    out_games = []
    for g in GAMES:
        if games and g not in games:
            continue
        drives, whitelist = build_drives(g, pool, max_steps, cap)
        out_games.append({
            "game": g, "human": HGAMES[g][1], "drives": drives,
            "train": build_transitions(g, "train", whitelist, pool),
            "test": build_transitions(g, "test", whitelist, pool)})
    return {"palette": {i: CSS.get(n, n) for n, i in pool.pal.items()},
            "pool": pool.pool, "games": out_games}


HTML = r"""<!doctype html><meta charset="utf-8">
<title>Training data — human drives & transitions</title>
<style>
:root{--paper:#f4f5f2;--ink:#1b1f27;--muted:#5c636e;--line:#d9dcd6;--card:#fff;
  --accent:#3a6ea5;--click:#e0a92e;--chip:#eceee9;--move:#2c7a68;--tgt:#3a6ea5;}
@media (prefers-color-scheme:dark){:root{--paper:#0f1217;--ink:#e6e9ef;--muted:#98a0ad;
  --line:#242a33;--card:#161b22;--accent:#6ea8dc;--click:#c8a24e;--chip:#1d232c;--move:#4bbfa3;--tgt:#6ea8dc;}}
:root[data-theme=dark]{--paper:#0f1217;--ink:#e6e9ef;--muted:#98a0ad;--line:#242a33;
  --card:#161b22;--accent:#6ea8dc;--click:#c8a24e;--chip:#1d232c;--move:#4bbfa3;--tgt:#6ea8dc;}
:root[data-theme=light]{--paper:#f4f5f2;--ink:#1b1f27;--muted:#5c636e;--line:#d9dcd6;
  --card:#fff;--accent:#3a6ea5;--click:#e0a92e;--chip:#eceee9;--move:#2c7a68;--tgt:#3a6ea5;}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font:14px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif}
.wrap{max-width:1400px;margin:0 auto;padding:24px 20px 80px}
h1{font:600 22px/1.2 Georgia,serif;margin:0 0 4px}
.sub{color:var(--muted);margin:0 0 18px;max-width:920px}
.mono{font-family:ui-monospace,Menlo,Consolas,monospace}
.bar{position:sticky;top:0;z-index:5;background:var(--paper);border-bottom:1px solid var(--line);
  display:flex;flex-wrap:wrap;gap:12px;align-items:center;padding:12px 0;margin-bottom:16px}
.bar label{font-size:12px;color:var(--muted);margin-right:4px}
select{font:inherit;color:var(--ink);background:var(--card);border:1px solid var(--line);
  border-radius:7px;padding:5px 8px}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:7px;overflow:hidden}
.seg button{font:inherit;color:var(--ink);background:var(--card);border:0;border-right:1px solid var(--line);
  padding:5px 10px;cursor:pointer}
.seg button:last-child{border-right:0}
.seg button.on{background:var(--accent);color:#fff}
.chk{font-size:12px;color:var(--muted);display:flex;align-items:center;gap:5px}
.count{margin-left:auto;color:var(--muted);font-size:12px}
.drive,.tr{background:var(--card);border:1px solid var(--line);border-radius:12px;
  padding:12px 14px;margin-bottom:14px}
.dh{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:8px}
.chip{background:var(--chip);border-radius:999px;padding:2px 10px;font-size:12px}
.chip.tr{color:#fff;background:var(--accent)}.chip.te{color:#fff;background:#7a6a9c}
.chip.click{color:#fff;background:var(--click)}.chip.move{color:#fff;background:var(--move)}
.strip{display:flex;gap:6px;overflow-x:auto;padding:4px 0 8px;align-items:flex-start}
.f{display:flex;flex-direction:column;align-items:center;gap:2px;flex:0 0 auto}
.f canvas{border:1px solid var(--line);border-radius:2px;display:block;image-rendering:pixelated}
.f.click canvas{outline:2px solid var(--click);outline-offset:1px}
.f.tgt canvas{outline:2px solid var(--tgt);outline-offset:1px}
.f.ctx{opacity:.6}
.a{font-size:9px;color:var(--muted);max-width:64px;text-align:center;white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis}
.a.click{color:var(--click);font-weight:600}.a.move{color:var(--move);font-weight:600}
.a.tgt{color:var(--tgt);font-weight:700;font-size:10px}
.idx{font-size:8px;color:var(--muted);opacity:.6}
.sep{align-self:stretch;border-left:1px dashed var(--line);margin:0 3px}
.arrow{align-self:center;display:flex;flex-direction:column;align-items:center;gap:2px;padding:0 2px}
.arrow .v{font-size:11px;font-weight:700}
</style>
<div class="wrap">
  <h1>Training data — human drives &amp; the transitions used</h1>
  <p class="sub"><b>Full drives</b>: each recorded human session. <b>Train / Test transitions</b>:
  the individual <span class="mono">(X_t, A_t, X_t+1)</span> examples the learner trains / tests on
  (K=9 prior context + 8 next), with the target transition <span class="mono">X_t → A_t → X_t+1</span>
  ringed in blue and its action shown; context frames are dimmed. Click actions are gold, moves green.</p>
  <div class="bar">
    <span><label>game</label><select id="fg"></select></span>
    <span class="seg" id="mode">
      <button data-m="drives" class="on">Full drives</button>
      <button data-m="train">Train transitions</button>
      <button data-m="test">Test transitions</button>
    </span>
    <span class="chk" id="setwrap"><label>set</label><select id="fs">
      <option value="">all</option><option value="train">train</option>
      <option value="test">test</option><option value="unselected">unselected</option></select></span>
    <span class="chk" id="verbwrap"><label>action</label><select id="fv"></select></span>
    <span class="seg" id="pager"><button id="prev">◀</button><button id="next">▶</button></span>
    <span class="count" id="count"></span>
  </div>
  <div id="list"></div>
</div>
<script>
const DATA=/*DATA*/{};
const PAL=DATA.palette, POOL=DATA.pool, G=DATA.games;
let MODE="drives";
function draw(pidx,cell){
  const cv=document.createElement("canvas");const CELL=cell||6;
  const grid=pidx>=0?POOL[pidx]:null;
  if(!grid){cv.width=cv.height=CELL*3;return cv;}
  const R=grid.length,C=grid[0].length;cv.width=C*CELL;cv.height=R*CELL;
  const x=cv.getContext("2d");
  for(let r=0;r<R;r++)for(let c=0;c<C;c++){x.fillStyle=PAL[grid[r][c]]||"#333";
    x.fillRect(c*CELL,r*CELL,CELL,CELL);}
  return cv;
}
function vclass(a){if(!a)return"";const v=a.split(" ")[0];
  return v=="click"?"click":["up","down","left","right"].includes(v)?"move":"";}
function frame(pidx,label,cls,cell){
  const f=document.createElement("div");f.className="f"+(cls?" "+cls:"");
  f.appendChild(draw(pidx,cell));
  if(label!=null){const l=document.createElement("div");l.className="a "+(cls&&cls.includes("tgt")?"tgt ":"")+vclass(label);
    l.textContent=label?(label.startsWith("click")?label.replace("click ","⊕"):label.split(" ")[0]):"·";
    l.title=label||"";f.appendChild(l);}
  return f;
}
// ---- full-drive card ----
function renderDrive(d){
  const box=document.createElement("div");box.className="drive";
  const h=document.createElement("div");h.className="dh";
  const cl=d.split=="train"?"tr":d.split=="test"?"te":"";
  const sp=document.createElement("span");sp.className="chip "+cl;sp.textContent=d.split;h.appendChild(sp);
  [["seed "+d.seed],["seg "+d.seg_idx],["task "+d.task],["user "+d.user],
   [d.frames.length+" frames"],[d.n_clicks+" clicks"]].forEach(([t])=>{const c=document.createElement("span");
    c.className="chip";c.textContent=t;h.appendChild(c);});
  box.appendChild(h);
  const strip=document.createElement("div");strip.className="strip";
  d.frames.forEach((p,i)=>{
    const a=d.actions[i];const f=frame(p,a,vclass(a));
    const ix=document.createElement("div");ix.className="idx";ix.textContent=i;f.appendChild(ix);
    strip.appendChild(f);});
  box.appendChild(strip);return box;
}
// ---- transition card ----
function renderTr(t,i){
  const box=document.createElement("div");box.className="tr";
  const h=document.createElement("div");h.className="dh";
  const nm=document.createElement("span");nm.className="chip tr";nm.textContent="#"+i;h.appendChild(nm);
  const ac=document.createElement("span");ac.className="chip "+(vclass(t.action)||"");
  ac.textContent="A_t = "+t.action;h.appendChild(ac);
  const dr=document.createElement("span");dr.className="chip";dr.textContent=t.drive;h.appendChild(dr);
  const cp=document.createElement("span");cp.className="chip";cp.textContent="ctx "+t.ctx_prev.length+"←/→"+t.ctx_next.length;h.appendChild(cp);
  box.appendChild(h);
  const strip=document.createElement("div");strip.className="strip";
  t.ctx_prev.forEach(([p,a])=>strip.appendChild(frame(p,a,"ctx "+vclass(a),5)));
  const s1=document.createElement("div");s1.className="sep";strip.appendChild(s1);
  strip.appendChild(frame(t.x_t,"X_t","tgt",8));
  const ar=document.createElement("div");ar.className="arrow";
  const v=document.createElement("div");v.className="v "+vclass(t.action);
  v.textContent="→"+(t.action.startsWith("click")?t.action.replace("click ","⊕"):t.action.split(" ")[0])+"→";
  ar.appendChild(v);strip.appendChild(ar);
  strip.appendChild(frame(t.x_t1,"X_t+1","tgt",8));
  const s2=document.createElement("div");s2.className="sep";strip.appendChild(s2);
  t.ctx_next.forEach(([a,p])=>strip.appendChild(frame(p,a,"ctx "+vclass(a),5)));
  box.appendChild(strip);return box;
}
const fg=document.getElementById("fg"),fv=document.getElementById("fv"),
  fs=document.getElementById("fs"),verbwrap=document.getElementById("verbwrap"),
  setwrap=document.getElementById("setwrap"),pager=document.getElementById("pager");
let page=0;const PER_DRIVES=8;
G.forEach(gg=>{const o=document.createElement("option");o.value=gg.game;
  o.textContent=gg.game+" / "+gg.human;fg.appendChild(o);});
function curGame(){return G.find(x=>x.game==fg.value)||G[0];}
function refreshVerbs(){
  const g=curGame();fv.innerHTML="";
  const o=document.createElement("option");o.value="";o.textContent="all actions";fv.appendChild(o);
  if(MODE=="drives")return;
  [...new Set(g[MODE].map(t=>t.verb))].sort().forEach(v=>{
    const e=document.createElement("option");e.value=v;e.textContent=v;fv.appendChild(e);});
}
function update(){
  const g=curGame();const list=document.getElementById("list");list.innerHTML="";
  const dm=MODE=="drives";
  verbwrap.style.display=dm?"none":"";setwrap.style.display=dm?"":"none";pager.style.display=dm?"":"none";
  if(dm){
    const items=fs.value?g.drives.filter(d=>d.split==fs.value):g.drives;
    const pages=Math.max(1,Math.ceil(items.length/PER_DRIVES));
    page=Math.max(0,Math.min(page,pages-1));
    items.slice(page*PER_DRIVES,page*PER_DRIVES+PER_DRIVES).forEach(d=>list.appendChild(renderDrive(d)));
    const n={train:0,test:0,unselected:0};g.drives.forEach(d=>n[d.split]++);
    document.getElementById("count").textContent=
      items.length+" drives ("+n.train+" train / "+n.test+" test / "+n.unselected+" unselected) · page "+(page+1)+"/"+pages;
  } else {
    let items=g[MODE];if(fv.value)items=items.filter(t=>t.verb==fv.value);
    items.forEach((t,i)=>list.appendChild(renderTr(t,i)));
    const byv={};g[MODE].forEach(t=>byv[t.verb]=(byv[t.verb]||0)+1);
    document.getElementById("count").textContent=
      items.length+"/"+g[MODE].length+" "+MODE+" transitions · "+JSON.stringify(byv).replace(/"/g,"");
  }
}
document.querySelectorAll("#mode button").forEach(b=>b.onclick=()=>{
  document.querySelectorAll("#mode button").forEach(x=>x.classList.remove("on"));
  b.classList.add("on");MODE=b.dataset.m;page=0;refreshVerbs();update();});
fg.addEventListener("change",()=>{page=0;refreshVerbs();update();});
fv.addEventListener("change",update);
fs.addEventListener("change",()=>{page=0;update();});
document.getElementById("prev").onclick=()=>{page--;update();};
document.getElementById("next").onclick=()=>{page++;update();};
refreshVerbs();update();
</script>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "logs/training_data_viz.html"))
    ap.add_argument("--games", default="", help="comma-separated subset")
    ap.add_argument("--max-steps", type=int, default=0, help="cap drive frames (0=all)")
    ap.add_argument("--unselected-cap", type=int, default=0,
                    help="cap unselected drives per game (0=all)")
    args = ap.parse_args()
    games = set(filter(None, args.games.split(",")))
    enc = encode(games, args.max_steps, args.unselected_cap)
    html = HTML.replace("/*DATA*/{}", json.dumps(enc))
    Path(args.out).write_text(html, encoding="utf-8")
    nd = sum(len(g["drives"]) for g in enc["games"])
    ntr = sum(len(g["train"]) + len(g["test"]) for g in enc["games"])
    print(f"{len(enc['games'])} games, {nd} drives, {ntr} transitions, "
          f"{len(enc['pool'])} pooled grids, {len(enc['palette'])} colours -> "
          f"{args.out} ({len(html)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
