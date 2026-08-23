#!/usr/bin/env python3
"""Render the plans generated during the coverage planning eval as grid filmstrips.

Reads an ENRICHED eval json (eval_coverage_plan.py stores per-arm `plan` + executed
`grids` + start/goal per problem) and emits a self-contained, filterable HTML page: for
each problem it draws the start grid, the goal grid, and -- for the correct trajectory
and each evaluated arm (raw / lmwm / wc) -- the plan as a filmstrip of rendered frames
(start -> after each action), labelling the action under each frame, ringing the frame
that first equals the goal, and badging success/failure. Filter by game / bucket /
mechanic / horizon / outcome.

    uv run python offline_learning/scripts/viz_coverage_plan.py \
        --in logs/coverage_plan_eval.json --out logs/coverage_plan_viz.html
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

MODEL_ARMS = ["raw", "lmwm", "wc"]
# color-name -> css (faithful but legible on a card); unknown names pass through as-is.
CSS = {"black": "#0d1017", "white": "#eef1f5", "gray": "#8b929c", "blue": "#2f6fd0",
       "lightblue": "#8fc4e8", "gold": "#e0a92e", "darkgreen": "#2e8b57", "red": "#d1442f",
       "darkorange": "#e07b1a", "mediumpurple": "#9575cd", "green": "#3fae5a",
       "orange": "#e08a1a", "yellow": "#e6c34a"}


def encode(data: dict, games: set[str], limit: int) -> dict:
    """Flatten the eval json into viz problems, mapping colours to a palette index."""
    pal: dict[str, int] = {}

    def enc(grid_json: str | None):
        if not grid_json:
            return None
        g = json.loads(grid_json)
        out = []
        for row in g:
            erow = []
            for name in row:
                if name not in pal:
                    pal[name] = len(pal)
                erow.append(pal[name])
            out.append(erow)
        return out

    problems = []
    per_game: dict[str, int] = {}
    coverage: dict[str, dict] = {}
    for res in data["results"]:
        for r in res["rows"]:
            if games and r["game"] not in games:
                continue
            if "start_grid" not in r or "raw" not in r or "grids" not in r.get("raw", {}):
                continue  # old (pre-enrichment) schema: no plan data
            if limit and per_game.get(r["game"], 0) >= limit:
                continue
            per_game[r["game"]] = per_game.get(r["game"], 0) + 1
            arms = {}
            gt = r.get("gt_actions") or []
            correct_grids = None
            correct_error = None
            if gt:
                try:
                    if r["game"] not in coverage:
                        coverage[r["game"]] = load_coverage(r["game"])
                    game_cov = coverage[r["game"]]
                    drive = game_cov["drives_by_seed"][r["seed"]]
                    prefix = drive["actions"][:r["t"]]
                    correct_grids = exec_plan(game_cov["program"], r["seed"], prefix, gt)
                except Exception as exc:  # keep the rest of the report usable
                    correct_error = f"correct-plan-execution:{type(exc).__name__}:{exc}"
            reached = next((i + 1 for i, g in enumerate(correct_grids or [])
                            if g == r["goal_grid"]), None)
            arms["correct"] = {
                "success": reached is not None,
                "reached_at": reached,
                "plan": gt or None,
                "plan_error": correct_error if gt else "no-ground-truth-plan",
                "grids": [enc(g) for g in correct_grids] if correct_grids else None,
            }
            for a in MODEL_ARMS:
                d = r.get(a, {})
                arms[a] = {"success": d.get("success"), "reached_at": d.get("reached_at"),
                           "plan": d.get("plan"), "plan_error": d.get("plan_error"),
                           "grids": [enc(g) for g in d["grids"]] if d.get("grids") else None}
            problems.append({
                "game": r["game"], "human": HGAMES[r["game"]][1], "bucket": r["bucket"],
                "mechanic": r["mechanic"], "h": r["h"], "seed": r["seed"], "t": r["t"],
                "gt": r.get("gt_actions"), "noop": r.get("noop_success"),
                "rand": r.get("random_success"),
                "start": enc(r["start_grid"]), "goal": enc(r["goal_grid"]), "arms": arms})
    palette = {i: CSS.get(name, name) for name, i in pal.items()}
    return {"palette": palette, "problems": problems}


HTML = r"""<!doctype html><meta charset="utf-8">
<title>Coverage planning — plan filmstrips</title>
<style>
:root{
  --paper:#f4f5f2; --ink:#1b1f27; --muted:#5c636e; --line:#d9dcd6;
  --card:#ffffff; --accent:#3a6ea5; --good:#2c7a68; --bad:#a94f38;
  --chip:#eceee9;
}
@media (prefers-color-scheme:dark){:root{
  --paper:#0f1217; --ink:#e6e9ef; --muted:#98a0ad; --line:#242a33;
  --card:#161b22; --accent:#6ea8dc; --good:#4bbfa3; --bad:#d98b6f; --chip:#1d232c;}}
:root[data-theme=dark]{--paper:#0f1217;--ink:#e6e9ef;--muted:#98a0ad;--line:#242a33;
  --card:#161b22;--accent:#6ea8dc;--good:#4bbfa3;--bad:#d98b6f;--chip:#1d232c;}
:root[data-theme=light]{--paper:#f4f5f2;--ink:#1b1f27;--muted:#5c636e;--line:#d9dcd6;
  --card:#ffffff;--accent:#3a6ea5;--good:#2c7a68;--bad:#a94f38;--chip:#eceee9;}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font:14px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif}
.wrap{max-width:1200px;margin:0 auto;padding:24px 20px 80px}
h1{font:600 22px/1.2 Georgia,"Times New Roman",serif;margin:0 0 4px}
.sub{color:var(--muted);margin:0 0 18px}
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
.head{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:10px}
.chip{background:var(--chip);border-radius:999px;padding:2px 10px;font-size:12px}
.chip.b{color:#fff;background:var(--accent)}
.pair{display:flex;gap:22px;align-items:flex-start;flex-wrap:wrap;margin-bottom:6px}
.frame{display:flex;flex-direction:column;align-items:center;gap:4px}
.frame .cap{font-size:11px;color:var(--muted)}
.arm{border-top:1px dashed var(--line);padding-top:10px;margin-top:10px}
.armhd{display:flex;gap:10px;align-items:center;margin-bottom:6px}
.armname{font-weight:600;width:52px}
.badge{font-size:11px;font-weight:600;padding:1px 8px;border-radius:999px}
.ok{color:#fff;background:var(--good)}
.no{color:#fff;background:var(--bad)}
.strip{display:flex;gap:8px;overflow-x:auto;padding-bottom:6px}
.step{display:flex;flex-direction:column;align-items:center;gap:3px;flex:0 0 auto}
.act{font-size:11px;padding:1px 6px;border-radius:5px;background:var(--chip)}
cnv,canvas{border:1px solid var(--line);border-radius:3px;display:block;image-rendering:pixelated}
.hit canvas{outline:2px solid var(--good);outline-offset:1px}
.err{color:var(--bad);font-size:12px}
.legend{font-size:12px;color:var(--muted);margin-top:2px}
</style>
<div class="wrap">
  <h1>Coverage planning — plan filmstrips</h1>
  <p class="sub">Each plan executed in the Autumn engine, frame by frame. The green-ringed
  frame is where the plan first reaches the goal. <span class="mono">correct</span> = dataset
  ground truth, <span class="mono">raw</span> = grid LLM, <span class="mono">wc</span> =
  worldcoder program search, <span class="mono">nlwm</span> = learned perception+beliefs
  (stored as <span class="mono">lmwm</span> in the evaluation data).</p>
  <div class="bar">
    <span><label>game</label><select id="fg"></select></span>
    <span><label>bucket</label><select id="fb"></select></span>
    <span><label>mechanic</label><select id="fm"></select></span>
    <span><label>horizon</label><select id="fh"></select></span>
    <span><label>outcome</label><select id="fo">
      <option value="">any</option><option value="split">arms disagree</option>
      <option value="lmwm1raw0">lmwm✓ raw✗</option><option value="allfail">all fail</option>
      <option value="allok">all succeed</option></select></span>
    <button id="prev">◀</button><button id="next">▶</button>
    <span class="count" id="count"></span>
  </div>
  <div id="list"></div>
</div>
<script>
const DATA = /*DATA*/{};
const PAL = DATA.palette, P = DATA.problems;
const MODEL_ARMS = ["raw","wc","lmwm"], ARMS = ["correct",...MODEL_ARMS];
const ARM_LABEL = {correct:"correct",raw:"raw",wc:"wc",lmwm:"nlwm"};
const CELL = 7;
function draw(grid){
  const cv=document.createElement("canvas");
  if(!grid){cv.width=cv.height=CELL*4;const x=cv.getContext("2d");
    x.fillStyle="#888";x.font="10px monospace";x.fillText("—",6,16);return cv;}
  const R=grid.length,C=grid[0].length;cv.width=C*CELL;cv.height=R*CELL;
  const x=cv.getContext("2d");
  for(let r=0;r<R;r++)for(let c=0;c<C;c++){x.fillStyle=PAL[grid[r][c]]||"#333";
    x.fillRect(c*CELL,r*CELL,CELL,CELL);}
  return cv;
}
function frame(grid,cap,ring){
  const d=document.createElement("div");d.className="frame"+(ring?" hit":"");
  d.appendChild(draw(grid));
  if(cap!=null){const s=document.createElement("div");s.className="cap";s.textContent=cap;d.appendChild(s);}
  return d;
}
function armBlock(name,a,goalStr){
  const wrap=document.createElement("div");wrap.className="arm";
  const hd=document.createElement("div");hd.className="armhd";
  const nm=document.createElement("span");nm.className="armname mono";nm.textContent=ARM_LABEL[name]||name;
  const bg=document.createElement("span");
  bg.className="badge "+(a.success?"ok":"no");
  bg.textContent=a.success?("✓ hit@"+a.reached_at):"✗ miss";
  hd.appendChild(nm);hd.appendChild(bg);
  if(a.plan_error){const e=document.createElement("span");e.className="err";e.textContent=a.plan_error;hd.appendChild(e);}
  wrap.appendChild(hd);
  if(a.grids){
    const strip=document.createElement("div");strip.className="strip";
    const plan=a.plan||[];
    a.grids.forEach((g,i)=>{
      const st=document.createElement("div");st.className="step";
      const ring=(JSON.stringify(g)===goalStr);
      st.appendChild(frame(g,null,ring));
      const act=document.createElement("div");act.className="act mono";act.textContent=(i+1)+". "+(plan[i]||"?");
      st.appendChild(act);strip.appendChild(st);
    });
    wrap.appendChild(strip);
  } else {
    const e=document.createElement("div");e.className="legend";e.textContent="(no plan produced)";wrap.appendChild(e);
  }
  return wrap;
}
function render(p){
  const c=document.createElement("div");c.className="card";
  const hd=document.createElement("div");hd.className="head";
  const bits=[["b",p.game+" · "+p.human],["",p.bucket],["",p.mechanic],
              ["","h="+p.h],["","noop "+fmt(p.noop)],["","rand "+fmt(p.rand)]];
  bits.forEach(([cl,t])=>{const s=document.createElement("span");s.className="chip"+(cl?" "+cl:"");
    s.textContent=t;hd.appendChild(s);});
  c.appendChild(hd);
  const pair=document.createElement("div");pair.className="pair";
  pair.appendChild(frame(p.start,"START",false));
  pair.appendChild(frame(p.goal,"GOAL",false));
  c.appendChild(pair);
  const goalStr=JSON.stringify(p.goal);
  ARMS.forEach(a=>c.appendChild(armBlock(a,p.arms[a],goalStr)));
  return c;
}
function fmt(v){return v==null?"—":(+v).toFixed(2);}

// ---- filtering ----
function opts(sel,vals,all){sel.innerHTML="";const o=document.createElement("option");
  o.value="";o.textContent=all;sel.appendChild(o);
  [...new Set(vals)].sort().forEach(v=>{const e=document.createElement("option");
    e.value=v;e.textContent=v;sel.appendChild(e);});}
const fg=document.getElementById("fg"),fb=document.getElementById("fb"),
  fm=document.getElementById("fm"),fh=document.getElementById("fh"),fo=document.getElementById("fo");
opts(fg,P.map(p=>p.game),"all games");
opts(fb,P.map(p=>p.bucket),"all buckets");
opts(fh,P.map(p=>String(p.h)),"all h");
function refreshMech(){const g=fg.value;opts(fm,P.filter(p=>!g||p.game==g).map(p=>p.mechanic),"all mechanics");}
refreshMech();
let page=0;const PER=12;
function match(p){
  if(fg.value&&p.game!=fg.value)return false;
  if(fb.value&&p.bucket!=fb.value)return false;
  if(fm.value&&p.mechanic!=fm.value)return false;
  if(fh.value&&String(p.h)!=fh.value)return false;
  const o=fo.value,s=MODEL_ARMS.map(a=>p.arms[a].success);
  if(o=="split"&&new Set(s).size<2)return false;
  if(o=="lmwm1raw0"&&!(p.arms.lmwm.success&&!p.arms.raw.success))return false;
  if(o=="allfail"&&s.some(x=>x))return false;
  if(o=="allok"&&s.some(x=>!x))return false;
  return true;
}
function update(){
  const sel=P.filter(match);
  const pages=Math.max(1,Math.ceil(sel.length/PER));
  page=Math.max(0,Math.min(page,pages-1));
  const list=document.getElementById("list");list.innerHTML="";
  sel.slice(page*PER,page*PER+PER).forEach(p=>list.appendChild(render(p)));
  document.getElementById("count").textContent=
    sel.length+" problems · page "+(page+1)+"/"+pages;
}
[fg,fb,fm,fh,fo].forEach(el=>el.addEventListener("change",()=>{
  if(el==fg)refreshMech();page=0;update();}));
document.getElementById("prev").onclick=()=>{page--;update();};
document.getElementById("next").onclick=()=>{page++;update();};
update();
</script>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(REPO / "logs/coverage_plan_eval.json"))
    ap.add_argument("--out", default=str(REPO / "logs/coverage_plan_viz.html"))
    ap.add_argument("--games", default="", help="comma-separated subset")
    ap.add_argument("--limit", type=int, default=0, help="cap problems per game")
    args = ap.parse_args()
    data = json.loads(Path(args.inp).read_text())
    games = set(filter(None, args.games.split(",")))
    enc = encode(data, games, args.limit)
    if not enc["problems"]:
        raise SystemExit("no enriched problems found (json predates plan-storage; re-run eval)")
    html = HTML.replace("/*DATA*/{}", json.dumps(enc))
    Path(args.out).write_text(html, encoding="utf-8")
    kb = len(html) / 1024
    print(f"{len(enc['problems'])} problems, {len(enc['palette'])} colours -> "
          f"{args.out} ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
