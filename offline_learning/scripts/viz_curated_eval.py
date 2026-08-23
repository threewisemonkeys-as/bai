#!/usr/bin/env python3
"""Render one eval seed as: the problem, then what each agent actually did on it.

Per problem, three bands:

  REFERENCE   start -> the ground-truth plan frame by frame -> goal, with the .sexp rules
              that fired at each step (identical to the dataset viz)
  ONLINE      one strip per arm: the frames the agent actually reached, one per EXECUTED
              action.  Under each frame sits the whole plan the model proposed that round --
              in a receding-horizon rollout only the first action of that plan is executed,
              so the gap between "what it planned" and "what it did next" is where the
              interesting behaviour lives.
  OFFLINE     the single open-loop plan each attempt submitted, and whether executing it
              ever hit the goal.

REASONING CAVEAT: the model's natural-language <reasoning> block is NOT here, because
`--reasoning-trace` defaults off in the reference harness and this run inherited that.  What
IS here is the proposed plan for every round, which is the model's intent in the only form
that was persisted.  Re-running with --reasoning-trace captures the prose.

    uv run python offline_learning/scripts/viz_curated_eval.py \
        --problems logs/2026-08-18/curated/problems.json \
        --online logs/2026-08-18/curated/eval/online.ckpt.jsonl \
        --offline logs/2026-08-18/curated/eval/offline.json \
        --attempt 0 --out logs/2026-08-18/curated/eval/seed0_viz.html
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from offline_learning.curated_plan import replay, trace  # noqa: E402
from offline_learning.human_replay import GAMES  # noqa: E402
from offline_learning.mechanics_rules import fired  # noqa: E402
from offline_learning.scripts.viz_curated import CSS, IDLE, is_scenery  # noqa: E402

ARMS = ["raw", "lmwm", "wc"]


def per_step(game: str, seed: int, plan: list[str]) -> list[list[dict]]:
    grids = [json.dumps([list(r) for r in s.grid]) for s in trace(game, seed, plan)]
    out = []
    for i, a in enumerate(plan):
        cf = json.dumps([list(r) for r in replay(game, seed, plan[:i] + ["noop"]).grid()])
        f = fired(game, grids[i], a, cf, grids[i + 1])
        row = [{"m": m, "a": True, "s": is_scenery(game, m)} for m in f.action if m not in IDLE]
        row += [{"m": m, "a": False, "s": is_scenery(game, m)} for m in f.passive if m not in IDLE]
        out.append(row)
    return out


def build(problems, online, offline, attempt: int) -> dict:
    pal: dict[str, int] = {}

    def enc(grid):
        out = []
        for row in grid:
            e = []
            for name in row:
                if name not in pal:
                    pal[name] = len(pal)
                e.append(pal[name])
            out.append(e)
        return out

    off_idx = {(r["game"], r["id"]): r for r in offline}
    on_idx: dict[tuple[str, str], dict] = {}
    on_all: dict[tuple[str, str], list] = defaultdict(list)
    for rec in online:
        pid, arm, k = rec["key"].split("|")
        if int(k) == attempt:
            on_idx[(pid, arm)] = rec["result"]
        on_all[(pid, arm)].append(bool(rec["result"]["success"]))

    cards = []
    for p in problems:
        g, seed, plan = p["game"], p["seed"], p["plan"]
        ref = [enc([list(x) for x in s.grid]) for s in trace(g, seed, plan)[1:]]
        card = {
            "game": g, "human": GAMES[g][1], "id": p["id"], "tier": p["tier"],
            "objective": p["objective"], "h": p["h"], "dec": p["n_decisions"],
            "rand": p["random_success"], "quiescent": p["quiescent"], "note": p["note"],
            "start": enc(p["start"]), "goal": enc(p["goal"]),
            "ref": ref, "ref_actions": plan, "ref_steps": per_step(g, seed, plan),
            "online": [], "offline": [],
            # success across ALL completed attempts, for the curve; the strips below still
            # show the single attempt this page was built for
            "all": {arm: on_all.get((p["id"], arm), []) for arm in ARMS},
        }
        for arm in ARMS:
            r = on_idx.get((p["id"], arm))
            if not r:
                continue
            rounds = []
            for rd in r["rounds"]:
                rounds.append({
                    "n": rd["n"],
                    "did": rd.get("executed"),
                    "grid": enc(json.loads(rd["grid_after"])) if rd.get("grid_after") else None,
                    "plan": rd.get("plan") or [],
                    "hit": bool(rd.get("reached_goal")),
                    "err": rd.get("plan_error"),
                })
            card["online"].append({
                "arm": arm, "success": r["success"], "reached_at": r["reached_at"],
                "used": r["actions_used"], "why": r["failed_reason"], "rounds": rounds,
            })
        o = off_idx.get((p["game"], p["id"]))
        if o:
            for arm in ARMS:
                if arm not in o:
                    continue
                for i, t in enumerate(o[arm]["attempts"]):
                    card["offline"].append({
                        "arm": arm, "i": i, "success": t["success"],
                        "reached_at": t.get("reached_at"), "plan": t.get("plan") or [],
                        "err": t.get("plan_error"),
                    })
        cards.append(card)
    return {"palette": {i: CSS.get(n, n) for n, i in pal.items()}, "cards": cards,
            "attempt": attempt}


HTML = r"""<!doctype html><meta charset="utf-8">
<title>Curated eval — agent trajectories</title>
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
.wrap{max-width:1500px;margin:0 auto;padding:24px 20px 80px}
h1{font:600 22px/1.2 Georgia,serif;margin:0 0 4px}
.sub{color:var(--muted);margin:0 0 16px;max-width:80ch}
.mono{font-family:ui-monospace,Menlo,Consolas,monospace}
.bar{position:sticky;top:0;z-index:5;background:var(--paper);border-bottom:1px solid var(--line);
  display:flex;flex-wrap:wrap;gap:10px 14px;align-items:center;padding:12px 0;margin-bottom:16px}
.bar label{font-size:12px;color:var(--muted);margin-right:4px}
select{font:inherit;color:var(--ink);background:var(--card);border:1px solid var(--line);
  border-radius:7px;padding:5px 8px}
.count{margin-left:auto;color:var(--muted);font-size:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
  padding:14px 16px;margin-bottom:18px}
.head{display:flex;flex-wrap:wrap;gap:8px;align-items:baseline;margin-bottom:6px}
.title{font-weight:600;font-size:15px}
.chip{background:var(--chip);border-radius:999px;padding:2px 10px;font-size:12px}
.chip.b{color:#fff;background:var(--accent)}
.chip.ok{background:var(--pass);color:var(--good)}
.chip.no{background:#f3e3de;color:var(--bad)}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]) .chip.no{background:#2a1c19}}
.chip.m{background:var(--act)}.chip.p{background:var(--pass)}
.band{margin-top:12px;border-top:1px solid var(--line);padding-top:10px}
.bandhead{display:flex;gap:8px;align-items:center;font-size:12px;color:var(--muted);
  margin-bottom:6px}
.run{display:flex;gap:8px;align-items:flex-start;overflow-x:auto;padding-bottom:8px}
.slot{display:flex;flex-direction:column;align-items:center;gap:3px;flex:0 0 auto;
  min-width:58px;max-width:150px}
.slot .cap{font-size:10px;color:var(--muted);text-align:center}
.slot .act{font-size:10px;padding:1px 6px;border-radius:5px;background:var(--chip);
  font-family:ui-monospace,Menlo,monospace;white-space:nowrap}
.mechs{display:flex;flex-direction:column;gap:2px;align-items:center}
.mech{font-size:9px;line-height:1.3;padding:0 5px;border-radius:4px;white-space:nowrap}
.mech.scen{opacity:.45;font-style:italic}
.plan{font-size:9px;color:var(--muted);font-family:ui-monospace,Menlo,monospace;
  max-width:150px;text-align:center;cursor:help;line-height:1.25}
.arrow{align-self:center;color:var(--muted);flex:0 0 auto;padding-top:20px}
canvas{border:1px solid var(--line);border-radius:3px;display:block;image-rendering:pixelated}
.goalring canvas{outline:2px solid var(--good);outline-offset:1px}
.charts{display:flex;flex-wrap:wrap;gap:14px;margin:0 0 20px}
.chart{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 10px}
.chart h4{margin:0 0 2px;font:600 12px/1.3 ui-sans-serif,system-ui,sans-serif}
.chart .cap{font-size:10px;color:var(--muted);margin-bottom:2px}
.legend{display:flex;gap:14px;font-size:12px;color:var(--muted);margin:0 0 8px;align-items:center}
.legend i{display:inline-block;width:14px;height:2px;vertical-align:middle;margin-right:5px}
table.scores{border-collapse:collapse;margin:0 0 18px;font-size:13px}
table.scores th,table.scores td{border:1px solid var(--line);padding:4px 10px;text-align:right}
table.scores th{background:var(--chip);font-weight:600;text-align:center}
table.scores td:first-child,table.scores th:first-child{text-align:left}
table.scores td.best{font-weight:700;color:var(--good)}
table.scores tr.tot td{border-top:2px solid var(--line);font-weight:600}
table.scores td.dim{color:var(--muted)}
.chartbox{background:var(--card);border:1px solid var(--line);border-radius:12px;
  padding:12px 14px 6px;margin:0 0 18px;max-width:760px}
.chartbox .ttl{font-size:12px;color:var(--muted);margin-bottom:2px}
.chartbox .lg{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;margin:2px 0 6px}
.chartbox .lg i{display:inline-block;width:22px;height:0;border-top:2px solid;
  vertical-align:middle;margin-right:5px}
.chartbox svg{display:block;width:100%;height:auto;overflow:visible}
.chartbox .ax{stroke:var(--line)}
.chartbox .gl{stroke:var(--line);stroke-dasharray:2 4}
.chartbox .tk{fill:var(--muted);font-size:10px}
.chartbox .al{fill:var(--muted);font-size:11px}
.offtab{font-size:12px;margin-top:4px}
.offtab div{padding:1px 0;color:var(--muted)}
.offtab .mono{color:var(--ink)}
</style>
<div class="wrap">
  <h1>Curated eval &mdash; agent trajectories</h1>
  <p class="sub">Per problem: the <strong>reference</strong> solution, then what each arm
  actually did. Online strips show one frame per <em>executed</em> action; the grey line
  under each frame is the <strong>whole plan the model proposed that round</strong> (hover
  for the full text). Receding-horizon executes only the first action of that plan, so the
  drift between successive proposals is the behaviour to read. A green ring marks the frame
  that hit the goal.<br>
  <em>The natural-language &lt;reasoning&gt; block is not shown: the reference harness
  defaults <span class="mono">--reasoning-trace</span> off and this run inherited that, so
  only the proposed plans were persisted.</em></p>
  <div id="summary"></div>
  <div class="legend" id="legend"></div>
  <div class="charts" id="charts"></div>
  <div id="chart"></div>
  <div class="bar">
    <span><label>game</label><select id="fg"></select></span>
    <span><label>tier</label><select id="ft"></select></span>
    <span><label>outcome</label><select id="fo">
      <option value="">all</option>
      <option value="any">any arm solved</option>
      <option value="none">no arm solved</option></select></span>
    <span><label>rounds shown</label><select id="fr">
      <option value="12">first 12</option>
      <option value="0">all</option></select></span>
    <span class="count" id="count"></span>
  </div>
  <div id="list"></div>
</div>
<script>
const DATA = /*DATA*/{};
const PAL = DATA.palette, C = DATA.cards;
const CELL = 6;
function draw(grid){
  const cv=document.createElement("canvas");
  if(!grid){cv.width=cv.height=CELL*4;return cv;}
  const R=grid.length,Cc=grid[0].length;cv.width=Cc*CELL;cv.height=R*CELL;
  const x=cv.getContext("2d");
  for(let r=0;r<R;r++)for(let c=0;c<Cc;c++){x.fillStyle=PAL[grid[r][c]]||"#333";
    x.fillRect(c*CELL,r*CELL,CELL,CELL);}
  return cv;
}
function slot(grid,cap,act,mechs,plan,ring){
  const d=document.createElement("div");d.className="slot"+(ring?" goalring":"");
  if(act!=null){const a=document.createElement("div");a.className="act";a.textContent=act;d.appendChild(a);}
  d.appendChild(draw(grid));
  if(cap!=null){const s=document.createElement("div");s.className="cap";s.textContent=cap;d.appendChild(s);}
  if(mechs&&mechs.length){const m=document.createElement("div");m.className="mechs";
    mechs.forEach(x=>{const e=document.createElement("div");
      e.className="mech "+(x.a?"chip m":"chip p")+(x.s?" scen":"");
      e.textContent=x.m;m.appendChild(e);});d.appendChild(m);}
  if(plan&&plan.length){const q=document.createElement("div");q.className="plan";
    q.textContent=plan.slice(0,3).join(" · ")+(plan.length>3?" +"+(plan.length-3):"");
    q.title="proposed plan ("+plan.length+" actions):\n"+plan.join("\n");d.appendChild(q);}
  return d;
}
function band(label,chips){
  const b=document.createElement("div");b.className="bandhead";
  const l=document.createElement("strong");l.textContent=label;b.appendChild(l);
  chips.forEach(([k,t])=>{const s=document.createElement("span");s.className="chip "+k;
    s.textContent=t;b.appendChild(s);});
  return b;
}
function card(p){
  const cap=+document.getElementById("fr").value;
  const c=document.createElement("div");c.className="card";
  const h=document.createElement("div");h.className="head";
  const t=document.createElement("span");t.className="title";t.textContent=p.objective;
  h.appendChild(t);
  [["b",p.tier],["",p.game+" / "+p.human],["","h = "+p.h],["","decisions "+p.dec],
   ["","random "+p.rand.toFixed(2)]].forEach(([k,x])=>{
    const s=document.createElement("span");s.className="chip "+k;s.textContent=x;h.appendChild(s);});
  c.appendChild(h);
  // reference
  const rb=document.createElement("div");rb.className="band";
  rb.appendChild(band("reference",[["","ground truth, "+p.h+" actions"]]));
  const rr=document.createElement("div");rr.className="run";
  rr.appendChild(slot(p.start,"START",null,null,null,false));
  const ar=document.createElement("div");ar.className="arrow";ar.textContent="→";rr.appendChild(ar);
  const rf=cap?p.ref.slice(0,cap):p.ref;
  rf.forEach((f,i)=>rr.appendChild(slot(f,"t+"+(i+1),p.ref_actions[i],p.ref_steps[i],null,
    i===p.ref.length-1)));
  if(cap&&p.ref.length>cap){const m=document.createElement("div");m.className="arrow";
    m.textContent="…+"+(p.ref.length-cap);rr.appendChild(m);}
  rb.appendChild(rr);c.appendChild(rb);
  // online arms
  p.online.forEach(o=>{
    const b=document.createElement("div");b.className="band";
    b.appendChild(band(o.arm+" · online",[
      [o.success?"ok":"no", o.success?("reached at "+o.reached_at):(o.why||"failed")],
      ["","actions "+o.used],["","rounds "+o.rounds.length]]));
    const run=document.createElement("div");run.className="run";
    run.appendChild(slot(p.start,"START",null,null,null,false));
    const a2=document.createElement("div");a2.className="arrow";a2.textContent="→";run.appendChild(a2);
    const rs=cap?o.rounds.slice(0,cap):o.rounds;
    rs.forEach(rd=>run.appendChild(slot(rd.grid,"r"+(rd.n+1),rd.did||rd.err||"—",null,rd.plan,rd.hit)));
    if(cap&&o.rounds.length>cap){const m=document.createElement("div");m.className="arrow";
      m.textContent="…+"+(o.rounds.length-cap);run.appendChild(m);}
    b.appendChild(run);c.appendChild(b);
  });
  // offline
  if(p.offline.length){
    const b=document.createElement("div");b.className="band";
    const nOk=p.offline.filter(x=>x.success).length;
    b.appendChild(band("offline (open loop)",[["",nOk+"/"+p.offline.length+" attempts reached the goal"]]));
    const tab=document.createElement("div");tab.className="offtab";
    p.offline.forEach(x=>{
      const d=document.createElement("div");
      d.innerHTML='<span class="chip '+(x.success?"ok":"no")+'">'+(x.success?"✓":"✗")+'</span> '
        +'<b>'+x.arm+'</b> #'+(x.i+1)+' &middot; <span class="mono">'
        +(x.plan.length?x.plan.join(" · "):("["+(x.err||"no plan")+"]"))+'</span>';
      tab.appendChild(d);});
    b.appendChild(tab);c.appendChild(b);
  }
  return c;
}
function opts(sel,vals,label){sel.innerHTML="";
  const a=document.createElement("option");a.value="";a.textContent="all "+label;sel.appendChild(a);
  vals.forEach(v=>{const o=document.createElement("option");o.value=v;o.textContent=v;sel.appendChild(o);});}
const fg=document.getElementById("fg"),ft=document.getElementById("ft"),
      fo=document.getElementById("fo"),fr=document.getElementById("fr");
opts(fg,[...new Set(C.map(p=>p.game))],"games");
opts(ft,[...new Set(C.map(p=>p.tier))].sort(),"tiers");
function pct(v){return v==null?"&mdash;":v.toFixed(2);}
function summary(s){
  const el=document.getElementById("summary");el.innerHTML="";
  if(!s.length)return;
  const games=[...new Set(s.map(p=>p.game))];
  const on=(rows,arm)=>{const v=rows.map(p=>{const o=p.online.find(x=>x.arm===arm);
      return o?(o.success?1:0):null;}).filter(x=>x!==null);
    return v.length?v.reduce((a,b)=>a+b,0)/v.length:null;};
  const off=(rows,arm)=>{const v=rows.map(p=>{const t=p.offline.filter(x=>x.arm===arm);
      return t.length?t.filter(x=>x.success).length/t.length:null;}).filter(x=>x!==null);
    return v.length?v.reduce((a,b)=>a+b,0)/v.length:null;};
  const t=document.createElement("table");t.className="scores";
  t.innerHTML='<thead><tr><th rowspan="2">game</th><th rowspan="2">n</th>'
    +'<th colspan="3">online &mdash; seed '+(DATA.attempt+1)+'</th>'
    +'<th colspan="3">offline &mdash; mean of attempts</th>'
    +'<th rowspan="2">random</th></tr>'
    +'<tr><th>raw</th><th>lmwm</th><th>wc</th><th>raw</th><th>lmwm</th><th>wc</th></tr></thead>';
  const tb=document.createElement("tbody");
  const row=(label,rows,cls)=>{
    const onv=["raw","lmwm","wc"].map(a=>on(rows,a));
    const offv=["raw","lmwm","wc"].map(a=>off(rows,a));
    const bo=Math.max(...onv.filter(x=>x!=null)), bf=Math.max(...offv.filter(x=>x!=null));
    const r=document.createElement("tr");if(cls)r.className=cls;
    r.innerHTML='<td>'+label+'</td><td>'+rows.length+'</td>'
      +onv.map(v=>'<td class="'+(v!=null&&v===bo&&bo>0?"best":"")+'">'+pct(v)+'</td>').join("")
      +offv.map(v=>'<td class="'+(v!=null&&v===bf&&bf>0?"best":"")+'">'+pct(v)+'</td>').join("")
      +'<td class="dim">'+(rows.reduce((a,b)=>a+b.rand,0)/rows.length).toFixed(2)+'</td>';
    tb.appendChild(r);
  };
  games.forEach(g=>row(g+" / "+s.find(p=>p.game===g).human, s.filter(p=>p.game===g)));
  if(games.length>1)row("all",s,"tot");
  t.appendChild(tb);el.appendChild(t);
}
const ARMCOL={raw:"#a94f38",lmwm:"#3a6ea5",wc:"#2c7a68"};
function chart(s){
  // ONLINE success vs the problem's DECISION count. `dec` is how many non-forced choices
  // the reference solution makes, so it is the difficulty axis the horizon `h` inflates
  // with waiting. One point per distinct dec present in the current filter; the marker
  // area scales with how many problems sit at that x, because most x carry only one or
  // two and a bare line would read as far more evidence than there is.
  const el=document.getElementById("chart");el.innerHTML="";
  const arms=["raw","lmwm","wc"];
  const onv=(p,arm)=>{const o=p.online.find(x=>x.arm===arm);return o?(o.success?1:0):null;};
  const usable=s.filter(p=>arms.some(a=>onv(p,a)!==null));
  if(usable.length<2){return;}
  const xs=[...new Set(usable.map(p=>p.dec))].sort((a,b)=>a-b);
  const series=arms.map(arm=>({arm,pts:xs.map(x=>{
      const v=usable.filter(p=>p.dec===x).map(p=>onv(p,arm)).filter(y=>y!==null);
      return v.length?{x,y:v.reduce((a,b)=>a+b,0)/v.length,n:v.length}:null;
    }).filter(Boolean)})).filter(d=>d.pts.length);
  if(!series.length){return;}
  const W=720,H=250,L=44,R=12,T=12,B=34;
  const xmin=xs[0],xmax=xs[xs.length-1];
  const px=x=>L+(xmax===xmin?0.5:(x-xmin)/(xmax-xmin))*(W-L-R);
  const py=y=>T+(1-y)*(H-T-B);
  let g="";
  [0,0.25,0.5,0.75,1].forEach(y=>{
    g+='<line class="gl" x1="'+L+'" y1="'+py(y)+'" x2="'+(W-R)+'" y2="'+py(y)+'"/>'
      +'<text class="tk" x="'+(L-7)+'" y="'+(py(y)+3)+'" text-anchor="end">'+y.toFixed(2)+'</text>';
  });
  xs.forEach(x=>{g+='<text class="tk" x="'+px(x)+'" y="'+(H-B+14)+'" text-anchor="middle">'+x+'</text>';});
  g+='<line class="ax" x1="'+L+'" y1="'+py(0)+'" x2="'+(W-R)+'" y2="'+py(0)+'"/>'
    +'<line class="ax" x1="'+L+'" y1="'+T+'" x2="'+L+'" y2="'+py(0)+'"/>'
    +'<text class="al" x="'+((L+W-R)/2)+'" y="'+(H-2)+'" text-anchor="middle">decisions in the reference solution</text>'
    +'<text class="al" transform="translate(11,'+((T+py(0))/2)+') rotate(-90)" text-anchor="middle">online pass rate</text>';
  series.forEach(d=>{
    const c=ARMCOL[d.arm];
    g+='<polyline fill="none" stroke="'+c+'" stroke-width="2" stroke-linejoin="round" points="'
      +d.pts.map(pt=>px(pt.x)+","+py(pt.y)).join(" ")+'"/>';
    d.pts.forEach(pt=>{g+='<circle cx="'+px(pt.x)+'" cy="'+py(pt.y)+'" r="'
      +(2.6+Math.sqrt(pt.n)*1.3)+'" fill="'+c+'" fill-opacity=".85"><title>'+d.arm
      +' — '+pt.n+' problem'+(pt.n>1?'s':'')+' at '+pt.x+' decision'+(pt.x>1?'s':'')
      +', pass '+pt.y.toFixed(2)+'</title></circle>';});
  });
  const att=(DATA.attempt+1);
  el.innerHTML='<div class="chartbox"><div class="ttl">Online pass rate vs decision count'
    +' &mdash; seed '+att+', '+usable.length+' problems (marker area &prop; problems at that x)</div>'
    +'<div class="lg">'+series.map(d=>'<span><i style="border-color:'+ARMCOL[d.arm]
    +'"></i>'+d.arm+'</span>').join("")+'</div>'
    +'<svg viewBox="0 0 '+W+' '+H+'" role="img" aria-label="online pass rate by decision count">'
    +g+'</svg></div>';
}
const ARMC={raw:"#686868",lmwm:"#1677b8",wc:"#d1493f"};
const ARMN={raw:"Raw LLM",lmwm:"NLWM (lmwm)",wc:"WorldCoder (wc)"};
function svgEl(n,a){const e=document.createElementNS("http://www.w3.org/2000/svg",n);
  for(const k in a)e.setAttribute(k,a[k]);return e;}
function chart(game,rows){
  const W=320,H=190,L=34,R=8,T=8,B=26;
  const xs=[...new Set(rows.map(p=>p.dec))].sort((a,b)=>a-b);
  const x0=Math.min(...xs),x1=Math.max(...xs);
  const px=v=>L+(x1===x0?(W-L-R)/2:(v-x0)/(x1-x0)*(W-L-R));
  const py=v=>T+(1-v)*(H-T-B);
  const svg=svgEl("svg",{width:W,height:H});
  [0,.5,1].forEach(g=>{
    svg.appendChild(svgEl("line",{x1:L,x2:W-R,y1:py(g),y2:py(g),
      stroke:"var(--line)","stroke-width":1}));
    const t=svgEl("text",{x:L-5,y:py(g)+3,"text-anchor":"end","font-size":9,
      fill:"var(--muted)"});t.textContent=g.toFixed(1);svg.appendChild(t);});
  xs.forEach(v=>{const t=svgEl("text",{x:px(v),y:H-B+13,"text-anchor":"middle",
    "font-size":9,fill:"var(--muted)"});t.textContent=v;svg.appendChild(t);});
  const ax=svgEl("text",{x:(L+W-R)/2,y:H-2,"text-anchor":"middle","font-size":9,
    fill:"var(--muted)"});ax.textContent="decisions (non-noop actions)";svg.appendChild(ax);
  // random floor
  const rnd=xs.map(v=>{const g=rows.filter(p=>p.dec===v);
    return [v,g.reduce((a,b)=>a+b.rand,0)/g.length];});
  svg.appendChild(svgEl("polyline",{points:rnd.map(([a,b])=>px(a)+","+py(b)).join(" "),
    fill:"none",stroke:"#b9b9b9","stroke-width":1.2,"stroke-dasharray":"3 3"}));
  ["raw","lmwm","wc"].forEach(arm=>{
    const pts=[];
    xs.forEach(v=>{
      const g=rows.filter(p=>p.dec===v);
      const vals=g.flatMap(p=>p.all[arm]||[]);
      if(vals.length)pts.push([v,vals.filter(Boolean).length/vals.length,vals.length]);
    });
    if(!pts.length)return;
    svg.appendChild(svgEl("polyline",{points:pts.map(([a,b])=>px(a)+","+py(b)).join(" "),
      fill:"none",stroke:ARMC[arm],"stroke-width":1.8}));
    pts.forEach(([a,b,n])=>{const c=svgEl("circle",{cx:px(a),cy:py(b),r:2.6,fill:ARMC[arm]});
      const ttl=svgEl("title");ttl.textContent=ARMN[arm]+"  decisions "+a+"  "+b.toFixed(2)+"  (n="+n+")";
      c.appendChild(ttl);svg.appendChild(c);});
  });
  const box=document.createElement("div");box.className="chart";
  const h=document.createElement("h4");h.textContent=game+" / "+rows[0].human;box.appendChild(h);
  const cp=document.createElement("div");cp.className="cap";
  cp.textContent=rows.length+" problems · "+rows.reduce((a,p)=>a+(p.all.raw||[]).length,0)
    +" raw attempts";box.appendChild(cp);
  box.appendChild(svg);return box;
}
function charts(s){
  const el=document.getElementById("charts");el.innerHTML="";
  const lg=document.getElementById("legend");lg.innerHTML="";
  if(!s.length)return;
  lg.innerHTML='<span>online success vs decisions &mdash; all completed attempts:</span>'
    +["raw","lmwm","wc"].map(a=>'<span><i style="background:'+ARMC[a]+'"></i>'+ARMN[a]+'</span>').join("")
    +'<span><i style="background:#b9b9b9"></i>random floor</span>';
  [...new Set(s.map(p=>p.game))].forEach(g=>el.appendChild(chart(g,s.filter(p=>p.game===g))));
}
function render(){
  const s=C.filter(p=>(!fg.value||p.game===fg.value)&&(!ft.value||p.tier===ft.value)
    &&(!fo.value||(fo.value==="any"?p.online.some(o=>o.success):!p.online.some(o=>o.success))));
  const list=document.getElementById("list");list.innerHTML="";
  // guarded: a throw in the header widgets must never take the trajectories with it,
  // which is exactly what a stray chart(s) call did
  try{ summary(s); }catch(e){ console.error("summary failed",e); }
  try{ charts(s); }catch(e){ console.error("charts failed",e); }
  s.forEach(p=>list.appendChild(card(p)));
  document.getElementById("count").textContent=s.length+" problems";
}
[fg,ft,fo,fr].forEach(e=>e.addEventListener("change",render));
render();
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="logs/2026-08-18/curated/problems.json")
    ap.add_argument("--online", default="logs/2026-08-18/curated/eval/online.ckpt.jsonl")
    ap.add_argument("--offline", default="logs/2026-08-18/curated/eval/offline.json")
    ap.add_argument("--attempt", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    problems = json.loads(Path(a.problems).read_text())
    online = [json.loads(l) for l in Path(a.online).read_text().splitlines() if l.strip()]
    offline = json.loads(Path(a.offline).read_text())["rows"]
    data = build(problems, online, offline, a.attempt)
    html = HTML.replace("/*DATA*/{}", json.dumps(data, separators=(",", ":")))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    n_on = sum(len(c["online"]) for c in data["cards"])
    n_rd = sum(len(o["rounds"]) for c in data["cards"] for o in c["online"])
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) — {len(data['cards'])} problems, "
          f"{n_on} agent rollouts, {n_rd} rounds")


if __name__ == "__main__":
    main()
