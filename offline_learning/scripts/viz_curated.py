#!/usr/bin/env python3
"""Render the curated planning set as start -> trajectory -> goal filmstrips.

One card per problem, grouped into each game's difficulty ladder (L1 one mechanic -> L4 the
game's objective).  The card shows the START frame, the reference plan replayed frame by
frame in the Autumn engine with the action above each frame and the .sexp rules that fired
underneath, and the GOAL -- which is a full exact frame, so the last trajectory frame and
the goal slot are the same picture by construction (V1 in validate_curated.py re-checks
that through a different engine driver).

Per-step rules are re-derived with a live replay rather than read from any stored summary:
a step whose rules had all fired earlier must still show its chips, otherwise it reads as
padding in a set whose whole point is not having any.

    uv run python offline_learning/scripts/viz_curated.py \
        --in logs/2026-08-18/curated/problems.json \
        --out logs/2026-08-18/curated/viz.html
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from offline_learning.curated_plan import replay, trace  # noqa: E402
from offline_learning.human_replay import GAMES  # noqa: E402
from offline_learning.mechanics_rules import fired  # noqa: E402

CSS = {"black": "#0d1017", "white": "#eef1f5", "gray": "#8b929c", "blue": "#2f6fd0",
       "lightblue": "#8fc4e8", "gold": "#e0a92e", "darkgreen": "#2e8b57", "red": "#d1442f",
       "darkorange": "#e07b1a", "mediumpurple": "#9575cd", "green": "#3fae5a",
       "orange": "#e08a1a", "yellow": "#e6c34a"}
IDLE = {"static-noop", "ant-idle", "particle-idle"}


def is_scenery(game: str, mech: str) -> bool:
    """A rule whose entire outcome lands in an autonomously-evolving class (only n2ntd's
    patrolling enemy, among these five).  It must still be predicted -- goals are full
    frames -- but it is not something the agent did."""
    return mech in _SCENERY.get(game, set())


_SCENERY = {"n2ntd": {"enemy-patrol", "enemy-bounce"}}


def per_step(game: str, seed: int, plan: list[str]) -> list[list[dict]]:
    grids = [json.dumps([list(r) for r in s.grid]) for s in trace(game, seed, plan)]
    out = []
    for i, a in enumerate(plan):
        cf = json.dumps([list(r) for r in replay(game, seed, plan[:i] + ["noop"]).grid()])
        f = fired(game, grids[i], a, cf, grids[i + 1])
        row = [{"m": m, "a": True, "s": is_scenery(game, m)} for m in f.action
               if m not in IDLE]
        row += [{"m": m, "a": False, "s": is_scenery(game, m)} for m in f.passive
                if m not in IDLE]
        out.append(row)
    return out


def encode(rows: list[dict]) -> dict:
    pal: dict[str, int] = {}

    def enc(grid):
        out = []
        for row in grid:
            erow = []
            for name in row:
                if name not in pal:
                    pal[name] = len(pal)
                erow.append(pal[name])
            out.append(erow)
        return out

    problems = []
    for r in rows:
        g, seed, plan = r["game"], r["seed"], r["plan"]
        frames = [enc([list(x) for x in s.grid])
                  for s in trace(g, seed, plan)[1:]]
        problems.append({
            "game": g, "human": GAMES[g][1], "id": r["id"], "tier": r["tier"],
            "objective": r["objective"], "h": r["h"], "dec": r["n_decisions"],
            "quiescent": r["quiescent"], "mech": r["mechanics"], "note": r["note"],
            "rand": r["random_success"],
            "seed": seed, "plan": plan,
            "start": enc(r["start"]), "goal": enc(r["goal"]),
            "frames": frames,
            "reached": bool(frames) and frames[-1] == enc(r["goal"]),
            "steps": per_step(g, seed, plan),
        })
    return {"palette": {i: CSS.get(n, n) for n, i in pal.items()}, "problems": problems}


HTML = r"""<!doctype html><meta charset="utf-8">
<title>Curated planning problems</title>
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
h2{font:600 15px/1.2 Georgia,"Times New Roman",serif;margin:26px 0 10px;
  padding-bottom:6px;border-bottom:1px solid var(--line)}
.sub{color:var(--muted);margin:0 0 18px;max-width:74ch}
.mono{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
.bar{position:sticky;top:0;z-index:5;background:var(--paper);border-bottom:1px solid var(--line);
  display:flex;flex-wrap:wrap;gap:10px 14px;align-items:center;padding:12px 0;margin-bottom:16px}
.bar label{font-size:12px;color:var(--muted);margin-right:4px}
select{font:inherit;color:var(--ink);background:var(--card);border:1px solid var(--line);
  border-radius:7px;padding:5px 8px}
.count{margin-left:auto;color:var(--muted);font-size:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
  padding:14px 16px;margin-bottom:16px}
.head{display:flex;flex-wrap:wrap;gap:8px;align-items:baseline;margin-bottom:4px}
.title{font-weight:600;font-size:15px}
.chip{background:var(--chip);border-radius:999px;padding:2px 10px;font-size:12px}
.chip.b{color:#fff;background:var(--accent)}
.chip.m{background:var(--act)}
.chip.p{background:var(--pass)}
.note{color:var(--muted);font-size:12px;margin:2px 0 12px;max-width:82ch}
.run{display:flex;gap:10px;align-items:flex-start;overflow-x:auto;padding-bottom:8px}
.slot{display:flex;flex-direction:column;align-items:center;gap:4px;flex:0 0 auto;
  min-width:64px;max-width:132px}
.slot .cap{font-size:11px;color:var(--muted);text-align:center}
.slot .act{font-size:11px;padding:1px 7px;border-radius:5px;background:var(--chip);
  font-family:ui-monospace,Menlo,monospace;white-space:nowrap}
.mechs{display:flex;flex-direction:column;gap:2px;align-items:center}
.mech{font-size:10px;line-height:1.35;padding:0 6px;border-radius:4px;white-space:nowrap}
.mech.scen{opacity:.45;font-style:italic}
.arrow{align-self:center;color:var(--muted);flex:0 0 auto;padding-top:22px}
canvas{border:1px solid var(--line);border-radius:3px;display:block;image-rendering:pixelated}
.goalring canvas{outline:2px solid var(--good);outline-offset:1px}
.warn{color:var(--bad);font-size:12px}
</style>
<div class="wrap">
  <h1>Curated planning problems</h1>
  <p class="sub">A short ladder per game, each rung built around something the game is
  actually about &mdash; L1 exercises one mechanic, L4 is the objective. A goal is a
  <strong>concrete end state</strong>: one full frame, compared for exact equality, so the
  last trajectory frame and the GOAL slot are the same picture. Chips under a frame name the
  <span class="mono">.sexp</span> rules that fired there &mdash;
  <span class="chip m">blue</span> triggered by the input,
  <span class="chip p">green</span> a clock rule; dimmed italic ones are scenery
  (n2ntd's patrolling enemy) which must still be predicted but is not something the agent
  did. Every plan is incompressible: no single action can be dropped and still complete the
  objective, so the horizon is difficulty rather than padding. A
  <span class="chip">tick-exact</span> badge marks a goal that is <em>not</em> absorbing
  &mdash; the frame names one tick and a solution has to land on it.</p>
  <div class="bar">
    <span><label>game</label><select id="fg"></select></span>
    <span><label>tier</label><select id="ft"></select></span>
    <span><label>mechanic</label><select id="fm"></select></span>
    <span><label>sort</label><select id="fs">
      <option value="ladder">ladder order</option>
      <option value="h">longest horizon</option>
      <option value="dec">most decisions</option></select></span>
    <span class="count" id="count"></span>
  </div>
  <div id="list"></div>
</div>
<script>
const DATA = /*DATA*/{};
const PAL = DATA.palette, P = DATA.problems;
const CELL = 7;
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
  const t=document.createElement("span");t.className="title";
  t.textContent=p.objective;h.appendChild(t);
  const bits=[["b",p.tier],["",p.game+" / "+p.human],["","h = "+p.h],
              ["","decisions "+p.dec],["","random "+p.rand.toFixed(2)]];
  if(!p.quiescent) bits.push(["","tick-exact"]);
  bits.forEach(([k,x])=>{const s=document.createElement("span");
    s.className="chip "+k;s.textContent=x;h.appendChild(s);});
  c.appendChild(h);
  const n=document.createElement("div");n.className="note";
  n.innerHTML='<span class="mono">'+p.id+'</span> &middot; '+p.mech.join(", ")
    +(p.note?' &middot; '+p.note:'');
  c.appendChild(n);
  if(!p.reached){const w=document.createElement("div");w.className="warn";
    w.textContent="replay does not reach the stored goal";c.appendChild(w);}
  const run=document.createElement("div");run.className="run";
  run.appendChild(slot(p.start,"START",null,null,false));
  const ar=document.createElement("div");ar.className="arrow";ar.textContent="→";
  run.appendChild(ar);
  p.frames.forEach((f,i)=>{
    run.appendChild(slot(f,"t+"+(i+1),p.plan[i],p.steps[i],
                         i===p.frames.length-1&&p.reached));
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
const fg=document.getElementById("fg"),ft=document.getElementById("ft"),
      fm=document.getElementById("fm"),fs=document.getElementById("fs");
opts(fg,[...new Set(P.map(p=>p.game))],"games");
opts(ft,[...new Set(P.map(p=>p.tier))].sort(),"tiers");
opts(fm,[...new Set(P.flatMap(p=>p.mech))].sort(),"mechanics");
function filtered(){
  let s=P.map((p,i)=>({p,i})).filter(({p})=>(!fg.value||p.game===fg.value)
    &&(!ft.value||p.tier===ft.value)
    &&(!fm.value||p.mech.includes(fm.value)));
  const k=fs.value;
  if(k==="h") s.sort((a,b)=>b.p.h-a.p.h);
  else if(k==="dec") s.sort((a,b)=>b.p.dec-a.p.dec);
  else s.sort((a,b)=>a.i-b.i);
  return s.map(x=>x.p);
}
function render(){
  const s=filtered();
  const list=document.getElementById("list");list.innerHTML="";
  let seen=null;
  s.forEach(p=>{
    if(fs.value==="ladder" && p.game!==seen){
      seen=p.game;
      const hh=document.createElement("h2");
      hh.textContent=p.game+" / "+p.human;list.appendChild(hh);
    }
    list.appendChild(card(p));
  });
  document.getElementById("count").textContent=s.length+" problems";
}
[fg,ft,fm,fs].forEach(e=>e.addEventListener("change",render));
render();
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = json.loads(Path(args.inp).read_text())
    enc = encode(rows)
    bad = [p["id"] for p in enc["problems"] if not p["reached"]]
    empty = sum(1 for p in enc["problems"] for s in p["steps"] if not s)
    html = HTML.replace("/*DATA*/{}", json.dumps(enc, separators=(",", ":")))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out}  ({len(enc['problems'])} problems, "
          f"{len(bad)} goal mismatches, {empty} steps with no rule fired)")
    if bad:
        print("  mismatched:", ", ".join(bad))


if __name__ == "__main__":
    main()
