#!/usr/bin/env python3
"""Render the v2 planning set as start -> trajectory -> goal filmstrips.

Same manner as viz_curated.py (one card per problem, grouped into each game's ladder),
extended for the planning-problems-v2.2 schema over all 15 selected games:

  * a problem may start after a replayable PREFIX (shown as a chip + the note line);
  * the caller explicitly chooses frame or NL presentation for the whole evaluation;
  * frame mode compares against the stored exact frame, while NL mode uses the registered
    checker and shows the reference plan's witness frame ("REF GOAL");
  * success mode "any" means the goal may be hit before the last frame: the ring lands on
    the frame where the replay first satisfies the goal, not necessarily the last one;
  * .sexp rule chips are only derivable for the mechanics_rules games
    (bt3gb / dq8gc / n2ntd / s2kt7); other games show plain frames.

Every problem is re-replayed live through the raw interpreter (planning_v2.rollout) and
re-scored with planning_v2.success; a card that no longer reproduces its stored start or
goal is flagged in red rather than silently drawn.

    uv run python offline_learning/scripts/viz_planning_v2.py \
        --in logs/2026-08-29/planning_v2/problems.json \
        --goal-presentation frame \
        --out logs/2026-08-29/planning_v2/viz.html
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from offline_learning.human_replay import GAMES  # noqa: E402
from offline_learning.mechanics_rules import _ACT as _RULES, fired  # noqa: E402
from offline_learning.planning_v2 import (  # noqa: E402
    quiescent_after, raw_trace, rollout, success,
)

RULE_GAMES = set(_RULES)

CSS = {"black": "#0d1017", "white": "#eef1f5", "gray": "#8b929c", "blue": "#2f6fd0",
       "lightblue": "#8fc4e8", "gold": "#e0a92e", "darkgreen": "#2e8b57", "red": "#d1442f",
       "darkorange": "#e07b1a", "mediumpurple": "#9575cd", "green": "#3fae5a",
       "orange": "#e08a1a", "yellow": "#e6c34a"}
IDLE = {"static-noop", "ant-idle", "particle-idle"}

_SCENERY = {"n2ntd": {"enemy-patrol", "enemy-bounce"}}


def is_scenery(game: str, mech: str) -> bool:
    return mech in _SCENERY.get(game, set())


def per_step(program: str, game: str, seed: int, prefix: list[str], plan: list[str],
             trace: list[list[list[str]]]) -> list[list[dict]]:
    """Rule chips per plan step (mechanics_rules games only), from the live replay."""
    if game not in RULE_GAMES:
        return [[] for _ in plan]
    base = len(prefix)
    grids = [json.dumps(trace[base + i]) for i in range(len(plan) + 1)]
    out = []
    for i, a in enumerate(plan):
        cf = json.dumps(raw_trace(program, seed, list(prefix) + plan[:i] + ["noop"])[-1])
        f = fired(game, grids[i], a, cf, grids[i + 1])
        row = [{"m": m, "a": True, "s": is_scenery(game, m)} for m in f.action
               if m not in IDLE]
        row += [{"m": m, "a": False, "s": is_scenery(game, m)} for m in f.passive
                if m not in IDLE]
        out.append(row)
    return out


def _prefix_summary(prefix: list[str]) -> str:
    if not prefix:
        return ""
    if all(a == "noop" for a in prefix):
        return f"{len(prefix)}×noop"
    return " ".join(prefix)


def encode(problems: list[dict], presentation: str) -> dict:
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

    rows = []
    for p in problems:
        prog, game, seed = p["program"], p["game"], p["seed"]
        prefix = list(p["prefix"])
        plan = list(p["plan"] if presentation == "frame" else p["nl_reference_plan"])
        start, frames, trace = rollout(prog, seed, prefix, plan, driver="raw")
        stable = None
        if presentation == "nl" and p.get("nl_require_quiescent"):
            stable = quiescent_after(prog, seed, prefix, plan, driver="raw")
        ok, at = success(p, presentation, start, frames, actions=plan,
                         stable_after_final=stable)
        rows.append({
            "game": game, "human": GAMES[game][1], "id": p["id"], "tier": p["tier"],
            "objective": p["objective"], "h": p["h"], "dec": p["n_decisions"],
            "quiescent": (p.get("frame_reference_quiescent") if presentation == "frame"
                           else p.get("nl_reference_quiescent")), "mech": p["mechanics"],
            "note": p.get("note") or "",
            # The floor the EVALUATORS use: random plans of the full PLAN_CAP, scored
            # through the same path model plans take. The rand@h number beside it was
            # measured with plans of exactly h actions, which is not the budget anyone
            # plans under, and under `nl` it is absent for the 54 rows whose original
            # floor was only ever measured as frames.
            "rand": p.get(f"{presentation}_random_success_cap50"),
            "randh": p.get(f"{presentation}_random_success"),
            "mode": presentation, "smode": (p["frame_success_mode"] if presentation == "frame"
                                                   else p["nl_success_mode"]),
            "stoch": bool(p.get("stochastic")),
            "nl": p.get("nl_goal") or "", "checker": p.get("nl_checker") or "",
            "seed": seed, "prefix": _prefix_summary(prefix), "plen": len(prefix),
            "plan": plan,
            "start": enc(start), "goal": enc(p["goal"] if presentation == "frame" else frames[-1]),
            "frames": [enc(f) for f in frames],
            "reached": bool(ok), "at": at,
            "startok": start == p["start"],
            "steps": per_step(prog, game, seed, prefix, plan, trace),
        })
    return {"palette": {i: CSS.get(n, n) for n, i in pal.items()}, "problems": rows}


HTML = r"""<!doctype html><meta charset="utf-8">
<title>Planning problems v2</title>
<style>
:root{--paper:#f4f5f2;--ink:#1b1f27;--muted:#5c636e;--line:#d9dcd6;--card:#fff;
  --accent:#3a6ea5;--good:#2c7a68;--bad:#a94f38;--chip:#eceee9;--pass:#e8f0ea;--act:#dfe9f4;
  --py:#6b5b9a}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--paper:#0f1217;--ink:#e6e9ef;
  --muted:#98a0ad;--line:#242a33;--card:#161b22;--accent:#6ea8dc;--good:#4bbfa3;--bad:#d98b6f;
  --chip:#1d232c;--pass:#17251f;--act:#182430;--py:#a292d6}}
:root[data-theme=dark]{--paper:#0f1217;--ink:#e6e9ef;--muted:#98a0ad;--line:#242a33;
  --card:#161b22;--accent:#6ea8dc;--good:#4bbfa3;--bad:#d98b6f;--chip:#1d232c;
  --pass:#17251f;--act:#182430;--py:#a292d6}
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
.chip.py{color:#fff;background:var(--py)}
.chip.m{background:var(--act)}
.chip.p{background:var(--pass)}
.note{color:var(--muted);font-size:12px;margin:2px 0 12px;max-width:82ch}
.nl{font-size:12px;margin:0 0 10px;max-width:82ch}
.nl .mono{color:var(--muted)}
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
  <h1>Planning problems v2</h1>
  <p class="sub">The v2 set over all 15 selected games: a short ladder per game, L1 one
  mechanic &rarr; L4 the game's objective. Deterministic games keep <strong>exact-frame
  goals</strong> (the last ringed trajectory frame and the GOAL slot are the same picture);
  the stochastic games (<span class="chip py">python goal</span>) score a registered Python
  predicate instead &mdash; their GOAL slot shows the reference plan's <em>witness</em>
  frame and the sentence above the filmstrip is the actual goal. A
  <span class="chip">prefix</span> chip means the problem starts after a replayed setup;
  START is the frame the planner sees. <span class="chip">any-step</span> goals may be hit
  before the horizon ends &mdash; the ring marks the first frame the replay satisfies the
  goal. Chips under a frame name the <span class="mono">.sexp</span> rules that fired
  (<span class="chip m">blue</span> input-triggered, <span class="chip p">green</span>
  clock; dimmed = scenery) &mdash; derivable only for the four human-origin rule games. A
  <span class="chip">tick-exact</span> badge marks a non-absorbing goal.</p>
  <div class="bar">
    <span><label>game</label><select id="fg"></select></span>
    <span><label>tier</label><select id="ft"></select></span>
    <span><label>goal</label><select id="fk">
      <option value="">all goals</option>
      <option value="exact_frame">exact frame</option>
      <option value="python">python</option></select></span>
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
  const floor = p.rand!=null ? "random "+p.rand.toFixed(2)+" @cap50"
              : p.randh!=null ? "random "+p.randh.toFixed(2)+" @h"
              : "random unmeasured";
  const bits=[["b",p.tier],["",p.game+" / "+p.human],["","h = "+p.h],
              ["","decisions "+p.dec],["",floor]];
  if(p.mode==="nl") bits.push(["py","python goal"]);
  if(p.stoch) bits.push(["","stochastic"]);
  if(p.smode==="any") bits.push(["","any-step"]);
  if(p.plen) bits.push(["","prefix "+p.plen]);
  if(!p.quiescent) bits.push(["","tick-exact"]);
  bits.forEach(([k,x])=>{const s=document.createElement("span");
    s.className="chip "+k;s.textContent=x;h.appendChild(s);});
  c.appendChild(h);
  const n=document.createElement("div");n.className="note";
  n.innerHTML='<span class="mono">'+p.id+'</span> &middot; seed '+p.seed
    +(p.prefix?' &middot; prefix: <span class="mono">'+p.prefix+'</span>':'')
    +' &middot; '+p.mech.join(", ")
    +(p.note?' &middot; '+p.note:'');
  c.appendChild(n);
  if(p.mode==="python"){
    const g=document.createElement("div");g.className="nl";
    g.innerHTML='<strong>goal:</strong> '+p.nl
      +' &nbsp;<span class="mono">['+p.checker+']</span>';
    c.appendChild(g);
  }
  if(!p.startok){const w=document.createElement("div");w.className="warn";
    w.textContent="replayed prefix does not reproduce the stored START frame";
    c.appendChild(w);}
  if(!p.reached){const w=document.createElement("div");w.className="warn";
    w.textContent="replayed reference plan does not satisfy the stored goal";
    c.appendChild(w);}
  const run=document.createElement("div");run.className="run";
  run.appendChild(slot(p.start,p.plen?"START (after prefix)":"START",null,null,false));
  const ar=document.createElement("div");ar.className="arrow";ar.textContent="→";
  run.appendChild(ar);
  p.frames.forEach((f,i)=>{
    run.appendChild(slot(f,"t+"+(i+1),p.plan[i],p.steps[i],
                         p.reached&&p.at===i+1));
  });
  const ar2=document.createElement("div");ar2.className="arrow";
  ar2.textContent=p.mode==="python"?"⊨":"=";
  run.appendChild(ar2);
  run.appendChild(slot(p.goal,p.mode==="python"?"REF GOAL":"GOAL",null,null,true));
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
      fk=document.getElementById("fk"),fm=document.getElementById("fm"),
      fs=document.getElementById("fs");
opts(fg,[...new Set(P.map(p=>p.game))],"games");
opts(ft,[...new Set(P.map(p=>p.tier))].sort(),"tiers");
opts(fm,[...new Set(P.flatMap(p=>p.mech))].sort(),"mechanics");
function filtered(){
  let s=P.map((p,i)=>({p,i})).filter(({p})=>(!fg.value||p.game===fg.value)
    &&(!ft.value||p.tier===ft.value)
    &&(!fk.value||p.mode===fk.value)
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
[fg,ft,fk,fm,fs].forEach(e=>e.addEventListener("change",render));
render();
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--goal-presentation", choices=("frame", "nl"), required=True)
    args = ap.parse_args()
    doc = json.loads(Path(args.inp).read_text())
    enc = encode(doc["problems"], args.goal_presentation)
    bad = [p["id"] for p in enc["problems"] if not p["reached"]]
    badstart = [p["id"] for p in enc["problems"] if not p["startok"]]
    html = HTML.replace("/*DATA*/{}", json.dumps(enc, separators=(",", ":")))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out}  ({len(enc['problems'])} problems, {len(bad)} goal failures, "
          f"{len(badstart)} start mismatches)")
    if bad:
        print("  goal not reached:", ", ".join(bad))
    if badstart:
        print("  start mismatch:", ", ".join(badstart))


if __name__ == "__main__":
    main()
