#!/usr/bin/env python3
"""Render the ONLINE (receding-horizon) planning rollouts as grid filmstrips.

Reads `logs/coverage_online_eval.json` (per arm, a `rounds` list where each round has
the model's full proposed `plan`, the `executed` first action, and the resulting
`grid_after`) and joins START/GOAL/ground-truth from `coverage_plan_problems.json`
(the online rows don't duplicate the grids). Emits a self-contained, filterable HTML:
per problem it draws START and GOAL, then for each arm the CLOSED-LOOP trajectory —
one frame per executed action — captioned with the action taken and the plan the model
proposed that round (so replanning/churn is visible), ringing the frame that first
reaches the goal, badging success (online vs the paired offline result). A `correct`
arm shows the ground-truth actions executed open-loop for reference.

REASONING PANELS: when the eval ran with `--reasoning-trace` (default), each round also
carries `reasoning` (the model's visible <reasoning> block) and `thinking` (the
provider's hidden reasoning tokens). Each arm gets a collapsible panel listing every
round — including rounds whose plan never executed, which the filmstrip necessarily
drops — and clicking a frame's action jumps to that round's reasoning. Visible
reasoning is embedded whole (~720 chars/round); hidden thinking averages ~70k
chars/round, so it is head+tail excerpted (`--max-thinking-chars`, 0 to omit).

    uv run python offline_learning/scripts/viz_coverage_online.py \
        --in logs/coverage_online_eval.json --out logs/coverage_online_viz.html
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
PKEY = ("game", "seed", "t", "bucket", "mechanic", "h")
CSS = {"black": "#0d1017", "white": "#eef1f5", "gray": "#8b929c", "blue": "#2f6fd0",
       "lightblue": "#8fc4e8", "gold": "#e0a92e", "darkgreen": "#2e8b57", "red": "#d1442f",
       "darkorange": "#e07b1a", "mediumpurple": "#9575cd", "green": "#3fae5a",
       "orange": "#e08a1a", "yellow": "#e6c34a"}


def excerpt(text: str | None, cap: int) -> dict | None:
    """Head+tail excerpt: keeps the opening framing AND the conclusion, which is where a
    planning trace commits to its actions (a head-only cut of a 70k-char think block is
    all preamble). Returns {t: shown text, n: true length}, or None when absent/disabled."""
    if not text or cap <= 0:
        return None
    n = len(text)
    if n <= cap:
        return {"t": text, "n": n}
    head = cap // 2
    return {"t": f"{text[:head]}\n\n… [{n - cap:,} chars elided] …\n\n{text[n - (cap - head):]}",
            "n": n}


def encode(online: dict, problems: dict, offline: dict | None, games: set[str], limit: int,
           max_steps: int, reason_cap: int, think_cap: int) -> dict:
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

    prob_by_key = {tuple(p[f] for f in PKEY): p for p in problems["problems"]}
    off_by_key = {}
    if offline:
        for res in offline.get("results", []):
            for r in res["rows"]:
                off_by_key[tuple(r[f] for f in PKEY)] = r

    out_problems, per_game, cov_cache = [], {}, {}
    for r in online["rows"]:
        game = r["game"]
        if games and game not in games:
            continue
        key = tuple(r[f] for f in PKEY)
        pr = prob_by_key.get(key)
        if pr is None:
            continue
        if limit and per_game.get(game, 0) >= limit:
            continue
        per_game[game] = per_game.get(game, 0) + 1
        goal_json = pr["goal_grid"]

        arms = {}
        # ground-truth reference (open-loop) — same computation as the offline viz.
        gt = pr.get("gt_actions") or []
        cg, cerr = None, None
        if gt:
            try:
                if game not in cov_cache:
                    cov_cache[game] = load_coverage(game)
                cov = cov_cache[game]
                prefix = cov["drives_by_seed"][r["seed"]]["actions"][:r["t"]]
                cg = exec_plan(cov["program"], r["seed"], prefix, gt)
            except Exception as exc:  # noqa: BLE001
                cerr = f"gt-exec:{type(exc).__name__}:{exc}"
        creached = next((i + 1 for i, g in enumerate(cg or []) if g == goal_json), None)
        arms["correct"] = {
            "success": creached is not None, "reached_at": creached,
            "failed_reason": cerr, "offline": None,
            "steps": ([{"grid": enc(g), "executed": (gt[i] if i < len(gt) else "?"),
                        "plan": None, "reached": g == goal_json}
                       for i, g in enumerate(cg)] if cg else None)}

        for a in MODEL_ARMS:
            d = r.get(a) or {}
            steps, rounds = [], []
            for ri, rd in enumerate(d.get("rounds") or []):
                why, think = excerpt(rd.get("reasoning"), reason_cap), excerpt(rd.get("thinking"), think_cap)
                ran = rd.get("grid_after") is not None
                if why or think or rd.get("plan"):
                    rounds.append({"i": ri, "n": rd.get("n"), "executed": rd.get("executed"),
                                   "plan": rd.get("plan"), "how": rd.get("how"), "ran": ran,
                                   "err": rd.get("plan_error"), "why": why, "think": think})
                if not ran:
                    continue                       # unusable-plan round: no executed frame
                steps.append({"grid": enc(rd["grid_after"]), "executed": rd.get("executed"),
                              "plan": rd.get("plan"), "reached": rd.get("reached_goal"), "ri": ri})
                if len(steps) >= max_steps:
                    break
            off = off_by_key.get(key, {}).get(a, {})
            arms[a] = {"success": d.get("success"), "reached_at": d.get("reached_at"),
                       "actions_used": d.get("actions_used"),
                       "failed_reason": d.get("failed_reason"),
                       "offline": off.get("success"),
                       "steps": steps or None, "rounds": rounds or None}

        out_problems.append({
            "game": game, "human": HGAMES[game][1], "bucket": r["bucket"],
            "mechanic": r["mechanic"], "h": r["h"], "seed": r["seed"], "t": r["t"],
            "noop": r.get("noop_success"), "rand": r.get("random_success"),
            "start": enc(pr["start_grid"]), "goal": enc(goal_json), "arms": arms})
    palette = {i: CSS.get(name, name) for name, i in pal.items()}
    return {"palette": palette, "problems": out_problems}


HTML = r"""<!doctype html><meta charset="utf-8">
<title>Coverage planning — online rollouts</title>
<style>
:root{--paper:#f4f5f2;--ink:#1b1f27;--muted:#5c636e;--line:#d9dcd6;
  --card:#ffffff;--accent:#3a6ea5;--good:#2c7a68;--bad:#a94f38;--chip:#eceee9;--warn:#9c7a32;}
@media (prefers-color-scheme:dark){:root{--paper:#0f1217;--ink:#e6e9ef;--muted:#98a0ad;
  --line:#242a33;--card:#161b22;--accent:#6ea8dc;--good:#4bbfa3;--bad:#d98b6f;--chip:#1d232c;--warn:#c8a24e;}}
:root[data-theme=dark]{--paper:#0f1217;--ink:#e6e9ef;--muted:#98a0ad;--line:#242a33;
  --card:#161b22;--accent:#6ea8dc;--good:#4bbfa3;--bad:#d98b6f;--chip:#1d232c;--warn:#c8a24e;}
:root[data-theme=light]{--paper:#f4f5f2;--ink:#1b1f27;--muted:#5c636e;--line:#d9dcd6;
  --card:#ffffff;--accent:#3a6ea5;--good:#2c7a68;--bad:#a94f38;--chip:#eceee9;--warn:#9c7a32;}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font:14px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif}
.wrap{max-width:1240px;margin:0 auto;padding:24px 20px 80px}
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
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin-bottom:16px}
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
.off{font-size:11px;color:var(--muted)}
.strip{display:flex;gap:8px;overflow-x:auto;padding-bottom:6px;align-items:flex-start}
.step{display:flex;flex-direction:column;align-items:center;gap:3px;flex:0 0 auto;max-width:150px}
.act{font-size:11px;padding:1px 6px;border-radius:5px;background:var(--chip)}
.plan{font-size:10px;color:var(--muted);max-width:150px;white-space:nowrap;overflow:hidden;
  text-overflow:ellipsis;cursor:help}
canvas{border:1px solid var(--line);border-radius:3px;display:block;image-rendering:pixelated}
.hit canvas{outline:2px solid var(--good);outline-offset:1px}
.err{color:var(--bad);font-size:12px}
.legend{font-size:12px;color:var(--muted);margin-top:2px}
.sep{color:var(--line);align-self:center;font-size:18px;padding:0 2px}
.act.jump{cursor:pointer;text-decoration:underline dotted;text-underline-offset:2px}
details.rz{margin-top:8px}
details.rz>summary{cursor:pointer;font-size:12px;color:var(--muted);list-style:none}
details.rz>summary::-webkit-details-marker{display:none}
details.rz>summary::before{content:"▸ ";color:var(--accent)}
details.rz[open]>summary::before{content:"▾ "}
.rzr{border-left:2px solid var(--line);padding:6px 0 6px 10px;margin:8px 0 0}
.rzr.on{border-left-color:var(--accent);background:var(--chip);border-radius:0 6px 6px 0}
.rzh{font-size:11px;color:var(--muted);display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.rzh b{color:var(--ink);font-weight:600}
.rzplan{font-size:11px;color:var(--muted);margin-top:2px;word-break:break-word}
.rztext{white-space:pre-wrap;word-break:break-word;font:12px/1.55 ui-monospace,Menlo,Consolas,monospace;
  margin:5px 0 0;max-height:340px;overflow-y:auto;background:var(--paper);
  border:1px solid var(--line);border-radius:6px;padding:8px 10px}
details.think{margin-top:5px}
details.think>summary{cursor:pointer;font-size:11px;color:var(--warn)}
.dim{color:var(--muted)}
</style>
<div class="wrap">
  <h1>Coverage planning — online (receding-horizon) rollouts</h1>
  <p class="sub">Each round the model replans to the GOAL; only its FIRST action is executed,
  then it replans from the observed state. Below, each arm's strip shows the executed
  trajectory (one frame per action). The caption is the executed action; the small grey line
  is the FULL plan the model proposed that round (hover for the whole thing) — watch it churn
  as the model replans. The green-ringed frame is where the goal is first reached.
  <span class="mono">on</span>/<span class="mono">off</span> badges = online vs the paired
  offline (open-loop) result. <span class="mono">correct</span> = ground truth (open-loop),
  <span class="mono">nlwm</span> = learned perception+beliefs (<span class="mono">lmwm</span>).
  Expand <b>reasoning</b> under an arm for its per-round trace — every round, including
  ones whose plan never executed (the strip can only show executed frames) — or click an
  underlined action caption to jump straight to that round. Hidden provider thinking is
  nested inside each round, head+tail excerpted.</p>
  <div class="bar">
    <span><label>game</label><select id="fg"></select></span>
    <span><label>bucket</label><select id="fb"></select></span>
    <span><label>mechanic</label><select id="fm"></select></span>
    <span><label>horizon</label><select id="fh"></select></span>
    <span><label>outcome</label><select id="fo">
      <option value="">any</option><option value="split">arms disagree</option>
      <option value="lmwm1raw0">nlwm✓ raw✗</option>
      <option value="onwin">online✓ offline✗ (any arm)</option>
      <option value="allfail">all fail</option><option value="allok">all succeed</option></select></span>
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
function txt(el,cls,s){const d=document.createElement(el);if(cls)d.className=cls;
  if(s!=null)d.textContent=s;return d;}
// One round's reasoning: header (executed action / plan / error) + the visible
// <reasoning> block, with the provider's hidden thinking nested one level deeper.
function roundBlock(rd){
  const b=txt("div","rzr");b.dataset.n=rd.i;
  const h=txt("div","rzh mono");
  h.appendChild(txt("b",null,"round "+((rd.n!=null?rd.n:rd.i)+1)));
  h.appendChild(txt("span",null,rd.ran?("executed: "+(rd.executed||"?")):"(not executed)"));
  if(rd.how)h.appendChild(txt("span","dim",rd.how));
  if(rd.err)h.appendChild(txt("span","err",rd.err));
  b.appendChild(h);
  if(rd.plan&&rd.plan.length)b.appendChild(txt("div","rzplan mono","plan: "+rd.plan.join(", ")));
  if(rd.why){
    b.appendChild(txt("pre","rztext",rd.why.t));
    if(rd.why.n>rd.why.t.length)b.appendChild(txt("div","legend","(reasoning truncated from "+rd.why.n+" chars)"));
  } else if(!rd.think) b.appendChild(txt("div","legend","(no reasoning captured)"));
  if(rd.think){
    const dt=document.createElement("details");dt.className="think";
    dt.appendChild(txt("summary",null,"hidden thinking — "+rd.think.n.toLocaleString()+" chars"
      +(rd.think.n>rd.think.t.length?" (head+tail excerpt)":"")));
    dt.appendChild(txt("pre","rztext",rd.think.t));
    b.appendChild(dt);
  }
  return b;
}
function armBlock(name,a){
  const wrap=document.createElement("div");wrap.className="arm";
  const hd=document.createElement("div");hd.className="armhd";
  const nm=document.createElement("span");nm.className="armname mono";nm.textContent=ARM_LABEL[name]||name;
  const bg=document.createElement("span");bg.className="badge "+(a.success?"ok":"no");
  bg.textContent=a.success?("✓ hit@"+a.reached_at):"✗ miss";
  hd.appendChild(nm);hd.appendChild(bg);
  if(a.offline!=null){const o=document.createElement("span");o.className="off";
    o.textContent="offline "+(a.offline?"✓":"✗");hd.appendChild(o);}
  if(a.failed_reason&&!a.success){const e=document.createElement("span");e.className="err";
    e.textContent=a.failed_reason;hd.appendChild(e);}
  wrap.appendChild(hd);
  // reasoning panel first, so the strip's action captions can link into it
  let det=null,blocks={};
  if(a.rounds&&a.rounds.length){
    det=document.createElement("details");det.className="rz";
    const nwhy=a.rounds.filter(r=>r.why).length;
    det.appendChild(txt("summary",null,"reasoning · "+a.rounds.length+" round"
      +(a.rounds.length==1?"":"s")+(nwhy<a.rounds.length?(" · "+nwhy+" with a <reasoning> block"):"")));
    a.rounds.forEach(rd=>{const b=roundBlock(rd);blocks[rd.i]=b;det.appendChild(b);});
  }
  if(a.steps){
    const strip=document.createElement("div");strip.className="strip";
    const st0=document.createElement("div");st0.className="step";
    st0.appendChild(frame(a._start,"start",false));strip.appendChild(st0);
    a.steps.forEach((s,i)=>{
      const sep=document.createElement("span");sep.className="sep";sep.textContent="›";strip.appendChild(sep);
      const st=document.createElement("div");st.className="step";
      st.appendChild(frame(s.grid,null,!!s.reached));
      const act=document.createElement("div");act.className="act mono";act.textContent=(i+1)+". "+(s.executed||"?");
      const b=blocks[s.ri];
      if(b){act.classList.add("jump");act.title="show this round's reasoning";
        act.onclick=()=>{det.open=true;
          det.querySelectorAll(".rzr.on").forEach(x=>x.classList.remove("on"));
          b.classList.add("on");b.scrollIntoView({block:"nearest",behavior:"smooth"});};}
      st.appendChild(act);
      if(s.plan){const pl=document.createElement("div");pl.className="plan mono";
        pl.textContent="plan: "+s.plan.join(", ");pl.title=s.plan.join("\n");st.appendChild(pl);}
      strip.appendChild(st);
    });
    wrap.appendChild(strip);
  } else {
    const e=document.createElement("div");e.className="legend";
    e.textContent=a.failed_reason?("(no executed step — "+a.failed_reason+")"):"(no rollout)";
    wrap.appendChild(e);
  }
  if(det)wrap.appendChild(det);
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
  ARMS.forEach(a=>{const arm=p.arms[a];arm._start=p.start;c.appendChild(armBlock(a,arm));});
  return c;
}
function fmt(v){return v==null?"—":(+v).toFixed(2);}

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
let page=0;const PER=10;
function match(p){
  if(fg.value&&p.game!=fg.value)return false;
  if(fb.value&&p.bucket!=fb.value)return false;
  if(fm.value&&p.mechanic!=fm.value)return false;
  if(fh.value&&String(p.h)!=fh.value)return false;
  const o=fo.value,s=MODEL_ARMS.map(a=>p.arms[a].success);
  if(o=="split"&&new Set(s).size<2)return false;
  if(o=="lmwm1raw0"&&!(p.arms.lmwm.success&&!p.arms.raw.success))return false;
  if(o=="onwin"&&!MODEL_ARMS.some(a=>p.arms[a].success&&p.arms[a].offline===false))return false;
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
  document.getElementById("count").textContent=sel.length+" problems · page "+(page+1)+"/"+pages;
}
[fg,fb,fm,fh,fo].forEach(el=>el.addEventListener("change",()=>{
  if(el==fg)refreshMech();page=0;update();}));
document.getElementById("prev").onclick=()=>{page--;update();};
document.getElementById("next").onclick=()=>{page++;update();};
update();
</script>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(REPO / "logs/coverage_online_eval.json"))
    ap.add_argument("--problems", default=str(REPO / "logs/coverage_plan_problems.json"))
    ap.add_argument("--offline", default=str(REPO / "logs/coverage_plan_eval.json"))
    ap.add_argument("--out", default=str(REPO / "logs/coverage_online_viz.html"))
    ap.add_argument("--games", default="", help="comma-separated subset")
    ap.add_argument("--limit", type=int, default=0, help="cap problems per game")
    ap.add_argument("--max-steps", type=int, default=20, help="cap executed frames per arm")
    ap.add_argument("--max-reasoning-chars", type=int, default=8000,
                    help="per-round cap on the visible <reasoning> block (0 = omit)")
    ap.add_argument("--max-thinking-chars", type=int, default=1200,
                    help="per-round cap on hidden provider thinking, head+tail (0 = omit; "
                         "these average ~70k chars, so embedding them whole is not viable)")
    args = ap.parse_args()
    online = json.loads(Path(args.inp).read_text())
    problems = json.loads(Path(args.problems).read_text())
    off_path = Path(args.offline)
    offline = json.loads(off_path.read_text()) if off_path.exists() else None
    games = set(filter(None, args.games.split(",")))
    enc = encode(online, problems, offline, games, args.limit, args.max_steps,
                 args.max_reasoning_chars, args.max_thinking_chars)
    if not enc["problems"]:
        raise SystemExit("no problems encoded (check --in / --problems keys line up)")
    rounds = [rd for p in enc["problems"] for a in MODEL_ARMS for rd in (p["arms"][a]["rounds"] or [])]
    # The payload now carries free-form model text, so a trace mentioning "</script>" would
    # close the tag and silently break the page. Escaping < / > keeps it valid JSON.
    blob = json.dumps(enc).replace("<", "\\u003c").replace(">", "\\u003e")
    html = HTML.replace("/*DATA*/{}", blob)
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"{len(enc['problems'])} problems, {len(enc['palette'])} colours, {len(rounds)} rounds "
          f"({sum(1 for r in rounds if r['why'])} with reasoning, "
          f"{sum(1 for r in rounds if r['think'])} with thinking) -> "
          f"{args.out} ({len(html)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
