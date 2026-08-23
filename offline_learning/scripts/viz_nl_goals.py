#!/usr/bin/env python3
"""Render the NL-goal pilot as problems and the attempts made at them.

One card per problem: the START frame and the sentence the agent was given -- which is the
whole of what it was told, since an NL goal shows no target frame. Under it, one row per
attempt: the plan replayed frame by frame in the Autumn engine, with the action above each
frame and the planner's own rationale for that action below it.

Frames are re-executed here rather than read from the eval JSON, through the same
`exec_plan` the scorer used, so what is drawn is what was scored. The frame where the
checker first accepts is ringed; nothing after it is greyed, because ANY-STEP scoring means
the run had already succeeded there and whatever follows is the planner talking past its own
success -- worth seeing, not worth hiding.

Rationales come from the numbered lines in the planner's `<reasoning>` block, keyed by the
model's OWN numbering. A step whose number the model never wrote shows no rationale rather
than borrowing its neighbour's.

    uv run python offline_learning/scripts/viz_nl_goals.py \
        --eval logs/2026-08-19/nl_pilot/eval/offline.json \
        --validation logs/2026-08-19/nl_pilot/validation.json \
        --out logs/2026-08-19/nl_pilot/viz.html
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(REPO), str(REPO / "offline_learning")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from offline_learning.coverage_plan import exec_plan  # noqa: E402
from offline_learning.human_replay import GAMES  # noqa: E402
from offline_learning.nl_goals import BY_PID, first_satisfied  # noqa: E402

CSS = {"black": "#0d1017", "white": "#eef1f5", "gray": "#8b929c", "blue": "#2f6fd0",
       "lightblue": "#8fc4e8", "gold": "#e0a92e", "darkgreen": "#2e8b57", "red": "#d1442f",
       "darkorange": "#e07b1a", "mediumpurple": "#9575cd", "green": "#3fae5a",
       "orange": "#e08a1a", "yellow": "#e6c34a"}

# Frames are the bulk of this file -- 5 problems x 5 attempts x up to 50 steps of 16x16 --
# so each is packed as one string, a character per cell, rows separated by '|'.  A plain
# nested-array encoding of the same data is ~15x larger and the page has to hold all of it.
_CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def pack(grid, pal: dict[str, int]) -> str:
    rows = []
    for row in grid:
        out = []
        for name in row:
            if name not in pal:
                pal[name] = len(pal)
            out.append(_CHARS[pal[name]])
        rows.append("".join(out))
    return "|".join(rows)


def build(ev: dict, val: dict, curated: dict, arms: list[str]) -> dict:
    pal: dict[str, int] = {}
    problems = []
    for r in ev["rows"]:
        goal = BY_PID[r["id"]]
        cur = curated[r["id"]]
        v = val.get(r["id"], {})
        start = cur["start"]
        attempts = []
        for arm in arms:
            if arm not in r:
                continue
            for i, t in enumerate(r[arm]["attempts"]):
                a = {"arm": arm, "n": i + 1, "error": t["plan_error"],
                     "reasoning": t.get("reasoning", ""),
                     "success": t["success"], "sat": t["satisfied_at"],
                     "frame_hit_at": t.get("frame_hit_at"),
                     "plan": t["plan"] or [], "frames": [], "why": []}
                if t["plan"]:
                    out = exec_plan(cur["program"], r["seed"], [], t["plan"])
                    grids = [start]
                    for g in out:
                        if g is None:
                            break
                        grids.append(json.loads(g))
                    a["frames"] = [pack(g, pal) for g in grids]
                    a["executed"] = len(grids) - 1
                    rl = t.get("reason_lines") or {}
                    a["why"] = [rl.get(str(k + 1), "") for k in range(len(t["plan"]))]
                attempts.append(a)
        # The reference solution rides along as a labelled row, not an attempt.  Two of the
        # five problems have no successful attempt at all, and a card showing five failures
        # and nothing else gives no sense of what the sentence actually asks for.
        ref_out = exec_plan(cur["program"], r["seed"], [], cur["plan"])
        ref_frames = [start] + [json.loads(g) for g in ref_out if g is not None]
        ref = {"arm": "reference", "n": 0, "error": None, "reasoning": "",
               "success": True, "frame_hit_at": len(ref_frames) - 1,
               "sat": first_satisfied(goal,
                                      [tuple(tuple(x) for x in g) for g in ref_frames],
                                      cur["plan"]),
               "plan": cur["plan"], "frames": [pack(g, pal) for g in ref_frames],
               "why": [""] * len(cur["plan"]), "is_ref": True}

        problems.append({
            "game": r["game"], "human": GAMES[r["game"]][1], "id": r["id"],
            "tier": r["tier"], "nl": r["nl"], "objective": r["objective"],
            "h": r["h"], "nl_h": (v.get("N4") or {}).get("nl_h"),
            "rand": (v.get("N5") or {}).get("floor_at_cap"),
            "seed": r["seed"], "start": pack(start, pal),
            "ref": ref,
            "pass1": {a: r[a]["pass_rate"] for a in arms if a in r},
            "attempts": attempts,
        })
    return {"palette": {i: CSS.get(n, n) for n, i in pal.items()},
            "problems": problems, "chars": _CHARS,
            "model": ev.get("config", {}).get("model", "?"),
            "arms": arms}


STYLE = r""":root{--paper:#f4f5f2;--ink:#1b1f27;--muted:#5c636e;--line:#d9dcd6;--card:#fff;
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
.wrap{max-width:1320px;margin:0 auto;padding:24px 20px 80px}
h1{font:600 22px/1.2 Georgia,"Times New Roman",serif;margin:0 0 4px}
h2{font:600 15px/1.2 Georgia,"Times New Roman",serif;margin:30px 0 12px;
  padding-bottom:6px;border-bottom:1px solid var(--line)}
.sub{color:var(--muted);margin:0 0 18px;max-width:76ch}
.mono{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
.bar{position:sticky;top:0;z-index:5;background:var(--paper);border-bottom:1px solid var(--line);
  display:flex;flex-wrap:wrap;gap:10px 14px;align-items:center;padding:12px 0;margin-bottom:16px}
.bar label{font-size:12px;color:var(--muted);margin-right:4px}
select{font:inherit;color:var(--ink);background:var(--card);border:1px solid var(--line);
  border-radius:7px;padding:5px 8px}
.count{margin-left:auto;color:var(--muted);font-size:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
  padding:16px 18px;margin-bottom:18px}
.head{display:flex;flex-wrap:wrap;gap:8px;align-items:baseline}
.title{font-weight:600;font-size:15px}
.chip{background:var(--chip);border-radius:999px;padding:2px 10px;font-size:12px;white-space:nowrap}
.chip.b{color:#fff;background:var(--accent)}
.chip.ok{background:var(--pass);color:var(--good)}
.chip.no{color:var(--bad)}
.chip.r{background:var(--good);color:#fff}
.att.ref{background:linear-gradient(90deg,var(--chip),transparent 45%);
  border-radius:8px;padding-left:10px}
.goal{display:flex;gap:16px;align-items:flex-start;margin:12px 0 4px;
  padding:12px 14px;border:1px solid var(--line);border-radius:10px;background:var(--paper)}
.goal .nl{font:italic 16px/1.45 Georgia,"Times New Roman",serif;max-width:70ch}
.goal .meta{color:var(--muted);font-size:12px;margin-top:6px}
.att{border-top:1px solid var(--line);margin-top:14px;padding-top:12px}
.att.pass{background:linear-gradient(90deg,var(--pass),transparent 45%);
  border-radius:8px;padding-left:10px}
.ahead{display:flex;flex-wrap:wrap;gap:8px;align-items:baseline;margin-bottom:6px}
.pre{color:var(--muted);font-size:12.5px;margin:0 0 8px;max-width:88ch;white-space:pre-wrap}
.run{display:flex;gap:10px;align-items:stretch;overflow-x:auto;padding:4px 0 10px}
.slot{display:flex;flex-direction:column;gap:4px;flex:0 0 auto;width:118px}
.slot .act{font-size:11px;padding:1px 7px;border-radius:5px;background:var(--act);
  font-family:ui-monospace,Menlo,monospace;white-space:nowrap;overflow:hidden;
  text-overflow:ellipsis;text-align:center}
.slot .cap{font-size:10.5px;color:var(--muted);text-align:center}
.slot .why{font-size:10.5px;line-height:1.35;color:var(--muted);
  max-height:4.6em;overflow:hidden;cursor:pointer}
.slot .why.open{max-height:none}
.slot .why:empty::before{content:"(no line)";opacity:.4;font-style:italic}
canvas{display:block;margin:0 auto;border:1px solid var(--line);border-radius:3px;
  image-rendering:pixelated}
.hit canvas{outline:2px solid var(--good);outline-offset:1px}
.after{opacity:.55}
.err{color:var(--bad);font-size:12px}
.foot{color:var(--muted);font-size:12px;margin-top:6px}
"""


HTML = r"""<!doctype html><meta charset="utf-8">
<title>NL-goal planning: problems and attempts</title>
<style>
""" + STYLE + r"""</style>
<div class="wrap">
  <h1>NL-goal planning: problems and attempts</h1>
  <p class="sub">One card per problem. The agent was shown the <strong>start state</strong>
  and the <strong>sentence</strong> &mdash; and nothing else: an NL goal shows no target
  frame, so everything to the right of the goal box is the planner working from the words
  alone. Each attempt is its plan replayed in the Autumn engine, the action above each frame
  and the planner's own line for that action below it. The frame where the checker
  <em>first accepts</em> is ringed in green; frames after it are dimmed &mdash; scoring is
  any-step, so the run had already succeeded there. <span class="mono">rand@50</span> is the
  random-plan floor at the 50-action budget: the number each score has to beat.</p>
  <div class="bar">
    <span><label>game</label><select id="fg"></select></span>
    <span><label>show</label><select id="fo">
      <option value="all">all attempts</option>
      <option value="pass">successes only</option>
      <option value="fail">failures only</option></select></span>
    <span><label>rationales</label><select id="fw">
      <option value="clip">clipped</option>
      <option value="full">expanded</option></select></span>
    <span><label for="fr">reference</label><input type="checkbox" id="fr" checked></span>
    <span class="count" id="count"></span>
  </div>
  <div id="list"></div>
</div>
<script>
const DATA = /*DATA*/{};
const PAL = DATA.palette, P = DATA.problems, CH = DATA.chars;
const CELL = 6;
function draw(packed){
  const rows = packed.split("|");
  const R = rows.length, C = rows[0].length;
  const cv = document.createElement("canvas");
  cv.width = C*CELL; cv.height = R*CELL;
  const x = cv.getContext("2d");
  for(let r=0;r<R;r++) for(let c=0;c<C;c++){
    x.fillStyle = PAL[CH.indexOf(rows[r][c])] || "#f0f";
    x.fillRect(c*CELL, r*CELL, CELL, CELL);
  }
  return cv;
}
function el(tag, cls, txt){
  const e = document.createElement(tag);
  if(cls) e.className = cls;
  if(txt !== undefined) e.textContent = txt;
  return e;
}
function slot(frame, action, cap, why, cls){
  const s = el("div", "slot" + (cls||""));
  s.appendChild(el("div", "act", action));
  s.appendChild(draw(frame));
  s.appendChild(el("div", "cap", cap));
  const w = el("div", "why", why || "");
  w.onclick = () => w.classList.toggle("open");
  s.appendChild(w);
  return s;
}
function attempt(a){
  const d = el("div", "att" + (a.is_ref ? " ref" : a.success ? " pass" : ""));
  const h = el("div", "ahead");
  h.appendChild(el("span", "chip" + (a.is_ref ? " r" : " b"),
    a.is_ref ? "reference solution — not an attempt" : a.arm + " · attempt " + a.n));
  if(a.error) h.appendChild(el("span", "chip no", a.error));
  else if(a.success) h.appendChild(el("span", "chip ok",
      "satisfied at step " + a.sat + " of " + a.plan.length));
  else h.appendChild(el("span", "chip no", "not satisfied in " + a.plan.length + " actions"));
  if(a.frame_hit_at && !a.is_ref) h.appendChild(el("span", "chip",
      "also hit the exact curated frame at " + a.frame_hit_at));
  d.appendChild(h);
  const pre = (a.reasoning||"").split("\n").filter(l => !/^\s*\d+\s*[.)]/.test(l))
                .join("\n").trim();
  if(pre) d.appendChild(el("p", "pre", pre));
  if(!a.frames.length){ d.appendChild(el("div","err","no plan was executed")); return d; }
  const run = el("div", "run");
  run.appendChild(slot(a.frames[0], "start", "t0", ""));
  for(let k=1;k<a.frames.length;k++){
    const after = a.sat !== null && k > a.sat;
    run.appendChild(slot(a.frames[k], a.plan[k-1], "t"+k, a.why[k-1],
      (a.sat === k ? " hit" : "") + (after ? " after" : "")));
  }
  d.appendChild(run);
  if(a.plan.length > a.frames.length-1)
    d.appendChild(el("div","foot","episode ended after "+(a.frames.length-1)+
      " of "+a.plan.length+" actions"));
  return d;
}
function card(p){
  const c = el("div", "card");
  const h = el("div", "head");
  h.appendChild(el("span", "title", p.game + " / " + p.id));
  h.appendChild(el("span", "chip", p.human));
  h.appendChild(el("span", "chip", p.tier));
  h.appendChild(el("span", "chip", "reference h " + p.h +
      (p.nl_h && p.nl_h !== p.h ? " · sentence needs " + p.nl_h : "")));
  if(p.rand !== null && p.rand !== undefined)
    h.appendChild(el("span", "chip" + (p.rand > 0.02 ? " no" : ""),
      "rand@50 " + p.rand.toFixed(3)));
  for(const a in p.pass1)
    h.appendChild(el("span", "chip" + (p.pass1[a] > 0 ? " ok" : " no"),
      a + " pass@1 " + p.pass1[a].toFixed(2)));
  c.appendChild(h);
  const g = el("div", "goal");
  g.appendChild(draw(p.start));
  const right = el("div");
  right.appendChild(el("div", "nl", "“" + p.nl + "”"));
  right.appendChild(el("div", "meta",
    "start state, seed " + p.seed + " · curated objective: " + p.objective));
  g.appendChild(right);
  c.appendChild(g);
  const show = document.getElementById("fo").value;
  if(document.getElementById("fr").checked) c.appendChild(attempt(p.ref));
  let n = 0;
  for(const a of p.attempts){
    if(show === "pass" && !a.success) continue;
    if(show === "fail" && a.success) continue;
    c.appendChild(attempt(a)); n++;
  }
  if(!n) c.appendChild(el("div", "foot", "no attempts match the current filter"));
  return c;
}
function render(){
  const game = document.getElementById("fg").value;
  const list = document.getElementById("list");
  list.textContent = "";
  const shown = P.filter(p => game === "all" || p.game === game);
  let games = [];
  for(const p of shown) if(!games.includes(p.game)) games.push(p.game);
  for(const gm of games){
    list.appendChild(el("h2", null, gm + " — " + GAMES_H[gm]));
    for(const p of shown.filter(x => x.game === gm)) list.appendChild(card(p));
  }
  const na = shown.reduce((s,p) => s + p.attempts.length, 0);
  document.getElementById("count").textContent =
    shown.length + " problems · " + na + " attempts · planner " + DATA.model;
  if(document.getElementById("fw").value === "full")
    document.querySelectorAll(".why").forEach(w => w.classList.add("open"));
}
const GAMES_H = {};
for(const p of P) GAMES_H[p.game] = p.human;
const fg = document.getElementById("fg");
fg.appendChild(new Option("all games", "all"));
for(const gm in GAMES_H) fg.appendChild(new Option(gm + " / " + GAMES_H[gm], gm));
for(const id of ["fg","fo","fw","fr"]) document.getElementById(id).onchange = render;
render();
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="logs/2026-08-19/nl_pilot/eval/offline.json")
    ap.add_argument("--validation", default="logs/2026-08-19/nl_pilot/validation.json")
    ap.add_argument("--problems", default="logs/2026-08-18/curated/problems.json")
    ap.add_argument("--out", default="logs/2026-08-19/nl_pilot/viz.html")
    a = ap.parse_args()

    ev = json.loads(Path(a.eval).read_text())
    val = {r["pid"]: r for r in json.loads(Path(a.validation).read_text())}
    curated = {r["id"]: r for r in json.loads(Path(a.problems).read_text())}
    arms = ev.get("config", {}).get("arms", ["lmwm"])

    data = build(ev, val, curated, arms)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HTML.replace("/*DATA*/{}", json.dumps(data, separators=(",", ":"))))
    n_att = sum(len(p["attempts"]) for p in data["problems"])
    n_fr = sum(len(at["frames"]) for p in data["problems"] for at in p["attempts"])
    print(f"wrote {out}  ({len(data['problems'])} problems, {n_att} attempts, "
          f"{n_fr} frames, {out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
