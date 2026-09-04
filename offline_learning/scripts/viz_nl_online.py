#!/usr/bin/env python3
"""Render the ONLINE (receding-horizon) NL-goal run: what was planned vs what was done.

The open-loop viz (`viz_nl_goals.py`) draws one plan per attempt, because there is one.  A
closed-loop rollout has a plan PER ROUND and executes only its first action, so the thing
worth looking at is not the trajectory alone but the gap between intention and execution:
each round shows the action that ran, the frame it produced, the planner's line for it, and
the TAIL IT THREW AWAY -- the rest of the plan it had just written and will now rewrite.

Each round is also marked `followed` or `replanned`, by comparing the executed action against
the first action of the tail carried over from the previous round.  A rollout that says
`replanned` every round is not executing a plan at all, it is being led one step at a time;
one that says `followed` throughout is running an open-loop plan through a closed-loop
harness.  Neither is visible in a pass rate.

    uv run python offline_learning/scripts/viz_nl_online.py \
        --eval logs/2026-08-19/nl_pilot/eval/online.json \
        --offline logs/2026-08-19/nl_pilot/eval/offline.json \
        --out logs/2026-08-19/nl_pilot/viz_online.html
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(REPO), str(REPO / "offline_learning"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from offline_learning.human_replay import GAMES  # noqa: E402
from viz_nl_goals import CSS, STYLE, _CHARS, pack  # noqa: E402


def build(ev: dict, off_idx: dict, curated: dict, arms: list[str]) -> dict:
    pal: dict[str, int] = {}
    problems = []
    for r in ev["rows"]:
        cur = curated[r["id"]]
        rollouts = []
        for arm in arms:
            if arm not in r:
                continue
            for i, a in enumerate(r[arm]["attempts"]):
                rounds = []
                carry: list[str] = []
                for rd in a["rounds"]:
                    plan = rd["plan"] or []
                    followed = bool(carry) and bool(plan) and carry[0] == rd["executed"]
                    rounds.append({
                        # Fall back to the whole <reasoning> block: only ~10% of rounds
                        # number their rationale lines, and online only the FIRST action of
                        # each plan executes, so the block is about that action anyway.
                        "n": rd["n"], "action": rd["executed"],
                        "why": (rd.get("why") or "").strip() or (rd.get("reasoning") or ""),
                        "tail": plan[1:], "remaining": rd["remaining"],
                        "error": rd["plan_error"], "satisfied": rd.get("satisfied", False),
                        "followed": followed, "first": not carry,
                        "frame": pack(json.loads(rd["grid_after"]), pal)
                                 if rd.get("grid_after") else None,
                    })
                    carry = plan[1:]
                rollouts.append({"n": i + 1, "arm": arm, "success": a["success"],
                                 "sat": a["satisfied_at"], "used": a["actions_used"],
                                 "failed": a["failed_reason"], "frame_hit": a["frame_hit"],
                                 "rounds": rounds})
        problems.append({
            "game": r["game"], "human": GAMES[r["game"]][1], "id": r["id"],
            "tier": r["tier"], "nl": r["nl"], "objective": r["objective"],
            "h": r["h"], "seed": r["seed"], "rand": r["rand"],
            "start": pack(cur["start"], pal),
            "arms": {a: {"on1": r[a]["pass_rate"], "on5": bool(r[a]["pass_any"]),
                         "used": r[a]["mean_used"], "frame": r[a]["frame_rate"],
                         "off1": off_idx.get((r["id"], a))}
                     for a in arms if a in r},
            "rollouts": rollouts,
        })
    return {"palette": {i: CSS.get(n, n) for n, i in pal.items()},
            "problems": problems, "chars": _CHARS, "armlist": arms,
            "model": ev.get("config", {}).get("model", "?"),
            "cap": ev.get("config", {}).get("max_actions")}


EXTRA = r"""
.tail{font-size:10px;line-height:1.3;color:var(--muted);font-family:ui-monospace,Menlo,monospace;
  max-height:3.2em;overflow:hidden;cursor:pointer;border-top:1px dashed var(--line);padding-top:2px}
.tail.open{max-height:none}
.tail:empty::before{content:"nothing left to plan";opacity:.4;font-style:italic}
.mark{font-size:9.5px;letter-spacing:.02em;text-align:center;border-radius:4px;padding:0 4px}
.mark.rep{color:var(--bad)}
.mark.fol{color:var(--good)}
.mark.non{color:var(--muted);opacity:.6}
.legend{color:var(--muted);font-size:12px;margin:2px 0 10px}
table.sum{border-collapse:collapse;margin:4px 0 18px;font-size:13px;width:100%;
  background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
table.sum th{text-align:right;font-weight:600;color:var(--muted);font-size:11.5px;
  letter-spacing:.03em;text-transform:uppercase;padding:9px 12px;
  border-bottom:1px solid var(--line);white-space:nowrap}
table.sum th:first-child,table.sum td:first-child{text-align:left}
table.sum td{text-align:right;padding:8px 12px;border-bottom:1px solid var(--line);
  font-variant-numeric:tabular-nums}
table.sum tr:last-child td{border-bottom:none}
table.sum tbody tr{cursor:pointer}
table.sum tbody tr:hover{background:var(--chip)}
table.sum td.big{font-weight:600;font-size:14px}
table.sum td.gain{color:var(--good);font-weight:600}
table.sum td.flat{color:var(--muted)}
table.sum td.warn{color:var(--bad)}
table.sum .id{font-family:ui-monospace,Menlo,monospace}
table.sum .g{color:var(--muted);font-size:12px}
.io-btn{margin-top:auto;border:1px solid var(--line);border-radius:5px;padding:3px 5px;
  color:var(--accent);background:var(--card);font:10px/1.2 ui-sans-serif,system-ui,sans-serif;
  cursor:pointer}
.io-btn:hover{background:var(--chip)}
dialog.io-modal{width:min(1120px,calc(100vw - 32px));max-height:calc(100vh - 32px);
  color:var(--ink);background:var(--card);border:1px solid var(--line);border-radius:12px;
  padding:0;box-shadow:0 18px 60px #0006}
dialog.io-modal::backdrop{background:#0008}
.io-head{position:sticky;top:0;z-index:1;display:flex;align-items:center;gap:12px;
  padding:12px 16px;background:var(--card);border-bottom:1px solid var(--line)}
.io-head strong{font-size:14px}.io-head span{color:var(--muted);font-size:12px}
.io-close{margin-left:auto;border:1px solid var(--line);border-radius:6px;padding:4px 9px;
  color:var(--ink);background:var(--chip);cursor:pointer}
.io-grid{display:grid;grid-template-columns:1fr 1fr;gap:0}
.io-pane{min-width:0;padding:14px 16px}.io-pane+.io-pane{border-left:1px solid var(--line)}
.io-pane h3{font-size:12px;text-transform:uppercase;letter-spacing:.04em;margin:0 0 3px}
.io-note{min-height:2.8em;color:var(--muted);font-size:11px;margin-bottom:8px}
.io-pane pre{margin:0;padding:12px;border:1px solid var(--line);border-radius:7px;
  background:var(--paper);color:var(--ink);font:11px/1.45 ui-monospace,Menlo,Consolas,monospace;
  white-space:pre-wrap;overflow-wrap:anywhere}
@media(max-width:760px){.io-grid{grid-template-columns:1fr}.io-pane+.io-pane{
  border-left:0;border-top:1px solid var(--line)}}
"""

HTML = r"""<!doctype html><meta charset="utf-8">
<title>NL-goal planning, online</title>
<style>
""" + STYLE + EXTRA + r"""</style>
<div class="wrap">
  <h1>NL-goal planning: online (receding horizon)</h1>
  <p class="sub">Same five problems and the same sentences as the open-loop run, closed into
  a loop: each round the planner writes a plan of at most <span class="mono">budget &minus;
  n</span> actions, <strong>only the first one runs</strong>, and it replans from what it
  then sees. So each slot below is one round &mdash; the action that executed, the frame it
  produced, the planner's line for it, and underneath, dashed off, the <em>tail it discarded</em>:
  the rest of the plan it had just written. A round is
  <span class="mark fol">followed</span> when the executed action is the one the previous
  round had queued up next, and <span class="mark rep">replanned</span> when it is not.
  The round where the checker accepts is ringed &mdash; the harness stops the rollout there.
  <span class="mono">replanned</span> below is the share of rounds that departed from the
  queued plan, and <span class="mono">rand@50</span> the random-plan floor the score has to
  beat. Click a row to filter to that game.</p>
  <table class="sum" id="sum"></table>
  <div class="bar">
    <span><label>game</label><select id="fg"></select></span>
    <span><label>show</label><select id="fo">
      <option value="all">all rollouts</option>
      <option value="pass">successes only</option>
      <option value="fail">failures only</option></select></span>
    <span><label>detail</label><select id="fw">
      <option value="clip">clipped</option>
      <option value="full">expanded</option></select></span>
    <span class="count" id="count"></span>
  </div>
  <div id="list"></div>
</div>
<dialog class="io-modal" id="ioModal">
  <form method="dialog" class="io-head">
    <strong id="ioTitle">Step model I/O</strong><span id="ioMeta"></span>
    <button class="io-close" value="close">close</button>
  </form>
  <div class="io-grid">
    <section class="io-pane"><h3>Prompt</h3><div class="io-note" id="promptNote"></div>
      <pre id="promptText"></pre></section>
    <section class="io-pane"><h3>Response</h3><div class="io-note" id="responseNote"></div>
      <pre id="responseText"></pre></section>
  </div>
</dialog>
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
function showIO(rd){
  document.getElementById("ioMeta").textContent =
    "round " + rd.n + " · " + rd.remaining + " actions left";
  document.getElementById("promptNote").textContent = rd.promptNote || "";
  document.getElementById("responseNote").textContent = rd.responseNote || "";
  document.getElementById("promptText").textContent = rd.prompt || "(unavailable)";
  document.getElementById("responseText").textContent = rd.response || "(unavailable)";
  document.getElementById("ioModal").showModal();
}
function roundSlot(rd){
  const s = el("div", "slot" + (rd.satisfied ? " hit" : ""));
  s.appendChild(el("div", "act", rd.action || "(no plan)"));
  if(rd.frame) s.appendChild(draw(rd.frame));
  s.appendChild(el("div", "cap", "round " + rd.n + " · " + rd.remaining + " left"));
  s.appendChild(el("div", "mark " + (rd.first ? "non" : rd.followed ? "fol" : "rep"),
    rd.first ? "first round" : rd.followed ? "followed" : "replanned"));
  const w = el("div", "why", rd.why || "");
  w.onclick = () => w.classList.toggle("open");
  s.appendChild(w);
  const t = el("div", "tail", rd.tail.length ? "then: " + rd.tail.join(", ") : "");
  t.onclick = () => t.classList.toggle("open");
  s.appendChild(t);
  if(rd.error) s.appendChild(el("div", "err", rd.error));
  if(rd.prompt !== undefined || rd.response !== undefined){
    const b = el("button", "io-btn", "prompt + response");
    b.type = "button"; b.onclick = () => showIO(rd); s.appendChild(b);
  }
  return s;
}
function rollout(ro){
  const d = el("div", "att" + (ro.success ? " pass" : ""));
  const h = el("div", "ahead");
  h.appendChild(el("span", "chip b", "rollout " + ro.n));
  h.appendChild(el("span", "chip " + (ro.success ? "ok" : "no"),
    ro.success ? "satisfied after " + ro.sat + " actions"
               : "failed: " + (ro.failed || "?") + " (" + ro.used + " actions used)"));
  const rep = ro.rounds.filter(r => !r.first && !r.followed).length;
  h.appendChild(el("span", "chip", rep + " of " + Math.max(ro.rounds.length-1,0) +
    " rounds replanned"));
  if(ro.arm) h.appendChild(el("span", "chip", ro.arm));
  if(ro.frame_hit) h.appendChild(el("span", "chip", "also hit the exact curated frame"));
  d.appendChild(h);
  const run = el("div", "run");
  for(const rd of ro.rounds) run.appendChild(roundSlot(rd));
  d.appendChild(run);
  return d;
}
function card(p){
  const c = el("div", "card");
  const h = el("div", "head");
  h.appendChild(el("span", "title", p.game + " / " + p.id));
  h.appendChild(el("span", "chip", p.human));
  h.appendChild(el("span", "chip", p.tier));
  h.appendChild(el("span", "chip", "reference h " + p.h));
  for(const arm of DATA.armlist){
    const d = p.arms[arm]; if(!d) continue;
    h.appendChild(el("span", "chip" + (d.on1 > 0 ? " ok" : " no"),
      arm + " online " + d.on1.toFixed(2) +
      (d.off1 === null || d.off1 === undefined ? "" : " (offline " + d.off1.toFixed(2) + ")") +
      " · " + d.used.toFixed(1) + " actions"));
  }
  if(p.rand !== null && p.rand !== undefined)
    h.appendChild(el("span", "chip" + (p.rand > 0.02 ? " no" : ""),
      "rand@50 " + p.rand.toFixed(3)));
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
  let n = 0;
  for(const ro of p.rollouts){
    if(show === "pass" && !ro.success) continue;
    if(show === "fail" && ro.success) continue;
    c.appendChild(rollout(ro)); n++;
  }
  if(!n) c.appendChild(el("div", "foot", "no rollouts match the current filter"));
  return c;
}
function replanRate(p, arm){
  let tot = 0, rep = 0;
  for(const ro of p.rollouts){
    if(arm && ro.arm !== arm) continue;
    for(const rd of ro.rounds){
      if(rd.first) continue;
      tot++; if(!rd.followed) rep++;
    }
  }
  return tot ? rep/tot : null;
}
function summary(){
  const t = document.getElementById("sum");
  t.textContent = "";
  const head = el("tr");
  for(const h of ["problem","arm","h","online @1","offline @1","\u0394",
                  "actions used","replanned","frame@1","rand@50"])
    head.appendChild(el("th", null, h));
  t.appendChild(el("thead")).appendChild(head);
  const body = el("tbody");
  for(const p of P) for(const arm of DATA.armlist){
    const d = p.arms[arm]; if(!d) continue;
    const tr = el("tr");
    tr.onclick = () => { document.getElementById("fg").value = p.game; render();
                         window.scrollTo({top:0,behavior:"smooth"}); };
    const name = el("td");
    name.appendChild(el("span", "g", p.game + " / "));
    name.appendChild(el("span", "id", p.id));
    tr.appendChild(name);
    tr.appendChild(el("td", "flat", arm));
    tr.appendChild(el("td", null, String(p.h)));
    tr.appendChild(el("td", "big", d.on1.toFixed(2)));
    const has = d.off1 !== null && d.off1 !== undefined;
    tr.appendChild(el("td", "flat", has ? d.off1.toFixed(2) : "--"));
    const dd = has ? d.on1 - d.off1 : null;
    tr.appendChild(el("td", dd > 0 ? "gain" : "flat",
      dd === null ? "--" : (dd > 0 ? "+" : "") + dd.toFixed(2)));
    tr.appendChild(el("td", null, d.used.toFixed(1)));
    const rr = replanRate(p, arm);
    tr.appendChild(el("td", null, rr === null ? "--" : Math.round(rr*100) + "%"));
    tr.appendChild(el("td", null, d.frame.toFixed(2)));
    tr.appendChild(el("td", p.rand > 0.02 ? "warn" : "flat",
      p.rand === null || p.rand === undefined ? "--" : p.rand.toFixed(3)));
    body.appendChild(tr);
  }
  t.appendChild(body);
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
  const nr = shown.reduce((s,p) => s + p.rollouts.reduce((t,r) => t + r.rounds.length, 0), 0);
  document.getElementById("count").textContent =
    shown.length + " problems · " + nr + " rounds · cap " + DATA.cap +
    " actions · planner " + DATA.model;
  if(document.getElementById("fw").value === "full")
    document.querySelectorAll(".why,.tail").forEach(w => w.classList.add("open"));
}
const GAMES_H = {};
for(const p of P) GAMES_H[p.game] = p.human;
const fg = document.getElementById("fg");
fg.appendChild(new Option("all games", "all"));
for(const gm in GAMES_H) fg.appendChild(new Option(gm + " / " + GAMES_H[gm], gm));
for(const id of ["fg","fo","fw"]) document.getElementById(id).onchange = render;
summary();
render();
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="logs/2026-08-19/nl_pilot/eval/online.json")
    ap.add_argument("--offline", default="logs/2026-08-19/nl_pilot/eval/offline.json")
    ap.add_argument("--problems", default="logs/2026-08-18/curated/problems.json")
    ap.add_argument("--out", default="logs/2026-08-19/nl_pilot/viz_online.html")
    a = ap.parse_args()

    ev = json.loads(Path(a.eval).read_text())
    curated = {r["id"]: r for r in json.loads(Path(a.problems).read_text())}
    arms = ev.get("config", {}).get("arms", ["lmwm"])
    off_idx = {}
    if Path(a.offline).exists():
        off = json.loads(Path(a.offline).read_text())
        off_idx = {(r["id"], arm): r[arm]["pass_rate"]
                   for r in off["rows"] for arm in arms if arm in r}

    data = build(ev, off_idx, curated, arms)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HTML.replace("/*DATA*/{}", json.dumps(data, separators=(",", ":"))))
    nr = sum(len(ro["rounds"]) for p in data["problems"] for ro in p["rollouts"])
    print(f"wrote {out}  ({len(data['problems'])} problems, "
          f"{sum(len(p['rollouts']) for p in data['problems'])} rollouts, {nr} rounds, "
          f"{out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
