#!/usr/bin/env python3
"""Build a self-contained browser viewer for eval_forward_modes.py results.

The evaluation JSON stores predictions, responses, targets, and scores, but the
prompts were not persisted. This tool deterministically reconstructs the exact
test split and prompt text using the same evaluator helpers, verifies that the
reconstructed current/next representations match the saved rows, and embeds
everything in one HTML file.

Usage:
    uv run python offline_learning/scripts/build_forward_eval_viz.py
"""
from __future__ import annotations

import os as _bos, sys as _bsys  # offline_learning/ on sys.path (flat-import the kept libs)
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import eval_forward_modes as efm  # noqa: E402
from invdyn_core import (  # noqa: E402
    DEFAULT_KNOWLEDGE,
    FWD_WINDOW_TMPL,
    INV_WINDOW_TMPL,
    _forward_transcript,
    _inverse_transcript,
    build_window,
)
from validate_beliefs import PREDICT_TMPL as ID_TWO_STATE_TMPL  # noqa: E402


def learned_prompt(win: dict, action: str, beliefs: str) -> str:
    return FWD_WINDOW_TMPL.format(
        beliefs=beliefs.strip() or "(empty)",
        default_knowledge=DEFAULT_KNOWLEDGE,
        transcript=_forward_transcript(win),
        action=action,
    )


def raw_prompt(tr) -> str:
    return efm.RAW_FORWARD_TMPL.format(
        default_knowledge=DEFAULT_KNOWLEDGE,
        transcript=efm.raw_transcript(tr),
        action=tr.action,
    )


def slim_row(row: dict, prompt: str) -> dict:
    return {
        "start": row["start"],
        "target": row["target"],
        "pred": row["pred"],
        "prompt": prompt,
        "response": row["response"],
        "changed": row["changed"],
        "exact": row["exact"],
        "partial": row["partial"],
        "stale_exact": row["stale_exact"],
        "stale_partial": row["stale_partial"],
        "perception_error": row["perception_error"],
        "cost": row["cost"],
    }


def load_inverse_trace(path: Path) -> dict:
    trace = json.loads(path.read_text())
    if not trace.get("records"):
        raise ValueError(f"inverse-dynamics trace has no records: {path}")
    return trace


def learned_inverse_prompt(win: dict, beliefs: str, choices: list[str]) -> str:
    return INV_WINDOW_TMPL.format(
        beliefs=beliefs.strip() or "(empty)",
        default_knowledge=DEFAULT_KNOWLEDGE,
        transcript=_inverse_transcript(win),
        choices="\n".join(f"- {choice}" for choice in choices),
    )


def raw_inverse_prompt(record: dict) -> str:
    return ID_TWO_STATE_TMPL.format(
        beliefs="(empty)",
        z_t=record.get("z_t") or "(empty)",
        z_t1=record.get("z_t1") or "(empty)",
        choices="\n".join(f"- {choice}" for choice in record["choices"]),
    )


def slim_inverse(record: dict) -> dict:
    return {
        "start": record.get("z_t", ""),
        "target": record.get("z_t1", ""),
        "pred": record.get("pred", ""),
        "prompt": record.get("prompt", ""),
        "response": record.get("response", ""),
        "response_note": "Full original model response saved by the test50 ID evaluator.",
        "choices": record.get("choices", []),
        "changed": str(record.get("z_t", "")).strip() != str(record.get("z_t1", "")).strip(),
        "exact": float(record.get("exact", bool(record.get("correct")))),
    }


def build_payload(source: dict, id_source: dict) -> dict:
    cfg = source["config"]
    id_by_game = {result["game"]: result for result in id_source["results"]}
    data_root = Path(cfg["data_root"])
    games = []
    errors = []

    for result in source["results"]:
        game = result["game"]
        transitions, train_total, test_total = efm.reproduce_test_split(
            game=game,
            data_root=data_root,
            test_dir=cfg["test_dir"],
            seed=cfg["seed"],
            context_k=cfg["context_k"],
            test_n=cfg["test_n"],
        )
        rows = {
            mode: {row["idx"]: row for row in result["rows"] if row["mode"] == mode}
            for mode in ("raw", "learned")
        }
        artifact_dir = Path(result["artifact_dir"])
        code = (artifact_dir / f"best_perception_gepa_seed{cfg['seed']}.py").read_text()
        beliefs_path = artifact_dir / f"best_beliefs_gepa_seed{cfg['seed']}.txt"
        beliefs = beliefs_path.read_text() if beliefs_path.exists() else ""
        if game not in id_by_game:
            raise ValueError(f"{game}: missing test50 ID result")
        id_result = id_by_game[game]
        if Path(id_result["artifact_dir"]) != artifact_dir:
            raise ValueError(
                f"{game}: FD artifact {artifact_dir} != ID artifact "
                f"{id_result[artifact_dir]}"
            )
        id_rows = {
            mode: {
                int(row["idx"]): row
                for row in id_result["rows"]
                if row["mode"] == mode
            }
            for mode in ("raw", "learned")
        }
        learned_id_n = len(id_rows["learned"])
        raw_id_n = len(id_rows["raw"])
        if learned_id_n != raw_id_n:
            raise ValueError(f"{game}: raw/learned inverse row lengths differ")
        learned_id_exact = id_result["summary"]["learned"]["exact"]
        raw_id_exact = id_result["summary"]["raw"]["exact"]
        composite = {
            "learned": (learned_id_exact + result["summary"]["learned"]["exact"]) / 2,
            "raw": (raw_id_exact + result["summary"]["raw"]["exact"]) / 2,
        }


        items = []
        for idx, tr in enumerate(transitions):
            raw = rows["raw"][idx]
            learned = rows["learned"][idx]
            win, perception_error = build_window(code, tr)

            checks = {
                "action": raw["action"] == learned["action"] == tr.action,
                "raw_start": raw["start"] == efm.canonical_grid(tr.x_t),
                "raw_target": raw["target"] == efm.canonical_grid(tr.x_t1),
                "learned_start": learned["start"] == win["z_t"],
                "learned_target": learned["target"] == win["z_t1"],
                "perception_error": learned["perception_error"] == perception_error,
            }
            if not all(checks.values()):
                errors.append({"game": game, "idx": idx, "checks": checks})

            items.append(
                {
                    "idx": idx,
                    "action": tr.action,
                    "raw": slim_row(raw, raw_prompt(tr)),
                    "learned": slim_row(
                        learned, learned_prompt(win, tr.action, beliefs)
                    ),
                }
            )

        if len(transitions) != learned_id_n:
            raise ValueError(
                f"{game}: reconstructed {len(transitions)} ID items, "
                f"expected {learned_id_n}"
            )
        id_items = []
        for idx, tr in enumerate(transitions):
            raw_record = id_rows["raw"][idx]
            learned_record = id_rows["learned"][idx]
            id_checks = {
                "truth": tr.action == raw_record.get("truth") == learned_record.get("truth"),
                "raw_start": raw_record.get("z_t") == tr.x_t[:6000],
                "raw_target": raw_record.get("z_t1") == tr.x_t1[:6000],
                "choices": raw_record.get("choices") == learned_record.get("choices"),
                "raw_grid_start": raw_record.get("raw_grid_start") == efm.canonical_grid(tr.x_t),
                "raw_grid_target": raw_record.get("raw_grid_target") == efm.canonical_grid(tr.x_t1),
            }
            if not all(id_checks.values()):
                errors.append(
                    {"game": game, "task": "id", "idx": idx, "checks": id_checks}
                )
            id_items.append(
                {
                    "idx": idx,
                    "action": tr.action,
                    "raw_grid_start": raw_record["raw_grid_start"],
                    "raw_grid_target": raw_record["raw_grid_target"],
                    "raw": slim_inverse(raw_record),
                    "learned": slim_inverse(learned_record),
                    "perception_error": learned_record.get("perception_error"),
                }
            )

        games.append(
            {
                "game": game,
                "artifact_dir": str(artifact_dir),
                "beliefs": beliefs,
                "train_total": train_total,
                "test_total": test_total,
                "summary": result["summary"],
                "inverse_exact": {"learned": learned_id_exact, "raw": raw_id_exact},
                "inverse_n": {"learned": learned_id_n, "raw": raw_id_n},
                "composite_exact": composite,
                "items": items,
                "id_items": id_items,
            }
        )

    games.sort(key=lambda item: (-item["composite_exact"]["learned"], item["game"]))
    return {
        "title": "Dynamics evaluation: raw vs learned",
        "config": cfg,
        "aggregate": source["aggregate"],
        "alignment_ok": not errors,
        "alignment_errors": errors,
        "games": games,
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f4f6f8;--panel:#fff;--ink:#17202a;--muted:#657384;--line:#dce2e8;
--blue:#1769d2;--green:#137333;--red:#b3261e;--amber:#9a6700;--raw:#7357c5;--learned:#087f8c}
*{box-sizing:border-box} body{margin:0;font:13px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;
background:var(--bg);color:var(--ink)} button,select,input{font:inherit}
header{position:sticky;top:0;z-index:20;background:#fff;border-bottom:1px solid var(--line);
padding:12px 18px;display:flex;align-items:center;gap:14px;box-shadow:0 1px 5px #0000000b}
h1{font-size:17px;margin:0;white-space:nowrap}.meta{color:var(--muted);font-size:12px}
.badge{padding:3px 8px;border-radius:999px;font-size:11px;font-weight:650}
.ok{background:#e6f4ea;color:var(--green)}.bad{background:#fce8e6;color:var(--red)}
.layout{display:grid;grid-template-columns:220px minmax(650px,1fr);min-height:calc(100vh - 57px)}
aside{border-right:1px solid var(--line);background:#fff;padding:12px;position:sticky;top:57px;height:calc(100vh - 57px);overflow:auto}
.game{width:100%;border:1px solid transparent;background:transparent;text-align:left;padding:8px;border-radius:7px;
cursor:pointer;margin-bottom:3px}.game:hover{background:#f4f7fb}.game.active{background:#eaf2ff;border-color:#bfd5f5}
.game b{display:block}.game small{color:var(--muted)}.mini{display:grid;grid-template-columns:1fr 1fr;gap:3px;margin-top:4px}
.mini span{background:#f3f5f7;border-radius:4px;padding:2px 4px;font-size:10px}
main{padding:16px 18px;min-width:0}.toolbar{background:#fff;border:1px solid var(--line);border-radius:9px;
padding:10px;display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:12px}
select,input{border:1px solid #c7d0da;border-radius:6px;padding:6px 8px;background:#fff}
input{min-width:220px}.count{margin-left:auto;color:var(--muted)}
.summary{display:grid;grid-template-columns:repeat(6,minmax(120px,1fr));gap:8px;margin-bottom:12px}
.stat{background:#fff;border:1px solid var(--line);border-radius:8px;padding:9px}.stat .k{color:var(--muted);font-size:11px}
.stat .v{font-size:18px;font-weight:700}.stat.raw{border-left:4px solid var(--raw)}.stat.learned{border-left:4px solid var(--learned)}
.beliefs{background:#fff;border:1px solid var(--line);border-radius:9px;margin-bottom:12px;overflow:hidden}
.beliefs summary{cursor:pointer;padding:10px 12px;font-weight:700;background:#f8fafb}.beliefs .inside{padding:10px 12px}
.beliefs .path{color:var(--muted);font-size:11px;margin-bottom:7px;overflow-wrap:anywhere}
.tablewrap{background:#fff;border:1px solid var(--line);border-radius:9px;overflow:auto}
table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:7px 9px;border-bottom:1px solid #e8edf1;text-align:left;white-space:nowrap}
th{position:sticky;top:0;background:#f8fafb;z-index:2;color:#52606e;font-size:11px}
tbody tr{cursor:pointer}tbody tr:hover{background:#f2f7ff}.selected{background:#eaf2ff!important}
.score{display:inline-block;min-width:47px;text-align:center;border-radius:4px;padding:2px 5px;font-variant-numeric:tabular-nums}
.s1{background:#dcefe2;color:var(--green)}.s0{background:#f8dddd;color:var(--red)}.spart{background:#fff2cc;color:var(--amber)}
.mode{font-weight:650}.rawtxt{color:var(--raw)}.learntxt{color:var(--learned)}
#drawer{position:fixed;right:0;top:0;z-index:50;width:min(1040px,82vw);height:100vh;background:#fff;
border-left:1px solid var(--line);box-shadow:-10px 0 35px #0002;display:none;overflow:auto}
.drawerhead{position:sticky;top:0;background:#fff;border-bottom:1px solid var(--line);padding:12px 16px;
display:flex;align-items:center;gap:9px;z-index:4}.drawerhead h2{font-size:15px;margin:0}.spacer{flex:1}
.btn{border:1px solid #c7d0da;background:#fff;border-radius:6px;padding:5px 9px;cursor:pointer}.btn:hover{background:#f2f5f8}
.drawerbody{padding:14px 16px 30px}.metrics{display:flex;gap:6px;flex-wrap:wrap;margin:7px 0 10px}
.metric{border:1px solid var(--line);border-radius:5px;padding:3px 7px;font-size:11px}
.columns{display:grid;grid-template-columns:1fr 1fr;gap:12px}.card{border:1px solid var(--line);border-radius:9px;overflow:hidden;min-width:0}
.cardhead{padding:9px 11px;background:#f8fafb;border-bottom:1px solid var(--line);font-weight:700}
.card.raw .cardhead{border-left:4px solid var(--raw)}.card.learned .cardhead{border-left:4px solid var(--learned)}
.section{padding:10px 11px;border-bottom:1px solid #e8edf1}.section:last-child{border:0}.section h3{font-size:11px;
text-transform:uppercase;letter-spacing:.05em;color:#536273;margin:0 0 6px}
pre{margin:0;background:#f6f8fa;border:1px solid #e4e9ee;border-radius:6px;padding:8px;white-space:pre-wrap;
overflow:auto;max-height:330px;font:11px/1.4 ui-monospace,SFMono-Regular,Consolas,monospace}
.prompt pre{max-height:520px}.copy{float:right;border:0;background:transparent;color:var(--blue);cursor:pointer;font-size:11px}
.grids{display:flex;gap:10px;overflow:auto;align-items:flex-start}.gridbox{flex:0 0 auto}.gridlabel{font-size:10px;color:var(--muted);margin-bottom:3px}
canvas{display:block;border:1px solid #ccd4dc;image-rendering:pixelated;background:#eee}.parsefail{color:var(--muted);font-size:11px;padding:8px}
.truth{color:var(--muted);font-size:12px}.empty{padding:40px;text-align:center;color:var(--muted)}
@media(max-width:900px){.layout{grid-template-columns:1fr}aside{display:none}.summary{grid-template-columns:1fr 1fr}
#drawer{width:100vw}.columns{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <h1>Dynamics evaluation viewer</h1>
  <span id="align"></span>
  <span class="meta" id="headerMeta"></span>
</header>
<div class="layout">
  <aside><div id="games"></div></aside>
  <main>
    <div class="toolbar">
      <select id="taskFilter"><option value="fd">forward dynamics (FD)</option><option value="id">inverse dynamics (ID)</option></select>
      <select id="modeFilter"><option value="all">raw + learned</option><option value="raw">raw only</option><option value="learned">learned only</option></select>
      <select id="exactFilter"><option value="all">all exact results</option><option value="correct">exact correct</option><option value="wrong">exact wrong</option></select>
      <select id="changeFilter"><option value="all">all transitions</option><option value="changed">changed target</option><option value="stale">unchanged target</option></select>
      <select id="actionFilter"><option value="all">all actions</option></select>
      <input id="search" placeholder="search action, prompt, response…">
      <span class="count" id="count"></span>
    </div>
    <div class="summary" id="summary"></div>
    <div id="beliefs"></div>
    <div class="tablewrap">
      <table><thead><tr id="headrow"></tr></thead>
      <tbody id="rows"></tbody></table>
    </div>
  </main>
</div>
<div id="drawer"><div class="drawerhead">
  <button class="btn" id="close">close ✕</button><button class="btn" id="prev">← prev</button><button class="btn" id="next">next →</button>
  <h2 id="drawerTitle"></h2><span class="spacer"></span><span class="truth" id="drawerTruth"></span>
</div><div class="drawerbody" id="drawerBody"></div></div>
<script>
const DATA=__DATA__;
let gameIndex=0, selectedIndex=null, visible=[];
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=x=>Number(x??0).toFixed(3);
const money=x=>'$'+Number(x??0).toFixed(4);
function score(x){return '<span class="score '+(x===1?'s1':x===0?'s0':'spart')+'">'+fmt(x)+'</span>'}
function currentGame(){return DATA.games[gameIndex]}
function currentTask(){return $('taskFilter').value}
function currentItems(){return currentTask()==='id'?currentGame().id_items:currentGame().items}
function copyText(text,button){navigator.clipboard.writeText(text).then(()=>{const old=button.textContent;button.textContent='copied';setTimeout(()=>button.textContent=old,900)})}
function copySection(button,key,mode){copyText(currentItems()[selectedIndex][mode][key],button)}
function parseGrid(s){
  if(!s)return null; let text=String(s).trim();
  const tag=text.match(/<next_state>\s*([\s\S]*?)\s*<\/next_state>/i);if(tag)text=tag[1].trim();
  const a=text.indexOf('[['),b=text.lastIndexOf(']]');if(a>=0&&b>=a)text=text.slice(a,b+2);
  try{const g=JSON.parse(text);return Array.isArray(g)&&Array.isArray(g[0])?g:null}catch{return null}
}
const ARC={0:'#fff',1:'#ccc',2:'#999',3:'#666',4:'#333',5:'#000',6:'#E53AA3',7:'#FF7BCC',8:'#F93C31',9:'#1E93FF',10:'#88D8F1',11:'#FFDC00',12:'#FF851B',13:'#921231',14:'#4FCC30',15:'#A356D6'};
const probe=document.createElement('span').style;
function color(v){if(typeof v==='number')return ARC[v]||'#e6e';probe.color='';probe.color=String(v);return probe.color||'#e6e'}
function drawGrid(grid){
  if(!grid)return null;const h=grid.length,w=Math.max(...grid.map(r=>r.length));let px=Math.max(3,Math.min(18,Math.floor(250/Math.max(h,w))));
  const c=document.createElement('canvas');c.width=w*px;c.height=h*px;const x=c.getContext('2d');
  for(let r=0;r<h;r++)for(let q=0;q<w;q++){x.fillStyle=color(grid[r][q]);x.fillRect(q*px,r*px,px,px)}
  if(px>=7&&w<=32){x.strokeStyle='#0002';x.lineWidth=1;for(let i=0;i<=w;i++){x.beginPath();x.moveTo(i*px,0);x.lineTo(i*px,h*px);x.stroke()}for(let i=0;i<=h;i++){x.beginPath();x.moveTo(0,i*px);x.lineTo(w*px,i*px);x.stroke()}}
  return c
}
function gridsSection(item){
  const specs=currentTask()==='id'
    ? [['STATE 1 raw grid',item.raw_grid_start],['STATE 2 raw grid',item.raw_grid_target]]
    : [['current raw grid',item.raw.start],['true next raw grid',item.raw.target],['raw predicted grid',item.raw.pred]];
  const wrap=document.createElement('div');wrap.className='grids';
  for(const [label,text] of specs){const box=document.createElement('div');box.className='gridbox';box.innerHTML='<div class="gridlabel">'+esc(label)+'</div>';
    const c=drawGrid(parseGrid(text));if(c)box.appendChild(c);else box.innerHTML+='<div class="parsefail">not parseable as a full grid</div>';wrap.appendChild(box)}
  return wrap
}
function renderGames(){
  $('games').innerHTML=DATA.games.map((g,i)=>`<button class="game ${i===gameIndex?'active':''}" onclick="chooseGame(${i})">
    <b>${esc(g.game)}</b><small>${g.items.length} FD · ${g.id_items.length} ID transitions</small><div class="mini">
    <span>raw C ${fmt(g.composite_exact.raw)}</span><span>learned C ${fmt(g.composite_exact.learned)}</span>
    <span>ID ${fmt(g.inverse_exact.raw)}</span><span>ID ${fmt(g.inverse_exact.learned)}</span>
    <span>FD ${fmt(g.summary.raw.exact)}</span><span>FD ${fmt(g.summary.learned.exact)}</span></div></button>`).join('')
}
function renderSummary(){
  const g=currentGame(),r=g.summary.raw,l=g.summary.learned;
  $('summary').innerHTML=`<div class="stat raw"><div class="k">raw composite exact</div><div class="v">${fmt(g.composite_exact.raw)}</div></div>
  <div class="stat raw"><div class="k">raw ID exact · n=${g.inverse_n.raw}</div><div class="v">${fmt(g.inverse_exact.raw)}</div></div>
  <div class="stat raw"><div class="k">raw FD exact · n=${r.n}</div><div class="v">${fmt(r.exact)}</div></div>
  <div class="stat learned"><div class="k">learned composite exact</div><div class="v">${fmt(g.composite_exact.learned)}</div></div>
  <div class="stat learned"><div class="k">learned ID exact · n=${g.inverse_n.learned}</div><div class="v">${fmt(g.inverse_exact.learned)}</div></div>
  <div class="stat learned"><div class="k">learned FD exact · n=${l.n}</div><div class="v">${fmt(l.exact)}</div></div>`
  $('beliefs').innerHTML=`<details class="beliefs"><summary>Learned beliefs / world knowledge</summary><div class="inside">
  <div class="path">artifact: ${esc(g.artifact_dir)}/best_beliefs_gepa_seed${DATA.config.seed}.txt <button class="copy" onclick="copyText(currentGame().beliefs,this)">copy</button></div>
  <pre>${esc(g.beliefs||'(empty)')}</pre></div></details>`
}
function rebuildActions(){
  const actions=[...new Set(currentItems().map(x=>x.action))].sort();$('actionFilter').innerHTML='<option value="all">all actions</option>'+actions.map(a=>`<option>${esc(a)}</option>`).join('')
}
function matches(item){
  const mode=$('modeFilter').value,exact=$('exactFilter').value,change=$('changeFilter').value,action=$('actionFilter').value,q=$('search').value.toLowerCase();
  const modes=mode==='all'?['raw','learned']:[mode];
  if(exact!=='all'&&!modes.some(m=>exact==='correct'?item[m].exact===1:item[m].exact!==1))return false;
  if(change==='changed'&&!modes.some(m=>item[m].changed))return false;if(change==='stale'&&!modes.some(m=>!item[m].changed))return false;
  if(action!=='all'&&item.action!==action)return false;
  if(q&&!modes.some(m=>(item.action+'\n'+item[m].prompt+'\n'+item[m].response+'\n'+item[m].pred).toLowerCase().includes(q)))return false;
  return true
}
function renderRows(){
  const task=currentTask(),items=currentItems();visible=items.filter(matches).map(x=>x.idx);
  if(task==='fd'){
    $('headrow').innerHTML='<th>#</th><th>action</th><th>changed</th><th class="rawtxt">raw FD exact</th><th class="rawtxt">raw FD partial</th><th class="learntxt">learned FD exact</th><th class="learntxt">learned FD partial</th><th>cost</th>';
    $('rows').innerHTML=visible.map(i=>{const x=items[i],cost=x.raw.cost+x.learned.cost;return `<tr class="${selectedIndex===i?'selected':''}" onclick="openItem(${i})">
    <td>${i}</td><td><b>${esc(x.action)}</b></td><td>${x.raw.changed||x.learned.changed?'yes':'no'}</td>
    <td>${score(x.raw.exact)}</td><td>${score(x.raw.partial)}</td><td>${score(x.learned.exact)}</td><td>${score(x.learned.partial)}</td><td>${money(cost)}</td></tr>`}).join('');
  }else{
    $('headrow').innerHTML='<th>#</th><th>true action</th><th>state changed</th><th class="rawtxt">raw ID exact</th><th class="rawtxt">raw prediction</th><th class="learntxt">learned ID exact</th><th class="learntxt">learned prediction</th><th>choices</th>';
    $('rows').innerHTML=visible.map(i=>{const x=items[i];return `<tr class="${selectedIndex===i?'selected':''}" onclick="openItem(${i})">
    <td>${i}</td><td><b>${esc(x.action)}</b></td><td>${x.raw.changed||x.learned.changed?'yes':'no'}</td>
    <td>${score(x.raw.exact)}</td><td>${esc(x.raw.pred||'?')}</td><td>${score(x.learned.exact)}</td><td>${esc(x.learned.pred||'?')}</td><td>${x.learned.choices.length}</td></tr>`}).join('');
  }
  $('count').textContent=visible.length+' / '+items.length+' '+task.toUpperCase()+' transitions'
}
function chooseGame(i){gameIndex=i;selectedIndex=null;$('drawer').style.display='none';rebuildActions();renderAll();location.hash=currentGame().game+':'+currentTask()}
function section(title,key,value,mode,cls=''){
  return `<div class="section ${cls}"><h3>${esc(title)}<button class="copy" onclick="copySection(this,'${key}','${mode}')">copy</button></h3><pre>${esc(value||'(empty)')}</pre></div>`
}
function modeCard(item,mode){
  const r=item[mode],task=currentTask();
  if(task==='id')return `<div class="card ${mode}"><div class="cardhead">${mode==='raw'?'Raw-frame inverse dynamics':'Learned perception + beliefs · inverse dynamics'}</div>
  <div class="section"><div class="metrics"><span class="metric">ID exact ${fmt(r.exact)}</span><span class="metric">${r.exact===1?'✓ correct':'✗ wrong'}</span><span class="metric">state changed ${r.changed?'yes':'no'}</span></div>
  ${item.perception_error&&mode==='learned'?'<div class="bad badge">perception error: '+esc(item.perception_error)+'</div>':''}</div>
  ${section('STATE 1 representation','start',r.start,mode)}
  ${section('STATE 2 representation','target',r.target,mode)}
  ${section('Predicted action','pred',r.pred,mode)}
  <div class="section"><h3>Action choices</h3><pre>${esc(r.choices.map(x=>'- '+x).join('\n'))}</pre></div>
  ${section('Exact model prompt','prompt',r.prompt,mode,'prompt')}
  <div class="section"><div class="truth">${esc(r.response_note||'')}</div></div>
  ${section(mode==='raw'?'Recorded response (reasoning only)':'Original model response','response',r.response,mode)}
  </div>`;
  return `<div class="card ${mode}"><div class="cardhead">${mode==='raw'?'Raw-frame forward dynamics':'Learned perception + beliefs · forward dynamics'}</div>
  <div class="section"><div class="metrics"><span class="metric">FD exact ${fmt(r.exact)}</span><span class="metric">FD partial ${fmt(r.partial)}</span>
  <span class="metric">stale exact ${fmt(r.stale_exact)}</span><span class="metric">changed ${r.changed?'yes':'no'}</span><span class="metric">cost ${money(r.cost)}</span></div>
  ${r.perception_error?'<div class="bad badge">perception error: '+esc(r.perception_error)+'</div>':''}</div>
  ${section('Current representation','start',r.start,mode)}
  ${section('True next representation','target',r.target,mode)}
  ${section('Parsed prediction','pred',r.pred,mode)}
  ${section('Exact model prompt','prompt',r.prompt,mode,'prompt')}
  ${section('Original model response','response',r.response,mode)}
  </div>`
}
function openItem(i){
  selectedIndex=i;const g=currentGame(),task=currentTask(),item=currentItems()[i];$('drawerTitle').textContent=g.game+' · '+task.toUpperCase()+' transition '+i;
  $('drawerTruth').textContent='true action: '+item.action;$('drawerBody').innerHTML='<div id="gridMount" class="section"><h3>Raw grids</h3></div><div class="columns">'+modeCard(item,'raw')+modeCard(item,'learned')+'</div>';
  $('gridMount').appendChild(gridsSection(item));$('drawer').style.display='block';renderRows();location.hash=g.game+':'+task+':'+i
}
function move(delta){if(selectedIndex===null||!visible.length)return;let p=visible.indexOf(selectedIndex);if(p<0)p=0;openItem(visible[Math.max(0,Math.min(visible.length-1,p+delta))])}
function renderAll(){renderGames();renderSummary();renderRows();$('headerMeta').textContent=DATA.games.length+' games · '+DATA.games.reduce((n,g)=>n+g.items.length,0)+' FD + '+DATA.games.reduce((n,g)=>n+g.id_items.length,0)+' ID transitions · ordered by learned composite exact'}
$('taskFilter').addEventListener('change',()=>{selectedIndex=null;$('drawer').style.display='none';rebuildActions();renderRows()});
for(const id of ['modeFilter','exactFilter','changeFilter','actionFilter'])$(id).addEventListener('change',renderRows);
$('search').addEventListener('input',renderRows);$('close').onclick=()=>{$('drawer').style.display='none';selectedIndex=null;renderRows()};$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);
document.addEventListener('keydown',e=>{if(e.key==='Escape')$('close').click();if($('drawer').style.display==='block'&&e.key==='ArrowLeft')move(-1);if($('drawer').style.display==='block'&&e.key==='ArrowRight')move(1)});
const hash=decodeURIComponent(location.hash.slice(1)),[hg,ht,hi]=hash.split(':');const gi=DATA.games.findIndex(g=>g.game===hg);if(gi>=0)gameIndex=gi;if(ht==='id'||ht==='fd')$('taskFilter').value=ht;
$('align').innerHTML='<span class="badge '+(DATA.alignment_ok?'ok':'bad')+'">'+(DATA.alignment_ok?'ID + FD prompt/data alignment verified':'alignment errors: '+DATA.alignment_errors.length)+'</span>';
rebuildActions();renderAll();if(gi>=0&&hi!==undefined&&!Number.isNaN(+hi))openItem(+hi);
</script>
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO / "logs" / "forward_eval_test50_raw_vs_learned.json",
    )
    parser.add_argument(
        "--id-input",
        type=Path,
        default=REPO / "logs" / "id_eval_test50_raw_vs_learned.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "logs" / "forward_eval_test50_raw_vs_learned_viz.html",
    )
    args = parser.parse_args()

    source = json.loads(args.input.read_text())
    id_source = json.loads(args.id_input.read_text())
    payload = build_payload(source, id_source)
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    html = HTML.replace("__TITLE__", payload["title"]).replace("__DATA__", data)
    args.out.write_text(html)
    print(
        f"wrote {args.out} ({args.out.stat().st_size / 1024 / 1024:.1f} MiB); "
        f"games={len(payload['games'])}, "
        f"fd_transitions={sum(len(g['items']) for g in payload['games'])}, "
        f"id_transitions={sum(len(g['id_items']) for g in payload['games'])}, "
        f"alignment_ok={payload['alignment_ok']}"
    )
    if payload["alignment_errors"]:
        print(json.dumps(payload["alignment_errors"][:10], indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
