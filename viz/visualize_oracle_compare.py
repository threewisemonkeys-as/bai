"""Render side-by-side comparison of multiple oracle runs.

Usage:
    # Auto-discover the latest run per task under logs/oracle_runs/:
    uv run viz/visualize_oracle_compare.py
    uv run viz/visualize_oracle_compare.py --out compare.html --open

    # Or pass explicit run dirs:
    uv run viz/visualize_oracle_compare.py logs/oracle_runs/ice/llm_XXX logs/oracle_runs/83WKQ/llm_YYY
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "logs" / "oracle_runs"

# A small categorical palette
COLORS = ["#58a6ff", "#3fb950", "#d29922", "#bc8cff", "#f85149", "#79c0ff", "#56d364"]


def _discover_latest_runs() -> list[Path]:
    runs = []
    if not RUNS_ROOT.exists():
        sys.exit(f"No oracle runs dir at {RUNS_ROOT}")
    for task_dir in sorted(RUNS_ROOT.iterdir()):
        if not task_dir.is_dir():
            continue
        children = sorted(task_dir.glob("*/config.json"))
        if children:
            runs.append(children[-1].parent)
    return runs


def _load_run(run_dir: Path) -> dict | None:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text())
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    scores: list[int | None] = []
    costs: list[float] = []
    answers: list[str] = []
    final_beliefs = ""
    final_bank: list[dict] = []
    for step_dir in sorted(run_dir.glob("step_*")):
        sj = step_dir / "step.json"
        if not sj.exists():
            continue
        s = json.loads(sj.read_text())
        scores.append(s.get("belief_score"))
        costs.append(s.get("step_cost", 0.0))
        ans = (s.get("answer") or {}).get("answer") or ""
        answers.append(ans)
        if s.get("beliefs_after"):
            final_beliefs = s["beliefs_after"]
        bank_path = step_dir / "qa.jsonl"
        if bank_path.exists():
            tmp: list[dict] = []
            for line in bank_path.read_text().splitlines():
                line = line.strip()
                if line:
                    tmp.append(json.loads(line))
            if tmp:
                final_bank = tmp
    return {
        "run_dir": run_dir,
        "task": cfg["task"],
        "selector": cfg["selector"],
        "cfg": cfg,
        "summary": summary,
        "scores": scores,
        "costs": costs,
        "answers": answers,
        "final_beliefs": final_beliefs,
        "final_bank": final_bank,
        "total_cost": (summary or {}).get("total_cost", sum(costs)),
        "running": summary is None,
    }


_CSS = """
:root {
  --bg: #0d1117; --surface: #161b22; --surface2: #21262d; --border: #30363d;
  --text: #e6edf3; --text-muted: #8b949e; --accent: #58a6ff; --accent2: #3fb950;
  --accent3: #d29922; --danger: #f85149; --purple: #bc8cff;
  --font-mono: 'SF Mono', 'Fira Code', monospace;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); line-height: 1.5; padding: 20px 28px; }
h1 { font-size: 20px; margin-bottom: 4px; }
.meta { color: var(--text-muted); font-size: 13px; margin-bottom: 14px; }
.section { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 16px; }
.section-header { padding: 10px 16px; font-size: 13px; font-weight: 600; border-bottom: 1px solid var(--border); }
.section-body { padding: 14px 16px; }
.legend { display: flex; gap: 16px; flex-wrap: wrap; padding: 0 16px 12px; }
.legend-item { font-size: 12px; color: var(--text-muted); display: flex; align-items: center; gap: 6px; }
.legend-item .swatch { width: 12px; height: 12px; border-radius: 2px; }
.grid { display: grid; grid-template-columns: repeat({n_cols}, minmax(260px, 1fr)); gap: 14px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 14px; display: flex; flex-direction: column; gap: 10px; min-width: 0; }
.card h3 { font-size: 14px; display: flex; align-items: center; gap: 8px; }
.card h3 .swatch { width: 12px; height: 12px; border-radius: 2px; }
.card .sub { font-size: 11px; color: var(--text-muted); }
.kpi-row { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.kpi { background: var(--surface2); border-radius: 4px; padding: 6px 8px; font-size: 11px; color: var(--text-muted); }
.kpi b { display: block; font-size: 14px; color: var(--text); font-weight: 700; }
.spark { background: var(--bg); border: 1px solid var(--border); border-radius: 4px; padding: 4px; }
.steps-strip { display: flex; flex-wrap: wrap; gap: 2px; }
.steps-strip .blip { width: 14px; height: 14px; font-size: 9px; line-height: 14px; text-align: center; font-family: var(--font-mono); border-radius: 2px; color: #0d1117; font-weight: 700; }
.steps-strip .blip.yes { background: var(--accent2); }
.steps-strip .blip.no { background: var(--danger); }
.steps-strip .blip.unknown { background: var(--accent3); }
pre { background: var(--bg); border: 1px solid var(--border); border-radius: 4px; padding: 8px 10px; font-family: var(--font-mono); font-size: 11px; line-height: 1.45; overflow: auto; max-height: 280px; white-space: pre-wrap; word-wrap: break-word; }
.bank-mini { font-size: 11px; max-height: 220px; overflow-y: auto; border: 1px solid var(--border); border-radius: 4px; padding: 4px 8px; background: var(--bg); }
.bank-mini .row { display: flex; gap: 6px; padding: 2px 0; border-bottom: 1px solid #1a1f26; }
.bank-mini .row:last-child { border-bottom: none; }
.bank-mini .qid { color: var(--text-muted); font-family: var(--font-mono); min-width: 32px; }
.bank-mini .ans { font-family: var(--font-mono); font-weight: 700; min-width: 22px; }
.bank-mini .ans.yes { color: var(--accent2); }
.bank-mini .ans.no { color: var(--danger); }
.bank-mini .ans.un { color: var(--text-muted); }
.running-badge { background: rgba(210,153,34,0.15); color: var(--accent3); padding: 1px 6px; border-radius: 3px; font-size: 10px; font-weight: 600; margin-left: 4px; }
a.runlink { color: var(--accent); text-decoration: none; font-size: 11px; }
a.runlink:hover { text-decoration: underline; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--surface2); border-radius: 3px; }
"""


def _overlay_curve(runs: list[dict], width: int = 980, height: int = 320) -> str:
    pad_l, pad_r, pad_t, pad_b = 40, 14, 18, 30
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b
    max_n = max((len(r["scores"]) for r in runs), default=1)
    max_n = max(max_n, 1)

    def x_of(i: int) -> float:
        return pad_l + (inner_w * i / max(1, max_n - 1))

    def y_of(s: int | None) -> float:
        s = 0 if s is None else s
        return pad_t + inner_h - (inner_h * s / 10)

    grid = []
    for tick in (0, 2, 4, 6, 8, 10):
        y = y_of(tick)
        grid.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + inner_w}" y2="{y:.1f}" stroke="#21262d" />'
            f'<text x="{pad_l - 6}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#8b949e">{tick}</text>'
        )
    x_ticks = []
    for i in range(0, max_n, max(1, max_n // 10)):
        x_ticks.append(
            f'<text x="{x_of(i):.1f}" y="{height - 10}" text-anchor="middle" font-size="11" fill="#8b949e">{i + 1}</text>'
        )

    paths = []
    for ri, run in enumerate(runs):
        c = COLORS[ri % len(COLORS)]
        pts = [(x_of(i), y_of(s)) for i, s in enumerate(run["scores"])]
        if not pts:
            continue
        path = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        paths.append(f'<path d="{path}" fill="none" stroke="{c}" stroke-width="2" />')
        for x, y in pts:
            paths.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{c}" />')

    return f'''<svg viewBox="0 0 {width} {height}" width="100%" preserveAspectRatio="xMidYMid meet">
  {"".join(grid)}{"".join(x_ticks)}
  <text x="{pad_l + inner_w / 2}" y="{height - 1}" text-anchor="middle" font-size="11" fill="#8b949e">step</text>
  {"".join(paths)}
</svg>'''


def _spark(scores: list[int | None], color: str, width: int = 240, height: int = 60) -> str:
    n = len(scores) or 1
    pts = [
        (i * width / max(1, n - 1), height - (height * (0 if s is None else s) / 10))
        for i, s in enumerate(scores)
    ]
    if not pts:
        return ""
    path = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="{color}" />' for x, y in pts
    )
    return f'<svg viewBox="0 0 {width} {height}" width="100%"><path d="{path}" fill="none" stroke="{color}" stroke-width="1.5" />{dots}</svg>'


def _bank_mini(bank: list[dict], limit: int = 80) -> str:
    if not bank:
        return "<em style='color:var(--text-muted)'>(empty)</em>"
    rows = []
    answered = [q for q in bank if q.get("answer") is not None]
    unanswered = [q for q in bank if q.get("answer") is None]
    # Show answered first, then a few unanswered
    for qa in (answered + unanswered)[:limit]:
        ans = qa.get("answer")
        if ans is True:
            cls, lbl = "yes", "Y"
        elif ans is False:
            cls, lbl = "no", "N"
        else:
            cls, lbl = "un", "—"
        rows.append(
            f'<div class="row"><span class="qid">Q{bank.index(qa) + 1}</span>'
            f'<span class="ans {cls}">{lbl}</span><span>{html.escape(qa.get("question", ""))}</span></div>'
        )
    note = (
        f"<div class='row' style='color:var(--text-muted);font-style:italic'>… {len(bank) - limit} more</div>"
        if len(bank) > limit else ""
    )
    return "".join(rows) + note


def _card(run: dict, color: str) -> str:
    scores = run["scores"]
    final = next((s for s in reversed(scores) if s is not None), None)
    peak = max((s for s in scores if s is not None), default=None)
    n_ans = sum(1 for q in run["final_bank"] if q.get("answer") is not None)
    n_steps = len(scores)
    target = run["cfg"].get("num_steps", "?")
    answers_strip = "".join(
        f'<div class="blip {ans or "unknown"}">{(ans or "?")[0].upper()}</div>'
        for ans in run["answers"]
    )
    running_tag = '<span class="running-badge">RUNNING</span>' if run["running"] else ""
    return f"""<div class="card">
  <h3><span class="swatch" style="background:{color}"></span>{html.escape(run["task"])}{running_tag}</h3>
  <div class="sub">{html.escape(run["selector"])} · steps {n_steps}/{target}
    · <a class="runlink" href="{run["run_dir"].name}/report.html">full report ↗</a>
  </div>
  <div class="kpi-row">
    <div class="kpi">final score<b>{final}/10</b></div>
    <div class="kpi">peak<b>{peak}/10</b></div>
    <div class="kpi">total cost<b>${run["total_cost"]:.2f}</b></div>
    <div class="kpi">bank · answered<b>{len(run["final_bank"])} · {n_ans}</b></div>
  </div>
  <div class="spark">{_spark(scores, color)}</div>
  <div>
    <div class="sub" style="margin-bottom:4px">answer strip (Y=yes, N=no, —=unknown)</div>
    <div class="steps-strip">{answers_strip}</div>
  </div>
  <div>
    <div class="sub" style="margin-bottom:4px">final beliefs</div>
    <pre>{html.escape(run["final_beliefs"]) or "<em>(empty)</em>"}</pre>
  </div>
  <div>
    <div class="sub" style="margin-bottom:4px">final question bank ({len(run["final_bank"])})</div>
    <div class="bank-mini">{_bank_mini(run["final_bank"])}</div>
  </div>
</div>"""


def render(runs: list[dict]) -> str:
    n = len(runs)
    legend = "".join(
        f'<div class="legend-item"><span class="swatch" style="background:{COLORS[i % len(COLORS)]}"></span>'
        f'{html.escape(r["task"])} · final {next((s for s in reversed(r["scores"]) if s is not None), "—")}/10</div>'
        for i, r in enumerate(runs)
    )
    overlay = _overlay_curve(runs)
    cards = "".join(_card(r, COLORS[i % len(COLORS)]) for i, r in enumerate(runs))
    css = _CSS.replace("{n_cols}", str(min(max(n, 1), 5)))
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Oracle comparison ({n} runs)</title>
<style>{css}</style></head>
<body>
<h1>Oracle runs comparison · {n} task{'s' if n != 1 else ''}</h1>
<div class="meta">Each card links to the per-run report with full prompt/response traces.</div>
<section class="section">
  <div class="section-header">Belief score (0-10) by step</div>
  <div class="section-body">{overlay}</div>
  <div class="legend">{legend}</div>
</section>
<div class="grid">{cards}</div>
</body></html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dirs", nargs="*", help="Run directories to compare. Empty = auto-discover latest per task.")
    p.add_argument("--out", default=None, help="Output HTML path. Default: logs/oracle_runs/compare.html")
    p.add_argument("--open", action="store_true")
    args = p.parse_args()

    if args.run_dirs:
        dirs = [Path(d).resolve() for d in args.run_dirs]
    else:
        dirs = _discover_latest_runs()
        if not dirs:
            sys.exit("No runs found under logs/oracle_runs/")

    runs = []
    for d in dirs:
        r = _load_run(d)
        if r is not None:
            runs.append(r)
        else:
            print(f"skip {d} (no config.json)", file=sys.stderr)

    out_path = Path(args.out).resolve() if args.out else RUNS_ROOT / "compare.html"
    out_path.write_text(render(runs))
    total = sum(r["total_cost"] for r in runs)
    print(f"wrote {out_path}  ({len(runs)} runs, total cost ${total:.2f})")
    for r in runs:
        final = next((s for s in reversed(r["scores"]) if s is not None), None)
        flag = " (running)" if r["running"] else ""
        print(f"  {r['task']:8s}  final={final}/10  steps={len(r['scores'])}  ${r['total_cost']:.2f}{flag}")
    if args.open:
        webbrowser.open(out_path.as_uri())


if __name__ == "__main__":
    main()
