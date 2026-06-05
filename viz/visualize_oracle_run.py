"""Render a single oracle-run directory as a self-contained HTML report.

Usage:
    uv run viz/visualize_oracle_run.py logs/oracle_runs/ice/llm_<ts>/
    uv run viz/visualize_oracle_run.py logs/oracle_runs/ice/llm_<ts>/ --out report.html
    uv run viz/visualize_oracle_run.py logs/oracle_runs/ice/llm_<ts>/ --open

If --out is omitted, writes to <run_dir>/report.html.

The report includes the score curve, the accumulated question bank, and per
step the full prompt+response for every LLM interaction
(question generation, selection, oracle answer, belief update, scoring).
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import webbrowser
from pathlib import Path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_run(run_dir: Path) -> dict:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        sys.exit(f"Not an oracle run directory (no config.json): {run_dir}")
    cfg = json.loads(cfg_path.read_text())
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    steps = []
    for step_dir in sorted(run_dir.glob("step_*")):
        step_json = step_dir / "step.json"
        if not step_json.exists():
            continue
        step = json.loads(step_json.read_text())
        step["_dir"] = step_dir.name
        calls_path = step_dir / "llm_calls.json"
        step["llm_calls"] = json.loads(calls_path.read_text()) if calls_path.exists() else []
        qa_path = step_dir / "qa.jsonl"
        bank: list[dict] = []
        if qa_path.exists():
            for line in qa_path.read_text().splitlines():
                line = line.strip()
                if line:
                    bank.append(json.loads(line))
        step["bank"] = bank
        steps.append(step)
    return {"cfg": cfg, "summary": summary, "steps": steps, "run_dir": str(run_dir)}


# ---------------------------------------------------------------------------
# CSS — dark theme (cribbed from viz/stepwise_eb_learn/index.html)
# ---------------------------------------------------------------------------


_CSS = """
:root {
  --bg: #0d1117; --surface: #161b22; --surface2: #21262d; --border: #30363d;
  --text: #e6edf3; --text-muted: #8b949e; --accent: #58a6ff; --accent2: #3fb950;
  --accent3: #d29922; --danger: #f85149; --purple: #bc8cff;
  --font-mono: 'SF Mono', 'Fira Code', monospace;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); line-height: 1.5; }
.app { display: flex; flex-direction: column; min-height: 100vh; }
.topbar { background: var(--surface); border-bottom: 1px solid var(--border); padding: 12px 24px; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
.topbar-title { font-size: 16px; font-weight: 700; }
.topbar-info { font-size: 13px; color: var(--text-muted); }
.topbar-info code { background: var(--surface2); padding: 1px 6px; border-radius: 4px; font-size: 12px; }
.kpis { display: flex; gap: 8px; margin-left: auto; flex-wrap: wrap; }
.kpi { background: var(--surface2); padding: 6px 12px; border-radius: 6px; font-size: 12px; color: var(--text-muted); }
.kpi b { display: block; color: var(--text); font-size: 15px; font-weight: 700; }
.content-area { display: flex; flex: 1; min-height: 0; }
.sidebar { width: 260px; min-width: 260px; background: var(--surface); border-right: 1px solid var(--border); overflow-y: auto; position: sticky; top: 0; max-height: 100vh; }
.sidebar h2 { padding: 12px 16px; font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid var(--border); font-weight: 600; }
.step-item { padding: 8px 14px; border-left: 3px solid transparent; cursor: pointer; font-size: 13px; display: flex; align-items: center; gap: 8px; border-bottom: 1px solid #1a1f26; }
.step-item:hover { background: var(--surface2); }
.step-item.active { background: var(--surface2); border-left-color: var(--accent); }
.step-item .num { font-family: var(--font-mono); color: var(--text-muted); min-width: 28px; }
.step-item .score { margin-left: auto; font-family: var(--font-mono); font-size: 11px; font-weight: 700; color: var(--accent); }
.step-item .ans { font-family: var(--font-mono); font-size: 10px; padding: 1px 5px; border-radius: 3px; }
.step-item .ans.yes { background: rgba(63,185,80,0.15); color: var(--accent2); }
.step-item .ans.no { background: rgba(248,81,73,0.15); color: var(--danger); }
.step-item .ans.unknown { background: rgba(139,148,158,0.15); color: var(--text-muted); }
.main { flex: 1; padding: 20px 28px; min-width: 0; }
.section { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 16px; }
.section-header { padding: 10px 16px; font-size: 13px; font-weight: 600; border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; }
.section-body { padding: 14px 16px; }
h2.step-title { font-size: 18px; margin-bottom: 4px; display: flex; align-items: center; gap: 10px; }
h2.step-title .pill { font-size: 11px; padding: 2px 8px; border-radius: 999px; background: var(--surface2); color: var(--text-muted); font-weight: 500; }
h2.step-title .pill.score { background: rgba(88,166,255,0.15); color: var(--accent); font-weight: 700; }
h2.step-title .pill.cost { background: rgba(210,153,34,0.15); color: var(--accent3); }
h3 { font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
p.q { font-size: 14px; padding: 10px 12px; background: var(--bg); border-radius: 4px; border-left: 3px solid var(--accent); margin-bottom: 12px; font-weight: 500; }
p.rationale { font-size: 13px; color: var(--text-muted); line-height: 1.5; margin: 6px 0; }
.score-curve { background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 10px; }
pre { background: var(--bg); border: 1px solid var(--border); border-radius: 4px; padding: 10px 12px; font-family: var(--font-mono); font-size: 12px; line-height: 1.5; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; max-height: 320px; overflow-y: auto; }
pre.tall { max-height: 540px; }
.llm-call { border: 1px solid var(--border); border-radius: 6px; margin-bottom: 10px; }
.llm-call > summary { padding: 10px 14px; cursor: pointer; display: flex; align-items: center; gap: 10px; list-style: none; }
.llm-call > summary::-webkit-details-marker { display: none; }
.llm-call > summary::before { content: "▶"; font-size: 9px; color: var(--text-muted); transition: transform .15s; }
.llm-call[open] > summary::before { transform: rotate(90deg); }
.llm-call .body { padding: 0 14px 14px; }
.llm-call .kind { font-weight: 600; font-size: 13px; }
.llm-call .meta { margin-left: auto; font-family: var(--font-mono); font-size: 11px; color: var(--text-muted); display: flex; gap: 10px; }
.llm-call .meta .model { color: var(--purple); }
.llm-call .meta .cost { color: var(--accent3); }
.io-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
@media (max-width: 1100px) { .io-grid { grid-template-columns: 1fr; } }
.io-grid h4 { font-size: 11px; color: var(--text-muted); text-transform: uppercase; margin-bottom: 4px; }
.bank-row { display: grid; grid-template-columns: 40px 56px 1fr 80px; gap: 10px; padding: 6px 12px; border-bottom: 1px solid var(--border); font-size: 12px; align-items: start; }
.bank-row:last-child { border-bottom: none; }
.bank-row .qid { font-family: var(--font-mono); color: var(--text-muted); }
.bank-row .ans { font-family: var(--font-mono); font-size: 11px; font-weight: 700; }
.bank-row .ans.yes { color: var(--accent2); }
.bank-row .ans.no { color: var(--danger); }
.bank-row .ans.unanswered { color: var(--text-muted); }
.bank-row .src { color: var(--text-muted); font-family: var(--font-mono); font-size: 11px; }
.bank-row.picked { background: rgba(88,166,255,0.07); }
.evidence { font-size: 11px; color: var(--text-muted); margin-top: 2px; line-height: 1.4; }
.unknown-tag { background: rgba(210,153,34,0.15); color: var(--accent3); padding: 1px 6px; border-radius: 3px; font-size: 10px; font-weight: 600; margin-left: 4px; }
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--surface2); border-radius: 4px; }
.toc { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 16px; }
.toc a { background: var(--surface2); border: 1px solid var(--border); padding: 4px 10px; border-radius: 4px; font-size: 11px; color: var(--text-muted); text-decoration: none; }
.toc a:hover { color: var(--accent); border-color: var(--accent); }
"""


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _esc(s: str | None) -> str:
    return html.escape(s or "")


def _score_curve_svg(scores: list[int | None], width: int = 880, height: int = 200) -> str:
    pad_l, pad_r, pad_t, pad_b = 38, 14, 16, 28
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b
    n = len(scores)
    if n == 0:
        return "<p style='color:var(--text-muted)'>no steps yet</p>"
    xs = [pad_l + (inner_w * i / max(1, n - 1)) for i in range(n)]

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
    pts = [(xs[i], y_of(scores[i])) for i in range(n)]
    path = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#58a6ff" />'
        f'<text x="{x:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-size="11" fill="#e6edf3">{0 if scores[i] is None else scores[i]}</text>'
        for i, (x, y) in enumerate(pts)
    )
    xlabels = "".join(
        f'<text x="{xs[i]:.1f}" y="{height - 8}" text-anchor="middle" font-size="11" fill="#8b949e">{i + 1}</text>'
        for i in range(n)
    )
    return f"""<svg viewBox="0 0 {width} {height}" width="100%" preserveAspectRatio="xMidYMid meet">
  {"".join(grid)}
  <path d="{path}" fill="none" stroke="#58a6ff" stroke-width="2" />
  {dots}{xlabels}
</svg>"""


def _bank_panel(bank: list[dict], picked_index: int | None) -> str:
    rows = []
    for i, qa in enumerate(bank):
        ans_val = qa.get("answer")
        if ans_val is True:
            ans_html = '<span class="ans yes">YES</span>'
        elif ans_val is False:
            ans_html = '<span class="ans no">NO</span>'
        else:
            ans_html = '<span class="ans unanswered">—</span>'
        evidence = qa.get("evidence") or ""
        ev_html = f'<div class="evidence">{_esc(evidence)}</div>' if evidence else ""
        picked_cls = " picked" if i == picked_index else ""
        rows.append(
            f'<div class="bank-row{picked_cls}">'
            f'<div class="qid">Q{i + 1}</div>'
            f'<div>{ans_html}</div>'
            f'<div>{_esc(qa.get("question"))}{ev_html}</div>'
            f'<div class="src">step {qa.get("source_step", "?")}</div>'
            f'</div>'
        )
    n_ans = sum(1 for q in bank if q.get("answer") is not None)
    return f"""<section class="section" id="bank">
  <div class="section-header">Accumulated question bank
    <span style="color:var(--text-muted);font-weight:400">{len(bank)} total · {n_ans} answered · {len(bank) - n_ans} unanswered</span>
  </div>
  <div>{"".join(rows) or "<p style='padding:14px;color:var(--text-muted)'>(empty)</p>"}</div>
</section>"""


def _llm_call_panel(call: dict) -> str:
    kind = call.get("kind", "?")
    model = call.get("model", "?")
    cost = call.get("cost", 0.0)
    prompt = call.get("prompt", "") or ""
    response = call.get("response", "") or ""
    cached_tag = (
        '<span class="unknown-tag">CACHED</span>' if call.get("cached") else ""
    )
    return f"""<details class="llm-call">
  <summary>
    <span class="kind">{_esc(kind)}</span>{cached_tag}
    <div class="meta">
      <span class="model">{_esc(model)}</span>
      <span class="cost">${cost:.4f}</span>
    </div>
  </summary>
  <div class="body">
    <div class="io-grid">
      <div><h4>Prompt</h4><pre class="tall">{_esc(prompt) or "<em>(empty)</em>"}</pre></div>
      <div><h4>Response</h4><pre class="tall">{_esc(response) or "<em>(empty)</em>"}</pre></div>
    </div>
  </div>
</details>"""


def _step_section(step: dict) -> str:
    idx = step["step_index"]
    ans = step.get("answer") or {}
    label = ans.get("answer", "")
    sel_src = step.get("selected_source_index")
    sel_lbl = f"Q{sel_src + 1}" if sel_src is not None else "—"
    new_qs = step.get("new_questions") or []
    new_qs_html = "".join(f"<li>{_esc(q)}</li>" for q in new_qs)
    beliefs = step.get("beliefs_after") or ""
    score = step.get("belief_score")
    score_rationale = step.get("belief_score_rationale") or ""
    cost = step.get("step_cost", 0.0)
    score_pill = f'<span class="pill score">score {score}/10</span>' if score is not None else ""
    bank = step.get("bank") or []
    llm_calls = step.get("llm_calls") or []
    bank_meta = (
        f"bank: {step.get('bank_size_before', '?')} → {step.get('bank_size_after', len(bank))} · "
        f"unanswered pool: {step.get('unanswered_pool_size', '?')}"
    )

    badge_class = label if label in ("yes", "no", "unknown") else "unknown"
    answer_pill = f'<span class="pill" style="background:rgba(63,185,80,0.15);color:var(--accent2)">{label.upper()}</span>' if label == "yes" else (
        f'<span class="pill" style="background:rgba(248,81,73,0.15);color:var(--danger)">{label.upper()}</span>' if label == "no" else
        f'<span class="pill">{_esc(label).upper() or "—"}</span>'
    )

    calls_html = "".join(_llm_call_panel(c) for c in llm_calls)

    return f"""
<a id="step-{idx}"></a>
<h2 class="step-title">Step {idx + 1}
  <span class="pill">picked {sel_lbl}</span>
  {answer_pill}
  {score_pill}
  <span class="pill cost">${cost:.4f}</span>
</h2>
<p style="color:var(--text-muted);font-size:12px;margin-bottom:14px">{bank_meta}</p>

<section class="section">
  <div class="section-header">Picked question · oracle answer</div>
  <div class="section-body">
    <p class="q">{_esc(step.get("selected_question") or "(none)")}</p>
    <h3>Oracle rationale</h3>
    <p class="rationale">{_esc(ans.get("rationale") or "(none)")}</p>
    <h3>Scorer rationale</h3>
    <p class="rationale">{_esc(score_rationale) or "<em>(not scored this step)</em>"}</p>
  </div>
</section>

<section class="section">
  <div class="section-header">New candidate questions generated this step ({len(new_qs)})</div>
  <div class="section-body"><ol>{new_qs_html or "<em>(none)</em>"}</ol></div>
</section>

<section class="section">
  <div class="section-header">Bank snapshot after this step</div>
  <div>{"".join(_bank_row(bank, picked=sel_src))}</div>
</section>

<section class="section">
  <div class="section-header">LLM interactions ({len(llm_calls)})</div>
  <div class="section-body">{calls_html or "<p style='color:var(--text-muted)'>(none)</p>"}</div>
</section>

<section class="section">
  <div class="section-header">Beliefs after step {idx + 1}</div>
  <div class="section-body"><pre>{_esc(beliefs) or "(empty)"}</pre></div>
</section>
"""


def _bank_row(bank: list[dict], picked: int | None) -> list[str]:
    rows = []
    for i, qa in enumerate(bank):
        ans_val = qa.get("answer")
        if ans_val is True:
            ans_html = '<span class="ans yes">YES</span>'
        elif ans_val is False:
            ans_html = '<span class="ans no">NO</span>'
        else:
            ans_html = '<span class="ans unanswered">—</span>'
        evidence = qa.get("evidence") or ""
        ev_html = f'<div class="evidence">{_esc(evidence)}</div>' if evidence else ""
        picked_cls = " picked" if i == picked else ""
        rows.append(
            f'<div class="bank-row{picked_cls}">'
            f'<div class="qid">Q{i + 1}</div>'
            f'<div>{ans_html}</div>'
            f'<div>{_esc(qa.get("question"))}{ev_html}</div>'
            f'<div class="src">step {qa.get("source_step", "?")}</div>'
            f'</div>'
        )
    return rows


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def render(report: dict) -> str:
    cfg = report["cfg"]
    summary = report["summary"] or {}
    steps = report["steps"]
    scores = [s.get("belief_score") for s in steps]
    final_score = next((s for s in reversed(scores) if s is not None), None)
    total_cost = summary.get("total_cost") or sum(s.get("step_cost", 0.0) for s in steps)
    n_answered = sum(
        1
        for s in steps
        if (s.get("answer") or {}).get("answer") in ("yes", "no")
    )

    sidebar = []
    for s in steps:
        idx = s["step_index"]
        sc = s.get("belief_score")
        sc_html = f'<span class="score">{sc}</span>' if sc is not None else ""
        ans_label = (s.get("answer") or {}).get("answer") or ""
        ans_html = (
            f'<span class="ans {ans_label}">{ans_label.upper()}</span>'
            if ans_label
            else ""
        )
        sidebar.append(
            f'<a class="step-item" href="#step-{idx}">'
            f'<span class="num">#{idx + 1}</span>{ans_html}{sc_html}</a>'
        )

    kpis = "".join(
        f'<div class="kpi">{label}<b>{value}</b></div>'
        for label, value in [
            ("task", _esc(cfg["task"])),
            ("selector", _esc(cfg["selector"])),
            ("steps", f"{len(steps)}/{cfg.get('num_steps', '?')}"),
            ("final score", f"{final_score}/10" if final_score is not None else "—"),
            ("answered", str(n_answered)),
            ("total cost", f"${total_cost:.4f}"),
        ]
    )
    meta = (
        f"<code>learner</code> {_esc(cfg.get('learner_model', '?'))} · "
        f"<code>oracle</code> {_esc(cfg.get('oracle_model', '?'))} · "
        f"<code>scorer</code> {_esc(cfg.get('scorer_model', '?'))} · "
        f"candidates/step {cfg.get('candidates_per_step', '?')} · seed {cfg.get('seed', '?')}"
    )

    # final bank from the last step (most complete)
    final_bank = steps[-1]["bank"] if steps else []
    bank_overview = _bank_panel(final_bank, picked_index=None)

    panels = "\n".join(_step_section(s) for s in steps)
    curve_svg = _score_curve_svg(scores)
    title = f"Oracle run · {_esc(cfg['task'])} · {_esc(cfg['selector'])}"
    toc = "".join(
        f'<a href="#step-{s["step_index"]}">step {s["step_index"] + 1}</a>'
        for s in steps
    )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>{_CSS}</style>
</head><body>
<div class="app">
  <div class="topbar">
    <div class="topbar-title">{title}</div>
    <div class="topbar-info">{meta}</div>
    <div class="kpis">{kpis}</div>
  </div>
  <div class="content-area">
    <nav class="sidebar">
      <h2>Steps</h2>
      <a class="step-item" href="#overview"><span class="num">▴</span> overview</a>
      <a class="step-item" href="#bank"><span class="num">★</span> bank</a>
      {"".join(sidebar)}
    </nav>
    <div class="main">
      <a id="overview"></a>
      <h2 class="step-title">Overview</h2>
      <section class="section">
        <div class="section-header">Belief-score curve (0-10)</div>
        <div class="section-body score-curve">{curve_svg}</div>
      </section>
      <div class="toc">{toc}</div>
      {bank_overview}
      {panels or "<p style='color:var(--text-muted)'>No steps yet.</p>"}
    </div>
  </div>
</div>
</body></html>
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", help="Path to an oracle run directory.")
    p.add_argument("--out", default=None, help="Output HTML path (default: <run_dir>/report.html).")
    p.add_argument("--open", action="store_true", help="Open the rendered HTML in the default browser.")
    args = p.parse_args()
    run_dir = Path(args.run_dir).resolve()
    report = _load_run(run_dir)
    html_doc = render(report)
    out_path = Path(args.out).resolve() if args.out else run_dir / "report.html"
    out_path.write_text(html_doc)
    print(f"wrote {out_path}  ({len(report['steps'])} steps, "
          f"{sum(len(s.get('llm_calls') or []) for s in report['steps'])} llm calls)")
    if args.open:
        webbrowser.open(out_path.as_uri())


if __name__ == "__main__":
    main()
