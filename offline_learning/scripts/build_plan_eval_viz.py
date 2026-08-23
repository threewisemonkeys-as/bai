#!/usr/bin/env python3
"""Self-contained HTML viz of the goal-conditioned PLANNING eval task
(eval_multistep_fd_plan.py output) — the eval itself, not the learning run.

For every window: the START grid and GOAL grid side by side, the horizon, the
logged (gold) action sequence, and per arm (learned / raw) the emitted plan,
whether the engine reached the goal (and at which step), plus the model's
reasoning in a collapsible block. Windows are grouped by horizon with a
success summary table at the top.

    uv run python offline_learning/scripts/build_plan_eval_viz.py \
        --shards logs/multistep_shards_aug5_s2kt7_seed5data.json --game s2kt7
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

CSS_COLOR = {
    "black": "#0c0e12", "white": "#e8ecf1", "gray": "#8a93a0", "grey": "#8a93a0",
    "red": "#ff4d4d", "gold": "#ffd45a", "green": "#54d98c",
    "mediumpurple": "#9b7fe0", "purple": "#7a5cc0", "blue": "#4d9dff",
    "lightblue": "#8fd0ff", "darkorange": "#ff9a3d", "yellow": "#ffe14d",
    "orange": "#ff9a3d", "pink": "#ff9ad5", "brown": "#a5714a",
}

PAGE_CSS = """
body{background:#0c0e12;color:#cdd3da;font:13px ui-monospace,Menlo,monospace;margin:16px}
h1{margin:0 0 4px;font-size:18px} h2{margin:22px 0 8px;font-size:15px;color:#ffd45a}
.sub{color:#8a93a0;margin-bottom:12px;max-width:1000px;line-height:1.5}
table.sum{border-collapse:collapse;margin:8px 0 4px}
table.sum td,table.sum th{border:1px solid #232a33;padding:3px 10px;text-align:right}
table.sum th{color:#8a93a0;font-weight:400}
.win{background:#151921;border:1px solid #232a33;border-radius:8px;padding:10px;margin:10px 0;
     display:flex;gap:16px;flex-wrap:wrap}
.gcol{width:180px}
.grid{display:grid;width:168px;height:168px;border:1px solid #2a3340;image-rendering:pixelated}
.grid > i{display:block}
.glab{color:#7fd1ff;font-size:11px;margin:3px 0 4px}
.meta{flex:1;min-width:340px;line-height:1.7}
.gold{color:#ffd45a}.ok{color:#54d98c;font-weight:700}.bad{color:#ff4d4d;font-weight:700}
.arm{margin:6px 0;padding:6px 8px;background:#10141a;border-radius:6px;border:1px solid #1d242e}
.armname{color:#7fd1ff;font-weight:700}
details{margin-top:4px} summary{cursor:pointer;color:#8a93a0;font-size:11px}
pre{white-space:pre-wrap;font-size:11px;color:#9aa3ad;max-height:260px;overflow-y:auto;
    background:#0c0e12;padding:6px;border-radius:4px}
.err{color:#ff9a3d;font-size:11px}
"""


def render_grid(grid_json: str) -> str:
    grid = json.loads(grid_json)
    rows, cols = len(grid), len(grid[0])
    cells = "".join(
        f'<i style="background:{CSS_COLOR.get(c, "#d949c4")}"></i>'
        for row in grid for c in row)
    return (f'<div class="grid" style="grid-template-columns:repeat({cols},1fr);'
            f'grid-template-rows:repeat({rows},1fr)">{cells}</div>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--game", required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    data = json.loads(args.shards.read_text())
    (res,) = [r for r in data["results"] if r["game"] == args.game]
    windows, plan_rows = res["windows"], res["plan_rows"]
    by_window: dict[int, dict[str, dict]] = {}
    for pr in plan_rows:
        by_window.setdefault(pr["window"], {})[pr["mode"]] = pr

    hs = sorted({w["h"] for w in windows})
    parts = ['<!doctype html><meta charset="utf-8">', f"<style>{PAGE_CSS}</style>",
             f"<h1>{args.game} — goal-conditioned planning eval</h1>",
             f'<div class="sub">Task: given a K-step history ending at the START '
             f'grid and the GOAL grid, emit &le;h fully-parameterized actions; the '
             f'plan is executed in the Autumn engine from the replayed drive state '
             f'— success iff the grid after the final action equals GOAL. '
             f'Source: {html.escape(str(args.shards))} | artifact: '
             f'{html.escape(res.get("artifact_dir", "?"))} | task model: '
             f'{html.escape(str(data["config"].get("task_model")))}</div>']

    s = res["summary"]
    parts.append('<table class="sum"><tr><th>arm</th>'
                 + "".join(f"<th>h={h}</th>" for h in hs) + "</tr>")
    for mode in ("learned", "raw"):
        cells = "".join(
            f"<td>{s[str(h)]['plan'][mode]['success']:.2f}</td>"
            if s[str(h)]["plan"][mode]["success"] is not None else "<td>—</td>"
            for h in hs)
        parts.append(f"<tr><td>plan {mode}</td>{cells}</tr>")
    rc = [f"<td>{s[str(h)]['plan']['raw']['random_success']:.2f}</td>"
          if s[str(h)]["plan"]["raw"]["random_success"] is not None else "<td>—</td>"
          for h in hs]
    parts.append(f"<tr><td>random floor</td>{''.join(rc)}</tr></table>")

    for h in hs:
        parts.append(f"<h2>horizon h={h}</h2>")
        for wi, w in enumerate(windows):
            if w["h"] != h:
                continue
            drive = Path(w["drive"]).parts
            drive_short = "/".join(drive[-3:-1]) if len(drive) > 3 else w["drive"]
            arms_html = ""
            for mode in ("learned", "raw"):
                pr = by_window.get(wi, {}).get(mode)
                if pr is None:
                    arms_html += (f'<div class="arm"><span class="armname">{mode}'
                                  f"</span>: no row</div>")
                    continue
                ok = pr["success"]
                verdict = ('<span class="ok">SUCCESS</span>' if ok
                           else '<span class="bad">FAIL</span>')
                reached = (f' (goal at step {pr["reached_at"]})'
                           if ok and pr.get("reached_at") is not None else "")
                errs = ""
                if pr.get("plan_error"):
                    errs += f'<div class="err">plan_error: {html.escape(str(pr["plan_error"]))}</div>'
                if pr.get("perception_error") and mode == "learned":
                    errs += f'<div class="err">perception_error: {html.escape(str(pr["perception_error"])[:200])}</div>'
                prompt = html.escape(str(pr.get("prompt", "")))
                resp = html.escape(str(pr.get("response", ""))[:2000])
                arms_html += (
                    f'<div class="arm"><span class="armname">{mode}</span>: '
                    f'plan = {html.escape(str(pr["plan"]))} &rarr; {verdict}{reached}'
                    f"{errs}<details><summary>prompt</summary>"
                    f"<pre>{prompt}</pre></details>"
                    f"<details><summary>model response</summary>"
                    f"<pre>{resp}</pre></details></div>")
            parts.append(
                f'<div class="win"><div class="gcol"><div class="glab">START '
                f'(t={w["t"]}, {html.escape(drive_short)})</div>'
                f'{render_grid(w["start_grid"])}</div>'
                f'<div class="gcol"><div class="glab">GOAL (t+{h})</div>'
                f'{render_grid(w["goal_grid"])}</div>'
                f'<div class="meta">window #{wi} &middot; h={h} &middot; '
                f'random floor {w["random_success"]:.2f}<br>'
                f'<span class="gold">gold actions: {html.escape(str(w["actions"]))}'
                f"</span>{arms_html}</div></div>")

    out = args.out or args.shards.with_name(
        args.shards.stem + f"_planviz_{args.game}.html")
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB, "
          f"{len(windows)} windows)")


if __name__ == "__main__":
    main()
