#!/usr/bin/env python3
"""Live sweep dashboard for the ARC image-mode clean_sweep run.

Aggregates every (game, seed) under a clean_sweep out-dir and renders ONE self-
refreshing HTML page so you can watch results land as each game progresses:

  - live progress from <game>_seed<N>/gepa_run_seed<N>/process_log.jsonl
    (iteration count, accept/reject tally, running-best score sparkline)
  - final metrics parsed from <game>_seed<N>/stdout.txt once a game finishes
    (GEPA test acc vs raw-frame ref vs chance vs start-P), reusing
    clean_sweep.parse_summary so the numbers match the sweep's own table.

Usage (one-shot):
  uv run python prototypes/perc_invdyn/arc_imgmode_dashboard.py \
      --out-dir logs/clean_sweep_gepa_padded_ctxk9_arc_imgmode

Add --watch N to regenerate every N seconds until the run's results.json appears
(the sweep writes it only after ALL games finish); the HTML also <meta>-refreshes.
"""
from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path

from clean_sweep import parse_summary  # reuse the exact summary regexes


def load_process_log(gdir: Path) -> list[dict]:
    p = gdir / "process_log.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:  # noqa: BLE001
            continue
    return rows


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def best_curve(rows: list[dict]) -> list[float]:
    """Running-best accepted score over iterations (for the sparkline)."""
    best, curve = None, []
    for r in rows:
        cand = _num(r.get("new_score")) if r.get("verdict") == "accepted" else None
        if cand is None:
            cand = _num(r.get("selected_score"))
        if cand is not None:
            best = cand if best is None else max(best, cand)
        curve.append(best if best is not None else 0.0)
    return curve


def sparkline(curve: list[float], w=180, h=34) -> str:
    if not curve:
        return '<span style="color:#667">no iterations yet</span>'
    lo, hi = min(curve), max(curve)
    rng = (hi - lo) or 1.0
    n = len(curve)
    step = w / max(n - 1, 1)
    pts = " ".join(
        f"{i*step:.1f},{h - 2 - (v - lo) / rng * (h - 4):.1f}" for i, v in enumerate(curve)
    )
    return (
        f'<svg width="{w}" height="{h}" style="vertical-align:middle">'
        f'<polyline fill="none" stroke="#5ad0ff" stroke-width="1.6" points="{pts}"/>'
        f"</svg>"
    )


def game_state(gdir_parent: Path, game: str, seed: int) -> dict:
    d = gdir_parent / f"{game}_seed{seed}"
    gdir = d / f"gepa_run_seed{seed}"
    rows = load_process_log(gdir)
    accepted = sum(1 for r in rows if r.get("verdict") == "accepted")
    rejected = sum(1 for r in rows if r.get("verdict") == "rejected")
    stdout = d / "stdout.txt"
    summ = parse_summary(stdout.read_text()) if stdout.exists() else {}
    done = stdout.exists()
    return {
        "game": game,
        "iters": len(rows),
        "accepted": accepted,
        "rejected": rejected,
        "curve": best_curve(rows),
        "done": done,
        "summ": summ,
        "started": d.exists(),
    }


def _cell(v, fmt="{:.2f}"):
    return fmt.format(v) if isinstance(v, (int, float)) else "&mdash;"


def render(out_dir: Path, games: list[str], seed: int, refresh: int) -> str:
    states = [game_state(out_dir, g, seed) for g in games]
    n_done = sum(s["done"] for s in states)
    rows_html = []
    for s in states:
        su = s["summ"]
        gepa, raw, chance, start = (
            su.get("gepa_acc"),
            su.get("raw"),
            su.get("chance"),
            su.get("start_acc"),
        )
        if s["done"]:
            status = '<span style="color:#6ee787">done</span>'
        elif s["started"]:
            status = '<span style="color:#ffd45a">running</span>'
        else:
            status = '<span style="color:#667">queued</span>'
        # highlight GEPA vs raw baseline
        gcol = "#6ee787" if (isinstance(gepa, float) and isinstance(raw, float) and gepa >= raw) else "#ff9a9a"
        rows_html.append(
            f"<tr>"
            f"<td class=g>{html.escape(s['game'])}</td>"
            f"<td>{status}</td>"
            f"<td class=n>{s['iters']}</td>"
            f"<td class=n style='color:#6ee787'>{s['accepted']}</td>"
            f"<td class=n style='color:#ff9a9a'>{s['rejected']}</td>"
            f"<td>{sparkline(s['curve'])}</td>"
            f"<td class=n style='color:{gcol};font-weight:700'>{_cell(gepa)}</td>"
            f"<td class=n>{_cell(raw)}</td>"
            f"<td class=n>{_cell(chance)}</td>"
            f"<td class=n>{_cell(start)}</td>"
            f"</tr>"
        )
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    meta = f'<meta http-equiv="refresh" content="{refresh}">' if refresh else ""
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">{meta}
<title>ARC image-mode sweep</title><style>
body{{background:#0d1117;color:#c9d1d9;font:13px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:24px}}
h1{{font-size:17px;margin:0 0 4px}}.sub{{color:#8b949e;margin-bottom:16px}}
table{{border-collapse:collapse;width:100%;max-width:1000px}}
th,td{{padding:7px 10px;border-bottom:1px solid #21262d;text-align:left}}
th{{color:#8b949e;font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.4px}}
td.n{{text-align:right;font-variant-numeric:tabular-nums}}td.g{{font-weight:700;color:#e6edf3}}
tr:hover td{{background:#161b22}}
</style></head><body>
<h1>ARC image-mode clean_sweep &mdash; live</h1>
<div class=sub>{html.escape(str(out_dir))} &middot; images to P&amp;B proposers, NOT to metric calls &middot;
{n_done}/{len(states)} games done &middot; updated {ts} (auto-refresh {refresh}s)</div>
<table>
<tr><th>game</th><th>status</th><th>iters</th><th>acc</th><th>rej</th><th>best-score</th>
<th>GEPA test</th><th>raw ref</th><th>chance</th><th>start-P</th></tr>
{''.join(rows_html)}
</table>
<div class=sub style="margin-top:14px">GEPA test = held-out inverse-dynamics accuracy of the best candidate;
green when &ge; raw-frame reference. Sparkline = running-best train score over GEPA iterations.</div>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="clean_sweep out-dir under logs/")
    ap.add_argument("--games", default="ft09,ls20,sp80,tn36,vc33")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--refresh", type=int, default=20, help="HTML auto-refresh seconds")
    ap.add_argument("--watch", type=int, default=0,
                    help="regenerate every N seconds until results.json appears (0=one-shot)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    games = [g for g in args.games.split(",") if g.strip()]
    html_path = out_dir / "dashboard.html"

    def once():
        out_dir.mkdir(parents=True, exist_ok=True)
        html_path.write_text(render(out_dir, games, args.seed, args.refresh))

    once()
    print(f"[dashboard] {html_path}")
    if args.watch:
        while not (out_dir / "results.json").exists():
            time.sleep(args.watch)
            once()
            print(f"[dashboard] refreshed {time.strftime('%H:%M:%S')}", flush=True)
        once()
        print("[dashboard] sweep complete (results.json present) -> final render written")


if __name__ == "__main__":
    main()
