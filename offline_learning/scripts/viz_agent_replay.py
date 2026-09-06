#!/usr/bin/env python3
"""One self-contained page holding every trajectory in an `agent`-arm run.

The sibling of `viz_plan_replay.py`, for the arm that arrived from the other direction.
That page shows a planner's prompt and its reply, because a prompt and a reply are all
there is to a planner. This one has to show a coding agent, whose visible work is mostly
NOT on the board: it reads a corpus of recorded transitions, greps it, writes scratch
files, runs code, and only then commits to a plan. Rendering only the plan would show the
last 2% of the session and call it the arm.

So a problem here is two interleaved sequences:

  * **steps** -- one per EXECUTED action, each carrying the board it produced;
  * **turns** -- one per agent call, each carrying that call's reasoning, the shell it
    ran with the output it got back, the files it changed, and the plan it committed to.

A step names the turn that planned it, so scrubbing the board moves the transcript to the
thinking that produced the frame you are looking at. That link is the whole point of the
page: it is what distinguishes "the agent got it right" from "the agent got it right for
a reason it wrote down".

Both come from `<run>/traces/*.json`, written per problem by `research.autumn.launch`.
`rows.jsonl` is deliberately not enough -- it is the online evaluator's row shape and has
nowhere to put any of this.

    uv run python offline_learning/scripts/viz_agent_replay.py \
        --run-root logs/2026-09-06/agent_full \
        --out logs/2026-09-06/agent_full/replay.html

    # a full 15-game run does not fit one page; split it
    uv run python offline_learning/scripts/viz_agent_replay.py \
        --run-root logs/2026-09-06/agent_full --per-game \
        --out logs/2026-09-06/agent_full/replay.html
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OFF = HERE.parent
REPO = OFF.parent
for _p in (str(REPO), str(OFF), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from viz_nl_goals import CSS, _CHARS, pack       # noqa: E402
from viz_plan_replay import Lines, delta_frames, human_name  # noqa: E402

BODY = HERE / "agent_replay_body.html"
STYLE = HERE / "agent_replay.css"

# The trace keeps enough to audit; the page keeps enough to read. A session that `cat`s
# sixty drive files would otherwise carry half a megabyte of JSON per problem into a
# document nobody scrolls -- the trace file is the archive, this is the view.
MAX_OUTPUT_SHOWN = 1_500
MAX_REASONING_SHOWN = 12_000


def _clip(text: str, limit: int) -> tuple[str, int]:
    text = text or ""
    return (text[:limit], len(text) - limit) if len(text) > limit else (text, 0)


def _event(ev: dict, lines: Lines) -> dict:
    """One transcript entry, with its text moved into the shared line table."""
    kind = ev.get("kind")
    if kind == "command":
        out, dropped = _clip(str(ev.get("output") or ""), MAX_OUTPUT_SHOWN)
        return {"k": "cmd",
                "cmd": lines.add(str(ev.get("command") or "")),
                "out": lines.add(out) if out else None,
                "more": dropped + (1 if ev.get("truncated") else 0),
                "exit": ev.get("exit_code")}
    if kind == "file_change":
        return {"k": "file",
                "paths": [c.get("path", "") for c in (ev.get("changes") or [])][:20],
                "types": [c.get("type", "") for c in (ev.get("changes") or [])][:20],
                "n": ev.get("n") or len(ev.get("changes") or [])}
    if kind in ("reasoning", "message", "error"):
        text, dropped = _clip(str(ev.get("text") or ""), MAX_REASONING_SHOWN)
        return {"k": {"reasoning": "think", "message": "say", "error": "err"}[kind],
                "text": lines.add(text), "more": dropped}
    return {"k": "other", "text": lines.add(json.dumps(ev)[:2000]), "more": 0}


def _turn(t: dict, lines: Lines) -> dict:
    ev = [_event(e, lines) for e in (t.get("events") or [])]
    tok = t.get("tokens") or {}
    return {
        "i": t.get("i"),
        "kind": t.get("kind"),
        "attempt": t.get("attempt"),
        "n": t.get("n"),
        "remaining": t.get("remaining"),
        "plan": t.get("plan") or [],
        "rejected": t.get("rejected") or [],
        "wall": t.get("wall_s"),
        "tok": [tok.get("input_tokens") or 0, tok.get("cached_tokens") or 0,
                tok.get("output_tokens") or 0, tok.get("reasoning_tokens") or 0],
        "prompt": lines.add(t.get("prompt") or "(not retained)"),
        "events": ev,
        # a turn is worth opening if it thought or did something, not if it only replied
        "nthink": sum(1 for e in ev if e["k"] == "think"),
        "ncmd": sum(1 for e in ev if e["k"] == "cmd"),
        "nfile": sum(1 for e in ev if e["k"] == "file"),
    }


def build(traces: list[dict], lines: Lines) -> dict:
    pal: dict[str, int] = {}
    problems = []
    for tr in traces:
        steps = tr.get("steps") or []
        start = json.loads(tr["start_grid"])
        # frame i is the board AFTER step i's action; the start board is on the card
        frames = delta_frames([s["grid_after"] for s in steps], pal) if steps else []
        problems.append({
            "uid": tr["task_uid"],
            "game": tr["game"],
            "human": human_name(tr["game"]),
            "label": tr.get("label"),
            "goal": tr.get("nl_goal") or "",
            "cap": tr.get("action_cap"),
            "rows": len(start), "cols": len(start[0]),
            "startFrame": pack(start, pal),
            "success": bool(tr.get("success")),
            "liveSuccess": tr.get("live_success"),
            "reachedAt": tr.get("reached_at"),
            "used": tr.get("actions_used") or 0,
            "failed": tr.get("failed_reason"),
            "studies": tr.get("study_rounds_used") or 0,
            "wall": tr.get("wall_s"),
            "usage": tr.get("usage") or {},
            "alphabet": tr.get("alphabet") or [],
            "frames": frames,
            "steps": [{"n": s.get("n"), "a": s.get("action"), "t": s.get("turn"),
                       "r": bool(s.get("reached")), "x": bool(s.get("terminated")),
                       "rem": s.get("remaining"),
                       "pi": s.get("plan_index"), "pt": s.get("plan_total")}
                      for s in steps],
            "turns": [_turn(t, lines) for t in (tr.get("turns") or [])],
        })
    problems.sort(key=lambda p: (p["human"], p["uid"]))
    return {
        "problems": problems,
        "palette": {_CHARS[i]: CSS.get(n, n) for n, i in pal.items()},
        "names": {_CHARS[i]: n for n, i in pal.items()},
        "lines": lines.order,
    }


def summary(traces: list[dict]) -> dict:
    """Per-game scoreboard, plus what the arm spent to get it.

    Cost is not a footnote for this arm. Inference compute is the one axis the study does
    not match across arms, so the page that shows the trajectories shows the bill.
    """
    by_game: dict[str, list[dict]] = {}
    for tr in traces:
        by_game.setdefault(tr["game"], []).append(tr)
    games = []
    for g, trs in sorted(by_game.items(), key=lambda kv: human_name(kv[0])):
        done = [t for t in trs if t.get("success") is not None]
        wins = sum(1 for t in trs if t.get("success"))
        usage = [t.get("usage") or {} for t in trs]
        games.append({
            "game": g, "human": human_name(g), "label": trs[0].get("label"),
            "n": len(trs), "wins": wins,
            "pass": wins / len(done) if done else None,
            "calls": sum(u.get("calls") or 0 for u in usage),
            "in": sum(u.get("in") or 0 for u in usage),
            "out": sum(u.get("out") or 0 for u in usage),
            "reasoning": sum(u.get("reasoning") or 0 for u in usage),
            "wall": sum(t.get("wall_s") or 0 for t in trs),
        })
    n = sum(g["n"] for g in games)
    wins = sum(g["wins"] for g in games)
    return {
        "games": games,
        "n": n, "wins": wins,
        "macro": (sum(g["pass"] for g in games if g["pass"] is not None)
                  / max(1, sum(1 for g in games if g["pass"] is not None))),
        "micro": wins / n if n else 0.0,
        "calls": sum(g["calls"] for g in games),
        "in": sum(g["in"] for g in games),
        "out": sum(g["out"] for g in games),
        "reasoning": sum(g["reasoning"] for g in games),
        "wall": sum(g["wall"] for g in games),
    }


def render(traces: list[dict], out: Path, title: str) -> None:
    lines = Lines()
    data = build(traces, lines)
    data["summary"] = summary(traces)
    data["page"] = {
        "title": title or "Agentic planning trajectories",
        "blurb": ("PRO-LONG (deepseek-v4-flash via codex) on the curated planning "
                  "battery. One session per problem, started from the same 60 recorded "
                  "transitions the world model was fit on. Scrub the board; the "
                  "transcript follows the turn that planned the frame."),
        "footer": ("Boards replayed from the executed action list. Reasoning, shell and "
                   "file edits are the codex event stream as recorded, clipped for the "
                   "page &mdash; <code>traces/*.json</code> holds the full record."),
    }
    body = BODY.read_text()
    css = STYLE.read_text()
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = body.replace("/*__CSS__*/", css).replace("/*__DATA__*/", payload)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    mb = out.stat().st_size / 1e6
    print(f"{out}  {mb:.1f} MB  {len(data['problems'])} problem(s), "
          f"{sum(len(p['turns']) for p in data['problems'])} turns, "
          f"{len(data['lines'])} unique lines")
    if mb > 16:
        print("  WARNING: over the 16 MB artifact limit; use --per-game")


INDEX = """<!doctype html>
<meta charset="utf-8"><title>{title}</title><style>{css}</style>
<header><h1>{title}</h1><p>{blurb}</p></header>
<section class="detail">
{summary}
<h2>every problem</h2>
<table class="sum">
<tr><th>world</th><th>problem</th><th>goal</th><th></th><th>actions</th><th>calls</th>
    <th>tokens in</th><th>reasoning</th><th>wall</th></tr>
{rows}
</table>
</section>
<footer>{footer}</footer>
"""


def render_index(traces: list[dict], out: Path, title: str) -> None:
    """The run-level page as an INDEX, not as every session at once.

    Measured at 10 problems in: a whole-run page carrying every transcript runs about
    290 KB per problem, so the full 86 would be a ~25 MB document that a browser opens
    slowly and the artifact limit rejects outright. The transcripts belong on the
    per-game pages, which stay a couple of megabytes each; what the run level owes the
    reader is the scoreboard and a way in.
    """
    s = summary(traces)
    def cell(v, d=2):
        return "&mdash;" if v is None else f"{v:.{d}f}"
    head = ["<tr><td colspan=9><b>the run</b></td></tr>"]
    for g in s["games"]:
        head.append(
            f"<tr><td><a href='{g['game']}/replay.html'>{g['human']}</a></td>"
            f"<td class=n>{g['wins']}/{g['n']}</td><td class=n>{cell(g['pass'])}</td>"
            f"<td colspan=3 class=n>{g['calls']} calls</td>"
            f"<td class=n>{g['in']/1e6:.1f}M</td><td class=n>{g['reasoning']/1e6:.2f}M</td>"
            f"<td class=n>{g['wall']/3600:.1f}h</td></tr>")
    head.append(
        f"<tr class=tot><td>all {len(s['games'])}</td><td class=n>{s['wins']}/{s['n']}</td>"
        f"<td class=n>{cell(s['macro'])}</td><td colspan=3 class=n>{s['calls']} calls</td>"
        f"<td class=n>{s['in']/1e6:.1f}M</td><td class=n>{s['reasoning']/1e6:.2f}M</td>"
        f"<td class=n>{s['wall']/3600:.1f}h</td></tr>")

    rows = []
    for tr in sorted(traces, key=lambda t: (human_name(t["game"]), t["task_uid"])):
        u = tr.get("usage") or {}
        ok = bool(tr.get("success"))
        rows.append(
            f"<tr><td><a href='{tr['game']}/replay.html'>{human_name(tr['game'])}</a></td>"
            f"<td>{tr['task_uid'].split(':')[1] if ':' in tr['task_uid'] else tr['task_uid']}</td>"
            f"<td>{(tr.get('nl_goal') or '')[:70]}</td>"
            f"<td style='color:var(--{'ok' if ok else 'bad'})'>{'solved' if ok else 'missed'}</td>"
            f"<td class=n>{tr.get('actions_used')}/{tr.get('action_cap')}</td>"
            f"<td class=n>{u.get('calls')}</td>"
            f"<td class=n>{(u.get('in') or 0)/1e6:.1f}M</td>"
            f"<td class=n>{(u.get('reasoning') or 0)/1e3:.0f}k</td>"
            f"<td class=n>{(tr.get('wall_s') or 0)/60:.0f}m</td></tr>")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(INDEX.format(
        title=title or "Agentic planning trajectories",
        css=STYLE.read_text(),
        blurb=("PRO-LONG (deepseek-v4-flash via codex) on the curated planning battery. "
               "Pick a world to watch its sessions: the board steps one action at a "
               "time and the transcript follows the turn that planned each frame."),
        summary="<h2>the run</h2><table class='sum'>" + "".join(head) + "</table>",
        rows="".join(rows),
        footer="Per-world pages hold the transcripts; <code>traces/*.json</code> holds "
               "the full record."))
    print(f"{out}  {out.stat().st_size/1e6:.2f} MB  index over {len(traces)} problem(s)")


def load_traces(paths: list[Path]) -> list[dict]:
    out = []
    for p in paths:
        try:
            out.append(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  skipping {p}: {exc}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", default="",
                    help="a launch.py --out directory; traces/*.json under it are used")
    ap.add_argument("--trace", action="append", default=[], help="one trace; repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--games", nargs="*", help="restrict to these worlds")
    ap.add_argument("--index", action="store_true",
                    help="render the run-level INDEX (scoreboard + links) instead of "
                         "every transcript at once; see render_index for why")
    ap.add_argument("--per-game", action="store_true",
                    help="one page per game under <out dir>/<game>/<name>")
    a = ap.parse_args()

    paths = [Path(p) for p in a.trace]
    if a.run_root:
        paths += sorted(Path(a.run_root).glob("traces/*.json"))
    if not paths:
        raise SystemExit("nothing to render: pass --run-root or --trace")
    traces = load_traces(paths)
    if a.games:
        traces = [t for t in traces if t.get("game") in set(a.games)]
    if not traces:
        raise SystemExit("no traces matched")

    out = Path(a.out)
    if a.index:
        render_index(traces, out, a.title)
        return
    if not a.per_game:
        render(traces, out, a.title)
        return
    by_game: dict[str, list[dict]] = {}
    for t in traces:
        by_game.setdefault(t["game"], []).append(t)
    for game, trs in sorted(by_game.items()):
        render(trs, out.parent / game / out.name,
               a.title or f"{human_name(game)}: agentic trajectories")


if __name__ == "__main__":
    main()
