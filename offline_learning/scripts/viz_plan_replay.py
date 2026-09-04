#!/usr/bin/env python3
"""One self-contained page holding every planning-eval trajectory in a run.

A companion to `viz_v2_online.py`, not a replacement: that page is the per-round
plan-vs-executed filmstrip, this one is the replay — a game x problem matrix, the board
stepping one executed action at a time, and beside it the exact prompt the model was
shown and the exact response it gave at that round. Modelled on the Claude-Code-on-Autumn
replay page (`cc_autumn/curated_replay.html`), with the agent's off-board work replaced by
the planner's prompt and response, which is what there is to look at in this eval.

Prompts are stored as indices into one shared line table. Within a rollout every prompt
repeats the same knowledge block and most of the same transcript, so the table comes to
about 6% of the raw prompt bytes and a whole 15-game run fits in a single page.

    uv run python offline_learning/scripts/viz_plan_replay.py \
        --run-root logs/2026-09-01/planning_v2_online_ds_nl \
        --out logs/2026-09-01/planning_v2_online_ds_nl/replay.html

    # or an explicit set of evaluator outputs
    uv run python offline_learning/scripts/viz_plan_replay.py \
        --eval logs/.../dino/online.json --eval logs/.../SET/online.json --out replay.html
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OFF = HERE.parent
REPO = OFF.parent
for _p in (str(REPO), str(OFF), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from offline_learning.human_replay import GAMES  # noqa: E402
from eval_curated_plan import load_eval_problems, select_goal_presentation  # noqa: E402
from viz_nl_goals import CSS, _CHARS, pack  # noqa: E402

BODY = HERE / "plan_replay_body.html"
STYLE = HERE / "plan_replay.css"
LLM_ARMS = ("raw", "lmwm", "wc")


def human_name(game: str) -> str:
    """The English name of a game, for every label the page shows.

    The benchmark's own IDs (`bt3gb`, `7www9`) are unreadable, so nothing user-facing
    uses them -- only paths, `task_uid`s and CLI filters, which have to keep matching
    the dataset. A game outside the table is shown as it is named."""
    return GAMES[game][1] if game in GAMES else game


class Lines:
    """One shared table of prompt/response lines; texts are stored as index lists."""

    def __init__(self) -> None:
        self.index: dict[str, int] = {}
        self.order: list[str] = []

    def add(self, text: str) -> list[int]:
        out = []
        for line in text.split("\n"):
            i = self.index.get(line)
            if i is None:
                i = self.index[line] = len(self.order)
                self.order.append(line)
            out.append(i)
        return out


def delta_frames(grids: list[str], pal: dict[str, int]) -> list[dict]:
    """Pack a rollout's frames as one full grid then per-cell edits against it."""
    out: list[dict] = []
    prev: str | None = None
    for g in grids:
        packed = pack(json.loads(g), pal).replace("|", "")
        if prev is None:
            out.append({"full": packed})
        else:
            d = [[i, c] for i, (c, p) in enumerate(zip(packed, prev)) if c != p]
            out.append({"d": d})
        prev = packed
    return out


def build(evals: list[dict], curated: dict, lines: Lines) -> dict:
    pal: dict[str, int] = {}
    problems, arms_seen = [], []
    for ev in evals:
        for r in ev["rows"]:
            cur = curated.get(r["task_uid"])
            if cur is None:                       # the row's problem file has moved on
                continue
            rollouts = []
            for arm in LLM_ARMS:
                cell = r.get(arm)
                if not isinstance(cell, dict):
                    continue
                for att in cell.get("attempts", []):
                    rounds = att.get("rounds", [])
                    if not rounds or not all(rd.get("grid_after") for rd in rounds):
                        continue              # a torn rollout has no board to step
                    if arm not in arms_seen:
                        arms_seen.append(arm)
                    carry: list[str] = []
                    # frame i is the board AFTER round i's action, so a round shows what
                    # its own action produced; the start board is on the problem card
                    packed = delta_frames([rd["grid_after"] for rd in rounds], pal)
                    steps = []
                    for i, rd in enumerate(rounds):
                        plan = rd.get("plan") or []
                        steps.append({
                            "n": rd["n"],
                            "action": rd.get("executed"),
                            "remaining": rd["remaining"],
                            "plan": plan,
                            "reached": bool(rd.get("reached_goal")),
                            "carried": bool(carry) and bool(plan) and carry == plan,
                            "error": rd.get("plan_error"),
                            "retries": rd.get("retry_errors") or [],
                            "prompt": lines.add(rd.get("prompt") or "(not retained)"),
                            "response": (lines.add(rd["response"])
                                         if rd.get("response") is not None else None),
                            "z": rd.get("z_after"),
                            "zerr": rd.get("z_error"),
                        })
                        carry = plan[1:]
                    rollouts.append({
                        "arm": arm,
                        "success": bool(att.get("success")),
                        "reached": att.get("reached_at"),
                        "used": att.get("actions_used", 0),
                        "failed": att.get("failed_reason"),
                        "frames": packed,
                        "rounds": steps,
                    })
            if not rollouts:
                continue
            start = cur["start"]
            problems.append({
                "uid": r["task_uid"], "game": r["game"],
                "human": human_name(r["game"]), "id": r["id"], "tier": r.get("tier"),
                "rows": len(start), "cols": len(start[0]),
                "mode": r.get("goal_presentation", "?"),
                "goal": r.get("nl_goal") or r.get("objective") or "",
                "startFrame": pack(start, pal),
                "goalFrame": (pack(cur["goal"], pal)
                              if r.get("goal_presentation") == "frame" else None),
                "h": r.get("h"),
                "cap": r.get("action_cap", (ev.get("config") or {}).get("max_actions")),
                "floor": (r.get("random_floor") if r.get("random_floor") is not None
                          else r.get("random_success_cap50")),
                "rollouts": rollouts,
            })
    return {
        "problems": problems,
        "arms": [a for a in LLM_ARMS if a in arms_seen],
        "palette": {_CHARS[i]: CSS.get(n, n) for n, i in pal.items()},
        "names": {_CHARS[i]: n for n, i in pal.items()},
        "lines": lines.order,
    }


def _pass_cells(rows: list[dict], arm: str) -> tuple[list[float], list[float]]:
    """(raw pass rates, floor-adjusted pass rates) for one arm over a game's rows.

    Mirrors the launcher's SUMMARY.md exactly: the floor is the one measured at the
    budget the rollouts actually ran under, and a row with no comparable floor scores
    into the raw column only."""
    pr, adj = [], []
    for r in rows:
        cell = r.get(arm)
        if not isinstance(cell, dict) or cell.get("pass_rate") is None:
            continue
        pr.append(cell["pass_rate"])
        fl = r.get("random_floor")
        if fl is None:
            fl = r.get("random_success_cap50")
            fl = r.get("random_success") if fl is None else fl
        if fl is not None:
            adj.append(max(0.0, (cell["pass_rate"] - fl) / (1 - fl)) if fl < 1 else 0.0)
    return pr, adj


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def summary(evals: list[dict], curated: dict, expect_games: list[str]) -> dict:
    """Game-wise raw/lmwm scoreboard plus how much of the run has landed.

    A game that has started is described by its own `online.json` — its post-exclusion
    row set and its rollout counter are authoritative, so a run that drops saturated
    rows still reaches 100%. A game that has not started is sized from the problems
    file, which is the only estimate available before its evaluator runs."""
    by_game = {}
    for ev in evals:
        cfg = ev.get("config") or {}
        for g in {r["game"] for r in ev["rows"]}:
            rows = [r for r in ev["rows"] if r["game"] == g]
            caps = [c for u, c in (cfg.get("action_caps") or {}).items()
                    if u.split(":")[0] == g]
            prev = by_game.get(g, {"rows": [], "cost": 0.0, "done": 0, "total": 0,
                                   "caps": []})
            by_game[g] = {
                "rows": prev["rows"] + rows,
                "cost": prev["cost"] + ev.get("cost", 0.0),
                # one evaluator invocation per game, so these do not accumulate
                "done": max(prev["done"], cfg.get("rollouts_done") or 0),
                "total": max(prev["total"], cfg.get("rollouts_total") or 0),
                "caps": prev["caps"] + caps,
            }

    order = list(expect_games) or sorted(by_game)
    for g in sorted(by_game):                      # a game nobody expected still shows
        if g not in order:
            order.append(g)

    n_arms = max(2, len({a for ev in evals for r in ev["rows"] for a in LLM_ARMS
                         if isinstance(r.get(a), dict)
                         and r[a].get("pass_rate") is not None}))
    games, tot_rows, tot_done, tot_total, tot_cost, done_games = [], [], 0, 0, 0.0, 0
    for g in order:
        seen = by_game.get(g)
        expect_rows = sum(1 for p in curated.values() if p["game"] == g)
        if seen is None:
            games.append({"game": g, "human": human_name(g),
                          "status": "pending", "n": 0,
                          "expect": expect_rows, "cap": None,
                          "raw": None, "lmwm": None, "adjRaw": None, "adjLmwm": None,
                          "cost": 0.0, "done": 0, "total": expect_rows * n_arms})
            tot_total += expect_rows * n_arms
            continue
        rows = seen["rows"]
        # the evaluator's own cap table is the row set it kept after any exclusions; an
        # older eval JSON has no cap table, but its rollout counter divides out to the
        # same thing. Only a game that never started is sized from the problems file.
        expect = (len(seen["caps"]) or (seen["total"] // n_arms if seen["total"] else 0)
                  or expect_rows)
        caps = sorted(set(seen["caps"])) or sorted(
            {r["action_cap"] for r in rows if r.get("action_cap") is not None})
        raw_pr, raw_adj = _pass_cells(rows, "raw")
        lm_pr, lm_adj = _pass_cells(rows, "lmwm")
        total = seen["total"] or expect * n_arms
        complete = seen["done"] >= total and total > 0
        done_games += complete
        games.append({
            "game": g, "human": human_name(g),
            "status": "done" if complete else "running",
            "n": len(rows), "expect": expect,
            "cap": (str(caps[0]) if len(caps) == 1
                    else (f"{caps[0]}–{caps[-1]}" if caps else None)),
            "raw": _mean(raw_pr), "lmwm": _mean(lm_pr),
            "adjRaw": _mean(raw_adj), "adjLmwm": _mean(lm_adj),
            "cost": seen["cost"], "done": seen["done"], "total": total,
        })
        tot_rows += rows
        tot_done += seen["done"]
        tot_total += total
        tot_cost += seen["cost"]

    raw_pr, raw_adj = _pass_cells(tot_rows, "raw")
    lm_pr, lm_adj = _pass_cells(tot_rows, "lmwm")
    return {
        "games": games,
        # row-weighted, so a 9-row game counts for more than a 3-row one
        "overall": {"n": len(tot_rows),
                    "expect": sum(x["expect"] for x in games),
                    "raw": _mean(raw_pr), "lmwm": _mean(lm_pr),
                    "adjRaw": _mean(raw_adj), "adjLmwm": _mean(lm_adj),
                    "cost": tot_cost},
        "progress": {"games": done_games, "gamesTotal": len(games),
                     "rollouts": tot_done, "rolloutsTotal": tot_total,
                     "problems": len(tot_rows),
                     "problemsTotal": sum(x["expect"] for x in games)},
    }


def render(evals: list[dict], curated: dict, out: Path, title: str,
           expect_games: list[str] | None = None) -> None:
    """Build one self-contained page from a set of evaluator outputs."""
    data = build(evals, curated, Lines())
    if not data["problems"]:
        raise SystemExit(f"{out}: no rollout in these evals carries a replayable board")

    caps = {p["cap"] for p in data["problems"] if p["cap"] is not None}
    models = {ev.get("config", {}).get("model", "?") for ev in evals}
    cost = sum(ev.get("cost", 0.0) for ev in evals)
    games = {p["game"] for p in data["problems"]}
    nr = sum(len(ro["rounds"]) for p in data["problems"] for ro in p["rollouts"])
    solved = sum(1 for p in data["problems"] for ro in p["rollouts"] if ro["success"])
    total = sum(len(p["rollouts"]) for p in data["problems"])
    presentation = evals[0].get("config", {}).get("goal_presentation")
    data["summary"] = summary(evals, curated, expect_games or [])
    data["page"] = {
        "title": title or f"Planning trajectories \u00b7 {presentation} goals",
        "blurb": ("Every rollout in the run, one executed action at a time, with the "
                  "prompt the planner saw and the answer it gave at that round. Pick a "
                  "problem; press a to switch arm."),
        "footer":
            f"<p>{len(data['problems'])} problems across {len(games)} game"
            f"{'' if len(games) == 1 else 's'} &middot; {total} rollouts &middot; {nr} "
            f"rounds &middot; {solved} solved &middot; planner "
            f"{', '.join(sorted(models))} &middot; ${cost:.2f} &middot; action budget "
            f"{'/'.join(str(c) for c in sorted(caps)) or '&mdash;'}.</p>"
            "<p>Scoring is any-step: a rollout ends the moment the goal first holds after "
            "an executed action, so an unsolved rollout always spends its whole budget. "
            "The bar under the board is that budget \u2014 one block per executed action, "
            "coloured by whether the model re-planned or played on from the plan it was "
            "already carrying, green where the goal held, with a red tick where a plan "
            "was rejected.</p>",
    }

    body = BODY.read_text()
    for marker, payload in (("/*__CSS__*/", STYLE.read_text()),
                            ("/*__DATA__*/", json.dumps(data, separators=(",", ":")))):
        if marker not in body:
            raise SystemExit(f"{BODY.name} has no {marker} to substitute")
        # `</` inside the JSON island or the stylesheet would close its element
        body = body.replace(marker, payload.replace("</", "<\\/"))

    out.parent.mkdir(parents=True, exist_ok=True)
    # atomic: a watcher redrawing a live run must never hand a browser half a page
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(body)
    os.replace(tmp, out)
    mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({len(data['problems'])} problems, {total} rollouts, "
          f"{nr} rounds, {len(data['lines'])} unique lines, {mb:.1f} MB)")
    if mb > 14:
        print("  close to the 16 MB page limit -- use --per-game")


def split_by_game(evals: list[dict]) -> dict[str, list[dict]]:
    """One evaluator-output list per game, so each game can get its own page."""
    out: dict[str, list[dict]] = {}
    for ev in evals:
        for g in sorted({r["game"] for r in ev["rows"]}):
            out.setdefault(g, []).append(
                dict(ev, rows=[r for r in ev["rows"] if r["game"] == g]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="",
                    help="a launcher out-root; every <game>/online.json under it is used")
    ap.add_argument("--eval", action="append", default=[],
                    help="an eval_curated_online.py JSON; repeatable")
    ap.add_argument("--problems", default="logs/2026-08-29/planning_v2/problems.json")
    ap.add_argument("--out", required=True,
                    help="the page to write; under --per-game, <out dir>/<game>/<name>")
    ap.add_argument("--title", default="")
    ap.add_argument("--per-game", action="store_true",
                    help="one page per game instead of one for the whole run -- a full "
                    "15-game run does not fit under the 16 MB page limit")
    ap.add_argument("--expect-games", default="",
                    help="comma-separated games the run will cover, in order; the ones "
                    "with no results yet appear in the summary as pending, so the "
                    "progress counter knows the whole run and not just what has landed")
    a = ap.parse_args()
    expect = [g for g in a.expect_games.split(",") if g.strip()]

    paths = [Path(p) for p in a.eval]
    if a.run_root:
        paths += sorted(Path(a.run_root).glob("*/online.json"))
    if not paths:
        raise SystemExit("nothing to render: pass --run-root or --eval")
    evals = [json.loads(p.read_text()) for p in paths]

    presentations = {ev.get("config", {}).get("goal_presentation") for ev in evals}
    if len(presentations) != 1 or presentations == {None}:
        raise SystemExit(f"evals must share one goal presentation, got {presentations}")
    presentation = presentations.pop()
    modes = {ev.get("config", {}).get("success_mode", "any") for ev in evals}
    mode = modes.pop() if len(modes) == 1 else "any"
    if mode == "online-any-step":
        mode = "any"

    _meta, selected = load_eval_problems(a.problems)
    selected = select_goal_presentation(selected, presentation, mode)
    curated = {p["task_uid"]: p for p in selected}

    out = Path(a.out)
    if not a.per_game:
        render(evals, curated, out, a.title, expect)
        return
    for game, evs in sorted(split_by_game(evals).items()):
        # a per-game page's progress is that game's, never the whole run's
        render(evs, curated, out.parent / game / out.name,
               a.title or f"{human_name(game)}: planning trajectories", [game])


if __name__ == "__main__":
    main()
