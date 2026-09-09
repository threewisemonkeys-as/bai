#!/usr/bin/env python3
"""Read the ONLINE planning-v2 run dirs and report what the paper's tables claim.

The evaluator writes one `<game>/online.json` per game and `viz_plan_replay.py` folds
those same files into `<run>/replay.html`, so a number visible in the replay page is a
number in an `online.json`: this reads the JSON directly and never parses HTML. Games
still queued simply have no file yet, which is why it is safe to run against a live run
-- it reports what has landed and marks the rest pending.

    # what the two NL runs currently say, plus the paired comparison
    uv run python offline_learning/scripts/report_planning_v2_online.py

    # regenerate the LaTeX between the AUTO markers in paper/main.tex
    uv run python offline_learning/scripts/report_planning_v2_online.py --write-tex

    # only the things that would make a number wrong
    uv run python offline_learning/scripts/report_planning_v2_online.py --check

`--run LABEL=PATH` replaces the default pair; the LABEL is what the LaTeX column is
called. `--raw-from LABEL` picks which run supplies the shared Raw column (each run has
its own raw rollouts and the table has one Raw column).

The Agent columns come from `--agent PATH` and `--agent-wm PATH`, `research.autumn.launch`
run dirs, which are a different shape: one `rows.jsonl` for the whole run rather than a
per-game `online.json`, because the agent plays one session per problem and never has a
per-game evaluator pass. The rows are the evaluator's own row shape, so they fold into the
same aggregates; `--no-agent` puts both columns back to em-dashes.

The two agent columns are the same harness under one difference: `--agent-wm` is the run
whose workspaces also held the beliefs and perception module the NLWM column plans with,
and it prints as `NLWM (Agentic)` at the far end of the table because that is what it is --
the same world model, planned with by an agent instead of by a single call. Agent vs
NLWM (Agentic) is therefore what the world model is worth to an agent that can compute over
the data it was learned from, and NLWM (Agentic) vs NLWM is what the agent loop is worth
given the same world model.

The default pair IS a matched comparison as of 2026-09-04: both runs score the same 86
problems under the same per-problem action budgets, and differ only in which reflector
built the artifacts they plan with (DeepSeek vs Opus-5). The superseded DeepSeek run
`logs/2026-09-01/planning_v2_online_ds_nl` was flat cap 50 on a different problems file,
so it shared neither the problem set nor the budgets; `--check` and the COMPARABILITY
block report any such mismatch, and `paired()` prints a warning if the budgets diverge.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from offline_learning.human_replay import GAMES as HGAMES  # noqa: E402

sys.path.insert(0, str(REPO / "offline_learning/launch"))
from launch_planning_v2_online import GAME_ORDER  # noqa: E402

LLM_ARMS = ["raw", "lmwm", "icl"]
ARMS = LLM_ARMS + ["wc"]
DEFAULT_COLUMN_ARM = "lmwm"
# how an arm prints in a table header (the internal names are not paper words)
ARM_DISPLAY = {"raw": "Raw", "lmwm": "NLWM", "icl": "ICL", "wc": "WorldCoder"}
# The agent arm is not an arm of this evaluator: it is its own run, written by
# `research.autumn.launch` as a single rows.jsonl in the evaluator's row shape. It is
# named here because the results table has always reserved a column for it.
AGENT_ARM = "agent"
TIERS = ["L1", "L2", "L3", "L4"]

# the paper's row order, and the display name each benchmark id prints as
PAPER_ORDER = ["eahcw", "egg", "bt3gb", "dq8gc", "7xf97", "n2ntd", "va6fq", "s2kt7",
               "colour_lines", "SET", "diffusion", "dino", "f5w3n", "logic_gates",
               "7www9"]
DISPLAY = {"colour_lines": "Colour Lines", "SET": "SET", "logic_gates": "Logic Gates",
           "f5w3n": "Space Invaders"}

DEFAULT_RUNS = [("NLWM", "logs/2026-09-03/planning_v2_online_ds_percap_nl"),
                ("NLWM (SL)", "logs/2026-09-02/planning_v2_online_opus5_nl")]
# label, run dir, and whether the column prints AFTER the --run columns. `Agent` is a
# baseline and sits with Raw; the agent holding NLWM's artifacts is a variant of NLWM and
# reads as one, so it sits at the far end beside the other NLWM columns.
DEFAULT_AGENT = ("Agent", "logs/2026-09-06/agent_full", False)
DEFAULT_AGENT_WM = ("NLWM (Agentic)", "logs/2026-09-08/agent_wm_full", True)

def display_name(game: str) -> str:
    """The paper's name for a game: the English name, title-cased, with overrides."""
    if game in DISPLAY:
        return DISPLAY[game]
    return HGAMES[game][1].replace("_", " ").title() if game in HGAMES else game


def floor_of(row: dict) -> float | None:
    """The random floor comparable to this row's scores.

    `emit()` already resolved it to the budget the rollouts actually ran under and stored
    it as `random_floor`; the cap50 fields only describe a flat-50 regime, so they are a
    fallback for rows written before per-problem caps existed."""
    if row.get("random_floor") is not None:
        return row["random_floor"]
    cap50 = row.get("random_success_cap50")
    return cap50 if cap50 is not None else row.get("random_success")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def adjusted(pass_rate: float | None, floor: float | None) -> float | None:
    """Headroom-normalised score: how much of the gap above chance the arm closed.

    This has to be applied to an AGGREGATE, never per problem. With one attempt a
    problem scores 0 or 1, and (1-f)/(1-f) = 1 while max(0, -f/(1-f)) = 0, so a per-row
    normalisation returns the row's own score and averaging it reproduces pass@1 exactly
    -- which is why the launcher's SUMMARY.md `adj` columns equal its `pass` columns."""
    if pass_rate is None or floor is None:
        return None
    return 0.0 if floor >= 1 else max(0.0, (pass_rate - floor) / (1 - floor))


# ------------------------------------------------------------------ loading
def arm_scores(rows: list[dict], arm: str) -> dict:
    """pass@1 / pass@any / floor-adjusted pass@1 over the rows this arm actually scored.

    The floor is averaged over the same rows the score is, so the two are comparable even
    when an arm skipped a problem the other one scored."""
    p1, pany, floors = [], [], []
    for r in rows:
        cell = r.get(arm)
        if not isinstance(cell, dict) or cell.get("pass_rate") is None:
            continue
        p1.append(cell["pass_rate"])
        if cell.get("pass_any") is not None:
            pany.append(float(cell["pass_any"]))
        floors.append(floor_of(r))
    pass1, floor = mean(p1), mean(floors)
    return {"n": len(p1), "pass1": pass1, "pass_any": mean(pany), "floor": floor,
            "adj": adjusted(pass1, floor)}


def load_run(label: str, root: Path, arm: str = DEFAULT_COLUMN_ARM) -> dict:
    """Every landed game of one run, plus the freshness of the pages built from it.

    `arm` is which arm of that run this column reports. It is usually `lmwm` (the world
    model), but the in-context baseline is a THIRD ARM of an existing run rather than a
    run of its own, so a column has to be able to name it."""
    games, order = {}, []
    for game in GAME_ORDER:
        f = root / game / "online.json"
        if not f.exists():
            continue
        ev = json.loads(f.read_text())
        rows, cfg = ev["rows"], ev.get("config", {})
        done, total = cfg.get("rollouts_done"), cfg.get("rollouts_total")
        # rows written before per-problem caps existed carry no `action_cap`; those runs
        # were flat-budget, so the config's `max_actions` is the cap they actually ran at
        for r in rows:
            r["_cap"] = r.get("action_cap") or cfg.get("max_actions")
        caps = sorted({r["_cap"] for r in rows if r["_cap"] is not None})
        games[game] = {
            "rows": rows,
            "config": cfg,
            "cost": ev.get("cost", 0.0),
            "complete": done is not None and done == total,
            "rollouts": (done, total),
            "cap": (f"{caps[0]}-{caps[-1]}" if len(caps) > 1 else
                    (str(caps[0]) if caps else "?")),
            "floor": mean(floor_of(r) for r in rows),
            "arms": {a: arm_scores(rows, a) for a in ARMS},
            "mtime": f.stat().st_mtime,
            "replay_mtime": (root / game / "replay.html").stat().st_mtime
                            if (root / game / "replay.html").exists() else None,
        }
        order.append(game)
    combined = root / "replay.html"
    return {
        "label": label, "root": root, "arm": arm, "games": games, "order": order,
        "pending": [g for g in GAME_ORDER if g not in games],
        "replay": combined if combined.exists() else None,
        "replay_mtime": combined.stat().st_mtime if combined.exists() else None,
        "config": games[order[0]]["config"] if order else {},
        # the early games of a run predate --cap-mode; they ran flat, like `fixed`
        "cap_mode": "/".join(sorted({(g["config"].get("cap_mode") or "fixed")
                                     for g in games.values()})) or "?",
    }


def load_agent_run(label: str, root: Path, reference: dict, *,
                   trailing: bool = False) -> dict:
    """The agent arm folded into the shape the tables already read.

    `research.autumn.launch` writes ONE `rows.jsonl` for the whole run instead of a
    per-game `online.json`, because its unit is a session per problem and it has no
    per-game evaluator pass -- but it writes the evaluator's row shape, so the same
    `arm_scores` reads it and the same `scored`/`run_totals` fold it.

    A game is complete when it has a row for every problem the Raw column scores. That
    is the only definition under which the columns are comparable, so `reference` is the
    run supplying Raw rather than the agent run's own idea of its size -- and the same
    reference answers the two questions that would silently invalidate the column: did
    this arm score the SAME problems, and under the SAME per-problem budgets?
    """
    # `rows.jsonl` is the ledger, but a run launched with --workers writes one file
    # per worker and merges them only at the end -- so a live parallel run is read here
    # exactly the way its own resume reads it, and the table tightens as it plays.
    files = [f for f in [root / "rows.jsonl", *sorted(root.glob("rows.w*.jsonl"))]
             if f.is_file()]
    if not files:
        raise SystemExit(f"no rows.jsonl in {root}")
    by_game: dict[str, list[dict]] = {}
    seen: set[str] = set()
    for f in files:
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:             # a torn last line on a live run
                continue
            if row.get("game") and row.get("task_uid") not in seen:
                seen.add(row["task_uid"])
                by_game.setdefault(row["game"], []).append(row)
    expected = {g: len(v["rows"]) for g, v in reference["games"].items()}
    games = {g: {"rows": rows,
                 "arms": {AGENT_ARM: arm_scores(rows, AGENT_ARM)},
                 "expected": expected.get(g),
                 "complete": expected.get(g) is not None and len(rows) >= expected[g]}
             for g, rows in by_game.items()}
    ref_caps = {r["task_uid"]: r["_cap"]
                for v in reference["games"].values() for r in v["rows"]}
    mine = {r["task_uid"]: r.get("action_cap")
            for rows in by_game.values() for r in rows}
    return {"label": label, "root": root, "arm": AGENT_ARM, "games": games,
            "trailing": trailing,
            "order": [g for g in GAME_ORDER if g in games],
            "pending": [g for g in GAME_ORDER if g not in games],
            "shared": len(set(mine) & set(ref_caps)),
            "only_mine": sorted(set(mine) - set(ref_caps)),
            "only_ref": sorted(set(ref_caps) - set(mine)),
            "cap_diff": sorted(u for u in set(mine) & set(ref_caps)
                               if mine[u] != ref_caps[u]),
            "reference": reference["label"],
            "mtime": max(f.stat().st_mtime for f in files)}


def run_totals(run: dict, arm: str, games: list[str] | None = None) -> dict:
    """Macro (per-game) and micro (per-problem) means over the games that have landed."""
    keys = [g for g in (games or run["order"]) if g in run["games"]]
    per_game = [run["games"][g]["arms"][arm]["pass1"] for g in keys]
    per_game_adj = [run["games"][g]["arms"][arm]["adj"] for g in keys]  # per-game floors
    flat = [r for g in keys for r in run["games"][g]["rows"]]
    micro = arm_scores(flat, arm)
    return {"games": len([x for x in per_game if x is not None]),
            "macro": mean(per_game), "macro_adj": mean(per_game_adj),
            "micro": micro["pass1"], "micro_adj": micro["adj"], "n": micro["n"]}


# ------------------------------------------------------------------ report
def fmt(v, d=2):
    return "  --" if v is None else f"{v:.{d}f}"


def per_game_table(runs: list[dict]) -> list[str]:
    """One block per run: n / cap / floor belong to the run, not to the game."""
    head = ["game"]
    for r in runs:
        head += [f"{r['label']}:n", "cap", "rand", *LLM_ARMS]
    L = ["  ".join(h.rjust(13) for h in head)]
    for game in PAPER_ORDER:
        cells = [display_name(game)]
        for r in runs:
            g = r["games"].get(game)
            if g is None:
                cells += ["pending", "", "", "", ""]
                continue
            mark = "" if g["complete"] else "*"
            cells += [str(len(g["rows"])) + mark, g["cap"], fmt(g["floor"])]
            cells += [fmt(g["arms"][a]["pass1"]) for a in LLM_ARMS]
        L.append("  ".join(c.rjust(13) for c in cells))
    return L


def paired(runs: list[dict]) -> list[str]:
    """Whether the runs can be read as one table: same problems, same budgets."""
    if len(runs) != 2:
        return []
    a, b = runs
    idx = [{r["task_uid"]: r for g in run["games"].values() for r in g["rows"]}
           for run in (a, b)]
    shared = sorted(set(idx[0]) & set(idx[1]))
    cap_mismatch = [u for u in shared if idx[0][u]["_cap"] != idx[1][u]["_cap"]]
    L = ["", "COMPARABILITY", "-" * 13,
         f"  {a['label']}: {len(idx[0])} problems over {len(a['games'])} games"
         f"  (caps {a['cap_mode']}, artifacts "
         f"{a['config'].get('artifact_root')})",
         f"  {b['label']}: {len(idx[1])} problems over {len(b['games'])} games"
         f"  (caps {b['cap_mode']}, artifacts "
         f"{b['config'].get('artifact_root')})",
         f"  shared task_uids: {len(shared)}"
         f"  ({a['label']}-only {len(set(idx[0]) - set(idx[1]))},"
         f" {b['label']}-only {len(set(idx[1]) - set(idx[0]))})",
         f"  of those, action budgets differ on: {len(cap_mismatch)}"]
    if cap_mismatch:
        L.append("  -> the shared rows are NOT a matched pair; scores below are"
                 " descriptive only")
    L.append("")
    L.append(f"  {'arm':>10}  {a['label']:>12}  {b['label']:>12}   (shared rows only)")
    for arm in LLM_ARMS:
        cells = []
        for run, index in zip((a, b), idx):
            rows = [index[u] for u in shared]
            cells.append(fmt(arm_scores(rows, arm)["pass1"]))
        L.append(f"  {arm:>10}  {cells[0]:>12}  {cells[1]:>12}")
    return L


def tiers(runs: list[dict]) -> list[str]:
    L = ["", "PER TIER (pass@1 over landed games)", "-" * 35,
         "  " + "  ".join(x.rjust(12) for x in
                          ["tier"] + [f"{r['label']}:{a}" for r in runs
                                      for a in LLM_ARMS])]
    for tier in TIERS:
        cells = [tier]
        for r in runs:
            rows = [x for g in r["games"].values() for x in g["rows"]
                    if x["tier"] == tier]
            cells += [f"{fmt(arm_scores(rows, a)['pass1'])} (n={len(rows)})"
                      for a in LLM_ARMS]
        L.append("  " + "  ".join(c.rjust(12) for c in cells))
    return L


def checks(runs: list[dict], agents: list[dict] = ()) -> list[str]:
    """Everything that would make a reported number wrong or stale."""
    L = ["", "CHECKS", "-" * 6]
    ok = True
    for run in runs:
        L.append(f"  {run['label']}  ({run['root']})")
        if run["pending"]:
            ok = False
            L.append(f"    ! {len(run['pending'])} games have no online.json yet: "
                     + ", ".join(run["pending"]))
        for game, g in run["games"].items():
            if not g["complete"]:
                ok = False
                L.append(f"    ! {game}: only {g['rollouts'][0]}/{g['rollouts'][1]} "
                         "rollouts done -- partial scores")
            for arm in LLM_ARMS:
                bad = {r["task_uid"]: r[arm].get("status") for r in g["rows"]
                       if isinstance(r.get(arm), dict)
                       and r[arm].get("status") not in (None, "evaluated")}
                if bad:
                    ok = False
                    L.append(f"    ! {game}/{arm}: {len(bad)} rows not evaluated "
                             f"({sorted(set(bad.values()))}): "
                             f"{', '.join(sorted(bad)[:3])}")
            missing = [r["task_uid"] for r in g["rows"] if floor_of(r) is None]
            if missing:
                ok = False
                L.append(f"    ! {game}: {len(missing)} rows have no random floor at "
                         "the cap in force -- excluded from the adjusted score")
            if g["replay_mtime"] is not None and g["replay_mtime"] < g["mtime"] - 1:
                ok = False
                L.append(f"    ! {game}/replay.html is older than its online.json "
                         f"({(g['mtime'] - g['replay_mtime']) / 60:.0f} min behind)")
        newest = max((g["mtime"] for g in run["games"].values()), default=None)
        if run["replay_mtime"] is None:
            L.append("    ! no combined replay.html")
        elif newest and run["replay_mtime"] < newest - 1:
            ok = False
            L.append(f"    ! replay.html is {(newest - run['replay_mtime']) / 60:.0f} "
                     "min behind the newest online.json (re-run watch_plan_replay.py)")
        else:
            L.append("    replay.html is level with the results")
    for agent in agents:
        L.append(f"  {agent['label']}  ({agent['root']})")
        if agent["pending"]:
            ok = False
            L.append(f"    ! {len(agent['pending'])} games have no rows yet: "
                     + ", ".join(agent["pending"]))
        if agent["only_mine"] or agent["only_ref"]:
            ok = False
            L.append(f"    ! problem set differs from {agent['reference']}: "
                     f"{agent['shared']} shared, {len(agent['only_mine'])} agent-only, "
                     f"{len(agent['only_ref'])} {agent['reference']}-only "
                     f"({', '.join((agent['only_mine'] + agent['only_ref'])[:3])})")
        if agent["cap_diff"]:
            ok = False
            L.append(f"    ! action budgets differ from {agent['reference']} on "
                     f"{len(agent['cap_diff'])} problems: "
                     f"{', '.join(agent['cap_diff'][:3])}")
        for game, g in sorted(agent["games"].items()):
            if not g["complete"]:
                ok = False
                L.append(f"    ! {game}: {len(g['rows'])}/{g['expected']} problems "
                         "recorded -- partial scores")
            # a session that died is not a miss; scoring it as one would understate the
            # arm, so it is called out rather than folded into the pass rate
            bad = {r["task_uid"]: r[AGENT_ARM].get("status") for r in g["rows"]
                   if isinstance(r.get(AGENT_ARM), dict)
                   and r[AGENT_ARM].get("status") not in (None, "done")}
            if bad:
                ok = False
                L.append(f"    ! {game}/agent: {len(bad)} sessions did not finish "
                         f"({sorted(set(bad.values()))}): {', '.join(sorted(bad)[:3])}")
    if ok:
        L.append("  no problems found")
    return L


def console(runs: list[dict], agents: list[dict] = ()) -> str:
    L = ["PLANNING v2 ONLINE -- results behind replay.html", "=" * 47, ""]
    for run in runs:
        cfg = run["config"]
        cost = sum(g["cost"] for g in run["games"].values())
        L.append(f"{run['label']}: {run['root']}")
        L.append(f"  planner {cfg.get('model')} | goals {cfg.get('goal_presentation')} "
                 f"| caps {run['cap_mode']} | artifacts {cfg.get('artifact_root')}")
        L.append(f"  {len(run['games'])}/{len(GAME_ORDER)} games landed | "
                 f"{sum(len(g['rows']) for g in run['games'].values())} problems | "
                 f"${cost:.2f}")
        for arm in LLM_ARMS:
            t = run_totals(run, arm)
            L.append(f"    {arm:>5}: pass@1 macro {fmt(t['macro'])} "
                     f"(micro {fmt(t['micro'])}, n={t['n']})   "
                     f"floor-adjusted macro {fmt(t['macro_adj'])}")
        L.append("")
    for agent in agents:
        t = run_totals(agent, AGENT_ARM)
        L.append(f"{agent['label']}: {agent['root']}")
        L.append(f"  {len(agent['games'])}/{len(GAME_ORDER)} games landed | "
                 f"{sum(len(g['rows']) for g in agent['games'].values())} problems")
        L.append(f"    {AGENT_ARM:>5}: pass@1 macro {fmt(t['macro'])} "
                 f"(micro {fmt(t['micro'])}, n={t['n']})")
        L.append("")
    common = [g for g in PAPER_ORDER if all(scored(r, g) for r in runs)]
    if len(runs) > 1 and common:
        L += [f"COMPLETE-CASE MEAN over the {len(common)} games every run has finished "
              "(what the paper's Mean row prints)", "-" * 70]
        for run in runs:
            L.append(f"  {run['label']:>12}:  raw "
                     f"{fmt(run_totals(run, 'raw', common)['macro'])}   lmwm "
                     f"{fmt(run_totals(run, 'lmwm', common)['macro'])}")
        L += ["  " + ", ".join(display_name(g) for g in common), ""]
    L += ["PER GAME (pass@1; * = partial)", "-" * 30]
    L += per_game_table(runs)
    L += tiers(runs)
    L += paired(runs)
    L += checks(runs, agents)
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------- tex
def tex(v, d=2):
    return r"\textemdash" if v is None else f"{v:.{d}f}"


def scored(run: dict, game: str) -> dict | None:
    """A game's cell, or None while it is still playing -- a partial game is not a score."""
    g = run["games"].get(game)
    return g if g and g["complete"] else None


def tex_results(runs: list[dict], raw_from: dict, agents: list[dict] = ()) -> str:
    """tab:autumn-results -- one row per game, Raw from the single designated run.

    A game still playing prints an em-dash, so the table is publishable mid-run and
    tightens as games land. The Mean row is a complete-case macro average: only games
    that have a number in every column, so no column is averaged over a different set --
    which is why an unfinished agent run narrows the mean for every column, rather than
    quietly giving the Agent column an easier set of games than the others."""
    complete = [g for g in PAPER_ORDER
                if scored(raw_from, g) and all(scored(r, g) for r in runs)
                and all(scored(a, g) for a in agents)]
    lead = [a for a in agents if not a.get("trailing")]
    trail = [a for a in agents if a.get("trailing")]
    L = [r"    \toprule",
         r"    Environment & Raw & " + " & ".join([a["label"] for a in lead]
                                                 + [r["label"] for r in runs]
                                                 + [a["label"] for a in trail])
         + r" \\",
         r"    \midrule"]

    def agent_cell(a, game):
        ga = scored(a, game)
        return tex(ga["arms"][AGENT_ARM]["pass1"]) if ga else r"\textemdash"

    for game in PAPER_ORDER:
        g = scored(raw_from, game)
        cells = [tex(g["arms"]["raw"]["pass1"]) if g else r"\textemdash"]
        cells += [agent_cell(a, game) for a in lead]
        for r in runs:
            gr = scored(r, game)
            cells.append(tex(gr["arms"][r["arm"]]["pass1"]) if gr else r"\textemdash")
        cells += [agent_cell(a, game) for a in trail]
        L.append(f"    {display_name(game):<14} & " + " & ".join(cells) + r" \\")
    L.append(r"    \midrule")
    means = [tex(run_totals(raw_from, "raw", complete)["macro"])]
    means += [tex(run_totals(a, AGENT_ARM, complete)["macro"]) for a in lead]
    means += [tex(run_totals(r, r["arm"], complete)["macro"]) for r in runs]
    means += [tex(run_totals(a, AGENT_ARM, complete)["macro"]) for a in trail]
    L.append(r"    \textbf{Mean} & " + " & ".join(means) + r" \\")
    L.append(r"    \bottomrule")
    L.append(f"    % complete-case mean over {len(complete)} games: "
             + ", ".join(complete))
    return "\n".join(L)


def tex_protocol(runs: list[dict]) -> str:
    """tab:autumn-protocol -- the audit trail the headline table cannot carry.

    The runs do not share a problem set or an action budget, so per-run $n$, budget and
    random floor have to be reported before the headline numbers can be read as a
    comparison."""
    L = [r"    \toprule",
         r"    & " + " & ".join(r"\multicolumn{5}{c}{" + r["label"] + "}" for r in runs)
         + r" \\",
         "    " + " ".join(r"\cmidrule(lr){%d-%d}" % (2 + 5 * i, 6 + 5 * i)
                          for i in range(len(runs))),
         r"    Game & " + " & ".join(r"$n$ & Budget & Rand & Raw & "
                                     + ARM_DISPLAY.get(r["arm"], r["arm"])
                                     for r in runs) + r" \\",
         r"    \midrule"]
    for game in PAPER_ORDER:
        cells = []
        for r in runs:
            g = r["games"].get(game)
            if g is None:
                cells += [r"\textemdash"] * 5
                continue
            partial = "" if g["complete"] else r"\textsuperscript{*}"
            cells += [str(len(g["rows"])) + partial, g["cap"].replace("-", "--"),
                      tex(g["floor"]),
                      tex(g["arms"]["raw"]["pass1"]),
                      tex(g["arms"][r["arm"]]["pass1"])]
        L.append(f"    {display_name(game):<14} & " + " & ".join(cells) + r" \\")
    L.append(r"    \bottomrule")
    return "\n".join(L)


# label -> (body builder, column spec for the number of runs)
BLOCKS = {
    "tab:autumn-results": (tex_results,
                           lambda n, k: "@{}l" + "c" * (1 + k + n) + "@{}"),
    "tab:autumn-protocol": (tex_protocol, lambda n, k: "@{}l" + "r" * (5 * n) + "@{}"),
}


def comment_prefix(line: str, marker: str) -> str:
    """The `% ` a wholesale-commented-out table carries in front of every one of its lines.

    A hidden table is still a table: it keeps its AUTO markers, so it keeps being
    regenerated -- under the same prefix -- rather than silently going stale against the
    run it claims to report. Without this, a table someone commented out in Overleaf
    would either be skipped or abort the write, since its `\\begin{tabular}` is no longer
    LaTeX the block regex can rewrite."""
    head = line[:line.index(marker)]
    return "% " if head.lstrip().startswith("%") else ""


def write_tex(path: Path, runs: list[dict], raw_from: dict,
              agents: list[dict] = ()) -> list[str]:
    """Replace each `% BEGIN AUTO <label>` .. `% END AUTO <label>` block in place.

    The `\\begin{tabular}` line that opens the block is rewritten too, so a table stays
    compilable when the number of runs changes the number of columns. A block that has
    been commented out is rewritten commented out, prefix intact."""
    lines = path.read_text().split("\n")
    changed = []
    for label, (build, colspec) in BLOCKS.items():
        begin, end = f"% BEGIN AUTO {label}", f"% END AUTO {label}"
        bi = next((i for i, l in enumerate(lines) if l.rstrip().endswith(begin)), None)
        if bi is None:
            continue
        ei = next((i for i, l in enumerate(lines[bi:], bi) if l.rstrip().endswith(end)),
                  None)
        if ei is None:
            raise SystemExit(f"{begin} in {path} has no matching {end}")
        if bi == 0 or r"\begin{tabular}{" not in lines[bi - 1]:
            raise SystemExit(f"{begin} in {path} does not directly follow a tabular")
        pfx = comment_prefix(lines[bi], begin)
        # the column spec itself contains `}` (`@{}lccc@{}`), so keep only what follows
        # the line's LAST brace -- that is the tabular's own closing one
        head, _, tail = lines[bi - 1].partition(r"\begin{tabular}{")
        opener = (head + r"\begin{tabular}{" + colspec(len(runs), len(agents))
                  + tail[tail.rindex("}"):])
        body = (build(runs, raw_from, agents) if label == "tab:autumn-results"
                else build(runs))
        block = [pfx + l for l in [f"    {begin}", *body.split("\n"), f"    {end}"]]
        if [opener] + block != lines[bi - 1:ei + 1]:
            changed.append(label)
        lines[bi - 1:ei + 1] = [opener] + block
    path.write_text("\n".join(lines))
    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", metavar="LABEL=PATH[@ARM]",
                    help="a run dir and the column name it gets (repeatable); "
                         "defaults to the two NL runs. Append @ARM to report an arm "
                         "other than lmwm -- the in-context baseline is a third arm of "
                         "an existing run, not a run of its own, e.g. "
                         "'ICL=logs/.../nl_icl@icl'")
    ap.add_argument("--raw-from", metavar="LABEL",
                    help="which run supplies the shared Raw column "
                         "(default: the first --run)")
    ap.add_argument("--tex", action="store_true", help="print the LaTeX table body")
    ap.add_argument("--write-tex", nargs="?", const="paper/main.tex", metavar="PATH",
                    help="rewrite the AUTO block of that .tex in place")
    ap.add_argument("--agent", metavar="PATH", default=DEFAULT_AGENT[1],
                    help="the agent arm's run dir; its rows.jsonl fills the Agent "
                         f"column (default {DEFAULT_AGENT[1]})")
    ap.add_argument("--agent-wm", metavar="PATH", default=DEFAULT_AGENT_WM[1],
                    help="the same harness with the NLWM world model in each workspace; "
                         f"fills the {DEFAULT_AGENT_WM[0]} column "
                         f"(default {DEFAULT_AGENT_WM[1]})")
    ap.add_argument("--no-agent", action="store_true",
                    help="leave both Agent columns em-dashed")
    ap.add_argument("--check", action="store_true",
                    help="only the staleness/completeness checks")
    ap.add_argument("--json", metavar="PATH", help="dump the aggregates as JSON")
    a = ap.parse_args()

    spec = [tuple(s.split("=", 1)) for s in a.run] if a.run else DEFAULT_RUNS
    runs = []
    for label, path in spec:
        path, _, arm = path.partition("@")
        arm = arm or DEFAULT_COLUMN_ARM
        if arm not in LLM_ARMS:
            raise SystemExit(f"unknown arm {arm!r}; choose from {', '.join(LLM_ARMS)}")
        root = Path(path) if Path(path).is_absolute() else REPO / path
        if not root.is_dir():
            raise SystemExit(f"no such run dir: {root}")
        runs.append(load_run(label, root, arm))
    raw_from = next((r for r in runs if r["label"] == a.raw_from), runs[0])

    agents = []
    if not a.no_agent:
        for (label, _d, trailing), path in ((DEFAULT_AGENT, a.agent),
                                            (DEFAULT_AGENT_WM, a.agent_wm)):
            if not path:
                continue
            root = Path(path) if Path(path).is_absolute() else REPO / path
            if not root.is_dir() or not (list(root.glob("rows.w*.jsonl"))
                                         + [f for f in [root / "rows.jsonl"]
                                            if f.is_file()]):
                # A column whose run has not started, or has not recorded its first
                # problem, is not an error: the table prints em-dashes for it and
                # tightens as the run lands.
                print(f"note: nothing recorded at {root} yet; {label} stays em-dashed")
                continue
            agents.append(load_agent_run(label, root, raw_from, trailing=trailing))

    if a.check:
        print("\n".join(checks(runs, agents)).lstrip("\n"))
    else:
        print(console(runs, agents), end="")
    if a.tex:
        print("\n% " + "tab:autumn-results\n" + tex_results(runs, raw_from, agents))
        print("\n% " + "tab:autumn-protocol\n" + tex_protocol(runs))
    if a.write_tex:
        p = Path(a.write_tex) if Path(a.write_tex).is_absolute() else REPO / a.write_tex
        moved = write_tex(p, runs, raw_from, agents)
        print(f"\n{('updated ' + ', '.join(moved)) if moved else 'unchanged'} in {p}"
              f"  (Raw column from {raw_from['label']})")
    if a.json:
        out = {r["label"]: {
            "root": str(r["root"]), "config": r["config"],
            "totals": {arm: run_totals(r, arm) for arm in LLM_ARMS},
            "games": {g: {"n": len(v["rows"]), "cap": v["cap"], "floor": v["floor"],
                          "complete": v["complete"], "cost": v["cost"],
                          "arms": v["arms"]} for g, v in r["games"].items()},
            "pending": r["pending"]} for r in runs}
        out["_generated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        Path(a.json).write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
