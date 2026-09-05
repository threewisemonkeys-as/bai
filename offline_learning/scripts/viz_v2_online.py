#!/usr/bin/env python3
"""Render an ONLINE planning-v2 run (eval_curated_online.py output) as the per-round
plan-vs-executed filmstrip. Reuses the v1 HTML shell from viz_nl_online.py; this module
only remaps the v2 result schema (task_uid / nl_goal / reached_at / reached_goal, per-arm
attempt dicts) into the DATA shape that shell renders.

    uv run python offline_learning/scripts/viz_v2_online.py \
        --eval logs/2026-08-30/planning_v2_online_ds/f5w3n/online.json \
        --problems logs/2026-08-29/planning_v2/problems.json \
        --out logs/2026-08-30/planning_v2_online_ds/f5w3n/viz.html
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(REPO), str(REPO / "offline_learning"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from offline_learning.human_replay import GAMES  # noqa: E402
from eval_coverage_online import WARM_TMPL  # noqa: E402
from eval_curated_online import build_resources, prepare  # noqa: E402
from eval_curated_plan import (  # noqa: E402
    build_prompt, icl_config, load_eval_problems, select_goal_presentation,
)
from viz_nl_goals import CSS, _CHARS, pack  # noqa: E402
from viz_nl_online import HTML  # noqa: E402

LLM_ARMS = ("raw", "lmwm", "icl")


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def _parsed_response(rd: dict) -> str:
    """Recreate the structured portion retained from the model response.

    Historical online checkpoints kept the parsed reasoning and normalized plan, not
    the original response bytes. Prefer an exact response when a newer run has one.
    """
    if rd.get("response") is not None:
        return str(rd["response"])
    reasoning = rd.get("reasoning") or rd.get("thinking") or ""
    plan = rd.get("plan")
    parts = [f"<reasoning>{reasoning}</reasoning>"]
    if plan:
        parts.append("<plan>\n" + "\n".join(plan) + "\n</plan>")
    else:
        parts.append(
            "<plan unavailable>\n"
            f"stored parse error: {rd.get('plan_error') or 'unknown'}"
        )
    return "\n".join(parts)


def _step_io(p: dict, arm: str, att: dict, resources: dict, config: dict) -> list[dict]:
    """Reconstruct the prompt visible at every historical receding-horizon step."""
    cur_grid = p["start_grid"]
    cur_z = p["_z_t"]
    hist_raw: list[tuple[str, str]] = []
    hist_z: list[tuple[str, str]] = []
    carry: list[str] = []
    context_k = int(config.get("context_k") or 9)
    out = []
    for rd in att.get("rounds", []):
        exact_prompt = rd.get("prompt")
        prompt = ""
        if exact_prompt is None:              # only pay to rebuild what was not stored
            prompt = build_prompt(
                p, arm, cur_grid, start_features=cur_z,
                goal_features=p["_z_goal"], beliefs=resources[p["game"]]["beliefs"],
                hist_raw=hist_raw[-context_k:], hist_z=hist_z[-context_k:],
                cap=rd["remaining"], icl_block=resources[p["game"]]["icl"],
            )
            if config.get("warm_start", True) and carry:
                warm = WARM_TMPL.format(cand="\n".join(carry), remaining=rd["remaining"])
                prompt = prompt.replace("\nRespond as:\n", f"\n{warm}\nRespond as:\n", 1)
        exact_response = rd.get("response")
        out.append({
            "prompt": str(exact_prompt) if exact_prompt is not None else prompt,
            "response": _parsed_response(rd),
            "promptNote": (
                "Exact prompt retained by the evaluator."
                if exact_prompt is not None else
                "Reconstructed from the saved state/history and evaluator template. "
                "A corrective retry suffix, if one was used, was not retained."
            ),
            "responseNote": (
                "Exact raw response retained by the evaluator."
                if exact_response is not None else
                "Reconstructed from the saved parsed reasoning and normalized plan; "
                "original whitespace and any extra text were not retained."
            ),
        })
        plan = rd.get("plan") or []
        carry = plan[1:]
        action = rd.get("executed")
        if action and rd.get("grid_after"):
            hist_raw.append((cur_grid, action))
            if arm == "lmwm":
                hist_z.append((cur_z, action))
            cur_grid = rd["grid_after"]
            if arm == "lmwm":
                cur_z = rd.get("z_after")
                if cur_z is None:
                    cur_z = resources[p["game"]]["perceive"](cur_grid)[0]
    return out


def build(ev: dict, off_idx: dict, curated: dict, resources: dict) -> dict:
    pal: dict[str, int] = {}
    problems = []
    arms = [a for a in LLM_ARMS
            if any(isinstance(r.get(a), dict) and r[a].get("attempts")
                   for r in ev["rows"])]
    for r in ev["rows"]:
        cur = curated[r["task_uid"]]
        rollouts = []
        for arm in arms:
            cell = r.get(arm)
            if not isinstance(cell, dict) or not cell.get("attempts"):
                continue
            for i, att in enumerate(cell["attempts"]):
                rounds = []
                carry: list[str] = []
                step_io = _step_io(cur, arm, att, resources, ev.get("config", {}))
                for ri, rd in enumerate(att.get("rounds", [])):
                    plan = rd.get("plan") or []
                    followed = bool(carry) and bool(plan) and carry[0] == rd.get("executed")
                    rounds.append({
                        "n": rd["n"], "action": rd.get("executed"),
                        "why": (rd.get("reasoning") or rd.get("thinking") or "").strip(),
                        "tail": plan[1:], "remaining": rd["remaining"],
                        "error": rd.get("plan_error"),
                        "satisfied": bool(rd.get("reached_goal")),
                        "followed": followed, "first": not carry,
                        "frame": pack(json.loads(rd["grid_after"]), pal)
                                 if rd.get("grid_after") else None,
                        **step_io[ri],
                    })
                    carry = plan[1:]
                rollouts.append({"n": i + 1, "arm": arm,
                                 "success": bool(att.get("success")),
                                 "sat": att.get("reached_at"),
                                 "used": att.get("actions_used", 0),
                                 "failed": att.get("failed_reason"),
                                 "frame_hit": False,   # v2: the goal is the explicitly selected target
                                 "rounds": rounds})
        if not rollouts:
            continue
        rand = r.get("random_success_cap50")
        if rand is None:
            rand = r.get("random_success")
        problems.append({
            "game": r["game"], "human": GAMES[r["game"]][1], "id": r["id"],
            "tier": r.get("tier"), "nl": r.get("nl_goal") or "",
            "objective": r.get("objective") or "", "h": r.get("h"),
            "seed": r.get("seed"), "rand": rand,
            "start": pack(cur["start"], pal),
            "arms": {a: {"on1": r[a]["pass_rate"], "on5": bool(r[a]["pass_any"]),
                         "used": _mean([t.get("actions_used") for t in r[a]["attempts"]]),
                         "mode": r.get("goal_presentation", "?"),
                         "frame": r[a]["pass_rate"],
                         "off1": off_idx.get((r["task_uid"], a))}
                     for a in arms
                     if isinstance(r.get(a), dict) and r[a].get("attempts")},
            "rollouts": rollouts,
        })
    return {"palette": {i: CSS.get(n, n) for n, i in pal.items()},
            "problems": problems, "chars": _CHARS, "armlist": arms,
            "model": ev.get("config", {}).get("model", "?"),
            "cap": ev.get("config", {}).get("max_actions")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--offline", default="")
    ap.add_argument("--problems", default="logs/2026-08-29/planning_v2/problems.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ev = json.loads(Path(a.eval).read_text())
    _meta, selected = load_eval_problems(a.problems)
    goal_presentation = ev.get("config", {}).get("goal_presentation")
    if goal_presentation not in {"frame", "nl"}:
        raise ValueError("evaluation JSON must specify goal_presentation=frame or nl")
    success_mode = ev.get("config", {}).get("success_mode", "any")
    if success_mode == "online-any-step":
        success_mode = "any"
    selected = select_goal_presentation(selected, goal_presentation, success_mode)
    wanted = {r["task_uid"] for r in ev["rows"]}
    selected = [p for p in selected if p["task_uid"] in wanted]
    arms = [a for a in LLM_ARMS if any(
        isinstance(r.get(a), dict) and r[a].get("attempts") for r in ev["rows"]
    )]
    by_game = defaultdict(list)
    for p in selected:
        by_game[p["game"]].append(p)
    resources = {}
    artifact_root = Path(ev.get("config", {}).get("artifact_root") or "")
    icl_cfg = ev.get("config", {}).get("icl")
    if icl_cfg:                               # rebuild with the run's own settings
        icl_cfg = {**icl_cfg, "data_root": Path(icl_cfg["data_root"]),
                   "context_k": int(icl_cfg["context_k"])}
    for game, ps in by_game.items():
        resources[game], _skipped = build_resources(game, artifact_root, arms, icl_cfg)
        prepare(ps, resources[game]["perceive"])
    curated = {p["task_uid"]: p for p in selected}
    off_idx = {}
    if a.offline and Path(a.offline).exists():
        off = json.loads(Path(a.offline).read_text())
        off_idx = {(r["task_uid"], arm): r[arm]["pass_rate"]
                   for r in off["rows"] for arm in LLM_ARMS
                   if isinstance(r.get(arm), dict) and r[arm].get("pass_rate") is not None}

    data = build(ev, off_idx, curated, resources)
    html = HTML.replace("<title>NL-goal planning, online</title>",
                        "<title>Planning v2, online</title>")
    html = html.replace("<h1>NL-goal planning: online (receding horizon)</h1>",
                        "<h1>Planning v2: online (receding horizon)</h1>")
    # summary column "frame@1" is meaningless in v2 (the selected goal may be the frame);
    # show the goal mode instead
    html = html.replace('"frame@1","rand@50"', '"goal","rand@50"')
    html = html.replace('tr.appendChild(el("td", null, d.frame.toFixed(2)));',
                        'tr.appendChild(el("td", "flat", d.mode));')
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace("/*DATA*/{}", json.dumps(data, separators=(",", ":"))))
    nr = sum(len(ro["rounds"]) for p in data["problems"] for ro in p["rollouts"])
    print(f"wrote {out}  ({len(data['problems'])} problems, "
          f"{sum(len(p['rollouts']) for p in data['problems'])} rollouts, {nr} rounds, "
          f"{out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
