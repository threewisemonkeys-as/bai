#!/usr/bin/env python3
"""OFFLINE (open-loop) planning eval with NATURAL-LANGUAGE goals.

Same protocol as `eval_curated_plan.py` -- same problems, same start states, same 50-action
budget, same 5 attempts, same parser -- with exactly two things changed:

  * THE GOAL IS A SENTENCE.  Where the frame-goal prompt pasted the target grid (or its
    perception features), this one states the goal in English and shows no target at all.
    That removes a channel the `lmwm` arm was quietly relying on: it used to receive the goal
    THROUGH ITS OWN PERCEPTION MODULE, so goal and state were expressed in the same
    vocabulary.  Here it must ground "coins" and "infected" into whatever its beliefs and
    features happen to call those things.
  * SCORING IS ANY-STEP, BY PREDICATE.  Success is `nl_goals.first_satisfied(...) is not
    None`: the checker holds after some prefix of the executed plan.  `frame_hit` is recorded
    alongside -- whether the run also reproduced the curated exact frame -- so the gap between
    the two says how much the relaxed goal is buying.

The `<reasoning>` block asks for ONE NUMBERED LINE PER ACTION so the rationale can be lined
up against the trajectory in `viz_nl_goals.py`.  That is a deliberate deviation from the
frame-goal prompt, which asked for free-form reasoning; it changes what the planner is asked
to produce, so numbers here are not strictly comparable to a run made without it.

Arms: `lmwm` only by default.  `wc` is not available at all: `prt.plan_search` needs a goal
GRID both as its termination test and as its beam heuristic, and a sentence gives it neither.
`raw` can be added with `--arms raw,lmwm` as a no-world-model control.

    uv run python offline_learning/scripts/eval_curated_nl.py \
        --out logs/2026-08-19/nl_pilot/eval/offline
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OFF = HERE.parent
REPO = OFF.parent
for _p in (str(OFF), str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from validate import run_perceive  # noqa: E402
from offline_learning.coverage_plan import exec_plan  # noqa: E402
from offline_learning.human_replay import GAMES as HGAMES  # noqa: E402
from offline_learning.nl_goals import GOALS, NLGoal, first_satisfied  # noqa: E402

import eval_coverage_plan as ecp  # noqa: E402
from eval_coverage_plan import (  # noqa: E402
    DEFAULT_KNOWLEDGE, feat_transcript, llm_call, parse_plan, raw_transcript,
    resolve_llm_config,
)
from eval_curated_plan import ATTEMPTS, PLAN_CAP, TIERS, gstr  # noqa: E402
from eval_coverage_plan import thinking_record  # noqa: E402

# The two prompts differ from PLAN_RAW_TMPL / PLAN_WIN_TMPL only in the GOAL block and in the
# sentence stating what "achieved" means -- any-step, because the checker is any-step and an
# agent that is not told so would waste actions holding a state it had already reached.
_GOAL_BLOCK = """=== GOAL (in words) ===
{goal}
=== END GOAL ===

Plan a sequence of AT MOST {cap} action(s) so that, starting from the CURRENT state and
executing your actions in order, the GOAL above is true at some point during your plan (the
last action may be the one that achieves it, or it may happen earlier). The environment's
passive dynamics keep running on every step (including noop), so timing can matter. Every
action must be fully specified: one of up, down, left, right, noop, or click ROW COL with
0-indexed integers.

Respond as:
<reasoning>
First, one or two sentences on what the goal means in terms of what you can see.
Then ONE NUMBERED LINE PER ACTION, in the same order as your plan, formatted exactly as
  N. <action> - why this action, and what you expect the state to look like after it
</reasoning>
<plan>
one action per line, at most {cap} line(s) - ACTIONS ONLY here, no numbering and no
explanation; the reasoning for each one belongs in the block above
</plan>"""

PLAN_NL_RAW_TMPL = """You control a grid environment and must achieve a goal stated in words.

=== DEFAULT KNOWLEDGE (always-true facts about this environment) ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

Below is a trajectory of consecutive RAW GRIDS in canonical JSON, ending at the
CURRENT grid. The action between each prior pair is shown. Use the whole history
to infer the dynamics (passive drift, momentum, selection, delayed effects).

{transcript}

""" + _GOAL_BLOCK

PLAN_NL_WIN_TMPL = """You control a grid environment and must achieve a goal stated in words.

=== DEFAULT KNOWLEDGE (always-true facts about this environment) ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== WORLD KNOWLEDGE (general facts; may be empty) ===
{beliefs}
=== END WORLD KNOWLEDGE ===

Below is a trajectory of consecutive states (summarized as features by a
perception module) ending at the CURRENT state, with the action taken between
each pair. Use the whole history to infer the dynamics (passive drift, momentum,
selection, delayed effects).

{transcript}

""" + _GOAL_BLOCK


_REASON_LINE = re.compile(r"^\s*(\d+)\s*[.)]\s*(.+)$")


def parse_reasoning(text: str) -> tuple[str, dict[int, str]]:
    """The <reasoning> block, plus its numbered lines keyed by action index (1-based).

    The mapping is by the model's OWN numbering, not by line order: a model that skips a
    number or writes a preamble line would otherwise have every subsequent rationale
    attached to the wrong action, which is worse than showing none."""
    body = ecp._parse_tag(text, "reasoning") or ""
    lines: dict[int, str] = {}
    for ln in body.splitlines():
        m = _REASON_LINE.match(ln.strip())
        if m:
            lines.setdefault(int(m.group(1)), m.group(2).strip())
    return body.strip(), lines


def _grids(start: str, out: list[str | None]) -> list:
    """Frames as the checker wants them: start first, truncated at termination."""
    grids = [tuple(tuple(r) for r in json.loads(start))]
    for g in out:
        if g is None:
            break
        grids.append(tuple(tuple(r) for r in json.loads(g)))
    return grids


def score(goal: NLGoal, program: str, seed: int, plan: list[str], start: str,
          goal_frame: str) -> dict:
    out = exec_plan(program, seed, [], plan)
    grids = _grids(start, out)
    acts = plan[:len(grids) - 1]
    k = first_satisfied(goal, grids, acts)
    hit = next((j + 1 for j, g in enumerate(out) if g == goal_frame), None)
    return {"success": k is not None, "satisfied_at": k, "frame_hit": hit is not None,
            "frame_hit_at": hit, "executed": len(acts)}


async def eval_goal(goal: NLGoal, row: dict, arms: list[str], sem, llm,
                    artifact_root: Path, attempts: int) -> dict:
    rex = artifact_root / "rexpure" / f"{goal.game}_s1"
    perc_code = (rex / "best_perception_rexpure_seed1.py").read_text()
    beliefs = (rex / "best_beliefs_rexpure_seed1.txt").read_text()

    start, goal_frame = gstr(row["start"]), gstr(row["goal"])
    dims = (len(row["start"]), len(row["start"][0]))
    z_t = run_perceive(perc_code, start)[0]

    prompts = {
        "raw": PLAN_NL_RAW_TMPL.format(
            cap=PLAN_CAP, default_knowledge=DEFAULT_KNOWLEDGE,
            transcript=raw_transcript([], start), goal=goal.nl),
        "lmwm": PLAN_NL_WIN_TMPL.format(
            cap=PLAN_CAP, default_knowledge=DEFAULT_KNOWLEDGE,
            beliefs=beliefs.strip() or "(empty)",
            transcript=feat_transcript([], z_t), goal=goal.nl),
    }

    got = await asyncio.gather(*(llm_call(prompts[a], sem, llm)
                                 for a in arms for _ in range(attempts)))
    calls = {a: got[i * attempts:(i + 1) * attempts] for i, a in enumerate(arms)}

    out = {"game": goal.game, "id": goal.pid, "tier": goal.tier, "nl": goal.nl,
           "seed": goal.seed, "h": row["h"], "n_decisions": row["n_decisions"],
           "objective": row["objective"], "cost": 0.0}
    for arm in arms:
        tries = []
        for text, think, c, errs in calls[arm]:
            out["cost"] += c
            plan, perr = parse_plan(text, dims)
            block, rlines = parse_reasoning(text)
            if plan is not None and len(plan) > PLAN_CAP:
                plan, perr = None, f"budget-exceeded:{len(plan)}>{PLAN_CAP}"
            t = {"success": False, "satisfied_at": None, "frame_hit": False,
                 "frame_hit_at": None, "executed": 0}
            if plan is not None:
                t = score(goal, row["program"], goal.seed, plan, start, goal_frame)
            t |= {"plan_len": len(plan) if plan else None, "plan_error": perr,
                  "retry_errors": errs, "plan": plan, "reasoning": block,
                  "reason_lines": {str(k): v for k, v in sorted(rlines.items())},
                  **thinking_record(think)}
            tries.append(t)
        out[arm] = {"attempts": tries,
                    "pass_rate": sum(t["success"] for t in tries) / len(tries),
                    "pass_any": any(t["success"] for t in tries),
                    "frame_rate": sum(t["frame_hit"] for t in tries) / len(tries)}
    return out


def load_floors(path: str) -> dict[str, float]:
    """Per-problem random-plan floor at the eval budget, from the validator.  Every score in
    the report is read against it: a set-valued goal scored any-step over 50 actions is not
    starting from zero (dq8gc's is several percent), and a table without the column invites
    exactly that mistake."""
    try:
        return {r["pid"]: r["N5"]["floor_at_cap"]
                for r in json.loads(Path(path).read_text())}
    except (OSError, KeyError, json.JSONDecodeError):
        return {}


def report(rows: list[dict], arms: list[str], llm, elapsed: float, cost: float,
           floors: dict[str, float], attempts: int) -> str:
    L = ["# Curated planning eval - NATURAL-LANGUAGE goals, OFFLINE (open-loop)", "",
         f"Planner: {llm.label} | plan cap {PLAN_CAP} | {attempts} attempt(s) per arm | "
         f"no history, CURRENT state only | goal stated in words, no target frame shown | "
         f"ANY-STEP predicate scoring | {len(rows)} problems | {elapsed / 60:.0f} min | "
         f"${cost:.2f}", "",
         "`pass@1` is the mean over attempts, `pass@n` is any-of-n. `frame@1` is how often "
         "the same run ALSO reproduced the curated exact goal frame -- the gap between the "
         "two columns is what the relaxed goal admits that the frame goal did not.", "",
         "| game | tier | id | h | " + " | ".join(
             f"{a} @1 | {a} @n | {a} frame@1" for a in arms) + " | rand@50 | goal |",
         "|---|---|---|--:|" + "--:|" * (3 * len(arms) + 1) + "---|"]
    for r in rows:
        cellsm = " | ".join(f"{r[a]['pass_rate']:.2f} | {r[a]['pass_any']:.2f} | "
                            f"{r[a]['frame_rate']:.2f}" for a in arms)
        fl = floors.get(r["id"])
        L.append(f"| {r['game']} | {r['tier']} | `{r['id']}` | {r['h']} | {cellsm} | "
                 f"{'--' if fl is None else f'{fl:.3f}'} | {r['nl']} |")

    L += ["", "## Per tier", "", "| tier | n | " + " | ".join(
        f"{a} @1 | {a} @n" for a in arms) + " |",
        "|---|--:|" + "--:|" * (2 * len(arms))]
    for tier in TIERS:
        s = [r for r in rows if r["tier"] == tier]
        if not s:
            continue
        L.append(f"| {tier} | {len(s)} | " + " | ".join(
            f"{sum(r[a]['pass_rate'] for r in s) / len(s):.2f} | "
            f"{sum(r[a]['pass_any'] for r in s) / len(s):.2f}" for a in arms) + " |")

    bad = defaultdict(int)
    for r in rows:
        for a in arms:
            for t in r[a]["attempts"]:
                if t["plan_error"]:
                    bad[f"{a}:{t['plan_error'].split(':')[0]}"] += 1
    if bad:
        L += ["", "## Unusable responses", ""] + [f"- {k}: {v}" for k, v in sorted(bad.items())]
    return "\n".join(L) + "\n"


async def main_async(a):
    curated = {r["id"]: r for r in json.loads(Path(a.problems).read_text())}
    goals = [g for g in GOALS if not a.pid or g.pid in a.pid]
    arms = [x for x in a.arms.split(",") if x]
    llm = resolve_llm_config(a)
    sem = asyncio.Semaphore(a.concurrency)

    t0 = time.time()
    rows = await asyncio.gather(*(eval_goal(g, curated[g.pid], arms, sem, llm,
                                            Path(a.artifact_root), a.attempts)
                                  for g in goals))
    rows = list(rows)
    cost = sum(r["cost"] for r in rows)
    md = report(rows, arms, llm, time.time() - t0, cost, load_floors(a.floors),
                a.attempts)
    print(md)
    stem = Path(a.out)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(
        {"config": {"model": llm.model, "backend": llm.backend, "plan_cap": PLAN_CAP,
                    "attempts": a.attempts, "arms": arms,
                    "scoring": "any-step-predicate"},
         "rows": rows, "cost": cost}, indent=1))
    stem.with_suffix(".md").write_text(md)
    print(f"wrote {stem}.json / {stem}.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="logs/2026-08-18/curated/problems.json")
    ap.add_argument("--out", default="logs/2026-08-19/nl_pilot/eval/offline")
    ap.add_argument("--pid", action="append", help="restrict to these problem ids")
    ap.add_argument("--arms", default="lmwm", help="comma-separated: lmwm[,raw]")
    ap.add_argument("--attempts", type=int, default=ATTEMPTS)
    ap.add_argument("--floors", default="logs/2026-08-19/nl_pilot/validation.json",
                    help="validator output, for the random-floor column")
    ap.add_argument("--artifact-root", default=str(ecp.ARTIFACT_ROOT))
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--llm-backend", choices=("claude", "openrouter"), default="openrouter")
    ap.add_argument("--llm-url", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--provider-order", default=",".join(ecp.DEFAULT_PROVIDER_ORDER))
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
