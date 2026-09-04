#!/usr/bin/env python3
"""OFFLINE (open-loop) planning eval for curated planning-problems v1 and v2.

V2 states are replay addresses: reset with ``seed``, execute ``prefix``, then execute the
candidate plan. Exact-frame goals are shown as frames; NL goals are shown only as
``nl_goal`` and scored by registered Python programs. Each checker's success mode decides
whether the goal must hold after the final action or may be reached at any step.

Config is inherited from `eval_coverage_plan` by IMPORT, not by copy: the prompt templates,
the LLM call, the plan parser, the transcript builders and the wc search constants are the
same objects the coverage run used. The evaluator hides the reference horizon and shows
only the CURRENT state (never the replay prefix). The plan-length budget is a flat 50 by
default; ``--cap-mode per-game|per-problem`` instead scales it off each row's measured
any-step reference reach (``action_cap``), which needs floors measured at the same budget.

WorldCoder's existing search requires an exact goal grid. It is evaluated only when the
run explicitly selects ``--goal-presentation frame`` and is reported as not applicable in
NL runs.

The startup oracle preflight replays every stored reference plan through the same wrapper
and scorer used for model plans before any paid LLM call is issued.

    uv run python offline_learning/scripts/eval_curated_plan.py \
        --problems logs/2026-08-29/planning_v2/problems.json \
        --artifact-root logs/2026-08-24/human_curated \
        --goal-presentation frame \
        --out logs/2026-08-29/planning_v2/eval/offline
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import replace as dc_replace
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OFF = HERE.parent
REPO = OFF.parent
for _p in (str(OFF), str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import icl_context  # noqa: E402
import program_runtime as prt  # noqa: E402
from validate import _parse_tag, run_perceive  # noqa: E402
from worldcoder_optimize import _clean_program  # noqa: E402
from offline_learning.human_replay import GAMES as HGAMES  # noqa: E402
from offline_learning.planning_nl_goals import (  # noqa: E402
    CHECKER_VERSION, get_python_goal, legacy_checker_id, score_python_goal,
    validate_problem_goal,
)
from offline_learning.planning_v2 import (  # noqa: E402
    SCHEMA_VERSION, load_problem_file, quiescent_after, rollout, success,
)

import eval_coverage_plan as ecp  # noqa: E402
from eval_coverage_plan import (  # noqa: E402
    CONTEXT_K, DEFAULT_KNOWLEDGE, PLAN_RAW_TMPL, PLAN_WIN_TMPL, WC_BEAM, WC_BUDGET,
    feat_transcript, llm_call, parse_plan, raw_transcript, resolve_llm_config,
)

PLAN_CAP = 50               # was 20; see module docstring
CAP_MODES = ("fixed", "per-game", "per-problem")
ATTEMPTS = 5
LLM_ARMS = ["raw", "lmwm", "icl"]
# "icl" is the raw planner with the world model's own training transitions pasted
# into the prompt -- the like-for-like control for "does the learned model do
# anything the data alone does not?". See offline_learning/icl_context.py.
RAW_LIKE_ARMS = ("raw", "icl")     # arms that read the raw grid, not P's features
ARMS = LLM_ARMS + ["wc"]
# "icl" is opt-in: its prompts carry ~20-160k extra tokens per call, so it must be
# asked for by name rather than inherited from a default that used to mean "all".
DEFAULT_ARMS = ["raw", "lmwm", "wc"]
TIERS = ["L1", "L2", "L3", "L4"]
DEFAULT_PROBLEMS = REPO / "logs/2026-08-29/planning_v2/problems.json"
DEFAULT_OUT = REPO / "logs/2026-08-29/planning_v2/eval/offline"
DEFAULT_ARTIFACT_ROOT = REPO / "logs/2026-08-24/human_curated"
WC_NL_POLICY = "not-applicable"


def gstr(grid: list[list[str]]) -> str:
    """Canonical grid string, byte-identical to what `_grid()` pulls out of a wrapper
    observation -- json.dumps' default ', ' separator is what the wrapper emits."""
    return json.dumps(grid)


def normalize_problem(row: dict) -> dict:
    """Return a representation-neutral v2 row; v1 lists are upgraded as frame+NL."""
    p = dict(row)
    required = {"game", "program", "id", "tier", "objective", "seed", "plan",
                "h", "n_decisions", "start", "goal"}
    missing = sorted(required - p.keys())
    if missing:
        raise ValueError(f"problem is missing required fields: {missing}")

    p.setdefault("schema_version", "curated-planning-v1")
    p.setdefault("task_uid", f"{p['game']}:{p['id']}:s{p['seed']}")
    p.setdefault("template_id", f"{p['game']}:{p['id']}")
    p.setdefault("prefix", [])
    p.setdefault("frame_success_mode", "any")
    p.setdefault("source", "curated-v1")
    p.setdefault("stochastic", False)
    p.setdefault("frame_reference_quiescent", None)
    p.setdefault("frame_noop_success", p.pop("noop_success", None))
    p.setdefault("nl_noop_success", None)
    p.setdefault("frame_random_success", p.pop("random_success", None))
    p.setdefault("frame_random_trials", p.pop("random_trials", None))
    p.setdefault("frame_random_success_cap50", p.pop("random_success_cap50", None))
    p.setdefault("frame_random_trials_cap50", p.pop("random_trials_cap50", None))
    p.setdefault("nl_random_success", None)
    p.setdefault("nl_random_trials", None)
    p.setdefault("nl_random_success_cap50", None)
    p.setdefault("nl_random_trials_cap50", None)
    p.setdefault("frame_anystep_reached_at", None)
    p.setdefault("nl_anystep_reached_at", None)
    p.setdefault("frame_random_floors", {})
    p.setdefault("nl_random_floors", {})

    if p["game"] not in HGAMES:
        raise ValueError(f"{p['task_uid']}: game is absent from human_replay.GAMES")
    if not isinstance(p["prefix"], list):
        raise ValueError(f"{p['task_uid']}: prefix must be a list")
    if not p.get("nl_checker"):
        checker_id = legacy_checker_id(p["game"], p["id"])
        goal = get_python_goal(checker_id)
        p["nl_checker"] = checker_id
        p["nl_checker_version"] = CHECKER_VERSION
        p["nl_success_mode"] = goal.success_mode
        p["nl_require_quiescent"] = goal.require_quiescent
        p["nl_reference_plan"] = list(goal.reference_plan or p["plan"])
        p["nl_goal"] = goal.nl
    else:
        p.setdefault("nl_checker_version", CHECKER_VERSION)
        goal = get_python_goal(p["nl_checker"])
        p.setdefault("nl_success_mode", goal.success_mode)
        p.setdefault("nl_require_quiescent", goal.require_quiescent)
        p.setdefault("nl_reference_plan", list(goal.reference_plan or p["plan"]))

    if p["frame_success_mode"] not in {"any", "final"}:
        raise ValueError(
            f"{p['task_uid']}: invalid frame_success_mode {p['frame_success_mode']!r}"
        )
    validate_problem_goal(p)
    if p["h"] != len(p["plan"]):
        raise ValueError(f"{p['task_uid']}: h={p['h']} but reference plan has "
                         f"{len(p['plan'])} actions")
    return p


def load_eval_problems(path: str | Path) -> tuple[dict, list[dict]]:
    """Load either a v1 list or the schema-versioned v2 container."""
    meta, raw = load_problem_file(path)
    schema = meta.get("schema_version")
    if schema is not None and schema != SCHEMA_VERSION:
        raise ValueError(f"unsupported planning schema {schema!r}; expected {SCHEMA_VERSION!r}")
    problems = [normalize_problem(row) for row in raw]
    uids = [p["task_uid"] for p in problems]
    duplicates = sorted(uid for uid, n in Counter(uids).items() if n > 1)
    if duplicates:
        raise ValueError(f"duplicate task_uid values: {duplicates}")
    return meta, problems


def _apply_success_override(p: dict, python_goal, override: str) -> dict:
    """Apply an optional any-step override to the selected representation."""
    if override == "reference":
        p["_eval_success_override"] = None
        return p
    if p["_eval_presentation"] == "frame":
        p["_eval_success_mode"] = "any"
    elif python_goal.require_quiescent:
        p["_eval_success_override"] = "kept-final-quiescent"
        return p
    else:
        p["_eval_success_mode"] = "any"
        p["_eval_python_goal"] = dc_replace(python_goal, success_mode="any")
    p["_eval_success_override"] = "any"
    return p


def configure_evaluation_goal(problem: dict, presentation: str,
                              success_override: str = "any") -> dict:
    """Attach one explicitly requested frame or NL prompt/scoring view."""
    if presentation not in {"frame", "nl"}:
        raise ValueError(f"unknown goal presentation {presentation!r}")
    if success_override not in {"any", "reference"}:
        raise ValueError(f"unknown success-mode override {success_override!r}")

    p = dict(problem)
    python_goal = validate_problem_goal(p)
    p["_eval_presentation"] = presentation
    # the flat budget unless a run resolves caps; keeps a configured row usable alone
    p["_eval_action_cap"] = PLAN_CAP
    p["_eval_python_goal"] = python_goal
    if presentation == "frame":
        p["_eval_success_mode"] = p["frame_success_mode"]
        p["_eval_oracle_plan"] = list(p["plan"])
        p["_eval_checker_source"] = "exact-frame"
    else:
        p["_eval_success_mode"] = python_goal.success_mode
        p["_eval_nl_goal"] = python_goal.nl
        p["_eval_oracle_plan"] = list(p["nl_reference_plan"])
        p["_eval_checker_source"] = "python-registry"
    return _apply_success_override(p, python_goal, success_override)


def select_goal_presentation(problems: list[dict], presentation: str,
                             success_override: str = "any") -> list[dict]:
    return [configure_evaluation_goal(p, presentation, success_override) for p in problems]


# ------------------------------------------------------- reference-scaled action caps
def action_cap(reach: int) -> int:
    """Action budget for a problem whose reference reaches the goal in `reach` actions.

    2x the reference up to 10 actions, 1.5x above it (rounded up). A flat cap is a
    different test per row: at 50 a 1-action row gets fifty chances to stumble onto its
    goal and a 40-action row gets one, so the same pass rate does not mean the same
    thing. Scaling the budget with the reference restores a common unit -- every row is
    "solve it with a little slack" -- and the slack shrinks proportionally as problems
    get longer, since a long plan that is 50% over reference is already lost."""
    if not isinstance(reach, int) or isinstance(reach, bool) or reach < 1:
        raise ValueError(f"reference reach must be a positive int, got {reach!r}")
    return 2 * reach if reach <= 10 else -(-3 * reach // 2)


def reference_reach(row: dict, presentation: str) -> int | None:
    """Actions the stored reference needs under ANY-STEP scoring, or None if unmeasured.

    Deliberately NOT `{pres}_reference_reached_at`: that field is measured under the
    row's own success mode, so a `final`-mode row stores the plan length even when the
    goal first holds much earlier (dino: stored 30, any-step 10). Online scoring is
    any-step by construction, so the any-step reach is the honest reference."""
    reach = row.get(f"{presentation}_anystep_reached_at")
    return reach if isinstance(reach, int) and not isinstance(reach, bool) else None


def resolve_action_caps(problems: list[dict], mode: str,
                        fixed_cap: int = PLAN_CAP) -> dict[str, int]:
    """{task_uid: action budget} under `mode`.

    `per-game` takes the max over the game's rows, so no row in a game is starved; that
    is looser than `per-problem` by exactly the within-game spread of reference lengths.
    Raises if a row lacks the measured any-step reach the scaled modes need."""
    if mode not in CAP_MODES:
        raise ValueError(f"unknown cap mode {mode!r}; choose from {', '.join(CAP_MODES)}")
    if mode == "fixed":
        return {p["task_uid"]: fixed_cap for p in problems}

    per_problem, missing = {}, []
    for p in problems:
        presentation = p.get("_eval_presentation")
        if presentation not in {"frame", "nl"}:
            raise ValueError(f"{p['task_uid']}: cap resolution needs a configured "
                             "presentation (call select_goal_presentation first)")
        reach = reference_reach(p, presentation)
        if reach is None:
            missing.append(p["task_uid"])
            continue
        per_problem[p["task_uid"]] = action_cap(reach)
    if missing:
        raise ValueError(
            f"--cap-mode {mode} needs a measured any-step reference reach; "
            f"{len(missing)} row(s) lack it (run annotate_action_caps.py): "
            + ", ".join(missing[:6]) + (" ..." if len(missing) > 6 else ""))
    if mode == "per-problem":
        return per_problem

    by_game: dict[str, int] = defaultdict(int)
    for p in problems:
        by_game[p["game"]] = max(by_game[p["game"]], per_problem[p["task_uid"]])
    return {p["task_uid"]: by_game[p["game"]] for p in problems}


def apply_action_caps(problems: list[dict], mode: str,
                      fixed_cap: int = PLAN_CAP) -> dict[str, int]:
    """Resolve caps and stamp each row with the budget its rollouts must obey."""
    caps = resolve_action_caps(problems, mode, fixed_cap)
    for p in problems:
        p["_eval_action_cap"] = caps[p["task_uid"]]
    return caps


def execute_and_score(problem: dict, plan: list[str], driver: str = "wrapper") \
        -> tuple[bool, int | None]:
    """Replay the state address, execute a candidate, and apply the selected scorer."""
    start, frames, _trace = rollout(
        problem["program"], problem["seed"], problem["prefix"], plan, driver,
    )
    if start != problem["start"]:
        raise ValueError(f"{problem['task_uid']}: replayed prefix does not reproduce start")

    presentation = problem.get("_eval_presentation")
    if presentation not in {"frame", "nl"}:
        raise ValueError("problem must be configured with frame or nl presentation")
    if presentation == "nl":
        goal = problem.get("_eval_python_goal") or validate_problem_goal(problem)
        stable = None
        if goal.require_quiescent:
            stable = quiescent_after(
                problem["program"], problem["seed"], problem["prefix"], plan, driver,
            )
        return score_python_goal(
            goal, start, frames, plan, stable_after_final=stable,
        )

    probe = dict(problem, frame_success_mode=problem["_eval_success_mode"])
    return success(probe, "frame", start, frames, plan)


def oracle_preflight(problems: list[dict]) -> dict:
    """Require every stored witness to pass through the exact model-plan execution path."""
    failures = []
    for p in problems:
        try:
            ok, at = execute_and_score(p, p.get("_eval_oracle_plan", p["plan"]))
        except Exception as exc:  # surface all malformed rows in one preflight report
            failures.append(f"{p['task_uid']}: {type(exc).__name__}: {exc}")
            continue
        if not ok:
            failures.append(f"{p['task_uid']}: reference plan failed")
        elif p.get("_eval_success_override") is None:
            stored = p.get(f"{p['_eval_presentation']}_reference_reached_at")
            if stored not in {None, at}:
                failures.append(
                    f"{p['task_uid']}: reached_at={at}, stored={stored}"
                )
    if failures:
        raise RuntimeError("oracle preflight failed:\n  " + "\n  ".join(failures))
    presentation = problems[0]["_eval_presentation"] if problems else None
    return {"passed": len(problems), "presentation": presentation}


def _exact_template(arm: str, success_mode: str) -> str:
    template = PLAN_RAW_TMPL if arm in RAW_LIKE_ARMS else PLAN_WIN_TMPL
    if success_mode == "final":
        return template
    replacements = {
        "raw": (
            "the grid after your FINAL action is\nEXACTLY the GOAL grid.",
            "the grid is EXACTLY the GOAL grid at some point during your plan\n"
            "(the last action may achieve it, or it may happen earlier).",
        ),
        "lmwm": (
            "the state after your FINAL action is\nEXACTLY the GOAL state.",
            "the state is EXACTLY the GOAL state at some point during your plan\n"
            "(the last action may achieve it, or it may happen earlier).",
        ),
    }
    old, new = replacements["raw" if arm in RAW_LIKE_ARMS else arm]
    if old not in template:
        raise RuntimeError("shared exact-goal prompt changed; update the any-step rewrite")
    return template.replace(old, new, 1)


_NL_RAW_TMPL = """You control a grid environment and must achieve a goal stated in words.

=== DEFAULT KNOWLEDGE (always-true facts about this environment) ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

Below is the CURRENT RAW GRID in canonical JSON.

{transcript}

=== GOAL (in words) ===
{goal}
=== END GOAL ===

Plan a sequence of AT MOST {cap} action(s) so that, starting from the CURRENT state and
executing your actions in order, {criterion} The environment's passive dynamics keep running
on every step (including noop), so timing can matter. Every action must be fully specified:
one of up, down, left, right, noop, or click ROW COL with 0-indexed integers.

Respond as:
<reasoning>how your actions achieve the stated goal, step by step</reasoning>
<plan>
one action per line, at most {cap} line(s)
</plan>"""


_NL_WIN_TMPL = """You control a grid environment and must achieve a goal stated in words.

=== DEFAULT KNOWLEDGE (always-true facts about this environment) ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== WORLD KNOWLEDGE (general facts; may be empty) ===
{beliefs}
=== END WORLD KNOWLEDGE ===

Below is the CURRENT state, summarized as features by a perception module.

{transcript}

=== GOAL (in words) ===
{goal}
=== END GOAL ===

Plan a sequence of AT MOST {cap} action(s) so that, starting from the CURRENT state and
executing your actions in order, {criterion} The environment's passive dynamics keep running
on every step (including noop), so timing can matter. Every action must be fully specified:
one of up, down, left, right, noop, or click ROW COL with 0-indexed integers.

Respond as:
<reasoning>how your actions achieve the stated goal, step by step</reasoning>
<plan>
one action per line, at most {cap} line(s)
</plan>"""


def add_icl_args(ap) -> None:
    """The `icl` arm's data knobs, shared by the offline and online evaluators."""
    ap.add_argument("--icl-pool", default=icl_context.DEFAULT_POOL,
                    help="training pool under human_data/<game>/ to paste in; must be "
                         "the pool the world model was fit on for the comparison to hold")
    ap.add_argument("--icl-data-root", default=str(icl_context.DEFAULT_DATA_ROOT))
    ap.add_argument("--icl-render", choices=icl_context.RENDERS, default="full",
                    help="full = both grids verbatim (default; no representational help); "
                         "diff = next state as the changed cells, ~half the tokens")
    ap.add_argument("--icl-context-k", type=int, default=0,
                    help="earlier frames to show per transition (0 = bare s,a,s\')")


def icl_config(a) -> dict:
    return {"pool": getattr(a, "icl_pool", icl_context.DEFAULT_POOL),
            "data_root": Path(getattr(a, "icl_data_root", icl_context.DEFAULT_DATA_ROOT)),
            "render": getattr(a, "icl_render", "full"),
            "context_k": getattr(a, "icl_context_k", 0)}


def load_icl_block(game: str, artifact_root: Path, cfg: dict | None) -> tuple[str, dict]:
    """The offline-data block for `game`, or ("", reason) if this game has no pool."""
    cfg = cfg or icl_config(argparse.Namespace())
    try:
        return icl_context.build_icl_block(
            game, pool=cfg["pool"], data_root=cfg["data_root"], render=cfg["render"],
            context_k=cfg["context_k"], artifact_root=artifact_root)
    except FileNotFoundError as exc:
        return "", {"error": str(exc)}


_ICL_ANCHOR = "=== END DEFAULT KNOWLEDGE ===\n"


def splice_icl(prompt: str, block: str) -> str:
    """Insert the offline-data block right after DEFAULT KNOWLEDGE.

    It goes at the FRONT, ahead of the per-round transcript, so the block is a stable
    prefix across a rollout's replanning rounds and a provider's prefix cache can serve
    it. An anchor that stopped matching means the shared template moved; that must fail
    loudly rather than silently drop the arm's only distinguishing input."""
    if not block.strip():
        raise ValueError("arm 'icl' requires a non-empty offline-data block")
    if _ICL_ANCHOR not in prompt:
        raise RuntimeError("shared prompt template changed; update the ICL splice anchor")
    i = prompt.index(_ICL_ANCHOR) + len(_ICL_ANCHOR)
    return prompt[:i] + "\n" + block.strip("\n") + "\n" + prompt[i:]


def build_prompt(problem: dict, arm: str, start_grid: str, *, start_features: str = "",
                 goal_features: str = "", beliefs: str = "",
                 hist_raw: list[tuple[str, str]] | None = None,
                 hist_z: list[tuple[str, str]] | None = None,
                 cap: int | None = None, icl_block: str = "") -> str:
    """Build a prompt without exposing an NL task's diagnostic reference frame.

    `hist_raw`/`hist_z`/`cap` exist for the ONLINE evaluator: same templates, but each
    round shows the executed history and plans within the remaining budget. The offline
    caller passes neither and gets the historical single-state prompt unchanged."""
    if arm not in LLM_ARMS:
        raise ValueError(f"unsupported LLM arm {arm!r}")
    cap = PLAN_CAP if cap is None else cap
    presentation = problem.get("_eval_presentation")
    if presentation not in {"frame", "nl"}:
        raise ValueError("problem must be configured with frame or nl presentation")
    eval_success = problem["_eval_success_mode"]
    eval_nl = problem.get("_eval_nl_goal", problem["nl_goal"])
    raw_like = arm in RAW_LIKE_ARMS
    if presentation == "frame":
        template = _exact_template(arm, eval_success)
        if raw_like:
            out = template.format(
                cap=cap, default_knowledge=DEFAULT_KNOWLEDGE,
                transcript=raw_transcript(hist_raw or [], start_grid),
                goal=gstr(problem["goal"]),
            )
            return splice_icl(out, icl_block) if arm == "icl" else out
        return template.format(
            cap=cap, default_knowledge=DEFAULT_KNOWLEDGE,
            beliefs=beliefs.strip() or "(empty)",
            transcript=feat_transcript(hist_z or [], start_features),
            goal=goal_features or "(empty)",
        )

    criterion = (
        "the GOAL above is true after your FINAL action."
        if eval_success == "final"
        else "the GOAL above is true at some point during your plan."
    )
    template = _NL_RAW_TMPL if raw_like else _NL_WIN_TMPL
    values = {
        "cap": cap,
        "default_knowledge": DEFAULT_KNOWLEDGE,
        "transcript": (raw_transcript(hist_raw or [], start_grid) if raw_like
                       else feat_transcript(hist_z or [], start_features)),
        "goal": eval_nl,
        "criterion": criterion,
        "beliefs": beliefs.strip() or "(empty)",
    }
    out = template.format(**values)
    return splice_icl(out, icl_block) if arm == "icl" else out


async def eval_game(game: str, problems: list[dict], sem, llm, artifact_root: Path,
                    arms: list[str], attempts_n: int,
                    a_reason: bool = True, a_keep: bool = False,
                    icl_cfg: dict | None = None) -> dict:
    llm_arms = [arm for arm in arms if arm in LLM_ARMS]
    need_lmwm = "lmwm" in llm_arms
    need_wc = ("wc" in arms and any(
        p["_eval_presentation"] == "frame" for p in problems
    ))
    # A missing artifact skips THAT arm for THAT game with a warning instead of killing
    # the whole run (an incomplete artifact tree -- e.g. a paused worldcoder relaunch --
    # is a completeness gap, not a reason to abort other games' paid work).
    skipped: dict[str, str] = {}

    perc_code = beliefs = ""
    if need_lmwm:
        rex = artifact_root / "rexpure" / f"{game}_s1"
        perception_path = rex / "best_perception_rexpure_seed1.py"
        beliefs_path = rex / "best_beliefs_rexpure_seed1.txt"
        for path, kind in ((perception_path, "perception"), (beliefs_path, "beliefs")):
            if not path.is_file():
                skipped["lmwm"] = f"missing {kind} artifact: {path}"
                break
        if "lmwm" in skipped:
            print(f"WARNING: {game}: skipping arm lmwm -- {skipped['lmwm']}", flush=True)
            llm_arms = [arm for arm in llm_arms if arm != "lmwm"]
            need_lmwm = False
        else:
            perc_code = perception_path.read_text()
            beliefs = beliefs_path.read_text()

    icl_block, icl_meta = "", {}
    if "icl" in llm_arms:
        icl_block, icl_meta = load_icl_block(game, artifact_root, icl_cfg)
        if not icl_block:
            skipped["icl"] = f"no training pool: {icl_meta.get('error')}"
            print(f"WARNING: {game}: skipping arm icl -- {skipped['icl']}", flush=True)
            llm_arms = [arm for arm in llm_arms if arm != "icl"]
        else:
            print(f"[icl] {game}: {icl_meta['n_transitions']} transitions, "
                  f"~{icl_meta['est_tokens']} tokens ({icl_meta['render']})", flush=True)

    rt = None
    verbs = HGAMES[game][2]
    if need_wc:
        wc_path = artifact_root / "worldcoder" / f"{game}_s1/best_transition_wc_seed1.py"
        if not wc_path.is_file():
            skipped["wc"] = f"missing WorldCoder artifact: {wc_path}"
            print(f"WARNING: {game}: skipping arm wc -- {skipped['wc']}", flush=True)
            need_wc = False
        else:
            rt = prt.ProgramRuntime(_clean_program(wc_path.read_text()), timeout_s=1.0)

    pcache: dict[str, str] = {}

    def perceive(g: str) -> str:
        if not perc_code:
            raise RuntimeError("perception requested without loading a perception artifact")
        if g not in pcache:
            pcache[g] = run_perceive(perc_code, g)[0]
        return pcache[g]

    prepared: dict[str, dict] = {}
    for p in problems:
        start_grid = gstr(p["start"])
        exact_goal = p["_eval_presentation"] == "frame"
        goal_grid = gstr(p["goal"]) if exact_goal else ""
        prepared[p["task_uid"]] = {
            "start": start_grid, "goal": goal_grid,
            "z_t": perceive(start_grid) if need_lmwm else "",
            "z_goal": perceive(goal_grid) if need_lmwm and exact_goal else "",
            "dims": (len(p["start"]), len(p["start"][0])),
        }

    async def attempts(p: dict, arm: str):
        prep = prepared[p["task_uid"]]
        prompt = build_prompt(
            p, arm, prep["start"], start_features=prep["z_t"],
            goal_features=prep["z_goal"], beliefs=beliefs,
            cap=p["_eval_action_cap"], icl_block=icl_block,
        )
        return await asyncio.gather(
            *(llm_call(prompt, sem, llm) for _ in range(attempts_n))
        )

    jobs = [(p, arm) for p in problems for arm in llm_arms]
    got = await asyncio.gather(*(attempts(p, arm) for p, arm in jobs))
    calls: dict[tuple[str, str], list] = {}
    for (p, arm), res in zip(jobs, got):
        calls[(p["task_uid"], arm)] = res

    rows, cost = [], 0.0
    for p in problems:
        prep = prepared[p["task_uid"]]
        presentation = p["_eval_presentation"]
        eval_success = p["_eval_success_mode"]
        row = {k: p.get(k) for k in (
            "game", "id", "task_uid", "template_id", "tier", "objective", "h",
            "n_decisions", "seed", "source", "stochastic",
        )}
        row.update({
            "goal_presentation": presentation,
            "nl_goal": p.get("_eval_nl_goal", p["nl_goal"]),
            "eval_success_mode": eval_success,
            "eval_success_override": p.get("_eval_success_override"),
            "checker_source": p["_eval_checker_source"],
            "nl_checker": p["nl_checker"],
            "nl_checker_version": p["nl_checker_version"],
            "nl_success_mode": p["nl_success_mode"],
            "nl_require_quiescent": p["nl_require_quiescent"],
            "reference_quiescent": p.get(f"{presentation}_reference_quiescent"),
            "random_success": p.get(f"{presentation}_random_success"),
            "random_trials": p.get(f"{presentation}_random_trials"),
            "random_success_cap50": p.get(f"{presentation}_random_success_cap50"),
            "random_trials_cap50": p.get(f"{presentation}_random_trials_cap50"),
            "action_cap": p["_eval_action_cap"],
            "random_floor": _floor(p),
            "noop_success": p.get(f"{presentation}_noop_success"),
            "prefix": list(p["prefix"]),
            "prefix_len": len(p["prefix"]),
            "start_grid": prep["start"],
            "goal_grid": prep["goal"] or None,
        })
        for arm in llm_arms:
            tries = []
            for text, think, c, errs in calls[(p["task_uid"], arm)]:
                cost += c
                plan, perr = parse_plan(text, prep["dims"])
                cap = p["_eval_action_cap"]
                if plan is not None and len(plan) > cap:
                    plan, perr = None, f"budget-exceeded:{len(plan)}>{cap}"
                ok, at, execution_error = False, None, None
                if plan is not None:
                    try:
                        ok, at = execute_and_score(p, plan)
                    except Exception as exc:  # keep one bad model plan from aborting the run
                        execution_error = f"{type(exc).__name__}:{exc}"
                if execution_error:
                    perr = f"execution-error:{execution_error}"
                rec = {"success": ok, "reached_at": at,
                       "plan_len": len(plan) if plan is not None else None,
                       "plan_error": perr, "execution_error": execution_error,
                       "retry_errors": errs, "plan": plan}
                if a_reason:
                    # Two different things, and both are kept.  `reasoning` is the model's
                    # own <reasoning> block: its stated justification, written for a reader.
                    # `thinking` is the provider's hidden chain, which on a reasoning model
                    # IS the deliberation -- the visible block is a summary composed after
                    # the fact.  It is stored capped (see eval_coverage_plan.REASONING_CAP)
                    # rather than dropped; --no-keep-thinking omits it entirely.
                    rec["reasoning"] = _parse_tag(text, "reasoning")
                    if a_keep:
                        rec.update(ecp.thinking_record(think))
                tries.append(rec)
            row[arm] = {"attempts": tries,
                        "pass_rate": sum(t["success"] for t in tries) / len(tries),
                        "pass_any": any(t["success"] for t in tries)}
        for arm, reason in skipped.items():
            row[arm] = {"status": "skipped-missing-artifact", "reason": reason,
                        "attempts": [], "pass_rate": None, "pass_any": None}
        if "wc" in arms and "wc" not in skipped:
            if presentation != "frame":
                row["wc"] = {
                    "status": "not-applicable",
                    "reason": "WorldCoder search requires an exact target frame",
                    "attempts": [], "pass_rate": None, "pass_any": None,
                }
            else:
                assert rt is not None
                start_g, goal_g = p["start"], p["goal"]
                universe = prt.build_action_universe(verbs, start_g, goal_g)
                search_error = None
                try:
                    found = prt.plan_search(
                        rt, [], start_g, goal_g, universe, p["_eval_action_cap"],
                        beam=WC_BEAM, node_budget=WC_BUDGET, context_k=CONTEXT_K,
                        allow_empty=False,
                    )
                except Exception as exc:
                    found = None
                    search_error = f"{type(exc).__name__}:{exc}"
                plan = ([prt.unparse_action(a) for a in found]
                        if found is not None else None)
                ok, at, execution_error = False, None, None
                if plan is not None:
                    try:
                        ok, at = execute_and_score(p, plan)
                    except Exception as exc:
                        execution_error = f"{type(exc).__name__}:{exc}"
                plan_error = (
                    f"search-error:{search_error}" if search_error else
                    f"execution-error:{execution_error}" if execution_error else
                    None if plan is not None else "no-plan-found"
                )
                row["wc"] = {
                    "status": "evaluated",
                    "attempts": [{
                        "success": ok, "reached_at": at,
                        "plan_len": len(plan) if plan is not None else None,
                        "plan_error": plan_error, "execution_error": execution_error,
                        "retry_errors": [], "plan": plan,
                    }],
                    "pass_rate": float(ok), "pass_any": ok,
                }
        rows.append(row)
    if rt is not None:
        rt.close()
    return {"game": game, "rows": rows, "cost": cost, "skipped": skipped}


def _mean(values) -> float | None:
    present = [float(v) for v in values if v is not None]
    return sum(present) / len(present) if present else None


def _floor(row: dict) -> float | None:
    """Random floor for the explicitly selected presentation, matched to the cap in force.

    A floor is only comparable to a model score when both were measured under the same
    action budget, so a row evaluated under a scaled cap uses the floor recomputed at
    THAT cap; the cap50 floor is the fallback for the flat-50 regime."""
    presentation = row.get("_eval_presentation")
    if presentation in {"frame", "nl"}:
        budget = row.get("_eval_action_cap")
        measured = row.get(f"{presentation}_random_floors") or {}
        if budget is not None and str(budget) in measured:
            return measured[str(budget)].get("success")
        cap = row.get(f"{presentation}_random_success_cap50")
        if budget not in (None, PLAN_CAP) and cap is not None:
            return None       # a cap50 floor does not describe a scaled-cap rollout
        return cap if cap is not None else row.get(f"{presentation}_random_success")
    # emitted report rows carry no _eval_presentation: they store the in-force floor
    # under `random_floor` (written by emit) and fall back to the flat-cap fields
    if row.get("random_floor") is not None:
        return row["random_floor"]
    cap = row.get("random_success_cap50")
    return cap if cap is not None else row.get("random_success")


def _cap_label(rows: list[dict], cap_mode: str | None = None) -> str:
    """"50" when every row shares a budget, "8-60 (per-game)" when they do not."""
    caps = {r["action_cap"] for r in rows if r.get("action_cap") is not None}
    if not caps:
        return str(PLAN_CAP)
    if len(caps) == 1:
        return str(caps.pop())
    return f"{min(caps)}-{max(caps)}" + (f" ({cap_mode})" if cap_mode else "")


def _cell(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _arm_headers(arms: list[str], attempts_n: int) -> list[str]:
    headers = []
    for arm in arms:
        if arm in LLM_ARMS:
            headers += [f"{arm} @1", f"{arm} @{attempts_n}"]
        else:
            headers.append("wc")
    return headers


def _arm_cells(rows: list[dict], arms: list[str]) -> list[str]:
    cells = []
    for arm in arms:
        if arm in LLM_ARMS:
            cells += [
                _cell(_mean(r[arm]["pass_rate"] for r in rows)),
                _cell(_mean(r[arm]["pass_any"] for r in rows)),
            ]
        else:
            cells.append(_cell(_mean(r[arm]["pass_rate"] for r in rows)))
    return cells


def report(all_rows: list[dict], llm, elapsed: float, cost: float, arms: list[str],
           attempts_n: int, presentation: str, oracle: dict | None,
           skipped_arms: dict[str, dict] | None = None,
           excluded_saturated: list[dict] | None = None) -> str:
    planner = llm.label if llm is not None else "none (no LLM arm requested)"
    llm_arms = [arm for arm in arms if arm in LLM_ARMS]
    sampling = (
        f"{attempts_n} attempts per LLM arm" if llm_arms else "no LLM sampling"
    )
    pass_note = (
        f"`pass@1` is the attempt mean; `pass@{attempts_n}` is any-of-{attempts_n}. "
        if llm_arms else ""
    )
    checkers = Counter(r["checker_source"] for r in all_rows)
    oracle_text = (
        f"{oracle['passed']}/{len(all_rows)} passed" if oracle is not None else "disabled"
    )
    L = [
        "# Curated planning eval - OFFLINE (open-loop)", "",
        f"Planner: {planner} | arms {','.join(arms)} | plan cap "
        f"{_cap_label(all_rows)} | "
        f"{sampling} | {len(all_rows)} problems | "
        f"{elapsed / 60:.1f} min | ${cost:.2f}", "",
        f"Goal presentation: `{presentation}` ({len(all_rows)} problems). "
        f"Oracle preflight: {oracle_text}.", "",
        "Checker sources: " + ", ".join(
            f"`{name}`={count}" for name, count in sorted(checkers.items())
        ) + ".", "",
        pass_note +
        "`rand` is the cap-matched random floor (random plans of the SAME length as the "
        "row's action budget, under the evaluated success mode) where the recompute has run, "
        "falling back to the stored rand@h floor otherwise. Floors are defined under the "
        "WorldCoder requires an exact target frame. NL-only rows are reported as N/A; "
        "their diagnostic reference frames are never used as targets.", "",
    ]
    overrides = Counter(r.get("eval_success_mode") for r in all_rows)
    if any(r.get("eval_success_override") for r in all_rows):
        kept = sum(r.get("eval_success_override") == "kept-final-quiescent"
                   for r in all_rows)
        L += [
            "Success mode: unified ANY-STEP (the legacy v1 rule) -- "
            f"{overrides.get('any', 0)} rows scored at-any-step"
            + (f", {kept} quiescence-requiring checkers kept final+quiescent "
               "(any-step quiescence is undefined)" if kept else "")
            + ". The selected representation's reference success mode is recorded per row.", "",
        ]
    if excluded_saturated:
        L += ["**Excluded saturated problems** (random floor above --max-floor; random "
              "play solves them, so they cannot discriminate planners):", ""]
        for p in excluded_saturated:
            L.append(f"- `{p['task_uid']}` (floor {_floor(p):.2f})")
        L.append("")
    if skipped_arms:
        L += ["**Skipped arms** (missing artifacts; per-game, run continued):", ""]
        for game, arm_reasons in sorted(skipped_arms.items()):
            for arm, reason in sorted(arm_reasons.items()):
                L.append(f"- {game} / {arm}: {reason}")
        L.append("")

    metric_headers = _arm_headers(arms, attempts_n)

    def add_summary(title: str, groups: list[tuple[str, list[dict]]]) -> None:
        L.extend([
            f"## {title}", "",
            "| group | n | " + " | ".join(metric_headers) + " | rand |",
            "|---|--:|" + "--:|" * (len(metric_headers) + 1),
        ])
        for label, rows in groups:
            values = [
                str(len(rows)),
                *_arm_cells(rows, arms),
                _cell(_mean(_floor(r) for r in rows)),
            ]
            L.append(f"| {label} | " + " | ".join(values) + " |")
        L.append("")

    games = []
    for game in dict.fromkeys(r["game"] for r in all_rows):
        games.append((
            f"{game} ({HGAMES[game][1]})",
            [r for r in all_rows if r["game"] == game],
        ))
    add_summary("Per game", games)

    if llm_arms and any(_floor(r) for r in all_rows):
        L += [
            "## Per game, floor-adjusted (skill above chance: (pass - rand) / (1 - rand), "
            "clipped at 0)", "",
            "| game | n | " + " | ".join(f"{a} @1" for a in llm_arms) + " |",
            "|---|--:|" + "--:|" * len(llm_arms),
        ]

        def _adj(row, arm):
            pr = row.get(arm, {}).get("pass_rate")
            fl = _floor(row) or 0.0
            if pr is None or fl >= 1.0:
                return None
            return max(0.0, (pr - fl) / (1.0 - fl))

        for game, rows_g in games:
            cells = [_cell(_mean(_adj(r, arm) for r in rows_g)) for arm in llm_arms]
            L.append(f"| {game} | {len(rows_g)} | " + " | ".join(cells) + " |")
        L.append("")

    tiers = [
        (tier, [r for r in all_rows if r["tier"] == tier])
        for tier in TIERS if any(r["tier"] == tier for r in all_rows)
    ]
    add_summary("Per tier", tiers)

    by_template: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        by_template[row["template_id"]].append(row)
    multi_seed = [(f"`{uid}`", rows) for uid, rows in by_template.items()
                  if len({r["seed"] for r in rows}) > 1]
    if multi_seed:
        add_summary("Multi-seed templates", multi_seed)

    L += [
        "## Per problem", "",
        "| task | tier | mode | checker | h | dec | prefix | "
        + " | ".join(metric_headers) + " | rand |",
        "|---|---|---|---|--:|--:|--:|" + "--:|" * (len(metric_headers) + 1),
    ]
    for row in all_rows:
        values = [
            f"`{row['task_uid']}`", row["tier"], row["goal_presentation"],
            row["checker_source"], str(row["h"]), str(row["n_decisions"]),
            str(row["prefix_len"]), *_arm_cells([row], arms),
            _cell(_floor(row)),
        ]
        L.append("| " + " | ".join(values) + " |")

    bad = defaultdict(int)
    for row in all_rows:
        for arm in arms:
            for attempt in row.get(arm, {}).get("attempts", []):
                if attempt["plan_error"]:
                    bad[f"{arm}:{attempt['plan_error'].split(':')[0]}"] += 1
    if bad:
        L += ["", "## Unusable responses", ""]
        L += [f"- {name}: {count}" for name, count in sorted(bad.items())]
    return "\n".join(L) + "\n"


async def main_async(a):
    meta, problems = load_eval_problems(a.problems)
    input_count = len(problems)
    available_games = {p["game"] for p in problems}

    if a.games:
        requested_games = [g.strip() for g in a.games.split(",") if g.strip()]
        unknown_games = sorted(set(requested_games) - available_games)
        if unknown_games:
            raise ValueError(f"requested games absent from input: {unknown_games}")
        wanted = set(requested_games)
        problems = [p for p in problems if p["game"] in wanted]

    problems = select_goal_presentation(problems, a.goal_presentation, a.success_mode)
    caps = apply_action_caps(problems, a.cap_mode, a.max_actions)
    if a.cap_mode != "fixed":
        unmatched = [p["task_uid"] for p in problems if _floor(p) is None]
        if unmatched:
            raise ValueError(
                f"--cap-mode {a.cap_mode} needs random floors measured at the same "
                f"budget; {len(unmatched)} row(s) have none (run "
                f"recompute_random_floors.py --cap-mode {a.cap_mode}): "
                + ", ".join(unmatched[:6]) + (" ..." if len(unmatched) > 6 else ""))
        print(f"action caps ({a.cap_mode}): {min(caps.values())}-{max(caps.values())} "
              f"over {len(caps)} rows", flush=True)
    excluded_saturated = []
    if a.max_floor is not None and a.max_floor >= 0:
        keep = []
        for p in problems:
            if (_floor(p) or 0.0) > a.max_floor:
                excluded_saturated.append(p)
            else:
                keep.append(p)
        problems = keep
        if excluded_saturated:
            print(f"excluding {len(excluded_saturated)} saturated problems "
                  f"(random floor > {a.max_floor}): "
                  + ", ".join(p["task_uid"] for p in excluded_saturated), flush=True)
    if a.limit:
        per_game = defaultdict(int)
        keep = []
        for p in problems:
            if per_game[p["game"]] < a.limit:
                keep.append(p)
                per_game[p["game"]] += 1
        problems = keep
    if not problems:
        raise ValueError("filters selected no planning problems")

    arms = list(dict.fromkeys(arm.strip() for arm in a.arms.split(",") if arm.strip()))
    unknown_arms = sorted(set(arms) - set(ARMS))
    if not arms or unknown_arms:
        raise ValueError(f"invalid --arms {a.arms!r}; choose from {','.join(ARMS)}")
    if a.attempts < 1:
        raise ValueError("--attempts must be positive")
    if a.concurrency < 1:
        raise ValueError("--concurrency must be positive")
    if a.oracle_only and not a.oracle_preflight:
        raise ValueError("--oracle-only cannot be combined with --no-oracle-preflight")

    oracle = oracle_preflight(problems) if a.oracle_preflight else None
    if oracle is not None:
        print(
            f"oracle preflight: {oracle['passed']} {oracle['presentation']} "
            "problems passed",
            flush=True,
        )
    checker_counts = Counter(p["_eval_checker_source"] for p in problems)
    if a.oracle_only:
        summary = {
            "input": str(Path(a.problems)),
            "schema_version": meta.get("schema_version"),
            "goal_presentation": a.goal_presentation,
            "selected": len(problems),
            "checker_sources": dict(sorted(checker_counts.items())),
            "oracle": oracle,
        }
        print(json.dumps(summary, indent=2))
        return summary

    llm_requested = any(arm in LLM_ARMS for arm in arms)
    llm = resolve_llm_config(a) if llm_requested else None
    sem = asyncio.Semaphore(a.concurrency) if llm_requested else None
    root = Path(a.artifact_root)
    by_game: dict[str, list] = defaultdict(list)
    for p in problems:
        by_game[p["game"]].append(p)

    ecp.CALL_STATS.clear()
    t0 = time.time()
    out = await asyncio.gather(*(
        eval_game(
            game, game_problems, sem, llm, root, arms, a.attempts,
            a.reasoning_trace, a.keep_thinking, icl_config(a),
        )
        for game, game_problems in by_game.items()
    ))
    elapsed = time.time() - t0
    rows = [row for game_result in out for row in game_result["rows"]]
    cost = sum(game_result["cost"] for game_result in out)
    skipped_arms = {g["game"]: g["skipped"] for g in out if g.get("skipped")}
    md = report(
        rows, llm, elapsed, cost, arms, a.attempts, a.goal_presentation, oracle,
        skipped_arms, excluded_saturated,
    )
    print(md, flush=True)

    stem = Path(a.out)
    stem.parent.mkdir(parents=True, exist_ok=True)
    served = Counter((call.get("provider") or "unknown") for call in ecp.CALL_STATS)
    walls = sorted(call["wall_s"] for call in ecp.CALL_STATS)
    config = {
        "input": str(Path(a.problems)),
        "input_schema_version": meta.get("schema_version"),
        "input_problem_count": input_count,
        "selected_problem_count": len(problems),
        "artifact_root": str(root),
        "goal_presentation": a.goal_presentation,
        "checker_sources": dict(sorted(checker_counts.items())),
        "oracle_preflight": oracle,
        "scoring_driver": "wrapper",
        "prefix_replayed": True,
        "wc_nl_policy": WC_NL_POLICY,
        "random_floor": ("cap50-any-step-recomputed (fallback stored-at-reference-h)"
                         if any(r.get("random_success_cap50") is not None for r in rows)
                         else "stored-at-reference-h-for-selected-presentation"),
        "success_mode": a.success_mode,
        "max_floor": a.max_floor,
        "excluded_saturated": [p["task_uid"] for p in excluded_saturated],
        "success_mode_composition": dict(Counter(r["eval_success_mode"] for r in rows)),
        "skipped_arms": skipped_arms,
        "arms": arms,
        "model": llm.model if llm is not None else None,
        "backend": llm.backend if llm is not None else None,
        "label": llm.label if llm is not None else None,
        "plan_cap": PLAN_CAP,
        "cap_mode": a.cap_mode,
        "action_caps": {p["task_uid"]: p["_eval_action_cap"] for p in problems},
        "attempts": a.attempts,
        "context_k": CONTEXT_K,
        "wc_budget": WC_BUDGET,
        "concurrency": a.concurrency,
        "providers_served": dict(served.most_common()),
        "call_p50_s": walls[len(walls) // 2] if walls else None,
        "call_mean_s": (sum(walls) / len(walls)) if walls else None,
        "elapsed_s": elapsed,
    }
    payload = {"config": config, "rows": rows, "cost": cost}
    stem.with_suffix(".json").write_text(json.dumps(payload, indent=1))
    stem.with_suffix(".md").write_text(md)
    print(f"wrote {stem}.json / {stem}.md", flush=True)
    return payload


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate curated planning-problems v1/v2 from replayed current states."
    )
    ap.add_argument("--problems", default=str(DEFAULT_PROBLEMS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    ap.add_argument("--games", default="",
                    help="optional comma-separated game filter")
    ap.add_argument("--limit", type=int, default=0,
                    help="maximum selected problems per game (0 means all)")
    ap.add_argument(
        "--goal-presentation", choices=("frame", "nl"), required=True,
        help="required: present and score every selected problem as frame or NL",
    )
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                    help=f"comma-separated subset of {','.join(ARMS)}; 'icl' is the raw "
                         "planner with the world model's training transitions in context "
                         "and is off by default")
    add_icl_args(ap)
    ap.add_argument("--max-actions", type=int, default=PLAN_CAP,
                    help="plan-length budget under --cap-mode fixed")
    ap.add_argument("--cap-mode", choices=CAP_MODES, default="fixed",
                    help="how the plan-length budget is set: 'fixed' is --max-actions "
                    "for every row; 'per-game' and 'per-problem' scale it off the "
                    "measured any-step reference reach (2x up to 10 actions, 1.5x "
                    "above), per-game taking the max over the game's rows. Scaled modes "
                    "need floors recomputed at the same budget")
    ap.add_argument(
        "--max-floor", type=float, default=0.95,
        help="exclude problems whose random floor (cap-matched when present) exceeds "
        "this; they are listed in the report, never evaluated. Negative disables.",
    )
    ap.add_argument(
        "--success-mode", choices=("any", "reference"), default="any",
        help="any=unified legacy any-step scoring for every row (default; "
        "quiescence-requiring NL checkers keep final+quiescent); "
        "reference=use the selected representation's stored success mode",
    )
    ap.add_argument("--attempts", type=int, default=ATTEMPTS,
                    help="independent samples per requested LLM arm")
    ap.add_argument("--oracle-preflight", action=argparse.BooleanOptionalAction, default=True,
                    help="replay and score every selected reference plan before evaluation")
    ap.add_argument("--oracle-only", action="store_true",
                    help="run preflight and exit without API calls or learned artifacts")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--llm-backend", choices=("claude", "openrouter"), default="openrouter")
    ap.add_argument("--llm-url", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--provider-order", default=None)
    ecp.add_llm_tuning_args(ap)
    ap.add_argument("--reasoning-trace", action=argparse.BooleanOptionalAction, default=True,
                    help="persist each attempt's <reasoning> block")
    ap.add_argument("--keep-thinking", action=argparse.BooleanOptionalAction, default=True,
                    help="persist the provider's hidden reasoning tokens, capped by "
                         "LLM_REASONING_CAP (default 8000 chars/call)")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
