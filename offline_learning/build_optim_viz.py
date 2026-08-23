#!/usr/bin/env python3
"""Build a self-contained, step-by-step HTML visualisation of a REx-pure perception/belief
optimisation run (the inverse-dynamics setup in rexpure_optimize.py).

It joins four things that are otherwise scattered across the run dir:
  - candidates.json            -> each candidate's perception code + world_knowledge + val score + lineage
  - run_log.json               -> per-iteration: which parent was selected, which candidate was born
  - generated_best_outputs_valset/task_*/iter_*_prog_*.json
                               -> every candidate's predicted action for every val transition
  - the raw trajectory.csv (--run) -> ground-truth action + the actual on-grid movement,
                                      reconstructed into the SAME train/val/test split the run used
                                      (deterministic, seeded) so task_N lines up with a real transition.

Output: <run-dir>/optim_viz.html  (open in any browser; no server needed).

The headline view is a TRANSITION x CANDIDATE matrix: rows are val transitions (with their
before/after grids + true action + green-move vector), columns are candidates in birth order,
cells are green/red for correct/wrong.

Usage (the dq8gc double-budget full-batch prompt-fix run):
  uv run python offline_learning/build_optim_viz.py \
      --run-dir logs/clean_sweep_fwd/dq8gc_seed1_fullbatch_promptfix/rexpure_run_seed1 \
      --run offline_learning/clean_data/dq8gc \
      --seed 1 --train-n 20 --test-n 10 --tie-train-val \
      --actions left,right,up,down,noop,click
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------------------
# Data loading + split -- replicated verbatim from validate.py / validate_beliefs.py /
# rexpure_optimize.py so this tool has no heavy deps and reproduces the exact same ordering.
# (Verified against the saved test_trace truths via an assert below.)
# --------------------------------------------------------------------------------------
@dataclass
class Transition:
    x_t: str
    x_t1: str
    action: str
    # K-step temporal context (mirrors validate.py). Empty unless context_k>0.
    # ctx_prev: steps BEFORE x_t, oldest->newest, each (state_raw, outgoing_action) so the
    # last entry's action leads into x_t. ctx_next: steps AFTER x_t1, each (incoming_action,
    # state_raw) so resulting future states are X_t+2..X_t+K. Both partial at episode edges.
    ctx_prev: list = field(default_factory=list)
    ctx_next: list = field(default_factory=list)


def _row_step(row, whitelist):
    """(observation, action) for a usable, in-vocab, non-terminal row; else None (boundary)."""
    action = (row.get("Action") or "").strip()
    obs = row.get("Observation") or ""
    done = (row.get("Done") or "").strip().lower() in ("true", "1")
    if done or not action or not obs.strip():
        return None
    if whitelist is not None and action.split()[0] not in whitelist:
        return None
    return obs, action


def load_transitions(run_dirs, whitelist, context_k=0):
    transitions = []
    for run_dir in run_dirs:
        # Preserve pathlib/glob iteration order: rexpure_optimize's validate.load_transitions
        # does not sort, and the seeded shuffle therefore depends on this input order.
        for csv_path in run_dir.glob("episode_*/trajectory.csv"):
            rows = list(csv.DictReader(csv_path.open()))
            for i in range(len(rows) - 1):
                r, nxt = rows[i], rows[i + 1]
                action = (r.get("Action") or "").strip()
                obs, obs_next = r.get("Observation") or "", nxt.get("Observation") or ""
                done = (r.get("Done") or "").strip().lower() in ("true", "1")
                if done or not action or not obs.strip() or not obs_next.strip():
                    continue
                if whitelist is not None and action.split()[0] not in whitelist:
                    continue
                tr = Transition(obs, obs_next, action)
                if context_k > 0:
                    prev = []
                    for j in range(i - 1, i - 1 - context_k, -1):
                        if j < 0:
                            break
                        step = _row_step(rows[j], whitelist)
                        if step is None:
                            break
                        prev.append(step)
                    tr.ctx_prev = list(reversed(prev))  # oldest -> newest
                    nxts = []
                    for m in range(1, context_k):
                        step = _row_step(rows[i + m], whitelist) if i + m < len(rows) else None
                        fut = rows[i + m + 1].get("Observation") if i + m + 1 < len(rows) else None
                        if step is None or not (fut and fut.strip()):
                            break
                        nxts.append((step[1], fut))  # (incoming action, resulting state)
                    tr.ctx_next = nxts
                transitions.append(tr)
    if not transitions:
        raise FileNotFoundError(f"no usable transitions under {run_dirs}")
    return transitions


def backfill_context_from_source(transitions, source_dirs, whitelist, context_k):
    """Attach full-trajectory context to curated transition-pair targets.

    This mirrors validate.backfill_context_from_source, kept local so the standalone
    visualizer remains free of the optimizer's heavier imports.
    """
    if context_k <= 0:
        return transitions
    source = load_transitions(source_dirs, whitelist, context_k=context_k)
    by_pair = defaultdict(list)
    for tr in source:
        by_pair[(tr.x_t, tr.action, tr.x_t1)].append(tr)
    for i, tr in enumerate(transitions):
        matches = by_pair.get((tr.x_t, tr.action, tr.x_t1), [])
        if not matches:
            raise ValueError(
                f"curated transition {i} was not found verbatim in context source {source_dirs}"
            )
        contexts = {(tuple(m.ctx_prev), tuple(m.ctx_next)) for m in matches}
        if len(contexts) != 1:
            raise ValueError(
                f"curated transition {i} has {len(contexts)} distinct source contexts; "
                "the source mapping is ambiguous"
            )
        tr.ctx_prev = list(matches[0].ctx_prev)
        tr.ctx_next = list(matches[0].ctx_next)
    return transitions


# Obs-metadata strip -- replicated verbatim from validate.py so the reconstructed
# transitions carry the SAME observation bytes the run hashed. rexpure_optimize strips this
# by default (unless --keep-obs-metadata); the per-prediction sidecar keys tr_hash on the
# STRIPPED grid, so the viz MUST strip too or every detail join misses. Marker + logic must
# stay byte-identical to strip_autumn_obs_metadata / strip_transitions_obs_metadata.
_OBS_METADATA_MARKER = "========== Start of Direct Observation =========="


def strip_autumn_obs_metadata(obs):
    """Keep only the JSON colour-name grid; drop the Task:/Step:/Phase: harness header.
    ARC integer grids and header-less strings pass through unchanged."""
    if not obs or _OBS_METADATA_MARKER not in obs:
        return obs
    tail = obs.split(_OBS_METADATA_MARKER, 1)[1]
    s, e = tail.find("[["), tail.rfind("]]")
    if s == -1 or e <= s:
        return obs
    payload = tail[s : e + 2]
    try:
        g = json.loads(payload)
    except Exception:
        return obs
    if not (
        isinstance(g, list) and g and isinstance(g[0], list)
        and g[0] and isinstance(g[0][0], str)
    ):
        return obs
    return payload


def strip_transitions_obs_metadata(transitions):
    """strip_autumn_obs_metadata over x_t / x_t1 / both context walks, in place. Must run
    AFTER backfill_context_from_source (backfill matches curated pairs verbatim, Step included)."""
    for tr in transitions:
        tr.x_t = strip_autumn_obs_metadata(tr.x_t)
        tr.x_t1 = strip_autumn_obs_metadata(tr.x_t1)
        tr.ctx_prev = [(strip_autumn_obs_metadata(s), a) for s, a in tr.ctx_prev]
        tr.ctx_next = [(a, strip_autumn_obs_metadata(s)) for a, s in tr.ctx_next]
    return transitions


def balanced_split(transitions, holdout_n, train_n, rng):
    by_action = defaultdict(list)
    for t in transitions:
        by_action[t.action].append(t)
    for v in by_action.values():
        rng.shuffle(v)
    actions = list(by_action)
    holdout = []
    i = 0
    while len(holdout) < holdout_n and any(by_action[a] for a in actions):
        a = actions[i % len(actions)]
        if by_action[a]:
            holdout.append(by_action[a].pop())
        i += 1
    rest = [t for a in actions for t in by_action[a]]
    rng.shuffle(rest)
    return rest[:train_n], holdout


def stratified_split(transitions, n_a, n_b, rng):
    """Replicated verbatim from rexpure_optimize.py (--stratified-split): ONE pool dealt
    into two sets that BOTH see every action -- each action's (shuffled) rows are dealt
    alternately, cycling actions round-robin, until the size targets are met."""
    by_action = defaultdict(list)
    for t in transitions:
        by_action[t.action].append(t)
    for v in by_action.values():
        rng.shuffle(v)
    actions = list(by_action)
    side_a, side_b = [], []
    caps = {id(side_a): n_a, id(side_b): n_b}
    nxt = {x: j % 2 for j, x in enumerate(actions)}
    i = 0
    while (len(side_a) < n_a or len(side_b) < n_b) and any(
        by_action[x] for x in actions
    ):
        x = actions[i % len(actions)]
        i += 1
        if not by_action[x]:
            continue
        t = by_action[x].pop()
        first, second = (
            (side_a, side_b) if nxt[x] == 0 else (side_b, side_a)
        )
        nxt[x] ^= 1
        if len(first) < caps[id(first)]:
            first.append(t)
        elif len(second) < caps[id(second)]:
            second.append(t)
    return side_a, side_b


def swap_click_into_train(train_tr, test_tr):
    """Move ONE click transition from test INTO train, and one 'up' from train INTO test,
    in place. Puts a click-IDENTIFICATION item into the scored train/val set (so a correct
    click belief can finally earn score under the objective) while keeping a click in the
    clean test set for generalization. Deterministic: swaps the first such transitions in the
    already-seed-shuffled split lists. Returns the click action moved to train, or None if
    there was no test click or no train 'up' to swap."""
    ci = next((i for i, t in enumerate(test_tr) if t.action.split()[0] == "click"), None)
    ui = next((i for i, t in enumerate(train_tr) if t.action.split()[0] == "up"), None)
    if ci is None or ui is None:
        return None
    test_tr[ci], train_tr[ui] = train_tr[ui], test_tr[ci]
    return train_tr[ui].action  # the click now sitting in train


def make_choices(true_action, action_pool, k, rng):
    distractors = [a for a in action_pool if a != true_action]
    rng.shuffle(distractors)
    choices = [true_action] + distractors[: max(0, k - 1)]
    rng.shuffle(choices)
    return choices


def bake_choices(transitions, action_pool, k, rng):
    return [{"tr": tr, "choices": make_choices(tr.action, action_pool, k, rng)} for tr in transitions]


def reconstruct_split(args):
    """Replay rexpure_optimize.py's exact RNG sequence and return its val/test transitions."""
    rng = random.Random(args.seed)
    whitelist = set(filter(None, args.actions.split(","))) or None
    run_dirs = [Path(p) for p in args.run.split(",") if p.strip()]
    context_k = getattr(args, "context_k", 0)
    context_source_run = getattr(args, "context_source_run", None)
    if context_source_run:
        source_dirs = [Path(p) for p in context_source_run.split(",") if p.strip()]
        if len(source_dirs) == len(run_dirs):
            transitions = []
            for target_dir, source_dir in zip(run_dirs, source_dirs):
                target = load_transitions([target_dir], whitelist, context_k=context_k)
                backfill_context_from_source(target, [source_dir], whitelist, context_k)
                transitions.extend(target)
        else:
            transitions = load_transitions(run_dirs, whitelist, context_k=context_k)
            backfill_context_from_source(transitions, source_dirs, whitelist, context_k)
    else:
        transitions = load_transitions(run_dirs, whitelist, context_k=context_k)
    # mirror rexpure_optimize._collapse: strip obs metadata (default on) BEFORE any action
    # collapse, so tr_hash over (x_t, x_t1, action) matches the logged predictions sidecar.
    if not getattr(args, "keep_obs_metadata", False):
        strip_transitions_obs_metadata(transitions)
    if args.collapse_action_params:
        for t in transitions:
            t.action = t.action.split()[0]

    test_run = getattr(args, "test_run", None)
    if test_run:  # cross-trajectory split: mirror rexpure_optimize.py's --test-run RNG sequence
        test_dirs = [Path(p) for p in test_run.split(",") if p.strip()]
        test_context_source_run = getattr(args, "test_context_source_run", None)
        if test_context_source_run:
            test_source_dirs = [
                Path(p) for p in test_context_source_run.split(",") if p.strip()
            ]
            if len(test_source_dirs) == len(test_dirs):
                test_transitions = []
                for target_dir, source_dir in zip(test_dirs, test_source_dirs):
                    target = load_transitions([target_dir], whitelist, context_k=context_k)
                    backfill_context_from_source(
                        target, [source_dir], whitelist, context_k
                    )
                    test_transitions.extend(target)
            else:
                test_transitions = load_transitions(
                    test_dirs, whitelist, context_k=context_k
                )
                backfill_context_from_source(
                    test_transitions, test_source_dirs, whitelist, context_k
                )
        else:
            test_transitions = load_transitions(
                test_dirs, whitelist, context_k=context_k
            )
        if not getattr(args, "keep_obs_metadata", False):
            strip_transitions_obs_metadata(test_transitions)
        if args.collapse_action_params:
            for t in test_transitions:
                t.action = t.action.split()[0]
        action_pool = sorted({t.action for t in transitions} | {t.action for t in test_transitions})
        train_pool = list(transitions); rng.shuffle(train_pool)
        test_pool = list(test_transitions); rng.shuffle(test_pool)
        test_n = args.test_n if args.test_n >= 0 else 10**9
        _, test_tr = balanced_split(test_pool, test_n, 10**9, rng)
        if args.stratified_split:
            train_tr, val_tr = stratified_split(
                train_pool, args.train_n, args.val_n, rng
            )
        else:
            rest, train_tr = balanced_split(train_pool, args.train_n, 10**9, rng)
            if args.tie_train_val:
                val_tr = train_tr
            else:
                _, val_tr = balanced_split(rest, args.val_n, 10**9, rng)
        train = bake_choices(train_tr, action_pool, args.k_choices, rng)
        val = train if args.tie_train_val else bake_choices(
            val_tr, action_pool, args.k_choices, rng
        )
        test = bake_choices(test_tr, action_pool, args.k_choices, rng)
        return val, test, action_pool

    action_pool = sorted({t.action for t in transitions})
    # mirror rexpure_optimize's pin-aware split: pinned transitions (by original index) are
    # held out of the shuffle/test carve and prepended to train.
    pin_idx = sorted({int(x) for x in getattr(args, "pin_train_idx", "").split(",") if x.strip()})
    pinned = [transitions[i] for i in pin_idx if 0 <= i < len(transitions)]
    pinned_ids = {id(t) for t in pinned}
    pool = [t for t in transitions if id(t) not in pinned_ids]
    rng.shuffle(pool)
    rest, test_tr = balanced_split(pool, args.test_n, 10**9, rng)
    if args.tie_train_val:
        _, train_core = balanced_split(rest, args.train_n, 10**9, rng)
        train_tr = pinned + train_core
        val_tr = train_tr
    else:
        train_pool, val_tr = balanced_split(rest, args.val_n, 10**9, rng)
        train_tr = pinned + train_pool[: args.train_n]
    if getattr(args, "swap_click_train", False):
        swap_click_into_train(train_tr, test_tr)
    train = bake_choices(train_tr, action_pool, args.k_choices, rng)
    val = train if args.tie_train_val else bake_choices(
        val_tr, action_pool, args.k_choices, rng
    )
    test = bake_choices(test_tr, action_pool, args.k_choices, rng)
    return val, test, action_pool


# --------------------------------------------------------------------------------------
# Grid parsing -> green/gray cells + movement vector
# --------------------------------------------------------------------------------------
def parse_grid(obs):
    """Parse a raw observation into a 2D grid, supporting both env formats:
      - AutumnBench: a single nested ``[[ "black", "green", ... ]]`` JSON array of
        color-NAME strings.
      - ARC-AGI-3: one or more ``<grid_N> ... </grid_N>`` blocks whose rows are
        single-bracket lists of INTEGER palette indices (0..15), one row per line.
    Returns a list-of-rows (strings for autumn, ints for ARC) or None."""
    start = obs.find("[[")
    end = obs.rfind("]]")
    if start >= 0 and end >= 0:
        try:
            return json.loads(obs[start : end + 2])
        except Exception:
            pass
    return _parse_arc_grid(obs)


def _parse_arc_grid(obs):
    """Parse the first ``<grid_N>`` block of an ARC-AGI-3 observation into a 2D int grid."""
    if "<grid_" not in obs:
        return None
    m = re.search(r"<grid_\d+>(.*?)</grid_\d+>", obs, re.S)
    body = m.group(1) if m else obs
    rows = []
    for line in re.findall(r"\[[\d,\s]+\]", body):
        try:
            rows.append([int(x) for x in line.strip()[1:-1].split(",") if x.strip()])
        except ValueError:
            continue
    if not rows:
        return None
    w = len(rows[0])
    if w == 0 or any(len(r) != w for r in rows):
        return None
    return rows


def _bg_color(grid):
    """Background token = the most common cell value ('black' for autumn, the modal
    palette index for ARC). Used to render the canvas fill and to drop background cells
    from the compact payload."""
    flat = [str(v) for row in grid for v in row]
    if not flat:
        return "black"
    return Counter(flat).most_common(1)[0][0]


def cells_of(grid, predicate):
    out = []
    if not grid:
        return out
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if predicate(v):
                out.append((r, c))
    return out


def movement(x_t, x_t1):
    """Classify a transition by its OBSERVABLE on-grid effect (the only thing any P can see):
      kind = nochange | move | conv | ungrid ; vec = (drow,dcol) of the cell that moved ;
      newly_green = a cell converted gray->green. Used to render the per-row move vector."""
    g1, g2 = parse_grid(x_t), parse_grid(x_t1)
    if g1 is None or g2 is None or len(g1) != len(g2):
        return {"vec": None, "newly_green": False, "kind": "ungrid", "sig": "ungrid"}
    bg = _bg_color(g1)  # 'black' for autumn; modal palette index for ARC
    diffs = [((r, c), g1[r][c], g2[r][c])
             for r in range(len(g1)) for c in range(len(g1[0])) if g1[r][c] != g2[r][c]]
    if not diffs:
        return {"vec": None, "newly_green": False, "kind": "nochange", "sig": "nochange"}
    newly_green = any(str(a).lower() == "gray" and "green" in str(b).lower() for _, a, b in diffs)
    vacated = [k for k, a, b in diffs if str(b) == bg]
    filled = [k for k, a, b in diffs if str(a) == bg]
    vec = None
    if len(vacated) == 1 and len(filled) == 1:
        vec = (filled[0][0] - vacated[0][0], filled[0][1] - vacated[0][1])
    if newly_green:
        return {"vec": vec, "newly_green": True, "kind": "conv", "sig": "conv"}
    return {"vec": vec, "newly_green": False, "kind": "move", "sig": f"move{vec}"}


def window_payload(tr):
    """Ordered K-step window for the side panel: the raw state grids X_t-K..X_t+K with the
    action between each pair. target_idx marks the action a_t that inverse dynamics must
    recover (it sits between X_t and X_t+1). Returns None if no context attached."""
    if not getattr(tr, "ctx_prev", None) and not getattr(tr, "ctx_next", None):
        return None
    states, actions = [], []
    npre = len(tr.ctx_prev)
    # prev states X_t-K..X_t-1 (oldest->newest) with their outgoing actions
    for k, (st, act) in enumerate(tr.ctx_prev):
        states.append({"label": f"t-{npre - k}", "grid": grid_summary(st)})
        actions.append({"action": verb(act), "target": False})
    # center: X_t --a_t--> X_t+1
    states.append({"label": "t", "grid": grid_summary(tr.x_t)})
    actions.append({"action": verb(tr.action), "target": True})
    states.append({"label": "t+1", "grid": grid_summary(tr.x_t1)})
    target_idx = npre  # action index between X_t and X_t+1
    # next states X_t+2..X_t+K, each reached by its incoming action
    for k, (act, st) in enumerate(tr.ctx_next):
        actions.append({"action": verb(act), "target": False})
        states.append({"label": f"t+{k + 2}", "grid": grid_summary(st)})
    return {"states": states, "actions": actions, "target_idx": target_idx}


def grid_summary(obs):
    """Compact colored-cell render payload for the tooltip (only non-background cells).
    Works for both autumn color-name grids and ARC integer grids; `bg` is the canvas
    fill the renderer paints under the stored cells."""
    g = parse_grid(obs)
    if not g:
        return None
    h, w = len(g), len(g[0]) if g else 0
    bg = _bg_color(g)
    cells = []
    for r, row in enumerate(g):
        for c, v in enumerate(row):
            if str(v) != bg:
                cells.append([r, c, str(v)])
    return {"h": h, "w": w, "cells": cells, "bg": bg}


# --------------------------------------------------------------------------------------
# Decoder-prompt reconstruction. The test traces store P's features, the baked choices and
# the decoder response, but NOT the prompt. The prompt is deterministic given (P code,
# beliefs, transition, choices), so we rebuild it here with templates copied VERBATIM from
# invdyn_core.py (DEFAULT_KNOWLEDGE, INV_WINDOW_TMPL, _inverse_transcript) and
# validate_beliefs.py (PREDICT_TMPL) plus validate.py's run_perceive. Exactness is verified
# per row: re-running the stored P on the center states must reproduce the stored z_t/z_t1
# (surfaced in the panel as best_prompt_exact).
# --------------------------------------------------------------------------------------
DEFAULT_KNOWLEDGE = (
    "The available actions are: up, down, left, right, noop, and click ROW COL, where ROW is the "
    "row index and COL is the column index (both 0-indexed; this matches the (row, col) order the "
    "perception reports cells in)."
)

PREDICT_TMPL = """You identify which action was taken between two states of a grid environment.

=== WORLD KNOWLEDGE (general facts about this environment; may be empty) ===
{beliefs}
=== END WORLD KNOWLEDGE ===

Two consecutive states are summarized by a perception module. Exactly one action was taken to get from STATE 1 to STATE 2.

=== STATE 1 features ===
{z_t}
=== STATE 2 features ===
{z_t1}

The action was one of:
{choices}

Use the world knowledge to interpret the features (e.g. coordinate conventions). Decide which single action was taken, using ONLY the features above. If undecidable, say what is missing and make your best guess.

Respond as:
<reasoning>what changed and what action that implies</reasoning>
<action>the chosen action, copied verbatim from the list</action>"""

INV_WINDOW_TMPL = """You identify a single HIDDEN action in a trajectory of a grid environment.

=== DEFAULT KNOWLEDGE (always-true facts about this environment) ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== WORLD KNOWLEDGE (general facts; may be empty) ===
{beliefs}
=== END WORLD KNOWLEDGE ===

Below is a trajectory of consecutive states (summarized as features by a perception module) with the action taken between each pair. ONE action is hidden and marked `??? (IDENTIFY THIS)`. Use the WHOLE trajectory -- the states and actions BEFORE and AFTER the gap -- to infer the hidden action; temporally-extended patterns (a selected/active object, momentum, periodicity) may only be visible across several steps.

{transcript}

The hidden action was one of:
{choices}

Respond as:
<reasoning>what the surrounding states and actions imply about the hidden action</reasoning>
<action>the chosen action, copied verbatim from the list</action>"""


def run_perceive(code, raw_obs):
    """Verbatim from validate.py: exec the perception module and run perceive([obs])."""
    if not code.strip():
        return "", None
    try:
        ns = {}
        exec(code, ns)  # noqa: S102 -- run artifacts' own perception code, as at eval time
        fn = ns.get("perceive")
        if not callable(fn):
            return "", "no callable perceive()"
        out = fn([raw_obs])
        return (out if isinstance(out, str) else str(out)), None
    except Exception as e:  # noqa: BLE001
        return "", f"{type(e).__name__}: {e}"


def _tlabel(i):
    return "t" if i == 0 else (f"t{i}" if i < 0 else f"t+{i}")


def _inverse_transcript(win):
    """Verbatim from rexpure_optimize.py: states X_t-K..X_t+K, a_t masked."""
    lines, n_prev = [], len(win["prev"])
    for k, (z, a) in enumerate(win["prev"]):
        idx = -(n_prev - k)
        lines.append(f"STATE[{_tlabel(idx)}] features:\n{z or '(empty)'}")
        lines.append(f"  action({_tlabel(idx)} -> {_tlabel(idx + 1)}): {a}")
    lines.append(f"STATE[t] features:\n{win['z_t'] or '(empty)'}")
    lines.append("  action(t -> t+1): ??? (IDENTIFY THIS)")
    lines.append(f"STATE[t+1] features:\n{win['z_t1'] or '(empty)'}")
    for k, (a, z) in enumerate(win["nxt"]):
        lines.append(f"  action({_tlabel(k + 1)} -> {_tlabel(k + 2)}): {a}")
        lines.append(f"STATE[{_tlabel(k + 2)}] features:\n{z or '(empty)'}")
    return "\n".join(lines)


def feature_window(code, tr):
    """Run P over the whole temporal window (mirrors rexpure_optimize.build_window, features only)."""
    prev = [(run_perceive(code, raw)[0], a) for raw, a in tr.ctx_prev]
    nxt = [(a, run_perceive(code, raw)[0]) for a, raw in tr.ctx_next]
    return {
        "prev": prev,
        "z_t": run_perceive(code, tr.x_t)[0],
        "z_t1": run_perceive(code, tr.x_t1)[0],
        "nxt": nxt,
    }


def reconstruct_test_prompts(r, raw, tr_i, best_code, best_beliefs, context_k):
    """(best_prompt, best_prompt_exact, raw_prompt) for one held-out test record.

    best: windowed runs re-run the stored P over the window (exactness checked against the
    stored center features); context_k==0 runs rebuild from the stored z_t/z_t1 verbatim.
    raw: rebuilt verbatim from the stored truncated raw frames + choices (beliefs were empty).
    Image-mode records ('<image>') carry prompts we cannot rebuild as text -> empty."""
    choices = r.get("choices") or []
    fmt_choices = "\n".join(f"- {c}" for c in choices)
    best_prompt, best_exact = "", None
    if tr_i is not None and choices and r.get("z_t") != "<image>":
        if context_k > 0:
            win = feature_window(best_code, tr_i)
            best_prompt = INV_WINDOW_TMPL.format(
                beliefs=best_beliefs.strip() or "(empty)",
                default_knowledge=DEFAULT_KNOWLEDGE,
                transcript=_inverse_transcript(win),
                choices=fmt_choices,
            )
            best_exact = (str(win["z_t"]) == str(r.get("z_t") or "")
                          and str(win["z_t1"]) == str(r.get("z_t1") or ""))
        else:
            best_prompt = PREDICT_TMPL.format(
                beliefs=best_beliefs.strip() or "(empty)",
                z_t=r.get("z_t") or "(empty)",
                z_t1=r.get("z_t1") or "(empty)",
                choices=fmt_choices,
            )
            best_exact = True
    raw_prompt = ""
    if raw and raw.get("choices") and raw.get("z_t") != "<image>":
        raw_prompt = PREDICT_TMPL.format(
            beliefs="(empty)",
            z_t=raw.get("z_t") or "(empty)",
            z_t1=raw.get("z_t1") or "(empty)",
            choices="\n".join(f"- {c}" for c in raw["choices"]),
        )
    return best_prompt, best_exact, raw_prompt


# --------------------------------------------------------------------------------------
# Read the run artifacts
# --------------------------------------------------------------------------------------
def _extract_js_array(text, start):
    """Return the JSON array literal beginning at the '[' at/after `start`, found by
    bracket-counting while respecting string literals + escapes. Robust to candidate
    text that contains ']'/'];' (which a non-greedy regex truncates on)."""
    i = text.index("[", start)
    depth, in_str, esc = 0, False, False
    for j in range(i, len(text)):
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[i:j + 1]
    raise ValueError("unterminated array")


def load_candidates(gdir):
    """Prefer the rich NODES embedded in candidate_tree.html (idx/score/parents/role/
    components, historical runs); else the rex_search candidates.jsonl (idx/train_score/
    parents + component keys); else the bare candidates.json (position == idx)."""
    tree = gdir / "candidate_tree.html"
    if tree.exists():
        txt = tree.read_text()
        k = txt.find("const NODES")
        if k != -1:
            try:
                return json.loads(_extract_js_array(txt, k))
            except (ValueError, json.JSONDecodeError):
                pass
    jl = gdir / "candidates.jsonl"
    if jl.exists():
        out = []
        for line in jl.open():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            meta = {"idx", "parents", "train_score", "expansions"}
            out.append({
                "idx": rec.get("idx"),
                "score": rec.get("train_score"),
                "parents": rec.get("parents", "?"),
                "role": "",
                "components": {k: v for k, v in rec.items() if k not in meta},
            })
        return out
    bare = json.load((gdir / "candidates.json").open())
    return [{"idx": i, "score": None, "parents": "?", "role": "", "components": c}
            for i, c in enumerate(bare)]


def load_runlog(gdir):
    p = gdir / "run_log.json"
    return json.load(p.open()) if p.exists() else []


def load_process_log(gdir):
    """Per-iteration search record incl. REJECTED proposals + the mistake feedback that
    produced each candidate (written by ProcessLogger in invdyn_core.py). Empty for runs
    that predate the logger; the viz then falls back to the bare accepted-only timeline."""
    p = gdir / "process_log.jsonl"
    out = []
    if p.exists():
        for line in p.open():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def load_reflection_calls(gdir):
    """The proposer (reflection-LM) calls -- full prompt + response per proposal --
    written by _log_reflection in invdyn_core.py (one JSON object per call, in
    iteration order). Returns {component: [calls in order]} so a proposal node can be
    matched to the exact call that produced it by per-component order (verified 1:1
    against process_log: round_robin proposes one component per iteration)."""
    p = gdir / "reflection_calls.jsonl"
    by_comp = {}
    if p.exists():
        for line in p.open():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            by_comp.setdefault(r.get("component"), []).append(r)
    return by_comp


def load_analysis_calls(gdir):
    """The --analyze-mistakes diagnosis calls -- full prompt + raw response per call --
    written by _log_analysis in invdyn_core.py. Each record is stamped with the
    `iteration` it was built for, so we key by (iteration -> [calls]) and attach to the
    node with the matching i. Empty for runs that predate the logger or ran without
    --analyze-mistakes; the viz then simply omits the analysis section."""
    p = gdir / "analysis_calls.jsonl"
    by_iter = {}
    if p.exists():
        for line in p.open():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            by_iter.setdefault(r.get("iteration"), []).append(r)
    return by_iter


def load_valset_predictions(gdir):
    """task_idx -> {prog_idx -> prediction}.  Latest iter wins if a prog repeats."""
    root = gdir / "generated_best_outputs_valset"
    preds = defaultdict(dict)
    order = defaultdict(dict)  # prog -> iter (to keep latest)
    for task_dir in root.glob("task_*"):
        tidx = int(task_dir.name.split("_")[1])
        for f in task_dir.glob("iter_*_prog_*.json"):
            parts = f.stem.split("_")
            it, prog = int(parts[1]), int(parts[3])
            if prog not in order[tidx] or it >= order[tidx][prog]:
                preds[tidx][prog] = json.load(f.open())
                order[tidx][prog] = it
    return preds


def content_hash(*parts):
    """MUST match InvDynAdapter._content_hash in invdyn_core.py so the per-prediction
    sidecar joins to the reconstructed split + candidate components."""
    h = hashlib.md5()
    for p in parts:
        h.update((p or "").encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def load_predictions(gdir):
    """(cand_hash, tr_hash) -> full prediction detail (inverse + forward), last write wins.
    Empty when the run predates the sidecar; the viewer then degrades gracefully."""
    p = gdir / "predictions.jsonl"
    out = {}
    if p.exists():
        for line in p.open():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[(r.get("cand_hash"), r.get("tr_hash"))] = r
    return out


def verb(s):
    return (str(s).strip().split() or [""])[0]


def pred_label(pred):
    """Human-readable predicted action(s). Runs launched with --id-set-loss log `pred`
    as an ordered list (a subset of the choices) instead of a single action string;
    verb() on str(list) would mangle it (e.g. "['noop'," from splitting the repr)."""
    if isinstance(pred, (list, tuple)):
        return "{" + ", ".join(str(p) for p in pred) + "}"
    return verb(pred)


def pred_hit(pred, truth):
    """Whether `truth` is consistent with `pred` -- set membership for --id-set-loss
    runs (pred is a list), exact verb match otherwise."""
    if isinstance(pred, (list, tuple)):
        return truth in pred
    return verb(pred) == truth


def load_runlog_scores(runlog):
    """Complete per-instance score matrix from run_log.json: cand_idx -> {val_idx -> composite score}.
    Every candidate (as the iteration's selected parent OR its newborn) has a score for every val
    index it was evaluated on -- unlike generated_best_outputs_valset, which only stores improvements."""
    scores = defaultdict(dict)
    for it in runlog:
        ids = it.get("subsample_ids", []) or []
        sp = it.get("selected_program_candidate")
        ps = it.get("subsample_scores", []) or []
        nb = it.get("new_program_idx")
        ns = it.get("new_subsample_scores", []) or []
        for k, vi in enumerate(ids):
            if sp is not None and k < len(ps):
                scores[sp][vi] = ps[k]
            if nb is not None and k < len(ns):
                scores[nb][vi] = ns[k]
    return scores


# --------------------------------------------------------------------------------------
# Assemble the model the HTML renders
# --------------------------------------------------------------------------------------
def build_model(args):
    gdir = Path(args.run_dir)
    val, test, action_pool = reconstruct_split(args)
    candidates = load_candidates(gdir)
    runlog = load_runlog(gdir)
    valpreds = load_valset_predictions(gdir)
    preds = load_predictions(gdir)  # (cand_hash, tr_hash) -> inverse+forward detail
    # prog_idx -> content hash of its components (to join the per-prediction sidecar)
    cand_hash_by_prog = {
        c["idx"]: content_hash(
            c["components"].get("perception", ""),
            c["components"].get("world_knowledge", ""),
        )
        for c in candidates
    }

    # ---- verification anchor: reconstructed test truths must match the saved test_trace ----
    # accept both the historical gepa naming and the rex_search naming.
    trace_path = (next(gdir.parent.glob("test_trace_gepa_seed*.json"), None)
                  or next(gdir.parent.glob("test_trace_rexpure_seed*.json"), None))
    # Raw-frame-mode baseline trace on the SAME held-out test set (decoder sees the raw
    # observation, no learned perception). Joined per-transition to compare against the best
    # candidate. Aligned by idx (both traces are baked from the identical test split/order).
    raw_trace_path = next(gdir.parent.glob("test_trace_raw_seed*.json"), None)
    raw_by_idx, raw_acc = {}, None
    if raw_trace_path:
        rawtr = json.load(raw_trace_path.open())
        raw_acc = rawtr.get("acc")
        raw_by_idx = {r["idx"]: r for r in rawtr.get("records", [])}
    verify = {"ok": None, "detail": ""}
    test_records = []
    test_acc = None
    if trace_path:
        tr = json.load(trace_path.open())
        test_acc = tr.get("acc")
        recs = tr.get("records", [])
        best_code = tr.get("perception") or ""
        best_beliefs = tr.get("beliefs") or ""
        recon = [verb(t["tr"].action) for t in test]
        saved = [verb(r["truth"]) for r in recs]
        verify["ok"] = recon == saved
        verify["detail"] = f"reconstructed test truths {'==' if verify['ok'] else '!='} test_trace truths"
        for i, r in enumerate(recs):
            tr_i = test[i]["tr"] if i < len(test) else None
            raw = raw_by_idx.get(r["idx"])
            best_prompt, best_exact, raw_prompt = reconstruct_test_prompts(
                r, raw, tr_i, best_code, best_beliefs, getattr(args, "context_k", 0))
            test_records.append({
                "idx": r["idx"], "truth": r["truth"], "pred": pred_label(r["pred"]), "correct": r["correct"],
                "z_t": r.get("z_t", ""), "z_t1": r.get("z_t1", ""), "reasoning": r.get("reasoning", ""),
                "choices": r.get("choices") or [],
                "best_prompt": best_prompt, "best_prompt_exact": best_exact,
                "raw_prompt": raw_prompt,
                "raw_pred": (pred_label(raw.get("pred")) if raw and raw.get("pred") is not None else None),
                "raw_correct": raw.get("correct") if raw else None,
                "raw_reasoning": raw.get("reasoning", "") if raw else "",
                "raw_z_t": raw.get("z_t", "") if raw else "",
                "grid_t": grid_summary(tr_i.x_t) if tr_i else None,
                "grid_t1": grid_summary(tr_i.x_t1) if tr_i else None,
                "move": movement(tr_i.x_t, tr_i.x_t1) if tr_i else None,
                "window": window_payload(tr_i) if tr_i else None,
            })

    # ---- val/train transitions with movement ----
    vinfo = []
    for i, item in enumerate(val):
        tr = item["tr"]
        mv = movement(tr.x_t, tr.x_t1)
        vinfo.append({
            "task": i, "action": verb(tr.action), "move": mv,
            "tr_hash": content_hash(tr.x_t, tr.x_t1, tr.action),
            "grid_t": grid_summary(tr.x_t), "grid_t1": grid_summary(tr.x_t1),
            "window": window_payload(tr),
        })

    # ---- complete per-cell matrix: score from run_log (authoritative, dense),
    #      predicted action overlaid from the valset store where it exists --------
    rl_scores = load_runlog_scores(runlog)
    cand_cols = sorted(rl_scores) or sorted({pp for d in valpreds.values() for pp in d})
    cells = {}  # (task, prog) -> {score, ok, pred, src}
    for task in range(len(vinfo)):
        truth = vinfo[task]["action"]
        for prog in cand_cols:
            score = rl_scores.get(prog, {}).get(task)
            pred = valpreds.get(task, {}).get(prog)
            if pred is not None:                       # exact: we know the predicted action
                ok, label, src = pred_hit(pred, truth), pred_label(pred), "pred"
            elif score is not None:                    # derive from composite: >0.5 correct, <0.5 wrong
                if score > 0.5 + 1e-9:
                    ok, src = True, "score"
                elif score < 0.5 - 1e-9:
                    ok, src = False, "score"
                else:
                    ok, src = None, "ambig"            # exactly 0.5 -> ID undetermined
                label = f"{score:.2f}"
            else:
                continue                               # genuinely not evaluated
            cell = {"ok": ok, "label": label, "pred": (pred_label(pred) if pred else None),
                    "score": score, "src": src}
            # full prediction detail (inverse reasoning + forward prediction), if logged
            det = preds.get((cand_hash_by_prog.get(prog), vinfo[task]["tr_hash"]))
            if det:
                det_pred = det.get("pred")
                cell["det"] = {
                    "pred": (pred_label(det_pred) if det_pred is not None else None),
                    "id_score": det.get("id_score"),
                    "reasoning": det.get("reasoning", ""),
                    "z_t": det.get("z_t", ""), "z_t1": det.get("z_t1", ""),
                    "z_hat": det.get("z_hat", ""), "fd_score": det.get("fd_score"),
                    "fd_reasoning": det.get("fd_reasoning", ""),
                    "score": det.get("score"),
                    # full prompt/response pairs F saw for each objective (windowed runs)
                    "inv_prompt": det.get("inv_prompt", ""), "inv_response": det.get("inv_response", ""),
                    "fwd_prompt": det.get("fwd_prompt", ""), "fwd_response": det.get("fwd_response", ""),
                }
            cells[f"{task},{prog}"] = cell

    # ---- candidate metadata ----
    cand_meta = []
    for c in candidates:
        cand_meta.append({
            "idx": c["idx"], "score": c.get("score"), "parents": c.get("parents"),
            "role": c.get("role", ""),
            "perception": c["components"].get("perception", ""),
            "world_knowledge": c["components"].get("world_knowledge", ""),
        })

    # ---- iteration timeline ----
    timeline = []
    for it in runlog:
        ns = it.get("new_subsample_scores") or []
        timeline.append({
            "i": it.get("i"), "selected": it.get("selected_program_candidate"),
            "born": it.get("new_program_idx"),
            "val_mean": round(sum(ns) / len(ns), 3) if ns else None,
        })

    # ---- search PROCESS tree (incl. rejected dead ends + mistake feedback) -------------
    # Each logged iteration becomes a node hanging off the parent it was proposed from:
    # accepted -> a real candidate node (cand N); rejected/skipped -> a dead-end leaf. Every
    # node carries the proposer's response (new component text) and the feedback shown to it.
    proc = load_process_log(gdir)
    refl_by_comp = load_reflection_calls(gdir)  # {component: [proposer calls in order]}
    refl_ptr = {}  # component -> next-call index (per-component order matches process_log)
    analysis_by_iter = load_analysis_calls(gdir)  # {iteration: [diagnosis calls]}
    cand_text = {c["idx"]: c["components"] for c in candidates}  # parent text -> for diffing

    tree_nodes = [{
        "id": "c0", "kind": "seed", "i": 0, "parent": None, "cand": 0,
        "comp": [], "proposed": {},
        "label": "cand 0", "sub": "seed",
    }]
    seen_cand = {0}
    for rec in proc:
        sel = rec.get("selected")
        parent = f"c{sel}" if sel is not None else "c0"
        comps = rec.get("components") or list((rec.get("proposed") or {}).keys())
        verdict = rec.get("verdict", "skipped")
        # match this iteration's proposal to the exact proposer call that produced it
        # (per-component order; round_robin -> one component proposed per iteration).
        proposer = {}
        for comp in comps:
            calls = refl_by_comp.get(comp) or []
            idx = refl_ptr.get(comp, 0)
            if idx < len(calls):
                c = calls[idx]
                proposer[comp] = {
                    "prompt": c.get("prompt", ""), "response": c.get("response", ""),
                    "attempts": c.get("attempts"), "ts": c.get("ts"),
                }
                refl_ptr[comp] = idx + 1
        node = {
            "i": rec.get("i"), "parent": parent, "kind": verdict,
            "selected": sel, "selected_score": rec.get("selected_score"),
            "comp": comps,
            "proposed": rec.get("proposed") or {},
            "parent_text": cand_text.get(sel, {}),
            "proposer": proposer,
            "analysis": analysis_by_iter.get(rec.get("i"), []),
            "old_score": rec.get("old_score"), "new_score": rec.get("new_score"),
            "reason": rec.get("reason"),
        }
        if verdict == "accepted" and rec.get("new_idx") is not None:
            ni = rec["new_idx"]
            node.update({"id": f"c{ni}", "cand": ni,
                         "label": f"cand {ni}", "sub": "+".join(comps) or "accepted"})
            seen_cand.add(ni)
        else:
            node.update({"id": f"x{rec.get('i')}", "cand": None,
                         "label": f"i{rec.get('i')} ✗", "sub": verdict})
        tree_nodes.append(node)

    return {
        "title": gdir.parent.name,
        "verify": verify,
        "action_pool": action_pool,
        "val": vinfo,
        "cand_cols": cand_cols,
        "cells": cells,
        "candidates": cand_meta,
        "timeline": timeline,
        "tree": tree_nodes,
        "test": test_records,
        "test_acc": test_acc,
        "raw_acc": raw_acc,
    }


# --------------------------------------------------------------------------------------
# HTML (single file, vanilla JS, data embedded)
# --------------------------------------------------------------------------------------
HTML = r"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Optimisation viewer — __TITLE__</title>
<style>
 *{box-sizing:border-box} body{margin:0;font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#f6f7f9;color:#1d2129}
 header{background:#fff;border-bottom:1px solid #dde;padding:12px 20px}
 h1{font-size:17px;margin:0} h2{font-size:14px;margin:22px 0 8px;color:#344}
 .wrap{padding:16px 20px;max-width:1500px}
 .verify{font-size:12px;padding:3px 8px;border-radius:4px;display:inline-block;margin-top:6px}
 .v-ok{background:#e6f4ea;color:#137333} .v-bad{background:#fce8e6;color:#c5221f}
 table{border-collapse:collapse;font-size:12px;background:#fff}
 .matrix td,.matrix th{border:1px solid #e6e8eb;padding:0;text-align:center}
 .matrix th{background:#f1f3f4;padding:5px 7px;font-weight:600;position:sticky;top:0}
 .rowhdr{padding:5px 8px !important;text-align:left !important;white-space:nowrap;cursor:pointer}
 .cell{width:54px;height:30px;cursor:pointer;font-size:11px}
 .ok{background:#cde6d4;color:#0b5} .bad{background:#f6cdcf;color:#a00} .na{background:#f0f0f0;color:#bbb} .ambig{background:#fff2cc;color:#a80}
 .tag{display:inline-block;font-size:10px;padding:1px 5px;border-radius:3px;margin-left:5px}
 .t-conv{background:#cfe8ff;color:#06c}
 .legend{font-size:12px;color:#556;margin:6px 0}
 .cands{display:flex;flex-wrap:wrap;gap:10px}
 .card{background:#fff;border:1px solid #dde;border-radius:6px;padding:10px;width:340px;font-size:12px}
 .card h3{margin:0 0 4px;font-size:13px} .score{float:right;font-weight:700;color:#06c}
 pre{background:#f6f8fa;border:1px solid #eee;border-radius:4px;padding:7px;font-size:11px;white-space:pre-wrap;max-height:160px;overflow:auto;margin:5px 0}
 .role{font-size:10px;padding:1px 6px;border-radius:3px;margin-left:6px}
 .r-Best{background:#cdf;color:#039} .r-ParetoFront{background:#fe9;color:#a60}
 .tl{font-size:12px} .tl td,.tl th{border:1px solid #e6e8eb;padding:4px 9px}
 #tip{position:fixed;display:none;background:#fff;border:1px solid #ccd;border-radius:8px;box-shadow:0 6px 24px rgba(0,0,0,.18);
      padding:12px;max-width:560px;z-index:99;font-size:12px}
 .grids{display:flex;gap:14px;align-items:flex-start;margin-top:6px}
 .gtitle{font-size:11px;color:#667;margin-bottom:3px} .gwrap{display:inline-block;border:1px solid #ddd}
 canvas{display:block;image-rendering:pixelated}
 .mv{font-family:monospace;font-size:11px;color:#334}
 .cell.has-det{position:relative} .cell.has-det::after{content:'';position:absolute;top:2px;right:2px;width:0;height:0;border-top:5px solid #06c5;border-left:5px solid transparent}
 #panel{position:fixed;right:0;top:0;width:560px;height:100%;background:#fff;border-left:1px solid #ccd;box-shadow:-6px 0 24px rgba(0,0,0,.14);
        padding:16px 18px;overflow:auto;display:none;z-index:120;font-size:12px}
 #panel h3{margin:0 0 2px;font-size:14px} #panel .sub{color:#778;margin-bottom:10px}
 #panel .sec{margin:12px 0;border:1px solid #e6e8eb;border-radius:6px;padding:10px}
 #panel .sec h4{margin:0 0 6px;font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#456}
 #panel .kv{margin:3px 0} #panel .ok2{color:#137333;font-weight:600} #panel .bad2{color:#c5221f;font-weight:600}
 #panel pre{background:#f6f8fa;border:1px solid #eee;border-radius:4px;padding:7px;font-size:11px;white-space:pre-wrap;max-height:240px;overflow:auto;margin:4px 0}
 #panel .close{float:right;cursor:pointer;border:1px solid #ccd;border-radius:4px;padding:2px 9px;background:#f6f7f9}
 .si-inv{border-left:3px solid #06c} .si-fwd{border-left:3px solid #2a8}
 .winrow{display:flex;align-items:flex-start;gap:2px;overflow-x:auto;padding:4px 0}
 .wincol{display:flex;flex-direction:column;align-items:center;flex:0 0 auto}
 .wincol .gtitle{font-size:10px;color:#667;margin-bottom:2px;white-space:nowrap}
 .winarrow{flex:0 0 auto;align-self:center;text-align:center;font-size:11px;color:#789;padding:0 1px;line-height:1.1}
 .winarrow span{font-size:9px;color:#789} .winarrow.wintarget{color:#06c;font-weight:700}
 .winarrow.wintarget span{color:#06c} .winmask{color:#c5221f;font-weight:700}
 details.pr{margin:6px 0} details.pr summary{cursor:pointer;font-size:11px;color:#06c;user-select:none}
 #tree{font-size:12px;line-height:1.2} #tree ul{list-style:none;margin:0;padding-left:18px;border-left:1px dotted #ccd}
 #tree>ul{padding-left:0;border-left:none} #tree li{margin:3px 0}
 .tnode{cursor:pointer;border:1px solid #ccd;border-radius:4px;padding:2px 8px;font-size:11px;display:inline-block;background:#fff}
 .tnode:hover{box-shadow:0 1px 5px rgba(0,0,0,.15)} .tnode .ts{color:#889;font-size:10px;margin-left:6px}
 .tk-seed{background:#eef;border-color:#aac;font-weight:600}
 .tk-accepted{background:#e6f4ea;border-color:#a8d8b9;color:#137333}
 .tk-rejected{background:#fce8e6;border-color:#f3b7b2;color:#c5221f}
 .tk-skipped{background:#f1f3f4;border-color:#ddd;color:#888}
 .deltaup{color:#137333;font-weight:600} .deltadn{color:#c5221f;font-weight:600}
 #panel .fbrow{border-bottom:1px solid #eee;padding:6px 0} #panel .fbrow:last-child{border:none}
 #testbody tr{cursor:pointer} #testbody tr:hover .rowhdr{background:#eef4ff}
</style></head><body>
<header><h1>Optimisation viewer — <span id="ttl"></span></h1>
<div id="verify"></div></header>
<div class="wrap">
 <h2 style="margin-top:0">Optimisation progress</h2>
 <div class="legend">Every search iteration's newly proposed candidate, scored on that iteration's minibatch (not the full val set) — points are noisy step to step.
  <span style="color:#06c">● accepted</span> into the candidate pool, <span style="color:#c7cbd1">● rejected/skipped</span>. The bold line tracks the best accepted score seen so far.</div>
 <div id="progresschart"></div>

 <div class="legend">Headline view: rows = val transitions, columns = <b>every candidate</b> produced over the whole run (not one iteration), in birth order.
  Green = action predicted correctly, red = wrong, <span style="background:#fff2cc">amber</span> = composite exactly 0.5 (action undetermined), gray = not evaluated.
  Cells show the predicted action where it was stored, otherwise the composite score (>0.5 ⟹ correct action). Coverage is complete because it's read from <code>run_log.json</code>'s per-instance scores.
  Cells with a <span style="color:#06c">▸ blue corner</span> have a logged prediction — <b>click</b> to see the inverse-dynamics chain (predicted action + reasoning) and the forward prediction (predicted next features vs true, fd_score) that produced the score. Hover any row header or cell.</div>

 <h2>Transition × Candidate matrix</h2>
 <table class="matrix"><thead><tr><th class="rowhdr">transition (true action)</th><th>move</th><th id="colhdr"></th></tr></thead>
 <tbody id="mbody"></tbody></table>

 <h2>Iteration timeline</h2>
 <table class="tl"><thead><tr><th>iter</th><th>selected parent</th><th>candidate born</th><th>full-val mean</th></tr></thead><tbody id="tlbody"></tbody></table>

 <h2>Search process tree</h2>
 <div class="legend" id="treelegend">Every proposal over the whole run, hung off the parent candidate it was mutated from.
  <span class="tnode tk-accepted" style="position:static;display:inline-block">cand N</span> = accepted (passed the subsample gate);
  <span class="tnode tk-rejected" style="position:static;display:inline-block">iN ✗</span> = <b>rejected dead end</b> (proposed but not better on the subsample);
  <span class="tnode tk-skipped" style="position:static;display:inline-block">iN ✗</span> = skipped (empty/failed proposal).
  <b>Click any node</b> to see the proposer's response (new component text) and the full proposer call (prompt + response) that produced it.</div>
 <div id="tree"></div>

 <h2>Candidates (perception + world knowledge)</h2>
 <div class="cands" id="cands"></div>

 <h2>Held-out test transitions <span id="testacc" style="font-weight:400;font-size:12px;color:#556"></span></h2>
 <div class="legend">Same held-out test set decoded two ways: the <b>final best candidate</b> (learned perception) vs the <b>raw-frame baseline</b> (decoder sees the raw observation, no learned perception).
  <b>Click any row</b> to open the detail panel: the transition's images (with the K-step context window the decoder saw), plus the full prompt + response for the best-candidate prediction and for the raw-frame prediction. Hover a row header for a quick before/after.</div>
 <table class="matrix"><thead><tr>
   <th class="rowhdr">true action</th>
   <th>best pred</th><th>raw pred</th><th>move</th>
   <th>best candidate reasoning</th><th>raw-frame reasoning</th>
 </tr></thead><tbody id="testbody"></tbody></table>
</div>
<div id="tip"></div>
<div id="panel"></div>
<script>
const D = __DATA__;
const tip = document.getElementById('tip');
document.getElementById('ttl').textContent = D.title;
const vr = D.verify;
const vdiv = document.getElementById('verify');
if (vr.ok !== null){ vdiv.innerHTML = '<span class="verify '+(vr.ok?'v-ok':'v-bad')+'">alignment check: '+vr.detail+'</span>'; }

// Official ARC-AGI-3 palette (arc_agi/rendering.py COLOR_MAP), keyed by integer index.
const ARC = {0:'#FFFFFF',1:'#CCCCCC',2:'#999999',3:'#666666',4:'#333333',5:'#000000',
  6:'#E53AA3',7:'#FF7BCC',8:'#F93C31',9:'#1E93FF',10:'#88D8F1',11:'#FFDC00',
  12:'#FF851B',13:'#921231',14:'#4FCC30',15:'#A356D6'};
// AutumnBench cell tokens are full CSS color names (white, mediumpurple, gold,
// darkorange, tan, sandybrown, ...) emitted straight from the game renderer, so
// returning the name lets the canvas resolve the EXACT game color. ARC grids use
// integer palette indices instead. Only validated CSS names are returned; anything
// the browser can't parse falls back to magenta so it stands out as unmapped.
const _CSSPROBE = document.createElement('span').style;
const _CSSOK = {};
function isCssColor(s){
  if(s in _CSSOK) return _CSSOK[s];
  _CSSPROBE.color=''; _CSSPROBE.color=s;       // browser rejects invalid -> stays ''
  return _CSSOK[s] = _CSSPROBE.color!=='';
}
// Resolve a cell/background token to a color: numeric -> ARC palette, name -> CSS color.
function colorOf(tok){
  if(tok===undefined||tok===null) return 'black';
  const s=String(tok);
  if(/^\d+$/.test(s)) return ARC[+s]||'#e6e';
  return isCssColor(s) ? s : '#e6e';
}
function drawGrid(g, scale){
  if(!g) return null;
  // Shrink large frames (ARC is 64x64) so before/after pairs fit the tooltip/panel.
  const MAXDIM=300;
  let px = scale||10;
  if(g.w*px > MAXDIM) px = Math.max(2, Math.floor(MAXDIM/g.w));
  const cv = document.createElement('canvas'); cv.width=g.w*px; cv.height=g.h*px;
  const ctx = cv.getContext('2d');
  ctx.fillStyle = colorOf(g.bg!==undefined?g.bg:'black'); ctx.fillRect(0,0,cv.width,cv.height);
  for(const [r,c,col] of g.cells){ ctx.fillStyle = colorOf(col); ctx.fillRect(c*px,r*px,px,px); }
  if(g.w<=32 && px>=5){  // gridlines only when cells are big enough to be legible
    ctx.strokeStyle='#0003'; for(let i=0;i<=g.w;i++){ctx.beginPath();ctx.moveTo(i*px,0);ctx.lineTo(i*px,cv.height);ctx.stroke();}
    for(let i=0;i<=g.h;i++){ctx.beginPath();ctx.moveTo(0,i*px);ctx.lineTo(cv.width,i*px);ctx.stroke();}
  }
  return cv;
}
function gridsHtml(){ return ''; }
function showTip(html, grids, e){
  tip.innerHTML = html;
  if(grids){ const box=document.createElement('div'); box.className='grids';
    for(const [lbl,g] of grids){ const d=document.createElement('div'); const t=document.createElement('div'); t.className='gtitle'; t.textContent=lbl; d.appendChild(t);
      const cv=drawGrid(g,12); if(cv){const w=document.createElement('div');w.className='gwrap';w.appendChild(cv);d.appendChild(w);} box.appendChild(d);} tip.appendChild(box); }
  tip.style.display='block';
  let x=e.clientX+14, y=e.clientY+14;
  if(x+tip.offsetWidth>innerWidth) x=innerWidth-tip.offsetWidth-10;
  if(y+tip.offsetHeight>innerHeight) y=innerHeight-tip.offsetHeight-10;
  tip.style.left=x+'px'; tip.style.top=y+'px';
}
function hideTip(){ tip.style.display='none'; }

// ---- per-prediction detail panel (inverse-dynamics + forward-dynamics) ----
const panel=document.getElementById('panel');
function closePanel(){ panel.style.display='none'; }
function fmt(x){ return (x===null||x===undefined)?'?':(typeof x==='number'?x.toFixed(3):x); }
// collapsible full prompt/response pair (only rendered when the run logged them)
function promptResp(label, prompt, resp){
  if(!prompt && !resp) return '';
  return '<details class="pr"><summary>'+esc(label)+'</summary>'+
         '<div class="kv">PROMPT →</div><pre>'+esc(prompt||'(none)')+'</pre>'+
         '<div class="kv">← RESPONSE</div><pre>'+esc(resp||'(none)')+'</pre></details>';
}
// Render the K-step window as a horizontal strip of raw state grids with action labels
// between them. mode 'inverse' shows X_t-K..X_t+K with a_t masked (truth revealed in parens);
// mode 'forward' shows only the history X_t-K..X_t then a_t as the action being conditioned on.
function drawWindow(container, win, mode){
  if(!container || !win) return;
  container.innerHTML='';
  const box=document.createElement('div'); box.className='winrow';
  const ti=win.target_idx;
  const last = (mode==='forward') ? ti : win.states.length-1; // last STATE index to draw
  for(let i=0;i<=last;i++){
    const s=win.states[i];
    const col=document.createElement('div'); col.className='wincol';
    const lab=document.createElement('div'); lab.className='gtitle'; lab.textContent='X_'+s.label; col.appendChild(lab);
    const cv=drawGrid(s.grid,8); if(cv){const w=document.createElement('div');w.className='gwrap';w.appendChild(cv);col.appendChild(w);}
    box.appendChild(col);
    const a=win.actions[i];                       // action between state i and i+1
    const drawArrow = (i<last) || (mode==='forward' && i===ti);
    if(a && drawArrow){
      const ar=document.createElement('div'); ar.className='winarrow'+(a.target?' wintarget':'');
      const lbl = (a.target && mode==='inverse') ? '<span class="winmask">???</span><br><span>('+esc(a.action)+')</span>'
                                                 : '<span>'+esc(a.action)+'</span>';
      ar.innerHTML='→<br>'+lbl; box.appendChild(ar);
    }
  }
  if(mode==='forward'){ const q=document.createElement('div'); q.className='wincol';
    q.innerHTML='<div class="gtitle">X_t+1</div><div style="font-size:18px;color:#2a8">?</div>'; box.appendChild(q); }
  container.appendChild(box);
}
function openPanel(v,p,cell){
  hideTip();
  const d=cell.det;
  const hasWin = v.window && (v.window.states||[]).length>1;
  const winNote = hasWin ? ('<div class="kv" style="color:#778">raw state window X_t-K..X_t+K (what F saw, perceived through P). a_t is masked for the decoder.</div><div id="winInv"></div>') : '';
  let h='<span class="close" onclick="closePanel()">close ✕</span>'+
        '<h3>cand '+p+' on transition t'+v.task+'</h3>'+
        '<div class="sub">true action <b>'+v.action+'</b> · composite score shown in matrix: <b>'+fmt(cell.score)+'</b></div>';
  if(!d){
    h+=(hasWin?'<div class="sec si-inv"><h4>Trajectory window (K-step)</h4>'+winNote+'</div>':'')+
       '<div class="sec">No per-prediction detail was logged for this cell.<br><br>'+
       'The matrix value here is '+(cell.src==='pred'?'the stored predicted action':'derived from the composite score')+
       ', but the reasoning / forward prediction was not captured. Re-run with the updated '+
       '<code>rexpure_optimize.py</code> (writes <code>predictions.jsonl</code>) to populate this.</div>';
    panel.innerHTML=h; panel.style.display='block';
    if(hasWin) drawWindow(document.getElementById('winInv'), v.window, 'inverse');
    return;
  }
  const idok = d.id_score>=1.0;
  h+='<div class="sec si-inv"><h4>Inverse dynamics — recover the action</h4>'+
     winNote+
     '<div class="kv">predicted action: <b>'+esc(d.pred==null?'?':d.pred)+'</b> · truth <b>'+v.action+'</b> · '+
        '<span class="'+(idok?'ok2':'bad2')+'">id_score '+fmt(d.id_score)+(idok?' ✓ correct':' ✗ wrong')+'</span></div>'+
     '<div class="kv">P(X_t) features:</div><pre>'+esc(d.z_t||'(empty)')+'</pre>'+
     '<div class="kv">P(X_t+1) features:</div><pre>'+esc(d.z_t1||'(empty)')+'</pre>'+
     '<div class="kv">decoder reasoning:</div><pre>'+esc(d.reasoning||'(none)')+'</pre>'+
     promptResp('full prompt F saw (inverse)', d.inv_prompt, d.inv_response)+'</div>';
  if(d.z_hat){  // only when a forward prediction was actually made (fd_weight>0)
    h+='<div class="sec si-fwd"><h4>Forward dynamics — predict the next state</h4>'+
       (hasWin?'<div class="kv" style="color:#778">history window X_t-K..X_t conditioned on a_t to predict X_t+1:</div><div id="winFwd"></div>':'')+
       '<div class="kv">fd_score (match of predicted vs true next features): <b>'+fmt(d.fd_score)+'</b></div>'+
       '<div class="kv">predicted next features P̂(X_t+1) — from P(X_t)+action+B:</div><pre>'+esc(d.z_hat||'(none)')+'</pre>'+
       '<div class="kv">true next features P(X_t+1):</div><pre>'+esc(d.z_t1||'(empty)')+'</pre>'+
       (d.fd_reasoning?'<div class="kv">judge reasoning:</div><pre>'+esc(d.fd_reasoning)+'</pre>':'')+
       promptResp('full prompt F saw (forward)', d.fwd_prompt, d.fwd_response)+'</div>';
  }
  h+='<div class="sec"><h4>Composite</h4><div class="kv">score = (1−w)·id_score + w·fd_score = <b>'+fmt(d.score)+'</b></div></div>';
  panel.innerHTML=h; panel.style.display='block';
  if(hasWin){ drawWindow(document.getElementById('winInv'), v.window, 'inverse');
    if(d.z_hat) drawWindow(document.getElementById('winFwd'), v.window, 'forward'); }
}

// ---- matrix ----
const colhdrCell = document.querySelector('#colhdr');
const headRow = colhdrCell.parentElement;
colhdrCell.remove();
for(const p of D.cand_cols){ const th=document.createElement('th'); const c=D.candidates.find(c=>c.idx===p);
  th.innerHTML='cand '+p+(c&&c.role?'<div class="role r-'+c.role.replace(/\s/g,'')+'">'+c.role+'</div>':'');
  headRow.appendChild(th); }
const mbody=document.getElementById('mbody');
for(const v of D.val){
  const tr=document.createElement('tr');
  const hdr=document.createElement('td'); hdr.className='rowhdr';
  hdr.innerHTML='t'+v.task+' · <b>'+v.action+'</b>';
  hdr.onmousemove=e=>showTip('<b>transition t'+v.task+'</b> — true action <b>'+v.action+'</b><br>move vector (Δrow,Δcol): <span class="mv">'+JSON.stringify(v.move.vec)+'</span>',
    [['before',v.grid_t],['after',v.grid_t1]], e);
  hdr.onmouseleave=hideTip; tr.appendChild(hdr);
  const mv=document.createElement('td'); mv.className='mv'; mv.textContent=v.move.vec?('('+v.move.vec[0]+','+v.move.vec[1]+')'):'·'; tr.appendChild(mv);
  for(const p of D.cand_cols){ const td=document.createElement('td'); const k=v.task+','+p; const cell=D.cells[k];
    if(!cell){ td.className='cell na'; td.textContent='–'; }
    else{ const cls = cell.ok===true?'ok':(cell.ok===false?'bad':'ambig'); td.className='cell '+cls+(cell.det?' has-det':''); td.textContent=cell.label;
      const predtxt = cell.pred!=null ? ('predicted <b>'+cell.pred+'</b>') : ('composite score <b>'+(cell.score!=null?cell.score.toFixed(3):'?')+'</b> ('+(cell.ok===true?'action correct':cell.ok===false?'action wrong':'undetermined')+')');
      td.onmousemove=e=>showTip('cand '+p+' on t'+v.task+': '+predtxt+' · truth <b>'+v.action+'</b><br><span style="color:#889">source: '+cell.src+(cell.src!=='pred'?' (derived from composite ID+FD; exact action not stored)':'')+'</span>'+(cell.det?'<br><b style="color:#06c">click to see the prediction (inverse + forward) that produced this score</b>':''),null,e);
      td.onmouseleave=hideTip;
      td.onclick=()=>openPanel(v,p,cell); }
    tr.appendChild(td); }
  mbody.appendChild(tr);
}

// ---- optimisation progress chart (every iteration's minibatch score, best-so-far envelope) ----
(function(){
  const host = document.getElementById('progresschart');
  const pts = (D.timeline || []).filter(t => t.val_mean != null);
  if(!pts.length){ host.innerHTML = '<div style="color:#778;font-size:12px">No <code>run_log.json</code> for this run — nothing to plot.</div>'; return; }
  const NS = 'http://www.w3.org/2000/svg';
  const W = 1400, H = 200, ML = 40, MR = 12, MT = 10, MB = 24;
  const iMax = Math.max(...pts.map(p => p.i)), iMin = Math.min(...pts.map(p => p.i));
  const xOf = i => ML + (iMax > iMin ? (i - iMin) / (iMax - iMin) : 0) * (W - ML - MR);
  const yOf = v => MT + (1 - v) * (H - MT - MB);
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  svg.style.width = '100%'; svg.style.maxWidth = W + 'px'; svg.style.display = 'block';
  [0, .25, .5, .75, 1].forEach(v => {
    const y = yOf(v);
    const gl = document.createElementNS(NS, 'line');
    gl.setAttribute('x1', ML); gl.setAttribute('x2', W - MR); gl.setAttribute('y1', y); gl.setAttribute('y2', y);
    gl.setAttribute('stroke', '#e6e8eb'); gl.setAttribute('stroke-width', 1);
    svg.appendChild(gl);
    const lbl = document.createElementNS(NS, 'text');
    lbl.setAttribute('x', ML - 6); lbl.setAttribute('y', y + 3); lbl.setAttribute('text-anchor', 'end');
    lbl.setAttribute('font-size', '10'); lbl.setAttribute('fill', '#889');
    lbl.textContent = Math.round(v * 100) + '%';
    svg.appendChild(lbl);
  });
  // best-accepted-so-far step envelope
  let best = null, step = [];
  for(const p of pts){
    if(p.born != null && (best == null || p.val_mean > best)){
      if(best != null) step.push([p.i, best]);
      best = p.val_mean;
      step.push([p.i, best]);
    }
  }
  if(step.length){
    step.push([iMax, best]);
    const path = document.createElementNS(NS, 'polyline');
    path.setAttribute('points', step.map(([i, v]) => xOf(i) + ',' + yOf(v)).join(' '));
    path.setAttribute('fill', 'none'); path.setAttribute('stroke', '#06c'); path.setAttribute('stroke-width', 2.5);
    svg.appendChild(path);
  }
  // per-iteration dots (blue = accepted, gray = rejected/skipped)
  for(const p of pts){
    const c = document.createElementNS(NS, 'circle');
    c.setAttribute('cx', xOf(p.i)); c.setAttribute('cy', yOf(p.val_mean));
    c.setAttribute('r', p.born != null ? 3.5 : 2);
    c.setAttribute('fill', p.born != null ? '#06c' : '#c7cbd1');
    c.style.cursor = 'pointer';
    c.addEventListener('mouseenter', e => showTip(
      'iter ' + p.i + (p.born != null ? ' — accepted as cand ' + p.born : ' — not accepted (parent cand ' + p.selected + ')') +
      '<br>minibatch mean score: <b>' + p.val_mean.toFixed(3) + '</b>', null, e));
    c.addEventListener('mouseleave', hideTip);
    svg.appendChild(c);
  }
  // x-axis ticks
  const nTicks = Math.min(10, iMax - iMin + 1);
  for(let k = 0; k <= nTicks; k++){
    const i = Math.round(iMin + (iMax - iMin) * k / nTicks);
    const t = document.createElementNS(NS, 'text');
    t.setAttribute('x', xOf(i)); t.setAttribute('y', H - 6); t.setAttribute('text-anchor', 'middle');
    t.setAttribute('font-size', '10'); t.setAttribute('fill', '#889');
    t.textContent = i;
    svg.appendChild(t);
  }
  host.appendChild(svg);
})();

// ---- timeline ----
const tlb=document.getElementById('tlbody');
for(const t of D.timeline){ const tr=document.createElement('tr');
  tr.innerHTML='<td>'+t.i+'</td><td>cand '+t.selected+'</td><td>'+(t.born!=null?'cand '+t.born:'— (rejected)')+'</td><td>'+(t.val_mean!=null?t.val_mean:'')+'</td>';
  tlb.appendChild(tr); }

// ---- search process tree ----
function deltaSpan(o,n){
  if(o==null||n==null) return '';
  const d=n-o, cls=d>0?'deltaup':(d<0?'deltadn':'');
  return ' <span class="'+cls+'">'+fmt(o)+'→'+fmt(n)+(d>0?' ▲':(d<0?' ▼':''))+'</span>';
}
function openProc(node){
  hideTip();
  const accepted = node.kind==='accepted';
  let h='<span class="close" onclick="closePanel()">close ✕</span>'+
    '<h3>'+node.label+' <span class="role '+('tk-'+node.kind)+'" style="position:static">'+node.kind+'</span></h3>'+
    '<div class="sub">iteration '+node.i+' · mutated from <b>cand '+node.selected+'</b>'+
      (node.comp&&node.comp.length?(' · updated <b>'+node.comp.join(', ')+'</b>'):'')+'</div>';
  // scores
  h+='<div class="sec"><h4>Subsample gate</h4>'+
     '<div class="kv">parent full-val score: <b>'+fmt(node.selected_score)+'</b></div>'+
     '<div class="kv">subsample before→after:'+(deltaSpan(node.old_score,node.new_score)||' <b>'+fmt(node.new_score)+'</b>')+
       (accepted?' — passed, added to pool':(node.kind==='rejected'?' — not better, rejected':' — skipped'))+'</div>'+
     (node.reason?'<div class="kv" style="color:#778">'+esc(node.reason)+'</div>':'')+'</div>';
  // proposer response (new component text), per component
  const prop=node.proposed||{};
  for(const comp of Object.keys(prop)){
    h+='<div class="sec si-fwd"><h4>Proposer response — new '+esc(comp)+'</h4>'+
       '<pre>'+esc(prop[comp])+'</pre>'+
       (node.parent_text&&node.parent_text[comp]!=null?
         '<details class="pr"><summary>parent ('+comp+') for comparison</summary><pre>'+esc(node.parent_text[comp])+'</pre></details>':'')+
       '</div>';
  }
  // mistake feedback shown to the proposer + the proposer's response, per component.
  // The prompt IS the full rendered reflective dataset (the mistake feedback); the
  // response is the proposer's reply. Shown in full rather than broken up.
  const pc=node.proposer||{};
  const pcomps=Object.keys(pc);
  if(pcomps.length){
    for(const comp of pcomps){
      const c=pc[comp]||{};
      const meta=(c.attempts!=null?('attempts '+c.attempts):'')+(c.ts?(' · '+c.ts):'');
      h+='<div class="sec si-inv"><h4>Mistake feedback prompt + response — '+esc(comp)+(meta?(' <span class="kv" style="color:#778;font-weight:400">('+esc(meta)+')</span>'):'')+'</h4>'+
         promptResp('full prompt (mistake feedback) + response', c.prompt, c.response)+'</div>';
    }
  } else {
    h+='<div class="sec"><h4>Mistake feedback prompt + response</h4><div class="kv" style="color:#778">No proposer call captured for this iteration. '+
       '(Run without <code>--log-reflection</code>, or it predates reflection_calls.jsonl.)</div></div>';
  }
  // mistake-analysis calls (--analyze-mistakes): the SEPARATE LLM call that diagnoses F's
  // mistakes into the ANALYSIS feedback embedded in the proposer prompt above. Full prompt +
  // raw response per call. Empty unless the run used --analyze-mistakes with analysis logging.
  const an=node.analysis||[];
  if(an.length){
    let inner='';
    an.forEach((a,i)=>{
      const tag=(a.component||'?')+' · '+(a.mode||'?')+(a.kind&&a.kind!=='combined'?(' '+a.kind):'')+
                (a.n_cases!=null?(' · '+a.n_cases+' mistake'+(a.n_cases==1?'':'s')):'')+(a.ts?(' · '+a.ts):'');
      const recon=a.reconstructed?' <span style="color:#b36b00;font-weight:600">[reconstructed — prompt not logged]</span>':'';
      inner+='<div class="fbrow"><div class="kv">analysis call '+(i+1)+' <span style="color:#778;font-weight:400">('+esc(tag)+')</span>'+recon+'</div>'+
             promptResp('diagnosis prompt + response', a.prompt, a.response)+'</div>';
    });
    h+='<div class="sec si-fwd"><h4>Mistake-analysis call(s) — diagnosis → ANALYSIS feedback</h4>'+inner+'</div>';
  }
  panel.innerHTML=h; panel.style.display='block';
}
(function renderTree(){
  const host=document.getElementById('tree');
  const nodes=D.tree||[];
  if(!nodes.length){ host.innerHTML='<div style="color:#778;font-size:12px">No <code>process_log.jsonl</code> for this run — re-run with the updated <code>rexpure_optimize.py</code> to capture rejected proposals + proposer calls. (The accepted-only lineage is in the timeline + candidate parents above.)</div>'; return; }
  const kids={}; nodes.forEach(n=>{ (kids[n.parent]=kids[n.parent]||[]).push(n); });
  function nodeHTML(n){
    const sub = n.sub?('<span class="ts">'+esc(n.sub)+'</span>'):'';
    const delta = (n.old_score!=null&&n.new_score!=null)?deltaSpan(n.old_score,n.new_score):'';
    return '<span class="tnode tk-'+n.kind+'" data-id="'+n.id+'">'+esc(n.label)+sub+delta+'</span>';
  }
  function build(id){
    const ch=kids[id]; if(!ch||!ch.length) return '';
    let s='<ul>';
    for(const n of ch){ s+='<li>'+nodeHTML(n)+build(n.id)+'</li>'; }
    return s+'</ul>';
  }
  const root=nodes.find(n=>n.parent===null) || nodes[0];
  host.innerHTML='<ul><li>'+nodeHTML(root)+build(root.id)+'</li></ul>';
  const byId={}; nodes.forEach(n=>byId[n.id]=n);
  host.querySelectorAll('.tnode').forEach(el=>{ el.onclick=()=>{ const n=byId[el.getAttribute('data-id')]; if(n) openProc(n); }; });
})();

// ---- candidate cards ----
const cc=document.getElementById('cands');
for(const c of D.candidates){ const d=document.createElement('div'); d.className='card';
  d.innerHTML='<h3>candidate '+c.idx+(c.role?'<span class="role r-'+c.role.replace(/\s/g,'')+'">'+c.role+'</span>':'')+
    '<span class="score">'+(c.score!=null?c.score:'')+'</span></h3>'+
    '<div style="color:#778">parent: '+c.parents+'</div>'+
    '<div class="gtitle">world_knowledge'+(c.world_knowledge?'':' (empty)')+'</div>'+(c.world_knowledge?'<pre>'+esc(c.world_knowledge)+'</pre>':'')+
    '<div class="gtitle">perception</div><pre>'+esc(c.perception)+'</pre>';
  cc.appendChild(d); }
function esc(s){ return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;'); }

// ---- test rows ----
const _pct=v=>(v===null||v===undefined)?'—':(v*100).toFixed(0)+'%';
const ta=document.getElementById('testacc');
if(ta){ ta.innerHTML='— best candidate <b>'+_pct(D.test_acc)+'</b> vs raw-frame baseline <b>'+_pct(D.raw_acc)+'</b>'; }
const tb=document.getElementById('testbody');
// side panel for one held-out test row: the transition (with its K-step context window
// rendered as images) + full decoder prompt/response for BOTH the best candidate and the
// raw-frame baseline. Prompts are rebuilt deterministically at viz build time (the traces
// only log features/choices/response); best_prompt_exact records the verification.
function gridStrip(el, pairs){
  if(!el) return;
  for(const [lbl,g] of pairs){ const d=document.createElement('div');
    const t=document.createElement('div'); t.className='gtitle'; t.textContent=lbl; d.appendChild(t);
    const cv=drawGrid(g,12); if(cv){const w=document.createElement('div');w.className='gwrap';w.appendChild(cv);d.appendChild(w);}
    el.appendChild(d); }
}
function openTest(r,ri){
  hideTip();
  const hasRaw = r.raw_pred!==null && r.raw_pred!==undefined;
  const hasWin = r.window && (r.window.states||[]).length>1;
  let h='<span class="close" onclick="closePanel()">close ✕</span>'+
    '<h3>held-out test #'+ri+'</h3>'+
    '<div class="sub">true action <b>'+esc(r.truth)+'</b> · best <span class="'+(r.correct?'ok2':'bad2')+'">'+esc(r.pred)+(r.correct?' ✓':' ✗')+'</span>'+
      (hasRaw?' · raw <span class="'+(r.raw_correct?'ok2':'bad2')+'">'+esc(r.raw_pred)+(r.raw_correct?' ✓':' ✗')+'</span>':'')+
      ' · move <span class="mv">'+(r.move&&r.move.vec?('('+r.move.vec[0]+','+r.move.vec[1]+')'):'·')+'</span></div>';
  h+='<div class="sec"><h4>Test transition'+(hasWin?' — raw K-step context window':'')+'</h4>'+
     (hasWin?'<div class="kv" style="color:#778">raw states X_t-K..X_t+K around the masked action — the context the best-candidate decoder saw (perceived through P). The raw baseline saw only X_t/X_t+1.</div><div id="twin"></div>'
            :'<div class="grids" id="tgrids"></div>')+
     ((r.choices&&r.choices.length)?'<div class="kv">choices: '+r.choices.map(c=>'<b>'+esc(c)+'</b>').join(' · ')+'</div>':'')+'</div>';
  const bnote = !r.best_prompt ? '' :
    (r.best_prompt_exact===false
      ?'<div class="kv" style="color:#b36b00;font-weight:600">prompt reconstructed, but re-running P gave different features than the stored ones — the prompt below may not be verbatim</div>'
      :'<div class="kv" style="color:#778">prompt rebuilt deterministically from the stored P/beliefs/choices'+(r.best_prompt_exact?' (verified against the stored features)':'')+'</div>');
  h+='<div class="sec si-inv"><h4>Best candidate — prediction</h4>'+
     '<div class="kv">predicted <b>'+esc(r.pred)+'</b> · truth <b>'+esc(r.truth)+'</b> · <span class="'+(r.correct?'ok2':'bad2')+'">'+(r.correct?'✓ correct':'✗ wrong')+'</span></div>'+
     '<div class="kv">P(X_t) features:</div><pre>'+esc(r.z_t||'(empty)')+'</pre>'+
     '<div class="kv">P(X_t+1) features:</div><pre>'+esc(r.z_t1||'(empty)')+'</pre>'+
     bnote+
     ((r.best_prompt||r.reasoning)?promptResp('full prompt + response (best candidate)', r.best_prompt, r.reasoning)
                                  :'<div class="kv" style="color:#778">no prompt/response available for this row</div>')+'</div>';
  h+='<div class="sec si-fwd"><h4>Raw-frame baseline — prediction</h4>'+
     (hasRaw?('<div class="kv">predicted <b>'+esc(r.raw_pred)+'</b> · truth <b>'+esc(r.truth)+'</b> · <span class="'+(r.raw_correct?'ok2':'bad2')+'">'+(r.raw_correct?'✓ correct':'✗ wrong')+'</span></div>'+
       (r.raw_prompt?'<div class="kv" style="color:#778">prompt rebuilt verbatim from the stored raw frames + choices; only the extracted &lt;reasoning&gt; of the response was logged</div>':'')+
       ((r.raw_prompt||r.raw_reasoning)?promptResp('full prompt + response (raw baseline)', r.raw_prompt, r.raw_reasoning)
                                       :'<div class="kv" style="color:#778">no prompt/response available for this row</div>'))
     :'<div class="kv" style="color:#778">no raw-frame trace for this run</div>')+'</div>';
  panel.innerHTML=h; panel.style.display='block';
  if(hasWin) drawWindow(document.getElementById('twin'), r.window, 'inverse');
  else gridStrip(document.getElementById('tgrids'), [['before',r.grid_t],['after',r.grid_t1]]);
}
(D.test||[]).forEach((r,ri)=>{ const tr=document.createElement('tr');
  const hasRaw = r.raw_pred!==null && r.raw_pred!==undefined;
  tr.onclick=()=>openTest(r,ri);
  const hdr=document.createElement('td'); hdr.className='rowhdr'; hdr.innerHTML='<b>'+r.truth+'</b>';
  hdr.onmousemove=e=>showTip('<b>held-out #'+ri+'</b> true <b>'+r.truth+'</b> · best <b>'+r.pred+'</b> · raw <b>'+(hasRaw?r.raw_pred:'—')+'</b><br><b style="color:#06c">click for the context window + full prompts/responses</b>',
    [['before',r.grid_t],['after',r.grid_t1]],e); hdr.onmouseleave=hideTip; tr.appendChild(hdr);
  const pc=document.createElement('td'); pc.className=r.correct?'ok':'bad'; pc.textContent=r.pred; tr.appendChild(pc);
  const rp=document.createElement('td'); rp.className=hasRaw?(r.raw_correct?'ok':'bad'):''; rp.textContent=hasRaw?r.raw_pred:'—'; tr.appendChild(rp);
  const mv=document.createElement('td'); mv.className='mv'; mv.textContent=r.move&&r.move.vec?('('+r.move.vec[0]+','+r.move.vec[1]+')'):'·'; tr.appendChild(mv);
  const rs=document.createElement('td'); rs.style.textAlign='left'; rs.style.padding='4px 8px'; rs.style.maxWidth='500px'; rs.textContent=r.reasoning; tr.appendChild(rs);
  const rr=document.createElement('td'); rr.style.textAlign='left'; rr.style.padding='4px 8px'; rr.style.maxWidth='500px'; rr.style.color='#556'; rr.textContent=hasRaw?r.raw_reasoning:'(no raw trace)'; tr.appendChild(rr);
  tb.appendChild(tr); });
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="the rexpure_run_seed<N> (or historical gepa_run_seed<N>) dir inside a run")
    ap.add_argument("--run", required=True, help="comma-separated data run dirs (same as the optimisation used)")
    ap.add_argument("--test-run", default=None,
                    help="match a cross-trajectory run launched with --test-run: TRAIN/VAL from "
                         "--run, the held-out TEST from these dirs. When set, the pooled in-pool test "
                         "carve / pin / swap-click are skipped (as in rexpure_optimize).")
    ap.add_argument("--context-source-run", default=None, help="comma-separated full trajectory dirs used to backfill temporal context for curated --run targets; matches rexpure_optimize")
    ap.add_argument("--test-context-source-run", default=None,
                    help="comma-separated full trajectory dirs used to backfill temporal "
                         "context for curated --test-run targets; matches rexpure_optimize")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--train-n", type=int, default=20)
    ap.add_argument("--val-n", type=int, default=20)
    ap.add_argument("--test-n", type=int, default=10)
    ap.add_argument("--k-choices", type=int, default=5)
    ap.add_argument("--actions", default="left,right,up,down,noop,click")
    ap.add_argument("--tie-train-val", action="store_true")
    ap.add_argument("--stratified-split", action="store_true",
                    help="mirror rexpure_optimize --stratified-split: per-action alternate deal of train/val")
    ap.add_argument("--swap-click-train", action="store_true",
        help="match a run launched with --swap-click-train so the reconstructed split aligns")
    ap.add_argument("--context-k", type=int, default=3,
                    help="K-step temporal window to reconstruct + render in the panel "
                         "(match the run's --context-k; 0 = single-step, no window)")
    ap.add_argument("--collapse-action-params", action="store_true")
    ap.add_argument(
        "--keep-obs-metadata", action="store_true",
        help="match a run launched with --keep-obs-metadata: keep the Task:/Step:/Phase: "
        "header in autumn obs (default: strip it, mirroring rexpure_optimize, so tr_hash "
        "joins the per-prediction sidecar)",
    )
    ap.add_argument("--pin-train-idx", default="",
                    help="must match the run: comma-separated original transition indices pinned into train")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = build_model(args)
    out = Path(args.out) if args.out else Path(args.run_dir).parent / "optim_viz.html"
    html = HTML.replace("__TITLE__", model["title"]).replace("__DATA__", json.dumps(model))
    out.write_text(html)
    v = model["verify"]
    print(f"[viz] wrote {out}")
    print(f"[viz] alignment check: ok={v['ok']} ({v['detail']})")
    print(f"[viz] {len(model['val'])} val transitions | {len(model['cand_cols'])} candidate columns")


if __name__ == "__main__":
    main()
