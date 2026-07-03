"""GEPA-based optimizer for perception P (+ world knowledge B), as a drop-in
alternative to the hand-rolled greedy loop in validate.py / validate_beliefs.py.

Why GEPA: our greedy held-out gate keeps a single best candidate, so it can't
fund a representational leap (single-cell -> full darkgreen SET) that dips before
it pays off -> the 0.85 plateau. GEPA keeps a PARETO FRONTIER of candidates that
each win on *some* held-out instances, samples from it for mutation, and uses a
strong reflection LM to propose targeted rewrites from execution traces. That
directly targets the "greedy + prior-locked search" failure mode (RC2), while we
still own signal quality (RC1) via make_reflective_dataset.

Mapping onto GEPA:
  candidate            = {"perception": <P code>, "world_knowledge": <B text>}   (multi-component)
  DataInst             = one transition (X_t, A_t, X_t+1) + a fixed choice set
  adapter.evaluate     = forward inverse-dynamics: P(X_t),P(X_t+1),B -> F -> Ahat; score = 1[Ahat==A_t]
  make_reflective_data = per-failure signal from INTERNAL observables only:
                         predicted vs TRUE action, F's reasoning, and an output-invariance
                         note (P emitted identical features for both states) -- no hand-coded
                         grid parser / ground-truth raw change is ever shown to the proposer
  reflection_lm        = strong model rewriting the requested component
  task_lm (F)          = weak frozen decoder (= the deployed agent) -- kept cheap

Comparison: this script reports baselines + GEPA's best on a CLEAN test split that
neither optimization nor selection ever touches. With --compare it ALSO runs the
legacy greedy P/B loop on the same train/val and scores it on the same test, for a
single-run head-to-head. validate.py / validate_beliefs.py are left intact.

Usage (reflection_lm defaults to --task-model, so GEPA and legacy share one model):
  uv run prototypes/perc_invdyn/gepa_optimize.py \
      --run "logs/dynamics_full/DQ8GC/...,logs/seed_autumn/DQ8GC/..." \
      --task-model google/gemini-2.5-flash --max-metric-calls 300 --compare
"""

import argparse
import asyncio
import base64
import difflib
import hashlib
import json
import random
import re
import time
from collections import Counter
from io import BytesIO
from pathlib import Path

import gepa

# Forward-dynamics scorers (validated in forward_objective.py): textdiff_delta is the
# deterministic, format-agnostic next-state change score; judge_score is the LLM judge.
from forward_objective import (  # noqa: E402
    judge_score,
    judge_score_reasoned,
    textdiff_delta_f1,
)
from gepa import Image as GepaImage
from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from PIL import Image as PILImage


def install_accept_ties_patch(eps: float = 1e-9) -> None:
    """Flip GEPA's strict acceptance gate (engine rejects when ``new_sum <= old_sum``)
    to ``<`` semantics, so a proposal that exactly TIES its parent on the subsample is
    still sent to full eval + the pareto front instead of being dropped.

    Why: the gate compares the AGGREGATE sum over the subsample. A candidate that is
    worse on one transition but better on another nets the same sum and is killed --
    even though that per-instance complementarity is exactly what the pareto front is
    built to keep. With a stochastic task LM, a tie is also partly F-sampling noise.

    How: we do NOT edit the engine's inline comparison. We wrap the reflective
    proposer so that on an EXACT tie it nudges the reported parent baseline
    (``subsample_scores_before``) down by ``eps``. The child's real score
    (``subsample_scores_after``) is untouched, genuinely-worse proposals
    (``new_sum < old_sum``) are unaffected -> still rejected, and the pareto front is
    unaffected regardless because accepted candidates are re-scored by a fresh FULL
    eval (subsample scores feed only the gate + a cosmetic log field)."""
    from gepa.proposer.reflective_mutation.reflective_mutation import (
        ReflectiveMutationProposer,
    )

    if getattr(ReflectiveMutationProposer, "_accept_ties_patched", False):
        return
    _orig_propose = ReflectiveMutationProposer.propose

    def propose(self, state):  # noqa: ANN001
        prop = _orig_propose(self, state)
        if prop is None:
            return prop
        before, after = prop.subsample_scores_before, prop.subsample_scores_after
        if before and after and abs(sum(after) - sum(before)) < eps:
            nudged = list(before)
            nudged[0] = nudged[0] - 2 * eps  # make new_sum strictly > old_sum on ties
            prop.subsample_scores_before = nudged
        return prop

    ReflectiveMutationProposer.propose = propose
    ReflectiveMutationProposer._accept_ties_patched = True


# Reuse the existing prototype plumbing verbatim (same dir; run via uv from repo root).
from validate_beliefs import (  # noqa: E402
    GOOD_P,
    Transition,
    _extract_action,
    _extract_code,
    _llm_call,
    _parse_tag,
    balanced_split,
    compute_g1,
    forward_eval,
    load_transitions,
    make_choices,
    make_config,
    perception_runs,
    predict_action,
    run_perceive,
    swap_click_into_train,
    update_beliefs,
    update_perception,
)


# ---------------------------------------------------------------------------
# Grid parsing -- used ONLY to RENDER the raw observation as an image for the
# proposer in --image-mode (parse_grid -> render_grid). It is never used to
# build the learning signal / ground-truth change. (See module docstring.)
# ---------------------------------------------------------------------------
def parse_grid(raw: str):
    """Parse the first grid out of an observation. Handles two formats:
    - AutumnBench: a JSON 2D array of color strings  [["black","gold",...],[...]]
    - ARC-AGI-3:   '<grid_0>' then integer rows on separate lines  [5, 5, 5, ...]
    """
    if "<grid_" in raw:  # ARC integer grid: take the first <grid_k> block
        start = raw.find("<grid_")
        block = raw[start:]
        nxt = block.find("<grid_", 5)
        if nxt != -1:
            block = block[:nxt]
        rows = re.findall(r"\[([0-9,\s]+)\]", block)
        grid = [[int(x) for x in r.split(",") if x.strip() != ""] for r in rows]
        grid = [row for row in grid if row]
        return grid or None
    s = raw.find("[[")  # Autumn string grid
    e = raw.rfind("]]") + 2
    if s == -1 or e <= 1:
        return None
    try:
        return json.loads(raw[s:e])
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Image rendering: turn a parsed grid into a PNG so prompts can show the IMAGE
# instead of the (huge, truncated) raw-grid text. P still runs on the text grid;
# only prompts that would display the raw grid get the image. (ARC / int grids.)
# ---------------------------------------------------------------------------
ARC_PALETTE = {
    0: (0, 0, 0),
    1: (0, 116, 217),
    2: (255, 65, 54),
    3: (46, 204, 64),
    4: (255, 220, 0),
    5: (170, 170, 170),
    6: (240, 18, 190),
    7: (255, 133, 27),
    8: (127, 219, 255),
    9: (135, 12, 37),
    10: (255, 255, 255),
    11: (90, 90, 90),
}


def _prefix_hint(text: str, n: int) -> str:
    """First n chars of an observation, with an explicit marker IFF it was cut,
    so the proposer never mistakes a prefix for the whole input."""
    if len(text) <= n:
        return text
    return (
        text[:n]
        + "\n... [TRUNCATED -- many more grid rows follow in the full observation at runtime]"
    )


def _clip_reasoning(text: str, head: int = 700, tail: int = 700) -> str:
    """Condense F's chain-of-thought for proposer feedback. F's reasoning is long
    (median ~2k chars) and its DECISION lives at the END, so a plain head-truncation
    drops the very part the proposer needs. Keep the head (state analysis) AND the
    tail (the conclusion), eliding only the redundant middle."""
    text = (text or "").strip()
    if len(text) <= head + tail:
        return text or "(none given)"
    cut = len(text) - head - tail
    return f"{text[:head]}\n... [elided {cut} chars of intermediate reasoning] ...\n{text[-tail:]}"


def _color_of(v):
    if isinstance(v, int):
        return ARC_PALETTE.get(v, (60, 60, 60))
    # named colors (autumn) -> let PIL parse the name, fall back to grey
    try:
        from PIL import ImageColor

        return ImageColor.getrgb(str(v))
    except Exception:  # noqa: BLE001
        return (60, 60, 60)


def render_grid(grid, cell=8) -> "PILImage.Image | None":
    if not grid:
        return None
    h, w = len(grid), max(len(r) for r in grid)
    img = PILImage.new("RGB", (w * cell, h * cell), (0, 0, 0))
    px = img.load()
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            col = _color_of(v)
            for dy in range(cell):
                for dx in range(cell):
                    px[c * cell + dx, r * cell + dy] = col
    return img


def grid_b64(raw: str, cell=8):
    """raw observation text -> (base64 png, media_type) of the rendered grid, or None."""
    img = render_grid(parse_grid(raw), cell=cell)
    if img is None:
        return None
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode(), "image/png"


def grid_pil(raw: str, cell=8):
    return render_grid(parse_grid(raw), cell=cell)


# ---------------------------------------------------------------------------
# Forward predictor Fwd: given P(X_t), the TRUE action, and B, predict P(X_t+1).
# Dual of predict_action (inverse). Self-supervised: the label is the logged next
# frame run through the SAME P (z_t1); Fwd never sees it. Uses the SAME frozen task_lm
# as the inverse decoder. Output is scored by textdiff_delta_f1 or judge_score, both of
# which touch only P's emitted symbols -> the forward signal stays pure.
# ---------------------------------------------------------------------------
# DEFAULT KNOWLEDGE: fixed, always-true conventions injected directly ABOVE the (learned)
# WORLD KNOWLEDGE block in every prompt that carries one. Unlike world knowledge it is never
# optimized -- it supplies facts the perception module strips out of the raw observation,
# notably the action space and the click coordinate order. For autumn games this is a simple
# sentence describing the available actions; other game families can override DEFAULT_KNOWLEDGE.
DEFAULT_KNOWLEDGE = (
    "The available actions are: up, down, left, right, noop, and click ROW COL, where ROW is the "
    "row index and COL is the column index (both 0-indexed; this matches the (row, col) order the "
    "perception reports cells in)."
)


FORWARD_TMPL = """You predict the NEXT-state features of a grid environment from the current features and an action.

=== DEFAULT KNOWLEDGE (always-true facts about this environment) ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== WORLD KNOWLEDGE (general facts; may be empty) ===
{beliefs}
=== END WORLD KNOWLEDGE ===

A perception module summarized the CURRENT state as these features:
=== CURRENT features ===
{z_t}
=== END CURRENT features ===

The action taken was: {action}

Predict the features of the resulting NEXT state, written in EXACTLY the same format and vocabulary the perception module uses above (same keys, same coordinate/colour conventions). Change only what the action changes; copy everything else unchanged. Do NOT add commentary.

<next_state>predicted next-state features, same format as CURRENT</next_state>"""


async def predict_next_state(cfg, z_t, action, beliefs, sem):
    prompt = FORWARD_TMPL.format(
        beliefs=beliefs.strip() or "(empty)",
        default_knowledge=DEFAULT_KNOWLEDGE,
        z_t=z_t or "(empty)",
        action=action,
    )
    async with sem:
        text, cost = await _llm_call(cfg, prompt)
    text = text or ""
    z_hat = _parse_tag(text, "next_state") or text.strip()
    return z_hat, cost


# ---------------------------------------------------------------------------
# Temporal-window (K-step) objectives. Instead of showing F just the two center
# states, show a transcript P(X_t-K),a_t-K,...,P(X_t),...,P(X_t+K): inverse MASKS
# a_t and asks F to recover it from the whole window; forward shows the history up to
# P(X_t) plus a_t and asks for P(X_t+1). This lets B encode temporally-extended dynamics
# (selection, momentum, periodicity) that a single (X_t, X_t+1) pair cannot reveal. The
# scoring labels are unchanged (id_score on a_t; fd_score on the center z_t1), so the
# signal stays pure -- only the CONTEXT shown to F is widened. Falls back to the two-state
# templates when context_k==0 (exact back-compat with the validated path).
# ---------------------------------------------------------------------------
def build_window(code, tr):
    """Run P over the whole temporal window of a Transition. Returns (win, perc_err) where
    win = {prev:[(z, a)], z_t, z_t1, nxt:[(a, z)]} in feature space, aligned with tr.ctx_*.
    The raw observation of every window state is also retained (prev_raw / x_t / x_t1 /
    nxt_raw, same order as the feature lists) so the image-augmented windowed predictors and
    the P-writer can render one image per state."""
    err = None
    prev = []
    for raw, a in tr.ctx_prev:
        z, e = run_perceive(code, raw)
        err = err or e
        prev.append((z, a))
    z_t, e1 = run_perceive(code, tr.x_t)
    z_t1, e2 = run_perceive(code, tr.x_t1)
    err = err or e1 or e2
    nxt = []
    for a, raw in tr.ctx_next:
        z, e = run_perceive(code, raw)
        err = err or e
        nxt.append((a, z))
    return {
        "prev": prev,
        "z_t": z_t,
        "z_t1": z_t1,
        "nxt": nxt,
        "prev_raw": [raw for raw, _ in tr.ctx_prev],
        "x_t": tr.x_t,
        "x_t1": tr.x_t1,
        "nxt_raw": [raw for _, raw in tr.ctx_next],
    }, err


def _tlabel(i: int) -> str:
    """Step label relative to the center: 0 -> 't', -2 -> 't-2', 1 -> 't+1'."""
    return "t" if i == 0 else (f"t{i}" if i < 0 else f"t+{i}")


def _inverse_transcript(win):
    """States X_t-K..X_t+K with the action between each consecutive pair; a_t is MASKED."""
    lines, n_prev = [], len(win["prev"])
    for k, (z, a) in enumerate(win["prev"]):
        idx = -(n_prev - k)  # -n_prev .. -1
        lines.append(f"STATE[{_tlabel(idx)}] features:\n{z or '(empty)'}")
        lines.append(f"  action({_tlabel(idx)} -> {_tlabel(idx + 1)}): {a}")
    lines.append(f"STATE[t] features:\n{win['z_t'] or '(empty)'}")
    lines.append("  action(t -> t+1): ??? (IDENTIFY THIS)")
    lines.append(f"STATE[t+1] features:\n{win['z_t1'] or '(empty)'}")
    for k, (a, z) in enumerate(win["nxt"]):
        lines.append(f"  action({_tlabel(k + 1)} -> {_tlabel(k + 2)}): {a}")
        lines.append(f"STATE[{_tlabel(k + 2)}] features:\n{z or '(empty)'}")
    return "\n".join(lines)


def _forward_transcript(win):
    """History X_t-K..X_t with the action between each consecutive pair (ends at CURRENT)."""
    lines, n_prev = [], len(win["prev"])
    for k, (z, a) in enumerate(win["prev"]):
        idx = -(n_prev - k)
        lines.append(f"STATE[{_tlabel(idx)}] features:\n{z or '(empty)'}")
        lines.append(f"  action({_tlabel(idx)} -> {_tlabel(idx + 1)}): {a}")
    lines.append(f"STATE[t] (CURRENT) features:\n{win['z_t'] or '(empty)'}")
    return "\n".join(lines)


def _window_frames(win):
    """Ordered [(label, raw_obs)] for EVERY state in the inverse window, t-K .. t+K, in the
    same order as _inverse_transcript so one image can be attached per state."""
    frames = [(_tlabel(-(len(win.get("prev_raw", [])) - k)), raw)
              for k, raw in enumerate(win.get("prev_raw", []))]
    frames.append(("t", win["x_t"]))
    frames.append(("t+1", win["x_t1"]))
    frames += [(_tlabel(k + 2), raw) for k, raw in enumerate(win.get("nxt_raw", []))]
    return frames


def _window_frames_forward(win):
    """Ordered [(label, raw_obs)] for the forward history t-K .. t (ends at CURRENT), matching
    _forward_transcript -- no next/future states are revealed."""
    frames = [(_tlabel(-(len(win.get("prev_raw", [])) - k)), raw)
              for k, raw in enumerate(win.get("prev_raw", []))]
    frames.append(("t", win["x_t"]))
    return frames


def _state_diff(pred: str, true: str) -> str:
    """Readable token-level diff between the PREDICTED and TRUE next-state feature
    strings. Format-agnostic (P defines the vocabulary): tokens are whitespace-split,
    so each comma-joined field stays intact, and changed/extra/missing fields are
    flagged inline. Returns '(identical)' when they match."""
    pred, true = (pred or "").strip(), (true or "").strip()
    if pred == true:
        return "(identical)"
    ptok, ttok = re.findall(r"\S+", pred), re.findall(r"\S+", true)
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=ptok, b=ttok).get_opcodes():
        if tag == "equal":
            out.append(" ".join(ptok[i1:i2]))
        elif tag == "replace":
            out.append(
                f"[predicted '{' '.join(ptok[i1:i2])}' != true '{' '.join(ttok[j1:j2])}']"
            )
        elif tag == "delete":
            out.append(f"[only-in-predicted '{' '.join(ptok[i1:i2])}']")
        elif tag == "insert":
            out.append(f"[only-in-true '{' '.join(ttok[j1:j2])}']")
    return " ".join(out)


def _forward_section(t, *, win) -> str:
    """The FORWARD-prediction block of a reflective example: the past states leading to
    CURRENT, the TRUE next state, the PREDICTED next state, and a field-level diff."""
    history = (
        _forward_transcript(win)
        if win
        else (f"STATE[t] (CURRENT) features:\n{t['z_t'] or '(empty)'}")
    )
    return (
        "presented past states (history ending at CURRENT):\n"
        f"{history}\n\n"
        f"true next state:\n{t['z_t1'] or '(empty)'}\n\n"
        f"predicted next state:\n{t['z_hat'] or '(empty)'}\n\n"
        f"diff (predicted vs true), match={t.get('fd_score', 0.0):.2f}:\n"
        f"{_state_diff(t['z_hat'], t['z_t1'])}"
    )


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


FWD_WINDOW_TMPL = """You predict the NEXT-state features of a grid environment from a trajectory and the action just taken.

=== DEFAULT KNOWLEDGE (always-true facts about this environment) ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== WORLD KNOWLEDGE (general facts; may be empty) ===
{beliefs}
=== END WORLD KNOWLEDGE ===

Below is a trajectory of consecutive states (summarized as features) ending at the CURRENT state, with the action taken between each pair. Use the WHOLE history to capture dynamics that depend on more than the current state alone.

{transcript}

The action now taken from the CURRENT state is: {action}

Predict the features of the resulting NEXT state, in EXACTLY the same format and vocabulary the perception module uses above (same keys, same coordinate/colour conventions). Change only what the action changes; copy everything else unchanged. Do NOT add commentary.

<next_state>predicted next-state features, same format as CURRENT</next_state>"""


async def predict_action_from_window(cfg, win, beliefs, choices, sem):
    prompt = INV_WINDOW_TMPL.format(
        beliefs=beliefs.strip() or "(empty)",
        default_knowledge=DEFAULT_KNOWLEDGE,
        transcript=_inverse_transcript(win),
        choices="\n".join(f"- {c}" for c in choices),
    )
    async with sem:
        text, cost = await _llm_call(cfg, prompt)
    text = text or ""
    pred = _extract_action(text, choices)
    return pred, text, cost, prompt  # text = raw response, prompt = exact prompt F saw


async def predict_next_state_from_window(cfg, win, action, beliefs, sem):
    prompt = FWD_WINDOW_TMPL.format(
        beliefs=beliefs.strip() or "(empty)",
        default_knowledge=DEFAULT_KNOWLEDGE,
        transcript=_forward_transcript(win),
        action=action,
    )
    async with sem:
        text, cost = await _llm_call(cfg, prompt)
    text = text or ""
    z_hat = _parse_tag(text, "next_state") or text.strip()
    return z_hat, cost, text, prompt  # text = raw response, prompt = exact prompt F saw


def exact_match_f1(z_hat, z_t1):
    """Forward reward: 1.0 iff the predicted next features EXACTLY equal the true P(X_t+1),
    else 0.0. Ends are stripped so trailing formatting whitespace doesn't spuriously fail an
    otherwise-identical prediction; the content itself must match character-for-character."""
    return 1.0 if (z_hat or "").strip() == (z_t1 or "").strip() else 0.0


# ---------------------------------------------------------------------------
# Optional LLM mistake analysis. When enabled, a model is shown the world
# knowledge, the predictor's reasoning, its (wrong) prediction and the ground
# truth, and asked to diagnose the root cause into concise feedback that the
# proposer can act on -- a step BETWEEN the raw mistake and the rewrite.
# ---------------------------------------------------------------------------
def _analysis_target(comp: str) -> str:
    return (
        "the WORLD KNOWLEDGE block (its dynamics / convention rules)"
        if comp == "world_knowledge"
        else "the PERCEPTION module (the text features perceive() emits)"
    )


ANALYZE_INVERSE_TMPL = """You are diagnosing a single mistake made by an inverse-dynamics predictor F in a grid environment.

F is given the WORLD KNOWLEDGE below plus a perception module's text features for consecutive states, and must name the ONE action taken between two states. On THIS case F was WRONG.

=== DEFAULT KNOWLEDGE (always-true facts F was given) ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== WORLD KNOWLEDGE (general facts F was given; may be empty) ===
{beliefs}
=== END WORLD KNOWLEDGE ===

=== STATES / FEATURE TRAJECTORY (the masked action is the one F had to recover) ===
{transcript}
=== END ===

F predicted action: {pred}
TRUE action:        {truth}
F's stated reasoning: {reasoning}

Diagnose the ROOT CAUSE of the error and give specific, GENERAL feedback that would help improve {target} so this whole class of mistake is avoided -- e.g. a wrong/missing dynamics rule, or features that fail to distinguish this action from the one F chose. Do not restate the prediction; give actionable guidance. Be concise (<120 words).

<feedback>your diagnosis + concrete fix</feedback>"""


ANALYZE_FORWARD_TMPL = """You are diagnosing a single mistake made by a forward-dynamics predictor in a grid environment.

Given the WORLD KNOWLEDGE and a perception module's features for the current state (with recent history) plus the action taken, the predictor must output the NEXT state's features. On THIS case the prediction was WRONG.

=== DEFAULT KNOWLEDGE (always-true facts about this environment) ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== WORLD KNOWLEDGE (general facts; may be empty) ===
{beliefs}
=== END WORLD KNOWLEDGE ===

=== HISTORY (features, ending at the CURRENT state) ===
{history}
=== END ===

Action taken:           {action}
PREDICTED next features: {pred}
TRUE next features:      {truth}
Field-level diff (predicted vs true): {diff}

Diagnose the ROOT CAUSE of the divergence and give specific, GENERAL feedback that would help improve {target} so the next state becomes predictable from the current one plus the action -- e.g. a wrong/missing dynamics rule, or features that drop state needed to predict the change. Be concise (<120 words).

<feedback>your diagnosis + concrete fix</feedback>"""


def _log_analysis(log_path, **rec):
    """Append one --analyze-mistakes diagnosis call (full prompt + raw response) to
    analysis_calls.jsonl. These are the LLM calls that DIAGNOSE F's mistakes into proposer
    feedback -- distinct from the proposer calls in reflection_calls.jsonl. No-op unless a
    path was provided. `iteration` (the GEPA iteration, stamped from the shared run context)
    lets the viz line each diagnosis up with the candidate it fed."""
    if not log_path:
        return
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec.setdefault("ts", time.strftime("%H:%M:%S"))
    with p.open("a") as f:
        f.write(json.dumps(rec) + "\n")


async def analyze_inverse(
    cfg, beliefs, comp, t, tr, win, sem, *, log_path=None, iteration=None, ti=None
):
    transcript = (
        _inverse_transcript(win)
        if win
        else f"CURRENT features:\n{t['z_t'] or '(empty)'}\nNEXT features:\n{t['z_t1'] or '(empty)'}"
    )
    prompt = ANALYZE_INVERSE_TMPL.format(
        beliefs=(beliefs or "").strip() or "(empty)",
        default_knowledge=DEFAULT_KNOWLEDGE,
        transcript=transcript,
        pred=t["pred"],
        truth=tr.action,
        reasoning=(t.get("reasoning") or "")[:500] or "(none given)",
        target=_analysis_target(comp),
    )
    async with sem:
        text, cost = await _llm_call(cfg, prompt)
    _log_analysis(
        log_path,
        iteration=iteration,
        component=comp,
        mode="per-mistake",
        kind="inv",
        ti=ti,
        n_cases=1,
        prompt=prompt,
        response=text or "",
        cost=cost,
    )
    fb = _parse_tag(text or "", "feedback") or (text or "").strip()
    return fb, cost


async def analyze_forward(
    cfg, beliefs, comp, t, tr, win, sem, *, log_path=None, iteration=None, ti=None
):
    history = (
        _forward_transcript(win)
        if win
        else f"CURRENT features:\n{t['z_t'] or '(empty)'}"
    )
    prompt = ANALYZE_FORWARD_TMPL.format(
        beliefs=(beliefs or "").strip() or "(empty)",
        default_knowledge=DEFAULT_KNOWLEDGE,
        history=history,
        action=tr.action,
        pred=t["z_hat"] or "(empty)",
        truth=t["z_t1"] or "(empty)",
        diff=_state_diff(t["z_hat"], t["z_t1"]),
        target=_analysis_target(comp),
    )
    async with sem:
        text, cost = await _llm_call(cfg, prompt)
    _log_analysis(
        log_path,
        iteration=iteration,
        component=comp,
        mode="per-mistake",
        kind="fwd",
        ti=ti,
        n_cases=1,
        prompt=prompt,
        response=text or "",
        cost=cost,
    )
    fb = _parse_tag(text or "", "feedback") or (text or "").strip()
    return fb, cost


ANALYZE_COMBINED_TMPL = """You are diagnosing a BATCH of mistakes made by inverse- and forward-dynamics predictors in a grid environment. They were ALL produced by the SAME perception module + world knowledge, so diagnosing them together lets you find the COMMON root cause instead of {n} isolated ones.

=== DEFAULT KNOWLEDGE (always-true facts the predictors were given) ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== WORLD KNOWLEDGE (general facts the predictors were given; may be empty) ===
{beliefs}
=== END WORLD KNOWLEDGE ===

There are {n} mistakes below, each tagged [mK INVERSE] (the predictor had to recover the masked action between two states) or [mK FORWARD] (the predictor had to output the NEXT state's features).

{cases}

Diagnose the ROOT CAUSE(S) and give specific, GENERAL feedback to improve {target} so these classes of mistake are avoided. First identify any pattern shared across several mistakes, then give one concrete, actionable fix per mistake (do not merely restate the prediction).

Respond in EXACTLY this format -- the synthesis first, then ONE tag per mistake listed above, keeping each fix under 80 words:
<common_root_causes>patterns shared across the mistakes (1-4 sentences); write "none" if each is independent</common_root_causes>
{tags}"""


async def analyze_combined(
    cfg, beliefs, comp, cases, sem, *, log_path=None, iteration=None
):
    """ONE LLM call diagnosing ALL shown mistakes for a component at once. Emits structured
    per-mistake feedback plus a shared root-cause synthesis, and returns it in the SAME shape
    as the per-mistake path -- {(comp, ti, 'inv'|'fwd'): feedback}, cost -- so the injection in
    make_reflective_dataset is unchanged. Cheaper (the world-knowledge + instructions prefix is
    sent once, not per mistake) and lets the model spot cross-mistake patterns.
    cases = [(ti, kind, t, tr, win), ...]."""
    blocks, tags = [], []
    for ti, kind, t, tr, win in cases:
        tags.append(f"<m{ti}_{kind}>fix for this mistake</m{ti}_{kind}>")
        if kind == "inv":
            transcript = (
                _inverse_transcript(win)
                if win
                else f"CURRENT features:\n{t['z_t'] or '(empty)'}\nNEXT features:\n{t['z_t1'] or '(empty)'}"
            )
            blocks.append(
                f"[m{ti} INVERSE]\n{transcript}\n"
                f"predicted: {t['pred']} | TRUE action: {tr.action}\n"
                f"predictor reasoning: {_clip_reasoning(t.get('reasoning') or '')}"
            )
        else:
            history = (
                _forward_transcript(win)
                if win
                else f"CURRENT features:\n{t['z_t'] or '(empty)'}"
            )
            blocks.append(
                f"[m{ti} FORWARD]\n{history}\n"
                f"action: {tr.action} | PREDICTED next: {t['z_hat'] or '(empty)'} | "
                f"TRUE next: {t['z_t1'] or '(empty)'}\n"
                f"field-level diff (predicted vs true): {_state_diff(t.get('z_hat'), t.get('z_t1'))}"
            )
    prompt = ANALYZE_COMBINED_TMPL.format(
        n=len(cases),
        beliefs=(beliefs or "").strip() or "(empty)",
        default_knowledge=DEFAULT_KNOWLEDGE,
        cases="\n\n".join(blocks),
        target=_analysis_target(comp),
        tags="\n".join(tags),
    )
    async with sem:
        text, cost = await _llm_call(cfg, prompt)
    text = text or ""
    _log_analysis(
        log_path,
        iteration=iteration,
        component=comp,
        mode="combined",
        kind="combined",
        ti=None,
        n_cases=len(cases),
        prompt=prompt,
        response=text,
        cost=cost,
    )
    common = _parse_tag(text, "common_root_causes")
    out, first = {}, True
    for ti, kind, t, tr, win in cases:
        fb = _parse_tag(text, f"m{ti}_{kind}")
        if not fb:
            continue
        # surface the shared synthesis ONCE (on the first parsed mistake) so the proposer sees
        # the cross-mistake pattern without it being repeated per entry.
        if first and common and common.strip().lower() != "none":
            fb = f"COMMON ROOT CAUSE(S) ACROSS THESE MISTAKES:\n{common}\n\nTHIS MISTAKE:\n{fb}"
            first = False
        out[(comp, ti, kind)] = fb
    return out, cost


# ---------------------------------------------------------------------------
# GEPA adapter: the single integration point with the engine.
# ---------------------------------------------------------------------------
class InvDynAdapter(GEPAAdapter):
    """Inverse-dynamics evaluation + RC1 reflective dataset for the P/B candidate.

    Optional FORWARD-dynamics composite: score = (1-fd_weight)*ID + fd_weight*FD, where
    ID = 1[action recovered] and FD scores a predicted next-state (Fwd) against the true
    P(X_t+1). fd_scorer selects the FD metric ('textdiff' deterministic | 'judge' LLM).
    fd_weight=0 (default) -> pure inverse dynamics, identical to the validated path."""

    def __init__(
        self,
        cfg,
        action_pool,
        concurrency=16,
        image_mode=False,
        cell=8,
        fd_scorer="none",
        fd_weight=0.0,
        fd_reflect=True,
        analyze_mistakes=False,
        analyze_mode="combined",
        pred_log_path=None,
        analysis_log_path=None,
        run_ctx=None,
        context_k=0,
        reuse_traces=True,
        f_image=False,
    ):
        self.cfg = cfg
        # --f-image: F also SEES the rendered state image(s) when scoring (inverse +
        # forward). Only honoured on the K==0 path (image-mode forces context_k=0).
        self.f_image = f_image
        self.action_pool = action_pool
        # K-step temporal window shown to F (0 = two-state, validated path)
        self.context_k = context_k
        # sidecar: per-(candidate, transition) prediction detail for the viewer.
        # Keyed by content hashes so build_optim_viz.py can join without indices.
        self.pred_log_path = pred_log_path
        # sidecar: per --analyze-mistakes diagnosis call (prompt + response) for the viewer.
        self.analysis_log_path = analysis_log_path
        # shared, mutable run context (ProcessLogger stamps the live GEPA iteration here so
        # each analysis call can be stamped + lined up with its candidate in the viz).
        self.run_ctx = run_ctx
        self.concurrency = concurrency
        self.image_mode = (
            image_mode  # show rendered IMAGE (not raw text) to the P-writer
        )
        self.cell = cell
        self.fd_scorer = fd_scorer  # "none" | "textdiff" | "judge"
        self.fd_weight = 0.0 if fd_scorer == "none" else fd_weight
        self.fd_reflect = fd_reflect  # also feed forward failures to the proposer
        self.analyze_mistakes = analyze_mistakes  # LLM diagnosis -> proposer feedback
        # "combined" (default): one diagnosis call per component for ALL its shown mistakes
        # (cheaper + cross-mistake synthesis); "per-mistake": one call per mistake (original).
        self.analyze_mode = analyze_mode
        self.total_cost = 0.0
        self.eval_calls = 0  # number of FRESH per-example F calls (actual LLM work)
        # Trace reuse: every per-(candidate, transition) result is the SAME work whether it
        # is computed for the acceptance/valset eval (capture_traces=False) or the reflection
        # eval (capture_traces=True) -- only the latter keeps the trajectory. So we cache the
        # full result keyed by (candidate-content-hash, transition-hash) and serve it back when
        # the same candidate is later re-evaluated on the same transition (e.g. a child accepted
        # on the full train batch, then re-selected as a parent for reflection). This removes the
        # redundant parent re-eval entirely in the full-batch + tie-train-val regime, at zero
        # extra cost. Tradeoff: reflection then runs on the trace SAMPLE captured at first eval
        # rather than a fresh draw of a stochastic F -- disable with --fresh-traces to restore
        # GEPA's always-fresh behaviour.
        self.reuse_traces = reuse_traces
        self._trace_cache = {}  # (cand_hash, tr_key) -> (score, pred, traj)
        self.reused_evals = 0  # per-example results served from cache (no LLM call)

    # ---- forward pass -----------------------------------------------------
    def evaluate(self, batch, candidate, capture_traces=False) -> EvaluationBatch:
        code = candidate.get("perception", "")
        beliefs = candidate.get("world_knowledge", "")
        sem = asyncio.Semaphore(self.concurrency)

        async def one(inst):
            tr, choices = inst["tr"], inst["choices"]
            # windowed (K>0) reuses one perceived window for both inverse & forward;
            # K==0 perceives just the two center states (validated path).
            win = None
            if self.context_k > 0:
                win, perc_err = build_window(code, tr)
                z_t, z_t1 = win["z_t"], win["z_t1"]
            else:
                z_t, err_t = run_perceive(code, tr.x_t)
                z_t1, err_t1 = run_perceive(code, tr.x_t1)
                perc_err = err_t or err_t1
            if perc_err:  # rung-1 gradient: P crashed -> score 0, record why
                return (
                    0.0,
                    "(perception error)",
                    {
                        "tr": tr,
                        "z_t": z_t,
                        "z_t1": z_t1,
                        "choices": choices,
                        "pred": None,
                        "reasoning": "",
                        "perc_err": perc_err,
                        "id_score": 0.0,
                        "fd_score": 0.0,
                        "z_hat": "",
                        "win": win,
                    },
                )

            # Inverse and forward predictions are INDEPENDENT (forward never uses the inverse
            # result), so issue them concurrently -- they contend for the same semaphore, which
            # keeps the concurrency gate saturated instead of serialising two calls per
            # transition. Each returns a normalised (… , prompt) shape so the non-windowed path
            # carries None prompts.
            async def _inverse():
                if win is not None:
                    if self.f_image:
                        return await predict_action_from_window_img_aug(
                            self.cfg, win, beliefs, choices, sem, self.cell
                        )
                    return await predict_action_from_window(
                        self.cfg, win, beliefs, choices, sem
                    )
                if self.f_image:
                    p, r, c = await predict_action_img_aug(
                        self.cfg, z_t, z_t1, beliefs, choices, sem,
                        grid_pil(tr.x_t, self.cell), grid_pil(tr.x_t1, self.cell),
                    )
                    return p, r, c, None
                p, r, c = await predict_action(self.cfg, z_t, z_t1, beliefs, choices, sem)
                return p, r, c, None  # (pred, reasoning, cost, inv_prompt)

            async def _forward():
                if self.fd_weight <= 0.0:
                    return (
                        "",
                        0.0,
                        "",
                        None,
                    )  # (z_hat, cost, fwd_raw, fwd_prompt): no call
                if win is not None:
                    if self.f_image:
                        return await predict_next_state_from_window_img_aug(
                            self.cfg, win, tr.action, beliefs, sem, self.cell
                        )
                    return await predict_next_state_from_window(
                        self.cfg, win, tr.action, beliefs, sem
                    )
                if self.f_image:
                    zh, c = await predict_next_state_img_aug(
                        self.cfg, z_t, tr.action, beliefs, sem, grid_pil(tr.x_t, self.cell)
                    )
                    return zh, c, zh, None
                zh, c = await predict_next_state(self.cfg, z_t, tr.action, beliefs, sem)
                return zh, c, zh, None

            (
                (pred, reasoning, cost, inv_prompt),
                (z_hat, c2, fwd_raw, fwd_prompt),
            ) = await asyncio.gather(_inverse(), _forward())
            self.total_cost += cost + c2
            id_score = 1.0 if pred == tr.action else 0.0
            # forward-dynamics scoring (optional): compare generated P(X_t+1) to true z_t1.
            # (The judge scorer is itself an LLM call and depends on z_hat, so it runs after.)
            fd_score, fd_reasoning = 1.0, ""
            if self.fd_weight > 0.0:
                if self.fd_scorer == "judge":
                    fd_score, fd_reasoning, c3 = await judge_score_reasoned(
                        self.cfg, z_t, z_hat, z_t1, sem
                    )
                    self.total_cost += c3
                elif (
                    self.fd_scorer == "exact"
                ):  # 1.0 iff z_hat == P(X_t+1), else 0.0 (free)
                    fd_score = exact_match_f1(z_hat, z_t1)
                else:  # "textdiff": deterministic, free
                    fd_score = textdiff_delta_f1(z_t, z_hat, z_t1)
            score = (1.0 - self.fd_weight) * id_score + self.fd_weight * fd_score
            traj = {
                "tr": tr,
                "z_t": z_t,
                "z_t1": z_t1,
                "choices": choices,
                "pred": pred,
                "reasoning": reasoning,
                "perc_err": None,
                "id_score": id_score,
                "fd_score": fd_score,
                "z_hat": z_hat,
                "fd_reasoning": fd_reasoning,
                "win": win,
                "inv_prompt": inv_prompt,  # exact prompt F saw (inverse)
                "fwd_prompt": fwd_prompt,  # exact prompt F saw (forward)
                "fwd_raw": fwd_raw,  # raw forward response (incl. text outside <next_state>)
            }
            return score, pred, traj

        cand_hash = self._content_hash(code, beliefs)

        def _tr_key(inst):
            tr = inst["tr"]
            # transition identity + the baked choices (the inverse prompt depends on them).
            return self._content_hash(
                tr.x_t, tr.x_t1, tr.action, "||".join(inst.get("choices", []))
            )

        async def run_all():
            results = [None] * len(batch)
            todo = []  # (batch_idx, inst) that must actually call F
            for bi, inst in enumerate(batch):
                key = (cand_hash, _tr_key(inst))
                if self.reuse_traces and key in self._trace_cache:
                    results[bi] = self._trace_cache[
                        key
                    ]  # served from cache, no LLM call
                else:
                    todo.append((bi, inst))
            fresh = (
                await asyncio.gather(*(one(inst) for _, inst in todo)) if todo else []
            )
            for (bi, inst), res in zip(todo, fresh):
                results[bi] = res
                self._trace_cache[(cand_hash, _tr_key(inst))] = (
                    res  # populate for reuse
                )
            return results, len(todo)

        results, n_fresh = asyncio.run(run_all())
        self.eval_calls += n_fresh
        self.reused_evals += len(batch) - n_fresh
        scores = [r[0] for r in results]
        outputs = [r[1] for r in results]
        trajs = [r[2] for r in results] if capture_traces else None
        # sidecar: persist every candidate x transition prediction (inverse + forward)
        # so the viewer can show WHY each displayed score came out as it did.
        self._log_predictions(candidate, [r[2] for r in results], scores)
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajs)

    @staticmethod
    def _content_hash(*parts: str) -> str:
        h = hashlib.md5()
        for p in parts:
            h.update((p or "").encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()[:16]

    def _log_predictions(self, candidate, trajs, scores):
        """Append one JSON line per (candidate, transition) with the full prediction
        detail. Keyed by content hashes (process-stable, unlike id()) so the offline
        viewer can join against the reconstructed split and the candidate components.
        No-op unless a path was provided -> zero overhead for plain runs."""
        if not self.pred_log_path:
            return
        cand_hash = self._content_hash(
            candidate.get("perception", ""), candidate.get("world_knowledge", "")
        )
        p = Path(self.pred_log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            for t, s in zip(trajs, scores):
                tr = t["tr"]
                rec = {
                    "cand_hash": cand_hash,
                    "tr_hash": self._content_hash(tr.x_t, tr.x_t1, tr.action),
                    "truth": tr.action,
                    "pred": t["pred"],
                    "id_score": t["id_score"],
                    "reasoning": t["reasoning"],  # inverse-dynamics chain
                    "z_t": t["z_t"],
                    "z_t1": t["z_t1"],  # TRUE next features
                    "z_hat": t.get("z_hat", ""),  # forward prediction
                    "fd_score": t.get("fd_score"),
                    "fd_reasoning": t.get("fd_reasoning", ""),  # judge chain (if any)
                    "score": s,  # composite that the matrix shows
                }
                # Full prompt/response pair F saw for EACH objective, so the viewer can show
                # exactly what was asked and answered (the per-step PERCEIVED feature window is
                # the body of each prompt). Reconstructed deterministically from the same module
                # templates + this candidate's beliefs + the baked choices. Present only for
                # windowed runs (context_k>0); fd_* present only when a forward pred was made.
                win = t.get("win")
                if win and (win.get("prev") or win.get("nxt")):
                    beliefs = (
                        candidate.get("world_knowledge", "") or ""
                    ).strip() or "(empty)"
                    # prefer the ACTUAL prompt/response captured at the LLM call site; fall back
                    # to a deterministic reconstruction (e.g. perception-error rows never called F).
                    rec["inv_prompt"] = t.get("inv_prompt") or INV_WINDOW_TMPL.format(
                        beliefs=beliefs,
                        default_knowledge=DEFAULT_KNOWLEDGE,
                        transcript=_inverse_transcript(win),
                        choices="\n".join(f"- {c}" for c in t.get("choices", [])),
                    )
                    rec["inv_response"] = t.get(
                        "reasoning", ""
                    )  # full raw inverse response
                    if t.get("fwd_prompt") or t.get("z_hat"):
                        rec["fwd_prompt"] = t.get(
                            "fwd_prompt"
                        ) or FWD_WINDOW_TMPL.format(
                            beliefs=beliefs,
                            default_knowledge=DEFAULT_KNOWLEDGE,
                            transcript=_forward_transcript(win),
                            action=tr.action,
                        )
                        # raw forward response (full text incl. anything outside <next_state>),
                        # falling back to the parsed answer if the raw wasn't captured.
                        rec["fwd_response"] = t.get("fwd_raw") or t.get("z_hat", "")
                f.write(json.dumps(rec) + "\n")

    def _analyze_failures(self, beliefs, failures, components):
        """Optional LLM diagnosis of shown mistakes -> proposer-ready feedback.
        Returns {(comp, failure_idx, 'inv'|'fwd'): feedback}. Two modes (self.analyze_mode):
          - "combined" (default): ONE call per component diagnosing all its shown mistakes
            together (cheaper -- shared prefix sent once -- and finds cross-mistake patterns);
            structured per-mistake output so injection is unchanged.
          - "per-mistake": one call per mistake (the original behaviour).
        Runs concurrently (make_reflective_dataset is sync, so we own the event loop)."""

        def _cases_for(comp):
            cases = []
            for ti, (t, _) in enumerate(failures):
                tr = t["tr"]
                win = (
                    t["win"]
                    if (t.get("win") and (t["win"]["prev"] or t["win"]["nxt"]))
                    else None
                )
                if t.get("id_score", 1.0) < 1.0:  # inverse-dynamics mistake
                    cases.append((ti, "inv", t, tr, win))
                if (
                    self.fd_reflect
                    and t.get("z_hat")
                    and t.get("fd_score", 1.0) < 0.999
                ):
                    cases.append((ti, "fwd", t, tr, win))
            return cases

        async def run_combined():
            sem = asyncio.Semaphore(self.concurrency)
            it = (self.run_ctx or {}).get("iteration")
            jobs = [
                analyze_combined(
                    self.cfg,
                    beliefs,
                    comp,
                    cases,
                    sem,
                    log_path=self.analysis_log_path,
                    iteration=it,
                )
                for comp in components
                if (cases := _cases_for(comp))
            ]
            results = await asyncio.gather(*jobs, return_exceptions=True)
            out = {}
            for res in results:
                if isinstance(res, Exception):
                    continue
                d, cost = res
                self.total_cost += cost
                out.update(d)
            return out

        async def run_per_mistake():
            sem = asyncio.Semaphore(self.concurrency)
            it = (self.run_ctx or {}).get("iteration")
            jobs = []  # (key, coroutine)
            for comp in components:
                for ti, kind, t, tr, win in _cases_for(comp):
                    fn = analyze_inverse if kind == "inv" else analyze_forward
                    jobs.append(
                        (
                            (comp, ti, kind),
                            fn(
                                self.cfg,
                                beliefs,
                                comp,
                                t,
                                tr,
                                win,
                                sem,
                                log_path=self.analysis_log_path,
                                iteration=it,
                                ti=ti,
                            ),
                        )
                    )
            results = await asyncio.gather(
                *(c for _, c in jobs), return_exceptions=True
            )
            out = {}
            for (key, _), res in zip(jobs, results):
                if isinstance(res, Exception):
                    continue
                fb, cost = res
                self.total_cost += cost
                if fb:
                    out[key] = fb
            return out

        runner = run_combined if self.analyze_mode == "combined" else run_per_mistake
        return asyncio.run(runner())

    # ---- backward pass: build the reflective dataset (RC1 signal) ----------
    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        trajs = eval_batch.trajectories or []
        scores = eval_batch.scores
        failures = [(t, s) for t, s in zip(trajs, scores) if s < 1.0]
        corrects = [(t, s) for t, s in zip(trajs, scores) if s >= 1.0]
        beliefs = (
            candidate.get("world_knowledge", "") if isinstance(candidate, dict) else ""
        )

        # Optional: diagnose each shown mistake with an LLM into proposer-ready feedback.
        analyses = (
            self._analyze_failures(beliefs, failures[:8], components_to_update)
            if self.analyze_mistakes
            else {}
        )

        dataset: dict = {}
        for comp in components_to_update:
            records = []
            for ti, (t, _) in enumerate(failures[:8]):
                tr = t["tr"]
                if comp == "perception":
                    # Feedback is split into two clearly-separated sections: an INVERSE
                    # DYNAMICS section (is the action recoverable from P's features?) and a
                    # FORWARD PREDICTION section (are the features Markov-sufficient to
                    # predict the next state?). A perceive() crash is reported up front since
                    # it is orthogonal to both.
                    win = (
                        t["win"]
                        if (t.get("win") and (t["win"]["prev"] or t["win"]["nxt"]))
                        else None
                    )

                    # ---- INVERSE DYNAMICS section: is the action recoverable from P's
                    # features? The predicted AND the true action are surfaced together as
                    # explicit fields below; here we collect the supporting notes.
                    inv = [
                        f"F (which sees ONLY P's text output, not the frames) reasoned when "
                        f"coming up with the predicted action: {_clip_reasoning(t['reasoning'])}"
                    ]
                    if t.get("id_score", 1.0) < 1.0:
                        inv.insert(
                            0,
                            "=> INVERSE MISS: the TRUE action was NOT recoverable from the "
                            "features below.",
                        )
                    if t["z_t"] == t["z_t1"]:
                        inv.append(
                            "P produced IDENTICAL output for both states, so the action could not "
                            "be recovered from the features. If the state changed between them, the "
                            "abstraction is not moving when the world moves -- surface whatever "
                            "distinguishes consecutive states."
                        )
                    # When windowed: show the feature trajectory so the P-writer can see
                    # whether the features carry enough state to make the action recoverable
                    # ACROSS several steps (e.g. a selected/active object persists in time).
                    if win is not None:
                        inv.append(
                            "FEATURE TRAJECTORY (your perceive() output over consecutive "
                            "states; the action between each pair is shown, a_t masked):\n"
                            + _inverse_transcript(win)
                        )
                    if (comp, ti, "inv") in analyses:
                        inv.append(
                            "ANALYSIS (inverse-dynamics mistake):\n"
                            + analyses[(comp, ti, "inv")]
                        )

                    # Assemble the INVERSE DYNAMICS section: raw-state reference, the
                    # features P produced for each state, and the predicted/TRUE action pair.
                    inverse_dynamics: dict = {}
                    if (
                        self.image_mode
                    ):  # show the rendered IMAGE instead of raw-grid text
                        # ONLY THE FIRST ~400 chars, as an orientation hint. The real
                        # observation perceive() gets at runtime is the FULL grid (see
                        # OBSERVATION SCHEMA above); do not assume this prefix is complete.
                        inverse_dynamics[
                            "observation_prefix_HINT_ONLY (first ~400 chars of STATE[t]; the FULL "
                            "observation perceive() receives at runtime is much longer -- see the "
                            "OBSERVATION SCHEMA -- do NOT treat this prefix as the whole input)"
                        ] = _prefix_hint(tr.x_t, 400)
                        if win is not None:
                            # windowed: one labeled image per state in the K-step window so the
                            # P-writer can line each picture up with the feature transcript above.
                            for lbl, raw in _window_frames(win):
                                b = grid_b64(raw, self.cell)
                                if b:
                                    inverse_dynamics[f"image_STATE[{lbl}]"] = GepaImage(
                                        base64_data=b[0], media_type=b[1]
                                    )
                        else:
                            b1, b2 = (
                                grid_b64(tr.x_t, self.cell),
                                grid_b64(tr.x_t1, self.cell),
                            )
                            if b1:
                                inverse_dynamics["image_state_1"] = GepaImage(
                                    base64_data=b1[0], media_type=b1[1]
                                )
                            if b2:
                                inverse_dynamics["image_state_2"] = GepaImage(
                                    base64_data=b2[0], media_type=b2[1]
                                )
                    else:
                        inverse_dynamics[
                            "raw_state_1 (PREFIX ONLY, first 1500 chars; full observation is longer at runtime)"
                        ] = _prefix_hint(tr.x_t, 1500)
                        inverse_dynamics[
                            "raw_state_2 (PREFIX ONLY, first 1500 chars; full observation is longer at runtime)"
                        ] = _prefix_hint(tr.x_t1, 1500)
                    inverse_dynamics["perceive(state_1)"] = t["z_t"] or "(empty)"
                    inverse_dynamics["perceive(state_2)"] = t["z_t1"] or "(empty)"
                    inverse_dynamics["predicted action"] = repr(t["pred"])
                    inverse_dynamics["TRUE action"] = repr(tr.action)
                    inverse_dynamics["notes"] = "\n".join(inv)

                    # ---- FORWARD PREDICTION section: are the features Markov-sufficient to
                    # predict the next state from P(X_t)+action? ----
                    show_forward = (
                        self.fd_reflect
                        and t.get("z_hat")
                        and t.get("fd_score", 1.0) < 0.999
                    )
                    if show_forward:
                        forward_prediction = _forward_section(t, win=win)
                        if (comp, ti, "fwd") in analyses:
                            forward_prediction += (
                                "\n\nANALYSIS (forward-prediction mistake):\n"
                                + analyses[(comp, ti, "fwd")]
                            )
                    else:
                        forward_prediction = (
                            "(next features matched the true next state; no forward error "
                            "on this transition.)"
                        )

                    # ---- FEEDBACK section: actionable guidance only ----
                    fb = []
                    if t["perc_err"]:
                        fb.append(
                            f"PERCEPTION CRASHED: {t['perc_err']} -> fix so perceive() never raises."
                        )
                    if t.get("id_score", 1.0) < 1.0:
                        fb.append(
                            "INVERSE: the TRUE action was not recoverable from the features -- "
                            "surface whatever distinguishes the two states so the action becomes "
                            "identifiable."
                        )
                    if show_forward:
                        fb.append(
                            "FORWARD: the predicted next state diverged from the TRUE next state "
                            "(see the diff). Make perceive()'s features capture enough state that "
                            "the NEXT features are predictable from the current ones plus the action."
                        )
                    if not fb:
                        fb.append(
                            "This transition scored imperfectly; make the features cleaner so the "
                            "action is recoverable and the next state is predictable."
                        )

                    records.append(
                        {
                            "Inverse Dynamics": inverse_dynamics,
                            "Forward Prediction": forward_prediction,
                            "Feedback": "\n".join(fb),
                        }
                    )
                else:  # world_knowledge: distill the general convention from labels
                    # Each example carries two clearly-separated sections: an INVERSE
                    # DYNAMICS section (recover the masked action) and a FORWARD
                    # PREDICTION section (predict the next state; show the diff).
                    win = (
                        t["win"]
                        if (t.get("win") and (t["win"]["prev"] or t["win"]["nxt"]))
                        else None
                    )

                    # ---- INVERSE DYNAMICS section ----
                    if win is not None:
                        # The window already contains STATE[t]/STATE[t+1], so no separate pair.
                        inverse_section = _inverse_transcript(win)
                    else:
                        inverse_section = (
                            f"features_state_1 (CURRENT):\n{t['z_t'] or '(empty)'}\n\n"
                            f"features_state_2 (NEXT):\n{t['z_t1'] or '(empty)'}"
                        )
                    inv_fb = []
                    # INVERSE signal: only when the action was actually mis-identified --
                    # otherwise the "chose X but TRUE was X" line is noise (the example may
                    # be here purely on a FORWARD miss).
                    if t.get("id_score", 1.0) < 1.0:
                        inv_fb += [
                            f"INVERSE: the predictor chose {t['pred']!r} but the TRUE action "
                            f"was {tr.action!r}.",
                            f"F (which sees ONLY P's text features and this world knowledge, "
                            f"not the frames) reasoned: \n\n{_clip_reasoning(t['reasoning'])}",
                        ]
                    if (comp, ti, "inv") in analyses:
                        inv_fb.append(
                            "ANALYSIS (inverse-dynamics mistake):\n"
                            + analyses[(comp, ti, "inv")]
                        )

                    inverse_dynamics = {
                        "predicted action": repr(t["pred"]),
                        "TRUE action": repr(tr.action),
                        "transition (a_t masked as '??? (IDENTIFY THIS)'; TRUE a_t given above)": inverse_section,
                    }
                    # In --image-mode, ADD a rendered image of each state ALONGSIDE the
                    # feature transcript above (the raw observation TEXT is still never shown
                    # to the belief writer -- images are purely additive). Labels line up with
                    # the STATE[...] / state_1/2 references in the transition section.
                    if self.image_mode:
                        if win is not None:
                            for lbl, raw in _window_frames(win):
                                b = grid_b64(raw, self.cell)
                                if b:
                                    inverse_dynamics[f"image_STATE[{lbl}]"] = GepaImage(
                                        base64_data=b[0], media_type=b[1]
                                    )
                        else:
                            b1, b2 = (
                                grid_b64(tr.x_t, self.cell),
                                grid_b64(tr.x_t1, self.cell),
                            )
                            if b1:
                                inverse_dynamics["image_state_1"] = GepaImage(
                                    base64_data=b1[0], media_type=b1[1]
                                )
                            if b2:
                                inverse_dynamics["image_state_2"] = GepaImage(
                                    base64_data=b2[0], media_type=b2[1]
                                )
                        inverse_dynamics["image_note"] = (
                            "A rendered IMAGE of each state above is ATTACHED, labeled to match "
                            "the STATE[...] (or state_1/state_2) references in the transition. "
                            "Use the images to sanity-check what P's features describe; the raw "
                            "observation text itself is deliberately not shown."
                        )

                    # ---- FORWARD PREDICTION section ----
                    fwd_fb = []
                    show_forward = (
                        self.fd_reflect
                        and t.get("z_hat")
                        and t.get("fd_score", 1.0) < 0.999
                    )
                    if show_forward:
                        forward_prediction = _forward_section(t, win=win)
                        if (comp, ti, "fwd") in analyses:
                            fwd_fb.append(
                                "ANALYSIS (forward-prediction mistake):\n"
                                + analyses[(comp, ti, "fwd")]
                            )
                        fwd_fb.append(
                            "FORWARD: the predicted next state diverged from the TRUE next state "
                            "(see the diff). Refine the world knowledge so its dynamics rules "
                            "predict the NEXT features from the current ones plus the action."
                        )
                    else:
                        forward_prediction = (
                            "(next features matched the true next state; no forward error "
                            "on this transition.)"
                        )

                    fb = inv_fb + fwd_fb
                    if (
                        not fb
                    ):  # selected as imperfect but neither signal is displayable
                        fb.append(
                            "This transition scored imperfectly; refine the world knowledge "
                            "so feature changes map cleanly to action names and dynamics."
                        )
                    records.append(
                        {
                            "Inverse Dynamics": inverse_dynamics,
                            "Forward Prediction": forward_prediction,
                            "Feedback": "\n".join(fb),
                        }
                    )
            # a couple of CORRECT cases for contrast (keep what already works)
            for t, _ in corrects[:2]:
                records.append(
                    {
                        "Inverse Dynamics": {
                            "perceive(state_1)": t["z_t"],
                            "perceive(state_2)": t["z_t1"],
                            "predicted action": repr(t["tr"].action),
                            "TRUE action": repr(t["tr"].action),
                        },
                        "Forward Prediction": "(correct case, shown for contrast)",
                        "Feedback": f"Correctly identified {t['tr'].action!r} -- preserve the behavior that made this work.",
                    }
                )
            dataset[comp] = records or [
                {
                    "Inverse Dynamics": "(no failures in this minibatch)",
                    "Forward Prediction": "(no failures in this minibatch)",
                    "Feedback": "No failures in this minibatch; keep the component as-is or make it more robust.",
                }
            ]
        return dataset


# ---------------------------------------------------------------------------
# reflection_lm: route GEPA's proposal calls through our OpenRouter plumbing.
# ---------------------------------------------------------------------------
_REFLECTION = {"cost": 0.0, "calls": 0}


def _data_uri_to_pil(uri: str):
    try:
        b64 = uri.split(",", 1)[1] if uri.startswith("data:") else uri
        return PILImage.open(BytesIO(base64.b64decode(b64)))
    except Exception:  # noqa: BLE001
        return None


def _flatten_prompt(prompt):
    """GEPA passes str, or a multimodal list (OpenAI vision parts) when the
    reflective dataset carries Images. Return (text, [PIL images])."""
    if isinstance(prompt, str):
        return prompt, []
    texts, imgs = [], []

    def walk(x):
        if isinstance(x, str):
            texts.append(x)
        elif isinstance(x, dict):
            t = x.get("type")
            if t in ("text", "input_text") and "text" in x:
                texts.append(x["text"])
            elif t in ("image_url", "input_image"):
                u = x.get("image_url") or x.get("url") or x.get("image")
                url = u.get("url") if isinstance(u, dict) else u
                im = _data_uri_to_pil(url) if isinstance(url, str) else None
                if im is not None:
                    imgs.append(im)
            elif "content" in x:  # a message dict
                walk(x["content"])
            elif "text" in x:
                texts.append(x["text"])
        elif isinstance(x, list):
            for it in x:
                walk(it)

    walk(prompt)
    return "\n".join(texts), imgs


def _log_reflection(log_path, prompt, response, attempts):
    """Append one proposer call (full prompt + response) to reflection_calls.jsonl.
    Off unless --log-reflection passes a path. component is inferred from the template text."""
    if not log_path:
        return
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    comp = (
        "world_knowledge"
        if "WORLD KNOWLEDGE" in prompt
        else "perception"
        if "perceive(" in prompt
        else "?"
    )
    rec = {
        "call": _REFLECTION["calls"],
        "ts": time.strftime("%H:%M:%S"),
        "component": comp,
        "attempts": attempts,
        "prompt": prompt,
        "response": response,
    }
    with p.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def make_reflection_lm(cfg, retries=4, log_path=None):
    def fn(prompt):
        text, images = _flatten_prompt(prompt)
        # models intermittently return None/empty (MALFORMED) -> retry, NEVER return
        # None (GEPA calls .strip() on the result).
        attempts = 0
        for _ in range(retries):
            out, cost = asyncio.run(_llm_call(cfg, text, images=images or None))
            _REFLECTION["cost"] += cost
            _REFLECTION["calls"] += 1
            attempts += 1
            if out and out.strip():
                _log_reflection(log_path, text, out, attempts)
                return out
        _log_reflection(
            log_path, text, "", attempts
        )  # also record failed/empty proposals
        return ""  # last resort: empty -> GEPA scores candidate as not-better, skips it

    return fn


class ProcessLogger:
    """GEPA callback that persists the FULL search process -- including REJECTED proposals --
    to <run_dir>/process_log.jsonl, one JSON record per iteration. GEPA's own state only keeps
    candidates that PASSED the subsample gate; the dead ends (and the feedback that produced
    them) are dropped after the callback fires. This logger records, per iteration:

      - the selected parent candidate (idx + score),
      - the mistake feedback shown to the proposer (the reflective_dataset, per component) --
        this is the analyze-mistakes diagnosis + inverse/forward transcripts,
      - the proposer's RESPONSE (the new component text), kept even when the candidate is
        rejected,
      - the accept / reject / skip verdict with the subsample scores.

    That is exactly what the viz needs to draw the search as a tree with every dead end and
    associate the proposer prompt-feedback + response with each candidate. Always on; local,
    cheap, append-only (so it continues across --max-metric-calls resumes)."""

    def __init__(self, path, run_ctx=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cur = None
        # shared dict the adapter reads to stamp the current GEPA iteration onto its
        # analysis-call log (so the viz can line each diagnosis up with its candidate).
        self.run_ctx = run_ctx

    @staticmethod
    def _safe(obj):
        # reflective datasets embed images / non-JSON objects -> coerce to str rather than crash.
        return json.loads(json.dumps(obj, default=str))

    def on_iteration_start(self, event):
        self.cur = {"i": event["iteration"]}
        if self.run_ctx is not None:
            self.run_ctx["iteration"] = event["iteration"]

    def on_candidate_selected(self, event):
        if self.cur is None:
            return
        self.cur["selected"] = event.get("candidate_idx")
        self.cur["selected_score"] = event.get("score")

    def on_minibatch_sampled(self, event):
        if self.cur is None:
            return
        self.cur["minibatch_ids"] = list(event.get("minibatch_ids") or [])

    def on_reflective_dataset_built(self, event):
        if self.cur is None:
            return
        self.cur["components"] = list(event.get("components") or [])
        try:
            self.cur["feedback"] = self._safe(event.get("dataset"))
        except Exception:  # noqa: BLE001
            self.cur["feedback"] = None

    def on_proposal_end(self, event):
        if self.cur is None:
            return
        try:
            self.cur["proposed"] = self._safe(event.get("new_instructions") or {})
        except Exception:  # noqa: BLE001
            self.cur["proposed"] = None

    def on_candidate_accepted(self, event):
        if self.cur is None:
            return
        self.cur["verdict"] = "accepted"
        self.cur["new_idx"] = event.get("new_candidate_idx")
        self.cur["new_score"] = event.get("new_score")
        self.cur["parent_ids"] = list(event.get("parent_ids") or [])

    def on_candidate_rejected(self, event):
        if self.cur is None:
            return
        self.cur["verdict"] = "rejected"
        self.cur["old_score"] = event.get("old_score")
        self.cur["new_score"] = event.get("new_score")
        self.cur["reason"] = event.get("reason")

    def on_iteration_end(self, event):
        if self.cur is None:
            return
        self.cur.setdefault("verdict", "skipped")
        self.cur["accepted"] = bool(event.get("proposal_accepted"))
        with self.path.open("a") as f:
            f.write(json.dumps(self.cur, default=str) + "\n")
        self.cur = None


# Observation schema: FORMAT-ONLY description of what perceive() receives at runtime.
# Describes structure + (ARC) the integer->colour palette so the writer can parse the grid.
# Deliberately says NOTHING about which colours are the agent/walls/targets or how any
# action behaves (no layout, no dynamics).
# SCOPED to the active env via observation_schema(env_name): an autumn run is NOT shown the
# ARC palette and an ARC run is NOT shown the autumn format. Showing both let the writer
# copy the WRONG env's format into P (e.g. dead ARC integer-grid code + palette in an autumn
# module) -- a cross-env knowledge leak. Unknown env -> both (offline main has no env handle).
_SCHEMA_INTRO = """OBSERVATION SCHEMA (this describes the string `observation_history[-1]` that perceive() is given at RUNTIME -- it is the COMPLETE observation, much longer than any snippet shown to you here):
- It is a single TEXT string (not a Python object). perceive() must parse it from text"""

_SCHEMA_ARC = """  (A) ARC integer grid -- a marker line `<grid_0>` (there may be several `<grid_k>` blocks), followed by the grid ROWS, each row on its own line as a bracketed comma-separated list of integers, e.g.:
        State: NOT_FINISHED
        Levels completed: 0/6
        Action count: 0
        ========== Start of Direct Observation ==========
        <grid_0>
        [14, 14, 14, ... , 14]
        [12, 0, 0, ... , 12]
        ... (many more rows)
      The grid is a rectangular list-of-lists of ints, often large (e.g. 64x64), so it spans MANY lines. Each int is a CELL COLOUR CODE in this fixed palette:
        0=black  1=blue  2=red  3=green  4=yellow  5=light-gray  6=magenta
        7=orange  8=light-blue  9=dark-red/maroon  10=white  11=dark-gray
        (integers outside 0-11 can occur; they render as a neutral gray.)
      The IMAGE you are shown is exactly these integers painted with the palette above, so a region you SEE as e.g. maroon corresponds to grid cells whose value is 9, sky-blue to 8, green to 3, etc. Use the palette to translate what you see in the image into the integer values your text parser must look for."""

_SCHEMA_AUTUMN = """  (B) Autumn string grid -- a JSON 2D array of lowercase colour-NAME strings embedded in the text, e.g. [["black","gold",...],["black","blue",...], ...]. Cells are names like "black","gray","blue","red","green","gold","lightblue","darkgreen".
      IMPORTANT: the whole 2D array is one CONTIGUOUS JSON value (it is NOT split one-row-per-line; do not iterate over text lines looking for a row each). The first row is NOT always black (e.g. it may start with [["gray",...). Parse it robustly and directly, e.g.:
          s = obs.find("[["); e = obs.rfind("]]") + 2
          grid = json.loads(obs[s:e])   # grid[r][c] is the colour-name string at row r, col c
      Then summarise the cells that are NOT the dominant/background colour (their (row,col,colour)), so the action between two states is recoverable from the two summaries."""

_SCHEMA_FOOTER = """- Parse defensively: the grid may be 64+ rows; never assume only the first row(s) exist; never assume a fixed length per row; on any parse error return a best-effort summary string (never raise, never return empty)."""


def observation_schema(env_name: str | None = None) -> str:
    """The OBSERVATION SCHEMA scoped to env_name: only the relevant grid format/palette."""
    if env_name == "autumn":
        body, conj = _SCHEMA_AUTUMN, ", encoded as the grid below:"
    elif env_name == "arc_agi":
        body, conj = _SCHEMA_ARC, ", encoded as the grid below:"
    else:  # unknown -> describe BOTH (back-compat; offline main has no env handle)
        body = _SCHEMA_ARC + "\n" + _SCHEMA_AUTUMN
        conj = " and must handle either of the two encodings below (detect which is present):"
    return f"{_SCHEMA_INTRO}{conj}\n{body}\n{_SCHEMA_FOOTER}"


OBSERVATION_SCHEMA = observation_schema(
    None
)  # module default (both formats), back-compat

# Per-component proposal templates: GEPA's default framing is "write a new
# instruction"; for the code component we reframe to "rewrite the Python module".
REFLECTION_TEMPLATES = {
    "perception": f"""You are improving a Python perception module for a grid environment.

It must define `perceive(observation_history: list[str]) -> str`; observation_history[-1] is the current raw observation. It must output a concise (<2000 char) text summary of decision-relevant features, never raise, never return empty. Its output (for two consecutive states) is consumed WITHOUT the raw grid to identify the action taken between them.

{OBSERVATION_SCHEMA}

Current module:
```
<curr_param>
```

Execution feedback (each failure shows the predicted vs TRUE action, the reasoning of F -- which sees ONLY P's text output, not the frames -- when coming up with the predicted action, and whether P emitted IDENTICAL features for both states -- read these carefully; if P's output does not change between two consecutive states, the abstraction is dropping decision-relevant state):
```
<side_info>
```

Rewrite the FULL module so its output moves whenever the world moves and makes the action recoverable from two consecutive outputs. Provide the complete module within ``` blocks.""",
    "world_knowledge": (
        "You maintain a WORLD KNOWLEDGE block: a concise and general understanding of how this environment works.\n\n"
        "=== DEFAULT KNOWLEDGE (always-true facts; fixed and always supplied to the predictors alongside your block) ===\n"
        f"{DEFAULT_KNOWLEDGE}\n"
        "=== END DEFAULT KNOWLEDGE ===\n\n"
        "Do NOT restate the DEFAULT KNOWLEDGE in your block -- it is always provided; add only what it does not already cover.\n\n"
        "Current world knowledge:\n"
        "```\n"
        "<curr_param>\n"
        "```\n\n"
        "Feedback:\n"
        "```\n"
        "<side_info>\n"
        "```\n\n"
        "CRITICAL -- GENERALIZE, do NOT memorize the dataset: never write rules that key off a specific step/timestep number or a specific concrete state/coordinate set and map it to an action (e.g. 'Step 3 -> left', 'when points are {(2,2),(2,6),...} -> right'). The step counter and absolute positions do NOT determine the action and such per-instance mappings will NOT transfer to held-out states -- they are memorization, not knowledge. State ONLY general, transferable dynamics: how each action transforms the features in a way that applies to ANY state (e.g. 'left shifts every point one column toward smaller x'). If an action's effect is genuinely indeterminate from the state change, say so as a general rule rather than enumerating instances.\n\n"
        "Rewrite the FULL world knowledge block: concise, general, sufficient to map feature changes to action names. Provide it within ``` blocks."
    ),
}


def build_reflection_templates(env_name: str | None = None) -> dict:
    """REFLECTION_TEMPLATES with the OBSERVATION SCHEMA scoped to env_name (autumn/arc_agi ->
    only that grid format; None -> both). Pass the running env to avoid leaking the other
    env's format/palette into the proposed perception code."""
    return {
        "perception": REFLECTION_TEMPLATES["perception"].replace(
            OBSERVATION_SCHEMA, observation_schema(env_name)
        ),
        "world_knowledge": REFLECTION_TEMPLATES["world_knowledge"],
    }


class PerceptionBiasedComponentSelector:
    """GEPA component selector that updates `belief_component` only ONCE every
    `belief_period` selections; every other iteration updates perception. Plain
    round-robin alternates 50/50, which over-trains the belief block and lets it
    overfit (memorizing per-step/per-state -> action mappings). Lowering the
    belief cadence keeps optimization pressure on the generalizable perception
    code. belief_period=1 reproduces "beliefs every iteration"; large values
    nearly freeze beliefs."""

    def __init__(self, belief_period: int = 4, belief_component: str = "world_knowledge"):
        self.belief_period = max(1, int(belief_period))
        self.belief_component = belief_component
        self._n = 0

    def __call__(self, state, trajectories, subsample_scores, candidate_idx, candidate):
        names = list(state.list_of_named_predictors)
        others = [n for n in names if n != self.belief_component]
        self._n += 1
        if self.belief_component in names and others and (self._n % self.belief_period == 0):
            return [self.belief_component]
        return [others[0]] if others else [names[0]]


def _clean_component(comp: str, text: str) -> str:
    """GEPA returns the proposed text already stripped of ``` fences for the
    instruction; but if a code fence slipped through, extract it for perception."""
    if comp == "perception" and "def perceive" not in text:
        return _extract_code(text)
    return text


# ---------------------------------------------------------------------------
# Image-mode helpers: prompts show the rendered IMAGE instead of raw-grid text.
# (P still parses the text grid at runtime.)
# ---------------------------------------------------------------------------
PREDICT_IMG_TMPL = """You identify which action was taken between two states of a grid environment, shown as two IMAGES.

=== DEFAULT KNOWLEDGE (always-true facts about this environment) ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== WORLD KNOWLEDGE (general facts; may be empty) ===
{beliefs}
=== END WORLD KNOWLEDGE ===

The FIRST image is STATE 1, the SECOND image is STATE 2. Exactly one action was taken to get from STATE 1 to STATE 2.

The action was one of:
{choices}

Decide which single action was taken, using the two images. Respond as:
<reasoning>what changed between the images and what action that implies</reasoning>
<action>the chosen action, copied verbatim from the list</action>"""


async def predict_action_img(cfg, img_t, img_t1, beliefs, choices, sem):
    prompt = PREDICT_IMG_TMPL.format(
        beliefs=beliefs.strip() or "(empty)",
        default_knowledge=DEFAULT_KNOWLEDGE,
        choices="\n".join(f"- {c}" for c in choices),
    )
    async with sem:
        text, cost = await _llm_call(cfg, prompt, images=[img_t, img_t1])
    text = text or ""
    pred = _extract_action(text, choices)
    return pred, text, cost


# ---------------------------------------------------------------------------
# F-sees-image variants of the SCORING predictors (--f-image). Unlike
# predict_action_img (the raw-frame BASELINE, which sees ONLY images), these show F
# BOTH P's text features AND the rendered state image(s): the experiment is whether
# letting F ground the features against the actual pixels changes the inverse/forward
# objective. NOTE: this relaxes the P bottleneck (F can read pixels directly), so it
# is NOT a pure-P signal -- use only for the deliberate "show images to F" comparison.
# ---------------------------------------------------------------------------
PREDICT_IMG_AUG_TMPL = """You identify which action was taken between two states of a grid environment.

=== DEFAULT KNOWLEDGE (always-true facts about this environment) ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== WORLD KNOWLEDGE (general facts; may be empty) ===
{beliefs}
=== END WORLD KNOWLEDGE ===

A perception module summarized two consecutive states. Exactly one action was taken to get from STATE 1 to STATE 2. You are ALSO shown the two raw states as IMAGES (the FIRST image is STATE 1, the SECOND image is STATE 2).

=== STATE 1 features ===
{z_t}
=== STATE 2 features ===
{z_t1}

The action was one of:
{choices}

Decide which single action was taken, using the features AND the images. Respond as:
<reasoning>what changed and what action that implies</reasoning>
<action>the chosen action, copied verbatim from the list</action>"""


async def predict_action_img_aug(cfg, z_t, z_t1, beliefs, choices, sem, img_t, img_t1):
    prompt = PREDICT_IMG_AUG_TMPL.format(
        beliefs=beliefs.strip() or "(empty)",
        default_knowledge=DEFAULT_KNOWLEDGE,
        z_t=z_t or "(empty)",
        z_t1=z_t1 or "(empty)",
        choices="\n".join(f"- {c}" for c in choices),
    )
    imgs = [im for im in (img_t, img_t1) if im is not None]
    async with sem:
        text, cost = await _llm_call(cfg, prompt, images=imgs or None)
    text = text or ""
    pred = _extract_action(text, choices)
    return pred, _parse_tag(text, "reasoning"), cost


FORWARD_IMG_AUG_TMPL = """You predict the NEXT-state features of a grid environment from the current features, the current state IMAGE, and an action.

=== DEFAULT KNOWLEDGE (always-true facts about this environment) ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== WORLD KNOWLEDGE (general facts; may be empty) ===
{beliefs}
=== END WORLD KNOWLEDGE ===

A perception module summarized the CURRENT state as these features. The IMAGE of the current state is also shown:
=== CURRENT features ===
{z_t}
=== END CURRENT features ===

The action taken was: {action}

Predict the features of the resulting NEXT state, written in EXACTLY the same format and vocabulary the perception module uses above (same keys, same coordinate/colour conventions). Change only what the action changes; copy everything else unchanged. Do NOT add commentary.

<next_state>predicted next-state features, same format as CURRENT</next_state>"""


async def predict_next_state_img_aug(cfg, z_t, action, beliefs, sem, img_t):
    prompt = FORWARD_IMG_AUG_TMPL.format(
        beliefs=beliefs.strip() or "(empty)",
        default_knowledge=DEFAULT_KNOWLEDGE,
        z_t=z_t or "(empty)",
        action=action,
    )
    imgs = [img_t] if img_t is not None else None
    async with sem:
        text, cost = await _llm_call(cfg, prompt, images=imgs)
    text = text or ""
    z_hat = _parse_tag(text, "next_state") or text.strip()
    return z_hat, cost


# ---------------------------------------------------------------------------
# Windowed + F-sees-image variants (--f-image with --context-k>0): F sees the K-step
# text transcript AND one rendered image per window state, labeled so it can line each
# picture up with the corresponding STATE[...] in the transcript. Same pure-P caveat as
# the two-state img_aug predictors -- F can read pixels directly, so use only for the
# deliberate "show images to F" comparison.
# ---------------------------------------------------------------------------
def _window_images(frames, cell):
    """(image_legend_text, [PIL]) for an ordered [(label, raw)] window: render each state and
    describe its position so the model can map image N -> STATE[label]. States that fail to
    render are skipped, and the legend only lists the images actually attached."""
    legend, imgs = [], []
    for lbl, raw in frames:
        im = grid_pil(raw, cell)
        if im is None:
            continue
        imgs.append(im)
        legend.append(f"  image {len(imgs)} = STATE[{lbl}]")
    return ("\n".join(legend) or "(no images could be rendered)"), imgs


INV_WINDOW_IMG_AUG_TMPL = """You identify a single HIDDEN action in a trajectory of a grid environment.

=== DEFAULT KNOWLEDGE (always-true facts about this environment) ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== WORLD KNOWLEDGE (general facts; may be empty) ===
{beliefs}
=== END WORLD KNOWLEDGE ===

Below is a trajectory of consecutive states (summarized as features by a perception module) with the action taken between each pair. ONE action is hidden and marked `??? (IDENTIFY THIS)`. Use the WHOLE trajectory -- the states and actions BEFORE and AFTER the gap -- to infer the hidden action; temporally-extended patterns (a selected/active object, momentum, periodicity) may only be visible across several steps.

{transcript}

You are ALSO shown the raw states as IMAGES, attached in this order:
{image_legend}

The hidden action was one of:
{choices}

Respond as:
<reasoning>what the surrounding states and actions imply about the hidden action</reasoning>
<action>the chosen action, copied verbatim from the list</action>"""


async def predict_action_from_window_img_aug(cfg, win, beliefs, choices, sem, cell):
    legend, imgs = _window_images(_window_frames(win), cell)
    prompt = INV_WINDOW_IMG_AUG_TMPL.format(
        beliefs=beliefs.strip() or "(empty)",
        default_knowledge=DEFAULT_KNOWLEDGE,
        transcript=_inverse_transcript(win),
        image_legend=legend,
        choices="\n".join(f"- {c}" for c in choices),
    )
    async with sem:
        text, cost = await _llm_call(cfg, prompt, images=imgs or None)
    text = text or ""
    pred = _extract_action(text, choices)
    return pred, text, cost, prompt  # text = raw response, prompt = exact prompt F saw


FWD_WINDOW_IMG_AUG_TMPL = """You predict the NEXT-state features of a grid environment from a trajectory and the action just taken.

=== DEFAULT KNOWLEDGE (always-true facts about this environment) ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== WORLD KNOWLEDGE (general facts; may be empty) ===
{beliefs}
=== END WORLD KNOWLEDGE ===

Below is a trajectory of consecutive states (summarized as features) ending at the CURRENT state, with the action taken between each pair. Use the WHOLE history to capture dynamics that depend on more than the current state alone.

{transcript}

You are ALSO shown the raw states as IMAGES, attached in this order:
{image_legend}

The action now taken from the CURRENT state is: {action}

Predict the features of the resulting NEXT state, in EXACTLY the same format and vocabulary the perception module uses above (same keys, same coordinate/colour conventions). Change only what the action changes; copy everything else unchanged. Do NOT add commentary.

<next_state>predicted next-state features, same format as CURRENT</next_state>"""


async def predict_next_state_from_window_img_aug(cfg, win, action, beliefs, sem, cell):
    legend, imgs = _window_images(_window_frames_forward(win), cell)
    prompt = FWD_WINDOW_IMG_AUG_TMPL.format(
        beliefs=beliefs.strip() or "(empty)",
        default_knowledge=DEFAULT_KNOWLEDGE,
        transcript=_forward_transcript(win),
        image_legend=legend,
        action=action,
    )
    async with sem:
        text, cost = await _llm_call(cfg, prompt, images=imgs or None)
    text = text or ""
    z_hat = _parse_tag(text, "next_state") or text.strip()
    return z_hat, cost, text, prompt  # text = raw response, prompt = exact prompt F saw


UPDATE_P_IMG_TMPL = """You are improving a Python perception module for a grid environment.

It must define `perceive(observation_history: list[str]) -> str`. `observation_history[-1]` is the current raw observation TEXT, which your code must parse. Output a concise (<2000 char) text summary of decision-relevant features; never raise, never return empty. Its output (for two consecutive states) is consumed WITHOUT the raw grid to identify the action taken.

A separate WORLD KNOWLEDGE block supplies conventions, so DO NOT put conventions here -- only EXTRACT state features:
=== DEFAULT KNOWLEDGE (always-true facts about this environment) ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== WORLD KNOWLEDGE ===
{beliefs}
=== END WORLD KNOWLEDGE ===

=== CURRENT PERCEPTION MODULE ===
```python
{code}
```

Diagnosed PERCEPTION deficiencies (the gradient):
{g1_p}

Below are IMAGES of example consecutive states so you can SEE what is in the grid and what changes. The raw observation your code receives is TEXT in this format (parse it to recover the grid):
{fmt_hint}

Rewrite the FULL module to fix the deficiencies so its output moves whenever the world moves. Respond with the full module in one block:
```python
<code>
```"""


async def update_perception_img(
    cfg, code, g1_p, beliefs, raw_examples, sem, cell=8, n_examples=2
):
    imgs, hint = [], ""
    for tr in raw_examples[:n_examples]:
        for raw in (tr.x_t, tr.x_t1):
            im = grid_pil(raw, cell)
            if im is not None:
                imgs.append(im)
        if not hint:
            hint = tr.x_t[:400]
    prompt = UPDATE_P_IMG_TMPL.format(
        beliefs=beliefs.strip() or "(empty)",
        default_knowledge=DEFAULT_KNOWLEDGE,
        code=code or "# (empty)",
        g1_p=g1_p or "(none)",
        fmt_hint=hint,
    )
    async with sem:
        text, cost = await _llm_call(cfg, prompt, images=imgs or None)
    return _extract_code(text or ""), cost


# ---------------------------------------------------------------------------
# Legacy greedy loop (for --compare): the current implementation's logic,
# re-driven here on the SAME split so we can score it on the SAME clean test.
# (validate_beliefs.py itself is left untouched.)
# ---------------------------------------------------------------------------
async def run_legacy_loop(
    cfg,
    train,
    holdout,
    action_pool,
    k,
    sem,
    rng,
    rounds,
    start_code,
    image_mode=False,
    cell=8,
):
    code, beliefs = start_code, ""
    cost = 0.0
    best_acc, best_code, best_beliefs, c = await _legacy_eval(
        cfg, code, beliefs, holdout, action_pool, k, sem, rng
    )
    cost += c
    for rnd in range(1, rounds + 1):
        active = "P" if rnd % 2 == 1 else "B"
        _, recs, c = await forward_eval(
            cfg, code, beliefs, train, action_pool, k, sem, rng
        )
        cost += c
        failures = [r for r in recs if not r.correct]
        g1_p, g1_b, c = await compute_g1(cfg, failures, beliefs, sem)
        cost += c
        if active == "P":
            exs = [r.tr for r in failures] or [r.tr for r in recs]
            if image_mode:
                cand, c = await update_perception_img(
                    cfg, code, g1_p, beliefs, exs, sem, cell=cell
                )
                cost += c
            else:
                cand, c = await update_perception(cfg, code, g1_p, beliefs, exs, sem)
                cost += c
            ok, _ = perception_runs(cand, [t.x_t for t in train[:4]])
            if not ok:
                continue
            new_acc, _, c = await forward_eval(
                cfg, cand, beliefs, holdout, action_pool, k, sem, rng
            )
            cost += c
            if new_acc >= best_acc:
                best_acc, best_code, code = new_acc, cand, cand
        else:
            cand, c = await update_beliefs(cfg, beliefs, g1_b, failures or recs, sem)
            cost += c
            new_acc, _, c = await forward_eval(
                cfg, code, cand, holdout, action_pool, k, sem, rng
            )
            cost += c
            if new_acc >= best_acc:
                best_acc, best_beliefs, beliefs = new_acc, cand, cand
    return best_code, best_beliefs, cost


async def _legacy_eval(cfg, code, beliefs, data, action_pool, k, sem, rng):
    acc, recs, cost = await forward_eval(
        cfg, code, beliefs, data, action_pool, k, sem, rng
    )
    return acc, code, beliefs, cost


# ---------------------------------------------------------------------------
async def eval_on(
    cfg,
    code,
    beliefs,
    baked,
    raw_mode=False,
    concurrency=16,
    image_mode=False,
    cell=8,
    log_path=None,
    context_k=0,
):
    """Evaluate on a list of baked {tr, choices} dicts using FIXED choice sets, so
    every method (GEPA, legacy, baselines) faces identical choices on the test set.
    raw_mode + image_mode -> the raw-frame reference shows the model the IMAGES.

    If log_path is given, write a self-contained per-item trace (the learned P and B
    plus, for every test item, P's output on both frames, F's reasoning, the chosen
    vs true action and the deterministic raw change). This lets later failure analysis
    read the trace directly -- no log reconstruction and no LLM replay needed."""
    sem = asyncio.Semaphore(concurrency)

    async def one(idx, inst):
        tr, choices = inst["tr"], inst["choices"]
        if raw_mode and image_mode:
            pred, reasoning, cost = await predict_action_img(
                cfg,
                grid_pil(tr.x_t, cell),
                grid_pil(tr.x_t1, cell),
                beliefs,
                choices,
                sem,
            )
            z_t = z_t1 = "<image>"
        elif (not raw_mode) and context_k > 0:
            # windowed test: same K-step transcript the optimizer trained F on
            win, _ = build_window(code, tr)
            z_t, z_t1 = win["z_t"], win["z_t1"]
            pred, reasoning, cost, _ = await predict_action_from_window(
                cfg, win, beliefs, choices, sem
            )
        else:
            if raw_mode:
                z_t, z_t1 = tr.x_t[:6000], tr.x_t1[:6000]
            else:
                z_t, z_t1 = (
                    run_perceive(code, tr.x_t)[0],
                    run_perceive(code, tr.x_t1)[0],
                )
            pred, reasoning, cost = await predict_action(
                cfg, z_t, z_t1, beliefs, choices, sem
            )
        rec = None
        if log_path is not None:
            rec = {
                "idx": idx,
                "truth": tr.action,
                "pred": pred,
                "correct": pred == tr.action,
                "choices": list(choices),
                "z_t": str(z_t),
                "z_t1": str(z_t1),
                "reasoning": reasoning,
            }
        return (1.0 if pred == tr.action else 0.0), cost, rec

    res = await asyncio.gather(*(one(i, inst) for i, inst in enumerate(baked)))
    acc = sum(r[0] for r in res) / max(1, len(res))
    if log_path is not None:
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "acc": acc,
                    "raw_mode": raw_mode,
                    "image_mode": image_mode,
                    "perception": code,
                    "beliefs": beliefs,
                    "records": [r[2] for r in res],
                },
                indent=2,
                default=str,
            )
        )
    return acc, sum(r[1] for r in res)


async def eval_fd_on(cfg, code, beliefs, baked, scorer, concurrency=16, context_k=0):
    """Mean forward-dynamics score on the clean test set: generate Z_hat = Fwd(history,
    A, B) and score it against the TRUE next features P(X_t+1). Secondary readout (the
    headline is test ID accuracy from eval_on); lets us see forward quality per arm."""
    sem = asyncio.Semaphore(concurrency)

    async def one(inst):
        tr = inst["tr"]
        if context_k > 0:
            win, _ = build_window(code, tr)
            z_t, z_t1 = win["z_t"], win["z_t1"]
            z_hat, c, _, _ = await predict_next_state_from_window(
                cfg, win, tr.action, beliefs, sem
            )
        else:
            z_t = run_perceive(code, tr.x_t)[0]
            z_t1 = run_perceive(code, tr.x_t1)[0]
            z_hat, c = await predict_next_state(cfg, z_t, tr.action, beliefs, sem)
        if scorer == "judge":
            s, c2 = await judge_score(cfg, z_t, z_hat, z_t1, sem)
            c += c2
        elif scorer == "exact":
            s = exact_match_f1(z_hat, z_t1)
        else:
            s = textdiff_delta_f1(z_t, z_hat, z_t1)
        return s, c

    res = await asyncio.gather(*(one(i) for i in baked))
    return sum(r[0] for r in res) / max(1, len(res)), sum(r[1] for r in res)


def bake_choices(transitions, action_pool, k, rng):
    """Fix one choice set per transition so scores are comparable across candidates."""
    return [
        {"tr": tr, "choices": make_choices(tr.action, action_pool, k, rng)}
        for tr in transitions
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="comma-separated run dirs (pooled)")
    ap.add_argument(
        "--test-run", default=None,
        help="comma-separated run dirs to draw the held-out TEST set from, kept entirely "
        "separate from --run (which then supplies only train/val). Use this for a "
        "cross-trajectory generalization test: train on one crafted trajectory, test on a "
        "distinct one demonstrating the same dynamics. When set, the in-pool test carve is "
        "skipped and --swap-click-train / --pin-train-idx are ignored.")
    ap.add_argument("--train-n", type=int, default=20)
    ap.add_argument(
        "--val-n", type=int, default=20
    )  # GEPA selection set (= our held-out gate)
    ap.add_argument(
        "--test-n", type=int, default=10
    )  # clean test neither method touches
    ap.add_argument("--k-choices", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument(
        "--task-model",
        default="google/gemini-2.5-flash",
        help="frozen decoder F (the agent)",
    )
    ap.add_argument(
        "--reflection-model",
        default=None,
        help="proposer LM (defaults to --task-model so GEPA and legacy share one model)",
    )
    ap.add_argument("--client", default="openrouter")
    ap.add_argument("--max-metric-calls", type=int, default=2000)
    ap.add_argument("--actions", default="left,right,up,down,noop,click")
    ap.add_argument(
        "--collapse-action-params",
        action="store_true",
        help="collapse parameterized actions to their verb (e.g. 'click 3 5' -> "
        "'click'). Default is to KEEP full action strings everywhere in the "
        "pipeline so the LOCATION is part of the prediction target / feedback.",
    )
    ap.add_argument(
        "--tie-train-val",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="low-data regime: val = the SAME samples as train (size --train-n); test unchanged. "
        "Default ON; --no-tie-train-val restores a distinct train/val split.",
    )
    ap.add_argument(
        "--swap-click-train",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="after the split, swap ONE click transition from test INTO train (and one 'up' "
        "from train INTO test) so the SCORED train/val set contains a click-identification "
        "item -- lets a correct click belief earn score. A click remains in the clean test set.",
    )
    ap.add_argument(
        "--good-baseline",
        action="store_true",
        help="also report the DQ8GC-specific single-cell GOOD_P ceiling (DQ8GC only)",
    )
    ap.add_argument(
        "--pin-train-idx",
        default="",
        help="comma-separated ORIGINAL (pre-shuffle) transition indices to force into the "
        "SCORED train set (and keep OUT of test). Use to guarantee key events -- level "
        "submits, wall bumps, landmark contacts -- are trained on. Test is carved from the rest.",
    )
    ap.add_argument(
        "--image-mode",
        action="store_true",
        help="prompts that would show the raw grid show the RENDERED IMAGE instead "
        "(raw-frame baseline + P-writer). P still parses the text grid. (ARC)",
    )
    ap.add_argument(
        "--cell-px", type=int, default=8, help="pixels per grid cell when rendering"
    )
    ap.add_argument(
        "--f-image",
        action="store_true",
        help="F (the inverse + forward scorer) ALSO sees the rendered state image(s) "
        "alongside P's text features. Relaxes the P bottleneck -- experiment only. "
        "With --context-k>0, F sees one image per window state.",
    )
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--start-perception", default="",
                    help="warm-start: path to a perception .py to seed the initial candidate. "
                    "Learning otherwise ALWAYS starts from an empty P; use this ONLY to continue "
                    "from a previously LEARNED artifact, never to inject a hand-written P.")
    ap.add_argument("--start-beliefs", default="",
                    help="warm-start: path to a beliefs/world_knowledge .txt to seed the candidate "
                    "(same rule as --start-perception: learned artifacts only, not hand-written B).")
    ap.add_argument("--belief-update-period", type=int, default=4,
                    help="update the world_knowledge (beliefs) component once every N component "
                    "selections; all other iterations update perception. 1 = plain round-robin-like "
                    "(50/50 is period 2); larger = less belief learning (curbs belief overfitting). "
                    "Default 4 (~1 belief update per 3 perception updates).")
    ap.add_argument(
        "--fd-scorer",
        choices=["none", "textdiff", "judge", "exact"],
        default="exact",
        help="forward-dynamics term: none=pure inverse dynamics (default); "
        "textdiff=deterministic next-state change diff (free); judge=LLM similarity; "
        "exact=1.0 iff predicted next features exactly equal P(X_t+1), else 0.0 (free)",
    )
    ap.add_argument(
        "--fd-weight",
        type=float,
        default=0.5,
        help="composite weight w: score=(1-w)*ID + w*FD (ignored if --fd-scorer none)",
    )
    ap.add_argument(
        "--fd-reflect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also feed forward mispredictions to the proposer's reflective dataset so it "
        "sees mistakes from ALL objectives (inverse + forward). Default ON; "
        "--no-fd-reflect restores inverse-only feedback.",
    )
    ap.add_argument(
        "--analyze-mistakes",
        action="store_true",
        help="run an extra LLM diagnosis on each shown inverse/forward mistake (world "
        "knowledge + reasoning + prediction + ground truth -> root-cause feedback) and "
        "inject it into the proposer's reflective dataset. Adds one LLM call per shown "
        "mistake; pairs with --fd-reflect for forward-mistake analysis.",
    )
    ap.add_argument(
        "--context-k",
        type=int,
        default=3,
        help="temporal window K shown to F: inverse sees P(X_t-K),a_t-K,...,P(X_t),...,"
        "P(X_t+K) with a_t masked; forward sees the history up to P(X_t). 0 = two-state "
        "(validated path). Lets beliefs learn temporally-extended dynamics. Composes with "
        "--image-mode/--f-image: every window state is also rendered as an image.",
    )
    ap.add_argument(
        "--analyze-mode",
        choices=["combined", "per-mistake"],
        default="combined",
        help="how --analyze-mistakes diagnoses failures. combined (default): ONE LLM call per "
        "component diagnosing all shown mistakes together -- cheaper (shared prefix sent once) "
        "and synthesises cross-mistake root causes, with structured per-mistake output. "
        "per-mistake: one call per mistake (original behaviour).",
    )
    ap.add_argument(
        "--fresh-traces",
        action="store_true",
        help="disable trace reuse: always re-evaluate the selected parent for reflection "
        "(GEPA's default) instead of reusing the trajectory captured when the candidate was "
        "first scored. Reuse (the default here) removes the redundant parent re-eval in the "
        "full-batch + tie-train-val regime at zero extra cost, but reflects on the cached F "
        "sample rather than a fresh draw.",
    )
    ap.add_argument(
        "--reflection-minibatch",
        type=int,
        default=None,
        help="reflection/acceptance minibatch size cap. Default (unset) == full-batch: "
        "the minibatch is the entire train set regardless of its size, so every proposal "
        "sees all train frames and the gate is deterministic. Set a smaller value to "
        "subsample; it is capped at the train size.",
    )
    ap.add_argument(
        "--accept-ties",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="accept proposals that TIE their parent on the subsample acceptance gate "
        "(flip GEPA's strict '<=' reject to '<') so a score-equal-but-different candidate "
        "still reaches full eval + the pareto front, which can decide whether its "
        "per-instance profile is worth keeping. Default ON; --no-accept-ties restores "
        "GEPA's strict-improvement behaviour.",
    )
    ap.add_argument(
        "--compare",
        action="store_true",
        help="also run the legacy greedy loop on the same split",
    )
    ap.add_argument(
        "--legacy-only",
        action="store_true",
        help="skip GEPA entirely; run only the legacy greedy P/B loop on the same split",
    )
    ap.add_argument("--legacy-rounds", type=int, default=4)
    ap.add_argument(
        "--log-reflection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="tee every proposer (reflection-LM) call -- full prompt + response -- to "
        "<gepa_run_dir>/reflection_calls.jsonl (one JSON object per call). Default ON; "
        "--no-log-reflection to disable.",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="where to write best_perception/best_beliefs artifacts, per-item test "
        "traces, and the full GEPA run_dir (all candidates + scores + lineage). "
        "Default: <repo>/logs/perc_invdyn/<timestamp>_seed<seed> (durable, NOT /tmp). "
        "Use a unique dir per game when running many games in parallel.",
    )
    args = ap.parse_args()
    if args.reflection_model is None:  # default: same model for decode and proposal
        args.reflection_model = args.task_model

    rng = random.Random(args.seed)
    task_cfg = make_config(args.task_model, args.client)
    refl_cfg = make_config(args.reflection_model, args.client)
    whitelist = set(filter(None, args.actions.split(","))) or None

    # image mode + context_k>0 now compose: every window state is rendered and shown
    # alongside the K-step transcript (to the P-writer, and to F when --f-image is set).
    context_k = args.context_k

    def _collapse(trs):  # opt-in: collapse parameterized actions (e.g. 'click 3 5' -> 'click')
        if args.collapse_action_params:
            for t in trs:
                t.action = t.action.split()[0]
                t.ctx_prev = [(s, a.split()[0]) for s, a in t.ctx_prev]
                t.ctx_next = [(a.split()[0], s) for a, s in t.ctx_next]
        return trs

    run_dirs = [Path(p) for p in args.run.split(",") if p.strip()]
    transitions = _collapse(load_transitions(run_dirs, whitelist, context_k=context_k))

    if args.test_run:
        # cross-trajectory generalization: TRAIN/VAL drawn ONLY from --run, the held-out TEST
        # drawn ONLY from a distinct --test-run. No pooling, no in-pool test carve; pin_train /
        # swap_click are pooled-mode features and do not apply here.
        test_dirs = [Path(p) for p in args.test_run.split(",") if p.strip()]
        test_transitions = _collapse(
            load_transitions(test_dirs, whitelist, context_k=context_k))
        # the choice pool must span BOTH trajectories: a test item's true action (e.g. a click
        # LOCATION unique to the test trajectory) has to be selectable as a candidate.
        action_pool = sorted(
            {t.action for t in transitions} | {t.action for t in test_transitions})
        train_pool = list(transitions)
        rng.shuffle(train_pool)
        test_pool = list(test_transitions)
        rng.shuffle(test_pool)
        test_n = args.test_n if args.test_n >= 0 else 10**9  # test_n<0 -> use ALL test items
        _, test_tr = balanced_split(test_pool, test_n, 10**9, rng)
        rest, train_core = balanced_split(train_pool, args.train_n, 10**9, rng)
        train_tr = train_core
        if args.tie_train_val:  # low-data regime: train and val are the SAME samples
            val_tr = train_tr
        else:  # distinct val carved from the TRAIN trajectory (never from test)
            _, val_tr = balanced_split(rest, args.val_n, 10**9, rng)
        print(
            f"[cross-traj] train/val <- {args.run} ({len(transitions)} transitions) | "
            f"test <- {args.test_run} ({len(test_transitions)} transitions)")
    else:
        # default: keep full string (e.g. 'click 3 5') -> location is part of the target
        action_pool = sorted({t.action for t in transitions})

        # pin: force specific transitions (by ORIGINAL pre-shuffle index) into the scored train
        # set and OUT of test, so key events (level submits, wall bumps, landmark contacts) are
        # always trained on. The remaining pool is shuffled and split as usual.
        pin_idx = sorted({int(x) for x in args.pin_train_idx.split(",") if x.strip()})
        pinned = [transitions[i] for i in pin_idx if 0 <= i < len(transitions)]
        pinned_ids = {id(t) for t in pinned}
        pool = [t for t in transitions if id(t) not in pinned_ids]

        rng.shuffle(pool)

        # balanced split: test (clean) carved first so it is identical regardless of mode.
        # balanced_split(data, holdout_n, train_n, rng) -> (train_pool, holdout).
        rest, test_tr = balanced_split(pool, args.test_n, 10**9, rng)
        if args.tie_train_val:  # low-data regime: train and val are the SAME samples
            _, train_core = balanced_split(rest, args.train_n, 10**9, rng)
            train_tr = pinned + train_core
            val_tr = train_tr
        else:  # train (reflection) / val (GEPA Pareto selection) distinct
            train_pool, val_tr = balanced_split(rest, args.val_n, 10**9, rng)
            train_tr = pinned + train_pool[: args.train_n]
        if pinned:
            print(
                f"[pin-train] forced {len(pinned)} key transition(s) into train (orig idx "
                f"{pin_idx}); test carved from the remaining {len(pool)}"
            )

        if args.swap_click_train:  # put a click-ID item into the SCORED train/val set
            moved = swap_click_into_train(train_tr, test_tr)
            print(
                f"[swap-click-train] moved click into train: {moved} (an 'up' moved to test)"
            )

    k = args.k_choices
    train = bake_choices(train_tr, action_pool, k, rng)
    val = train if args.tie_train_val else bake_choices(val_tr, action_pool, k, rng)
    test = bake_choices(test_tr, action_pool, k, rng)

    mode = "TIED train==val" if args.tie_train_val else "balanced 3-way"
    print(
        f"transitions: {len(transitions)} | train={len(train)} val={len(val)} test={len(test)} ({mode})"
    )
    if context_k > 0:
        avg_prev = sum(len(t.ctx_prev) for t in transitions) / len(transitions)
        avg_next = sum(len(t.ctx_next) for t in transitions) / len(transitions)
        print(
            f"temporal window K={context_k} | avg ctx_prev={avg_prev:.1f}/{context_k} "
            f"avg ctx_next={avg_next:.1f}/{context_k - 1} (partial at episode edges)"
        )
    print(f"test action balance: {dict(Counter(t['tr'].action for t in test))}")
    print(f"action pool ({len(action_pool)}): {action_pool}")
    print(
        f"task_lm (F) = {args.task_model} | reflection_lm = {args.reflection_model}\n"
    )

    # The learning process ALWAYS starts from an empty perception/beliefs candidate
    # (no hand-coded world knowledge seeded into the loop). The --start-perception /
    # --start-beliefs flags are the only way to seed a non-empty start, and are meant
    # ONLY for continuing from a previously *learned* artifact -- never to inject a
    # hand-written P/B.
    if args.start_perception:
        seed_code = Path(args.start_perception).read_text()
        print(f"[warm-start] seed perception <- {args.start_perception}")
    else:
        seed_code = ""
    seed_beliefs = Path(args.start_beliefs).read_text() if args.start_beliefs else ""
    if args.start_beliefs:
        print(f"[warm-start] seed beliefs <- {args.start_beliefs}")
    seed_candidate = {"perception": seed_code, "world_knowledge": seed_beliefs}

    if args.out_dir:
        outd = Path(args.out_dir)
    else:  # durable default under the repo (never /tmp, which gets wiped)
        repo_root = Path(__file__).resolve().parents[2]
        outd = (
            repo_root
            / "logs"
            / "perc_invdyn"
            / f"{time.strftime('%Y%m%d-%H%M%S')}_seed{args.seed}"
        )
    outd.mkdir(parents=True, exist_ok=True)
    print(f"[out] artifacts + traces + GEPA run_dir -> {outd}")

    # ---- baselines on the CLEAN test split -------------------------------
    chance = 1.0 / k
    raw_acc, _ = asyncio.run(
        eval_on(
            task_cfg,
            "",
            "",
            test,
            raw_mode=True,
            image_mode=args.image_mode,
            cell=args.cell_px,
            log_path=outd / f"test_trace_raw_seed{args.seed}.json",
        )
    )
    start_acc, _ = asyncio.run(
        eval_on(task_cfg, seed_code, "", test, context_k=context_k)
    )
    good_acc = None
    if args.good_baseline:
        good_acc, _ = asyncio.run(
            eval_on(task_cfg, GOOD_P, "", test, context_k=context_k)
        )
    print(
        f"[test baselines] random={chance:.2f} | start-P={start_acc:.2f} | "
        f"raw-frame={raw_acc:.2f}"
        + (f" | single-cell GOOD_P={good_acc:.2f}" if good_acc is not None else "")
        + "\n"
    )

    # ---- GEPA optimization ------------------------------------------------
    gepa_test_acc = gepa_cost = gepa_secs = None
    best_code, best_beliefs = seed_code, ""
    if args.legacy_only:
        print("[gepa] --legacy-only: skipping GEPA, running legacy greedy loop only.\n")
    if not args.legacy_only:
        # shared run context: ProcessLogger writes the live iteration here, the adapter reads
        # it to stamp each mistake-analysis call so the viz can line them up with candidates.
        run_ctx = {"iteration": None}
        adapter = InvDynAdapter(
            task_cfg,
            action_pool,
            concurrency=args.concurrency,
            image_mode=args.image_mode,
            cell=args.cell_px,
            fd_scorer=args.fd_scorer,
            fd_weight=args.fd_weight,
            fd_reflect=args.fd_reflect,
            analyze_mistakes=args.analyze_mistakes,
            analyze_mode=args.analyze_mode,
            pred_log_path=outd / f"gepa_run_seed{args.seed}" / "predictions.jsonl",
            analysis_log_path=(
                outd / f"gepa_run_seed{args.seed}" / "analysis_calls.jsonl"
                if args.analyze_mistakes
                else None
            ),
            run_ctx=run_ctx,
            context_k=context_k,
            reuse_traces=not args.fresh_traces,
            f_image=args.f_image,
        )
        fd_desc = (
            "pure inverse-dynamics"
            if args.fd_scorer == "none"
            else f"composite {1 - args.fd_weight:.2f}*ID + {args.fd_weight:.2f}*FD[{args.fd_scorer}]"
            + (" +reflect" if args.fd_reflect else "")
        )
        fd_desc += f" | context_k={context_k}" + (
            " (windowed)" if context_k > 0 else ""
        )
        if args.accept_ties:
            install_accept_ties_patch()
        print(
            f"[gepa] optimizing (max_metric_calls={args.max_metric_calls}, pareto frontier, "
            f"image_mode={args.image_mode}, objective={fd_desc}"
            f"{' | accept-ties' if args.accept_ties else ' | strict-gate'})..."
        )
        gepa_t0 = time.perf_counter()
        gepa_run_dir = outd / f"gepa_run_seed{args.seed}"
        refl_log = (
            str(gepa_run_dir / "reflection_calls.jsonl")
            if args.log_reflection
            else None
        )
        result = gepa.optimize(
            seed_candidate=seed_candidate,
            trainset=train,
            valset=val,
            adapter=adapter,
            reflection_lm=make_reflection_lm(refl_cfg, log_path=refl_log),
            reflection_prompt_template=REFLECTION_TEMPLATES,
            candidate_selection_strategy="pareto",
            # bias component updates toward perception; beliefs only every Nth selection
            # (plain round_robin is 50/50 and over-trains beliefs into memorization).
            module_selector=PerceptionBiasedComponentSelector(args.belief_update_period),
            reflection_minibatch_size=(
                len(train) if args.reflection_minibatch is None
                else min(len(train), args.reflection_minibatch)
            ),
            max_metric_calls=args.max_metric_calls,
            display_progress_bar=False,
            seed=args.seed,
            cache_evaluation=True,  # candidate scores are reused -> less cost/noise
            raise_on_exception=False,
            # persist the FULL run: every candidate, its val subscores, the selection /
            # proposal lineage and per-iter outputs -> durable, no stdout scraping, resumable.
            run_dir=str(gepa_run_dir),
            track_best_outputs=True,
            # persist the full search process -- incl. rejected proposals + the mistake
            # feedback that produced each candidate -> process_log.jsonl (for the tree viz).
            callbacks=[
                ProcessLogger(gepa_run_dir / "process_log.jsonl", run_ctx=run_ctx)
            ],
        )

        gepa_secs = time.perf_counter() - gepa_t0
        best = dict(result.best_candidate)
        best_code = _clean_component("perception", best.get("perception", ""))
        best_beliefs = best.get("world_knowledge", "")
        gepa_test_acc, _ = asyncio.run(
            eval_on(
                task_cfg,
                best_code,
                best_beliefs,
                test,
                log_path=outd / f"test_trace_gepa_seed{args.seed}.json",
                context_k=context_k,
            )
        )
        gepa_cost = adapter.total_cost + _REFLECTION["cost"]

        try:
            best_val = result.val_aggregate_subscores[result.best_idx]
        except Exception:  # noqa: BLE001
            best_val = "?"
        print(
            f"\n[gepa] best val score (selection set) = {best_val} | "
            f"candidates explored = {result.num_candidates} | total metric calls = {result.total_metric_calls}"
        )
        print(f"[gepa] CLEAN test acc (inverse) = {gepa_test_acc:.2f}")
        if args.fd_scorer != "none":
            fd_test, fd_cost = asyncio.run(
                eval_fd_on(
                    task_cfg,
                    best_code,
                    best_beliefs,
                    test,
                    args.fd_scorer,
                    args.concurrency,
                    context_k=context_k,
                )
            )
            gepa_cost += fd_cost
            print(f"[gepa] CLEAN test FD[{args.fd_scorer}] = {fd_test:.2f}")
        print(
            f"[gepa] F (task_lm) cost=${adapter.total_cost:.4f} ({adapter.eval_calls} fresh F evals"
            + (
                f", {adapter.reused_evals} reused from trace cache"
                if adapter.reuse_traces
                else ""
            )
            + f") | reflection cost=${_REFLECTION['cost']:.4f} ({_REFLECTION['calls']} calls)"
        )

        tag = f"gepa_seed{args.seed}"
        (outd / f"best_perception_{tag}.py").write_text(best_code)
        (outd / f"best_beliefs_{tag}.txt").write_text(best_beliefs)
        print(
            f"[gepa] best P -> best_perception_{tag}.py | best B -> best_beliefs_{tag}.txt"
        )

    # ---- optional head-to-head vs the legacy greedy loop -----------------
    legacy_test_acc = legacy_cost = legacy_secs = None
    if args.compare or args.legacy_only:
        print(
            f"\n[legacy] running greedy P/B loop ({args.legacy_rounds} rounds) on same train/val..."
        )
        sem = asyncio.Semaphore(args.concurrency)
        lrng = random.Random(args.seed)
        leg_t0 = time.perf_counter()
        lcode, lbeliefs, legacy_cost = asyncio.run(
            run_legacy_loop(
                task_cfg,
                train_tr,
                val_tr,
                action_pool,
                k,
                sem,
                lrng,
                args.legacy_rounds,
                seed_code,
                image_mode=args.image_mode,
                cell=args.cell_px,
            )
        )
        legacy_secs = time.perf_counter() - leg_t0
        legacy_test_acc, _ = asyncio.run(
            eval_on(
                task_cfg,
                lcode,
                lbeliefs,
                test,
                log_path=outd / f"test_trace_legacy_seed{args.seed}.json",
            )
        )
        (outd / f"best_perception_legacy_seed{args.seed}.py").write_text(lcode)
        (outd / f"best_beliefs_legacy_seed{args.seed}.txt").write_text(lbeliefs)
        print(
            f"[legacy] CLEAN test acc = {legacy_test_acc:.2f} | cost=${legacy_cost:.4f} | time={legacy_secs:.0f}s"
        )

    note = ""
    print("\n=== HEAD-TO-HEAD (clean test) ===")
    print(
        f"task_lm(F)={args.task_model}  reflection_lm={args.reflection_model}  seed={args.seed}{note}"
    )
    print(f"{'method':<34} {'test_acc':>8} {'cost($)':>9} {'time(s)':>8}")
    print(f"{'raw-frame ref':<34} {raw_acc:>8.2f} {'-':>9} {'-':>8}")
    if not args.legacy_only:
        print(
            f"{'GEPA (pareto + strong reflection)':<34} {gepa_test_acc:>8.2f} {gepa_cost:>9.4f} {gepa_secs:>8.0f}"
        )
    if args.compare or args.legacy_only:
        print(
            f"{'legacy greedy P/B loop':<34} {legacy_test_acc:>8.2f} {legacy_cost:>9.4f} {legacy_secs:>8.0f}"
        )


if __name__ == "__main__":
    main()
