"""Shared core for the inverse-dynamics perception optimizer (self-contained).

Holds the InvDynAdapter (evaluate + make_reflective_dataset), the REx / component
selectors, the reflection LM + templates + prompt render/extract, ProcessLogger,
the eval_on / eval_fd_on scorers, data baking, and the `rex_search` REx-pure loop.
This module has no external optimizer dependency; it is the single backend for rexpure_optimize.py
(perception/belief WM), worldcoder_optimize.py (program WM) and stepwise_eb_learn's
frontier mode. See rex_search for the search loop (faithful REx, Tang et al. 2024).
"""
import argparse
import asyncio
import base64
import difflib
import hashlib
import json
import math
import os
import random
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Small self-contained data types (EvaluationBatch, Image) so this module has no
# external optimizer dependency. rexpure_optimize.py uses Image for VLM reflection
# side_info in --image-mode; the adapter's image_cls param lets a caller inject an
# alternative Image implementation.
# ---------------------------------------------------------------------------
import base64 as _base64
import os as _os
from dataclasses import dataclass as _dataclass
from typing import Any as _Any


@_dataclass
class EvaluationBatch:
    outputs: list
    scores: list
    trajectories: "list | None" = None
    objective_scores: "list | None" = None


_MEDIA_TYPE_BY_EXT = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
}


def _guess_media_type(path: str) -> str:
    return _MEDIA_TYPE_BY_EXT.get(_os.path.splitext(path)[1].lower(), "image/png")


@_dataclass
class Image:
    """Image side_info for VLM reflection (used in --image-mode)."""
    url: "str | None" = None
    path: "str | None" = None
    base64_data: "str | None" = None
    media_type: "str | None" = None

    def __post_init__(self) -> None:
        sources = sum(x is not None for x in [self.url, self.path, self.base64_data])
        if sources != 1:
            raise ValueError("Exactly one of url, path, or base64_data must be provided.")
        if self.base64_data is not None and self.media_type is None:
            raise ValueError("media_type is required when using base64_data.")

    def to_openai_content_part(self) -> "dict[str, _Any]":
        if self.url is not None:
            return {"type": "image_url", "image_url": {"url": self.url}}
        if self.path is not None:
            mt = self.media_type or _guess_media_type(self.path)
            with open(self.path, "rb") as f:
                data = _base64.b64encode(f.read()).decode("utf-8")
            return {"type": "image_url", "image_url": {"url": f"data:{mt};base64,{data}"}}
        assert self.base64_data is not None and self.media_type is not None
        return {"type": "image_url",
                "image_url": {"url": f"data:{self.media_type};base64,{self.base64_data}"}}


from forward_objective import (  # noqa: E402
    judge_score,
    judge_score_reasoned,
    textdiff_delta_f1,
)
from PIL import Image as PILImage


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
from validate import (  # noqa: E402
    backfill_context_from_source,
    _extract_action_set,
    llm_hedge_stats,
    run_async,
    strip_transitions_obs_metadata,
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


def infer_env_name(transitions) -> "str | None":
    """'arc_agi' | 'autumn' | None, from the observation format of the logged frames."""
    for tr in list(transitions)[:8]:
        raw = tr.x_t
        if "<grid_" in raw:
            return "arc_agi"
        if "[[" in raw:
            return "autumn"
    return None


def infer_background(transitions, n: int = 8):
    """DIAGNOSTIC ONLY -- never fed to a prompt (decision 2026-08-23: the background is
    derivable from the frames, so stating it would be feature engineering). (dominant
    colour, (rows, cols)) over the first n logged autumn frames, or (None, None) when
    nothing parses; use it to audit whether a learned P found the background itself. The
    dominant colour is the empty-cell colour in every catalogue world except those with a
    large static object (rink's 484-cell ice)."""
    counts: Counter = Counter()
    shape = None
    for tr in list(transitions)[:n]:
        g = parse_grid(tr.x_t)
        if not g or not isinstance(g[0], list) or not g[0] or not isinstance(g[0][0], str):
            continue
        shape = (len(g), len(g[0]))
        for row in g:
            counts.update(row)
    if not counts:
        return None, None
    return counts.most_common(1)[0][0], shape


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


def _inverse_transcript(win, reveal_action=None):
    """States X_t-K..X_t+K with the action between each consecutive pair; a_t is MASKED.

    `reveal_action` prints the true a_t in place of the mask. Used by the --no-id
    ablation's reflective dataset, where the proposer is never asked to make the action
    recoverable and the "??? (IDENTIFY THIS)" framing would smuggle the dropped inverse
    objective back into the prompt."""
    lines, n_prev = [], len(win["prev"])
    for k, (z, a) in enumerate(win["prev"]):
        idx = -(n_prev - k)  # -n_prev .. -1
        lines.append(f"STATE[{_tlabel(idx)}] features:\n{z or '(empty)'}")
        lines.append(f"  action({_tlabel(idx)} -> {_tlabel(idx + 1)}): {a}")
    lines.append(f"STATE[t] features:\n{win['z_t'] or '(empty)'}")
    lines.append("  action(t -> t+1): "
                 + (str(reveal_action) if reveal_action is not None
                    else "??? (IDENTIFY THIS)"))
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


# Set-prediction instruction tail shared by both set-mode inverse templates: the
# scoring rule is stated explicitly so the smallest defensible set is the
# incentive-compatible answer (credit / |S|, ~zero on a miss).
_SET_ANSWER_TAIL = """Predict the SET of plausible hidden actions. SCORING: your credit on this case is divided by the number of actions you list, and is nearly ZERO if the true action is not among them. So list the SMALLEST set you can defend: exactly ONE action when the features determine it; if the features genuinely cannot distinguish several candidates, include exactly those indistinguishable candidates and no more.

Respond as:
<reasoning>what the states and actions imply; if several candidates are indistinguishable, say which feature is missing</reasoning>
<actions>
one action per line, each copied verbatim from the list
</actions>"""


INV_WINDOW_SET_TMPL = """You identify a single HIDDEN action in a trajectory of a grid environment.

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

""" + _SET_ANSWER_TAIL


INV_SET_TMPL = """You identify which action was taken between two states of a grid environment.

=== DEFAULT KNOWLEDGE (always-true facts about this environment) ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== WORLD KNOWLEDGE (general facts; may be empty) ===
{beliefs}
=== END WORLD KNOWLEDGE ===

Two consecutive states are summarized below. Exactly one action was taken to get from STATE 1 to STATE 2.

=== STATE 1 features ===
{z_t}
=== STATE 2 features ===
{z_t1}

The action was one of:
{choices}

""" + _SET_ANSWER_TAIL


async def predict_action_set_from_window(cfg, win, beliefs, choices, sem):
    """Set-mode analogue of predict_action_from_window: returns (pred_set, raw
    response, cost, prompt) with pred_set an ordered subset of `choices`."""
    prompt = INV_WINDOW_SET_TMPL.format(
        beliefs=beliefs.strip() or "(empty)",
        default_knowledge=DEFAULT_KNOWLEDGE,
        transcript=_inverse_transcript(win),
        choices="\n".join(f"- {c}" for c in choices),
    )
    async with sem:
        text, cost = await _llm_call(cfg, prompt)
    text = text or ""
    pred_set = _extract_action_set(text, choices)
    return pred_set, text, cost, prompt


async def predict_action_set(cfg, z_t, z_t1, beliefs, choices, sem):
    """Set-mode analogue of the two-state predict_action (raw baseline / K==0):
    returns (pred_set, reasoning, cost)."""
    prompt = INV_SET_TMPL.format(
        beliefs=beliefs.strip() or "(empty)",
        default_knowledge=DEFAULT_KNOWLEDGE,
        z_t=z_t or "(empty)",
        z_t1=z_t1 or "(empty)",
        choices="\n".join(f"- {c}" for c in choices),
    )
    async with sem:
        text, cost = await _llm_call(cfg, prompt)
    text = text or ""
    pred_set = _extract_action_set(text, choices)
    return pred_set, _parse_tag(text, "reasoning"), cost


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


# ---------------------------------------------------------------------------
# CREDITED SCORING (--credited-scoring): a P that discards real content can still
# make FD trivially predictable (a step counter makes "next = step+1" free) or make ID
# partly guessable from the visible action-history window alone (momentum/repetition),
# independent of whether z carries any genuine information. Both are additive-composite
# exploits: score = (1-w)*ID + w*FD lets one term fully compensate for the other's
# collapse. The fix has two parts: (1) subtract a candidate-independent BLIND floor from
# each raw score -- how much is achievable with z's content withheld -- crediting only
# the genuine LIFT z provides; (2) combine the two credited terms with min() (a
# conjunction), not addition, so a candidate must show real lift on BOTH axes to score
# well; maxing one out no longer buys a pass on the other. See id-set-loss-jul21 memory.
#
# The ID blind floor withholds STATE content but keeps the action-history window (the
# empty-string seed candidate already measures this: e.g. bt3gb mean ID 0.414 from
# context alone). The FD blind floor is different in kind: a constant/blank P would
# trivially "predict" itself, so a naive P-level blind reference degenerately maxes FD
# out (~1.0) instead of providing a meaningful floor. The correct analogue instead
# withholds z_t specifically (not the whole candidate) when asking F to predict the
# real z_t1: history t-K..t-1 + the action survive, only the CURRENT state is masked.
# If z_t1 is basically position-determined (recoverable from the window's other states
# and the action alone, e.g. a step counter), a z_t-blind F predicts it just as well as
# a sighted one and credited_fd collapses to ~0 -- correctly flagging no real dependence
# on P's content.
# ---------------------------------------------------------------------------
# Fixed content-free placeholder for the ID blind floor. Deliberately "" (matching the
# codebase's own existing convention for zero perceptual content, e.g. seed_code=""),
# not a custom string -- every prompt template already renders falsy z as "(empty)"
# (see _inverse_transcript, predict_action_set's z_t=z_t or "(empty)", etc.), so this
# gives the blind floor the exact same prompt framing an empty-string P would produce.
# A distinct placeholder wording (e.g. "(no perception)") would risk confounding "zero
# information" with "the LLM reacts differently to a different phrase" -- a real
# discrepancy observed in a live smoke test (blind_id_score=1.0 vs the seed P's own
# id_score=0.214 on the identical transition, both nominally content-free).
_BLIND_Z = ""


def _forward_transcript_blind(win):
    """Same history as _forward_transcript but the CURRENT (z_t) state is withheld --
    only t-K..t-1 + their actions survive. Used to measure how much of FD's score
    depends on z_t specifically versus the history + action alone."""
    lines, n_prev = [], len(win["prev"])
    for k, (z, a) in enumerate(win["prev"]):
        idx = -(n_prev - k)
        lines.append(f"STATE[{_tlabel(idx)}] features:\n{z or '(empty)'}")
        lines.append(f"  action({_tlabel(idx)} -> {_tlabel(idx + 1)}): {a}")
    lines.append("STATE[t] (CURRENT) features:\n(withheld)")
    return "\n".join(lines)


FWD_WINDOW_BLIND_TMPL = """You predict the NEXT-state features of a grid environment from a trajectory history and the action just taken. The CURRENT state itself has been withheld -- use ONLY the preceding history and the action to make your best-guess prediction.

=== DEFAULT KNOWLEDGE (always-true facts about this environment) ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== WORLD KNOWLEDGE (general facts; may be empty) ===
{beliefs}
=== END WORLD KNOWLEDGE ===

Below is a trajectory of consecutive states (summarized as features) ending BEFORE the CURRENT state. The CURRENT state itself is intentionally withheld.

{transcript}

The action now taken from the (withheld) CURRENT state is: {action}

Predict the features of the resulting NEXT state, written in EXACTLY the same format and vocabulary the perception module uses above (same keys, same coordinate/colour conventions), based only on the historical pattern and the action. Do NOT add commentary.

<next_state>predicted next-state features, same format as the history above</next_state>"""


FORWARD_BLIND_TMPL = """You predict the NEXT-state features of a grid environment from an action alone -- the CURRENT state has been withheld and no history is available.

=== DEFAULT KNOWLEDGE (always-true facts about this environment) ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== WORLD KNOWLEDGE (general facts; may be empty) ===
{beliefs}
=== END WORLD KNOWLEDGE ===

The CURRENT state is withheld. The action taken was: {action}

Make your best-guess prediction of the resulting NEXT state's features, in the format and vocabulary a perception module for this environment would use. Do NOT add commentary.

<next_state>predicted next-state features</next_state>"""


async def predict_next_state_from_window_blind(cfg, win, action, beliefs, sem):
    """z_t-blind analogue of predict_next_state_from_window: predicts z_t1 WITHOUT
    seeing z_t (only the t-K..t-1 history + the action), scored against the SAME true
    z_t1 by the caller. See CREDITED SCORING above."""
    prompt = FWD_WINDOW_BLIND_TMPL.format(
        beliefs=beliefs.strip() or "(empty)",
        default_knowledge=DEFAULT_KNOWLEDGE,
        transcript=_forward_transcript_blind(win),
        action=action,
    )
    async with sem:
        text, cost = await _llm_call(cfg, prompt)
    text = text or ""
    z_hat = _parse_tag(text, "next_state") or text.strip()
    return z_hat, cost


async def predict_next_state_blind(cfg, action, beliefs, sem):
    """K==0 analogue: no history window exists at all, so this has only the action +
    beliefs to go on. Included for completeness; with no window there is little for a
    blind predictor to exploit, so credited_fd collapses toward raw fd on this path."""
    prompt = FORWARD_BLIND_TMPL.format(
        beliefs=beliefs.strip() or "(empty)",
        default_knowledge=DEFAULT_KNOWLEDGE,
        action=action,
    )
    async with sem:
        text, cost = await _llm_call(cfg, prompt)
    text = text or ""
    z_hat = _parse_tag(text, "next_state") or text.strip()
    return z_hat, cost


def _blind_window(win):
    """Same shape as a build_window() result but every perceived feature is replaced
    by a fixed, content-free placeholder -- only the action history survives. Used to
    compute the ID blind floor: how much action-recovery accuracy is available from
    action-history alone, independent of what P actually reports."""
    return {
        "prev": [(_BLIND_Z, a) for _, a in win["prev"]],
        "z_t": _BLIND_Z,
        "z_t1": _BLIND_Z,
        "nxt": [(a, _BLIND_Z) for a, _ in win["nxt"]],
    }


# ---------------------------------------------------------------------------
# CONTRASTIVE FD (--contrastive-fd): identify the TRUE next observation among
# baked decoy frames, given the candidate's feature history + the action.
#
# Decoys are BAKED once per transition at split time (see bake_decoys) so every
# candidate answers the identical question. Two TARGET RENDERINGS:
#
# - DEFAULT: candidates are shown as P(option) -- the candidate's own perceive()
#   run on each baked frame. This tests directly whether P preserves the
#   information that distinguishes the true next state from plausible decoys:
#   a P that renders the true option identically to a decoy makes the question
#   unanswerable and eats the miss (surfaced to the proposer as cfd_ambiguous).
#   Only safe now that observations are metadata-stripped (a Step:-style counter
#   in the record would let any P win by stamp continuity) and the baked frames
#   themselves stay raw, so the question is still candidate-independent.
# - --cfd-raw-targets: candidates are shown as the raw frames. Target space is
#   candidate-independent all the way to the prompt, so the chance floor
#   (1/n_options) is a constant for ANY P; but a lossy P can still score well
#   here because the judge may discriminate decoys using raw information P does
#   not encode.
# ---------------------------------------------------------------------------
CONTRASTIVE_FD_TMPL = """You identify which candidate is the TRUE next observation of a grid environment, given a trajectory and the action just taken.

=== DEFAULT KNOWLEDGE (always-true facts about this environment) ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== WORLD KNOWLEDGE (general facts; may be empty) ===
{beliefs}
=== END WORLD KNOWLEDGE ===

Below is a trajectory of consecutive states (summarized as features by a perception module) ending at the CURRENT state, with the action taken between each pair.

{transcript}

The action now taken from the CURRENT state is: {action}

{candidates_intro}

{options}

Respond as:
<reasoning>what the current state and the action's effect imply the next observation must contain</reasoning>
<option>the single number of the true candidate</option>"""


def _extract_option_index(text: str, n: int):
    """Recover the chosen candidate number, robust to format drift: <option> tag,
    then \\boxed{{}}, then the last in-range integer near the end of the text."""
    m = re.search(r"<option>\s*(\d+)\s*</option>", text, re.IGNORECASE)
    if not m:
        m = re.search(r"\\boxed\{\s*(\d+)\s*\}", text)
    if m:
        idx = int(m.group(1))
        return idx if 1 <= idx <= n else None
    tail = [int(s) for s in re.findall(r"\b(\d+)\b", text[-200:]) if 1 <= int(s) <= n]
    return tail[-1] if tail else None


_CFD_INTRO_RAW = (
    "Exactly ONE of the following {n} candidate RAW observations is the true NEXT "
    "observation; the others are drawn from other moments of play. Work out what the "
    "next state must look like and pick the candidate consistent with it."
)
_CFD_INTRO_P = (
    "Below are the outputs of the SAME perception module (the one that produced the "
    "trajectory features above) run on {n} candidate next observations. Exactly ONE "
    "was produced from the true NEXT observation; the others come from other moments "
    "of play. Work out what the next state's features must look like and pick the "
    "candidate consistent with it. If several candidates read identically, the "
    "perception module failed to preserve the distinguishing information; make your "
    "best guess among them."
)


async def predict_true_next_frame(
    cfg, win, z_t, action, beliefs, options, sem, rendered=None
):
    """Contrastive FD call: n-way identification of the true next observation. When
    `rendered` is given (default target mode) the candidates are shown as P(option)
    feature summaries instead of the raw frames; the option ORDER is the baked one
    either way, so the returned index refers to the raw option list. Returns
    (pred_index_1based_or_None, cost, prompt, raw_response)."""
    shown = rendered if rendered is not None else options
    opts_txt = "\n\n".join(
        f"=== CANDIDATE {i + 1} ===\n{(o or '').strip()}" for i, o in enumerate(shown)
    )
    transcript = (
        _forward_transcript(win)
        if win is not None
        else f"STATE[t] (CURRENT) features:\n{z_t or '(empty)'}"
    )
    intro = _CFD_INTRO_P if rendered is not None else _CFD_INTRO_RAW
    prompt = CONTRASTIVE_FD_TMPL.format(
        beliefs=beliefs.strip() or "(empty)",
        default_knowledge=DEFAULT_KNOWLEDGE,
        transcript=transcript,
        action=action,
        candidates_intro=intro.format(n=len(options)),
        options=opts_txt,
    )
    async with sem:
        text, cost = await _llm_call(cfg, prompt)
    text = text or ""
    return _extract_option_index(text, len(options)), cost, prompt, text


def exact_match_f1(z_hat, z_t1):
    """Forward reward: 1.0 iff the predicted next features EXACTLY equal the true P(X_t+1),
    else 0.0. Ends are stripped so trailing formatting whitespace doesn't spuriously fail an
    otherwise-identical prediction; the content itself must match character-for-character."""
    return 1.0 if (z_hat or "").strip() == (z_t1 or "").strip() else 0.0


# ---------------------------------------------------------------------------
# Set-based inverse-dynamics loss (--id-set-loss). F predicts a SET of plausible
# actions instead of one; L = -log(eps/N + (1-eps)*1[a* in S]/|S|) where N is the
# FULL per-game action universe (buttons + every click cell), independent of S.
# The per-transition score is p/p_max in [0,1]: a correct singleton
# scores 1.0 (so `score < 1.0` failure selection is unchanged), a hedged hit
# ~1/|S|, a miss ~0. On aliased/unobservable transitions F can hedge honestly
# instead of playing a 0/1 lottery over indistinguishable candidates.
# ---------------------------------------------------------------------------
def id_set_metrics(truth, pred_set, eps, n_actions):
    """Returns {hit, set_size, p, loss, score}; empty set == explicit miss."""
    m = len(pred_set or [])
    hit = truth in (pred_set or [])
    floor = eps / max(1, n_actions)
    p = floor + ((1.0 - eps) / m if (hit and m > 0) else 0.0)
    p_max = floor + (1.0 - eps)
    return {"hit": hit, "set_size": m, "p": p, "loss": -math.log(p), "score": p / p_max}


_GRID_BOUND_RE = re.compile(r"both in 0\.\.(\d+)")


def compute_action_universe(whitelist, collapse, transitions, override=None):
    """N for the eps/N floor: |non-click whitelisted verbs| + click cells (1 when
    --collapse-action-params folds `click R C` -> `click`, else grid^2 with the grid
    side parsed from the raw observation's action menu `both in 0..K`). Returns
    (N, grid_side_or_None)."""
    if override:
        return override, None
    if not whitelist:
        raise ValueError("--id-set-loss needs --actions or an explicit --id-n-actions")
    n = len(set(whitelist) - {"click"})
    grid = None
    if "click" in whitelist:
        if collapse:
            n += 1
        else:
            for tr in transitions:
                m = _GRID_BOUND_RE.search(tr.x_t)
                if m:
                    grid = int(m.group(1)) + 1
                    break
            if grid is None:
                # Metadata-stripped records carry no action menu -- measure the
                # grid itself (click bounds span the full grid; side = max dim).
                for tr in transitions:
                    s, e = tr.x_t.find("[["), tr.x_t.rfind("]]")
                    if s == -1 or e <= s:
                        continue
                    try:
                        g = json.loads(tr.x_t[s : e + 2])
                        grid = max(len(g), max((len(r) for r in g), default=0))
                        break
                    except Exception:
                        continue
            if grid is None:
                raise ValueError(
                    "could not parse the click grid size ('both in 0..K' menu or "
                    "the grid itself) from any observation; pass --id-n-actions "
                    "explicitly"
                )
            n += grid * grid
    return n, grid


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


def _pred_desc(t) -> str:
    """Inverse prediction rendered for feedback/analysis prompts. Set mode (pred is
    a list): states whether the true action was included and the hedging cost;
    single mode: the predicted action string as before."""
    pred = t.get("pred")
    if isinstance(pred, list):
        m = len(pred)
        if t.get("id_hit"):
            return (
                f"the SET {pred!r} ({m} candidates; the TRUE action WAS included, "
                f"but credit is divided by {m})"
            )
        return f"the SET {pred!r} ({m} candidates; the TRUE action was MISSING)"
    return repr(pred)


ANALYZE_INVERSE_TMPL = """You are diagnosing a single mistake made by an inverse-dynamics predictor F in a grid environment.

F is given the WORLD KNOWLEDGE below plus a perception module's text features for consecutive states, and must name the ONE action taken between two states. On THIS case F was WRONG.

The TRUE action comes from the environment engine and is never wrong. Any diagnosis that implies the TRUE label should have been a different action is invalid. Never propose a rule under which the TRUE action could not have produced the observed transition.

=== DEFAULT KNOWLEDGE (always-true facts F was given) ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== WORLD KNOWLEDGE (general facts F was given; may be empty -- written by a previous model from partial evidence and EXPECTED to contain wrong rules; auditing it is part of your job) ===
{beliefs}
=== END WORLD KNOWLEDGE ===

=== STATES / FEATURE TRAJECTORY (the masked action is the one F had to recover) ===
{transcript}
=== END ===

F predicted action: {pred}
TRUE action:        {truth}
F's stated reasoning: {reasoning}

Diagnose in this order:
1. Quote the specific WORLD KNOWLEDGE rule(s) F relied on (or F's stated assumption, if no rule covers the case).
2. Check each quoted rule against THIS case: is it consistent with the TRUE action producing the observed change? If not, the rule is FALSIFIED -- your feedback must begin "REVISE:" or "DELETE:" naming that rule, with the minimal replacement consistent with this case AND the other labeled steps shown in the trajectory.
3. Only if no existing rule is implicated, begin "ADD:".
Never rescue a rule by inventing unobservable machinery (off-screen objects, moving viewports, coordinate-frame shifts, out-of-grid effects) to explain the evidence away. Give feedback that improves {target} so this whole class of mistake is avoided. Do not restate the prediction. Be concise (<150 words).

<feedback>your diagnosis + concrete fix</feedback>"""


ANALYZE_FORWARD_TMPL = """You are diagnosing a single mistake made by a forward-dynamics predictor in a grid environment.

Given the WORLD KNOWLEDGE and a perception module's features for the current state (with recent history) plus the action taken, the predictor must output the NEXT state's features. On THIS case the prediction was WRONG.

=== DEFAULT KNOWLEDGE (always-true facts about this environment) ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== WORLD KNOWLEDGE (general facts; may be empty -- written by a previous model from partial evidence and EXPECTED to contain wrong rules; auditing it is part of your job) ===
{beliefs}
=== END WORLD KNOWLEDGE ===

=== HISTORY (features, ending at the CURRENT state) ===
{history}
=== END ===

Action taken:           {action}
PREDICTED next features: {pred}
TRUE next features:      {truth}
Field-level diff (predicted vs true): {diff}

The TRUE next features come from the engine-logged next observation and are never wrong. Diagnose in this order: (1) quote the WORLD KNOWLEDGE rule(s) the prediction relied on; (2) check each against THIS case -- if the rule is inconsistent with the TRUE next features, it is FALSIFIED and your feedback must begin "REVISE:" or "DELETE:" naming it, with the minimal replacement; (3) only if no existing rule is implicated, begin "ADD:". Never rescue a rule by inventing unobservable machinery (off-screen objects, moving viewports, coordinate-frame shifts). Give feedback that improves {target} so the next state becomes predictable from the current one plus the action. Be concise (<150 words).

<feedback>your diagnosis + concrete fix</feedback>"""


def _log_analysis(log_path, **rec):
    """Append one --analyze-mistakes diagnosis call (full prompt + raw response) to
    analysis_calls.jsonl. These are the LLM calls that DIAGNOSE F's mistakes into proposer
    feedback -- distinct from the proposer calls in reflection_calls.jsonl. No-op unless a
    path was provided. `iteration` (the search iteration, stamped from the shared run context)
    lets the viz line each diagnosis up with the candidate it fed."""
    if not log_path:
        return
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec.setdefault("ts", time.strftime("%H:%M:%S"))
    with p.open("a") as f:
        f.write(json.dumps(rec) + "\n")


_ANALYSIS_CACHE_STATS = {"calls": 0, "hits": 0}


def analysis_cache_stats() -> dict:
    """Snapshot of diagnosis-call memo counters for end-of-run summaries."""
    return dict(_ANALYSIS_CACHE_STATS)


async def _analysis_llm(cfg, prompt, sem, cache):
    """One diagnosis call, memoized on the exact prompt text.

    REx re-selects the same parent often (14-26% of the diagnosis prompts in the five aug8
    runs were byte-identical repeats), and a re-selection rebuilds the SAME reflective
    dataset from the SAME cached traces -- so the prompt is identical and the call is pure
    repeated work on the critical path. A hit returns (text, 0.0): the cost was already
    charged the first time.

    Trade-off (why it is switchable): re-wording the diagnosis was one source of proposal
    diversity across repeat visits to a parent. With the memo, a repeat visit differs only
    through the proposer's own sampling. cache=None restores the always-fresh behaviour."""
    _ANALYSIS_CACHE_STATS["calls"] += 1
    if cache is not None:
        key = hashlib.md5(prompt.encode()).hexdigest()
        hit = cache.get(key)
        if hit is not None:
            _ANALYSIS_CACHE_STATS["hits"] += 1
            return hit, 0.0
    async with sem:
        text, cost = await _llm_call(cfg, prompt)
    if cache is not None and text:
        cache[key] = text
    return text, cost


async def analyze_inverse(
    cfg, beliefs, comp, t, tr, win, sem, *, log_path=None, iteration=None, ti=None,
    cache=None,
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
        pred=_pred_desc(t),
        truth=tr.action,
        reasoning=(t.get("reasoning") or "")[:500] or "(none given)",
        target=_analysis_target(comp),
    )
    text, cost = await _analysis_llm(cfg, prompt, sem, cache)
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
    cfg, beliefs, comp, t, tr, win, sem, *, log_path=None, iteration=None, ti=None,
    cache=None,
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
    text, cost = await _analysis_llm(cfg, prompt, sem, cache)
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

The TRUE labels below come from the environment engine and are never wrong. Any diagnosis that implies a TRUE label should have been a different action is invalid. Never propose a rule under which a TRUE action could not have produced its observed transition.

=== DEFAULT KNOWLEDGE (always-true facts the predictors were given) ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== WORLD KNOWLEDGE (general facts the predictors were given; may be empty -- written by a previous model from partial evidence and EXPECTED to contain wrong rules; auditing it is part of your job) ===
{beliefs}
=== END WORLD KNOWLEDGE ===

There are {n} mistakes below, each tagged [mK INVERSE] (the predictor had to recover the masked action between two states) or [mK FORWARD] (the predictor had to output the NEXT state's features).

{cases}

Diagnose in this order:
1. For each mistake, quote the specific WORLD KNOWLEDGE rule(s) the predictor relied on (or its stated assumption, if no rule covers the case).
2. Check each quoted rule against the case: is it consistent with the TRUE label producing the observed change? If not, the rule is FALSIFIED -- that mistake's fix must begin "REVISE:" or "DELETE:" naming the rule, with the minimal replacement consistent with the case AND the other labeled steps shown in its trajectory.
3. Only if no existing rule is implicated, begin the fix with "ADD:".
Never rescue a rule by inventing unobservable machinery (off-screen objects, moving viewports, coordinate-frame shifts, out-of-grid effects) to explain the evidence away. First identify any pattern shared across several mistakes, then give one concrete, actionable fix per mistake (do not merely restate the prediction). Feedback should improve {target} so these classes of mistake are avoided.

Respond in EXACTLY this format -- the synthesis first, then ONE tag per mistake listed above, keeping each fix under 80 words:
<common_root_causes>patterns shared across the mistakes (1-4 sentences); write "none" if each is independent</common_root_causes>
{tags}"""


async def analyze_combined(
    cfg, beliefs, comp, cases, sem, *, log_path=None, iteration=None, cache=None
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
                f"predicted: {_pred_desc(t)} | TRUE action: {tr.action}\n"
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
    text, cost = await _analysis_llm(cfg, prompt, sem, cache)
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
# Adapter: the single integration point between scoring and the search loop.
# ---------------------------------------------------------------------------
class InvDynAdapter:
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
        analysis_cfg=None,
        pred_log_path=None,
        analysis_log_path=None,
        run_ctx=None,
        context_k=0,
        reuse_traces=True,
        f_image=False,
        gate_train_x=None,
        id_set_loss=False,
        id_eps=0.1,
        id_n_actions=None,
        credited_scoring=False,
        composite="additive",
        softmin_tau=0.25,
        contrastive_fd=False,
        cfd_raw_targets=False,
        no_id=False,
        reflect_max_failures=8,
        reflect_raw_prefix=1500,
        analysis_memo=True,
        image_cls=None,
    ):
        self.cfg = cfg
        # Mistake-analysis calls belong to the reflection stack (they write the
        # proposer's feedback), so a mixed-model run routes them to the reflection
        # model's config; default (None) keeps the historical task-cfg behaviour.
        self.analysis_cfg = analysis_cfg if analysis_cfg is not None else cfg
        # --composite min: per-transition score = min over the SIGHTED terms (ID,
        # FD[pred], and contrastive FD when enabled) -- a conjunction with no blind
        # floors, so competence is never taxed (contrast credited scoring, whose
        # candidate-dependent FD floor rose with candidate quality).
        # --composite softmin: Boltzmann-weighted softmin over the same terms,
        # sum(t*exp(-t/tau))/sum(exp(-t/tau)). tau->0 recovers the hard min;
        # with near-binary terms a one-term-only candidate earns ~sigmoid(-1/tau)
        # per transition instead of 0, so progress on the non-binding term stays
        # visible to selection (the hard min's "min-veto" hid it) while the
        # single-term-gaming payoff stays capped at that same epsilon.
        self.composite = composite
        self.softmin_tau = softmin_tau
        # --contrastive-fd: one extra F call per transition identifying the TRUE next
        # observation among baked decoys. Default target rendering is P(option) --
        # the candidate's perceive() over each baked frame (tests P's discriminative
        # sufficiency directly); --cfd-raw-targets shows the raw frames instead
        # (candidate-independent all the way to the prompt). See the CONTRASTIVE FD
        # comment above CONTRASTIVE_FD_TMPL.
        self.contrastive_fd = contrastive_fd
        self.cfd_raw_targets = cfd_raw_targets
        self.image_cls = image_cls or Image
        self._popt_cache = {}  # (cand_hash, option) -> P(option) rendering
        if contrastive_fd and f_image:
            raise ValueError("--contrastive-fd does not support --f-image")
        if composite in ("min", "softmin") and credited_scoring:
            raise ValueError(
                f"--composite {composite} and --credited-scoring are mutually exclusive "
                "(credited already combines its own terms with min)"
            )
        if composite == "softmin" and softmin_tau <= 0:
            raise ValueError("--softmin-tau must be > 0")
        # --id-set-loss: F predicts a SET of plausible actions; per-transition
        # id_score = p/p_max from id_set_metrics (correct singleton -> 1.0).
        self.id_set_loss = id_set_loss
        self.id_eps = id_eps
        self.id_n_actions = id_n_actions
        if id_set_loss and f_image:
            raise ValueError("--id-set-loss does not support --f-image")
        # --credited-scoring: score = min(credited_id, credited_fd), each credited_*
        # = raw_* minus a blind (z-content-withheld) floor. See CREDITED SCORING above
        # predict_next_state_from_window_blind. Blind ID floor is candidate-independent
        # (only depends on the transition + choices), so it is cached once and reused
        # for every candidate; blind FD is candidate-dependent (the target z_t1 comes
        # from THIS candidate's own perceive()) and is recomputed per candidate x
        # transition, alongside the real FD call.
        self.credited_scoring = credited_scoring
        if credited_scoring and f_image:
            raise ValueError("--credited-scoring does not support --f-image")
        self._blind_id_cache = {}  # tr_key -> blind ID score (candidate-independent)
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
        # shared, mutable run context (ProcessLogger stamps the live search iteration here so
        # each analysis call can be stamped + lined up with its candidate in the viz).
        self.run_ctx = run_ctx
        self.concurrency = concurrency
        self.image_mode = (
            image_mode  # show rendered IMAGE (not raw text) to the P-writer
        )
        self.cell = cell
        self.fd_scorer = fd_scorer  # "none" | "textdiff" | "judge"
        self.fd_weight = 0.0 if fd_scorer == "none" else fd_weight
        # --no-id (OBJECTIVE ABLATION): drop the inverse-dynamics term from the composite
        # AND from everything the proposer reads. The ID call itself still runs, so
        # id_score stays in predictions.jsonl as a diagnostic and the end-of-run held-out
        # ID protocol (eval_on) is untouched -- what changes is only the training signal.
        # Suppressing the reflection side matters as much as the score: leaving the
        # predicted-vs-true action, F's decoder reasoning and the INVERSE feedback in the
        # prompt would let the proposer keep optimizing ID by hand, which would make this
        # an ablation of the scorer rather than of the objective.
        self.no_id = no_id
        if no_id and not (self.contrastive_fd or self.fd_weight > 0.0):
            raise ValueError(
                "--no-id removes the only term from the composite: enable a forward term "
                "(--contrastive-fd, or --fd-scorer textdiff|judge|exact) alongside it"
            )
        # Heading for the reflective record's evidence block. render_reflection_prompt
        # turns record keys into `## <key>` markdown headings, so this key is literally
        # what names the task the proposer believes it is solving.
        self._evidence_key = "Transition" if no_id else "Inverse Dynamics"
        self.fd_reflect = fd_reflect  # also feed forward failures to the proposer
        self.analyze_mistakes = analyze_mistakes  # LLM diagnosis -> proposer feedback
        # "combined" (default): one diagnosis call per component for ALL its shown mistakes
        # (cheaper + cross-mistake synthesis); "per-mistake": one call per mistake (original).
        self.analyze_mode = analyze_mode
        # Proposer-prompt size. The reflection prompt is the run's biggest single request
        # (~80k chars on the aug8 games) and prefill is a real share of its latency, so both
        # of its bulk terms are tunable: how many failing transitions are shown at all, and
        # how much of each raw observation is pasted as an orientation prefix (the encoding
        # itself is already spelled out by the OBSERVATION SCHEMA block, so the prefix is a
        # hint, not the spec). Defaults reproduce the historical prompt exactly.
        self.reflect_max_failures = max(1, int(reflect_max_failures))
        self.reflect_raw_prefix = max(0, int(reflect_raw_prefix))
        # prompt-keyed memo for the diagnosis calls (see _analysis_llm): a re-selected
        # parent rebuilds a byte-identical prompt, so the second call is repeated work.
        self._analysis_cache: "dict | None" = {} if analysis_memo else None
        self._counter_lock = threading.Lock()
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
        # the always-fresh behaviour.
        self.reuse_traces = reuse_traces
        self._trace_cache = {}  # (cand_hash, tr_key) -> (score, pred, traj)
        self.reused_evals = 0  # per-example results served from cache (no LLM call)
        # Constant-output degeneracy gate: ALL raw train observations. Every evaluate()
        # runs the candidate's perceive() locally over this list (no LLM calls; cached
        # per perception code). If the observations are NOT all identical yet perceive()
        # maps every one to the SAME string, the whole batch score is zeroed: a constant
        # z makes FD[exact] a free 1.0 and leaves ID guessable from the visible action
        # history, so any credit earned by such a P is spurious (the s2kt7
        # "error_in_perception" collapse won the run exactly this way).
        self.gate_train_x = list(gate_train_x or [])
        self._gate_cache = {}  # perception-code hash -> (fired, constant_output)

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
                err_traj = {
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
                }
                if self.id_set_loss:  # keep set fields uniform for downstream .get()s
                    idm = id_set_metrics(tr.action, [], self.id_eps, self.id_n_actions)
                    err_traj.update(
                        id_hit=idm["hit"], id_set_size=idm["set_size"],
                        id_p=idm["p"], id_loss=idm["loss"],
                    )
                if self.credited_scoring:  # keep credited fields uniform too
                    err_traj.update(
                        blind_id_score=0.0, credited_id=0.0,
                        blind_fd_score=0.0, credited_fd=0.0,
                    )
                if self.contrastive_fd:  # keep contrastive fields uniform too
                    err_traj.update(cfd_score=0.0, cfd_pred=None, cfd_ambiguous=False)
                return (0.0, "(perception error)", err_traj)

            # Inverse and forward predictions are INDEPENDENT (forward never uses the inverse
            # result), so issue them concurrently -- they contend for the same semaphore, which
            # keeps the concurrency gate saturated instead of serialising two calls per
            # transition. Each returns a normalised (… , prompt) shape so the non-windowed path
            # carries None prompts.
            async def _inverse():
                if self.id_set_loss:  # set mode: pred is an ordered SUBSET of choices
                    if win is not None:
                        return await predict_action_set_from_window(
                            self.cfg, win, beliefs, choices, sem
                        )
                    p, r, c = await predict_action_set(
                        self.cfg, z_t, z_t1, beliefs, choices, sem
                    )
                    return p, r, c, None
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
                        "",
                        0.0,
                    )  # (z_hat, cost, fwd_raw, fwd_prompt, blind_z_hat, blind_cost): no call
                if win is not None:
                    if self.f_image:
                        zh, c, raw, prompt = await predict_next_state_from_window_img_aug(
                            self.cfg, win, tr.action, beliefs, sem, self.cell
                        )
                        return zh, c, raw, prompt, "", 0.0  # blind not supported w/ f_image
                    zh, c, raw, prompt = await predict_next_state_from_window(
                        self.cfg, win, tr.action, beliefs, sem
                    )
                    blind_zh, blind_c = "", 0.0
                    if self.credited_scoring:
                        blind_zh, blind_c = await predict_next_state_from_window_blind(
                            self.cfg, win, tr.action, beliefs, sem
                        )
                    return zh, c, raw, prompt, blind_zh, blind_c
                if self.f_image:
                    zh, c = await predict_next_state_img_aug(
                        self.cfg, z_t, tr.action, beliefs, sem, grid_pil(tr.x_t, self.cell)
                    )
                    return zh, c, zh, None, "", 0.0
                zh, c = await predict_next_state(self.cfg, z_t, tr.action, beliefs, sem)
                blind_zh, blind_c = "", 0.0
                if self.credited_scoring:
                    blind_zh, blind_c = await predict_next_state_blind(
                        self.cfg, tr.action, beliefs, sem
                    )
                return zh, c, zh, None, blind_zh, blind_c

            async def _blind_id():
                if not self.credited_scoring:
                    return 0.0
                return await self._blind_id_score(tr, choices, win, sem)

            async def _contrastive():
                # (pred_idx, cost, prompt, raw_response, rendered); no call unless
                # enabled AND the instance carries baked options (test-eval
                # instances don't). Default: options shown as P(option) -- rendered
                # with THIS candidate's perceive(), cached per (candidate, frame)
                # since the same pool frames recur across transitions.
                if not self.contrastive_fd or not inst.get("cfd_options"):
                    return None, 0.0, None, "", None
                rendered = None
                if not self.cfd_raw_targets:
                    rendered = []
                    for o in inst["cfd_options"]:
                        key = (cand_hash, o)
                        r = self._popt_cache.get(key)
                        if r is None:
                            z, e = run_perceive(code, o)
                            r = "(perception error)" if e else (z.strip() or "(empty)")
                            self._popt_cache[key] = r
                        rendered.append(r)
                pred_i, c, prompt, raw = await predict_true_next_frame(
                    self.cfg, win, z_t, tr.action, beliefs, inst["cfd_options"], sem,
                    rendered=rendered,
                )
                return pred_i, c, prompt, raw, rendered

            (
                (pred, reasoning, cost, inv_prompt),
                (z_hat, c2, fwd_raw, fwd_prompt, blind_z_hat, c_blind_fd),
                blind_id_score,
                (cfd_pred, c_cfd, cfd_prompt, cfd_raw, cfd_rendered),
            ) = await asyncio.gather(_inverse(), _forward(), _blind_id(), _contrastive())
            self.total_cost += cost + c2 + c_blind_fd + c_cfd
            if self.id_set_loss:
                idm = id_set_metrics(tr.action, pred, self.id_eps, self.id_n_actions)
                id_score = idm["score"]
            else:
                idm = None
                id_score = 1.0 if pred == tr.action else 0.0
            # forward-dynamics scoring (optional): compare generated P(X_t+1) to true z_t1.
            # (The judge scorer is itself an LLM call and depends on z_hat, so it runs after.)
            fd_score, fd_reasoning = 1.0, ""
            blind_fd_score = 0.0
            if self.fd_weight > 0.0:
                if self.fd_scorer == "judge":
                    fd_score, fd_reasoning, c3 = await judge_score_reasoned(
                        self.cfg, z_t, z_hat, z_t1, sem
                    )
                    self.total_cost += c3
                    if self.credited_scoring:
                        blind_fd_score, _, c3b = await judge_score_reasoned(
                            self.cfg, z_t, blind_z_hat, z_t1, sem
                        )
                        self.total_cost += c3b
                elif (
                    self.fd_scorer == "exact"
                ):  # 1.0 iff z_hat == P(X_t+1), else 0.0 (free)
                    fd_score = exact_match_f1(z_hat, z_t1)
                    if self.credited_scoring:
                        blind_fd_score = exact_match_f1(blind_z_hat, z_t1)
                else:  # "textdiff": deterministic, free
                    fd_score = textdiff_delta_f1(z_t, z_hat, z_t1)
                    if self.credited_scoring:
                        blind_fd_score = textdiff_delta_f1(z_t, blind_z_hat, z_t1)
            # contrastive-FD scoring: 1.0 iff the chosen candidate IS the true next
            # frame (content match against the baked RAW options -- the index refers
            # to the baked list regardless of how the options were rendered; decoys
            # are deduped against the truth so the match is unambiguous).
            cfd_score = None
            cfd_ambiguous = False
            cfd_tied_decoy = None
            if self.contrastive_fd and inst.get("cfd_options"):
                cfd_score = 0.0
                opts = inst["cfd_options"]
                truth_key = (tr.x_t1 or "").strip()
                if (
                    cfd_pred is not None
                    and 1 <= cfd_pred <= len(opts)
                    and (opts[cfd_pred - 1] or "").strip() == truth_key
                ):
                    cfd_score = 1.0
                if cfd_rendered is not None:
                    # P rendered the TRUE option identically to a decoy: the
                    # question was unanswerable from the features -- P's fault.
                    # Keep the (first) tied decoy's RAW frame so the proposer can
                    # be shown exactly which two frames P failed to tell apart.
                    ti = next(
                        (i for i, o in enumerate(opts)
                         if (o or "").strip() == truth_key),
                        None,
                    )
                    if ti is not None:
                        tied_j = next(
                            (i for i, r in enumerate(cfd_rendered)
                             if i != ti and r == cfd_rendered[ti]),
                            None,
                        )
                        cfd_ambiguous = tied_j is not None
                        if tied_j is not None:
                            cfd_tied_decoy = opts[tied_j]
            if self.credited_scoring:
                credited_id = id_score - blind_id_score
                credited_fd = fd_score - blind_fd_score
                score = (
                    min(credited_id, credited_fd)
                    if self.fd_weight > 0.0
                    else credited_id
                )
            elif self.composite in ("min", "softmin"):
                credited_id = credited_fd = None
                # --no-id drops the ID term; the ctor guarantees a forward term survives,
                # but an instance with no baked cfd_options (never true for the train
                # split, which bake_decoys covers) would otherwise leave terms empty.
                terms = [] if self.no_id else [id_score]
                if self.fd_weight > 0.0:
                    terms.append(fd_score)
                if cfd_score is not None:
                    terms.append(cfd_score)
                if not terms:
                    raise ValueError(
                        "--no-id: transition has no forward term to score "
                        "(no baked contrastive options and no predictive FD scorer)"
                    )
                if self.composite == "min":
                    score = min(terms)
                else:
                    # Boltzmann softmin: equal terms score themselves (softmin(a,a)=a,
                    # unlike LSE, so a both-perfect transition still scores exactly 1.0
                    # and the `score < 1.0` failure selection is unchanged).
                    ws = [math.exp(-t / self.softmin_tau) for t in terms]
                    score = sum(t * w for t, w in zip(terms, ws)) / sum(ws)
            else:
                credited_id = credited_fd = None
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
            if idm is not None:  # set mode: pred is a list; keep the raw metrics too
                traj.update(
                    id_hit=idm["hit"], id_set_size=idm["set_size"],
                    id_p=idm["p"], id_loss=idm["loss"],
                )
            if self.credited_scoring:
                traj.update(
                    blind_id_score=blind_id_score, credited_id=credited_id,
                    blind_fd_score=blind_fd_score, credited_fd=credited_fd,
                )
            if cfd_score is not None:
                traj.update(
                    cfd_score=cfd_score, cfd_pred=cfd_pred,
                    cfd_prompt=cfd_prompt, cfd_response=cfd_raw,
                    cfd_ambiguous=cfd_ambiguous,
                )
                if cfd_tied_decoy is not None:
                    traj["cfd_tied_decoy"] = cfd_tied_decoy
            return score, pred, traj

        cand_hash = self._content_hash(code, beliefs)

        def _tr_key(inst):
            tr = inst["tr"]
            # transition identity + the baked choices (the inverse prompt depends on
            # them) + the baked contrastive options (the contrastive prompt does too).
            return self._content_hash(
                tr.x_t, tr.x_t1, tr.action, "||".join(inst.get("choices", [])),
                "||".join(inst.get("cfd_options") or []),
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

        results, n_fresh = run_async(run_all())
        # these two run on the CALLER's thread (several at once under --propose-batch),
        # unlike the in-coroutine cost counters, which the single LLM loop serializes.
        with self._counter_lock:
            self.eval_calls += n_fresh
            self.reused_evals += len(batch) - n_fresh
        scores = [r[0] for r in results]
        gate_fired, _ = self._constant_p_gate(code)
        if gate_fired:
            scores = [0.0] * len(scores)
        outputs = [r[1] for r in results]
        trajs = [r[2] for r in results] if capture_traces else None
        # sidecar: persist every candidate x transition prediction (inverse + forward)
        # so the viewer can show WHY each displayed score came out as it did.
        self._log_predictions(
            candidate, [r[2] for r in results], scores, gate_zeroed=gate_fired
        )
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajs)

    @staticmethod
    def _content_hash(*parts: str) -> str:
        h = hashlib.md5()
        for p in parts:
            h.update((p or "").encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()[:16]

    def _constant_p_gate(self, code):
        """Constant-output degeneracy gate over ALL training observations.

        Returns (fired, constant_output). Fires iff the training observations are
        not all identical yet perceive() returns the SAME string for every one of
        them (a collapsed P -- e.g. a blanket `except` returning a constant
        sentinel, which never surfaces as a runtime error). Local perceive()
        execution only, no LLM calls; result cached per perception code."""
        if not self.gate_train_x:
            return False, ""
        key = self._content_hash(code)
        if key in self._gate_cache:
            return self._gate_cache[key]
        fired, const_z = False, ""
        if len(set(self.gate_train_x)) > 1:
            outs = set()
            for x in self.gate_train_x:
                z, _err = run_perceive(code, x)  # a crash yields "" -- still an output
                outs.add(z)
                if len(outs) > 1:
                    break
            if len(outs) == 1:
                fired, const_z = True, next(iter(outs))
        self._gate_cache[key] = (fired, const_z)
        return fired, const_z

    async def _blind_id_score(self, tr, choices, win, sem):
        """Candidate-independent ID floor for one transition: mean recoverability of
        the hidden action with z's content withheld entirely (only the action-history
        window + choices survive). Cached per (transition, choices) so it is computed
        ONCE across the whole run and reused for every candidate -- it does not depend
        on P or B (beliefs is fixed to "" here for the same reason: a stable reference,
        not something the learned components can shift). See CREDITED SCORING."""
        key = self._content_hash(
            tr.x_t, tr.x_t1, tr.action, "||".join(choices), "blind-id"
        )
        if key in self._blind_id_cache:
            return self._blind_id_cache[key]
        if win is not None:
            blind_win = _blind_window(win)
            if self.id_set_loss:
                pred, _, cost, _ = await predict_action_set_from_window(
                    self.cfg, blind_win, "", choices, sem
                )
            else:
                pred, _, cost, _ = await predict_action_from_window(
                    self.cfg, blind_win, "", choices, sem
                )
        else:
            if self.id_set_loss:
                pred, _, cost = await predict_action_set(
                    self.cfg, _BLIND_Z, _BLIND_Z, "", choices, sem
                )
            else:
                pred, _, cost = await predict_action(
                    self.cfg, _BLIND_Z, _BLIND_Z, "", choices, sem
                )
        if self.id_set_loss:
            score = id_set_metrics(tr.action, pred, self.id_eps, self.id_n_actions)["score"]
        else:
            score = 1.0 if pred == tr.action else 0.0
        self.total_cost += cost
        self._blind_id_cache[key] = score
        return score

    def _log_predictions(self, candidate, trajs, scores, gate_zeroed=False):
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
                if gate_zeroed:  # constant-P gate zeroed the composite; components above are the raw values
                    rec["gate_zeroed"] = True
                if self.id_set_loss and "id_hit" in t:
                    rec["id_hit"] = t["id_hit"]
                    rec["id_set_size"] = t["id_set_size"]
                    rec["id_p"] = t["id_p"]
                    rec["id_loss"] = t["id_loss"]
                if self.credited_scoring and "credited_id" in t:
                    rec["blind_id_score"] = t["blind_id_score"]
                    rec["credited_id"] = t["credited_id"]
                    rec["blind_fd_score"] = t["blind_fd_score"]
                    rec["credited_fd"] = t["credited_fd"]
                if self.contrastive_fd and "cfd_score" in t:
                    rec["cfd_score"] = t["cfd_score"]
                    rec["cfd_pred"] = t.get("cfd_pred")
                    rec["cfd_ambiguous"] = t.get("cfd_ambiguous", False)
                    if t.get("cfd_tied_decoy"):
                        rec["cfd_tied_decoy"] = t["cfd_tied_decoy"]
                    if t.get("cfd_prompt"):
                        rec["cfd_prompt"] = t["cfd_prompt"]
                        rec["cfd_response"] = t.get("cfd_response", "")
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
                    inv_tmpl = INV_WINDOW_SET_TMPL if self.id_set_loss else INV_WINDOW_TMPL
                    rec["inv_prompt"] = t.get("inv_prompt") or inv_tmpl.format(
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
                # --no-id: never diagnose an inverse-dynamics mistake -- the diagnosis text
                # is injected straight into the proposer prompt, so it is ID signal.
                if not self.no_id and t.get("id_score", 1.0) < 1.0:
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
                    self.analysis_cfg,
                    beliefs,
                    comp,
                    cases,
                    sem,
                    log_path=self.analysis_log_path,
                    iteration=it,
                    cache=self._analysis_cache,
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
                                self.analysis_cfg,
                                beliefs,
                                comp,
                                t,
                                tr,
                                win,
                                sem,
                                log_path=self.analysis_log_path,
                                iteration=it,
                                ti=ti,
                                cache=self._analysis_cache,
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
        return run_async(runner())

    # ---- backward pass: build the reflective dataset (RC1 signal) ----------
    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        trajs = eval_batch.trajectories or []
        scores = eval_batch.scores
        failures = [(t, s) for t, s in zip(trajs, scores) if s < 1.0]
        if self.id_set_loss:
            # hedged hits score close to 1.0 and would crowd the [:8] window; sort
            # ascending so true misses (lowest composite) are always shown first.
            failures.sort(key=lambda ts: ts[1])
        corrects = [(t, s) for t, s in zip(trajs, scores) if s >= 1.0]
        beliefs = (
            candidate.get("world_knowledge", "") if isinstance(candidate, dict) else ""
        )

        # Optional: diagnose each shown mistake with an LLM into proposer-ready feedback.
        analyses = (
            self._analyze_failures(
                beliefs, failures[: self.reflect_max_failures], components_to_update
            )
            if self.analyze_mistakes
            else {}
        )

        gate_fired, gate_z = self._constant_p_gate(
            candidate.get("perception", "") if isinstance(candidate, dict) else ""
        )

        dataset: dict = {}
        for comp in components_to_update:
            records = []
            if gate_fired:
                # Shown FIRST so the proposer reads the real reason every score below
                # is 0 before any per-transition detail.
                n_obs = len(self.gate_train_x)
                if comp == "perception":
                    # why a constant P earns no credit differs by objective: under the
                    # full composite it makes FD trivially exact AND leaves ID to the
                    # visible action history; under --no-id there is no ID term, and the
                    # honest statement is that every contrastive option renders alike.
                    why = (
                        "A constant output renders every candidate next observation "
                        "identically, so the contrastive check cannot be answered from "
                        "the features at all and none of the credit is real."
                        if self.no_id else
                        "A constant output makes the forward prediction trivially exact "
                        "and leaves the action guessable only from the visible action "
                        "history, so none of the credit is real."
                    )
                    gate_fb = (
                        "Score was zeroed: perception output was constant over the "
                        f"batch. perceive() returned the identical output {gate_z!r} "
                        f"for all {n_obs} training observations even though the "
                        f"observations themselves differ. {why} The composite score "
                        "stays 0 while the output is constant."
                    )
                else:  # world_knowledge
                    gate_fb = (
                        "Score was zeroed: perception output was constant over the "
                        f"batch (perceive() returned {gate_z!r} for all {n_obs} "
                        "training observations, which differ). No world-knowledge "
                        "edit can lift the score while perception is collapsed -- the "
                        "fix belongs in the perception component."
                    )
                records.append(
                    {
                        self._evidence_key: "(all scores zeroed by the constant-output gate)",
                        "Forward Prediction": "(all scores zeroed by the constant-output gate)",
                        "Feedback": gate_fb,
                    }
                )
            for ti, (t, _) in enumerate(failures[: self.reflect_max_failures]):
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
                    # Under --no-id this section is renamed TRANSITION and carries only the
                    # shared evidence (raw states, P's features, the true action): F's
                    # decoder reasoning, its prediction and every INVERSE note below are
                    # inverse-dynamics signal and are withheld.
                    inv = [] if self.no_id else [
                        f"F (which sees ONLY P's text output, not the frames) reasoned when "
                        f"coming up with the predicted action: {_clip_reasoning(t['reasoning'])}"
                    ]
                    if not self.no_id and t.get("id_score", 1.0) < 1.0:
                        if self.id_set_loss and t.get("id_hit"):
                            m = t.get("id_set_size", 0)
                            inv.insert(
                                0,
                                f"=> INVERSE HEDGE: the TRUE action WAS in the predicted "
                                f"set, but the predictor hedged over {m} candidates "
                                f"(credit 1/{m}). The features were not sharp enough to "
                                f"single it out -- surface the discriminating detail so a "
                                f"singleton prediction becomes defensible.",
                            )
                        elif self.id_set_loss:
                            inv.insert(
                                0,
                                "=> INVERSE MISS: the TRUE action was NOT in the predicted "
                                "set -- none of the listed candidates matched. Surface "
                                "whatever distinguishes the two states so the true action "
                                "becomes at least a candidate.",
                            )
                        else:
                            inv.insert(
                                0,
                                "=> INVERSE MISS: the TRUE action was NOT recoverable from the "
                                "features below.",
                            )
                    if t["z_t"] == t["z_t1"]:
                        inv.append(
                            "P produced IDENTICAL output for both states. If the state changed "
                            "between them, the abstraction is not moving when the world moves -- "
                            "surface whatever distinguishes consecutive states."
                            if self.no_id else
                            "P produced IDENTICAL output for both states, so the action could not "
                            "be recovered from the features. If the state changed between them, the "
                            "abstraction is not moving when the world moves -- surface whatever "
                            "distinguishes consecutive states."
                        )
                    # When windowed: show the feature trajectory so the P-writer can see
                    # whether the features carry enough state to make the action recoverable
                    # ACROSS several steps (e.g. a selected/active object persists in time).
                    # Under --no-id a_t is shown rather than masked (nothing is being asked
                    # to identify it).
                    if win is not None:
                        inv.append(
                            "FEATURE TRAJECTORY (your perceive() output over consecutive "
                            "states, with the action between each pair):\n"
                            + _inverse_transcript(win, reveal_action=tr.action)
                            if self.no_id else
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
                                    inverse_dynamics[f"image_STATE[{lbl}]"] = self.image_cls(
                                        base64_data=b[0], media_type=b[1]
                                    )
                        else:
                            b1, b2 = (
                                grid_b64(tr.x_t, self.cell),
                                grid_b64(tr.x_t1, self.cell),
                            )
                            if b1:
                                inverse_dynamics["image_state_1"] = self.image_cls(
                                    base64_data=b1[0], media_type=b1[1]
                                )
                            if b2:
                                inverse_dynamics["image_state_2"] = self.image_cls(
                                    base64_data=b2[0], media_type=b2[1]
                                )
                    else:
                        n_pfx = self.reflect_raw_prefix
                        for lbl, raw in (("1", tr.x_t), ("2", tr.x_t1)):
                            if n_pfx:
                                inverse_dynamics[
                                    f"raw_state_{lbl} (PREFIX ONLY, first {n_pfx} chars; "
                                    "full observation is longer at runtime)"
                                ] = _prefix_hint(raw, n_pfx)
                    inverse_dynamics["perceive(state_1)"] = t["z_t"] or "(empty)"
                    inverse_dynamics["perceive(state_2)"] = t["z_t1"] or "(empty)"
                    if not self.no_id:  # F's prediction IS the inverse-dynamics signal
                        pred_key = (
                            "predicted action set" if self.id_set_loss else "predicted action"
                        )
                        inverse_dynamics[pred_key] = repr(t["pred"])
                    inverse_dynamics["TRUE action"] = repr(tr.action)
                    if inv:
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
                    elif t.get("cfd_score") == 0.0 and not t.get("perc_err"):
                        forward_prediction = (
                            "(contrastive forward check FAILED on this transition: the TRUE "
                            "next observation was not identifiable from the features + action "
                            "among decoy frames -- see the CONTRASTIVE note under Feedback.)"
                        )
                    else:
                        forward_prediction = (
                            "(no forward error shown for this transition.)"
                        )

                    # ---- FEEDBACK section: actionable guidance only ----
                    fb = []
                    if t["perc_err"]:
                        fb.append(
                            f"PERCEPTION CRASHED: {t['perc_err']} -> fix so perceive() never raises."
                        )
                    if not self.no_id and t.get("id_score", 1.0) < 1.0:
                        if self.id_set_loss and t.get("id_hit"):
                            fb.append(
                                f"INVERSE: the predictor could only narrow the action down to "
                                f"{t.get('id_set_size', 0)} candidates -- make the features "
                                f"discriminate among {t['pred']!r} so exactly one survives."
                            )
                        else:
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
                    if t.get("cfd_score") == 0.0 and not t.get("perc_err"):
                        if t.get("cfd_ambiguous"):
                            fb.append(
                                "CONTRASTIVE: perceive() rendered the TRUE next observation "
                                "IDENTICALLY to a decoy frame from another moment of play, so "
                                "the true next state was unidentifiable from the features no "
                                "matter how good the dynamics reasoning. The features destroy "
                                "the information that distinguishes these two frames -- the "
                                "decoy's COMPLETE raw observation is shown in this example's "
                                "'Contrastive decoy' section: compare it against raw_state_2, "
                                "find what differs, and surface that in the features."
                            )
                        else:
                            fb.append(
                                "CONTRASTIVE: given the feature history and the action, the TRUE next "
                                "observation could not be picked out among decoy candidates from other "
                                "moments of play. The features are not grounded enough in the actual "
                                "grid content (which objects/colors are where) to pin down the real "
                                "next state -- preserve the state details that distinguish this moment "
                                "from others, do not abstract them away."
                            )
                    if not fb:
                        fb.append(
                            "This transition scored imperfectly; make the features cleaner so the "
                            "next state is predictable and distinguishable from near-miss frames."
                            if self.no_id else
                            "This transition scored imperfectly; make the features cleaner so the "
                            "action is recoverable and the next state is predictable."
                        )

                    record = {
                        self._evidence_key: inverse_dynamics,
                        "Forward Prediction": forward_prediction,
                    }
                    if t.get("cfd_ambiguous") and t.get("cfd_tied_decoy"):
                        # The COMPLETE raw frame P confused with the true next state
                        # (raw_state_2 above is the truth) -- proposer-prompt-only
                        # evidence; F and the scorer never see this.
                        record[
                            "Contrastive decoy (COMPLETE raw observation; perceive() "
                            "rendered it IDENTICALLY to the TRUE next state)"
                        ] = t["cfd_tied_decoy"]
                    record["Feedback"] = "\n".join(fb)
                    records.append(record)
                else:  # world_knowledge: distill the general convention from labels
                    # Each example carries two clearly-separated sections: an INVERSE
                    # DYNAMICS section (recover the masked action) and a FORWARD
                    # PREDICTION section (predict the next state; show the diff).
                    win = (
                        t["win"]
                        if (t.get("win") and (t["win"]["prev"] or t["win"]["nxt"]))
                        else None
                    )

                    # ---- INVERSE DYNAMICS section ---- (TRANSITION under --no-id, where
                    # a_t is revealed rather than masked and no ID outcome is shown)
                    if win is not None:
                        # The window already contains STATE[t]/STATE[t+1], so no separate pair.
                        inverse_section = _inverse_transcript(
                            win, reveal_action=tr.action if self.no_id else None)
                    else:
                        inverse_section = (
                            f"features_state_1 (CURRENT):\n{t['z_t'] or '(empty)'}\n\n"
                            f"features_state_2 (NEXT):\n{t['z_t1'] or '(empty)'}"
                        )
                    inv_fb = []
                    # INVERSE signal: only when the action was actually mis-identified --
                    # otherwise the "chose X but TRUE was X" line is noise (the example may
                    # be here purely on a FORWARD miss). Withheld entirely under --no-id.
                    if not self.no_id and t.get("id_score", 1.0) < 1.0:
                        if self.id_set_loss and t.get("id_hit"):
                            inv_line = (
                                f"INVERSE: the predictor hedged over {t['pred']!r}; the TRUE "
                                f"action {tr.action!r} was among them but credit is divided by "
                                f"the set size. Add or refine the world-knowledge rule that "
                                f"disambiguates these candidates in this situation."
                            )
                        elif self.id_set_loss:
                            inv_line = (
                                f"INVERSE: the predictor's candidate set {t['pred']!r} did NOT "
                                f"contain the TRUE action {tr.action!r}."
                            )
                        else:
                            inv_line = (
                                f"INVERSE: the predictor chose {t['pred']!r} but the TRUE action "
                                f"was {tr.action!r}."
                            )
                        inv_fb += [
                            inv_line,
                            f"F (which sees ONLY P's text features and this world knowledge, "
                            f"not the frames) reasoned: \n\n{_clip_reasoning(t['reasoning'])}",
                        ]
                    if (comp, ti, "inv") in analyses:
                        inv_fb.append(
                            "ANALYSIS (inverse-dynamics mistake):\n"
                            + analyses[(comp, ti, "inv")]
                        )

                    if self.no_id:
                        inverse_dynamics = {
                            "action taken (a_t)": repr(tr.action),
                            "transition (a_t shown in place)": inverse_section,
                        }
                    else:
                        inverse_dynamics = {
                            (
                                "predicted action set" if self.id_set_loss else "predicted action"
                            ): repr(t["pred"]),
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
                                    inverse_dynamics[f"image_STATE[{lbl}]"] = self.image_cls(
                                        base64_data=b[0], media_type=b[1]
                                    )
                        else:
                            b1, b2 = (
                                grid_b64(tr.x_t, self.cell),
                                grid_b64(tr.x_t1, self.cell),
                            )
                            if b1:
                                inverse_dynamics["image_state_1"] = self.image_cls(
                                    base64_data=b1[0], media_type=b1[1]
                                )
                            if b2:
                                inverse_dynamics["image_state_2"] = self.image_cls(
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
                    elif t.get("cfd_score") == 0.0 and not t.get("perc_err"):
                        forward_prediction = (
                            "(contrastive forward check FAILED on this transition: the TRUE "
                            "next observation was not identifiable from the features + action "
                            "among decoy frames -- see the CONTRASTIVE note under Feedback.)"
                        )
                    else:
                        forward_prediction = (
                            "(no forward error shown for this transition.)"
                        )
                    if t.get("cfd_score") == 0.0 and not t.get("perc_err"):
                        if t.get("cfd_ambiguous"):
                            fwd_fb.append(
                                "CONTRASTIVE: the perception module rendered the TRUE next "
                                "observation identically to a decoy, so no world knowledge "
                                "could have identified it -- this miss is a perception "
                                "limitation, not a dynamics-rule error."
                            )
                        else:
                            fwd_fb.append(
                                "CONTRASTIVE: given the feature history, the world knowledge and "
                                "the action, the TRUE next observation could not be picked out "
                                "among decoy candidates from other moments of play. The dynamics "
                                "rules do not pin down what actually changes -- revise the world "
                                "knowledge so the next state is predictable from the current state "
                                "plus the action."
                            )

                    fb = inv_fb + fwd_fb
                    if (
                        not fb
                    ):  # selected as imperfect but neither signal is displayable
                        fb.append(
                            "This transition scored imperfectly; refine the world knowledge "
                            "so its dynamics rules pin down what each action changes."
                            if self.no_id else
                            "This transition scored imperfectly; refine the world knowledge "
                            "so feature changes map cleanly to action names and dynamics."
                        )
                    records.append(
                        {
                            self._evidence_key: inverse_dynamics,
                            "Forward Prediction": forward_prediction,
                            "Feedback": "\n".join(fb),
                        }
                    )
            # a couple of CORRECT cases for contrast (keep what already works)
            for t, _ in corrects[:2]:
                if self.no_id:
                    # A "correct" case here scored 1.0 on the forward term(s) only, so the
                    # contrast to draw is about the features that made the true next state
                    # identifiable -- not about an action the scorer never looked at.
                    correct_id = {
                        "perceive(state_1)": t["z_t"],
                        "perceive(state_2)": t["z_t1"],
                        "action taken (a_t)": repr(t["tr"].action),
                    }
                    correct_fb = (
                        f"Scored perfectly on this transition after {t['tr'].action!r} -- the "
                        f"features carried enough of the state to pin down the true next "
                        f"observation against near-miss alternatives. Note WHAT made that "
                        f"work and keep it intact when revising the rest."
                    )
                    records.append(
                        {
                            self._evidence_key: correct_id,
                            "Forward Prediction": "(correct case, shown for contrast)",
                            "Feedback": correct_fb,
                        }
                    )
                    continue
                if self.id_set_loss:
                    correct_id = {
                        "perceive(state_1)": t["z_t"],
                        "perceive(state_2)": t["z_t1"],
                        "predicted action set": repr(t["pred"]),
                        "TRUE action": repr(t["tr"].action),
                    }
                    correct_fb = (
                        f"Correctly identified {t['tr'].action!r} with a SINGLETON set -- "
                        f"note WHAT made this decisive (which rule or feature); keep that "
                        f"intact when revising the rest."
                    )
                else:
                    correct_id = {
                        "perceive(state_1)": t["z_t"],
                        "perceive(state_2)": t["z_t1"],
                        "predicted action": repr(t["tr"].action),
                        "TRUE action": repr(t["tr"].action),
                    }
                    correct_fb = (
                        f"Correctly identified {t['tr'].action!r} -- note WHAT made this "
                        f"decisive (which rule or feature); keep that intact when revising "
                        f"the rest."
                    )
                records.append(
                    {
                        self._evidence_key: correct_id,
                        "Forward Prediction": "(correct case, shown for contrast)",
                        "Feedback": correct_fb,
                    }
                )
            dataset[comp] = records or [
                {
                    self._evidence_key: "(no failures in this minibatch)",
                    "Forward Prediction": "(no failures in this minibatch)",
                    "Feedback": "No failures in this minibatch; keep the component as-is or make it more robust.",
                }
            ]
        return dataset


# ---------------------------------------------------------------------------
# reflection_lm: route the proposer's calls through our OpenRouter plumbing.
# ---------------------------------------------------------------------------
_REFLECTION = {"cost": 0.0, "calls": 0}


def _data_uri_to_pil(uri: str):
    try:
        b64 = uri.split(",", 1)[1] if uri.startswith("data:") else uri
        return PILImage.open(BytesIO(base64.b64decode(b64)))
    except Exception:  # noqa: BLE001
        return None


def _flatten_prompt(prompt):
    """The prompt is a str, or a multimodal list (OpenAI vision parts) when the
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
        # None (the caller calls .strip() on the result).
        attempts = 0
        for _ in range(retries):
            out, cost = run_async(_llm_call(cfg, text, images=images or None))
            _REFLECTION["cost"] += cost
            _REFLECTION["calls"] += 1
            attempts += 1
            if out and out.strip():
                _log_reflection(log_path, text, out, attempts)
                return out
        _log_reflection(
            log_path, text, "", attempts
        )  # also record failed/empty proposals
        return ""  # last resort: empty -> caller treats the proposal as a no-op

    return fn


class ThreadScopedCtx:
    """Mapping shim for the adapter's `run_ctx`. Writes land in THREAD-LOCAL storage, so
    concurrent --propose-batch iterations each stamp their OWN iteration number onto the
    analysis calls they make (a plain shared dict would have the last writer win and
    mislabel every other worker's diagnoses). Readers in a thread that never wrote fall
    back to the shared initial value. Single-threaded behaviour is unchanged."""

    def __init__(self, **initial):
        self._shared = dict(initial)
        self._local = threading.local()

    def _d(self) -> dict:
        d = getattr(self._local, "d", None)
        if d is None:
            d = self._local.d = {}
        return d

    def __setitem__(self, key, value):
        self._d()[key] = value

    def __getitem__(self, key):
        d = self._d()
        return d[key] if key in d else self._shared[key]

    def get(self, key, default=None):
        d = self._d()
        return d.get(key, self._shared.get(key, default))


class ProcessLogger:
    """Callback that persists the FULL search process -- including REJECTED proposals --
    to <run_dir>/process_log.jsonl, one JSON record per iteration. The pool only keeps
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
    cheap, append-only (so it continues across --max-nodes resumes).

    THREAD-SAFE: with --propose-batch>1 several iterations are in flight at once, so the
    in-progress record lives in thread-local state (each worker builds its own) and the
    append at on_iteration_end is taken under a lock. Single-threaded behaviour is
    byte-identical."""

    def __init__(self, path, run_ctx=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._wlock = threading.Lock()
        # shared dict the adapter reads to stamp the current search iteration onto its
        # analysis-call log (so the viz can line each diagnosis up with its candidate).
        self.run_ctx = run_ctx

    @property
    def cur(self):
        return getattr(self._local, "cur", None)

    @cur.setter
    def cur(self, value):
        self._local.cur = value

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

    def detach(self):
        """Hand the in-progress record back to the caller and clear it. Used when the
        record must be COMPLETED on another thread: with --propose-batch>1 a worker builds
        the record but only the main thread knows the admitted node index, and `cur` is
        thread-local so the main thread cannot reach into the worker's slot."""
        rec, self.cur = self.cur, None
        return rec

    def close_accepted(self, rec, event):
        """Finish + write a record detached from a worker thread (see detach)."""
        if rec is None:
            return
        rec["verdict"] = "accepted"
        rec["new_idx"] = event.get("new_candidate_idx")
        rec["new_score"] = event.get("new_score")
        rec["parent_ids"] = list(event.get("parent_ids") or [])
        rec["accepted"] = True
        self._write(rec)

    def _write(self, rec):
        with self._wlock, self.path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def on_iteration_end(self, event):
        if self.cur is None:
            return
        self.cur.setdefault("verdict", "skipped")
        self.cur["accepted"] = bool(event.get("proposal_accepted"))
        self._write(self.cur)
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

_SCHEMA_AUTUMN = """  (B) Autumn string grid -- a JSON 2D array of lowercase colour-NAME strings embedded in the text, e.g. [["black","gold",...],["black","blue",...], ...] in a black-background world, or [["skyblue","skyblue",...],["skyblue","brown",...], ...] in a skyblue one. Cells are names like "black","white","gray","grey","skyblue","lightblue","blue","darkblue","red","green","darkgreen","limegreen","gold","yellow","orange","brown","tan","pink","mediumpurple","purple".
      The BACKGROUND (empty-cell) colour DIFFERS BETWEEN WORLDS (black, white, skyblue, gray, ...). Never hardcode "black": determine the background as the dominant colour of the grid and filter on that.
      IMPORTANT: the whole 2D array is one CONTIGUOUS JSON value (it is NOT split one-row-per-line; do not iterate over text lines looking for a row each). The first row is NOT always black (e.g. it may start with [["gray",...). Parse it robustly and directly, e.g.:
          s = obs.find("[["); e = obs.rfind("]]") + 2
          grid = json.loads(obs[s:e])   # grid[r][c] is the colour-name string at row r, col c
      Then summarise the cells that are NOT the background colour (their (row,col,colour)), so the action between two states is recoverable from the two summaries."""

_SCHEMA_FOOTER = """- Parse defensively: the grid may be 64+ rows; never assume only the first row(s) exist; never assume a fixed length per row; on any parse error return a best-effort summary string (never raise, never return empty)."""


def observation_schema(env_name: str | None = None) -> str:
    """The OBSERVATION SCHEMA scoped to env_name: only the relevant grid format/palette."""
    if env_name == "autumn":
        body, conj = _SCHEMA_AUTUMN, ", encoded as the grid below:"
    elif env_name == "arc_agi":
        body, conj = _SCHEMA_ARC, ", encoded as the grid below:"
    else:  # unknown -> describe BOTH (back-compat)
        body = _SCHEMA_ARC + "\n" + _SCHEMA_AUTUMN
        conj = " and must handle either of the two encodings below (detect which is present):"
    return f"{_SCHEMA_INTRO}{conj}\n{body}\n{_SCHEMA_FOOTER}"


OBSERVATION_SCHEMA = observation_schema(
    None
)  # module default (both formats), back-compat

# Per-component proposal templates: the default framing is "write a new
# instruction"; for the code component we reframe to "rewrite the Python module".
REFLECTION_TEMPLATES = {
    "perception": f"""You are improving a Python perception module for a grid environment.

It must define `perceive(observation_history: list[str]) -> str`; observation_history[-1] is the current raw observation. It must output a concise (<2000 char) text summary of decision-relevant features, never raise, never return empty. Its output over a window of consecutive states (several steps before and after a transition, when context is available) is consumed WITHOUT the raw grid to identify the action taken between the two center states.

{OBSERVATION_SCHEMA}

Current module:
```
<curr_param>
```

Execution feedback (each failure shows the predicted vs TRUE action, the reasoning of F -- which sees ONLY P's text output, not the frames -- when coming up with the predicted action, and whether P emitted IDENTICAL features for both states -- read these carefully; if P's output does not change between two consecutive states, the abstraction is dropping decision-relevant state):
```
<side_info>
```

Rewrite the FULL module so its output moves whenever the world moves and makes the action recoverable from the feature trajectory -- including cases where the evidence lies several steps away from the masked action. Provide the complete module within ``` blocks.""",
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
        "If -- and only if -- an action produces no visible change anywhere in the shown window, it may act on latent state (selection, mode toggles, arming); in that case describe the dynamics of the latent states underlying the environment. Verify against the window first: the predictor sees several steps before and after the masked action, and an action's effect may be delayed by several steps or may change many cells at once -- explain visible changes as visible dynamics rather than attributing them to hidden state.\n\n"
        "The current block was written by a previous model from partial evidence and is EXPECTED to contain wrong rules. Deleting a wrong rule is worth as much as adding a right one.\n\n"
        "Step 1 -- AUDIT (plain text, no code fences): check every rule in the current block against EVERY labeled step in the shown windows (not only the masked centers). List each rule that any shown step contradicts. A wrong prediction alone does not falsify a rule -- only the shown windows can.\n\n"
        "Step 2 -- TABULATE (plain text, no code fences): for each action label appearing anywhere in the windows, note what feature change followed it in every instance. State any exceptionless regularity, and flag any action whose current rule disagrees with these instances. An action's parameters (e.g. click coordinates) need not have any local observable effect at those coordinates.\n\n"
        "Step 3 -- REWRITE the FULL world knowledge block: concise, general, sufficient to map feature changes -- across the whole shown window, not just the center pair -- to action names. Contradicted rules must be corrected or removed -- never preserved behind exceptions, and never rescued by unobservable machinery (off-screen objects, moving viewports, coordinate-frame shifts). Provide the block within ``` fences -- the ONLY fenced block in your reply -- and AFTER the closing fence a <changes> list naming each rule you removed, revised, or added and the shown case motivating it."
    ),
}


# --no-id template surgery. The proposer's OWN INSTRUCTIONS are inverse-dynamics-framed
# ("identify the action taken", "makes the action recoverable", "map feature changes to
# action names"), so suppressing the ID signal from the reflective dataset is not enough:
# the task statement itself has to be restated in terms of the surviving contrastive
# forward objective, or the proposer keeps optimizing for an objective the scorer no
# longer measures. Each swap asserts its source string is present, so an edit to
# REFLECTION_TEMPLATES that moves one of these sentences fails loudly instead of silently
# leaving the ablation with ID framing.
_NO_ID_TEMPLATE_SWAPS = {
    "perception": [
        (
            "Its output over a window of consecutive states (several steps before and "
            "after a transition, when context is available) is consumed WITHOUT the raw "
            "grid to identify the action taken between the two center states.",
            "Its output over a window of consecutive states (several steps before and "
            "after a transition, when context is available) is consumed WITHOUT the raw "
            "grid to identify which of several candidate observations is the TRUE next "
            "state, given the history and the action just taken. The decoys are near "
            "misses drawn from other moments of the same game, so the features must "
            "preserve exactly what distinguishes this next state from a similar one.",
        ),
        (
            "Execution feedback (each failure shows the predicted vs TRUE action, the "
            "reasoning of F -- which sees ONLY P's text output, not the frames -- when "
            "coming up with the predicted action, and whether P emitted IDENTICAL "
            "features for both states -- read these carefully; if P's output does not "
            "change between two consecutive states, the abstraction is dropping "
            "decision-relevant state):",
            "Execution feedback (each failure shows the transition and the action taken, "
            "whether the TRUE next observation could be picked out among near-miss "
            "decoys, whether P rendered the true next state IDENTICALLY to a decoy, and "
            "whether P emitted IDENTICAL features for both states -- read these "
            "carefully; if P's output does not change between two consecutive states, "
            "the abstraction is dropping decision-relevant state):",
        ),
        (
            "Rewrite the FULL module so its output moves whenever the world moves and "
            "makes the action recoverable from the feature trajectory -- including cases "
            "where the evidence lies several steps away from the masked action. Provide "
            "the complete module within ``` blocks.",
            "Rewrite the FULL module so its output moves whenever the world moves and "
            "preserves enough of the grid's actual content that the true next state is "
            "distinguishable from near-miss frames of the same game -- including cases "
            "where the evidence lies several steps away from the transition. Do not "
            "abstract away the state details that separate one moment of play from "
            "another. Provide the complete module within ``` blocks.",
        ),
    ],
    "world_knowledge": [
        (
            "Verify against the window first: the predictor sees several steps before and "
            "after the masked action, and an action's effect may be delayed by several "
            "steps or may change many cells at once",
            "Verify against the window first: the predictor sees several steps before and "
            "after the transition, and an action's effect may be delayed by several "
            "steps or may change many cells at once",
        ),
        (
            "Step 2 -- TABULATE (plain text, no code fences): for each action label "
            "appearing anywhere in the windows, note what feature change followed it in "
            "every instance. State any exceptionless regularity, and flag any action "
            "whose current rule disagrees with these instances.",
            "Step 2 -- TABULATE (plain text, no code fences): for each action label "
            "appearing anywhere in the windows, note exactly what feature change followed "
            "it in every instance, including what stayed the same. State any exceptionless "
            "regularity, and flag any action whose current rule disagrees with these "
            "instances.",
        ),
        (
            "Step 3 -- REWRITE the FULL world knowledge block: concise, general, "
            "sufficient to map feature changes -- across the whole shown window, not just "
            "the center pair -- to action names.",
            "Step 3 -- REWRITE the FULL world knowledge block: concise, general, "
            "sufficient to PREDICT the next features from the current ones plus the "
            "action -- across the whole shown window, not just the center pair -- "
            "precisely enough to tell the true next state apart from a near-miss "
            "alternative.",
        ),
    ],
    # The OBSERVATION SCHEMA is spliced into the perception template by
    # observation_schema(); its closing sentence motivates the summary by inverse
    # dynamics, so it needs the same treatment. Keyed separately because only the autumn
    # variant carries such a sentence, so the swap is presence-conditional.
    "_schema": [
        (
            "so the action between two states is recoverable from the two summaries.",
            "so the true next state is distinguishable from a near-miss frame of the "
            "same game.",
        ),
    ],
}


# Backstop for the swaps above: after rewriting, no proposer template may still ask for
# inverse dynamics. Cheaper to maintain than the per-sentence asserts and catches new ID
# framing added anywhere in the templates or in a future OBSERVATION SCHEMA.
_NO_ID_BANNED = re.compile(
    r"identify the action|action taken between|masked action|recover\w* the action"
    r"|the action .{0,40}\bis recoverable|action is recoverable|recoverable from the "
    r"(?:feature|two)|to action names|predicted vs TRUE action|predicted action",
    re.I,
)


def _apply_no_id_swaps(comp: str, text: str, *, required: bool = True) -> str:
    for src, dst in _NO_ID_TEMPLATE_SWAPS[comp]:
        if src not in text:
            if not required:
                continue
            raise RuntimeError(
                f"--no-id: the {comp} reflection template no longer contains the "
                f"inverse-dynamics sentence this ablation must replace; update "
                f"_NO_ID_TEMPLATE_SWAPS. Missing: {src[:80]!r}..."
            )
        text = text.replace(src, dst)
    return text


def build_reflection_templates(env_name: str | None = None, no_id: bool = False) -> dict:
    """REFLECTION_TEMPLATES with the OBSERVATION SCHEMA scoped to env_name (autumn/arc_agi ->
    only that grid format; None -> both). Pass the running env to avoid leaking the other
    env's format/palette into the proposed perception code.

    no_id=True (the --no-id objective ablation) additionally restates the proposer's task
    in terms of the contrastive forward objective; see _NO_ID_TEMPLATE_SWAPS."""
    out = {
        "perception": REFLECTION_TEMPLATES["perception"].replace(
            OBSERVATION_SCHEMA, observation_schema(env_name)
        ),
        "world_knowledge": REFLECTION_TEMPLATES["world_knowledge"],
    }
    if not no_id:
        return out
    out = {c: _apply_no_id_swaps(c, t) for c, t in out.items()}
    # the schema is spliced into perception; only the autumn variant carries an
    # inverse-dynamics motivation, so this swap is presence-conditional
    out["perception"] = _apply_no_id_swaps("_schema", out["perception"], required=False)
    for comp, text in out.items():
        m = _NO_ID_BANNED.search(text)
        if m:
            raise RuntimeError(
                f"--no-id: inverse-dynamics framing survives in the {comp} reflection "
                f"template ({m.group(0)!r} at char {m.start()}). Add a swap to "
                "_NO_ID_TEMPLATE_SWAPS -- leaving it in makes the proposer optimize an "
                "objective the scorer no longer measures."
            )
    return out


class RExPureCandidateSelector:
    """Faithful REx (Tang et al. 2024, arXiv:2405.17503) for --selector rex_pure:
    theta_i ~ Beta(1 + C*h_i, 1 + C*(1-h_i) + N_i) with the paper's semantics on
    both terms:

      h_i  -- the candidate's full-TRAIN composite. rex_pure aliases the valset
              to the trainset, so state.program_full_scores_val_set is the mean
              score over all train rows; the val split is never evaluated.
      N_i  -- the number of times candidate i has been SELECTED for refinement
              (every expansion counts: progressive widening). The gated selector
              instead counts only gate-FAILED proposals, which (a) has no signal
              once the gate is removed and (b) lets an accept-lucky lineage
              monopolize selection (n2ntd branch B took 16/20 accepts) while a
              parent hit by noise-ties is starved (candidate 8: 3 pulls, 0 left).

    With no acceptance gate, pruning pressure comes only from here: low-h arms are
    rarely drawn, heavily-expanded arms decay, and every candidate (junk included)
    keeps nonzero probability forever."""

    def __init__(self, c: float = 5.0, rng: random.Random | None = None):
        self.c = float(c)
        self.rng = rng or random.Random(0)
        self.expansions: dict[int, int] = {}

    def select_candidate_idx(self, state) -> int:
        n = len(state.program_candidates)
        h = state.program_full_scores_val_set
        assert len(h) == n

        def draw(i):
            hi = min(1.0, max(0.0, h[i]))
            return self.rng.betavariate(
                1.0 + self.c * hi,
                1.0 + self.c * (1.0 - hi) + self.expansions.get(i, 0),
            )

        idx = max(range(n), key=draw)
        self.expansions[idx] = self.expansions.get(idx, 0) + 1
        return idx


class PerceptionBiasedComponentSelector:
    """Component selector that updates `belief_component` only ONCE every
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


class _NullProcessLogger:
    """No-op stand-in for ProcessLogger when rex_search runs without a run_dir."""

    def __getattr__(self, _name):
        return lambda *a, **k: None


class SingleComponentSelector:
    """Component selector for single-component candidates (e.g. worldcoder's
    {"transition_program": ...}): always returns the one component to mutate."""

    def __init__(self, name: str):
        self.name = name

    def __call__(self, state, trajectories, subsample_scores, candidate_idx, candidate):
        return [self.name]


# ---------------------------------------------------------------------------
# Reflection prompt rendering + output extraction -- the one piece of proposer
# logic the proposer needs (render the reflective dataset into the prompt, extract
# the fenced reply). render_reflection_prompt
# fills a component template's <curr_param>/<side_info> from the reflective dataset
# records (collecting any Image side_info into an OpenAI multimodal messages list);
# extract_proposed_text pulls the proposed component out of the LM's fenced reply.
# ---------------------------------------------------------------------------
def _render_value(value, images, level=3):
    if isinstance(value, Image):
        images.append(value)
        return f"[IMAGE-{len(images)} — see visual content]\n\n"
    if isinstance(value, dict):
        s = ""
        for k, v in value.items():
            s += f"{'#' * level} {k}\n"
            s += _render_value(v, images, min(level + 1, 6))
        return s or "\n"
    if isinstance(value, (list, tuple)):
        s = ""
        for i, item in enumerate(value):
            s += f"{'#' * level} Item {i + 1}\n"
            s += _render_value(item, images, min(level + 1, 6))
        return s or "\n"
    return f"{str(value).strip()}\n\n"


def render_reflection_prompt(template, current_instruction, records):
    """Mirror InstructionProposalSignature.prompt_renderer: fill <curr_param> +
    <side_info>, return a plain string, or an OpenAI multimodal messages list when
    any record carries an Image (so the reflection VLM sees it inline)."""
    images = []
    blocks = []
    for i, sample in enumerate(records):
        s = f"# Example {i + 1}\n"
        for key, val in sample.items():
            s += f"## {key}\n"
            s += _render_value(val, images, level=3)
        blocks.append(s)
    formatted = "\n\n".join(blocks)
    if images:
        formatted = (
            f"The evaluation data below includes visual content ({len(images)} image(s)). "
            "Analyze both the text and images when suggesting improvements.\n\n" + formatted
        )
    prompt = template.replace("<curr_param>", current_instruction).replace(
        "<side_info>", formatted
    )
    if images:
        content = [{"type": "text", "text": prompt}]
        content += [img.to_openai_content_part() for img in images]
        return [{"role": "user", "content": content}]
    return prompt


def extract_proposed_text(lm_out):
    """Mirror InstructionProposalSignature.output_extractor: the LAST fenced block
    (falling back to a lone/leading/trailing fence, else the whole reply)."""
    start = lm_out.find("```") + 3
    end = lm_out.rfind("```")
    if start >= end:
        stripped = lm_out.strip()
        if stripped.startswith("```"):
            m = re.match(r"^```\S*\n?", lm_out)
            return lm_out[m.end():].strip() if m else stripped
        if stripped.endswith("```"):
            return stripped[:-3].strip()
        return stripped
    content = lm_out[start:end]
    m = re.match(r"^\S*\n", content)
    if m:
        content = content[m.end():]
    return content.strip()


# ---------------------------------------------------------------------------
# Node-level resume for rex_search. After the seed and every admitted child we
# checkpoint the full search state so an interrupted run can continue from the
# exact node instead of restarting from the seed. Two files under run_dir:
#   resume_batches.jsonl -- append-only, one line per pooled candidate: its P/B,
#     scores, outputs, and per-transition trajectories. The `tr` (Transition) is
#     DROPPED from each traj: the batch is evaluated over `train` in order, so
#     trajectory j belongs to train[j]; on load we reattach train[j].tr by index.
#     This makes the checkpoint independent of the (large) K-step context blobs.
#   resume_state.json -- overwritten atomically: h, parents, counters, the two
#     selectors' state, and the running cost totals. Written AFTER the batch line
#     is appended, so on load we trust its pool size N and take the first N batch
#     lines (a crash between the two leaves an extra/partial batch line we drop).
# A fingerprint of `train` (length + action sequence) guards against resuming
# onto a different split; a mismatch falls back to a fresh run.
# ---------------------------------------------------------------------------
def _train_fingerprint(train):
    h = hashlib.md5()
    h.update(str(len(train)).encode())
    for inst in train:
        # train items are dicts-with-"tr" (invdyn/rexpure) OR bare PreparedTransition
        # objects (worldcoder program arm); both expose `.action`. For a Transition
        # `.action` is a str; for a PreparedTransition it's an (verb,row,col) tuple.
        # Keep the str path byte-identical to the original so existing dict-format
        # resume checkpoints still match.
        tr = inst["tr"] if isinstance(inst, dict) else inst
        act = getattr(tr, "action", None)
        if not isinstance(act, str):
            act = "" if act is None else repr(act)
        h.update((act or "").encode())
        h.update(b"|")
    return h.hexdigest()[:16]


def _serialize_traj(t):
    # Keep only JSON-serializable values ("tr" is dropped and reattached by index on
    # load). invdyn/rexpure trajectory dicts are fully JSON-safe once "tr" is removed,
    # so this is a no-op for them. The worldcoder program arm stashes live objects
    # (ItemResult, PreparedTransition) that are needed in-memory but can't round-trip;
    # drop them from the checkpoint rather than crash the whole search.
    out = {}
    for k, v in t.items():
        if k == "tr":
            continue
        try:
            json.dumps(v)
        except (TypeError, ValueError):
            continue
        out[k] = v
    return out


def _save_resume(run_dir, *, pool, batches, h, parents, it, errors,
                 selector, module_selector, adapter, train_fp, append_last_only):
    """Persist search state. append_last_only=True appends just the newest
    candidate's batch line (seed or freshly admitted child); False rewrites the
    whole batches file (only used if it is missing/short)."""
    if run_dir is None:
        return
    bpath = run_dir / "resume_batches.jsonl"
    if append_last_only and bpath.exists():
        k = len(pool) - 1
        with bpath.open("a") as f:
            f.write(json.dumps({
                "cand_idx": k, "candidate": pool[k], "scores": batches[k].scores,
                "outputs": batches[k].outputs,
                "trajectories": [_serialize_traj(t) for t in (batches[k].trajectories or [])],
            }) + "\n")
    else:
        with bpath.open("w") as f:
            for k in range(len(pool)):
                f.write(json.dumps({
                    "cand_idx": k, "candidate": pool[k], "scores": batches[k].scores,
                    "outputs": batches[k].outputs,
                    "trajectories": [_serialize_traj(t) for t in (batches[k].trajectories or [])],
                }) + "\n")
    state = {
        "train_fingerprint": train_fp,
        "n_nodes": len(pool),
        "h": h, "parents": parents, "it": it, "errors": errors,
        "selector": {"expansions": {str(k): v for k, v in getattr(selector, "expansions", {}).items()},
                     "rng_state": getattr(selector, "rng", None).getstate() if getattr(selector, "rng", None) else None},
        "module_selector": {"_n": getattr(module_selector, "_n", 0)},
        "cost": {"total_cost": getattr(adapter, "total_cost", 0.0),
                 "eval_calls": getattr(adapter, "eval_calls", 0),
                 "reused_evals": getattr(adapter, "reused_evals", 0),
                 "reflection_cost": _REFLECTION.get("cost", 0.0),
                 "reflection_calls": _REFLECTION.get("calls", 0)},
    }
    tmp = run_dir / "resume_state.json.tmp"
    tmp.write_text(json.dumps(state))
    os.replace(tmp, run_dir / "resume_state.json")


def _load_resume(run_dir, train, selector, module_selector, adapter, train_fp):
    """Reload a checkpoint. Returns (pool, batches, h, parents, it, errors) or None
    if there is no usable checkpoint (or it was written for a different split)."""
    if run_dir is None:
        return None
    spath = Path(run_dir) / "resume_state.json"
    bpath = Path(run_dir) / "resume_batches.jsonl"
    if not spath.exists() or not bpath.exists():
        return None
    try:
        state = json.loads(spath.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if state.get("train_fingerprint") != train_fp:
        print("[rex-resume] train fingerprint mismatch -> ignoring checkpoint, fresh start")
        return None
    n = int(state["n_nodes"])
    lines = []
    for ln in bpath.read_text().splitlines():
        if not ln.strip():
            continue
        try:
            lines.append(json.loads(ln))
        except json.JSONDecodeError:
            break  # tolerate a partial trailing line from a mid-append crash
    lines = [d for d in lines if d["cand_idx"] < n][:n]
    if len(lines) < n:
        print(f"[rex-resume] have {len(lines)} batch lines < {n} nodes -> fresh start")
        return None
    lines.sort(key=lambda d: d["cand_idx"])
    pool, batches = [], []
    for d in lines:
        trajs = []
        for j, st in enumerate(d["trajectories"]):
            st = dict(st)
            # reattach by position (batch order == train order); train items are
            # dicts-with-"tr" (invdyn/rexpure) or bare PreparedTransition (worldcoder)
            st["tr"] = train[j]["tr"] if isinstance(train[j], dict) else train[j]
            trajs.append(st)
        pool.append(d["candidate"])
        batches.append(EvaluationBatch(outputs=d["outputs"], scores=d["scores"],
                                       trajectories=trajs or None))
    # restore selector / component-selector / cost counters
    sel = state.get("selector", {})
    if hasattr(selector, "expansions"):
        selector.expansions = {int(k): v for k, v in sel.get("expansions", {}).items()}
    if getattr(selector, "rng", None) is not None and sel.get("rng_state") is not None:
        rs = sel["rng_state"]
        selector.rng.setstate((rs[0], tuple(rs[1]), rs[2]))  # JSON turned the inner tuple into a list
    if hasattr(module_selector, "_n"):
        module_selector._n = state.get("module_selector", {}).get("_n", 0)
    c = state.get("cost", {})
    adapter.total_cost = c.get("total_cost", 0.0)
    adapter.eval_calls = c.get("eval_calls", 0)
    adapter.reused_evals = c.get("reused_evals", 0)
    _REFLECTION["cost"] = c.get("reflection_cost", 0.0)
    _REFLECTION["calls"] = c.get("reflection_calls", 0)
    return pool, batches, list(state["h"]), [list(p) for p in state["parents"]], \
        int(state["it"]), int(state["errors"])


# ---------------------------------------------------------------------------
# The REx-pure search loop -- the standalone optimizer under
# --selector rex_pure, shared by rexpure_optimize.py (perception/belief WM),
# worldcoder_optimize.py (program WM) and stepwise_eb_learn.py (frontier mode).
# Faithful REx (Tang et al. 2024): every child is admitted as a bandit arm; the
# parent is Thompson-sampled from {h_i}; ship = argmax train score. See the module
# docstrings of those callers for the loop's derivation vs a bare REx loop.
# ---------------------------------------------------------------------------
def rex_search(
    *,
    adapter,
    seed_candidate,
    train,
    reflection_lm,
    templates,
    selector,
    module_selector,
    max_nodes,
    run_dir=None,
    perfect_score=1.0,
    log_prefix="rex",
    max_iters=None,
    resume=False,
    propose_batch=1,
):
    """Run REx-pure search over the components of `seed_candidate`. Returns a dict
    with the best-by-train-score candidate, the full pool + per-candidate train
    scores + parents (so a caller can apply its own ship rule), and counters.

    Budget = nodes explored (evaluated candidates): the seed is node 1, then +1 per
    admitted child; the search stops once `max_nodes` candidates have been evaluated.
    skip-perfect re-selections and errored iterations add no node (`len(pool)` IS the
    node count). The parent's reflection reuses its cached EvaluationBatch (no re-eval).

    propose_batch=B>1 runs B iterations (analysis LLM call -> proposer LLM call -> child
    eval) CONCURRENTLY per round instead of one at a time. The node budget, the selector
    RNG stream and the admission order are unchanged; what changes is that the B parents
    of a round are all drawn against the SAME (one-round-stale) pool, i.e. batched
    Thompson sampling -- children admitted this round only inform the next round's draws.
    That is the intended approximation: a rex_pure node is a serial chain of two blocking
    LLM calls plus an eval, so wall-clock is ~(nodes/B) rounds instead of `nodes`. Each
    in-flight eval opens its own `--concurrency` fan-out, so B*concurrency requests can be
    live at once.

    run_dir=None skips all disk persistence (process_log.jsonl / candidates.jsonl) --
    used by internal relearns (e.g. stepwise frontier mode) that don't need the viz."""
    propose_batch = max(1, int(propose_batch))
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        plog = ProcessLogger(run_dir / "process_log.jsonl", run_ctx=getattr(adapter, "run_ctx", None))
    else:
        plog = _NullProcessLogger()

    comp_names = list(seed_candidate.keys())
    ncomp = len(train)  # rex_pure: minibatch == valset == full train
    train_fp = _train_fingerprint(train)

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    loaded = _load_resume(run_dir, train, selector, module_selector, adapter, train_fp) if resume else None
    if loaded is not None:
        pool, batches, h, parents, it, errors = loaded
        print(f"[{log_prefix}] RESUMED from checkpoint at node {len(pool)}/{max_nodes} "
              f"(best h={max(h):.4f}, it={it})")
    else:
        # --- seed: one full-train eval (traces kept so it can later act as a parent) ---
        seed_batch = adapter.evaluate(train, seed_candidate, capture_traces=True)
        pool = [dict(seed_candidate)]
        batches = [seed_batch]
        h = [mean(seed_batch.scores)]
        parents = [[]]
        it = errors = 0
        # budget = nodes explored (evaluated candidates); the seed is node 1 and each
        # admitted child adds one, so len(pool) IS the running node count.
        print(f"[{log_prefix}] seed full-train score h0={h[0]:.4f} | nodes_explored={len(pool)}")
        _save_resume(run_dir, pool=pool, batches=batches, h=h, parents=parents, it=it,
                     errors=errors, selector=selector, module_selector=module_selector,
                     adapter=adapter, train_fp=train_fp, append_last_only=False)

    # skip-perfect / errored iterations add no node, so the node budget alone cannot
    # bound the loop if every reachable arm is already perfect -- cap total iterations too.
    if max_iters is None:
        max_iters = 100 + 40 * max_nodes

    consec_errors = 0

    def _one_iteration(it_no, i, comps, state):
        """(d)-(e) for ONE drawn parent: reflective dataset -> analysis+proposer LLM calls
        -> child eval. Runs on a worker thread when propose_batch>1, so it must touch only
        thread-local logger state and the snapshotted `state` -- never pool/h/batches, and
        never the stateful selectors (both are driven on the main thread)."""
        parent, parent_batch = state.program_candidates[i], batches[i]
        if getattr(adapter, "run_ctx", None) is not None:
            adapter.run_ctx["iteration"] = it_no
        plog.on_iteration_start({"iteration": it_no})
        plog.on_candidate_selected({"candidate_idx": i, "score": state.program_full_scores_val_set[i]})
        plog.on_minibatch_sampled({"minibatch_ids": list(range(ncomp))})

        # A transient LLM / eval error here skips this iteration (skip-and-continue, not
        # fail-the-run) instead of killing the whole run -- the selection already counted.
        try:
            # (d) reflective dataset -> reflection prompt -> LM -> new component text.
            refl = adapter.make_reflective_dataset(parent, parent_batch, comps)
            plog.on_reflective_dataset_built({"components": list(comps), "dataset": refl})
            new_texts = {}
            for name in comps:
                records = refl.get(name)
                if not records:  # skip a component with no reflective data
                    continue
                prompt = render_reflection_prompt(templates[name], parent.get(name, ""), records)
                out = reflection_lm(prompt)
                # strip the ``` fence (the proposer's output_extractor step)
                # BEFORE _clean_component -- otherwise a fenced 'def perceive' is kept
                # verbatim with the fence and execs to a SyntaxError -> empty output.
                new_texts[name] = _clean_component(name, extract_proposed_text(out))
            plog.on_proposal_end({"new_instructions": new_texts})

            child = dict(parent)
            for name, text in new_texts.items():
                child[name] = text

            # (e) score the child on the full train batch (one fresh eval).
            child_batch = adapter.evaluate(train, child, capture_traces=True)
        except Exception as e:  # noqa: BLE001
            plog.on_iteration_end({"proposal_accepted": False})
            return {"status": "error", "it": it_no, "i": i,
                    "msg": f"[{log_prefix}] it={it_no}: iteration error, skipping -> "
                           f"{type(e).__name__}: {str(e)[:200]}"}
        # the accepted-record needs the node index, known only after admission on the main
        # thread -> hand the (thread-local) record over instead of closing it here.
        return {"status": "ok", "it": it_no, "i": i, "child": child,
                "child_batch": child_batch, "comps": comps, "rec": plog.detach()}

    while len(pool) < max_nodes and it < max_iters:
        if consec_errors >= 30:  # provider likely down -> stop burning the budget
            print(f"[{log_prefix}] {consec_errors} consecutive iteration errors -> aborting search")
            break
        # state shim: exactly the attributes the selectors read. Snapshotted (copies) so a
        # concurrent round sees ONE fixed pool -- children admitted this round are only
        # visible to the NEXT round's draws.
        state = SimpleNamespace(
            program_candidates=list(pool),
            program_full_scores_val_set=list(h),
            list_of_named_predictors=comp_names,
        )

        # (a)-(c) parent draw, skip-perfect check and component choice all happen HERE, on
        # the main thread and in draw order: both selectors carry mutable state (REx
        # expansion counts, the belief-cadence counter) whose update order must not depend
        # on which worker finishes first. Draws are sequential so each sees the previous
        # draw's expansion increment (widening still applies WITHIN a round) and the RNG
        # stream is unchanged; only the h/pool they condition on are one round stale.
        n_draw = min(propose_batch, max_nodes - len(pool), max_iters - it)
        picks, skipped = [], []
        for _ in range(max(1, n_draw)):
            it += 1
            i = selector.select_candidate_idx(state)
            pb = batches[i]
            # (b) skip a parent already perfect on every train row (skip_perfect_score).
            # The expansion still counts toward progressive widening but adds no node
            # (only admitted children advance len(pool)); no eval runs.
            if all(s is not None and s >= perfect_score for s in pb.scores):
                plog.on_iteration_start({"iteration": it})
                plog.on_candidate_selected({"candidate_idx": i, "score": h[i]})
                plog.on_iteration_end({"proposal_accepted": False})
                skipped.append(f"[{log_prefix}] it={it}: parent {i} perfect on all train rows -> skip")
                continue
            # (c) which component to mutate.
            picks.append((it, i, module_selector(state, pb.trajectories, pb.scores, i, pool[i])))
        for msg in skipped:
            print(msg)
        if not picks:
            continue

        if len(picks) == 1:
            results = [_one_iteration(*picks[0], state)]
        else:
            with ThreadPoolExecutor(max_workers=len(picks)) as ex:
                results = list(ex.map(lambda p: _one_iteration(p[0], p[1], p[2], state), picks))

        round_errors = 0
        for res in results:  # admit in draw order -> node indices stay deterministic
            if res["status"] != "ok":
                print(res["msg"])
                errors += 1
                round_errors += 1
                continue
            i, ch = res["i"], mean(res["child_batch"].scores)
            pool.append(res["child"])
            batches.append(res["child_batch"])
            h.append(ch)
            parents.append([i])
            new_idx = len(pool) - 1
            _save_resume(run_dir, pool=pool, batches=batches, h=h, parents=parents, it=it,
                         errors=errors, selector=selector, module_selector=module_selector,
                         adapter=adapter, train_fp=train_fp, append_last_only=True)
            plog.close_accepted(
                res.get("rec"),
                {"new_candidate_idx": new_idx, "new_score": ch, "parent_ids": [i]},
            )
            print(
                f"[{log_prefix}] it={res['it']}: parent {i} (h={h[i]:.3f}) -> child {new_idx} "
                f"h={ch:.3f} [{','.join(res['comps'])}] | nodes_explored={len(pool)}"
            )
        consec_errors = 0 if round_errors < len(results) else consec_errors + round_errors

    best_idx = max(range(len(h)), key=lambda j: h[j])
    # persist the full pool for inspection / viz (self-describing).
    if run_dir is not None:
        (run_dir / "candidates.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "idx": j,
                        "parents": parents[j],
                        "train_score": h[j],
                        "expansions": getattr(selector, "expansions", {}).get(j, 0),
                        **{c: pool[j].get(c, "") for c in comp_names},
                    },
                    default=str,
                )
                + "\n"
                for j in range(len(pool))
            )
        )
    return {
        "best_idx": best_idx,
        "best_candidate": pool[best_idx],
        "best_train_score": h[best_idx],
        "num_candidates": len(pool),
        "nodes_explored": len(pool),
        "iterations": it,
        "errors": errors,
        "pool": pool,
        "train_scores": h,
        "parents": parents,
        "batches": batches,
    }


def _clean_component(comp: str, text: str) -> str:
    """The proposer returns the text already stripped of ``` fences for the
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

It must define `perceive(observation_history: list[str]) -> str`. `observation_history[-1]` is the current raw observation TEXT, which your code must parse. Output a concise (<2000 char) text summary of decision-relevant features; never raise, never return empty. Its output over a window of consecutive states (several steps before and after a transition, when context is available) is consumed WITHOUT the raw grid to identify the action taken between the two center states.

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
    id_set_loss=False,
    id_eps=0.1,
    id_n_actions=None,
    credited_scoring=False,
):
    """Evaluate on a list of baked {tr, choices} dicts using FIXED choice sets, so
    every method (learned, legacy, baselines) faces identical choices on the test set.
    raw_mode + image_mode -> the raw-frame reference shows the model the IMAGES.

    id_set_loss: F predicts a SET of plausible actions; each item scores the
    normalized set score from id_set_metrics (correct singleton -> 1.0) and `acc`
    becomes its mean. The same metric is used in training, so eval matches.

    credited_scoring (diagnostic only; does not change `acc`, the headline ID metric
    every comparison in this project is built on): also compute, per item, the
    candidate-independent blind-ID floor (action recoverable from context alone, P's
    content withheld) and report credited_id = item_score - blind. Skipped in
    raw_mode (there is no P to credit). See CREDITED SCORING near predict_next_state_
    from_window_blind for the full rationale.

    If log_path is given, write a self-contained per-item trace (the learned P and B
    plus, for every test item, P's output on both frames, F's reasoning, the chosen
    vs true action and the deterministic raw change). This lets later failure analysis
    read the trace directly -- no log reconstruction and no LLM replay needed."""
    sem = asyncio.Semaphore(concurrency)
    _blind_cache = {}  # (tr identity, choices) -> blind ID score, local to this call

    async def _blind_id_for(tr, choices, win):
        key = (tr.x_t, tr.x_t1, tr.action, tuple(choices))
        if key in _blind_cache:
            return _blind_cache[key]
        if win is not None:
            blind_win = _blind_window(win)
            if id_set_loss:
                pred, _, _c, _ = await predict_action_set_from_window(
                    cfg, blind_win, "", choices, sem
                )
            else:
                pred, _, _c, _ = await predict_action_from_window(
                    cfg, blind_win, "", choices, sem
                )
        else:
            if id_set_loss:
                pred, _, _c = await predict_action_set(
                    cfg, _BLIND_Z, _BLIND_Z, "", choices, sem
                )
            else:
                pred, _, _c = await predict_action(
                    cfg, _BLIND_Z, _BLIND_Z, "", choices, sem
                )
        score = (
            id_set_metrics(tr.action, pred, id_eps, id_n_actions)["score"]
            if id_set_loss
            else (1.0 if pred == tr.action else 0.0)
        )
        _blind_cache[key] = score
        return score

    async def one(idx, inst):
        tr, choices = inst["tr"], inst["choices"]
        win = None
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
            if id_set_loss:
                pred, reasoning, cost, _ = await predict_action_set_from_window(
                    cfg, win, beliefs, choices, sem
                )
            else:
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
            if id_set_loss:
                pred, reasoning, cost = await predict_action_set(
                    cfg, z_t, z_t1, beliefs, choices, sem
                )
            else:
                pred, reasoning, cost = await predict_action(
                    cfg, z_t, z_t1, beliefs, choices, sem
                )
        if id_set_loss:
            idm = id_set_metrics(tr.action, pred, id_eps, id_n_actions)
            item_score = idm["score"]
        else:
            idm = None
            item_score = 1.0 if pred == tr.action else 0.0
        blind_id_score = credited_id = None
        if credited_scoring and not raw_mode:
            blind_id_score = await _blind_id_for(tr, choices, win)
            credited_id = item_score - blind_id_score
        rec = None
        if log_path is not None:
            rec = {
                "idx": idx,
                "truth": tr.action,
                "pred": list(pred) if idm is not None else pred,
                # strict in set mode (hit AND singleton) so downstream viz keeps
                # its accuracy semantics; `hit` carries the set-membership signal.
                "correct": (
                    (idm["hit"] and idm["set_size"] == 1)
                    if idm is not None
                    else pred == tr.action
                ),
                "choices": list(choices),
                "z_t": str(z_t),
                "z_t1": str(z_t1),
                "reasoning": reasoning,
            }
            if idm is not None:
                rec.update(
                    hit=idm["hit"], set_size=idm["set_size"],
                    id_p=idm["p"], id_loss=idm["loss"], id_score=idm["score"],
                )
            if credited_id is not None:
                rec.update(blind_id_score=blind_id_score, credited_id=credited_id)
        return item_score, cost, rec, credited_id

    res = await asyncio.gather(*(one(i, inst) for i, inst in enumerate(baked)))
    acc = sum(r[0] for r in res) / max(1, len(res))
    if log_path is not None:
        payload = {
            "acc": acc,
            "raw_mode": raw_mode,
            "image_mode": image_mode,
            "perception": code,
            "beliefs": beliefs,
            "records": [r[2] for r in res],
        }
        if credited_scoring and not raw_mode:
            credited_vals = [r[3] for r in res if r[3] is not None]
            payload["credited_id_acc"] = sum(credited_vals) / max(1, len(credited_vals))
            payload["blind_id_floor_mean"] = acc - payload["credited_id_acc"]
        if id_set_loss:
            recs = [r[2] for r in res]
            n = max(1, len(recs))
            payload["id_metric"] = "set"
            payload["id_eps"] = id_eps
            payload["id_n_actions"] = id_n_actions
            payload["hit_rate"] = sum(1 for r in recs if r["hit"]) / n
            payload["mean_set_size"] = sum(r["set_size"] for r in recs) / n
            payload["mean_loss"] = sum(r["id_loss"] for r in recs) / n
            payload["strict_singleton_accuracy"] = (
                sum(1 for r in recs if r["correct"]) / n
            )
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2, default=str))
    return acc, sum(r[1] for r in res)


async def eval_fd_on(
    cfg,
    code,
    beliefs,
    baked,
    scorer,
    concurrency=16,
    context_k=0,
    log_path=None,
    credited_scoring=False,
):
    """Mean forward-dynamics score on the clean test set: generate Z_hat = Fwd(history,
    A, B) and score it against the TRUE next features P(X_t+1). Secondary readout (the
    headline is test ID accuracy from eval_on); lets us see forward quality per arm.

    credited_scoring (diagnostic only; does not change `fd_score`): also predicts
    z_t1 with z_t itself withheld (only history + action survive) and reports
    credited_fd = fd_score - blind_fd -- how much of FD's score is genuinely
    attributable to z_t's content vs. recoverable from history/action alone. See
    CREDITED SCORING near predict_next_state_from_window_blind.

    If log_path is given, persist the aggregate and a self-contained record for every test
    transition. This mirrors eval_on's durable ID trace so final FD results never exist only
    in terminal output.
    """
    sem = asyncio.Semaphore(concurrency)

    async def one(idx, inst):
        tr = inst["tr"]
        fwd_prompt = None
        fwd_response = None
        blind_z_hat = None
        if context_k > 0:
            win, _ = build_window(code, tr)
            z_t, z_t1 = win["z_t"], win["z_t1"]
            z_hat, c, fwd_response, fwd_prompt = await predict_next_state_from_window(
                cfg, win, tr.action, beliefs, sem
            )
            if credited_scoring:
                blind_z_hat, c_blind = await predict_next_state_from_window_blind(
                    cfg, win, tr.action, beliefs, sem
                )
                c += c_blind
        else:
            z_t = run_perceive(code, tr.x_t)[0]
            z_t1 = run_perceive(code, tr.x_t1)[0]
            z_hat, c = await predict_next_state(cfg, z_t, tr.action, beliefs, sem)
            # predict_next_state historically returns only its parsed answer. Reconstruct the
            # deterministic prompt and retain the parsed response so the trace is still useful
            # without changing that widely-used helper's return contract.
            fwd_prompt = FORWARD_TMPL.format(
                beliefs=beliefs.strip() or "(empty)",
                default_knowledge=DEFAULT_KNOWLEDGE,
                z_t=z_t or "(empty)",
                action=tr.action,
            )
            fwd_response = z_hat
            if credited_scoring:
                blind_z_hat, c_blind = await predict_next_state_blind(
                    cfg, tr.action, beliefs, sem
                )
                c += c_blind
        if scorer == "judge":
            s, c2 = await judge_score(cfg, z_t, z_hat, z_t1, sem)
            c += c2
        elif scorer == "exact":
            s = exact_match_f1(z_hat, z_t1)
        else:
            s = textdiff_delta_f1(z_t, z_hat, z_t1)
        blind_s = credited_fd = None
        if credited_scoring:
            if scorer == "judge":
                blind_s, c3 = await judge_score(cfg, z_t, blind_z_hat, z_t1, sem)
                c += c3
            elif scorer == "exact":
                blind_s = exact_match_f1(blind_z_hat, z_t1)
            else:
                blind_s = textdiff_delta_f1(z_t, blind_z_hat, z_t1)
            credited_fd = s - blind_s
        rec = None
        if log_path is not None:
            rec = {
                "idx": idx,
                "action": tr.action,
                "z_t": str(z_t),
                "z_t1": str(z_t1),
                "z_hat": str(z_hat),
                "score": s,
                "scorer": scorer,
                "fwd_prompt": fwd_prompt,
                "fwd_response": fwd_response,
            }
            if credited_fd is not None:
                rec.update(blind_z_hat=str(blind_z_hat), blind_fd_score=blind_s, credited_fd=credited_fd)
        return s, c, rec, credited_fd

    res = await asyncio.gather(*(one(i, inst) for i, inst in enumerate(baked)))
    fd_score = sum(r[0] for r in res) / max(1, len(res))
    if log_path is not None:
        payload = {
            "fd_score": fd_score,
            "scorer": scorer,
            "context_k": context_k,
            "perception": code,
            "beliefs": beliefs,
            "records": [r[2] for r in res],
        }
        if credited_scoring:
            credited_vals = [r[3] for r in res if r[3] is not None]
            payload["credited_fd_acc"] = sum(credited_vals) / max(1, len(credited_vals))
            payload["blind_fd_floor_mean"] = fd_score - payload["credited_fd_acc"]
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2, default=str))
    return fd_score, sum(r[1] for r in res)


# Held-out contrastive options are baked with a DIFFERENT rng offset from the train
# ones (rexpure_optimize.build_data uses seed+9173), so the two splits never share a
# draw. Both the in-run --cfd-test path and the standalone scorer go through
# bake_test_decoys, so a run's held-out cFD is reproducible from (seed, split) alone
# and the two paths cannot drift apart.
CFD_TEST_SEED_OFFSET = 9174


def bake_test_decoys(test, transitions, n_decoys, seed, hard=False):
    """bake_decoys over the TEST split under the fixed held-out convention."""
    return bake_decoys(test, transitions, n_decoys,
                       random.Random(seed + CFD_TEST_SEED_OFFSET), hard=hard)


async def eval_cfd_on(
    cfg,
    code,
    beliefs,
    baked,
    concurrency=16,
    context_k=0,
    raw_targets=False,
    log_path=None,
):
    """HELD-OUT contrastive-FD score: on each baked test item, identify the TRUE next
    observation among that item's baked `cfd_options`.

    The training loop scores cFD only on the train split (bake_decoys is called on
    `train`), and the end-of-run summary reports only inverse accuracy, so a run trained
    under `--fd-scorer none` has no held-out forward number at all. That is fine while
    every arm optimises ID -- test ID is then everyone's own metric -- but it leaves the
    --no-id ablation with no held-out metric on the objective it actually trained on.
    This scores exactly the term the training loop uses, on the clean test split.

    Two TARGET RENDERINGS, the same pair the training scorer offers:
      raw_targets=False (default) -- options shown as P(option), so the score measures
        whether THIS P preserves what distinguishes the true next state from a decoy.
        Cross-arm comparison is then slightly apples-to-oranges (each arm answers a
        question posed in its own feature language), which is exactly why the ablation
        table should also carry the raw variant.
      raw_targets=True -- options shown as raw frames, candidate-independent all the way
        to the prompt, so the chance floor 1/n_options is a constant for ANY P and the
        arms answer the identical question. A lossy P can still score here on
        information it does not itself encode; read the two together.

    `baked` items must carry `cfd_options` (bake_decoys). Returns (mean_score, cost).
    """
    missing = [i for i, inst in enumerate(baked) if not inst.get("cfd_options")]
    if missing:
        raise ValueError(
            f"eval_cfd_on: {len(missing)} of {len(baked)} items carry no baked "
            "cfd_options -- run bake_decoys over the split first"
        )
    sem = asyncio.Semaphore(concurrency)
    _popt = {}  # frame -> P(frame), reused across items (pool frames recur as decoys)

    def _render(o):
        r = _popt.get(o)
        if r is None:
            z, e = run_perceive(code, o)
            r = "(perception error)" if e else (z.strip() or "(empty)")
            _popt[o] = r
        return r

    async def one(idx, inst):
        tr, opts = inst["tr"], inst["cfd_options"]
        win = None
        if context_k > 0:
            win, _ = build_window(code, tr)
            z_t = win["z_t"]
        else:
            z_t = run_perceive(code, tr.x_t)[0]
        rendered = None if raw_targets else [_render(o) for o in opts]
        pred_i, cost, prompt, raw = await predict_true_next_frame(
            cfg, win, z_t, tr.action, beliefs, opts, sem, rendered=rendered
        )
        truth_key = (tr.x_t1 or "").strip()
        score = 1.0 if (
            pred_i is not None
            and 1 <= pred_i <= len(opts)
            and (opts[pred_i - 1] or "").strip() == truth_key
        ) else 0.0
        # P collapsed the true option onto a decoy: the question was unanswerable from
        # the features. Only meaningful in the P-rendered mode.
        ambiguous = False
        if rendered is not None:
            ti = next((i for i, o in enumerate(opts)
                       if (o or "").strip() == truth_key), None)
            if ti is not None:
                ambiguous = any(r == rendered[ti] for i, r in enumerate(rendered)
                                if i != ti)
        rec = None
        if log_path is not None:
            rec = {
                "idx": idx, "action": tr.action, "n_options": len(opts),
                "pred_index": pred_i, "score": score, "ambiguous": ambiguous,
                "cfd_prompt": prompt, "cfd_response": raw,
            }
        return score, cost, rec, ambiguous

    res = await asyncio.gather(*(one(i, inst) for i, inst in enumerate(baked)))
    n = max(1, len(res))
    cfd = sum(r[0] for r in res) / n
    if log_path is not None:
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "cfd_score": cfd,
            "n_test": len(res),
            "targets": "raw" if raw_targets else "perceived",
            "chance": 1.0 / max(1, len(baked[0]["cfd_options"])),
            "ambiguous_rate": sum(1 for r in res if r[3]) / n,
            "context_k": context_k,
            "perception": code,
            "beliefs": beliefs,
            "records": [r[2] for r in res],
        }, indent=2, default=str))
    return cfd, sum(r[1] for r in res)


def stratified_split(transitions, n_a, n_b, rng):
    """Split ONE pool into two sets that BOTH see every action: each action's
    (shuffled) rows are dealt alternately to the two sides, cycling actions
    round-robin, until the size targets are met. Carving side A to balance
    first (balanced_split) exhausts rare actions -- with 5 'up' rows in the
    pool, train takes all 5 and val degenerates to the leftovers; the deal
    gives 3/2 instead. Alternation parity varies by action so odd counts don't
    systematically favor one side. Returns (side_a, side_b)."""
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


def bake_choices(transitions, action_pool, k, rng):
    """Fix one choice set per transition so scores are comparable across candidates."""
    return [
        {"tr": tr, "choices": make_choices(tr.action, action_pool, k, rng)}
        for tr in transitions
    ]


def _frame_sim(a, b):
    """Content similarity between two raw frames: cell-wise match fraction when the
    strings are shape-compatible, difflib ratio otherwise. Generic (no task
    knowledge) -- used only to RANK candidate decoys by closeness to the truth."""
    a = (a or "").strip()
    b = (b or "").strip()
    if a and len(a) == len(b):
        return sum(x == y for x, y in zip(a, b)) / len(a)
    return difflib.SequenceMatcher(None, a, b).ratio()


def bake_decoys(insts, frame_pool_transitions, n_decoys, rng, hard=False):
    """Attach a FIXED contrastive-FD option list to each baked instance: the true next
    frame plus up to n_decoys DISTINCT decoy frames drawn from other moments of the
    frame pool, shuffled once. Baked at split time with a seeded rng (the same
    convention as bake_choices) so every candidate answers the identical question and
    cached per-(candidate, transition) results stay valid. Frames identical to the
    true next frame are excluded, so on a no-change transition the correct pick is
    exactly the frame matching the CURRENT state. Returns the count of items whose
    decoy pool came up short (fewer distinct frames than requested).

    hard=True replaces uniform decoy sampling with near-miss decoys: the CURRENT
    frame (rejecting it requires detecting that something changed) plus the pool
    frames most similar to the true next frame (rejecting them requires knowing
    exactly what changed, e.g. drift advanced by exactly one step). Uniform decoys
    saturate on games whose frames are mutually far apart (bt3gb cFD 0.83-0.93)."""
    frames, seen = [], set()
    for tr in frame_pool_transitions:
        for x in (tr.x_t, tr.x_t1):
            key = (x or "").strip()
            if key and key not in seen:
                seen.add(key)
                frames.append(x)
    short = 0
    for inst in insts:
        tr = inst["tr"]
        truth_key = (tr.x_t1 or "").strip()
        pool = [f for f in frames if (f or "").strip() != truth_key]
        if hard:
            cur_key = (tr.x_t or "").strip()
            picks = [tr.x_t] if cur_key and cur_key != truth_key else []
            for f in sorted(pool, key=lambda f: _frame_sim(f, tr.x_t1), reverse=True):
                if len(picks) >= n_decoys:
                    break
                if (f or "").strip() != cur_key:
                    picks.append(f)
        else:
            picks = rng.sample(pool, min(n_decoys, len(pool)))
        if len(picks) < n_decoys:
            short += 1
        options = picks + [tr.x_t1]
        rng.shuffle(options)
        inst["cfd_options"] = options
    return short


