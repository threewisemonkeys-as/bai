"""Plan A — Bayesian experiment design over a *persistent* theory ensemble.

This module implements the "theory generation drives action selection" idea
described in ``notes/theory_generation_for_action_selection.md`` (Plan A), as a
self-contained layer on top of the primitives already in
``theory_exploration.py`` (Plan B). It deliberately touches nothing in
``stepwise_eb_learn.py`` — the live loop can adopt it later by importing these
functions.

Where Plan B (``theory_exploration.py``) is *stateless* — it regenerates
competing theories at each selection point and uses them only to seed crux
questions — Plan A makes the **theory ensemble itself the persistent state**: a
posterior ``p(T_i)`` carried across steps and episodes. The unit of exploration
is no longer a binary question but a full world-model, and action selection is
the classic active-learning move: take the action whose predicted outcome the
high-probability theories *most disagree on*.

The loop per (critical) step is:

  1. ``select_discriminating_action`` — given the current ensemble, propose
     candidate actions, predict each theory's observable outcome for each, and
     return the single most-discriminating action together with each theory's
     *pre-registered* predicted outcome for it. The plan text is the artifact
     injected into the agent (same type as ``current_experiment`` today).
  2. The agent acts; we observe the real outcome.
  3. ``update_theory_posterior`` — an LLM judge compares each theory's
     pre-registered prediction against the observed outcome (text + image),
     marks it consistent / violated / partial, applies a multiplicative weight
     update, renormalizes, and drops dead theories. It also reports whether the
     MAP (top-weighted) theory was violated — the *surprise* signal used to gate
     regeneration.
  4. ``refill_theories`` — on surprise (or whenever the ensemble has shrunk
     below N), generate fresh theories that differ from the survivors and merge
     them in, so the ensemble is replenished to N.

All LLM calls route through ``mixed_improve._llm_call`` for mock-mode / logging
/ cost parity, exactly as the rest of the codebase does.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field

from omegaconf import DictConfig

from mixed_improve import _llm_call
from theory_exploration import (
    Theory,
    _dk_section,
    _extract_block,
    parse_theories,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Persistence (mirrors stepwise_eb_learn_improve.serialize_eb_qa_pairs)
# ---------------------------------------------------------------------------


def serialize_theories(theories: list[Theory]) -> list[dict]:
    """Serialize a theory ensemble to JSON-friendly dicts (preserves weights)."""
    return [
        {
            "world_knowledge": t.world_knowledge,
            "rationale": t.rationale,
            "rank": t.rank,
            "likelihood": t.likelihood,
            "weight": t.weight,
        }
        for t in theories
    ]


def deserialize_theories(data: list[dict]) -> list[Theory]:
    """Inverse of :func:`serialize_theories`."""
    theories = [
        Theory(
            world_knowledge=d.get("world_knowledge", ""),
            rationale=d.get("rationale", ""),
            rank=int(d.get("rank", i + 1)),
            likelihood=d.get("likelihood", ""),
            weight=float(d.get("weight", 0.0)),
        )
        for i, d in enumerate(data)
    ]
    return theories


# ---------------------------------------------------------------------------
# Ensemble bookkeeping
# ---------------------------------------------------------------------------


def reindex_ranks(theories: list[Theory]) -> None:
    """Sort by weight (desc) and reassign contiguous ranks 1..N in place.

    Rank is used as the stable key that ties a theory to its pre-registered
    prediction within a single select/update cycle, so it must stay unique.
    Keeping rank aligned with weight order also means "rank 1" is always the
    current MAP theory.
    """
    theories.sort(key=lambda t: t.weight, reverse=True)
    for i, t in enumerate(theories, start=1):
        t.rank = i


def renormalize(theories: list[Theory]) -> None:
    """Renormalize weights to sum to 1 (in place). No-op on an empty list."""
    total = sum(t.weight for t in theories)
    if total <= 0:
        # Degenerate: fall back to uniform so the posterior stays well-defined.
        n = len(theories) or 1
        for t in theories:
            t.weight = 1.0 / n
        return
    for t in theories:
        t.weight /= total


def map_theory(theories: list[Theory]) -> Theory | None:
    """Return the maximum-a-posteriori (highest-weight) theory, or None."""
    return max(theories, key=lambda t: t.weight, default=None)


# ---------------------------------------------------------------------------
# Discriminating-action selection
# ---------------------------------------------------------------------------


@dataclass
class DiscriminatingAction:
    """The chosen maximally-discriminating action plus pre-registered outcomes.

    ``plan`` is the free-text action plan injected into the agent (mirrors the
    ``current_experiment`` artifact). ``predictions`` maps a theory's rank to
    its predicted observable outcome for ``plan``; these are handed to
    :func:`update_theory_posterior` after the action is taken.
    """

    plan: str
    rationale: str = ""
    predictions: dict[int, str] = field(default_factory=dict)
    candidate_actions: list[str] = field(default_factory=list)


def _theory_blocks(theories: list[Theory]) -> str:
    return "\n\n".join(
        f"THEORY {t.rank} (posterior weight {t.weight:.3f}):\n{t.world_knowledge.strip()}"
        for t in theories
    )


def _parse_selected_predictions(text: str, valid_ranks: set[int]) -> dict[int, str]:
    """Parse the per-theory predicted outcomes for the selected action.

    Looks inside ``<selected_action>`` for ``<theory rank="..">`` blocks whose
    body contains an ``<outcome>``. Falls back to any ``<theory rank=..>`` block
    in the whole response if the selected-action wrapper is missing.
    """
    region = _extract_block(text, "selected_action") or text
    predictions: dict[int, str] = {}
    for m in re.finditer(r"<theory\b([^>]*)>(.*?)</theory>", region, re.IGNORECASE | re.DOTALL):
        attrs, body = m.group(1), m.group(2)
        rank_m = re.search(r'rank\s*=\s*["\']?\s*(\d+)', attrs, re.IGNORECASE)
        if not rank_m:
            continue
        rank = int(rank_m.group(1))
        if rank not in valid_ranks:
            continue
        outcome = _extract_block(body, "outcome") or body.strip()
        if outcome:
            predictions[rank] = outcome.strip()
    return predictions


def _parse_candidate_actions(text: str) -> list[str]:
    region = _extract_block(text, "candidate_actions")
    if not region:
        return []
    actions: list[str] = []
    for m in re.finditer(r"<action\b[^>]*>(.*?)</action>", region, re.IGNORECASE | re.DOTALL):
        a = m.group(1).strip()
        if a:
            actions.append(a)
    return actions


def _parse_candidates_with_groups(
    text: str, valid_ranks: set[int]
) -> list[dict]:
    """Parse ``<candidate>`` blocks into structured per-theory outcome groups.

    Returns a list of ``{"id", "action", "per_theory"}`` where ``per_theory``
    maps ``rank -> (group_label, outcome_text)``. A theory whose ``group``
    attribute is missing is placed in its own singleton group so it never
    spuriously "agrees" with another theory.
    """
    region = _extract_block(text, "candidates") or text
    out: list[dict] = []
    for m in re.finditer(r"<candidate\b([^>]*)>(.*?)</candidate>", region, re.IGNORECASE | re.DOTALL):
        attrs, body = m.group(1), m.group(2)
        id_m = re.search(r'id\s*=\s*["\']?\s*(\w+)', attrs, re.IGNORECASE)
        cid = id_m.group(1) if id_m else str(len(out) + 1)
        action = (_extract_block(body, "action") or "").strip()
        per_theory: dict[int, tuple[str, str]] = {}
        for tm in re.finditer(r"<theory\b([^>]*)>(.*?)</theory>", body, re.IGNORECASE | re.DOTALL):
            tattrs, tbody = tm.group(1), tm.group(2)
            rank_m = re.search(r'rank\s*=\s*["\']?\s*(\d+)', tattrs, re.IGNORECASE)
            if not rank_m:
                continue
            rank = int(rank_m.group(1))
            if rank not in valid_ranks:
                continue
            grp_m = re.search(r'group\s*=\s*["\']?\s*([^"\'>\s]+)', tattrs, re.IGNORECASE)
            group = grp_m.group(1).strip().upper() if grp_m else f"__SOLO_{rank}"
            outcome = (_extract_block(tbody, "outcome") or tbody).strip()
            per_theory[rank] = (group, outcome)
        out.append({"id": cid, "action": action, "per_theory": per_theory})
    return out


def _candidate_eig(
    per_theory: dict[int, tuple[str, str]], weights: dict[int, float]
) -> tuple[float, int]:
    """Expected info gain proxy: Shannon entropy of posterior mass over the
    candidate's predicted-outcome groups.

    Each theory backs the outcome of its group; the mass of a group is the sum
    of its theories' posterior weights. Entropy is maximized when the groups
    split the weight evenly (most informative) and is 0 when every theory agrees
    (a single group). Returns ``(entropy_nats, num_groups)``.
    """
    group_mass: dict[str, float] = {}
    for rank, (group, _outcome) in per_theory.items():
        group_mass[group] = group_mass.get(group, 0.0) + weights.get(rank, 0.0)
    total = sum(group_mass.values())
    if total <= 0.0 or len(group_mass) < 2:
        return 0.0, len(group_mass)
    entropy = 0.0
    for mass in group_mass.values():
        p = mass / total
        if p > 0.0:
            entropy -= p * math.log(p)
    return entropy, len(group_mass)


async def select_discriminating_action(
    config: DictConfig,
    *,
    theories: list[Theory],
    beliefs: str,
    default_knowledge: str,
    steps_context: str = "",
    current_observation: str | None = None,
    current_image=None,
    steps_context_images: list | None = None,
    num_candidate_actions: int = 4,
    action_space_description: str = "",
    goal: str = "",
) -> tuple[DiscriminatingAction | None, float, dict]:
    """Choose the action whose predicted outcome most splits the top theories.

    Single LLM call: present the current weighted ensemble + state, ask the
    model to (a) propose ``num_candidate_actions`` candidate actions/short
    plans, (b) predict each theory's observable outcome for each, then (c)
    select the single most-discriminating action and restate every theory's
    *pre-registered* predicted outcome for it.

    Returns ``(DiscriminatingAction | None, cost, log)``. Returns ``None`` for
    the action only if fewer than 2 theories exist (nothing to discriminate) or
    the response was unparseable.
    """
    images: list = []
    if steps_context_images:
        images.extend(steps_context_images)
    if current_image is not None:
        images.append(current_image)

    if len(theories) < 2:
        return None, 0.0, {
            "kind": "select_discriminating_action",
            "note": "need >=2 theories to discriminate",
            "num_theories": len(theories),
        }

    obs_section = ""
    if current_observation:
        obs_section = (
            f"\n=== CURRENT STATE (text) ===\n{current_observation}\n=== END CURRENT STATE ===\n"
        )
    history_section = ""
    if steps_context:
        history_section = (
            f"\n=== RECENT HISTORY OF STATES AND ACTIONS ===\n{steps_context}\n"
            "=== END RECENT HISTORY ===\n"
        )
    action_space_section = ""
    if action_space_description.strip():
        action_space_section = (
            f"\n=== AVAILABLE ACTIONS ===\n{action_space_description.strip()}\n"
            "=== END AVAILABLE ACTIONS ===\n"
        )
    image_note = " The current game state is shown in the attached image(s)." if images else ""
    goal_section = (
        f"\n=== GOAL / WIN CONDITION ===\n{goal.strip()}\n=== END GOAL ===\n"
        if goal and goal.strip() else ""
    )
    goal_note = (
        " The theories disagree about what STRATEGY advances the goal, so prefer experiments "
        "that test those disagreements: when predicting each theory's outcome, include whether "
        "the action makes PROGRESS toward the goal (e.g. score/level/target change) under that "
        "theory, since that is the disagreement that matters most."
        if goal and goal.strip() else ""
    )

    prompt = f"""You are an agent trying to figure out how a game works by running the most \
informative experiment. You hold several COMPETING THEORIES about the game's mechanics, each \
with a posterior weight (probability). Your job is to pick the single action (or short action \
plan) whose observable outcome would best DISCRIMINATE between the high-weight theories — i.e. \
the theories predict different things will happen, so observing the real outcome will tell you \
which theories are right.{goal_note}

CRITICAL PRINCIPLE — what makes an experiment informative:
- A GOOD experiment is one the theories DISAGREE about: at least one theory predicts outcome A \
and at least one other predicts a clearly DIFFERENT outcome B. Whatever happens, you can mark \
some theories consistent and others violated, so the result tells you WHICH theory is right.
- A USELESS experiment is one where ALL theories predict the SAME outcome (the result confirms \
everything and distinguishes nothing).
- An equally USELESS experiment is one you expect ALL theories to get WRONG / be surprised by \
(e.g. poking at something none of the theories say anything specific about). If every theory is \
violated, you learn that they are all wrong but NOTHING about which one is closer to the truth — \
this is wasted. Do NOT pick an action just because it is novel or exploratory.
- Therefore: only pick an action whose outcome each relevant theory can CONFIDENTLY and \
DIFFERENTLY predict. Favor actions that split the highest-weight posterior mass roughly evenly.
{_dk_section(default_knowledge)}{goal_section}
=== CURRENT WORLD KNOWLEDGE (confirmed beliefs all theories must respect) ===
{beliefs.strip() if beliefs and beliefs.strip() else "(none yet)"}
=== END CURRENT WORLD KNOWLEDGE ===

=== COMPETING THEORIES (with posterior weights) ===
{_theory_blocks(theories)}
=== END COMPETING THEORIES ==={history_section}{obs_section}{action_space_section}
{image_note}

Do the following:
1. Propose up to {num_candidate_actions} distinct CANDIDATE actions (or short action plans) that \
are possible from the current state. Each candidate should be something whose outcome the \
theories DISAGREE about.
2. For EACH candidate, predict the concrete observable outcome under EACH theory. Then assign \
each theory to an outcome GROUP: theories that predict the SAME observable result share the same \
group label (A, B, C, ...); theories predicting clearly DIFFERENT results get different group \
labels. If all theories predict the same thing for a candidate, they all share group A (this \
candidate is uninformative).

Do NOT pick the action yourself — just lay out every candidate with its per-theory predicted \
outcomes and group labels. The most informative candidate will be selected automatically from \
the group structure (the one that splits posterior weight across the most groups, most evenly).

Respond in exactly this format:
<candidates>
<candidate id="1">
<action>Concrete action or short plan, written as a direct instruction the agent can execute.</action>
<theory rank="1" group="A"><outcome>Precise predicted observable outcome under THEORY 1.</outcome></theory>
<theory rank="2" group="B"><outcome>Predicted observable outcome under THEORY 2.</outcome></theory>
... (one <theory> block per theory above, each with a group label)
</candidate>
<candidate id="2">
...
</candidate>
... (up to {num_candidate_actions} candidates)
</candidates>"""

    text, cost = await _llm_call(config, prompt, images=images or None)

    valid_ranks = {t.rank for t in theories}
    weights = {t.rank: t.weight for t in theories}
    parsed = _parse_candidates_with_groups(text, valid_ranks)

    # Score every candidate by expected information gain = entropy of the
    # posterior mass distributed over its predicted-outcome groups. A candidate
    # on which the theories all agree has one group (EIG 0); one that splits the
    # weight evenly across distinct groups scores highest. We pick in code,
    # overriding any choice the model might express, so "discriminating" is a
    # computed quantity rather than an eyeballed one.
    scored = []
    for c in parsed:
        eig, ngroups = _candidate_eig(c["per_theory"], weights)
        scored.append({"id": c["id"], "action": c["action"], "eig": eig,
                       "num_groups": ngroups, "per_theory": c["per_theory"]})
    scored.sort(key=lambda s: (s["eig"], s["num_groups"]), reverse=True)
    candidates = [c["action"] for c in parsed]

    log = {
        "kind": "select_discriminating_action",
        "num_theories": len(theories),
        "num_candidate_actions_requested": num_candidate_actions,
        "num_candidate_actions_parsed": len(parsed),
        "theory_weights": weights,
        "candidate_scores": [
            {"id": s["id"], "action": s["action"], "eig": s["eig"],
             "num_groups": s["num_groups"]} for s in scored
        ],
        "prompt": prompt,
        "response": text,
        "candidate_actions": candidates,
    }

    best = scored[0] if scored else None
    if best is None or not best["action"]:
        logger.warning("[multi_theory] select_discriminating_action: no parseable candidate")
        log["selected_plan"] = None
        log["predictions"] = {}
        return None, cost, log

    if best["eig"] <= 0.0:
        logger.warning(
            "[multi_theory] select_discriminating_action: best candidate has EIG=0 "
            "(no action splits the theories); running it anyway."
        )

    predictions = {rank: outcome for rank, (_grp, outcome) in best["per_theory"].items()}
    # Human-readable split summary for the rationale: group -> ranks.
    group_ranks: dict[str, list[int]] = {}
    for rank, (grp, _o) in best["per_theory"].items():
        group_ranks.setdefault(grp, []).append(rank)
    split_desc = "; ".join(
        f"group {g}: theories {sorted(rs)}" for g, rs in sorted(group_ranks.items())
    )
    rationale = (
        f"Selected by max expected-info-gain (EIG={best['eig']:.3f}, "
        f"{best['num_groups']} outcome groups). Split — {split_desc}."
    )
    log["selected_plan"] = best["action"]
    log["selected_eig"] = best["eig"]
    log["predictions"] = predictions

    action = DiscriminatingAction(
        plan=best["action"].strip(),
        rationale=rationale,
        predictions=predictions,
        candidate_actions=candidates,
    )
    return action, cost, log


# ---------------------------------------------------------------------------
# Goal-directed action selection (EXPLOIT)
# ---------------------------------------------------------------------------


async def select_goal_action(
    config: DictConfig,
    *,
    map_theory: Theory,
    beliefs: str,
    default_knowledge: str,
    goal: str = "",
    steps_context: str = "",
    current_observation: str | None = None,
    current_image=None,
    steps_context_images: list | None = None,
) -> tuple[DiscriminatingAction | None, float, dict]:
    """EXPLOIT: pick the action that best advances the goal, ASSUMING the MAP
    theory is the correct model of the game.

    Unlike :func:`select_discriminating_action` (which maximizes information),
    this maximizes progress toward the win condition under the single best
    theory. It still pre-registers the MAP theory's predicted outcome (keyed by
    its rank) so the posterior keeps updating — if the action's result violates
    the MAP theory, the loop drops it and falls back to exploring.
    """
    images: list = []
    if steps_context_images:
        images.extend(steps_context_images)
    if current_image is not None:
        images.append(current_image)

    obs_section = (
        f"\n=== CURRENT STATE (text) ===\n{current_observation}\n=== END CURRENT STATE ===\n"
        if current_observation else ""
    )
    history_section = (
        f"\n=== RECENT HISTORY OF STATES AND ACTIONS ===\n{steps_context}\n"
        "=== END RECENT HISTORY ===\n" if steps_context else ""
    )
    goal_section = (
        f"\n=== GOAL / WIN CONDITION ===\n{goal.strip()}\n=== END GOAL ===\n"
        if goal and goal.strip() else ""
    )
    image_note = " The current game state is shown in the attached image(s)." if images else ""

    prompt = f"""You are an agent playing a game you now understand well. Treat the theory below \
as the CORRECT model of how the game works, and choose the single action (or short action plan) \
that best makes PROGRESS TOWARD THE GOAL / WIN CONDITION under that model. This is an \
EXPLOITATION step: act to win, not to gather information.
{_dk_section(default_knowledge)}{goal_section}
=== CONFIRMED BELIEFS ===
{beliefs.strip() if beliefs and beliefs.strip() else "(none yet)"}
=== END CONFIRMED BELIEFS ===

=== HOW THE GAME WORKS (your current best model — assume it is correct) ===
{map_theory.world_knowledge.strip()}
=== END MODEL ==={history_section}{obs_section}
{image_note}

Choose the action that, under this model, most advances the goal from the current state. Then \
state the precise observable outcome you expect (under this model) so it can be checked against \
reality.

Respond in exactly this format:
<goal_action>
<plan>The exact action / short plan to execute now, written as a direct instruction.</plan>
<rationale>Why this best advances the goal under the model.</rationale>
<expected_outcome>Precise, checkable observable outcome you expect after the action.</expected_outcome>
</goal_action>"""

    text, cost = await _llm_call(config, prompt, images=images or None)
    region = _extract_block(text, "goal_action") or text
    plan = _extract_block(region, "plan")
    rationale = _extract_block(region, "rationale")
    expected = _extract_block(region, "expected_outcome")

    log = {
        "kind": "select_goal_action",
        "map_rank": map_theory.rank,
        "map_weight": map_theory.weight,
        "prompt": prompt,
        "response": text,
        "selected_plan": plan,
        "expected_outcome": expected,
    }
    if not plan:
        logger.warning("[multi_theory] select_goal_action: no <plan> parsed")
        log["predictions"] = {}
        return None, cost, log

    predictions = {map_theory.rank: (expected or "").strip()} if expected else {}
    log["predictions"] = predictions
    action = DiscriminatingAction(
        plan=plan.strip(),
        rationale=(rationale or "").strip(),
        predictions=predictions,
        candidate_actions=[plan.strip()],
    )
    return action, cost, log


# ---------------------------------------------------------------------------
# Posterior update via pre-registered predictions (LLM judge, text + image)
# ---------------------------------------------------------------------------

# Verdict -> multiplicative weight factor, parameterized by ``violation_penalty``
# p in (0, 1): a violated theory keeps a (1 - p) fraction of its weight; a
# partial/ambiguous match keeps (1 - p/2); a consistent prediction is unchanged.
_VERDICT_CONSISTENT = "CONSISTENT"
_VERDICT_VIOLATED = "VIOLATED"
_VERDICT_PARTIAL = "PARTIAL"


def _verdict_factor(verdict: str, violation_penalty: float) -> float:
    v = verdict.strip().upper()
    if v == _VERDICT_VIOLATED:
        return max(0.0, 1.0 - violation_penalty)
    if v == _VERDICT_PARTIAL:
        return max(0.0, 1.0 - violation_penalty / 2.0)
    # CONSISTENT or anything unrecognized -> no penalty.
    return 1.0


def _parse_verdicts(text: str, valid_ranks: set[int]) -> dict[int, tuple[str, str]]:
    """Parse ``<theory rank=..>`` -> (verdict, explanation) from the judge."""
    region = _extract_block(text, "verdicts") or text
    out: dict[int, tuple[str, str]] = {}
    for m in re.finditer(r"<theory\b([^>]*)>(.*?)</theory>", region, re.IGNORECASE | re.DOTALL):
        attrs, body = m.group(1), m.group(2)
        rank_m = re.search(r'rank\s*=\s*["\']?\s*(\d+)', attrs, re.IGNORECASE)
        if not rank_m:
            continue
        rank = int(rank_m.group(1))
        if rank not in valid_ranks:
            continue
        verdict = (_extract_block(body, "verdict") or "").strip().upper()
        explanation = (_extract_block(body, "explanation") or "").strip()
        if verdict:
            out[rank] = (verdict, explanation)
    return out


async def update_theory_posterior(
    config: DictConfig,
    *,
    theories: list[Theory],
    predictions: dict[int, str],
    action_taken: str,
    observed_outcome: str | None = None,
    observed_image=None,
    default_knowledge: str = "",
    violation_penalty: float = 0.7,
    min_weight: float = 0.02,
) -> tuple[list[Theory], dict]:
    """Reweight the ensemble against the observed outcome and drop dead theories.

    An LLM judge compares each theory's *pre-registered* prediction (from
    :func:`select_discriminating_action`) against the action actually taken and
    the observed result (text + image). Each theory is marked CONSISTENT /
    VIOLATED / PARTIAL; weights are multiplied by the corresponding factor and
    renormalized. Any theory whose prediction was VIOLATED is then dropped
    (regenerate-on-falsification), as is any theory below ``min_weight``; the
    ensemble may empty out entirely on an all-violated step and is refilled by
    the caller next step. Ranks are reindexed by the new weights.

    Returns ``(theories, log)``. ``log`` includes ``surprise`` (True iff the
    pre-update MAP theory was VIOLATED *or* every theory was VIOLATED) for
    gating regeneration, ``all_violated``, ``num_dropped``, and ``cost``.
    """
    log: dict = {
        "kind": "update_theory_posterior",
        "num_theories": len(theories),
        "violation_penalty": violation_penalty,
        "min_weight": min_weight,
        "action_taken": action_taken,
        "weights_before": {t.rank: t.weight for t in theories},
    }

    if not theories:
        log["note"] = "no theories to update"
        log["surprise"] = False
        log["num_dropped"] = 0
        log["cost"] = 0.0
        return theories, log

    map_before = map_theory(theories)
    map_rank_before = map_before.rank if map_before else None

    # Only judge theories that actually registered a prediction this step. In
    # exploit mode just the MAP theory is tested; untested theories keep their
    # weight unchanged (factor 1.0) and are excluded from the all-violated /
    # surprise / falsification logic below.
    tested_ranks = {t.rank for t in theories if t.rank in predictions}

    # Build the judge prompt: each tested theory's prediction vs the real outcome.
    pred_blocks = [
        f"THEORY {t.rank} (weight {t.weight:.3f}) PREDICTED:\n"
        f"{predictions.get(t.rank, '').strip()}"
        for t in theories
        if t.rank in tested_ranks
    ]
    predictions_text = "\n\n".join(pred_blocks)

    observed_section = ""
    if observed_outcome:
        observed_section = (
            f"\n=== OBSERVED RESULT (text) ===\n{observed_outcome}\n=== END OBSERVED RESULT ===\n"
        )
    image_note = (
        " The actual resulting game state is shown in the attached image(s)."
        if observed_image is not None
        else ""
    )
    images = [observed_image] if observed_image is not None else None

    prompt = f"""You are judging which competing theories about a game survived an experiment. \
Each theory made a prediction about what would be observed after a specific action. The action \
was taken and the real result is given below. For EACH theory, decide whether its prediction was \
CONSISTENT, VIOLATED, or PARTIAL with respect to the observed result.

- CONSISTENT: the observed result matches what the theory predicted.
- VIOLATED: the observed result clearly contradicts the theory's prediction.
- PARTIAL: partially right, or the prediction was too vague to fully confirm or refute.
{_dk_section(default_knowledge)}
=== ACTION TAKEN ===
{action_taken.strip()}
=== END ACTION TAKEN ===

=== THEORY PREDICTIONS ===
{predictions_text}
=== END THEORY PREDICTIONS ==={observed_section}
{image_note}

Judge each theory strictly against the observed result (do not reward a theory for being \
plausible — only for predicting what actually happened).

Respond in exactly this format:
<verdicts>
<theory rank="1">
<verdict>CONSISTENT or VIOLATED or PARTIAL</verdict>
<explanation>One sentence on why.</explanation>
</theory>
... (one block per theory)
</verdicts>"""

    text, cost = await _llm_call(config, prompt, images=images)

    valid_ranks = {t.rank for t in theories}
    verdicts = _parse_verdicts(text, valid_ranks)

    # Apply the multiplicative update. Untested theories (no registered
    # prediction this step) are marked UNTESTED and left unchanged (factor 1.0).
    per_theory_log = []
    for t in theories:
        if t.rank in tested_ranks:
            verdict, explanation = verdicts.get(t.rank, ("UNKNOWN", "no verdict parsed"))
        else:
            verdict, explanation = ("UNTESTED", "no prediction registered this step")
        factor = _verdict_factor(verdict, violation_penalty)
        weight_before = t.weight
        t.weight *= factor
        per_theory_log.append({
            "rank": t.rank,
            "verdict": verdict,
            "explanation": explanation,
            "factor": factor,
            "weight_before": weight_before,
            "weight_after_unnorm": t.weight,
        })

    # Harvest falsification evidence — for every VIOLATED theory, what it
    # claimed, what it predicted, and how the observation contradicted it. The
    # loop accumulates these and feeds them back into regeneration so fresh
    # theories must explain what actually happened instead of re-proposing a
    # mechanic that was just ruled out.
    falsifications = [
        {
            "claim": t.world_knowledge.strip(),
            "prediction": predictions.get(t.rank, "").strip(),
            "contradiction": verdicts.get(t.rank, ("", ""))[1].strip(),
            "action": action_taken.strip(),
        }
        for t in theories
        if verdicts.get(t.rank, ("", ""))[0] == _VERDICT_VIOLATED
    ]

    # Surprise = the MAP theory (before update) was violated, OR every TESTED
    # theory was violated. Both gate regeneration. Untested theories don't count.
    map_verdict = verdicts.get(map_rank_before, ("UNKNOWN", ""))[0] if map_rank_before else "UNKNOWN"
    tested_verdicts = [pt["verdict"] for pt in per_theory_log if pt["rank"] in tested_ranks]
    all_violated = bool(tested_verdicts) and all(
        v == _VERDICT_VIOLATED for v in tested_verdicts
    )
    surprise = (map_verdict == _VERDICT_VIOLATED) or all_violated

    renormalize(theories)

    # Regenerate-on-falsification: drop any theory whose pre-registered
    # prediction was VIOLATED (it has been falsified), plus any theory that
    # decayed below ``min_weight``. The ensemble is allowed to empty out
    # completely on an all-violated step — the loop refills it next step with
    # fresh hypotheses conditioned on the latest observation. This is what
    # turns an all-violated step from a no-op shrink into real learning.
    violated_ranks = {r for r, (v, _) in verdicts.items() if v == _VERDICT_VIOLATED}
    kept = sorted(theories, key=lambda t: t.weight, reverse=True)
    survivors = [
        t for t in kept if t.weight >= min_weight and t.rank not in violated_ranks
    ]
    num_dropped = len(theories) - len(survivors)
    dropped = [t for t in theories if t not in survivors]
    theories[:] = survivors
    renormalize(theories)
    reindex_ranks(theories)

    log.update({
        "verdicts": verdicts_to_log(verdicts),
        "per_theory": per_theory_log,
        "map_rank_before": map_rank_before,
        "map_verdict": map_verdict,
        "surprise": surprise,
        "all_violated": all_violated,
        "falsifications": falsifications,
        "num_dropped": num_dropped,
        "dropped_world_knowledge": [t.world_knowledge for t in dropped],
        "weights_after": {t.rank: t.weight for t in theories},
        "prompt": prompt,
        "response": text,
        "cost": cost,
    })
    logger.info(
        f"[multi_theory] posterior update: surprise={surprise}, dropped={num_dropped}, "
        f"survivors={len(theories)}, cost=${cost:.6f}"
    )
    return theories, log


def verdicts_to_log(verdicts: dict[int, tuple[str, str]]) -> dict:
    return {str(r): {"verdict": v, "explanation": e} for r, (v, e) in verdicts.items()}


def _falsification_key(f: dict) -> str:
    """Normalized dedup key: the observed contradiction (the actual fact)."""
    return " ".join((f.get("contradiction") or "").lower().split())


def merge_falsifications(
    memory: list[dict], new: list[dict], cap: int = 16
) -> list[dict]:
    """Append new falsifications in place, skipping ones whose observed
    contradiction is already recorded, then keep only the most recent ``cap``
    DISTINCT entries.

    Each step contributes one raw falsification per VIOLATED theory, but they are
    all judged against the same observation, so their contradictions describe the
    same fact; the same falsified prediction also recurs across steps. Deduping on
    the observed contradiction keeps the memory a compact list of distinct
    observations instead of the same few facts repeated 5x — which is what was
    saturating the regeneration prompt and causing the late-episode collapse.
    """
    seen = {k for k in (_falsification_key(f) for f in memory) if k}
    for f in new:
        key = _falsification_key(f)
        if not key or key in seen:
            continue
        seen.add(key)
        memory.append(f)
    if len(memory) > cap:
        del memory[:-cap]
    return memory


def _falsifications_section(falsifications: list[dict] | None) -> str:
    """Render accumulated falsification evidence for the regeneration prompt."""
    if not falsifications:
        return ""
    blocks = []
    for i, f in enumerate(falsifications, 1):
        pred = (f.get("prediction") or "").strip()
        contra = (f.get("contradiction") or "").strip()
        act = (f.get("action") or "").strip()
        claim = (f.get("claim") or "").strip()
        blk = f"RULED-OUT #{i}:"
        if act:
            blk += f"\n  after action: {act}"
        if pred:
            blk += f"\n  a theory predicted: {pred}"
        if contra:
            blk += f"\n  but the observation showed: {contra}"
        if claim:
            blk += f"\n  (falsified claim: {claim[:300]})"
        blocks.append(blk)
    body = "\n".join(blocks)
    return (
        "\n=== RULED-OUT MECHANICS (these predictions were already FALSIFIED by "
        "experiments) ===\n"
        f"{body}\n"
        "Your new theories MUST be consistent with every observation above and MUST "
        "NOT repeat any of these falsified predictions or the mechanics behind them.\n"
        "=== END RULED-OUT MECHANICS ===\n"
    )


# ---------------------------------------------------------------------------
# Refill the ensemble back to N (anchored on survivors for diversity)
# ---------------------------------------------------------------------------


async def refill_theories(
    config: DictConfig,
    *,
    theories: list[Theory],
    beliefs: str,
    default_knowledge: str,
    num_theories: int = 5,
    steps_context: str = "",
    current_observation: str | None = None,
    current_image=None,
    steps_context_images: list | None = None,
    falsifications: list[dict] | None = None,
    goal: str = "",
    new_theory_weight_fraction: float = 0.5,
) -> tuple[list[Theory], float, dict]:
    """Generate fresh theories that differ from the survivors and merge them in.

    Survivors keep their learned posterior weights. The newly generated theories
    are seeded with a small prior — collectively ``new_theory_weight_fraction``
    of the survivors' total mass (so a freshly-replenished hypothesis can win but
    does not immediately dominate a theory that has already earned its weight) —
    after which the whole ensemble is renormalized and reindexed.

    If the ensemble is already at or above ``num_theories`` this is a no-op.
    """
    need = num_theories - len(theories)
    log: dict = {
        "kind": "refill_theories",
        "num_existing": len(theories),
        "num_requested_new": max(0, need),
        "num_theories_target": num_theories,
    }
    if need <= 0:
        log["note"] = "ensemble already full"
        return theories, 0.0, log

    images: list = []
    if steps_context_images:
        images.extend(steps_context_images)
    if current_image is not None:
        images.append(current_image)

    existing_section = "\n\n".join(
        f"EXISTING THEORY {i + 1}:\n{t.world_knowledge.strip()}"
        for i, t in enumerate(theories)
    ) or "(none)"

    obs_section = ""
    if current_observation:
        obs_section = (
            f"\n=== CURRENT STATE (text) ===\n{current_observation}\n=== END CURRENT STATE ===\n"
        )
    history_section = ""
    if steps_context:
        history_section = (
            f"\n=== RECENT HISTORY OF STATES AND ACTIONS ===\n{steps_context}\n"
            "=== END RECENT HISTORY ===\n"
        )
    image_note = " The current game state is shown in the attached image(s)." if images else ""
    falsified_section = _falsifications_section(falsifications)
    goal_section = (
        f"\n=== GOAL / WIN CONDITION ===\n{goal.strip()}\n=== END GOAL ===\n"
        if goal and goal.strip() else ""
    )
    goal_requirement = (
        " Each new theory MUST be GOAL-AWARE: state (a) what it hypothesizes is required to make "
        "progress / score / complete the level, (b) the concrete STRATEGY (what actions advance "
        "the goal under it, and why), and (c) the minimal mechanics that strategy relies on. The "
        "new theories should propose DIFFERENT strategies for reaching the goal than the existing "
        "and ruled-out ones."
        if goal and goal.strip() else ""
    )

    prompt = f"""You are an agent trying to understand how a game works. Some theories about the \
game have already been proposed (and some were ruled out by experiments). Propose {need} NEW, \
DISTINCT theories that are genuinely different from the existing ones below — do not rephrase \
them. Each must be consistent with the default knowledge and confirmed beliefs, and should offer \
a different explanation of the unknown mechanics / win condition.{goal_requirement}
{_dk_section(default_knowledge)}{goal_section}
=== CONFIRMED BELIEFS (all theories must respect these) ===
{beliefs.strip() if beliefs and beliefs.strip() else "(none yet)"}
=== END CONFIRMED BELIEFS ===

=== EXISTING THEORIES (propose alternatives to these) ===
{existing_section}
=== END EXISTING THEORIES ==={history_section}{obs_section}{falsified_section}
{image_note}

Respond in exactly this format:
<theories>
<theory rank="1" likelihood="<rough probability or short phrase>">
<content>
- ...
</content>
<rationale>Why this is a plausible alternative.</rationale>
</theory>
... (ranks 2..{need})
</theories>"""

    text, cost = await _llm_call(config, prompt, images=images or None)
    new_theories = parse_theories(text)[:need]

    # Seed the new theories with a fraction of the survivors' total weight,
    # split evenly, so they are testable but don't swamp earned posteriors.
    survivor_mass = sum(t.weight for t in theories) or 1.0
    if new_theories:
        seed_each = (new_theory_weight_fraction * survivor_mass) / len(new_theories)
        for t in new_theories:
            t.weight = seed_each
        theories.extend(new_theories)
        renormalize(theories)
        reindex_ranks(theories)

    log.update({
        "num_parsed_new": len(new_theories),
        "new_world_knowledge": [t.world_knowledge for t in new_theories],
        "num_falsifications_used": len(falsifications or []),
        "weights_after": {t.rank: t.weight for t in theories},
        "prompt": prompt,
        "response": text,
        "cost": cost,
    })
    logger.info(
        f"[multi_theory] refilled {len(new_theories)} theories "
        f"(target {num_theories}); cost=${cost:.6f}"
    )
    return theories, cost, log


# ---------------------------------------------------------------------------
# Convenience: fresh ensemble from scratch (full regeneration on surprise)
# ---------------------------------------------------------------------------


async def init_theory_ensemble(
    config: DictConfig,
    *,
    beliefs: str,
    default_knowledge: str,
    num_theories: int = 5,
    decay: float = 0.6,
    steps_context: str = "",
    current_observation: str | None = None,
    current_image=None,
    steps_context_images: list | None = None,
    goal: str = "",
) -> tuple[list[Theory], float, dict]:
    """Generate a fresh weighted ensemble of ``num_theories`` theories.

    Thin wrapper over ``theory_exploration.generate_theories`` for the cold-start
    / full-regeneration case; weights come from the rank prior. Reindexes ranks
    by weight so rank 1 is the MAP theory, matching the rest of this module. When
    ``goal`` is given, the generated theories are goal-aware (win condition +
    strategy), not just mechanic descriptions.
    """
    from theory_exploration import generate_theories

    theories, cost, log = await generate_theories(
        config,
        beliefs=beliefs,
        default_knowledge=default_knowledge,
        steps_context=steps_context,
        current_observation=current_observation,
        current_image=current_image,
        steps_context_images=steps_context_images,
        num_theories=num_theories,
        decay=decay,
        goal=goal,
    )
    reindex_ranks(theories)
    return theories, cost, log
