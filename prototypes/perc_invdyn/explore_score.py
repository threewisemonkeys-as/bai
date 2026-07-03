"""Active-exploration ACQUISITION scoring for the inverse/forward-dynamics learner.

Given the current raw observation X_t and the candidate actions available there, score
each action by its expected VALUE TO LEARNING and return the argmax (the "decide optimal
step" box of the explore->act->learn loop). This is online experimental design for the
world model, not reward-seeking exploration: the objective is purely epistemic.

WHY these signals (all traceable to the offline findings):
  * observability  -- defeats the 64%-unobservable killer (gepa-sweep-failure-attribution):
                      an action whose effect leaves no trace yields an unusable transition
                      for inverse dynamics AND zero gradient for forward dynamics (stale
                      frame = the blind-P degeneracy). Prefer actions predicted to MOVE.
  * disagreement   -- the GEPA pareto frontier IS a free ensemble. Where its members
                      predict DIFFERENT next-states, the resulting transition resolves the
                      most: classic disagreement / value-of-information.
  * coverage       -- restore per-action balance (the held-out gate needs it to kill the
                      frequency-prior cheat) and avoid extending a `stuck` run.

The model-based signals (observability, disagreement) reuse predict_next_state -- ONE
frozen forward LLM rollout per (frontier member x candidate action). That is the cost
knob; a deterministic history-based observability fallback is provided for cheap ranking.

PURITY (invdyn-no-external-knowledge): every signal is computed over P's OWN emitted
symbols (run_perceive output) and predicted next-states -- no raw-grid parse, no
hand-coded background, no game facts.

Sketch / prototype: the env-stepping side (apply chosen action, append transition,
re-run GEPA every k steps) is left as documented TODOs at the bottom -- this file is the
acquisition core those hooks call.
"""

import asyncio
import hashlib
from collections import Counter
from dataclasses import dataclass, field

# Reuse the validated plumbing verbatim (same dir; run via uv from repo root).
from gepa_optimize import predict_next_state          # async forward rollout: (cfg,z_t,action,beliefs,sem)->(z_hat,cost)
from forward_objective import tok_multiset            # content-agnostic atomizer over P's output
from validate import Transition, run_perceive         # P over a single frame: (code,raw)->(out,err)


# A candidate is a GEPA frontier member. Accept either key spelling seen in gepa_optimize.
def _perc(c: dict) -> str:
    return c.get("perception", "") or ""

def _belief(c: dict) -> str:
    return c.get("world_knowledge", c.get("beliefs", "")) or ""


# ---------------------------------------------------------------------------
# Deterministic primitives over P-output token multisets (no game semantics).
#
# PURITY (invdyn-no-external-knowledge): these operate ONLY on P's emitted symbols, as-is.
# The acquisition scorer does NOT edit P's output -- no decoy masking, no token deletion,
# no "this field is a counter, ignore it" heuristic. An always-moving counter is P's
# problem to abstract away, driven by the LEARNING objective (an action-independent field
# carries zero inverse-dynamics signal, so a P that wastes representation on it earns no
# accuracy and is selected against by the held-out gate). The scorer must not paper over a
# weak P with a side-channel filter -- that just hides the very signal the loop exists to
# fix. See the COUNTER-DECOY note below score_actions.
# ---------------------------------------------------------------------------
def _symdiff(a: str, b: str) -> Counter:
    """Symmetric-difference of two P-output token multisets, unedited."""
    return (tok_multiset(a) - tok_multiset(b)) | (tok_multiset(b) - tok_multiset(a))

def _delta_size(a: str, b: str) -> int:
    """Size of the symmetric-difference. 0 => P sees the two states as identical (a
    predicted no-op in P's OWN representation)."""
    return sum(_symdiff(a, b).values())

def _delta_agree(z_t: str, zi: str, zj: str) -> float:
    """F1 agreement of two members' PREDICTED CHANGES (each vs the shared start z_t).
    1.0 => same predicted delta; 0.0 => disjoint. Grades dynamics over P's own symbols."""
    di, dj = _symdiff(zi, z_t), _symdiff(zj, z_t)
    if not di and not dj:
        return 1.0                      # both predict no-op: agree (and obs handles the penalty)
    inter = sum((di & dj).values())
    p = inter / max(1, sum(di.values()))
    r = inter / max(1, sum(dj.values()))
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)

def _hash_state(z: str) -> str:
    return hashlib.md5((z or "").encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Per-action score record.
# ---------------------------------------------------------------------------
@dataclass
class ActionScore:
    action: str
    observability: float = 0.0     # expected |delta| in P's own representation         [0,1]
    disagreement: float = 0.0      # 1 - mean pairwise delta-agreement among members  [0,1]
    coverage: float = 0.0          # novelty/balance bonus from buffer action counts   [0,1]
    contrast: float = 0.0          # completes the contrast set at the current context [0,1]
    revisit_pen: float = 0.0       # predicted to land on a recently-seen state        [0,1]
    total: float = 0.0
    z_hats: list = field(default_factory=list)   # per-member predicted next-state (for inspection/contrastive logic)


# ---------------------------------------------------------------------------
# Forward-rollout cache: perceive X_t once per member, predict z_hat per (member, action).
# All LLM calls fan out concurrently under the shared semaphore.
# ---------------------------------------------------------------------------
async def rollout(cfg, frontier: list[dict], x_t: str, actions: list[str], sem):
    """Returns (z_t_by_member, z_hat[(action, m)] , total_cost). One predict_next_state
    per (member, action); members' z_t computed once and reused across actions."""
    z_t = [run_perceive(_perc(c), x_t)[0] for c in frontier]      # deterministic, no LLM
    jobs, keys = [], []
    for m, c in enumerate(frontier):
        for a in actions:
            keys.append((a, m))
            jobs.append(predict_next_state(cfg, z_t[m], a, _belief(c), sem))
    out = await asyncio.gather(*jobs)
    z_hat = {k: zh for k, (zh, _) in zip(keys, out)}
    cost = sum(c for _, c in out)
    return z_t, z_hat, cost


# ---------------------------------------------------------------------------
# Coverage / balance from the buffer (deterministic, free).
# ---------------------------------------------------------------------------
def coverage_bonus(action: str, action_counts: Counter) -> float:
    """UCB-style novelty: under-sampled actions score high, decaying as they accumulate.
    Restores the action balance the held-out gate needs."""
    n = action_counts.get(action, 0)
    return 1.0 / (1.0 + n) ** 0.5


# ---------------------------------------------------------------------------
# The scorer: combine signals into a single ranking. Signals are pure functions of P's
# UNEDITED output and the frontier's forward predictions (no token masking).
#
# COUNTER-DECOY, handled on the LEARNING side (not here): an action-independent field
# (step counter) inflates observability with a constant pedestal -- a real move still
# outscores a blocked one (the move adds its own tokens on top), so the RANKING survives,
# just softer. We deliberately do NOT mask it: an honest acquisition scorer reports P as
# it is. The counter is removed by P LEARNING to stop emitting it -- the inverse-dynamics
# objective gives an action-independent feature zero credit and the held-out gate selects
# against a P that wastes representation on it. As P abstracts the counter, observability
# sharpens (no-ops -> 0) AND identical-but-for-the-counter states finally cluster, so the
# contrast machinery below switches on. That coupling is the point: the loop fixes the
# decoy at its source instead of hiding it from the metric.
# ---------------------------------------------------------------------------
def score_actions(
    actions: list[str],
    z_t: list[str],
    z_hat: dict,
    action_counts: Counter,
    recent_hashes: set,
    *,
    contrast_actions: set | None = None,  # actions still missing at the current context cluster
    w_obs: float = 1.0,
    w_dis: float = 1.0,
    w_cov: float = 0.5,
    w_contrast: float = 0.7,
    w_revisit: float = 0.5,
    obs_norm: int = 8,                # |delta| token count that saturates observability to 1.0
) -> list[ActionScore]:
    M = len(z_t)
    contrast_actions = contrast_actions or set()
    scored = []
    for a in actions:
        zh = [z_hat[(a, m)] for m in range(M)]

        # observability: mean predicted change magnitude across members, in P's own symbols.
        obs = sum(min(1.0, _delta_size(z_t[m], zh[m]) / obs_norm) for m in range(M)) / max(1, M)

        # disagreement: 1 - mean pairwise delta-agreement (only meaningful with >=2 members).
        if M >= 2:
            agrees = [_delta_agree(z_t[m], zh[m], zh[n])
                      for m in range(M) for n in range(m + 1, M)]
            dis = 1.0 - sum(agrees) / len(agrees)
        else:
            dis = 0.0

        # contrast: completing a recurring context's action set isolates this action's
        # effect across action-diverse examples (the no-reset minimal-pair surrogate).
        contrast = 1.0 if a.split()[0] in {c.split()[0] for c in contrast_actions} else 0.0

        # revisit: predicted next-state already seen recently -> extends a stuck run.
        revisit = sum(_hash_state(zh[m]) in recent_hashes for m in range(M)) / max(1, M)

        cov = coverage_bonus(a, action_counts)
        total = (w_obs * obs + w_dis * dis + w_cov * cov
                 + w_contrast * contrast - w_revisit * revisit)
        scored.append(ActionScore(a, obs, dis, cov, contrast, revisit, total, zh))
    scored.sort(key=lambda s: s.total, reverse=True)
    return scored


async def select_action(cfg, frontier, x_t, actions, action_counts, recent_hashes, sem,
                        *, contrast_actions=None, **weights):
    """Top-level 'decide optimal step'. Returns (best ActionScore, ranked list, cost)."""
    z_t, z_hat, cost = await rollout(cfg, frontier, x_t, actions, sem)
    ranked = score_actions(actions, z_t, z_hat, action_counts, recent_hashes,
                           contrast_actions=contrast_actions, **weights)
    return ranked[0], ranked, cost


# ---------------------------------------------------------------------------
# Cheap deterministic observability fallback (no LLM) -- for warmup / budget control.
# ---------------------------------------------------------------------------
def historical_observability(action: str, buffer: list[Transition], perc_code: str) -> float:
    """Fraction of past transitions with this action where P's output changed. Lets you
    rank by observability with zero rollout cost once the buffer has a few of each action."""
    seen = [tr for tr in buffer if tr.action.split()[0] == action.split()[0]]
    if not seen:
        return 1.0   # untried -> optimistic, so warmup actually tries it
    moved = sum(_delta_size(run_perceive(perc_code, tr.x_t)[0],
                            run_perceive(perc_code, tr.x_t1)[0]) > 0 for tr in seen)
    return moved / len(seen)


# ---------------------------------------------------------------------------
# Contrastive targeting WITHOUT reset.
#
# With reset you'd revisit one state and branch each action so the counter cancels. With
# NO reset the surrogate is OPPORTUNISTIC contrast: a long rollout re-passes through
# near-duplicate P-contexts on its own; when it does, prefer the action still MISSING from
# that cluster so the per-action effect is identifiable across action-diverse examples
# (statistical separation, not exact cancellation).
#
# Context identity is P's UNEDITED output (no masking). Consequence, by design: while P
# still emits a per-step counter, NO two states are identical, so clustering finds nothing
# and this machinery stays dormant. It SWITCHES ON precisely as the learning objective
# drives P to abstract the counter away -- the same event that sharpens observability. We
# accept that coupling rather than force clustering with an external filter: contrast
# effectiveness is a read-out of P quality, kept inside the learned pipeline.
#
# missing_at_context() supplies `contrast_actions` to score_actions when the current state
# matches a recurring cluster. To ACTIVELY return to an under-contrasted cluster rather
# than wait to drift back, see the return-to-context note in the loop skeleton.
# ---------------------------------------------------------------------------
def _ctx_key(z: str) -> str:
    """Context identity over P's unedited output. Order-invariant over P's token multiset."""
    return hashlib.md5(" ".join(sorted(tok_multiset(z).elements())).encode()).hexdigest()[:12]


def missing_at_context(x_t: str, actions: list[str], perc_code: str,
                       buffer: list[Transition]) -> set:
    """Actions not yet observed from a context whose P-output matches x_t's. Feeding these
    as contrast_actions completes the cluster's contrast set -> isolates the per-action
    effect without reset. Only fires once P's output actually repeats across states."""
    key = _ctx_key(run_perceive(perc_code, x_t)[0])
    tried = {tr.action.split()[0] for tr in buffer
             if _ctx_key(run_perceive(perc_code, tr.x_t)[0]) == key}
    return {a for a in actions if a.split()[0] not in tried}


def under_contrasted_clusters(buffer: list[Transition], perc_code: str, action_pool: list[str],
                              min_seen: int = 2) -> list[tuple[str, set]]:
    """Recurring contexts (seen >= min_seen times) that still lack actions. Return
    (ctx_key, missing_actions) sorted by how close each is to a complete contrast set --
    the navigation targets for return-to-context. The forward model would then plan a short
    path from the current state to one of these (the no-reset substitute for reset+branch);
    planning itself needs the env handle and is left to the loop side."""
    seen: dict[str, Counter] = {}
    for tr in buffer:
        seen.setdefault(_ctx_key(run_perceive(perc_code, tr.x_t)[0]), Counter())[tr.action.split()[0]] += 1
    pool = {a.split()[0] for a in action_pool}
    out = [(k, pool - set(c)) for k, c in seen.items() if sum(c.values()) >= min_seen and pool - set(c)]
    return sorted(out, key=lambda kv: len(kv[1]))   # nearest-to-complete first


# ---------------------------------------------------------------------------
# LOOP SKELETON -- NO-RESET edition (the env side; TODO, needs the env handle).
#
# No reset => (a) a fresh EPISODE is the only coarse reset; budget across many; start a
# new one when dead/stuck. (b) the counter is NOT masked -- P learns to drop it. (c)
# irreversible actions are costly (one life), so a reversibility penalty guards the run.
# ---------------------------------------------------------------------------
# async def explore_learn(cfg, make_env, action_pool, sem, budget, k_relearn=10, warmup=8):
#     buffer, frontier = [], [{"perception": SEED_P, "world_knowledge": ""}]
#     action_counts, recent = Counter(), []
#     env = make_env(); x_t = env.observe(); step = 0
#     while step < budget:
#         if env.done() or _stuck(recent):            # NO reset -> new episode = coarse reset
#             env = make_env(); x_t = env.observe(); recent.clear(); continue
#         actions = env.available_actions()           # ARC: ACTION1-5; ft09 needs a PROPOSER
#         if step < warmup:                           # chicken-egg: model too weak to steer early
#             a = random.choice(actions)              # or rank by historical_observability
#         else:
#             contrast = missing_at_context(x_t, actions, _perc(frontier[0]), buffer)
#             best, ranked, _ = await select_action(
#                 cfg, frontier, x_t, actions, action_counts, set(recent[-5:]), sem,
#                 contrast_actions=contrast)
#             a = best.action
#             # REVERSIBILITY guard (no reset = one life): if the model predicts a is a
#             # one-way/dead-end move and its info gain is modest, fall back to the next
#             # ranked reversible action. Reversibility ~ predicted next-state still has
#             # observable actions available (not all-no-op). Loop-side; needs lookahead.
#         x_t1 = env.step(a)                          # ACT & OBSERVE
#         buffer.append(Transition(x_t, x_t1, a)); action_counts[a] += 1
#         recent.append(_hash_state(run_perceive(_perc(frontier[0]), x_t1)[0]))
#         x_t = x_t1; step += 1
#         if step >= warmup and step % k_relearn == 0:            # LEARN
#             frontier = run_gepa(cfg, buffer, action_pool, ...)  # held-out gate; refresh ensemble
#     return buffer, frontier
#
# RETURN-TO-CONTEXT (optional, replaces passive drift): instead of waiting to re-pass a
# recurring context, periodically pick a target from under_contrasted_clusters() and use
# the forward model to plan a short path to it, then take a missing action there. The
# no-reset substitute for reset+branch; only as reliable as the current forward model.
#
# Metric that proves the loop (mirrors every other validation here): held-out
# inverse-dynamics accuracy vs ENV-STEPS, active selection vs the passive
# stepwise_eb_learn buffer at equal step budget. Win = higher acc/step + usable
# (action-observable) fraction well above 36% as P sheds the counter.
