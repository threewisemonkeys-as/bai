"""Inverse-dynamics with a SECOND optimized input B (world knowledge).

Extends validate.py: the predictor F now reads (P(X_t), P(X_t+1), B) -> Ahat.
B is a shared text block of general environment facts (coordinate conventions,
what the agent is, what actions do). Two gradients now:
  - g1_P : perception deficiency (features missing/misrepresenting state)
  - g1_B : world-knowledge deficiency (a general rule needed to interpret features)
attributed by a single feedback call, then applied by coordinate descent
(alternate P / B updates), each accepted only by a held-out gate.

Soundness controls (B is optimized against labels, so guard against cheating):
  - B is one block shared across ALL transitions -> structurally cannot encode
    per-instance answers; only general rules or dataset priors.
  - held-out is BALANCED across actions -> an action-frequency prior in B buys
    nothing, so any B gain must come from genuine disambiguation.

Question: does adding+optimizing B push the score BEYOND the raw-frame
reference (~0.55) toward the analytic ceiling (1.00) by supplying the
coordinate convention F was forced to guess?

Usage:
  uv run offline_learning/validate_beliefs.py --run "A,B" --rounds 4
"""

import argparse
import asyncio
import random
from collections import defaultdict
from pathlib import Path

# Reuse pure helpers + LLM plumbing from the base prototype / repo.
from validate import (  # noqa: E402  (same dir; run via uv from repo root)
    GOOD_P,
    Transition,
    _extract_code,
    _extract_action,
    _parse_tag,
    _llm_call,
    load_transitions,
    make_choices,
    make_config,
    perception_runs,
    run_perceive,
)

# ---------------------------------------------------------------------------
# F: predictor now also reads world knowledge B.
# ---------------------------------------------------------------------------
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


async def predict_action(cfg, z_t, z_t1, beliefs, choices, sem):
    prompt = PREDICT_TMPL.format(
        beliefs=beliefs.strip() or "(empty)",
        z_t=z_t or "(empty)",
        z_t1=z_t1 or "(empty)",
        choices="\n".join(f"- {c}" for c in choices),
    )
    async with sem:
        text, cost = await _llm_call(cfg, prompt)
    text = text or ""  # models intermittently return None (MALFORMED) -> treat as a failed prediction
    pred = _extract_action(text, choices)
    return pred, _parse_tag(text, "reasoning"), cost


class Rec:
    __slots__ = ("tr", "z_t", "z_t1", "pred", "reasoning", "correct")

    def __init__(self, tr, z_t, z_t1, pred, reasoning):
        self.tr, self.z_t, self.z_t1 = tr, z_t, z_t1
        self.pred, self.reasoning = pred, reasoning
        self.correct = pred == tr.action


async def forward_eval(cfg, code, beliefs, transitions, action_pool, k_choices, sem, rng,
                       raw_mode=False):
    async def one(tr):
        if raw_mode:
            z_t, z_t1 = tr.x_t[:6000], tr.x_t1[:6000]
        else:
            z_t = run_perceive(code, tr.x_t)[0]
            z_t1 = run_perceive(code, tr.x_t1)[0]
        choices = make_choices(tr.action, action_pool, k_choices, rng)
        pred, reasoning, cost = await predict_action(cfg, z_t, z_t1, beliefs, choices, sem)
        return Rec(tr, z_t, z_t1, pred, reasoning), cost

    results = await asyncio.gather(*(one(tr) for tr in transitions))
    records = [r for r, _ in results]
    cost = sum(c for _, c in results)
    acc = sum(r.correct for r in records) / max(1, len(records))
    return acc, records, cost


# ---------------------------------------------------------------------------
# Attributed gradient: split failures into perception vs world-knowledge.
# ---------------------------------------------------------------------------
G1_TMPL = """A perception module summarizes grid states into text features. A separate WORLD KNOWLEDGE block holds general facts about the environment. A predictor uses (features of two consecutive states + world knowledge) to identify the action taken, WITHOUT seeing the raw grid. Below are FAILED cases.

=== CURRENT WORLD KNOWLEDGE ===
{beliefs}
=== END WORLD KNOWLEDGE ===

{cases}

Attribute each deficiency to exactly one of:
- PERCEPTION: the features are missing or misrepresenting information that is in the state (e.g. agent position not reported).
- WORLD_KNOWLEDGE: the features are adequate, but a GENERAL rule/convention needed to interpret them is missing (e.g. which coordinate index is the column; whether increasing column means right or left; what entity the agent is).

Respond as:
<perception_deficiencies>
- concrete bullets, or the single word: none
</perception_deficiencies>
<world_knowledge_deficiencies>
- concrete bullets naming the missing GENERAL rule/convention, or the single word: none
</world_knowledge_deficiencies>"""


async def compute_g1(cfg, failures, beliefs, sem, max_cases=8):
    if not failures:
        return "", "", 0.0
    cases = []
    for i, r in enumerate(failures[:max_cases], 1):
        cases.append(
            f"--- case {i} ---\n"
            f"STATE 1 features: {r.z_t or '(empty)'}\n"
            f"STATE 2 features: {r.z_t1 or '(empty)'}\n"
            f"predicted: {r.pred or '(none)'} | TRUE action: {r.tr.action}\n"
            f"predictor reasoning: {r.reasoning[:300]}"
        )
    prompt = G1_TMPL.format(beliefs=beliefs.strip() or "(empty)", cases="\n\n".join(cases))
    async with sem:
        text, cost = await _llm_call(cfg, prompt)
    return _parse_tag(text, "perception_deficiencies"), _parse_tag(text, "world_knowledge_deficiencies"), cost


# ---------------------------------------------------------------------------
# update P (perception code) -- knows B so it does not duplicate conventions.
# ---------------------------------------------------------------------------
UPDATE_P_TMPL = """You are improving a Python perception module for a grid environment.

It must define `perceive(observation_history: list[str]) -> str`. `observation_history[-1]` is the current raw observation. Output a concise (<2000 char) text summary of decision-relevant features. Never raise, never return empty.

A separate WORLD KNOWLEDGE block already supplies general conventions, so DO NOT put coordinate conventions or rules here -- only EXTRACT state features:
=== WORLD KNOWLEDGE ===
{beliefs}
=== END WORLD KNOWLEDGE ===

=== CURRENT PERCEPTION MODULE ===
```python
{code}
```

Diagnosed PERCEPTION deficiencies (the gradient):
{g1_p}

Raw observation examples (see what is extractable; do NOT echo the raw grid):
{raw_examples}

Rewrite the FULL module to fix the deficiencies while keeping what worked. Respond with the full module in one block:
```python
<code>
```"""


async def update_perception(cfg, code, g1_p, beliefs, raw_examples, sem, n_examples=2):
    ex = []
    for i, tr in enumerate(raw_examples[:n_examples], 1):
        ex.append(f"--- example {i} (action: {tr.action}) ---\nX_t: {tr.x_t[:2200]}\nX_t+1: {tr.x_t1[:2200]}")
    prompt = UPDATE_P_TMPL.format(
        beliefs=beliefs.strip() or "(empty)", code=code or "# (empty)",
        g1_p=g1_p or "(none)", raw_examples="\n\n".join(ex),
    )
    async with sem:
        text, cost = await _llm_call(cfg, prompt)
    return _extract_code(text), cost


# ---------------------------------------------------------------------------
# update B (world knowledge) -- distills general rules from LABELED examples.
# This is where the coordinate convention gets extracted from the action labels.
# ---------------------------------------------------------------------------
UPDATE_B_TMPL = """You maintain a WORLD KNOWLEDGE block: concise GENERAL facts about how this grid environment works (coordinate conventions, what entities are, what each action does). It is shared across ALL states, so it must contain only general truths -- NEVER facts about one specific state, and NEVER action frequencies/priors.

=== CURRENT WORLD KNOWLEDGE ===
{beliefs}
=== END CURRENT WORLD KNOWLEDGE ===

A predictor uses this knowledge to map perception features to the action taken between two states. It errs due to missing GENERAL knowledge:
{g1_b}

Labeled examples (perception features of two consecutive states + the TRUE action). Infer the general rule(s) that explain ALL of them -- especially how feature changes map to action names:
{labeled_cases}

Rewrite the FULL world knowledge block. Keep it concise and GENERAL: state what entity the agent is, the coordinate convention, and how each action changes the features -- but INFER every such fact ONLY from the labeled examples above; do not assume any specific mapping. Do NOT include state-specific facts or which action is most common.

Respond as:
<world_knowledge>
- ...
</world_knowledge>"""


async def update_beliefs(cfg, beliefs, g1_b, labeled, sem, n_examples=8):
    cases = []
    for i, r in enumerate(labeled[:n_examples], 1):
        cases.append(
            f"--- example {i} ---\nSTATE 1 features: {r.z_t or '(empty)'}\n"
            f"STATE 2 features: {r.z_t1 or '(empty)'}\nTRUE action: {r.tr.action}"
        )
    prompt = UPDATE_B_TMPL.format(
        beliefs=beliefs.strip() or "(empty)", g1_b=g1_b or "(none)",
        labeled_cases="\n\n".join(cases),
    )
    async with sem:
        text, cost = await _llm_call(cfg, prompt)
    return (_parse_tag(text, "world_knowledge") or text.strip()), cost


# ---------------------------------------------------------------------------
# Balanced-action holdout: equalize action counts so frequency priors in B
# give no edge -> any B gain reflects genuine disambiguation.
# ---------------------------------------------------------------------------
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


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="comma-separated run dirs (pooled)")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--train-n", type=int, default=18)
    ap.add_argument("--holdout-n", type=int, default=18)
    ap.add_argument("--k-choices", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--model", default="google/gemini-2.5-flash")
    ap.add_argument("--client", default="openrouter")
    ap.add_argument("--actions", default="left,right,up,down,noop")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cfg = make_config(args.model, args.client)
    sem = asyncio.Semaphore(args.concurrency)
    whitelist = set(filter(None, args.actions.split(","))) or None

    run_dirs = [Path(p) for p in args.run.split(",") if p.strip()]
    transitions = load_transitions(run_dirs, whitelist)
    rng.shuffle(transitions)
    train, holdout = balanced_split(transitions, args.holdout_n, args.train_n, rng)
    action_pool = sorted({t.action for t in transitions})
    from collections import Counter
    print(f"transitions: {len(transitions)} | train={len(train)} holdout={len(holdout)} (balanced)")
    print(f"holdout action balance: {dict(Counter(t.action for t in holdout))}")
    print(f"action pool ({len(action_pool)}): {action_pool}\n")

    total = 0.0
    chance = 1.0 / args.k_choices
    raw_acc, _, c = await forward_eval(cfg, "", "", holdout, action_pool, args.k_choices, sem, rng, raw_mode=True); total += c
    print(f"[ref] random={chance:.2f} | raw-frame={raw_acc:.2f}\n")

    code = ""  # learning always starts from an empty perception module
    beliefs = ""
    h_acc, _, c = await forward_eval(cfg, code, beliefs, holdout, action_pool, args.k_choices, sem, rng); total += c
    print(f"[round 0] start (P=empty, B=empty): held-out = {h_acc:.2f}")
    start_acc = h_acc
    best_acc, best_code, best_beliefs = h_acc, code, beliefs

    for rnd in range(1, args.rounds + 1):
        active = "P" if rnd % 2 == 1 else "B"  # P,B,P,B,...
        tr_acc, recs, c = await forward_eval(cfg, code, beliefs, train, action_pool, args.k_choices, sem, rng); total += c
        failures = [r for r in recs if not r.correct]
        g1_p, g1_b, c = await compute_g1(cfg, failures, beliefs, sem); total += c

        if active == "P":
            cand, c = await update_perception(cfg, code, g1_p, beliefs, [r.tr for r in failures] or train, sem); total += c
            ok, err = perception_runs(cand, [t.x_t for t in train[:4]])
            if not ok:
                print(f"[round {rnd}|P] update REJECTED (runtime: {err}); keep")
                continue
            new_acc, _, c = await forward_eval(cfg, cand, beliefs, holdout, action_pool, args.k_choices, sem, rng); total += c
            verdict = "ACCEPT" if new_acc >= best_acc else "reject"
            print(f"[round {rnd}|P] train={tr_acc:.2f} fails={len(failures)} | cand held-out={new_acc:.2f} (best={best_acc:.2f}) -> {verdict}")
            print(f"          g1_P: {g1_p[:200].replace(chr(10),' ')}")
            if new_acc >= best_acc:
                best_acc, best_code, code = new_acc, cand, cand
        else:
            cand, c = await update_beliefs(cfg, beliefs, g1_b, failures or recs, sem); total += c
            new_acc, _, c = await forward_eval(cfg, code, cand, holdout, action_pool, args.k_choices, sem, rng); total += c
            verdict = "ACCEPT" if new_acc >= best_acc else "reject"
            print(f"[round {rnd}|B] train={tr_acc:.2f} fails={len(failures)} | cand held-out={new_acc:.2f} (best={best_acc:.2f}) -> {verdict}")
            print(f"          g1_B: {g1_b[:200].replace(chr(10),' ')}")
            print(f"          B'  : {cand[:200].replace(chr(10),' ')}")
            if new_acc >= best_acc:
                best_acc, best_beliefs, beliefs = new_acc, cand, cand

    print(f"\n=== SUMMARY ===")
    print(f"start held-out      : {start_acc:.2f}")
    print(f"final held-out (P+B): {best_acc:.2f}  (delta {best_acc - start_acc:+.2f})")
    print(f"random / raw-frame  : {chance:.2f} / {raw_acc:.2f}")
    print(f"beat raw-frame?     : {'YES' if best_acc > raw_acc else 'no'}")
    print(f"total LLM cost      : ${total:.4f}")
    outd = Path(__file__).resolve().parent
    (outd / "best_perception_B.py").write_text(best_code)
    (outd / "best_beliefs_B.txt").write_text(best_beliefs)
    print(f"final B:\n{best_beliefs}")


if __name__ == "__main__":
    asyncio.run(main())
