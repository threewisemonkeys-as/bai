# Using "Theory Generation" to Drive Action Selection

Notes on replacing / augmenting the current question-generation + experiment-selection
machinery in `stepwise_eb_learn.py` with a "theory generation" mechanism for choosing
the next action in the environment.

Context: a one-shot "brainstorm 5 ranked theories" prompt (see
`scripts/simulate_theories.py`) surfaces the *correct* game mechanic as its top-ranked
theory at steps 5/9/12 of the gemini-2.5-flash ARC run — well before the real run
committed to it. The real run generated that mechanic as a *question* at step 9 but its
B-difference score never won selection, so it was never tested. This motivates using
full competing world-models (theories) as the unit of exploration.

---

## How the current mechanism works (what we'd be replacing)

The current "unit of exploration intent" is a **binary question**. Pipeline:

1. Agent acts under a standing **experiment** (an action plan).
2. After each action, `identify_critical_transition` (LLM judge, ~line 1840) fires
   "critical" if the transition was *surprising given beliefs* **or** the experiment is
   *stale*.
3. On critical: improve beliefs -> generate new questions -> dedup/trim the question bank
   -> score unanswered questions by **B-difference** (`question_scoring.py`: how much
   resolving q_i flips the predicted answers to the *other* questions — a value-of-
   information proxy over the question graph) -> select top-k ->
   `formulate_experiment_from_question` turns it into an action plan.

So exploration is driven by *"which question, if answered, most reshuffles my answers to
the other questions."*

Failure mode observed: the correct mechanic was a high-ranked **theory** at step 9, but as
a *question* it never won the B-diff ranking, so it was never tested.

Theory generation gives us something the question bank doesn't: explicit, competing, full
world-models with predictions. That unlocks the classic active-learning move — pick the
action where the candidate models **most disagree**.

---

## Plan A — Bayesian experiment design over theories (discriminating action)

The most direct translation. Theories become the thing you're trying to discriminate,
replacing the question bank entirely.

- **State:** maintain a ranked ensemble of N theories with probabilities `p(T_i)` (a
  posterior over world-models). The confirmed/`<world_knowledge>` facts become shared
  *axioms* every theory must respect; theories differ only on the unknown mechanics +
  win condition.
- **Action selection:** prompt the LLM: *"For each theory, predict the observable outcome
  of candidate actions; propose the single action (or short plan) whose predicted
  outcomes differ most across the high-probability theories."* Pick that maximally-
  discriminating action. This is expected-information-gain / max-disagreement selection.
- **Update:** after observing the real outcome, down-weight theories whose prediction was
  violated, renormalize, drop dead ones, and regenerate to refill to N. Sequential
  Bayesian model selection.
- **Replaces:** question generation, B-diff scoring, and `formulate_experiment_from_question`
  all collapse into "generate theories -> choose discriminating action -> reweight."
- **Why it'd have helped at step 9:** the target-pattern theory and the all-perimeter-red
  theory predict different things when you set a grid to match its center — the
  discriminator would force that exact test instead of letting the agent fixate.
- **Cost/risk:** generating N theories + per-action predictions every step is expensive;
  mitigate by gating regeneration behind critical states (see cross-cutting notes).
  Theories can collapse to rephrasings -> reuse the existing dedup/diversity machinery.

## Plan B — Theory-seeded VOI (minimal change, keep the question rails)

Keep almost all current plumbing; only change *where questions come from* and *how
they're scored*.

- Generate theories, then extract the **crux questions** — the binary questions on which
  the top theories most disagree (one per contradicting pair/cluster).
- Score a question by **expected reduction in entropy over the theory posterior** (how
  much its YES/NO answer splits theory probability mass), instead of B-difference over the
  question bank.
- Feed the winning crux question into the existing `formulate_experiment_from_question`
  -> unchanged downstream.
- **Pro:** smallest diff, reuses experiment/critical machinery, directly fixes the "right
  idea never selected" problem because the scoring now rewards questions that separate
  full models.
- **Con:** still bottlenecked through a single binary question per step; loses the richer
  "predict-and-compare" signal of Plan A.

## Plan C — Theories as the belief representation + Thompson sampling

Replace the single `<world_knowledge>` block with an **ensemble** of K weighted theories
(a particle filter over world-models), and skip explicit VOI entirely.

- **Action selection:** each step, **sample one theory proportional to its probability**,
  then ask the agent to take the action that best *makes progress assuming that theory is
  true*. Thompson sampling gives exploration–exploitation balance for free: likely
  theories are exploited often, but minority theories occasionally get acted on (and
  thereby tested).
- **Update:** reweight theories by how well they predicted the transition;
  resample/regenerate particles periodically.
- **Pro:** no separate discriminating-action prompt, naturally interleaves "play to win"
  with "explore"; elegant.
- **Con:** less sample-efficient at *deliberately* killing wrong theories than Plan A's
  max-disagreement; needs a sane prior over K.

## Plan D — Two-level: theories decide *what to test*, experiments decide *how*

A hybrid that preserves the experiment/critical abstraction already trusted.

- At critical states, generate theories. Take the **top-2 high-probability but mutually
  contradictory** theories and ask for a **discriminating experiment** (the action
  sequence whose outcome differs between exactly those two).
- Hand that to the existing experiment-execution + critical-transition loop. The critical
  judge's "is the experiment stale?" criterion becomes "has the discriminator resolved
  which of the two theories holds?"
- **Pro:** keeps multi-step experiments (not just one action), reuses the standing-
  experiment + staleness machinery; theories supply *targets*, experiments supply
  *procedure*.
- **Con:** only discriminates two theories at a time.

## Plan E — Active-inference / free-energy loop (predictions do double duty)

Use the theories' explicit predictions to *also* replace the LLM critical-judge.

- Every step the MAP theory pre-registers a prediction; **prediction error = the surprise
  signal**, so `identify_critical_transition` collapses into "did the top theory
  mispredict?" (cheaper, principled).
- Action selection minimizes **expected free energy = pragmatic value (game progress) +
  epistemic value (expected info-gain over theories)** — one scalar trades off "win" vs
  "learn."
- **Pro:** unifies surprise detection, exploration, and exploitation under one objective;
  removes a separate LLM call.
- **Con:** most engineering; needs a calibrated way to turn LLM judgments into the two
  value terms.

---

## Cross-cutting design decisions (apply to all plans)

- **How to update the theory posterior.** Cleanest is **pre-registered predictions**:
  before acting, each surviving theory states its expected outcome; after observing, an
  LLM (or exact image/text diff) marks each consistent/violated -> multiplicative weight
  update. More honest than post-hoc "which theory fits."
- **Cost gating.** Don't regenerate the full theory set every step. Regenerate only on
  surprise (a violated MAP prediction); between surprises just act under the current
  ensemble. Keeps per-step cost near today's.
- **Anchoring & diversity.** Force theories to be consistent with default knowledge +
  *confirmed* facts (shared axioms), and run dedup so the N theories are genuinely
  distinct hypotheses, not paraphrases — reuse the existing question-dedup code.
- **Granularity.** Decide whether the discriminating "experiment" is a single action
  (Plan A/C) or a multi-step plan (Plan D) — multi-step is needed for mechanics that only
  reveal themselves after a setup sequence (e.g., "match the center then click to
  confirm").
- **Evaluation.** Reuse the eval harness; the clean A/B is theory-driven selection vs.
  current question/B-diff selection on the same ARC games, measuring levels completed and
  steps-to-first-correct-theory. The step-9 case is a ready-made unit test: does the new
  selector actually *test* the target-pattern theory?

---

## Recommended sequencing

1. Prototype **Plan B** first (smallest diff, reuses the rails, directly fixes the
   selection bottleneck diagnosed at step 9).
2. If entropy-over-theories beats B-diff, graduate to **Plan A or D** to capture the full
   predict-and-discriminate loop.
3. **Plan C (Thompson)** is the best bet if exploration and *winning* should be tied
   together rather than run as a pure-exploration phase.

---

## Relevant code integration points

- Trigger: `identify_critical_transition` (`stepwise_eb_learn.py` ~line 1840).
- Selection: `select_qa_pairs_and_formulate_experiments` (~line 2295) and the B-diff
  scorer in `question_scoring.py`.
- Experiment -> action: `formulate_experiment_from_question`
  (`stepwise_eb_learn_improve.py`); the experiment plan is passed as `current_experiment`
  / `current_experiment_question` and injected into the agent rollout.
- Beliefs: single `<world_knowledge>` block today; theory plans make this an ensemble.
- LLM calls: route through `mixed_improve._llm_call` for mock-mode / logging / cost
  parity (as `scripts/simulate_theories.py` already does).
