# Converging to Useful Code with a Weak Generator

*Context: RGB-Agent's read/grep/python-over-the-log scaffold lets Claude Opus 4.6
one-shot ft09 Level 1 and win all three preview games (1,069 actions). The same
scaffold with gemini-2.5-flash never leaves score 0 on ft09 Level 1. The scaffold
is necessary but not sufficient — the binding capability is the model's. This note
asks: can we design an **objective** under which gemini eventually generates useful
code through convergence, despite being a poor one-shot hypothesis-former?*

## Why the RGB run failed (the constraints any objective must satisfy)

From the gemini ft09 logs (`0608T150653_swarm_rgb_agent/ft09`), four failures:

1. **Genre prior instead of structure.** Opus discovered four 3×3 panels, three
   already solved, with a `$`/`#` center indicator encoding the target — recovered
   by checking the correlation *extensionally* across panels. gemini pattern-matched
   to "match-3 / block-clearing," clicked the first non-`O` cell at a border, never
   formed the panel hypothesis.
2. **Code used for trivia, not the load-bearing inference.** Same `python` tool;
   gemini used it to find "first non-O char" and diff dimensions, never to
   downsample the 2×2-doubled grid or compute the cross-panel correlation.
3. **Evidence couldn't falsify its model.** A 68-vs-64 row wobble (a *parser
   artifact* from inconsistent board-section boundaries) was misread as a game
   effect and recruited to keep block-clearing alive; an `IndexError` was waved off
   as "expected." Proposer and judge shared a failure mode → no self-correction.
   (This is the early-hypothesis-lock-in the blog names, and the belief-retraction
   problem in `stepwise_eb_learn`.)
4. **No coordinate/representation discipline.** Raw single-pixel clicks, no cell-size
   model, no x/y-convention hedge.

So an objective that drives convergence must (a) not route scoring through the
model, (b) reward local/partial progress so a weak model only has to be locally
lucky, (c) deliver its gradient as concrete signal the weak model can act on.

## Reframe: convergence is about the *selected* artifact, not the generator

Treat it as search: generator `G` (gemini) = noisy proposal distribution; objective
`J` scores; an **elitist archive** keeps the best and is fed back as context. We do
not need `G` to be good — we need the right code *reachable* across cheap draws and
`J` to keep it once drawn. Four jointly-sufficient conditions, each targeting a
failure above:

1. **`J` is model-free and environment-grounded** — a pure function of
   (code, logged trajectory buffer), no LLM in the scoring path. Breaks the
   correlated-error trap (failure 3): a grounded `J` can't be talked out of the
   evidence.
2. **Elitist archive + strict-improvement acceptance** — only replace the incumbent
   if a candidate *strictly* improves `J` on held-out data; never lose a good
   program. Converts gemini's re-derive-and-forget thrashing into monotone progress.
3. **`J` bounded and decomposable** — bounded (finite buffer, predictions in [0,1])
   so "monotone non-decreasing + bounded ⇒ converges" applies; decomposable
   (per-action operator, per-concept predicate) so the model only has to be lucky
   *locally*, not solve the game in one shot.
4. **Coverage** — the good program has nonzero probability under `G` in the round
   budget. The only condition the objective can't manufacture; addressed below.

## The objective

Score a candidate program `p` (parser, predicate, or action-operator) by grounded
forward-prediction with an abstention floor and a simplicity penalty:

```
J(p) =  Σ_{(s,a,s') ∈ D_held}  score( p(s,a), s' )   −   λ · DL(p)
```

`score` is **three-valued**: correct committed prediction > abstain (MAYBE) >
wrong committed prediction. The abstention floor is the lever for a *weak*
generator: gemini confidently asserts false things, and if right/wrong commitments
are scored symmetrically its proposal noise becomes *errors*. Make MAYBE strictly
better than a confident error and the same noise becomes *abstentions* — the
selected library stays high-precision even though the generator is low-precision.
That is what lets a bad model produce a good knowledge base: it is allowed to
not-know. (Proper-scoring-rule territory; absence-of-evidence must map to MAYBE,
never NO — the `qa_codegen` prototype's failure mode.)

## The loop: CEGIS (counterexample-guided), not invention

Decisive move for gemini specifically: **never ask it to invent from scratch** —
that's exactly failure 1/2. The RGB logs show gemini is reliably good at *repair
given a crisp signal* (every traceback led to a correct fix). So make synthesis a
counterexample-guided repair task:

1. Propose `p` (or repair the incumbent).
2. Run `p` over the whole buffer; if `J` doesn't strictly beat the incumbent on
   held-out, reject and resample.
3. Else find a specific mispredicted transition —
   `state X, action A, predicted Y, truth Z` — and feed *that* back as the next
   prompt.

Each step is a concrete grounded repair (gemini's strength), not an open hypothesis
task (its weakness), and the gradient is real environment data, not an LLM
narrative. **Convergence argument:** misprediction count is a non-negative integer;
strict-improvement acceptance makes it strictly decrease on a finite buffer ⇒ the
loop terminates; the archive guarantees best-so-far never regresses. Anti-overfit
(held-out split + DL term + abstention floor) prevents the trivial buffer-memorizer
solution. Accept a repair only on strict *global* improvement (regression-test the
whole buffer), so a repair that fixes the shown counterexample but breaks others is
rejected — hill-climb with grounded fitness + diverse resampling to escape plateaus.

## Why coverage is reachable even for gemini

Two structural facts drop the bar from "gemini must one-shot the game" to "gemini
must occasionally be locally right":

- **Decomposition** → small targets. It needn't discover the whole panel/indicator
  rule; it needs, among many draws, one predicate like "cell toggles between two
  colors on click" — reachable in a few samples. The objective then *selects* it and
  cheaply *rejects* wrong siblings ("cell disappears" = match-3) via bad prediction
  score. `J` can't stop gemini proposing match-3; it doesn't need to.
- **Shared-concept reuse** → one lucky draw *per concept*, not per use. A concept
  gemini gets right in one question's context enters the library and serves every
  question referencing it. Convergence rate is governed by per-concept success
  probability, not per-program — a large multiplier at low per-draw probability.

CEGIS counterexample-conditioning + decomposition + reuse together turn "gemini
almost never proposes the right thing" into "gemini eventually proposes each small
right thing, and the ratchet keeps it."

## What it converges *to* (the honest limit)

Selection can only choose among what's proposed ⇒ convergence is to **the best
program reachable under `G`'s support**, not necessarily the true one. For gemini:
it reliably locks in simple structurally-checkable concepts (toggle/transition
counts, conservation invariants, action-availability preconditions) and reliably
**abstains** on concepts it can't form — instead of confidently pursuing a wrong one
for 1,500 actions. Strictly better than the RGB failure: "confidently wrong forever"
becomes "honestly partial," which is exactly what belief-retraction needs.

To get true-model convergence on the hard concepts too, the only fix is widening
`G`'s support: diversify proposals (different decompositions / prompt framings so
errors decorrelate), seed candidate predicates from the question bank, and — cheapest
high-leverage — pin a handful of hand-specified structural probes as anchors (the
few-gold-labels trick that stabilizes the whole factor graph).

## First instantiation in this codebase

The three tiers already exist:

- **`perception.py`** = parser tier, scored by structural-invariant satisfaction over
  the buffer (would reject the 68-vs-64 artifact *before* any game logic runs).
- **`prototypes/qa_codegen`** = predicate verifier, but driven by CEGIS — feed
  mispredicted transitions back rather than asking for whole answers.
- **`pending_predictions`** (in `stepwise_eb_learn.py`) = pre-registration making each
  operator's prediction checkable.

**Minimal experiment to test whether convergence holds for gemini:** take the may29
ft09 buffer, run the CEGIS loop on ~20 predicates with an elitist archive, and plot
the *selected* set's held-out prediction accuracy across rounds. If the curve is
monotone non-decreasing and flattens even though individual gemini proposals are
noisy, convergence is real; where it flattens is gemini's support ceiling.

## Related

- Noisy-verifier program selection / shared-concept consistency:
  `notes/program_synthesis_noisy_verifier.{tex,pdf}`.
- VisualPredicator (NSP, ICLR 2025): operator-fit objective (cluster-and-intersect,
  classification-accuracy score Eq. 5) is the grounded `J` for the operator tier;
  failure-driven online proposal = the CEGIS counterexample source.
- `qa-codegen-prototype` memory: code-synthesis beats LLM log-reading on global
  structural invariants, fails on fuzzy geometry (re-derives concepts) and
  absence→NO — both fixed here by the parser/perception tier + the MAYBE floor.
