# Executing the probing analyses

Planning review, 2026-09-15. Companion to [the original proposal](probing_analysis_plan.md).
This document specifies proposed experiments; it does not report new probe results.
The repository and saved-run checks below were read-only. No model evaluations or
simulator replays were launched during this review.

## 1. Recommendation and the claims to aim for

Start with the 15 existing NLWM (Plain) runs and their saved ablations. Run both
proposed probes, with **prediction of a common raw-grid target as the main comparison
across checkpoints**, and **reconstruction as a measurement of recoverability under a
specified decoder**. Native feature prediction is useful as a separate diagnostic.

The questions should be distinguished explicitly:

| Question | Evidence needed | Defensible interpretation if supported |
|---|---|---|
| Does learning improve prediction? | Fixed test windows, fixed output target, same evaluator and budgets, first-working versus final incumbent | The selected artifacts improve action-conditioned prediction under this evaluation protocol. |
| What can be recovered from P(X)? | Held-out reconstruction with controlled demonstrations, decoder controls, and collision checks | Raw observations are recoverable to the measured accuracy by this decoder with this side information. |
| Does one-step performance transfer to longer horizons? | Same starts and action prefixes at every horizon; direct and recursive prediction distinguished | One-step performance is associated with, or improvements persist into, longer-horizon prediction. |
| Do the training objectives matter? | Evaluate the existing matched -FD/-ID training arms on common targets; quantify uncertainty | These training ablations change performance in the evaluated runs. |
| Does the representation help planning? | Downstream planning under controlled feature/model interventions or training ablations | The tested intervention changes planning performance. Probe correlation alone is insufficient. |
| Does P discard irrelevant information while retaining useful information? | Controlled nuisance variations, retained task predicates, and prediction/planning tests | Evidence for selective abstraction over the tested variations. Raw reconstruction alone cannot establish this. |

High prediction and high reconstruction can be a good result: P may make information
easier for the LM to use while retaining the grid. Low reconstruction is not inherently
desirable, and low reconstruction with one LM does not establish information loss.

## 2. What already exists, and what the audit changes

### Saved artifacts and data

- Reference: `logs/2026-08-24/human_curated/rexpure/<game>_s1`.
  All 15 runs have 30 candidates, process logs, and a shipped P/K pair matching exactly
  one saved candidate. Process-log lengths range from 33 to 51 entries. The shipped
  candidate is often earlier than candidate 29; do not equate the last proposal with
  the selected model.
- Each dataset manifest reports 60 train and 50 test transition targets. The test
  sources comprise 3–5 drives but only **3–4 distinct users per game**, with no
  train/test user overlap within any of the 15 manifests checked.
- Full test drives exist. Counting rows gives 464–896 possible horizon-8 starts per
  game with nine preceding steps, before checking continuity, terminal flags, or
  replay validity. These are overlapping windows, not independent observations.
- All four ablation directories under `logs/2026-09-09/ablations/`
  (`nofd`, `noid`, `noperc`, `nobeliefs`) have 15 perception artifacts and 30-node
  candidate logs per run. Their saved train/test paths, context paths, context length,
  pool sizes, and seed match the corresponding reference commands. This establishes
  availability and matching flags, not a full content/provenance audit. Older
  statements in [the ablation plan](ablations-plan.md) that nothing was launched are stale.

### Reusable implementation

| Existing code | Reuse | Required change or qualification |
|---|---|---|
| [`eval_heldout_cfd.py`](../offline_learning/scripts/eval_heldout_cfd.py), `rexpure_optimize.build_data` | Reconstruct each run's data from its launch arguments | Add observation/context/source content hashes; the old training fingerprint hashes only length and action sequence. |
| [`perception_metrics.py`](../offline_learning/scripts/perception_metrics.py) | Candidate inventory, frame cache, P execution, shipped-artifact joins | Its cached pairs are not a contiguous trajectory dataset. Preserve source positions and actions separately. |
| [`eval_multistep_fd_plan.py`](../offline_learning/scripts/eval_multistep_fd_plan.py) | Action-conditioned endpoint prompts, replay checks, per-window traces | Replace legacy test50 source resolution with the human-drive manifests; separate FD from planning calls; replace horizon-specific sampling. |
| [`eval_forward_modes.py`](../offline_learning/scripts/eval_forward_modes.py) | Raw-grid serialization and simple prediction scaffolding | Existing modes are X-to-X and P-to-P; **P-to-X is new**. Its split logic targets older GEPA runs. |
| [`invdyn_core.py`](../offline_learning/invdyn_core.py) | Forward-only history formatting; cFD evaluation | `build_window` also contains future states for ID. Only the forward view may enter a forecast prompt. |
| [`field_checks.py`](../analysis/wm_quant/inquiry/semantic_changes/field_checks.py) | Examples of collision and deterministic reconstruction checks | These are selected-case audits with stated assumptions, not a universal inverse of P. |

Two important protocol corrections:

1. The reference commands use `--fd-scorer none --contrastive-fd --cfd-hard-decoys`:
   the forward training term is **contrastive next-state selection**, not generative
   exact-match prediction. Free-form and multi-step generation are transfer tests.
   Report training cFD and generative FD under different labels. The manuscript's
   current generative-objective equations need reconciliation with the actual runs.
2. `rexpure` uses train == selection/scoring set, with no separate validation split.
   The 50 external test targets are distinct. Also, `build_data` constructs the ID
   action pool using train and test actions. Audit/disclose this label-vocabulary
   sharing; it is different from exposing test target frames. A claim of completely
   test-independent training needs a documented legal-action universe or train-only
   vocabulary, or a new evaluation set uninvolved in that construction.

## 3. Freeze the experimental definition

Use **K** for learned dynamics/world knowledge, matching the manuscript, and reserve
**D** for the dataset. At checkpoint c, define Z_t = P_c(X_t). Here X is the observed
grid, not the simulator's complete hidden state.

For a history length k and horizon h, the forecast input is:

`(Z[t-k:t], past actions a[t-k:t-1], future actions a[t:t+h-1], K_c)`.

Future actions are supplied; future observations are withheld. This is conditional
prediction, not action selection. Use k=9 initially to match training, with k=0 as a
diagnostic. Execute P afresh on each individual frame through `run_perceive`, matching
the learner. A history-aware P runtime would be a separate intervention.
Check output determinism across fresh Python processes, pin the Python environment and
`PYTHONHASHSEED`, and record them in the cache. Some saved programs iterate sets or
emit hashes. The same raw frame must have the same target bytes throughout an exact
comparison; a newly pinned runtime is an evaluation convention, not proof of matching
an undocumented historical runtime.

**Raw target:** a parsed rectangular grid, with fixed row/column and colour conventions.
Exclude recorder headers, absolute step, phase, user/session identity, and timestamps.
Match the training metadata-stripping pipeline before running P. Do not supply a
target-specific background colour or other missing information by inspecting held-out X.
Any fixed dimensions, vocabulary, or encoding documentation supplied to the decoder
must be recorded as side information and provided consistently.

### Forward modes

| Mode | Input and output | Purpose |
|---|---|---|
| Native direct | P-history + K + h actions -> predicted Z[t+h] | Predictability in the candidate's own language. |
| Raw-target direct | P-history + K + h actions -> predicted X[t+h] | Main comparison across learning checkpoints and ablations. |
| Raw baseline | Raw history + h actions -> predicted X[t+h] | Controls the LM's inference from observations without learned artifacts. |
| Recursive | Repeated one-step predictions, feeding back predicted states only | Measures accumulation of rollout error. |
| Teacher-forced diagnostic | One-step predictions with true history restored at each step | Separates local prediction failures from recursive accumulation. |

Start with the first three modes. Add recursion after the pilot establishes reliable
output parsing. A single request for the whole future trajectory is another mode, not
equivalent to recursive calls. Record it separately if used.

For recursive Z prediction, shift the history and append predicted Z; never re-encode
a true future frame. For raw-state recursion, either append raw predictions to a raw
predictor or run P on predicted grids for the P-input variant; name the variant.
An invalid prediction terminates that rollout, and remaining horizons fail. A rollout
to h=8 can yield the h=1,2,4,8 measurements in eight calls, not fifteen.
Report call and token budgets for direct versus recursive modes; extra inference
compute can affect their difference. A compute-matched sensitivity run is needed if
attributing that difference specifically to the rollout procedure.

For P-to-X, supply a small fixed set of **training-only paired examples (P_c(X), X)**
to explain the representation. Reuse the same raw example identities across checkpoints
and render them under each P. Use the same mapping support for reconstruction. The raw
baseline receives the corresponding raw-to-raw examples and the same task history.
Include a baseline with access to the same training transition examples if claiming an
advantage over learning the dynamics in context. Equal output targets alone do not
equalize training information. Keep this comparison distinct from the simpler raw baseline.

### Metrics

- **Raw grid:** exact equality after JSON parsing/canonicalization, cell accuracy,
  and cell-change precision/recall/F1 relative to the start grid. For change-F1, a
  change is `(row, column, new_colour)`; wrong colour or position must not match.
  Two empty change sets score 1; only one empty set scores 0. Report unchanged and
  changed-target strata, defined on X and shared by all candidates.
- **Reconstruction:** parsed grid exact match and cell accuracy, plus non-background
  cell precision/recall/F1 under a frozen scorer definition. A background-only guess
  can obtain high cell accuracy, so cell accuracy alone is inadequate.
- **Native Z:** exact match after trimming outer whitespace, plus copy-Z baseline. Report
  output length, collisions, and failures beside it. Different P's define different
  tasks, so native exact cannot be read as an improvement in common state accuracy.
  Any semantic parser or field-level score must be developed on training/development
  outputs, versioned, and shown separately from literal exact match.
- Save parse failure, P execution failure, empty output, and truncation rates.
  Model-generated malformed answers count as failures, not discarded trials. Provider
  failures follow a fixed retry rule and remain visible as missing coverage if unresolved.

Include **copy-current-X** as a raw forecast baseline. On a predeclared action-sensitive
subset, also remove the future actions or shuffle their order in the prompt to measure
whether forecasts use the specified actions. Keep the original targets fixed for this
diagnostic and distinguish it from replaying a different action sequence. Passive or
order-invariant examples need not get worse under these controls.

The existing textdiff score measures textual edits and has documented permutation
failures ([audit](../offline_learning/audit_forward_findings.md)); use grid-level
scoring for the common target. Do not silently replace it with token overlap, which
can confuse different coordinate/colour assignments.

Native exact has a particularly relevant failure mode here: several learned P's emit
hashes/checksums. Predicting a checksum in text can be difficult even when the grid is
correct. Preserve the literal score; diagnose with a predeclared, train-developed
parser or deterministic re-encoding of predicted X through P. Do not remove nuisance
fields only after seeing which removal improves the result.

## 4. Data: reuse the human trajectories, rebuild the evaluation windows

**Answer to Q1 in the draft:** the original simulator data need not be changed, but
the two-row target slices are insufficient for multi-step evaluation. Recover
contiguous windows from their full source drives, carrying every intervening action.
Never concatenate independently selected transitions to manufacture a trajectory.

1. Resolve each selected slice to `(game, user, session/reset segment, drive, step)`
   using the manifest and original rows before stripping metadata. Require an
   unambiguous mapping. Preserve click coordinates and no-op ticks.
2. Verify chronology, action/frame alignment, reset/terminal boundaries, program
   identity, seeds and simulator replay. Keep observations from the recorded replay;
   do not silently replace mismatches with newly generated frames.
3. Build train exposure sets covering targets, **past and future ID context**, cFD
   decoys, reflection examples, and other data actually shown during search. Audit
   overlap with every test window's context and targets, including canonical-grid
   duplicates that metadata differences would conceal.
4. Freeze a forecast window manifest independent of P, K, and probe scores. Suggested
   starting budget: 50 starts per game, spread across held-out users/drives, requiring
   k=9 history and h=8 continuation. Reuse every start at h=1,2,4,8. Publish exclusions
   and available coverage; shorter-horizon extra windows can be a separate analysis.
5. Preserve an all-valid-window sample and report activity strata. The old planning
   sampler removes static and noop-solvable windows, which removes legitimate passive
   dynamics from an FD study. If retaining the curated informative targets as anchors,
   label that sampling distribution and report a uniformly sampled eligible-window
   sensitivity analysis. Neither manually selected drives nor informative targets
   represent unfiltered human play.
6. Keep a state-level reconstruction manifest with fixed frame identities, provenance,
   and demonstration IDs. Start with up to 100 query time points per game; additionally
   report unique-state and train-unseen-state results so repeated boards do not make
   reconstruction look like generalization.

For horizon plots, changing h should not also change the start-state distribution.
Use the same windows across candidates, arms, and models. Keep a non-overlapping-window
sensitivity analysis, and retain the full user/drive structure for uncertainty estimates.

### What “held out” can mean

The existing external test users are suitable for a retrospective artifact analysis.
They are not automatically a fresh confirmation set: these datasets and related
results have already supported curation and project decisions. Record prior use.
Develop prompts/parsers on training or a separate development partition; freeze the
protocol before the final pass. For stronger confirmation, reserve previously unused
users/sessions, excluding all earlier exposure, or collect independently generated
trajectories under a declared policy. New windows from old users are not new users.

### Stochasticity and hidden state

Audit this per game. A reproducible seeded replay does not mean the next frame is
uniquely determined by the visible history. Ants is already a documented stochastic
case; the human-replay methodology also notes uncertainty in reproducing the original
human study's random outcomes.

Report deterministic and stochastic/aliased cases separately. Exact match to one
realized stochastic future is a valid realized-outcome score, not an observation-only
oracle ceiling. If making distributional claims, use repeated continuations of the
same conditioned start/hidden-state setup with controlled random draws and a defined
distributional or event-level score. Arbitrarily changing the initial seed can change
the starting state and is not that experiment. A simulator with inaccessible RNG/hidden
state is a privileged reference, not a fair LM baseline.

For a controlled stochastic forecast study, choose open-loop action sequences before
the future randomness. Logged human future actions can themselves reveal reactions to
future events. Label the original task as prediction conditioned on recorded actions.

## 5. Reconstruction: disentangle loss from decoder difficulty

Primary probe: `R(P_c(X); E_c) -> X`, where E_c contains training-only example pairs.
No K, preceding/future frames, query raw grid, or query-specific source information.
This measures P under a fixed decoder and example budget. A version with K or history
answers a different question and should be reported separately.

Use a fixed evaluator, prompt, temperature/reasoning budget, and primary example count
(provisional m=8). Choose demonstration identities without looking at test answers.
Use nested m=0,2,8 sets and multiple example orders on a smaller sensitivity subset.
Report context-token consumption and truncation; if m=8 does not fit all arms, choose
a common supported budget on development data instead of silently dropping examples.

Necessary controls:

- **Raw identity and simple lossless grid encoding:** measure output-format difficulty
  and distinguish useful formatting from recovery of genuinely missing information.
- **Constant/empty input and background/mode-state guesses:** measure prior knowledge
  and the benefit of easy, repetitive boards. A valid constant P can score perfectly
  on its own native FD task; show this degeneracy explicitly.
- **Shuffled query representations:** preserve format but break the query-target pairing;
  test whether successful reconstruction actually uses the query representation.
- **Hash-only input:** distinguishes unique state identifiers from readily decodable
  spatial content; uniqueness is not evidence of accessible grid information.
- **Simple deterministic inverse where available:** freeze a parser from P's code and
  training outputs; verify it on held-out states. State assumptions about dimensions,
  background and colour dictionaries. It is a recoverability audit, not the same LM probe.

For fixed side information, group equal Z values and inspect whether their raw targets
differ. The empirical best deterministic reconstruction accuracy is

`C_emp = (1/N) * sum_z max_x count(Z=z, X=x)`.

This is a finite-sample collision bound, not a population guarantee. Observed collisions
prove ambiguity for those inputs; no collisions on a small sample do not prove lossless
encoding. A hash can have no sample collisions while being unusable by the decoder.

An optional diagnostic composes native prediction with reconstruction:
`predicted Z[t+h] -> R -> predicted X[t+h]`, compared with `true Z[t+h] -> R`.
This helps locate failures, but the score gap is not a clean additive dynamics error:
predicted feature strings can be outside the decoder's training distribution. Likewise,
reconstruction accuracy is not a universal ceiling on forward accuracy, and dividing
one by the other has no justified interpretation.

Call the measurement **reconstruction accuracy** or **recoverability**, not information
content or mutual information. Probe capacity and controls affect its interpretation;
this is the central concern of [Hewitt and Liang (2019)](https://aclanthology.org/D19-1275/).
If an information-theoretic claim is essential, a separately specified conditional
codelength/MDL study with decoder cost and held-out likelihoods is more appropriate;
few-shot accuracy and gzip size are not substitutes for that study
([Voita and Titov, 2020](https://aclanthology.org/2020.emnlp-main.14/)).

## 6. Checkpoints, controls, statistics, and figures

### Checkpoint policy

Freeze up to five checkpoints per reference run: first working incumbent, and the
incumbents at 25%, 50%, 75%, and 100% of the recorded search iterations. Determine
“working” from training-frame execution and non-collapse, not test accuracy. If several
milestones select the same P/K, evaluate it once and reuse the result. Deduplicate
reconstruction by P alone. Record actual iteration and candidate IDs, tie-breaking,
artifact hashes, and the shipped-pair match; show blank/failed initial candidates separately.

Incumbent means best under the **original training selection rule** at that point.
Do not pick the highest probing score. Use iteration counts within each game; when
comparing training arms, also report actual objective-call/token budgets, since equal
node counts do not guarantee equal compute. A learning curve need not be monotonic
on held-out probes even if its training objective is monotonic.

### Comparison order

1. First-working versus final P/K on the same windows.
2. Final P/K versus raw and fixed lossless-encoding baselines under matched evaluation.
3. Final artifacts from all four saved training ablations, on the same raw targets.
4. A secondary evaluator and context/example-budget sensitivities on preselected cases.
5. Optional feature-removal experiments for specific semantic claims, e.g. derived
   spatial summaries or hashes, preserving the rest of the input and including a
   comparable formatting/token-removal control.

Removing K at evaluation time tests reliance on K; training without K tests a different
question. Likewise, crossing early P with final K can introduce incompatible vocabulary.
Treat such cross-pairs as a compatibility diagnostic unless the interfaces are aligned.

### Statistical unit and primary contrasts

Provisional primary prediction contrast: final minus first-working incumbent in macro
cell-change F1 at h=4. Report raw exact match and the full h=1,2,4,8 curves as secondary
outcomes. The main reconstruction contrast is final minus first-working grid exact
match at fixed m. Freeze these choices after the development pilot, before confirmation.
A paper claiming *exact state prediction* should make exact match its primary endpoint.

Compute paired differences per game on common queries; give each of the 15 games equal
weight in the headline macro average. Include per-game values and absolute baseline
scores. Keep unchanged-target results visible; a large average driven by passive copying
is a different finding from better action-dependent changes.

For the fixed benchmark, bootstrap user clusters (and drives/blocks within them where
supported) with all arm/checkpoint predictions paired. Preserve repeated-user identities
across games when they recur; they are not necessarily nested independent clusters.
A separate resampling of games addresses variation over the chosen environments, not
random sampling of all possible games. With only 3–4 users/game, per-game intervals
are fragile; show user-level results and leave-one-user-out sensitivity. Hundreds of
windows do not repair that limitation.

The current one-training-seed-per-game design supports conclusions about these saved
runs. To claim robust behavior of the learning algorithm, add independent training
seeds across the full planned suite and arms: start with 3, prefer 5 or more if feasible,
and set the final count using pilot variance and a meaningful effect/interval target.
Decoder repeats and additional windows are not substitutes for training seeds. This
separation between point estimates and run uncertainty follows the evaluation concerns
in [Agarwal et al. (2021)](https://arxiv.org/abs/2108.13264).

Predeclare a small family of hypothesis tests and adjust for multiple primary contrasts
if testing several arms. Treat remaining plots/subgroups as exploratory. Failed and
missing arms must be shown with denominators, not silently removed from macro averages.

The current manuscript table displays 0.74 for full NLWM and 0.73 for -FD. Those
rounded aggregates alone do not establish the importance of FD. Inspect paired task
outcomes and uncertainty, and allow the conclusion to be a small or unresolved effect.

### Figure set

1. **Prediction versus horizon:** common raw-target score, paired first/final curves,
   raw/lossless baselines, per-game panels, uncertainty, and separate activity strata.
2. **Reconstruction versus learning:** fixed queries and example budget; exact and
   foreground-sensitive scores, with validity/parse coverage and output length alongside.
3. **FD-h versus FD-1:** one panel per h>1; points identify game and checkpoint, with
   within-game trajectories. Use the common raw metric. Correlations pooled over all
   candidates confound game difficulty and repeated checkpoints; use within-game
   summaries and clustered uncertainty. A disjoint-query sensitivity check can assess
   shared evaluation noise. `FD-h = FD-1^h` is not an identity without strong assumptions.
4. **Prediction versus reconstruction:** distinguish useful lossless re-encoding,
   potentially useful lossy abstraction, and decoder/dynamics failures.

An association with existing planning scores is exploratory, particularly with only
15 games and different planning interfaces. Behavioral/causal claims require controlled
downstream interventions, consistent with the distinction made by
[Elazar et al. (2021)](https://arxiv.org/abs/2006.00995).

## 7. Work packages and estimated evaluation size

The direct-prediction and reconstruction pipeline is implemented. See
[the implementation guide](probing_implementation.md) for commands, controls, and
the current scope. The stages below also include follow-up experiments and evidence
requirements that are not satisfied by implementing the evaluator alone.

| Stage | Deliverable | Acceptance condition |
|---|---|---|
| A. Inventory and data audit | `build_probe_manifest.py`; run/checkpoint/window/example manifests and source hashes | Unambiguous artifact and source joins; recorded exposure/overlap audit; eligible horizons and independent-unit counts known. |
| B. Evaluator core | `probing_common.py`; raw scorers, forward-only prompts, isolated P execution, cache keys, trace schema | Meaningful tests for action alignment, no future leakage, resets, canonical equality, moved-cell penalties, failure accounting, and recursive feedback. |
| C. Development pilot | `eval_probing.py` on bt3gb, dq8gc, eahcw, s2kt7: two checkpoints, 10 starts, h=1,4; 20 reconstruction queries | Prompts/formatting and baselines work; stochasticity and checksum issues characterized; measured token, call, and latency budget available. |
| D. Freeze protocol | Versioned configuration and explicit primary contrasts | Data/model/example/checkpoint choices fixed before confirmation; remaining limitations written down. |
| E. Reference analysis | All 15 games, up to five checkpoints, h=1,2,4,8; fixed reconstruction queries | Complete traces and coverage; baselines evaluated on exactly the same queries. |
| F. Ablations and robustness | Four existing final ablation artifacts/game; secondary decoder and selected sensitivities | Same targets and split manifest; method-specific information and compute differences disclosed. |
| G. Paper evidence | `report_probing.py`; per-item data, tables, figures, and claim/evidence summary | Every reported number regenerates from frozen records; unsupported stronger wording narrowed. |

Trace records should contain game/user pseudonym/drive/start, raw-target hash, past/future
action hashes, P/K hashes, iteration, model/provider/version and decoding settings,
demonstration IDs, prompt/response, parsing/errors/retries, metrics, usage and latency.
Cache keys must include all of these evaluation-defining settings. A checkpoint name
or horizon list alone is insufficient to validate a resumed result.

Illustrative call counts, **not a measured dollar or wall-time estimate**:

- Pilot direct forecasts: `4 games * 2 checkpoints * 10 starts * 2 horizons * 2 targets
  = 320` calls; raw baseline 80; primary reconstruction 160. Controls/repeats are extra.
- Reference direct forecasts: `15 * 5 * 50 * 4 * 2 = 30,000` calls before deduplication.
  Raw baseline adds 3,000. `15 * 5 * 100 = 7,500` primary reconstruction calls.
- Four final ablations on raw-target direct FD: `4 * 15 * 50 * 4 = 12,000` calls.
- Adding recursive rollout to h=8 costs eight calls per start/checkpoint/representation.
  Full checkpoint curves for every ablation would multiply the budget substantially.

Price using actual pilot input/output token counts, provider rates, retries, and measured
throughput. Keep the main reconstruction budget to one example count and decoder;
run example-count/order and evaluator sensitivities on a frozen subset first.

## 8. Decisions to settle before implementation or a full run

| Decision | Recommended starting choice | What would change it |
|---|---|---|
| Primary claim | Better predictive representations under a specified LM; measured recoverability | A stronger information-loss or causal planning claim requires additional experiments above. |
| Scope | Existing 15 reference runs plus final saved ablations | General claims about the learner require training-seed replication. |
| Evaluation population | Existing user-disjoint drives for retrospective analysis; fresh users for confirmation | If new users are unavailable, explicitly retain the retrospective scope. |
| Raw state definition | Observable grid; recorder metadata excluded | Hidden-state recovery is a distinct task requiring labels and observability analysis. |
| Main target/metric | Common X; h=4 change-F1 primary, exact secondary | Exact-state claims should prioritize exact; task sufficiency also needs fixed task predicates. |
| History and horizon | k=9, h=1,2,4,8, shared eligible starts | Increase h only after checking boundary coverage, stochasticity and output budgets. |
| Rollout regime | Direct endpoint first; recursive and teacher-forced diagnostics second | Error-accumulation claims require recursive evaluation. |
| Probe decoder | Freeze the reference evaluation configuration first; second model as robustness | The current endpoint must be available and support all prompts; otherwise document a replacement. |
| Reconstruction support | No K/history; m=8 training pairs, with smaller nested budgets in sensitivity tests | Context limits or a focus on joint P/K recoverability require an explicitly different protocol. |
| Checkpoints | Training-selected incumbents at frozen milestones | All-candidate curves are useful diagnostics but cost more and remain statistically dependent. |
| Stochastic evaluation | Separate realized-outcome results and fixed task/event scores | Distributional claims require controlled repeated continuations and appropriate scoring. |
| Independence and precision | Paired user-cluster analysis; disclose one training seed | Wider claims need more independent users/seeds, not merely more windows. |

The first implementation step is Stage A: establish the exact evaluation population
and frozen artifact/window manifest. The most consequential scientific choice is
whether the paper aims to show **information retained**, **information accessible to
an LM**, or **information used for successful prediction/planning**. The two proposed
probes directly address the second and part of the third; they need the controls and
additional interventions above to support the other interpretations.
