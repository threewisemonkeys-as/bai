# NLWM ablation plan — objectives (−FD, −ID) and representations (−Perception, −Beliefs)

Status: **code landed 2026-09-06, nothing launched.** Written against the repo as of `28ad918`.

Implemented since the first draft of this plan (§2 and §4.4 are now descriptions of code, not
proposals):

| what | where |
|---|---|
| `--no-id` — drops ID from the composite, from the reflective dataset, from the diagnosis calls, and from the proposer's task statement | `invdyn_core.py`, `rexpure_optimize.py` |
| `--no-beliefs` — `world_knowledge` dropped from the candidate, selector pinned to perception | `rexpure_optimize.py` |
| held-out cFD — `eval_cfd_on` + `bake_test_decoys`, in-run via `--cfd-test` and post hoc via a standalone scorer | `invdyn_core.py`, `offline_learning/scripts/eval_heldout_cfd.py` |
| `--ablation {nofd,noid,noperc,nobeliefs}` on the reference-command rebuild | `offline_learning/launch/launch_human_origin.py` |
| tests, including the split-invariance claim | `tests/test_ablation_flags.py` (32 tests) |

Verified: `--ablation none` reproduces the shipped reference command byte-for-byte; each arm's
diff against it is exactly its one documented delta; all four arms rebuild bit-identical
train/test splits (fingerprint-checked against the reference run's own checkpoint); the
held-out scorer runs end-to-end on a real artifact (colour_lines NLWM: raw 0.42, perceived
0.56, chance 0.25, $0.04); full suite 166 passed.

Goal: fill the four em-dashes in `paper/main.tex::tab:ablations` with numbers that are
directly comparable to the NLWM column of `tab:autumn-results` (macro 0.62, 86 problems,
online, NL goals, per-problem caps).

The comparability requirement dictates the whole design: **each ablation arm is the
reference run with exactly one flag delta, evaluated by the exact same command as NLWM,
against the exact same problems file.** Everything below is organised around making that
delta auditable.

---

## 0. The reference the ablations are measured against

Two artifacts define "NLWM":

| phase | artifact |
|---|---|
| learning | `logs/2026-08-24/human_curated/rexpure/<game>_s1` (15 games, seed 1, `informative_curated` pools) |
| planning | `logs/2026-09-03/planning_v2_online_ds_percap_nl` (86 problems, NL goals, per-problem caps, deepseek planner) |

The learning command is byte-identical across all 15 games except `--run/--context-source-run/
--test-run/--test-context-source-run/--out-dir/--actions` (verified: the `--`-flag list is
literally the same string for eahcw, dino and logic_gates). Reconstructed from
`logs/2026-08-24/human_curated/rexpure/bt3gb_s1/launch.json`:

```
rexpure_optimize.py
  --run/--context-source-run/--test-run/--test-context-source-run   (informative_curated)
  --actions left,right,up,down,noop,click        --out-dir <...>
  --test-n 50 --k-choices 5 --context-k 9 --seed 1 --max-nodes 30 --train-n 60
  --concurrency 12 ... --concurrency 4           (argparse keeps the last: 4)
  --fd-scorer none
  --task-model openai/gpt-oss-20b --task-provider-order groq
  --task-reasoning-json '{"effort": "low"}'
  --reflection-model deepseek/deepseek-v4-flash
  --reflection-provider-order deepseek,baidu,fireworks
  --reflection-hedge-delay 120 --reflection-timeout 300
  --analyze-mistakes --no-analysis-memo
  --start-perception offline_learning/autumn_seed_perception.py
  --id-set-loss --id-eps 0.1
  --composite min --contrastive-fd --cfd-hard-decoys
  --rex-c 5 --propose-batch 3
```
with `OPENROUTER_PROVIDER_ORDER=groq` in the environment.

### FINDING 1 — the plan's objective-ablation labels are stale

`experimental_plan.md` says:

> **[X]** No FD: `--fd-scorer none` (composite reduces to ID).

That is wrong, and it is marked done because it describes the *reference*, not an ablation.
The shipped run already passes `--fd-scorer none`; the forward term in the shipped objective
is the **contrastive** FD, supplied by `--contrastive-fd --cfd-hard-decoys`. Reading
`invdyn_core.py:1631-1645`:

```python
elif self.composite in ("min", "softmin"):
    terms = [id_score]
    if self.fd_weight > 0.0:        # False: --fd-scorer none sets fd_weight = 0
        terms.append(fd_score)
    if cfd_score is not None:       # True: --contrastive-fd
        terms.append(cfd_score)
    score = min(terms)
```

so the live objective is `min(ID_set, cFD_hard)` — two terms, matching the prose in the
Ablations section ("The shipped objective is min(ID, cFD-hard)") but not the flag it names.

**Therefore −FD = drop `--contrastive-fd --cfd-hard-decoys`** (leaving `--fd-scorer none`
in place), which reduces `terms` to `[id_score]` and `min([id]) == id`. **No code change
is required for −FD.**

### FINDING 2 — the flag deltas provably do not perturb the data split

This matters more than anything else for comparability. `build_data()` draws the
train/test split and the ID choice sets from `rng = Random(args.seed)`, and it runs
*before* any of the four deltas take effect:

* `bake_decoys` uses its own `random.Random(args.seed + 9173)` (`rexpure_optimize.py:138`),
  so switching `--contrastive-fd` off does **not** advance the split RNG.
* `--no-perception` is applied at `rexpure_optimize.py:340`, after `build_data` returns.
* `--no-id` and `--no-beliefs` (below) touch only the adapter and the seed candidate.

So all five arms see **the same 60 train / 50 test transitions, the same `k=5` choice sets,
the same K=9 windows**. This is assertable in a unit test (§2.3) and should be, because it
is the single claim the ablation table rests on.

---

## 1. The four arms

| arm | flag delta vs the reference command | code needed |
|---|---|---|
| **−FD** | remove `--contrastive-fd`, `--cfd-hard-decoys` | none |
| **−ID** | add `--no-id` | ~40 lines (§2.2) |
| **−Perception** | remove `--start-perception <path>`, add `--no-perception` (+ I/O-only knobs, below) | none — flag exists |
| **−Beliefs** | add `--no-beliefs` | ~20 lines (§2.1) |

Resulting objectives: NLWM `min(ID, cFD)` · −FD `ID` · −ID `cFD` · −Perception `min(ID, cFD)`
with P frozen to identity · −Beliefs `min(ID, cFD)` with B frozen to empty.

### −Perception carries one documented I/O-only deviation

With P frozen to the identity module every F prompt carries 19 raw grids instead of a
compact feature transcript (~10x the prompt). The aug19 ablation
(`offline_learning/launch/launch_aug19_noperc.py`) found that under the reference's 30 s
hedge every call out-ran the delay, raced a duplicate, and killed 4 of 5 games with
unhandled 429s from groq. Carry its fix forward verbatim:

```
--hedge-delay 120 --llm-timeout 180        env: LLM_RETRIES=10
```

Same model, same provider pin, same prompts — only how long we wait and how often we
re-ask. It must be recorded in the arm's `launch.json` and named in the paper's protocol
note.

### What each arm is expected to show (hypotheses, so a surprise is legible)

* **−FD**: ID alone rewards action-discriminative features but not fidelity. Expect drift
  toward the collapse attractors already documented (`fd-exact-error-collapse-attractor`,
  `click-mechanism-objective-bottleneck` in memory): lossy P, colour collapse on dq8gc,
  and planning worse than the ID metric suggests.
* **−ID**: cFD alone rewards *fidelity* (pick the true next frame among hard decoys) with
  no pressure to make the action recoverable. Expect verbose/near-lossless P and beliefs
  that describe appearance rather than dynamics. Note `forward-objective-scoring` in memory:
  a forward objective without ID pairing is blind-P gameable; here the only guard is the
  constant-P gate (`_constant_p_gate`), which is *not* part of either term and stays on.
  If this arm collapses on most games, that is the result, not a bug.
* **−Perception**: the aug19 5-game run measured mean test ID falling from ~0.85 to ~0.53.
  Expect the largest single drop of the four.
* **−Beliefs**: bt3gb and n2ntd historically shipped empty beliefs anyway
  (`minclick-contrastive-validated`), so this arm should be near-NLWM on those and cost
  most on the games where beliefs carry the hidden-state rule (dq8gc's click=select,
  eahcw's latched paint colour, egg's latched height).

---

## 2. Code work — DONE

All opt-in flags; the default path is unchanged, so the reference run stays reproducible
(asserted: `--ablation none` rebuilds the shipped command byte-for-byte).

### 2.1 `--no-beliefs` (`offline_learning/rexpure_optimize.py`, ~20 lines)

Mirror of the existing `--no-perception`, which is the exact template:

```python
ap.add_argument("--no-beliefs", action="store_true",
                help="ABLATION: do not learn beliefs. The world-knowledge block is fixed "
                     "to empty, so every F prompt shows '(empty)' where beliefs would be "
                     "and the search mutates ONLY perception. Mutually exclusive with "
                     "--start-beliefs and --no-perception.")
```

and in `main()`:

* error on `--no-beliefs --start-beliefs ...` and on `--no-beliefs --no-perception`
  (that combination is the `raw` arm, not an ablation of the method);
* `seed_candidate = {"perception": seed_code}` — drop the `world_knowledge` key entirely;
* `module_selector = SingleComponentSelector("perception")`.

Nothing else needs touching: every read of the belief text in `invdyn_core.py` is
`candidate.get("world_knowledge", "")` (lines 1405, 1818, 1868, 2003), so a missing key
yields `""` everywhere, and `best_beliefs_rexpure_seed1.txt` is still written (empty) so the
evaluator's `build_resources` finds the artifact instead of skipping the arm. On the eval
side `build_prompt` already renders `beliefs.strip() or "(empty)"`, so the lmwm prompt keeps
its structure with an empty knowledge block — which is the ablation semantics we want.

### 2.2 `--no-id` (`rexpure_optimize.py` + `invdyn_core.py`, ~40 lines)

CLI:

```python
ap.add_argument("--no-id", action="store_true",
                help="ABLATION: drop the inverse-dynamics term from the training "
                     "composite and from the proposer's feedback. Requires a surviving "
                     "term (--contrastive-fd or --fd-scorer != none). The held-out ID "
                     "test at the end of the run is unchanged, so the metric stays "
                     "comparable across arms.")
```

`InvDynAdapter.__init__` takes `no_id=False` and refuses a configuration where ID was the
only term. **Suppression reaches four places, not one** — an ablation that only changed the
score would leave the proposer optimising ID by hand, which would make it an ablation of
the scorer rather than of the objective:

1. **Composite** (`invdyn_core.py`): `terms = [] if self.no_id else [id_score]`, then
   `fd_score`/`cfd_score` as before.
2. **Reflective dataset** (`make_reflective_dataset`), for BOTH components: drop F's
   predicted action, its decoder reasoning, the `INVERSE …` notes and feedback bullets, and
   the ID-framed "Correctly identified X" contrast cases; rename the evidence block from
   `## Inverse Dynamics` to `## Transition`; reveal `a_t` in the window transcript instead
   of masking it `??? (IDENTIFY THIS)` (`_inverse_transcript(win, reveal_action=…)`);
   restate the constant-P gate's justification in contrastive terms. What stays is the
   shared evidence a forward model needs — `raw_state_1/2`, both `perceive()` renderings,
   the true action, the whole window.
3. **Diagnosis calls** (`_analyze_failures`): inverse cases are never diagnosed. That text
   goes straight into the proposer prompt, so it is ID signal even though the composite
   never sees it.
4. **Proposer templates** (`build_reflection_templates(env, no_id=True)`): the task
   statement itself is restated — "identify the action taken between the two center states"
   → "identify which of several candidate observations is the TRUE next state"; "makes the
   action recoverable" → "preserves … that the true next state is distinguishable from
   near-miss frames"; "map feature changes … to action names" → "PREDICT the next features
   … precisely enough to tell the true next state apart". Each swap asserts its source
   sentence exists, and a regex backstop (`_NO_ID_BANNED`) re-scans the finished templates
   and raises if any inverse-dynamics framing survives — including framing added later
   anywhere in the templates or the observation schema.

**The ID F-call still runs.** `id_score` stays computed and logged in `predictions.jsonl`,
it just never enters the score or any prompt, and the end-of-run held-out ID protocol is
untouched — so test ID stays comparable across all five arms. Costs ~40% more F calls
(≈ +$10 over the arm) and buys the diagnostic "what did ID do while nothing optimised it".

`--no-id` composes with `--no-perception` / `--no-beliefs`, but we run only the four
single-delta arms.

### 2.3 Tests (`tests/test_ablation_flags.py`) — 32 passing

1. **Split invariance (the load-bearing test).** `build_data()` is run on the reference
   argv and on each of the four deltas; the train and test transition identities and their
   baked choice sets must be equal, as must `context_k`, the action pool and `id_n`. A
   companion test checks the reference rebuild against the `train_fingerprint` the run
   itself checkpointed, so the comparison is anchored to the real shipped split, not just
   to itself. This is what licenses "same data" in the paper — if it fails, no ablation
   number is comparable to the NLWM column.
2. −FD leaves train items with no `cfd_options` and nothing else changed.
3. `--no-id` composite reduces to cFD; a regex sweep over the *rendered* proposer prompt
   (via `render_reflection_prompt`, i.e. exactly the bytes the LM sees) must find zero ID
   leaks for both components, while the same sweep must find them in the reference prompt;
   the true action and both feature renderings must survive; the window transcript reveals
   `a_t`; inverse diagnosis cases are skipped; the templates change for every env and pass
   `_NO_ID_BANNED`; a moved template sentence raises.
4. `--no-beliefs` selector/candidate wiring.
5. Every rejected and every accepted ablation flag combination, through the real
   `validate_args` (extracted from `main()` so it runs without a dataset).
6. Held-out cFD: train and test decoys use different rng offsets; `eval_cfd_on` refuses
   unbaked items rather than silently scoring an easier question.

`validate_args()` was split out of `rexpure_optimize.main()` for (5); that is the only
refactor of existing behaviour.

---

## 3. Training runs

### 3.1 Launcher

Extend `offline_learning/launch/launch_human_origin.py` with
`--ablation {none,nofd,noid,noperc,nobeliefs}` rather than writing a fifth bespoke
launcher. That file already rebuilds each command from the reference `launch.json` and
swaps only named flags, which is exactly the guarantee we need; the ablation applies one
more documented delta on top and records it in the per-run `launch.json`:

```python
ABLATION = {
    "nofd":       {"drop_flags": {"--contrastive-fd", "--cfd-hard-decoys"}},
    "noid":       {"add": ["--no-id"]},
    "noperc":     {"drop": {"--start-perception"},        # value-taking
                   "add": ["--no-perception", "--hedge-delay", "120",
                           "--llm-timeout", "180"],
                   "env": {"LLM_RETRIES": "10"}},
    "nobeliefs":  {"add": ["--no-beliefs"]},
}
```

Out-root layout must match what the evaluator expects (`<root>/rexpure/<game>_s1`):

```
logs/2026-09-XX/ablations/nofd/rexpure/<game>_s1
logs/2026-09-XX/ablations/noid/rexpure/<game>_s1
logs/2026-09-XX/ablations/noperc/rexpure/<game>_s1
logs/2026-09-XX/ablations/nobeliefs/rexpure/<game>_s1
```

Commands (all 15 games, `--max-parallel 3` as the reference used):

```bash
for A in nofd noid nobeliefs; do
  uv run python offline_learning/launch/launch_human_origin.py \
      --learner rexpure --ablation $A --max-parallel 3 \
      --out-root logs/2026-09-XX/ablations/$A
done
uv run python offline_learning/launch/launch_human_origin.py \
    --learner rexpure --ablation noperc --max-parallel 2 \
    --out-root logs/2026-09-XX/ablations/noperc
```

Run `--dry-run` first for every arm and **diff the printed command against the reference**;
the diff must be exactly the delta in the table above and nothing else. That diff is the
artifact to paste into the paper's appendix.

### 3.2 Cost and wall (measured, not guessed)

Reference rexpure run, from the 15 `stdout.txt` summary lines:
**$25.62 total, 44–65 min/game, 13.4 h serial ≈ 4.5 h at `--max-parallel 3`.**

The aug19 no-perception arm vs its own reference on the same 5 games:
$6.18 vs $1.33 mean per game (**4.6x**) and 118 vs 38 min (**3.1x**).

| arm | est. cost | est. wall @ parallel 3 |
|---|---|---|
| −FD | ~$20 (cFD calls dropped) | ~4 h |
| −ID | ~$28 (ID call kept for logging) | ~5 h |
| −Beliefs | ~$25 | ~4.5 h |
| −Perception | **~$130–150** | ~10 h (parallel 2) |
| **total** | **~$210** | ~24 h if arms run sequentially |

Concurrency budget: each run is `--concurrency 4 × --propose-batch 3` = up to 12 in flight,
so `--max-parallel 3` is ~36 in flight to groq. The measured groq ceiling is ~64 in flight
(`propose-batch-and-throughput` in memory), so **run at most two arms at a time**, and run
−Perception alone.

### 3.3 Risks

* **−Perception on the big grids.** logic_gates is 24x24 (~1450 tok/frame, 2.2x ice) and SET
  is 20x20 with the heaviest obs of the set. With K=9 windows the identity-P F prompt is
  ~27k tokens; gpt-oss-20b's context holds it, but this is where the 4.6x cost multiplier
  will be worst and where 429s are likeliest. Watch these two first, and if a game fails,
  report it as a gap rather than quietly changing its config.
* **−ID may look degenerate.** See the hypothesis above. Pre-register that reading so a
  collapse is reported as a finding, not patched around mid-run.
* **Single seed.** The reference is seed 1 only, so the ablations are seed 1 only. State it.

---

## 4. Planning evaluation

### 4.1 One run per arm, `lmwm` only

The Raw column is shared: the `raw` arm reads no artifacts, so NLWM's published raw
rollouts are the control for every ablation. Do not re-roll it.

```bash
uv run python offline_learning/launch/launch_planning_v2_online.py \
  --goal-presentation nl \
  --cap-mode per-problem \
  --problems logs/2026-09-03/planning_v2_online_ds_percap_nl/problems.per-problem-floors.json \
  --artifact-root logs/2026-09-XX/ablations/<ARM> \
  --arms lmwm --attempts 1 \
  --model deepseek/deepseek-v4-flash \
  --provider-only parasail/fp8,novita/fp8,alibaba/fp8 \
  --concurrency 24 \
  --out-root logs/2026-09-XX/planning_v2_online_abl_<ARM>
```

Every flag except `--artifact-root` and `--out-root` is copied from the NLWM driver line in
`logs/2026-09-03/planning_v2_online_ds_percap_nl/driver_nl.log`. Reusing that run's own
`problems.per-problem-floors.json` (rather than rebuilding from
`logs/2026-08-29/planning_v2/problems.json`) is what pins the 86 rows, the per-problem
action caps and the measured random floors to the published values.

### 4.2 GOTCHA — do not `--seed-from` the NLWM run

`--seed-from` copies whole per-game checkpoints and the rollout key is
`(task, arm, attempt, cap)`. Seeding an ablation from NLWM would make the evaluator
"resume" **NLWM's own lmwm rollouts** as the ablation's, silently producing a perfect tie.
Seeding is safe only when adding a *new* arm name (that is why the ICL runs could use it).
Start each ablation from an empty out-root.

### 4.3 Cost and wall (measured)

NLWM's 2-arm run: **$14.16 and 20.7 h serial** (12–190 min/game). An lmwm-only run is about
half the rollouts but the same serial game loop, so budget **~$7–9 and ~13 h per arm**.
Running two arms concurrently at `--concurrency 24` (48 in flight on the deepseek route,
which carried 24 comfortably) gives ~26 h for all four. Total ≈ **$30**.

### 4.4 Held-out cFD — BUILT

The training loop bakes decoys on `train` only (`rexpure_optimize.py:139`) and `eval_fd_on`
runs only when `--fd-scorer != none`, so the reference reports no held-out forward number
at all (`"forward_score": null` in every `test_summary_rexpure_seed1.json`). Fine while
every arm optimises ID; not fine for **−ID, whose own objective would otherwise go
unmeasured on held-out data**.

`offline_learning/scripts/eval_heldout_cfd.py` closes it with no retraining. It re-parses a
finished run's `launch.json` argv through `rexpure_optimize.build_parser()`, feeds it back
to `build_data()` — the function that produced the split in the first place — and
**fingerprint-checks the rebuild against the run's own `resume_state.json`** before scoring,
so a number can never come off a different test set. Then it bakes test decoys
(`bake_test_decoys`, its own rng offset so train and test never share a draw) and scores
both target renderings via `eval_cfd_on`:

* **raw** — options shown as raw frames. Candidate-independent all the way to the prompt, so
  the 1/(n+1) floor is a constant for any P and every arm answers the identical question.
  **This is the cross-arm column.**
* **perceived** — options shown as `P(option)`. Measures whether *this* P preserves what
  separates the true next state from a near miss; each arm answers in its own feature
  language, so read it per-arm rather than across arms.

Per-game results are cached as `heldout_cfd_seed1.json` next to the artifacts (re-runs
resume free) plus a combined table at `--out`. Smoke on the reference: colour_lines raw
0.42 / perceived 0.56 against a 0.25 chance floor, $0.04 and ~2 min for 100 F calls — so
all 15 games × 5 arms is roughly **$3 and a couple of hours**.

```bash
OPENROUTER_PROVIDER_ORDER=groq uv run python offline_learning/scripts/eval_heldout_cfd.py \
  --artifact-root "NLWM=logs/2026-08-24/human_curated" \
  --artifact-root "-FD=logs/2026-09-XX/ablations/nofd" \
  --artifact-root "-ID=logs/2026-09-XX/ablations/noid" \
  --artifact-root "-Perception=logs/2026-09-XX/ablations/noperc" \
  --artifact-root "-Beliefs=logs/2026-09-XX/ablations/nobeliefs" \
  --out logs/2026-09-XX/heldout_cfd
```

New training runs can also compute it inline with `--cfd-test` (same `eval_cfd_on`, same
`bake_test_decoys`, so the two paths are interchangeable). It is off by default so existing
reference commands stay byte-identical; the ablation launcher does **not** add it, for the
same reason — use the standalone scorer so all five arms are measured by one code path.

---

## 5. Reporting

### 5.1 Numbers to put in `tab:ablations`

Per arm, two columns rather than one, because the interesting result is likely the gap
between them (as it already was for NLWM (SL): +0.086 test ID, −0.02 planning):

| column | source |
|---|---|
| mean credited test ID | `<ablation>/rexpure/<game>_s1/test_summary_rexpure_seed1.json::inverse_accuracy`, macro over 15 games — free, no extra runs |
| online planning macro pass@1 | `report_planning_v2_online.py` over the ablation's run dir |

Plus, in the text: floor-adjusted macro, the L1–L4 tier breakdown, and a **paired sign test
over the 86 shared rows against NLWM** — all four arms score the identical rows, so the
paired test is strictly better powered than comparing macros, and the existing report
warns that per-game cells at 1 attempt are one draw of a coin.

### 5.2 Tooling

The console/LaTeX report already takes N runs:

```bash
uv run python offline_learning/scripts/report_planning_v2_online.py \
  --run "NLWM=logs/2026-09-03/planning_v2_online_ds_percap_nl" \
  --run "-FD=logs/2026-09-XX/planning_v2_online_abl_nofd" \
  --run "-ID=logs/2026-09-XX/planning_v2_online_abl_noid" \
  --run "-Perception=logs/2026-09-XX/planning_v2_online_abl_noperc" \
  --run "-Beliefs=logs/2026-09-XX/planning_v2_online_abl_nobeliefs" \
  --raw-from NLWM --check
```

Two small additions to make the ablation table auto-generated like the main one:

1. a `tab:ablations` entry in `BLOCKS` (`report_planning_v2_online.py:407`) whose builder
   emits one row per arm — arm name, mean credited test ID, macro pass@1 — reading the
   training summaries from a `--learning-root` argument;
2. `% BEGIN AUTO tab:ablations` / `% END AUTO tab:ablations` markers around the tabular body
   in `paper/main.tex:137-151`, so the table stops being hand-maintained.

`paired()` currently only prints for exactly two runs; extend it to compare every run
against `--raw-from`'s partner (or just invoke the report twice, NLWM + one ablation, to get
the paired block per arm — cheaper and needs no code).

---

## 6. Sequencing

1. Code: `--no-beliefs`, `--no-id`, tests (§2). Half a day; the split-invariance test is the
   one that must pass before any money is spent.
2. `--ablation` support in `launch_human_origin.py`; `--dry-run` all four; diff each command
   against the reference and save the diff (§3.1).
3. Train −FD and −Beliefs together (cheapest, lowest risk) → sanity-check
   `test_summary_rexpure_seed1.json` on all 15 games.
4. Train −ID; then −Perception alone (§3.2/3.3).
5. Eval: two arms at a time, `lmwm` only, from empty out-roots (§4).
6. Report + paper table (§5); update the Ablations section and the TODO in
   `experimental_plan.md`.

Total ≈ **$240** and ~3–4 days elapsed, dominated by the −Perception training arm and the
serial planning evals.

## 7. Decisions

1. **−ID reflection suppression** — DECIDED, implemented: ID evidence, diagnosis and task
   framing are all withheld from the proposer, so this is an ablation of the objective and
   not merely of the scorer (§2.2). The paper should say so explicitly, since the weaker
   score-only reading is what a reader would otherwise assume.
2. **Held-out cFD** — DECIDED, built (§4.4). Score all five arms with the standalone script
   so one code path produces every cell; report the `raw`-target column across arms.
3. **SL (Opus) ablations** — not planned. The ablations are on the DeepSeek NLWM reference,
   which is the paper's headline column.
4. **Seeds** — one seed, matching the reference. If the table needs error bars, the paired
   sign test over the 86 shared rows is the affordable substitute for a second seed.

Still open, and cheap: `report_planning_v2_online.py` has no `tab:ablations` block builder
yet, and `paper/main.tex:137-151` still has no AUTO markers (§5.2). Neither blocks the runs.
