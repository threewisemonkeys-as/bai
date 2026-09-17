# Probing implementation

Implements the first executable study from [the execution plan](probing_analysis_execution_plan.md):
frozen human-trajectory windows, training-selected checkpoints, direct multi-step
forward prediction, controlled reconstruction, and reproducible reporting.

## Entry points

| Command | What it does |
|---|---|
| `offline_learning/scripts/build_probe_manifest.py` | Audits saved runs, replays source drives, selects fixed windows/checkpoints/examples, writes a content-hashed manifest. |
| `offline_learning/scripts/eval_probing.py` | Prepares prompts, evaluates deterministic baselines, and scores cached responses. Add `--execute` for model calls. |
| `offline_learning/scripts/report_probing.py` | Produces coverage-aware JSON/CSV/Markdown tables and PDF/PNG figures. |

Run from the repository root using the existing environment (`uv run --no-sync python`).
The data builder and prompt preparation are local operations with no model calls.

## 1. Build the reference dataset and saved ablations

```bash
uv run --no-sync python offline_learning/scripts/build_probe_manifest.py \
  --include-ablations \
  --out analysis/probing/manifest.json.gz
```

Defaults: 15 reference games, training seed 1, evaluation sample seed 0, k=9,
horizons 1/2/4/8, 50 starts/game, 100 reconstruction query time points/game,
8 training demonstration states. Starts are shared across horizons and all arms.
Sampling balances users, then drives within users; it does not filter static frames.
`--nonoverlapping` requires the complete context-through-target intervals to be disjoint.
Shortfalls are recorded, not padded with duplicate windows.

The first artifact root receives first-working, 25%, 50%, 75%, and final incumbents.
Additional arms receive final checkpoints. Duplicate P/K pairs share forecast requests;
duplicate P's share reconstruction requests. Checkpoint aliases remain in the results.
Supply alternate roots with repeated `--artifact-root LABEL=PATH` arguments.

The builder verifies:

- Original split reproduction and its legacy action fingerprint.
- Raw states **and past/future context actions** against every saved training batch.
- Every selected slice's two rows against its exact full-drive position.
- Shared train/test context contents and dataset manifests across arms.
- Train/test user separation, program-source identity, chronology and simulator replay.
- Source hashes for the compiled simulator and its Autumn standard library.
- Shipped P/K equality to the unique final training incumbent.
- Perception execution on training frames in two fresh Python processes with hash seed 0.

It also records overlap with training target/context/decoy frames. Identical grids across
users are marked as previously seen; they are not silently removed. Reflection is assumed
to derive from the audited batches; arbitrary external reads inside learned programs are
not audited. These remain retrospective data, not a new confirmation set.

Verify a saved manifest and all of its source hashes:

```bash
uv run --no-sync python offline_learning/scripts/build_probe_manifest.py \
  --verify analysis/probing/manifest.json.gz
```

Source or evaluator implementation changes require a new manifest/protocol where applicable.
Use a new output path for a new protocol. `--overwrite` explicitly replaces a manifest.
`--no-replay` is available for inspecting environments without the simulator, but a
model run requires a replay-verified manifest.

## 2. Prepare a small development pilot

Use training drives for prompt development; do not tune the protocol on test outcomes.

```bash
uv run --no-sync python offline_learning/scripts/build_probe_manifest.py \
  --games bt3gb,dq8gc,eahcw,s2kt7 --split train \
  --horizons 1,4 --windows 10 --reconstruction 20 --examples 8 \
  --out analysis/probing/pilot_manifest.json.gz

uv run --no-sync python offline_learning/scripts/eval_probing.py \
  --manifest analysis/probing/pilot_manifest.json.gz \
  --out analysis/probing/pilot \
  --checkpoints first_working,final \
  --forward-modes learned_raw,learned_native,raw,copy \
  --reconstruction-modes learned,raw,constant,shuffled
```

This writes `protocol.json`, `prepared.json`, `jobs.jsonl`, `results.jsonl`, and
`coverage.json`. Model-dependent results are explicitly `pending`. Deterministic
copy/inverse/mode-state baselines can be scored immediately.

`prepared.json` gives unique model requests after deduplication, cached/uncached
counts, and prompt character sizes. It is not a token or price estimate. Record actual
input/output usage and cost in a small execution before choosing the full-study budget.

To execute the same prepared pilot, repeat its `eval_probing.py` command with:

```text
--execute --max-calls 1200 --attempts 2 --concurrency 4
```

`--max-calls` caps uncached logical requests, before any call is sent; retries can use
up to `--attempts` API calls per request. The initial adapter uses the project's
OpenRouter chat endpoint with the model, provider, and reasoning settings from the
reference run. Credentials are read from the existing environment. Override with
`--model`, `--provider`, and `--reasoning-json` when freezing a different evaluator.
Output tokens and prompt characters have explicit caps; inputs are never truncated.

## 3. Supported measurements

**Forward:** `learned_raw`, `learned_native`, `raw`, `lossless`, `copy`.
The first predicts the raw future grid from learned P-history, K, and the supplied
future actions. Native predicts the P-output string. All are direct endpoint requests;
the history ends at the current state. Native prompts receive no raw-grid decoding
examples. Other model modes receive the same frozen training example identities in
their own representation.

**Reconstruction:** `learned`, `raw`, `lossless`, `constant`, `hash`, `shuffled`,
`lossless_inverse`, `train_mode`. The latter two are deterministic controls. Learned
reconstruction receives P(X) and training examples, with no K or trajectory history.
Shuffled controls use another query's representation and record whether the shuffle
actually changed it. `--examples 0` or `--examples 2` uses a nested prefix of the frozen
demonstration list; use a different output directory for each setting.

Raw scoring uses parsed-grid exact match, cell accuracy, foreground F1, and (for
forecasts) cell-change F1. Changes include coordinate and new colour. Native exact
only strips outer whitespace. Empty/malformed answers and perception failures score
zero; provider failures have missing metrics and visible coverage. Responses and
attempt history are cached atomically per request, with exact prompt/model/protocol
identity. Use `--retry-errors` to retry previously failed provider requests.

## 4. Reports

```bash
uv run --no-sync python offline_learning/scripts/report_probing.py analysis/probing/pilot
```

Outputs: `REPORT.md`, `summary.json`, `summary.csv`, horizon figures, and one combined
FD-1 versus FD-h scatter with its plotted values in `fd1_vs_multistep.csv`.
Reconstruction is reported in tables. Environment names are English labels; derived
tables retain source identifiers in a separate `game_id` field.

`python analysis/probing/build_paper_appendix.py` combines the final selected models'
probe scores with the planning values already printed in `nlwm_paper/main.tex`,
producing CSV/JSON/Markdown tables and the included `nlwm_paper/probing_appendix.tex`.
Defaults are raw FD change F1, native/reconstruction exact match, and the Plain/Agentic
NLWM planning columns. Incomplete groups receive no headline score. Partial means remain
available as diagnostics in JSON. Tables include changed/unchanged and train-unseen
strata, parse failures, truncation, users and unique target counts. Reconstruction's
unique-state metric gives each distinct grid equal weight after averaging its occurrences.
Representation diagnostics include output length and an empirical exact-reconstruction
bound from observed collisions, with the valid-query denominator. Native reports include
the fraction of unchanged feature strings and a separate horizon figure, because the
targets vary with P. These diagnostics do not establish general invertibility.

Paired comparisons keep queries aligned and use common random user weights across
games, preserving repeated-user identities. The reported 95% range is an exponential-
weight sensitivity interval conditional on these runs/games. It is not training-seed
uncertainty or a claim of population-wide significance. Source users remain few.

## Scope of this implementation

This is the direct-prediction/reconstruction stage of the plan. Recursive/teacher-forced
rollouts, action-removal interventions, bespoke inverses of learned P, stochastic
distributional forecasts, new held-out users, and new training seeds remain separate
experiments. The generic lossless inverse does not establish invertibility of learned P.
The raw baseline has a limited history and representation examples, not the complete
training information available to NLWM. These limits should remain visible in paper claims.

Targeted correctness checks:

```bash
uv run --no-sync python -m pytest tests/test_probing.py -q
```
