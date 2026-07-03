# perc_invdyn — How It Works

Inverse-dynamics perception optimization: learning a perception module **P** and a
beliefs block **B** for an LLM game agent on **AutumnBench** / **ARC-AGI-3** games,
*without* any hand-coded grid parser or game facts entering the objective.

This directory is a research prototype. This document explains the mental model, the
core modules, the data/objective flow, the datasets, and how to run everything.

---

## 1. Mental model

We learn two text artifacts for the agent:

- **P** — a Python module exposing `perceive(observation_history: list[str]) -> str`
  that turns a raw grid observation into a short **text feature summary**.
- **B** — a `world_knowledge` / beliefs text block describing the game's dynamics.

Both are learned from logged transitions `(X_t, a_t, X_{t+1})` using two
self-supervised objectives — **no ground-truth grid diff, no hand-written game
facts** ever touch the score (the *purity invariant*, see §6):

- **Inverse-dynamics (ID):** show a frozen decoder `F` the features `P(X_t)`,
  `P(X_{t+1})` and a choice set of actions; score `1` if it recovers the true `a_t`.
- **Forward-dynamics (FD):** given `P(X_t)` and `a_t`, predict `P(X_{t+1})`; score the
  prediction against the true next-state features.
- **Composite:** `score = (1 - w) * ID + w * FD`  (default `w = 0.5`).

A **GEPA** (pareto reflective) optimizer — or a population/greedy variant — rewrites P
and B to maximize that composite on a held-out split. An optional **online loop**
(`explore_loop`) actively steps a live environment to *collect* the most
learning-useful transitions instead of reading them from static logs.

---

## 2. Pipeline at a glance

```
raw run logs  (episode_*/trajectory.csv: Step, Action, Observation, Done, ...)
        │
        │  load_transitions(run_dirs, whitelist, context_k)        [validate.py]
        ▼
Transition(X_t, a_t, X_{t+1}, ctx_prev/next)   one per consecutive row pair, filtered
        │
        │  balanced_split → train / val / test   +   bake_choices (freeze ID choices)
        ▼
GEPA optimize  (InvDynAdapter)                             [gepa_optimize.py]
   candidate = {perception P, world_knowledge B}, seeded EMPTY
   per transition, per candidate:
       ID:  F(P(X_t), P(X_{t+1}), B, choices) == a_t ?          → 1/0
       FD:  Fwd(P(X_t), a_t, B) vs P(X_{t+1})                   → [0,1]
       composite = (1-w)*ID + w*FD
   reflective dataset = internal-only failure signals ("gradient")   [+ --analyze-mistakes]
   reflection LM rewrites P or B (belief-update-period), pareto-select
        ▼
best candidate → clean TEST split (never touched during search)
        ▼
best_perception_*.py   +   best_beliefs_*.txt
```

**Objectives recap**

- **ID** pressures P/B to make actions recoverable from feature changes. Reference
  points: `random` (`1/k`), `raw-frame` (F sees the truncated raw grid), and the
  `GOOD_P` ceiling.
- **FD** pressures P toward Markov sufficiency. Four scorers: `none`, `exact`,
  `textdiff`, `judge`. **FD alone is degenerate** — it rewards a blind constant P
  (`forward_objective.BLIND_P` demonstrates this) — so it is only ever used as the
  `w`-weighted partner of ID.
- The **active loop** (`explore_loop` + `explore_score`) is an alternative
  data-*collection* front-end feeding the same GEPA learner online.

---

## 3. Core modules

### `validate.py` — reference inverse-dynamics learner (from-scratch greedy)
Standalone proof that the ID backward-signal (forward → loss → textual gradient →
P-update, gated on held-out accuracy) genuinely improves P rather than resampling it.
It also owns the **canonical data primitives reused everywhere**:

- `@dataclass Transition(x_t, x_t1, action, ctx_prev, ctx_next)` — atomic data unit;
  `ctx_prev/ctx_next` hold up to K steps of within-episode temporal window (only when
  `context_k > 0`).
- `load_transitions(run_dirs, action_whitelist, context_k=0)` — **the** loader used by
  all optimizers. Globs `run_dir/episode_*/trajectory.csv`, makes one transition per
  consecutive row pair with `rows[i].Action` as `a_t`, drops episode boundaries / empty
  obs / out-of-whitelist verbs, and attaches shrinking-at-boundary windows.
- `run_perceive(code, raw_obs) -> (output, error)` — `exec`s P and calls
  `perceive([raw_obs])`; the single-frame forward pass reused everywhere.
- `make_config(model_id, client_name)` — minimal OmegaConf so the repo's `_llm_call`
  works without Hydra.
- `make_choices`, `predict_action` (frozen `F`), `compute_g1` (textual "gradient" =
  LLM failure diagnosis), `update_perception` (LLM "optimizer step"), `forward_eval`.
- `_parse_tag` / `_extract_action` — robust decoder answer extraction (`<action>` →
  `\boxed{}` → last-mentioned choice; `None` = abstention, never a silent default).
- `GOOD_P` — a known-good DQ8GC perception; a **measurement-only ceiling, never a seed**.

`_llm_call` is imported from the repo-root `mixed_improve` module.

`validate_beliefs.py` extends this so `F` also reads `B` — i.e. it adds the beliefs
component to the greedy learner.

### `gepa_optimize.py` — the primary optimizer (the workhorse)
Replaces the greedy single-best gate with GEPA's pareto-frontier reflective search over
the multi-component candidate `{"perception", "world_knowledge"}`, optimizing the ID+FD
composite.

- `class InvDynAdapter(GEPAAdapter)` — the heart. `evaluate(...)` runs the ID pass (and
  optional FD) per transition and returns scores; `make_reflective_dataset(...)` builds
  per-failure feedback from **internal observables only** (predicted vs true action, F's
  reasoning, output-invariance notes) — never a ground-truth grid diff. Key flags:
  `fd_scorer`, `fd_weight`, `fd_reflect`, `analyze_mistakes`, `analyze_mode`,
  `context_k`, `image_mode`, `f_image`, `reuse_traces`.
- `predict_action` / `predict_next_state` (+ windowed and `_img` variants) — the ID and
  FD LLM calls.
- FD scorers: `exact_match_f1` plus `textdiff_delta_f1` / `judge_score` (from
  `forward_objective`).
- `bake_choices` — freezes a choice set per transition into GEPA DataInsts.
- `balanced_split` — balanced-by-action train/val/test sampling.
- `eval_on` (ID accuracy) / `eval_fd_on` (FD score) — reused by `explore_loop` and
  `pop_optimize`.
- `build_reflection_templates`, `make_reflection_lm`, `observation_schema` — the
  proposer LM and the env-scoped **format-only** schema (no dynamics) handed to the
  P-writer.
- `PerceptionBiasedComponentSelector`, `install_accept_ties_patch`, `run_legacy_loop`
  (the `--compare` legacy greedy), grid rendering helpers.

Always starts from **EMPTY** P/B. Writes `best_perception_gepa_seed<N>.py`,
`best_beliefs_gepa_seed<N>.txt`, per-item traces, and a GEPA run_dir for
`build_optim_viz`.

### `forward_objective.py` — FD scoring library + degeneracy probe
Defines and validates ways to score a predicted next-state against the true `P(X_{t+1})`,
operating **only on P's own emitted symbols**.

- `tok_multiset(s)` — words+integers multiset (content-agnostic atomizer; reused by
  `explore_score`).
- `textdiff_delta_f1(start, pred, true)` — format-agnostic change-F1 via `difflib` (the
  deterministic FD scorer wired into GEPA); plus `delta_f1`, `tok_delta_f1`, `frame_f1`.
- `judge_score` / `judge_score_reasoned` — LLM judge (generic rubric, no game facts).
- `spearman(xs, ys)` — rank correlation (no scipy).
- `BLIND_P` — constant-output P that demonstrates the FD-alone degeneracy.

### `explore_score.py` — active-exploration acquisition scoring (the DECIDE step)
Scores the actions available at `X_t` by expected **value to learning** (epistemic, not
reward) and returns the argmax.

- `score_actions(...)` → `ActionScore(observability, disagreement, coverage, contrast,
  revisit_pen, total)`. Signals: observability (predicted |Δ| in P's symbols — defeats
  unobservable transitions), disagreement (frontier-ensemble VOI), coverage (UCB),
  contrast (complete a recurring context's action set), revisit penalty.
- `select_action(...)` — top-level DECIDE returning `(best, ranked, cost)`.
- `missing_at_context(...)` — no-reset contrastive targeting: actions not yet seen from
  a P-context matching `x_t`.

Imports `predict_next_state` from `gepa_optimize`, `tok_multiset` from
`forward_objective`, `Transition`/`run_perceive` from `validate`.

### `explore_loop.py` — the runnable DECIDE→ACT→LEARN active loop
Closes the loop: step a live env to collect the best transitions, GEPA-learn P/B on the
buffer, repeat.

- `class LiveEnv` — boots one BALROG/autumn/arc env; raw text obs =
  `obs["text"]["long_term_context"]` (identical to trajectory.csv → in-distribution with
  offline data).
- `relearn(...)` — runs `gepa.optimize` on the buffer; returns a frontier (top-N
  distinct candidates = the disagreement ensemble) + held-out ID accuracy.
- `explore_learn(args)` — warmup round-robins actions, then `missing_at_context` +
  `select_action` pick each step; relearns every `k-relearn` steps; starts a new episode
  on done/stuck (no reset). Writes `final_perception.py`, `final_beliefs.txt`,
  `loop_log.json`.

### `pop_optimize.py` — population / beam optimizer
A simpler-to-reason-about alternative to GEPA's pareto engine: generational beam search
over `(P, B)` reusing GEPA's reflective proposal + the same ID/FD objective verbatim.
`legacy_pop.py` is a related "missing middle" variant.

### `clean_data3_tools.py` — data-curation helpers
- `grid_at(obs)` — parses BOTH autumn `[["color", ...]]` arrays and ARC `<grid_N>`
  integer blocks into a 2D list.
- `classify(g0, g1)` — short per-color add/remove/move change-tag.
- `dump_transitions(...)` / `verify_pool(...)` — inspect what actually gets *scored*
  (the latter loads a curated dir exactly as GEPA will, via `load_transitions`).

### Core direct dependencies
- `mixed_improve.py` (repo root) — `_llm_call`, `set_meta_temperature`.
- `autumn_seed_perception.py` / `start_perc_autumn_robust.py` — general Autumn grid
  scaffolds passed to `--warm-start-perception` / `--start-perception` to avoid the
  parse-collapse local optimum. External: `gepa`, `hydra`, BALROG env factories.

---

## 4. Datasets: `clean_data*/`

All three hold `<game>/…/episode_*/trajectory.csv` consumed by `load_transitions`, in
increasing order of curation:

- **`clean_data/`** — pooled single-trajectory format (`<game>/episode_*/trajectory.csv`
  + `dynamics.txt` + `viz.html`). Used with an in-pool balanced test carve
  (`clean_sweep --no-cross-traj`).
- **`clean_data2/`** — cross-trajectory format (`<game>/train/…`, `<game>/test/…`).
  **This is `clean_sweep.py`'s default `--data-root`** (`--cross-traj`): train on one
  crafted trajectory, test on a distinct one showing the same dynamics.
- **`clean_data3/`** — the most curated (see `clean_data3_METHODOLOGY.md`). Each
  `train/episode_*` is a **short contiguous slice copied verbatim** from the original
  trajectory, chosen so every scored target pair exercises a CORE dynamic from
  `dynamics.txt` under BOTH ID and FD, plus **contrastive near-miss negatives** (defeats
  step-parity clock shortcuts — the `nrdf6` finding). Pools kept ~18–22 so `--train-n 20`
  keeps them all.

**Why slicing matters:** `load_transitions` scores every consecutive CSV row pair;
windows (`context_k`) are *context only*. So making a dynamic a *scored target* (not just
window context) means placing it as an internal pair of a slice.
`clean_data3_ARC_METHODOLOGY.md` covers ARC specifics (`<grid_N>` integer grids,
`arc_drive.py` driver, camera-scroll/level-transition caveats, `keep_action_params=True`
for click games where the click *location* is the ID label).

---

## 5. Running things

Run everything with `uv run` from the repo root (`/home/ays57/bai`).

**Main sweep — `clean_sweep.py`** (fans `gepa_optimize.py` over games×seeds, parses
summaries → `logs/<out-name>/results.json`):
```bash
# default antimemo GEPA-only sweep, seed 1, 7 parallel:
uv run python prototypes/perc_invdyn/clean_sweep.py --seeds 1 --parallel 7
# head-to-head vs legacy greedy:
uv run python prototypes/perc_invdyn/clean_sweep.py --compare --max-metric-calls 120
```
Notable defaults: `--train-n 20 --test-n 10 --task-model deepseek/deepseek-v4-flash
--max-metric-calls 2000 --context-k 9 --data-root clean_data2 --cross-traj
--fd-scorer exact --analyze --concurrency 16 --parallel 7`. The per-game whitelist /
`keep_action_params` / budget table lives in the `GAMES` dict at the top of the file.

**Single optimizer run — `gepa_optimize.py`:**
```bash
uv run prototypes/perc_invdyn/gepa_optimize.py --run <dir> \
    --task-model google/gemini-2.5-flash --max-metric-calls 2000 [--compare]
```
Key flags: `--train-n/--val-n/--test-n`, `--k-choices 5`, `--actions`,
`--fd-scorer {none,textdiff,judge,exact}`, `--fd-weight 0.5`, `--fd-reflect`,
`--analyze-mistakes`, `--context-k 3`, `--belief-update-period 4`, `--image-mode`,
`--f-image`, `--good-baseline`, `--start-perception/--start-beliefs` (continue from
*learned* artifacts only), `--seed`. Always starts from EMPTY P/B.

**Other runnable optimizers:** `pop_optimize.py`, `validate.py`, `validate_beliefs.py`,
`explore_loop.py`, `forward_objective.py` (standalone scorer margins).

Example active loop:
```bash
uv run prototypes/perc_invdyn/explore_loop.py --env autumn --task DQ8GC \
    --budget 40 --warmup 12 --k-relearn 14 --max-metric-calls 80 \
    --task-model google/gemini-2.5-flash
```

**Launch scripts** (each `cd`s to repo root and wraps `gepa_optimize.py`):
`launch_newgames_gepa.sh`, `launch_imagemode_gemini.sh`, `launch_pinned_imgmode.sh`,
`launch_resume_imgmode.sh`.

**Data drivers / generators:** `autumn_drive.py <task> <out>`,
`arc_drive.py <game> <OUTDIR> --actions "<csv>"`, `gen_clickmove.py`,
`_rowmajor_clean_data.py`, `render_arc.py`.

**Visualization builders:** `build_optim_viz.py` (GEPA run_dir → optim_viz.html),
`build_pop_viz.py`, `build_dataset_viz.py <dir>`, `build_clean_data3_viz.py <game>`,
`gen_optim_viz.py`, `gen_optim_viz_cd3.py`, `mark_train_split_viz.py`,
`arc_imgmode_dashboard.py` (live sweep dashboard).

---

## 6. Invariants & gotchas

- **Purity invariant** (`invdyn-no-external-knowledge`): every objective and acquisition
  signal is computed only over P's emitted symbols + the logged next frame run through
  the same P — no raw-grid parser, no hand-coded background, no game facts in
  judge/reflection. `GOOD_P` and `--good-baseline` are measurement-only ceilings, never
  seeds.
- **Learning always starts from EMPTY** P and B. `--start-perception` /
  `--start-beliefs` / `--warm-start-perception` only continue from previously *learned*
  artifacts or the format-only scaffold — never for injecting hand-written knowledge.
- **`--tie-train-val`** (default on) is the low-data regime where val == train; the clean
  test split is always untouched.
- **The `nrdf6` finding** motivates `clean_data3`: a dynamic present only as window
  context, or gameable by a step-parity clock, is never actually learned — hence the
  "score every core mechanic as a contrastive target pair" recipe.

---

## 7. File map

- **Core library:** `validate.py`, `validate_beliefs.py`, `gepa_optimize.py`,
  `forward_objective.py`, `explore_score.py`, `explore_loop.py`, `pop_optimize.py`,
  `legacy_pop.py`, `clean_data3_tools.py`; seed scaffolds `autumn_seed_perception.py`,
  `start_perc_autumn_robust.py`.
- **Entrypoints:** `clean_sweep.py`, `run_sweep.py`, the `launch_*.sh` scripts, the env
  drivers, and the `build_*`/`gen_*`/`*_viz`/`*_dashboard` visualizers.
- **Analysis (one-off diagnostics):** `analyze_*`, `diag_*`, `diagnose_*`, `dump_*`,
  `finalize_seeds5.py`, `link_diagnosis.py`, `reconstruct_analysis_calls.py`,
  `raw_history_diag.py`, `audit_forward.py`, `probe_dynamics.py`, `compare_alt_joint.py`,
  `greedy_fd.py`, `greedy_matched.py`.
- **Artifacts (regenerable):** `best_perception_*.py`, `best_beliefs_*.txt`,
  `*_results.md`, findings write-ups.
- **Methodology docs:** `clean_data3_METHODOLOGY.md`,
  `clean_data3_ARC_METHODOLOGY.md`, `clean_data3_REGEN_METHODOLOGY.md`,
  `forward_composite_runplan.md`, `exploration_runs_catalog.md`.
- **Datasets:** `clean_data/`, `clean_data2/`, `clean_data3/`.
