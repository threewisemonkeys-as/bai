# BAI

BAI is an LLM agent evaluation and self-improvement framework built on top of
[BALROG](https://github.com/balrog-ai/BALROG). It runs LLM agents in interactive
environments and uses a stepwise "explore-and-build" loop (`explore.stepwise_eb_learn`)
to iteratively learn beliefs (instruction prompts) about how each environment
works. Supported environments include MiniHack, [AutumnBench](https://github.com/basis-research/MARAProtocol)
(via MARAProtocol + Autumn.cpp), and ARC-AGI 3.

## 1. Setup

### Prerequisites

Install these on the host before running setup:

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- `git`, `curl`, `jq`, `cmake`, `make`, and a C++ compiler (needed to build the
  Autumn interpreter)
- API keys for at least one LLM provider (OpenAI, Anthropic, Google, or OpenRouter)

### One-shot setup

The repo includes a `setup.sh` that performs the full bootstrap:

```bash
# Clone with submodules (BALROG, MARAProtocol, Autumn.cpp)
git clone --recurse-submodules <repo-url>
cd bai

# Runs all of the steps below
./setup.sh
```

`setup.sh` does the following:

1. Initializes git submodules (`BALROG`, `MARAProtocol`, `Autumn.cpp`).
2. Installs Python dependencies with `uv sync` (installs `balrog` and
   `maraprotocol` as editable packages).
3. Builds the `Autumn.cpp` Python interpreter module (needed for Autumn).
4. Generates the MARAProtocol protobuf Python stubs.
5. Downloads the AutumnBench example dataset.

### API keys

Add your provider keys to a `.env` file in the repo root (loaded via
`python-dotenv`). All scripts route through OpenRouter by default, so an
`OPENROUTER_API_KEY` is usually enough:

```bash
# .env
OPENROUTER_API_KEY=sk-or-...
# Or, if calling providers directly:
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
```

### Verify the install

```bash
# Quick smoke test on Autumn
uv run python -m explore.stepwise_eb_learn envs.names=autumn tasks.autumn_tasks=[ice]
```

## 2. Running experiments with `explore.stepwise_eb_learn`

`explore.stepwise_eb_learn` (run with `uv run python -m explore.stepwise_eb_learn`) is the main entry point. It runs the agent through an
environment for a number of steps and, along the way, generates theories /
questions, runs experiments to discriminate between them, and accumulates a set
of beliefs about the environment. It is configured with
[Hydra](https://hydra.cc/) — defaults live in
`BALROG/balrog/config/config.yaml`, and any value can be overridden on the
command line with `key=value`.

The two most important overrides are:

- `envs.names` — which environment (`minihack`, `autumn`, or `arc_agi`)
- `tasks.<env>_tasks=[<id>]` — which task/game within that environment

Common shared knobs:

| Override | Meaning |
|----------|---------|
| `client.client_name` | LLM provider (`openrouter`, `openai`, `claude`, `gemini`) |
| `client.model_id` | Model id, e.g. `google/gemini-2.5-flash`, `anthropic/claude-sonnet-4.6` |
| `eval.evolve.n_environment_steps` | Number of environment steps to run the loop for |
| `eval.evolve.num_theories` | Number of theories generated per step |
| `eval.output_dir` | Where logs are written (defaults under `logs/`) |
| `agent.max_text_history` / `agent.max_image_history` | History window sizes |

Output lands in a timestamped subdirectory under `eval.output_dir` (default
`logs/...`), containing the learned beliefs, per-step details, trajectories, and
the question/experiment timelines that the viz module reads.

### MiniHack

MiniHack tasks are text-based (no images). Pick a task from
`tasks.minihack_tasks` and optionally fix the seed via
`envs.minihack_kwargs.seeds`.

```bash
uv run python -m explore.stepwise_eb_learn \
    envs.names=minihack \
    tasks.minihack_tasks=[MiniHack-Quest-Easy-v0] \
    client.client_name=openrouter \
    client.model_id=google/gemini-2.5-flash \
    eval.evolve.n_environment_steps=100 \
    eval.evolve.hide_obs_when_image=false \
    agent.max_text_history=4 agent.max_image_history=0
```

### Autumn (AutumnBench)

Autumn programs run through the locally built Autumn interpreter and use
image+text observations. Task ids are AutumnBench program ids (see
`MARAProtocol/python_examples/autumnbench/example_benchmark/programs/`). Games that
come from `autumn_programs_55.zip` rather than the benchmark download (rink,
logic_gates, balloon, colour_lines, diffusion, dino, tetris) are tracked under
`autumn_programs/` and installed into that directory with
`uv run tools/install_autumn_programs.py`. Set
`autumn_eval_after_learn=true` to run a frozen evaluation after learning.

```bash
uv run python -m explore.stepwise_eb_learn \
    envs.names=autumn \
    tasks.autumn_tasks=[ice] \
    client.client_name=openrouter \
    client.model_id=google/gemini-2.5-flash \
    eval.evolve.n_environment_steps=100 \
    eval.evolve.hide_obs_when_image=true \
    agent.max_text_history=4 agent.max_image_history=4 \
    eval.evolve.autumn_eval_after_learn=true \
    eval.evolve.autumn_eval_max_steps=501
```

### ARC-AGI 3

ARC-AGI 3 games use image+text observations. Each game is identified by its game
id (e.g. `ls20`, `sp80`, `tn36`, `vc33`, `ft09`) passed via
`tasks.arc_agi_tasks`.

```bash
uv run python -m explore.stepwise_eb_learn \
    envs.names=arc_agi \
    tasks.arc_agi_tasks=[ls20] \
    client.client_name=openrouter \
    client.model_id=google/gemini-2.5-flash \
    eval.evolve.n_environment_steps=50 \
    eval.evolve.hide_obs_when_image=true \
    agent.max_text_history=4 agent.max_image_history=4
```

### Launching a matrix of experiments

`launch/launch.py` runs a cross-product of `{env, model}` cells as parallel
subprocesses, applying the tuned per-cell overrides shown above. This is the
easiest way to fan out over several ARC-AGI games or environments at once.

```bash
# Preview the commands without running them
uv run launch/launch.py --log-dir logs/matrix --dry-run

# Run MiniHack + Autumn with a single model
uv run launch/launch.py --log-dir logs/matrix \
    --scripts eb_learn --envs minihack,autumn --models gemini-2.5-flash

# Run 5 ARC-AGI 3 games in parallel
uv run launch/launch.py --log-dir logs/matrix \
    --scripts eb_learn --envs arc_agi --models gemini-2.5-flash \
    --arc-games ls20,sp80,tn36,vc33,ft09 --parallel 5
```

Each cell writes to `<log-dir>/<timestamp>/<cell_name>/` with `stdout.log`,
`stderr.log`, and `cmd.txt` (the exact command that was run).

## 3. Question selection & perception improvement modes

At every step `explore.stepwise_eb_learn` decides (a) **which question to investigate
next** and (b) **how to fold what it learned back into the agent**. Both are
configurable via `eval.evolve.*` overrides; the defaults live in
`BALROG/balrog/config/config.yaml`.

### Improvement tracks and `perception_enabled`

Each improvement pass runs up to three "tracks":

- **Track 1a** — beliefs improvement from the recent steps (always on).
- **Track 1b** — perception improvement: generates/edits a Python *perception
  module* (code that post-processes raw observations). Only runs when
  `perception_enabled=true`.
- **Track 2** — QA-based improvement from answered questions.

```bash
# Beliefs-only (default): no perception code module is generated or applied;
# Track 1b is skipped and Track 2 runs beliefs-only.
eval.evolve.perception_enabled=false

# Enable the code perception module (Tracks 1a + 1b + 2).
eval.evolve.perception_enabled=true
```

When perception is on, `critical_transitions_enabled` and
`critical_id_min_for_perception` control which observations feed Track 1b: with
critical transitions enabled, the loop gates improvement and experiment
generation on a per-step LLM "is this a critical transition?" decision instead of
a fixed interval.

### Question scoring methods (`question_scoring_method`)

Controls how unanswered questions in the bank are ranked so the loop knows which
one is worth an experiment.

| Method | What it does |
|--------|--------------|
| `llm_trim` | An LLM trims/selects the question bank directly — no numeric scoring. Cheapest. |
| `b_diff_light` | Scores each unanswered question by **B-difference**: how much answering it would shift the agent's beliefs. "Light" is the cheaper projection. |
| `b_diff_full` | Same B-difference idea with the fuller, heavier scoring pass. |
| `theory_entropy` | Generates `num_theories` competing world-models, predicts each theory's YES/NO/UNKNOWN answer per question, and scores questions by the **mutual information** (expected info gain, in bits) between the answer and theory identity. Highest = best discriminator. |

The `theory_entropy` method has its own knobs:

| Override | Meaning |
|----------|---------|
| `eval.evolve.num_theories` | Competing world-models generated per selection point (default 5) |
| `eval.evolve.num_crux_questions` | Crux questions seeded into the bank to split theories |
| `eval.evolve.theory_weight_decay` | Prior over theories by rank; `1.0` = uniform |
| `eval.evolve.theory_gen_current_state_only` | If true, the theory generator sees only the current state, not recent history |
| `eval.evolve.num_theory_seed_questions` | Lever #1 MI-residual seeding: regenerate theories seeded with up to N all-UNKNOWN questions so the ensemble can model mechanisms no theory currently predicts; `0` disables |

There is also an experimental `theory_disagreement` method (Plan A): a
*persistent* theory ensemble drives **action selection** (it picks the most
discriminating action) and is reweighted from each theory's pre-registered
predictions every step (`theory_violation_penalty`, `theory_min_weight`,
`num_candidate_actions`, and the `exploit_*` explore→exploit switch). See
`multi_theory_exploration.py`.

### Experiment selection mode (`experiment_selection_mode`)

Once questions are scored, this controls how experiments are formulated:

| Mode | What it does |
|------|--------------|
| `single` | Score the questions, pick the single best, and formulate one experiment for it. |
| `score_topk` | Formulate experiments across many/all unanswered questions and let the scoring pass rank them. `score_topk_filter_questions=true` adds an LLM pre-filter; `experiment_scoring_max_concurrent` caps the fan-out. |

### Example: theory-entropy + single selection

This is the tuned setup used by `launch/launch.py`:

```bash
uv run python -m explore.stepwise_eb_learn \
    envs.names=arc_agi tasks.arc_agi_tasks=[ls20] \
    eval.evolve.question_scoring_method=theory_entropy \
    eval.evolve.experiment_selection_mode=single \
    eval.evolve.num_theories=5 \
    eval.evolve.num_crux_questions=5 \
    eval.evolve.perception_enabled=false \
    eval.evolve.critical_transitions_enabled=true
```

## 4. Visualizing logs with the viz module

The `viz/` module provides a local web viewer for `stepwise_eb_learn.py` runs.
It reads a log directory directly from disk and serves a frontend that shows the
learned beliefs, per-step details, trajectories, and the question/experiment
timelines.

### Local viewer (dynamic mode)

```bash
# Point it at a run's output directory
uv run viz/visualize_stepwise_eb_learn.py logs/dev/apr8/2026-04-08_16-48-43_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn

# Custom host/port, open the browser automatically
uv run viz/visualize_stepwise_eb_learn.py <log_dir> --port 8766 --open-browser
```

By default the server binds `127.0.0.1:8766`. The log directory is optional on
the command line — you can also paste a path into the browser UI.

### Static export (publishable snapshot)

To turn one run into a curated, committable snapshot (e.g. for GitHub Pages):

```bash
uv run viz/export_stepwise_eb_static.py <log_dir> \
    --run-id apr8-gemini-flash \
    --title "Apr 8 Gemini 2.5 Flash"
```

This writes a snapshot under `data/stepwise_eb_runs/<run-id>/` plus an
`index.json`. The committed static page (`viz/index.html`) loads one of these
curated runs from `data/`.

See `viz/README.md` for details on the module structure, the data flow, and how
to edit how individual tabs render (`viz/stepwise_eb_learn/app.js`).

## Project Structure

```
bai/
├── explore/                # Exploration/self-improvement package (run entry points with -m):
│   ├── explore.py          #   python -m explore.explore — main evolution loop
│   ├── stepwise_eb_learn.py#   python -m explore.stepwise_eb_learn — stepwise explore-and-build loop
│   ├── stepwise_b_learn.py, stepwise_explore.py, *_improve.py, improve.py, offline.py, ...
│   └── theory_exploration.py, multi_theory_exploration.py, question_scoring.py
├── autumn_env.py           # Autumn (AutumnBench) environment wrapper (root module)
├── arc_agi_env.py          # ARC-AGI 3 environment wrapper (root module)
├── rollout.py, llm_utils.py, run_utils.py, diff_utils.py, goal_prompts.py  # shared root modules
├── setup.sh                # One-shot bootstrap (submodules, deps, Autumn, protobuf, dataset)
├── launch/                 # Experiment-matrix launchers (env × model cells): launch.py, launch_ee.py, ...
├── eval_runners/           # Per-method artifact eval scripts (eval_stepwise_eb_artifacts.py, ...)
├── uncertainty/            # Question-uncertainty scoring (score_uncertainty.py, ...)
├── tests/                  # Pytest suite (uv run python -m pytest tests/)
├── BALROG/                 # Git submodule — BALROG benchmark framework + Hydra config
├── MARAProtocol/           # Git submodule — AutumnBench harness + protobuf
├── Autumn.cpp/             # Git submodule — Autumn language interpreter (built locally)
├── viz/                    # Local web viewer + static exporter for eb_learn runs
├── curated/                # Hand-written beliefs and perception modules
├── scripts/                # Evaluation, simulation, play/replay utilities
├── archive/                # Dormant/superseded prototypes kept for reference
└── logs/                   # Output from runs
```
