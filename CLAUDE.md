# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BAI is an LLM agent evaluation and self-improvement framework built on top of BALROG (Benchmarking Agentic LLM and VLM Reasoning On Games). It runs LLM agents in game environments (NetHack/MiniHack, TextWorld, BabyAI, Crafter, BabaIsAI), then uses an evolutionary "explore" loop to iteratively improve the agent's beliefs (instruction prompts) and perception module (Python code that processes observations).

## Commands

```bash
# Install dependencies (uses uv, not pip)
uv sync

# Run the main explore/evolution loop (explore/ is a package: run entry points with -m)
uv run python -m explore.explore

# Run a single evaluation (with Hydra overrides)
uv run rollout.py

# Evaluate all steps of an explore run
uv run scripts/eval_explore.py <explore_run_dir> --num-episodes 20

# Interactive play in terminal
uv run scripts/play.py MiniHack-Quest-Easy-v0

# Replay a saved trajectory
uv run scripts/replay.py trajectories/<task>/<file>.json

# Override config with Hydra syntax
uv run python -m explore.explore client.model_id=anthropic/claude-sonnet-4-20250514 envs.names=minihack eval.evolve.num_steps=10

# The stepwise learners are also modules under the explore package, e.g.
uv run python -m explore.stepwise_eb_learn envs.names=autumn tasks.autumn_tasks=[ice]
```

## Architecture

### Core Loop (`explore/explore.py`)

The main evolution loop (`explore/explore.py`) runs iterative self-improvement steps:
1. **Rollout** — runs the LLM agent in game environments via `rollout.py` (uses `balrog.evaluator.EvaluatorManager` + `balrog.agents.AgentFactory`)
2. **Improve** — analyzes trajectories and proposes improved beliefs/perception via `explore/improve.py` (LLM-based generation using LiteLLM)
3. **Evaluate** — tests improvements against baselines

Each step produces a directory under `logs/` with beliefs.txt, perception.py, trajectories, and summaries.

### Key Files

- **`explore/explore.py`** — Main evolution/explore loop orchestrator. Uses Hydra for config. Run with `python -m explore.explore`.
- **`explore/improve.py`** — LLM-powered improvement: generates candidate beliefs, experiments, analyzes trajectories, scores candidates.
- **`rollout.py`** — Runs agent evaluation episodes (stays at the repo root). `one_step()` takes instruction + perception and returns summary stats. `run_explore_rollouts()` parallelizes with ProcessPoolExecutor.
- **`llm_utils.py`** — Shared utilities for LLM interaction (build prompts, extract XML-tagged responses).

### BALROG Submodule (`./BALROG/`)

A git submodule (editable install via `uv`). Key components:
- `balrog/agents/` — Agent implementations (naive, chain-of-thought, robust_cot, few_shot, etc.)
- `balrog/environments/` — Environment wrappers (nle, minihack, textworld, babyai, crafter, babaisai)
- `balrog/evaluator.py` — `EvaluatorManager` runs episodes and collects results
- `balrog/config/config.yaml` — Default Hydra configuration (agent type, LLM client, env settings, eval params)
- `balrog/prompt_builder/` — Constructs prompts for agents

### Configuration

Uses Hydra with config at `BALROG/balrog/config/config.yaml`. Key config sections:
- `agent` — type (robust_cot), history settings
- `client` — LLM provider, model_id, temperature, max_tokens
- `envs` — which environments and their kwargs
- `eval` — output_dir, num_workers, num_episodes, evolve settings
- `eval.evolve` — num_steps, num_experiments, improve_mode ("both"=beliefs+perception)

### Supporting Directories

- `explore/` — The exploration / self-improvement package: the main loop (`explore.explore`), the stepwise learners (`explore.stepwise_eb_learn`, `explore.stepwise_b_learn`, `explore.stepwise_explore`, + their `_improve`/`_oracle` modules), the `explore.improve` engine, `explore.b_learn_improve`/`explore.mixed_improve`, the alt agents (`explore.simple_stepwise`, `explore.openhands_stepwise`), `explore.offline`, and the theory modules (`explore.theory_exploration`, `explore.multi_theory_exploration`, `explore.question_scoring`). Run entry points as modules: `uv run python -m explore.<module>`.
- `autumn_programs/` — The 55 Autumn `.sexp` sources from `autumn_programs_55.zip` (tracked). The harness only reads `MARAProtocol/python_examples/autumnbench/example_benchmark/programs/`, which is gitignored in the submodule, so run `uv run tools/install_autumn_programs.py` (defaults to the zip-sourced games in `experimental_plan.md`; `--check` reports drift) after a clone or dataset re-download.
- `curated/beliefs/` — Hand-written belief/instruction prompts for specific tasks
- `curated/perc/` — Hand-written perception modules
- `scripts/` — Evaluation, visualization, and utility scripts
- `launch/` — Experiment-matrix launchers (`launch.py`, `launch_baselines.py`, `launch_ee.py`, `launch_eval.py`, `launch_g.py`) that fan out `{env, model}` cells as subprocesses. Invoke from the repo root, e.g. `uv run launch/launch.py`.
- `eval_runners/` — Per-method artifact eval scripts (`eval_stepwise_eb_artifacts.py`, `eval_simple_artifacts.py`, `eval_openhands_artifacts.py`) driven by `launch/launch_ee.py`.
- `uncertainty/` — Question-uncertainty scoring over run dirs (`score_uncertainty.py`, `score_conditional_uncertainty.py`, `variance_conditional_uncertainty.py`).
- `tests/` — Pytest suite; run with `uv run python -m pytest tests/`.
- `archive/` — Dormant/superseded prototypes kept for reference, not part of the active pipeline (`agentic/` alternative agents, `wm/` world-model, `qa_codegen/`)

Note: the project **installs itself editable** — `[build-system]` + `[tool.setuptools]` (`packages = ["explore"]` + a `py-modules` list) in `pyproject.toml`, applied by `uv sync` — so both the `explore` package and the root modules are importable venv-wide with no `sys.path` shim. The exploration cluster is the `explore/` **package** (imported as `explore.<module>`, e.g. `from explore.stepwise_eb_learn import ...`); the shared libs (`llm_utils.py`, `run_utils.py`, `diff_utils.py`), env wrappers (`autumn_env.py`, `arc_agi_env.py`, ...), `goal_prompts.py`, `rollout.py`, and `get_stats_g.py` stay as flat top-level modules at the repo root (imported bare). Because `explore/` contains an `explore.py` module (same name as the package dir), its entry points must be run with `python -m explore.<module>` (running a script by path from inside `explore/` would shadow the package). When you add a root module, add its stem to `py-modules`; a new `explore/` submodule needs no config change.

## Environment Variables

Uses `.env` file (loaded via python-dotenv). Requires API keys for LLM providers (OpenAI, Anthropic, Google, OpenRouter, etc.).
