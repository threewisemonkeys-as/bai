# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BAI is an LLM agent evaluation and self-improvement framework built on top of BALROG (Benchmarking Agentic LLM and VLM Reasoning On Games). It runs LLM agents in game environments (NetHack/MiniHack, TextWorld, BabyAI, Crafter, BabaIsAI), then uses an evolutionary "explore" loop to iteratively improve the agent's beliefs (instruction prompts) and perception module (Python code that processes observations).

## Commands

```bash
# Install dependencies (uses uv, not pip)
uv sync

# Run the main explore/evolution loop
uv run explore.py

# Run a single evaluation (with Hydra overrides)
uv run rollout.py

# Evaluate all steps of an explore run
uv run scripts/eval_explore.py <explore_run_dir> --num-episodes 20

# Interactive play in terminal
uv run scripts/play.py MiniHack-Quest-Easy-v0

# Replay a saved trajectory
uv run scripts/replay.py trajectories/<task>/<file>.json

# Override config with Hydra syntax
uv run explore.py client.model_id=anthropic/claude-sonnet-4-20250514 envs.names=minihack eval.evolve.num_steps=10
```

## Architecture

### Core Loop (`explore.py`)

The main evolution loop (`explore.py`) runs iterative self-improvement steps:
1. **Rollout** — runs the LLM agent in game environments via `rollout.py` (uses `balrog.evaluator.EvaluatorManager` + `balrog.agents.AgentFactory`)
2. **Improve** — analyzes trajectories and proposes improved beliefs/perception via `improve.py` (LLM-based generation using LiteLLM)
3. **Evaluate** — tests improvements against baselines

Each step produces a directory under `logs/` with beliefs.txt, perception.py, trajectories, and summaries.

### Key Files

- **`explore.py`** — Main evolution/explore loop orchestrator. Uses Hydra for config.
- **`improve.py`** — LLM-powered improvement: generates candidate beliefs, experiments, analyzes trajectories, scores candidates.
- **`rollout.py`** — Runs agent evaluation episodes. `one_step()` takes instruction + perception and returns summary stats. `run_explore_rollouts()` parallelizes with ProcessPoolExecutor.
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

- `curated/beliefs/` — Hand-written belief/instruction prompts for specific tasks
- `curated/perc/` — Hand-written perception modules
- `agentic/` — Alternative agent implementations (base_agent, hprop_agent, simple_agent)
- `scripts/` — Evaluation, visualization, and utility scripts

## Environment Variables

Uses `.env` file (loaded via python-dotenv). Requires API keys for LLM providers (OpenAI, Anthropic, Google, OpenRouter, etc.).
