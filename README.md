# BAI

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- API keys for at least one LLM provider (OpenAI, Anthropic, Google, or OpenRouter)

## Setup

```bash
# Clone with the BALROG submodule
git clone --recurse-submodules <repo-url>
cd bai

# Install all dependencies
uv sync

# Add your API keys to .env (already in .gitignore)
cp .env.example .env   # or create .env manually
# Required keys depend on which provider you use:
#   OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY, etc.
```

## Running

All commands use `uv run` to execute within the project's virtual environment. Configuration is managed by [Hydra](https://hydra.cc/) — you can override any setting with `key=value` arguments on the command line.

### Explore (self-improvement loop)

The main entry point. Runs iterative steps where the agent plays episodes, then an LLM analyzes the results and proposes improved beliefs and perception.

```bash
# Run the explore loop with defaults (uses config from BALROG/balrog/config/config.yaml)
uv run explore.py eval.mode=explore

# Specify model, environment, and number of evolution steps
uv run explore.py eval.mode=explore \
    client.model_id=anthropic/claude-sonnet-4-20250514 \
    client.client_name=openrouter \
    envs.names=minihack \
    eval.evolve.num_steps=10

# Resume a previous run
uv run explore.py eval.mode=explore eval.resume_from=logs/dev/mar04/run_name
```

Output goes to `logs/` with one subdirectory per step containing `beliefs.txt`, `perception.py`, trajectories, and summaries.

### Eval (single evaluation)

Run the agent on episodes without the improvement loop, optionally with custom beliefs and perception.

```bash
# Evaluate with default settings
uv run explore.py eval.mode=eval

# Evaluate with specific beliefs and perception from a previous explore run
uv run explore.py eval.mode=eval \
    eval.beliefs_path=logs/dev/mar04/run/step_5/explore/beliefs.txt \
    eval.perception_path=logs/dev/mar04/run/step_5/explore/perception.py \
    eval.num_episodes=20

# Use curated (hand-written) beliefs
uv run explore.py eval.mode=eval \
    eval.beliefs_path=curated/beliefs/minihack_quest_manual.txt \
    eval.perception_path=curated/perc/perc.py
```

### Evaluate an explore run

After an explore run finishes, evaluate each step's beliefs and perception side-by-side to see which step performed best.

```bash
uv run scripts/eval_explore.py logs/dev/mar04/some_run --num-episodes 20
uv run scripts/eval_explore.py logs/dev/mar04/some_run --steps 5-15 --max-workers 4

# Pass extra Hydra overrides after --
uv run scripts/eval_explore.py logs/dev/mar04/some_run -- client.model_id=openai/gpt-4o
```

## Key Configuration Options

Override any of these on the command line (Hydra syntax). Defaults are in `BALROG/balrog/config/config.yaml`.

| Setting | Default | Description |
|---------|---------|-------------|
| `client.client_name` | `openrouter` | LLM provider (`openai`, `openrouter`, `gemini`, `claude`) |
| `client.model_id` | `google/gemini-2.5-flash` | Model to use |
| `client.generate_kwargs.temperature` | `0.0` | Sampling temperature |
| `envs.names` | `nle` | Environment (`nle`, `minihack`, `textworld`, `babyai`, `crafter`, `babaisai`) |
| `eval.num_episodes` | `20` | Episodes per evaluation |
| `eval.num_workers` | `32` | Parallel rollout workers |
| `eval.mode` | `eval` | `eval` (single run) or `explore` (self-improvement loop) |
| `eval.evolve.num_steps` | `20` | Number of evolution steps |
| `eval.evolve.num_experiments` | `3` | Experiments per step |
| `eval.evolve.improve_mode` | `both` | What to improve: `both`, `beliefs`, or `perception` |
| `eval.evolve.two_step` | `true` | Two-phase steps (baseline then explore) |

## Scripts

### Play

Interactively play a MiniHack/NetHack environment in the terminal, seeing the same observation the agent sees.

```bash
uv run scripts/play.py                          # default: MiniHack-Quest-Easy-v0
uv run scripts/play.py MiniHack-Room-5x5-v0     # specify a task
uv run scripts/play.py --skip-more               # auto-handle --More-- prompts
uv run scripts/play.py --seed 42                 # set a seed for reproducibility
```

Trajectories are saved as JSON to `trajectories/<task>/` by default. Disable with `--no-log`.

### Replay

Replay a saved trajectory log in the terminal with the same two-pane display.

```bash
uv run scripts/replay.py trajectories/MiniHack-Quest-Easy-v0/<file>.json
uv run scripts/replay.py trajectories/MiniHack-Quest-Easy-v0/<file>.json --speed 1.0  # auto-play at 1s/step
```

Controls during replay:
- **Enter** — next step
- **b** — go back one step
- **g N** — jump to step N
- **a** — auto-play (0.5s/step, Ctrl-C to pause)
- **q** — quit

## Project Structure

```
bai/
├── explore.py          # Main entry point — explore loop and eval mode
├── improve.py          # LLM-based trajectory analysis and belief/perception improvement
├── rollout.py          # Runs agent episodes via BALROG's evaluator
├── llm_utils.py        # Shared LLM utilities (prompt building, XML parsing)
├── BALROG/             # Git submodule — BALROG benchmark framework
│   └── balrog/
│       ├── agents/     # Agent implementations (naive, chain-of-thought, robust_cot, etc.)
│       ├── environments/   # Environment wrappers (nle, minihack, textworld, etc.)
│       ├── evaluator.py    # Episode runner
│       └── config/         # Default Hydra config
├── curated/            # Hand-written prompts and perception modules
│   ├── beliefs/        # Instruction prompts for specific tasks
│   └── perc/           # Perception modules
├── scripts/            # Evaluation, visualization, play/replay utilities
└── logs/               # Output from explore and eval runs
```
