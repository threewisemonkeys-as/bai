# Multi-horizon planning task — specification

Reference for the WorldCoder (program WM) vs NLWM (language WM) comparison.
Implementation: `offline_learning/scripts/eval_multistep_fd_plan.py`.

---

## 1. What the task is

A **window** is a tuple sampled from a verified Autumn drive:

```
w = (drive d, step t, horizon h)
  context   : up to K=9 preceding (grid, action) pairs, ending at step t-1
  start     : s_t     = grid at step t
  goal      : s_{t+h} = grid at step t+h
  truth     : a_t .. a_{t+h-1}, the h logged actions that actually produced the goal
```

Two sub-tasks are scored on every window.

### 1a. `msfd` — multi-step forward dynamics (secondary)

Given the context, `s_t`, **and the true h actions**, predict the state after all h of
them. Scored by exact match against the target, plus a partial-credit `textdiff_delta_f1`.

This is *not* planning — the action sequence is handed over. It measures whether the world
model can simulate h steps forward.

### 1b. `plan` — goal-conditioned planning (**the headline metric**)

Given the context, `s_t`, and the **goal** `s_{t+h}` (but **not** the true actions), emit a
fully-parameterized action sequence of length ≤ `cap` (`cap = h` by default, `--plan-cap N`
overrides).

Scoring is **execution in the real Autumn engine**, not self-report:

1. Reset the engine at the drive's seed, replay `d`'s logged actions for steps `0..t-1`.
2. Execute the emitted plan from there, action by action (`canon_click` de-canonicalized at
   the `env.step` boundary).
3. **success** ⟺ the rendered grid after the plan's **final** action equals the recorded goal
   grid, exactly.
4. `reached_at` also records the first step at which the goal was hit → an **ever-hit**
   variant exists in the rows. Use one or the other consistently; the online-plan eval
   compares against ever-hit ([[online-plan-eval]]).

Shortcuts count. A plan shorter than `h` that lands on the goal grid succeeds. A plan longer
than `cap` is rejected as `budget-exceeded` before execution.

### Window admission filters (what makes the task non-trivial)

From all `(d, t)` with h complete actions and h+1 complete grids, shuffled uniformly, keep
the first `--windows 10` per (game, h) that survive:

- **static excluded** — `s_t == s_{t+h}` is dropped. Emitting nothing can never score.
- **noop-solvable excluded** — `noop^h` is *executed in the engine*; if it reproduces the
  goal, the window is dropped. Passive drift alone can never score.

These two filters are the reason the metric is meaningful; they are also why sampling stats
(`candidates / static / noop_solvable / sampled`) are recorded per game per horizon.

### Floor and ceiling

- **Ceiling = 1.0 by construction.** The logged actions reach the goal — every drive is
  engine-verified frame-by-frame, so a perfect planner scores 1.0.
- **Floor = `rand`.** 3 uniformly random length-h plans per window (verb uniform over the
  game's verb list, then uniform cell for `click`), executed in the engine, averaged.
  The RNG is keyed `msplan-rand:{seed}:{game}:{h}:{drive}:{t}`, so the floor is **identical
  across runs** that share the window set — it is a per-window property, not a per-arm one.

---

## 2. The arms

| arm | world model | how a plan is produced | what it sees as the goal |
|---|---|---|---|
| `raw` | none | LLM prompted on raw canonical grids | raw goal grid |
| `learned` (NLWM) | perception `P` + beliefs `B` | LLM prompted on `P(·)` features + `B` | **`P(s_{t+h})` — features, not the grid** |
| `program` (WorldCoder) | `transition(prev, grid, action) -> grid` | **search** over `T̂` (no LLM) | raw goal grid |
| `hybrid` | `T̂` as verifier | rolls out each LLM plan + the search plan under `T̂`, picks a candidate `T̂` says reaches the goal (shortest wins) | — |

All arms are scored by the same in-engine execution rule.

### `program` search (`program_runtime.plan_search`)

- Action universe = non-click verbs + **every cell** as `click r c`. 16×16 games → 261 (or 257
  where the only non-click verb is `noop`); n2ntd 12×12 → 149.
- Clicks are ordered goal-differing-cells → non-background → rest (a mild goal-informed
  ordering advantage over the LLM arms).
- BFS if |universe| ≤ 8, otherwise **beam = 64** ranked by `cell_f1` to goal.
- Dedup on `(canonical grid, depth)`; depth stays in the key because passive/periodic
  dynamics make the same grid at different times non-equivalent.
- **Node budget = 5000 `T̂` calls per window**, memoized on `(grid, history, action)`.
- Returns the action list, or `None` = `no-plan-found` (unreachable *or* budget exhausted).

### `learned` prompt asymmetry — must be stated in any writeup

The NLWM planner is given `P(goal)` while being graded on exact **raw-grid** equality. If `P`
is lossy the goal it is aiming at is under-specified relative to the success criterion. This
handicaps `learned` and advantages `program`/`raw`, which see the literal goal grid.

Conversely, `program` searches with full knowledge of the goal grid *and* a goal-ordered
action universe.

### FD comparability

`msfd` exact-match is **not comparable across arms**: `raw` and `program` predict canonical
grids, `learned` predicts `P(x_{t+h})` in feature space. Only `program` vs `raw` is an
apples-to-apples FD-exact comparison. `stale_exact` (score of copying `s_t`) is stored per
row as the do-nothing floor.

### z-blindness check (validity gate for `learned`)

If `P(s_t) == P(s_{t+h})` the goal is invisible in feature space — `learned` FD is a fake
ceiling and `learned` planning is structurally 0. Check `z_t == z_goal` fraction and
`perception_errors` before trusting any `learned` number. **The batch3 run is clean: 0/200
z-blind windows, 0 perception errors.**

---

## 3. Data

### 3a. Evaluation data — the drives windows are cut from

Two sourcing modes, chosen per game:

- **recorded (`--env-seed 0`)** — the verified seed-0 source drives behind the test50 pools,
  resolved via `rescore_test50_id_sim.resolve_sources`. Every drive is re-verified
  frame-by-frame against a seed-0 engine reset before use (`SeqSim.problems`); the run aborts
  on any mismatch.
- **generated (`--gen-games G --gen-seed N`)** — fresh random-policy rollouts from the real
  engine under unseen seeds (6 drives × 40 steps, per-drive seeds `N..N+5`). Used for the two
  **truly stochastic** games, whose seed-0 recorded drives are degenerate
  ([[seed0-degenerate-data-gen]], [[autumn-games-uncertainty-taxonomy]]).

Current 5-game configuration (`seed=1`, `K=9`, `h ∈ {1,2,4,8}`, 10 windows per game per h =
**200 windows**):

| game | grid | verbs | drive source | drives |
|---|---|---|---|---|
| bt3gb | 16×16 | up/down/left/right/noop/click | recorded seed-0 | `fulltraj_context_remaining_autumn_inputs/bt3gb/test50_sources/drive{A,B,C}` |
| dq8gc | 16×16 | up/down/left/right/noop/click | recorded seed-0 | `…/dq8gc/test50_sources/drive{A..F}` |
| n2ntd | 12×12 | up/down/left/right/noop/click | recorded seed-0 | `…/n2ntd/test50_sources/drive{A,B,C,E,F,G}` |
| 83wkq | 16×16 | noop/click | **generated, seed 700001–700006** | random-policy, 40 steps |
| s2kt7 | 16×16 | noop/click | **generated, seed 700001–700006** | random-policy, 40 steps |

Window sampling is RNG-keyed `msplan:{seed}:{game}:{h}` → **the window set is reproducible and
shared across runs with the same config.** Verified: `multistep_wc.json` and
`multistep_batch3.json` have byte-identical `(drive, t, h)` lists for all 5 games.

### 3b. Training data — what each world model was fit on

**This is where the two arms are *not* matched.** Per game (seed 1, `K=9`):

| game | NLWM (`rexpure`, `logs/batch3_consolidated`) | WorldCoder (`logs/wc_seed1_consolidated`) |
|---|---|---|
| bt3gb | `clean_data3/bt3gb/{train,train2}` + `old_test10` → pool 62, **train 30** | `clean_data3/bt3gb/train` → pool 20, **train 14 / val 6** |
| dq8gc | `clean_data3/dq8gc/train` + `old_test10` → pool 30, **train 30** | same sources → pool 30, **train 20 / val 10** |
| n2ntd | `clean_data3/n2ntd/{train,train2}` + `old_test10` → pool 61, **train 30** | `train` + `old_test10` → pool 30, **train 20 / val 10** |
| 83wkq | `clean_data3/83wkq/train` + `old_test10` → pool 30, **train 30** — *seed-0 data* | `clean_data3/83wkq_seed1/train` → pool 104, **train 74 / val 30** — *regenerated* |
| s2kt7 | `clean_data3/s2kt7_seed5/train` (leak-fixed) → pool 104, **train 30** | `clean_data3/s2kt7_seed1/train` → pool 104, **train 74 / val 30** |

Additional config asymmetries:
- WC used `--collapse-action-params` on bt3gb/dq8gc/n2ntd (click params collapsed in its
  training ID signal); NLWM did not.
- `rex_pure` uses train == scoring set with no val; WC holds out a val split.
- The two arms saw **different data for both stochastic games** (83wkq seed-0 vs seed1-regen;
  s2kt7 seed5 vs seed1) and different pool sizes for bt3gb/n2ntd.

**Consequence:** the current head-to-head is *artifact vs artifact*, not *method vs method*.
Any claim of the form "programs are worse planners than language models" needs either a
matched-training-pool rerun or an explicit statement of this asymmetry.

### 3c. Decoder (only the LLM arms)

`raw` and `learned` share one decoder per run: `openai/gpt-oss-120b` (openrouter), 16-way
concurrency, up to 4 retries per call. A `deepseek-v4-pro` (`reasoning=low`) variant exists at
`logs/batch3_consolidated/multistep_batch3_dsv4pro.json` on the same windows. `program` and
`hybrid` make **zero LLM calls** at decision time.

---

## 4. Current results (already run, same 200 windows)

Planning success, mean over the 5 games. `program` numbers from
`logs/wc_seed1_consolidated/multistep_wc.{json,md}` (run with `--no-llm`, so its `raw`/
`learned`/`hybrid` columns are dead); NLWM from `logs/batch3_consolidated/multistep_batch3.*`.

| arm | h=1 | h=2 | h=4 | h=8 |
|---|---|---|---|---|
| rand (floor) | 0.15 | 0.10 | 0.09 | 0.01 |
| raw | 0.62 | 0.34 | 0.18 | 0.06 |
| **learned (NLWM)** | **0.94** | **0.62** | **0.36** | **0.12** |
| **program (WC)** | 0.48 | 0.40 | 0.20 | 0.02 |

But the aggregate hides the mechanism. Decomposed into *does the arm emit a plan at all* and
*is that plan right*:

| arm | h | emit rate | success | **P(success \| emitted)** |
|---|---|---|---|---|
| learned | 1 / 2 / 4 / 8 | 100% / 96% / 86% / 64% | 0.94 / 0.62 / 0.36 / 0.12 | 0.94 / 0.65 / 0.42 / 0.19 |
| program | 1 / 2 / 4 / 8 | 68% / 52% / 24% / **2%** | 0.48 / 0.40 / 0.20 / 0.02 | **0.71 / 0.77 / 0.83 / 1.00** |

**WorldCoder's planning deficit is dominated by coverage, not per-plan quality** — it abstains
(`no-plan-found`) on 49/50 windows at h=8, and the plans it does emit succeed at a rate that
*rises* with horizon. Two caveats before quoting this, both detailed below: the two arms'
"emitted" events are not the same kind of event, and the NLWM's low emit rate at h≥4 is
substantially a parse bug rather than abstention.

### "Emitted" is not the same event across arms

- **`program`**: `plan_search` returned a sequence whose rollout **under its own model**
  reaches the goal — a self-verified commitment made before emission.
- **`learned`/`raw`**: the response merely **parsed** (a `<plan>` tag of valid, in-bounds
  actions, non-empty, ≤ cap). No verification whatsoever; the LLM has no way to abstain.

So `P(success | emitted)` conditions on different things per arm — `P(success | my model says
this works)` vs `P(success | the text parsed)`. The abstention-rate gap is real; the precision
comparison is softer than the raw table implies.

### HARNESS BUG — `empty-plan` is a grab-bag (biases the NLWM emit rate)

`validate._parse_tag` returns `""` (not `None`) when the tag is absent, so `parse_plan`'s
`if body is None: return "no-plan-tag"` branch is **unreachable** — a missing `<plan>` tag is
mislabelled `empty-plan`. Splitting the non-emitted NLWM rows by what actually happened:

| arm | h | non-emit | no `<plan>` tag (markdown `**Plan**` + fence) | blank response | genuinely empty |
|---|---|---|---|---|---|
| learned | 4 | 7 | 4 | 1 | 2 |
| learned | 8 | 18 | **15** | 1 | 2 |
| raw | 8 | 3 | 3 | 0 | 0 |

**15 of learned's 18 h=8 non-emissions are format non-compliance, not abstention** — one bt3gb
case contains a complete valid 8-action plan that was discarded on formatting. The `learned`
emit rate (and hence its `P(success|emitted)` denominator) is therefore contaminated.

*Fix:* add a fenced-block fallback to `parse_plan`, and distinguish missing-tag from
empty-body. Re-scoring is **LLM-free** — every response is stored verbatim in the run json, so
plans can be re-derived and re-executed in the engine at zero API cost.

### The search-budget confound (h=4, h=8; h=2 for the click-only games)

`no-plan-found` conflates "my model says unreachable" with "I ran out of nodes". Measured
`T̂` calls per window against the 5000 budget:

| game | h=1 | h=2 | h=4 | h=8 | windows hitting budget (of 10) |
|---|---|---|---|---|---|
| bt3gb | 262 | 1307 | 5004 | 5008 | h2 0 · h4 8 · h8 10 |
| dq8gc | 6 | 267 | 4185 | 5008 | h2 0 · h4 4 · h8 9 |
| n2ntd | 4 | 304 | 5004 | 5008 | h2 0 · h4 8 · h8 10 |
| 83wkq | 4 | 5002 | 5004 | 5008 | h2 7 · h4 9 · h8 10 |
| s2kt7 | 258 | 5002 | 5004 | 5008 | h2 9 · h4 10 · h8 10 |

- **h=1**: never binds (full depth-1 expansion ≤ 261 calls). `no-plan-found` here is a
  **genuine, complete model failure** — trustworthy.
- **h=2**: binds only for the click-only games (83wkq, s2kt7 — universe 257 with no cheap
  movement branching). The movement trio's h=2 numbers are honest.
- **h=4 / h=8**: binds almost everywhere. **These program numbers are search-limited and are a
  lower bound on what the program world model supports.**

Sweeping `--search-budget` / `--search-beam` is the cheapest way to separate model error from
search error (the arm is LLM-free, so it costs only wall-clock).

### Other caveats

- `hybrid` is currently identical to `program` — the WC run was `--no-llm`, so there were no
  LLM candidate plans to verify. A real hybrid needs **one process** with both
  `--program-artifact` and the LLM enabled.
- s2kt7 has a random floor of 0.43/0.40/0.23/0.07 — essentially no signal for any arm.
- 83wkq and s2kt7 are truly stochastic; a deterministic `transition` program cannot represent
  them (83wkq's shipped program scores val 0.28, below the 0.50 identity floor). Report the
  **deterministic trio {bt3gb, dq8gc, n2ntd}** separately from the 5-game mean.
- FD-exact: `program` 0.48/0.46/0.34/0.18 vs `raw` 0.34/0.22/0.06/0.00 — the program is the
  far better simulator on the comparable metric, which is consistent with the precision
  finding above.

---

## 5. Reproducing / extending

```bash
# NLWM arms (LLM) on the canonical 5-game window set
uv run python offline_learning/scripts/eval_multistep_fd_plan.py \
  --id-json logs/batch3_consolidated/id_multistep_batch3.json \
  --model openai/gpt-oss-120b --horizons 1,2,4,8 --windows 10 \
  --env-seed 0 --gen-games s2kt7,83wkq --gen-seed 700001 \
  --out logs/<run>/multistep

# add the WorldCoder arm (same flags + --program-root) -> program + a REAL hybrid
  --program-root logs/wc_seed1_consolidated --search-beam 64 --search-budget 5000
```

`--seed 1 --context-k 9 --windows 10 --env-seed 0 --gen-games s2kt7,83wkq --gen-seed 700001`
is what pins the window set. Change any of those and the arms stop being comparable.
