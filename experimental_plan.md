# Experimental Plan

We want to evaluate our approach of learning from offline interaction data from games.
Our approach currently uses a text optimisation algorithm (Rex) to learn beliefs (a piece of text describing how the game works) and perception (python code that parses raw game state and outputs natural language strings describing the state).
We want to present our idea as a scientific paper, comparing it against baselines, exploring different tasks and performing various ablations.
In this document we outline the details of our plan for experiments.

Status legend used throughout: **[X]** done, **[P]** partial, **[ ]** not started. Status was last reconciled against the repo on 2026-09-04.


## Tasks

After a learning phase, where our methods (and baselines) learn from training data, we evaluate them on various planning problems in the games -

- Curated Planning Tasks: Accomplishing semantically useful goals in each game. The selected 15-game set is stored in `logs/2026-08-29/planning_v2/problems.json` (86 problems: 28 retained from the original four selected games and 58 new problems for the other 11; build and audit procedure in `notes/planning-problems-15-games.md`). The set was promoted from 83 to 86 rows on 2026-09-01 under schema v2.2, when `f5w3n/shoot-enemy` was retired as a one-tick render-occlusion exploit and replaced by the two invader-count checkers (`exactly-one`, `exactly-three`, three seeds each). The historical five-game v1 set remains in `logs/2026-08-18/curated/problems.json`.
  - Goal as required state: Tasks are specified by a start state and end state and task completion is exact match with the end state. Start state = (program, seed, action prefix); there is no interpreter snapshot, so every state is addressed by replay.
  - Goal as NL statement: Tasks are specified by a start state and a natural language statement indicating the goal. Completion is checked by ordinary Python trajectory programs with the legacy `check(grids, actions) -> bool` interface. `offline_learning/planning_nl_goals.py` provides one registry for all 86 selected-game problems (68 distinct checker programs; multi-seed stochastic rows share semantics deliberately) and adapts the original programs in `offline_learning/nl_goals.py`; the evaluator no longer interprets declarative predicate payloads.
- Autumn Tasks: CD/MFP/Masked Planning. These exist only for the 20 AutumnBench programs (60 tasks, `MARAProtocol/.../example_benchmark/`); games that come from `autumn_programs_55.zip` have none and need them authored. Known caveats from the 2026-08-22 audit: AutumnBench planning is scored on any-step reach (12/21 goals reachable by chance; zero floor for grow/ice/disease/sand), CD reward is a ratio with a per-game ceiling (disease 0.71), ants' CD never renders differently at seed 0, mario's CD crashes if the first action is a click.

Every curated problem carries both goal representations. Evaluators must explicitly select `--goal-presentation frame` or `--goal-presentation nl`; there is no per-problem preferred or declared mode.
- Offline: Represent the start and the goal to the method and require it to generate a plan that goes from the start to the goal. (`offline_learning/scripts/eval_curated_plan.py`)
- Online: At each state, we present the current state and the goal to the approach and ask it to generate a plan. However, we only execute the first action of the plan, after which we present the model with the updated state and the goal and ask it to again generate a plan. (`offline_learning/scripts/eval_curated_online.py`)

Two findings from the first NL-goal run (`notes/nl-goal-planning-set.md`) that the final design must account for: (1) with a frame goal our method receives the goal through its own perception module (`_z_goal`); an NL goal deletes that channel, so `lmwm` drops while `raw` does not; (2) online replanning helps on frame goals (a progress signal) but not on NL goals. Both are results in their own right, but the NL condition should give the learned model some way to ground the goal.

### Game catalogue

Regular [16]: Games amenable to NL expressible goals. Usually have well defined semantics and behaviour.
- Easy [6]:
  - lights [16x16]
  - rink [28x28]
  - arc_slack [16x16]
  - paint [16x16]
  - egg [16x16]
  - chomp [7x7]
- Moderate [5]:
  - ice [16x16]
  - carrace [13x13]
  - waterplug [16x16]
  - disease [16x16]
  - grow [16x16]
- Hard [5]:
  - mario [12x12]
  - chinese_checkers [8x8]
  - sand [10x10]
  - buoyancy [7x7]
  - pacman [10x10]

- Stochastic subset within regular [9]: Some aspects of the game (such as reset states, reshuffles) are stochastic, hence frame goals do not work unless seed specific. But overall the game still has semantic goals
- Easy [3]
  - ants [16x16]
  - colour_lines [10x10]
  - SET [20x20]
- Moderate [3]
  - diffusion [9x9]
  - dino [20x20]
  - snake [16x16]
- Hard [3]
  - block_breaker [13x13]
  - tetris [16x16]
  - space_invaders [16x16]


Some games have simple dynamics, could evaluate whether we learn abstract meaning which is not apparent [6]:
- logic_gates [24x24]
- magnets [16x16]
- gameOfLife [16x16]
- balloon [16x16]
- twiddle [3x3]
- tictactoe [5x5]

Some games are so stochastic that (1) frame based goals would be impossible and (2) NL goals would be degenerate (such as spawn N particles) [5]:
- kaleidoscope [25x25]
- particles [16x16]
- chaos_game [50x50]
- crystallization [10x10]
- minesweeper [10x10]

Hard to represent for planning [2]:
- scotland_yard [20x20]: maze game where parsing, representing and then planning in maze could be hard
- ricochet_robots [24x24]: similar to scotland yard, representation for llm planning is hard, and llm planning itself is hard

Long horizon memory required [5]:
- coins [16x16]: counting number of coins collected and number of bullets shot
- gravity [17x17]: keeping track of off screen particles
- balls [12x12]: is the wall active or not?
- charge [7x7]: how many timesteps since last click (counting charge)
- lightning_rod [9x9]: charge on cloud is number of timesteps

Too simple or not semantically meaningful [8]:
- nim [17x17]
- wind [17x17]
- hatch [16x16]
- blicket [11x11]
- bottle [11x11]
- lock [9x9]
- bbq [7x7]
- peg_soltaire [5x5]

### Selected games

For initial experimentation, we select the following [15] games from the Regular, Stochastic and Abstract groups. The table records, per game, where the program comes from (AutumnBench id or the 55-game zip), and what infrastructure exists today. "data" = training drives + test50 under `offline_learning/clean_data3/` (the artificial corpus; NOTE the reference runs since 2026-08-11 train on human play instead, see "human"); "human" = the game is in the human study (`basis_data.zip`, 43 games x 3 task types x 20 sessions), which is what `offline_learning/human_data/<game>/informative_unified` (60 train / 50 test targets) is built from — H = `informative_unified3` pool built 2026-08-24 (all 15), — = not in the study; "curated" = rows in the 83-problem v2.2 artifact; "NL" = registered Python goal programs in `planning_nl_goals.py`; "CD/MFP" = AutumnBench tasks.

| group | game | grid | source id | data | human | curated | NL | CD/MFP | notes |
|---|---|---|---|---|---|---|---|---|---|
| Regular / Easy | paint | 16 | EAHCW | X | H | X (4) | X | X | arrows set the paint colour (hidden until the next click); replaced lights 2026-08-23 |
| Regular / Easy | egg | 16 | zip | | H | X (5) | X | | replaced rink 2026-08-23. Deterministic; arrows move a 21-cell egg while gravity is off, clicking the button toggles gravity (button colour shows it) and latches the egg's height — above row 10 it shatters into gold pieces that fall (hidden: latched height). **Gray background** (relies on the background generalisation) |
| Regular / Moderate | ice | 16 | ice (== BT3GB) | X | H | X (8) | X | X | reference game |
| Regular / Moderate | disease | 16 | DQ8GC | X | H | X (7) | X | X | hidden state (click invisible for >= 6 frames) |
| Regular / Moderate | grow | 16 | 7XF97 | X | H | X (5) | X | X | agent can walk off-grid |
| Regular / Hard | mario | 12 | N2NTD | X | H | X (7) | X | X | white background; CD first-click crash; enemy `movingLeft` hidden |
| Regular / Hard | sand | 10 | VA6FQ | X | H | X (7) | X | X | clean on all 5 diagnostics |
| Stochastic / Easy | ants | 16 | S2KT7 | X (seed 5) | H | X (6) | X | X | seed-0 data is degenerate; CD never renders at seed 0 |
| Stochastic / Easy | colour_lines | 10 | zip | | H | X (3) | X | | random ball injected on every empty click; human pipeline dry run: 249 train / 227 test informative candidates (need 60/50) |
| Stochastic / Easy | SET | 20 | zip | | H | X (3) | X | | replaced tetris 2026-08-23. Atlas rated it Tier X (deck reduced to 3 card types, random refills, template cards rendered off-screen) — run the engine diagnostics before investing; heaviest obs of the set (~123 cells) |
| Stochastic / Moderate | diffusion | 9 | zip | | H | X (11) | X | | pure random walk every tick; distributional scoring only |
| Stochastic / Moderate | dino | 20 | zip | | H | X (3) | X | | bird row random per wrap; absorbing death |
| Stochastic / Hard | space_invaders | 16 | F5W3N | X | H | X (9) | X | X | |
| Abstract | logic_gates | 24 | zip | | H | X (4) | X | | **USABLE WITH FIXES** (`notes/testbed_investigations/logic_gates.md`): 13 frames / 36 transitions, not 4 states; 8 clickable cells of 576 → authored drives; a 51-action drive covers every transition, so the split must hold out the both-switches-on state (only one where OR ≠ XOR); L4 needs `logic_gates_v2` |
| Abstract | magnets | 16 | 7WWW9 | X | H | X (4) | X | X | replaced balloon 2026-08-23. Arrow-move a 2-cell magnet; opposite poles 2 cells apart pull one step, like poles adjacent cancel the move; edges do NOT block (slides off-grid, contra its dynamics.txt); atlas dropped it for a 40% planning floor — ladder needs deeper goals |

Facts that constrain the work on the 6 zip-sourced games (egg, colour_lines, diffusion, dino, logic_gates, SET): the harness loads programs only from `example_benchmark/programs/` (gitignored inside the MARAProtocol submodule), so the zip sources are tracked under `autumn_programs/` and installed there by `tools/install_autumn_programs.py` (done 2026-08-23 for logic_gates/colour_lines/diffusion/dino plus the since-dropped rink/balloon/tetris; SET and egg still need adding to the installer's SELECTED list and to `clean_sweep.py::GAMES`); all of the stochastic games use `uniformChoice`/`randomPositions`, which at seed 0 always returns the first element, so data and tasks must be generated at seed >= 1 (human sessions all start from a `reset` with a non-zero seed); every replay must call `render_all()` after every `step()` (collision primitives read an occupancy set only rendering rebuilds).

Decision 2026-08-23: keep rink, logic_gates and balloon despite the atlas ratings. Three engine-level investigations (`notes/testbed_investigations/{rink,logic_gates,balloon}.md`) found every flagged problem either an artefact of the atlas probe (rink: unguarded random policy; logic_gates: "4 states" is really 13 frames / 36 transitions) or fixable by data/eval design (authored drives for the tiny click surfaces, a held-out-state split for logic_gates, click-at-rest discipline for balloon, background generalisation). Common thread across all three: **random play produces no signal** (rink 78% off-grid, logic_gates 0.85 switch hits per 60 actions, balloon 2% click hits and the threshold never reached), so the training drives for the new games must be authored or object-targeted, not sampled.

Decision 2026-08-23 (later): **lights, tetris and balloon replaced by paint (EAHCW), SET (zip) and magnets (7WWW9)** so that every selected game except rink has human play data. Verified against `basis_data.zip`: the study covers 43 games (all of them zip programs; identity confirmed by byte-identical sources for paint/magnets and click extents = GRID_SIZE−1 for the rest); lights is absent (`lights_new` is a different 24x24 Wire/Switch program), and rink/tetris/balloon are absent from the human data and from the LLM baselines (balloon has only an Opus-4.6 baseline). The investigations of balloon remain valid but the game is out of the selection. **Then rink replaced by egg** (zip, 16x16 gray-background, in the study with 60 sessions, deterministic latched-threshold rule): with that, **all 15 selected games have human play data**, so no artificial training pool is needed for the paper's main runs; rink's investigation stands but the game is out.

The 55-game atlas (`notes/autumn55_game_characteristics.md`) also recommends nine new Tier-A worlds (pacman, charge, lightning_rod, gameOfLife, blicket, arc_slack, balls, chomp, twiddle) that are not in the selection; they remain candidates if a selected game turns out to be unusable.


## Baselines

We would like to compare our learning approach with other approaches -

- **[X]** Raw LLM: During planning tasks, we present an LLM with the specification directly (states / NL goals) and evaluate its planning abilities. (`raw` arm in the curated evals; 0.32 macro on the 86-problem online NL set.)
- **[ ]** Raw LLM + in-context examples: Similar to raw LLM, however, we also provide trajectories from the training data for this game in the context of the prompt.
- **[P]** Agentic LLM: In this baseline we provide the training data to an agent while also giving it the planning problem. The agent can learn from the training data autonomously to solve the planning problem. RUN 2026-09-02 with Claude Code as the agent over the curated battery — 0.895 macro on 78 of the 86 problems (`cc_autumn/curated_results.json`, harness `cc_autumn/autumn-code/rig/curated.py`). It is not yet budget- or row-matched to the other arms; see "Agentic baseline" under Results for exactly what has to be closed. Note this agent explores the live world (50 actions) rather than reading the offline training pool, so it is a stronger baseline than the one described here. (The `RGB-Agent` submodule is a separate agentic analyser running on AutumnBench in dynamics mode, still not wired to planning problems.)
- **[P]** WorldCoder: This is a program learning approach (https://arxiv.org/abs/2402.12275) that learns a program as the world model using a similar optimisation algorithm. We evaluate this for planning by using the learned program as a world model inside a search for action sequences to satisfy the goal. (`offline_learning/worldcoder_optimize.py`; `wc` arm, offline and online.) The learners have run (15/15 deepseek, 4/15 Opus-5), but **the `wc` arm was disabled in all four planning-v2 online runs** — every `wc` cell in those runs reads `N/A`. Enabling it on the 86-problem set is the remaining work.
- **[ ]** WorldCoder + LLM Planning: This is built on top of WorldCoder where we use the learned program, but instead of using a search on the program, we provide the program in context to a language model and ask the language model to plan to solve the planning problems.

Our method is the `lmwm` arm (`offline_learning/rexpure_optimize.py`, search in `offline_learning/invdyn_core.py::rex_search`).


## Ablations

### Objectives
Ablations by experimenting with different variations of our learning objectives. The shipped objective is min(ID, cFD-hard) (`--composite min`).
- **[X]** No FD: `--fd-scorer none` (composite reduces to ID).
- **[ ]** No ID: no flag yet. Needs an `--no-id` switch and a decision on what the min-composite becomes with one term.

### Representations
- **[ ]** No Beliefs (only perception): no flag yet. `--start-beliefs ""` + a huge `--belief-update-period` approximates it but the proposer can still write beliefs; needs a real `--no-beliefs` that fixes B to empty.
- **[X]** No Perception (only beliefs): `--no-perception`.

### LLMs
The pipeline uses an LLM in three roles that were benchmarked separately and have different winners, so model choices must be stated per role:
1. Eval/decoder (the inverse-dynamics classifier and next-frame scorer, the bulk of the calls): current gpt-oss-120b@cerebras, effort=low. Benchmarked 2026-08-06/07 (`notes` + memory `evalmodel-full-catalog-sweep-aug7`): gpt-oss-20b@groq+low and ling-3.0-flash are cheaper at tied quality.
2. Reflection/proposer (diagnoses failures, proposes new P/B): current deepseek-v4-flash with thinking ON. Benchmarked 2026-08-09; every faster alternative lost end-to-end (−0.09 to −0.24 paired test).
3. Planner (the arm that solves planning problems): current deepseek-v4-flash @deepseek,baidu,fireworks. Benchmarked 2026-08-19/20 (`offline_learning/scripts/bench_planner_models.py`, `logs/2026-08-19/planner_bench/`): reasoning tokens, not model identity, buy planning quality.

- Big Model: Claude (Opus 5, via the local CLI proxy `scripts/claude_cli_proxy.py`) — **benchmarked in the
  reflection/proposer role 2026-08-30**: +0.086 mean credited test ID over deepseek-v4-flash on the same
  pools, 14/15 wins (`logs/2026-08-24/human_curated_opus5/RESULTS.md`). It is the "strong learner" arm,
  NLWM (SL), in the paper. Not benchmarked in the eval/decoder or planner roles. Claude also drives the
  agentic baseline (`cc_autumn`), where it is the agent rather than a component.
- Small Models:
  - DeepSeek V4 Flash
  - Gemini 3.7 Flash
  - GPT OSS 120b


## Results so far

Every number below is a macro (per-game, then across games) average, and every eval is ONLINE
(receding horizon, any-step scoring) on the 86-problem curated set with NL goal presentation and
per-problem action budgets, unless stated otherwise.

### Learning phase (what the arms train on)

Both learners train on the same manually curated human-play pools
(`offline_learning/human_data/<game>/informative_curated`, 60 train / 50 test targets, all 15 games).
The two rexpure arms differ ONLY in the reflection model; F (the eval/decoder) and the rex config are
identical, so the deltas are like-for-like.

| arm | reflector | learner | status | mean credited test ID |
|---|---|---|---|---|
| NLWM | deepseek-v4-flash | rexpure | **complete 15/15** (`logs/2026-08-24/human_curated`) | 0.754 |
| NLWM (SL) | claude-opus-5 | rexpure | **complete 15/15** (`logs/2026-08-24/human_curated_opus5`) | **0.840** (+0.086, wins 14/15) |
| WorldCoder | deepseek-v4-flash | worldcoder | complete 15/15 (`logs/2026-08-24/human_curated`) | 0.448 |
| WorldCoder (SL) | claude-opus-5 | worldcoder | **INCOMPLETE — 4/15 games** | 0.801 over those 4 (+0.126, 2/4 wins) |

The WorldCoder Opus-5 relaunch of the remaining 11 games (started 2026-08-30 09:30) died: only
`7www9_s1` and `7xf97_s1` have run dirs and neither reached a test summary; the other nine were never
written. Re-launch with `launch_human_origin.py --resume` before this arm can be reported.

### Planning evaluation

Four completed online runs. Only the last two are a matched pair and only they back the paper's table.

| run | goals | budgets | rows | raw | lmwm |
|---|---|---|---|---|---|
| `logs/2026-08-30/planning_v2_online_ds` | frame | flat 50 | 74 | 0.48 | 0.66 |
| `logs/2026-09-01/planning_v2_online_ds_nl` | NL | flat 50 | 79 | 0.48 | 0.61 |
| `logs/2026-09-03/planning_v2_online_ds_percap_nl` (**NLWM**) | NL | per-problem | 86 | 0.32 | **0.62** |
| `logs/2026-09-02/planning_v2_online_opus5_nl` (**NLWM (SL)**) | NL | per-problem | 86 | 0.36 | **0.60** |

The first two runs predate the 83→86 promotion and use a flat 50-action budget, so they share neither
the problem set nor the budgets with the pair below them and must not be pooled with it. The pair is
matched: 86 shared `task_uid`s, 0 rows with differing budgets, mean random floor 0.06 — they differ
only in which reflector built the artifacts the planner plans with. Both use the same
deepseek-v4-flash planner, and the Raw column is that planner with no learned model.

Headline: **the learned world model roughly doubles online planning pass@1 over the raw planner**
(0.32 → 0.62), and it does so on every tier (L1 0.60→0.93, L2 0.29→0.53, L3 0.20→0.60, L4 0.25→0.33).
The strong learner's large ID advantage (+0.086) does **not** carry into planning: NLWM (SL) scores
0.60 vs 0.62, a wash, winning on disease/mario/egg/space_invaders and losing on paint/SET. Better
one-step dynamics accuracy is not the same thing as a better planning model, and that gap is a result
worth reporting rather than a run to repeat.

`offline_learning/scripts/report_planning_v2_online.py` reads the run dirs, checks that
`replay.html` is level with the results, and writes the LaTeX between the AUTO markers in
`paper/main.tex` (`--write-tex`). `--check` reports any comparability breach.

### Agentic baseline (Claude Code on the same problems)

`cc_autumn/` holds a separate harness (`cc_autumn/autumn-code`, its own git repo) that plays Autumn
with Claude Code as the agent. It now runs the curated planning battery: one session per problem,
50 exploration actions in the live world, then a closed-loop test phase scored by importing this
repo's own loader, goal configuration and success test (`cc_autumn/autumn-code/rig/curated.py`), so
the +1/0 is the same verdict the online evaluator gives every other arm.

Result (`cc_autumn/curated_results.json`, launched 2026-09-02): **0.895 macro over 78 problems**,
78/78 audit-clean, 78/78 agreeing with an independent re-score, $166 and a median of 63 turns /
7.5 min per problem.

**It is not yet a fair column in the table.** Two protocol differences have to be closed first:

* **Budget.** The agent plays at a flat 50-action cap; the NLWM pair plays at per-problem caps
  (2–60). Flat 50 is the looser budget on most rows.
* **Row set.** The agent scores 78 of the 86. One row (`va6fq:settle-sand`) is dropped for a
  closed-loop random floor of 0.995, and seven more (`bt3gb:nightfall`, five `diffusion` rows,
  `s2kt7:spawn`) for a cap-50 random floor at or above 0.99 — i.e. the rows it drops are ones a
  random plan already solves at that budget, which the per-problem caps make hard again.

On the shared 78 rows the arms read: Raw 0.30, NLWM 0.586, NLWM (SL) 0.564, Agent 0.895. Closing the
comparison means re-running the agent under per-problem caps (or re-scoring the pair at flat 50 on
those 78); until then the agent's number is on a materially easier protocol and the paper's Agent
column stays em-dashed.

### Paper

`paper/` is a checkout of the Overleaf project (its own git remote, not tracked here). `main.tex`
currently carries the auto-generated results table and a commented-out protocol appendix table; the
prose sections are still TODO.

## TODO [X: completed, P: partially completed]
- [P] Set up the final evaluation set of planning tasks.
  - [X] Go through all autumn games and select a subset to evaluate on.
  - [X] Resolve the three atlas-flagged selections (rink, logic_gates, balloon): engine investigations done 2026-08-23, all three USABLE WITH FIXES; decision = keep all three. Reports in `notes/testbed_investigations/`, probe scripts in `scripts/testbed_probes/`. The fixes are folded into the items below.
    - [X] rink → USABLE WITH FIXES (DROPPED from the selection 2026-08-23 in favour of egg — no human data; kept for the record). Rules verified (walk 1 cell off-ice; entering the ice starts a 2-cell/tick slide that stops 1–2 cells past the far edge; perpendicular turns mid-slide; no 180° reversal except by re-entry from the overshoot cell; the skater can never rest on ice; off-grid frames are byte-identical for every off-grid position). Fixes: (1) on-grid guard — reject an arrow that would move the red cell out of [0,28)², from the frame alone — in the automatic drive sampler (`offline_learning/scripts/eval_multistep_fd_plan.py::generate_drive`/`random_plan`) and optionally `autumn_drive.py --on-grid-guard`; unguarded random drives go off-grid in 20/20 seeds (median step 2); (2) `--reflect-raw-prefix ≈9200 --reflect-max-failures 4` (the 1500-char default shows the P-proposer only the top 4 rows, never the skater); (3) curated ladder + NL checkers (8 verified problems, L4 needs 22–31 actions and the ice-free walk is 52 > PLAN_CAP so the mechanic is necessary; every checker must require the red cell to be present). Markov at K=2 (26% ambiguous at K=1, 0% at K=2; pipeline runs K=9). Credited-ID ceiling 0.68. Probes in `scripts/testbed_probes/rink/`.
    - [X] balloon → USABLE WITH FIXES (DROPPED from the selection 2026-08-23 in favour of magnets — no human data; kept for the record). Rules verified (35-cell sprite, origin y∈[2,7], 6 clickable cells at `x∈{6,7,8} × y∈{oy+6,oy+7}`, threshold counts a 5x8 column not the basket, settle 5 ticks, absorbing top/bottom, arrows are exact no-ops). Fixes: (1) the background items above; (2) authored drives — 12/607 random clicks hit and 0 random frames ever reach 3 rocks; recipe = 5 noops → 3 clicks at (6/7/8, oy+6) → 5 noops → remove → 5 noops; (3) drive/plan rule: click only while at rest, because a click tick freezes rock motion (strands rocks mid-air) and rocks can stack invisibly in one cell (real hidden state, 0.06% of noop transitions). Ladder: drop "rise to the ceiling" (floor 1.00); use one-rock (typed floor 0.08) / land (0.00) / exact-frame landed with 3 rocks (absorbing) / land-lighten-return (14-16 actions, 0.00). Probes in `scripts/testbed_probes/balloon/`.
    - [X] logic_gates → USABLE WITH FIXES. Fixes: (1) held-out-state split — train never turns both switches on, test on the 5 states with s=(1,1) or w=(1,1) (7/5 states, 17/19 transitions), the only split where a memoriser and a truth-table belief disagree; (2) authored 58-action drive (28/36 coverage vs 4.2/36 random); (3) variants `autumn_programs/variants/logic_gates_v1.sexp` (NAND/NOR/BUF/XNOR, tests B-content relearning under unchanged P) and `logic_gates_v2.sexp` (two-stage composition, 22 frames, planning depth 4 — the only version that carries an L1–L4 ladder); (4) score NL goals at the endpoint and never at t=0 (the t=0 frame violates the truth table). Obs ≈ 1450 tok/frame (2.2x ice). Two zero-floor NL goals identified. Probes in `scripts/testbed_probes/logic_gates/`.
  - [P] Wire the zip-sourced programs into the harness / data-gen / planning tooling: done 2026-08-23 for colour_lines, diffusion, dino, logic_gates (and the since-dropped rink/tetris/balloon) — `autumn_programs/` + `tools/install_autumn_programs.py`, per-game action alphabets in `offline_learning/clean_sweep.py::GAMES`, lower-case names accepted by `game_profile.py` (which now also renders after every step). Still to add: SET (installer SELECTED list; `clean_sweep.GAMES` alphabet `noop,click` keep=True — its only handler is `(on (clicked cards))`; `human_replay.GAMES`) and egg (alphabet `left,right,up,down,noop,click`; click is the gravity button so collapse to the verb like ice).
  - [P] Training data for the new games — pools BUILT 2026-08-24 for all 15 games (`offline_learning/human_data/<game>/informative_unified3`, build logs + pool table + viz in `logs/2026-08-24/human_unified3_build/`); all 15 pass the validation gates and their oracle ceilings are in the README; the training runs on these pools are the remaining part; `launch_human_origin.py` is already rewired for them — 15 games, `informative_unified3` default, archived-reference fallback, out-root `logs/2026-08-24/human_unified3` — and dry-runs clean for both learners, NOT launched. **Decision 2026-08-24 (data quality):** after manual review found coverage gaps (mario's coin→shoot→kill chain absent, grow's covered-watering unsampled, magnets' repulsion structurally excluded by the informativeness rule), drives are now selected MANUALLY per game (`offline_learning/curated_drives.json`, variant `informative_curated`; session-level picks only, within-drive sampling unchanged). ALL 15 games done 2026-08-24: mario picked by hand (shot clicks scored 5 train / 3 test), the other 14 by one review subagent per game over the per-drive sheets (`logs/2026-08-24/human_unified3_build/drive_sheets/`), picks + rationale in `curated_drives.json`, pools rebuilt at 60/50 with machine-verified disjointness/fill. `informative_curated` is now the training variant of record; unified3 stays as the uncurated control arm. **Runs 2026-08-24/25**: curated arm (deepseek-v4-flash reflection) COMPLETE for both learners — `logs/2026-08-24/human_curated/RESULTS.md` (rexpure mean credited test ID 0.754/15 games; worldcoder 0.448, collapsing on the stochastic tier + logic_gates). Claude-reflection arm (Opus 5 via `scripts/claude_cli_proxy.py`, `--reflection-client vllm`, F unchanged) in `logs/2026-08-24/human_curated_opus5/`: **rexpure COMPLETE 2026-08-30 for all 15 games — mean credited test ID 0.840 vs deepseek 0.754 on the same pools (+0.086, wins 14/15; only logic_gates loses 0.54 vs 0.60; biggest lifts f5w3n +0.28, colour_lines +0.25, s2kt7 +0.17; zero P errors/collapses; ~$47/game nominal, subscription-billed)** — table + notes in `human_curated_opus5/RESULTS.md`. Worldcoder: 4 valid games (bt3gb 0.79 +0.08, dq8gc 0.75 −0.09, eahcw 0.69 −0.02, n2ntd 0.97 +0.53), remaining 11 relaunched 2026-08-30 (0-call dirs archived in `_limit_hit3/`) — **that relaunch died**: as of 2026-09-04 only `7www9_s1` and `7xf97_s1` have run dirs and neither reached a test summary, so the WorldCoder (SL) arm still stands at 4/15 and needs `launch_human_origin.py --resume`. Ops: the subscription 5-h windows each buy ~1.5–2 h of Opus reflection; the proxy HOLDs requests through limit refusals (fix `8edb8e3` also parses rc≠0 refusals + 'session limit' wording) so runs sleep and resume without loss; one 22 h stall (Aug 28–29) was an expired `~/.claude/.credentials.json` OAuth token misclassified as a limit — recovered on refresh, no work lost. The reference runs since 2026-08-11 (`logs/2026-08-11/human_unified`, the artifacts every curated/NL/planner eval loads) train on HUMAN play — `offline_learning/human_data/<game>/informative_unified`, 60 train / 50 test targets built by `offline_learning/human_replay.py` (segment at resets → replay → noop-counterfactual observability filter → verb round-robin → 2-row slices + drive context) — not on `clean_data3`. Plan: (a) the 6 zip games (egg, colour_lines, diffusion, dino, logic_gates, SET — all have human sessions) get entries in `human_replay.GAMES` and run through the same pipeline (colour_lines dry run 2026-08-23: 249 train / 227 test informative candidates from the top-3/next-3 drives, so the 60/50 pools are easily filled); (b) paint (eahcw), magnets (7www9), grow, sand, space_invaders — benchmark games with human sessions — get the same human build as the 5 reference games; (c) egg's gray background and SET's off-screen template cards must be checked through the replay/render path before the pools are trusted. All 15 games are human-sourced; no artificial pool is needed for the main runs. Every human session starts with a `reset` carrying a non-zero seed. **Decisions 2026-08-24** (variant `informative_unified3`, recipe in `HUMAN_DATA_METHODOLOGY.md`): informativeness horizon 8 for every game (horizon 1 deletes every latent-effect action — paint's colour arrows, colour_lines' select-click — so paint would have no arrow targets at all); logic_gates keeps the plain user split, i.e. train and test are in-distribution — the abstraction question is whether the learner must see every switch combination to tell the gates apart, not a held-out-state generalisation; out-of-whitelist human inputs are kept as `noop` ticks (`--oov noop`) rather than deleted, so the replay preserves the human's timeline. The 4 already-built reference games are rebuilt under the same recipe so all 15 share one dataset definition (their `informative_unified` artifacts from 2026-08-11 stay as they are until retrained). Wiring done 2026-08-24: `human_replay.GAMES` (11 new entries), `clean_sweep.GAMES` + installer SELECTED for SET/egg (both installed, replay verified: SET 20x20, egg 16x16 gray background).
  - [X] Generalise background handling so non-black backgrounds (mario white, balloon skyblue) work without per-game patches. Done 2026-08-23 in two layers. (A) Plumbing: MARAProtocol `env_utils.py` renderers now REQUIRE `background_color` (omission is a TypeError, not a black grid) and all 12 render calls in `concrete_envs.py` pass it — including the CD/planning image renders, the planning goal image (the historical mario bug, now fixed at the source) and the MFP renders (which derive it from the program via `program_background`); `offline_learning/program_meta.py` reads `background`/`GRID_SIZE` from the .sexp and backs `mechanics._BG` / `mechanics_rules.BG,SIZE` as derived tables, `coverage_exam` no longer treats an unlisted game as all-foreground, `rescore_test50_id_sim` orders by the modal colour, `autumn_drive.py` prints the modal colour as background. (B) Prompt de-biasing: `invdyn_core._SCHEMA_AUTUMN` no longer leads with black (examples for black and skyblue worlds, the full colour list, explicit "backgrounds differ between worlds — detect the dominant colour"); `rexpure_optimize.py` scopes the proposer schema to the loaded frames' format (`infer_env_name`) instead of showing both envs. DECIDED AGAINST (C): injecting the measured background / grid shape into DEFAULT KNOWLEDGE — both are derivable from any frame, so stating them is feature engineering (and the dominant colour is wrong on rink, where it is the ice); `infer_background` is kept as a diagnostic to audit whether a learned P found the background itself. Verified: all four harness envs render text+image for balloon/N2NTD with the right background (pixel-checked), parity + 46 pytest pass. Pre-fix perception artifacts still hardcode black — re-learn, do not reuse, on non-black worlds. The MARAProtocol edits are committed as `81a26af` on `bai-integration` and this repo's submodule pointer records them.
  - [X] Create curated planning tasks for the remaining selected autumn games. Done 2026-08-30: `logs/2026-08-29/planning_v2/problems.json` has 86 problems across all 15 selected games — the 28 accepted problems for ice/disease/mario/ants are retained rather than regenerated, and 58 new problems cover paint, egg, grow, sand, colour_lines, SET, diffusion, dino, space_invaders, logic_gates, and magnets. (83 at first build; the three `f5w3n/shoot-enemy` rows were retired on 2026-09-01 as a render-occlusion exploit and replaced by six invader-count rows, promoting the set to 86 under schema v2.2.) Every row carries both a goal frame and a registered NL trajectory goal; evaluation requires an explicit presentation choice. Pipeline and counts: `notes/planning-problems-15-games.md`.
    - Legacy migration detail: five redundant Mario witness plans were normalized length-preservingly with all replacements recorded; their tasks, starts, goals, seeds, and horizons are unchanged.
  - [X] Create natural language problem versions for these curated planning tasks. All 86 v2.2 rows carry both a frame goal and a registered NL checker (68 distinct checker programs in `offline_learning/planning_nl_goals.py`); offline and online evaluators require `--goal-presentation frame` or `nl`. Four full online runs have been scored through the NL path, so the adapter work this item was tracking is finished.
  - [ ] Setup the CD/MFP/Masked Planning tasks from autumn for these games (exist for the 9 benchmark-sourced games incl. paint/magnets; must be authored for the 6 zip games).
  - [X] Thoroughly test the formulations of the planning tasks. The independent wrapper audit (`logs/2026-08-29/planning_v2/validation.json`) passes 86/86 rows and all global checks: raw/wrapper parity, prefix replay, reference success, nontrivial/noop failure, frame- and task-deletion minimality, action substitution, quiescence, reproducible random floors, composite-ID uniqueness, semantic goals for stochastic games, and three-seed stochastic templates. Easy L1/L2 random floors remain recorded rather than hidden.
- [P] Baselines. The agentic arm has run (Claude Code, 0.895 on 78/86) but is not budget- or row-matched; the `wc` arm is implemented but was disabled in every planning-v2 online run; raw + in-context trajectories and WorldCoder + LLM planning are not started.
  - [ ] Re-run the agentic arm under per-problem action caps on all 86 rows, or re-score the NLWM pair at flat cap 50 on the shared 78, so the Agent column is comparable.
  - [ ] Enable the `wc` arm in an online planning run on the 86-problem set, and finish the 11 missing WorldCoder (SL) training runs first if that arm is to be reported.
  - [ ] Raw LLM + in-context trajectories.
  - [ ] WorldCoder + LLM planning.
- [ ] Ablations: add `--no-id` and `--no-beliefs`. Nothing has been run for the paper's ablation table yet;
  every cell in `paper/main.tex::tab:ablations` is an em-dash.
- [P] Paper. `paper/` is an Overleaf checkout with its own remote (not tracked in this repo). The results
  table is auto-generated by `report_planning_v2_online.py --write-tex`; every prose section is TODO.
- [P] Clean up codebase to get rid of code that is not relevant to our core approach or baselines or the evaluations described in this document. A dead-code sweep and the `offline_learning/` rename happened 2026-08-07. The planning-v2 tooling and eval changes were committed 2026-09-04; `cc_autumn/autumn-code` (its own git remote) and `paper/` (Overleaf) are deliberately not tracked here, and `cc_autumn`'s multi-megabyte bundles/replays/zips are gitignored as regenerable. The restructuring itself has not started.
- [P] Figure out what models to use
   - [P] Run profiling experiments on a variety of models on OpenRouter to understand the tradeoffs between time, money and performance. Select the model that best satisfies these. Done for the eval, reflection and planner roles with the small models (see LLMs section), and Claude (Opus 5) is now benchmarked in the reflection role — it wins the learning objective decisively (+0.086 credited test ID, 14/15) but that advantage does not reach planning (0.60 vs 0.62). Still open: Claude in the eval/decoder and planner roles, and writing the final per-role choice down.
