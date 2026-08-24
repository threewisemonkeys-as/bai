# Experimental Plan

We want to evaluate our approach of learning from offline interaction data from games.
Our approach currently uses a text optimisation algorithm (Rex) to learn beliefs (a piece of text describing how the game works) and perception (python code that parses raw game state and outputs natural language strings describing the state).
We want to present our idea as a scientific paper, comparing it against baselines, exploring different tasks and performing various ablations.
In this document we outline the details of our plan for experiments.

Status legend used throughout: **[X]** done, **[P]** partial, **[ ]** not started. Status was last reconciled against the repo on 2026-08-23.


## Tasks

After a learning phase, where our methods (and baselines) learn from training data, we evaluate them on various planning problems in the games -

- Curated Planning Tasks: Accomplishing semantically useful goals in each game. Tasks for each game are created by understanding the dynamics of the game, and coming up with a handful of semantically meaningful goals. The tasks are stored in `logs/2026-08-18/curated/problems.json` (30 problems, L1-L4 ladders; build procedure in `notes/curated-planning-procedure.md`, builder `offline_learning/curated_plan.py`).
  - Goal as required state: Tasks are specified by a start state and end state and task completion is exact match with the end state. Start state = (program, seed, action prefix); there is no interpreter snapshot, so every state is addressed by replay.
  - Goal as NL statement: Tasks are specified by a start state and a natural language statement indicating the goal. Completion is checked with crafted predicates (`offline_learning/nl_goals.py`) designed to check whether the final state the plan arrives at satisfies the natural language goal. NL versions exist for all 30 curated problems (`logs/2026-08-20/nl_full/`).
- Autumn Tasks: CD/MFP/Masked Planning. These exist only for the 20 AutumnBench programs (60 tasks, `MARAProtocol/.../example_benchmark/`); games that come from `autumn_programs_55.zip` have none and need them authored. Known caveats from the 2026-08-22 audit: AutumnBench planning is scored on any-step reach (12/21 goals reachable by chance; zero floor for grow/ice/disease/sand), CD reward is a ratio with a per-game ceiling (disease 0.71), ants' CD never renders differently at seed 0, mario's CD crashes if the first action is a click.

All of these planning problems can have 2 modes.
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

For initial experimentation, we select the following [15] games from the Regular, Stochastic and Abstract groups. The table records, per game, where the program comes from (AutumnBench id or the 55-game zip), and what infrastructure exists today. "data" = training drives + test50 under `offline_learning/clean_data3/`; "curated" = frame-goal ladder in `problems.json`; "NL" = predicate checkers in `nl_goals.py`; "CD/MFP" = AutumnBench tasks.

| group | game | grid | source id | data | curated | NL | CD/MFP | notes |
|---|---|---|---|---|---|---|---|---|
| Regular / Easy | lights | 16 | E3V6M | X | | | X | clean on all 5 diagnostics; best next ladder candidate |
| Regular / Easy | rink | 28 | zip | | | | | **USABLE WITH FIXES** (`notes/testbed_investigations/rink.md`): the atlas numbers were an artefact of its unguarded random policy (78% of ticks off-grid, invisible); with a frame-computable on-grid guard vis 0.14→0.99, 2,324-state space BFS-closed with 0 escapes, cleaner than DQ8GC on the 5 diagnostics; real cost = 2,837 tok/frame (3.7x ice) |
| Regular / Moderate | ice | 16 | ice (== BT3GB) | X | X (8) | X | X | reference game |
| Regular / Moderate | disease | 16 | DQ8GC | X | X (7) | X | X | hidden state (click invisible for >= 6 frames) |
| Regular / Moderate | grow | 16 | 7XF97 | X | | | X | agent can walk off-grid |
| Regular / Hard | mario | 12 | N2NTD | X | X (7) | X | X | white background; CD first-click crash; enemy `movingLeft` hidden |
| Regular / Hard | sand | 10 | VA6FQ | X | | | X | clean on all 5 diagnostics |
| Stochastic / Easy | ants | 16 | S2KT7 | X (seed 5) | X (6) | X | X | seed-0 data is degenerate; CD never renders at seed 0 |
| Stochastic / Easy | colour_lines | 10 | zip | | | | | random ball injected on every empty click |
| Stochastic / Moderate | diffusion | 9 | zip | | | | | pure random walk every tick; distributional scoring only |
| Stochastic / Moderate | dino | 20 | zip | | | | | bird row random per wrap; absorbing death |
| Stochastic / Hard | tetris | 16 | zip | | | | | piece shape+column random per landing; deterministic between landings |
| Stochastic / Hard | space_invaders | 16 | F5W3N | X | | | X | |
| Abstract | logic_gates | 24 | zip | | | | | **USABLE WITH FIXES** (`notes/testbed_investigations/logic_gates.md`): 13 frames / 36 transitions, not 4 states; 8 clickable cells of 576 → authored drives; a 51-action drive covers every transition, so the split must hold out the both-switches-on state (only one where OR ≠ XOR); L4 needs `logic_gates_v2` |
| Abstract | balloon | 16 | zip | | | | | **USABLE WITH FIXES** (`notes/testbed_investigations/balloon.md`): clean, deterministic, zero-aliased count-threshold physics with two absorbing states; 6-cell click surface that moves with the balloon (random play never reaches the 3-rock threshold → authored drives, clicks only at rest); three real black-background biters listed in the TODO |

Facts that constrain the work on the 7 zip-sourced games: the harness loads programs only from `example_benchmark/programs/` (gitignored inside the MARAProtocol submodule), so the zip sources are tracked under `autumn_programs/` and installed there by `tools/install_autumn_programs.py` (done 2026-08-23, all 7 load and step through `autumn_drive.py`); all of the stochastic games use `uniformChoice`/`randomPositions`, which at seed 0 always returns the first element, so data and tasks must be generated at seed >= 1; every replay must call `render_all()` after every `step()` (collision primitives read an occupancy set only rendering rebuilds).

Decision 2026-08-23: keep rink, logic_gates and balloon despite the atlas ratings. Three engine-level investigations (`notes/testbed_investigations/{rink,logic_gates,balloon}.md`) found every flagged problem either an artefact of the atlas probe (rink: unguarded random policy; logic_gates: "4 states" is really 13 frames / 36 transitions) or fixable by data/eval design (authored drives for the tiny click surfaces, a held-out-state split for logic_gates, click-at-rest discipline for balloon, background generalisation). Common thread across all three: **random play produces no signal** (rink 78% off-grid, logic_gates 0.85 switch hits per 60 actions, balloon 2% click hits and the threshold never reached), so the training drives for the new games must be authored or object-targeted, not sampled.

The 55-game atlas (`notes/autumn55_game_characteristics.md`) also recommends nine new Tier-A worlds (pacman, charge, lightning_rod, gameOfLife, blicket, arc_slack, balls, chomp, twiddle) that are not in the selection; they remain candidates if a selected game turns out to be unusable.


## Baselines

We would like to compare our learning approach with other approaches -

- **[X]** Raw LLM: During planning tasks, we present an LLM with the specification directly (states / NL goals) and evaluate its planning abilities. (`raw` arm in the curated evals.)
- **[ ]** Raw LLM + in-context examples: Similar to raw LLM, however, we also provide trajectories from the training data for this game in the context of the prompt.
- **[ ]** Agentic LLM: In this baseline we provide the training data to an agent while also giving it the planning problem. The agent can learn from the training data autonomously to solve the planning problem. (The `RGB-Agent` submodule is an agentic analyser already running on AutumnBench in dynamics mode, but it is not wired to planning problems.)
- **[X]** WorldCoder: This is a program learning approach (https://arxiv.org/abs/2402.12275) that learns a program as the world model using a similar optimisation algorithm. We evaluate this for planning by using the learned program as a world model inside a search for action sequences to satisfy the goal. (`offline_learning/worldcoder_optimize.py`; `wc` arm, offline and online.)
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

- Big Model: Claude — not yet benchmarked in any role.
- Small Models:
  - DeepSeek V4 Flash
  - Gemini 3.7 Flash
  - GPT OSS 120b


## TODO [X: completed, P: partially completed]
- [P] Set up the final evaluation set of planning tasks.
  - [X] Go through all autumn games and select a subset to evaluate on.
  - [X] Resolve the three atlas-flagged selections (rink, logic_gates, balloon): engine investigations done 2026-08-23, all three USABLE WITH FIXES; decision = keep all three. Reports in `notes/testbed_investigations/`, probe scripts in `scripts/testbed_probes/`. The fixes are folded into the items below.
    - [X] rink → USABLE WITH FIXES. Rules verified (walk 1 cell off-ice; entering the ice starts a 2-cell/tick slide that stops 1–2 cells past the far edge; perpendicular turns mid-slide; no 180° reversal except by re-entry from the overshoot cell; the skater can never rest on ice; off-grid frames are byte-identical for every off-grid position). Fixes: (1) on-grid guard — reject an arrow that would move the red cell out of [0,28)², from the frame alone — in the automatic drive sampler (`offline_learning/scripts/eval_multistep_fd_plan.py::generate_drive`/`random_plan`) and optionally `autumn_drive.py --on-grid-guard`; unguarded random drives go off-grid in 20/20 seeds (median step 2); (2) `--reflect-raw-prefix ≈9200 --reflect-max-failures 4` (the 1500-char default shows the P-proposer only the top 4 rows, never the skater); (3) curated ladder + NL checkers (8 verified problems, L4 needs 22–31 actions and the ice-free walk is 52 > PLAN_CAP so the mechanic is necessary; every checker must require the red cell to be present). Markov at K=2 (26% ambiguous at K=1, 0% at K=2; pipeline runs K=9). Credited-ID ceiling 0.68. Probes in `scripts/testbed_probes/rink/`.
    - [X] balloon → USABLE WITH FIXES. Rules verified (35-cell sprite, origin y∈[2,7], 6 clickable cells at `x∈{6,7,8} × y∈{oy+6,oy+7}`, threshold counts a 5x8 column not the basket, settle 5 ticks, absorbing top/bottom, arrows are exact no-ops). Fixes: (1) the background items above; (2) authored drives — 12/607 random clicks hit and 0 random frames ever reach 3 rocks; recipe = 5 noops → 3 clicks at (6/7/8, oy+6) → 5 noops → remove → 5 noops; (3) drive/plan rule: click only while at rest, because a click tick freezes rock motion (strands rocks mid-air) and rocks can stack invisibly in one cell (real hidden state, 0.06% of noop transitions). Ladder: drop "rise to the ceiling" (floor 1.00); use one-rock (typed floor 0.08) / land (0.00) / exact-frame landed with 3 rocks (absorbing) / land-lighten-return (14-16 actions, 0.00). Probes in `scripts/testbed_probes/balloon/`.
    - [X] logic_gates → USABLE WITH FIXES. Fixes: (1) held-out-state split — train never turns both switches on, test on the 5 states with s=(1,1) or w=(1,1) (7/5 states, 17/19 transitions), the only split where a memoriser and a truth-table belief disagree; (2) authored 58-action drive (28/36 coverage vs 4.2/36 random); (3) variants `autumn_programs/variants/logic_gates_v1.sexp` (NAND/NOR/BUF/XNOR, tests B-content relearning under unchanged P) and `logic_gates_v2.sexp` (two-stage composition, 22 frames, planning depth 4 — the only version that carries an L1–L4 ladder); (4) score NL goals at the endpoint and never at t=0 (the t=0 frame violates the truth table). Obs ≈ 1450 tok/frame (2.2x ice). Two zero-floor NL goals identified. Probes in `scripts/testbed_probes/logic_gates/`.
  - [X] Wire the 7 zip-sourced programs into the harness / data-gen / planning tooling (rink, colour_lines, diffusion, dino, tetris, logic_gates, balloon): `autumn_programs/` + `tools/install_autumn_programs.py`, per-game action alphabets in `offline_learning/clean_sweep.py::GAMES`, lower-case names accepted by `game_profile.py` (which now also renders after every step).
  - [ ] Generate training drives + test50 for the 7 new games at seed >= 1. Random play is useless on three of them (see the investigations): logic_gates needs the authored covering drive AND the held-out-state split; balloon needs object-targeted clicks at rest; rink needs the frame-computable on-grid guard in the sampler. Check the other four (colour_lines, diffusion, dino, tetris) for the same failure before trusting a random drive.
  - [X] Generalise background handling so non-black backgrounds (mario white, balloon skyblue) work without per-game patches. Done 2026-08-23 in two layers. (A) Plumbing: MARAProtocol `env_utils.py` renderers now REQUIRE `background_color` (omission is a TypeError, not a black grid) and all 12 render calls in `concrete_envs.py` pass it — including the CD/planning image renders, the planning goal image (the historical mario bug, now fixed at the source) and the MFP renders (which derive it from the program via `program_background`); `offline_learning/program_meta.py` reads `background`/`GRID_SIZE` from the .sexp and backs `mechanics._BG` / `mechanics_rules.BG,SIZE` as derived tables, `coverage_exam` no longer treats an unlisted game as all-foreground, `rescore_test50_id_sim` orders by the modal colour, `autumn_drive.py` prints the modal colour as background. (B) Prompt de-biasing: `invdyn_core._SCHEMA_AUTUMN` no longer leads with black (examples for black and skyblue worlds, the full colour list, explicit "backgrounds differ between worlds — detect the dominant colour"); `rexpure_optimize.py` scopes the proposer schema to the loaded frames' format (`infer_env_name`) instead of showing both envs. DECIDED AGAINST (C): injecting the measured background / grid shape into DEFAULT KNOWLEDGE — both are derivable from any frame, so stating them is feature engineering (and the dominant colour is wrong on rink, where it is the ice); `infer_background` is kept as a diagnostic to audit whether a learned P found the background itself. Verified: all four harness envs render text+image for balloon/N2NTD with the right background (pixel-checked), parity + 46 pytest pass. Pre-fix perception artifacts still hardcode black — re-learn, do not reuse, on non-black worlds. MARAProtocol edits sit in the submodule working tree and need a submodule commit + pointer bump.
  - [P] Create curated planning tasks for the remaining selected autumn games. Done: ice, disease, mario, ants (4/15). Remaining 11: lights, rink, grow, sand, colour_lines, diffusion, dino, tetris, space_invaders, logic_gates, balloon. Run `offline_learning/scripts/game_profile.py` diagnostics first for each.
  - [P] Create natural language problem versions for these curated planning tasks. Done for the existing 30 problems; remaining games follow their ladders.
  - [ ] Setup the CD/MFP/Masked Planning tasks from autumn for these games (exist for the 8 benchmark-sourced games; must be authored for the 7 zip games).
  - [ ] Thoroughly test the formulations of the planning tasks to make sure that they evaluate planning in a convincing way (random floors per problem, no chance-reachable goals, absorbing end states for frame goals, predicates tight enough that compression cannot delete the intended route).
- [ ] Baselines: add the three missing arms (raw + in-context trajectories, agentic LLM, WorldCoder + LLM planning).
- [ ] Ablations: add `--no-id` and `--no-beliefs`.
- [P] Clean up codebase to get rid of code that is not relevant to our core approach or baselines or the evaluations described in this document. A dead-code sweep and the `offline_learning/` rename happened 2026-08-07, and the working tree currently holds ~1,500 uncommitted changes from it (archive moves, `data/oracle` deletions); commit that state before restructuring further.
- [P] Figure out what models to use
   - [P] Run profiling experiments on a variety of models on OpenRouter to understand the tradeoffs between time, money and performance. Select the model that best satisfies these. Done for the eval, reflection and planner roles with the small models (see LLMs section); Claude has not been benchmarked in any role, and the final per-role choice has not been written down.
