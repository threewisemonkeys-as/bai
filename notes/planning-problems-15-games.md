# Planning problems for the selected 15 Autumn games

Status: built 2026-08-30, revised to v2.2 on 2026-09-01 (see "Revision" below). The
schema-versioned generated artifact is
`logs/2026-08-29/planning_v2/problems.json`; its independent audit is
`logs/2026-08-29/planning_v2/validation.json`. Schema v2.2 gives every row a registered
Python NL-goal program; no declarative `goal_spec` payloads are shipped.

## Decision about the original four games

The v2 set does **not** generate replacement problems for ice, disease, mario, or ants.
It migrates their 28 accepted v1 rows without changing the stored start frame, goal frame,
seed, or horizon, then adds v2 addressing fields (`task_uid`, `prefix=[]`, goal frame and NL
goal). The final audit found redundant movement in five Mario witness plans, so those plans
were normalized length-preservingly by replacing only proven-unnecessary moves with timing
noops. Each affected row records `legacy_original_plan` and `migration_repairs`; no task or
goal was added or regenerated. The 58 newly authored rows cover the other 11 selected games.
Particles/83wkq is not
one of the selected 15 and remains outside this set.

| game | rows | goal representations |
|---|---:|---|
| paint (`eahcw`) | 4 | frame + NL |
| egg | 5 | frame + NL |
| ice (`bt3gb`) | 8 | frame + NL |
| disease (`dq8gc`) | 7 | frame + NL |
| grow (`7xf97`) | 5 | frame + NL |
| mario (`n2ntd`) | 7 | frame + NL |
| sand (`va6fq`) | 7 | frame + NL |
| ants (`s2kt7`) | 6 | frame + NL |
| colour_lines | 3 | frame + NL |
| SET | 3 | frame + NL |
| diffusion | 11 | frame + NL |
| dino | 3 | frame + NL |
| space_invaders (`f5w3n`) | 9 | frame + NL |
| logic_gates | 4 | frame + NL |
| magnets (`7www9`) | 4 | frame + NL |
| **total** | **86** | **28 retained + 58 new** |

## Revision: 83 -> 86 rows (2026-09-01, schema v2.2)

The set was first built at 83 rows. Auditing the space_invaders rows found that
`f5w3n/shoot-enemy` was satisfiable by a one-tick render occlusion -- a just-spawned orange
bullet drawn over an invader reads as a kill -- so the checker was retired rather than
patched. Its three seeded rows were replaced by six rows under two checkers that count
invaders across two guarded frames (`exactly-one`, `exactly-three`, three seeds each), which
is what takes space_invaders from 6 to 9 rows and the set from 83 to 86. The staged artifact
is `logs/2026-09-01/planning_v22_staging/problems.json`; it was promoted in place to
`logs/2026-08-29/planning_v2/problems.json`, which is the file every evaluator loads.

Runs made before the promotion are therefore scored on a smaller set and are not comparable
to the 86-row runs: `logs/2026-08-30/planning_v2_online_ds` covers 74 rows and
`logs/2026-09-01/planning_v2_online_ds_nl` 79. `offline_learning/scripts/rescore_online.py`
re-scores a finished run against the current checkers where the recorded frames allow it;
the three retired `shoot-enemy` rows cannot be re-judged, because the buggy checker stopped
each rollout at action 4 and only four frames were ever recorded.

## Construction pipeline

1. **Profile before authoring.** `game_profile.py` measures reset drift, seeded divergence,
   rendered-state ambiguity, overlap, and the live input surface. The profiler now uses a
   nonzero seed, distinguishes quiet-at-reset from eventual settling, varies seeds in the
   hidden-state probe, tests state-dependent movement, and scans sparse click surfaces. This
   catches SET's deal on its first tick and logic_gates' two small switch hitboxes.
2. **Read the program and probe routes.** Objectives and mechanics come from each `.sexp`;
   candidate action sequences are replayed before they become specs. The manual route tool
   accepts all 15 selected games and records states as `(program, seed, prefix)`.
3. **Author both goal representations.** Every task has an ordinary Python checker program
   with the legacy `check(grids, actions) -> bool` interface and a reference endpoint frame. Prefixes prepare states such as SET's nine-card
   deal, a raised egg, and a powered logic input.
4. **Compress against the task, not the rendered frame.** The raw interpreter greedily
   deletes actions to a fixpoint. For goals intended to settle, quiescence means that the
   reached state survives a *future* noop; the probe noop is not shipped in the plan.
5. **Ship both representations without a preference.** Every row contains the compressed
   route's endpoint frame plus its NL sentence and registered trajectory checker. The
   evaluator, not the problem row, must explicitly choose which one to present and score.
6. **Measure stable floors.** Each new row stores a random-plan success rate from 24 trials.
   The RNG seed is CRC32 of `task_uid`, not Python's process-salted `hash`, so another process
   can reproduce it exactly.
7. **Validate with another driver.** The builder drives the raw interpreter. The validator
   replays prefixes and candidates through `AutumnBenchEnvWrapper` and requires all ten
   checks below. It also requires unique composite task IDs, all 15 games, exactly 28
   retained rows, semantic goals for the stochastic tier, and a three-seed template in
   every stochastic game.

Build and validate with:

```bash
PYTHONPATH=. uv run python offline_learning/scripts/build_planning_v2.py
PYTHONPATH=. uv run python offline_learning/scripts/validate_planning_v2.py
```

## Independent audit screens

- `A1-schema`: required v2 fields, known game, nonempty NL goal, and horizon consistency.
- `A2-drivers`: raw-interpreter and wrapper traces agree exactly.
- `A3-reference`: the replayed prefix reproduces `start`; the frame reference and the
  explicit NL reference each satisfy their respective scorer.
- `A4-nontrivial`: the current state does not already satisfy the goal.
- `A5-noop-fails`: waiting for the reference horizon does not solve the problem.
- Frame-reference deletions are recorded as a diagnostic; semantic compression can leave actions that are irrelevant to the endpoint frame.
- `A7-task-delete`: no single deletion satisfies the Python NL-goal program.
- `A8-substitute`: no consequential action can be replaced with noop. This is enforced on
  all 55 newly generated rows. Non-minimality in an accepted v1 NL witness is diagnostic;
  the accepted exact-frame task is not rewritten to optimize a different scorer.
- `A9-quiescence`: the recorded flag matches a real future noop.
- `A10-random-floor`: wrapper execution exactly reproduces the stored stable random floor.

The final full run passes 86/86 row audits and every global check.

## Why the NL and frame conditions can differ

An exact future frame would score uncontrollable details in these games:

- **colour_lines:** moving the original blue ball to a destination is scored; the random
  color/location of the extra spawned ball is ignored.
- **SET:** any valid triple may be removed. Each board is dealt by a nine-noop prefix, and
  the checker observes card removal rather than one memorized triple.
- **diffusion:** controlled membrane density and red/blue additions are scored while random
  walks are ignored.
- **dino:** survival through two cactus passes is scored at the second reset; the random
  bird row is not an exact-frame target.
- **space_invaders:** an invader count decrease, resolved player bullet, and living hero are
  scored; the random enemy-bullet origin is not.

The three dino witnesses use two timed jumps over a 30-step horizon. A 48-plan probe per
seed found a 0/48 random floor, replacing an earlier one-cactus formulation whose floor was
12.5–20.8%.

## Schema notes for evaluation

`task_uid` is the join/checkpoint key; `id` alone is not globally unique. Always replay
`prefix` before presenting `start` or executing a candidate plan. A frame run uses
`goal` and `frame_success_mode`; an NL run resolves `nl_checker` in
`offline_learning/planning_nl_goals.py` and uses its scheduling and quiescence metadata. Every row also has an `nl_reference_plan`, so the
all-NL presentation uses the same Python scoring path for all 86 tasks. There are 68 unique
checker programs because multi-seed stochastic rows intentionally share semantics.

`eval_curated_plan.py` requires one of two explicit views: frame or NL. Every problem is
included in either view, and its oracle preflight must pass every selected reference.

## Action budget: reference-scaled caps

Runs up to and including `logs/2026-09-01/planning_v2_online_ds_nl` used a flat 50-action
budget for every row (`PLAN_CAP`). That is a different test per row: at 50 actions a
4-action problem gets fifty chances to stumble onto its goal and a 40-action one gets one,
so the same pass rate does not mean the same thing across the ladder. It also inflates the
random floor exactly where the problem is shortest — SET's rows measure a 0.46–0.54 floor
at cap 50 and 0.01–0.02 at their own cap of 8.

`--cap-mode` (both evaluators, default `fixed` = the historical flat budget) scales the
budget off each row's reference instead:

    cap = 2 x reach          if reach <= 10
    cap = ceil(1.5 x reach)  if reach > 10

`per-game` takes the max over the game's rows (one budget per evaluator invocation, which
is how the launcher fans out); `per-problem` uses each row's own. Over the 86-row NL set:
worst-case executed actions drop 23% under `per-game` and 60% under `per-problem`.

`reach` is the **any-step** reference reach measured by
`offline_learning/scripts/annotate_action_caps.py` and stored as
`{frame,nl}_anystep_reached_at` — deliberately not `{pres}_reference_reached_at`, which is
recorded under the row's own success mode and therefore stores the plan length even when
the goal first holds much earlier. dino stores 30 and first holds at 10; n2ntd `coin-air`
stores 11 and holds at 5. Online scoring is any-step by construction, so the any-step reach
is the honest reference.

Two consequences that are enforced, not documented-and-hoped:

  * **A floor only compares to a score measured at the same budget.** Floors live in
    `{pres}_random_floors` keyed by cap; `recompute_random_floors.py --cap-mode ...`
    measures them, and a scaled-cap run refuses to start if any selected row lacks one.
    The `_cap50` fields remain the flat-50 regime's floors.
  * **A rollout is only reusable under the budget it ran with.** The checkpoint key gains
    a `|cap<N>` suffix under a scaled mode, so runs at different budgets never resume into
    each other. Flat-50 keys are unchanged, so existing checkpoints still resume.

`per-game` is looser than `per-problem` by exactly the within-game spread of reference
lengths, and on the heterogeneous games that spread is most of the point: 7xf97's rows
reach in 1–40 actions, so its per-game cap of 60 still gives the 1-action row sixty
chances. Prefer `per-problem` where the comparison is per-row; `per-game` is the right
knob when a whole game must share one budget.

## Looking at the trajectories

Two pages are written per game, from the same `online.json`:

  * `viz.html` (`viz_v2_online.py`) — the per-round plan-vs-executed filmstrip.
  * `replay.html` (`viz_plan_replay.py`) — the replay, modelled on the Claude-Code-on-Autumn
    page (`cc_autumn/curated_replay.html`): a game x problem matrix, the board stepped one
    executed action at a time, and beside it **the exact prompt the planner was shown and
    the exact response it gave at that round**. That substitution is the whole difference
    from the CC page, where the same rail holds the agent's off-board work; in this eval
    the prompt and the response are all there is to look at.

The budget bar under the board is the run's action cap, one block per executed action,
coloured by whether the model re-planned or replayed the candidate it was already carrying
— which is the thing worth seeing at a glance in a receding-horizon run — green where the
goal held, with a red tick where a plan was rejected. `a` switches arm on the same problem.

Prompts are stored as indices into one shared line table. Every round repeats the same
knowledge block and most of the same transcript, so the table is about 6% of the raw
prompt bytes: a 1875-round, 5-game run is a 4.9 MB page. A full 15-game run will not fit
under the 16 MB page limit, so `--per-game` writes one page per game and the launcher does
that automatically.

`offline_learning/launch/watch_plan_replay.py <run-root>` keeps the pages level with a run
that is still playing. It re-renders only the games whose `online.json` has moved (the
evaluator rewrites it every `--emit-every` rollouts), writes atomically so a browser never
gets half a page, and stops when the launcher leaves the process list and nothing moves.
