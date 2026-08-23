# Curated Autumn planning set

**Built 2026-08-18.** 30 problems, `logs/2026-08-18/curated/problems.json`, all 30 passing
every check in `validate_curated.py`.  Replaces the 470-problem compositional set, whose
data has been erased.

    uv run python -m offline_learning.curated_plan build --out logs/<date>/curated/problems.json
    uv run python offline_learning/scripts/validate_curated.py logs/<date>/curated/problems.json
    uv run python offline_learning/scripts/viz_curated.py --in ... --out ...

## What changed and why

The compositional set targeted CHAINS OF MECHANICS and grew a window forward until the chain
completed.  Nothing in that construction asked whether the chain added up to anything, so it
produced `move-up -> move-left -> contagion-spread` and never `every particle is infected`;
`jump -> gravity-fall` and never `the enemy is dead`.  Every screen it passed was about the
plan, not the goal.

This set is ~6 problems per game, each authored around something the game is about, ordered
L1 (one mechanic) -> L4 (the objective).

## Goals are concrete end states

One full rendered frame, exact equality.  No masks, no partial goals, no predicate
evaluation at eval time.  Predicates exist only inside the generator: they define the
subgoals the solver chains through, and they carry the incompressibility proof.

This works because **every one of these games has an absorbing end state**.  Killing n2ntd's
enemy removes the only autonomously-moving object in any of the five and the world goes
completely still; bt3gb water settles, dq8gc particles move only on input, s2kt7 ants freeze
once the food is gone.  20 of 30 goals are absorbing; the other 10 are recorded
`quiescent: false` and flagged `tick-exact` in the viz -- the frame names one tick and a
solution has to land on it.  All 10 are either h=1 or a live-enemy mario problem or
s2kt7/intercept, which is deliberately mid-forage.

## The set

| game | n | h range | notes |
|---|---|---|---|
| n2ntd / mario | 7 | 1-34 | `all-coins-kill` (h=34) is the crown: 3 coins, then a timed shot |
| bt3gb / ice | 8 | 1-33 | `ice-tower` vs `freeze-pool` are the discriminating pair |
| dq8gc / disease | 7 | 3-19 | `gather` is unsolvable without repeated click-select |
| s2kt7 / ants | 6 | 1-21 | `intercept` is the only non-absorbing goal by design |
| 83wkq / particles | 2 | 1-2 | see the honesty note below |

35 distinct `.sexp` rules fire across the set.

## Padding control

Every plan is greedily compressed to a fixpoint: no single action can be deleted and still
complete the objective.  Compression runs against the authored **predicate**, not the frame.
That distinction is load-bearing on n2ntd: the enemy patrols on an 18-tick cycle regardless
of the agent, so deleting any action re-phases it and misses the frame whether or not the
action mattered -- a frame-based screen would pass vacuously on exactly the game that needs
it most.  The validator reports both (`V4f` frame, `V4t` task) so the difference stays
visible.

Compression does real work: `staircase` came out of the solver at 60 actions and shipped at
33; `three-rounds` 25 -> 21.

## Screens (validate_curated.py)

Deliberately re-runs every plan through `AutumnBenchEnvWrapper` -- the *other* engine driver
-- rather than the raw-interpreter `Sim` the builder used, so a disagreement between the two
surfaces as a failure instead of a silently wrong dataset.

- **V1** the reference plan reproduces the stored goal frame exactly, and the stored start
- **V2** the goal frame occurs at no earlier step
- **V3** noop^h misses the goal
- **V4f** frame-incompressible / **V4t** task-incompressible (see above)
- **V5** the recorded `quiescent` flag is correct
- **V6** random plans of the same length miss.  Enforced only for h >= 2: `platform`,
  `nightfall` and `spawn` are h=1 and a random plan guesses them 23-33% of the time.  That
  is a true fact about a one-action problem, so `random_success` ships on every problem as
  the floor any result has to beat, rather than being hidden.

## Two engine facts worth keeping

**Unrendered state breaks naive search.**  n2ntd's enemy carries a `movingLeft` flag that is
not drawn: it sits on the same column going both ways (t=12 and t=14 render identically and
evolve oppositely), so BFS deduplicating on the frame silently pruned the only branch that
kills it.  Mario is also invisible for the one frame he stands on a coin (coins render after
him), and an s2kt7 ant is invisible while standing on food.  Both games therefore carry the
tick in the search key, making it a sound BFS over the time-expanded graph.

**bt3gb drops must be spaced.**  `on down` assigns water from the CURRENT list and overrides
the fall clause, so two `down`s on consecutive ticks put both drops in the same cell and they
travel together for ever -- three of them render as one.

## Honest limits

- **83wkq is not a planning game.**  Particles random-walk every tick with no bounds check;
  the same click under four seeds puts the particle in four different cells by t+2.  Under
  exact-frame goals it supports exactly one solvable problem (`spawn-one`, h=1, goal one tick
  after the only click).  `spawn-two` is included as requested and is close to unsolvable by
  construction: the second particle sits where it was clicked, but the first has already
  taken a uniform random step, so the frame cannot be derived from the rules -- only
  memorised from the RNG.  Its note says so.
- **Invisible state cannot appear in a goal.**  Mario's ammo and which dq8gc particle is
  active are not rendered, which rules out e.g. "kill the enemy with ammo to spare".
- **s2kt7 uses seed 1.**  Seed 0 is degenerate -- `randomPositions` collapses both foods onto
  (0,0).
- **Click convention.**  Plans are stored row-major (`click ROW COL`), matching
  `AutumnBenchEnvWrapper`.  The raw interpreter is column-first; `autumn_env.py` transposes.
  Any export to MARA's own harness must transpose.

## A bug in AutumnBench's own N2NTD planning problem

Its goal asserts `black` across all 24 masked cells (rows 0-1) but N2NTD's background is
`white`, and `render_grid_to_matrix` fills empty cells with the literal background name.
`check_grid_same` can therefore never pass: the public N2NTD planning problem is unreachable
as shipped.  The intent reading is still right -- rows 0-1 empty means the enemy is dead --
but the encoding is broken.  The other four games have background `black` and are consistent.

## Not done

- No eval harness for this schema.  `eval_coverage_plan.py` hardcodes the old schema and
  compares only at the final step; a curated goal should be scored the way MARA's own
  planning env scores (`reached_goal()` every step, terminate with reward 1 on match).
- No MARA `{GAME}_planning.json` export.  Every problem starts from a reset, so all 30 are
  drop-in compatible with an all-ones mask; the exporter is a few lines.
- `offline_learning/compose_plan.py` and its scripts are left in place even though the data
  is erased -- `offline_learning/manual_plan/` (untracked, written during this session)
  references its conventions.
