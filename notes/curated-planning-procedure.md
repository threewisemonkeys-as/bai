# Building curated planning problems for an Autumn game

Procedure behind `logs/2026-08-18/curated/` (5 games, 30 problems, 30/30 validated),
written so it can be run on any of the 23 programs in
`MARAProtocol/python_examples/autumnbench/example_benchmark/programs/`.

The output is a short **ladder** per game -- L1 exercises one mechanic, L4 is the thing the
game is *about* -- where a goal is a **concrete end state**: one full rendered frame,
compared for exact equality. Predicates are used freely while generating; nothing
predicate-shaped ships.

Budget roughly a day per game: an hour reading the program, an hour probing, a couple of
hours authoring, and mostly-unattended solve/validate.

---

## Step 0 — Profile the game before you design anything

    uv run python offline_learning/scripts/game_profile.py --game DQ8GC

Five properties decide everything downstream. Each one, if missed, produces a dataset that
looks fine and is quietly broken.

| flag | meaning | what it forces |
|---|---|---|
| `DRIFT` | the world moves with no input | the goal frame stamps the elapsed tick → **compress against a predicate, not the frame** |
| `RNG` | frames are not a function of (seed, actions) | goals past the randomised step are memorisation → cap the ladder there |
| `HIDDEN` | the rendered frame does not determine the next frame | **carry the tick in the search key**, or BFS prunes correct branches |
| `OCCLUSION` | two objects share a cell | two different states render alike; also warns of object stacking |
| `QUIET-AT-RESET` | a noop at t=0 changes nothing | exact-frame goals are well-posed and absorbing goals are reachable |

`HIDDEN` is detected empirically and honestly: collect `(frame, action) → next frame` over
random drives and look for a key with two different successors. That is a *proof* of
unrendered state, not a guess.

Current readings (`logs/2026-08-18/curated/game_profiles.json`):

    27VWC     7  DRIFT  HIDDEN(15/129)  QUIET
    7WWW9    16  HIDDEN(2/393)  QUIET
    7XF97    16  HIDDEN(7/597)  OCCLUSION  QUIET
    83WKQ    16  RNG  QUIET
    ADA85    11  HIDDEN(2/91)  OCCLUSION  QUIET
    AW9WD    16  HIDDEN(7/139)  OCCLUSION(18)  QUIET
    BT3GB    16  OCCLUSION(5)  QUIET
    DGG2C    17  DRIFT  HIDDEN(14/650)
    DQ8GC    16  HIDDEN(10/430)  OCCLUSION  QUIET
    E3V6M    16  QUIET                                  <- cleanest in the catalogue
    EAHCW    16  HIDDEN(1/379)  QUIET
    F5W3N    16  DRIFT  RNG  HIDDEN(3/699)  OCCLUSION
    GLACIER  16  DRIFT  HIDDEN(26/529)
    N2NTD    12  DRIFT  HIDDEN(24/648)
    NRDF6     7  OCCLUSION(5)  QUIET
    NTQ4Y    16  QUIET                                  <- cleanest in the catalogue
    ORCHRD   16  HIDDEN(1/472)  OCCLUSION  QUIET
    QFSVC    16  HIDDEN(2/449)  QUIET
    QQM74    21  HIDDEN(27/615)  QUIET
    S2KT7    16  RNG  OCCLUSION(6)  QUIET
    VA6FQ    10  QUIET                                  <- cleanest in the catalogue
    VQJH6    17  HIDDEN(4/534)  OCCLUSION  QUIET
    ice      16  == BT3GB byte for byte

**`HIDDEN` fires on 17 of 23.** Treat "carry the tick in the search key" as the default and
plain frame-dedup as the exception you justify, not the other way round.

`E3V6M`, `NTQ4Y` and `VA6FQ` are clean on every axis — start there when adding games.

---

## Step 1 — Read the program and write down the objective

Open the `.sexp`. Three things to extract, in order:

1. **Objects and their `next` clauses** — what evolves on its own.
2. **`on` handlers** — everything the agent can cause. This is the mechanic list.
3. **What the game is *for*.** Follow the causal chain to its end: n2ntd's coins give ammo,
   ammo kills the enemy, and the enemy is the only thing that can leave the board. That is
   the objective. Ice's `on down` + day/night gives you a material with two phases and a
   physics for each. Disease's contagion is permanent and reaches every particle.

If you cannot name the objective in one sentence, the game may not have one. 83wkq does not
— it spawns particles that random-walk, nothing accumulates and nothing terminates. Worse,
it turned out to admit **no non-trivial planning problem at all** (see the failure
catalogue), so it should be carried by the ID exam and dropped from the planning set rather
than padded out with token rungs.

**Look specifically for an absorbing state**, and prefer it as L4. All five built games have
one, and killing n2ntd's enemy removes the only autonomously-moving object in any of them,
so the hardest goal in the set is also the cleanest frame: reach it and it holds for ever,
and "matched at step h" and "matched at any step" coincide.

---

## Step 2 — Probe feasibility in the engine, before authoring

Do not design a goal you have not seen happen. Script the route by hand and look at it:

    uv run python -c "
    from offline_learning.curated_plan import trace
    for s in trace('n2ntd', 0, ['left']*5+['up']+['noop']*3):
        print(...)"

This is where the real content gets found. Probing produced: the exact firing window that
kills mario's enemy (t=12..15, dead at t=22); that a 3-drop ice tower melts into a flat
3-wide pool; that all four disease particles are infectable in 12 actions; that s2kt7 seed 0
is degenerate and seed 1 is not. Each of those became a rung.

It is also where you learn the game's traps. **Budget real time here** — it is cheaper than
discovering the trap in a validated dataset.

---

## Step 3 — Author the ladder

5–8 rungs. L1 one mechanic, L2 two composed, L3 a real sub-goal, L4 the objective.

Rules that earned their place:

- **Prefer absorbing goals.** Screen with `step(goal, noop) == goal`. Where it holds the
  problem is timing-robust; where it does not, record it (`quiescent: false`) so a reader
  knows the frame names one tick.
- **Make at least one pair route-discriminating.** `ice-tower` (a stacked column) and
  `freeze-pool` (a flat run) are both three ice blocks at night, but a column can only be
  made by freezing *before* the drops and a run only by freezing *after* they settle.
  Without such a pair you cannot tell whether a solver understands the mechanic or just
  found any route to a shape.
- **Force composition explicitly.** `gather` is unsolvable without repeated click-select,
  because moving a particle requires controlling it. If a mechanic pair never co-occurs in
  your ladder, no result will tell you anything about it.
- **Keep L1 honest.** An h=1 problem is guessable — measured 23–33% by random plan. Ship
  the `random_success` floor with every problem instead of hiding it.

---

## Step 4 — Encode as subgoal chains, not as plans

Each problem is an ordered list of `Seg(goal_predicate, action_alphabet, cap)`. The solver
BFSes each segment and concatenates.

**The per-segment alphabet is the whole trick.** The engine has no usable state snapshot
(`restore_environment` exists but its own `get_environment_string` dump does not round-trip
— it raises `ParseError`), so every node costs a replay from reset at ~880 steps/s. A
14-tick wait under `["noop"]` is a straight line; the same wait under a 5-verb alphabet is
5^14. Segments look like:

    Seg(lambda s: len(water_solid(s)) == 3, ["down", "noop"], 8, "snow x3"),
    Seg(lambda s: tower(s, 4, 3),           ["noop"],        24, "stack"),

Keep every segment ≤ 10 actions. If one will not fit, split it.

---

## Step 5 — Solve, compress, snapshot

1. BFS each segment. **Key on (frame, hidden state) including the tick** wherever Step 0
   flagged `HIDDEN` — see the failure catalogue for what happens if you don't.
2. **Compress against the predicate**, greedily deleting any single action whose removal
   still completes the objective, to fixpoint.
3. Snapshot the frame the compressed plan reaches. That frame is the goal.

Compression is not cosmetic: `staircase` came out of the solver at 60 actions and shipped at
33; `three-rounds` at 25 shipped at 21.

**Compress against the predicate, never the frame.** Where the game has `DRIFT`, deleting
any action re-phases the drifting object and misses the frame whether or not the action
mattered — so a frame-based screen passes vacuously on exactly the game that needs it most.
That hole is what let an earlier generation ship 12-tick problems averaging 2.23 real
actions.

---

## Step 6 — Validate through a *different* engine driver

    uv run python offline_learning/scripts/validate_curated.py <problems.json>

The builder drives the raw interpreter; the validator drives `AutumnBenchEnvWrapper`. A
disagreement between them then surfaces as a failed check rather than a silently wrong
dataset. Checks: V1 the plan reproduces the stored goal and start; V2 the goal occurs at no
earlier step; V3 noop^h misses; V4f/V4t frame- and task-incompressible (report both — the
gap between them *is* the drift story); V5 the `quiescent` flag is right; V6 random plans of
the same length miss, enforced only for h ≥ 2.

---

## Failure catalogue

Every one of these was hit for real. The signature is what you will actually observe.

**An action tick freezes the dynamics it writes to.** An `on` handler that assigns a
variable suppresses that variable's `next` clause for that tick. So `click 8 8` three times
in a row in 83wkq leaves the particle at (8,8) — diffusion frozen — and two `down`s on
consecutive ticks in bt3gb put both drops in the same cell, where they travel together for
ever and three of them render as one.
*Signature:* a goal you designed to exercise passive dynamics is solved instantly by every
arm. *Fix:* interleave noops, and make the predicate require the passive effect.

**The compressor will undo your fix.** `spawn-two` was authored as click/noop/click and the
compressor deleted the noop, because the predicate `len(particles) == 2` was satisfied
without it — collapsing the goal back to "the two cells you clicked", which all three eval
arms then solved 5/5. *Fix:* tighten the predicate until the intended route is the only one
that satisfies it (`... and (4,4) not in particles`). **After compression, re-read the plan
and ask whether it still does the thing you designed.**

**The goal may be reachable by a route that ignores the mechanic.** This is the most
dangerous failure here, because the dataset validates clean and the eval still measures
nothing. In 83wkq every goal frame with k particles is reproducible by clicking those k
cells directly -- clicks freeze diffusion, so the shortcut always works. Both rungs scored
1.00 on all three eval arms, including a program-search arm that models no dynamics at all.
Authoring the reference plan to exercise diffusion did not help: the shortcut is a property
of the GAME, not of the plan.
*Test for it:* before shipping, try to reach the goal by the dumbest route that ignores the
mechanic you meant to exercise -- click the goal cells, walk straight at the target, wait.
If that works, the problem measures goal-reading, not planning. `V6`'s random baseline will
NOT catch this, because the shortcut is targeted rather than random.
*Fix:* find a goal the shortcut cannot reach, or accept that the game affords no planning
problem and say so.

**Frame-dedup prunes the answer.** n2ntd's enemy carries an unrendered `movingLeft` flag; it
renders identically at t=12 and t=14 while moving opposite ways, so a BFS keyed on the frame
merged them and silently killed the only branch that shoots it. *Signature:* a segment you
know is reachable comes back `UNREACHED`. *Fix:* carry the tick in the search key.

**Unrendered latents are invisible to a windowed world model too.** The same latents that
break frame-dedup break cold-start prediction. Measured 1-step accuracy with an empty window
vs one previous frame: dq8gc 0.25 → 0.88, n2ntd 0.35 → 0.60, the other three unchanged. One
frame recovers everything; more adds nothing. And a self-referential window (`prev = current`)
buys exactly zero, because the latent lives in the *difference* between frames.

**Occlusion aliases states.** Mario is invisible for the frame he stands on a coin (coins
render after him); an ant is invisible on food. Two genuinely different states render alike.
The tick in the search key covers this too. Note the profiler's occlusion check is
drive-dependent and under-reports — it missed mario-on-coin because its random drive never
landed there.

**Seed 0 may be degenerate.** s2kt7's `randomPositions` collapses both foods onto (0,0) at
seed 0. Always look at the opening frames of the seed you pick.

**Clicks are row-major here, column-first natively.** Plans are stored `click ROW COL`
matching `AutumnBenchEnvWrapper`; the raw interpreter takes `click(x=col, y=row)` and
`autumn_env.py` transposes. Any export to MARA's own harness must transpose.

**Even the reference benchmark has broken goals.** AutumnBench's own N2NTD planning problem
asserts `black` across its 24 masked cells while N2NTD's background is `white`, so
`check_grid_same` can never pass. Do not assume a shipped problem is reachable — check.

---

## Checklist

- [ ] `game_profile.py` run; flags read and each one's consequence chosen
- [ ] Objective named in one sentence, or the game declared objective-less
- [ ] Absorbing state identified and used as L4
- [ ] Every rung probed in the engine before it was authored
- [ ] At least one route-discriminating pair
- [ ] At least one rung that forces two mechanics to compose
- [ ] Search key carries the tick if `HIDDEN` fired
- [ ] Compression runs against the predicate
- [ ] Post-compression plans re-read: do they still do the intended thing?
- [ ] Dumbest-route check: is the goal reachable while ignoring the mechanic under test?
- [ ] `validate_curated.py` green on every check
- [ ] `random_success` shipped per problem
- [ ] Viz eyeballed end to end — every step should visibly do something

---

## Files

- `offline_learning/scripts/game_profile.py` — Step 0
- `offline_learning/curated_plan.py` — Steps 3–5 (`_problems()` holds the ladders)
- `offline_learning/scripts/validate_curated.py` — Step 6
- `offline_learning/scripts/viz_curated.py` — Step 7
- `notes/curated-planning-set.md` — what the 5-game build actually produced
