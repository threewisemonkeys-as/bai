# Evaluating the Claude Code harness on the humanRL prior games

Plan for porting the `cc_autumn` harness to the six games of Dubey et al., ICML 2018,
*Investigating Human Priors for Playing Video Games*
([repo](https://github.com/rachit-dubey/humanRL_prior_games), [paper](https://arxiv.org/abs/1802.10217)).

Everything below the "Findings" heading was measured against a clone, not read off the
README. Working clone during this investigation:
`/tmp/claude-.../scratchpad/humanRL_prior_games` (throwaway — the build step re-clones).

---

## 0. What the benchmark actually is

Six PyGame/PLE games, all forks of PLE's `MonsterKong`. One goal: a player sprite must
reach a princess sprite. Five keys (`left`, `right`, `jump`, `up`, `down`) plus PLE's
noop. Reward is `+1` on touching the princess and `0` everywhere else. Screen is
230×230 RGB; the level is a 16×16 grid of 15px cells loaded from a `map*.txt`.

| game | map | what it removes |
|---|---|---|
| `originalgame` | `map.txt` | nothing — full semantics |
| `nosemantics` | `map.txt` | character identity: princess → white rectangle, enemy → magenta square, player → cyan square, fire → dark blocks. Geometry and platform/ladder art unchanged |
| `noobject` | `map_3.txt` | object-ness: a mosaic of coloured rectangles is blitted behind everything, so object boundaries stop coinciding with colour boundaries |
| `noaffordance` | `map_3.txt` | affordance: photographic textures everywhere, ladders drawn as `ladder2.png` so they no longer look climbable |
| `nosimilarity` | `map_similarity.txt` | similarity: the four wall codes 4/5/6/7 render as four *different* textures and ladders come in two appearances, so same-function objects look unlike |
| `continualgame` | random of `map0..19.txt` | nothing — 20 layouts, player/princess positions read from the map (codes 21/20). A generalization arm, not a prior manipulation |

---

## Findings from the clone (these drive every design choice)

**F1. The five prior games are the same game.** One hand-written 199-action sequence
solves `originalgame`, `nosemantics`, `noobject`, `noaffordance` *and* `nosimilarity`
verbatim. The maps differ slightly and the start/goal pixels differ by 10-15px, but the
route and the timing are identical. **The manipulation is purely perceptual.**

Two consequences:
- an agent given non-visual state would score identically on all five — which makes a
  symbolic arm a *perfect* control, not a weak one;
- sessions leak catastrophically into each other. One solved trajectory is the answer key
  for the other four. Workspaces must not be reachable from one another, and results must
  not be pooled into anything an agent can read (`cc_autumn` already has this problem with
  siblings on one program; here it is total).

**F2. Nothing moves except the player.** `enemy.animateenemy()` is never called; enemies
(`11`) and fire (`12`) are static. So the world is fully deterministic and static, and
frame-differencing isolates the player in one action. That is a large advantage our agent
has over the human subjects, and it should be reported rather than engineered away.

**F3. Physics is continuous, not grid-aligned.** One action = one 30fps frame.
`left`/`right` move 7px; a jump is `v=7`, gravity 1, i.e. +28px over 7 frames and ~14
frames total, carrying ~98px horizontally if you hold a direction. Ladder climbing is
3.5px/frame. Landing snaps to `wall.y - 16`. So a 15px cell grid is **too coarse to plan
a jump with** — any observation must carry sub-cell precision or the game is unplayable.

**F4. Death is cheap and instant, and one life is not a measurement.** A random policy
died in every one of 30 episodes (mean 313 steps, 0 wins). Walking right from the start
of `originalgame` falls into the fire pit at cols 9-11 within 8 actions. With a single
life the score would mostly measure whether the agent happened to walk left.

**F5. `reset` is exact.** Replaying a prefix after `p.reset_game()` reproduces the state
bit-for-bit; death → reset restores `lives=1`. So `cc_autumn`'s "reset is your cheapest
instrument" doctrine transfers unchanged.

**F6. A per-cell tile hash — the obvious `cc_autumn` transplant — collapses.** Hashing
each 15×15 tile gives 58 distinct tiles on `originalgame` (readable), 110 on `noobject`,
and **229/231 out of 256 on `noaffordance`/`nosimilarity`** — i.e. every cell unique, i.e.
noise. There is no compact lossless text encoding of the hard games. The observation has
to be the image.

**F7. A quantised thumbnail degrades along exactly the right axis.** Modal colour per
10px cell, 24-entry palette, rendered as characters: on `originalgame` the platforms and
ladder columns are legible at a glance; on `noobject` boundaries blur; on `noaffordance`
it is noise. Since it is a pure function of the frame buffer, it adds no information the
agent could not compute itself — it only saves tokens — and it preserves the manipulation.

**F8. Speed and size are not a problem.** ~1250 act+capture/s; a 230×230 PNG is ~5KB, so a
2000-action run writes ~10MB of frames. A full run is seconds of simulator time.

**F9. Engine quirks to handle in the actuator.**
- `Board.populateMap()` does `np.loadtxt("map.txt")` — **cwd-relative**. The daemon must
  run in a directory holding the maps (which must not be the agent's workspace).
- `continualgame` picks its map with the global `random.randint`, and `print`s the
  filename. Seed `random` before `init()` to pin a map; the print lands in `daemon.log`.
- `continualgame.resetGroups2()` never restores `lives`, so it terminates after one level
  anyway; treat it as "20 layouts, one level each", selected by seed.
- `game_over()` has a side effect (`numactions = 0` on the 2000-step cap), and PLE
  silently no-ops actions once `game_over()` is true. Do not use either for budgeting —
  the actuator owns the budget.
- `Person.updateWH` swaps in an unscaled 16×16 sprite but never recomputes `rect`, so the
  hitbox stays 15×15 while the drawn player is 16×16. 1px, harmless, worth knowing.

---

## 1. The two decisions

### Decision A — observation space

Given F1/F6, the observation space *is* the experiment. Proposal: **two arms**, run on the
same six games with everything else identical.

**Arm P — pixels (primary).** Every action writes `frames/NNNNNN.png` (the raw 230×230
frame, lossless) and one `logs.txt` block containing the frame path, a frame hash, the
count of changed pixels since the previous frame, and the quantised thumbnail from F7.
Nothing that is not a pure function of the frame buffer ever reaches the log. This is the
only arm that preserves all four manipulations, and it is the honest analogue of what the
human subjects saw — plus programmatic access, which is where our agent beats them (F2).

**Arm O — anonymised objects (control).** The same run, but instead of pixels the log
carries a list of every sprite as `kind=<appearance-hash> box=(x0,y0,x1,y1)`, where the
kind is an arbitrary stable label with no name attached — `A`, `B`, `C`, not `wall`,
`ladder`, `princess`. Perception is removed; semantics are still hidden. By F1 this arm
should be *flat across the five games*; Arm P should reproduce the human ordering. The gap
between them is the number the whole experiment is for: **how much of the prior deficit is
perceptual**.

Optionally **Arm S** as a ceiling: the 16×16 map with anonymised cell codes plus exact
sprite pixel positions — pure planning and control, no perception at all.

Deliberately rejected: a per-cell tile-hash grid as the primary observation (F6 — it is
noise on the two hardest games); naming roles in any arm (that is the prior under test).

### Decision B — episode structure, since there is no exploration phase

`cc_autumn`'s explore-then-one-scored-test does not apply: the user's framing is one
goal-driven game, "complete it". But F4 says a single life measures luck. Proposal:

- **one phase, one shared budget** of *B* actions (default **2000**, the game's own cap —
  the reference solution is 199, a random policy dies at ~313);
- **death restarts the level and costs nothing but the actions already spent**, logged as
  an event. `reset` is available on demand and costs one action (F5);
- the run ends on the win or when the budget is exhausted.

This matches both comparison populations: humans played until they solved it and were
scored on time and deaths; RL agents were scored on frames-to-solve.

Metrics, mirroring the paper so the numbers sit next to the human ones:

| metric | how |
|---|---|
| `solved` | reward 1 reached |
| `actions_to_solve` | budget spent at the win |
| `deaths` | restart events |
| `unique_cells` | distinct 15px cells the player's box entered ("states visited" in the paper) |
| `max_progress` | best cell-graph distance to the princess achieved — the graded score for failures, so the table is not all zeros |

`continualgame` gets a fixed set of seeds → one map each; its score is per-map solved.

### Smaller decisions worth making explicitly

- **Action names.** `jump` is a semantic leak — it names an affordance the no-affordance
  game exists to remove. Default to neutral labels for the fifth key (e.g. `left right up
  down act noop`) and keep `--semantic-actions` as an ablation. Directions are cheap to
  discover and not the prior under test; "jump" is.
- **Repeat syntax.** `./act do right*12 act right*10` — locomotion is 33 actions across
  the board, and a 200-action solution should not need 200 tokens of command line.
- **Budget pilot.** Run one session per game at B=2000 before committing; if every arm
  ceilings or floors, the experiment says nothing.

---

## 2. What to build

Sibling of `cc_autumn/`, same shape, because the shape is the part that already works.

```
cc_humanrl/
  humanrl-code/            # the harness repo (git, like cc_autumn/autumn-code)
    act.py                 # actuator + daemon  <- port of cc_autumn/autumn-code/act.py
    run.py                 # session launcher   <- ~unchanged
    GAME.md                # the brief          <- new, replaces AUTUMN.md
    PROMPT.md              # playing doctrine   <- ~unchanged
    rig/{agents,audit}.py  # agents unchanged; audit needs a new pattern table
    tools/{results,replay,bundle,chart,baselines}.py
    game/                  # vendored humanRL_prior_games + maps + assets, pinned
```

### Reused unchanged
`run.py` (workspace creation, isolated `CLAUDE_CONFIG_DIR`, `claude -p` with the
allow/deny tool lists, stream-json capture, `Report`, replay/rotation), `rig/agents.py`,
the daemon-over-unix-socket design, the `logs.txt` block format, `--plan` recording,
batch-stops-on-event, log rotation across attempts, `tools/results.py` and the bundle/chart
pipeline.

### Rewritten

**`act.py`.** `AutumnGame` → `PleGame` (owns one PLE instance, `chdir`s to a private maps
dir, seeds `random` for `continualgame`). `Legend` → the frame encoders (palette fit on
frame 0, thumbnail, hash, changed-pixel count) for Arm P and the anonymised-sprite dumper
for Arm O. Phase machinery collapses to one phase but the `stop_reason` mechanism stays —
it now fires on death, win and budget. New `[event]` blocks for death/restart. New
`frames/` directory. Keep `RunState.PRIVATE` discipline: the map file, the game name and
the arm must not appear in anything the agent reads.

**`GAME.md`.** New brief, in the register of `AUTUMN.md` and naming nothing:
> You are given a 230×230 image and six keys. One of the things in the image moves when
> you act. Something in the image ends the task when you reach it. Nobody will tell you
> which is which. Frames are written to `frames/`; read them with code.

**`rig/audit.py`.** New `SUSPICIOUS` table. The answer key here is *public*: the game
source, the maps, the assets and the paper are all on GitHub and arXiv, and the model
plausibly knows the paper. Patterns for `humanRL`, `ple/games`, `map*.txt`, `MonsterKong`,
`PLE`, `pygame`, `assets/`, `originalgame|nosemantics|noobject|noaffordance|nosimilarity`,
`princess`, `Dubey`, plus the vendored `game/` path — on top of the network patterns, which
carry over verbatim.

**Baselines** (`tools/baselines.py`): random policy, noop policy, and the 199-action
reference solution as a ceiling check that the environment is wired correctly.

**Replay** (`tools/replay.py`): easier than `cc_autumn`'s — the frames are already PNGs, so
the replay page is a scrubber over them with the action and plan beside each.

### Build steps

1. Vendor the games. Clone at a pinned commit into `cc_humanrl/humanrl-code/game/`,
   apply a small patch series: absolute map paths (F9), deterministic map choice for
   `continualgame`, drop the stray `print`, and a `getGameState()` that returns the sprite
   list Arm O needs. Keep the patches as patches so the upstream diff stays legible.
2. `PleGame` + the encoders, with a test that the 199-action reference sequence still wins
   on all five games (it is the regression test for the whole port).
3. `act.py` port; `tests/` mirroring `cc_autumn/tests` (calibration, selection, session
   config, results, bundle).
4. `GAME.md` / `PROMPT.md` / audit table.
5. Baselines and floors, then a **1-session-per-game pilot on Arm P** before the matrix.
6. Full matrix: 6 games × 2 arms × N seeds, `continualgame` × its map seeds.

### Risks

- **Ceiling or floor.** If Opus solves all six on Arm P, the manipulations are invisible
  and the interesting result is the comparison to the human ordering. If it solves none,
  raise B and add a scaffolded arm. The pilot decides this, cheaply.
- **Recognition.** The model may recognise the paper from a screenshot of
  `originalgame`. That is not cheating — semantic recognition *is* the prior
  `originalgame` tests — but it must not turn into looking the map up. Audit + no web.
- **Cross-session leakage** (F1). Workspaces isolated, results not readable from any
  workspace, one game per session.
- **Vision cost.** Arm P sessions read many PNGs. `PROMPT.md`'s "read it with code, not
  with your eyes" is load-bearing here; watch the compaction count in the reports.

---

## 3. What this gets us

A table of six games × two arms with `solved` / `actions_to_solve` / `deaths` /
`unique_cells`, set beside the human numbers in the paper. Three questions it answers:

1. Does an LLM coding agent show the *same ordering* of prior dependence as humans?
2. How much of the deficit is perception? (Arm P vs Arm O — clean, because of F1.)
3. Does programmatic pixel access (F2: frame-differencing finds the player instantly)
   substitute for the priors humans lose, or not?

---

## Appendix — the reference solution (199 actions, wins on all five prior games)

Verified against the clone with `SDL_VIDEODRIVER=dummy`, `PLE(game, fps=30,
force_fps=True, rng=24)`, one `p.act(key)` per action:

```python
PLAN = (["noop"]*8 + ["left"]*13 + ["up"]*40 + ["right"]*3 + ["noop"]*6
        + ["right"]*13 + ["jump"] + ["right"]*10 + ["noop"]*10 + ["right"]*10
        + ["up"]*45 + ["left"]*3 + ["noop"]*8 + ["left"]*8 + ["noop"]*6
        + ["jump"] + ["left"]*14)
```

Route: fall to the floor, walk left to the left ladder, climb it, step right onto the
middle platform, jump the two-cell gap, climb the right ladder, walk left along the top
platform, then one jump that clears both the enemy and the second gap and lands on the
princess. It is padded with `noop`s to let falls settle, so a tighter solution exists.
