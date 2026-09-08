# Building the raw-pixel arm of the humanRL harness

Implementation plan for `notes/humanrl-priors-harness-plan.md`, taking the path where
the agent is given the frame and nothing else.

Working clone for everything measured below:
`https://github.com/rachit-dubey/humanRL_prior_games` at `c9fff26af5df3e1cdea023536da5fda114ba5c60`
(the repo's only branch, last commit 2018-08-20). Run under this project's venv —
pygame 2.6.1 / SDL 2.28.4 / Python 3.12.11 — with `SDL_VIDEODRIVER=dummy`.

---

## 0. What "the raw state, least assistance" is being read as

Arm P of the parent plan, with every derived channel removed. The agent is given the
230x230 RGB frame as a PNG, the action that produced it, and the budget. Not the
quantised thumbnail (F7), not the frame hash, not the changed-pixel count, not the
anonymised sprite list, not a name for anything on the screen.

The rule that decides every later question: **the log tells the agent about the
protocol, never about the world.** How to act, what a batch does, how much budget is
left, whether the run restarted — protocol, and it is told. What moved, what a colour
means, where the goal is, that one action changes only a small region — world, and it
is not.

That reading makes Arm O and Arm S strictly later work, not parallel work. Nothing
below forecloses them; the sprite dumper is the only thing they need that this arm
does not build.

---

## 1. Findings from the clone

F1-F9 in the parent plan were re-measured where they drive the build. Five held as
written (F1, F2, F4, F5, F8 in substance). What follows is what is new, corrected, or
load-bearing enough to restate.

**G1. The reference solution reproduces in our venv, unmodified.** The 199-action
`PLAN` scores `+1` on `originalgame`, `nosemantics`, `noobject`, `noaffordance` and
`nosimilarity`, at 3800-4900 act/s with no PNG writing. So the port has a regression
test on day one, and the vendored game needs no compatibility patch to run under a
2026 pygame.

**G2. `getScreenRGB()` is transposed.** It is `pygame.surfarray.array3d`, which returns
`(width, height, 3)` — indexed `[x][y]`, not `[row][col]`. Saved straight to a PNG the
level comes out mirrored about its diagonal. Every frame must go through
`np.transpose(frame, (1, 0, 2))` before it is written or measured. Verified by eye:
transposed, the princess is top-left, the player bottom-centre and the fire pit
bottom-right, which is the level as the paper shows it; untransposed it is not. This
also means any thumbnail built on the raw array — F7's included — had its rows and
columns swapped.

**G3. Frame 0 is blank.** PLE draws only inside `step()`, so the surface after `init()`
is entirely black, and so is the surface immediately after `reset_game()`. Logging the
start state requires calling `game.newGame.redrawScreen(game.screen, game.width,
game.height)` from the actuator. Without it, action 0 and every post-death restart
write a black PNG, and the first thing the agent ever sees is nothing.

**G4. Death and victory are told apart by the reward, not by `game_over()`.** Both set
`lives = 0`. Walking right from the start of `originalgame` dies on the **6th** action,
not the 8th. On death the reward stays `0` and the player is teleported back to the
start position; on victory the reward is `+1`. Once `lives = 0`, `PLE._oneStepAct`
returns `0.0` and plays nothing, silently, so an unnoticed death turns the rest of a
batch into no-ops.

**G5. `game_over()` is a write.** It sets `self.numactions = 0` on *either* branch that
returns 1 — the 2000-action cap and `lives <= 0` alike. The actuator must never call
it. Read `game.newGame.lives` and the accumulated reward instead.

**G6. Determinism holds three ways.** Two fresh games given the same prefix, and one
game reset and replayed through the same prefix, produce a bit-identical frame. So
`cc_autumn`'s "reset is your cheapest instrument" transfers unchanged, and a
trajectory can be reconstructed from `(game, seed, action list)` alone.

**G7. Frame differencing is clean after frame 0.** One action changes 100-330 pixels,
inside a box roughly 20x20 around the player, on a frame carrying 210 distinct
colours. The only frame that changes wholesale is the first one, because it is
compared against black. This is the advantage the agent has over the human subjects,
and the reason none of it is written into the log.

**G8. PNG writing is the cost, and it does not matter.** act + encode + write runs at
**209 actions/s**, not the 1250 the parent plan measured without the encode. A
2000-action run is ~10 s of simulator time and ~9.5 MB of frames at ~4.8 KB each.

**G9. The raw arm needs no patch series.** Every F9 quirk is already handled by
machinery `cc_autumn` has:

| quirk | what handles it |
|---|---|
| `np.loadtxt("map.txt")` is cwd-relative | the daemon already `chdir`s to `env_dir` before building the environment; put the maps there |
| `continualgame` `print`s its map filename | `env_dir/daemon.log`, which is outside the workspace |
| `continualgame` picks a map with global `random` | seed `random` before construction. It draws **twice** (once in `Board.__init__`, once on `init()`), so pin the seed and *record* `game.newGame.map` rather than predicting which draw lands |
| `game_over()` side effect, PLE no-ops after death | G5: never call it; the actuator owns the budget and the restart |

Patches become necessary only when Arm O needs a `getGameState()` returning sprite
boxes. Until then the vendored tree is upstream, byte for byte, which is worth keeping.

**G10. The agent has no way to open a PNG.** `/usr/bin/python3` here has neither numpy
nor Pillow. An agent told "the observation is an image" and handed no interpreter that
can read one cannot play at all — this arm is the one that turns tooling into a
blocker. And the project venv is the wrong thing to hand it: once the game is vendored,
`import ple` from that interpreter is the answer key. **The workspace needs a `python`
shim pointing at an interpreter of its own, holding numpy and Pillow and nothing
else.** This is a build item, not a footnote.

**G12. The renderer lags the logic on the terminal step, and that is the only
reason the end of a trajectory is visible at all.** Upstream teleports the player
back to its start position on death *and* on victory, inside the same `step()` that
then draws — so on the face of it neither the moment of contact with the fire nor
the moment of contact with the princess is ever drawn. But `Person.setPosition`
updates the position it stores without moving the `rect` the sprite groups blit at
(the same class of bug as `updateWH` in F9), so the drawing lags one event behind
the logic: `player_pos()` reports the start while the frame still shows the player
standing in the fire, or standing on the princess. Verified by eye on both.

Two consequences. The frame an ending action produced is the *only* view of how a
run ended, so it is written before the restart and kept — the block for that action
carries two frames, how it ended and what it restarted into. And
`min_goal_distance`, which reads the logical position, is right to record the
sample before contact rather than the teleported one.

**G13. `continualgame` deals a fresh level on every reset.** `populateMap` calls
`randint(0, 19)` itself, and `resetGroups2` runs it again — and upstream calls that
from *inside* `step()` when a run ends. So left alone, `reset` would not return the
run to its initial state, replay would not be bit-exact, `map_id` would go stale,
and dying would change the level underneath the agent. Its layout is now chosen
from the seed and pinned for the life of the run, by patching `randint` on the
board module at runtime rather than editing the vendored tree — which is what the
parent plan asks of this arm anyway ("20 layouts, selected by seed"). The board
also prints the map filename it loaded, from three different call sites; all three
are swallowed, because `map_id()` is the reliable way to know and there is no
reason for a second copy of the level's name to exist in a log.

**G14. A session need not drive through `./act` directly, and one of them did
not.** `nosimilarity` wrote a `drive.py` that shells out to `./act do` and prints
the player's position after every frame; all 434 of its actions arrived through
that, and `noaffordance` and `continualgame` wrap `./act` in compound commands
that play several batches at once. So the agent's stream cannot be cut at
commands that *look* like they play — anything reading a trajectory back has to
take the batch structure from the log's own `step i/k` counter and treat "the
budget this call reported went up" as the definition of having played.

**G15. Thinking blocks come back encrypted.** Every one of the 330 in the pilot is
an empty string beside a signature, on every session. Whatever a session's private
reasoning was, it is not in the record and cannot be shown or audited. What *is*
recorded is what it wrote down — and the harness is fortunate here by accident:
`--plan` is mandatory to move, so every batch carries a stated intent, and the
observation can only be read with code, so the scripts are the perception.

**G11. The metrics are computable without a movement model.** `map.txt` is 16x16 with
codes `{0,1,2,9,11,12}`; solid floors are rows 4, 9 and 14; the princess sits at pixel
`(30, 47)` and the player starts at `(120, 190)`; the player rect is 15x15. The
reference route visits **49** distinct 15px cells. So `unique_cells` is a set of
`(y//15, x//15)` over logged positions, and progress can be graded by the highest floor
the player has stood on together with the closest it has come to the princess in
pixels — neither needs a reachability search, and the second is well defined on
`continualgame`'s twenty layouts too, where "highest floor" is not.

---

## 2. The arm, precisely

### What is written per action

```
================================================================================
action 137 | budget 137/500 | k2 | step 5/12

plan: k2 moved the sprite right twice, so drive it to the orange column

[frame] frames/000137.png
```

and nothing else. The header carries the protocol; `[frame]` carries a path. `reward`
is appended only when it is non-zero, and `[event]` lines appear only for restarts and
for the end of the run.

The action that ends a trajectory carries two frames, in the order they happened —
how the run ended, then what it restarted into (G12):

```
================================================================================
action 6 | budget 6/500 | k1 | step 6/20

plan: walk k1 as far as it goes

[frame] frames/000006.png
[event] restart
[frame] frames/000006-restart.png
```

Overwriting the first with the second would throw away the only record of how a
trajectory ended, which is the one thing about a failed run worth knowing. A win
writes only the first, because nothing follows it.

### What is deliberately absent

| absent | why |
|---|---|
| quantised thumbnail | a token-saving convenience that survives only on the easy games; on `noaffordance` it is noise, so it would grade the arm unevenly across the very manipulation being measured |
| frame hash | the agent can hash a file; giving it one says "compare consecutive frames" |
| changed-pixel count | this is G7 — the single largest advantage we have over the human subjects — handed over for free |
| sprite list, object boxes | that is Arm O |
| any name for a key | `jump` names the affordance `noaffordance` exists to remove, and `left`/`up` name a direction-to-screen mapping that is itself a prior. The five game keys are `k1`..`k5` |
| any name for the game, the arm or the map | F1: one solved trajectory is the answer key for the other four |

### Actions

`k1 k2 k3 k4 k5 noop reset`. The five game keys are opaque: not `jump`, which names the
affordance `noaffordance` exists to remove, and not `left`/`up` either, because the
mapping from a key to a direction on the screen is itself a prior. Discovering the
mapping costs the agent something the paper's subjects were not charged, and that is
the price of the strict reading — it is a known asymmetry to report, not one to
engineer away. `--semantic-actions`, restoring `left right up down act`, is built as the
ablation that measures exactly what the asymmetry is worth.

`noop` and `reset` keep their names. They are not game keys — `noop` is PLE's absence of
input and `reset` is a harness verb — so naming them says nothing about the world, and
`noop` has to be nameable for a controlled experiment to be possible at all.

Batches take a repeat suffix — `./act do right*12 act right*10` — because locomotion is
33 actions wide and a 200-action solution should not cost 200 tokens of command line.

### Episode structure

One phase, one budget of **B = 500** actions — 2.5x the 199-action reference solution,
and well under the game's own 2000-action cap, so `numactions` is never approached and
G5 stays a thing we avoid rather than a thing we patch. Discovering the opaque key
mapping should cost 5-15 actions, 1-3% of the budget. Death restarts the level, costs
nothing but the actions already spent, and is logged as `[event] restart`. `reset` is
available on demand and costs one action. The run ends on the win or on the budget.

A failure at 500 is ambiguous between "could not" and "ran out", and the pilot is what
tells the two apart: if sessions end at the budget while still making progress, B is
the thing to raise.

A batch **stops on death**, dropping the rest — the state it was planned against is
gone, and G4 says the alternative is a silent run of no-ops. It also stops on the win.

### Metrics, all computed harness-side and never shown

`solved`, `actions_to_solve`, `deaths`, `unique_cells`, `max_floor`,
`min_goal_distance_px`. Recorded from the player position after each action, which the
actuator reads directly and keeps out of `state.json`.

---

## 3. What gets built, in order

Sibling of `cc_autumn/`, same shape:

```
cc_humanrl/humanrl-code/
  game/            vendored humanRL_prior_games @ c9fff26 (ple/ + map*.txt + LICENSE)
  ple_game.py      the wrapper: one PLE instance, frames, restart, positions
  act.py           actuator + daemon
  run.py           session launcher
  GAME.md          the brief
  PROMPT.md        playing doctrine
  rig/{agents,audit}.py
  tools/{baselines,results,replay,bundle,chart}.py
  tests/
```

### M0 — vendor, and give the agent an interpreter

Clone at the pinned commit into `game/`, keeping `ple/` and `map*.txt` and dropping
`docker/ docs/ examples/ tests/ setup.py`. Do **not** install it into the project venv.
Build the agent's interpreter once (`uv venv` + numpy + Pillow) at a path the harness
owns; `make_workspace` writes a `python` shim to it beside the `act` shim.

*Acceptance*: the 199-action `PLAN` scores `+1` on all five prior games through the
vendored tree; two runs of the same prefix write byte-identical PNGs;
`<agent-venv>/bin/python -c "import ple"` fails.

### M1 — `ple_game.py`

`PleGame(name, seed)` owns one PLE instance: seeds `random` before construction for
`continualgame`, calls `init()`, forces the first `redrawScreen` (G3). `frame()`
returns the transposed array (G2); `write(path)` encodes the PNG. `step(token)` returns
`(reward, alive)` reading `lives` rather than `game_over()` (G4, G5). `restart()` is
`reset_game()` plus a forced redraw. `player_pos()` and `map` are exposed for metrics
and are private to the harness.

*Acceptance*: `tests/test_game.py` — reference plan wins through the wrapper; `right*6`
kills the player and the 7th action changes nothing; reset-then-replay is bit-exact;
an orientation assertion that fails on the untransposed array.

### M2 — `act.py`

Cut down from `cc_autumn/autumn-code/act.py`, which is 1565 lines and should land near
600. **Kept verbatim**: `ActError`, the unix-socket daemon (`serve`/`read_line`/`call`),
`cmd_init`, the `restart`/`ATTEMPT` log rotation, `RunState.save` with its `PRIVATE`
discipline, the argparse skeleton, `--plan` recording.

**Rewritten**: `RunState` fields (`game`, `arm`, `budget`, `deaths`, `solved` — with
`game`, `arm` and the map name all `PRIVATE`, per F1); `parse_tokens` for the new
vocabulary and the `TOKEN*N` suffix; `Session.play`, which now writes a PNG and a
three-line block instead of a grid; `preamble`, which describes one phase and one
budget; `cmd_status`; `cmd_board`, which prints the path of the latest frame.
**Dropped**: `Legend`, `render_goal`, `render_choices`, the video and choose and submit
commands, `CuratedProblem`, the whole phase machine — `stop_reason` survives, firing on
death, win and budget.

*Acceptance*: `tests/test_act.py` — a scripted client plays the reference plan through
`./act do` in batches and the run reports solved at 199 actions; a batch that dies
stops early with the remainder dropped and a `[event] restart` block in the log; an
over-long batch is refused before a single action reaches the game; `state.json`
contains neither the game name, nor the arm, nor a map filename.

### M3 — brief, launcher, audit

`GAME.md` in the register of `AUTUMN.md`, naming nothing: an image, six keys, one of
the things in it moves when you act, something in it ends the task, frames are in
`frames/`. `PROMPT.md` adapted — the "read it with code" section becomes about images
and diffing PNGs, the reset and experiment-design doctrine carries over unchanged, the
grid-specific advice goes.

`run.py` is `cc_autumn`'s with the curated battery stripped; `Report` gains the six
metrics of §2. One thing it adds: **a workspace is named after nothing.** cc_autumn
calls a workspace `<program>_<task>`, which is safe because a program's name says
little; here the name would say which manipulation is in play, and F1 makes a
sibling's `notes.md` a finished answer rather than a hint. Workspaces are opaque
labels drawn at random per launch — not derived from the game, because the six names
and the paper behind them are public and anything invertible would be inverted — and
the label mapping, the summary and every session's verdict live under `<launch>/.rig`,
which is not the launch root, because the root is one `..` from every workspace. `rig/agents.py` is copied verbatim. `rig/audit.py` keeps the network,
credentials, another-agent and filesystem-sweep patterns verbatim and takes a new
answer-key table: `humanRL`, `ple/games`, `map\d*\.txt`, `map_similarity`,
`MonsterKong`, `\bPLE\b`, `pygame`, `assets/`, the six game names, `princess`, `Dubey`,
and the vendored `game/` path. `grids_in_context` becomes `frames_in_context` — on this
arm, looking at a PNG is a legitimate channel (it is what the subjects had), so it is
counted rather than flagged.

*Acceptance*: `--dry-run` builds six isolated workspaces; the audit tests port.

### M4 — floors and ceiling

`tools/baselines.py`: random (30 episodes), noop, and the reference plan. Random is the
floor every table needs; the reference is the check that the environment is wired
correctly end to end.

### M5 — the pilot, and the gate

One session per game on the raw arm at B = 500, Opus 5. Read `solved`, cost,
compactions, `frames_in_context`, and the audit. **This is a decision point, not a
result**: if every game is solved the manipulations are invisible at this budget and
the interesting number becomes the comparison to the human ordering; if none is, raise
B or add a scaffolded arm. Committing to the full matrix before this is how the
experiment ends up saying nothing.

### M5 result — the pilot, 2026-09-08

`~/humanrl-runs/20260908-160328`, Opus 5, B = 500, seed 0, one session per game.
Audit clean on all six (after the launch-root fix below), zero web requests, zero
unapproved tools, zero compactions.

| game | actions | deaths | cells | turns | frames read | min | cost |
|---|---|---|---|---|---|---|---|
| originalgame | 159 | 0 | 51 | 85 | 6 | 8.8 | $2.22 |
| noobject | 214 | 1 | 47 | 79 | 6 | 12.7 | $2.89 |
| nosemantics | 258 | 2 | 58 | 94 | 4 | 13.5 | $3.14 |
| noaffordance | 267 | 2 | 57 | 171 | 22 | 30.9 | $8.31 |
| continualgame | 291 | 2 | 35 | 162 | 11 | 18.2 | $5.30 |
| nosimilarity | 434 | 2 | 73 | 233 | 22 | 55.7 | $15.18 |

Against a random floor of 0/30 on every game and a reference ceiling of 199,
`originalgame` was solved in **159** — faster than the hand-written route, from
pixels alone, with permuted opaque keys.

**The gate says: keep B = 500, and do not headline `solved`.** Six of six is a
ceiling, so the paper's own primary column does not discriminate here. Two things
do:

1. **Effort, not actions.** The action counts are bunched (159-434, a 2.7x spread)
   while the work is not: 85 -> 233 turns, $2.22 -> $15.18, 8.8 -> 55.7 minutes, a
   **6.8x spread in cost and 6.3x in wall-clock**. Whatever the manipulations cost
   this agent, they cost it in reasoning rather than in actions.
2. **`frames_in_context` tracks the manipulation.** Four to six frames read by eye
   on the three games whose objects have flat colours, and **22 on both
   `noaffordance` and `nosimilarity`** — the two photographic-texture games. That
   is the agent falling back to the human channel exactly where code-based
   perception stops working, and it is the sharpest single signal in the pilot.
   Decision 4 (allow it, count it) is what made it visible; fencing it off would
   have destroyed the measurement.

Caveats that bound all of this: **n = 1 per game**, so the ordering is suggestive
and nothing more, and the effort columns are plan-reported rather than billed. The
comparison to the paper's human ordering is not made here — it needs the paper's
own table, which this run has no access to and which should not be recalled from
memory.

Two consequences for M6: seeds are what the matrix buys (variance on one game is
currently unknown and could swallow the whole ordering), and the results table has
to carry turns, cost and frames-read beside `actions_to_solve`, or it will report
a flat 6/6 and say nothing.

### Replay — one page per launch

`tools/replay.py <launch> --out replay.html` builds a self-contained viewer, in the
shape of cc_autumn's: a row per game, then a scrubber over the trajectory with the
plan behind each batch, the tokens as played, and every command, script and file the
session wrote in its workspace *between* batches. Built to `cc_humanrl/replay.html`,
gitignored like cc_autumn's for the same reason — it is regenerable and it is 3.2 MB.

The size is the interesting part. Six runs are 1,638 frames and 65 MB of PNGs,
because `noaffordance` and `nosimilarity` are photographic and cost 85 KB a frame.
But an action moves a sprite and nothing else: consecutive frames differ by ~212
pixels even on the worst game, and by 92,000 over the whole of it. So one frame is
stored whole and the rest as five bytes per changed pixel, and 65 MB arrives in 2.5.
A test rebuilds every frame of a live run and compares it against the PNG on disk;
a viewer showing an approximation of what a session saw would be worse than none.

Two things it makes visible that the summary table cannot:

- **Where the effort went.** A bar per batch for the work done before it, on one
  scale across the six runs — the same finding as the cost column, but located.
- **What a session did to see.** The scripts are the perception, and they are the
  only place the manipulations show up as difficulty rather than as a number.

Frames read by eye needed a correction to be worth marking: a 230x230 PNG is
unreadable at native size, so a session crops and magnifies one first and reads
`/tmp/sprite.png`. Counting direct reads found *one* of the twenty-two images
`nosimilarity` looked at, so a crop is now traced back to the command that made it.

### M6 — the matrix

6 games x N seeds on the raw arm, `continualgame` across its map seeds. Arm O follows
as the control it was designed to be, once the raw arm has a number.

---

## 4. Decisions, settled 2026-09-07

1. **Opaque keys.** `k1 k2 k3 k4 k5 noop reset`; `--semantic-actions` is the ablation.
   The strict reading of the arm, taken knowingly: it puts the agent below the human
   condition, which the ablation is what measures.
2. **Death restarts automatically**, rather than leaving the game dead until the agent
   plays `reset`. PLE silently no-ops after death (G4), so the alternative spends the
   agent's actions on a rule about our harness rather than about the world.
3. **`[event] restart` is logged.** Protocol, not world. Silence would read as the world
   teleporting the player, which is a fact about the harness the agent has no way to
   discover.
4. **Vision is allowed and counted.** `Read` on a PNG stays approved — it is the channel
   the human subjects had, and closing it would measure something else. `report.json`
   carries `frames_in_context`; `PROMPT.md` still says to diff them with code rather
   than read hundreds by eye.
5. **B = 500** for the pilot.

---

## 5. Deferred, deliberately

- Arm O (anonymised sprites) and Arm S (map + positions) — the controls the gap is
  measured against, and the only reason the vendored tree will ever need a patch.
- The `--opaque-actions` ablation.
- Any comparison to the paper's human numbers, which is a table over finished runs.
