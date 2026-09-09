# Running the Claude Code agent on Craftax

Plan for a third sibling harness, `cc_craftax/`, after `cc_autumn/` (AutumnBench) and
`cc_humanrl/` (the six prior-ablated MonsterKong games). Target:
[Craftax](https://github.com/MichaelTMatthews/Craftax), Matthews et al. ICML 2024
([paper](https://arxiv.org/abs/2402.16801)).

**Scope: the game and its interface as shipped.** No renamed or permuted actions, no
substituted assets, no withheld reward, and every observation the agent receives is one the
package itself produces. What changes is only who is playing it — a coding-agent session
instead of a policy network — and the number we get out is in the benchmark's own units,
next to the benchmark's own baselines.

Everything under "Findings" was measured against `craftax==1.6.1` installed into a
throwaway venv on this box, not read off the README. Where a number is quoted, the probe
that printed it is named.

---

## 0. What the benchmark actually is

Two environments in one package, sharing a code shape and differing in scale.

| | **Craftax-Classic** | **Craftax** (full) |
|---|---|---|
| origin | JAX re-implementation of Crafter (Hafner 2021) | Crafter + NetHack-style depth |
| view | 7×9 blocks + 2 inventory rows | 9×11 blocks + 4 inventory rows |
| pixel obs | 63×63×3 | 130×110×3 |
| symbolic obs | flat vector | 8268 floats |
| actions | 17 | 43 |
| achievements | 22 | 67 |
| max achievement reward | 22 | **226** |
| world | one 64×64 map | 48×48 × **9 levels** (overworld, mines, dungeon, sewers, vault, troll mines, fire realm, ice realm, graveyard) |
| `max_timesteps` | 10 000 | 100 000 |

There is no goal state. The task is open-ended: unlock as many achievements as you can
before you die. The tech tree is Minecraft's — wood → crafting table → wood pickaxe →
stone → furnace → iron → diamond — extended in the full game with ladders down through
nine levels, a bow, three schools of magic, potions, enchanting, attributes and a
Necromancer boss.

**The baselines we are joining.** Published results are RL and are reported as mean episode
return as a percentage of the 226:

| Craftax-1B (a billion env steps of training) | | Craftax-1M (sample efficiency) | |
|---|---|---|---|
| PPO-GTrXL | 18.3% | Simulus | 6.6% |
| PQN-RNN | 16.0% | Efficient MBRL | 5.4% |
| PPO-RNN | 15.3% | PPO-RNN | 2.3% |
| RND | 12.0% | | |
| PPO | 11.9% | | |

The paper reports **no human baseline** — unlike Crafter, whose paper does.

---

## Findings from the install

**F1 — Reward is tiered, and it is not only achievements.**
`reward = Σ(newly unlocked achievements × coefficient) + 0.1 × Δhealth`, in both variants
(`craftax/game_logic.py:3074`, `craftax_classic/game_logic.py:1695`). Coefficients come from
`achievement_mapping`: **25 achievements worth 1, 18 worth 3, 15 worth 5, 9 worth 8**,
summing to exactly 226 (verified: `Counter(ACHIEVEMENT_REWARD_MAP) == {1:25, 3:18, 5:15,
8:9}`). Classic gives a flat 1 each, max 22. The health term means the reward stream goes
negative on damage — so the headline number is the achievement-weighted sum, and the
per-step scalar is what the agent sees, exactly as an RL policy does.

**F2 — Termination is death, timeout, or the boss.**
`done = timestep ≥ max_timesteps | health ≤ 0 | has_beaten_boss`. No princess, no solve
condition short of the Necromancer. The `solved` column from `cc_humanrl` has no analogue
here; the score is the achievement set.

**F3 — Achievements are per-episode and die with you.**
`state.achievements` is a bool vector the world generator resets. A run spanning several
lives therefore has two numbers, not one: the per-episode return (which is what the
baselines report) and the union over the run.

**F4 — The environment is pure JAX, and that changes the harness.**
`env.step(key, state, action, params)` is functional; state is a pytree. Measured
(`probe4.py`): **2.0 ms/step on CPU** after JIT, and replaying a 120-action prefix twice
gives a bit-identical state when the per-step key is derived as
`jax.random.fold_in(PRNGKey(seed), i)`. Two consequences: replay is exact and costs ~1 s
per 500 actions; and **the action sequence plus the seed is the whole record**, so frames
on disk are a cache rather than the archive — a stronger invariant than either previous
harness had.

**F5 — First-call compile is seconds, everything after is free.**
`reset` 5.1 s, `step` 9.0 s, the 64 px renderer 6.7 s, all one-off. The daemon must warm up
before the first `./act do` returns, as `cc_humanrl`'s does.

**F6 — The package ships three observation channels, all native.**

* **Pixels.** `make_craftax_pixel_renderer(block_pixel_size)` at three cached sizes:
  10 px/block (the RL agent's 130×110 observation), 16 px, and 64 px — the size
  `play_craftax` renders for a human. Jitted: 4.3 ms at 16 px, 25.5 ms at 64 px; a 64 px
  PNG is ~17 KB (832×704 for Craftax, 576×576 for Classic). I looked at all three: the
  130×110 is illegible, 208×176 marginal, and the 64 px frame clean — player, cow, trees,
  and a HUD of heart/food/drink/energy/mana with digits.
* **Symbolic.** An 8268-float vector, designed for a neural net and meaningless raw.
* **Text.** `render_craftax_text(state)` returns ~1.8 KB naming every visible cell
  (`"0, 3: tree"`) and every inventory count — the symbolic content in readable form.

They do not cost the same, which matters for the arm that is off. Per action, jitted:
the 64 px render **25 ms**, the symbolic vector **0.2 ms**, the text render **381 ms** —
it returns a Python string, so it cannot be jitted and re-dispatches its jax work every
call. On a 3000-action run that is nineteen minutes of wall clock. It stays upstream's
function anyway: a faster re-implementation would stop being the package's own, which is
the only reason the channel is worth having.

**F7 — Partial observability is doubled.**
The view is a window centred on the player, *and* a light map darkens everything outside a
radius at night and underground. The world is 48×48 × 9 levels. The agent has to build a
map from a moving keyhole — the operation code does well and a human player does badly.

**F8 — Floors, measured (`floors.py`, 1000 actions, restart on death, 3 seeds).**

| variant | policy | deaths | best episode | union |
|---|---|---|---|---|
| Craftax | random | 3 | 3–4 | **4** |
| Craftax-Classic | random | 5–7 | 3–6 | **4–6** |
| Craftax | noop | 3 | 0 | **0** |

Noop still dies three times in 1000 actions: hunger, thirst and fatigue kill you for
standing still. Random survives ~250 actions a life. Staying alive is part of the score.

**F9 — No GPU is needed.** CPU JAX at 2 ms/step means six concurrent sessions cost a few
hundred MB each and no accelerator. A 3000-action session is ~6 s of environment time and
~140 s of rendering; the wall clock is entirely the agent's.

**F10 — What a human spends, from the six trajectories the README links.**
The dataset is a Drive link off the README (*"run1 is the only trajectory to complete the
game"*), and it is the only record of human play in existence for this environment. All
six, and the README's "mixed-skill" is not a hedge — the spread is enormous:

| run | actions | deaths | first life | final | % of 226 |
|---|---|---|---|---|---|
| run1 | 23 225 | 1 | **23 209** | 222.0 | **98.2%** |
| run2 | 13 471 | 1 | 13 471 | 196.1 | 86.8% |
| run5 | 3 805 | 2 | 1 851 | 72.2 | 31.9% |
| run6 | 3 644 | 2 | 1 819 | 48.2 | 21.3% |
| run3 | 3 337 | 1 | 3 336 | 44.1 | 19.5% |
| run4 | 766 | 3 | 203 | 14.3 | 6.3% |

And the curve along the way, which is the calibration the budget question needs:

| action | run1 | run2 | run3 | run4 | run5 | run6 | **median** |
|---|---|---|---|---|---|---|---|
| 250 | 5.3% | 5.4% | 6.2% | 4.5% | 5.8% | 5.3% | 5.4% |
| 1 000 | 19.0% | 19.9% | 7.5% | — | 8.4% | 8.0% | **8.4%** |
| 2 000 | 27.8% | 23.5% | 14.6% | — | 13.8% | 20.8% | 20.8% |
| **3 000** | 28.3% | 31.3% | 17.2% | — | 26.5% | 21.7% | **26.5%** |
| 5 000 | 30.5% | 57.7% | — | — | — | — | |
| 10 000 | 59.5% | 73.0% | — | — | — | — | |

Three readings. **3000 actions is where the median human (26.5%) clears the best published
RL number** — PPO-GTrXL's 18.3% after a billion training steps — while the weakest run
still going at that point is at 17.2%, level with it. At 1000 only the two experts are
there (19–20%); the median human is at 8.4%, well below PPO. Second, **four of the six
sessions are 3300–3800 actions long**, so ~3000 is roughly what a person actually spends
on this in a sitting. Third, the two expert runs are a *single life* — dying is a novice's
problem here, which makes the multi-life budget a concession to the agent rather than a
property of the game.

Reading the dataset needs two shims, and one of them matters. It predates the package
layout, so its pickles name `craftax.craftax_state`; it predates this jax, so its stored
avals carry a `named_shape` that `ShapedArray.update` no longer takes. And — the one that
would silently corrupt an analysis — **its achievement indices are an older enum**:
decoded against 1.6.1's, `defeat_necromancer` appears to fire at action 863, before the
graveyard is entered. Only the reward channel is index-independent, so every number above
is cumulative reward.

**F11 — The model has read about this game.**
The assets are Crafter's, the tech tree is in every model's training data, and
`pip install craftax` ships a constants file listing all 67 achievements by name. That is
not a flaw to be engineered around here — it is a property of the thing being measured, and
it belongs in the write-up rather than in the design. What the harness must prevent is the
session *fetching* the source or the wiki mid-run, which the existing audit already covers.

**F12 — The environment's reward is per episode, and a new life re-earns all of it.**
`craftax_step` computes `achievement_reward` as `(state.achievements - init_achievements)
* ACHIEVEMENT_REWARD_MAP`, where `init_achievements` is the vector at the *start of the
step* and the vector itself is part of the episode's state. So a fresh episode starts with
nothing unlocked and every achievement pays again. Summed across episodes, reward counts
one achievement once per life, which makes a run-cumulative total meaningless — and, if
shown to a session, actively harmful: the first pilot found that `reset left do` paid +1
every three actions and spent 2200 actions on it (M5). The quantity the benchmark reports
is return on *one* episode, and that is the only aggregate a session should ever see.

**F13 — What a turn costs, measured.**
From the first pilot's 153 turns over 2757 actions, obs `pixels`:

| | |
|---|---|
| actions per turn, genuine play | **4.2** (life 1: 557 actions in ~132 turns) |
| context per turn | **25k → 235k** cache-read tokens, still climbing |
| compactions | **0** — a 1M window and it never got close |
| seconds per turn | 12 early → **29** at 235k |
| cost, input side, at list price | **$11.6** for 18.0M cache reads |

Two things the stream will not tell you. One API response arrives as several `assistant`
events sharing a `message.id`, all carrying the same usage, and their `output_tokens` is a
pre-completion stub — over one finished session they sum to 297 where the result event
reports 33,784. And thinking reaches the stream as a *signature* with zero characters of
text (422KB of signature over 119 blocks), while being 69% of the output tokens in the one
session that reported a total. So per-turn output is not measurable, only apportionable.

---

## 1. The decisions

Three, and only the first is genuinely open.

### Decision A — the observation channels are a switch, and it starts on pixels

The environment produces three observations natively (F6), and which of them a session gets
is **configuration, not a fork of the harness**. `--obs` takes any combination:

    --obs pixels                 # the default, and what we run now
    --obs pixels,text
    --obs text
    --obs pixels,text,symbolic

Every enabled channel is written per action; the log block names the channels it carries so
a run is self-describing. Adding a channel must never change how the environment steps —
the channels are renderings of one state, and a test pins that the same action sequence
produces an identical state under every combination.

**Now: `--obs pixels`.** The state at each timestep is the pixel frame and nothing else,
written as a PNG per action, with the 43 actions under their real names from the `Action`
enum and the scalar reward as `env.step` returns it.

The frame is rendered at **64 px/block** — the package's own `play_craftax` human setting,
not the 10 px the RL observation uses. Same pixels, same information, at the size the
package renders for a person: the 130×110 is unreadable as an image (F6) and downsampling
to it would be a decision, not a default. 832×704 for Craftax, 576×576 for Classic, ~17 KB
a frame.

Keeping the other two wired but off costs almost nothing — `render_craftax_text` is a
function call and the symbolic vector is the env's own `get_obs` — and it means the
comparison against the text-observation condition most published LLM-agent work uses is a
flag on a later run rather than a second harness.

### Decision B — episode structure

The environment's own: an episode runs until death or `max_timesteps`. On death the
harness starts a fresh episode and keeps spending the same action budget. The benchmark's
metric is per-episode return, so episodes are the scoring unit; the session is a sequence
of them.

The one departure from the baselines, and it should be stated in any table we produce:
**PPO is trained on 10⁹ steps and then evaluated; the agent is not trained at all.** Its
episodes-within-a-session are the closest thing it has to learning, which makes the
per-episode curve across a run interesting in its own right — episode 1 versus episode 10
is the agent's learning curve, and it is a curve no RL baseline has an analogue for.

**What the benchmark prescribes, and what it does not.** One number, in Appendix D:
*"The maximum episode length is 100,000 at which the episode is truncated"* (10 000 for
Classic). That is a truncation limit, not a play budget, and nothing reaches it. The
1B and 1M in Craftax-1B / Craftax-1M are *training* budgets for a learner — *"a budget of
1 billion environment interactions is permitted"* — and have no analogue for an agent that
is not trained. So there is no play budget to inherit; ours has to be justified on its own
terms.

Two more things the paper settles while we are here. The challenges are defined on
**Craftax-Symbolic** (*"we also limit the benchmarks to considering symbolic
observations"*), so PPO-RNN's 15.3% is a symbolic-observation number and a pixel run is a
different condition — sayable in a table, not ignorable. And the prose says 65
achievements where 1.6.1 ships 67; both normalise by 226, and the M0 pin is on the
package's own map.

**The budget is a spending decision, not a reporting one.** The actuator records the score
after every action, so a 3000-action run *contains* the 1000-action result: the
achievements-against-actions curve comes out of any run for free, and no single number has
to be chosen for the write-up. What the budget buys is how far along that curve the run
goes, and what it costs.

Proposed: **3000 actions** on full Craftax, 1500 on Classic — and the human curve (F10) is
what makes that number a choice rather than a guess. At 3000 the median human sits at
26.5% of the 226, clear of the 18.3% that a billion training steps buys, and the weakest
run still going is level with it; at 1000 the median human is at 8.4%, *below* PPO, so a
shorter budget would be measuring the wrong end of the curve. 3000 is also about what four
of the six people actually played in a sitting. And it is 13% of the 23 209 actions the
completed run took: it buys the early and middle tree, not the game. Random dies every ~236 actions (median over 61 complete lives,
5 seeds), so it is also roughly a dozen lives — enough for a curve across several of them. Environment cost is nothing
(F9); the bill is the agent. From the `cc_humanrl` pilot — 1623 actions over six sessions
for $37.04 — the rate is **$0.023 per action** at 3.6 actions per tool call, which puts a
3000-action session at $60–90 and a six-seed matrix at $400–550. Craftax should batch
harder than those games did (walking twenty steps is one plan), which is the main thing
that could move it, and the main thing to read off the pilot.

### Decision C — what the log carries

`cc_humanrl`'s rule holds unchanged: the log tells the agent about the protocol, never
about the world. Per action: the action index, the budget, the action played, whichever
observation channels are enabled (now: the frame path), the scalar reward, and `done`.
That is precisely what `env.step` returns, and nothing else — no hash, no changed-pixel
count, nothing computed from the frame.

### Metrics, harness-side

| metric | how |
|---|---|
| `episode_return_pct` | mean episode achievement-weighted return ÷ 226 — **the number that sits beside PPO-RNN's 15.3%** |
| `best_episode_pct` | the best single life |
| `achievements_union` | anything unlocked anywhere in the run |
| `per_achievement` | which ones, by name, per episode — the Crafter-style success-rate table across seeds |
| `episode_curve` | return per episode in order, the within-session learning curve |
| `deaths`, `actions_used`, `max_level`, `unique_cells` | survival, depth, exploration |
| `obs_channels` | which channels the session was given — recorded on every report, since it is the axis later runs will vary |
| cost, tokens, turns, compactions | from the stream, as in the other two harnesses |

---

## 2. What to build

Sibling of `cc_humanrl/`, same shape, because the shape is the part that already works.

```
cc_craftax/
  craftax-code/               # its own git checkout, gitignored from bai
    act.py                    # actuator + daemon   <- port, contract unchanged
    run.py                    # session launcher    <- ~unchanged
    craftax_game.py           # the ple_game.py analogue
    GAME.md / PROMPT.md       # brief + doctrine
    rig/{agents,audit}.py     # agents unchanged; new audit table
    tools/{replay,baselines,results}.py
    tests/
    .agent-venv/              # numpy + Pillow, no jax
    .env-venv/                # craftax + jax — the daemon's interpreter, never the agent's
```

### Reused unchanged
`run.py` (workspace creation, isolated `CLAUDE_CONFIG_DIR`, `claude -p` with the allow/deny
lists, stream capture, `Report`, replay/rotation), `rig/agents.py`, the
daemon-over-unix-socket design, the `logs.txt` block format, `--plan` recording,
batch-stops-on-event, log rotation, `tools/regrade.py`, and the three-file replay-page
pattern (`replay.py` + `replay.css` + `replay_body.html` — the pixel-delta encoder suits
832×704 frames even better than 230×230).

### New

**`craftax_game.py`** — owns one env, its params and its rng discipline. `play(action) →
(observation, reward, done)` where the observation is whichever channels are on, plus
`restart()`, `frame()`, `write_frame()`, `text()`, `symbolic()` and the score readers. The
responsibilities are all findings above: the NoAutoReset envs (F2), the deterministic
per-step key (F4), achievement diffing per episode (F1/F3), the jitted 64 px renderer (F6),
warm-up on construction (F5). Channel selection lives here and nowhere else, so `act.py`
only ever asks for what the run is configured to give.

**`act.py`** — same contract; the daemon holds a JAX process. Batch stops on death and on
budget.

**`tools/readout.py`** — the score at *every* action, which nothing on disk holds
(`result.json` is rewritten in place after each one), recovered by replaying the recorded
history through a fresh engine and checked against the record. Plus the agent side from
the stream: actions per tool call, context per turn, cost apportioned by depth (F13).

**`tools/replay.py`** — the page. Same shape as `cc_humanrl`'s — pick a run, scrub it,
and every action carries the plan its batch was given, the actions in it and the work
done in the workspace before playing them — with two differences forced by the game.
The pictures cannot be inlined as deltas, because the view scrolls with the player
(292,365 of 585,728 pixels change per action, measured), so the default writes a
directory of WebP beside the page and `--inline` builds a single publishable file that
thins only runs of plain movement, never a moment that paid, unlocked, descended or
died. And the page carries the replayed score, level and condition, so the transport
shows where the curve moved rather than only where the actions were.

**`GAME.md`** — the brief. Unlike the humanRL games there is nothing to withhold, so it
says what the environment is: an open-ended survival and crafting world, 43 actions listed
by name, reward arrives when you achieve something, you die of damage or hunger or thirst,
and the run continues into a fresh world when you do. It also states which observation
channels this run gives, generated from the `--obs` setting rather than written by hand, so
the brief cannot drift from what the log actually carries.

**`rig/audit.py`** — a new pattern table. The fence is not about the game's identity (F10)
but about fetching things mid-run: `craftax|crafter` source, `MichaelTMatthews`,
`2402.16801`, `import jax`, `texture_cache`, the wiki — on top of the network,
package-install, second-agent and model-API patterns, which carry over verbatim.
`pip install craftax` would put the world generator inside the workspace and is already
caught by `fetching packages`. The agent's `.agent-venv` gets numpy and Pillow and not jax.

**`tools/baselines.py`** — random and noop (F8, already measured) plus the ceiling below.

### Build steps

* **M0** — pin `craftax==1.6.1` into `.env-venv`, build `.agent-venv`, scaffold the repo.
  *Done 2026-09-08 (`933d260`).* Both interpreters build from `tools/make_*_venv.sh`;
  `.env-venv` is 776 MB of JAX, which is the reason Craftax is not a dependency of the
  parent project. `tests/test_env.py` (12 tests) stands in for `cc_humanrl`'s vendored
  commit: it pins the action and achievement counts, the reward tiers and their sum to
  226, the 64 px frame shapes, that an episode ends only on death or timeout, that a
  prefix replays bit-exactly under `fold_in(PRNGKey(seed), i)`, and that the agent's
  interpreter can open a PNG but cannot `import craftax` or `import jax`.
* **M1** — `craftax_game.py` and its tests: bit-exact replay from a prefix; death reported
  and the game silent afterwards; a fresh episode starts clean; achievement diffing catches
  a known unlock; frame size and orientation; every `--obs` combination parses and writes
  what it claims; **the same action sequence yields an identical state under every channel
  combination**; both variants.
  *Done 2026-09-08 (`3fc3f81`), 27 tests, 39 in the suite.* Settled while building it:
  the world key and the step keys come from two independent splits of the seed, so what a
  seed *draws* and what it *plays out* cannot be confused; a restart deals the same world
  by default (`fresh_world=True` is the generalisation setting); the jitted engine is
  cached per variant, since a `CraftaxGame` that compiled its own would pay 9 s for `step`
  and 7 s for the renderer every time one was made; observations are written to files per
  channel rather than inlined, because 1.8 KB of text render per action would put five
  megabytes of world into the one file `PROMPT.md` tells the session to read back; and the
  package's "Loading Craftax textures" chatter is swallowed, since `./act` is a command
  whose stdout the agent reads.
* **M2** — `act.py` and its tests, mirroring `tests/test_act.py`.
  *Done 2026-09-08, 17 tests, 56 in the suite.* A life ending is not the run ending:
  the batch stops, the block holds the state the life ended in and then the state the
  next life began in, and the budget carries on. The reward goes in the block header —
  it is the channel a policy is given — while the achievement set stays in `result.json`
  beside the environment, because naming what fired would hand over the tech tree a rung
  at a time. `--obs` is threaded through `init`/`serve`, and `restart` rotates
  `frames/`, `text/` and `symbolic/` together with the log.
* **M3** — `GAME.md`, `PROMPT.md`, launcher wiring, audit table and `tests/test_audit.py`.
  *Done 2026-09-09, 31 tests, 87 in the suite.* The audit's line moved: it grades
  reaching outside and never knowledge, and `named_the_game` records recognition as a
  measurement (F11). Two false positives are pinned — a session's own launch paths, and
  a session printing its own `act` shim, which names `<harness>/.env-venv`; both roots
  are scrubbed before any pattern runs, and before `named_the_game` too. `GAME.md`'s
  first paragraph is generated from `--obs`. The dry run builds workspaces, starts both
  variants and records the label mapping under `.rig/`.
* **M4** — floors and a **ceiling**. `cc_humanrl` had a 199-action reference solution that
  wins on all five games and was the regression test for the whole port; the analogue is a
  scripted route that reliably reaches iron on a fixed seed, written greedily against the
  symbolic state (harness-side only) and then frozen as a `(seed, actions, achievements)`
  golden triple.
  *Done 2026-09-09, 7 tests, 94 in the suite.* The route could not be a recording — the
  world moves while you plan against it — so `tools/route.py` re-plans from the block grid
  after every step, verifies each move happened, tunnels through rock when the pickaxe
  allows it (seed 0's iron has no walkable neighbour), drinks before thirst kills it, and
  hits whatever walks up at night. Over ten seeds: **iron pickaxe on nine, all ten alive,
  median 13 achievements in 184 actions**. Seed 0's 115 actions are frozen as a fixed list
  so a router change cannot quietly become the environment's regression test.

  | policy | actions | best life | union | lives | deaths |
  |---|---|---|---|---|---|
  | random | 3000 | 3–5 (1.3–2.2%) | 4–6 | 10–16 | 9–15 |
  | noop | 3000 | 0 | 0 | 12–15 | 11–14 |
  | route | 115–214 | 12–14 (5.3–6.2%) | 12–14 | 1 | 0 |

  Two things this settles. The route sits at roughly where a human is at 250 actions
  (5.3%, F10), so it is a competence floor rather than a ceiling for the game. And **noop
  dies eleven to fourteen times over a 3000-action budget** — thirst does not care whether
  you move — so a session that merely fails to drink will lose everything it has unlocked,
  repeatedly.
* **M5** — pilot: one session on full Craftax at budget 3000. *Classic dropped from the
  pilot — the main variant is the one being evaluated.*

  **The first attempt (`20260909-043025`, workspace 8E3WG) is not a result; it is the
  reason the interface changed.** It played 2757 actions and then stopped being about
  Craftax. What it did before that was good: `look.py` for cell segmentation and sprite
  prototypes, `nav.py`, `world.py`, a `./go` wrapper, and **17 achievements by action
  247 in one unbroken life** — the whole stone-tool chain by action 53 where the
  scripted route needs 115, then drink, cow, coal, a survived night (`wake_up` at 223),
  furnace and torches. That is 7.5% of 226 in 247 actions, above the human median at 250
  (5.3%, F10) and above the route's ceiling.

  | action | union | % | life | depth |
  |---|---|---|---|---|
  | 100 | 12 | 5.3 | 1 | 0 |
  | **247** | **17** | **7.5** | 1 | 0 |
  | 300–557 | 17 | 7.5 | 1 | 0 |
  | 700–2757 | 18 | 8.0 | 30 → 715 | 0 |

  Then it plateaued for 310 actions, and at action 558 it began pressing `reset left do`.
  Its notes call this a decisive finding and every word is true (F12): a new life
  re-earns everything, `reset` cost one action, so wood paid +1 every three actions. It
  spent 226 resets and 715 lives on it while the union score sat at 18 and `max_level`
  never left 0.

  **Three defects, all ours.**
  1. **`reset` was a forty-fourth action.** Craftax has 43 and none of them abandons a
     life; the scope of this port is the game as shipped, so offering one was an
     addition to the interface. Removed — a life now ends only when the world ends it.
  2. **The only aggregate a session could see was the farmable one.** `./act status`
     totalled reward over the whole run, and nothing said what the run was for. `status`
     now shows the *current life's* return, the run total is kept for analysis and no
     longer written to `state.json`, and the brief says the run is judged on its best
     single life — which is the unit the paper reports and gives away nothing about the
     world. `PROMPT.md` gained one clause: repeating the shortest thing that paid is a
     diagnosis, not a plan.
  3. **A hung-up client killed the daemon, and the dead daemon destroyed the report.**
     `sendall` raised `BrokenPipeError` out of the accept loop, so `act stop` failed, the
     exception left `run.py`'s `finally`, and the report of a 2757-action run was never
     written. Both halves fixed, both tested.

  Also found: `mean_episode_pct` — the number the plan names as sitting beside
  PPO-RNN@1B's 15.3% — was never computed. It is now, over the lives the world ended,
  with its n printed beside it, because on a 3000-action run n is often 1 or 2.

  **What the curve actually says about budget — and it is not "spend more".** All 18
  achievements it unlocked are one-pointers, 18 of the 25 that exist. The seven it
  missed are the iron chain (`collect_iron`, `make_iron_pickaxe`, `make_iron_sword`),
  `collect_diamond`, `eat_plant`, and the two combat ones. Everything else — **201 of
  the 226 points** — is in the 3/5/8 tiers, and reading those tiers by name shows they
  are almost entirely underground: `enter_gnomish_mines` and `enter_dungeon` are 5 points
  *each*, and they gate the mobs, chests, potions, bow, enchanting, diamond gear, the
  spells and the four deeper realms.

  So **the highest-value action in this game is going down a level**, and a single one of
  those two is worth more than a quarter of everything this session scored in 2757
  actions. `max_level` was 0 for all of them. That reframes the axis: the question is not
  whether 3000 actions is enough surface time, it is whether a session descends at all.
  3000 stands (it is not what bound this run), and depth is the first thing to read off
  the second attempt.

  *Second attempt: `20260909-053629`, workspace DLT8F, on the fixed interface.*
* **M6** — the matrix: N seeds on Craftax, and the results table. Sizing from F13: a
  genuine 3000-action session is ~710 turns at 4.2 actions/turn, and since context was
  still climbing at 235k with no compaction, **$100–250 and 4–6 hours per seed** — with
  compaction behaviour the dominant uncertainty in that range.

### Risks

* **Floor or ceiling.** If the agent lands near random (4/67) the run says nothing except
  that agentic play does not transfer; if it clears the whole basic tier immediately, the
  interesting axis becomes depth rather than breadth. The pilot decides, cheaply.
* **Token cost.** These are much longer sessions than the humanRL games — 3000 actions
  against 500, with a HUD and a text dump to re-read every batch. `PROMPT.md`'s "read the
  frames with code, not with your eyes" is load-bearing; watch the compaction counter.
* **Comparability.** Our number and PPO's are both mean episode return as % of 226, but
  the training regimes are not comparable at all (Decision B). Say so in the table rather
  than letting the shared denominator imply more than it should.
* **Episode count.** A 3000-action budget is ~12 random-quality lives but could be 2 or 3
  competent ones, and episode-return variance in this environment is large. Several seeds
  are needed before any mean is worth quoting.

---

## 3. What this gets us

1. **An agentic number on Craftax**, in the benchmark's own units, beside PPO-RNN@1B's
   15.3% and the 1M track's 6.6% — the first thing anyone will ask for.
2. **A within-session learning curve** — return per episode across the lives of one run.
   No RL baseline has an analogue, and it is the part of agentic play that is actually
   different in kind: the agent carries notes across deaths where a policy carries weights.
3. **A per-achievement table** across seeds, which says *where* the tech tree stops — the
   diagnostic the aggregate percentage hides.
4. **Continuity.** Third benchmark, same harness, same doctrine, same replay page. The
   marginal cost of the fourth is smaller again.

---

## 4. Deferred, deliberately

* Running the other channels. They are wired (Decision A) and off. The symbolic vector in
  particular is designed for a network rather than a reader, and `render_craftax_text` is
  the same content in a form a reader can use — but both stay one flag away.
* Checkpoint/rewind as an agent-facing instrument. F4 makes it possible and it would be a
  genuinely new capability — but it changes the game rather than measuring the agent.
* `god_mode` / `always_diamond` and the other `EnvParams` levers as difficulty settings.
* Any comparison to human play. Craftax reports none, and Crafter's human-expert numbers
  are on a different environment.
