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

**F10 — The model has read about this game.**
The assets are Crafter's, the tech tree is in every model's training data, and
`pip install craftax` ships a constants file listing all 67 achievements by name. That is
not a flaw to be engineered around here — it is a property of the thing being measured, and
it belongs in the write-up rather than in the design. What the harness must prevent is the
session *fetching* the source or the wiki mid-run, which the existing audit already covers.

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

Budget: **3000 actions** on full Craftax, 1500 on Classic. Random dies every ~250 actions,
so 3000 is roughly a dozen lives — enough for a curve, and enough to reach the mines. The
constraint is agent tokens, not environment steps (F9), so the pilot sets the final number.

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
* **M2** — `act.py` and its tests, mirroring `tests/test_act.py`.
* **M3** — `GAME.md`, `PROMPT.md`, launcher wiring, audit table and `tests/test_audit.py`.
* **M4** — floors and a **ceiling**. `cc_humanrl` had a 199-action reference solution that
  wins on all five games and was the regression test for the whole port; the analogue is a
  scripted route that reliably reaches iron on a fixed seed, written greedily against the
  symbolic state (harness-side only) and then frozen as a `(seed, actions, achievements)`
  golden triple. Fallback: record one route by hand through `play_craftax` and freeze that.
* **M5** — pilot: one session on full Craftax, budget 3000, plus one on Classic. Read the
  per-episode curve and set the real budget.
* **M6** — the matrix: N seeds on Craftax, N on Classic, and the results table.

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
