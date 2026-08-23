# AutumnBench 55-program characterisation for offline belief + perception learning
_Generated 2026-08-22 from `autumn_programs_55.zip` (all 55 `.sexp` sources read in full) plus an empirical probe on the real interpreter (`scripts/autumn55_probe.py`, results `scripts/autumn55_probe_results_s1.json`). Static labels live in `scripts/autumn55_static.py`; this file is produced by `scripts/autumn55_make_report.py`._

## TL;DR

**Recommended core set (Tier A, 19 games).** Keep the 10 benchmark worlds that already work in the pipeline — `ice` (reference), `paint` (control),
`lights`, `hatch`, `disease`, `mario`, `coins`, `grow`, `sand`, `bbq` — and add **nine new worlds** that each cover a characteristic the current set lacks:

| Add | Fills this gap | Grid |
|---|---|---|
| `pacman` | state-dependent autonomous dynamics (ghosts chase every 3rd tick) + agent + walls; cannot terminate (capture bug) | 10 |
| `charge` | hidden accumulating resource (energy, never rendered) + timer + gravity; revealed only through jump height | 7 |
| `lightning_rod` | banded hidden counter + threshold-gated click + 20-tick countdown — the temporal complement of `bbq` | 9 |
| `gameOfLife` | neighbourhood-counting rule; dynamics only on a button = a perfectly observable "step" action with a large effect | 16 |
| `blicket` | causal-structure discovery: a disjunctive 2-factor rule with visibility toggles | 11 |
| `arc_slack` | `disease`'s invisible click-to-select, generalised to 7 objects + 2 buttons + deletion | 16 |
| `balls` | autonomous bouncing physics with a latent velocity (needs a 2-frame window) and a toggleable obstacle | 12 |
| `chomp` | turn-taking hidden mode (rendered), geometric quadrant selection, irreversible removal, game-over state | 7 |
| `twiddle` | 9-colour 2×2 permutation rule on a 3×3 grid — the smallest perception/belief sanity check in the corpus | 3 |

Suggested order if budget is limited: `pacman`, `charge`, `lightning_rod`, `gameOfLife`, `blicket`, `arc_slack`, then `balls`, `chomp`, `twiddle`.

**Tier B (20)** are sound but second-choice (large static clutter, delayed effects, reset-time randomness, non-black backgrounds, or already-measured
weak testbeds). **Tier S (8)** are genuinely stochastic and must be scored distributionally (or on their deterministic sub-rules) — never by exact next-frame match.
**Tier X (8)** are excluded: one crashes, one is unplayable at 50×50 with click-independent random dynamics, others are degenerate or near-stateless.

**Two interpreter facts discovered while probing (both affect existing tooling):**

1. **`render_all()` is not side-effect free.** The collision primitives (`isFreePos`, `isFreeExcept`, every `*NoCollision` move, `nextLiquid`) read an
   occupancy set that is rebuilt *only* inside `renderAll()`; `step()` never refreshes it. The benchmark harness renders after every step, so the "true" game
   is the render-every-step one. Any replay that steps without rendering (e.g. `offline_learning/scripts/game_profile.py::_step`, or a prefix replay that
   only renders the final frame) silently plays a different game: after `left`, `right` is blocked by the object's own stale cells (`mario`, `magnets`,
   `block_breaker`, `lightning_rod` all reproduce this). `curated_plan.py` renders before and after each step and is safe. Rule: call `render_all()` after
   every `step()`, always.
2. **Seed 0 makes `uniformChoice` return the first element, every time** (generalising the known `randomPositions`→(0,0) artifact). At seed 0 `kaleidoscope`
   paints only gold, `tetris` spawns only the first piece, `crystallization` particles all drift the same way off-grid, `minesweeper` is ~all mines. Never generate
   data or measure stochastic games at seed 0. Affected programs (any `uniformChoice`/`randomPositions`): `ants`, `block_breaker`, `bottle`, `chaos_game`, `colour_lines`, `crystallization`, `diffusion`, `dino`, `exp_particles`, `kaleidoscope`, `masters_logic`, `minesweeper`, `particles`, `peg_solitaire`, `SET`, `snake`, `space_invaders`, `tetris`, `twiddle`.

## Characteristics and why each matters for this pipeline

The pipeline learns a perception module P (code) and beliefs B (text) from logged transitions with inverse-dynamics (ID), forward (FD) and contrastive-FD
objectives over a K-step window, then evaluates by multistep planning and the benchmark's MFP / CD / planning tasks. Each characteristic below is a known
failure or success lever of that setup.

| Characteristic | Definition (how it was measured) | Why it matters here |
|---|---|---|
| **Stochasticity** | From source: `uniformChoice`/`randomPositions` sites, classified as *init* (reset only), *event* (on a trigger), *step* (every tick). From probe: first step at which seeds 1 vs 2 diverge under a noop-only policy (`nDiv`) and under a fixed random action sequence (`aDiv`). | Exact-frame FD punishes a correct model; ID/FD ceilings are depressed. *init*-only randomness is fine within a seed but is memorisable (leak) across a single-seed dataset. |
| **Passive dynamics** | From source: none / *scripted* (timers, oscillators — state-independent) / *conditional* (only player-created objects evolve, e.g. falling water) / *autonomous* (always running, state-dependent). From probe: frame-change rate under noops in steps 0–40 and 40–80, and the noop-change rate at in-play states. | Static windows are FD freebies; timer rhythms let ID be gamed by padding; scripted motion is learnable by a counter, autonomous state-dependent motion needs a real model. |
| **Partial observability / hidden state** | From source: unrendered variables (modes, counters, velocities, secret bits, occluded or off-screen objects) and how they are recoverable — from action history (memory), by exploration, or never. | Memory-recoverable latents test the belief window; exploration-recoverable ones test directed exploration; never-recoverable ones (off-screen blobs, secret coin flips) cap every objective. |
| **Action visibility / latency** | Probe: at 16 branch points along a random trajectory, replay the prefix and branch with noop, each arrow, and 8 clicks (4 fixed corners/centre + 2 on occupied cells + 2 on empty cells). `vis h1` = fraction of branches whose t+1 frame differs from the noop branch; `vis h3` = same by t+3. `arrow distinct/5` = mean number of distinct t+1 frames among {noop, 4 arrows} (aliasing); `click distinct/8` = same among the 8 clicks (does click *location* matter). | An action with no visible effect at t+1 is unobservable to h=1 ID (the dominant failure in the GEPA sweep); h3 > h1 means a multistep window is required; low `distinct` means actions alias (ID ceiling < 1). |
| **Relational rules** | From source: adjacency contagion, containment, collision, neighbour counting, graph adjacency, symmetry. | These are what perception code has to compute; where raw frames suffice, P cannot pay off. |
| **Quantitative rules** | From source: counters, thresholds, timers, velocities (and whether they are rendered exactly, in bands, or not at all). | Banded or hidden counters need beliefs that track integers over many steps. |
| **Mode-dependent inputs** | From source: the same key means different things depending on a (possibly hidden) mode. | Forces conditional beliefs; combined with a hidden mode it creates invisible transitions. |
| **Observation size** | Probe: mean/max non-background cells, distinct colours, objects, and cells rendered off-grid (invisible but live). | Perception only pays when raw observations overwhelm the decoder; off-screen cells are unrecoverable state. |
| **Irreversibility / absorbing states** | From source: deletions, deaths, game-over, auto-resets. | Absorbing states truncate useful data; resets create discontinuities a model must explain. |
| **Benchmark membership** | Whether the program is one of the 20 in the public manifest (each of those has an MFP + CD + planning task). | New worlds need curated tasks authored (see the curated-planning procedure) before they can be scored on benchmark metrics. |

## Master table — source-level characteristics

Tier: A core · B secondary · S stochastic tier · X exclude. Inputs: L/R/U/D arrows, C click. Click: what the click location means (global = ignored, button = fixed cells, object = on game objects, free = anywhere).

| Game | Bench ID | Tier | Grid | Bg | Stoch (src) | Stoch (probe) | Passive | Hidden | Relational | Quant | Mode | Inputs | Click |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **arc_slack** | – | A | 16 | black | none | none | none | med | none | none | yes | LRUD C | object+button |
| **balls** | – | A | 12 | black | none | none | autonomous | med | med | med | no | LR C | object |
| **bbq** | 27VWC | A | 7 | black | none | none | autonomous | med | none | high | no | C | object+button |
| **blicket** | – | A | 11 | black | none | none | none | low | med | none | yes | UD C | object |
| **charge** | – | A | 7 | black | none | none | autonomous | high | low | high | no | LR C | global |
| **chomp** | – | A | 7 | black | none | none | none | low | med | none | yes | C | object+button |
| **coins** | QFSVC | A | 16 | black | none | none | conditional | med | low | med | no | LRUD C | global |
| **disease** | DQ8GC | A | 16 | black | none | none | conditional | med | high | none | yes | LRUD C | object |
| **gameOfLife** | – | A | 16 | black | none | none | none | none | high | med | no | C | free+button |
| **grow** | 7XF97 | A | 16 | black | none | none | conditional | low | med | none | no | LRD C | object |
| **hatch** | AW9WD | A | 16 | black | none | none | conditional | high | med | none | no | C | object |
| **ice** | BT3GB | A | 16 | black | none | none | conditional | none | med | none | yes | LRD C | global |
| **lightning_rod** | – | A | 9 | black | none | none | autonomous | high | none | high | no | LR C | object |
| **lights** | E3V6M | A | 16 | black | none | none | none | none | none | low | yes | LRUD C | global |
| **mario** | N2NTD | A | 12 | white | none | none | autonomous | med | med | med | no | LRU C | global |
| **pacman** | – | A | 10 | black | none | none | autonomous | med | med | low | no | LRUD | none |
| **paint** | EAHCW | A | 16 | black | none | none | none | med | none | none | yes | LRUD C | free |
| **sand** | VA6FQ | A | 10 | black | none | none | conditional | med | high | none | yes | C | free+button |
| **twiddle** | – | A | 3 | black | init | none | none | none | high | none | no | C | button |
| **balloon** | – | B | 16 | skyblue | none | none | autonomous | low | med | med | no | C | object |
| **block_breaker** | – | B | 13 | black | init | init | autonomous | med | med | med | no | LR | none |
| **buoyancy** | NRDF6 | B | 7 | black | none | none | conditional | low | high | med | no | C | free |
| **carrace** | – | B | 13 | black | none | none | scripted | med | low | high | no | LRUD | none |
| **chinese_checkers** | – | B | 8 | black | none | none | none | low | high | none | yes | C | object |
| **dino** | – | B | 20 | black | event | step (t=28) | scripted | med | low | low | no | U | none |
| **egg** | – | B | 16 | gray | none | none | conditional | low | low | med | yes | LRUD C | button |
| **gravity** | VQJH6 | B | 17 | black | none | none | autonomous | med | none | none | yes | C | free+button |
| **gravity_3** | QQM74 | B | 21 | black | none | none | autonomous | med | none | med | no | LRUD C | free |
| **kaleidoscope** | – | B | 25 | black | event | event (t=2) | none | none | high | none | no | C | free |
| **lights_new** | – | B | 24 | black | none | none | none | none | med | low | no | C | button |
| **magnets** | 7WWW9 | B | 16 | black | none | none | conditional | none | high | none | no | LRUD | none |
| **nim** | – | B | 17 | black | none | none | none | none | low | none | no | C | object+button |
| **peg_solitaire** | – | B | 5 | black | init | init | none | none | high | none | yes | C | object |
| **ricochet_robots** | – | B | 24 | black | none | none | conditional | med | high | none | yes | LRUD C | object |
| **scotland_yard** | – | B | 20 | black | none | none | conditional | low | med | none | no | LRUD | none |
| **snake** | – | B | 16 | black | event | none | autonomous | low | med | none | no | LRUD | none |
| **tetris** | – | B | 16 | black | event | init | autonomous | none | med | none | no | LRUD | none |
| **waterplug** | NTQ4Y | B | 16 | black | none | none | conditional | med | med | none | yes | C | free+button |
| **wind** | DGG2C | B | 17 | black | none | none | scripted | med | none | med | no | LR | none |
| **ants** | S2KT7 | S | 16 | black | event | event (t=2) | conditional | none | med | none | no | C | global |
| **bottle** | ADA85 | S | 11 | black | event | event (t=50) | conditional | high | low | none | no | C | button+object |
| **colour_lines** | – | S | 10 | black | event | event (t=2) | conditional | med | low | none | yes | C | object+free |
| **crystallization** | – | S | 10 | black | step | step (t=1) | autonomous | none | med | none | no | C | free |
| **diffusion** | – | S | 9 | black | step | step (t=1) | autonomous | none | low | low | no | U C | free |
| **minesweeper** | – | S | 10 | black | init | event (t=19) | conditional | high | low | none | no | C | object |
| **particles** | 83WKQ | S | 16 | black | step | event (t=6) | autonomous | none | none | none | no | C | free |
| **space_invaders** | F5W3N | S | 16 | black | event | step (t=4) | scripted | med | low | low | no | LRU | none |
| **chaos_game** | – | X | 50 | black | event | init | none | med | low | none | no | C | global |
| **exp_particles** | – | X | 16 | black | step | crash | autonomous | none | none | none | no | C | global |
| **lock** | – | X | 9 | black | none | none | conditional | none | low | none | no | LRUD | none |
| **logic_gates** | – | X | 24 | black | none | none | none | none | med | none | no | C | button |
| **masters_logic** | – | X | 12 | black | init+event | init | none | none | med | low | no | C | object+button |
| **rink** | – | X | 28 | black | none | none | conditional | low | low | low | no | LRUD | none |
| **SET** | – | X | 20 | black | event | step (t=1) | conditional | none | med | none | no | C | object |
| **tictactoe** | – | X | 5 | gray | none | none | none | low | low | none | yes | C | object |

## Master table — probe: action visibility and passive dynamics (seed 1, render-every-step)

`vis h1/h3`: fraction of counterfactual branches where the action changed the t+1 / t+3 frame relative to a noop. `distinct`: mean distinct next frames among 5 arrow options / 8 click options. `Passive`: noop-only frame-change rate. `Noop-change (in play)`: at branch points of the random trajectory, how often a noop still changes the frame.

| Game | Tier | Arrow vis h1 | Arrow vis h3 | Arrow distinct/5 | Click vis h1 | Click vis h3 | Click-on-obj h1 | Click-on-empty h1 | Click distinct/8 | Passive 0-40 | Passive 40-80 | Noop-change (in play) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **arc_slack** | A | 0.83 | 0.83 | 4.31 | 0.16 | 0.16 | 0.16 | 0.00 | 2.12 | 0.00 | 0.00 | 0.00 |
| **balls** | A | 0.36 | 0.36 | 2.44 | 0.16 | 0.16 | 0.62 | 0.00 | 2.00 | 1.00 | 1.00 | 1.00 |
| **bbq** | A | 0.00 | 0.00 | 1.00 | 0.23 | 0.24 | 0.44 | 0.00 | 2.62 | 0.05 | 0.10 | 0.00 |
| **blicket** | A | 0.09 | 0.09 | 1.38 | 0.02 | 0.02 | 0.06 | 0.00 | 1.12 | 0.00 | 0.00 | 0.00 |
| **charge** | A | 0.38 | 0.38 | 2.50 | 0.38 | 0.38 | 0.38 | 0.38 | 1.00 | 0.00 | 0.00 | 0.12 |
| **chomp** | A | 0.00 | 0.00 | 1.00 | 0.30 | 0.30 | 0.81 | 0.00 | 3.44 | 0.00 | 0.00 | 0.00 |
| **coins** | A | 1.00 | 1.00 | 5.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| **disease** | A | 0.72 | 0.72 | 3.81 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| **gameOfLife** | A | 0.00 | 0.00 | 1.00 | 0.93 | 0.93 | 0.78 | – | 5.75 | 0.00 | 0.00 | 0.00 |
| **grow** | A | 0.64 | 0.75 | 3.56 | 0.16 | 0.16 | 0.31 | 0.00 | 1.75 | 0.00 | 0.00 | 0.56 |
| **hatch** | A | 0.00 | 0.00 | 1.00 | 0.08 | 0.20 | 0.31 | 0.00 | 1.31 | 0.00 | 0.00 | 0.31 |
| **ice** | A | 0.67 | 0.67 | 3.69 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.81 |
| **lightning_rod** | A | 0.47 | 0.47 | 2.88 | 0.03 | 0.05 | 0.12 | 0.00 | 1.12 | 0.03 | 0.03 | 0.00 |
| **lights** | A | 1.00 | 1.00 | 4.56 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| **mario** | A | 0.66 | 0.66 | 3.62 | 0.06 | 0.06 | 0.06 | 0.06 | 1.00 | 1.00 | 1.00 | 1.00 |
| **pacman** | A | 0.92 | 0.92 | 4.62 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.17 | 0.00 | 0.38 |
| **paint** | A | 0.00 | 0.00 | 1.00 | 0.94 | 0.94 | 0.73 | 1.00 | 7.75 | 0.00 | 0.00 | 0.00 |
| **sand** | A | 0.00 | 0.00 | 1.00 | 0.53 | 0.53 | 0.00 | 1.00 | 5.00 | 0.00 | 0.00 | 0.75 |
| **twiddle** | A | 0.00 | 0.00 | 1.00 | 0.67 | 0.67 | 0.50 | – | 4.38 | 0.00 | 0.00 | 0.00 |
| **balloon** | B | 0.00 | 0.00 | 1.00 | 0.12 | 0.12 | 0.00 | 0.00 | 1.94 | 0.12 | 0.00 | 0.06 |
| **block_breaker** | B | 0.50 | 0.50 | 3.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **buoyancy** | B | 0.00 | 0.00 | 1.00 | 0.50 | 0.50 | 0.00 | 1.00 | 4.69 | 0.00 | 0.00 | 0.31 |
| **carrace** | B | 0.45 | 0.53 | 2.81 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.10 | 0.10 | 0.06 |
| **chinese_checkers** | B | 0.00 | 0.00 | 1.00 | 0.17 | 0.17 | 0.50 | 0.00 | 2.00 | 0.00 | 0.00 | 0.00 |
| **dino** | B | 0.19 | 0.19 | 1.75 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **egg** | B | 0.89 | 0.89 | 4.56 | 0.14 | 0.14 | 0.06 | 0.00 | 2.00 | 0.00 | 0.00 | 0.00 |
| **gravity** | B | 0.00 | 0.00 | 1.00 | 0.73 | 0.75 | 0.00 | 1.00 | 6.69 | 0.00 | 0.00 | 0.94 |
| **gravity_3** | B | 0.00 | 0.64 | 1.00 | 0.51 | 0.51 | 0.00 | 0.97 | 5.00 | 0.00 | 0.00 | 0.88 |
| **kaleidoscope** | B | 0.00 | 0.00 | 1.00 | 0.94 | 0.94 | 0.90 | 1.00 | 7.81 | 0.00 | 0.00 | 0.00 |
| **lights_new** | B | 0.00 | 0.00 | 1.00 | 0.01 | 0.01 | 0.03 | 0.00 | 1.06 | 0.00 | 0.00 | 0.00 |
| **magnets** | B | 1.00 | 1.00 | 5.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| **nim** | B | 0.00 | 0.00 | 1.00 | 0.21 | 0.21 | 0.84 | 0.00 | 2.69 | 0.00 | 0.00 | 0.00 |
| **peg_solitaire** | B | 0.00 | 0.00 | 1.00 | 0.80 | 0.80 | 0.91 | – | 5.44 | 0.00 | 0.00 | 0.00 |
| **ricochet_robots** | B | 0.14 | 0.14 | 1.56 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.69 |
| **scotland_yard** | B | 0.73 | 0.73 | 3.94 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.12 |
| **snake** | B | 1.00 | 1.00 | 2.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **tetris** | B | 0.91 | 0.91 | 3.88 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **waterplug** | B | 0.00 | 0.00 | 1.00 | 0.77 | 0.77 | 0.06 | 1.00 | 7.12 | 0.00 | 0.00 | 0.00 |
| **wind** | B | 0.00 | 0.36 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.95 | 1.00 | 0.94 |
| **ants** | S | 0.00 | 0.00 | 1.00 | 0.69 | 0.69 | 0.69 | 0.69 | 1.00 | 0.00 | 0.00 | 0.94 |
| **bottle** | S | 0.00 | 0.00 | 1.00 | 0.12 | 0.28 | 0.31 | 0.00 | 1.62 | 0.00 | 0.00 | 0.38 |
| **colour_lines** | S | 0.00 | 0.00 | 1.00 | 0.91 | 0.91 | 0.81 | 1.00 | 4.31 | 0.00 | 0.00 | 0.75 |
| **crystallization** | S | 0.00 | 0.00 | 1.00 | 0.60 | 0.60 | 0.12 | 1.00 | 5.69 | 1.00 | 1.00 | 0.12 |
| **diffusion** | S | 0.23 | 0.25 | 1.94 | 1.00 | 1.00 | 1.00 | 1.00 | 6.75 | 0.70 | 0.80 | 1.00 |
| **minesweeper** | S | 0.00 | 0.00 | 1.00 | 0.70 | 0.70 | 0.78 | – | 5.00 | 0.00 | 0.00 | 0.06 |
| **particles** | S | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 6.88 | 0.00 | 0.00 | 0.94 |
| **space_invaders** | S | 0.72 | 0.72 | 3.88 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.95 | 0.95 | 1.00 |
| **chaos_game** | X | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| **exp_particles** | X | – | – | – | – | – | – | – | – | – | – | – |
| **lock** | X | 0.56 | 0.56 | 3.25 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| **logic_gates** | X | 0.00 | 0.00 | 1.00 | 0.01 | 0.01 | 0.03 | 0.00 | 1.06 | 0.03 | 0.00 | 0.06 |
| **masters_logic** | X | 0.00 | 0.00 | 1.00 | 0.13 | 0.13 | 0.53 | 0.00 | 2.06 | 0.00 | 0.00 | 0.00 |
| **rink** | X | 0.25 | 0.25 | 1.88 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| **SET** | X | 0.00 | 0.00 | 1.00 | 0.36 | 0.36 | 0.97 | 0.03 | 3.25 | 0.23 | 0.00 | 0.31 |
| **tictactoe** | X | 0.00 | 0.00 | 1.00 | 0.08 | 0.08 | 0.22 | 0.00 | 1.44 | 0.00 | 0.00 | 0.00 |

## Master table — observation size (random policy, 80 steps, seed 1)

| Game | Tier | Grid | Mean cells | Max cells | Colours | Max objects | Max off-screen cells | Dim |
|---|---|---|---|---|---|---|---|---|
| **arc_slack** | A | 16 | 9.00 | 9 | 4 | 3 | 0 | low |
| **balls** | A | 12 | 5.65 | 6 | 4 | 4 | 2 | low |
| **bbq** | A | 7 | 18.00 | 18 | 9 | 4 | 0 | low |
| **blicket** | A | 11 | 60.00 | 60 | 6 | 3 | 0 | low |
| **charge** | A | 7 | 6.00 | 6 | 2 | 2 | 0 | low |
| **chomp** | A | 7 | 21.96 | 23 | 7 | 6 | 1 | low |
| **coins** | A | 16 | 11.00 | 11 | 2 | 2 | 0 | low |
| **disease** | A | 16 | 4.57 | 5 | 2 | 2 | 1 | low |
| **gameOfLife** | A | 16 | 256.00 | 256 | 4 | 3 | 0 | med |
| **grow** | A | 16 | 30.22 | 32 | 4 | 4 | 3 | low |
| **hatch** | A | 16 | 37.00 | 37 | 3 | 2 | 0 | low |
| **ice** | A | 16 | 8.43 | 12 | 4 | 3 | 0 | low |
| **lightning_rod** | A | 9 | 21.00 | 21 | 3 | 2 | 0 | low |
| **lights** | A | 16 | 5.00 | 5 | 2 | 1 | 0 | low |
| **mario** | A | 12 | 18.64 | 19 | 5 | 5 | 1 | low |
| **pacman** | A | 10 | 8.96 | 11 | 5 | 4 | 0 | low |
| **paint** | A | 16 | 24.67 | 45 | 5 | 1 | 0 | low |
| **sand** | A | 10 | 43.35 | 54 | 5 | 4 | 0 | low |
| **twiddle** | A | 3 | 9.00 | 9 | 9 | 1 | 0 | low |
| **balloon** | B | 16 | 35.00 | 35 | 3 | 1 | 0 | med |
| **block_breaker** | B | 13 | 10.23 | 21 | 5 | 3 | 1 | low |
| **buoyancy** | B | 7 | 35.65 | 40 | 3 | 3 | 0 | low |
| **carrace** | B | 13 | 9.46 | 10 | 4 | 7 | 0 | low |
| **chinese_checkers** | B | 8 | 20.42 | 22 | 3 | 3 | 0 | low |
| **dino** | B | 20 | 7.85 | 10 | 4 | 3 | 6 | low |
| **egg** | B | 16 | 22.00 | 22 | 2 | 2 | 0 | med |
| **gravity** | B | 17 | 22.15 | 36 | 5 | 5 | 186 | low |
| **gravity_3** | B | 21 | 7.75 | 19 | 1 | 1 | 42 | low |
| **kaleidoscope** | B | 25 | 198.75 | 346 | 5 | 2 | 36 | high |
| **lights_new** | B | 24 | 115.00 | 115 | 6 | 9 | 0 | high |
| **magnets** | B | 16 | 4.00 | 4 | 2 | 2 | 0 | low |
| **nim** | B | 17 | 19.21 | 20 | 3 | 8 | 0 | low |
| **peg_solitaire** | B | 5 | 25.00 | 25 | 4 | 1 | 0 | low |
| **ricochet_robots** | B | 24 | 159.00 | 159 | 6 | 18 | 0 | high |
| **scotland_yard** | B | 20 | 142.00 | 142 | 3 | 3 | 0 | high |
| **snake** | B | 16 | 3.00 | 3 | 2 | 2 | 0 | low |
| **tetris** | B | 16 | 9.64 | 16 | 2 | 2 | 2 | low |
| **waterplug** | B | 16 | 35.04 | 50 | 5 | 7 | 0 | med |
| **wind** | B | 17 | 42.95 | 46 | 2 | 2 | 57 | low |
| **ants** | S | 16 | 7.93 | 13 | 2 | 2 | 0 | low |
| **bottle** | S | 11 | 7.47 | 9 | 6 | 5 | 0 | low |
| **colour_lines** | S | 10 | 20.44 | 33 | 5 | 1 | 0 | low |
| **crystallization** | S | 10 | 29.86 | 47 | 2 | 1 | 0 | low |
| **diffusion** | S | 9 | 29.95 | 49 | 3 | 3 | 0 | low |
| **minesweeper** | S | 10 | 100.00 | 100 | 3 | 1 | 0 | med |
| **particles** | S | 16 | 24.91 | 47 | 1 | 1 | 0 | low |
| **space_invaders** | S | 16 | 11.31 | 14 | 4 | 5 | 8 | low |
| **chaos_game** | X | 50 | 27.04 | 50 | 4 | 2 | 0 | high |
| **exp_particles** | X | 16 | – | – | – | – | – | low |
| **lock** | X | 9 | 17.00 | 17 | 3 | 3 | 0 | low |
| **logic_gates** | X | 24 | 99.00 | 99 | 6 | 13 | 0 | high |
| **masters_logic** | X | 12 | 9.00 | 9 | 6 | 3 | 0 | low |
| **rink** | X | 28 | 484.21 | 485 | 2 | 2 | 1 | high |
| **SET** | X | 20 | 123.33 | 135 | 5 | 2 | 45 | high |
| **tictactoe** | X | 5 | 20.06 | 25 | 3 | 2 | 0 | low |

## Tier A: core

Deterministic, every latent is recoverable from history or exploration, action effects are visible within 3 steps, and the rules need real perception or belief content.

- **arc_slack** — 16×16, black bg. 7 particles (green/gold); clicking a particle makes it the sole 'active' one (no visual change); arrows move the active one; gray button toggles its colour, orange-red button deletes it. _(hidden: active flag never rendered; recoverable from the last click (memory))_ Probe: arrows visible 0.83/0.83 (h1/h3), clicks 0.16/0.16, passive 0.00, noop-change in play 0.00, ~9.00 cells, 4 colours. **Why A:** the disease 'click=select' pattern generalised: 7 selectable objects + 2 buttons; clean and deterministic.
- **balls** — 12×12, black bg. Two balls bounce diagonally off grid edges and each other; L/R move a 3-cell wall, click on the wall toggles it solid (white) so balls bounce off it. _(hidden: ball direction (8-way) inferable from two consecutive frames; balls step one cell off-grid before bouncing; quirk: edge test uses ==GRID_SIZE so balls leave the grid for one tick)_ Probe: arrows visible 0.36/0.36 (h1/h3), clicks 0.16/0.16, passive 1.00, noop-change in play 1.00, ~5.65 cells, 4 colours. **Why A:** autonomous physics with a latent velocity and a player-controlled obstacle; needs a 2-frame window.
- **bbq** (`27VWC`) — 7×7, black bg. Gas counter burns down while the fire is on, meat 'cooked' counter rises; clicking bbq toggles fire, fill button adds gas, clicking meat feeds a person (+1/-1/-2 health by doneness) and resets the meat. _(hidden: gas and cooked are integers rendered as 2 / 4 colour bands; prior: benchmark; CD ceiling 0.83; planning floor 20%)_ Probe: arrows visible 0.00/0.00 (h1/h3), clicks 0.23/0.24, passive 0.05, noop-change in play 0.00, ~18.00 cells, 9 colours. **Why A:** cleanest hidden-counter + threshold world; tiny grid keeps perception cheap.
- **blicket** — 11×11, black bg. Two blickets and a machine. Click a blicket to select/hide it, up/down change the selected blicket's shape (plus vs diagonal); the machine lights green iff (blicket1 is plus & visible) or (blicket2 is diagonal & visible). _(hidden: 'selected' is not rendered but follows the last click)_ Probe: arrows visible 0.09/0.09 (h1/h3), clicks 0.02/0.02, passive 0.00, noop-change in play 0.00, ~60.00 cells, 6 colours. **Why A:** blicket-detector causal discovery: a disjunctive 2-factor rule, deterministic, small; random play barely touches it (clicks 0.02) so it needs authored drives.
- **charge** — 7×7, black bg. A 2x2 jumper gains 1 energy every 3 ticks while grounded (cap 4); click launches it up by 'energy' cells, then gravity pulls it down; touching the top 'stop' toggles its colour. L/R move. _(hidden: energy never rendered; revealed by jump height)_ Probe: arrows visible 0.38/0.38 (h1/h3), clicks 0.38/0.38, passive 0.00, noop-change in play 0.12, ~6.00 cells, 2 colours. **Why A:** hidden accumulating resource with a timer, observable only through its effect; 7x7.
- **chomp** — 7×7, black bg. Click a block to select it and every block below-right of it (coloured by current player); the remove button deletes the selection and passes the turn; clicking the poisoned block ends the game. _(hidden: current player is rendered on the player markers)_ Probe: arrows visible 0.00/0.00 (h1/h3), clicks 0.30/0.30, passive 0.00, noop-change in play 0.00, ~21.96 cells, 7 colours. **Why A:** 2-player turn state + a geometric selection rule + irreversible removal, all on 7x7.
- **coins** (`QFSVC`) — 16×16, black bg. Arrow-move an agent over coins to bank bullets (hidden counter); click fires a bullet upward if any are banked. _(hidden: numBullets never rendered; agent can walk off-grid; prior: benchmark; in current pipeline (qfsvc))_ Probe: arrows visible 1.00/1.00 (h1/h3), clicks 0.00/0.00, passive 0.00, noop-change in play 0.00, ~11.00 cells, 2 colours. **Why A:** hidden resource gates whether an action has any effect.
- **disease** (`DQ8GC`) — 16×16, black bg. Arrows move the active particle; clicking an inactive particle swaps which one is active (no visual change); sickness spreads to adjacent particles. _(hidden: which particle is active is never rendered; quirk: MARA env transposes click args; prior: benchmark; core game in current pipeline)_ Probe: arrows visible 0.72/0.72 (h1/h3), clicks 0.00/0.00, passive 0.00, noop-change in play 0.00, ~4.57 cells, 2 colours. **Why A:** already core; hidden selection + contagion.
- **gameOfLife** — 16×16, black bg. All 256 cells are particles (dead=black); click sets a cell alive; the green button applies one Conway step; the silver button clears. Probe: arrows visible 0.00/0.00 (h1/h3), clicks 0.93/0.93, passive 0.00, noop-change in play 0.00, ~256.00 cells, 4 colours. **Why A:** the only neighbourhood-counting rule in the corpus; the 'step' button is a perfectly observable action with a large effect.
- **grow** (`7XF97`) — 16×16, black bg. Cloud moves L/R, 'down' drops water; water reaching a green leaf grows a new leaf above it unless the sun overlaps the cloud; clicking the sun steps it (direction flips at edges). _(hidden: sun's movingLeft bit; quirk: cloud can leave the grid; prior: benchmark; conditional tier)_ Probe: arrows visible 0.64/0.75 (h1/h3), clicks 0.16/0.16, passive 0.00, noop-change in play 0.56, ~30.22 cells, 4 colours. **Why A:** multi-condition rule; already supported.
- **hatch** (`AW9WD`) — 16×16, black bg. Click eggshell cells to break them; broken shells fall if unsupported or vanish when over a feather, revealing the hidden chick. _(hidden: chick occluded; 'broken' flag invisible until a hole opens below; prior: benchmark; most headroom; hardest MFP)_ Probe: arrows visible 0.00/0.00 (h1/h3), clicks 0.08/0.20, passive 0.00, noop-change in play 0.31, ~37.00 cells, 3 colours. **Why A:** exploration-reduced uncertainty; click effects mostly appear at t+2 and only bottom/over-feather shells do anything, so it needs authored drives.
- **ice** (`BT3GB`) — 16×16, black bg. Cloud L/R, 'down' drops water that is liquid by day and ice by night; click toggles day/night and flips every drop. _(prior: reference game)_ Probe: arrows visible 0.67/0.67 (h1/h3), clicks 1.00/1.00, passive 0.00, noop-change in play 0.81, ~8.43 cells, 4 colours. **Why A:** reference / baseline.
- **lightning_rod** — 9×9, black bg. Cloud charge rises 1 every 3 ticks (3 colour bands); L/R move the cloud; clicking it resets charge and, if charge>=15, lights the rod gold for 20 ticks. _(hidden: exact charge, rod countdown and tick phase are invisible; clicking the cloud resets charge, which is only visible when the colour band drops)_ Probe: arrows visible 0.47/0.47 (h1/h3), clicks 0.03/0.05, passive 0.03, noop-change in play 0.00, ~21.00 cells, 3 colours. **Why A:** banded counter + threshold-gated click + a 20-tick countdown: the temporal complement of bbq (L/R are the visible channel; most clicks are invisible resets).
- **lights** (`E3V6M`) — 16×16, black bg. A 5-cell bar: click toggles it on; when off arrows translate it, when on arrows rotate it through 8 orientations. _(prior: benchmark; recommended)_ Probe: arrows visible 1.00/1.00 (h1/h3), clicks 1.00/1.00, passive 0.00, noop-change in play 0.00, ~5.00 cells, 2 colours. **Why A:** mode-dependent arrows, every action visible.
- **mario** (`N2NTD`) — 12×12, white bg. Platformer: gravity, L/R, jump when grounded; three platforms and an enemy oscillate; coins bank bullets (hidden), click fires. _(hidden: bullet count; oscillation direction bits; quirk: white background; CD first-click crash; prior: benchmark; n2ntd core)_ Probe: arrows visible 0.66/0.66 (h1/h3), clicks 0.06/0.06, passive 1.00, noop-change in play 1.00, ~18.64 cells, 5 colours. **Why A:** richest agent+passive world already in the pipeline.
- **pacman** — 10×10, black bg. Arrows move Pac-Man (walls block); two ghosts step toward him every 3rd tick; pellets vanish on contact. _(hidden: tick phase (mod 3); score; quirk: ghost capture never kills (result unassigned) - non-terminating)_ Probe: arrows visible 0.92/0.92 (h1/h3), clicks 0.00/0.00, passive 0.17, noop-change in play 0.38, ~8.96 cells, 5 colours. **Why A:** the corpus' only state-dependent autonomous pursuer; non-terminating thanks to the bug.
- **paint** (`EAHCW`) — 16×16, black bg. Click paints a cell in the current colour; arrows set the colour (up gold, down purple, left green, right blue), the opposite arrow resets to red. _(hidden: colour mode invisible until the next click; prior: benchmark; control world)_ Probe: arrows visible 0.00/0.00 (h1/h3), clicks 0.94/0.94, passive 0.00, noop-change in play 0.00, ~24.67 cells, 5 colours. **Why A:** control.
- **sand** (`VA6FQ`) — 10×10, black bg. Two buttons choose sand or water for clicks; water flows; sand touching water liquefies and flows too. _(hidden: click-type mode; prior: benchmark; conditional tier)_ Probe: arrows visible 0.00/0.00 (h1/h3), clicks 0.53/0.53, passive 0.00, noop-change in play 0.75, ~43.35 cells, 5 colours. **Why A:** hidden mode + contagion + liquid physics on 10x10.
- **twiddle** — 3×3, black bg. 3x3 of 9 colours; clicking a corner rotates that 2x2 block clockwise. Starts one random rotation from solved. _(stochastic: 1 of 4 scrambles; quirk: shuffle applies only one rotation (head of map))_ Probe: arrows visible 0.00/0.00 (h1/h3), clicks 0.67/0.67, passive 0.00, noop-change in play 0.00, ~9.00 cells, 9 colours. **Why A:** 9-colour permutation rule on the smallest grid: a perception/belief sanity check.

## Tier B: secondary

Sound worlds that lose to a Tier-A sibling on cost or signal: heavy static clutter, long action→effect chains, reset-time randomness, non-black backgrounds, or a measured weak testbed in the earlier audit. Use when a specific characteristic is needed.

- **balloon** — 16×16, skyblue bg. Click inside the basket to add a rock (click on a rock removes it); >=3 rocks make the balloon sink 1/tick, fewer make it rise 1/tick. _(hidden: 'weight' bool is a threshold of the visible rock count)_ Probe: arrows visible 0.00/0.00 (h1/h3), clicks 0.12/0.12, passive 0.12, noop-change in play 0.06, ~35.00 cells, 3 colours. **Why B:** nice count-threshold physics but a 36-cell sprite, sky-blue background, and a 15-cell click surface.
- **block_breaker** — 13×13, black bg. Ball bounces off edges and a 3-wide paddle (L/R); 20 balloons at random cells/colours are popped when the ball comes within 1 cell. _(hidden: ball direction inferable from 2 frames; stochastic: balloon positions/colours drawn at reset only)_ Probe: arrows visible 0.50/0.50 (h1/h3), clicks 0.00/0.00, passive 1.00, noop-change in play 1.00, ~10.23 cells, 5 colours. **Why B:** like balls but reset-time randomness means each seed is a different level; deterministic within a seed.
- **buoyancy** (`NRDF6`) — 7×7, black bg. Click to drop a rock; rocks fall, water is displaced upward; each rock inside the crate sinks it one row (max 5). _(hidden: addWeight equals visible rocks in crate; prior: benchmark; CD ceiling 0.20 (effect ~18 frames after click); planning floor 84%)_ Probe: arrows visible 0.00/0.00 (h1/h3), clicks 0.50/0.50, passive 0.00, noop-change in play 0.31, ~35.65 cells, 3 colours. **Why B:** rich liquid physics but the action->effect chain is long; weak for h=1 objectives, decent for multistep.
- **carrace** — 13×13, black bg. Car advances one row every UPDATE_RATE ticks and obstacles descend; L/R steer, up/down change the speed (3..18); crash or finish resets the board and adds a score pip. _(hidden: frame-phase and UPDATE_RATE are invisible; speed only shows as motion period; quirk: obstacles never respawn until a reset)_ Probe: arrows visible 0.45/0.53 (h1/h3), clicks 0.00/0.00, passive 0.10, noop-change in play 0.06, ~9.46 cells, 4 colours. **Why B:** tests timer-phase inference (like wind/space_invaders) without stochasticity; but the board resets itself.
- **chinese_checkers** — 8×8, black bg. Click a peg to select it: legal jump targets (2 cells away over an occupied cell) light up yellow; click a target to jump. _(hidden: active peg only shown via highlights)_ Probe: arrows visible 0.00/0.00 (h1/h3), clicks 0.17/0.17, passive 0.00, noop-change in play 0.00, ~20.42 cells, 3 colours. **Why B:** clean relational rule with rendered affordances; very sparse random-click signal (must hit pegs).
- **dino** — 20×20, black bg. Cactus and bird scroll left (bird re-enters at a random row); 'up' makes the dino jump 7 then fall 1/tick; any collision makes the dino invisible forever. _(hidden: bird spawns off-screen (x=27) and is invisible for 8 ticks; stochastic: bird row on each wrap)_ Probe: arrows visible 0.19/0.19 (h1/h3), clicks 0.00/0.00, passive 1.00, noop-change in play 1.00, ~7.85 cells, 4 colours. **Why B:** one useful key, absorbing death; mostly a timing/reflex world.
- **egg** — 16×16, gray bg. Arrows move a 21-cell egg while gravity is off; the button toggles gravity (button colour shows it) and latches the egg's height - if the egg was above row 10 it shatters into gold pieces that fall. _(hidden: latched height)_ Probe: arrows visible 0.89/0.89 (h1/h3), clicks 0.14/0.14, passive 0.00, noop-change in play 0.00, ~22.00 cells, 2 colours. **Why B:** good latched-threshold rule but a gray background and a large sprite.
- **gravity** (`VQJH6`) — 17×17, black bg. Four edge buttons set the gravity direction; click adds a 2x2 blob; blobs move in the gravity direction each tick and persist off-screen. _(hidden: gravity mode invisible until a blob moves; off-screen blobs; prior: benchmark; planning floor 88%)_ Probe: arrows visible 0.00/0.00 (h1/h3), clicks 0.73/0.75, passive 0.00, noop-change in play 0.94, ~22.15 cells, 5 colours. **Why B:** mode + off-screen hidden state; in pipeline already.
- **gravity_3** (`QQM74`) — 21×21, black bg. Arrows add -1/0/+1 to a shared x/y velocity; every blob moves by it each tick; click adds a blob. No bounds: blobs persist off-screen. _(hidden: velocity only visible through motion; off-screen blobs invisible; prior: benchmark; pair-level ID vacuous; planning floor 100%)_ Probe: arrows visible 0.00/0.64 (h1/h3), clicks 0.51/0.51, passive 0.00, noop-change in play 0.88, ~7.75 cells, 1 colours. **Why B:** arrow effect is one step delayed and saturates at |v|=1; off-screen state is unrecoverable.
- **kaleidoscope** — 25×25, black bg. Click places a 3-cell glitter of random colour; its mirror images in the other three quadrants are recomputed every tick. _(stochastic: colour per click; quirk: reflections can land off-grid)_ Probe: arrows visible 0.00/0.00 (h1/h3), clicks 0.94/0.94, passive 0.00, noop-change in play 0.00, ~198.75 cells, 5 colours. **Why B:** symmetry is a strong perception test, but colour is random and the grid is 25x25 with up to 346 cells.
- **lights_new** — 24×24, black bg. Four click-switches drive three lamps through a fixed combinational rule (switch3 is a master, switch4 kills the supply permanently); wires are static decoration. Probe: arrows visible 0.00/0.00 (h1/h3), clicks 0.01/0.01, passive 0.00, noop-change in play 0.00, ~115.00 cells, 6 colours. **Why B:** combinational-logic discovery but 115 static cells and an 8-cell click surface (1.4%).
- **magnets** (`7WWW9`) — 16×16, black bg. Arrow-move a 2-cell magnet; when opposite poles line up 2 cells apart it is pulled one step; like poles adjacent cancel the move. _(prior: benchmark; dropped in audit (planning floor 40%))_ Probe: arrows visible 1.00/1.00 (h1/h3), clicks 0.00/0.00, passive 0.00, noop-change in play 0.00, ~4.00 cells, 2 colours. **Why B:** small state but a genuinely relational force rule.
- **nim** — 17×17, black bg. Four rows of matches; click toggles a match selected; each row's red button removes that row's selected matches. Probe: arrows visible 0.00/0.00 (h1/h3), clicks 0.21/0.21, passive 0.00, noop-change in play 0.00, ~19.21 cells, 3 colours. **Why B:** paint-level control world with selection + per-row buttons.
- **peg_solitaire** — 5×5, black bg. 5x5 pegs with one hole; click a peg to select it (legal jumps turn pink), click a pink cell to jump and remove the middle peg. Initial selection is random among 4. _(stochastic: one of 4 starting selections)_ Probe: arrows visible 0.00/0.00 (h1/h3), clicks 0.80/0.80, passive 0.00, noop-change in play 0.00, ~25.00 cells, 4 colours. **Why B:** tiny relational puzzle with rendered affordances.
- **ricochet_robots** — 24×24, black bg. Click a robot to make it active (not rendered); an arrow launches it sliding until it hits a wall/robot. _(hidden: active robot; direction while sliding)_ Probe: arrows visible 0.14/0.14 (h1/h3), clicks 0.00/0.00, passive 0.00, noop-change in play 0.69, ~159.00 cells, 6 colours. **Why B:** best multi-step action effect in the corpus, but 24x24 with 159 static wall cells.
- **scotland_yard** — 20×20, black bg. Mr X moves along a fixed rail network: an arrow starts him moving and he keeps going until the next station. _(hidden: direction)_ Probe: arrows visible 0.73/0.73 (h1/h3), clicks 0.00/0.00, passive 0.00, noop-change in play 0.12, ~142.00 cells, 3 colours. **Why B:** travel-to-station is a clean multi-step effect; 142 static cells.
- **snake** — 16×16, black bg. Classic snake with wrap-around; arrows set direction; eating food grows the snake and respawns food at a random free cell. No death. _(hidden: direction; stochastic: food respawn)_ Probe: arrows visible 1.00/1.00 (h1/h3), clicks 0.00/0.00, passive 1.00, noop-change in play 1.00, ~3.00 cells, 2 colours. **Why B:** almost deterministic (one random cell per meal) autonomous mover with growth.
- **tetris** — 16×16, black bg. Tetromino falls 1/tick; L/R move, up/down rotate; on landing a random piece spawns at a random column; no line clears; stack to the top ends play. _(stochastic: piece shape+column per landing)_ Probe: arrows visible 0.91/0.91 (h1/h3), clicks 0.00/0.00, passive 1.00, noop-change in play 1.00, ~9.64 cells, 2 colours. **Why B:** deterministic between landings; good rotation/collision rules.
- **waterplug** (`NTQ4Y`) — 16×16, black bg. Buttons choose vessel/plug/water for clicks; water flows and is held by vessel walls and plugs; buttons also remove plugs or clear all. _(hidden: click-type mode; prior: benchmark; CD never fired in audit)_ Probe: arrows visible 0.00/0.00 (h1/h3), clicks 0.77/0.77, passive 0.00, noop-change in play 0.00, ~35.04 cells, 5 colours. **Why B:** sand's sibling with more buttons; in pipeline.
- **wind** (`DGG2C`) — 17×17, black bg. Every 5 ticks four drops spawn under the cloud and fall, drifting by the wind (-1/0/+1 set by L/R). _(hidden: wind only visible through drift; tick phase; prior: benchmark; dropped in audit)_ Probe: arrows visible 0.00/0.36 (h1/h3), clicks 0.00/0.00, passive 0.95, noop-change in play 0.94, ~42.95 cells, 2 colours. **Why B:** arrow effects delayed one tick; timer gives free FD reward.

## Tier S: stochastic

Fresh randomness enters during play. Score with a distributional metric (likelihood of the logged frame, set-based credit) or only on the deterministic sub-rules (e.g. crystallisation contagion, ant chasing, rock flight); never by exact next-frame match. Always seed ≠ 0.

- **ants** (`S2KT7`) — 16×16, black bg. 2 ants step toward the nearest food each tick; a click spawns 2 food at random cells (click location ignored); food eaten on contact. _(stochastic: randomPositions on every click; quirk: seed 0 collapses randomPositions to (0,0); prior: benchmark; CD never renders at seed 0; planning floor 100%)_ Probe: arrows visible 0.00/0.00 (h1/h3), clicks 0.69/0.69, passive 0.00, noop-change in play 0.94, ~7.93 cells, 2 colours. **Why S:** stochastic spawn location; chase rule is deterministic and learnable with set-based/partial scoring.
- **bottle** (`ADA85`) — 11×11, black bg. Click Suzie/Billy to throw a rock that travels to the bottle over ~10 ticks; each rock secretly is/isn't a breaker (50/50, never rendered); click the broken bottle to mend it. _(hidden: breaksBottle bit is unobservable until impact; stochastic: hidden coin flip per throw; prior: benchmark; flagship uncertainty game; planning floor 96%)_ Probe: arrows visible 0.00/0.00 (h1/h3), clicks 0.12/0.28, passive 0.00, noop-change in play 0.38, ~7.47 cells, 6 colours. **Why S:** the only game that needs probabilistic planning; exact next-frame scoring is wrong here.
- **colour_lines** — 10×10, black bg. Click a ball to make it the mover; click an empty cell to set a destination (the mover walks there 1 cell/tick) AND spawn a random-coloured ball at a random cell. _(hidden: moving flag and destination are not rendered; stochastic: new ball colour+position per empty click)_ Probe: arrows visible 0.00/0.00 (h1/h3), clicks 0.91/0.91, passive 0.00, noop-change in play 0.75, ~20.44 cells, 5 colours. **Why S:** deterministic walk-to-target is learnable but every empty click also injects a random ball.
- **crystallization** — 10×10, black bg. 16 blue particles random-walk (no bounds); a particle adjacent to a red crystal becomes a crystal; click seeds a crystal. _(stochastic: random walk every tick; quirk: particles drift off-grid and are lost)_ Probe: arrows visible 0.00/0.00 (h1/h3), clicks 0.60/0.60, passive 1.00, noop-change in play 0.12, ~29.86 cells, 2 colours. **Why S:** random walk like particles, plus a deterministic contagion rule worth scoring separately.
- **diffusion** — 9×9, black bg. Red and blue cells random-walk (4-way, no collision); 'up' cycles how many membrane segments exist (0..2); click adds red on the left half / blue on the right. _(stochastic: random walk every tick)_ Probe: arrows visible 0.23/0.25 (h1/h3), clicks 1.00/1.00, passive 0.70, noop-change in play 1.00, ~29.95 cells, 3 colours. **Why S:** pure random walk.
- **minesweeper** — 10×10, black bg. 100 hidden cells, each a mine w.p. 1/4 at reset; click reveals one (green safe / red mine); a revealed mine also reveals its 3x3 neighbourhood. _(hidden: layout only recoverable by exploration; stochastic: mine layout per seed; quirk: no numbers/clues - it is a pure lottery)_ Probe: arrows visible 0.00/0.00 (h1/h3), clicks 0.70/0.70, passive 0.00, noop-change in play 0.06, ~100.00 cells, 3 colours. **Why S:** deterministic within a seed (memorisable), unpredictable across seeds.
- **particles** (`83WKQ`) — 16×16, black bg. Click adds a particle; every particle random-walks one cell per tick. _(stochastic: step; prior: benchmark; stochastic)_ Probe: arrows visible 0.00/0.00 (h1/h3), clicks 1.00/1.00, passive 0.00, noop-change in play 0.94, ~24.91 cells, 1 colours. **Why S:** irreducible random walk.
- **space_invaders** (`F5W3N`) — 16×16, black bg. Enemy rows oscillate on a 10-tick cycle; hero L/R, 'up' fires; every 15 ticks a random enemy fires downward. _(hidden: tick phase; stochastic: shooter chosen every 15 ticks; prior: benchmark; weak testbed)_ Probe: arrows visible 0.72/0.72 (h1/h3), clicks 0.00/0.00, passive 0.95, noop-change in play 1.00, ~11.31 cells, 4 colours. **Why S:** periodic randomness.

## Tier X: exclude

Excluded: unrunnable, degenerate, near-stateless, or dominated by a better sibling.

- **chaos_game** — 50×50, black bg. Each click adds the midpoint between the last point and a uniformly random vertex (click location is ignored) - a Sierpinski chaos game. _(hidden: which point is 'last' is not visually marked; stochastic: random vertex per click + random start; quirk: start point is black on black)_ Probe: arrows visible 0.00/0.00 (h1/h3), clicks 1.00/1.00, passive 0.00, noop-change in play 0.00, ~27.04 cells, 4 colours. **Why X:** 50x50, click location irrelevant, every transition random; nothing deterministic to learn.
- **exp_particles** — 16×16, black bg. Particles random-walk; click doubles every particle. _(stochastic: step; quirk: CRASHES in the interpreter (updateObj lambda error) - unrunnable)_ **Why X:** does not run.
- **lock** — 9×9, black bg. Arrow-move a 2-cell key; when it sits in the keyhole the wall slides left one cell per tick - forever, off the grid. _(quirk: wall never stops/returns (condition can never be met))_ Probe: arrows visible 0.56/0.56 (h1/h3), clicks 0.00/0.00, passive 0.00, noop-change in play 0.00, ~17.00 cells, 3 colours. **Why X:** one trigger, then a degenerate absorbing state.
- **logic_gates** — 24×24, black bg. Two click-switches; AND/OR/NOT/XOR outputs and wires light accordingly. Probe: arrows visible 0.00/0.00 (h1/h3), clicks 0.01/0.01, passive 0.03, noop-change in play 0.06, ~99.00 cells, 6 colours. **Why X:** only 4 reachable states; 99 static cells; 8-cell click surface.
- **masters_logic** — 12×12, black bg. Mastermind: click guess cells to cycle colours, press enter to get 4 hint pegs (placed in random order); previous guesses scroll down. _(stochastic: code at reset; hint peg order per guess; quirk: CorrectAnswer.show initialised true - the code is visible)_ Probe: arrows visible 0.00/0.00 (h1/h3), clicks 0.13/0.13, passive 0.00, noop-change in play 0.00, ~9.00 cells, 6 colours. **Why X:** the deduction is moot (answer visible) and hint placement is random.
- **rink** — 28×28, black bg. A skater steps with arrows; stepping onto the 22x22 rink makes it slide 2 cells/tick in that direction until it leaves the ice. _(hidden: slide direction; quirk: skater can walk off the grid)_ Probe: arrows visible 0.25/0.25 (h1/h3), clicks 0.00/0.00, passive 0.00, noop-change in play 0.00, ~484.21 cells, 2 colours. **Why X:** 484 static cells for a 1-cell agent; momentum already covered by ricochet_robots.
- **SET** — 20×20, black bg. 9 card slots refilled at random from only 3 card types; select 3 cards - they vanish if colours are all-same/all-different, otherwise deselect. _(stochastic: refill draws; quirk: deck reduced to 3 cards; template cards rendered off-screen)_ Probe: arrows visible 0.00/0.00 (h1/h3), clicks 0.36/0.36, passive 0.23, noop-change in play 0.31, ~123.33 cells, 5 colours. **Why X:** degenerate deck and random refills.
- **tictactoe** — 5×5, gray bg. Click an empty board cell to place the current player's mark; play stops after a win or a full board. _(hidden: turn parity; quirk: winner border never renders)_ Probe: arrows visible 0.00/0.00 (h1/h3), clicks 0.08/0.08, passive 0.00, noop-change in play 0.00, ~20.06 cells, 3 colours. **Why X:** <=9 transitions per episode; gray background.

## Practical notes for adding new worlds

- **Tasks.** Only the 20 manifest programs have MFP/CD/planning instances. A new world needs a curated planning ladder (frame goals) and, if wanted, CD/MFP
  instances — the curated-planning procedure (5 diagnostics + failure catalogue) applies directly. `gameOfLife`, `chomp`, `twiddle`, `blicket` have natural
  exact-frame goals; `pacman` (eat all pellets), `charge` (reach the stop), `lightning_rod` (light the rod) have natural predicate goals.
- **Backgrounds.** `mario` (white), `balloon` (skyblue), `egg`/`tictactoe` (gray) need the non-black goal-renderer fix; everything in Tier A except `mario` is black.
- **Click surfaces.** Random exploration finds nothing in `lights_new`/`logic_gates` (8 clickable cells of 576), `balloon` (~15), `bottle` (3 cells), `hatch`
  (only bottom/over-feather shells do anything). Directed exploration or authored drives are required there.
- **Off-screen state.** `gravity` (186 cells), `wind` (57), `gravity_3` (42), `kaleidoscope` (36), `SET` (45 template cards) keep live objects outside the
  grid; `coins`/`grow`/`rink` let the agent walk off. Those states are unrecoverable from frames — prefer drives that stay on-grid.
- **Click args are (col,row)** at the interpreter; the MARA env for `disease` transposes them (known).
- **Replay faithfully:** render after every step (see TL;DR #1), and never at seed 0 for any program in the stochastic list.

## Method

- Static: every `.sexp` read in full; randomness sites grepped (`uniformChoice`, `randomPositions` are the interpreter's only RNG entry points); stdlib semantics
  (`nextLiquid`, `nextSolid`, `*NoCollision`, `isFreeExcept`) read from `Autumn.cpp/autumnstdlib/stdlib.sexp`.
- Probe (`scripts/autumn55_probe.py`): per game, seed 1 base; determinism check (same seed + actions twice); seed 1 vs 2 vs 3 divergence under noop-only and
  random policies; 80-step noop run and 80-step random run (40 % arrows / 60 % uniform clicks, same action RNG for all games); 16 counterfactual branch points
  with 13 branches each (noop, 4 arrows, 8 clicks), prefix replayed from reset with `render_all()` after every step, outcomes compared at t+1 and t+3.
  Interpreter: `Autumn.cpp/build/interpreter_module` with the harness stdlib `MARAProtocol/python_examples/autumnbench/autumnstdlib.py`.
- One program (`exp_particles`) raises inside the interpreter and is reported as a crash.
