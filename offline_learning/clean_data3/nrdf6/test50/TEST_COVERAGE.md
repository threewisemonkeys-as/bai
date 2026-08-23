# nrdf6 — TEST50 held-out coverage

Large held-out test pool for **nrdf6**. Whitelist `noop,click`, `keep_action_params=TRUE`
(the full `click ROW COL` string is the ID label). All trajectories are **freshly driven**
from a clean seed-0 reset with `autumn_drive.py NRDF6` (row-major `click_ROW_COL`), then
sliced verbatim — distinct action sequences and configurations from `train/` (a
cross-trajectory generalization test).

Verified: `T.verify_pool('.../clean_data3/nrdf6/test50','noop,click',context_k=9)` →
**49 scored target transitions** (within the 50 ± 2 spec).

## 1. Core dynamics (from dynamics.txt)

- **D1 — Click spawns a rock** at the clicked (row,col) iff that cell is free
  (`isFreePos`). Clicks on crate wall/floor, water, or an existing rock do nothing.
  Click is the only meaningful action (left/right/up/down have no handlers).
- **D2 — Gravity on rocks (passive, every step):** each rock falls one cell if the cell
  below is in-bounds and unoccupied; else it stays.
- **D3 — Crate weight / sinking (passive, every step):** if `addWeight < (#rocks inside
  the crate interior)` AND that count `< 5`, `addWeight++` and the whole crate moves DOWN
  one row. Max 4 sinks (floor reaches row 6); then the `<5` gate **freezes** it.
- **D4 — Water reacts (passive, every step):** water flows and is displaced upward by
  rocks sharing its cell; visible as `blue~move` slosh and the `black+N blue-N` swap when
  the crate sinks into the water.

Initial layout (seed 0): crate walls at cols 1,5 (rows 0–2), floor row 2 (cols 1–5);
water fills rows 3–6; no rocks. Free cells: rows 0–1 cols 0,2,3,4,6 and row 2 cols 0,6.

## 2. Pool composition (16 episodes → 49 targets)

Action histogram: **17 click / 32 noop**; 14 distinct click locations
(`0 3`×4, `0 2`, `0 4`, `1 2`, `1 4`, `0 0`, `0 6`, `2 0`, `2 6`, `1 0`, `1 6`,
plus negatives `0 5`, `6 3`, `2 3`). The passive dynamics (D2/D3/D4) can only be scored on
`noop` targets, so the pool is deliberately noop-heavy (that is the whole point of this
game's test) while click LOCATIONS are spread across rows 0,1,2,6 and cols 0–6.

| ep | drive slice | targets (action → role) |
|---|---|---|
| 0 | col3 `[0-3]` | `click 0 3` D1+ ; `noop` **D3+ SINK** (y1→2)+D2+D4 ; `noop` D2+ fall / D3− |
| 1 | col2 `[1-4]` | `click 0 2` D1+ ; `noop` **D3+ SINK** ; `noop` D2+ fall / D3− |
| 2 | col4 `[2-5]` | `click 0 4` D1+ ; `noop` **D3+ SINK** ; `noop` D2+ fall / D3− |
| 3 | i12  `[3-6]` | `click 1 2` D1+ ; `noop` **D3+ SINK** ; `noop` NO_CHANGE D2−/D3− (settled) |
| 4 | i14  `[0-3]` | `click 1 4` D1+ ; `noop` **D3+ SINK** ; `noop` NO_CHANGE D2−/D3− (settled) |
| 5 | col0 `[0-4]` | `click 0 0` D1+ ; 3× `noop` D2+ fall / **D3− (rock outside crate, no sink)** + D4 |
| 6 | col6 `[0-4]` | `click 0 6` D1+ ; 3× `noop` D2+ fall / **D3−** + D4 |
| 7 | o20  `[0-2]` | `click 2 0` D1+ ; `noop` D2+ fall / D3− + D4 |
| 8 | o26  `[0-2]` | `click 2 6` D1+ ; `noop` D2+ fall / D3− + D4 |
| 9 | s1   `[0-3]` | `click 1 0` D1+ ; 2× `noop` D2+ fall / D3− + D4 |
| 10| s2   `[0-3]` | `click 1 6` D1+ ; 2× `noop` D2+ fall / D3− + D4 |
| 11| slow `[6-9]` | `click 0 3` D1+ (busy) ; `noop` **D3+ SINK (y2→3)** ; `noop` D2+ / D3− |
| 12| slow `[12-15]`| `click 0 3` D1+ ; `noop` **D3+ SINK (y3→4)** ; `noop` D2+ / D3− |
| 13| slow `[26-30]`| `click 0 3` D1+ (5th rock) ; `noop` D2+ settle / **D3− (crate FULL, frozen)** ; 2× `noop` NO_CHANGE (frozen) |
| 14| negprobe `[0-3]`| **`click 0 5` D1− (wall)** ; `noop` NC ; **`click 6 3` D1− (water)** |
| 15| rockneg  `[3-6]`| `noop` NC D2−/D3− ; **`click 2 3` D1− (on existing rock)** ; `noop` NC |

## 3. Per-dynamic scored-target coverage (positives / negatives)

| Dynamic | Positives (scored ≥4) | Contrastive negatives |
|---|---|---|
| **D1 click-spawn** | **14** spawn targets, 11 distinct free locations across the grid (rows 0/1/2, cols 0–6); ID label = where the new silver cell appears | **3** no-spawn: `click 0 5` (wall), `click 6 3` (water), `click 2 3` (existing rock) → NO_CHANGE, defeat "click always spawns" |
| **D2 gravity** | **>20** rock-fall targets (in-crate during sinks + outside-crate free-fall, all `silver~move(1.0/…)`) | **6** settled `noop` where a rock is present and does NOT fall (ep3/4 settle, ep13 frozen ×2, ep15 ×2) → defeat "noop always moves silver down" |
| **D3 crate-sink** | **7** sink targets `brown~move(1.0,0)` — 5 first-sinks (y1→2, cols 2/3/4) + 2 deeper sinks (y2→3, y3→4). Step%4 of sinks = {0,1,2,3} (all residues) | **~18** `noop` where a crate is present but does NOT sink: rock falling **outside** the crate (ep5–10, 12 targets), rock falling inside but weight already caught up (ep0/1/2 fall, ep11/12), **crate FULL & frozen** (ep13, 3 targets). Non-sink noops at step%4=1: **10** vs 3 sinks → the `step%4==1` clock that fooled the original train scores far WORSE than the true rule |
| **D4 water** | **7** `black+N blue-N` displacements on the sinks + many `blue~move` sloshes as rocks fall into the water (ep5,7–13) | rides on D2/D3; no separate negative (water reacts only to rocks) |

**Key contrastive (the nrdf6 gap):** the motivating failure was a `step%4==1` clock that
explained the crate motion. Here sinks occur at all four step-parities, and at **every**
parity there are more non-sink `noop`s than sink `noop`s (e.g. parity 1: 10 non-sink vs 3
sink) — including the strongest negative, a **crate packed full of rocks that will not
sink** (ep13, floor on row 6, `<5` weight gate frozen). A clock or a "noop always sinks the
crate / moves silver" shortcut is punished under both ID and FD.

## 4. How TEST50 differs from train

- **Fresh trajectories, new situations.** `train/` reuses the original clean_data2/nrdf6
  recorded run (clicks almost all at `(0,3)`, only first-sinks y1→2). TEST50 is 14 freshly
  driven scenarios with **new click locations** — interior `(1,2)/(1,4)`, outside drops
  `(2,0)/(2,6)/(1,0)/(1,6)`, and new negatives `(0,5)` wall, `(6,3)` water, `(2,3)` rock.
- **Deeper crate states.** Train only sinks the crate once (y1→2). TEST50 adds y2→3 and
  y3→4 sinks and a **full/frozen** crate — states train never reaches as targets.
- **Parity-scattered sinks + matched negatives** (via leading noops that shift the sink's
  Step value), directly targeting the clock shortcut rather than merely re-using it.

## 5. Not covered / caveats

- **D1 no-spawn negatives are ID-aliased with `noop`.** A click on an occupied cell
  produces NO_CHANGE, identical to a `noop` NO_CHANGE, so an ID oracle cannot recover the
  action for those 3 targets (6% of the pool). They are included for their **FD** value
  (predict no-change; defeat "click always spawns"). This aliasing is intrinsic to the
  game (a no-op click leaves no trace) and is minimized to 3 targets. Every other target
  is ID-observable (spawn location recoverable; noop = no new silver).
- **Deep sinks (ep11, ep12) carry water-intrusion noise** (`blue~move`, fractional
  `silver~move` as multiple rocks move) — the `brown~move(1.0,0)` sink signal is still the
  dominant, legible change, but these frames are busier than the clean first-sinks. This is
  the true D4 behaviour as the crate submerges and is unavoidable for depth variety.
- **D4 water is never an isolated target** (water only reacts to rocks); it is always
  scored jointly with a D2 fall or D3 sink. That matches the SEXP (no independent water
  goal).

viz.html is a filmstrip of the whole pool (all 16 episodes concatenated; episode
boundaries are visible as Step-number resets).
