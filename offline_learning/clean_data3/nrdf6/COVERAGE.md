# nrdf6 — clean_data3 coverage

Whitelist: `noop,click`. `keep_action_params=TRUE` (click LOCATION is part of the
label, e.g. `click 0 3`). Click coordinates are `click ROW COL`.

## 1. Core dynamics (extracted from dynamics.txt)

- **D1 — Click spawns a rock.** On `click`, if the clicked cell is a free position
  (`isFreePos`) a new silver Rock appears at that exact (row,col). If the cell is not
  free (crate wall/floor, water, or an existing rock) nothing happens. Click is the ONLY
  meaningful action; left/right/up/down have no handlers.
- **D2 — Gravity on rocks (passive, every step).** Each rock falls one cell per step if
  the cell below is in bounds and not occupied (by another rock or the crate
  wall/floor); otherwise it stays. Rocks settle on the crate floor, other rocks, or the
  grid bottom.
- **D3 — Crate weight / sinking (passive, every step).** Count rocks inside the crate
  interior. If `addWeight < (#rocks inside)` AND that count `< 5`, `addWeight++` and the
  whole crate moves DOWN one row. The crate sinks at most 4 rows (floor reaches row 6)
  and then **freezes** once the inside-count hits 5 (the `< 5` weight gate).
- **D4 — Water reacts (passive, every step).** Water flows as a liquid each step and is
  displaced upward by rocks that share its cell. Visible as `blue` shifts and the
  `black+N blue-N` swap when a rock pushes water out as the crate sinks.

(No color changes, no win/termination/reward — free-running sandbox.)

## 2. Coverage of each dynamic as a SCORED TARGET, and the gap in the ORIGINAL train pool

The original `clean_data2/nrdf6/train` is one 84-row episode. GEPA scores a
**balanced-20** sample of consecutive `noop/click` pairs. Change-tag histogram of the
original whitelisted pairs: 32 NO_CHANGE, 12 spawns (`black-1 silver+1`), and the rest
small `noop` motion; the crate sink `brown~move(1.0,0)` appears in only **4** pairs
(36→37, 40→41, 44→45, 48→49) — all `noop`.

| Dynamic | TARGET under ID? | TARGET under FD? | In a balanced-20 of the original? | Gap |
|---|---|---|---|---|
| D1 click-spawn | Yes — label = where the new silver cell appears | Yes — predict the new silver cell at the clicked (row,col) | Likely (12 spawns exist), but mixed with 32 NO_CHANGE pairs | The NO_CHANGE "click did nothing" pairs are rarely paired against spawns at matched locations, so a "click always spawns" shortcut isn't punished. |
| D2 gravity | Weak — passive (fires on `noop`), so action is NOT recoverable from the fall alone | Yes — predict the rock one row lower | Present but as `noop`, swamped | Few clean `silver~move(1.0)` falls survive a balanced sample; no settled "rock does NOT fall" negative to defeat "noop always moves silver down". |
| D3 crate-sink | Weak — passive `noop` only | Yes — predict crate floor one row lower + water displaced | **Major gap**: only 4 sink pairs exist and balanced-20 may grab ≤1; the original lets a spurious `step % 4 == 1` clock explain the motion | Need all 4 sinks scored AND step-parity-matched non-sink `noop`s (rock above the crate / crate already full) so the clock shortcut scores worse than the true conditional rule. |
| D4 water | Context (rides on D2/D3 pairs) | Yes — water shift accompanies sink/fall | Present as context | Adequately covered wherever D2/D3 are targets; not curated separately. |

**Summary of gaps fixed:** force all 4 sinks to be scored targets; add step-parity /
surface-matched negatives (rocks falling above a stationary crate; clicks on occupied
cells; a settled crate where passive dynamics are frozen) so the ID/FD objectives cannot
be satisfied by a "noop always moves X" or "every k-th step the crate sinks" shortcut.

## 3. Curated slices (each = one episode of verbatim original rows)

Pool = **20 scored targets**, balanced **12 click / 8 noop**. Slice = consecutive
ORIGINAL steps; each internal pair is a scored target (windows never bleed across slices).

| ep | steps | target pair(s) | dynamic / role |
|---|---|---|---|
| 0 | 35,36,37,38 | 35→36 spawn `click 0 3`; 36→37 SINK `noop`; 37→38 settle `noop` | D1+, D3+ (`brown~move 1.0`, water displaced), D2+ |
| 1 | 39,40,41 | 39→40 spawn `click 0 3`; 40→41 SINK `noop` | D1+, D3+ |
| 2 | 43,44,45 | 43→44 spawn `click 0 3`; 44→45 SINK `noop` | D1+, D3+ |
| 3 | 47,48,49 | 47→48 spawn `click 0 3`; 48→49 SINK `noop` | D1+, D3+ |
| 4 | 27,28,29 | 27→28 spawn `click 0 0`; 28→29 fall `noop` (`silver~move 1.0`) | D1+ (location≠0,3), D2+; **D3 negative** (rock falling above crate → crate does NOT move) |
| 5 | 31,32,33 | 31→32 spawn `click 0 6`; 32→33 fall `noop` (`silver~move 1.0`) | D1+ (location), D2+; **D3 negative** |
| 6 | 55,56 | 55→56 spawn `click 0 2` | D1+ (location) |
| 7 | 59,60 | 59→60 spawn `click 0 4` | D1+ (location) |
| 8 | 51,52 | 51→52 spawn `click 0 3` | D1+ |
| 9 | 19,20 | 19→20 `click 0 1` NO_CHANGE | **D1 negative** — clicked a brown crate-wall cell (not free) → no spawn |
| 10 | 23,24 | 23→24 `click 5 3` NO_CHANGE | **D1 negative** — clicked a water cell (not free) → no spawn |
| 11 | 75,76 | 75→76 `click 0 3` NO_CHANGE | **D1 negative** — clicked a cell already holding a silver rock (not free) → no spawn |
| 12 | 76,77 | 76→77 `noop` NO_CHANGE | **D2 + D3 negative** — crate full & settled (floor on row 6, inside-count ≥5): passive dynamics FROZEN, nothing falls, crate does not sink |

### Contrastive structure (per dynamic)

- **D1 (spawn):** 9 positives across distinct click locations (`0 3`×5, `0 0`, `0 6`,
  `0 2`, `0 4`) — the label varies with where the silver cell appears, forcing ID/FD to
  learn click→spawn-at-coordinate. 3 NO_CHANGE negatives at occupied cells (crate wall,
  water, existing rock) defeat "click always spawns".
- **D2 (gravity):** clean `silver~move(1.0)` falls (ep4, ep5) + the fall inside every
  sink pair. Negative: ep12 settled `noop` where rocks do NOT move → defeats "noop
  always moves silver down".
- **D3 (sink):** all 4 sink pairs are scored targets (`brown~move(1.0,0)`). Negatives:
  step-parity-matched `noop`s where the crate is present but does NOT sink — ep4/ep5
  (rock still falling above the crate, addWeight not yet incremented) and ep12 (crate
  full, weight gate `<5` frozen). Together these punish the spurious `step % 4` clock the
  original pool allowed.
- **D4 (water):** rides on the sink targets (`black+N blue-N`) and the fall targets
  (`blue~move`).

## 4. Verification

```
T.verify_pool('prototypes/perc_invdyn/clean_data3/nrdf6/train','noop,click')
→ 20 scored target transitions
  by action: click 0 3:6, noop:8, click 0 0:1, click 0 6:1, click 0 2:1,
             click 0 4:1, click 0 1:1, click 5 3:1
  4 noop targets show brown~move(1.0,0) (sinks); 2 show silver~move(1.0) (clean falls);
  3 click + 1 noop show NO_CHANGE (negatives).
```

## Note vs reference (clean_data2/nrdf6_key)

This reuses the proven nrdf6_key slice design (same 9 spawns / 4 sinks / 3 click
negatives) and **improves** it by adding ep12 — a settled `noop` NO_CHANGE — as an
explicit contrastive negative for D2 (settled rock doesn't fall) and D3 (crate frozen by
the `<5` weight gate when full), which the reference lacked. To keep the pool at exactly
20 (so `--train-n 20` `balanced_split` returns the whole pool), the redundant micro-move
gravity `noop` 52→53 was dropped (ep8 trimmed to `[51,52]`). Action balance is unchanged
(12 click / 8 noop).
