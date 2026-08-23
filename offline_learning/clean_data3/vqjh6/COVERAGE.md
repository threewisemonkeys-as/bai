# vqjh6 — clean_data3 coverage

Game: gravity / blob sandbox. 17x17 grid. Four edge-midpoint Buttons set gravity
(leftButton red @ (8,0), rightButton darkorange @ (8,16), upButton gold @ (0,8),
downButton green @ (16,8)). Blobs = 2x2 blue clusters. Background black.

Config: whitelist = `left,right,up,down,noop,click`; **keep_action_params = FALSE**
(movement game — every click collapses to the verb `click`; the click LOCATION is NOT
the label). `verify_pool` prints raw `click R C` strings, but under the real GEPA run
all 8 clicks are one action class `click`, so the scored balance is noop=12 / click=8.

## Core dynamics extracted from dynamics.txt

- **D1 Passive gravity motion (default down):** every step, all blobs move 1 cell in the
  current gravity direction; default is "down".
- **D2 Gravity=left** via leftButton click; subsequent blobs move left.
- **D3 Gravity=right** via rightButton click.
- **D4 Gravity=up** via upButton click.
- **D5 Gravity=down** via downButton click (also the default).
- **D6 One-tick lag** on a gravity change: on the click tick blobs still move per the OLD
  gravity; the new direction takes effect the next tick.
- **D7 Spawn:** clicking a free (unoccupied, non-button) cell spawns a 2x2 blob at the
  clicked anchor; it appears on the next observation.
- **D8 Occupied-cell click → no effect** (spawn only fires on free cells).
- **D9 Button click → gravity change only, no blob spawn** (and on an empty grid, no
  visible change at all on the click tick due to lag/no blobs).
- **D10 Arrow actions left/right/up/down → no-op** (no movement handler).
- **D11 noop → no-op** (action itself does nothing; blobs still move via gravity).
- **D12 No despawn:** blobs persist; they may move off-grid via edge motion (blue cell
  count falls) but there is no removal rule.
- **D13 No win / termination / reward** (open-ended sandbox).

## Scorability under ID (action recoverable from change) and FD (state visibly changes)

| Dynamic | TARGET under ID? | TARGET under FD? | In original train as a scored target? | Gap |
|---|---|---|---|---|
| D1 passive down | no (noop; passive, indistinguishable from arrows) | YES (blob +1 row) | yes, but ~all noops were "down" | direction monoculture: a "blobs always fall" rule scores FD high |
| D2 left / D3 right / D4 up | no (manifests on a post-click noop) | YES (blob moves new dir) | present but rare vs down; relies on the setting-click being in the window | nrdf6-style: passive direction change, easy to swamp with down |
| D5 down via downButton | no | YES | yes | — |
| D6 one-tick lag | no (click tick looks like a noop-down) | partial (no change vs prior tick) | only as context | the click tick is ID-invisible; useful as a click no-spawn negative |
| D7 spawn (free cell) | YES (blue+4 appears; noop never adds blobs) | partial (knows a blob appears; location stripped, so WHERE is unknown) | yes (4 clean spawns) | — |
| D8 occupied click no-spawn | YES as a click NEGATIVE (click + no blob added) | YES (predict: no new blob) | present (steps 27, 68) | needed as contrast or "click ⇒ +blob" wins |
| D9 button click no-spawn / empty no-effect | YES as a click NEGATIVE | YES (no blob; on empty, NO_CHANGE) | present | same contrast role |
| D10 arrows = no-op | **NO — unidentifiable** (identical to noop and to each other) | NO (no change beyond gravity, same as noop) | n/a | **inherent GAP**: not includable as a discriminating target; context only |
| D11 noop no-op | partial (noop is the passive default) | n/a (empty grid → NO_CHANGE) | yes | — |
| D12 off-grid attrition | no | weak (blob count falls at boundary) | present but messy (green-button overlap) | low value; a side-effect of D1, omitted |
| D13 no termination | n/a | n/a | n/a | shown implicitly (NO_CHANGE noops) |

**Gaps found in the original pool** (would a balanced-20 sample score the rules?):
1. **Direction monoculture (the nrdf6 trap).** Default gravity is down, so the vast
   majority of noop targets move blobs DOWN. A balanced-20 sample is dominated by
   "blob → +1 row"; the FD objective is satisfied by a constant "blobs fall" rule and
   never has to express that gravity is a button-set state with left/up/right variants.
2. **D10 arrow no-ops are unscorable.** left/right/up/down arrows produce exactly the
   same observation transition as noop, so they can neither be identified (ID) nor used to
   test any rule (FD). Including them as targets only injects ID noise → excluded from the
   curated pool; documented here as an inherent gap (context only).
3. **Click is over-positive without negatives.** If only spawning clicks are scored, ID/FD
   learn "click ⇒ a blob appears". The button-gravity clicks (lag, no spawn) and the
   occupied/empty clicks (no spawn) must be scored as click NEGATIVES so the conditional
   spawn rule beats the lazy one.

## Curated slices (verbatim original rows; each slice = one episode)

Each consecutive pair inside a slice is a scored target; windows never bleed across slices.
The button-click that sets a direction is kept INSIDE the same slice so the post-click
noop target has the cause in `ctx_prev` (verified: each direction noop has prev≥1).

| episode | original steps | target pairs (action → change) | dynamics |
|---|---|---|---|
| 0 | 6,7,8,9,10 | (6→7) noop DOWN; (7→8) click leftBtn — blob still DOWN (lag, no spawn); (8→9) noop LEFT; (9→10) noop left | D1, D6, D9-neg, D2 |
| 1 | 11,12,13,14 | (11→12) click upBtn — still LEFT (lag, no spawn); (12→13) noop UP; (13→14) noop up | D6, D9-neg, D4 |
| 2 | 15,16,17,18 | (15→16) click rightBtn — still UP (lag, no spawn); (16→17) noop RIGHT; (17→18) noop right | D6, D9-neg, D3 |
| 3 | 19,20,21 | (19→20) click downBtn — still RIGHT (lag, no spawn); (20→21) noop DOWN | D6, D9-neg, D5 |
| 4 | 3,4,5 | (3→4) click 3 8 SPAWN (blue+4, empty grid); (4→5) noop down | D7 +, D1 |
| 5 | 76,77,78 | (76→77) click 8 8 SPAWN (blue+4, center free cell, other blob present); (77→78) noop down | D7 +, D1 |
| 6 | 26,27,28 | (26→27) noop down; (27→28) click 8 5 on an OCCUPIED blob cell — no spawn | D1, D8-neg |
| 7 | 48,49,50 | (48→49) click 8 0 leftBtn on EMPTY grid — NO_CHANGE; (49→50) noop empty — NO_CHANGE | D9-neg, D11, D13 |

## Final pool (verify_pool, context_k=9)

- **20 scored targets**, pool == train-n=20 → balanced_split keeps ALL of them.
- By action class (keep_action_params=FALSE): **noop = 12, click = 8**.
- Noop FD direction balance: down 5, left 2, up 2, right 2, NO_CHANGE(empty) 1 — all four
  gravity directions appear as targets, defeating a constant "blobs fall down" rule.
- Click contrast: **2 spawn positives** (ep4 click 3 8, ep5 click 8 8 → blue+4) vs **6
  spawn negatives** — 4 button-gravity clicks (lag, no spawn), 1 occupied-cell click
  (no spawn), 1 empty-grid button click (NO_CHANGE). This defeats "click ⇒ +blob".
- Each gravity-change noop target carries its setting button-click in `ctx_prev`
  (prev≥1 confirmed), so FD can recover the gravity state.

### Contrastive negatives summary
- **Direction:** left/up/right noop targets vs the dominant down → "always down" fails.
- **Spawn:** spawning clicks (blue+4) vs button/occupied/empty clicks (no blob) →
  "click always spawns" fails.
- **Lag (D6):** click-on-button tick shows the OLD-direction move, doubling as the click
  no-spawn negative and the evidence that a gravity change is delayed one tick.

### Documented inherent gap
- **D10 (arrow actions = no-op):** intentionally NOT included as a scored target — arrows
  are observationally identical to noop and to one another, so they are unidentifiable (ID)
  and test no rule (FD). Including them would only add ID noise. Context-only dynamic.
- **D12 (off-grid attrition):** omitted; it is a boundary side-effect of D1 and the clean
  examples are confounded by green-button overlap.

## TRAIN2 (untied-val expansion)

**Config note (supersedes the header above for this pool):** the header's "keep_action_params
= FALSE / whitelist left,right,up,down,noop,click" describes how the *original* `train/` was
curated. The actual sweep config in `clean_sweep.py` (`GAMES["vqjh6"]`) is
**whitelist = `noop,click`, keep_action_params = TRUE** — identical to `test50`'s config (see
`test50/TEST_COVERAGE.md`). `train2/` is built for this real config: click LOCATION is the ID
label, arrows are outside the whitelist entirely (so never driven), and every click coordinate
is a distinct scored action-class. (Running `verify_pool` on `train/` under `noop,click` still
returns the same 20 targets — the arrow rows in the original drive were never used as slice
targets — so `train/` and `train2/` are compatible additions under the real config.)

### Source: `train_regen2/` (fresh single continuous drive, seed 0)

A brand-new 35-action drive (`train_regen2/episode_0/trajectory.csv`, 36 rows, filmstrip at
`train_regen2/viz.html`), driven with `autumn_drive.py VQJH6`, entirely distinct from the
original train drive and from test50's 4 drives (different spawn cells, different button-click
order/precedent-gravity combos, different blob counts at each event). Verified clean via the
printed ASCII + blue-cell-count audit: every spawn lands exactly at its anchor formula
(`rows{R-1,R} x cols{C,C+1}`), every button-click lag shows the OLD-direction move on the click
tick, and — after one fix (an early draft let a corner blob clip the right edge on click `0 8`;
re-picked its spawn anchor with more margin) — **no blob ever exits the grid** (total blue-cell
count holds at each spawn plateau, e.g. reaches 20 = 5 spawns x4 and stays there through the end),
so no "vanish/reappear" artifact contaminates any later gravity reversal (the failure mode this
game is prone to: an off-grid blob can drift back into view many ticks later if gravity later
reverses over the same axis — avoided here by design/verification, not by luck).

### Curated `train2/` (6 episodes, verified via `verify_pool`)

```
T.verify_pool('prototypes/perc_invdyn/clean_data3/vqjh6/train2','noop,click',context_k=9)
-> 32 scored target transitions
   by action: {'click 5 10':1, 'noop':17, 'click 9 3':1, 'click 8 0':2, 'click 0 8':2,
               'click 6 9':1, 'click 8 16':2, 'click 5 2':1, 'click 16 8':2,
               'click 13 4':1, 'click 8 11':1, 'click 3 9':1}
```

| episode | source steps | target pairs (action -> change) | dynamics |
|---|---|---|---|
| 0 | 0-7 | `click 5 10` **D7+** (blue+4, empty grid); noop D1-down x2; `click 9 3` **D7+ compound** (2nd spawn while 1st continues down); noop D1-down; `click 8 0` **D2 cause + D6 lag** (still moves DOWN); noop **D2 left reveal** | D7 x2, D1(down)x3, D2, D6 |
| 1 | 8-12 | `click 0 8` **D4 cause + D6 lag #1** (from LEFT, still moves LEFT); noop **D4 up reveal** x2 (2nd one transiently covers leftButton red -> `red-1`, blob-over-button compound, same phenomenon test50 calls out for the orange button); `click 6 9` **D7+** (spawn under UP gravity) | D4, D6, D7, D1(up)x2 |
| 2 | 13-19 | `click 8 16` **D3 cause + D6 lag #1** (from UP, still moves UP); noop **D3 right reveal** x2; `click 5 2` **D8- occupied negative #1** (targets Y's own cell mid-motion, right ctx, no spawn); `click 16 8` **D5 cause + D6 lag #1** (from RIGHT, still moves RIGHT); noop **D5 down reveal** | D3, D6, D1(right)x2, D8-, D5 |
| 3 | 20-26 | `click 13 4` **D7+** (spawn under DOWN gravity, 4th blob); noop D1-down; `click 8 11` **D8- occupied negative #2** (targets X's cell, adjacent-but-not-overlapping with the spawn-3 blob at the same row, down ctx, no spawn); `click 8 0` **D2 cause + D6 lag #2** (from DOWN this time, NEW precedent vs episode-0's D2, 4-blob scene); noop **D2 left reveal** x2 | D7, D1(down), D8-, D2, D6 |
| 4 | 26-30 | `click 3 9` **D7+** (5th spawn, under LEFT gravity, top-of-grid region, clear of all other blobs); `click 8 16` **D3 cause + D6 lag #2** (from LEFT this time, NEW precedent vs episode-2's D3, 5-blob scene); noop **D3 right reveal** x2 | D7, D3, D6, D1(right) |
| 5 | 30-35 | `click 0 8` **D4 cause + D6 lag #2** (from RIGHT this time, NEW precedent vs episode-1's D4); noop **D4 up reveal**; `click 16 8` **D5 cause + D6 lag #2** (from UP this time, NEW precedent vs episode-2's D5); noop **D5 down reveal** x2 | D4, D6, D1(up), D5, D1(down)x2 |

### Coverage tally (every core dynamic scored >= 2x, in varied situations)

| Dynamic | Scored-target count | Varied how |
|---|---|---|
| D1 passive motion | pure-`noop` targets by direction: down x7 (ep0 x3, ep2 x1, ep3 x1, ep5 x2), left x3 (ep0 x1, ep3 x2), up x3 (ep1 x2, ep5 x1), right x4 (ep2 x2, ep4 x2) — sums to 17 = all noops | all 4 directions, irregular click cadence (steps 1,4,6,9,12,14,17,18,21,23,24,27,28,31,33 — gaps of 1-3, never a fixed period) |
| D2 leftButton | 2 (ep0, ep3) | 1st from DOWN in a 2-blob scene, 2nd from DOWN again but a 4-blob scene (repeated cell, different game state, per the task's ask) |
| D3 rightButton | 2 (ep2, ep4) | 1st from UP (3-blob scene), 2nd from LEFT (5-blob scene) -- different precedent gravity |
| D4 upButton | 2 (ep1, ep5) | 1st from LEFT (2-blob scene), 2nd from RIGHT (5-blob scene) -- different precedent gravity |
| D5 downButton | 2 (ep2, ep5) | 1st from RIGHT (3-blob scene), 2nd from UP (5-blob scene) -- different precedent gravity |
| D6 one-tick lag | 8 (every button click above) | each shows the OLD-direction move on the click tick; the reveal noop(s) in the same episode show the NEW direction, so a no-lag rule mis-predicts the click pair specifically |
| D7 spawn (free cell) | 5 (`5 10`, `9 3`, `6 9`, `13 4`, `3 9`) | under 3 different gravities (down x2, up x1, down x1, left x1); rows {3,5,6,9,13} all distinct, cols {3,4,9,10} (9 used twice, by different rows) -- every spawn has `|row-col| >= 3` so a row/col-transposed reading of the click lands far from the true anchor and is punishable |
| D8 occupied-cell click (negative) | 2 (`5 2` on Y, `8 11` on X) | 2 different blobs, 2 different gravities (right, down); both show pure gravity-move with NO `+4`, contrasting directly against the 5 spawns |

### Contrastive negatives

- **Click != spawn:** 10/15 clicks are non-spawning (4 buttons x2 + 2 occupied) vs 5/15
  spawning -> two-thirds of clicks are negatives (10/32 = 31% of ALL scored targets are
  click-negatives, noop included in the denominator, matching the ~20-30%-of-pool guideline)
  -- "click always spawns" fails on the majority of click examples.
- **Direction monoculture defeated:** noop targets split down 7 / left 3 / up 3 / right 4 (17
  total) -- no single direction dominates enough to let a constant-direction FD rule win.
- **Lag defeats "no-lag" rules:** every one of the 8 button-click pairs moves in the OLD
  direction, and the immediately-following reveal noop(s) in the same episode show the NEW
  direction, so scoring both pairs correctly requires the lag rule specifically.
- **Row/col spread defeats transposition:** across all 15 clicks (4 buttons reused x2 + 5
  spawns + 2 occupied), 8 distinct rows {0,3,5,6,8,9,13,16} and 9 distinct columns
  {0,2,3,4,8,9,10,11,16} appear; every non-button click has `|row-col| >= 3`.

### How train2 differs from `train/` and `test50/` (no situation reuse)

- **Click coordinates:** all 7 non-button coordinates in `train/` (`3 8`, `8 8` spawns, `8 5`
  occupied) and all 14 non-button coordinates in `test50` (`3 2`, `12 6`, `2 12`, `10 1`, `6 7`,
  `7 7`, `13 3`, `14 11`, `9 14`, `10 12`, `5 5`, `7 4`, `1 10`, `12 9`) are avoided; `train2`
  uses 7 entirely new non-button cells (`5 10`, `9 3`, `6 9`, `13 4`, `3 9`, `5 2`, `8 11`). Only
  the 4 fixed button cells are necessarily reused (there is only one possible coordinate per
  button) -- each reuse is in a new precedent-gravity / blob-count context per the table above.
- **Button-order / precedent-gravity combos:** `train/`'s single cycle was
  down->left->up->right->down. `train2` visits left->up->right->down->left->right->up->down
  across two full passes with DIFFERENT precedent gravity each time a button is reused (e.g.
  rightButton once from UP, once from LEFT), which `train/`'s single cycle never repeated.
- **Blob-count scenes:** `train/` never exceeds 1 simultaneous blob at a scored target;
  `train2` builds up to 5 simultaneous blobs (spawns 1-5 all persist), giving multi-blob
  compound scenes at every later click/noop -- closer to `test50`'s 2-4-blob scenes but with
  its own distinct configuration.
- **Fresh drive:** `train_regen2` is an independent `autumn_drive.py` invocation (seed 0, its
  own reset) from both `train/`'s original drive and each of `test50`'s 4 drives.

### Caveats

- **Button-overlap compound (ep1's 2nd up-reveal noop):** blob Y transiently occupies the
  leftButton's cell (`red-1` in the diff tag) while also moving up -- the same
  blob-passes-over-a-button phenomenon `test50/TEST_COVERAGE.md` documents for the darkorange
  button. Not a defect; both signals (`blue~move` and `red-1`) are present in the diff, so FD
  must explain both.
- **X/Spawn3 adjacency (ep3's `click 8 11` frame):** at that tick X and Spawn3 sit in
  touching-but-not-overlapping columns (`11-12` and `13-14` on the same row), rendering as one
  contiguous 4-wide blue run. The clicked cell (`8 11`) is unambiguously X's leftmost column,
  so the occupied-negative target is still unambiguous, but a human skimming the raw ASCII
  should not mistake this for one wide blob.
- **`train_regen2` is a single continuous drive, not fully independent mini-drives** (unlike
  test50's 4 separate resets) -- chosen deliberately so blob persistence/compounding across
  scenes gives the multi-blob variety noted above; verified frame-by-frame (see the ASCII +
  blue-count audit above) to contain no accidental edge-exit or reappearance artifacts in any
  curated slice.
- **Pool-size note:** the task brief that requested this expansion assumed an existing pool of
  "~30"; the actual `train/` (under the real `noop,click` / keep=TRUE config) is 20. `train2`
  alone is 32 (within the requested 28-32 range); `train/` + `train2` combined = 52.
