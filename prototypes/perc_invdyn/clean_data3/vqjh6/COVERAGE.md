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
