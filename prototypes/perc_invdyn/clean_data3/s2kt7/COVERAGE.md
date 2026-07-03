# clean_data3 coverage — s2kt7 (Ant foraging, 16x16)

Whitelist: `noop,click`  ·  keep_action_params: **TRUE** (click LOCATION is part of the label).
Pool: **19 scored targets** (≤ train-n=20 ⇒ balanced_split keeps the whole pool).
By action: `click 3 3`×1, `click 9 12`×1, `click 2 14`×1, `noop`×16.

## 1. Core dynamics (from dynamics.txt)

- **D1 — Click spawns food at the FIXED cell (0,0), regardless of click coords.** Only the
  fact of a click matters; ROW/COL in the label are irrelevant (seed-0 `randomPositions`
  resolves to (0,0)).
- **D2 — Click while food already exists at (0,0) ⇒ no visible change** (still one red cell).
- **D3 — Ant movement (passive, every tick):** each ant steps one cell toward its nearest
  food via the ant→food unit vector. Two ants; both target the global food set.
- **D4 — Eating / food removal (passive):** a food is removed once an ant occupies its cell.
  Render: red shows on the tick the ant arrives (ant hidden ⇒ gray count drops), food
  disappears on the FOLLOWING tick.
- **D5 — No food ⇒ ants idle** (stay put); they remain wherever they ended until a new click.
- **D6 — Movement keys / noop have no handler** (no-ops). `left,right,up,down` are not even
  whitelisted; only `click` affects world state.

## 2. Target coverage under ID & FD, and gaps in the ORIGINAL train pool

| Dynamic | FD-informative target? | ID-informative target? | In a balanced-20 of ORIGINAL? | Gap |
|---|---|---|---|---|
| D1 click→food@(0,0) | yes (red+1 at (0,0)) | partial: click vs noop recoverable (food appears); **exact coords NOT recoverable** (food always @0,0) | 3 clicks exist (3->4, 27->28, 38->39); likely sampled | coords inherently unidentifiable under keep_action_params — see note |
| D2 click-on-existing-food = no change | n/a | n/a | **never occurs** (no two consecutive clicks anywhere in the trajectory) | **GAP: not constructible** from verbatim contiguous slices; documented, omitted |
| D3 ant movement | yes (gray centroid shift) | passive (noop); recoverable as noop iff paired w/ idle negatives | movement is only steps 4-14; in a balanced-20, the ~10 NO_CHANGE idle noops swamp the ~8 movement noops ⇒ **D3 mostly appears as window context, not as scored targets** (the nrdf6 failure mode) | **GAP fixed** by upweighting movement targets + matched idle negatives |
| D4 eating/despawn | yes (gray-1 arrival; red-1 despawn) | passive (noop) | the eat events (13->15, 28->29, 39->40) are a small minority; easily under-sampled | **GAP fixed** by including all eat-arrival/despawn pairs as targets |
| D5 no-food idle | yes (predict NO_CHANGE) | noop | abundant (26 NO_CHANGE noops) | none — but these are exactly what crowds out D3/D4 in a random sample |
| D6 no-op keys | n/a (`left/right/up/down` dropped pre-pool) | noop | noop present | none |

**nrdf6-style risks for s2kt7 (the reason for this curation):**
1. **D3 and D4 are purely PASSIVE** (fire only on the auto `on true` clock, observed under
   `noop`). They are never the *cause* of the action, so ID can only ever label them `noop`;
   they carry their teaching signal entirely through **FD**. In the original pool they are a
   minority of noop targets, dwarfed by idle NO_CHANGE noops → a balanced-20 would relegate
   them to window context (exactly nrdf6). This pool inverts that ratio.
2. **Step-counter shortcut:** in the original, movement is a single contiguous block
   (steps 4-14) and idle is everywhere else, so a "move during steps a..b" clock could fit.
   Defeated here by making movement and idle co-vary with the **food-presence covariate**:
   every movement/eat target has red present in its window; every idle negative has no red.
   The only rule consistent across the pool is *"ants move ⇔ a food exists"*, not a clock.
3. **Click LOCATION is unidentifiable under ID (keep_action_params=TRUE).** Food always
   materializes at (0,0) no matter the clicked ROW/COL, so the exact coords can never be
   recovered from `X_t→X_t+1`. This is inherent to the game, not fixable by data. Instead the
   pool teaches the *real* rule contrastively: three clicks with **different** labels
   (`click 3 3`, `click 9 12`, `click 2 14`) **all** produce red at (0,0) ⇒ location is
   irrelevant, only click-vs-noop is signal.

## 3. Curated slices (verbatim contiguous original rows → one episode each)

No slice contains a non-whitelisted row (steps 18=left, 22=right, 33=up, 45=empty are all
excluded), so no window is truncated by a dropped action.

| episode | original steps | target pairs → dynamics |
|---|---|---|
| 0 | 3,4,5,6 | 3->4 `click 3 3`: food spawns @(0,0) [**D1**]; 4->5,5->6 noop: ants step toward (0,0) [**D3**] |
| 1 | 9,10,11,12 | 9->10,10->11,11->12 noop: vertical descent toward food (unit-vector) [**D3**] |
| 2 | 12,13,14,15 | 12->13 move [D3]; 13->14 noop: ant arrives onto food, gray 2→1 [**D4 arrival**]; 14->15 noop: food despawns red→0, ant2 idles [**D4 despawn + D5**] |
| 3 | 0,1,2 | 0->1,1->2 noop: no food ⇒ no movement [**D5 / contrastive negative for D1+D3**] |
| 4 | 15,16,17 | 15->16,16->17 noop: idle after eating [**D5 negative**] |
| 5 | 23,24,25 | 23->24,24->25 noop: idle [**D5 negative**] |
| 6 | 27,28,29 | 27->28 `click 9 12`: food @(0,0) onto ant already there, gray-1 red+1 [**D1 location-irrelevant + D4**]; 28->29 noop despawn [**D4**] |
| 7 | 38,39,40 | 38->39 `click 2 14`: food @(0,0) onto ant, gray-1 red+1 [**D1 location-irrelevant + D4**]; 39->40 noop despawn [**D4**] |

## 4. Contrastive structure (what defeats each shortcut)

- **D1 "food appears on click":** positives = 3->4, 27->28, 38->39 (red+1). Negatives = every
  idle noop (0->1, 1->2, 15->16, 16->17, 23->24, 24->25) where no food appears → click≠noop.
- **D1 "location irrelevant":** the 3 click labels differ (3,3 / 9,12 / 2,14) yet all yield red
  at (0,0) → the model cannot tie the spawn cell to the clicked coords.
- **D3 "ants move":** positives = 6 movement noops (food present in window). Matched negatives
  = 6 idle noops (no food). Same action (noop), opposite outcome, discriminated only by food
  presence → kills the step-clock explanation.
- **D4 "eat removes food":** arrival pair (13->14, gray-1) + despawn pairs (14->15, 28->29,
  39->40, red-1). The despawns are noops whose ONLY change is red→0, forcing FD to encode
  ant-overlap removal rather than a generic decay.

## 5. Pool verification

`T.verify_pool('prototypes/perc_invdyn/clean_data3/s2kt7/train','noop,click')` →
19 targets: `click 3 3`×1, `click 9 12`×1, `click 2 14`×1, `noop`×16.
noop breakdown: 6 movement (gray~move), 1 arrival (black+1 gray-1), 3 despawn (gray+1 red-1),
6 idle (NO_CHANGE). All windows intact (no dropped-action truncation). Pool 19 ≤ 20 ⇒ used whole.

---
## REGENERATED trajectory (click-on-occupied no-op, D2)

**The gap.** The original `clean_data2/s2kt7` rollout never demonstrated **D2 — clicking
while a food already exists at (0,0) produces no visible change.** It contained no two
consecutive (or otherwise overlapping) clicks, so the state "food already at (0,0), then
click again" never occurred and could not be recovered by any verbatim contiguous slice
(see §2 above: "D2 ... never occurs ... GAP: not constructible"). D2 is the defining
quirk of this game (the click LOCATION is irrelevant and a redundant click is a no-op), so
it must be a SCORED target — which required regenerating the trajectory.

**Fix.** Drove the real `S2KT7.sexp` with `autumn_drive.py` (seed 0) and saved the full
trajectory to `train_regen/episode_0/trajectory.csv` (filmstrip `train_regen/viz.html`).
The scripted rollout deliberately enters the rare state: spawn a food, then click AGAIN on
the now-occupied (0,0) cell (twice, with two different click coords) and show the food count
stay at exactly one red cell. dynamics.txt and test/ are unchanged (test/ is still the
original rollout and still lacks the regenerated double-click state).

**Action sequence** (19 actions → 20 frames; click_ROW_COL is row-major):
```
noop, noop,                 # steps 0-2: D5 idle, no food -> ants stationary at (5,5),(14,1)
click_8_8,                  # 2->3: D1 SPAWN  -> red 0->1 at (0,0); ants do NOT move yet
click_3_10,                 # 3->4: D2 OCCUPIED click (diff coords) -> red stays 1, ants step
noop,                       # 4->5: D3 ant movement toward (0,0)
click_12_2,                 # 5->6: D2 OCCUPIED click #2 (third distinct coords) -> red stays 1
noop x6,                    # 6->12: D3 ants converge column-first toward (0,0)
noop,                       # 12->13: D4 ARRIVAL -> ant1 onto (0,0), hidden under red (gray 2->1)
noop,                       # 13->14: D4 DESPAWN -> red 1->0, ant1 reappears (gray 1->2)
noop x4                     # 14->18: D5 idle after eating -> both ants stationary (no food)
```
Rationale: the two back-to-back/near clicks make D1 (spawn, red+1) and D2 (occupied, red
unchanged) adjacent scored targets sharing the same verb, so the conditional rule
"click spawns food ONLY if (0,0) is empty" is forced. All three click coords differ
(8 8 / 3 10 / 12 2) yet every effect lands on (0,0) -> click LOCATION is irrelevant. The
spawn click is isolated so ants don't move on that tick (clean red+1). Eating is left to
pure noops (no click colliding with the arrival tick) so D4 arrival/despawn read cleanly.

**Curated pool** (sliced from train_regen; `T.verify_pool` → **19 scored targets**, pool ≤
train-n=20 so balanced_split keeps all):

| episode | train_regen steps | target pairs → dynamic |
|---|---|---|
| 0 | 0,1,2,3,4 | 0->1,1->2 noop NO_CHANGE [**D5** idle, no food]; 2->3 `click 8 8` red+1 [**D1** spawn]; 3->4 `click 3 10` red stays 1 [**D2** occupied no-op] |
| 1 | 4,5,6,7 | 4->5 noop gray~move [**D3**]; 5->6 `click 12 2` red stays 1 [**D2** occupied #2]; 6->7 noop gray~move [**D3**] |
| 2 | 7,8,9,10,11,12 | 7->8…11->12 noop gray~move ×5 [**D3** converge] |
| 3 | 11,12,13,14,15 | 11->12 noop move [**D3**]; 12->13 noop gray-1 [**D4** arrival]; 13->14 noop gray+1 red-1 [**D4** despawn]; 14->15 noop NO_CHANGE [**D5**] |
| 4 | 15,16,17,18 | 15->16,16->17,17->18 noop NO_CHANGE ×3 [**D5** idle negatives] |

By action: `noop`×16, `click 8 8`×1 (spawn), `click 3 10`×1 (D2), `click 12 2`×1 (D2). The
**gap dynamic D2 is now a SCORED target** — `click 3 10` (ep0) and `click 12 2` (ep1), both
classified `gray~move` with NO red change, vs the spawning `click 8 8` (`red+1`).

**Contrastive structure**
- **D1 vs D2 (the gap):** `click 8 8` red 0->1 vs `click 3 10` / `click 12 2` red 1->1.
  Same verb, effect conditional on (0,0) being empty — defeats "click always spawns food".
- **Location irrelevance:** three clicks at (8,8)/(3,10)/(12,2) all act on (0,0); food never
  appears at the clicked coords (inherently ID-unidentifiable under keep_action_params=TRUE).
- **D3 move vs D5 idle:** 7 movement noops (food present, gray~move) vs 6 idle noops (no food,
  NO_CHANGE). Same action, opposite outcome ⇒ "ants move ⇔ a food exists", not a step-clock.
- **D4 arrival vs despawn:** gray 2->1 (ant absorbed under food) then gray 1->2 + red-1 (food
  removed on ant overlap); FD must encode ant-overlap removal, not a generic decay.
