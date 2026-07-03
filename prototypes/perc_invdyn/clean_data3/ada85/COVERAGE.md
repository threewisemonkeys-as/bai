# ada85 — clean_data3 coverage

Whitelist: `noop,click`  •  `keep_action_params=TRUE` (click LOCATION is part of the
action label, e.g. `click 0 3`). Action pool in the curated pool: `noop`, `click 0 0`
(Suzie), `click 10 0` (Billy), `click 5 10` (bottle middle / BottleSpot).

Grid 11x11, coords are (row, col). Objects: Suzie=blue button at (0,0); Billy=red button
at (10,0); Bottle = 5 vertical cells at col 10 rows 3..7; BottleSpot = (5,10) (hidden
inside bottle). Rocks = gray cells.

## 1. Core dynamics (extracted from dynamics.txt)

- **D1 — Click Suzie spawns a rock.** `click 0 0` spawns a rock at (0,0). It is HIDDEN on
  the spawn frame; it first becomes visible one step later at (0,1). (delayed effect)
- **D2 — Click Billy spawns a rock.** `click 10 0` spawns a rock at (10,0); visible one
  step later at (10,1). (delayed effect; location differs from D1)
- **D3 — Rock horizontal movement.** Every step each rock advances col 0->10 along its
  spawn row (one column/step). Passive (runs on any action incl. noop).
- **D4 — Rock vertical movement.** After reaching col 10 the rock advances toward row 5:
  a Suzie/row-0 rock moves DOWN (row+1/step), a Billy/row-10 rock moves UP (row-1/step).
  Passive. Direction depends on spawn row.
- **D5 — Rock hidden behind bottle.** When a rock enters col 10 rows 3..7 it is drawn
  behind the bottle and vanishes from view (last visible just outside, e.g. (2,10) or
  (8,10)). Passive.
- **D6 — Bottle breaks.** When a `breaksBottle=true` rock reaches BottleSpot (5,10) the
  bottle becomes broken starting the next step: palette flips and a GOLD cell appears at
  (5,10). Under seed 0 the first rock has breaksBottle=true. (passive, delayed, on noop)
- **D7 — Bottle stays broken.** Once broken it persists broken every subsequent step with
  no action; never auto-repairs (NO_CHANGE while broken).
- **D8 — Click bottle while broken => momentary repair.** `click 5 10` (any bottle cell)
  while broken shows the intact palette for exactly THAT one frame (gold disappears).
- **D9 — Repair reverts next step.** The step after a click-repair the bottle reverts to
  broken (gold reappears) on its own. (passive)
- **D10 — Idle / noop does nothing.** With no rocks in flight and an intact, unchanged
  bottle, noop produces NO_CHANGE. (the baseline negative for the spawn clicks)
- (No win/termination/reward — open-ended sandbox; all rewards 0.)

## 2. Dynamic x {ID, FD} target coverage + the gap in the ORIGINAL train pool

ID uses the WHOLE window (predict_action_from_window: the masked action is identified
from the states+actions before AND after the gap). FD predicts X_t+1 from history up to t.

| Dynamic | Tested as TARGET under ID | Tested under FD | ORIGINAL-pool gap |
|---|---|---|---|
| D1 Suzie spawn | YES — `click 0 0` (3->4) NO_CHANGE pair; window carries the rock appearing at (0,1) next step => spawn-at-Suzie recoverable | YES — noop 4->5 (rock appears at (0,1)) | Only 1 spawn each in 56 rows; a balanced-20 sample drops most noops, and the click pair alone (NO_CHANGE) is ID-degenerate without its aftermath in the SAME episode window. Spurious "noop never adds objects" / step-clock shortcuts unpunished. |
| D2 Billy spawn | YES — `click 10 0` (23->24); window carries rock at (10,1) => recoverable; LOCATION-contrasts D1 | YES — noop 24->25 (rock at (10,1)) | Same as D1; plus original never juxtaposes the two spawn locations as targets, so `keep_action_params` (row-0 vs row-10 click) is untested. |
| D3 horizontal move | YES — noop 13->14, 25->26 (rock +1 col, no new object => noop) | YES — same pairs (predict rock advanced 1 col) | Present in original but only as a long run of near-identical noops that a step-clock could fit. |
| D4 vertical move | YES — noop 14->15,15->16 (DOWN, Suzie) vs 34->35,35->36 (UP, Billy) — direction contrast | YES — same pairs | Originally interior context only; the up-vs-down (spawn-row-dependent) contrast was never two scored targets side by side. |
| D5 hidden behind bottle | YES — noop 16->17, 36->37 (gray vanishes => noop) | YES — same pairs (predict rock disappears) | Context-only in original; rarely a scored target. |
| D6 bottle breaks | YES — noop 19->20 (gold APPEARS); contrasts the click-repair (opposite direction + click) | YES — 19->20; window (13..19) carries the descending+hidden rock so the delayed break is groundable, not spontaneous | THE nrdf6-style gap: in 56 rows the break is one passive noop; a balanced-20 sample likely never scores it, and if it does, a `gold appears => click 5 10` or step-clock shortcut is never contradicted. |
| D7 bottle stays broken | YES — noop 17->18,18->19,47->48 NO_CHANGE while broken | YES — predict no change (persistence) | Present as filler noops; fine in original, but needed here as the negative for D8/D9. |
| D8 click-repair | YES — `click 5 10` (45->46): gold DISAPPEARS, palette->intact => recoverable & caused by click | YES — predict intact palette for one frame | One occurrence; a balanced sample may keep it (only click) but without the break/revert juxtaposition the gold-direction rule is untestable. |
| D9 repair reverts | YES — noop 46->47 (gold REAPPEARS => passive, noop) | YES — predict revert to broken | Context-only; never paired against the click that caused it. |
| D10 idle noop | YES — noop 1->2, 2->3 NO_CHANGE (no rocks) | YES — predict no change | Plenty in original; here it is the deliberate negative for D1/D2. |

**Summary of gaps fixed:** the spawn clicks (D1/D2) and the bottle break (D6) — the
load-bearing, mostly-passive/delayed dynamics — were effectively context-only / sampled
away in the original 56-row pool, and the break was a lone noop a step-clock could fake.
The curated pool forces each as a scored target inside a self-contained window, and adds
the contrastive negatives below so palette-direction / spawn-location / "noop is static"
shortcuts all score worse than the true conditional rules.

## 3. Curated slices (each becomes one episode; internal consecutive pairs = targets)

Rows copied verbatim from the original `train/episode_0`; slice steps are contiguous so
windows are real consecutive frames and never bleed across slices.

**episode_0 = original steps [1,2,3,4,5]** (Suzie spawn + idle baseline)
- 1->2  noop  NO_CHANGE — **D10 idle negative** (no rock, nothing happens)
- 2->3  noop  NO_CHANGE — **D10 idle negative**
- 3->4  `click 0 0` NO_CHANGE — **D1 ID** (spawn click; window's next frame = rock at (0,1))
- 4->5  noop  rock appears at (0,1) — **D1 FD** (delayed spawn becomes visible)

**episode_1 = original steps [13,14,15,16,17,18,19,20]** (Suzie descent -> hidden -> break)
- 13->14 noop  rock (0,9)->(0,10) — **D3 horizontal move** (reaching col 10)
- 14->15 noop  rock (0,10)->(1,10) — **D4 vertical move DOWN** (corner turn)
- 15->16 noop  rock (1,10)->(2,10) — **D4 vertical move DOWN**
- 16->17 noop  rock (2,10) vanishes — **D5 hidden behind bottle**
- 17->18 noop  NO_CHANGE — **D7 persistence / noop negative** (rock hidden, bottle intact)
- 18->19 noop  NO_CHANGE — **D7 / noop negative**
- 19->20 noop  GOLD appears at (5,10) — **D6 bottle breaks** (delayed; descent is in window)

**episode_2 = original steps [23,24,25,26]** (Billy spawn)
- 23->24 `click 10 0` NO_CHANGE — **D2 ID** (spawn click; window's next frames = rock at (10,1),(10,2)). Contrasts D1: click row-10 => rock row-10.
- 24->25 noop  rock appears at (10,1) — **D2 FD** (delayed spawn; row-10 vs D1's row-0)
- 25->26 noop  rock (10,1)->(10,2) — **D3 horizontal move** (Billy row)

**episode_3 = original steps [34,35,36,37]** (Billy ascent -> hidden)
- 34->35 noop  rock (10,10)->(9,10) — **D4 vertical move UP** (contrasts Suzie's DOWN)
- 35->36 noop  rock (9,10)->(8,10) — **D4 vertical move UP**
- 36->37 noop  rock (8,10) vanishes — **D5 hidden behind bottle**

**episode_4 = original steps [45,46,47,48]** (repair / revert / persistence)
- 45->46 `click 5 10` gold DISAPPEARS, palette->intact — **D8 click-repair** (ID+FD)
- 46->47 noop  gold REAPPEARS — **D9 repair reverts** (passive)
- 47->48 noop  NO_CHANGE (broken) — **D7 bottle stays broken**

## 4. Contrastive negatives (so shortcuts score worse)

- **Spawn click vs idle noop (D1/D2 vs D10):** `click 0 0` (3->4) and `click 10 0` (23->24)
  are BOTH NO_CHANGE on their own pair, identical to the idle noops 1->2 / 2->3. The only
  discriminator is the aftermath carried in the window (a rock appears near Suzie/Billy).
  Defeats "click never changes anything => noop" and any step-counter labeling.
- **Click LOCATION (D1 vs D2):** Suzie click => rock at row 0; Billy click => rock at
  row 10. Forces the perception to encode WHERE the rock appears so `click 0 0` and
  `click 10 0` are separable (the point of `keep_action_params`).
- **Bottle break vs click-repair vs revert (D6 vs D8 vs D9):** all three move the bottle
  palette, but: break (19->20, gold APPEARS, **noop**), revert (46->47, gold APPEARS,
  **noop**), repair (45->46, gold DISAPPEARS, **click 5 10**). A lazy "bottle palette
  changed => click 5 10" rule mislabels the break and the revert; a lazy "noop never
  touches the bottle" rule fails on break and revert. The true rule must use DIRECTION
  (gold disappears = click-caused intact; gold appears = passive break/revert).
- **Vertical direction (D4):** same action (noop), opposite displacement — DOWN for a
  Suzie/row-0 rock, UP for a Billy/row-10 rock — so direction must be tied to the rock's
  row, not to a global clock.
- **Persistence (D7):** NO_CHANGE-while-broken noops (17->18, 18->19, 47->48) are the
  negatives proving the broken state holds without any action (no auto-repair).
