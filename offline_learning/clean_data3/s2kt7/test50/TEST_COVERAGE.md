# TEST50 coverage — s2kt7 (Ant foraging, 16x16)

Whitelist: `noop,click` · keep_action_params **TRUE** (the full `click ROW COL` string is the ID label).
Large held-out test pool: **50 scored target transitions** (`verify_pool(..., context_k=9)` → 50).
Freshly driven with `autumn_drive.py S2KT7` (seed 0) across **4 new drives** (T1–T4); all click
coords and slice windows are disjoint from `train/` (train used `8 8 / 3 10 / 12 2`).

By action: **39 noop + 11 distinct click labels ×1 each**
(`0 15, 3 7, 14 6, 12 9, 15 0, 0 1, 11 4, 8 13, 2 2, 5 5, 9 2` — spread over corners/edges/interior).

(verb, observed change) histogram of the pool:

| verb | change (classify) | dynamic | n |
|---|---|---|---|
| click | `red+1` (spawn) | D1 | 6 |
| click | `gray~move` (no red change) | D2 (near-miss) | 5 |
| noop | `gray~move` | D3 | 21 |
| noop | `gray-1` (arrival) | D4 | 4 |
| noop | `gray+1 red-1` (despawn) | D4 | 5 |
| noop | `NO_CHANGE` (idle) | D5 | 9 |

## 1. Core dynamics (from dynamics.txt)

- **D1 — Click spawns food at the FIXED cell (0,0)**, regardless of clicked ROW/COL (seed-0
  `randomPositions` resolves to (0,0)). Only the *fact* of a click matters; coords are irrelevant.
- **D2 — Click while a food already sits at (0,0) ⇒ no visible spawn** (red count unchanged); the
  redundant click is a no-op. Frame-identical to a `noop` on that tick.
- **D3 — Ant movement (passive, every tick a food exists):** each ant steps ONE cell toward its
  nearest food. Empirically the step is **column-first, then row** (Manhattan): decrement COL to 0,
  then decrement ROW to 0. Both ants target the single food at (0,0). All motion is up/left only.
- **D4 — Eating / food removal (passive):** an ant reaching (0,0) hides under the food on the
  ARRIVAL tick (`gray 2→1`, red stays), and the food is removed on the FOLLOWING tick
  (`red 1→0`, ant reappears `gray 1→2`); the other ant loses its target and idles.
- **D5 — No food ⇒ ants idle** (stay put). Two distinct idle configs: START `(5,5),(14,1)` and
  POST-EAT/PARKED `(0,0),(5,0)`.
- **D6 — `left/right/up/down`/`noop` have no handler** (no-ops). Movement keys aren't whitelisted,
  so they can never be scored targets; only `noop`/`click` transitions exist in the pool.

## 2. Dynamic → target coverage (positives AND negatives, under FD and ID)

`ep_i s→s` = a scored pair in `episode_i/trajectory.csv`. Every internal consecutive row pair is a
target; all slice rows are whitelisted (noop/click), so no window is truncated.

| Dynamic | FD-informative targets (positives) | ID role | Contrastive negatives |
|---|---|---|---|
| **D1 click→food@(0,0)** | 6: clean spawns ep0 `0 15`, ep2 `3 7`, ep4 `14 6`, ep6 `12 9` (`red+1`, ants stay); spawn-onto-parked-ant ep1 `15 0`, ep5 `0 1` (`gray-1 red+1`) | click *verb* recoverable (red appears); **exact coords NOT recoverable** (§4) | D2 clicks (below) + all 9 idle noops (no red ever) ⇒ click≠noop only when red appears |
| **D2 click-on-occupied = no-op** | 5: ep2 `11 4`,`8 13`,`2 2`; ep4 `5 5`,`9 2` (click, `gray~move`, red stays 1) | **ID-aliased with noop** (identical frame to a move) — inherent, documented | vs D1 spawns: same verb, `red+1` only when (0,0) was empty ⇒ kills "click always spawns food" |
| **D3 ant movement** | 21 noop `gray~move` across ep0 (full 9-step convergence), ep2, ep3, ep4, ep5, ep7 (varied ant positions + windows) | noop (recoverable as "no red appeared") | vs D5 idle: same verb `noop`, opposite outcome, split by food-presence covariate ⇒ "ants move ⇔ a food exists", not a step-clock |
| **D4 eating/removal** | arrival ×4 (ep1,ep3,ep5,ep7 `gray-1`) + despawn ×5 (ep1×2, ep3, ep5×2 `gray+1 red-1`) | noop (passive) | despawn's only change is `red→0` under ant overlap ⇒ FD must encode ant-overlap removal, not decay |
| **D5 no-food idle** | 9 noop `NO_CHANGE`: START config ep2 (×2), ep6 (×3); PARKED config ep1 (×3), ep5 (×1) | noop | these ARE the D3 negatives; both idle configs present so it isn't a single memorized frame |
| D6 no-op keys | — (movement keys unwhitelisted; never targets) | n/a | n/a |

Every core dynamic is a scored TARGET **≥4×** (D1=6, D2=5, D3=21, D4=9, D5=9).

## 3. Contrastive structure (what defeats each shortcut)

- **"click always spawns food":** D1 spawns (`red+1`) vs D2 occupied-clicks (`red` unchanged) —
  same verb, effect conditional on (0,0) being empty.
- **"food appears where you click":** 11 different click labels, spread across the grid, ALL land at
  (0,0). Anything predicting food at the clicked cell fails FD on every click.
- **"ants move on a step-clock":** movement and idle share the `noop` verb and co-vary ONLY with
  food presence (21 move noops all have red in-window; 9 idle noops have no red). No `step % k`
  window fits — the drives spawn at different steps (0 / 2 / 3) and the two idle configs recur at
  non-periodic steps.
- **"noop never changes the frame":** 21 move noops + 4 arrivals + 5 despawns are all `noop`s that
  DO change the frame, against 9 `NO_CHANGE` noops.

## 4. How this test differs from `train/`, and what is uncoverable (and why)

**Distinct from train.** All 11 click coords are new (train: `8 8/3 10/12 2`); slice windows,
episode composition, and emphasis differ: TEST adds the full 9-step convergence in one window (ep0),
**4 separate eat/arrival cycles**, both re-spawn-onto-parked-ant clicks, both idle configs, and D2
clicks at **5 different path positions**. Cross-trajectory generalization within the same rules.

**Unavoidable limits — this game is fully deterministic under seed 0 with a fixed start
`(5,5),(14,1)` and food ALWAYS at (0,0):**

1. **Exact click LOCATION is inherently ID-unidentifiable.** Food materializes at (0,0) no matter
   the clicked ROW/COL (verified: 1st/2nd/3rd fresh spawns all land at (0,0); RNG does not advance
   to a new cell). Under keep_action_params=TRUE the label is `click R C`, but `X_t→X_t+1` carries
   zero information about R,C. An oracle can recover the click *verb* (red appears) but never the
   coords — so per-item exact-match ID on clicks is capped near chance. This is the game, not the
   data; the pool tests the *real* rule (click⇔spawn, location-irrelevant) contrastively instead.
2. **D2 clicks are ID-aliased with noop.** A click on an already-occupied (0,0) produces exactly a
   move frame. Kept anyway (5×) because D2 is a core dynamic and is FD-informative (predict "no new
   food"); its ID score is expected to look like noop. Minimized to 10% of the pool.
3. **No directional / positional diversity.** All motion is up/left toward (0,0); food never appears
   elsewhere; there is only ONE deterministic convergence line. "Varied situations" therefore means
   varied *ant positions along that line*, varied *windows*, varied *click coords/timing*, and
   varied *idle configs* — not different geometries. The underlying state sequence is necessarily
   shared with train (a deterministic single-trajectory game admits no truly novel trajectory).
4. **ant2 never reaches (0,0) / never eats.** Once ant1 parks at (0,0) it instantly eats every
   re-spawn before ant2's move-step, so ant2 stalls at (5,0) forever. Every eat event is ant1's;
   the pool covers arrival+despawn thoroughly but cannot show a second, differently-positioned eater.
5. **Small distinct-state space (~15 frame-pairs).** To reach 50 targets the pool re-uses center
   changes across different episodes/windows/labels (genuinely different decoder inputs via the
   context window and click label), which is the only way to build a large pool for this game.

## 5. Verification

`verify_pool('prototypes/perc_invdyn/clean_data3/s2kt7/test50','noop,click',context_k=9)` → 50
targets; by action = 39 noop + 11 distinct `click R C` (×1). Windows intact, shrinking only at
episode boundaries (no cross-episode bleed). Filmstrip: `test50/viz.html`.

| episode | drive[steps] | targets |
|---|---|---|
| ep0 | T1[0..10]  | D1-clean `0 15` + full 9-step convergence [D3] |
| ep1 | T1[10..17] | arrival + despawn + 3 idle + D1-onto-ant `15 0` + despawn [D4/D5/D1] |
| ep2 | T2[0..9]   | 2 start-idle + D1-clean `3 7` + D3/D2 interleave (`11 4`,`8 13`,`2 2`) |
| ep3 | T2[9..14]  | 3 moves + arrival + despawn [D3/D4] |
| ep4 | T3[0..6]   | D1-clean `14 6` + D3 + D2 (`5 5`,`9 2`) |
| ep5 | T3[9..15]  | move + arrival + despawn + idle + D1-onto-ant `0 1` + despawn |
| ep6 | T4[0..4]   | 3 start-idle + D1-clean `12 9` [D5/D1] |
| ep7 | T4[11..14] | 2 moves + arrival [D3/D4] |
