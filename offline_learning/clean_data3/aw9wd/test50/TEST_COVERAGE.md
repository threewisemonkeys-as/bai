# aw9wd — TEST50 held-out coverage

Large held-out test pool for **aw9wd**. Whitelist `noop,click`, `keep_action_params=TRUE`
(the full `click ROW COL` string is the ID label). All trajectories are **freshly driven**
from a clean seed-0 reset with `autumn_drive.py AW9WD` (row-major `click_ROW_COL`), then
sliced verbatim — click locations, break order and resulting shell configurations are all
distinct from `train/` (a cross-trajectory generalization test).

Verified: `T.verify_pool('.../clean_data3/aw9wd/test50','noop,click',context_k=9)` →
**51 scored target transitions** (within the 50 ± 2 spec).

## 1. Core dynamics (from dynamics.txt)

- **D1 — click breaks an eggshell.** Clicking an (unbroken) eggshell cell sets
  `broken=true`. Location matters; clicking empty space or a bare feather does nothing.
  The break itself is INVISIBLE (color stays tan) — its visible effect is delayed to the
  following noop (GAP-1, intrinsic; see caveats).
- **D2 — broken eggshells fall like liquid (passive, on noop).** Down one cell if the
  cell below is free; else one lateral step along a FREE path toward the nearest
  reachable hole; if no free path, the shell is STUCK and nothing moves. Other
  eggshells — and FEATHERS (verified empirically, see ep5) — block the liquid.
- **D3 — despawn-on-feather → uncover (passive, on noop).** A broken eggshell on a
  feather cell is removed the next step, revealing the orange/yellow feather at exactly
  the clicked cell. Only route (a) — click directly on a feather-covered shell — is
  physically reachable (see caveats on route (b)).
- **D4 — statics.** Unbroken shells and feathers never move; settled/stuck states give
  noop → NO_CHANGE.

## 2. Pool composition (8 episodes → 51 targets)

Action histogram: **17 click / 34 noop**; 15 distinct click locations
(`13 9`, `15 8`, `12 8`, `11 10`, `13 5`, `13 11`, `9 9`×2, `12 7`, `11 7`, `10 9`,
`14 10`, `13 10`, `14 9`×2, `2 3`, `15 14`). Both repeats are deliberate contrasts:
`click 9 9` in two different world states (ep3 vs ep5), `click 14 9` twice in the same
drive (shell present → uncover; then bare feather → nothing). Zero overlap with train's
click set (`12 6`, `11 11`, `12 9`, `0 0`, `9 7`, `9 8`). The passive dynamics (D2/D3)
can only be scored on `noop` targets, so the pool is deliberately noop-heavy while click
locations spread over rows 2–15 and cols 3–14, both egg regions and empty space.

| ep | drive slice | targets (action → role) |
|---|---|---|
| 0 | uncover tour `[1-8]` | `click 13 9` D1+ ; noop **D3+ uncover Y(13,9)** ; noop NC D4 ; `click 15 8` D1+ ; noop **D3+ uncover O(15,8)** (orange!) ; `click 12 8` D1+ ; noop **D3+ uncover Y(12,8)** |
| 1 | uncover tour `[9-12]` | noop NC D4 (pre-click baseline) ; `click 11 10` D1+ ; noop **D3+ uncover Y(11,10)** |
| 2 | drain tour `[0-6]` | `click 13 5` D1+ ; noop **D2+ fall (13,5)→(14,5)** ; noop **D2+ settle →(15,5)** ; noop NC **D2− (settled shell does not move)** ; `click 13 11` D1+ ; noop **D2+ fall (13,11)→(14,11)** |
| 3 | drain tour `[7-13]` | `click 9 9` D1+ ; noop **D2+ sideways (9,9)→(9,10)** ; noop **D2+ sideways →(9,11)** ; noop **D2+ down →(10,11)** ; noop **D2+ sideways →(10,12)** ; noop **D2+ down →(11,12)** (liquid trek: lateral hole-seeking, not plain gravity) |
| 4 | stuck shell `[0-6]` | `click 12 7` D1+ ; noop **D3+ uncover Y(12,7)** ; noop NC D4 ; `click 11 7` **D1+/D2− STUCK** (feather below + walled laterally) ; noop NC **D2− (broken shell, no free path → no motion)** ; noop NC **D2−** |
| 5 | feather blocks `[0-6]` | `click 10 9` D1+ ; noop **D3+ uncover Y(10,9)** ; noop NC D4 ; `click 9 9` D1+ ; noop **D2+ (9,9)→(9,10) — does NOT enter the bare feather cell (10,9) below: feathers block liquid** ; noop **D2+ →(9,11)** |
| 6 | stacking `[0-7]` | `click 14 10` D1+ ; noop **D2+ fall→(15,10)** ; noop NC D4 ; `click 13 10` D1+ ; noop **D2+ fall→(14,10)** ; noop **D2+ blocked by settled shell → detour (14,10)→(14,11)** ; noop **D2+ settle →(15,11)** |
| 7 | negatives `[0-10]` | **`click 2 3` D1− (empty cell)** ; noop NC ; noop NC ; `click 14 9` D1+ ; noop **D3+ uncover Y(14,9)** ; noop NC ; **`click 14 9` D1− (bare feather — same location, 2nd click does nothing)** ; noop NC ; **`click 15 14` D1− (empty)** ; noop NC |

## 3. Per-dynamic scored-target coverage (positives / negatives)

| Dynamic | Positives (scored ≥4) | Contrastive negatives |
|---|---|---|
| **D1 click-break** | **13** effective break clicks at 12 distinct locations (rows 9–15, cols 5–11); each cause pair is NO_CHANGE (GAP-1) but the delayed effect appears AT the clicked cell inside the slice's ctx window | **4** no-effect clicks: `2 3` (empty), `15 14` (empty), `14 9` 2nd (bare feather), plus `11 7` (breaks but shell is stuck → no visible effect EVER) → defeat "a click always changes the next frame at the clicked cell" |
| **D2 liquid fall** | **14** `tan~move` noop targets: 4 straight falls, 5 lateral hole-seeking steps (ep3's trek + ep5's detour), 2 down-steps mid-trek, 3 settles — incl. blocking by a settled shell (ep6) and by feathers (ep5) | **3** noops with a BROKEN shell present that does not move: ep2 settled shell, ep4 stuck shell ×2 → defeat "noop always moves tan" / "broken ⇒ moving" |
| **D3 despawn-uncover** | **7** uncover targets (`tan-1 yellow+1` ×6, `orange+1 tan-1` ×1) at 7 distinct cells — (13,9),(15,8),(12,8),(11,10),(12,7),(10,9),(14,9) — both feather colors, at varied within-episode offsets and step parities | **6** same-structure `click→noop` sequences where the next noop shows a MOVE not an uncover (ep2 ×2, ep3, ep5, ep6 ×2) → forces the on-feather condition; plus the bare-feather re-click (ep7) |
| **D4 statics / no-op** | **10** settled/static NC noops at varied offsets (0–3 steps after clicks, plus pre-click baselines) | — (they ARE the negatives for D2/D3 timing shortcuts) |

**Timing.** Clicks fall at irregular within-slice offsets (gaps of 1–4 noops; ep7 clicks at
offsets 0,3,6,8) and the visible changes land at step parities {0,1,2,3}, with NC noops at
matching parities — no `step % k` clock explains the pool.

## 4. How TEST50 differs from train

- **Disjoint click locations.** Train's 6 clicks (`12 6`, `11 11`, `12 9`, `0 0`, `9 7`,
  `9 8`) vs TEST50's 15 fresh locations; no reuse.
- **New dynamics variants never targeted in train:** the multi-step lateral liquid trek
  with hole-seeking (ep3), feather-as-obstacle (ep5), eggshell-on-eggshell stacking with
  detour (ep6), the STUCK broken shell (ep4), the orange feather (ep0 — train only ever
  uncovered yellow at target pairs... train ep1 had orange; here a different orange cell
  `15 8`), and the same-cell click contrast (ep7).
- **Richer world states.** Train slices break at most one shell per slice from the intact
  egg; TEST50 episodes chain 2–4 breaks so later targets occur in partially-eroded
  configurations (holes, settled piles) that train never reaches.

## 5. Not covered / caveats

- **GAP-1 (intrinsic): click ID is never recoverable from the scored pair.** Every
  `click R C` pair is NO_CHANGE (the break is invisible; the effect is delayed one step).
  The location is recoverable only from the ctx_next window (uncover/fall appears AT the
  clicked cell), which every slice preserves by keeping cause + delayed effect together.
  The 4 no-effect clicks are additionally aliased with `noop` even through the window
  (~8% of the pool); they are included for FD value and to defeat "click ⇒ change".
- **D3 route (b) — a draining shell landing on a feather cell — is unreachable.**
  Verified empirically (ep5's setup): feather cells BLOCK the liquid (`isFreePos` counts
  feathers), so a broken shell can never enter one; the despawn filter only ever fires on
  shells broken directly on a feather (route a). dynamics.txt's route (b) description is
  not realizable in play; the underlying despawn rule is fully covered via route (a) ×7.
  (Train's COVERAGE reached the same conclusion — its GAP-4.)
- **Broken-vs-unbroken is invisible in `X_t`.** A stuck/settled broken shell renders
  identically to an unbroken one; the NC negatives in ep2/ep4 are only interpretable via
  the in-slice history (the click in ctx_prev). This is the game's intrinsic partial
  observability, not a data artifact.
- **No terminal/reward signal exists in this game** (sandbox; nothing to cover).

viz.html is a filmstrip of the whole pool (all 8 episodes with separators).
