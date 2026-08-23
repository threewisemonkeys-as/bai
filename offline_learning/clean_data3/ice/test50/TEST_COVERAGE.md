# clean_data3 / ice — TEST50 held-out pool coverage

Large held-out test carve for the Autumn game **ice** (`ice.sexp`), built per
`clean_data3_TEST50_METHODOLOGY.md`. Whitelist = `left,right,up,down,noop,click`;
**keep_action_params = FALSE** (every `click R C` collapses to the bare verb `click`;
the click location is location-independent in ice, so the verb is the only ID label).

**Pool = exactly 50 scored target transitions** (`verify_pool(..., context_k=9)`).
Freshly DRIVEN with `autumn_drive.py ice ... --actions "..."` (seed 0) across 6 themed
trajectories, then sliced into 12 episodes (contiguous verbatim rows, one slice per episode).

Rendering legend (ASCII / `[[...]]`): sun = 2x2 top-left block (`gold`=day, `gray`=night);
cloud = 1x3 gray bar on row 0; water = `blue` (liquid) / `lightblue` (solid/ice).
`black` = background (complement noise, ignored in the diff tags below).

## CORE dynamics (from dynamics.txt)

1. **left** moves the cloud one cell left, if in bounds; **1b** clamped at the left edge → no move.
2. **right** moves the cloud one cell right, if in bounds; **2b** clamped at the right edge → no move.
3. **down** spawns a water drop below the cloud; **3a** day (gold sun) → **blue** drop,
   **3b** night (gray sun) → **lightblue** (solid) drop.
4. **click** (with coord) toggles global day/night: flips sun gold↔gray AND flips the
   `liquid` flag of every drop (blue↔lightblue), location-independent. **4a** day→night, **4b** night→day.
5. **up** — no handler → no-op (cloud unchanged; passive water dynamics still run).
6. **noop** — no-op action (passive per-tick dynamics still run).
7. **passive falling (per tick):** **7a** solid (lightblue) drops fall straight down 1 cell
   if the cell below is free, else stay; **7b** liquid (blue) drops fall, and when blocked
   step 1 cell/tick sideways toward the nearest floor hole (water settling / spreading).
8. **nothing-happens:** with no water (or all water settled / cloud clamped) a step leaves the board unchanged.
9. **win/termination/reward:** NONE in the program — nothing to cover.

## Dynamic → target coverage (episode.pair; N = counts)

| # | Dynamic | ID-informative targets (positives) | FD-informative / negatives |
|---|---------|-----------------------------------|----------------------------|
| 1  | left moves cloud −1 col | ep0.2→3, ep0.3→4, ep2.20→21, ep2.21→22, ep3.31→32, ep6.17→18 (w/water) — **6** | cloud centroid shifts −1 col (gray~move(0,−1)) |
| 1b | left CLAMPED at edge (near-miss) | — (aliased, NO_CHANGE) | ep3.32→33, ep3.33→34 — **2 negatives** (left pressed at leftmost → NO_CHANGE) |
| 2  | right moves cloud +1 col | ep0.0→1, ep0.1→2, ep1.15→16, ep1.16→17, ep1.17→18, ep9.10→11 — **6** | cloud centroid shifts +1 col |
| 2b | right CLAMPED at edge (near-miss) | — (aliased, NO_CHANGE) | ep9.11→12, ep9.12→13 — **2 negatives** (right at rightmost → NO_CHANGE). **NEW vs train** (train lacked any right-clamp). |
| 3a | down → blue drop (day) | ep4.3→4, ep4.6→7, ep6.18→19 — **3** | new blue cell at (row1, cloud-col) |
| 3b | down → lightblue drop (night) | ep5.12→13, ep10.13→14, ep10.16→17 — **3** | new lightblue cell — CONTRAST vs 3a: spawn color = current day flag |
| 4a | click day→night (gold→gray, all blue→lightblue) | ep0.4→5, ep5.9→10 (w/2 drops), ep11.3→4, ep11.5→6 — **4** | mass recolor gold−4/gray+4 (+ drops blue→lightblue) |
| 4b | click night→day (gray→gold, all lightblue→blue) | ep0.5→6, ep5.15→16 (w/3 drops), ep11.4→5, ep11.6→7 — **4** | reverse recolor — both directions so FD can't learn one-way |
| 5  | up = no-op | ep0.6→7 (NO_CHANGE, no water), ep4.8→9, ep5.14→15, ep11.7→8 (water falls, cloud still) — **4** | FD: cloud unchanged; passive fall still shown. **ID: aliased with noop** (see gaps) |
| 6  | noop = no-op (passive runs) | 16 noop targets (below) | — |
| 7a | passive SOLID fall (straight, stack) | ep5.10→11, ep5.11→12, ep5.13→14, ep10.14→15, ep10.15→16 — **5** | lightblue moves +1 row; stacks vertically (no sideways) |
| 7b | passive LIQUID fall + SIDEWAYS settle | falls: ep4.4→5, ep4.5→6, ep4.7→8, ep6.19→20; **sideways**: ep7.23→24, ep8.39→40; **fall-into-hole**: ep7.24→25, ep8.40→41 — **8** | blue moves +1 row; when blocked steps sideways to nearest floor hole then drops in (water spreading) |
| 8  | nothing-happens (settled / no water / clamp) | — (aliased) | ep0.7→8 (noop, no water), ep7.25→26, ep8.41→42 (water settled at floor) + the 4 clamp NO_CHANGEs above — **negatives** defeating "noop always moves water" & step-clock shortcuts |

## Action histogram (verb-collapsed, keep_action_params=False)

`right 8` (6 move + 2 clamp) · `left 8` (6 move + 2 clamp) · `down 6` (3 day-blue + 3 night-lightblue) ·
`click 8` (4 day→night + 4 night→day) · `up 4` · `noop 16`. **Total = 50.**

`verify_pool` raw `by action` shows the click coords un-collapsed
(`click 2 3`×2, `click 5 5`×2, `click 8 8`×4) — they all become `click` under keep_action_params=False.

## How TEST50 differs from `train/` (cross-trajectory generalization)

- **Fresh drives, new situations.** 6 new seed-0 rollouts with irregular action timing; none
  replicate the train slices (train = re-sliced original clean_data2 rollout, steps 0–76).
- **New cloud regions / states train never showed:** cloud driven to the **right edge**
  (cols 13–15) and both edges clamped; water spawned in cols 6–8, 10, 14 (train hugged cols 2–5).
- **Deep water configurations absent from train:** drops falling all the way to the **floor**
  and **stacking** (liquid spreads sideways into floor holes: ep7, ep8; solids stack vertically: ep10);
  liquid **sideways-settling** shown twice from independent pile geometries.
- **Right-edge clamp (2b): entirely new** — train's COVERAGE.md flags it as absent; here it is 2 targets.
- **Repeated day/night toggling of one airborne drop** (ep11: 4 alternating clicks) — a state train lacks.
- More balanced/expanded per-verb counts (50 targets vs train's 18).

## Documented gaps / unavoidable aliasing (ice-inherent)

- **up ↔ noop are ID-indistinguishable.** `up` has no handler, so an `up` frame is byte-identical
  to a `noop` frame (either NO_CHANGE, or the same passive water fall). This is the ice analogue of
  nrdf6's "unidentifiable passive noop." Mitigation: `up` is kept minimal (4 targets) so a
  "passive-frame ⇒ noop" reader is right on the vast majority of passive frames; these dynamics are
  carried under **FD** (predict the fall / NO_CHANGE), not ID.
- **NO_CHANGE frames are mutually ID-aliased.** The 8 still-frame targets (up-NC, noop-NC, 2 left-clamps,
  2 right-clamps, 2 settled-water noops) are indistinguishable under ID (identical X_t→X_t+1). They are
  intentional **contrastive negatives** for the boundary/settling rules and score under FD. An oracle
  therefore cannot get 100% ID; ID is genuinely capped by these no-op/clamp equivalences (~11/50 frames
  are up/noop/clamp-ambiguous). This is minimized and is a property of the game, not the carve.
- **Passive fall taints every airborne frame.** While water is in the air it falls on left/right/click/down
  steps too — but those actions still carry their own distinctive signal (cloud shift / mass recolor /
  new top cell), so left/right/down/click remain ID-recoverable; only the pure no-op class collapses.
- **No reward/termination dynamic** exists in the program (#9) — nothing to cover.
