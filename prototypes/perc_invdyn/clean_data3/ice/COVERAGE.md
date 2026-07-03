# clean_data3 / ice — coverage analysis

Config: whitelist = `left,right,up,down,noop,click`; **keep_action_params = FALSE**
(movement game — `click 8 8` / `click 3 3` collapse to the verb `click`; the click
LOCATION is NOT the label, consistent with the global, location-independent click effect).

Rendering note: in the `[[...]]` grids the sun is the 2x2 block top-left (`gold`=day,
`gray`=night); the cloud is a 1x3 bar on row 0 (`gray`); water is `blue` (liquid) /
`lightblue` (solid/ice). `black` = background and is pure noise in the diff tags
(it is the complement of every other color), so it is ignored in the analysis below.

## CORE dynamics (from dynamics.txt)

1. **left** moves the cloud one cell left, if in bounds (else clamp / no change).
2. **right** moves the cloud one cell right, if in bounds (else clamp / no change).
3. **down** spawns one new water drop just below the cloud; its `liquid` flag = sun's
   `day` value → `day`(gold sun) gives a **blue** drop, `night`(gray sun) gives a
   **lightblue** (solid) drop.
4. **click** (with coord) toggles global day/night: flips the sun gold↔gray AND flips
   the `liquid` flag of EVERY existing drop (blue↔lightblue), all at once. Location
   irrelevant. (A bare `click` with no coord is rejected → no-op; not present in data.)
5. **up** — no handler → no-op (cloud unchanged).
6. **noop** — no-op action; passive per-tick dynamics still run.
7. **passive falling (per tick, on any step):** solid (lightblue) drops fall straight
   down 1 cell if the cell below is free; liquid (blue) drops fall, and when blocked
   step one cell/tick toward the nearest hole (sideways settling along the floor).
8. **no-op / nothing-happens:** with no water (or all water settled), a step leaves the
   board unchanged.
9. **win/termination/reward:** NONE — no goal, no score, no terminal state. Nothing to
   cover (no target dynamic).

## Dynamic → objective coverage + gap in the ORIGINAL train pool

| # | Dynamic | ID target? | FD target? | Gap in original balanced-20 |
|---|---------|-----------|-----------|------------------------------|
| 1 | left moves cloud −1 col | YES (cloud gray shifts −1) | YES (cloud position changes) | OK, present (43→44,47→48,55→56) |
| 1b| left clamped at edge = no move | weak (aliased w/ noop/up) | YES (cloud does NOT move while water still falls) | 67→68 exists but 1/81 — a balanced-20 sample would likely drop this near-miss |
| 2 | right moves cloud +1 col | YES (gray shifts +1) | YES | OK (19→20,27→28) |
| 2b| right clamped negative | — | — | **Not in data** (no clamped-right step in the trajectory); left-clamp carries the boundary contrast |
| 3a| down spawns blue drop (day) | YES (new cell at row 1) | YES (cell count +1) | OK, many (7→8,11→12,…) |
| 3b| down spawns lightblue drop (night) | YES | YES (color of spawn depends on day flag) | **only 1 night-down (35→36)** — a balanced sample mixing it with 5 day-downs could fail to score the day-flag→spawn-color rule contrastively |
| 4a| click day→night (gold→gray, all blue→lightblue) | YES (mass recolor) | YES | only 1 (31→32) |
| 4b| click night→day (reverse) | YES | YES | only 1 (39→40); both directions needed so FD can't learn one-way recolor |
| 5 | up = no-op | weak (aliased w/ noop) | YES-as-negative (cloud unchanged) | present but rare; **inherently unidentifiable under ID (see gap)** |
| 6 | noop = no-op | weak | — | dominant action (≈45/81) |
| 7a| passive liquid fall on noop | weak (passive) | YES (drop moves down ~1 row) | abundant as noop targets |
| 7b| passive solid fall on noop | weak (passive) | YES (lightblue moves down) | present only after the night-click window |
| 8 | nothing-happens (settled / no water) | — | YES-as-negative (NO_CHANGE) | present (0→1, 68→69) but a balanced sample of mostly-falling noops underweights it |

### Documented gaps (nrdf6-style)

- **Unidentifiable passive no-op (ID alias):** `noop`, `up`, and `left/right-when-clamped`
  all produce the SAME observable change — the cloud stays put while passive water
  settling continues. Under ID these three are genuinely confusable; the falling water
  is passive (not caused by the action). This dynamic is therefore carried mainly under
  **FD** (predict the fall) plus the **NO_CHANGE near-miss negatives** so a rule can't
  claim "noop always moves water" or "noop never changes". Kept and balanced, but flagged
  as not ID-separable — same class as nrdf6's unidentifiable passive noop.
- **Step-counter shortcut risk:** while water is airborne it moves every tick, so a
  `step % k` clock could "explain" motion. Defeated by including noop/up targets where
  NOTHING moves (no water present / water settled) at otherwise-identical surface cues.
- **Right-clamp negative absent:** the original trajectory never issues `right` at the
  right edge, so only the left-clamp near-miss is available for the boundary rule.
- **No reward/termination dynamic** to cover (the program has none).

## Curated pool — 15 slices → 18 scored targets

Each slice = consecutive ORIGINAL step numbers copied verbatim; every internal pair is a
scored target. Pool size 18 ≤ train-n 20 ⇒ balanced_split keeps ALL of them.

| episode | orig steps | target pair(s) → dynamic covered |
|---|---|---|
| 0 | 7,8,9   | 7→8 **down**(day→blue spawn) [3a, ID+FD]; 8→9 **noop**(liquid drop falls 1 row) [7a, FD] |
| 1 | 11,12,13| 11→12 **down**(day→blue) [3a]; 12→13 **noop**(liquid fall) [7a] |
| 2 | 51,52   | 51→52 **down**(day→blue, clean) [3a] |
| 3 | 35,36   | 35→36 **down**(night→**lightblue** solid spawn) [3b] — CONTRAST vs day-down spawn color |
| 4 | 19,20   | 19→20 **right** (cloud +1 col) [2, ID+FD] |
| 5 | 27,28   | 27→28 **right** (cloud +1 col) [2] |
| 6 | 43,44   | 43→44 **left** (cloud −1 col) [1, ID+FD] |
| 7 | 47,48   | 47→48 **left** (cloud −1 col) [1] |
| 8 | 67,68   | 67→68 **left CLAMPED** at left edge → cloud does NOT move (water still settles) [1b] — NEAR-MISS NEGATIVE for left/movement |
| 9 | 31,32,33| 31→32 **click** day→night (sun gold→gray, all blue→lightblue) [4a, ID+FD]; 32→33 **noop**(solid drop falls straight) [7b, FD] |
| 10| 39,40   | 39→40 **click** night→day (sun gray→gold, all lightblue→blue) [4b] — REVERSE direction so FD can't learn one-way recolor |
| 11| 3,4     | 3→4 **up** = no-op, NO_CHANGE (no water) [5] |
| 12| 75,76   | 75→76 **up** = cloud unchanged while water passively settles [5] — up behaves as noop |
| 13| 0,1     | 0→1 **noop** NO_CHANGE (no water) [8] — NEGATIVE vs falling |
| 14| 68,69   | 68→69 **noop** NO_CHANGE (water settled at floor) [8] — NEGATIVE vs falling |

### Pool composition (verify_pool, whitelist left,right,up,down,noop,click)

- **18 scored target transitions.**
- by action (click collapsed to verb): **down 4, noop 5, left 3, right 2, click 2, up 2**.
- Contrastive pairs built in:
  - **down spawn color:** 3 day→blue vs 1 night→lightblue (tests day-flag→spawn-color).
  - **click:** day→night vs night→day (both directions).
  - **movement vs clamp:** left/right that move the cloud vs left-at-edge that does NOT.
  - **passive motion vs stillness:** noop/up with falling water (FD must predict the fall)
    vs noop/up with NO_CHANGE (defeats "always moves" and step-counter shortcuts).
