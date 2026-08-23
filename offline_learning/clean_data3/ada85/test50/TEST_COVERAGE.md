# ada85 — TEST50 held-out pool coverage

Large held-out test carve for GEPA inverse/forward-dynamics scoring.
**Whitelist** `noop,click` • **keep_action_params=TRUE** (the full `click ROW COL` string is
the action label). Grid 11x11, coords (row, col).

Verified: `verify_pool(.../ada85/test50, 'noop,click', context_k=9)` → **exactly 50 scored
target transitions** across 14 episodes. Pool size ≤ test-n=50, so `balanced_split` returns
the WHOLE pool (nothing subsampled).

Objects: Suzie = blue button (0,0), Billy = red button (10,0), Bottle = 5 vertical cells
col10 rows3..7 (intact `[W,Y,Y,Y,W]` / broken `[Y,W,gold,W,Y]`, gold at (5,10) = the
unambiguous broken signal), BottleSpot = (5,10) (hidden inside bottle). Rocks = gray cells.

Freshly driven with `autumn_drive.py ADA85 ... --seed 0`; five source drives
(srcA..srcE) sliced into contiguous verbatim episodes. This pool does NOT reuse the train
slices — see "How this differs from train" below.

## 1. Core dynamics (from dynamics.txt)

- **D1 — Click Suzie spawns a rock** (`click 0 0`): rock spawns HIDDEN under Suzie; first
  visible one step later at (0,1). Delayed effect.
- **D2 — Click Billy spawns a rock** (`click 10 0`): rock spawns hidden; visible next step
  at (10,1). Location contrasts D1 (row 0 vs row 10).
- **D3 — Rock horizontal move** (passive/noop): each rock advances col 0→10 along its spawn
  row, one column/step.
- **D4 — Rock vertical move** (passive/noop): on col 10 the rock advances toward row 5 —
  a Suzie/row-0 rock moves DOWN (+1 row), a Billy/row-10 rock moves UP (−1 row).
- **D5 — Rock hidden behind bottle** (passive/noop): entering col10 rows3..7 the rock
  vanishes from view (last visible (2,10) for Suzie, (8,10) for Billy).
- **D6 — Bottle breaks** (passive/noop, delayed): a breaksBottle=true rock reaching (5,10)
  flips the bottle to broken next step — gold appears at (5,10), palette flips. Under seed 0
  the FIRST rock of every episode has breaksBottle=true.
- **D7 — Bottle stays broken** (passive/noop): once broken it persists every step
  (NO_CHANGE); never auto-repairs.
- **D8 — Click-repair** (`click 5 10` while broken): the bottle shows intact for exactly THAT
  one frame (gold disappears, palette → intact).
- **D9 — Repair reverts** (passive/noop): the step after a click-repair the bottle reverts to
  broken on its own (gold reappears).
- **D10 — Idle / no-op** (noop, nothing in flight, bottle unchanged): NO_CHANGE.
- **Negatives (no-effect actions):** click on an empty cell, or on the bottle while intact,
  fire no handler → no effect (see §4).
- (No win / reward — open-ended sandbox; all rewards 0.)

Empirically re-verified every rule by reading the driver's ASCII per step (scratch drives):
spawn→hidden→visible-at-col1 timing, the full 16-step journey to the break, gold-appears at
break, the one-frame repair flash + auto-revert (confirmed repeated), and — critically for
ID — that **clicking a spawner FREEZES in-flight rocks for that step** (they don't advance),
while a click on an empty cell does NOT freeze them.

## 2. Dynamic × {ID, FD} target coverage

ID = `predict_action_from_window`: the masked center action a_t is identified from the WHOLE
K=9 feature window (states+actions before AND after the gap). FD = predict X_t+1 features from
history+action. Episodes listed as `epN` with the internal pair (source Step→Step).

| Dyn | Scored TARGETs (positives) | ID-informative? | FD-informative? | count |
|---|---|---|---|---|
| **D1** Suzie spawn | ep0 (0→1), ep6 (0→1), ep6 (3→4), ep11 (4→5) | YES — window's next frame(s) show a rock appear at (0,1); ep6(3→4) also shows the in-flight rock FREEZE. Row-0 location ⇒ Suzie, separating it from D2. | via the following noop (rock appears at (0,1)) | 4 |
| **D2** Billy spawn | ep3 (0→1), ep8 (0→1), ep8 (3→4), ep12 (8→9) | YES — rock appears at (10,1) (row 10 ⇒ Billy); ep8(3→4) & ep12(8→9) also show a frozen in-flight rock. | rock appears at (10,1) next step | 4 |
| **D3** horizontal move | ep0 (1→2), ep3 (1→2), ep6 (1→2,2→3,4→5), ep8 (1→2,2→3,4→5), ep11 (5→6), ep12 (9→10), ep13 (11→12) | YES — rock advances 1 col, no new object/flip ⇒ noop | YES — predict rock +1 col | 11 |
| **D4** vertical move | DOWN: ep1 (11→12,12→13); UP: ep4 (11→12,12→13) | YES — noop; displacement sign ties to spawn row | YES — predict rock ±1 row | 4 (2 dn, 2 up) |
| **D5** hidden behind bottle | ep1 (13→14), ep4 (13→14), ep7 (16→17), ep9 (16→17) | YES — gray vanishes, no spawn/flip ⇒ noop | YES — predict rock disappears | 4 |
| **D6** bottle breaks | ep1 (16→17), ep4 (16→17), ep7 (17→18), ep9 (17→18) | YES — gold APPEARS ⇒ passive noop (opposite direction to the click-repair); descent is in ctx_prev so the delayed break is grounded, not spontaneous | YES — predict gold appears / palette flip | 4 |
| **D7** stays broken | ep7 (18→19,19→20), ep9 (18→19,19→20) | YES — NO_CHANGE while broken ⇒ noop (persistence) | YES — predict no change | 4 |
| **D8** click-repair | ep2 (17→18,19→20), ep5 (17→18,19→20) | YES — gold DISAPPEARS, palette→intact ⇒ caused by `click 5 10` (only a click removes gold) | YES — predict intact palette for one frame | 4 |
| **D9** repair reverts | ep2 (18→19,20→21), ep5 (18→19,20→21) | YES — gold REAPPEARS on a noop ⇒ passive | YES — predict revert to broken | 4 |
| **D10** idle / no-op | ep10 (0→1), ep1 (14→15,15→16), ep4 (14→15,15→16) | YES — NO_CHANGE, bottle intact ⇒ noop | YES — predict no change | 5 |

Every core dynamic is a scored TARGET **≥ 4 times** in varied situations (different spawn
row, single vs. double rock, Suzie-caused vs. Billy-caused break, different break timing, two
repair episodes).

## 3. Action histogram (the scored pool)

```
noop        36
click 5 10   5   (4 repair-while-broken [D8] + 1 intact-bottle no-op [neg])
click 0 0    4   (Suzie spawn [D1])
click 10 0   4   (Billy spawn [D2])
click 5 5    1   (empty-cell no-op [neg])
```

noop dominates (72%) because ada85's substantive dynamics — every rock move, the break, the
revert, and persistence — are PASSIVE (noop-triggered); the only meaningful actions are the
three clicks. Click LOCATION diversity = 4 distinct labels: the two spawners `(0,0)/(10,0)`,
the bottle `(5,10)`, and an empty cell `(5,5)`. The spawn labels (`click 0 0` / `click 10 0`)
are structurally fixed to the corners — they must repeat, but they occur in varied situations
(empty board, single rock in flight = frozen-rock cue, and as the first vs. second of a
double spawn). `click 5 10` is reused as BOTH the repair (broken) and the intact no-op — a
state-dependent conditional near-miss on the SAME label.

## 4. Contrastive negatives & near-misses (~22% of pool)

- **Bottle-palette DIRECTION (D6/D9 vs D8):** break (ep1/ep4 16→17, ep7/ep9 17→18) and revert
  (ep2/ep5 18→19, 20→21) both make gold APPEAR on a **noop**; repair (ep2/ep5 17→18, 19→20)
  makes gold DISAPPEAR on **`click 5 10`**. A lazy "bottle palette changed ⇒ click 5 10"
  mislabels break+revert; a lazy "noop never touches the bottle" fails on break+revert. The
  true rule must use direction (gold appears = passive; gold disappears = click).
- **`click 5 10` conditional (D8 vs intact no-op):** ep10 (1→2) clicks the bottle while
  INTACT → NO_CHANGE. Same label as the repair, opposite outcome, so an unconditional
  "click 5 10 ⇒ flip" scores worse (correct: only flips when broken).
- **Empty-cell click (ep13 12→13, `click 5 5`):** fires no handler; the in-flight rocks keep
  moving exactly as on a noop. Punishes "any click ⇒ an effect".
- **D6 near-miss — rock present, no break:** ep1/ep4 (14→15, 15→16) are noop NO_CHANGE while
  the rock is hidden IN the bottle region but has not yet reached (5,10) — a shortcut
  "rock enters bottle region ⇒ gold" is contradicted. And ep7/ep9 (18→19, 19→20) are the
  trailing 2nd rock reaching the ALREADY-broken bottle → still NO_CHANGE (no second break).
- **Vertical DIRECTION (D4):** same action (noop), opposite displacement — DOWN for the
  row-0 (Suzie) rock, UP for the row-10 (Billy) rock — so direction must tie to the rock row,
  not a global clock.
- **Varied break timing (defeats step-clocks):** the break lands at pair-offset 16→17 in the
  single-rock episodes (ep1/ep4) but 17→18 in the double-spawn episodes (ep7/ep9, the extra
  click shifts the schedule by one), and each break is in a DIFFERENT episode — no `step % k`
  rule fits across the pool.

## 5. How this differs from train (cross-trajectory generalization)

Train pool (20 targets) = 1 Suzie spawn, 1 Billy spawn, a single **Suzie-caused** break, one
repair, one revert; single rock throughout; never uses `click 5 5`; `click 5 10` only ever
appears on a broken bottle. TEST50 exercises the SAME rules in NEW configurations absent from
train:
- **Billy-caused breaks** (srcB, srcD) — train's Billy rock only ascended/hid, never broke
  the bottle; here Billy breaks it twice.
- **Double-rock spawns** (ep6, ep8): two rocks in flight simultaneously, including the
  freeze-on-second-click cue and a trailing rock arriving at an already-broken bottle — train
  never has >1 rock.
- **Repeated repair/revert** (ep2, ep5: two repair→revert cycles each) vs. one in train.
- **Click negatives** `click 5 5` (empty) and `click 5 10`-while-intact — neither exists in
  train.

Individual single-rock horizontal-move frames necessarily resemble train frames (the
dynamics are deterministic), but the transitions-in-context (multi-rock windows, Billy break,
negatives, varied timing) do not replicate any train slice.

## 6. Observability limits (what could not be fully covered, and why)

The full-sweep flag that ada85 is "partly action-indistinguishable" was chased down to three
concrete sources; the pool is built to minimize and document them:

1. **Spawn clicks are NO_CHANGE on their own frame** (the new rock is hidden under the
   spawner). This is NOT true aliasing here: recovery uses the K=9 window, which carries the
   rock appearing at (0,1)/(10,1) one step later — and, when a rock is already in flight, the
   spawner click FREEZES it (ep6 3→4, ep8 3→4, ep12 8→9), a second independent cue. An oracle
   holding the spawn belief recovers every spawn. All spawn targets were verified to have a
   forward window that includes the appearance frame (`win next ≥ 1`).
2. **Empty-cell / intact-bottle clicks are genuinely ID-aliased with noop.** `click 5 5` and
   `click 5 10`-while-intact fire no handler, so the frame evolves identically to a noop (rocks
   move, or nothing changes). An oracle CANNOT distinguish these from a noop by observation —
   they truly ARE no-ops in the game. Kept to the **bare minimum (2 total, 4% of pool)** as
   documented negatives; their worth is FD-contrastive ("predict no effect") and punishing
   "click ⇒ effect" shortcuts, not ID discrimination.
3. **The exact bottle cell clicked for a repair is unrecoverable** — clicking ANY of the 5
   bottle cells (rows 3..7, col 10) flips the WHOLE bottle identically. To avoid seeding a
   cluster of mutually-aliased labels, all repairs use a SINGLE label `click 5 10` (the gold
   cell, where the change is most localized). The specific row is intentionally not part of
   the test.
4. **breaksBottle=FALSE rock → "reaches spot, no break" is not cleanly observable under seed
   0.** The first rock of every episode is breaksBottle=true and breaks the bottle, which then
   stays broken forever; the flag of any later rock is unobservable (it reaches an
   already-broken bottle and produces NO_CHANGE, invisible in the hidden region). The pool
   represents this only as the trailing-rock persistence near-miss (ep7/ep9 18→19,19→20:
   "a rock reaches the spot but no new break because already broken"), which is the best the
   engine permits without a false-flag first rock.

Aside from the two intentional aliased negatives (#2), an oracle that knows the rules recovers
every ID target.
