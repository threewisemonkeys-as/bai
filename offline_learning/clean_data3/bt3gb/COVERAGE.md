# bt3gb — coverage analysis (clean_data3)

Game: **bt3gb** — weather / day-night water sandbox (16x16). No goal/score/termination.
Config: whitelist = `left,right,up,down,noop,click`; **keep_action_params = FALSE**
(movement game — `click ROW COL` collapses to the verb `click`; the click LOCATION is
irrelevant to the dynamic, it is a GLOBAL toggle, so location is NOT the label).

## 1. Core dynamics (extracted from dynamics.txt)

| # | Dynamic | Rule |
|---|---------|------|
| D1 | **left** moves Cloud (3-cell gray bar) one cell left, bounds-clamped at cols 0,1,2 |
| D2 | **right** moves Cloud one cell right, bounds-clamped at cols 13,14,15 |
| D3 | **down** spawns a Water droplet below the cloud origin; the droplet's color = current day state (blue if day, lightblue if night) |
| D4 | **up** has NO handler — it is a no-op (only passive water update fires) |
| D5 | **noop** — no player effect (only the passive water update fires) |
| D6 | **click** GLOBAL toggle: flips CelestialBody day (gold<->gray) AND flips `liquid` of EVERY droplet (blue<->lightblue) at once; location ignored |
| D7 | **passive/clock**: every tick each Water droplet falls one cell — liquid (blue) slides sideways along the floor (spread), solid (lightblue) stacks vertically; settled/empty water => no motion |

## 2. Target-coverage of the ORIGINAL train pool (the gaps)

Original `clean_data2/bt3gb/train` = 64 rows / 63 transitions. Action distribution:
~45 noop, 7 down, 4 left, 1 right, 1 up, 2 click. A balanced-by-action sample of 20
would take all of {up=1, right=1, click=2} but subsample the 7 downs / 4 lefts / many
noops at random.

| Dynamic | TARGET under ID? | TARGET under FD? | In a balanced-20 of the original? | GAP |
|---|---|---|---|---|
| D1 left-move | yes (cloud displaces -1 col) | yes (gray bar shifts) | likely (≥1 of 4 lefts) | ok |
| D1 left-**clamp** (near-miss) | n/a (no change) | yes (action fires, NO move => defeats "left always moves") | only 1 of 4 lefts (31->32) — easily dropped | **GAP: clamp negative may be absent** |
| D2 right-move | yes | yes | yes (only right, kept) | ok |
| D3 down-**day** (blue spawn) | yes (new cell at cloud-origin col) | yes (new blue cell) | yes (6 of 7 downs are day) | ok |
| D3 down-**night** (lightblue spawn) | yes | yes (predicting LIGHTBLUE requires day=false) | only 1 of 7 downs (51->52) — random subsample may drop it | **GAP: day/night contrastive for down may be lost** |
| D4 up no-op | up indistinguishable from noop (same passive-only effect) | only passive water | kept (only up) | inherent: **up unidentifiable from noop under ID** (kept as context) |
| D6 click day->night | yes (mass recolor recovers click) | yes (gold->gray, blue->lightblue) | yes (kept) | ok |
| D6 click night->day | yes | yes (lightblue->blue, gray->gold) | yes (kept) | ok — but pairing both directions matters |
| D7 passive water fall | NOT recoverable (any action also ticks water => noop not identifiable from motion) | yes (must predict droplet falling) | falling noops dominate | inherent ID gap; FD-only |
| D7 passive **no-motion** (settled/empty) | n/a | yes — defeats "water always falls each tick" / `step%k` clock shortcut (the nrdf6 trap) | the 6 NO_CHANGE noops are clustered at start + steps 32-34; a random noop subsample may pick only falling noops | **GAP: passive contrastive negative not guaranteed** |

Summary of original-pool gaps (nrdf6-style): the *conditional* and *near-miss* variants
(down-at-night, left-at-clamp, settled/empty noop) are each represented by only 1–6 of many
same-verb transitions, so a balanced-20 random draw can easily leave a dynamic testable only
as window context and let a lazy "noop always moves water" / `step%k` clock rule survive.

## 3. Curated slices (verbatim original rows; each internal pair = a scored target)

Each slice is its own `episode_N`; windows are real consecutive frames and never bleed
across slices.

| episode | original steps | target pairs (action @ first step) | dynamic(s) covered |
|---|---|---|---|
| 0 | 2,3,4,5,6 | 2->3 noop (NO_CHANGE, empty); 3->4 **down**(blue spawn, day); 4->5 noop(blue fall); 5->6 noop(blue fall) | D3 down-day (+ID col), D5/D7 passive liquid fall, **D7 no-motion negative** |
| 1 | 18,19,20 | 18->19 noop(fall); 19->20 **left** (cloud -1 col) | D1 left-move, D7 |
| 2 | 30,31,32,33 | 30->31 noop(blue **slides** +0.25 col); 31->32 **left** CLAMP (no gray move); 32->33 noop(NO_CHANGE settled) | **D1 left-clamp negative**, D7 liquid-spread, **D7 no-motion negative** |
| 3 | 38,39,40 | 38->39 noop(fall); 39->40 **right** (cloud +1 col) | D2 right-move, D7 |
| 4 | 43,44,45 | 43->44 **up** (only passive water); 44->45 noop(fall) | **D4 up==noop**, D7 |
| 5 | 47,48,49 | 47->48 **click** day->night (gold->gray, blue->lightblue x5); 48->49 noop(lightblue/**solid** fall) | D6 click d->n, D7 solid fall |
| 6 | 50,51,52 | 50->51 noop(lightblue fall); 51->52 **down**(**lightblue** spawn, night) | **D3 down-night (color contrast vs ep0)**, D7 |
| 7 | 54,55,56,57 | 54->55 noop(lightblue fall); 55->56 **click** night->day (lightblue->blue, gray->gold); 56->57 noop(blue fall) | D6 click n->d (reverse), D7 |

### Contrastive pairs (defeat shortcuts)
- **left move (ep1) vs left clamp (ep2)** — same verb, cloud moves vs does not => "left always shifts gray" scores worse than the bounds-clamp rule.
- **down-day blue (ep0) vs down-night lightblue (ep6)** — spawn color is conditional on day state; a color-blind "down spawns blue" rule scores worse.
- **click d->n (ep5) vs click n->d (ep7)** — both toggle directions present so the rule is the flip, not a fixed target color.
- **up (ep4) ≈ noop** — establishes up is a true no-op (passive-only), the only contrast available since up is ID-unidentifiable from noop.
- **passive fall noops vs NO_CHANGE noops (ep0 2->3, ep2 32->33)** — water falls only when unsettled; defeats the nrdf6 `step%k` / "water always moves each tick" clock shortcut.
- **liquid slide (ep2 30->31, +col) vs solid straight fall (ep5/6 lightblue)** — exercises the liquid-spread vs solid-stack passive distinction.

## 4. Final pool composition (verify_pool)

20 scored target transitions (pool == --train-n 20, so balanced_split keeps ALL of them).

By verb (keep_action_params=FALSE collapses `click R C` -> `click`):
`noop=12, down=2, left=2 (1 move + 1 clamp), right=1, up=1, click=2 (d->n + n->d)`.

All 7 core dynamics are exercised as scored targets under their applicable objective
(D4 up and D7-on-noop are ID-unidentifiable by construction and are covered under FD +
as contrast), with explicit near-miss negatives for left-clamp, down-color, click-direction,
and passive no-motion. The 12 noops are not redundant filler: 9 carry the passive
fall/spread/solid dynamics (FD) and provide the in-episode context windows for the adjacent
action targets, and 2 are the NO_CHANGE clock-shortcut negatives.

---
## REGENERATED trajectory (ice stacking)

The original clean_data2/bt3gb rollout never demonstrated the defining SOLID behavior —
ice droplets piling into a vertical column. Its single short night phase only froze an
already-spread liquid layer (a flat ice row), and the one fresh solid droplet melted before
stacking. So "solid stacks vertically / liquid spreads sideways" was never a scored target.

Fixed by **regenerating the trajectory** with autumn_drive.py driving the real BT3GB.sexp
(saved at `train_regen/episode_0/trajectory.csv`, filmstrip `train_regen/viz.html`). The
scripted rollout shows the full arc in one column:
  DAY liquid spreads (flat row) -> click FREEZE (day->night, all blue->lightblue)
  -> NIGHT solid droplets STACK into a vertical ice column -> click MELT (night->day,
  all lightblue->blue) -> column collapses and spreads as liquid.

The curated train/ is now sliced from train_regen (20 targets). Key distinctions are now
scored: FREEZE (`blue->lightblue,gold->gray`), MELT (`lightblue->blue,gray->gold`),
solid STACK (`lightblue~move(X,0.0)` pure vertical), liquid SPREAD (`blue~move(X,Y)` lateral).
(dynamics.txt unchanged; test/ still the original — regenerate it too for full consistency.)

---
## TRAIN2 (untied-val expansion)

Goal: add a second, independent ~30-transition pool (`train2/`) so the total per-game
corpus supports an untied 30-train/30-val GEPA split, without replicating any situation
already used by `train/`, `train_regen/`, or `test50/`.

### Fresh drive: `train_regen2/`

Regenerated with `autumn_drive.py BT3GB` (seed 0), 85 actions / 86 rows, filmstrip at
`train_regen2/viz.html`. Deliberately themed differently from `train_regen/` (which stays
centered on cols 2-6 with all water in column 4, clicks always at `8 8`, no clamps) and
from `test50`'s three drives (right-half/left-first/mid-air themes; clicks at `0 15`,
`7 0`, `3 12`, `10 3`, `14 1`, `5 8`; water columns 2, 3, 4, 9, 14). The new drive:

1. Traverses the cloud **right to the wall clamp on an EMPTY grid** (no water at all
   yet) — a clamp variant neither `train/` nor `test50` has (theirs always have some
   water on screen during a clamp).
2. Rains **day** water twice at the wall column (col 14), with an `up` fired mid-fall
   (water still drops — the "up-during-motion" contrast).
3. Moves the cloud back **left across the whole grid to the opposite wall clamp**,
   passing over the CelestialBody (occluding it, then revealing it later in step 9) and
   demonstrating clean left moves once the day water has settled.
4. **Clicks FREEZE at `5 1`** while the cloud sits at the left clamp, occluding 2 of the
   CelestialBody's 4 cells — a partial-occlusion click (`gold-2`, not the usual `gold-4`)
   not exercised anywhere else.
5. Rains **night** water twice at the (occluded) left-clamp column (col 1); the second
   droplet's fall is timed to land directly on top of the first, building a clean 2-high
   ice **stack** in a column no other pool uses. (Bonus finding: because the cloud is
   clamped over the moon, the spawn cell coincides with a CelestialBody cell, so the new
   droplet visibly overwrites it — `gray-1 lightblue+1` instead of the usual `black-1`.)
6. **Clicks MELT at `11 11`**, which simultaneously thaws the settled flat day-cluster
   (col 14, unaffected in shape) AND the 2-stack (col 1), which collapses: this is
   captured as two distinct ticks — a **lateral slide** (blocked-from-below droplet
   shifts one column sideways) followed by a **vertical drop into the now-open cell** —
   making the "seek nearest open cell in the row below" rule explicit as two separate
   scored pairs instead of one instantaneous diagonal jump.
7. Ends with a **static `up`** (fully-settled scene, `NO_CHANGE`) and three more `right`
   moves that **reveal** the CelestialBody cells the cloud had hidden in step 3
   (`gold+1` each tick) — the reverse of the occlusion dynamic, not shown elsewhere.

### Curated pool: `train2/` (10 episodes, verified below)

```
uv run python -c "import sys; sys.path.insert(0,'prototypes/perc_invdyn'); \
    import clean_data3_tools as T; T.verify_pool('prototypes/perc_invdyn/clean_data3/bt3gb/train2','left,right,up,down,noop,click', context_k=9)"
### POOL prototypes/perc_invdyn/clean_data3/bt3gb/train2: 32 scored target transitions
    by action: {'right': 7, 'down': 4, 'up': 2, 'noop': 11, 'left': 6, 'click 5 1': 1, 'click 11 11': 1}
```
By verb (collapsing the two click locations, matching keep_action_params=FALSE at GEPA
run time): `right=7, left=6, noop=11, down=4, up=2, click=2` — total **32**.

| ep | train_regen2 steps | scored pairs (action @ step) | dynamic(s) covered |
|---|---|---|---|
| 0 | 9,10,11,12,13 | 9->10 **right** move, 10->11 **right** move (reaches wall), 11->12 **right CLAMP** (NC), 12->13 **right CLAMP** (NC) | D2 move x2 (empty grid), **D2 clamp negative x2** |
| 1 | 13,14,15,16,17 | 13->14 **down** (day blue spawn col14), 14->15 **up** (water still falls — moving-up), 15->16 noop (liquid fall), 16->17 **down** (2nd day blue spawn col14; 1st droplet does NOT advance this tick) | D3 down-day x2, **D4 up (moving variant)**, D7 liquid-fall, bonus: down-tick freezes existing water |
| 2 | 32,33,34 | 32->33 **left** move, 33->34 **left** move (water already settled — clean, no confound) | D1 move x2 (clean) |
| 3 | 39,40,41,42,43 | 39->40 **left** move (occludes 1 gold cell), 40->41 **left** move (occludes 2nd, reaches wall), 41->42 **left CLAMP** (NC), 42->43 **left CLAMP** (NC) | D1 move x2 (occlusion variant), **D1 clamp negative x2** |
| 4 | 43,44,45,46 | 43->44 **click 5 1** FREEZE (blue-2->lightblue, gold-2->gray+2, partial-occlusion), 44->45 noop (NC), 45->46 noop (NC) | D6 click day->night, **D7 no-motion negative x2** |
| 5 | 46,47,48 | 46->47 **down** (night lightblue spawn col1, overwrites a moon cell), 47->48 noop (solid fall) | D3 down-night #1, D7 solid-fall |
| 6 | 60,61,62,63 | 60->61 noop (1st solid reaches floor), 61->62 **down** (2nd night spawn col1; 1st droplet frozen this tick), 62->63 noop (2nd solid falling) | D7 solid-fall, D3 down-night #2, bonus: down-tick freezes existing water (2nd instance) |
| 7 | 73,74,75,76 | 73->74 noop (2nd solid falling), 74->75 noop (2nd solid **lands** atop the 1st -> 2-stack complete), 75->76 noop (**NC**, stack at rest) | D7 solid-fall + **stack landing**, **D7 no-motion negative** |
| 8 | 76,77,78,79 | 76->77 **click 11 11** MELT (both clusters: flat pair unchanged-shape + 2-stack begins collapsing), 77->78 noop (lateral slide, blocked-from-below search), 78->79 noop (vertical drop into the opened cell) | D6 click night->day, **D7 liquid-collapse (2-tick mechanic)** |
| 9 | 80,81,82,83,84 | 80->81 **up** (fully static scene -> **NC**), 81->82 **right** move (reveals 1 occluded gold cell), 82->83 **right** move (reveals 2nd), 83->84 **right** move (clean, past the CelestialBody) | **D4 up (static variant)**, D2 move x3 (reverse-occlusion + clean) |

### Dynamic -> target-pair coverage (train2 only)

| dynamic | positives (>=2 required) | negatives |
|---|---|---|
| D1 left move | 4: ep2 x2 (clean), ep3 x2 (occlusion) | **2 clamp NC** (ep3) |
| D2 right move | 5: ep0 x2 (empty grid), ep9 x3 (2 reveal + 1 clean) | **2 clamp NC** (ep0) |
| D3 down-day (blue) | 2: ep1 (col14, both spawns) | color-contrasted by D3 down-night |
| D3 down-night (lightblue) | 2: ep5, ep6 (col1, both spawns) | color-contrasted by D3 down-day |
| D4 up == passive-only | 1 moving (ep1, water still falls) + 1 static (ep9, NC) | the static one IS the negative (defeats "up sometimes does X") |
| D6 click day->night (freeze) | 1: ep4 (partial-occlusion variant, `gold-2`) | — (only one direction here; melt below is the paired contrast) |
| D6 click night->day (melt) | 1: ep8 (dual-cluster: flat pair + stack collapse) | freeze vs melt = the direction contrast |
| D7 liquid fall/spread | ep1 (straight fall) + ep8 x2 (lateral-then-drop collapse) = 3 | — |
| D7 solid fall/stack | ep5, ep6 x2, ep7 x2 (incl. the landing pair) = 5 | — |
| D7 no-motion (settled) | — (noop class) | **5 negatives**: ep4 x2 (post-freeze), ep7 (post-stack), ep9 (static up) — kills the "water/action always changes something" clock shortcut |

Contrastive/negative total: 2 (right-clamp) + 2 (left-clamp) + 2 (post-freeze NC) + 1
(post-stack NC) + 1 (static-up NC) = **8 / 32 ≈ 25%**, within the 20-30% target.
Action balance (`right=7, left=6, noop=11, down=4, up=2, click=2`) mirrors the shape of
`test50` (noop-heavy but every verb present, none starved to 0).

### How train2 differs from train/train_regen and test50

- **Columns**: day water at col **14** (via a from-scratch right-wall clamp) and night
  water at col **1** (via a from-scratch left-wall clamp) — neither is `train`'s column 4.
  Col 14 for a down-spawn also appears once in `test50` (ep1, drive A) but there it is a
  single spawn with no follow-on stack/freeze; here it's a 2-droplet cluster later frozen
  near-settled and then melted together with an unrelated 2-stack — a different scene.
  Col 1 (night, left-clamp) is new — `test50`'s night-first drive rains at col 2.
- **Click coordinates**: `5 1` and `11 11`, disjoint from `train`'s `8 8` and all six of
  `test50`'s coordinates.
- **New situations present in neither existing pool**: bounds traversal on a fully empty
  grid before any water exists (ep0); a click that fires while the cloud **occludes** part
  of the CelestialBody, so the toggle reads `gold-2` instead of `gold-4` (ep4); a single
  melt tick that acts on **two differently-shaped clusters at once** (a flat pair and a
  2-stack) rather than one homogeneous cluster; the liquid "seek nearest open cell" rule
  resolved as two separate, clean scored ticks (lateral search, then the vertical drop)
  instead of being buried in a longer cascade; the **reverse**-occlusion reveal as the
  cloud moves away (ep9); and `up` captured as exactly one moving instance and one static
  instance, the minimal contrastive pair for the action hardest to identify from `noop`.
- `train/train_regen` has **no clamp transitions at all** (verified via `verify_pool` —
  left=2/right=1, both pure moves) and no partial-occlusion click, so train2 also closes
  a real gap in the training-only corpus, not just a formal requirement.

### Caveats

- D6 click has only 1 scored instance per direction in train2 (not 2) — the freeze/melt
  *direction* pair itself is the contrast (as in `train`/`test50`), and each instance here
  is a richer/novel configuration (partial-occlusion freeze; dual-cluster melt) rather than
  a repeated plain toggle, which is why only one of each was budgeted against the 28-32 cap.
- Two original `train_regen2` steps (13 and 43) each appear as the **last** row of one
  episode and the **first** row of the next (ep0/ep1 share step 13; ep3/ep4 share step
  43) — this is intentional (the shared row's outgoing pair is scored in only one of the
  two slices) and matches the recipe's allowance for independent slice files.
- `test/` and `test50/` are untouched; only `train_regen2/`, `train2/`, and this section
  were added.
