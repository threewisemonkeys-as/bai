# bt3gb — TEST50 held-out test pool

Game: **bt3gb** — weather / day-night water sandbox (16x16, no goal/score).
Config: whitelist = `left,right,up,down,noop,click`; **keep_action_params = FALSE**
(`click R C` collapses to the verb `click`).

**Pool: 52 scored target transitions** (50±2 ✓), verified with
`clean_data3_tools.verify_pool('.../bt3gb/test50', 'left,right,up,down,noop,click', context_k=9)`.

All trajectories were **freshly driven** with `autumn_drive.py BT3GB` (seed 0) — three
themed drives, none replicating the train/train_regen action sequences or object
configurations (see "How this differs from train" below). Slices are verbatim contiguous
rows of the driven trajectory.csv files; each `episode_*` is one slice.

## Action histogram (by verb, as scored)

`noop=20, down=8, left=7, right=6, click=6, up=5` — total 52.
Negatives ≈ 25% of the pool: 8 NO_CHANGE noops + 2 right-clamps + 2 left-clamps + 1 up-NC.

## Source drives

| drive | theme |
|---|---|
| A | day rain on the RIGHT half (cols 6→14), right-clamp at the wall, freeze/melt of 3 settled separated droplets, lefts over settled water |
| B | night-first: click with NO water, night ice rain at col 2, 4-solid vertical stack, melt → cascading liquid collapse, left-occlusion move, left-clamps |
| C | mid-air freeze of 2 falling droplets, ups interleaved with falling water, up on a fully static scene, melt of a 2-stack → sideways slide, day rain at col 4 over settled water |

## Episodes (source rows are original driven Step numbers)

| ep | src steps | scored pairs (action @ step) | dynamics |
|---|---|---|---|
| 0 | A 6-10 | 6 **right**(+blue midfall), 7 **up**(water falls only), 8 **right**, 9 **down**(blue spawn c9) | D2 x2, D4, D3-day |
| 1 | A 29-34 | 29 **right**, 30 **right**(reach origin 14), 31 **right CLAMP**(NC), 32 **right CLAMP**(NC), 33 **down**(blue spawn c14 at the wall) | D2 x2 + **2 clamp negatives**, D3-day |
| 2 | A 50-58 | 50 **click** FREEZE settled (blue-3→lightblue, gold→gray), 51/52 **noop NC**(night, settled solids), 53 **click** MELT, 54/55 **noop NC**(day, settled liquids), 56/57 **left** x2 | D6 both directions, **D7 no-motion negatives x4** (solid-at-rest AND liquid-at-rest), D1 x2 |
| 3 | B 0-6 | 0 **click** day→night with NO water (gold→gray ONLY), 1 **left**(night), 2 **noop NC**(empty night), 3 **left**(cloud merges over moon: gray-1), 4 **down**(lightblue spawn c2), 5 **noop**(solid fall) | D6 no-droplet variant, D1 x2 (incl. occlusion-merge), **D5/D7 empty-scene negative**, D3-night, D7-solid |
| 4 | B 30-34 | 30/31 **noop**(solid falls), 32 **noop**(solid LANDS on stack r14), 33 **down**(night spawn onto 2-stack) | D7-solid fall + **stack landing**, D3-night |
| 5 | B 46-49 | 46/47 **noop NC**(3-stack column at rest), 48 **down**(night spawn) | **D7 no-motion negative w/ column present**, D3-night |
| 6 | B 59-62 | 59 **noop**(4th solid lands → 4-stack complete), 60 **click** MELT column (lightblue-4→blue, gray→gold; gold+3 due to cloud occlusion), 61 **noop**(collapse: liquid slides sideways) | D7 stack landing, D6 melt, D7-liquid slide |
| 7 | B 65-69 | 65 **noop**(collapse settling), 66 **left**(day, occludes gold col0 + concurrent water settle), 67/68 **left CLAMP**(NC) | D7-liquid, D1 move + **2 clamp negatives** |
| 8 | C 4-10 | 4/5 **up**(only water falls — up==passive), 6 **down**(2nd blue spawn c3), 7/8 **noop**(2 liquids fall), 9 **click** FREEZE MID-AIR (blue-2→lightblue in place; no fall on click tick) | D4 x2, D3-day, D7-liquid, D6 freeze mid-air |
| 9 | C 22-26 | 22 **up NC**(fully static night scene), 23 **noop NC**, 24 **click** MELT 2-stack, 25 **noop**(melted r14 droplet slides sideways to r14c4) | **D4 static negative**, D7-NC, D6 melt, D7-liquid slide |
| 10 | C 28-33 | 28 **down**(blue spawn c4 over settled floor water), 29/30 **noop**(fall), 31 **down**(2nd spawn), 32 **up**(both droplets fall) | D3-day x2, D7-liquid, D4 mid-fall |

## Dynamic -> target coverage (positives / negatives)

| dynamic | ID-informative positives | FD positives | negatives |
|---|---|---|---|
| D1 left moves cloud (bounds-clamped) | 5 moves: ep2 x2 (day, settled water), ep3 x2 (night; one merges over moon), ep7 x1 (day + concurrent water motion) | same 5 (gray bar shifts -1 col) | **2 left-clamps** ep7 (NC at origin col 1) |
| D2 right moves cloud (bounds-clamped) | 4 moves: ep0 x2 (during fall), ep1 x2 | same 4 | **2 right-clamps** ep1 (NC at origin col 14) |
| D3 down spawns droplet below cloud origin, color = day state | 8 spawns, ID from new cell at r1 under cloud origin: day-blue x5 (c9, c14, c3, c4 x2), night-lightblue x3 (c2 x3) | same 8 (predicting the spawn COLOR requires the day-state rule) | day/night color contrast is internal (5 blue vs 3 lightblue spawns) |
| D4 up = no handler (passive only) | — (inherently unidentifiable vs noop, see below) | 4 mid-fall ups (water still falls: ep0, ep8 x2, ep10) | 1 **up on static scene = NC** (ep9) |
| D5 noop | — (same class as up) | 12 noops with passive water motion | **8 NO_CHANGE noops** (empty night ep3; settled liquids ep2; settled solids ep2; 3-stack column ep5; static 2-stack ep9) |
| D6 click = GLOBAL toggle (celestial day flip + every droplet liquid flip) | 6 clicks, ID from mass recolor: freeze x3 (settled ep2, NO-water ep3 = gold→gray only, MID-AIR ep8), melt x3 (settled ep2, 4-column ep6 w/ gold+3 occlusion signature, 2-stack ep9) | same 6; also FD must predict droplets do NOT fall on a click tick | the no-water click (ep3) is the near-miss: no lightblue/blue change at all, only the celestial flips |
| D7 passive: liquid falls then spreads sideways; solid falls and stacks vertically | — (noop class) | solid: 4 fall pairs incl. **2 stack landings** (ep4 r14, ep6 r12→4-stack); liquid: 8 fall pairs + **3 sideways-slide pairs** (ep6 collapse, ep7 settle, ep9 slide) | the 8 NC noops above (water at rest does NOT move — kills "water always falls"/step-clock shortcuts); solid-at-rest vs liquid-at-rest both present |

Contrastive structure (why shortcuts lose): move-vs-clamp for left AND right; spawn-blue
vs spawn-lightblue conditioned on the celestial state; click-with-droplets vs
click-with-none; falling vs settled noops in four different scene types; solid-stacking
vs liquid-spreading under the same noop verb; up-with-motion vs up-NC. Action timing is
irregular throughout the drives (no fixed cadence to exploit).

## Known aliasing / uncoverable (by construction of the game)

- **up vs noop**: `up` has no handler; an (X_t, X_t+1) pair can never distinguish it from
  `noop`. 5 ups kept anyway (4 with passive motion, 1 NC) so the decoder's best play is
  the up/noop class, not a hallucinated rule. This caps oracle ID at ~47/52 ≈ 0.90.
- **clamped left/right vs static noop/up**: the 4 clamp pairs are NO_CHANGE frames, so
  they alias with static noop/up under ID (further capping the oracle by up to 4 items).
  They are kept deliberately as the FD near-miss negatives for the bounds-clamp rule; an
  oracle that knows the clamp rule still predicts X_t+1 perfectly (FD-informative).
- On a **click tick water does not fall** (the on-clicked update replaces that tick's
  water update) — covered by ep8's mid-air freeze; documented here since it surprises
  frame-prediction rules.
- ep3 pair 3->4 (left at night into the moon) is visually subtle (gray-1/black+1 at the
  edge) but real and unambiguous.

## How this differs from train/

train/ (sliced from train_regen) keeps the cloud in cols 2-6 and ALL water in column 4
(3-droplet liquid spread, 3-solid stack), clicks at `8 8`, no clamps, no no-water click,
no mid-air toggles, 3-high stack max. TEST50: right-half cloud positions incl. wall
(cols 6-14), water columns 2, 3, 9, 14 (+ c4 only atop pre-existing settled floor water,
a state train never had), all four clamp events, a night-first opening, a 4-high stack,
mid-air freeze AND melt-with-column, melt-collapse cascade, click coords spread over the
grid (0 15, 7 0, 3 12, 10 3, 14 1, 5 8), and irregular action gaps. Same rules,
different situations — a cross-trajectory generalization test.
