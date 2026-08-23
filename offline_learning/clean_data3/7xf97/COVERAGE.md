# 7xf97 — clean_data3 coverage

Config: whitelist=`left,right,up,down,noop,click`; `keep_action_params=FALSE` (movement game,
click collapsed to verb `click`; click LOCATION is NOT the label).

Render legend used during curation: `.`=black(bg) `G`=gold(Sun 3x3) `C`=gray(Cloud 4x3)
`w`=blue(Water) `g`=green(Leaf) `p`=mediumpurple(Leaf grown into row 12).

## 1. CORE dynamics (from dynamics.txt)

- **D1 left**: moves the Cloud one cell left (gray block x-1).
- **D2 right**: moves the Cloud one cell right (gray block x+1).
- **D3 down**: spawns a Water cell one below the cloud origin (a raindrop).
- **D4 click-on-Sun**: moves the Sun one cell left if `movingLeft` else right; click elsewhere = no-op.
- **D5 up**: no handler -> no-op.
- **D6 noop**: no action effect (but passive/clock dynamics still fire).
- **D7 passive gravity**: every step each Water cell falls one row; off-grid water is removed.
- **D8 passive Sun edge-bounce**: at Sun origin x==0 set `movingLeft=false`, at x==13 set true.
  (Flips intended direction only; the Sun moves only when clicked.)
- **D9 Water-vs-Leaf collision (despawn)**: water whose next-down cell is a Leaf is removed.
- **D10 watering grows a Leaf**: descending water about to land on a GREEN leaf, AND Sun not
  overlapping Cloud -> a new Leaf grows one cell above; mediumpurple if it grows into row 12,
  else green. If Sun overlaps Cloud, no growth.
- **D11 spawn/despawn bookkeeping**: water only spawns via `down`; despawns off-grid or on leaf-hit.
- **D12 win/termination**: none defined.

## 2. Original-train coverage and GAPS (balanced-20 view)

`T.dump_transitions('7xf97', ...)` on the original 126-row trajectory. ID = action recoverable
from the X_t->X_t+1 change; FD = state visibly changes per the rule.

| Dynamic | TARGET under ID? | TARGET under FD? | In original pool / GAP |
|---|---|---|---|
| D1 left | YES — gray block shifts x-1 | YES | well represented (many lefts) |
| D2 right | YES — gray block shifts x+1 | YES | present but SCARCE (only 2 rights: 23->24, 27->28) |
| D3 down | **NO** — every `down` transition is NO_CHANGE | **NO** | **GAP (nrdf6-type).** Water spawns one below cloud origin, INSIDE the 3-tall gray Cloud, so it is occluded for ~2-3 steps and only becomes visible (blue+1) later, always on a `noop`. So `down` is unidentifiable from the grid at its own step, and the visible spawn is mislabeled `noop`. |
| D4 click-on-Sun | YES — gold block shifts | YES | present (3 clicks: 3->4, 7->8, 11->12); all move RIGHT (Sun stays near x=0..3, `movingLeft` never set true). No click-elsewhere no-op exists in the trajectory, and no left-moving click -> can't contrast click-direction or click-miss. |
| D5 up | n/a | n/a | **GAP** — `up` never pressed; indistinguishable from noop anyway. Context only. |
| D6 noop | trivially noop | mixed (passive fires) | abundant |
| D7 gravity | NOT action-specific (fires on every action incl. left/right) | YES on noop | present; risk = read as "noop moves water". Needs contrast with a MOVE step that also has water falling. |
| D8 Sun edge-bounce | NO — hidden boolean, Sun never moves | NO | **GAP** — Sun never reaches an edge in this trajectory; the flag flip is unobservable. Not scorable. |
| D9 collision-despawn | weak (water vanishes on noop) | YES on noop | present (66->67, 112->113, ...); confounds with D10 (both are blue-1). |
| D10 watering-grow | weak (leaf appears on noop) | YES on noop | green growth 49->50, 53->54; purple-at-row-12 only ONCE (56->57). Must keep the rare purple case. |
| D11 spawn/despawn | see D3/D9 | see D3/D9 | spawn side unidentifiable (D3 gap); despawn side ok. |
| D12 termination | n/a | n/a | nothing to score. |

Main GAPS carried into the curated set: **D3 down is occluded/delayed (nrdf6 pattern)**;
**D5 up and D8 edge-bounce are never exercised / unobservable** (context only, cannot be made
targets from this trajectory); right and purple-watering are scarce so both are pinned in.

## 3. Curated slices (verified pool = 20 scored targets)

Each slice = consecutive ORIGINAL step numbers copied verbatim into its own `episode_*`.
`T.verify_pool(...)` confirms 20 targets: collapsed by action = click 2, left 3, right 2, down 1,
noop 12 (verify_pool prints the raw `click 1 1`/`click 1 3` strings; both collapse to `click`).

| episode | steps | target pairs -> dynamic | role |
|---|---|---|---|
| 0 | 3,4,5 | 3->4 click (gold +1 col) ; 4->5 noop NO_CHANGE | D4 positive + noop near-miss |
| 1 | 11,12,13 | 11->12 click (gold +1 col) ; 12->13 noop NO_CHANGE | D4 positive + noop near-miss |
| 2 | 15,16,17 | 15->16 left (gray -1 col) ; 16->17 noop NO_CHANGE | D1 positive + move-stops contrast |
| 3 | 67,68 | 67->68 left (gray -1 col), clean (no water) | D1 positive, uncluttered |
| 4 | 23,24,25 | 23->24 right (gray +1 col) ; 24->25 noop NO_CHANGE | D2 positive + direction contrast vs D1 |
| 5 | 27,28,29 | 27->28 right (gray +1 col) ; 28->29 noop NO_CHANGE | D2 positive (right is scarce) |
| 6 | 31,32,33,34 | 31->32 down NO_CHANGE ; 32->33 noop NO_CHANGE ; 33->34 noop **blue+1 (spawn appears)** | D3/D11 delayed-spawn: `down` sits in the spawn-noop's back-window (prev=2) and in `down`'s own fwd-window (next=2). Contrast: the other NO_CHANGE noops do NOT spawn. |
| 7 | 48,49,50 | 48->49 noop **blue moves down (gravity)** ; 49->50 noop **blue-1 green+1 (watering)** | D7 gravity + D10 green-grow |
| 8 | 55,56,57 | 55->56 left **+ blue moves down** ; 56->57 noop **blue-1 mediumpurple+1** | D10 purple-at-row-12 (the only instance) + D7 gravity firing ON a left step (proves gravity is action-independent) |
| 9 | 65,66,67 | 65->66 noop **blue moves down (gravity)** ; 66->67 noop **blue-1 black+1 (despawn on leaf)** | D9 collision-despawn, contrasted against D10: same blue-1 cue but leaf does NOT grow (water hit a purple, not green, leaf) |

### Contrastive structure (defeat shortcuts)
- **Direction**: left (ep 2,3,8) vs right (ep 4,5) -> "gray moved" is not enough, sign matters.
- **click vs noop**: each click is paired with an immediately-following NO_CHANGE noop -> Sun
  moves only on click, not on idle steps.
- **Move-is-one-shot**: each move is followed by a NO_CHANGE noop -> the block does not keep drifting.
- **down vs noop (spawn)**: ep 6 spawn-noop (blue+1) vs the 5 other NO_CHANGE noops and the
  gravity noops -> "noop spawns water" is false; the spawn correlates with the `down` carried in
  the window, defeating any `step % k` clock shortcut.
- **gravity is action-independent**: water falls on noops (ep 7,8,9) AND on a left (ep 8) ->
  "noop moves water down" is wrong; gravity is unconditional per-step.
- **watering vs despawn**: green/purple growth (ep 7,8: blue-1 +leaf) vs leaf-collision removal
  (ep 9: blue-1 +black) -> both are "a water cell disappears", but the outcome is conditional
  on the leaf it lands on (green -> grow; purple/non-green -> just removed).

### Known residual gaps (cannot be fixed from this trajectory; documented, not hidden)
- **D3 down** remains unidentifiable at its own step (occlusion); best achievable is the
  cause-in-window construction in ep 6.
- **D5 up** and **D8 Sun edge-bounce** are never exercised / unobservable in the source
  trajectory, so they appear only as context, never as scored targets.
- **click direction (D4 left-move)** and **Sun-overlaps-Cloud blocks watering** never occur in
  the source, so only the right-move click and the unblocked-watering cases are represented.

---
## REGENERATED trajectory (sun edge-bounce, click-left, blocked watering, `up`)

The original `clean_data2/7xf97` rollout never produced the states four core dynamics need, so
re-slicing (sections 1-3 above) could not score them. They are now filled by **regenerating the
trajectory** with `autumn_drive.py` driving the real `7XF97.sexp` (seed 0). Full trajectory:
`train_regen/episode_0/trajectory.csv` (111 frames), filmstrip `train_regen/viz.html`. The
curated `train/` below is sliced verbatim from `train_regen` (NOT from clean_data2). `test/` and
`dynamics.txt` are left as the original verbatim copies (test/ still lacks the regenerated states).

### The four gaps (none occur in the original rollout)
1. **`up` never pressed** — now step 1 (`up`), a NO_CHANGE no-op.
2. **Sun edge-bounce never observable** — the Sun is driven by 11 right-clicks to overlap the
   cloud, 2 more to the right edge (origin x=13, fully under the cloud), then clicked left back
   across the open grid, then driven all the way to the LEFT edge (origin x=0) where the
   `x==0 -> movingLeft=false` rule fires (visibly): a subsequent click then moves it RIGHT again.
3. **Click never moved the Sun LEFT** — after the right-edge bounce sets `movingLeft=true`,
   clicks move the Sun left (steps 41-44 in the open grid, and 107->108).
4. **Sun-overlaps-Cloud blocks watering** never happened — with the Sun parked at origin x=11
   (cols 11,12,13 — overlapping the cloud at cols 12-15, col 11 visible), a raindrop in col 13
   lands on the fresh green leaf (15,13) and is blocked: the water despawns but NO leaf grows.

### Action sequence (rationale)
`autumn_drive.py 7XF97 ... --actions <seq>` (seed 0), 110 transitions, in phases:
- **P1 (steps 0-6)**: `noop, up, left, noop, left, right, right` — `up`/`noop` no-ops, cloud
  left/right moves (returns to home origin 13), all with no water so D1/D2 are clean.
- **P2 (7-17)**: 11 right-clicks drive the Sun x0->x11 (open-grid click-RIGHT targets; the Sun
  slides under the cloud at the end and is hidden).
- **P3 (18-33)**: `down` + 15 noops — the raindrop spawns INSIDE the 3-tall cloud (occluded
  steps 18-20, visible blue at row 3 by step 21), falls col 13, and is **BLOCKED** at the leaf
  (step 32->33) because the Sun (origin 11) overlaps the cloud: water despawns, green count stays 8.
- **P4 (34-36)**: 2 right-clicks to the right edge (x=13) + a noop so the `x==13 -> movingLeft=true`
  rule fires (the right-edge bounce; the Sun is hidden under the cloud here).
- **P5 (37-45)**: 8 left-clicks drive the Sun x13->x5, emerging from the cloud — **click-LEFT**
  targets in the open grid (proves the bounce flipped the direction).
- **P6 (45-89)**: three UNBLOCKED waterings in col 13 (Sun parked far left at origin 5): grow
  green (14,13), green (13,13), then **mediumpurple (12,13)** (grows INTO row 12). A cloud
  `left,right` is injected mid-fall (steps 50-51) so **gravity is shown firing during a move**,
  not only on noop.
- **P7 (90-98)**: a 4th raindrop col 13 lands on the now-topmost **mediumpurple** leaf -> water
  despawns, NO growth (D10 needs a *green* leaf): pure-despawn (D9) contrast.
- **P8 (99-110)**: 5 left-clicks drive the Sun to the LEFT edge (origin 0), a noop fires
  `x==0 -> movingLeft=false`, then a click moves it **RIGHT** — the fully-visible left-edge bounce.

### Curated pool (`verify_pool` -> 20 scored targets; keep_action_params=FALSE)
By verb (clicks collapse to `click`): `noop=10, click=5, left=2, up=1, right=1, down=1`.

| ep | regen steps | target pair(s) -> change | dynamic / role |
|----|-------------|--------------------------|----------------|
| 0  | 0,1,2       | 0->1 noop NC ; 1->2 **up** NC | D6 noop ; **D5 `up`** (gap1; ID-identical to noop) |
| 1  | 2,3,4       | 2->3 **left** gray-1 ; 3->4 noop NC | D1 ; move-is-one-shot negative |
| 2  | 7,8         | 7->8 **click-RIGHT** gold+1 | D4 click-right (movingLeft=false) |
| 3  | 41,42,43    | 41->42, 42->43 **click-LEFT** gold-1 | **gap3** click-left (movingLeft=true, open grid) |
| 4  | 18,19,20,21 | 18->19 **down** NC(occl) ; 19->20 noop NC ; 20->21 noop **blue+1 spawn** | D3 occluded/delayed spawn (cause in window) |
| 5  | 50,51,52    | 50->51 **left**+blue-fall ; 51->52 **right**+blue-fall | D1/D2 + **D7 gravity is action-independent** |
| 6  | 31,32,33    | 31->32 noop blue-fall ; 32->33 noop **BLOCKED** (blue-1, green stays 8) | D7 gravity-on-noop ; **gap4** blocked (Sun origin 11 overlaps cloud) |
| 7  | 59,60       | 59->60 noop **blue-1 green+1** | D10 green grow (UNBLOCKED, Sun origin 5 clear) |
| 8  | 88,89       | 88->89 noop **blue-1 mediumpurple+1** | D10 grows into row 12 = mediumpurple |
| 9  | 101,102     | 101->102 noop **blue-1 black+1** | D9 despawn on a non-green (mediumpurple) leaf, NO grow |
| 10 | 107,108,109,110 | 107->108 **click-LEFT** to edge ; 108->109 noop NC (edge flip) ; 109->110 **click-RIGHT** | **gap2** left-edge bounce (same click, direction reverses at x=0) |

### Contrastive structure (defeats shortcuts)
- **click-RIGHT (ep2, ep10 109->110) vs click-LEFT (ep3, ep10 107->108)** — click direction is
  conditional on the hidden `movingLeft`, flipped by the edge bounce; not a fixed direction.
- **BLOCKED (ep6 32->33, Sun origin 11 over cloud, col 11 gold visible) vs UNBLOCKED green-grow
  (ep7, Sun origin 5 far)** — SAME col-13 fresh green leaf; only the Sun's overlap differs, so a
  "watering always grows" rule scores worse than the sun-block conditional.
- **green (ep7) vs mediumpurple-at-row-12 (ep8) vs despawn-on-non-green (ep9)** — D10 is
  conditional on a *green* target leaf and the row (purple at 12); D9 removes water regardless.
- **gravity on noop (ep6 31->32) vs on left/right (ep5)** — water falls every tick, action-
  independent; defeats "noop moves water" / any `step%k` clock.
- **down occluded (ep4 18->19 NC) -> visible spawn (20->21)** — the `down` cause sits in the
  spawn-noop's back-window (nrdf6 cause-in-window construction).

### Residual unidentifiabilities (documented, not hidden)
- **`up` (gap1) is ID-identical to `noop`** — both are NO_CHANGE; the regenerated `up` is a
  scored target but recoverable only as "a no-op", not distinguishable from `noop`. Inherent.
- **The bounce flag itself is hidden state.** `movingLeft` flipping (ep6/P4 right edge under the
  cloud; ep10 108->109 left edge) is never directly visible; it is identifiable ONLY via the
  subsequent click-direction reversal (the scored ep10 109->110 vs 107->108, and ep3 vs ep2).
- **`down` is occluded at its own step.** The raindrop spawns inside the 3-tall cloud, so the
  `down` transition is NO_CHANGE; the visible spawn one step later is labeled `noop`. Best
  achievable is the ep4 cause-in-window construction (unchanged from the original analysis).

---
## PATCH: enriched tree-growth coverage (central mechanic)
Tree/leaf growth — a raindrop landing on the tree's top leaf grows a NEW leaf one cell higher
(green, or mediumpurple at row 12) — is the game's central mechanic but was only scored as 2
bare 2-frame slices. Now all THREE growth events are scored, each as a 3-frame slice showing
raindrop-descends-onto-leaf-top -> new leaf appears, so the tree visibly climbs row 15->14->13->12:
  ep7 [58,59,60] GREEN grow (row15->14) ; ep8 [73,74,75] GREEN grow (row14->13) ;
  ep9 [87,88,89] MEDIUMPURPLE grow (row13->12).
Each growth target has prev=1 (the raindrop-on-leaf-top cause is in-window). Pool is now **24
targets** (> default 20), so 7xf97 must be run with **--train-n 24 --val-n 24** to keep all.

---
## TRAIN2 (untied-val expansion)

A SECOND regenerated trajectory + curated pool, built so the per-game pool (`train/` 24 +
`train2/` 32 = 56) supports an **untied 30-train/30-val split** instead of `--tie-train-val`.
Full drive: `train_regen2/episode_0/trajectory.csv` (126 frames, `autumn_drive.py 7XF97`, seed
0), filmstrip `train_regen2/viz.html`. Curated: `train2/episode_0..14/trajectory.csv` (15
slices, verbatim rows from `train_regen2`), filmstrip `train2/viz.html`.

### Verified pool
```
T.verify_pool('prototypes/perc_invdyn/clean_data3/7xf97/train2','left,right,up,down,noop,click', context_k=9)
-> 32 scored target transitions
   by action (raw): {'noop': 18, 'up': 2, 'left': 3, 'down': 3, 'right': 2,
                      'click 1 12': 1, 'click 1 13': 1, 'click 1 1': 1, 'click 1 0': 1}
   by verb (clicks collapse): noop=18, click=4, left=3, down=3, right=2, up=2   (sum=32)
```

### Design: why the situations are NEW (not a re-slice of train/ or test50/)
The whole regenerated trajectory pivots on **one fresh column, 11** (never the raindrop column
in train [13] or test50 [7, 5, 6, 9]), with the cloud parked there for the entire tree saga, and
the Sun/Cloud overlap (needed for the blocked-watering gap) engineered as a **stationary
mid-grid overlap** (Sun clicked to origin 9 while the Cloud sits at 11) — a materially different
mechanism from train's "Sun hides under the Cloud during the edge-bounce trip" and test50 D's
"cloud parked at col 8, sun at cols 6-8". A second, distinct even column (12) is used for the
off-grid despawn (test50 used col 6). The Sun's full round trip is also decoupled from the
Cloud's position (Cloud never moves during the clicking phases here, unlike test50 B where the
Cloud transits 13->2->13 alongside the Sun).

### Curated slices (32 scored targets; `T.verify_pool` confirms)

| ep | regen steps | target pair(s) -> change | dynamic / role |
|----|-------------|---------------------------|----------------|
| 0  | 0,1,2       | 0->1 noop NC ; 1->2 **up** NC | D6 ; D5 up #1 |
| 1  | 2,3,4       | 2->3 **left** gray-1 ; 3->4 **left** gray-1 | D1 #1,#2 — clean cloud move 13->12->11 (settles at the NEW tree column, no water present) |
| 2  | 8,9,10,11   | 8->9 **down** NC(occl) ; 9->10 noop NC(occl) ; 10->11 noop **blue+1 spawn** | D3 #1 cause-in-window (col 11, first drop) |
| 3  | 21,22,23    | 21->22 noop blue-fall ; 22->23 noop **blue-1 green+1** (row14) | D7 ; D10 #1 (first growth) |
| 4  | 23,24,25,26 | 23->24 **down** NC ; 24->25 noop NC ; 25->26 noop **blue+1 spawn** | D3 #2 cause-in-window (2nd drop, tree already 1 leaf tall) |
| 5  | 36,37       | 36->37 noop **blue-1 green+1** (row13) | D10 #2 (second growth, same leaf column) |
| 6  | 57,58,59    | 57->58 noop blue-fall ; 58->59 noop **BLOCKED despawn** (blue-1, green count UNCHANGED) | D9 negative #1 — Sun clicked to x=9 overlaps Cloud@11; SAME leaf/column as ep3/ep5, only the Sun's position differs |
| 7  | 63,64,65,66 | 63->64 **click-RIGHT** gold+3 (arrive x=13) ; 64->65 noop NC (**right-edge bounce**) ; 65->66 **click-LEFT** gold-3 (reversal) | D4 both directions ; D8 #1 (right edge) |
| 8  | 83,84       | 83->84 noop **blue-1 mediumpurple+1** (row12) | D10 #3 — grows into row 12 (now UNBLOCKED again, Sun clicked away to x=7) |
| 9  | 96,97       | 96->97 noop **blue-1 black+1** (no leaf-color change) | D9 negative #2 — pure despawn on the now-topmost mediumpurple (non-green) leaf |
| 10 | 98,99,100,101,102 | 98->99 **right** gray+1 (cloud 11->12) ; 99->100 **down** NC ; 100->101 noop NC ; 101->102 noop **blue+1 spawn** | D2 #1 ; D3 #3 cause-in-window (fresh EVEN column 12, no leaf) |
| 11 | 106,107,108 | 106->107 **left**+blue-fall ; 107->108 **right**+blue-fall | D1 #3 ; D2 #2 ; D7 gravity fires DURING both moves (action-independence) |
| 12 | 113,114,115 | 113->114 noop blue-fall ; 114->115 noop **blue-1 black+1** (OFF-GRID, no leaf involved) | D7 ; D9/D11 negative #3 — despawn-by-bounds, distinct mechanism from the leaf-collision despawns above |
| 13 | 121,122,123,124 | 121->122 **click-LEFT** gold-1-ish move ; 122->123 noop NC (**left-edge bounce**) ; 123->124 **click-RIGHT** (reversal) | D4 both directions ; D8 #2 (left edge) |
| 14 | 124,125     | 124->125 **up** NC | D5 up #2 |

### Dynamic -> coverage summary

| Dynamic | train2 target pairs (>=2 where applicable) | new situation vs train/test50 |
|---|---|---|
| D1 left | ep1 x2, ep11 (compound w/ fall) = 3 | fresh anchor column 11 |
| D2 right | ep10, ep11 (compound w/ fall) = 2 | one compound instance proves gravity fires during a right too |
| D3 down (occluded, cause-in-window) | ep2, ep4, ep10 = 3 | 2 instances at col 11 (different tree heights) + 1 at fresh even col 12 |
| D4 click-on-Sun | ep7 x2 (R+L), ep13 x2 (L+R) = 4 | round trip decoupled from Cloud motion (Cloud stationary throughout) |
| D5 up | ep0, ep14 = 2 | inherently NO_CHANGE / ID-identical to noop (same documented caveat as train/test50) |
| D6 noop | 18 (incl. all NO_CHANGE + passive-event noops) | abundant, as in both prior pools |
| D7 gravity (action-independent) | ep3, ep6 (plain falls) + ep11 (during left AND right) | ep11's compound is the clearest action-independence proof in this pool |
| D8 Sun edge-bounce | ep7 (right edge), ep13 (left edge) = 2 (1/edge) | IDed only via the surrounding click-direction reversal (flag is hidden state, inherent) |
| D9 leaf-collision despawn (no grow) | ep6 (blocked), ep9 (non-green) = 2 | two DIFFERENT reasons for "no growth" on the SAME col-11 tree lineage — a minimal pair train/test50 don't have (their blocked/unblocked contrasts are cross-episode at a different column) |
| D10 watering-grow | ep3 (green@14), ep5 (green@13), ep8 (purple@12) = 3 | continuous single-column narrative: grow, grow, [BLOCKED near-miss], grow-purple, [pure-despawn near-miss] |
| D11 spawn/despawn bookkeeping | spawn = D3; off-grid despawn = ep12 | fresh even column (12, vs test50's 6) |
| D12 termination | n/a | none defined |

### Contrastive negatives
Strict `NO_CHANGE` pairs: ep0 (0->1, 1->2), ep2 (8->9, 9->10), ep4 (23->24, 24->25), ep7
(64->65), ep10 (99->100, 100->101), ep13 (122->123), ep14 (124->125) = **11/32 (34%)**.
Plus 3 additional "wrong-outcome" near-misses that DO show a visible change but defeat a naive
rule rather than a literal no-op: ep6's blocked despawn (same leaf as ep5's growth, only the
Sun's overlap differs), ep9's pure despawn (same mechanism as ep3/5/8 but the wrong leaf color),
and ep12's off-grid despawn ("water disappears" for a bounds reason, not a leaf reason).
Counting both kinds, 14/32 (44%) of targets play a negative/contrastive role — above the
20-30% guideline, but consistent with the methodology's emphasis on defeating shortcuts for a
game whose central mechanic (watering) is conditional on hidden state (Sun/Cloud overlap) and
leaf color.

### Residual caveats (inherited from train/test50, unavoidable)
- **`up`** (ep0, ep14) is ID-identical to `noop` — both NO_CHANGE; a no-op verb with no visible
  or distinguishing effect. Included twice purely for whitelist-verb coverage.
- **D8's `movingLeft` flag is hidden state.** The bounce itself (ep7 64->65, ep13 122->123) is a
  NO_CHANGE noop; it is identifiable ONLY via the subsequent click-direction reversal in the SAME
  slice (ep7 63->64 vs 65->66; ep13 121->122 vs 123->124).
- **`down` (D3) remains occluded at its own step** — the raindrop spawns inside the 3-tall Cloud,
  so the `down` transition itself is always NO_CHANGE; the visible spawn one step later is
  labeled `noop`. All three D3 instances here use the cause-in-window construction (down's NC
  sits in the visible-spawn noop's back-window), same as train/test50.
- **Some click transitions are themselves aliased** (e.g. regen-trajectory steps 61->62,
  67->68 where the Sun moves entirely within the Cloud's footprint, 0 visible cells before AND
  after) — these were deliberately excluded from `train2`'s curated slices, matching the
  "avoid aliased scored pairs" requirement.
