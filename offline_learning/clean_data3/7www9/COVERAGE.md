# clean_data3 coverage — 7www9 (magnet attraction/repulsion)

Config: whitelist = `left,right,up,down,noop` (NO click handler in this game),
`keep_action_params = FALSE` (movement game → action label collapsed to verb).
Test/ and dynamics.txt copied verbatim from clean_data2/7www9.

Grid convention: observation `[row][col] = [y][x]`. red (fixedMagnet) is stationary at
posPole (x=7,y=7) / negPole (x=7,y=8) for the entire trajectory. blue (mobileMagnet) is the
only controllable object, a 2-cell vertical pair (posPole above negPole). Magnetic effects
fire **on the same step as the move** (passive rule evaluated after the move), so every
magnetic dynamic is observable on a directional action — not on a later noop. This is the key
difference from nrdf6 and is what makes the dynamics ID-identifiable here.

## CORE DYNAMICS (extracted from dynamics.txt)

- **D1 Normal movement** — left/right/up/down shift blue exactly 1 cell in that direction
  (moveNoCollision) when blue is far from red so no magnetic rule triggers.
- **D2 Blocked move / move-cancel** — a move is fully undone (state unchanged) when it would
  either (a) overlap red / leave the grid (moveNoCollision), or (b) land a *like* pole
  orthogonally adjacent (Manhattan = 1) to red's matching pole (REPULSION cancels the step).
- **D3 Attraction — inline 2-cell snap** — a move that ends with an *opposite* pole aligned
  exactly 2 cells from red along the shared cardinal axis triggers an auto-pull of +1 cell in
  the same direction, so blue visibly travels **2 cells** in one step.
- **D4 Attraction — sideways/diagonal snap** — a move that lands an opposite pole 2 cells from
  its counterpart *perpendicular* to the motion triggers a perpendicular auto-pull onto red's
  axis, so blue moves **diagonally** (1 in the action direction + 1 toward red's column/row).
- **D5 No-op / passive-does-nothing** — noop (and click) have no handler; passive forces only
  act when geometry aligns, and at rest nothing aligns, so noop leaves the state unchanged.
- **D6 No spawn/despawn/recolor, no win/termination** — objects persist, colors fixed, no goal
  state. Context only; nothing changes per-step, so it is not a scorable ID/FD target.

## DYNAMIC × OBJECTIVE coverage and the GAP in the original pool

The original train trajectory is 120 rows but **94 are noop NO_CHANGE** and only 26 are moves.
Each move is separated by 3 padding noops. A `balanced-20` sample balances by action, but the
rare informative magnetic pairs are singletons hidden among many plain moves, so they are
likely **missed** by a random balanced draw → GEPA would learn "a move shifts blue 1 cell" and
never the magnet rule (the nrdf6 failure: key dynamic present only as easily-skipped context,
and a noop-flooded null class).

| Dynamic | TARGET under ID? | TARGET under FD? | In ORIGINAL balanced-20? | Gap |
|---|---|---|---|---|
| D1 normal move | YES (displacement → verb) | YES (state changes) | Yes (plentiful) | none |
| D2 blocked move | NO (state unchanged → action not recoverable; aliases noop) | YES (predict no-change from geometry) | only 4 of 120 rows; likely under-sampled | weak; provides FD contrast only |
| D3 inline 2-cell snap | YES (+2 still implies the axis direction) | YES (must predict 2 cells, not 1) | only 2 rows (down@31, up@95); easily missed | **MISSED** without curation |
| D4 diagonal snap | YES (y-component still implies up) | YES (must predict diagonal, not straight) | only 2 rows (up@11, up@111); easily missed | **MISSED** without curation |
| D5 noop no-change | NO (no change; irreducibly ambiguous w/ D2) | YES (predict no-change) | over-represented (94 rows) | flooding risk |
| D6 no spawn/win | n/a | n/a | n/a | context only; not scored |

nrdf6-style failure modes present here: (1) the magnet dynamics (D3/D4) appear only as a
handful of singleton targets among many plain moves and are easily skipped; (2) D2 blocked
moves and D5 noops are ID-unidentifiable (X_t == X_t+1) and the original is noop-flooded.
There is **no** unidentifiable-passive-noop trap (magnetics fire on the move, not a later
noop) and **no** step-counter shortcut risk (effects are conditioned on geometry, not cadence)
— but the singleton-coverage and noop-flooding gaps are real, hence the curation below.

## CURATED POOL (18 scored targets; verified with T.verify_pool)

by action: right 3, left 2, up 6, down 4, noop 3. Pool 18 ≤ train_n 20 → all 18 are scored
(tie_train_val: train == val; test drawn from the separate verbatim test/). Each episode is a
2-row slice `[N, N+1]`; the target is row N's action applied to grid[N]→grid[N+1]. Magnetic
effects are same-step so a 2-row slice fully exhibits the rule (no prev-delay → no window
needed; windows correctly shrink to prev=0/next=0 at the slice boundary).

| episode | steps | action | result (blue Δ in [y,x]) | dynamic |
|---|---|---|---|---|
| 0 | 3→4    | right | +1 col            | D1 normal (near-miss for D2 right-blocked) |
| 1 | 79→80  | right | +1 col            | D1 normal |
| 2 | 39→40  | left  | −1 col            | D1 normal |
| 3 | 47→48  | left  | −1 col            | D1 normal |
| 4 | 19→20  | up    | −1 row            | D1 normal (near-miss for D3/D4 up) |
| 5 | 91→92  | up    | −1 row            | D1 normal |
| 6 | 51→52  | down  | +1 row            | D1 normal (near-miss for D3 down) |
| 7 | 63→64  | down  | +1 row            | D1 normal |
| 8 | 31→32  | down  | **+2 row**        | **D3 inline snap** (opposite poles 2-away on y) |
| 9 | 95→96  | up    | **−2 row**        | **D3 inline snap** |
| 10| 11→12  | up    | **−1 row, +1 col**| **D4 diagonal snap** (pull right onto red column) |
| 11| 111→112| up    | **−1 row, −1 col**| **D4 diagonal snap** (pull left onto red column) |
| 12| 7→8    | right | NO_CHANGE         | **D2 blocked** (repulsion: like pole would be adjacent to red) |
| 13| 35→36  | down  | NO_CHANGE         | **D2 blocked** (collision: neg pole would overlap red) |
| 14| 115→116| up    | NO_CHANGE         | **D2 blocked** (repulsion: like pole would be adjacent to red) |
| 15| 0→1    | noop  | NO_CHANGE         | D5 passive-no-change |
| 16| 16→17  | noop  | NO_CHANGE         | D5 passive-no-change |
| 17| 84→85  | noop  | NO_CHANGE         | D5 passive-no-change |

### Contrastive structure (defeats per-action shortcuts)

- **down** can be +1 (ep6/7), +2 (ep8, D3), or 0 (ep13, D2) → forces the geometry-conditional
  rule; "down = +1" scores worse.
- **up** can be −1 (ep4/5), −2 (ep9, D3), diagonal (ep10/11, D4), or 0 (ep14, D2) → the richest
  contrastive set; "up = −1" cannot explain it.
- **right** can be +1 (ep0/1) or 0 (ep12, D2) → "right always shifts" scores worse.
- **noop** is exactly 3 NO_CHANGE targets (vs 94 in the original): keeps the ID null class
  represented without flooding the objective.

### Known limitation

No **left** blocked/attraction target exists in the source trajectory (blue only ever
approaches red from the left/below via up/down/right; no left move ever lands near red), so
left is covered by D1 normal moves only. Documented rather than fabricated — slices are
verbatim original rows per the recipe.

---
## REGENERATED trajectory (left-side magnetic interaction)

The "Known limitation" above was the real gap: in the original `clean_data2/7www9` rollout the
blue magnet only ever pressed toward red from **up/down/right**. It never approached red **from
the left**, so the three left-approach magnetic cases — left-side **block** (repulsion when a
like pole would land Manhattan-1 left of its red counterpart), left-side **diagonal snap**
(opposite pole 2 left of red → auto-pull right onto red's column), and left-side **inline
2-cell snap** (a `right` move that lands an opposite pole exactly 2 cells left of red snaps an
extra cell right, +2 cols in one step) — were **never** demonstrated, as scored target or even
as context. Re-slicing clean_data2 cannot create states the rollout never produced.

Fixed by **regenerating the trajectory** with `autumn_drive.py` driving the real `7WWW9.sexp`
(seed 0), saved at `train_regen/episode_0/trajectory.csv` (38 frames; filmstrip
`train_regen/viz.html`). Geometry (red fixed: posPole (x7,y7) / negPole (x7,y8)) was derived
from the SEXP and every transition verified frame-by-frame against the interpreter (magnetic
effects are **same-step** — the snap/block resolves on the move itself, no settle noops needed).

### Action sequence (37 actions) and what each demonstrates
`noop, left, right, right, right, up, up, up, left, left, down, down, right, up, up, up, right,`
`down, down, left, left, left, down, down, down, down, down, down, right, right, right, up, up,`
`right, right, up, noop`

Key scripted states (regen Step → effect):
- **4→5 `right` BLOCK [GAP]** — blue at (x5,y7) on red's rows; a `right` would land its posPole
  Manhattan-1 left of red's posPole → repulsion cancels the whole step → NO_CHANGE.
- **5→6 `up` LEFT DIAGONAL SNAP [GAP]** — `up` from (x5,y7) lands an opposite pole 2 cells left
  of red → auto-pull right onto red's column → blue moves Δ(−1 row, +1 col).
- **12→13 `right` LEFT INLINE 2-SNAP [GAP]** — `right` from (x4,y6) ends an opposite pole 2 left
  of red → snap +1 more → blue travels **+2 cols** in one step.
- **17→18 `down` INLINE 2-SNAP** Δ(+2,0); **18→19 `down` BLOCK** (collision into red) NO_CHANGE.
- **31→32 `up` INLINE 2-SNAP** Δ(−2,0); **32→33 `up` BLOCK** (collision) NO_CHANGE.
- **35→36 `up` DIAGONAL SNAP (pull-left)** Δ(−1,−1) — right-side counterpart of the left diagonal.
- plus normal 1-cell L/R/U/D moves far from red, and noop NO_CHANGE.

### Curated `train/` (re-sliced from train_regen; 19 scored targets, verify_pool confirmed)

`train/` is now 11 episodes sliced **verbatim from train_regen** (the old clean_data2 slices were
replaced). Same-step magnetic effects mean 2-row slices (prev=0) fully exhibit a rule; contrastive
verbs are grouped into 3–4-row slices so the near-miss negatives share real context windows.

By action: **noop 2, left 2, right 6, up 5, down 4** (pool 19 ≤ `--train-n 20` → all 19 scored).

| episode | regen steps | target pair(s) | dynamic |
|---|---|---|---|
| 0 | 0,1 | noop NO_CHANGE | D5 |
| 1 | 1,2 | left −1 col | D1 |
| 2 | 3,4,5,6 | right +1; **right BLOCK [GAP]**; **up DIAG-snap (−1,+1) [GAP]** | D1, **D2-left**, **D4-left** |
| 3 | 6,7 | up −1 row | D1 |
| 4 | 8,9 | left −1 col | D1 |
| 5 | 11,12,13 | down +1; **right INLINE-2snap (0,+2) [GAP]** | D1, **D3-left** |
| 6 | 16,17,18,19 | right +1; **down INLINE-2snap (+2,0)**; **down BLOCK** | D1, D3, D2 |
| 7 | 22,23 | down +1 row | D1 |
| 8 | 30,31,32,33 | right +1; **up INLINE-2snap (−2,0)**; **up BLOCK** | D1, D3, D2 |
| 9 | 34,35,36 | right +1; **up DIAG-snap pull-left (−1,−1)** | D1, D4 |
| 10 | 36,37 | noop NO_CHANGE | D5 |

### Contrastive structure (now includes left)
- **right**: +1 (normal) vs **0 (left BLOCK)** vs **+2 (left INLINE snap)** → a "right = +1" rule fails.
- **up**: −1 vs **−2 (inline)** vs **(−1,+1) left diagonal** vs **(−1,−1) right diagonal** vs **0 (block)**.
- **down**: +1 vs **+2 (inline)** vs **0 (block)**.
- **left/noop**: normal-move and null-class anchors (left stays the "away" direction by geometry).

The left-side block, left diagonal snap, and left inline 2-snap are now **scored targets**
(episodes 2 and 5), closing the gap. `dynamics.txt` unchanged; `test/` remains the original
verbatim clean_data2 copy and still lacks the regenerated left-approach states.

---
## TRAIN2 (untied-val expansion)

Purpose: a second, independently-driven curated pool (`train2/`) so the combined
`train/` (19) + `train2/` (31) = 50 transitions support an **untied** 30-train/30-val
GEPA split (rather than `--tie-train-val` reusing the same 19-20 transitions for both).
`train/`, `train_regen/`, `test/`, `test50/`, `dynamics.txt` are all **unchanged**.

### Method

A fresh full trajectory was driven with `autumn_drive.py 7WWW9` (seed 0, 72 actions) and
saved verbatim at `train_regen2/episode_0/trajectory.csv` (73 rows; filmstrip
`train_regen2/viz.html`). Every scored transition below was **derived analytically from
the SEXP** (closed-form geometry: red fixed at posPole (7,7)/negPole (7,8); the two
repulsion conditions and the two attraction conditions each reduce to a small, exhaustive
set of trigger cells given red's fixed position) and then **verified frame-by-frame**
against the real interpreter's printed ASCII grids before being cut into `train2/`. All 8
"special" (non-D1/non-plain-noop) transitions matched the analytical prediction exactly on
first drive — no corrections needed.

Exhaustive geometry fact used throughout: on this 16x16 board with red fixed, there are
only **3** repulsion-block trigger cells `{(6,7),(8,7),(7,6)}` and **7** attraction trigger
cells `{(7,4),(7,10),(5,6),(9,6),(5,8),(9,8),(7,6)}` (the 8th combinatorially-possible one,
`(7,8)`, coincides with red's own negPole cell and is unreachable/collision-only; `(7,6)`
sits in *both* the repulsion and attraction sets — driven and confirmed empirically at
train_regen (original) step 18→19: the real interpreter resolves it as a **block**, not a
pull). Cross-referencing `train/`'s actual `verify_pool` output against
`test50/TEST_COVERAGE.md`'s documented cells shows **all 6 reachable "pure inline 2-cell
snap" trigger+direction combinations are already scored** — 5 in `test50/` (held out) and 1
in `train/` (the `(5,6)`-via-`right` case). There is therefore **no way to add a second,
non-duplicate pure-inline example** to `train2/` without either leaking a `test50/`
held-out pair into train or exactly re-scoring a `train/` pair. train2 does not attempt
this (see Caveats). Instead train2 fills out the picture with 8 fresh scored situations that
were geometrically available: 3 fresh **diagonal-snap-family** examples via the attraction
rule's "snap-back hold" corollary (documented as part of the D3 family in
`test50/TEST_COVERAGE.md` §1: "Corollary (same rule, opposite sign)... snap-back hold"), 2
fresh **diagonal snaps** (D4) at previously-unused approach directions, and 3 fresh **block**
(D2) examples (2 repulsion, 1 collision) at previously-unused approach directions — all
confirmed live against the interpreter (see log excerpts below the table).

### Pool composition (`verify_pool` output, `context_k=9`)

```
### POOL prototypes/perc_invdyn/clean_data3/7www9/train2: 31 scored target transitions
    by action: {'noop': 3, 'right': 8, 'down': 9, 'left': 4, 'up': 7}
```
31 ∈ [28,32]. NO_CHANGE count = 9/31 = 29% (noop×3 + 6 magnetic-NO_CHANGE specials),
matching `test50`'s own 29% NO_CHANGE ratio and the recipe's 20–30% target band.

### Dynamic → scored-target coverage (train2 only)

| Dynamic | train2 target pairs (action, position, outcome) | count |
|---|---|---|
| **D1** normal move | 20 plain ±1-cell moves across all 4 verbs, in 3 fresh regions: far-north row y=2 (cols 4→9, ep2), far-east col x=14 (rows 2→5, ep3), row y=5 cols 10→7 (ep4), plus the D1 moves that sandwich each special below (ep5,6,8,9,10,11,12) | 20 |
| **D2** blocked/collision (NO_CHANGE) | `down`@(8,6) repulsion-block [ep6, fresh]; `up`@(6,8) repulsion-block [ep8, fresh]; `up`@(7,9) collision-block into red's negPole cell [ep9, fresh] | 3 |
| **D3** attraction — snap-back hold (corollary; NO_CHANGE) | `up`@(7,5) [ep5, fresh — trigger cell (7,4), never previously used as a snap-back]; `right`@(8,6) [ep6, fresh — trigger cell (9,6), previously only scored as a pure-inline `left` approach]; `left`@(6,6) [ep12, fresh — trigger cell (5,6), previously only scored as `right`-inline (train) / diagonal (train+test50)] | 3 |
| **D4** attraction — diagonal snap | `up` (9,9)→(8,8), Δ(−1,−1) [ep10, fresh — trigger (9,8) had only been scored as inline-`left`/snapback-`right` before]; `down` (5,7)→(6,8), Δ(+1,+1) [ep11, fresh — trigger (5,8) had only been scored as inline-`right`/diagonal-`up` before] | 2 |
| **D5** noop (NO_CHANGE) | rest (4,2) far-north [ep0]; rest (14,2) far-east corner [ep1]; rest (6,6) near-red [ep12, fresh cell — never a scored noop before] | 3 |
| **D6** no spawn/win | context only, never scored (unchanged) | n/a |

Every core dynamic is a scored target **≥ 2×** in train2 alone (D1 20×, D2 3×, D3-family 3×,
D4 2×, D5 3×); combined with `train/`'s own D3 pure-inline coverage (2× — the `(7,4)`-down
and `(7,10)`-up cases) the full `train ∪ train2` pool covers every dynamic richly.

### Contrastive structure (defeats per-action shortcuts)

- **up**: −1 normal (×3) vs **snap-back hold → 0** (ep5, trigger (7,4)) vs **repulsion-block →
  0** (ep8, trigger (6,7)) vs **collision-block → 0** (ep9, into red's negPole) vs
  **diagonal(−1,−1)** (ep10). Five distinct outcomes for one verb — "up = −1" cannot survive.
- **down**: +1 normal (×5) vs **repulsion-block → 0** (ep6, trigger (8,7)) vs
  **diagonal(+1,+1)** (ep11). ep6 alone contains a normal-down immediately followed by a
  blocked-down at the adjacent cell (rows 29→30→31): the tightest possible near-miss pair.
- **right**: +1 normal (×7) vs **snap-back hold → 0** (ep6, trigger (9,6)) — same slice as the
  down-block above, so ep6 shows down AND right both failing at the same rest cell (8,6),
  sandwiched between normal down-in and up-out moves (rows 29–33, 4 targets in one 5-row
  slice).
- **left**: −1 normal (×3) vs **snap-back hold → 0** (ep12, trigger (5,6)). ep12 also
  sandwiches a normal `down` immediately before the blocked `left` (rows 69→70→71).
- **noop**: 3/3 NO_CHANGE at three cells never used as a noop rest anywhere else in
  `train/`/`test50/` (far-north (4,2), far-east (14,2), near-red (6,6)).

Two more built-in near-misses: ep8 (rows 42–46) has a normal `up` (43→44) immediately
followed by a blocked `up` at the adjacent cell (44→45, trigger (6,7)) — same pattern as the
down/right case in ep6. ep9→ep10 (rows 47–52) chains a collision-block `up` (47→48)
immediately before a genuinely different mechanism, the diagonal-snap `up` two slices later
(50→51) — same verb, three different outcomes (0, and later +diagonal) depending on exact
cell.

### How train2's situations differ from `train/` and `test50/`

- **Regions**: far-north row y=2 (cols 4–14) and far-east col x=14 (rows 2–5) are used for
  D1/D5 filler and are **not touched by train/, train_regen/, or test50/ at all** (those
  pools stay within roughly cols 0–10, rows 0–14, never venturing to col ≥ 11 or the y=2
  latitude at cols > 7).
- **D4 diagonal directions**: train2's two diagonal snaps use trigger cells `(9,8)` (via
  `up`) and `(5,8)` (via `down`) — both cells had previously only been exercised via their
  *other* two directions (inline-approach and the opposite diagonal), never these two. The
  resulting absolute position pairs — (9,9)→(8,8) and (5,7)→(6,8) — do not appear in
  `train/` or `test50/`.
- **D3 family**: train2 contributes **snap-back holds**, a mechanism `train/` never scores
  at all and `test50/` only scores twice (at trigger cells (7,10) and (9,8)). train2's three
  snap-back holds use trigger cells (7,4), (9,6), (5,6) — all three previously scored *only*
  as pure-inline or diagonal-landing points, never as a hold. This is a genuinely new
  manifestation of the rule, not a re-score of an existing pair.
- **D2 approach directions**: `train/`'s two blocks are `right`-approach (regen step 4→5)
  and (from the original, pre-regen table) a `down`/`up`-approach pair; `test50/`'s five
  blocks use `down,left,up,right,left`. train2's three use `down`@(8,6) [trigger (8,7), a
  cell test50 only approached via `up`/`left`], `up`@(6,8) [trigger (6,7), previously only
  `right`(train)/`down`(test50)], and `up`@(7,9) [collision into red's negPole cell from
  below — the only collision-block in any pool approached from that side].

### Caveats

- **No fresh pure-inline (D3 "+2 straight travel") example.** As derived above, the game's
  fixed geometry has exactly 6 reachable pure-inline trigger+direction combinations and all
  6 are already scored (5 in the held-out `test50/`, 1 in `train/`). Adding a 7th would
  require either leaking a `test50/` pair into training data or re-scoring `train/`'s own
  pair verbatim; train2 does neither. The combined `train ∪ train2` pool still covers this
  sub-case via `train/`'s existing 2 instances; train2's contribution to the D3 rule is via
  the snap-back-hold corollary (3 fresh instances) instead.
- **Three pre-frames coincide with existing `test50/` rest cells** (same static frame, no
  visible difference at that state), though every one of them pairs with a **different
  action** than test50's use of the same cell, so the scored (X_t, action, X_t+1) triple is
  not a duplicate: (6,8) [train2 `up`→block vs test50's `right`→collision-block at the same
  cell], (7,9) [train2 `up`→collision-block vs test50's `down`→snap-back-hold], and (6,6)
  [train2 `left`→snap-back-hold vs test50's `down`→repulsion-block]. All three are
  NO_CHANGE outcomes on both sides, which is consistent with (not a worsening of) the
  pre-existing NO_CHANGE/ID-aliasing property already documented in
  `test50/TEST_COVERAGE.md` §4 — a static frame can never disambiguate which blocked action
  was attempted, regardless of which pool it appears in.
- `(7,6)`'s dual-condition ambiguity (satisfies both the repulsion-adjacency and the
  attraction-2-away tests simultaneously) was not re-exercised in train2 — it's already
  scored in the original (pre-regen) `train/` history and the real-interpreter resolution
  (block wins) is noted here for the record in case a future regeneration touches it again.
