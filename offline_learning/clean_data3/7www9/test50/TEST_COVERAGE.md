# 7www9 — TEST50 held-out pool coverage

Game: **7WWW9**, magnet attraction/repulsion (16x16). Red fixedMagnet at posPole
(x7,y7)/negPole (x7,y8) forever; blue mobileMagnet (2-cell vertical, posPole above
negPole) is the only controllable object. All magnetic effects resolve **on the same
step as the move** (passive rules evaluated after the move), so every dynamic here is
observable on the directional action itself.

Config: whitelist = `left,right,up,down,noop`; **keep_action_params = FALSE** (movement
game — the verb is the ID label). Positions below are `(x=col, y=row)` of blue's posPole.

Scored pool (what `verify_pool(test50,'left,right,up,down,noop',context_k=9)` reports):
**52 scored target transitions** across **14 episodes** (curated contiguous slices of
freshly driven trajectories, seed 0). Pool 52 ∈ 50±2; `balanced_split` returns the whole
pool when pool ≤ `--test-n 50`, so the pool IS the test set.

Sources: 5 independent `autumn_drive.py 7WWW9` drives (A–E) from the fixed start (4,7);
every episode is verbatim rows from one drive. Every transition was verified frame-by-frame
against the printed ASCII grids AND against a rule simulator that reproduces all 37
train_regen transitions exactly.

## 1. Core dynamics (from dynamics.txt / COVERAGE.md)

- **D1 normal move** — left/right/up/down shift blue exactly 1 cell when no magnetic rule
  triggers.
- **D2 blocked move** — the step is fully undone (NO_CHANGE) when the move would (a) overlap
  red (moveNoCollision) or (b) land a *like* pole orthogonally adjacent (Manhattan 1) to
  red's matching pole (repulsion cancels the step).
- **D3 attraction, inline 2-snap** — a move ending with an *opposite* pole exactly 2 cells
  from its red counterpart along the shared cardinal axis auto-pulls +1 in that direction:
  blue travels **2 cells in one step**. Corollary (same rule, opposite sign): stepping
  **away** from the stable 1-gap rest re-opens the 2-gap and the pull fires immediately,
  yielding NO_CHANGE (**snap-back hold**).
- **D4 attraction, diagonal snap** — a move landing an opposite pole 2 cells from its
  counterpart *perpendicular* to the motion pulls blue onto red's axis: **diagonal** net
  displacement (1 action-direction + 1 toward red).
- **D5 noop** — no handler; at any rest state nothing aligns, so NO_CHANGE.
- **D6 no spawn/despawn/recolor/win** — context only, never a scorable target.

## 2. Coverage map — every dynamic is a scored TARGET ≥ 4×

Notation: `epN sS` = the pair whose Action row is driven step S in episode_N (blue Δ given
as (dx,dy)). All change tags verified with `classify` on the stored frames.

| Dyn | Scored TARGET pairs | ID-informative? | FD-informative? |
|---|---|---|---|
| **D1** normal (24) | down: ep0 (6,5)→(6,6) · ep5 (8,8)→(8,9),(8,9)→(8,10) · ep10 (1,5)→(1,6),(1,6)→(1,7) — up: ep1 (6,6)→(6,5),(6,5)→(6,4) · ep3 (7,12)→(7,11) · ep4 (10,9)→(10,8) · ep8 (8,9)→(8,8) · ep9 (10,7)→(10,6) · ep13 (8,6)→(8,5),(8,5)→(8,4) — left: ep0 (7,5)→(6,5) · ep2 (7,5)→(6,5),(6,5)→(5,5) · ep6 (7,9)→(6,9),(6,9)→(5,9) · ep12 (10,7)→(9,7) — right: ep8 (6,9)→(7,9),(7,9)→(8,9) · ep10 (0,5)→(1,5) · ep11 (5,10)→(6,10) · ep13 (8,5)→(9,5) | yes (unit displacement = verb) | yes |
| **D2** blocked (5) | ep0 s20 **down** rep-block @(6,6) · ep4 s23 **left** coll-block @(8,8) · ep5 s25 **up** rep-block @(8,8) · ep7 s2 **right** coll-block @(6,8) · ep12 s14 **left** rep-block @(9,7) | aliased (NO_CHANGE; see §4) | yes (geometry ⇒ no-change) |
| **D3** inline snap (5 pos) | ep0 s16 **down** (7,3)→(7,5) · ep3 s16 **up** (7,11)→(7,9) · ep7 s1 **right** (4,8)→(6,8) row8 · ep4 s22 **left** (10,8)→(8,8) row8 · ep9 s16 **left** (10,6)→(8,6) row6 | yes (2-cell displacement still encodes the axis+direction) | yes (must predict 2 cells, not 1) |
| **D3** snap-back holds (2 neg) | ep5 s30 **down** @(7,9) (pull-up cancels) · ep8 s8 **right** @(8,8) (pull-left cancels) | aliased (NO_CHANGE) | yes (only the attraction rule predicts no-change here) |
| **D4** diag snap (8) | ep1 s23 **right**(+1,+1) (6,4)→(7,5) · ep2 s27 **down**(+1,+1) (5,5)→(6,6) · ep5 s28 **left**(−1,−1) (8,10)→(7,9) · ep6 s33 **up**(+1,−1) (5,9)→(6,8) · ep11 s5 **right**(+1,−1) (6,10)→(7,9) · ep12 s15 **up**(−1,−1) (9,7)→(8,6) · ep13 s19 **down**(−1,+1) (9,5)→(8,6) · ep13 s22 **left**(−1,+1) (8,4)→(7,5) | yes (the action-axis component of the diagonal encodes the verb) | yes (must predict diagonal, not straight) |
| **D5** noop (8) | ep0 @(7,5) · ep3 @(7,9) · ep5 @(7,9) · ep7 @(6,8) · ep9 @(10,6) · ep11 @(7,9) · ep12 @(9,7) · ep13 @(9,5) | aliased (NO_CHANGE) | yes (predict no-change) |

Every dynamic ≥ 4 scored targets in varied situations: D1 ×24 (all four verbs, rows 5–10,
cols 0–10), D2 ×5 (four verbs; both repulsion and collision, both sides of red), D3 ×5
positives (all four verbs; approaches from above/below/left/right of red) + 2 rule-negatives,
D4 ×8 (all four verbs × both perpendicular pull directions), D5 ×8 (six distinct rest cells,
near-red gaps and far field).

### Contrastive structure (15/52 = 29% NO_CHANGE negatives, req. 20–30%)

Per-verb outcome spread — no `verb → fixed Δ` shortcut survives:

- **down**: +1 (ep0,ep5×2,ep10×2) vs **+2** (ep0 snap) vs **diag** (ep2 +right, ep13 +left)
  vs **0** (ep0 rep-block, ep5 snap-back hold). ep0 alone contains +2/+1/0 for `down`.
- **up**: −1 (×8) vs **−2** (ep3 snap; ep3's other up is the −1 near-miss one row further out)
  vs **diag** (ep6 +right, ep12 +left) vs **0** (ep5 rep-block).
- **left**: −1 (×6) vs **−2** (ep4 row8, ep9 row6) vs **diag** (ep5 up-pull, ep13 down-pull)
  vs **0** (ep4 coll-block, ep12 rep-block). ep12 is a minimal pair: left (10,7)→(9,7) moves,
  then left AT (9,7) is repulsion-cancelled — same verb, adjacent frames.
- **right**: +1 (×5) vs **+2** (ep7 snap) vs **diag** (ep1 down-pull, ep11 up-pull) vs **0**
  (ep7 coll-block — the +2 and the 0 are consecutive frames; ep8 snap-back hold).
- **noop**: 8/8 NO_CHANGE, including at the 1-gap rest cells (7,5)/(7,9)/(6,8)/(9,7) right
  next to red — defeats "near red ⇒ something moves".

**No step-clock shortcut.** Effects are geometry-conditioned; slices start at driven steps
1..25 of five different drives, so leaked `Step` values map to different actions/outcomes
across episodes (e.g. step 16 is a down-snap in ep0, an up-snap in ep3, a left-snap in ep9,
a D1 up in ep13's drive). Noops sit at irregular offsets (steps 3,6,13,15,17,18,29), never
a fixed cadence.

## 3. Action histogram (pool of 52)

```
left   12   (6 D1 + 2 inline snap + 2 diag snap + 2 block)
up     12   (8 D1 + 1 inline snap + 2 diag snap + 1 block)
down   10   (5 D1 + 1 inline snap + 2 diag snap + 1 block + 1 snap-back hold)
right  10   (5 D1 + 1 inline snap + 2 diag snap + 1 block + 1 snap-back hold)
noop    8   (all NO_CHANGE)
```

## 4. Uncoverable / aliased — minimized & documented

- **NO_CHANGE ID-aliasing (15/52).** D2 blocks, D3 snap-back holds and D5 noops all leave
  the frame identical, so ID cannot distinguish them from each other. Intrinsic to the game
  (a cancelled move has no visible trace). They are exactly the FD-informative contrastive
  negatives; the other 37/52 pairs are fully ID-recoverable.
- **Fixed trigger cells.** The inline-snap geometry is rigid: e.g. a `down` inline snap can
  ONLY fire from (7,3), an `up` snap from (7,11). Test items on those cells necessarily share
  the trigger cell with train's — variation comes from the approach path/window (train reached
  (7,3) sideways via `right`; test descends col 7 from the top) and from the surrounding
  slice. The row-8 right/left snaps and both left-side snaps use cells train never scored.
- **Grid-edge behavior EXCLUDED.** dynamics.txt says moveNoCollision blocks at the boundary;
  the real interpreter instead lets blue slide **off-grid** (posPole at y=−1 renders 1 cell;
  fully off-grid at x=−1 renders 0 cells). Observed at (7,0)+up, (4,14)+down, (0,7)+left
  during driving. These clipped/invisible frames contradict the documented dynamics and make
  windows degenerate, so no off-grid transition (and no frame within one) is in any slice.
  Consequence: D2 is covered by magnetic/collision blocks only, no edge blocks.

## 5. How this TEST differs from `train/` (train_regen re-slice)

Same rules, systematically different situations:

| | train (from train_regen) | test50 |
|---|---|---|
| inline snaps | right@row6 (4,6)→(6,6); down/up@col7 approached sideways | right@row8, left@row8, left@row6 (never scored in train); col7 snaps approached vertically (down-chain from (7,0) top, up-chain from (7,14) bottom) |
| diag snaps | `up` only: (5,7)→(6,6) and (9,9)→(8,8) | 8 snaps over ALL four verbs incl. left/right/down diagonals train never exhibits; different cells ((6,4),(5,5),(8,10),(5,9),(6,10),(9,7),(9,5),(8,4)) |
| blocks | right@(5,7) rep; down@(7,5) coll; up@(7,9) coll | down@(6,6) rep, left@(8,8) coll, up@(8,8) rep, right@(6,8) coll, left@(9,7) rep — all-new verbs×cells×mechanisms (incl. the first left-blocks) |
| snap-back holds | none (train always left rest cells sideways) | 2 (down@(7,9), right@(8,8)) — new consequence of the same attraction rule |
| regions | mid-board cols 4–7, one col-4 descent, row 11 | col-7/col-8/col-10 verticals, row-9/row-10 laterals right of red, far-left cols 0–1 (ep10), top row 0 approach context |
| noops | (4,7) start, (8,8) end | 6 distinct cells, incl. 4 at the 1-gap rest positions around red |

No scored (state, action) pair replicates a train scored pair except the two unavoidable
fixed-trigger inline snaps (documented in §4), and even those differ in window context.

---
Additive only: `train/`, `train_regen/`, `test/`, `dynamics.txt`, `COVERAGE.md` unchanged.
Rebuild: 5 deterministic `autumn_drive.py 7WWW9` drives (seed 0) —
A `up×7,right×3,up,noop,down×5,noop,left,down,down,up,up,right,noop,left,left,down,noop`;
B `down×8,noop,right×3,up×5,noop,right×3,up,left,left,noop,up,down,down,left,noop,down,left,left,up,up,noop`;
C `down,right,right,noop,down,right,right,up,right,down,right,right,up,up,up,noop,left,left,down,noop,up,left,up,left,noop`;
D `left×5,noop,up,right,up,noop,right,down,down,noop,right×3,noop,down,down,left,left,up,noop`;
E `down×3,right×3,noop,right×3,up,up,left,noop,left,up,up,right,noop,down,up,up,left,noop`
→ slices per episode header comments in the build script; verify with
`clean_data3_tools.verify_pool('.../7www9/test50','left,right,up,down,noop',context_k=9)` → 52 targets.
