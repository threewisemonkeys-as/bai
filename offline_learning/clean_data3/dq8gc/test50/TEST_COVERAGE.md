# dq8gc — test50 held-out pool (52 scored transitions)

Whitelist: `left,right,up,down,noop,click` • keep_action_params = FALSE (the ID label is the
bare verb; `click R C` collapses to `click`). Verified with
`clean_data3_tools.verify_pool(<...>/test50, 'left,right,up,down,noop,click', context_k=9)`:
**52 scored targets** — `noop 15, down 9, right 7, up 7, left 6, click 8`.

All transitions are FRESHLY DRIVEN (autumn_drive.py, seed 0) — none replicate the train
slices or the original test trajectory. Six themed drives were run and sliced; each
`episode_*/trajectory.csv` is a verbatim contiguous slice of one drive. Deterministic
regeneration commands (seed 0) for the six source drives:

```
A: down,noop,down,noop,down,noop,noop,right,right,noop,right,noop,right,noop,up,noop,up
B: right,down,click_3_4,noop,right,noop,down,down,right,click_6_6,noop,down,noop
C: click_6_6,down,left,left,left,noop,up,noop,up,up,up,noop,up,noop,noop
D: down,right,click_5_7,noop,left,left,down,right,down,noop,click_6_6,up,up,noop
E: click_3_4,down,down,noop,click_5_7,left,left,noop,click_2_2,down,down,down,noop,noop,noop,noop,left
F: click_5_3,up,left,up,noop,noop
```

## 1. Core dynamics (from dynamics.txt)

- **D1–D4 moves** — activeParticle col∓1 / row∓1, unconditional, no collision check.
- **D5 overlap / no-collision** — moving onto an occupied cell merges to one rendered cell
  (count −1); moving off reveals the underlying particle (count +1).
- **D6 click-swap** — clicking an inactive particle makes it the new active, releasing the
  old active into the inactive list. The immediate frame is NO_CHANGE.
  **New mechanism exercised here (verified empirically, consistent with the SEXP):** the
  click handler rebuilds `inactiveParticles` from the PREVIOUS state, so a click
  **suppresses any pending infection for that tick** (the recolor that a noop/move would
  have shown is delayed by one step). This makes clicks in pending-infection states
  uniquely ID-recoverable (see §3).
- **D7 passive infection** — every tick, an inactive orthogonally adjacent to any
  darkgreen particle (previous state) turns darkgreen; 1-step lag; monotone; spreads one
  cell per tick along orthogonal chains.
- **D8 active-particle health rule** — a gray ACTIVE orthogonally adjacent to a darkgreen
  particle turns darkgreen (the active-side counterpart of D7).
- **D9 noop** — advances the clock only; inert unless D7/D8 pending.
- **D10 color=health** — rendered consequence of D7/D8.

## 2. Episode map (dynamic -> scored target pairs)

Pairs are named `ep<k>: rowA->rowB` using the Step column of the slice (verbatim from the
source drive). POS = the dynamic visibly fires; NEG = contrastive near-miss.

| ep | src rows | scored pairs (action → change) | covers |
|---|---|---|---|
| 0 | A 3..9 | 3->4 noop NC **NEG: dg(4,2) DIAGONAL to gray(5,3), no infection**; 4->5 down (arrival creates adjacency); 5->6 noop **D7 POS: (5,3) gray→dg**; 6->7 noop NC **NEG: adjacency persists but already dg (monotone, no re-fire)**; 7->8 right **D5 MERGE (5,2)→(5,3), dg −1**; 8->9 right **D5 REVEAL, dg +1 at (5,4)** | D4, D5×2, D7 POS+2 NEG |
| 1 | B 0..5 | 0->1 right (dg clean); 1->2 down → (3,3), creates pending adjacency to gray(3,4); 2->3 **click 3 4 SUPPRESSION (recoverable)**; 3->4 noop **D8 POS: clicked cell (3,4), now the gray ACTIVE, turns dg**; 4->5 right (dg clean, the swapped-in particle moves) | D2×2, D4, D6, D8 POS |
| 2 | B 8..12 | 8->9 right → (5,6), creates DOUBLE pending (gray(5,7), gray(6,6)); 9->10 **click 6 6 SUPPRESSION of both (recoverable)**; 10->11 noop **D7 POS on (5,7) + D8 POS on new active (6,6) — two cells recolor in one tick**; 11->12 down (new dg active (6,6)→(7,6): reveals the swap) | D2, D4, D6, D7+D8 POS |
| 3 | C 0..5 | 0->1 **click 6 6 quiet (aliased at pair level)**; 1->2 down **gray (6,6) moves — reveals the swap in ctx_next of the click**; 2->3,3->4,4->5 left×3 (gray active transit (7,6)→(7,3)) | D1×3, D4, D6 |
| 4 | C 8..14 | 8->9 up **D5 MERGE gray-gray (6,3)→(5,3)**; 9->10 up **D5 REVEAL, gray +1 at (4,3)**; 10->11 up (clean, passes gray(3,4) — adjacency without any dg = nothing); 11->12 noop NC **NEG: gray ACTIVE (3,3) DIAGONAL to dg(2,2) + orthogonal to gray(3,4) — neither D8 nor D7 fires**; 12->13 up → (2,3) creates D8 adjacency; 13->14 noop **D8 POS via movement: gray active (2,3) turns dg** | D3×4, D5×2, D8 POS+NEG |
| 5 | D 1..5 | 1->2 right → (3,3), pending on gray(3,4); 2->3 **click 5 7 SUPPRESSION-AT-A-DISTANCE (recoverable): clicked particle is FAR from the pending one, recolor still suppressed**; 3->4 noop **D7 POS: (3,4) recolors — proves the pending fires when NOT clicked**; 4->5 left (gray (5,7)→(5,6): reveals which particle was selected) | D1, D2, D6, D7 POS |
| 6 | D 7..12 | 7->8 right **D5 MERGE gray-gray (6,5)→(6,6)**; 8->9 down **D5 REVEAL at (7,6)**; 9->10 noop NC **NEG: quiet-state certificate (nothing pending right before the click)**; 10->11 **click 6 6 quiet (aliased)**; 11->12 up **gray (6,6) moves up — reveals the swap (old active (7,6) stays)** | D2, D3, D4, D5×2, D6 |
| 7 | E 0..4 | 0->1 **click 3 4 quiet (aliased)**; 1->2 down **gray (3,4) moves — reveals swap**; 2->3 down → (5,4) builds gray chain next to gray(5,3); 3->4 noop NC **NEG: two GRAY particles orthogonally adjacent — adjacency alone never recolors, needs an unhealthy neighbor** | D4×2, D6 |
| 8 | E 8..11 | 8->9 **click 2 2 quiet (aliased) — clicking the DARKGREEN inactive**; 9->10 down **the dg (2,2) moves — reveals dg is now the active**; 10->11 down (dg transit) | D4×2, D6 |
| 9 | E 12..17 | 12->13 noop **D7 POS: (5,3) recolors (dg arrived at (5,2))**; 13->14 noop **D7 POS: (5,4) — chain tick 2, active did NOT move**; 14->15 noop **D7 POS: (5,5) — chain tick 3**; 15->16 noop NC **NEG: chain-end — gray(6,6) DIAGONAL to dg(5,5), spread stops**; 16->17 left (dg active (5,2)→(5,1)) | D1, D7 POS×3 + NEG |
| 10 | F 0..5 | 0->1 **click 5 3 quiet (aliased)**; 1->2 up **gray (5,3) moves — reveals swap**; 2->3 left; 3->4 up → (3,2) creates D8 adjacency from BELOW dg(2,2); 4->5 noop **D8 POS: (3,2) recolors** | D1, D3×2, D6, D8 POS |

### Per-dynamic scored-target counts

| dynamic | POS targets | NEG (near-miss) targets |
|---|---|---|
| D1 left | 6 (3 gray transit, 1 reveal-of-swap, 1 gray, 1 dg) | clean moves passing adjacent particles double as D5 negatives |
| D2 right | 7 (clean dg ×4, into-adjacency ×2, gray MERGE) | — |
| D3 up | 7 (gray MERGE, gray REVEAL, clean ×3, into-adjacency ×2) | — |
| D4 down | 9 (clean, arrival-into-adjacency, reveal-of-swap ×3, REVEAL, transit) | — |
| D5 overlap | 6 = 3 MERGE (right dg-dg, up gray-gray, right gray-gray) + 3 REVEAL (right, up, down) | every clean move (count conserved) is the standing contrast |
| D6 click-swap | 8 clicks: 3 suppression (recoverable), 5 quiet (aliased, each with the revealing move in ctx_next) | the 9 pending-state noop/move recolor pairs are the "no click → dynamic fires" contrast |
| D7 infection | 6 (single ×2, double 1 tick, chain ×3) | 4 (diagonal ×2, already-dg monotone, gray-gray adjacency) |
| D8 active-health | 4 (post-click ×2 incl. the double, via-movement ×2 from two approach directions) | 1 (gray active diagonal to dg) |
| D9 noop inert | — | 6 pure NO_CHANGE noops, at irregular spacing |

noop split: 15 = 9 POS (D7/D8 recolors) + 6 NEG (NO_CHANGE). Positives occur at irregular
offsets within and across slices (never a fixed cadence), and every recolor is explained
only by prev-state orthogonal adjacency — a step-parity clock or a "noop always/never
recolors" rule mislabels ≥6 pairs.

## 3. Click ID-recoverability (the honest accounting)

Click frames in dq8gc are ALWAYS NO_CHANGE (positions and colors are untouched by the
swap; the click handler also clobbers that tick's infection update). Under
keep_action_params=False the label is just `click`, so at the pair level:

**ID-RECOVERABLE clicks (3)** — clicked in a **pending-infection state**:
- ep1 2->3 `click 3 4` (pending: gray(3,4) orth-adjacent dg(3,3))
- ep2 9->10 `click 6 6` (double pending: gray(5,7) + gray(6,6) adjacent dg(5,6))
- ep5 2->3 `click 5 7` (pending: gray(3,4) adjacent dg(3,3); clicked particle far away)

In these states a noop or a move MUST show the pending recolor (D7 fires from prev-state
adjacency) and any move must show displacement. The observed frame shows neither →
`click` is the only verb consistent with x_t -> x_t+1. An oracle needs no knowledge of
the suppression mechanism — pure process of elimination over the whitelist. ep0 5->6,
ep5 3->4 and ep9 12->13..14->15 are the matched positives proving pendings DO fire on
noop; ep2/ep5's own next pair shows the suppressed recolor landing one tick late.

**ALIASED clicks (5)** — clicked in a **quiet state** (nothing pending, nothing moves):
- ep3 0->1 `click 6 6`, ep6 10->11 `click 6 6`, ep7 0->1 `click 3 4`,
  ep8 8->9 `click 2 2` (the darkgreen inactive), ep10 0->1 `click 5 3`
- At the pair level these are indistinguishable from an inert noop (both NO_CHANGE).
  Each is placed IMMEDIATELY before the swap-revealing move, so ctx_next (context_k=9)
  contains the proof a swap occurred (a different particle moves while the old active
  stays). Residual aliasing: an inert noop directly preceding such a click (e.g. ep6
  9->10) carries nearly the same window evidence one step earlier — with 5 quiet clicks
  vs 6 inert noops, a decoder that spots "NO_CHANGE + next-frames show a new mover" can
  narrow to {click, noop} but may confuse the exact step. This is the maximum visible
  attribution the game allows: dq8gc's click never changes the frame it acts on.

Not included (unobservable by construction, per req.4): clicks on empty cells or on the
active particle (no handler fires — a pure noop alias with zero window evidence).

## 4. How test50 differs from train/

- **All states are freshly driven** (6 new drives), none reuse the original clean_data2
  rollout that train/ was sliced from. Particle configurations unseen in train: the
  relocated gray at (7,3)/(6,3) column walk, the manually assembled 3-gray chain
  (5,3),(5,4),(5,5) infected end-to-end, gray actives walked INTO dg adjacency from two
  directions, dg active walked down column 2, double-pending infections, gray-gray
  merges/reveals (train's overlaps were dg-dg only).
- **Clicks**: train clicked (5,7) and (6,6) in mid-rollout states; test50 clicks 6 cells
  — (3,4)×2, (6,6)×2 (different states), (5,7), (5,3), (2,2) — including the pristine
  initial state, pending-infection states (never clicked in train), and the darkgreen
  inactive (train only ever clicked grays).
- **Suppression-of-pending-infection is exercised 3 times** (never a scored target in
  train, where both clicks were quiet-state).
- **Chain propagation without agent motion** (ep9: 3 consecutive recolor ticks while the
  active stands still) — train's infections were all single-cell, adjacent to the mover.
- **Irregular timing**: noop gaps of 0–3 between actions, positives/negatives interleaved
  (the original test/ used a rigid every-4th-step cadence; train had its own cadence).

## 5. Known limits / uncoverable items

- **Quiet-state click vs inert-noop aliasing** (5 pairs) — inherent to the game: the swap
  is invisible until the next move. Mitigated via ctx_next evidence; documented above.
- **Gray active -> dg via click-swap onto a pending cell** is covered, but a gray active
  that STARTS gray (pre-infection) never exists at seed 0 t=0 except via swap — all D8
  positives here necessarily post-date a click; that is the only way the game can produce
  a gray active.
- **No win/termination/reward** (game defines none) — nothing to cover.
- **Same-cell "adjacency"** (active overlapping an inactive) was deliberately NOT relied
  on for any recolor target: the SEXP's `adj ... 1` semantics for distance-0 are untested
  and would make the pair unexplainable; all recolor targets use clean distance-1
  orthogonal adjacency.
