# qqm74 — test50 (large held-out test pool)

Config: whitelist = `left,right,up,down,noop,click`; keep_action_params = **FALSE**
(movement game — `click` collapses to the bare verb; the ID label is the verb, not the
click location). Pool: **50 scored target transitions** across 7 curated episodes, each a
verbatim contiguous slice of a FRESHLY DRIVEN trajectory (`autumn_drive.py QQM74`, seed 0).
Verified with `T.verify_pool(..., context_k=9)` → 50 targets.

This is the large held-out **TEST** set (additive; `train/`, `test/`, `dynamics.txt`,
`COVERAGE.md` untouched). Same rules as train, deliberately DIFFERENT situations.

## The game in one line
Global velocity (xVel=col, yVel=row), each clamped to {-1,0,+1}, nudged ±1 by
left/right/up/down. Every step ALL blobs move by the **previous tick's** velocity
(one-step delay; inertia persists). `click` on a FREE cell spawns a blue blob (also moved
on its spawn tick → renders at `(row+yVel, col+xVel)`); click on an OCCUPIED cell does
nothing. `noop` has no handler. No win/score/termination.

## CORE dynamics (from dynamics.txt / train COVERAGE)
1. **Stationary @ vel0** — noop leaves the grid unchanged when xVel=yVel=0.
2. **Inertial drift (passive)** — every step each blob moves by (prev yVel, prev xVel); +col / -col / +row / -row / diagonal.
3. **One-step delay** — a movement key changes velocity but motion appears the FOLLOWING tick; the key's own tick moves by the OLD velocity.
4. **Key→sign** — right=+col, left=-col, down=+row, up=-row (seen in the drift a tick later).
5. **Velocity clamp ±1** — same key at the limit adds nothing; drift stays unit, never doubles.
6. **Deceleration / reversal (delayed)** — opposite key reduces/reverses velocity, again one-tick delayed (key tick still moves by old velocity, then it stops/reverses).
7. **Click FREE @ vel0** — spawns a blob exactly at the clicked cell.
8. **Click FREE @ vel≠0** — spawns a blob offset by velocity: `(clickRow+yVel, clickCol+xVel)`.
9. **Click OCCUPIED** — no spawn, no change (the discriminating negative).
10. **Off-grid drift / despawn-from-view** — velocity carries a blob past an edge; it stops rendering, blue count drops (no wrap/clamp).
11. **noop = no handler** — never changes velocity (covered by the #1/#2 contrast).
12. **Win/termination/reward** — none defined (not a scorable transition).

## Episodes → target pairs (each internal consecutive pair is a scored target)

Each episode is a verbatim contiguous slice of one driven trajectory. Clicks below are
`click ROW COL`; "→(r,c)" is where the spawned blob actually renders.

| Ep (src) | pair | action | change | dynamic(s) |
|---|---|---|---|---|
| **0** vel-0 clicks | 0→1 | noop | NO_CHANGE | #1 stationary |
| | 1→2 | click 5 5 | blob→(5,5) | #7 exact spawn (ID+FD) |
| | 2→3 | click 5 5 | NO_CHANGE | **#9 occupied (neg)** |
| | 3→4 | noop | NO_CHANGE | #1 stationary |
| | 4→5 | click 15 15 | blob→(15,15) | #7 exact spawn |
| | 5→6 | click 17 12 | blob→(17,12) | #7 exact spawn |
| | 6→7 | click 17 12 | NO_CHANGE | **#9 occupied (neg)** |
| | 7→8 | noop | NO_CHANGE | #1 stationary |
| | 8→9 | click 8 3 | blob→(8,3) | #7 exact spawn |
| | 9→10 | noop | NO_CHANGE | #1 stationary |
| **1** x-axis | 0→1 | right | NO_CHANGE | **#3 delay (neg)** |
| | 1→2 | right | +col | #5 clamp, #4 right→+col |
| | 2→3 | right | +col | #5 clamp |
| | 3→4 | left | +col | #6 decel (moves by OLD +1) |
| | 4→5 | left | NO_CHANGE | **#6 reversal / #3 delay (neg)** |
| | 5→6 | left | -col | #5 clamp, #4 left→-col |
| | 6→7 | noop | -col | #2 drift |
| | 7→8 | noop | -col | #2 drift |
| **2** y-axis | 0→1 | down | NO_CHANGE | **#3 delay (neg)** |
| | 1→2 | down | +row | #5 clamp, #4 down→+row |
| | 2→3 | down | +row | #5 clamp |
| | 3→4 | up | +row | #6 decel (moves by OLD +1) |
| | 4→5 | up | NO_CHANGE | **#6 reversal / #3 delay (neg)** |
| | 5→6 | up | -row | #5 clamp, #4 up→-row |
| | 6→7 | noop | -row | #2 drift |
| **3** diagonal | 0→1 | right | NO_CHANGE | **#3 delay (neg)** |
| | 1→2 | down | +col | #3 inertia (moves +col by prev xVel; down effect delayed) |
| | 2→3 | noop | diag (1,1) | #2 diagonal drift |
| | 3→4 | click 4 4 | blob→(5,5) | **#8 OFFSET spawn** (ID+FD) |
| | 4→5 | noop | diag | #2 diagonal drift |
| | 5→6 | click 12 5 | blob→(13,6) | **#8 OFFSET spawn** |
| | 6→7 | noop | diag | #2 diagonal drift |
| **4** right-edge | 0→1 | right | NO_CHANGE | #3 delay |
| | 1→2 | click 10 18 | blob→(10,19) | #8 offset spawn |
| | 2→3 | click 6 18 | blob→(6,19) | #8 offset spawn |
| | 3→4 | noop | blue 3→2 | **#10 off-grid despawn (right)** |
| | 4→5 | noop | blue 2→1 | **#10 off-grid despawn (right)** |
| | 5→6 | noop | +col | #2 drift |
| **5** top-edge | 0→1 | up | NO_CHANGE | #3 delay |
| | 1→2 | click 2 9 | blob→(1,9) | #8 offset spawn |
| | 2→3 | noop | -row | #2 drift |
| | 3→4 | noop | blue 2→1 | **#10 off-grid despawn (top)** |
| | 4→5 | noop | -row | #2 drift |
| **6** left-edge | 0→1 | click 10 2 | blob→(10,2) | #7 exact spawn @vel0 |
| | 1→2 | click 10 2 | NO_CHANGE | **#9 occupied (neg)** |
| | 2→3 | click 10 2 | NO_CHANGE | **#9 occupied (neg)** |
| | 3→4 | left | NO_CHANGE | **#3 delay (neg)** |
| | 4→5 | noop | -col | #2 drift |
| | 5→6 | noop | -col | #2 drift |
| | 6→7 | noop | blue 2→1 | **#10 off-grid despawn (left)** |

## Per-dynamic coverage counts (scored targets)

| # | Dynamic | positives | negatives / near-miss | total |
|---|---------|-----------|-----------------------|-------|
| 1 | Stationary @vel0 | — | 4 noop NC (defeat "noop drifts") | **4** |
| 2 | Inertial drift | 11 (+col,-col,+row,-row,diagonal) | — | **11** |
| 3 | One-step delay | 1 (inertia, ep3 1→2) | 7 movement-key NC ticks (defeat "key moves now") | **8** |
| 4 | Key→sign | ≥6 (right/left/down/up + diagonal) | — | **≥6** |
| 5 | Velocity clamp ±1 | 6 (right×2,left×1,down×2,up×1) | — | **6** |
| 6 | Decel / reversal | 2 decel + 2 reversal | (reversal ticks double as delay-NC negs) | **4** |
| 7 | Click FREE @vel0 | 5 exact spawns | — | **5** |
| 8 | Click FREE @vel≠0 | 5 offset spawns | — | **5** |
| 9 | Click OCCUPIED | — | 4 NC clicks (defeat "click always spawns") | **4** |
| 10 | Off-grid despawn | 4 (right×2, top, left) | — | **4** |
| 11 | noop no-handler | — | covered by #1/#2 contrast | — |
| 12 | Win/term/reward | none exists | — | n/a |

Every core dynamic #1–#10 is a scored target **≥4 times** in varied situations.

## Action histogram (verb-collapsed, keep_action_params=FALSE)

`noop:19  click:14  right:5  left:4  down:4  up:4`  = **50**

noop 38%, click 28%, movement (r/l/u/d) 34% — every whitelist verb ≥4. noop is the
plurality because inertial drift is **passive** (fires on noop ticks) and click is well
represented because it is the **only** ID-recoverable verb (see aliasing note). This is a
markedly flatter distribution than the train pool (noop 53%, movement each ≤2).

## Contrastive negatives (~32% of the pool are NO_CHANGE, all intentional)

16 of 50 pairs are NO_CHANGE and each defeats a shortcut:
- **4 stationary noops** (ep0) vs the 11 drift-noops → "noop always drifts" scores worse.
- **8 movement-key delay ticks** (ep1 0→1,4→5; ep2 0→1,4→5; ep3 0→1; ep4 0→1; ep5 0→1;
  ep6 3→4) → "movement key moves the blob immediately" scores worse; the motion lands the
  NEXT tick.
- **4 occupied clicks** (ep0 2→3,6→7; ep6 1→2,2→3) vs 10 spawning clicks → "click always
  spawns" scores worse.
Also: drift directions vary (+col,-col,+row,-row,diagonal) so "always +col" fails; clamp
targets (drift stays unit after repeated same-key presses) defeat "velocity accumulates";
offset spawns (#8) vs exact spawns (#7) defeat "click spawns at the clicked cell". Action
timing is irregular (no fixed every-k cadence) so a `step % k` clock is punished.

## How this test differs from train (cross-trajectory generalization)
- **All-new drives, seed 0, new click locations.** Train clicked (3,15) and (2,2); test
  clicks (5,5),(15,15),(17,12),(8,3),(4,4),(12,5),(10,18),(6,18),(2,9),(10,2) — disjoint.
- **Denser mechanics coverage.** Train showed each movement dynamic once at a fixed spot;
  test uses clamp RUNS on all four axes, decel+reversal chains on both axes, a diagonal
  (1,1) drift carrying TWO offset spawns, and off-grid despawns on **three** edges
  (right, top, left) vs train's single right-edge exit.
- **50 targets, ≥4 per dynamic** (train had 19, most dynamics 1–2×). 4 occupied-click
  negatives and 4 off-grid despawns vs train's 1 each.
- Same objects/rules, different positions, configurations and timing throughout.

## Aliasing / uncoverable (documented, minimized where inherent)
- **ID is structurally weak on movement verbs (unavoidable).** Because every velocity
  update is delayed one tick, each movement key's OWN transition moves the blob by the
  *previous* velocity — so a delay tick (key from vel0) is NO_CHANGE, indistinguishable
  from noop and from the other three movement keys; a clamp tick shows displacement in the
  key's direction but is identical to a noop at that same standing velocity. An oracle
  therefore **cannot uniquely recover left/right/up/down vs noop** from the immediate pair.
  Only **click** is cleanly ID-recoverable (a blob appears). This is the same headline gap
  documented for the train set. Mitigations applied: (a) every movement verb has clamp
  targets showing the correct-direction displacement; (b) the causing key sits in the
  window of every drift pair (delayed-effect recipe) so the movement dynamics score cleanly
  under **FD** (predict the drift from the velocity carried in context).
- **Off-grid re-entry** (a hidden off-grid blob drifting back into view) is deliberately
  EXCLUDED — it is unpredictable under FD without memory of invisible state. Only the clean
  despawn-from-view (exit) event is scored (#10). Same choice as train.
- **Button** object is defined but never instantiated → no target. **Win/termination/reward
  (#12)** does not exist in the program → nothing to score.

## Verify
```
uv run python -c "import sys; sys.path.insert(0,'prototypes/perc_invdyn'); import clean_data3_tools as T; T.verify_pool('prototypes/perc_invdyn/clean_data3/qqm74/test50','left,right,up,down,noop,click',context_k=9)"
```
