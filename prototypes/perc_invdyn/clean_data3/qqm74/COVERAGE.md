# qqm74 — clean_data3 coverage

Config: whitelist = `left,right,up,down,noop,click`; keep_action_params = FALSE
(movement game — `click` collapsed to the verb; click LOCATION is not the label).
Pool: 19 scored target transitions across 10 curated episodes (slices of the original
train trajectory, copied verbatim). Verified with `T.verify_pool`.

## The game in one line
A global velocity (xVel=col, yVel=row), each clamped to {-1,0,+1}, is nudged ±1 by
left/right/up/down. Every step ALL blobs move by the **previous tick's** velocity
(one-step delay; inertia persists). `click` on a FREE cell spawns a blue blob, which is
also moved on its spawn tick (renders at (row+yVel, col+xVel)); click on an OCCUPIED cell
does nothing. `noop` has no handler. No win/score/termination.

## CORE dynamics extracted from dynamics.txt
1. **Stationary at vel 0** — with xVel=yVel=0, noop leaves the grid unchanged.
2. **Inertial drift (passive)** — every step each blob moves by (prev yVel, prev xVel);
   covers +col / -col / +row / -row / diagonal.
3. **One-step delay** — a movement key changes velocity, but motion only appears the
   FOLLOWING tick; on the key's own tick the blob moves by the OLD velocity.
4. **Key→sign mapping** — right=+col, left=-col, down=+row, up=-row (seen in the drift the
   key produces a tick later).
5. **Velocity clamp at ±1** — pressing the same key while already at the limit adds nothing
   (drift stays unit, never doubles).
6. **Deceleration / reversal (delayed)** — opposite key reduces/reverses velocity, again
   with the one-step delay (key tick still moves by old velocity, then it stops/reverses).
7. **Click FREE @ vel 0** — spawns a blob exactly at the clicked cell.
8. **Click FREE @ vel≠0** — spawns a blob offset by velocity: (clickRow+yVel, clickCol+xVel).
9. **Click OCCUPIED** — no spawn, no change.
10. **Off-grid drift / despawn-from-view** — velocity carries a blob past an edge; it stops
    rendering and the blue count drops (no wraparound/clamp). (Persistence + re-entry of an
    off-grid blob also occurs but requires hidden state — see GAPS; kept out of the pool.)
11. **noop = no handler** — never changes velocity; only inertia acts.
12. **Win/termination/reward** — none defined (not a transition dynamic; nothing to score).

## Dynamic × objective table (and the GAP in the original train pool)

| # | Dynamic | TARGET under ID? | TARGET under FD? | Gap in ORIGINAL pool |
|---|---------|------------------|------------------|----------------------|
| 1 | Stationary @ vel0 | weak (noop, no change — only as "nothing happened") | yes (predict no motion) | present but only as filler noops |
| 2 | Inertial drift | NO — change happens on a **noop** step (labeled noop, not the key) | **yes** — must predict displacement from velocity in context | present, but the drift is the workhorse and was mixed with re-entry confounds |
| 3 | One-step delay | NO — key tick shows no change → confusable with noop | yes (predict that nothing moves yet, or moves by old vel) | present but never isolated; cause/effect split across steps |
| 4 | Key→sign | NO (delay, see #3) | yes (drift direction reveals it) | present |
| 5 | Velocity clamp ±1 | NO (key tick = ongoing drift, looks like noop) | yes (predict +1 not +2 when already at limit) | present but never paired with a non-clamp accel to make it discriminative |
| 6 | Decel/reversal | NO (delay) | yes (predict residual motion then stop) | present, buried |
| 7 | Click free @ vel0 | **yes** — new blob appears at click loc | **yes** — predict the new blob exactly at click | present (steps 3,87) |
| 8 | Click free @ vel≠0 | **yes** — new blob appears | **yes** — predict it OFFSET by velocity | present once (step 67); easily missed by a balanced-20 sample |
| 9 | Click occupied | partial — no change (the discriminating negative) | yes (predict NO spawn) | present once (step 7); would likely be dropped by sampling, losing the contrast |
| 10 | Off-grid despawn | NO (occurs on noop) | yes (edge blob + velocity ⇒ leaves view, count drops) | present but tangled with re-entry of a hidden blob |
| 11 | noop no-handler | — | covered by #1/#2 contrast | — |
| 12 | Win/term/reward | n/a | n/a | none exists |

**Headline gap (nrdf6-style): the ID objective is structurally weak on this game.**
Because every velocity update is delayed by one tick, the movement keys
(left/right/up/down) produce NO change on their own transition — they are indistinguishable
from `noop` (and from each other) in the immediate `X_t -> X_t+1` pair. The visible motion
lands on the SUBSEQUENT step, which is labeled `noop`. So under ID only **click** (a blob
appears) is cleanly recoverable; the movement verbs are unidentifiable from the scored pair.
The movement dynamics are therefore exercised primarily under **FD** (predicting the drift
requires the velocity, read from the window context that carries the causing key), with the
movement-key targets kept so the cause sits in the window of the drift pairs (the
delayed-effects recipe) and so a `noop=no-change` shortcut is punished.

Secondary gap: in the original pool the off-grid blob re-enters the visible grid at
step 26→27 purely from hidden off-grid state — an **unpredictable** target (FD cannot
recover it without memory of where the invisible blob is). I deliberately kept this out of
the scored pool and used the clean **despawn-from-view** (exit) event instead.

## Curated slices (episodes) → targets and the dynamic each covers

Each episode is a verbatim contiguous slice of the original train trajectory; internal
consecutive pairs are the scored targets, windows are real frames bounded by the slice.

| Ep | Steps | Target pair(s) | Action | Dynamic(s) covered |
|----|-------|----------------|--------|--------------------|
| 0 | 2,3,4 | 2→3 | noop | **#1 stationary @vel0** (negative for "noop moves") |
|   |       | 3→4 | click(3,15) | **#7 click free @vel0 → exact spawn** |
| 1 | 7,8 | 7→8 | click(3,15) | **#9 click OCCUPIED → no change** (contrastive neg vs #7/#8) |
| 2 | 11,12,13 | 11→12 | right | **#3 one-step delay** (right issued, NO motion this tick) |
|   |          | 12→13 | noop | **#2 +col drift**, **#4 right→+col** |
| 3 | 14,15,16 | 14→15 | noop | **#2 +col drift** (establishes "already +1" in window) |
|   |          | 15→16 | right | **#5 CLAMP** (right at xVel=1 ⇒ still +1, NOT +2) — contrast vs ep2 11→12 |
| 4 | 23,24,25 | 23→24 | left | **#3 delay** (left issued, NO motion) |
|   |          | 24→25 | noop | **#2 -col drift**, **#4 left→-col** |
| 5 | 35,36,37 | 35→36 | down | **#3 delay** (down issued, NO motion) |
|   |          | 36→37 | noop | **#2 +row drift**, **#4 down→+row** |
| 6 | 43,44,45 | 43→44 | up | **#6 decel/reversal** (up vs prev yVel=1 ⇒ residual +row this tick, delayed) |
|   |          | 44→45 | noop | **#6 result**: now stops (vel→0) — proves motion follows PREV velocity |
| 7 | 47,48,49 | 47→48 | up | **#3 delay** (up issued, NO motion) |
|   |          | 48→49 | noop | **#2 -row drift**, **#4 up→-row** |
| 8 | 16,17,18 | 16→17 | noop | **#2 +col drift** (window establishes xVel=+1, two blobs) |
|   |          | 17→18 | noop | **#10 off-grid despawn** (right-edge blob exits view; blue count 2→1) |
| 9 | 66,67,68 | 66→67 | noop | **#2 DIAGONAL drift** (vel=(1,1)) |
|   |          | 67→68 | click(2,2) | **#8 click free @vel≠0 → OFFSET spawn** at (3,3)=(2+yVel,2+xVel) — contrast vs ep0 exact spawn |

## Contrastive negatives (shortcut defeats)
- **"noop = no change"** → noops drift (eps 2,3,4,5,7,8,9) AND noops don't (eps 0, 6 44→45).
  The discriminator is velocity (window context), not the action verb.
- **"blob always moves +col every step"** → drift directions vary: +col, -col, +row, -row,
  diagonal, and stationary segments.
- **"click always spawns at the clicked cell"** → ep1 click on occupied = NO_CHANGE; ep9
  click spawns OFFSET (not at the cell) under nonzero velocity, vs ep0 exact at vel0.
- **"movement key causes immediate motion"** → eps 2,4,5,7 show the key's own tick as
  NO motion (delay); ep3 shows the key during ongoing drift adding nothing (clamp); ep6
  shows the key while motion still runs on old velocity then stops.
- **step-counter / clock shortcut** → drift is governed by action-set velocity at varied
  cadence; stationary and moving noops appear at unrelated step numbers.

## Final pool composition (verified)
- **19 scored target transitions**, 10 episodes.
- By action verb: noop ×10, click ×3 (1 spawn-exact, 1 occupied-negative, 1 spawn-offset),
  right ×2 (delay + clamp), left ×1 (delay), down ×1 (delay), up ×2 (delay + decel).
- Every core dynamic #1–#10 appears as a scored target; #2 (drift) and #8/#9 (click
  variants) appear with explicit contrastive partners. ID is informative for click;
  FD is informative for every movement/drift/clamp/decel/off-grid target.
- Pool size ≤ `--train-n 20`, so a balanced sample keeps all 19.

Verify:
`T.verify_pool('prototypes/perc_invdyn/clean_data3/qqm74/train','left,right,up,down,noop,click')`
