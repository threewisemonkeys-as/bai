# qfsvc — test50 held-out pool (52 scored transitions)

Config: whitelist = `left,right,up,down,noop,click`; `keep_action_params=FALSE`
(`click R C` collapses to the verb `click` — click location is irrelevant in this game).
Verified with `clean_data3_tools.verify_pool(<...>/test50, 'left,right,up,down,noop,click', context_k=9)`:
**52 scored target transitions** (50 ± 2), every episode a contiguous verbatim slice of a
freshly driven trajectory (`autumn_drive.py QFSVC`, seed 0).

## Source trajectories (all freshly driven, distinct from train)

The train slices come from the ORIGINAL rollout, which had a rigid 1-action/3-noop cadence,
collected coins (4,6)/(4,4)/(2,6) via **left-onto** and **up-onto** approaches, and fired from
low/central positions. test50 uses three NEW drives with irregular action timing:

- **Drive A** (38 tr): left corridor; collects (4,6) **right-onto** and (2,6) **down-onto**;
  fires from (4,6) and (5,6); bullet flight passes OVER coin (2,6) (gold flicker); ammo-0
  clicks on quiet frames.
- **Drive B** (30 tr): right-mid coins never touched by train; collects (4,10) **left-onto**
  and (2,10) **up-onto**; fires twice from (2,10) with the second click while the first
  bullet is still in flight (two bullets airborne, staggered despawns).
- **Drive C** (37 tr): far-right coins never touched by train; collects (4,12) **right-onto**
  with the delayed effect landing on a MOVE step (agent walks off the coin as it vanishes),
  and (2,12) **down-onto**; fires from (2,12) (short flight) and (5,11) (long flight with
  movement during flight); ends ammo-exhausted (collected 2, fired 2, click -> NO_CHANGE).

Overlap with train situations: coins (4,6)/(2,6) are also collected in the original train
trajectory, but from DIFFERENT directions along different paths; all other collected coins,
all spawn positions ((4,6),(5,6),(2,10) x2,(2,12),(5,11)), and all action sequences are new.

## Core dynamics (from dynamics.txt / COVERAGE.md numbering)

D1 left | D2 right | D3 up | D4 down | D5 click spawns bullet at agent cell iff ammo>0 |
D6 click with ammo==0 does nothing | D7 bullets move up 1/step, EVERY step |
D8 bullet despawns off the top edge | D9 move-onto-coin -> ammo+1, coin removed with a
one-step rendering delay (red hidden under coin on the cause tick) | D10 noop with nothing
active -> NO_CHANGE.

## Episodes -> target pairs (drive step s->s+1; POS = dynamic fires, NEG = near-miss)

| ep | drive steps | targets (action -> change tag) | dynamics |
|---|---|---|---|
| 0 | A 0..4 | noop NC (NEG) ; up ; left ; left | D10; D3; D1 x2 (clean moves) |
| 1 | A 9..16 | right black+1 red-1 (POS cause) ; noop gold-1 red+1 (POS eff) ; noop NC (NEG) ; click purple+1 red-1 (POS) ; noop fly ; noop fly gold-1 ; noop fly gold+1 | D9 right-onto (4,6); D10; D5 spawn @(4,6); D7 x3 — incl. bullet passing OVER coin (2,6): gold hidden then restored (rendering flicker, FD must not read it as coin destruction) |
| 2 | A 17..20 | noop purple-1 (POS) ; click NC (NEG) ; right | D8 despawn col 6; D6 ammo-0 quiet frame; D2 |
| 3 | A 25..31 | down black+1 red-1 (POS cause) ; noop gold-1 red+1 (POS eff) ; down x3 ; click purple+1 red-1 (POS) | D9 down-onto (2,6); D4 x3 (one re-crosses the former coin cell (4,6) — nothing happens, D9 NEG); D5 spawn @(5,6) |
| 4 | B 9..16 | right ; up (NEG: lands (4,11) BETWEEN coins, no collect) ; left cause (POS) ; noop eff (POS) ; up ; up cause (POS) ; noop eff (POS) | D2; D9 near-miss; D9 left-onto (4,10); D3; D9 up-onto (2,10) |
| 5 | B 18..25 | noop fly ; click purple+1 red-1 (POS, spawn WHILE bullet 1 in flight) ; noop purple-1 red+1 (POS desp + bullet-2 fly) ; noop fly ; noop purple-1 (POS desp) ; noop NC (NEG) ; click NC (NEG) | D7; D5 concurrent spawn @(2,10) — also shows D7 firing on a CLICK step; D8 x2 staggered; D10; D6 |
| 6 | C 8..13 | up ; up (NEG near-miss (4,11)) ; noop NC (NEG: agent ADJACENT to two coins, noop collects nothing) ; right cause (POS) ; right gold-1 red+1 moved (POS eff-ON-MOVE) | D3 x2; D10/D9 near-miss; D9 right-onto (4,12) with the delayed effect landing on a MOVE action — collection is not noop-bound |
| 7 | C 18..24 | down cause (POS) ; noop eff (POS) ; click purple+1 red-1 (POS) ; noop fly ; noop fly ; noop purple-1 (POS) | D9 down-onto (2,12); D5 spawn @(2,12); D7 x2; D8 despawn col 12 |
| 8 | C 30..37 | right red+purple both move (POS comp) ; noop fly ; noop fly ; left red+purple both move (POS comp) ; noop purple-1 (POS) ; click NC (NEG) ; noop NC (NEG) | D2/D1 WITH bullet in flight — D7 fires on movement steps too; D7 x2; D8 despawn col 11; D6 ammo-exhausted (collected 2, fired 2); D10 |

## Per-dynamic coverage (scored targets: positives / negatives)

| dynamic | positives | negatives (near-miss) |
|---|---|---|
| D1 left | 4 (2 clean, 1 D9-cause, 1 with bullet in flight) | — |
| D2 right | 6 (2 clean, 2 D9-cause, 1 D9-eff-on-move, 1 with bullet) | — |
| D3 up | 6 (4 clean, 1 D9-cause) | 2 of them land adjacent to coins w/o collecting (D9 near-miss) |
| D4 down | 5 (2 D9-cause, 3 clean) | 1 re-crosses a collected coin cell (D9 near-miss) |
| D5 click spawn | 4, at 4 distinct agent cells: (4,6),(5,6),(2,10) concurrent,(2,12) | D6 below is the negative |
| D6 click ammo-0 | — | 3 NO_CHANGE clicks (ep2 after despawn, ep5 quiet, ep8 ammo-exhausted) |
| D7 bullet-up | 9 noop-fly targets over 4 columns (6,10,11,12), incl. the over/off-coin flicker pair; ALSO evidenced on a click step (ep5) and two move steps (ep8) -> "bullets move only on noop" scores worse | 5 NO_CHANGE noops (D10) with no bullet present |
| D8 despawn | 5 (cols 6,10 x2,11,12; one compound with a second bullet still flying) | NO_CHANGE noops as above |
| D9 collection | 6 causes (right-onto x2, down-onto x2, left-onto, up-onto) + 6 effects (5 on noop, 1 on a move) | 2 moves landing between coins; 1 noop adjacent to coins; 1 move across a former coin cell; NO_CHANGE noops |
| D10 noop NC | n/a (it IS the negative) | 5 NO_CHANGE noops, irregular timing (never a fixed cadence) |

Contrastive-negative share: 5 NC noops + 3 ammo-0 clicks + 4 near-miss moves ≈ 12/52 ≈ 23%.

## Action histogram of the pool (collapsed verbs)

`noop: 24, click: 7, up: 6, right: 6, down: 5, left: 4` = 52.

noop is deliberately the largest bucket because ALL passive dynamics ride on it — its 24 are
5 NC negatives (D10), 5 collect-effects (D9), 9 bullet-fly (D7) and 5 despawn/desp+fly (D8);
the sixth collect-effect is scored under `right` (ep6 effect-on-move). Movement verbs are
4-6 each; click is 7 (4 spawns + 3 ammo-0 negatives).

## Known aliasing / limitations (unavoidable, minimized)

1. **Ammo-0 click vs quiet noop**: a `click` with ammo==0 and a `noop` with nothing active
   both produce NO_CHANGE — the frames alone cannot separate them (true in the game itself).
   3 click-NEG + 5 noop-NEG targets share this: an oracle gets the other 44/52 exactly and
   can at best guess within this NC set (ID ceiling ~= 48/52 ≈ 0.92 if it splits NC 50/50;
   FD is NOT ambiguous — predicting NO_CHANGE is correct for both).
2. **Collect-cause direction**: on the cause tick the agent hides under the coin, so the verb
   is recoverable from the vacated cell + which adjacent coin was entered; where two coins
   flank the start cell the ctx window (the following effect tick reveals the agent) is
   needed. All 6 causes keep cause+effect inside the same slice.
3. The ammo counter is internal/unrendered — D5 vs D6 conditioning is only visible through
   history (collect/fire events in the ctx window). All click targets keep the relevant
   evidence within their episode's context_k=9 window.

Nothing else in dynamics.txt is uncovered: no win/termination/color-change rules exist.
