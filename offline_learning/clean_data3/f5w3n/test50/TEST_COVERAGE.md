# f5w3n — TEST50 held-out pool (52 scored transitions)

Config: whitelist `left,right,up,noop` (`down`/`click` are no-ops in this game and were never
driven), `keep_action_params=FALSE` (verb-only labels), `context_k=9`. Coordinates are
`(row, col)`; hero row is 15. Verified with
`T.verify_pool('prototypes/perc_invdyn/clean_data3/f5w3n/test50','left,right,up,noop')`:
**52 scored target transitions** — `right:6, left:6, up:6, noop:34`.

## Provenance — freshly driven, distinct from train

Five NEW trajectories were driven with `autumn_drive.py F5W3N` (seed 0), each a theme:

| drive | actions (frame indices) | theme |
|---|---|---|
| A (36 rows) | right×5, noop×4, up@9, left×2, noop×2, up@14, noop×15, right@30, noop, left, noop×2 | right-side double kill: enemies2 at **(3,13)** (coloc f22) and **(3,11)** (coloc f28) |
| B (67 rows) | left×7, right, left, noop×11, up@20, noop×8, right×2, up@31, left×2, noop×30, up@64, noop | cancel at **(9,1)**, miss-shot up col3 (clean red off-grid), death at **(15,1)** f63 |
| C2 (54 rows) | left×6, noop×2, up@8, right, noop×25, right×2, noop×12, up@49, noop, up@51, noop | kill leftmost enemies1 at **(1,2)** (coloc f23) -> t=33 spawn SHIFTS to (1,5) -> death at **(15,5)** f48; dead-ups f49/f51 |
| D (48 rows) | left×3, noop, up@4, noop×5, right×6, noop×12, up@28, left×8, noop, right, noop×8 | kills at **(3,5)** (coloc f17) and **(1,11)** (coloc f43); no death |
| E (50 rows) | noop, right×2, noop, left×2, noop×3, left×2, noop, left×2, noop×6, left, noop×7, left, noop×14, up@43, noop×5 | hero walks under the t=33 col-2 orange and SAVES HIMSELF: cancel at **(13,2)** f46 |

Differences from `train/` (which is sliced from `train_regen`, all left-side col-2 action):
- **Kills**: train has ONE kill, enemies2 at (3,2). Test50 has FOUR, at (3,13), (3,5),
  (1,11), (1,2) — both enemy rows, left/center/right columns (the C-drive kill that landed
  on (3,2) was deliberately NOT sliced to avoid replicating the train cell).
- **Cancels**: train (10,2) f13; test50 (9,1) and (13,2) — different columns/rows/times.
- **Deaths**: train (15,2) f48; test50 (15,1) f63 and (15,5) f48 — the (15,5) death only
  exists because a prior kill shifted the enemy-fire draw (rule composition train never shows).
- **Fires**: train fires from (15,8)/(15,2); test50 scored fires from (15,11), (15,1),
  (15,3), (15,2) — the (15,2) one (ep14) is the same hero cell as train's but at t=43 with a
  different board (orange at (10,2), all 10 enemies, cancel at (13,2) not (10,2)).
- **Hero range**: train hero stays cols 2-3; test50 exercises cols 1-13.
- **Action cadence** is irregular in every drive (no fixed every-k-steps rhythm).

## Engine facts confirmed while driving (READ from the ASCII frames)

- March transitions fire out of frames with `time%10==0` (e1 RIGHT/e2 LEFT) and `%10==5`
  (e1 LEFT/e2 RIGHT); static the other 8/10 steps.
- Enemy fire out of `time%15==3` frames (3->4, 18->19, 33->34, 48->49); with seed 0 the draw
  is deterministic and always lands on the **leftmost remaining enemies1** enemy — so killing
  that enemy MOVES later spawn columns ((1,2)/(1,1) baseline; (1,5)/(1,4) after C2's kill).
- Orange travel row1->row15 takes 14 steps and the fire period is 15, so an orange's
  off-grid despawn is ALWAYS on the same transition as the next spawn
  (`orange~move(-14,*)` tag, bottom-vanish + top-appear, spatially far apart). The only
  clean off-grid despawn is a missed RED (ep6 47->48).
- Collisions are 1-step prev-delayed: co-locate frame N (victim occluded), removal N->N+1.
- On a removal step the collision rule reassigns `bullets = removeObj(prev bullets)`, so any
  OTHER red in flight is **frozen for that tick** (visible in ep2 22->23: the second red
  stays at (8,11)).
- Bullets pass through enemies without interacting unless the kill rule fires (occlusion
  only, e.g. orange crossing row 3).

## Core dynamics -> scored-target coverage

Episodes below; "pair" = Step_i -> Step_i+1 inside the episode. P=positive, N=near-miss/negative.

| # | dynamic | positives (scored pairs) | negatives / near-misses (scored pairs) |
|---|---------|--------------------------|----------------------------------------|
| 1 | left (x-1) | ep1 11->12; ep5 32->33, 33->34; ep12 30->31(+march), 31->32, 32->33 — 6, cols 12->11, 3->2->1, 10->7 | every hero-static noop |
| 2 | right (x+1) | ep0 2->3, 3->4(+spawn), 4->5; ep9 35->36(+march); ep11 14->15, 15->16(+march) — 6, cols 10->13, 4->5, 9->11 | every hero-static noop |
| 3 | up = fire (alive only) | LIVE (red+1 at hero cell): ep1 14->15 @(15,11); ep3 20->21 @(15,1); ep5 31->32 @(15,3) (PURE: only gray-1 red+1); ep14 43->44 @(15,2) — 4 | DEAD (no red+1): ep7 64->65, ep10 49->50 — the alive-guard contrast |
| 4 | noop / clock | 34 noops, each carrying a specific passive/collision dynamic (see below) | — |
| 5 | hero bullet up (y-1 each step) | ep1 12->13, 13->14; ep8 21->22; ep13 41->42; ep14 44->45; ep6 45->46, 46->47 | ep4 28->29 & post-removal frames: NO red motion once despawned |
| 6 | enemy bullet down (y+1 each step) | ep3 19->20; ep9 34->35; ep7 61->62; + present in ~25 pairs | ep4 28->29 NO_CHANGE (no orange in flight) |
| 7 | enemy march (time-keyed, 2/10 steps) | ep1 15->16 (n); ep2 20->21 (n); ep6 45->46 (n); ep3 20->21 (up); ep9 35->36 (R); ep11 15->16 (R); ep12 30->31 (L) — 7 | static-formation noops flanking each one (ep1 12->13/13->14, ep3 19->20, ep8, ep13, ...) — kills any `step%k` clock |
| 8 | enemy firing (time%15==3, at a REMAINING enemy) | ep3 18->19 (1,1); ep6 48->49 (1,1); ep7 63->64 (1,2); ep9 33->34 **(1,5)** and ep10 48->49 **(1,4)** (post-kill shifted draws — the "remaining" clause) — 5 | all other noops are no-spawn; ep4 28->29 NO_CHANGE |
| 9 | hero bullet ∩ enemy (both removed) | 4 kills x (co-locate + removal): ep2 21->22/22->23 @(3,13); ep8 22->23/23->24 @(1,2); ep11 16->17/17->18 @(3,5); ep13 42->43/43->44 @(1,11) — 8 | ep6 45->46: red reaches (1,3) BESIDE the marched formation — no kill (miss), then exits |
| 10 | hero ∩ enemy bullet (alive->false, bullet NOT consumed) | 2 deaths x (co-locate + flip): ep7 62->63/63->64 @(15,1); ep10 47->48/48->49 @(15,5) — 4 | ep3 17->18: orange lands at (15,2) ADJACENT to hero at (15,1) — hero survives |
| 11 | hero bullet ∩ enemy bullet (both removed) | 2 cancels x (co-locate + removal): ep4 26->27/27->28 @(9,1); ep14 45->46/46->47 @(13,2) — 4 (ep14 keeps the CAUSE fire 43->44 in the same slice) | ep6: red and nothing to cancel (col 3 empty) climbs through — no cancel |
| 12 | bullet off-grid despawn | RED clean: ep6 47->48 (red-1 from (0,3), nothing else changes); ORANGE (entangled with next spawn, see engine facts): ep3 18->19, ep6 48->49, ep7 63->64, ep9 33->34, ep10 48->49 — 6 | ep6 46->47: red at (0,3) still on-grid (edge row is not yet gone) |
| 13 | no reward/terminal | nothing to score (reward 0 everywhere; hero death does not end the episode — ep7/ep10 continue past it) | — |

## Episode map

| ep | src drive | steps | pairs (action: content) |
|----|-----------|-------|--------------------------|
| 0 | A | 2-5 | R clean; R + t=3 spawn (1,2); R clean |
| 1 | A | 11-16 | L 12->11; n climb x2 (red col13 + orange col2); U fire (15,11) 2nd red; n march+ |
| 2 | A | 20-23 | n march+ (+orange occluding (3,1)); n KILL coloc (3,13); n removal (+2nd red FROZEN at (8,11)) |
| 3 | B | 17-21 | n near-miss (orange (15,2), hero (15,1)); n off+spawn (1,1); n descend; U fire (15,1) + march |
| 4 | B | 26-29 | n CANCEL coloc (9,1) (orange-1); n removal (red-1); n NO_CHANGE (quiet board) |
| 5 | B | 31-34 | U PURE fire (15,3); L 3->2; L 2->1 + t=33 spawn |
| 6 | B | 45-49 | n miss: red (1,3) beside enemies + march; n red (0,3); n RED OFF-GRID clean; n orange off + spawn |
| 7 | B | 61-65 | n approach (orange (14,1)); n DEATH coloc (15,1) gray-1; n flip (hero stays gone) + spawn; U DEAD (no red+1) |
| 8 | C2 | 21-24 | n climb (red (3,2)->(2,2)); n KILL coloc (1,2); n removal |
| 9 | C2 | 33-36 | n off + SHIFTED spawn (1,5) (post-kill draw); n descend; R 4->5 + march |
| 10 | C2 | 47-50 | n DEATH coloc (15,5) gray-1; n flip + spawn (1,4); U DEAD (no red+1) |
| 11 | D | 14-18 | R 9->10; R 10->11 + march; n KILL coloc (3,5); n removal |
| 12 | D | 30-33 | L 10->9 + march; L 9->8; L 8->7 |
| 13 | D | 41-44 | n climb (red (3,11)->(2,11)); n KILL coloc (1,11); n removal |
| 14 | E | 43-47 | U fire (15,2); n converge (red 14,2 / orange 12,2); n CANCEL coloc (13,2) + march; n removal |

## Pool composition

- **52 scored transitions** (= the whole test set; `--test-n 50` with balanced_split keeps a
  pool <= test-n intact, and the sweep's loader consumes pools of this size wholesale).
- **By action:** `left:6, right:6, up:6 (4 live + 2 dead), noop:34` — movement verbs evenly
  balanced; every left/right shows the hero's 1-cell displacement, every live up shows
  `red+1` at the hero cell. The 34 noops are not filler: 8 kill pairs, 4 cancel pairs,
  4 death pairs, 5 spawn positives, 6 off-grid, 3 march+, plus climbs/descents/approaches
  and the explicit negatives (near-miss death, kill-miss, NO_CHANGE).
- **Negatives/near-misses ~25-30%**: 2 dead-ups, death-adjacent survival, kill-miss at row1,
  NO_CHANGE, no-spawn/static-march noops flanking every conditional positive.

## Known limitations (documented, not fixable in this game)

- **Dead-up aliasing (2 pairs).** `up` while dead produces a frame identical to `noop`
  (fires nothing, hero invisible). ep7 64->65 and ep10 49->50 are deliberate alive-guard
  FD/belief contrasts; an ID oracle can only call them noop-or-dead-up (ceiling ~50/52 if it
  never guesses dead-up). All other 50 pairs are action-recoverable by an oracle.
- **Edge-blocked movement is NOT tested.** A blocked left at col0 / right at col15 renders
  identically to noop (pure aliasing), violating the observability requirement, so no scored
  movement pair lacks displacement. The edge-block clause of dynamics #1/#2 is therefore
  uncovered by design.
- **Orange off-grid despawn is never isolated** — it always co-occurs with the next enemy
  fire (travel 14 vs period 15, see engine facts). The clean off-grid case is covered by the
  missed red (ep6). Train has the same entanglement.
- **Hero-death flip is evidenced by ABSENCE** (gray never returns after the orange moves on)
  because a dead hero renders black-on-black; the co-locate pair carries the positive signal.
- **Dead-move (left/right while dead) is not scored** — the hero is invisible, so the pair
  would be fully aliased with noop; excluded for the same reason as edge-blocking.
