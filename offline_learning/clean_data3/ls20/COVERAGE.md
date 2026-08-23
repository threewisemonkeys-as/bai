# ls20 — clean_data3 coverage (ARC, curated from regenerated trajectory)

ls20 is a 4-direction maze-nav puzzle. A 5x5 player block moves one 5px cell per action;
gray (color 4) walls block movement (a blocked move is a no-op frame). The camera does NOT
scroll in level 1, so the player block (top rows color 12, bottom rows color 9) displaces by
exactly 5px in the action's axis on a successful move; on a blocked move it stays put. The
bottom UI carries a step-budget bar (color 11) that ticks down and the carried-token preview.
A rotation-modifier tile cycles the carried token when stepped on; clearing the goal pad ends
the level (whole-board change). keep_action_params=FALSE ⇒ labels collapse to the four verbs.

Action map (verified from color-12 displacement): ACTION1=UP (row -5), ACTION2=DOWN (row +5),
ACTION3=LEFT (col -5), ACTION4=RIGHT (col +5).

## Core mechanics

| # | mechanic | ID (recover the action) | FD (predict next state) |
|---|----------|-------------------------|--------------------------|
| M1 | successful move: player block displaces 5px in the action's axis | YES — direction = sign+axis of the color-12 displacement | YES — block translates deterministically |
| M2 | wall-blocked move: action issued but destination is a wall ⇒ player stays put | negative — no displacement, direction NOT recoverable from change (contrastive) | YES — predict "player unchanged" (wall ahead) |
| M3 | step-budget bar (color 11) ticks down each spent step | n/a (action-independent side effect) | YES — bar shrinks (present on nearly every target) |
| M4 | rotation-modifier tile: stepping onto it occludes its glyph (colors 0/1) and cycles the carried token | partial — the extra 0/1 occlusion co-occurs with a normal UP move | YES — glyph hides on entry, reappears on exit |
| (M5) | goal-pad match → level advance (whole board swaps, Done=True) | — | whole-board change; NOT a scored target (sliced around: pair 29→30 excluded) |

## Gap in the original rollout

The clean_data2 train rollout (13 moves: LEFT×3, UP×4, RIGHT×3, UP×3) is an all-effective,
no-wall-bump solve. It NEVER shows a wall-blocked move for ANY direction, so under a
balanced-by-action sample M2 (the "action fired but nothing moved" contrast) is entirely
absent — a decoder could learn "ACTION1 always moves the block up" with no counterexample.
RIGHT/DOWN blocked cases and the full 4-way success set were also thin. The 31-frame regen
drives the player into a wall in all four directions and shows every direction's clean move,
so both M1 and its M2 near-miss are scored targets per direction.

## Curated pool (20 scored targets, keep_action_params=FALSE)

Verified by `verify_pool` → 20 targets, by action ACTION1:7, ACTION2:5, ACTION3:4, ACTION4:4.
Change tags: `12~move(±5,0)` = UP/DOWN success; `12~move(0,±5)` = LEFT/RIGHT success;
`NO_CHANGE` / `11-2 3+2` (no color-12 move) = wall-blocked; `0/1 ±` = rotation-tile occlusion.

| episode | steps (regen) | targets (mechanic) |
|---|---|---|
| ep0 | 4,5,6,7 | UP success ×3 (M1) ; **6→7 UP wall-blocked, NO_CHANGE (M2)** |
| ep1 | 9,10,11,12,13,14 | DOWN success ×4 (M1) ; **13→14 DOWN wall-blocked (M2)** |
| ep2 | 14,15,16,17,18,19 | LEFT success ×3 (M1) ; **16→17 UP wall-blocked (M2)** ; **18→19 LEFT wall-blocked (M2)** |
| ep3 | 19,20,21 | UP success (M1) ; **20→21 RIGHT wall-blocked (M2)** |
| ep4 | 22,23,24 | UP success ×2 through the rotation-modifier tile (M1+M4 glyph occlusion) |
| ep5 | 24,25,26,27 | RIGHT success ×3 (M1) |

Contrastive structure: every direction has BOTH a clean success (block displaces 5px) and a
wall-blocked near-miss (same verb, block does not move) — UP has two blocked flavors
(pure NO_CHANGE at 6→7 and step-tick-only 11-2 3+2 at 16→17). This forces the objective to
encode "the move happens only if the destination cell is open" rather than "ACTIONk always
translates the block." M4 is exposed as the color-0/1 occlusion riding on an UP move.

## Inherent limits

- **M2 (wall-blocked) is ID-degenerate by construction** — with no player displacement the
  direction cannot be recovered from the change; it is the deliberate FD-only negative
  ("predict no motion, wall ahead"). This is the point of the contrast, not a defect.
- **M3 (step bar)** is action-independent (ticks every spent step) ⇒ ID-uninformative;
  FD-only always-on side effect.
- **M4 (rotation modifier)** cannot be cleanly isolated: the carried-token preview is a tiny
  scale-2 bottom-left glyph and the tile effect co-occurs with a normal UP move, so its signal
  is entangled with M1 (kept as the 0/1 occlusion pair, not a standalone target).
- **M5 (level advance)** is a whole-board swap (pair 29→30, Done=True) — deliberately excluded
  from the scored pool.
- Camera does not scroll in level 1, so there is no camera-relative-motion confound here.

`dynamics.txt` and `test/` are verbatim clean_data2 originals. Full regen filmstrip:
`train_regen/viz.html`; selection viz: `train/viz.html`.
