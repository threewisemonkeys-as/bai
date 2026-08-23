# sp80 — clean_data3 coverage (REGENERATED)

Water-pour / plumbing puzzle. Reposition a movable bar during the interactive "change"
phase, then release water (ACTION5); water splits off the bar ends and must fill every
bucket without overflow. Solve → advance level (+0.1). Train is built on **level 1**
(unrotated, literal), plus a small **level-2** slice used ONLY for the click-select
mechanic (which is rotation-invariant). Level-2 *moves* are deliberately excluded because
level 2 is rotated 180° and would invert the move semantics.

## Core mechanics (from dynamics.txt, confirmed live)

1. **ACTION1 = move selected bar UP** — bar (color 9) displaces −4 display rows (1 board cell).
2. **ACTION2 = move selected bar DOWN** — +4 rows.
3. **ACTION3 = move selected bar LEFT** — −4 cols.
4. **ACTION4 = move selected bar RIGHT** — +4 cols.
5. **Move BLOCKED** — at board row 3 (top limit, y<3), an UP press does not move the bar;
   only the step meter (row 0, color 14) ticks. Near-miss negative for the move mechanic.
6. **ACTION5 = release water (spill).** Two outcomes, contrastive:
   - **WIN**: bar straddles the faucet (board cols 6–10) so its ends drop into both buckets
     → level solved, **whole board changes to the next level**, reward **+0.1**.
   - **FAIL**: wrong bar position → spill resolves off-screen, same layout reappears; only the
     meter ticks (visually a no-op).
7. **ACTION6 = click-select.** Clicking a DESELECTED piece (color 8) recolors it SELECTED
   (color 9) **exactly at the click x,y**, and the previously selected piece reverts 9→8.
   Clicking empty space, or the already-selected piece, is a no-op (meter only).

## Mechanic × objective table

| mechanic | ID (recover action from ΔX) | FD (predict X_t+1 from rule) |
|---|---|---|
| ACTION1 up | YES — bar shifts up 1 cell | YES |
| ACTION2 down | YES — bar shifts down 1 cell | YES |
| ACTION3 left | YES — bar shifts left 1 cell | YES |
| ACTION4 right | YES — bar shifts right 1 cell | YES |
| move blocked (up@row3) | NO (meter-only, aliases other no-ops) | YES — rule: at row 3, up ⇒ no move |
| ACTION5 fail | NO (meter-only, aliases other no-ops) | YES — rule: wrong position ⇒ no change |
| ACTION5 win / level-advance | YES — only ACTION5 causes whole-board change | NO — next layout is new/unpredictable |
| ACTION6 click-select (L2) | YES — new blue (9) region = click x,y | YES — recolor at clicked piece |
| ACTION6 click no-op (L1/empty) | NO (meter-only) | YES — rule: no deselected piece ⇒ no change |

## The gap in the original clean_data2 train pool

Original train = 10 rows / 9 transitions, a single scripted solve:
`ACTION6(click bar), ACTION3, ACTION1, ACTION2, ACTION4×4, ACTION5(win)`.

- **ACTION6 was a pure no-op** — the level-1 bar is already the only selectable piece, so the
  click recolored nothing. The click-select mechanic (recolor 8→9 at the click location, the
  whole point of `keep_action_params`) was **never demonstrated** → ACTION6 entirely
  ID-unidentifiable in the original.
- **No blocked move** — every move succeeded, so a "moves always" shortcut is never penalized.
- **No spill-FAIL** — only the winning ACTION5 appeared, so ACTION5 had no contrastive negative;
  the objective could treat "ACTION5 ⇒ board changes" unconditionally.
- Only ONE instance of each of ACTION1/2/3 — thin ID signal, and no near-miss contrast anywhere.

## Regeneration action sequence (train_regen/episode_0, 18 transitions)

`ACTION4,ACTION4,ACTION3,ACTION4,ACTION1,ACTION1,ACTION2,ACTION2,ACTION1,ACTION3,ACTION4,`
`click_22_18,ACTION5,ACTION4,ACTION5,click_13_17,click_30_48,click_33_25`

- steps 0–10: level-1 moves through every direction incl. an **up that blocks at row 3**.
- step 11 `click_22_18`: click on the already-selected bar → **L1 no-op** (ACTION6 negative).
- step 12 `ACTION5`: bar at cols 5–9 (ends miss the col-11 bucket) → **spill FAIL** (negative).
- steps 13–14: move to cols 6–10, `ACTION5` → **WIN / level-advance** (+0.1).
- steps 15–17 (level 2): `click_13_17` and `click_33_25` select two different deselected bars
  (**click-select positives**, recolor at the click x,y); `click_30_48` on empty = **negative**.

## Curated slices (train/, 18 scored targets)

| episode | train_regen steps | targets (mechanic) |
|---|---|---|
| 0 | 0–11 | R, R, L, R, U, **U-BLOCKED**, D, D, U, L, R — all 4 move dirs + blocked negative (level-1 windows only) |
| 1 | 11–13 | **ACTION6 L1 no-op** (neg) ; **ACTION5 FAIL** (neg) — level-1 no-op contrasts |
| 2 | 13–15 | R (winning move) ; **ACTION5 WIN / level-advance** (deliberate whole-board target, ID-only) |
| 3 | 15–18 | **click-select POS** (x=13,y=17) ; **click-empty NEG** (x=30,y=48) ; **click-select POS** (x=33,y=25) |

Pool by action (verify_pool): ACTION4=5, ACTION1=3 (2 pos + 1 blocked), ACTION3=2, ACTION2=2,
ACTION5=2 (fail+win), ACTION6=4 (1 L1 no-op, 2 select pos, 1 empty neg). Total **18**.

Contrastive coverage: move-succeeds vs move-blocked (ep0); ACTION5 win vs ACTION5 fail (ep2 vs ep1);
click-selects vs click-no-op (ep3 select pos vs ep3 empty neg + ep1 L1 no-op). The uniform
per-step meter tick (row 0, color 14) is present on EVERY transition, so it carries no action
information and cannot be used as a shortcut clock.

## Inherent limits / notes

- **ACTION5-fail, blocked-up, and L1/empty ACTION6** are all **ID-unidentifiable**: they change
  only the deterministic step meter, so ID cannot tell them apart. They are included as **FD /
  rule negatives** (the correct model must know these are conditional no-ops), never as ID targets.
- **ACTION5-win (level-advance)** is **ID-informative but FD-uninformative**: only ACTION5 triggers
  a whole-board change, but the next level's layout is not rule-predictable. It is included as one
  deliberate "level-advance" target (dynamics.txt treats win→advance as a core mechanic), sliced so
  the level-1→level-2 boundary is not smeared across other targets' windows.
- **ACTION6 is only ID-identifiable on multi-piece levels** (level 2+). Level 1 has a single
  always-selected bar, so a level-1 click can never reveal its location. The two positive
  click-select targets therefore come from level 2; because click→recolor-at-location is
  rotation-invariant, no level-2 *move* transitions are included (those would invert the move ID).
- `test/` and `dynamics.txt` are **verbatim copies of clean_data2/sp80/**. The held-out test still
  reflects the original level-1 rollout and does not contain the regenerated click-select or
  spill-fail states.
