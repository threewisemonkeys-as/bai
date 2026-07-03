# ft09 — clean_data3 coverage (ARC, regenerated)

ft09 is a click-only constraint-satisfaction colouring puzzle. Clicking a colour TILE cycles
its colour through the level's palette (2-colour palette ⇒ a toggle; 3-colour ⇒ a 3-way cycle);
a level is solved when all clue constraints are satisfied. A health/budget bar on the bottom
row shrinks by a step on every click.

## Core mechanics

| # | mechanic | ID (recover the ACTION6 x,y) | FD (predict next state) |
|---|----------|------------------------------|--------------------------|
| M1 | click a tile → its 3×3 block cycles to the next palette colour (toggle / 3-cycle) | YES — the flipped ~36-cell block locates the click | YES — colour advances deterministically |
| M2 | click off a tile (background/UI) → no colour change | NO (nothing moves ⇒ location unrecoverable — inherent) | YES — predict "no change" |
| M3 | budget bar (bottom row, colours 11/12) recedes one step on every click | n/a (action-independent side effect) | YES — bar shrinks each click |
| (M4) | level solved → whole board swaps to the next level's palette | — | whole-board change; NOT used as a scored target (sliced around) |

## Gap in the original rollout

The original 12-row clean_data2 rollout clicked few tiles and never cleanly separated the
toggle directions, an off-tile no-op, or a second level's palette — so the click→cycle rule and
its location-dependence were thinly exercised. Regenerated (23 frames) to drive many distinct
tile clicks, both toggle directions, a NO_CHANGE off-tile negative, and a second palette.

## Curated pool (20 scored targets, keep_action_params=TRUE)

Each target is a click whose location is recoverable from where the 3×3 block flips, plus the
budget bar tick. By slice (change tags: `8/9` = blue↔red toggle, level THR; `9/12` = blue↔orange,
level hxv; `11/12` = budget bar):

| episode | steps | mechanic exposed |
|---|---|---|
| ep0 | 0–4 | M1 toggle 9↔8 at distinct tiles (+M3) ; ends on **M2 NO_CHANGE** (off-tile) |
| ep1 | 5–10 | M1 toggle both directions across tiles ; ends **M2 NO_CHANGE** |
| ep2 | 10–13 | M1 repeated toggles at new tile locations |
| ep3 | 14–18 | M1 second palette (9↔12, level hxv) |
| ep4 | 18–22 | **M2 NO_CHANGE** first ; then M1 9↔12 toggles |

Contrastive structure: same action verb (ACTION6) yields a colour flip at the clicked tile vs.
NO_CHANGE off-tile — forcing "click cycles the tile UNDER the cursor" rather than "click always
flips." Distinct click x,y across tiles make the location genuinely part of the target.

## Inherent limits

- **M2 (off-tile no-op) is ID-unidentifiable** — with nothing changing, the click x,y can't be
  recovered; it is learnable only under FD ("predict no change"). Kept as the negative.
- **M3 (budget bar)** is action-independent (fires on every click), so it carries no ID signal;
  it's an FD-only always-on side effect.
- Level-advance (M4) is a whole-board swap, deliberately not a scored target.

`dynamics.txt` and `test/` are the verbatim clean_data2 originals. Full trajectory:
`train_regen/viz.html`; selection: `train/viz.html`.
