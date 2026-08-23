# tn36 — clean_data3 coverage (ARC, regenerated)

tn36 is a click-only "visual program the robot" puzzle. You toggle bits in a row of editor
cells below the board, then press a RUN button; the program executes and moves a robot sprite.
You win when the robot matches a target. A timer bar (row 1, colour 9/blue) recedes one column
on EVERY click; if it empties you lose.

## Core mechanics

| # | mechanic | ID (recover ACTION6 x,y) | FD (predict next state) |
|---|----------|--------------------------|--------------------------|
| M1 | click an editor cell → toggles a bit (segment display flips colours 1↔5) | YES — which segment cell changed locates the click | YES — the toggled bit's display flips deterministically |
| M2 | timer bar recedes one column (colour 9) on every click | n/a (action-independent side effect) | YES — bar shortens each click |
| M3 | click RUN → the program executes; the board/robot change wholesale (and level can advance) | partial — click on the RUN button, but the effect is a whole-board change | whole-board; treated as the "run" mechanic, sliced as one target |

## Gap in the original rollout

The original clean_data2 rollout was short and did not cleanly separate bit-toggle clicks (with
their segment change) from the RUN execution, nor show the timer's always-on recede against the
bit toggles. Regenerated (20 frames) to drive a series of editor-bit toggles (both directions),
timer-only clicks, and the RUN execution.

## Curated pool (19 scored targets, keep_action_params=TRUE)

Change tags: `1/5` = segment bit flip (editor cell), `9-1` = timer recede, `3` = padding;
the big `0+250 2+544 3+838 …` at ep6 = the RUN executing (board change).

| episode | steps | mechanic exposed |
|---|---|---|
| ep0 | 0–4 | M1 bit toggles both directions at editor cells (+M2 timer each click) |
| ep1 | 4–6 | M2 timer-only clicks (bit already set / minimal display change) |
| ep2 | 6–10 | M1 more bit toggles across cells |
| ep3 | 10–12 | M2 timer-only then M1 toggle |
| ep4 | 12–15 | M1 bit toggles (set direction) |
| ep5 | 15–18 | M2 timer + M1 toggles |
| ep6 | 18–19 | **M3 RUN → program executes / board changes** |

Contrastive structure: a click that flips a specific editor segment (M1, location-recoverable)
vs. a click that only advances the timer (M2) vs. the RUN click that transforms the board (M3) —
so the same ACTION6 verb maps to three distinguishable outcome classes.

## Inherent limits

- **M2 (timer recede)** fires on every click regardless of where — it is action-independent and
  carries no ID signal (FD-only always-on effect).
- Clicks whose only visible change is the timer are ID-weak (location hard to recover); they are
  retained to teach the always-on timer and as the contrast to M1's locatable bit flips.
- **M3 RUN** is a whole-board change; it is included as the single "execute the program" target
  rather than sliced away, because running the program is itself a core mechanic here.

`dynamics.txt` and `test/` are the verbatim clean_data2 originals. Full trajectory:
`train_regen/viz.html`; selection: `train/viz.html`.
