# Ground truth vs. learned beliefs — top-5 games (stepwise vs RGB-Agent)

Date: 2026-06-17
Games: the top-5 by the stepwise 1–5 re-score (7WWW9, F5W3N, DQ8GC, QQM74, VQJH6).
Setup (both methods, identical): AutumnBench, 50 env-step budget, gemini-2.5-flash, free
exploration (stepwise = `theory_disagreement` + perception, dynamics mode; RGB =
`run_autumn.py --task interactive`).

Ground truth read from the `.sexp` programs in
`MARAProtocol/python_examples/autumnbench/example_benchmark/programs/`.
Stepwise belief source: final episode `beliefs.txt`. RGB belief source: the analyzer's
"Inferred Dynamics" strategic briefings in `logs_analyzer.txt` (RGB keeps no belief store).

**Rubric (1–5):** how much of the *core dynamics* is covered correctly. Different
verbalization is fine — a correct mechanical description counts even if it doesn't name the
concept (magnet/gravity/momentum/etc.).

## Score summary

| Game  | Stepwise | RGB | RGB steps used |
|-------|:--------:|:---:|:--------------:|
| 7WWW9 | 4 | 2 | 14 (self-terminated) |
| F5W3N | 3 | 1 | 10 (self-terminated) |
| DQ8GC | 2 | 3 | 36 (self-terminated) |
| QQM74 | 2 | 1 | 22 (self-terminated) |
| VQJH6 | 2 | 2 | 50 |
| **Mean** | **2.6** | **1.8** | |

Stepwise higher on 3 (7WWW9, F5W3N, QQM74), RGB higher on 1 (DQ8GC), tie on 1 (VQJH6).
RGB self-terminated early (`go-to-test` / "task complete") on 4 of 5, capping its evidence.

---

## 7WWW9 — magnets

**Ground-truth dynamics (.sexp):**
- A mobile blue magnet (vertical 2-cell, init (4,7)) is moved by arrow keys.
- A red magnet (vertical 2-cell, (7,7)) is fixed.
- Like poles adjacent → movement blocked (mobile stays put).
- Opposite poles separated by distance 2 along an axis (attractVectors = {(0,±2),(±2,0)})
  → the mobile magnet auto-moves one step toward the fixed one (attraction).

**Stepwise beliefs — score 4/5**
- Blue is player-controlled, moved by arrows; red is a static obstacle. ✓
- "right is blocked when the blue's rightmost col would be adjacent to red (dist==2)" ✓
  (like-pole block, captured mechanically).
- "up moves up one row; if red_col − blue_col == 2 the column also shifts +1" ✓
  (this is exactly the distance-2 attraction shift, captured operationally).
- Captures all three observable behaviors of the controllable magnet (move / block / attract)
  without naming "magnetism". Misses only the general framing.

**RGB beliefs — score 2/5**
- "The blue cells move with directional commands, red cells are static, and clicks have no
  effect." ✓ movement + red-is-fixed.
- Misses the blocking and the attraction entirely; declared exploration done and quit at
  step 14. Covers 1 of 3 mechanics.

---

## F5W3N — space invaders

**Ground-truth dynamics (.sexp):**
- Hero (gray) moves left/right; `up` fires a red bullet that travels up each tick.
- Two enemy rows (blue): row-1 enemies where x%3==1, row-3 where x%3==2.
- Enemies oscillate on a timer: at time%10==5 row-1 moves left & row-3 right; at time%10==0
  row-1 right & row-3 left.
- Enemy bullets (orange) spawn periodically (time%15==3) from a random enemy and move down.
- Collisions: bullet↔enemy and bullet↔enemy-bullet remove both; hero hit by enemy bullet dies.

**Stepwise beliefs — score 3/5**
- "Blue cells follow two configs, 'initial' and 'shifted'; in a shift row-1 moves one col right
  and row-3 one col left." ✓ — precisely the enemy oscillation (got the two-row, opposite-
  direction structure), though it attributed the trigger to click/noop rather than a timer.
- "Orange cells move down one row on any action; disappear at row 15." ✓ enemy-bullet fall.
- Misses hero movement, shooting, and collisions. Covers ~2 of 5, but the 2 captured are right.

**RGB beliefs — score 1/5**
- Got movement + grid boundaries.
- But the reported "dynamics" are a fabricated color-toggle scheme that does not exist:
  "green↔red, blue↔yellow on click; white→green, black→red." (It misread the enemy
  oscillation / bullets as its own clicks recoloring cells.) Declared "complete understanding,"
  scored the test, looped "Score 100, done." Core mechanics essentially all wrong.

---

## DQ8GC — infection / conversion

**Ground-truth dynamics (.sexp):**
- The active particle (darkgreen, health=false) is moved by arrow keys.
- Each tick, any inactive particle (gray, health=true) adjacent to a non-health (darkgreen)
  particle becomes darkgreen — infection spreads to neighbors.
- Clicking an inactive particle makes it the new active particle (the old active joins the
  inactive set) — a swap.

**Stepwise beliefs — score 2/5**
- Arrow keys move the darkgreen "player" by one cell. ✓
- Explicitly states "click does not change the player position" and never captures the
  infection/spread or the click-to-swap-active. ✗✗
- Covers only movement (1 of 3); misses the defining spread mechanic.

**RGB beliefs — score 3/5** *(RGB's best game)*
- States the operational rule: "move the darkgreen agent adjacent to a gray cell, then click
  the gray cell → the gray turns darkgreen," and that this converts all grays over time. ✓
- Captures the conversion/spread (the defining mechanic) plus agent movement, framed as a
  manual adjacency+click operation rather than automatic per-tick infection. Covers ~2 of 3.

---

## QQM74 — momentum

**Ground-truth dynamics (.sexp):**
- A single blue blob starts at center (10,10); background is plain black (no other objects).
- left/right change a capped x-velocity by ∓1; up/down change a capped y-velocity by ∓1.
- Every tick (including `noop`) the blob moves by its (xVel,yVel) — momentum/drift.
- Clicking a free cell adds a new blob there. up/quit/reset return to the initial state.

**Stepwise beliefs — score 2/5**
- "noop causes passive continued movement, with direction influenced by prior arrow actions;
  a right action can suppress/redirect it." ✓ — a fuzzy but genuine capture of velocity/momentum.
- But models arrows as *direct* position moves and clicks as *teleport-to-(row,col)* — both
  wrong (arrows set velocity; click *adds a blob*). Only the momentum observation is correct.

**RGB beliefs — score 1/5**
- Describes the blob as moving one cardinal step per arrow with edge-wrap, plus a large set of
  click color-toggles (red↔white "togglable obstacle," green↔bluegreen, purple↔bluepurple)
  and "moving onto red bounces back." None of those colored cells/obstacles exist in this game
  (it's a black field with one blob); the whole obstacle/toggle model is hallucinated. Misses
  momentum and the click-adds-blob rule. Terminated at step 22.

---

## VQJH6 — gravity

**Ground-truth dynamics (.sexp):**
- Four edge buttons — left(red), right(darkorange), up(gold), down(green) — set the gravity
  direction when clicked.
- 2×2 blue blobs fall one step in the current gravity direction every tick.
- Clicking an empty cell adds a 2×2 blue blob there.

**Stepwise beliefs — score 2/5**
- "Clicking a black cell makes a 2×2-ish block of cells turn blue" ✓ (blob spawn), and "blobs
  then rearrange/move over time" (vaguely the falling). 
- Treats the gold/darkorange/green/red buttons as *static fixed* cells — misses that they set
  gravity direction. Covers ~1.5 of 3.

**RGB beliefs — score 2/5**
- "Clicks around gold create blue patterns; clicks around darkorange erase the blue pattern,
  turning cells black; click 0 9 turned red into blue." Partially captures blob creation and —
  notably — that the gold/darkorange buttons drive the changes (closer to the truth that
  buttons matter), but frames it as create/erase and misses the gravity-direction + falling
  dynamic. Used the full 50 steps. Covers ~1.5 of 3.

---

## Caveats
- 1–5 scores are a manual judgment, not a calibrated metric.
- The two belief sources are not symmetric: stepwise has a distilled `beliefs.txt`; RGB's
  "beliefs" are exploration-time briefings (and on 4/5 games it stopped exploring early).
- See `notes/beliefs_vs_groundtruth_planA_vs_rgb.md` for the original all-10-game 1–10 pass.
