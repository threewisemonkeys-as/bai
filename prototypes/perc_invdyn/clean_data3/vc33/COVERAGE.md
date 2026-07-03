# vc33 — clean_data3 coverage analysis (REGENERATED)

Whitelist: `ACTION6`  •  keep_action_params = **TRUE** (the full click string
`ACTION6 x=X y=Y` is the label; the click LOCATION is the target, so ID must
recover x,y from what changed).

vc33 is a **click-only gravity/sliding puzzle**. The only action in every state is
ACTION6(x,y). You slide coloured balls onto matching goals by clicking buttons; a
per-level step budget (pink top row) counts down 1 per click; solving a level
advances the board.

## Why regenerated (the gap in the original)

`clean_data2/vc33/train/episode_0` is only **6 rows / 5 transitions**:
`top(right), bottom(left), bottom(left), bottom(left), bottom(left)=SOLVE`.

- The **5th transition IS the whole-board level-advance** (11→14, walls/buttons all
  redraw, 276 background cells change). A balanced-pool sample would score this
  whole-board swap as a target — it is neither ID- nor FD-learnable (the click that
  triggered it, a bottom-button, is drowned by the board redraw). This is exactly the
  "don't score the level transition" case.
- **No contrastive negatives at all:** every scored transition moves the ball, so
  "any click moves the ball" fits the data with zero penalty. No blocked move, no
  no-op click (empty / wall / dead button), no near-miss.
- **Only one direction pair, one level, one ball colour.** Never shows the button→
  direction binding is conditional on ball position (blocked-at-wall) or on which
  chamber the ball is in (dead button in L2).

## Regenerated trajectory (`train_regen/episode_0`, 21 rows / 20 transitions)

Driven with `arc_drive.py` (seed 0, deterministic). Action sequence + what each
transition demonstrates (transition `i→i+1` is caused by row `i`'s action):

| steps | action | outcome |
|---|---|---|
| 0→1,1→2,2→3 | `x=60 y=25` (L1 top btn) | ball(11) slides **RIGHT** +2 logical, chambers resize |
| 3→4,4→5 | `x=60 y=25` | ball now at right wall → **BLOCKED** (no move, only bar ticks) |
| 5→6 | `x=10 y=15` (empty bg) | **NO-OP** click (bar ticks, ball still) |
| 6→7 | `x=46 y=29` (on a wall) | **NO-OP** click on wall |
| 7→8…11→12 | `x=60 y=33` (L1 bottom btn) | ball slides **LEFT** −2 logical |
| 12→13 | `x=60 y=33` | ball reaches goal column → **SOLVE → Level 2** (+0.1, board redraws, 11→14). *Sliced AROUND — never a scored target.* |
| 13→14,14→15,15→16 | `x=60 y=33` | in L2 that spot is empty → **NO-OP** (dead ex-button) |
| 16→17,18→19 | `x=1 y=45` (L2 btn) | green ball(14) slides **RIGHT** +2 |
| 17→18,19→20 | `x=1 y=37` (L2 btn) | green ball(14) slides **LEFT** −2 |

Reversal note: reversing direction *at the wall* can absorb the first click
(engine settle); this sequence inserts the two no-op clicks (5,6) before reversing,
and every left move 7→8…11→12 registers immediately (verified frame-by-frame).

## Core mechanics (from dynamics.txt)

- **M1 button-move** — clicking a button adjacent to the ball's chamber slides the ball
  by the level's gravity magnitude along the gravity axis; different buttons drive
  opposite directions. (L1 horizontal: top=+x, bottom=−x. L2 horizontal, green ball.)
- **M2 blocked move** — a ball against the chamber-end wall doesn't move when clicked
  further into the wall; the click still burns a step. (Conditional on ball position.)
- **M3 no-op / dead click** — clicking empty background, a wall, or a location that
  isn't an *active* button (e.g. a spot that was a button in a previous level) does
  nothing to the ball; only the budget bar ticks. (Conditional on WHERE you click.)
- **M4 step-budget bar** — every ACTION6 decrements the pink top-row (colour 7) bar.
  It is proportional/quantised (≈1.28 display-cells per click → alternating −1/−2), so
  it is a *coarse* per-click clock, not a per-click ±1 signal.
- **M5 level solve/advance** — when the ball's on-axis coord matches a same-colour
  goal on an adjacent wall, the whole board switches to the next level (+0.1 reward,
  budget refills, ball recolours 11→14). Whole-board change → **not scored** (sliced around).
- **M6 bridge blocks** (levels 4–7) — described in dynamics.txt only; unreachable
  without solving L1–L3, so genuinely absent here too (as in clean_data2).

## Mechanic × objective table

| mechanic | ID (recover x,y from change)? | FD (predict next state from rule)? | in ORIGINAL pool? |
|---|---|---|---|
| M1 button-move (L1 right, top) | **yes** — RIGHT motion ⇒ top button ⇒ `x=60 y=25` | **yes** — ball col +2 | partial (1 right) |
| M1 button-move (L1 left, bottom) | **yes** — LEFT motion ⇒ bottom ⇒ `x=60 y=33` | **yes** — ball col −2 | yes (3 lefts) |
| M1 button-move (L2 right/left, green) | **yes** — direction ⇒ `x=1 y=45`/`x=1 y=37` | **yes** — ball(14) ±2 | **NO (L2 never shown)** |
| M2 blocked | **no** — no ball change ⇒ location unrecoverable (see limits) | **yes** — rule predicts "at wall ⇒ no move" | **NO** |
| M3 no-op (empty/wall/dead btn) | **no** — location unrecoverable from bar-only change | **yes** — predict "ball unchanged" | **NO** |
| M4 budget bar | n/a (fires on every click, location-independent) | weak (quantised) | present but unscored |
| M5 level-advance | no (whole board) | no (whole board) | **scored in original (bad)** — fixed by slicing around |

## Curated pool — 5 episodes → 18 scored targets

`T.verify_pool(...'train','ACTION6')`:

| ep | steps | scored targets | mechanic |
|---|---|---|---|
| 0 | 0,1,2,3,4 | 0→1,1→2,2→3 R; 3→4 blocked | M1 L1-right ×3 + **M2 blocked** (same label, no move) |
| 1 | 4,5,6,7 | 4→5 blocked; 5→6 empty; 6→7 wall | **M2 blocked** + **M3 no-op empty + no-op wall** |
| 2 | 7,8,9,10,11 | 7→8…10→11 L ×4 | M1 L1-left ×4 |
| 3 | 13,14,15,16 | 13→14,14→15,15→16 no-op ×3 | **M3 dead-ex-button in L2** (label `x=60 y=33`, no move) |
| 4 | 16,17,18,19,20 | R,L,R,L | M1 L2 green-ball right(`x=1 y=45`) ×2 + left(`x=1 y=37`) ×2 |

By label: `x=60 y=25`×5 (3 move + 2 blocked), `x=10 y=15`×1, `x=46 y=29`×1,
`x=60 y=33`×7 (4 L1-left move + 3 L2 no-op), `x=1 y=45`×2, `x=1 y=37`×2.  Pool = 18 ≤
train-n 20 ⇒ all used.

### Contrastive structure (defeats shortcuts)

- **`x=60 y=25` is BOTH a right-move (steps 0–3) AND a no-move (steps 3–5, ball at
  wall)** in the same/adjacent slices. Defeats "the top button always moves the ball
  right"; forces the *conditional* (moves only when not blocked). The wall state is
  carried in-window so FD can learn the condition.
- **`x=60 y=33` is BOTH a left-move (L1, ep2) AND a no-move (L2, ep3).** Defeats
  "clicking this location always slides the ball left"; the outcome depends on the
  ball's chamber/level.
- **No-op empty (`x=10 y=15`) & no-op wall (`x=46 y=29`) vs. real moves.** Defeats
  "any click moves the ball" — a click that lands on nothing leaves the ball fixed.
- **Direction distinguishes co-located buttons.** In L1 both buttons share x=60
  (differ only in y=25 vs y=33); in L2 both share x=1 (y=45 vs y=37). ID must read the
  ball's motion *direction* to recover y — a pure "where did something appear" cue is
  insufficient, since the ball moves, nothing appears at the click.
- **Two ball colours (11 yellow L1, 14 green L2)** exercise the same move rule, so the
  mechanic can't be bound to a specific colour.

## Inherent limits (important)

- **ID is fundamentally unrecoverable for every no-move click** (M2 blocked, M3
  no-op). A blocked top-click, a click on empty space, a click on a wall, and a dead
  L2 button ALL produce the identical observable change (ball unchanged; only the
  budget bar ticks). The click LOCATION cannot be recovered from the state pair — the
  bar shrink is location-independent. These transitions are included **only as FD
  targets and as contrastive negatives**; scoring them under ID is impossible by
  construction (this is a property of the game: clicks act through buttons, and a
  missed/blocked click leaves no positional trace). The move transitions (M1) ARE
  ID-recoverable via the ball's displacement direction.
- **Budget bar is a coarse clock**, not a clean ±1 per click (quantised ≈1.28
  cells/click). It is deliberately NOT used as the discriminative signal for anything.
- **Level-advance (M5) is unscored** — a whole-board redraw is neither ID- nor
  FD-learnable as a single-click mechanic; every slice stops before step 12→13.
- **M6 bridge blocks (L4–7) remain unreachable** and are absent, same as clean_data2.
- **test/ is the verbatim clean_data2 copy** and still lacks the regenerated
  blocked/no-op/L2 states; it only shows the original L1 solve path.
