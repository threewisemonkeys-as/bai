# e3v6m — clean_data3 coverage

Game: **e3v6m** ("Lights" beam). Whitelist = `left,right,up,down,noop,click`.
`keep_action_params = FALSE` (movement game): the click LABEL collapses to the verb
`click`; click LOCATION is NOT the target.

## 1. Core dynamics (from dynamics.txt)

A single object "Lights" = a 5-cell line/beam on a 16x16 grid. State: `turnedOn` (Bool,
color white=off / yellow=on) and `dir` (0-7, one of 8 compass orientations). Anchor starts
at (7,7), dir=1 (down-right diagonal), turnedOn=false.

- **D1 click → toggle.** `click ROW COL` flips turnedOn (white↔yellow). The coordinate is
  required but irrelevant (any cell toggles). Bare `click` is a no-op. Position unchanged.
- **D2 move while OFF (white) → TRANSLATE.** up/down/left/right shift the whole beam one
  cell (up=row−1, down=row+1, left=col−1, right=col+1). dir unchanged.
- **D3 move while ON (yellow) → ROTATE (no translation).** Same keys change `dir` about the
  fixed anchor: up=dir+1, down=dir−1, left=dir+2, right=dir−2 (mod 8). Centroid unchanged.
- **D4 noop → nothing.** No handler.
- **D5 no passive/clock, no spawn/despawn, no collision, no win/termination.** State only
  changes in response to player actions; the beam otherwise persists unchanged.

The defining feature is the **state-conditional action semantics**: the *same* arrow key
does TWO different things depending on `turnedOn` (D2 vs D3). This is the dynamic that must
be covered contrastively or a "up always translates" shortcut goes unpunished.

## 2. Is each dynamic a scored TARGET under ID/FD in the ORIGINAL balanced-20 pool?

Original train = 88 rows, 66 of them noop NO_CHANGE; only 2 click rows. Counts available:
up≈6, down≈4, left≈5, right≈4, click=2, noop=66. A `balanced_split` to 20 samples ~3-4 per
action at random.

| Dynamic | ID informative? | FD informative? | In original balanced-20 pool? | Gap |
|---|---|---|---|---|
| D1 click toggle | yes — only action that recolors w/o moving | yes — 5 cells change color | fragile: only 2 click rows exist; a balanced sample usually keeps both, but the toggle DIRECTION (on vs off) is left to chance | toggle present but not guaranteed in both directions |
| D2 translate-OFF | yes — each dir = unique 1-cell shift | yes — positions move | likely sampled | — |
| D3 rotate-ON | up(+1)/down(−1) distinguishable; **left(+2)/right(−2) visually identical** (line is symmetric, dir≡dir+4) | yes — orientation changes | likely sampled | left-ON vs right-ON inherently unidentifiable under ID (see note) |
| D2 vs D3 conditional | the whole point — needs BOTH regimes for the SAME key | yes | **NOT guaranteed** — a random balanced sample need not contain both up-OFF and up-ON, etc. | **MAIN GAP (nrdf6-style):** the conditional rule appears only by luck; the "arrow=translate" shortcut can score fine, exactly the failure mode clean_data3 exists to fix |
| D4 noop | trivial (no change) | trivial | massively over-represented (66 rows) | none, but floods the raw pool |
| D5 (none) | n/a | n/a | n/a | no passive dynamic ⇒ no step-parity/clock shortcut risk (unlike nrdf6) |

**Note on the ID limit for D3:** the beam is a symmetric 5-cell line, so orientation `dir`
and `dir+4` render identically. left=+2 and right=−2≡+6 differ by 4 ⇒ they produce the
*same visible* orientation change. left-ON and right-ON are therefore not separable by ID
(confirmed: 55→56 left and 63→64 right both map anti-diagonal→diagonal). They remain
distinguishable from up/down/translate/click/noop, and FD is fully informative for them, and
the OFF/ON contrast still holds — so they are kept, with this caveat documented.

## 3. Curated slices (clean_data3/e3v6m/train) — 20 episodes, 20 scored targets

Each episode is a 2-row verbatim slice `[s, s+1]` from the original train trajectory, so each
yields exactly one scored target `s→s+1` with no window bleed. State (white/yellow) is fully
visible in `x_t`, so zero-length windows are sufficient (no delayed effects in this game).

| episode | steps | action | regime | what it tests |
|---|---|---|---|---|
| 0 | 3→4   | up    | OFF | D2 translate up (row−1) |
| 1 | 7→8   | up    | OFF | D2 translate up |
| 2 | 11→12 | left  | OFF | D2 translate left (col−1) |
| 3 | 15→16 | left  | OFF | D2 translate left |
| 4 | 19→20 | down  | OFF | D2 translate down (row+1) |
| 5 | 23→24 | down  | OFF | D2 translate down |
| 6 | 27→28 | right | OFF | D2 translate right (col+1) |
| 7 | 31→32 | right | OFF | D2 translate right |
| 8 | 35→36 | click | —   | D1 toggle white→yellow |
| 9 | 75→76 | click | —   | D1 toggle yellow→white |
| 10 | 39→40 | up    | ON  | D3 rotate up (dir+1) |
| 11 | 43→44 | up    | ON  | D3 rotate up |
| 12 | 51→52 | down  | ON  | D3 rotate down (dir−1) |
| 13 | 71→72 | down  | ON  | D3 rotate down |
| 14 | 55→56 | left  | ON  | D3 rotate left (dir+2) |
| 15 | 59→60 | left  | ON  | D3 rotate left |
| 16 | 63→64 | right | ON  | D3 rotate right (dir−2) |
| 17 | 67→68 | right | ON  | D3 rotate right |
| 18 | 0→1   | noop  | white | D4 noop = no change |
| 19 | 36→37 | noop  | yellow | D4 noop = no change (yellow doesn't passively rotate) |

By action: up 4, left 4, down 4, right 4, click 2, noop 2 = **20**.

### Contrastive structure (defeats shortcuts)

- **D2 vs D3 per direction (the core conditional):** every arrow key appears as BOTH a
  translate (OFF, ep 0-7) and a rotate (ON, ep 10-17). up: ep0/1 (move) vs ep10/11 (rotate);
  down: ep4/5 vs ep12/13; left: ep2/3 vs ep14/15; right: ep6/7 vs ep16/17. A rule that says
  "up always translates the beam" scores worse than the true state-conditional rule, because
  half of each key's targets reorient without moving.
- **D1 both toggle directions:** ep8 white→yellow, ep9 yellow→white — the toggle can't be
  collapsed to "click → yellow".
- **D4 as the null/near-miss negative:** ep18 (white) and ep19 (yellow) show that taking a
  step with no arrow/click leaves the beam unchanged — defeating any "every step the beam
  moves / a clock advances" shortcut. (There is no passive dynamic here, so no nrdf6-style
  step-parity hijack is possible, but the noops nail that down explicitly.)

## 4. Verification

`T.verify_pool('prototypes/perc_invdyn/clean_data3/e3v6m/train','left,right,up,down,noop,click')`
reports exactly 20 scored target transitions: by action {up:4, left:4, down:4, right:4,
click:2, noop:2}; the 8 OFF moves show `white~move(...)` (translation), the 8 ON moves show
`yellow~recolor` (rotation), clicks show `white∓5 yellow±5` (toggle), noops `NO_CHANGE`.
Pool size = 20 = `--train-n`, so `balanced_split` keeps all of them.

## TRAIN2 (untied-val expansion)

A second, independently-driven pool so the per-game total (~20 original + ~29 here ≈ 49)
supports an **untied** 30-train/30-val split for GEPA runs, instead of `--tie-train-val`.
Built per `clean_data3_REGEN_METHODOLOGY.md`: a single fresh `autumn_drive.py E3V6M ...`
trajectory (seed 0) saved verbatim to `train_regen2/episode_0/trajectory.csv` (51 rows, 50
transitions), then curated into 4 contiguous slices under `train2/`.

### Pool size and action histogram

`T.verify_pool('prototypes/perc_invdyn/clean_data3/e3v6m/train2','left,right,up,down,noop,click', context_k=9)`
reports **29 scored target transitions** (within the requested 28–32 band):

| verb | count | OFF (translate) | ON (rotate) |
|---|---|---|---|
| up | 4 | 2 | 2 |
| down | 6 | 4 | 2 |
| left | 5 | 3 | 2 |
| right | 4 | 2 | 2 |
| click | 4 | — (2 W→Y, 2 Y→W) | — |
| noop | 6 | — (4 white, 2 yellow; 3 DIAG-orientation, 3 VERT-orientation) | — |

(`verify_pool`'s raw dump shows 4 distinct `click ROW COL` strings, one occurrence each —
consistent with `keep_action_params=FALSE`: the label collapses to the bare verb `click` at
scoring time, so location diversity is a hygiene choice, not a labeling requirement.)

Verified with `load_transitions`: **0 internal duplicate** `(x_t, action, x_t+1)` triples,
**0 shared** with `train/`, **0 shared** with `test50/`.

### Dynamic -> target-pair coverage (every core dynamic ≥2 as a SCORED target)

| dynamic | positives (targets) | count | contrastive negative |
|---|---|---|---|
| D1 click W→Y | ep0 row12→13 `click 4 13` (on-beam, anchor (4,13)); ep3 row41→42 `click 13 1` (far-empty) | 2 | position/orientation never change (every click pair) |
| D1 click Y→W | ep1 row19→20 `click 9 2` (far cell); ep3 row46→47 `click 2 9` (far cell) | 2 | same as above |
| D2 up-OFF (translate) | ep2 rows2→3, 3→4 (col 7 constant, rows 5→4→3) | 2 | ep0 rows14→15,15→16 (up while ON = rotate, no move) |
| D2 down-OFF | ep0 row10→11; ep1 row23→24; ep3 rows40→41, 48→49 | 4 | ep1 row16→17, ep3 row45→46 (down while ON = rotate) |
| D2 left-OFF | ep1 rows20→21, 21→22; ep3 row47→48 | 3 | ep1 row17→18, ep3 row43→44 (left while ON = rotate) |
| D2 right-OFF | ep0 row11→12; ep3 row39→40 | 2 | ep1 row18→19, ep3 row44→45 (right while ON = rotate) |
| D3 up-ON (rotate, dir+1) | ep0 rows14→15 (DIAG→VERT), 15→16 (VERT→ANTI) | 2 | ep2 rows2→3,3→4 (up while OFF = translate, no reorient) |
| D3 down-ON (dir−1) | ep1 row16→17 (ANTI→VERT); ep3 row45→46 (VERT→DIAG) | 2 | ep0/ep1/ep3 down-OFF pairs above |
| D3 left-ON (dir+2) | ep1 row17→18 (VERT→HORIZ); ep3 row43→44 (VERT→HORIZ) | 2 | ep1/ep3 left-OFF pairs above |
| D3 right-ON (dir−2) | ep1 row18→19 (HORIZ→VERT); ep3 row44→45 (HORIZ→VERT) | 2 | ep0/ep1/ep3 right-OFF pairs above |
| D4 noop = nothing | ep0 rows9→10 (white DIAG @(3,12)), 13→14 (yellow DIAG @(4,13)); ep1 row22→23 (white VERT @(4,11)); ep3 rows38→39 (white VERT @(11,3)), 42→43 (yellow VERT @(12,4)), 49→50 (white DIAG @(13,3)) | 6 | every non-noop pair in the pool changes something |

Same left-ON/right-ON aliasing caveat as documented for `train/`/`test50/` applies here
(the 5-cell line is symmetric, `dir` ≡ `dir+4` visually, so left(+2)/right(−2≡+6) rotate to
the *same* visual class from any given state) — both are still included for regime coverage
(distinguishable from up/down/click/noop and fully FD-informative), never literally duplicated
within this pool (different anchors: (4,13) in ep0/ep1, (12,4) in ep3).

### Curated slices (train_regen2 rows -> train2 episodes)

| episode | source rows (train_regen2) | rows/pairs | region / anchor path | what it demonstrates |
|---|---|---|---|---|
| 0 | 9–16 | 8 rows / 7 pairs | (3,12) DIAG→(4,12)→(4,13) DIAG, toggle, up,up rotate to ANTI | D4 neg, D2 down+right, D1 W→Y, D4 neg, D3 up×2 |
| 1 | 16–24 | 9 rows / 8 pairs | pivot at (4,13): rotate down/left/right, toggle back, translate away to (5,11) | D3 down/left/right, D1 Y→W, D2 left×2, D4 neg, D2 down |
| 2 | 2–4 | 3 rows / 2 pairs | column translate up, (5,7)→(4,7)→(3,7) | D2 up×2 |
| 3 | 38–50 | 13 rows / 12 pairs | (11,3)→(11,4)→(12,4) pivot: translate, toggle, rotate left/right/down, toggle back, translate to (13,3) | D4 neg, D2 right+down, D1 W→Y, D4 neg, D3 left/right/down, D1 Y→W, D2 left+down, D4 neg |

### How train2's situations differ from train/ and test50/

| aspect | train | test50 | train2 |
|---|---|---|---|
| OFF-translate anchors/orientation | box rows 5–7, cols 5–7, DIAG only | rows 5–11, cols 4–11, all 4 orientations | rows 3–13, cols 3–13 (extends past both row<5 and row>11), DIAG **and** VERT |
| ON-rotate anchors | single anchor (7,7) | 5 anchors: (5,8),(9,5),(8,9),(6,5),(10,11) | 2 new anchors: **(4,13)** and **(12,4)** — neither matches train's (7,7) nor any of test50's 5 |
| click locations | both at `click 7 7` | 9 distinct: (1,13),(12,2),(15,15),(0,0),(13,4),(5,8),(8,9),(6,5),(9,7) | 4 distinct, none overlapping either list: `click 4 13` (on-beam), `click 9 2`, `click 13 1`, `click 2 9` |
| noop positions/colors | 2 (both at (7,7)) | 9, spread over positions/orientations/colors | 6, at (3,12)/(4,13)/(4,11)/(11,3)/(12,4)/(13,3) — none reused from either prior pool |
| slice shape | twenty 2-row slices, zero window | 12/12/12/12/7-row slices | 8/9/3/13-row slices (real windows within-episode, context_k=9) |
| rotation cadence | scripted "each direction, ON then done" | scripted per-episode themes | one long rotate-burst per pivot (up,up / down,left,right / left,right,down) with irregular noop placement, no fixed step-parity |

Verified programmatically (see above): 0 scored `(x_t, action, x_t+1)` triples shared with
`train/` or `test50/`, and 0 internal duplicates in `train2/`.

### Reproduce

```
uv run python prototypes/perc_invdyn/autumn_drive.py E3V6M prototypes/perc_invdyn/clean_data3/e3v6m/train_regen2 \
  --actions "up,up,up,up,right,right,right,right,right,noop,down,right,click_4_13,noop,up,up,down,left,right,click_9_2,left,left,noop,down,down,down,down,down,down,down,left,left,left,left,left,left,left,left,noop,right,down,click_13_1,noop,left,right,down,click_2_9,left,down,noop"

uv run python -c "import sys; sys.path.insert(0,'prototypes/perc_invdyn'); \
  import clean_data3_tools as T; \
  T.verify_pool('prototypes/perc_invdyn/clean_data3/e3v6m/train2','left,right,up,down,noop,click', context_k=9)"
```
reports 29 scored targets; by action {noop:6, down:6, right:4, up:4, left:5, click *:1 each
(x4)} — collapsing click to the bare verb per `keep_action_params=FALSE` gives {up:4, down:6,
left:5, right:4, click:4, noop:6}.
