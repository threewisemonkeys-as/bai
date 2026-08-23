# e3v6m — TEST50 coverage (large held-out test pool)

Game: **e3v6m** ("Lights" beam). Whitelist = `left,right,up,down,noop,click`;
`keep_action_params = FALSE` (the click label collapses to the bare verb `click`).
Pool = **50 scored target transitions** across 5 episodes (verified with
`verify_pool(..., context_k=9)`). All trajectories freshly driven with
`autumn_drive.py E3V6M` (seed 0); slices are verbatim rows of those drives.

## 1. Core dynamics (from dynamics.txt)

- **D1 click → toggle.** Any `click ROW COL` flips `turnedOn` (white↔yellow); the
  location is irrelevant; position/orientation unchanged.
- **D2 move while OFF (white) → TRANSLATE** one cell (up=row−1, down=row+1,
  left=col−1, right=col+1); orientation unchanged.
- **D3 move while ON (yellow) → ROTATE in place** about the anchor
  (up: dir+1, down: dir−1, left: dir+2, right: dir−2, mod 8); centroid unchanged.
  Visually the symmetric 5-cell line has 4 distinct orientations (dir mod 4):
  HORIZ(0), DIAG\\(1), VERT(2), ANTI/(3).
- **D4 noop → nothing.** No handler; no passive/clock dynamics at all (D5).

The defining structure is the **state-conditional action semantics** (D2 vs D3): the
same arrow key translates when white and rotates when yellow.

## 2. How TEST50 differs from train (cross-trajectory generalization)

Checked programmatically: **zero** scored (full-grid-state, verb) pairs are shared
with `train/`, and no two test items are identical either.

| aspect | train | test50 |
|---|---|---|
| OFF translations | all in DIAG\\ orientation, anchors confined to the (5–7, 5–7) box | all four orientations translated (DIAG\\, VERT, HORIZ, ANTI/), anchors spread over rows 5–11, cols 4–11 |
| ON rotations | all at anchor (7,7) | at 5 different anchors: (5,8), (9,5), (8,9), (6,5), (10,11) |
| toggles | 2 clicks, both at `click 7 7` (on-anchor), beam at (7,7) | 9 clicks at 9 distinct locations — far-empty cells ((1,13), (12,2), (15,15), (0,0), (13,4)) and on-beam cells (anchor (5,8)/(8,9)/(6,5), beam tip (9,7)) — proving location-irrelevance |
| noops | 2, both at (7,7) DIAG\\ | 9, spread over positions/orientations/colors (white DIAG×3, white HORIZ, yellow DIAG×2, yellow VERT, yellow ANTI×2) |
| slice shape | twenty 2-row slices (no window context) | 12/12/12/12/7-row slices (windows up to 9 within episode) |

## 3. Episodes and per-pair coverage

Each row of a slice stores the action taken FROM that frame. Anchor = middle cell.

**episode_0** (drive A, 12 rows, 11 targets) — north; DIAG→VERT→ANTI via up-ON:
| pair | action | regime | covers |
|---|---|---|---|
| 0→1 | right | OFF DIAG (7,7)→(7,8) | D2 right |
| 1→2, 2→3 | up ×2 | OFF DIAG →(5,8) | D2 up |
| 3→4 | noop | white DIAG (5,8) | D4 negative |
| 4→5 | click 1 13 | W→Y, far-empty cell | D1 |
| 5→6 | up | ON: DIAG→VERT | D3 up (+1) |
| 6→7 | up | ON: VERT→ANTI | D3 up (+1) |
| 7→8 | noop | yellow ANTI | D4 negative (ON state persists) |
| 8→9 | down | ON: ANTI→VERT | D3 down (−1) |
| 9→10 | click 5 8 | Y→W, on-anchor | D1 |
| 10→11 | right | OFF **VERT** (5,8)→(5,9) | D2 right, non-train orientation |

**episode_1** (drive B, 12 rows, 11 targets) — southwest; HORIZ/ANTI via down-ON:
| pair | action | regime | covers |
|---|---|---|---|
| 0→1 | down | OFF DIAG (7,7)→(8,7) | D2 down |
| 1→2, 2→3 | left ×2 | OFF DIAG →(8,5) | D2 left |
| 3→4 | down | OFF DIAG →(9,5) | D2 down |
| 4→5 | click 12 2 | W→Y, empty SW cell | D1 |
| 5→6 | down | ON: DIAG→HORIZ | D3 down (−1) |
| 6→7 | down | ON: HORIZ→ANTI | D3 down (−1) |
| 7→8 | up | ON: ANTI→HORIZ | D3 up (+1) |
| 8→9 | click 9 7 | Y→W, on beam TIP | D1 |
| 9→10 | noop | white **HORIZ** (9,5) | D4 negative |
| 10→11 | left | OFF **HORIZ** →(9,4) | D2 left, non-train orientation |

**episode_2** (drive C rows 1–12, 12 rows, 11 targets) — east; the aliased left/right-ON pair; ANTI translations:
| pair | action | regime | covers |
|---|---|---|---|
| 0→1, 1→2 | right ×2 | OFF DIAG (8,7)→(8,9) | D2 right |
| 2→3 | click 15 15 | W→Y, far corner | D1 |
| 3→4 | left | ON: DIAG→ANTI (dir 1→3) | D3 left (+2) *aliased* |
| 4→5 | noop | yellow ANTI | D4 negative |
| 5→6 | left | ON: ANTI→DIAG (dir 3→5) | D3 left (+2) *aliased* |
| 6→7 | right | ON: DIAG→ANTI (dir 5→3) | D3 right (−2) *aliased* |
| 7→8 | click 8 9 | Y→W, on-anchor | D1 |
| 8→9 | right | OFF **ANTI** (8,9)→(8,10) | D2 right, non-train orientation |
| 9→10, 10→11 | down ×2 | OFF ANTI →(10,10) | D2 down |

**episode_3** (drive D, 12 rows, 11 targets) — northwest; noop-heavy, irregular timing:
| pair | action | regime | covers |
|---|---|---|---|
| 0→1 | left | OFF DIAG (7,7)→(7,6) | D2 left |
| 1→2 | up | OFF DIAG →(6,6) | D2 up |
| 2→3 | noop | white DIAG (6,6) | D4 negative |
| 3→4 | left | OFF DIAG →(6,5) | D2 left |
| 4→5 | click 0 0 | W→Y, corner (0,0) | D1 |
| 5→6 | noop | yellow DIAG (6,5) | D4 negative |
| 6→7 | up | ON: DIAG→VERT | D3 up (+1) |
| 7→8 | noop | yellow VERT | D4 negative |
| 8→9 | down | ON: VERT→DIAG | D3 down (−1) |
| 9→10 | click 6 5 | Y→W, on-anchor | D1 |
| 10→11 | left | OFF DIAG →(6,4) | D2 left |

**episode_4** (drive E rows 7–13, 7 rows, 6 targets) — far southeast (ferry rows excluded from the slice):
| pair | action | regime | covers |
|---|---|---|---|
| 0→1 | noop | white DIAG (11,10) | D4 negative |
| 1→2 | up | OFF DIAG →(10,10) | D2 up |
| 2→3 | right | OFF DIAG →(10,11) | D2 right |
| 3→4 | click 13 4 | W→Y, empty SW cell | D1 |
| 4→5 | noop | yellow DIAG (10,11) | D4 negative |
| 5→6 | right | ON: DIAG→ANTI (dir 1→7) | D3 right (−2) *aliased* |

## 4. Per-dynamic totals (all as SCORED targets, FD- and ID-informative)

| dynamic | positives | contrastive negatives |
|---|---|---|
| D1 click toggle | 9 (5 W→Y, 4 Y→W; 9 distinct click locations, on-beam and far-empty) | position/orientation never change on click (every pair); toggles at 5 different beam states |
| D2 translate-OFF | 20 (up 4, down 4, left 6, right 6; in all 4 orientations) | the 12 same-verb ON pairs, where the beam does NOT translate |
| D3 rotate-ON | 12 (up 4, down 4, left 2, right 2; at 5 anchors) | the 20 same-verb OFF pairs, where the beam does NOT rotate |
| D4 noop = nothing | 9 (18%; both colors, 4 orientations, 7 positions) | every non-noop pair (something always changes) |

Verb histogram (the ID label space): **up 8, down 8, left 8, right 8, click 9, noop 9 = 50.**
Every arrow verb appears in BOTH regimes, so "up always translates" (or "always
rotates") misclassifies half its items — the D2/D3 conditional is the near-miss
structure of this game, on top of the 9 noop nulls. Noop timing is irregular
(gaps of 1–4 actions, at drive starts/middles/ends), so no step-parity clock fits.

## 5. Known limitation: left-ON / right-ON aliasing (inherent)

The beam is a symmetric 5-cell line, so orientation `dir` renders identically to
`dir+4`. left-ON (+2) and right-ON (−2 ≡ +6) therefore produce the SAME visible
orientation change (DIAG↔ANTI or HORIZ↔VERT) — no oracle can separate them from
the frames. Kept at the minimum needed for regime coverage: **4 aliased items**
(ep2 pairs 3→4, 5→6, 6→7; ep4 pair 5→6), i.e. an oracle ID ceiling of ~48/50 (96%),
expected oracle score ≈ 46/50 with 50/50 guessing between left/right on those.
They remain fully FD-informative (the rule "left/right while ON flips the diagonal"
must be known to predict X_t+1) and fully distinguishable from up/down/click/noop.
up-ON vs down-ON is NOT aliased (±1 mod 4 differ) and carries the unambiguous
rotation coverage (8 items).

Everything else in the game is coverable and covered; there are no passive
dynamics, spawns, collisions, or terminations to test (D5).

## 6. Verification

```
uv run python -c "import sys; sys.path.insert(0,'prototypes/perc_invdyn'); \
  import clean_data3_tools as T; \
  T.verify_pool('prototypes/perc_invdyn/clean_data3/e3v6m/test50','left,right,up,down,noop,click', context_k=9)"
```
reports 50 scored targets; by verb up 8 / down 8 / left 8 / right 8 / click 9 / noop 9;
8 OFF-moves per direction-pair show `white~move(±1,0)/(0,±1)`, the 12 ON-moves show
`yellow~recolor` (in-place reorientation), clicks show `white∓5 yellow±5`, noops
`NO_CHANGE`. Duplicate check: 0 scored (state, verb) pairs shared with `train/`,
0 internal duplicates. Windows (context_k=9) extend to the slice boundaries.
