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
