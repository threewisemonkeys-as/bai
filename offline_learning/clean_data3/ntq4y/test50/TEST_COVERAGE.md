# ntq4y — TEST50 held-out pool coverage

Large held-out test pool for the sandbox water-painter **ntq4y**. Whitelist = `noop,click`;
**keep_action_params = TRUE** (the full `click ROW COL` string is the ID label). Actions/obs
use (row, col); dynamics.txt writes positions as (col, row).

Pool = **50 scored target transitions** (`verify_pool(..., context_k=9)`), across 8 freshly
DRIVEN episodes (`autumn_drive.py NTQ4Y ...`, seed 0). This is a *cross-trajectory*
generalization test: placements, water configurations and action timing are DISTINCT from
`train/` (see §4). Additive only — `train/`, `test/`, `dynamics.txt`, `COVERAGE.md` untouched.

Two objectives (recap):
- **ID**: recover the hidden action from `X_t -> X_t+1`. Recoverable when the change reveals
  the action (placed cell at the click loc; a unique mass-delete signature per button).
- **FD**: predict `X_t+1` from `X_t` + action. Informative when the action actually changes
  the state and the change needs the rule (placed color, water motion, mass deletion).

## 1. CORE dynamics (from dynamics.txt)

- **D1 — place particle on a free cell.** `click` an empty cell places ONE particle of the
  selected type at the clicked coords; color = currentParticle (vessel=purple / plug=orange /
  water=blue).
- **D2 — toolbar selection.** `click` a row-0 button (vessel col2 / plug col5 / water col8)
  sets the hidden `currentParticle`. NO grid change on the click; effect shows only as the
  COLOR of the next placed particle.
- **D3 — removeButton (0,11).** deletes ALL plug (orange) cells.
- **D4 — clearButton (0,14).** deletes ALL vessels + plugs + water (funnel walls included).
- **D5 — occupied / non-free click no-op.** `click` an occupied non-button cell does nothing
  (isFreePos guard). The D1 negative.
- **D6 — passive water sim (every step).** Each water cell falls straight down if the cell
  below is free, else steps diagonally toward the nearest reachable hole, else stays. Runs on
  EVERY step (noop incl.), independent of the action.
- **D7 — drain on unplug.** Removing the plugs (D3) opens the funnel bottom so water held
  above the plugs drains down. (D3 then D6 composed; precondition = plugs gone.)

(No color changes; no spawn/despawn beyond clicks/buttons; reward always 0 — nothing to score
there.)

## 2. Per-dynamic coverage (episode → transition → role)

| Ep (theme) | transition (action → change) | dyn / role |
|---|---|---|
| **0** placements + occupied | `click 1 6`→purple+1 · `click 2 9`→purple+1 | D1 vessel ×2 (pos) |
| | `click 1 6`→NC · `click 10 12`→NC | **D5** occupied ×2 (neg; `click 1 6` = same label as the place above) |
| **1** plug select/place/remove | `click 0 5`→NC | **D2** select-plug (FD via t1/t2 color) |
| | `click 5 7`→orange+1 · `click 6 11`→orange+1 | D1 plug ×2 (pos) |
| | `click 5 7`→NC | **D5** occupied plug (neg; same label as place) |
| | `click 0 11`→orange-8 | **D3** removeButton (pos; 6 orig + 2 placed) |
| | `click 0 11`→NC | **D3 negative** (remove with no plugs left) |
| **2** water fall | `click 0 8`→NC | D2 select-water |
| | `click 2 7`→blue+1 | D1 water (pos) |
| | `noop`→blue↓ | **D6** water fall (pos) |
| **3** drain event #1 | `click 0 8`→NC | D2 select-water |
| | `click 12 7`→blue+1 · `click 12 8`→blue+1 | D1 water ×2 (pos) |
| | `noop`→blue settle | D6 water motion (pos) |
| | `noop`→NC | **D6/D7 negative** (water present, plugs IN → confined static) |
| | `click 0 11`→orange-6 | **D3** removeButton (pos) |
| | `noop`→blue shift · `noop`→blue↓ · `noop`→blue↓ | **D7** drain ×3 (pos) |
| **4** fall + clear | `click 0 8`→NC | D2 select-water |
| | `click 3 7`→blue+1 | D1 water (pos) |
| | `noop`→blue↓ · `noop`→blue↓ | **D6** water fall ×2 (pos) |
| | `click 0 14`→purple-12 orange-6 blue-1 | **D4** clearButton (pos) |
| | `noop`→NC | **D6 negative** (empty grid) |
| **5** clear-and-rebuild | `click 0 2`→NC | **D2** select-vessel (explicit button; train used default only) |
| | `click 2 4`→purple+1 | D1 vessel (pos) |
| | `click 0 14`→purple-13 orange-6 | **D4** clearButton (pos) |
| | `noop`→NC | D6 negative (empty) |
| | `click 5 5`→purple+1 | D1 vessel on EMPTY grid (novel state) |
| | `click 0 5`→NC | D2 select-plug |
| | `click 5 6`→orange+1 | D1 plug on empty grid |
| | `click 0 14`→purple-1 orange-1 | **D4** clearButton (pos; clears the 2 placed) |
| **6** drain event #2 | `click 0 8`→NC | D2 select-water |
| | `click 11 7`→blue+1 · `click 11 8`→blue+1 | D1 water ×2 (pos; row-11 config ≠ ep3) |
| | `noop`→blue settle · `noop`→blue settle | **D6** water motion ×2 (pos) |
| | `click 0 11`→orange-6 | **D3** removeButton (pos) |
| | `noop`→blue shift · `noop`→blue↓ | **D7** drain ×2 (pos) |
| **7** place/occupied/remove/clear | `click 0 5`→NC | D2 select-plug |
| | `click 2 5`→orange+1 | D1 plug (pos) |
| | `click 2 5`→NC | **D5** occupied (neg; same label as place) |
| | `click 0 11`→orange-7 | **D3** removeButton (pos; 6 orig + 1 placed) |
| | `click 0 14`→purple-12 | **D4** clearButton (pos; plugs already gone) |
| | `noop`→NC | D6 negative (empty) |

### Coverage counts (positives / negatives), every core dynamic ≥ 4 as a TARGET

| Dynamic | positive targets | negative / near-miss targets |
|---|---|---|
| D1 place | **14** (vessel 4, plug 4, water 6) | — (its negative is D5) |
| D2 selection | **8** selects, each with a same-color placement inside its window (FD) | (own step is NC by design; not ID-scorable — see §5) |
| D3 removeButton | **4** (ep1,3,6,7) | **1** remove-with-no-plugs (ep1 t5) |
| D4 clearButton | **4** (ep4, ep5 ×2, ep7) | — |
| D5 occupied no-op | — | **4** (ep0 ×2, ep1, ep7); 3 reuse a placement's exact label |
| D6 water sim | **6** water-motion noops (varied offsets) | **4** no-motion noops (1 confined-static + 3 empty-grid) |
| D7 drain on unplug | **5** (ep3 ×3, ep6 ×2) | (contrasted by ep3/ep6 confined-static noop before the remove) |

Pure near-miss negatives (D5 + D3-neg + D6-neg) = **9 / 50 ≈ 18 %**. Adding the 8 D2 select
clicks (also NO_CHANGE, i.e. "a click that places nothing") → 17/50 (34 %) NO_CHANGE targets;
the remaining **33/50 (66 %) are visibly-changing, ID-informative** targets.

## 3. Action histogram (pool)

`noop`:15 · `click 0 11`:5 · `click 0 8`:4 · `click 0 14`:4 · `click 0 5`:3 · `click 1 6`:2 ·
`click 5 7`:2 · `click 2 5`:2 · `click 2 9`:1 · `click 10 12`:1 · `click 6 11`:1 ·
`click 2 7`:1 · `click 12 7`:1 · `click 12 8`:1 · `click 3 7`:1 · `click 0 2`:1 ·
`click 2 4`:1 · `click 5 5`:1 · `click 5 6`:1 · `click 11 7`:1 · `click 11 8`:1

20 distinct click labels + noop. Placement clicks span the grid (rows 1-12, cols 3-13) at
distinct cells; the 5 fixed toolbar buttons (0 2 / 0 5 / 0 8 / 0 11 / 0 14) necessarily recur
(their positions are hard-coded in the SEXP).

### Contrastive shortcuts each set defeats
- **"click always places"** → D5 occupied clicks (ep0 `click 1 6`, ep1 `click 5 7`, ep7
  `click 2 5` each reuse a placement's own label but do nothing) + the 8 button clicks that
  place nothing.
- **"noop always moves water" / step%k clock** → D6 no-motion noops at varied step offsets
  (confined-static + empty-grid) vs D6 water-motion noops at varied offsets.
- **D7 precondition** → ep3/ep6 each pair a confined-static noop (plugs IN) against draining
  noops (plugs OUT); the only intervening event is the removeButton, so drain must be
  conditional on unplug, not a clock.
- **"removeButton always deletes plugs"** → ep1's second `click 0 11` is NC (no plugs left).
- **D2 color** → the only way to predict a placed particle's COLOR under FD is to track the
  last toolbar button in the window (the selection rule), not guess.

## 4. How TEST50 differs from train/

- **Placement cells are disjoint from train.** Train placed vessel@(2,2), plug@(4,6),
  water@(10,7). TEST50 places vessel@{(1,6),(2,9),(2,4),(5,5)}, plug@{(5,7),(6,11),(5,6),(2,5)},
  water@{(2,7),(12,7),(12,8),(3,7),(11,7),(11,8)} — no overlap.
- **Denser mass-event coverage.** Train had 1 remove, 1 clear, 1 drain; TEST50 has 4 removes,
  4 clears, **2** drain events with different water configs (row-12 vs row-11 placements).
- **Novel states not present in train:** post-clear *rebuild on the empty grid* (ep5), an
  explicit vessel-button select `click 0 2` (train relied on the default vessel), the
  remove-with-no-plugs negative (ep1), and same-label place→occupied contrasts (ep0, ep7).
- **Randomized noop-run lengths** (1 / 2 / 3), no fixed cadence, so a `step%k` clock is worse
  than the real conditional rules.
- Toolbar-button labels (0 2 / 0 5 / 0 8 / 0 11 / 0 14) are shared with train because the
  buttons are fixed cells — unavoidable and not a leakage of the scored *placement/water*
  situations.

## 5. Known limits / uncoverable

- **ID aliasing among NO_CHANGE targets is inherent.** The 8 D2 selection clicks, 4 D5
  occupied clicks, the 1 D3-negative, and the 4 D6-negative noops all render as the identical
  NO_CHANGE frame and are mutually ID-indistinguishable *by construction* — the SEXP hides
  `currentParticle` in a String state, and an occupied-click / empty-noop produces no pixel
  change. An ID oracle cannot separate them; they are the contrastive negatives / FD-only
  carriers, not ID signal-bearers. The 33 visibly-changing targets keep ID informative.
- **Button ID is by change-TYPE, not location.** removeButton/clearButton effects are global
  (mass deletion), not at the clicked cell. They are still ID-recoverable because each
  mass-change signature maps to exactly one fixed button label (a bijection), but the click
  location itself is not "where the effect appears".
- **D2 selection is not directly ID-scorable** (its own step is NC). It is covered under **FD
  via the downstream placement COLOR**, with the button click guaranteed inside the
  placement target's window (verified — e.g. ep3's select sits at prev=8 of the placement).
