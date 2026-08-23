# va6fq — TEST50 held-out pool coverage

Game: **VA6FQ**, falling-sand / liquid cellular automaton, 10x10 grid.
Config: whitelist = `noop,click`; **keep_action_params = TRUE** (the full `click ROW COL`
string is the ID label; row-major, confirmed by the spawned object always appearing at the
clicked cell).

Scored pool (what `verify_pool(test50,'noop,click',context_k=9)` reports):
**51 scored target transitions** across **13 episodes** (curated contiguous slices of freshly
driven trajectories, seed 0). Pool size 51 ∈ 50±2. `balanced_split` returns the whole pool
when pool ≤ `--test-n 50`, so effectively the pool == the test set (every pair is scored).

Freshly driven & distinct from `train/` (see "How this differs from train" below). Sources are
5 independent `autumn_drive.py VA6FQ` runs (A–E) in the scratch dir; every episode is verbatim
rows from one drive.

## 1. Core dynamics (from dynamics.txt)

- **D1** sandButton switch — click red button (0,2) → clickType="sand". No spawn, grid unchanged.
- **D2** waterButton switch — click green button (0,7) → clickType="water". No spawn, grid unchanged.
- **D3** spawn sand — click a free cell while brush=sand → `tan` (solid) Sand appears AT the click.
- **D4** spawn water — click a free cell while brush=water → `skyblue` Water appears AT the click.
- **D5** inert click — click an occupied non-button cell → nothing (no spawn, no brush change).
- **D6** solid-sand straight fall — dry `tan` sand steps straight DOWN 1/step; never sideways.
- **D7** water straight fall — `skyblue` water steps straight down 1/step if below is empty.
- **D8** wetting — dry `tan` sand orthogonally adjacent to Water flips `tan`→`sandybrown` (permanent).
- **D9** liquid diagonal flow — water AND wet sand, when straight-down is blocked, slip one cell
  diagonally toward the nearest reachable hole (empty cell whose own cell-above is empty). No slip
  if no reachable hole; solid (dry) sand NEVER slips.

## 2. Coverage map — every dynamic is a scored TARGET ≥ 4×

Notation: `epE stepA→stepB` = transition in `episode_E`, driven Step A→B. `[loc]` = the clicked
cell (ID-recoverable location). Change tags verified by `classify` on the actual frames.

| Dyn | Scored TARGET pairs (positives) | ID-informative? | FD-informative? |
|---|---|---|---|
| **D1** sandBtn | ep1 15→16 · ep9 1→2 · ep9 4→5 · ep11 14→15 — all `click 0 2`→NO_CHANGE | aliased (see §4) | yes (predict no spawn / brush=sand) |
| **D2** waterBtn | ep3 7→8 · ep9 2→3 · ep10 5→6 · ep12 2→3 — all `click 0 7`→NO_CHANGE | aliased (see §4) | yes (predict no spawn / brush=water) |
| **D3** spawn sand | ep0 0→1 [2,1] · ep1 12→13 [5,6 col6-pit] · ep2 17→18 [5,3 col3-pit] · ep11 15→16 [2,9] — `tan+1` | yes (loc = new tan cell) | yes (needs brush=sand) |
| **D4** spawn water | ep3 8→9 [2,9] · ep5 31→32 [1,9] · ep10 6→7 [2,9] · ep12 3→4 [2,5] — `skyblue+1` | yes (loc = new skyblue cell) | yes (needs brush=water; waterBtn in same-episode window for ep3/ep10/ep12) |
| **D5** inert | ep2 20→21 [8,5] · ep9 0→1 [9,5] · ep9 3→4 [7,4] · ep12 1→2 [6,4] — click occupied→NO_CHANGE | aliased (see §4) | yes (predict no spawn) |
| **D6** solid fall | ep0 1→2,2→3,3→4 (col1) · ep1 13→14 · ep2 18→19 · ep6 26→27,27→28,28→29 (col0) · ep11 16→17,17→18 (~10 total) | yes (noop: a grain steps down, no new cell) | yes (grain shifts down 1) |
| **D7** water fall | ep3 9→10 · ep4 12→13,13→14,14→15,16→17 · ep5 32→33,33→34 · ep7 35→36,36→37,37→38 · ep8 47→48 · ep10 7→8 · ep12 4→5 (~12) | yes (noop; water steps down, no new cell) | yes (water shifts down 1) |
| **D8** wetting | ep4 15→16 (wet 9,9) · ep4 16→17 (wet 8,7) · ep4 17→18 (wet 9,7) · ep7 38→39 · ep8 48→49 · ep8 49→50 — `sandybrown+1 tan-1` (6) | yes (noop) | yes (tan→sandybrown next to water) |
| **D9** diag slip | ep4 15→16 water (8,9)→(8,8) [clean single-drop] · ep7 38→39 [col9 2-tall double-slip] · ep8 48→49 · ep8 49→50 [col0 2-tall] (4 pos) | yes (noop; cells move, none spawned) | yes (liquid slips into hole) |

### Contrastive negatives (15 / 51 = 29%, req. 20–30%)

| Negative | Pairs | Shortcut it defeats |
|---|---|---|
| non-spawn clicks (D1/D2/D5) | 12 clicks → NO_CHANGE (see D1/D2/D5 rows) | "every click adds an object" |
| **brush-conditioned same cell** | `click 2 9`→**skyblue** (ep3 8→9, ep10 6→7, brush=water) vs `click 2 9`→**tan** (ep11 15→16, brush=sand) | coordinate memorization; forces hidden brush state |
| settled-NC noop | ep1 14→15 · ep2 19→20 → NO_CHANGE | "noop always moves a cell" / step-parity motion clock |
| **D9-refuse (D6/D9 contrast)** | ep6 29→30: DRY grain at (8,0) rests on obstacle (9,0), IGNORES reachable hole (9,1) → NO_CHANGE | "blocked-below liquid rule applies to solids too" — same blocked+hole cue, solid stays vs liquid slips (ep4/7/8) |

**Step-clock defeated (the whole point for va6fq).** Verified over the pool: for every k∈2..7,
*every* residue class `Step % k` contains BOTH clicks and noops (no split), and 10 distinct
leaked `Step:` values map to ≥2 different action labels across episodes (e.g. Step 0 → `click 2 1`
or `click 9 5`; Step 2 → `click 0 7` or `noop`). So no `step % k → click` rule beats the true
spawn-location rule. Because clicks are labelled with full coordinates (keep_action_params), a
clock could at best yield "a click" — pinning `R C` still needs the real rule; and mixed noop
outcomes (moves vs NO_CHANGE at all parities) make FD immune too.

## 3. Action histogram (pool of 51)

```
noop        31
click 0 2    4   (D1 sandButton, fixed cell)
click 0 7    4   (D2 waterButton, fixed cell)
click 2 9    3   (2× water spawn + 1× sand spawn = brush contrast)
click 2 1    1   (D3)        click 1 9   1  (D4)
click 5 6    1   (D3)        click 2 5   1  (D4)
click 5 3    1   (D3)        click 8 5   1  (D5)
click 9 5    1   (D5)        click 7 4   1  (D5)   click 6 4  1  (D5)
```
20 clicks over 12 distinct locations (spread across cols 0–9, rows 0–9, buttons + pits + edge
lanes + occupied pile). 31 noops carry the 4 passive dynamics (D6–D9) plus 3 NO_CHANGE negatives.

## 4. Uncoverable / aliased — minimized & documented (req. 4)

- **NO_CHANGE ID-aliasing.** D1 (sandBtn), D2 (waterBtn) and D5 (inert) all leave the grid
  *identical*, so under inverse-dynamics they are mutually aliased with each other and with a
  settled noop — an oracle cannot recover the click coordinates or tell them from `noop`. This is
  intrinsic (button/inert clicks have no visible effect) and is exactly why they are used as the
  FD-informative contrastive negatives. Minimized to 15/51 = 29% of the pool; the other 71% are
  ID-recoverable (spawn ⇒ location = clicked cell; motion ⇒ noop).
- **Clean single-cell slips (D9) are geometry-limited.** A pristine, single-cell diagonal slip is
  only isolable in a pile-*buffered* edge lane (col0 or col9, whose inner neighbour column is
  empty so the falling drop wets nothing until it reaches the obstacle). col0 replicates the train
  slip, so only **col9 gives a clean single-drop slip (ep4 15→16)**. The other 3 D9 positives use
  2-tall obstacles (ep7 col9, ep8 col0) and are "busy" — the wetted top-obstacle grain slips in the
  same tick as the water — genuine D9 in a varied situation but multi-cell. Any slip elsewhere
  touches the porous pile (gaps at cols 3,6) and triggers a wetting cascade. The ID label stays a
  clean `noop` in every case (cells move, none spawned); only FD legibility is reduced.
- **Fixed buttons.** D1/D2 can only ever be clicked at (0,2)/(0,7); their "varied situations" are
  varied surrounding grid state + timing, not varied click location.

## 5. How this TEST differs from `train/`

`train/` is sliced from `train_regen` (single col0 lane). This TEST50 is 5 fresh drives with
different situations under the same rules:

| | train (train_regen) | test50 |
|---|---|---|
| sand-fall lanes | col0 (2,0), col3-pit (4,3) | col1 (2,1), col8 (3,8 in source), col6-pit (5,6), col3-pit (5,3), col9 (2,9) |
| water spawn cells | (2,0) col0 | (2,9),(1,9),(2,5) — different cells & lanes |
| diagonal slip | col0 obstacle, slip **RIGHT** (8,0)→(8,1)→(9,1), 1-tall | col9 slip **LEFT** (8,9)→(8,8) [1-tall]; col9 & col0 **2-tall double-slips** (slip at row 7, not row 8) |
| inert clicks | (9,0) col0 grain | (8,5),(9,5),(7,4),(6,4) — pile cells |
| brush contrast | `click 2 0`→tan/skyblue | `click 2 9`→tan/skyblue (different cell) |
| timing | clicks steps 0,8,16,18,23,26,36,38 | 5 drives, irregular gaps, mixed parities; same Step# ↦ different actions across episodes |

Object positions, obstacle heights, slip directions/rows, click coordinates and timing are all
different — a genuine cross-trajectory generalization test.

---
Additive only: `train/`, `test/`, `dynamics.txt`, `COVERAGE.md` are unchanged.
Rebuild: 5 `autumn_drive.py VA6FQ` runs (A–E, seed 0) → curated slices; verify with
`clean_data3_tools.verify_pool('.../test50','noop,click',context_k=9)` → 51 targets.
