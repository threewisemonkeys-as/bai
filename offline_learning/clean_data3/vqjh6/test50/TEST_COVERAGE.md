# vqjh6 — TEST50 held-out coverage

Large held-out test pool for **vqjh6** (gravity / blob sandbox, 17x17). Whitelist
`noop,click`, **keep_action_params = TRUE** (the full `click ROW COL` string is the ID
label — unlike `train/`, which was curated for the movement-game keep=FALSE config, here
the click LOCATION must be recoverable). All trajectories are **freshly driven** from a
clean seed-0 reset with `autumn_drive.py VQJH6` (row-major `click_ROW_COL`), then sliced
verbatim — action sequences, spawn locations, button order and blob configurations are all
distinct from `train/` (a cross-trajectory generalization test).

Verified: `T.verify_pool('.../clean_data3/vqjh6/test50','noop,click',context_k=9)` →
**50 scored target transitions** (exactly; balanced_split at `--test-n 50` keeps all).

## 1. Core dynamics (from dynamics.txt)

- **D1 Passive gravity motion:** every step all blobs move 1 cell in the current gravity
  direction (default "down"); gravity persists until a button is clicked.
- **D2/D3/D4/D5 Gravity buttons:** clicking leftButton red (8,0) / rightButton darkorange
  (8,16) / upButton gold (0,8) / downButton green (16,8) sets gravity left/right/up/down.
- **D6 One-tick lag:** on the button-click tick blobs still move per the OLD gravity; the
  new direction takes effect the next tick.
- **D7 Spawn:** clicking a free non-button cell (R,C) spawns a 2x2 blue blob on the next
  observation at rows {R-1,R} x cols {C,C+1} (it does not move on its spawn tick).
- **D8 Occupied-cell click → no effect** (spawn only fires on free cells).
- **D12 Off-grid attrition:** no despawn rule, but blobs slide off the grid edge under
  gravity (blue count falls 2 then 2).
- (D10 arrow actions are no-ops and are outside the `noop,click` whitelist; D13 no
  win/termination.)

## 2. Pool composition (4 drives → 9 episodes → 50 targets)

Action histogram: **23 click / 27 noop**; **18 distinct click locations** — 12 spawn
clicks (all distinct cells, spread rows 1–14 x cols 1–14), 4 buttons ((8,16)x3, (0,8)x2,
(8,0)x2, (16,8)x2), 2 occupied-cell clicks. Noop FD direction balance: **down 8 / right 8
/ up 6 / left 4 / NO_CHANGE(empty) 1** — all four directions are repeatedly scored, so a
constant "blobs fall down" rule fails 19/27 noops.

| ep | drive slice | targets (action → change / role) |
|---|---|---|
| 0 | A `[0-6]` | `click 3 2` **D7+** (empty grid); 2x `noop` down D1; `click 8 16` **D3 cause + D6 lag** (moves DOWN); `noop` right D3; `click 12 6` **D7+ under RIGHT gravity** (compound: spawn + other blob moves) |
| 1 | A `[8-14]` | `noop` right; `click 0 8` **D4 cause + D6** (moves RIGHT); 2x `noop` up D4; `click 8 0` **D2 cause + D6** (moves UP); `noop` left D2 |
| 2 | A `[15-18]` | `noop` left; `click 16 8` **D5 cause + D6** (moves LEFT); `noop` down D5 |
| 3 | B `[0-3]` | `click 2 12` **D7+** (empty); `noop` down; `click 10 1` **D7+** (second blob) |
| 4 | B `[5-11]` | `click 6 7` **D7+** (3-blob scene); `noop` down; **`click 7 7` D8− OCCUPIED** (cell inside the blob just spawned by `click 6 7` — adjacent minimal pair); `noop` down; `click 13 3` **D7+ compound** (blue+4 spawn while another blob half-exits bottom, net +2); `noop` down **D12** (full exit, blue−2) |
| 5 | C `[0-11]` | `click 14 11` **D7+**; `click 0 8` **D4+D6** (clicked 1 tick after spawn — moves DOWN); `noop` up; `click 9 14` **D7+ under UP gravity**, beside rightButton; 2x `noop` up; **`click 10 12` D8− OCCUPIED under UP**; `noop` up; `click 8 16` **D3+D6** (moves UP); `noop` right; `click 5 5` **D7+ compound** (spawn + another blob half-exits right, net +2) |
| 6 | C `[12-15]` | `noop` right (blob covers darkorange button: `darkorange-1`); `click 16 8` **D5+D6** (moves RIGHT, +D12 exit blue−2); `noop` down |
| 7 | D `[2-6]` | **`click 8 16` D3 cause on EMPTY grid → NO_CHANGE** (recovered only from ctx_next: a blob spawned 2 ticks later moves RIGHT, revealing gravity persistence); `noop` empty **NO_CHANGE D1−**; `click 7 4` **D7+**; `noop` right (the reveal) |
| 8 | D `[7-15]` | `noop` right; `click 1 10` **D7+ top edge**, beside upButton; 2x `noop` right; `click 8 0` **D2+D6** (moves RIGHT); `noop` left; `click 12 9` **D7+ under LEFT gravity**; `noop` left |

## 3. Per-dynamic scored-target coverage (positives / negatives)

| Dynamic | Positives (scored) | Contrastive negatives |
|---|---|---|
| **D7 click-spawn** | **12** spawns, 12 distinct free cells, under all four gravities (down x6, right x4, up x1, left x1; 4 of them on an empty grid); ID label = the anchor where the 2x2 appears (`blue+4`, or `blue+2` compound with attrition) | **11** clicks with NO spawn at the clicked cell: 9 button clicks + 2 occupied-cell clicks → "click ⇒ blob at click location" fails 11/23 clicks |
| **D1 passive motion** | **26** noop targets where every blob moves exactly 1 cell in the current gravity direction (8 down / 8 right / 6 up / 4 left, in 1–4-blob scenes) | **1** empty-grid `noop` NO_CHANGE (nothing to move; ep7 also has the NC button-click pair); direction mix defeats "always down" |
| **D2 left** | leftButton clicked **2x** (from gravity=up in ep1, from gravity=right in ep8) + **4** left-move noops | every other click (buttons at other cells, spawns) + 23 non-left noops |
| **D3 right** | rightButton clicked **3x** (from down, from up, on empty grid) + **8** right-move noops | — |
| **D4 up** | upButton clicked **2x** (from down x2, different blob configs, once 1 tick after a spawn) + **6** up-move noops | — |
| **D5 down** | downButton clicked **2x** (from left in ep2, from right in ep6) + **8** down-move noops | down is the default: the button's effect is only credited via the lag pair + ctx |
| **D6 one-tick lag** | **8** button-click targets with blobs on grid, each showing the OLD-direction move on the click tick (down, right x2, up x2, left x2, right-with-exit); FD must apply the lag rule or predicts the wrong direction | the following noop in the same slice shows the NEW direction — a no-lag rule gets the pair wrong, a lag rule gets both right |
| **D8 occupied no-spawn** | — (it IS a negative) | **2** occupied-cell clicks: `click 7 7` (gravity down, on the blob spawned 2 ticks earlier by `click 6 7` — near-identical location, opposite outcome) and `click 10 12` (gravity up) |
| **D12 attrition** | **3** targets where a blob slides off an edge (bottom exit noop blue−2 in ep4; right-edge partial exits inside `click 5 5` and `click 16 8` pairs) + 2 partial-exit compounds | no removal rule to contrast; documented as pure D1 side-effect |

**Key contrasts:**
- **Direction monoculture (the train gap):** 19 of 27 noops move in a non-down direction
  or not at all; every direction appears ≥4x as a scored FD target with its setting click
  in ctx_prev.
- **"Click ⇒ spawn at click location":** 11/23 clicks are non-spawning (buttons/occupied),
  and 2 spawns are compounds where blue count changes by +2 not +4.
- **Lag:** all 8 blob-visible button clicks move the blobs the OLD way on the click tick;
  timing of clicks is irregular (gaps of 0–3 noops, incl. a click 1 tick after a spawn),
  killing step-parity shortcuts.

## 4. How TEST50 differs from train

- **Different label semantics:** train was curated for keep=FALSE (click collapses to the
  verb); TEST50 is built for **keep=TRUE** — 18 distinct `click R C` labels whose location
  must be read off the frame change (spawn anchor) or the ctx-visible gravity change
  (which button).
- **Fresh drives, new situations:** train's single trajectory used spawns at (3,8)/(8,8),
  occupied (8,5), button order left→up→right→down, single-blob scenes. TEST50 uses 12 new
  spawn cells, occupied clicks at new cells under two gravities, button orders
  right→up→left→down / up→right→down / right→left, and 2–4-blob scenes.
- **New state types never scored in train:** spawns while gravity is right/left/up (train:
  down only), spawn 1 tick before a button click, spawns at the top edge and beside
  buttons, blob passing OVER the darkorange button (`darkorange-1`/`+1`), off-grid
  attrition at bottom and right edges, empty-grid button click later revealed by a spawned
  blob (train's empty-grid click was never revealed).

## 5. Not covered / caveats

- **D8 occupied clicks are ID-aliased with `noop`** (2/50 = 4%): a click that spawns
  nothing leaves no trace, so the pair is identical to a noop move. Included for FD value
  (predict: NO new blob despite click) and as the free-vs-occupied condition on D7;
  aliasing is intrinsic and minimized (nrdf6 precedent: 6%).
- **`click 8 16` on the empty grid (ep7) is pair-invisible** (NO_CHANGE): recoverable
  only from ctx_next (blob spawned at +2 moves right at +3, and noops cannot change
  gravity). An oracle using the window gets it; a pair-only decoder cannot. 1/50 item,
  kept as the gravity-persistence probe.
- **A few slice-start noops have empty ctx_prev under non-default gravity** (ep1 first
  right-noop, ep6 first right-noop, ep8 first right-noop): ID is unaffected (a pure move
  is a noop), but FD cannot know the current direction for that first pair. Intrinsic to
  slicing; 3/50 items.
- **D10 (arrow no-ops):** outside the `noop,click` whitelist — untestable by
  construction, exactly as in train/COVERAGE.md.
- **Blob-overlap states** (two blobs sharing cells) were deliberately avoided — the SEXP
  has no collision rule and overlapping renders are illegible.

viz.html is a filmstrip of the whole pool (9 episodes with separators).
