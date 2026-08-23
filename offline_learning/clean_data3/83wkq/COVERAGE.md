# 83wkq — clean_data3 coverage

Config: whitelist = `noop,click`; `keep_action_params=TRUE` (click LOCATION is part of the
label, e.g. `click 0 3`). Move verbs (left/right/up/down) exist in the trajectory but are NOT
whitelisted; they drop out. Original train = `clean_data2/83wkq/train/episode_0` (71 rows,
steps 0..70). Move-verb rows (dropped, and would truncate windows) sit at steps 12, 21, 34, 43
— every curated slice avoids them so all internal pairs are scored and windows are intact.

## 1. CORE dynamics extracted from dynamics.txt

- **D1 — Spawn at click cell.** `click ROW COL` adds ONE new blue particle at exactly
  (ROW,COL). Only way to introduce particles. Count grows by 1 per click.
- **D2 — Click-frame movement suppression.** On a click frame the existing particles do NOT
  move (the `on clicked` list-reassign supersedes the random walk); only the new particle
  appears. They resume walking on the next non-click frame.
- **D3 — Passive random walk on non-click frames.** On every noop/arrow tick, EVERY existing
  particle moves to a uniformly random in-bounds orthogonal neighbor. It ALWAYS moves; it can
  never stay in place.
- **D4 — Adjacency / boundary constraint.** A move target is restricted to the in-bounds
  orthogonal neighbors only (interior 4 choices, edge 3, corner 2). The current cell is never
  a candidate; off-grid neighbors are excluded.
- **D5 — Persistence / no despawn.** Particles are never removed; true count is monotonically
  non-decreasing and equals #clicks issued.
- **D6 — No collisions / overlap.** Particles pass through each other freely. When two
  particles land on the same cell the blue CELL count reads BELOW the true particle count, so
  the on-screen blue-cell count is non-monotonic even though the true count only grows.
- **D7 — No color change** (always blue) — context only, not a scorable transition dynamic.

## 2. Coverage of each dynamic, and the GAP in the ORIGINAL balanced-20

`balanced_split` keys on the FULL action string. With `keep_action_params` the original pool
has 11 unique click labels (1 instance each) + `noop` (~56 instances). A balanced-20 holdout
round-robins the keys, so it grabs ALL 11 clicks + only ~9 of the ~56 noops, chosen at random.

| Dynamic | TARGET under ID? | TARGET under FD? | In original balanced-20? | GAP |
|---|---|---|---|---|
| D1 spawn@loc | yes — new cell at click loc = the label | yes — a cell appears at (R,C) | yes (all 11 clicks sampled) | none |
| D2 suppression | yes — on a click frame the ONLY change is the new cell (existing frozen ⇒ click loc unambiguous) | yes — predict existing stay + new appears | clicks scored, but the contrastive "same particle MOVES on the neighbouring noop" is not guaranteed adjacent | weak/uncontrasted |
| D3 random walk | yes — move + no new cell ⇒ noop | yes — particle visibly displaces | yes (generic noops sampled) | none |
| D4 boundary/corner | yes — corner particle's move reveals the in-bounds-only constraint | yes — only an in-bounds neighbor can be next | corner SPAWN yes, but the corner-WALK noop is 1 specific frame in ~56 ⇒ ~16% chance | **likely missed** |
| D5 persistence | weak (implicit; no despawn target) | implicit across a slice | partial | implicit only |
| D6 overlap/no-collision | the action is still noop | yes — blue-cell count DROPS (two particles merge) | only 2 of ~56 noops show it ⇒ both almost surely dropped | **essentially absent** |
| D7 color | n/a | n/a | n/a | n/a (context) |

**nrdf6-style failure modes this set defeats:**
- *"particles move every step / every k steps"* → defeated by 8 click-frame suppression
  targets where the existing particles FREEZE (movement is tied to action CLASS, not a step
  clock; clicks/noops interleave at varied parities so no `step%k` explains it).
- *"a new cell appears every frame"* → defeated by 12 noop targets with NO new cell.
- *"blue-cell count = #clicks and only grows"* → defeated by 2 overlap noops where the cell
  count drops (D6).
- *"a particle can move anywhere / off-grid"* → defeated by 2 corner-walk targets where the
  only legal next cell is an in-bounds neighbor (D4).
- *"predict no change for a noop"* → defeated by abundant clear-displacement noops (D3).

## 3. Curated slices (verbatim consecutive original rows; 1 slice = 1 episode)

Verified via `T.verify_pool(...)`: **20 scored targets — 8 distinct click labels + 12 noops**.
Each consecutive pair inside a slice is a scored target (action = the first row of the pair).

| episode | original steps | target pairs (action) | dynamics covered |
|---|---|---|---|
| 0 | 3,4,5,6 | `click 4 4` (3→4); noop (4→5); noop (5→6) | D1 spawn on EMPTY grid (cleanest ID — single new cell); D3 two clean single-particle walks; D5 persistence |
| 1 | 7,8,9,10 | noop (7→8); `click 8 8` (8→9); noop (9→10) | **D2 contrast**: particle MOVES (noop) → FREEZES while new spawns (click) → MOVES again (noop) |
| 2 | 17,18,19 | `click 12 12` (17→18); noop (18→19) | D1 interior spawn + D2 suppress (2 existing frozen); D3 walk |
| 3 | 30,31,32 | `click 6 10` (30→31); noop (31→32) | D1 interior spawn + D2 suppress; D3 walk |
| 4 | 47,48,49 | `click 14 4` (47→48); noop (48→49) | D1 EDGE spawn (row 14) + D2 suppress; D3 walk |
| 5 | 57,58,59 | `click 8 2` (57→58); noop (58→59) | D1 spawn + D2 suppress (8 existing frozen); **D6 OVERLAP** noop: blue cells 9→8 |
| 6 | 61,62,63,64 | `click 0 0` (61→62); noop (62→63); noop (63→64) | D1 CORNER spawn + D2; **D4 corner-walk** (0,0)→(0,1)→(0,0): only in-bounds neighbors |
| 7 | 66,67,68,69 | `click 15 15` (66→67); noop (67→68); noop (68→69) | D1 CORNER spawn + D2; **D4 corner-walk** (15,15)→(15,14); **D6 OVERLAP** noop: blue cells 10→9 |

**Contrastive negatives included:**
- Suppression (frozen existing) on 8 click frames vs. movement on 12 noop frames — the
  near-miss that kills any "always moves / step-clock" rule.
- 12 noops with no new cell — near-miss that kills "a cell always appears."
- 2 overlap noops (cell count drops) — near-miss that kills "count = #clicks, monotonic."
- 2 corner-walk noops (in-bounds only) — near-miss that kills "moves anywhere/off-grid."

Click-location coverage: empty-grid (4,4), interior (8,8),(12,12),(6,10), edge (14,4),
near-edge (8,2), corners (0,0),(15,15) — exercises the click-LOCATION label across the grid.
