# 83wkq — TEST50 held-out pool coverage

Config: whitelist = `noop,click`; `keep_action_params=TRUE` (the FULL click string, e.g.
`click 5 4`, is the ID label). Move verbs are not whitelisted (and are never emitted here).
Pool consumed by `gepa_optimize.py --test-run .../test50 --test-n 50` (pool ≤ test-n ⇒ the
whole pool is scored). **Verified: `verify_pool(...test50, 'noop,click', context_k=9)` = 50
scored target transitions**, 0 NO_CHANGE targets, 0 aliased (occupied-cell) clicks.

Freshly driven with `autumn_drive.py 83WKQ ... --seed 0` (6 independent drives → 6 episodes,
verbatim contiguous slices). Distinct from train and the original test: **zero overlap** with
the 8 train click locations or the 8 original-test click locations.

## The seed-0 dynamics realization (read this first)

83WKQ's rule is a uniform random walk (dynamics.txt D3/D4), but under the interpreter's
per-frame RNG at **seed 0 the walk is DETERMINISTIC and reproducible**, and this same
realization is what the sweep consumes from the CSV. Empirically (verified frame-by-frame):

- **Every noop: each particle stays in its ROW and moves one column LEFT (col→col-1)** if
  col>0; **at the left wall (col 0) it bounces RIGHT (col 0→col 1)**. Rows never change; no
  vertical moves ever occur; particles drift left and oscillate col0↔col1 at the wall.
- **Every click: the existing particles FREEZE** (D2) while exactly one new blue cell appears
  at the clicked cell (D1).

Consequence for the test: FD is oracle-predictable (good — the regularity to be learned is
"shift left, bounce at the left wall"). The **only** place movement deviates from "shift
left" is the **left-wall / corner bounce (D4)** — so those boundary noops carry the entire
anti-shortcut load for movement, and the pool deliberately over-samples them.

## 1. CORE dynamics (from dynamics.txt)

- **D1 — Spawn at click cell.** `click R C` adds one new blue particle at exactly (R,C).
- **D2 — Click-frame suppression.** On a click frame all existing particles hold position.
- **D3 — Passive walk on noop.** Every existing particle moves to an in-bounds orthogonal
  neighbor every noop (here: col-1, the seed-0 realization). It never stays in place.
- **D4 — Adjacency / boundary constraint.** Move target restricted to in-bounds neighbors
  (interior 4, edge 3, corner 2); the current cell is never a candidate. ⇒ a col-0 particle
  cannot go left, so it goes right; a corner particle has only 2 candidates.
- **D5 — Persistence / no despawn.** Particle count is monotonically non-decreasing = #clicks.
- **D6 — No collision / overlap.** Two particles may occupy one cell; the blue-CELL count then
  reads BELOW the true count (non-monotonic on-screen count).
- **D7 — No color change** (always blue) — context only, not a scorable transition dynamic.

## 2. Episode composition (verbatim slices; one drive = one episode)

| ep | source drive[rows] | trans | clicks (label) | noops | dynamics targeted |
|----|--------------------|-------|----------------|-------|-------------------|
| 0 | S1[0..10]    | 10 | (2,9)(4,13)(7,11)(10,14)(12,8) | 5 | D1 spawn ×5, D2 suppress ×4 (1→4 existing frozen), D3 interior walk ×5, D5 count 1→5 |
| 1 | S2[0..11]    | 11 | (1,14)(6,9)(9,15)(13,10)(3,6)(11,13) | 5 | D1 ×6, D2 ×5, D3 walk ×5 incl right-edge (9,15)→(9,14) start, D5 count 1→6 |
| 2 | CORN2[0..7]  | 7  | (0,15)(15,0)(0,3) | 4 | **D4 corner-walk** (0,15)→(0,14) [2 choices]; **D4 corner-bounce** (15,0)→(15,1) [rightward]; D1/D2/D3 |
| 3 | WALL2[0..6]  | 6  | (4,1)(8,3) | 4 | **D4 clean left-wall bounce** (4,0)→(4,1) [single particle, rightward]; walk-to-wall; D1/D2 |
| 4 | OV2[0..6]    | 6  | (5,0)(5,2)(5,4) | 3 | **D6 overlap** 3→2 and 2→1 (same-row merge at (5,1)); D4 col0 bounces; D1/D2 |
| 5 | OV[0..10]    | 10 | (6,0)(6,2)(9,1)(9,3) | 6 | **D6 overlap** 2→1 and 3→2; **D4 clean bounce** (6,0)→(6,1); D3 walks; D1/D2 |

Totals: **50 transitions = 23 clicks + 27 noops.**

## 3. Per-dynamic coverage under ID and FD (positives / negatives)

| dynamic | ID-informative? | FD-informative? | positives (count) | contrastive negative in pool |
|---|---|---|---|---|
| **D1** spawn@loc | yes — new cell location = the label | yes — a cell appears at (R,C) | **23** (every click; new cell verified at click loc) | noops never add a cell (27) ⇒ "a cell always appears" scores worse |
| **D2** suppression | yes — on a click frame the ONLY change is the new cell ⇒ loc unambiguous | yes — predict existing HOLD + new appears | **17** (clicks with ≥1 existing frozen) | the 27 noops where particles DO move ⇒ movement is tied to action CLASS, defeating "always moves"/`step%k` clocks (clicks/noops interleave at irregular parities) |
| **D3** passive walk | yes — move + no new cell ⇒ noop | yes — particle visibly displaces (col-1) | **23** count-preserving moves | the 23 clicks where existing FREEZE ⇒ "noop = no change" and "everything always moves" both fail |
| **D4** boundary/corner | yes — the constrained move reveals the in-bounds-only rule | yes — only an in-bounds neighbor is legal; at col0 the move is RIGHT not left | **9** boundary transitions: corner-walk (0,15)→(0,14); col0 rightward bounces (4,0)→(4,1), (6,0)→(6,1), (15,0)→(15,1) + embedded col0 bounces in D6 merges | these ARE the near-miss that kills "on noop everything shifts left" (they shift RIGHT) and "a particle can move anywhere / off-grid" (row fixed, in-bounds only) |
| **D5** persistence | weak (no despawn target) | implicit across a slice | implicit — every multi-step episode has monotonic non-decreasing count (ep0 1→5, ep1 1→6); no despawn ever | not a discrete scored change (as in train COVERAGE); no negative possible (despawn never fires) |
| **D6** overlap | action is still noop, recoverable (cell count DROPS, no new cell) | yes — blue-cell count DROPS as two particles merge | **4** count-drop noops (ep4 3→2, 2→1; ep5 2→1, 3→2) | the 23 non-drop noops (count preserved) ⇒ "blue-cell count = #clicks, monotonic" scores worse |
| D7 color | n/a | n/a | n/a (always blue) | n/a |

Every scorable core dynamic (D1–D4, D6) is a scored TARGET ≥4 times, in varied
situations. D5 is implicit/persistence (consistent with the train set's treatment).

## 4. Action histogram of the pool

`noop`: 27. Plus 23 distinct single-instance click labels:
(2,9)(4,13)(7,11)(10,14)(12,8)(1,14)(6,9)(9,15)(13,10)(3,6)(11,13)(0,15)(15,0)(0,3)(4,1)
(8,3)(5,0)(5,2)(5,4)(6,0)(6,2)(9,1)(9,3). Click locations span rows 0–15 and cols 0–15
(corners, top/bottom/left/right edges, all interior quadrants; low-col clicks are the D4/D6
scenarios). **Total 50** (verified by `verify_pool`).

## 5. How this test differs from train

- **Fresh drives, disjoint locations.** 23 NEW click locations, **zero overlap** with the 8
  train or 8 original-test click cells. Different (irregular) action cadence.
- **2.5× larger and harder.** Train = 20 targets (8 click + 12 noop); test50 = 50 (23 + 27).
- **Deliberately authored rare states.** Train's D4/D6 came from incidental original-rollout
  frames; test50 engineers them: single-particle left-wall/corner transitions for CLEAN D4
  bounces, and same-row same-parity particles that collide at the wall for D6 count-drops.
- **Larger multi-particle regimes** (up to 6 simultaneous in ep1; dense same-row clusters in
  ep4/ep5) that the short train slices never produced.

## 6. Not covered / caveats

1. **Seed-0 determinism / horizontal bias.** As above, the walk is a deterministic leftward
   column drift (+ left-wall bounce); no vertical or rightward-interior moves exist under seed
   0. Unavoidable (changing the seed would desync from train, which uses the same interpreter
   realization). Documented so a reviewer knows the D4 boundary bounces are the sole movement
   negative. The FD objective is therefore oracle-solvable, which is intended.
2. **Occupied-cell click is the one aliased case.** Clicking an already-occupied cell yields
   NO visible change (existing freeze + new particle overlaps) and is not ID-recoverable; per
   req 4 these are excluded (0 in the pool).
3. **D5 persistence** has no negative (despawn never fires) and no isolated scored change; it
   is covered implicitly by monotonic counts across multi-step episodes.
4. **D7 color**: no color dynamic exists in 83WKQ; context-only.

## 7. viz.html

`viz.html` is a filmstrip of all 6 episodes concatenated in order (56 frames = 50 transitions
+ 6 per-episode terminal seam markers, shown as `(terminal)` so no false cross-episode
transition is implied). Built with `build_dataset_viz.py` on a scratch concatenation of the
episodes (the tool renders a single `episode_0`); the per-episode CSVs under `episode_*/` are
the authoritative verbatim pool.
