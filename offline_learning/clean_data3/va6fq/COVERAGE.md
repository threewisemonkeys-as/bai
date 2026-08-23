# va6fq — clean_data3 coverage analysis

Game: VA6FQ, a falling-sand / liquid cellular automaton on a 10x10 grid.
Config: whitelist = `noop,click`; `keep_action_params=TRUE` (click LOCATION is part of
the label, e.g. `click 1 4`). Action string is `click ROW COL` (row first), confirmed from
the env header and from the spawned object always appearing exactly at the clicked cell.

## 1. Core dynamics extracted from dynamics.txt

- **D1 — sandButton brush switch.** Click on sandButton (red, row0 col2 → `click 0 2`) sets
  clickType="sand". No object spawned; grid unchanged by the click itself.
- **D2 — waterButton brush switch.** Click on waterButton (green, row0 col7 → `click 0 7`)
  sets clickType="water". No object spawned.
- **D3 — spawn sand.** Click a free cell while brush=sand → a `tan` (solid, liquid=false)
  Sand appears AT the clicked cell.
- **D4 — spawn water.** Click a free cell while brush=water → a `skyblue` Water appears AT
  the clicked cell.
- **D5 — inert click.** Click on an occupied non-button cell does nothing (no spawn, no
  brush change). Passive physics still run that step.
- **D6 — solid-sand straight fall.** Each step, dry (`tan`) sand moves straight DOWN one cell
  if the cell below is empty; never sideways/diagonal (grains stack into vertical columns).
- **D7 — water straight fall.** Each step, Water (`skyblue`) moves straight down one cell if
  below is empty.
- **D8 — wetting.** Each step, any dry `tan` sand orthogonally adjacent to Water flips
  liquid=true → recolors `tan` → `sandybrown` (permanent).
- **D9 — liquid diagonal flow.** Water AND wet (`sandybrown`) sand: if straight-down is
  blocked, slip one step DIAGONALLY-down into the nearest reachable hole (an empty cell whose
  own cell-above is also empty). They do not crawl across a flat fully-supported surface.

(No win/score/termination rule exists — open sandbox.)

## 2. Original-train coverage and the gaps

The original `clean_data2/va6fq/train` is 84 rows: **20 clicks, all at step ≡ 3 (mod 4)**, and
~63 noops at the other parities. A balanced-20 sample (`balanced_split`, train_n=20) draws
~10 clicks + ~10 noops at random.

| Dynamic | TARGET under ID? | TARGET under FD? | reliably in a balanced-20 sample? | GAP |
|---|---|---|---|---|
| D1 sandBtn switch | weak (grid unchanged by click → not recoverable from diff) | yes (predict NO spawn / brush=sand) | NO — only 1 such click (step55) among 20 | likely **dropped**; brush-state rule untested |
| D2 waterBtn switch | weak (same) | yes | NO — only 1 (step27) among 20 | likely **dropped** |
| D3 spawn sand | yes (loc = where tan appears) | yes | yes (many) | covered |
| D4 spawn water | yes (loc = where skyblue appears) | yes (needs brush=water) | yes (several) | covered, but FD needs the brush-switch in window — easily lost if D2 not in pool |
| D5 inert click | adversarial (looks like noop) | yes (predict no spawn) | NO — only 1 (step23) | **dropped**; "click always spawns" shortcut unpunished |
| D6 dry fall | yes (noop; grain steps down 1) | yes | yes (many noops) | covered but easily diluted by NO_CHANGE noops |
| D7 water fall | noop | yes | maybe — only the few single-water noops (steps32–34) | risk of **context-only** |
| D8 wetting | noop | yes (tan→sandybrown) | NOT guaranteed — only a handful of noops wet, scattered in steps37–82 | nrdf6-style **context-only** risk |
| D9 liquid diagonal flow | noop | yes | NOT guaranteed; and most flow steps are 7+-cell chaos (multiple water cells) → no clean single-rule test | **context-only / unisolable** in original |
| step-clock shortcut | — | — | — | clicks are PERFECTLY at step ≡ 3 (mod 4): a `step%4==3 → click` clock recovers click-vs-noop for free (the nrdf6 trap) |

**Key gaps:** (a) the three NON-spawning click types (D1/D2/D5) are each a single transition
in 84 rows and almost surely dropped by a balanced sample, so the brush-state machinery and
the "occupied click = no-op" rule never get scored; (b) the passive water dynamics (D7/D8/D9)
appear mostly as window context, and where they are targets they are buried in multi-cell
chaos; (c) the action cadence makes `step%4==3` a perfect free shortcut for ID.

**How this pool closes them:**
- All three non-spawning clicks (D1 `click 0 2`, D2 `click 0 7`, D5 `click 9 3`) are forced
  into the pool as scored targets and act as the contrastive negatives against the spawning
  clicks ("a click does NOT always spawn").
- D4 (spawn water) and D3-after-sandBtn (spawn sand) are each placed so the relevant button
  switch sits in their `ctx_prev` window (verified: both have prev=4) — so FD has the brush
  state available and the water/sand spawn distinction is learnable, not guessed.
- D7/D8/D9 are pulled in as **clean, single-cell-trackable** noop targets (single falling
  water cell; an isolated tan→sandybrown wetting; a dry grain falling straight while one wet
  grain slips diagonally) rather than the 7-cell chaos transitions.
- The step-clock is intrinsic to the source cadence and **cannot be broken with verbatim
  rows** (no source click exists off ≡3 mod4). It is instead defeated by `keep_action_params`:
  the ID label is the full `click R C` string, so `step%4` only yields "a click" — pinning
  the exact coordinates still requires the real spawn-location rule. FD is likewise immune
  (some noops move a cell, others are NO_CHANGE — no clock predicts the grid). This residual
  is documented rather than removable.

## 3. Curated slices (each = one episode of verbatim original rows)

Verified pool = **20 scored targets** (`T.verify_pool(...)`), 7 click + 13 noop, all kept
(pool size = train_n=20). Windows are real consecutive frames and stop at slice boundaries.

| ep | original steps | target pairs → dynamic(s) covered |
|---|---|---|
| 0 | 3,4,5,6,7 | `3→4 click 1 4` spawn SAND **(D3, ID-loc)**; `4→5`,`5→6`,`6→7` noop: grain (1,4)→(2,4)→(3,4)→(4,4) straight fall **(D6)**. Clean, no water. |
| 1 | 23,24 | `23→24 click 9 3` on OCCUPIED floor → no spawn; only a passive grain falls **(D5; contrastive negative — a click that does NOT spawn / looks like noop)**. |
| 2 | 27,28,29,30,31,32,33 | `27→28 click 0 7` waterButton **(D2 switch)**; `28→29`,`29→30`,`30→31` noop NO_CHANGE **(negatives: noop ≠ always moves; step advances, grid static)**; `31→32 click 1 2` spawn WATER **(D4; window carries the waterBtn switch)**; `32→33` noop water (1,2)→(2,2) straight fall **(D7)**. |
| 3 | 55,56,57,58,59,60 | `55→56 click 0 2` sandButton **(D1 switch; non-spawn click)**; `57→58` noop tan(7,6),(7,7)→sandybrown **(D8 wetting)**; `56→57`,`58→59` noop liquid motion **(D7/D9)**; `59→60 click 1 3` spawn SAND **(D3; window carries the sandBtn switch — contrasts D4: same free-cell click, sand not water)**. |
| 4 | 79,80,81,82,83 | `79→80 click 1 4` spawn SAND **(D3)**; `80→81`,`81→82`,`82→83` noop, each a clean DOUBLE event: dry grain (1,4)→(2,4)→(3,4)→(4,4) straight fall **(D6)** while a wet grain slips (5,6)→(5,7)→(5,8)→(6,8) **(D9 liquid diagonal flow)**. |

### Contrastive structure (defeats shortcuts)
- **Spawn vs non-spawn click:** spawning clicks (ep0 `1 4`, ep2 `1 2`, ep3 `1 3`, ep4 `1 4`)
  vs non-spawning clicks (ep3 sandBtn `0 2`, ep2 waterBtn `0 7`, ep1 occupied `9 3`).
  Punishes "every click adds an object."
- **Brush-conditioned spawn:** waterBtn→spawn skyblue (ep2) vs sandBtn→spawn tan (ep3), same
  "click a free cell" surface. Punishes "free-click always spawns sand."
- **noop-moves vs noop-NO_CHANGE:** moving noops (falls/flow/wetting, eps 0/2/3/4) vs the
  three NO_CHANGE noops (ep2). Punishes "noop always settles a grain" and any step-parity
  "motion clock."
- **step-clock:** unbreakable in-source (all clicks ≡3 mod4) but neutralised by
  `keep_action_params` (full-coordinate label) + mixed noop outcomes; documented above.

---
## REGENERATED trajectory (irregular timing + isolated single-drop diagonal flow)

The original `clean_data2/va6fq` rollout had two structural defects that **re-slicing
cannot fix**, so the trajectory was regenerated with `autumn_drive.py` driving the real
`VA6FQ.sexp` (seed 0). Full trajectory: `train_regen/episode_0/trajectory.csv` (44 rows / 43
transitions), filmstrip `train_regen/viz.html`. The curated `train/` below is now sliced from
`train_regen`. (`dynamics.txt` and `test/` unchanged — test still lacks these states.)

### The two gaps that forced regeneration
1. **Step-parity click shortcut.** In the original, *every* click fell on `step ≡ 3 (mod 4)`,
   so a `step % 4 == 3 -> click` clock recovered click-vs-noop for free. **Fix:** irregular
   action timing — clicks land on regenerated steps 0, 8, 16, 18, 23, 26, 36, 38 (gaps
   8,8,2,5,3,10,2; parities 0,0,0,2,3,2,0,2). No `step % k` predicts a click.
2. **Liquid diagonal flow only ever shown in 7+-cell chaos.** The single-droplet flow rule was
   never isolable. **Fix:** the flow is driven with a SINGLE water drop in the otherwise-empty
   **col-0 lane** (col-0's only neighbour col-1 is empty, so there is NO pile-edge wetting
   during the fall): one grain is first dropped to form an obstacle at (9,0); a single water
   drop then falls straight (5,0)->(8,0) and, blocked by the obstacle, **slips one cell into
   the lone reachable hole-column**: (8,0)->(8,1) [regen step 33], then falls into the hole
   (8,1)->(9,1) [step 34]. (This engine realises "diagonal flow" as a horizontal slip toward
   the hole followed by a fall — net diagonal over two ticks.)

### Action sequence (autumn_drive, seed 0)
`click_2_0` ·7 noop  (D3 spawn sand + D6 straight fall -> obstacle at (9,0)) ›
`click_2_9` ·7 noop  (D3+D6 -> obstacle at (9,9)) ›
`click_9_0` ·1 noop  (D5 inert click on occupied cell, NO_CHANGE) ›
`click_6_9` ·4 noop  (D6 dry grain lands at (8,9) and **REFUSES** the reachable hole (9,8)) ›
`click_0_7` ·2 noop  (D2 waterButton switch, NO_CHANGE) ›
`click_2_0` ·9 noop  (D4 spawn WATER; D7 straight fall col-0; **D9 slip (8,0)->(8,1)**; D8 grain wets) ›
`click_0_2` ·1 noop  (D1 sandButton switch, NO_CHANGE) ›
`click_4_3` ·4 noop  (D3 spawn SAND into the col-3 pit; D6 straight fall/stack)

### Curated slices (each = one episode of verbatim train_regen rows; regen Step numbers)
| ep | regen steps | scored targets -> dynamic(s) |
|---|---|---|
| 0 | 0,1,2,3 | `0->1 click 2 0` spawn SAND (D3, ID-loc); `1->2`,`2->3` noop grain falls (D6) |
| 1 | 16,17,18 | `16->17 click 9 0` on occupied grain -> NO spawn (**D5 inert; non-spawn click**); `17->18` noop NO_CHANGE (settle neg) |
| 2 | 19,20,21,22 | `19->20`,`20->21` noop dry grain falls onto obstacle (D6); `21->22` noop grain at (8,9) **REFUSES** hole (9,8) -> NO_CHANGE (**D6/D9 contrast negative**) |
| 3 | 23,24,25,26,27 | `23->24 click 0 7` waterBtn NO_CHANGE (**D2 switch**); `24->25`,`25->26` noop NO_CHANGE (negs); `26->27 click 2 0` spawn **WATER** (D4, ID-loc; window carries the D2 switch) |
| 4 | 31,32,33,34,35 | `31->32`,`32->33` noop water straight fall col-0 (D7); **`33->34` noop water (8,0)->(8,1) SLIP into hole-column (D9) + obstacle wets (D8)**; `34->35` noop water (8,1)->(9,1) fall into hole + pile-edge wets (D7/D8) |
| 5 | 36,37,38,39,40 | `36->37 click 0 2` sandBtn NO_CHANGE (**D1 switch**); `37->38` noop NO_CHANGE (neg); `38->39 click 4 3` spawn **SAND** (D3, ID-loc; window carries D1 switch — contrast vs ep3 WATER spawn); `39->40` noop grain falls (D6) |

### Final pool (verify_pool, whitelist `noop,click`, keep_action_params=TRUE)
**20 scored targets** (pool == train_n=20 -> balanced_split keeps ALL).
By action label: `noop=14, "click 2 0"=2, "click 9 0"=1, "click 0 7"=1, "click 0 2"=1, "click 4 3"=1`.

Every core dynamic is now a scored target under its applicable objective:
D1 (ep5 switch + ep5 sand spawn), D2 (ep3 switch + ep3 water spawn), D3 (ep0/ep3/ep5 spawns, ID-loc),
D4 (ep3 water spawn, ID-loc), D5 (ep1 inert), D6 (ep0/ep2/ep5 falls + ep2 refuse), D7 (ep4 falls),
D8 (ep4 wetting), **D9 (ep4 step 33 single-drop slip into hole)**.

### Contrastive structure (defeats shortcuts)
- **D9 slip vs D6 refuse:** water at (8,0) slips into hole-column (ep4 `33->34`) vs DRY grain at
  (8,9) refuses an identical reachable hole (ep2 `21->22` NO_CHANGE) — same blocked-below cue,
  liquid moves / solid stays. The single-drop slip is now a clean scored target, not 7-cell chaos.
- **Brush-conditioned spawn (same action, different result):** `click 2 0`->**tan** (ep0,
  brush=sand) vs `click 2 0`->**skyblue** (ep3, brush=water, switch in window). Identical
  action+coordinates; punishes coordinate-memorization, forces use of the hidden brush state.
- **Spawn vs non-spawn click:** spawns (ep0/ep3/ep5) vs non-spawns (ep1 inert, ep3 waterBtn,
  ep5 sandBtn) — punishes "every click adds an object".
- **noop-moves vs noop-NO_CHANGE:** falling/slip noops vs the NO_CHANGE noops (ep1, ep2 refuse,
  ep3 x2, ep5) — punishes "noop always moves a cell" and any step-parity motion clock.
- **Step-parity defeated:** clicks span parities {0,2,3} on irregular gaps; no `step % k`
  recovers click-vs-noop (the original's `step%4==3` shortcut is gone).
