# qfsvc — clean_data3 coverage

Config: whitelist = `left,right,up,down,noop,click`; `keep_action_params=FALSE`
(movement game — `click X Y` collapses to the verb `click`; the click LOCATION is
NOT the label, consistent with dynamics.txt: "the click location does not matter").

## 1. Core dynamics extracted from dynamics.txt

- **D1 left** — agent (red) moves one cell left (col −1).
- **D2 right** — agent moves one cell right (col +1).
- **D3 up** — agent moves one cell up (row −1).
- **D4 down** — agent moves one cell down (row +1).
- **D5 click (ammo > 0)** — spend 1 ammo, spawn a bullet (mediumpurple) at the
  agent's current cell. The bullet is drawn over the agent that tick, so red is
  hidden (rendering note). Click location is irrelevant.
- **D6 click (ammo == 0)** — no ammo, click does nothing (no-op). Contrastive
  negative for D5.
- **D7 passive bullet motion (clock, every step incl. noop)** — every existing
  bullet moves up one cell each step, automatically.
- **D8 bullet despawn** — a bullet at row 0 moves up off the top edge and vanishes
  (a consequence of D7 at the boundary).
- **D9 coin collection** — agent moving onto a gold coin → ammo +1 and the coin is
  removed. This is rendered with a one-step delay: on the move tick the agent's old
  cell goes black and red is hidden behind the still-drawn coin; on the FOLLOWING
  step the coin disappears and red reappears on that cell. ammo is internal/unrendered.
- **D10 noop with nothing active** — noop with no bullet present and agent not landing
  on a coin → NO_CHANGE. Contrastive negative for D7 / D9.
- (No win / termination / reward; no color changes — nothing to score.)

## 2. Coverage under the objectives in the ORIGINAL train pool, and the gaps

A balanced-20 sample of the original 100-row trajectory would be dominated by
NO_CHANGE noops (63 of 99 transitions are NO_CHANGE) with the action cadence
"1 real action then 3 noops", which is the nrdf6 trap.

| Dynamic | ID target? | FD target? | Gap in original pool |
|---|---|---|---|
| D1 left | yes (col −1) | yes | fine, but easily crowded out by noops |
| D2 right | yes (col +1) | yes | same |
| D3 up | yes (row −1) | yes | same |
| D4 down | yes (row +1) | yes | same |
| D5 click spawn | yes (new purple at agent cell) | yes (conditional on ammo) | present but rare (3 of 99); a balanced sample may pick the ammo-0 NO_CHANGE click and miss the spawn |
| D6 click no-op | n/a (NO_CHANGE) | yes (the ammo==0 condition) | only ONE instance (3→4); without it the spawn rule looks unconditional |
| D7 bullet motion | no (fires on noop) | yes | **gap**: only visible as window context around the few clicks; as a *target* it is a noop, indistinguishable from the 63 NO_CHANGE noops → the "noop = nothing" shortcut wins, and the regular 4-step cadence invites a spurious `step % 4` clock (the nrdf6 failure mode) |
| D8 despawn | no (noop) | yes | same noop-shortcut gap |
| D9 coin collection | partially (the move-onto pair hides red; the *collection* itself fires on the next noop) | yes | **gap**: the scored visible change (gold −1) lands on a noop one step after the move; as a passive noop it is unidentifiable and gets explained away as NO_CHANGE / clock unless paired contrastively |
| D10 noop NO_CHANGE | n/a | yes (must NOT change) | over-represented; needs to be balanced against the change-producing noops, not eliminated |

**Net gap:** all the *interesting* dynamics (D7, D8, D9, and the D5/D6 condition)
are either rare or surface only on noops, where a balanced-20 sample drowns them in
NO_CHANGE noops and a step-parity clock can fake the bullet motion. The curated pool
forces each as a scored target and pairs every conditional dynamic with a near-miss
negative under the SAME action.

## 3. Curated slices (11 episodes → 20 scored targets)

Each slice is a contiguous verbatim copy of original train rows; every internal
consecutive pair is a scored target. `a→b` uses `Action[a]` as the label.

| episode | orig steps | target(s) → dynamic |
|---|---|---|
| 0 | [7,8] | up 7→8 → **D3** (clean move, ID col-recoverable) |
| 1 | [31,32] | left 31→32 → **D1** (clean move) |
| 2 | [55,56] | right 55→56 → **D2** (clean move) |
| 3 | [59,60] | right 59→60 → **D2** (2nd, balance) |
| 4 | [87,88] | down 87→88 → **D4** (clean move) |
| 5 | [82,83,84] | noop 82→83 → **D8** bullet despawn off top (POS); down 83→84 → **D4** (2nd) |
| 6 | [2,3,4] | noop 2→3 → **D10** NO_CHANGE (NEG); click 3→4 → **D6** ammo-0 no-op (NEG, NO_CHANGE) |
| 7 | [43,44,45,46] | click 43→44 → **D5** spawn (POS, purple at agent cell); noop 44→45 → **D7** bullet-up; noop 45→46 → **D7** bullet-up (pure, only bullet moves) |
| 8 | [26,27,28,29] | noop 26→27 → **D10** NEG; left 27→28 → **D9** cause (move-onto-coin, red hidden); noop 28→29 → **D9** effect (gold −1, red reappears) |
| 9 | [70,71,72,73] | noop 70→71 → **D10** NEG; up 71→72 → **D9** cause; noop 72→73 → **D9** effect (collect via a different direction) |
| 10 | [79,80,81] | click 79→80 → **D5** spawn (POS, 2nd); noop 80→81 → **D7** bullet-up |

Delayed-effect handling (D9): the cause (move-onto-coin) and the effect (collection
on the next noop) are kept in the SAME slice, so the cause sits in the effect target's
prev-window (verified: the `gold-1 red+1` noops have prev-window = 2).

## 4. Pool composition (verified via `T.verify_pool`)

20 scored target transitions (= train-n, so balanced_split keeps all of them).

By action (click collapses to the verb under keep_action_params=FALSE):
`up:2, left:2, right:2, down:2, click:3, noop:9` = 20.

The 9 noops are a deliberate mix, not filler:
- 3 × NO_CHANGE negatives (2→3, 26→27, 70→71) — D10
- 3 × bullet-up positives (44→45, 45→46, 80→81) — D7
- 1 × despawn positive (82→83) — D8
- 2 × coin-collect positives (28→29, 72→73) — D9

### Contrastive structure (defeats the shortcuts)

- **click POS vs NEG:** spawn (43→44, 79→80) vs ammo-0 NO_CHANGE click (3→4) →
  the spawn rule must be conditioned on ammo, not "click always spawns".
- **noop changes vs noop NO_CHANGE:** the same `noop` label both produces change
  (bullet-up, despawn, coin-collect) and produces NO_CHANGE (3 negatives) →
  defeats "noop = nothing" AND defeats a `step % k` clock, because change is
  conditioned on bullet-presence / coin-underfoot, not on step parity.
- **coin collection from two directions** (left-onto and up-onto) with the collect
  effect as a noop target, each flanked by a NO_CHANGE noop negative.

Every core dynamic (D1–D10) appears as a scored target under at least one objective,
each conditional dynamic is paired with its near-miss negative, and the regular
1-action/3-noop cadence of the original trajectory is broken so no step-parity clock
can explain the passive motion.

## TRAIN2 (untied-val expansion)

`train2/` is a SECOND, independently regenerated curated pool (built from a fresh
`autumn_drive.py QFSVC` run, seed 0) added so the combined `train/` (20) + `train2/` (32)
pool supports an untied 30-train/30-val split. It does not touch `train/`, `test/`,
`test50/`, or `dynamics.txt`.

### Why a regen was needed, and which coins/positions it targets

The original 100-step raw rollout (source of `train/`) collects only 3 of the 10 coins
before it ends — confirmed by diffing the coin set between its first and last frame:
`(2,6)`, `(4,4)`, `(4,6)` go missing; `(2,4)`, `(2,8)`, `(4,10)`, `(4,8)`, `(2,10)`,
`(4,12)`, `(2,12)` remain gold throughout. `test50`'s three fresh drives (see its
`TEST_COVERAGE.md`) additionally collect `(4,6)`/`(2,6)` (Drive A), `(4,10)`/`(2,10)`
(Drive B), and `(4,12)`/`(2,12)` (Drive C). Combined, `train/` + `test50/` have already
touched 7 of the 10 coins: `{(2,6),(4,4),(4,6),(4,10),(2,10),(4,12),(2,12)}`. That leaves
exactly **3 untouched coins: `(2,4)`, `(2,8)`, `(4,8)`** — `train2`'s new drive is built
specifically to collect all three, each via a different approach direction, so the
scored coin-collection targets sit on genuinely new board configurations rather than
re-slicing positions already seen by GEPA (train) or reserved for held-out scoring
(test50).

### The new drive (`train_regen2/episode_0/trajectory.csv`, 44 actions, seed 0)

Action sequence (verified frame-by-frame against `autumn_drive.py` ASCII + legend output;
every gold/red/mediumpurple count matches the intended dynamic at each step):

```
click_9_7, noop, noop, up,up,up,up,up, right, noop, click_4_8, noop,noop,noop,noop,noop,
up,up, noop, up, left,left,left,left, down, noop, click_2_4, noop,noop,noop,
up, right,right, click_1_6, noop,noop, click_1_6, noop,noop,
right, down,down,down, noop
```

- Agent starts at `(9,7)` (row,col). Path: up×5 to `(4,7)`, **right-onto `(4,8)`**
  (collect #1) → click-spawns a bullet at `(4,8)` that needs the **longest flight (5
  ticks)** and, on its way up, passes directly over the still-uncollected coin `(2,8)` —
  a one-tick **flicker** (gold hidden under the bullet, confirmed via the legend:
  `gold` count dips from 9→8 then back to 9 with `mediumpurple` unchanged at 1) — then
  despawns off the top. The agent then walks up **onto `(2,8)`** (the same coin that was
  just flickered over) via **up-onto** (collect #2), demonstrating directly that transit
  ≠ collection.
- Continues left along row 1 (always coin-free, since coins only exist at rows 2/4) to
  `(1,4)`, then **down-onto `(2,4)`** (collect #3) — a **down-onto** approach, the third
  distinct direction used for a fresh coin. Click-spawns a bullet at `(2,4)` (**medium,
  3-tick flight**), despawns, then moves up+right to `(1,6)` and click-spawns a third
  bullet (**shortest, 2-tick flight** — row 1 is a fire row neither `train/` nor
  `test50/` ever uses) which despawns quickly. A same-cell click immediately after
  (ammo now 0) is a clean **D6 ammo-exhausted negative**. Ends with clean right/down
  moves through coin-free columns and a trailing quiet noop.
- Action cadence is irregular throughout (double-noop open, a 5-move dash, three
  different-length bullet flights, a double-noop before the tail) — no fixed
  step-parity pattern.
- Filmstrip: `train_regen2/viz.html` (45 frames).

### Curated slices (`train2/episode_0..8` — 9 episodes, 32 scored targets)

Each episode is a short contiguous verbatim slice of `train_regen2`; steps below are
`train_regen2` row numbers (0-indexed, matching its CSV `Step` column).

| ep | orig steps | target(s) → dynamic |
|---|---|---|
| 0 | [0,1,2,3] | click 0→1 → **D6 neg#1** (ammo-0 before any collection, NO_CHANGE); noop 1→2 → **D10 neg#1**; noop 2→3 → **D10 neg#2** |
| 1 | [7,8,9,10] | up 7→8 → **D3** clean; right 8→9 → **D9 cause** (RIGHT-onto `(4,8)`) + **D2**; noop 9→10 → **D9 eff#1** (gold 10→9, ammo 0→1) |
| 2 | [10,11,12,13,14] | click 10→11 → **D5 spawn#1** @ `(4,8)`; noop 11→12 → **D7** tick1 (row4→3); noop 12→13 → **D7** tick2 **FLICKER** (bullet passes over uncollected `(2,8)`: gold hidden, NOT removed); noop 13→14 → **D7** tick3 restore (gold reappears) |
| 3 | [16,17,18,19] | up 16→17 → **D3** clean; up 17→18 → **D9 cause** (UP-onto `(2,8)`, the just-flickered coin) + **D3**; noop 18→19 → **D9 eff#2** (gold 9→8, ammo 0→1) |
| 4 | [22,23,24,25,26] | left 22→23 → **D1**; left 23→24 → **D1**; down 24→25 → **D9 cause** (DOWN-onto `(2,4)`) + **D4**; noop 25→26 → **D9 eff#3** (gold 8→7, ammo 1→2) |
| 5 | [26,27,28,29,30] | click 26→27 → **D5 spawn#2** @ `(2,4)`; noop 27→28 → **D7** (row2→1); noop 28→29 → **D7** (row1→0); noop 29→30 → **D8 despawn#2** (col 4) |
| 6 | [31,32,33,34,35,36] | right 31→32 → **D2**; right 32→33 → **D2**; click 33→34 → **D5 spawn#3** @ `(1,6)`; noop 34→35 → **D7** (row1→0); noop 35→36 → **D8 despawn#3** (col 6, shortest 2-tick flight) |
| 7 | [36,37,38,39] | click 36→37 → **D6 neg#2** (ammo exhausted after 3 spawns, NO_CHANGE); noop 37→38 → **D10 neg#3**; noop 38→39 → **D10 neg#4** |
| 8 | [41,42,43,44] | down 41→42 → **D4**; down 42→43 → **D4**; noop 43→44 → **D10 neg#5** (final) |

Every target above was checked against `T.verify_pool`'s per-transition `classify()`
output (e.g. `right | black+1 red-1` for the collect-cause, `noop | gold-1 red+1` for the
collect-effect, `click 4 8 | mediumpurple+1 red-1` for the spawn, `noop | black+1
gold-1 mediumpurple~move(-1.0,0.0)` for the flicker tick, `NO_CHANGE` for every D6/D10
negative) — all match the intended dynamic exactly, with no accidental extra coin
collisions along any transit leg (rows 1 and 3/5+ never host coins, so every "clean"
move/transit step was routed to avoid the 7 already-used coin cells).

### Pool composition (verified via `T.verify_pool`)

**32 scored target transitions** (pool < `--train-n 20`? no — this pool is meant to pair
with `train/`'s 20 for an untied 30/30 split at the harness level; in isolation
`balanced_split` over `train2` alone would sample down to `--train-n`, which is expected).

By action (click collapses to the verb under `keep_action_params=FALSE`):
`click: 5, noop: 16, up: 3, right: 3, left: 2, down: 3` = 32.

- The 5 clicks: 3 spawns (`(4,8)`, `(2,4)`, `(1,6)`) + 2 ammo-0 negatives (pre-collection,
  post-exhaustion).
- The 16 noops: 5 NO_CHANGE negatives (D10: 2 opening, 2 post-exhaustion, 1 final) + 3
  D9 collection-effects + 6 D7 bullet-fly ticks (incl. the flicker pair) + 2 D8 despawns.

### Per-dynamic coverage

| dynamic | positives (train2) | negatives (train2) |
|---|---|---|
| D1 left | 2 (clean, row 1) | — |
| D2 right | 3 (1 as D9-cause via right-onto `(4,8)`, 2 clean) | — |
| D3 up | 3 (2 clean, 1 as D9-cause via up-onto `(2,8)`) | — |
| D4 down | 3 (1 as D9-cause via down-onto `(2,4)`, 2 clean) | — |
| D5 click spawn | 3, at 3 distinct cells/flight-lengths: `(4,8)` 5-tick, `(2,4)` 3-tick, `(1,6)` 2-tick (shortest, fired from row 1 — a row never used for firing in `train/` or `test50/`) | D6 below is the negative |
| D6 click ammo-0 | — | 2 (before any collection; after all 3 spawns/ammo exhausted) |
| D7 bullet-up | 6 across all 3 flights, incl. the flicker-tick and its restore-tick over `(2,8)` | the 5 D10 NO_CHANGE noops (no bullet present) |
| D8 despawn | 2 (col 4, col 6) | — |
| D9 coin collection | 3 causes (right-onto `(4,8)`, up-onto `(2,8)`, down-onto `(2,4)`) + 3 effects | D6/D10 negatives double as "nothing collected" contrast |
| D10 noop NO_CHANGE | n/a (IS the negative) | 5, spread irregularly (2 / 2 / 1), never a fixed cadence |

Contrastive-negative share: 2 D6-neg + 5 D10-neg = **7/32 ≈ 22%**.

### How train2's situations differ from `train/` and `test50/`

- **Coins collected are the 3 the other two pools never touch**: `(2,4)` down-onto,
  `(2,8)` up-onto, `(4,8)` right-onto — verified by diffing `clean_data2/qfsvc`'s raw
  100-step source (start-vs-end coin sets) against `test50/TEST_COVERAGE.md`'s explicit
  coin list. `train/` touches `(2,6)`,`(4,4)`,`(4,6)`; `test50` touches
  `(4,6)`,`(2,6)`,`(4,10)`,`(2,10)`,`(4,12)`,`(2,12)`.
- **Fire positions/flight lengths are new**: `train/` fires from `(4,3)` and `(2,7)`;
  `test50` fires from `(4,6)`,`(5,6)`,`(2,10)`×2,`(2,12)`,`(5,11)`. `train2` fires from
  `(4,8)` (longest, 5-tick), `(2,4)` (3-tick), and `(1,6)` (shortest, 2-tick — row 1 is
  new to the pool).
- **Despawn columns are new**: `test50` despawns at cols 6, 10(×2), 11, 12; `train2`
  despawns at cols 4 and 6, at different rows/flight-lengths than any prior despawn.
  (`train/`'s single despawn column is not documented in its COVERAGE table.)
- **The flicker-then-later-collect narrative is repeated on a different coin/column**:
  `test50` Drive A already shows a bullet flickering over `(2,6)` which is later
  collected via down-onto; `train2` shows the same structural contrast (transit ≠
  collection) for coin `(2,8)` via up-onto, fired from a different column/origin
  entirely, so no frame or agent position repeats — but this parallel is noted here for
  transparency.
- **Action cadence** never repeats the original 1-action/3-noop rhythm nor `test50`'s
  timings: double-noop open, a 5-move dash, 5-/3-/2-tick flights back to back, a
  double-noop before the tail.

### Verification

```
uv run python -c "import sys; sys.path.insert(0,'prototypes/perc_invdyn'); import clean_data3_tools as T; T.verify_pool('prototypes/perc_invdyn/clean_data3/qfsvc/train2','left,right,up,down,noop,click', context_k=9)"
```
reports **32 scored target transitions**; by action (raw, uncollapsed click params):
`{'click 9 7': 1, 'noop': 16, 'up': 3, 'right': 3, 'click 4 8': 1, 'left': 2, 'down': 3,
'click 2 4': 1, 'click 1 6': 2}` — collapsing the 4 click variants to the verb gives
`click: 5`, matching the table above. Every per-transition `classify()` tag matches the
intended dynamic (spawns show `mediumpurple+1 red-1`, collections show `black+1 red-1`
then `gold-1 red+1`, the flicker shows `gold-1 ... mediumpurple~move` then `gold+1 ...
mediumpurple~move`, all 7 negatives show `NO_CHANGE`). `train2/viz.html` (41 frames)
and `train_regen2/viz.html` (45 frames) hold the filmstrips.
