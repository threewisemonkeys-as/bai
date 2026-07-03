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
