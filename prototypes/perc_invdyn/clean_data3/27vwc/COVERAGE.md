# 27vwc coverage — "Cook the meat on the BBQ to feed the person"

Config: whitelist = `noop,click`; keep_action_params = TRUE (click LOCATION is part of the
label, e.g. `click 4 3`). Move verbs (left/right/up/down) are no-op handlers and are dropped
by the whitelist; the only such rows in the original train traj are steps 197/201/205/209,
all OUTSIDE every curated slice, so no window is truncated.

## 1. Core dynamics extracted from dynamics.txt

Interactive (click) dynamics:
- **D1 Fire toggle ON** — click any of the 9 BBQ cells (burner `click 4 3` etc.) when gas>0:
  burner cell obs[4][3] white -> orange.
- **D2 Fire toggle OFF** — same click when fire is on: burner orange -> white.
- **D3 Toggle blocked at gas==0** — clicking the BBQ when gas==0 does nothing (cannot relight
  an empty tank). NO_CHANGE. *(near-miss negative for D1/D2.)*
- **D4 Refuel (+5 gas)** — `click 6 0` (FillBBQ button); when the +5 pushes gas across a gauge
  threshold the gauge cell turns yellow (obs[5][3] or obs[6][3] white -> yellow).
- **D5 Refuel with no threshold crossing** — `click 6 0` that adds gas but stays in the same
  gauge band: NO_CHANGE. *(near-miss negative for D4 — defeats "click 6 0 always changes gauge".)*
- **D6 Feed well-cooked (+1 health)** — `click 3 3` when meat cooked in [30,60] (sandybrown):
  health bar gains a blue cell AND meat resets to lightblue (cooked->0).
- **D7 Feed raw/undercooked (-1 health)** — `click 3 3` when cooked<30 (lightblue/pink):
  health bar loses a blue cell; meat resets.
- **D8 Feed burnt (-2 health)** — `click 3 3` when cooked>60 (brown): health -2; meat resets.

Passive / per-tick (clock) dynamics:
- **D9 Cooking raw->rare** — on any tick with fire on, cooked++; at cooked=10 meat lightblue->pink.
- **D10 Cooking rare->good** — at cooked=30 meat pink->sandybrown.
- **D11 Cooking good->burnt** — at cooked=60 meat sandybrown->brown.
- **D12 Gas burn, gauge crosses 20** — fire on => gas-- each tick; at gas=20 upper gauge
  obs[5][3] yellow->white.
- **D13 Gas burn, gauge crosses 0** — at gas=0 lower gauge obs[6][3] yellow->white.
- **D14 Fire auto-extinguish at gas==0** — when gas reaches 0, fire is forced off the next
  tick: burner orange->white on a noop (no click).
- **D15 No cooking when fire off** — noop with fire off => meat color unchanged.
  *(contrastive negative — defeats "noop always advances the meat".)*
- **D16 Sub-threshold cooking** — noop with fire on but cooked not at a 10/30/60 boundary =>
  NO_CHANGE (cooking is gradual; only crossings are observable). *(natural negative.)*

## 2. Was each dynamic a scored TARGET in the ORIGINAL balanced-20 pool? (the gap)

The original train traj is 215 rows, 193 of 214 transitions are NO_CHANGE noops. A balanced-20
sample (`keep_action_params`, 4 distinct action labels) would draw ~5 per label; the ~5 noops
would almost certainly be drawn from the 193 NO_CHANGE majority, so the passive dynamics would
appear almost only as window CONTEXT, never as the scored target — the nrdf6 failure mode.

| Dynamic | informative pairs in original traj | scored TARGET likely in balanced-20? | gap |
|---|---|---|---|
| D1 toggle ON   | s7->8, s107->108            | maybe (click 4 3 label) | shared label with D2/D3, may be crowded out |
| D2 toggle OFF  | s3->4, s59->60              | maybe | same |
| D3 blocked g0  | s193->194 (only 1)          | unlikely (1 of ~5 click43) | near-miss almost never scored |
| D4 refuel +    | s67->68 (only 1)            | unlikely | 1 informative vs ~9 NO_CHANGE click 6 0 |
| D5 refuel NC   | s63/71/75/... (many)        | yes (dominates click 6 0) | so click 6 0 reduces to a "does nothing" prior |
| D6 feed +1     | s43->44 (only 1)            | maybe (1 of 3 click 3 3) | thin |
| D7 feed -1     | s47->48 (only 1)            | maybe | thin |
| D8 feed -2     | s189->190 (only 1)          | maybe | thin |
| D9 raw->rare   | s13->14, s57->58            | **no** (lost in 193 NC noops) | passive, context-only |
| D10 rare->good | s33->34, s125->126          | **no** | passive, context-only |
| D11 good->burnt| s155->156 (only 1)          | **no** | passive, context-only |
| D12 gas->20    | s49->50, s153->154          | **no** | passive, context-only |
| D13 gas->0     | s173->174 (only 1)          | **no** | passive, context-only |
| D14 auto-off   | s174->175 (only 1)          | **no** | passive noop; also gameable by step-counter |
| D15 fire-off NC| abundant                    | yes (majority) | over-represented, no contrast |
| D16 sub-thresh | abundant                    | yes (majority) | over-represented |

Gaps: every passive dynamic (D9–D14) and the three near-miss negatives that pin down the click
conditionals (D3 blocked, D4-vs-D5 refuel) are essentially never scored as targets in a default
balanced-20 sample — they survive only as context. The curated pool forces each to be a target,
contrastively.

## 3. Curated pool — slices and the dynamic each target pair covers

Pool = 19 scored targets (`verify_pool`): click 4 3 ×4, click 3 3 ×3, click 6 0 ×2, noop ×10.
Pool size ≤ train-n (20) so balanced_split keeps ALL of them; slices are their own episodes so
windows never bleed across slices. (Action labels = action AT the first step of each pair.)

| episode | orig steps | target pair(s) | action | change | dynamic |
|---|---|---|---|---|---|
| 0  | 3,4        | 3->4    | click 4 3 | orange-1 white+1            | **D2** toggle OFF (gas full) |
| 1  | 6,7,8      | 6->7    | noop      | NO_CHANGE (fire OFF)        | **D15** no-cook negative |
|    |            | 7->8    | click 4 3 | orange+1 white-1            | **D1** toggle ON |
| 2  | 12,13,14   | 12->13  | noop      | NO_CHANGE (fire ON)         | **D16** sub-threshold negative |
|    |            | 13->14  | noop      | lightblue-1 pink+1          | **D9** cook raw->rare |
| 3  | 33,34      | 33->34  | noop      | pink-1 sandybrown+1         | **D10** cook rare->good |
| 4  | 43,44      | 43->44  | click 3 3 | black-1 blue+1 sandybrown->lightblue | **D6** feed +1 + meat reset |
| 5  | 47,48      | 47->48  | click 3 3 | black+1 blue-1             | **D7** feed -1 (raw) |
| 6  | 48,49,50   | 48->49  | noop      | NO_CHANGE (fire ON)         | **D16** sub-threshold negative |
|    |            | 49->50  | noop      | white+1 yellow-1            | **D12** gas gauge crosses 20 |
| 7  | 63,64      | 63->64  | click 6 0 | NO_CHANGE                   | **D5** refuel no-threshold negative |
| 8  | 67,68      | 67->68  | click 6 0 | white-1 yellow+1            | **D4** refuel +5 (gauge up) |
| 9  | 107,108    | 107->108| click 4 3 | orange+1 white-1            | **D1** toggle ON (reinforce) |
| 10 | 154,155,156| 154->155| noop      | NO_CHANGE                   | sub-threshold negative |
|    |            | 155->156| noop      | brown+1 sandybrown-1        | **D11** cook good->burnt |
| 11 | 173,174,175| 173->174| noop      | white+1 yellow-1            | **D13** gas gauge crosses 0 |
|    |            | 174->175| noop      | orange-1 white+1            | **D14** fire auto-off at gas==0 |
| 12 | 189,190    | 189->190| click 3 3 | black+2 blue-2 brown->lightblue | **D8** feed -2 (burnt) + reset |
| 13 | 193,194    | 193->194| click 4 3 | NO_CHANGE                   | **D3** toggle blocked at gas==0 negative |

## Contrastive structure (defeats shortcuts)

- **Click-BBQ conditional (D1/D2 vs D3):** identical `click 4 3` label produces orange+1
  (on), orange-1 (off), and NO_CHANGE (blocked at gas==0, ep13 — burner white, gauge both
  white). Forces the rule "toggle iff gas>0", not "click 4 3 toggles".
- **Refuel conditional (D4 vs D5):** identical `click 6 0` label gives gauge yellow+1 (ep8)
  vs NO_CHANGE (ep7). Defeats both "click 6 0 always changes the gauge" and "click 6 0 never
  matters" — effect depends on crossing a gas threshold.
- **Cooking conditional (D9/D10/D11 vs D15/D16):** meat advances on noop ONLY with fire on
  AND at a 10/30/60 boundary. Negatives: noop with fire OFF (ep1, D15) and noop with fire ON
  but sub-threshold (ep2/ep6/ep10, D16). Defeats "noop always advances meat" and any
  `step % k` clock shortcut (the boundary steps 13/33/155 are not periodic).
- **Gas/auto-off (D12/D13/D14):** gauge-yellow-loss noops (ep6/ep11) contrast with refuel's
  gauge-yellow-gain click (ep8); the auto-off noop (ep11 174->175) sits in the SAME slice as
  the gas->0 noop (173->174) so the cause (gauge reaching both-white) is in-window, not a clock.
- **Feed outcomes (D6/D7/D8):** all three `click 3 3` outcomes (+1 sandybrown, -1 lightblue,
  -2 brown) are present, so the health delta must be read from the meat color, not the click.
