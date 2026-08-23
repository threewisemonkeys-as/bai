# 27vwc TEST50 — large held-out test pool (50 scored transitions)

"Cook the meat on the BBQ to feed the person." Config: whitelist = `noop,click`;
**keep_action_params = TRUE** (the full click string, e.g. `click 4 3`, is the ID label).

- **Pool size (verify_pool, context_k=9): 50** scored target transitions.
- **Freshly DRIVEN**, distinct from `train/` (see §5). Built by slicing seven fresh
  `autumn_drive.py 27VWC` rollouts (seed 0) with hand-designed action sequences:
  `driveT` (toggles), `driveCook`/`driveA`-style burns (shifted-timing passives),
  `driveRefuel`/`driveRef2` (gas-0 interactions + refuel crossings), `driveFeed`/
  `driveFeed2`/`driveFeed3` (feeds at varied doneness/health), `driveExtra`
  (toggles at pink meat + empty clicks + a 2nd gas-0 tail). Rows copied verbatim.

## 1. Core dynamics (mechanics from dynamics.txt) and their sub-cases

Interactive (click) mechanics:
- **Toggle fire** (click any BBQ cell — canonical burner `click 4 3`): D1 ON (white->orange),
  D2 OFF (orange->white); **negative** D3 blocked when gas==0 (NO_CHANGE, cannot relight).
- **Refuel** (`click 6 0` FillBBQ): D4 = +5 gas crosses a gauge threshold (gauge white->yellow);
  **negative** D5 = +5 that stays in the same gauge band (NO_CHANGE).
- **Feed** (`click 3 3` meat): outcome depends on meat doneness read from its color, then meat
  resets to lightblue. D6 well-cooked/sandybrown -> health +1; D7 raw or pink (<30) -> -1;
  D8 burnt/brown (>60) -> -2.

Passive (per-tick, on `noop`) mechanics:
- **Cooking** (fire on => cooked++): D9 lightblue->pink (cooked=10), D10 pink->sandybrown (30),
  D11 sandybrown->brown (60).
- **Gas burn** (fire on => gas--): D12 upper gauge yellow->white (gas=20), D13 lower gauge
  yellow->white (gas=0).
- **Auto-extinguish**: D14 burner orange->white the tick after gas hits 0.
- **Contrastive passives**: D15 = noop with fire OFF (meat unchanged); D16 = noop with fire ON
  but not at a 10/30/60 boundary (NO_CHANGE — gradual cooking is only observable at crossings).

## 2. Coverage map — dynamic -> episodes, count, ID/FD role

| dyn | pos/neg | count | episodes (step ranges) | ID-informative? | FD-informative? |
|---|---|---|---|---|---|
| D1 toggle ON    | pos | 4 | ep0[2-6], ep1[8-12], ep2[14-21], ep3[64-74] | yes (burner white->orange => click 4 3) | yes (burner flip) |
| D2 toggle OFF   | pos | 4 | ep0, ep1(x2), ep2 | yes (orange->white, gauge yellow => click 4 3, distinct from D14) | yes |
| D4 refuel+cross | pos | 4 | ep3[64-74] (cross0 + cross20), ep20[48-49], ep21[66-67] | yes (gauge white->yellow => click 6 0) | yes (gauge gain) |
| D6 feed +1      | pos | 3 | ep6[48-49], ep7[55-56] (h4->5), ep8[106-107] (h1->2) | yes (meat->lightblue + health up => click 3 3) | yes (health+1 iff sandybrown) |
| D7 feed -1      | pos | 3 | ep4[65-71], ep5[13-14], ep11[62-63] | yes (meat reset + health down1) | yes (health-1 iff cooked<30) |
| D8 feed -2      | pos | 2 | ep9[62-63] (h3->1), ep10[74-75] (shifted) | yes (meat reset + health down2) | yes (health-2 iff brown) |
| D9 cook l->pink | pos | 3 | ep4(@s65), ep12[17-18], ep13[15-16] | trivial (noop) | yes (meat color advance) |
| D10 cook p->sand| pos | 3 | ep14[37-38], ep15[43-45], ep16[29-30] | trivial (noop) | yes |
| D11 cook s->brown| pos| 2 | ep18[66-68], ep19[59-60] | trivial (noop) | yes |
| D12 gas cross20 | pos | 2 | ep15[43-45] (@s44), ep17[53-54] | trivial (noop) | yes (upper gauge) |
| D13 gas cross0  | pos | 2 | ep3[64-74] (@s64), ep4[65-71] (@s68) | trivial (noop) | yes (lower gauge) |
| D14 auto-off    | pos | 2 | ep3 (@s65), ep4 (@s69) | trivial (noop; gauge=white distinguishes from D2) | yes (burner off) |
| D3 blocked g0   | **neg** | 2 | ep3[64-74] (two `click 4 3` at gas0) | aliased (NO_CHANGE) | yes (predict NO_CHANGE vs D1/D2) |
| D5 refuel noCross| **neg**| 3 | ep3[64-74] (three `click 6 0`) | aliased (NO_CHANGE) | yes (predict NO_CHANGE vs D4) |
| D15 noCook fireOFF| **neg**| 4 | ep0(x2), ep2(x2) | trivial (noop) | yes (predict NO_CHANGE — defeats "noop cooks") |
| D16 subThreshold | **neg**| 5 | ep1, ep2, ep4(x2), ep18 | trivial (noop) | yes (predict NO_CHANGE — defeats step-clock) |
| EMPTY click     | **neg** | 2 | ep2 (`click 2 2`, `click 5 5`) | aliased (NO_CHANGE) | yes (not every click acts) |

Positives = 34, Negatives = 16 (**32%**).

## 3. Action histogram of the pool (keep_action_params)

`noop`: 23, `click 4 3` (burner/toggle): 10, `click 3 3` (meat/feed): 8, `click 6 0`
(fill/refuel): 7, `click 2 2` (empty): 1, `click 5 5` (empty): 1.

Click locations span the five interactive/empty targets. Toggles are ALWAYS on the burner
cell `4 3` where the fire effect appears (so the location is ID-recoverable); feeds on the
meat cell `3 3` (which visibly resets there); refuels on the fill button `6 0`.

## 4. Contrastive structure (defeats shortcuts)

- **Toggle conditional (D1/D2 vs D3):** identical `click 4 3` gives white->orange, orange->white,
  and NO_CHANGE (blocked at gas0, ep3). Forces "toggle iff gas>0", not "click 4 3 toggles".
- **Refuel conditional (D4 vs D5):** identical `click 6 0` gives gauge-yellow-gain vs NO_CHANGE
  (ep3, ep20, ep21). Defeats both "click 6 0 always fills the gauge" and "never matters".
- **Feed outcomes (D6/D7/D8):** all three `click 3 3` health deltas (+1 sandybrown, -1 raw/pink,
  -2 brown) are present at DIFFERENT health levels (h1..h5), so the delta must be read from the
  meat color, not the click label.
- **Cooking vs no-cook (D9/D10/D11 vs D15/D16):** meat advances on `noop` ONLY with fire on AND
  at a 10/30/60 boundary. Negatives: fire-OFF noops (D15) and fire-ON sub-threshold noops (D16).
- **Anti-clock timing:** the crossings are placed at NON-round, non-periodic steps by inserting
  fire-off gaps in the drives (D9 at driven steps 65/17/15; D10 at 37/43/29; D11 at 67/59;
  D12 at 44/53; D13/D14 at 64-65 and 68-69). A `step % k` or "meat goes pink at step 10" rule
  scores worse than the true cooked-counter rule.
- **D14 vs D2 (both burner orange->white):** kept distinguishable by gauge state — D14 occurs
  with the lower gauge already WHITE (gas=0) in both frames, D2 with it YELLOW (gas>0).

## 5. How the test differs from train (cross-trajectory generalization)

`train/` was carved from the ORIGINAL clean_data2 rollout (original steps 3-194, ~19 targets,
mostly ~1 instance per dynamic at that rollout's fixed timing). TEST50 is freshly driven and
differs in:
- **Timing:** every passive crossing occurs at a DIFFERENT, non-round driven step than train's
  (train D9 at orig s13; test D9 at driven s15/17/65). Explicitly anti-clock.
- **State configs:** feeds occur across the full health range h1..h5 (train had isolated single
  feeds); toggles occur on PINK meat and as a gas-25 relight (train toggled at gas-full raw meat).
- **New situations:** refuel-from-gas-0 (D4 crossing 0, absent from train which only crossed 20);
  two blocked `click 4 3` at gas 0 inside a full gas-0 -> refuel -> relight sequence (ep3);
  a second independent gas-0 auto-off tail (ep4).
- **Density:** 2-4 instances per dynamic vs train's ~1, so the pool ranks beliefs, not just
  detects presence.

## 6. Windows and uncoverable items

- **Window context:** 35/50 transitions carry a temporal window (ctx_prev/ctx_next up to 9);
  15 are 2-row episodes with empty windows. The windowless ones are visible-outcome feeds/refuels
  (outcome is fully in X_t) or DUPLICATE crossings whose primary instance is windowed elsewhere,
  so every dynamic has >=1 windowed target (e.g. D9 windowed in ep4, D10/D12 in ep15, D11 in ep18,
  D4/D13/D14 in the 11-row ep3).
- **ID aliasing (unavoidable, minimized to 14%):** the 7 NO_CHANGE non-noop negatives (D3 x2,
  D5 x3, EMPTY x2) are ID-ambiguous — a NO_CHANGE frame cannot reveal which click occurred, so an
  ID oracle guessing `noop` is wrong on them. They are retained only for their FD contrast (they
  defeat "click X always changes the frame") and kept to ~1/4 of the negatives. All 43 other
  transitions are ID-recoverable by an oracle who reads the rules.
- **D14 auto-off (2, not >=4):** each auto-off requires burning gas from 65 to 0, i.e. ~65 fire
  ticks; producing 4 would need ~4 full gas-0 burns and balloon the pool. Covered 2x at different
  driven steps/contexts; the gas-burn+auto-off MECHANIC (D12+D13+D14) is collectively covered 6x.
- **`keep_action_params` remote effects:** refuel's effect (gauge) and feed's health effect are
  NOT spatially at the click cell (the buttons are remote controls); they are still UNIQUELY
  recoverable (gauge-gain <=> `click 6 0`; health-change+meat-reset <=> `click 3 3`), so ID holds
  even though the effect is not co-located with the click. Only the burner toggle and the meat
  reset land ON the clicked cell.
