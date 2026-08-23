# n2ntd — clean_data3 coverage

Config: whitelist = `left,right,up,down,noop,click`; `keep_action_params=FALSE`
(movement game — `click R C` collapses to the verb `click`; the click LOCATION is NOT the label).

Grid colors: red=Mario, blue=enemy(3x2), gold=coin, darkorange=step platform (static),
mediumpurple=bullet, white=background. The passive clock ticks on EVERY action.

## 1. Core dynamics extracted from dynamics.txt

| # | dynamic |
|---|---------|
| D1 | **left** — Mario moves 1 col left (blocked by edge) |
| D2 | **right** — Mario moves 1 col right (blocked by edge) |
| D3 | **up = JUMP** — Mario moves up 4 rows (dy -4), ONLY when resting on floor (row 11); mid-air up is a no-op |
| D4 | **click = FIRE** — if ammo>0: spawn bullet at Mario's cell, ammo-1, nudge Mario down 1; LOCATION ignored. ammo==0 -> complete no-op |
| D5 | **down** — no handler, pure no-op (clock still ticks) |
| D6 | **noop** — no-op (clock still ticks) |
| D7 | **gravity** — every frame Mario falls 1 row until row 11 (falls THROUGH steps & coins; only bottom edge stops him) |
| D8 | **enemy patrol** — enemy moves 1 col/frame, flips direction at patrol ends (left edge col0 / right edge col11), independent of player action |
| D9 | **bullet rise** — bullet moves up 1/frame; FREEZES permanently one cell below a Step if its next-up cell hits a platform |
| D10 | **coin pickup** — Mario intersecting a coin removes it and gives +1 ammo |
| D11 | **bullet hits enemy** — enemy intersecting a bullet removes the bullet; enemyLives==1 so the enemy is killed (removed) |
| (D12) | **steps are STATIC** under seed 0 — non-dynamic (no observable motion); not a learning target |
| (D13) | **win/termination** — none defined (reward 0, done False); not a learning target |

D7/D8/D9/D11 + coin-side of D10 are PASSIVE: they fire under every action uniformly, so they are
**FD-informative but NOT ID-recoverable** (the action doesn't cause them). D3/D4 are the only player
effects that are also passive-coupled (jump/fire nudge Mario).

## 2. Original-train coverage vs the gap (the nrdf6 failure mode)

Original train = 1 episode, 57 rows. Whitelisted transitions are noop-dominated (~36 noop vs
4 up / 6 left / 6 right / 3 click / 1 down). A balanced-20 sample caps noop to ~3–4, so the
**rare passive events that live on single noop targets get dropped**:

| dynamic | TARGET under ID? | TARGET under FD? | in balanced-20 of ORIGINAL? | gap |
|---|---|---|---|---|
| D1 left | yes (red col-1) | yes | likely (left rare) | ok |
| D2 right | yes (red col+1) | yes | likely | ok |
| D3 jump | yes (red row-4) | yes | likely (up rare) | floor-vs-midair contrast absent |
| D4 fire | yes (bullet appears) | yes | maybe (click rare) | fire vs ammo0-no-op rarely BOTH sampled |
| D5 down | no (nothing happens) | yes (NO_CHANGE) | unlikely (only 1 down) | usually absent |
| D6 noop | n/a (no-op) | yes | yes | n/a |
| D7 gravity-fall | no (passive) | yes | falls live on noop -> mostly dropped | **fall pairs rarely scored** |
| D8 enemy patrol/flip | no (passive, every frame) | yes | move sampled; FLIP buried in noops | **flip target dropped** |
| D9 bullet rise | no (passive) | yes | rises buried in noops | **rise dropped** |
| D9 bullet FREEZE | no (passive) | yes (bullet stays) | freeze is 1 noop NO_CHANGE among 36 | **almost never sampled — the key gap** |
| D10 coin pickup | no (passive collision) | yes (gold vanishes) | pickup on noop/right, rare | **likely dropped** |
| D11 bullet kills enemy | no (passive collision) | yes (enemy+bullet vanish) | 1 noop target among 36 | **almost never sampled — key gap** |

**Shortcut risk (nrdf6-style):** because the enemy moves and bullets rise on *every* frame, a lazy
model can explain all motion with a per-step "clock" and never learn the CONDITIONAL rules
(bullet freezes only under a step; bullet/enemy vanish only on intersection; Mario falls only when
above row 11; click fires only when ammo>0). The curated pool below scores each conditional rule as
a target AND pairs it with a near-miss negative so the clock shortcut scores worse.

## 3. Curated slices (verbatim original rows) -> scored targets

9 slices, **20 scored targets** (pool == train-n=20, so balanced_split keeps all).
By action: noop 9, left 3, right 3, up 2, click 2, down 1.

| episode | orig steps | target pair(s) -> dynamic | contrastive role |
|---|---|---|---|
| 0 | 0,1 | 0->1 noop: enemy moves LEFT, Mario on floor does NOT fall | D8 + gravity NEG (on-floor -> no fall) |
| 1 | 3,4 | 3->4 up: red row11->row7 | D3 jump POSITIVE (from floor) |
| 2 | 12,13,14 | 12->13 noop: Mario falls, enemy reaches right edge; 13->14 noop: Mario falls, enemy FLIPS to leftward | D7 fall + D8 patrol FLIP |
| 3 | 27,28,29 | 27->28 noop: bullet rises into enemy; 28->29 noop: enemy(5 cells)+bullet vanish | D9 rise + D11 KILL POSITIVE |
| 4 | 30,31,32 | 30->31 click(ammo0): NO_CHANGE; 31->32 noop: NO_CHANGE | D4 fire NEG (ammo0) + D6 pure-noop NEG |
| 5 | 34,35,36 | 34->35, 35->36 left: red col-1 | D1 left |
| 6 | 38,39,40 | 38->39 left col-1; 39->40 up row11->row7 | D1 left + D3 jump (2nd, enemy dead) |
| 7 | 42,43,44,45 | 42->43 right: gold-1 (coin gone) red+1; 43->44, 44->45 right: red col+1 | D10 coin pickup + D2 right (2 clean) |
| 8 | 49,50,51,52,53,54 | 49->50 click(ammo>0): bullet appears at Mario; 50->51,51->52 noop: bullet rises; 52->53 noop: bullet FROZEN below step (NO move); 53->54 down: NO_CHANGE | D4 fire POSITIVE + D9 rise + D9 FREEZE NEG + D5 down no-op |

### Contrastive pairs (defeat shortcuts)
- **click**: fire (49->50, bullet+1) vs ammo0 no-op (30->31, NO_CHANGE) — both label `click`. Defeats "click always fires".
- **bullet**: rises (50->51, 51->52, 27->28) vs FREEZES below a step (52->53, NO move) — window carries the rising bullet, so FD must use the platform to predict the stop. Defeats "bullet moves up every frame".
- **noop / gravity**: Mario falls mid-air (12->13, 13->14) vs Mario on floor does not move (0->1); plus pure no-op when nothing is active (31->32, NO_CHANGE). Defeats "noop always changes state" and "Mario always falls".
- **enemy patrol**: steady move (0->1 left, 12->13 right) vs direction FLIP at the right edge (13->14). Defeats "enemy column = step*k" clock.
- **collision rules** as positives: coin vanishes only where Mario is (42->43); enemy+bullet vanish only on intersection (28->29).

### Known unidentifiable / unavailable cases (inherent, not fixable from this trajectory)
- **down vs noop** are fundamentally ID-indistinguishable (both pure no-ops); `down` is included only for FD (NO_CHANGE) coverage. Three NO_CHANGE targets (click-ammo0, noop, down, freeze) are ID-confusable but FD-correct and truthful to the game.
- **D3 mid-air jump = no-op** has NO clean negative in this trajectory: the only mid-air `up` in the original (step 6->7) anomalously still moves Mario up (engine/frame artifact), so it would teach the wrong rule and is deliberately excluded. Jump is covered by two from-floor positives only.
- All passive dynamics (D7/D8/D9/D10/D11) are FD-only targets (action not recoverable); they rely on the window + contrastive negatives rather than ID.

---
## PATCH: left-edge patrol bounce added
The initial set scored only the RIGHT-edge enemy direction-flip (13->14). The LEFT-edge
bounce existed in the trajectory (steps 4->5: enemy reaches cols 0-2 then reverses right)
but episode_1 was [3,4] — it stopped one frame before the reversal. Fixed by extending that
slice to [3,4,5] (4->5 now a scored target, prev=1 so the leftward approach is in-window and
the reversal is predictable) and dropping the redundant [0,1] steady-left noop. Both edge
bounces are now scored; still exactly 20 targets, same action histogram.

---
## TRAIN2 (untied-val expansion)

A second, independently-driven pool built so the combined train/+train2/ (~51 transitions)
supports an untied 30-train/30-val split for GEPA runs. Built per
`clean_data3_REGEN_METHODOLOGY.md`: a fresh `autumn_drive.py N2NTD` rollout (seed 0,
`train_regen2/`, 77 rows / 76 raw transitions, irregular action cadence) walking through
every core dynamic in situations that do not replicate `train/`'s original single rollout
or `test50/`'s 6 fresh drives, then curated down to `train2/` (10 episodes, **31 scored
target transitions**, pool ≤ `--train-n` so `balanced_split` keeps the whole pool).

### Engine facts discovered while driving (not previously documented here)

- **Mario rests ON platforms** (matches test50's note, confirmed independently): after a
  jump or fall, if the cell directly below is a Step, Mario stops one row above it and can
  jump again from there (`up` re-checked via "would moveDownNoCollision change position").
  Verified for all three platforms (row10, row8, **and row6** — row6 not previously jump-
  tested at all).
- **Interpreter quirk (new finding, worth flagging for future regen work on this game):**
  immediately after a coin-pickup's nudge-down resolves (the frame where Mario is
  repositioned via the pickup handler's own `moveDownNoCollision`), a same-frame `up`
  jump-attempt silently fails even when Mario is genuinely resting (reproduced twice,
  `/tmp` scratch drives). A single interposed **horizontal move** (even one that keeps the
  same row, e.g. sliding one cell further along the platform top) "cleanses" the object and
  the very next `up` succeeds. Root-caused empirically, not from the SEXP text; worked
  around in the final drive (see episode 6 below — `right` cleansing move before the jump).
- **CLI gotcha:** `autumn_drive.py` only rewrites `click_ROW_COL` (underscore form) into
  `click ROW COL`; a bare `click` action is passed through literally and the env's `(clicked)`
  predicate never fires (silent no-op, ammo untouched). Two early scratch drives wasted a
  freeze-fire and a kill-fire this way before the bug was found — always use `click_R_C`.

### Core dynamics covered as SCORED targets (fresh situations vs. train/ + test50/)

| Dyn | train2 scored pairs | Fresh vs. train/ + test50/? |
|---|---|---|
| D1 left | ep0 (2 clean, col2→1→0) — **[NEG]** ep0 (blocked at col0) | col0-blocked also appears in train/test50, but reached via a different action/coin/enemy history here |
| D2 right | ep1 (onto coin 9,1, background only), ep5 (onto coin 4,7), ep6 (onto row6-platform, onto coin 5,9, cleansing slide) | fresh coin(5,9) target never scored anywhere before; fresh row6-platform entry column (col8) |
| D3 up/jump | ep3 (floor jump col3, **open air**, no platform underneath) · ep5 (platform jump row8→row3, col6) · ep6 (**platform jump row6→row1, col10 — NOVEL**, row6 never jump-tested in train/test50) — **[NEG]** ep3 (midair-up at col3/row8, fresh column vs test50's col11) |
| D4 click/fire | ep4 (floor fire col5) · ep9 (fire col10, 3rd bullet) — **[NEG]** ep9 (ammo-0 click, same slice as the positive fire — direct contrast) |
| D5 down | ep7 (confirm-rest negative; enemy alive+patrolling in background) |
| D6 noop | 15/31 (carrier of all passive dynamics below) |
| D7 gravity | ep2 (leaves row10-platform support, falls to floor) · ep3 (open-air fall) · ep5 (fall row3→4, col6) · ep6/ep7 (fall back onto row6-platform) |
| D8 enemy patrol | ep1 (**LEFT-edge flip at t=22** — fresh cycle; train used t13, test50 used t4/t13/t31) + continuous background motion in nearly every other episode's window |
| D9 bullet rise | ep4 (2 rises, col5) · ep8 (2 rises, col10) · ep9 (2 rises, col10) |
| D9 FREEZE | ep4 (**freezes at (9,5)** — fresh location; train used (9,4), test50 used (9,6)) |
| D10 coin pickup | ep5 (coin 4,7: landing + resolve, **nudge succeeds** row4→row5) · ep6 (coin 5,9 — **never scored in train/ or test50/**: landing + resolve, **nudge BLOCKED** by row6-platform) — direct contrastive pair (same dynamic, opposite nudge outcome depending on platform presence) |
| D11 kill | ep8 (**col10**, enemy at its right patrol edge origin10/cols9-11 — fresh column vs train's col7 and test50's cols 3,7,11,1; single well-verified instance, timed via the enemy's exact deterministic patrol clock) |
| (D12/D13) | steps static, no win condition — unchanged, non-targets |

### Action histogram (pool of 31)

```
noop   15   (falls, rises, freeze+persist, kill, pickups, flip, down/ammo0 contexts)
right   5   (2 coin approaches + row6-platform entry + coin(5,9) + cleansing slide)
up      4   (floor-open-air, platform-row8, platform-row6[NOVEL], midair-NEG)
left    3   (2 clean + 1 blocked)
click   3   (2 fires + 1 ammo-0 negative — collapsed from `click 0 0`/`click 5 5` as
            reported by verify_pool; keep_action_params=FALSE at scoring time)
down    1
```

### Contrastive negatives (6 / 31 ≈ 19%)

| Negative | Pair | Shortcut it defeats |
|---|---|---|
| blocked left | ep0 (col0) | "left always moves Mario" |
| midair up | ep3 (col3, falls instead of jumping) | "up always jumps" |
| bullet freeze | ep4 (vs. the 2 rises earlier in the same slice) | "bullet always rises every frame" |
| nudge blocked by platform | ep6 coin(5,9) (vs. ep5 coin(4,7) nudge succeeding) | "pickup always nudges Mario down visibly" |
| down no-op | ep7 (enemy still patrolling in background) | "every action changes something" |
| ammo-0 click | ep9 (same slice as the positive fire two rows earlier) | "click always spawns a bullet" |

Ratio is slightly below the 20-30% target band (19%); the single available `down` action
and the single achievable fresh kill in this one-drive trajectory capped how many more
negatives could be added without a second regen drive (documented as a known limitation,
same spirit as the `down`/no-op caveats already noted for `train/` above).

### How train2/'s situations differ from train/ and test50/

| | train/ (original) | test50/ (6 fresh drives) | train2/ (this pool) |
|---|---|---|---|
| jumps | floor col6, col1 | floor col7, col11; platform row10→row5 col1 | floor col3 (**open air**, no landing); platform row8→row3 col6; **platform row6→row1 col10 (first-ever row6 jump)** |
| fires | floor (11,4) | floor (11,9),(11,6); midair (7,7),(7,11) | floor col5 (freeze) + col10 fired from **row6-platform height** (kill-setup) |
| kills | col7, enemy moving right | cols 3,7,11,1; moving left + both patrol edges | **col10**, enemy at the right patrol edge, bullet fired from a jumped-and-fallen-back platform position rather than the floor |
| freeze | (9,4) under row8 platform | (9,6) under row8 platform, (7,8) under row6 platform | **(9,5)** under row8 platform (third distinct column under that platform) |
| pickups | (9,1) via `right` (overridden) | (9,1) and (4,7) via clean `noop`; nudge-blocked vs nudge-visible variants | (9,1) unscored background + **(4,7) nudge-succeeds** and **(5,9) nudge-BLOCKED** scored as a direct contrastive pair — coin(5,9) never scored elsewhere |
| enemy flip | right-edge only (step13) | left/right edges at t4,13,31 (5 flip targets total) | **left-edge at t=22** — a cycle neither pool used |
| movement | floor col6 region, row9 platform (9,1)-(9,3) | floor cols8-11, platform row9 leftward, midair right landing on row6 | floor cols 0-3 region (left run to the wall), row6-platform lateral entry/exit at cols 7-10 |

No scored train2 pair reproduces a train/ or test50/ scored pair's (state, action) — the one
surface-level echo (blocked-left at col0, and the row10-platform jump concept) occurs with a
different action/coin/ammo/enemy-timing history in every case, so the underlying board state
differs even where the qualitative dynamic-name repeats.
