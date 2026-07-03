# bt3gb — coverage analysis (clean_data3)

Game: **bt3gb** — weather / day-night water sandbox (16x16). No goal/score/termination.
Config: whitelist = `left,right,up,down,noop,click`; **keep_action_params = FALSE**
(movement game — `click ROW COL` collapses to the verb `click`; the click LOCATION is
irrelevant to the dynamic, it is a GLOBAL toggle, so location is NOT the label).

## 1. Core dynamics (extracted from dynamics.txt)

| # | Dynamic | Rule |
|---|---------|------|
| D1 | **left** moves Cloud (3-cell gray bar) one cell left, bounds-clamped at cols 0,1,2 |
| D2 | **right** moves Cloud one cell right, bounds-clamped at cols 13,14,15 |
| D3 | **down** spawns a Water droplet below the cloud origin; the droplet's color = current day state (blue if day, lightblue if night) |
| D4 | **up** has NO handler — it is a no-op (only passive water update fires) |
| D5 | **noop** — no player effect (only the passive water update fires) |
| D6 | **click** GLOBAL toggle: flips CelestialBody day (gold<->gray) AND flips `liquid` of EVERY droplet (blue<->lightblue) at once; location ignored |
| D7 | **passive/clock**: every tick each Water droplet falls one cell — liquid (blue) slides sideways along the floor (spread), solid (lightblue) stacks vertically; settled/empty water => no motion |

## 2. Target-coverage of the ORIGINAL train pool (the gaps)

Original `clean_data2/bt3gb/train` = 64 rows / 63 transitions. Action distribution:
~45 noop, 7 down, 4 left, 1 right, 1 up, 2 click. A balanced-by-action sample of 20
would take all of {up=1, right=1, click=2} but subsample the 7 downs / 4 lefts / many
noops at random.

| Dynamic | TARGET under ID? | TARGET under FD? | In a balanced-20 of the original? | GAP |
|---|---|---|---|---|
| D1 left-move | yes (cloud displaces -1 col) | yes (gray bar shifts) | likely (≥1 of 4 lefts) | ok |
| D1 left-**clamp** (near-miss) | n/a (no change) | yes (action fires, NO move => defeats "left always moves") | only 1 of 4 lefts (31->32) — easily dropped | **GAP: clamp negative may be absent** |
| D2 right-move | yes | yes | yes (only right, kept) | ok |
| D3 down-**day** (blue spawn) | yes (new cell at cloud-origin col) | yes (new blue cell) | yes (6 of 7 downs are day) | ok |
| D3 down-**night** (lightblue spawn) | yes | yes (predicting LIGHTBLUE requires day=false) | only 1 of 7 downs (51->52) — random subsample may drop it | **GAP: day/night contrastive for down may be lost** |
| D4 up no-op | up indistinguishable from noop (same passive-only effect) | only passive water | kept (only up) | inherent: **up unidentifiable from noop under ID** (kept as context) |
| D6 click day->night | yes (mass recolor recovers click) | yes (gold->gray, blue->lightblue) | yes (kept) | ok |
| D6 click night->day | yes | yes (lightblue->blue, gray->gold) | yes (kept) | ok — but pairing both directions matters |
| D7 passive water fall | NOT recoverable (any action also ticks water => noop not identifiable from motion) | yes (must predict droplet falling) | falling noops dominate | inherent ID gap; FD-only |
| D7 passive **no-motion** (settled/empty) | n/a | yes — defeats "water always falls each tick" / `step%k` clock shortcut (the nrdf6 trap) | the 6 NO_CHANGE noops are clustered at start + steps 32-34; a random noop subsample may pick only falling noops | **GAP: passive contrastive negative not guaranteed** |

Summary of original-pool gaps (nrdf6-style): the *conditional* and *near-miss* variants
(down-at-night, left-at-clamp, settled/empty noop) are each represented by only 1–6 of many
same-verb transitions, so a balanced-20 random draw can easily leave a dynamic testable only
as window context and let a lazy "noop always moves water" / `step%k` clock rule survive.

## 3. Curated slices (verbatim original rows; each internal pair = a scored target)

Each slice is its own `episode_N`; windows are real consecutive frames and never bleed
across slices.

| episode | original steps | target pairs (action @ first step) | dynamic(s) covered |
|---|---|---|---|
| 0 | 2,3,4,5,6 | 2->3 noop (NO_CHANGE, empty); 3->4 **down**(blue spawn, day); 4->5 noop(blue fall); 5->6 noop(blue fall) | D3 down-day (+ID col), D5/D7 passive liquid fall, **D7 no-motion negative** |
| 1 | 18,19,20 | 18->19 noop(fall); 19->20 **left** (cloud -1 col) | D1 left-move, D7 |
| 2 | 30,31,32,33 | 30->31 noop(blue **slides** +0.25 col); 31->32 **left** CLAMP (no gray move); 32->33 noop(NO_CHANGE settled) | **D1 left-clamp negative**, D7 liquid-spread, **D7 no-motion negative** |
| 3 | 38,39,40 | 38->39 noop(fall); 39->40 **right** (cloud +1 col) | D2 right-move, D7 |
| 4 | 43,44,45 | 43->44 **up** (only passive water); 44->45 noop(fall) | **D4 up==noop**, D7 |
| 5 | 47,48,49 | 47->48 **click** day->night (gold->gray, blue->lightblue x5); 48->49 noop(lightblue/**solid** fall) | D6 click d->n, D7 solid fall |
| 6 | 50,51,52 | 50->51 noop(lightblue fall); 51->52 **down**(**lightblue** spawn, night) | **D3 down-night (color contrast vs ep0)**, D7 |
| 7 | 54,55,56,57 | 54->55 noop(lightblue fall); 55->56 **click** night->day (lightblue->blue, gray->gold); 56->57 noop(blue fall) | D6 click n->d (reverse), D7 |

### Contrastive pairs (defeat shortcuts)
- **left move (ep1) vs left clamp (ep2)** — same verb, cloud moves vs does not => "left always shifts gray" scores worse than the bounds-clamp rule.
- **down-day blue (ep0) vs down-night lightblue (ep6)** — spawn color is conditional on day state; a color-blind "down spawns blue" rule scores worse.
- **click d->n (ep5) vs click n->d (ep7)** — both toggle directions present so the rule is the flip, not a fixed target color.
- **up (ep4) ≈ noop** — establishes up is a true no-op (passive-only), the only contrast available since up is ID-unidentifiable from noop.
- **passive fall noops vs NO_CHANGE noops (ep0 2->3, ep2 32->33)** — water falls only when unsettled; defeats the nrdf6 `step%k` / "water always moves each tick" clock shortcut.
- **liquid slide (ep2 30->31, +col) vs solid straight fall (ep5/6 lightblue)** — exercises the liquid-spread vs solid-stack passive distinction.

## 4. Final pool composition (verify_pool)

20 scored target transitions (pool == --train-n 20, so balanced_split keeps ALL of them).

By verb (keep_action_params=FALSE collapses `click R C` -> `click`):
`noop=12, down=2, left=2 (1 move + 1 clamp), right=1, up=1, click=2 (d->n + n->d)`.

All 7 core dynamics are exercised as scored targets under their applicable objective
(D4 up and D7-on-noop are ID-unidentifiable by construction and are covered under FD +
as contrast), with explicit near-miss negatives for left-clamp, down-color, click-direction,
and passive no-motion. The 12 noops are not redundant filler: 9 carry the passive
fall/spread/solid dynamics (FD) and provide the in-episode context windows for the adjacent
action targets, and 2 are the NO_CHANGE clock-shortcut negatives.

---
## REGENERATED trajectory (ice stacking)

The original clean_data2/bt3gb rollout never demonstrated the defining SOLID behavior —
ice droplets piling into a vertical column. Its single short night phase only froze an
already-spread liquid layer (a flat ice row), and the one fresh solid droplet melted before
stacking. So "solid stacks vertically / liquid spreads sideways" was never a scored target.

Fixed by **regenerating the trajectory** with autumn_drive.py driving the real BT3GB.sexp
(saved at `train_regen/episode_0/trajectory.csv`, filmstrip `train_regen/viz.html`). The
scripted rollout shows the full arc in one column:
  DAY liquid spreads (flat row) -> click FREEZE (day->night, all blue->lightblue)
  -> NIGHT solid droplets STACK into a vertical ice column -> click MELT (night->day,
  all lightblue->blue) -> column collapses and spreads as liquid.

The curated train/ is now sliced from train_regen (20 targets). Key distinctions are now
scored: FREEZE (`blue->lightblue,gold->gray`), MELT (`lightblue->blue,gray->gold`),
solid STACK (`lightblue~move(X,0.0)` pure vertical), liquid SPREAD (`blue~move(X,Y)` lateral).
(dynamics.txt unchanged; test/ still the original — regenerate it too for full consistency.)
