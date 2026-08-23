# dgg2c — TEST50 held-out test pool coverage

Config: whitelist = `left,right,up,down,noop,click`; **`keep_action_params=TRUE`**
(click games — the full action string `click ROW COL` is the ID label, so `click 11 3`
and `click 2 9` are distinct labels). Pool = **49 scored target transitions** across 8
freshly-driven episodes (seed 0). `balanced_split` returns the whole pool when pool
size ≤ `--test-n`, so run with `gepa_optimize.py --test-run .../dgg2c/test50 --test-n 50`
and every pair below is scored.

Verified: `T.verify_pool('.../dgg2c/test50','left,right,up,down,noop,click',context_k=9)`
→ 49 scored transitions; the causing `left`/`right` of every drift / clamp / despawn
target sits inside that target's `ctx_prev` window (prev distances 1–6, all < context_k=9),
so the delayed wind effect and its cause live in the same slice.

dgg2c is a **rain** game: a static gray cloud on rows 0–1; lightblue water spawns at row 2
(cols 2,6,10,14) every 5 steps and falls 1 row/step under gravity; a hidden integer
`wind ∈ [−1,1]` (set by `left`=−1 / `right`=+1; `up/down/click/noop` are no-ops) steers the
fall (0→straight, +1→down-right, −1→down-left); off-grid water despawns. No win/collision/
color rules.

## 1. Core dynamics (from dynamics.txt)

| # | dynamic |
|---|---------|
| D1 | Gravity straight fall (wind 0): every water cell moves down 1 row, col unchanged |
| D2 | Wind-right drift (wind +1): water moves down AND right (diagonal) |
| D3 | Wind-left drift (wind −1): water moves down AND left (diagonal) |
| D4 | Spawn: when the clock hits, 4 water cells appear at row 2, cols 2,6,10,14 |
| D5 | No-spawn step (negative of D4): nothing new appears on the other 4/5 steps |
| D6 | Off-grid despawn: water leaving the 17×17 bounds is removed (count drops) |
| D7 | `left` → wind = max(wind−1,−1)  (incl. clamp at −1) |
| D8 | `right` → wind = min(wind+1,+1) (incl. clamp at +1) |
| D9 | `up` = no-op |
| D10 | `down` = no-op |
| D11 | `click` = no-op, LOCATION irrelevant |
| D12 | `noop` = no effect (passive clock still runs) |

## 2. Intrinsic observability limit (read BEFORE the table)

**No action in dgg2c is ID-recoverable from a single `(X_t, X_t+1)` pair.** The engine
applies the clock from the START-of-step state, so a `left`/`right` write to `wind` only
changes water motion on the *next* step (verified: `right`@ep1 s8→9 shows STRAIGHT fall, the
down-right turn lands on the following `noop` s9→10). Consequences:

- `up`, `down`, `click`, `noop` never move a cell → **mutually ID-aliased** with each other.
- `left`/`right` have **no same-step visible effect** (the wind write is delayed) → at their
  own target the water just *continues its prior direction* (a plain fall or the existing
  drift), so they are **ID-aliased with `noop` too**. The only visible direction *change*
  always lands one step later, on a `noop`.
- `click` produces **nothing at the clicked cell** → the click LOCATION is unrecoverable
  (D11: location irrelevant), so `keep_action_params=TRUE` splits the 4 clicks into 4
  singleton labels that an oracle still cannot place.

So an oracle that knows every rule scores **~chance on ID**. This is the game's defining
property (dynamics.txt: delayed wind + a no-op-heavy action set; see `COVERAGE.md` "ID is
fundamentally weak/adversarial here"), **not** a curation defect, and cannot be removed by
data selection. The pool's value is concentrated on **FD** — water moves, spawns, or
despawns on essentially every step, and predicting the *direction* requires reading `wind`
from the window (the prior fall direction / the last `left`/`right` in `ctx_prev`). The set
is built to make that window signal maximally legible and every shortcut strictly worse.

## 3. Per-dynamic coverage (positives AND negatives, under ID and FD)

Episode → source drive: ep0 straight-rain (wind 0) · ep1 wind-right · ep2 wind-left ·
ep3/ep4 up/down/click no-ops · ep5 anti-shortcut reversal · ep6 wind-right (diff phase) ·
ep7 wind-left (diff phase). Step pairs below are the verbatim driven `Step` values.

| Dynamic | Scored TARGET positives | ID | FD | Contrastive negatives |
|---|---|---|---|---|
| **D1** straight fall (wind 0) | **≥18** wind-0 down-only targets: all of ep0 (s8→12), the delayed `left`/`right` "cause" steps, both `noop`s in ep5 (s7→9), ep6 s8→10, ep7 s8→9, and every up/down/click fall in ep3/ep4 | aliased (no action signal) | YES — predict down 1 row, col unchanged | the drift steps (D2/D3) are "wind≠0 ⇒ NOT straight"; wind read from window |
| **D2** right drift (wind +1) | **10** down-right targets: ep1 s9→10,10→11,11→12,12→13; ep6 s10→11,11→12,12→13,13→14,14→15,15→16 | aliased | YES — predict down-AND-right; requires wind=+1 from window | vs 11 down-left (D3) & ≥18 straight (D1) with the SAME `noop` label ⇒ direction is window-only, not action-only |
| **D3** left drift (wind −1) | **11** down-left targets: ep2 s9→10,10→11,11→12,12→13; ep5 s4→5,5→6,6→7; ep7 s10→11,11→12,12→13,13→14 | aliased | YES — predict down-AND-left | mirror of D2; defeats "noop ⇒ straight" |
| **D4** spawn (+4) | **8** clean +4 spawns: ep0 s12→13; ep1 s7→8,12→13; ep2 s7→8,12→13; ep3 s7→8; ep4 s12→13; ep5 s7→8 (+2 masked, see D6) | passive (no action) | YES — predict 4 new cells at row 2 cols 2,6,10,14 | 39 no-spawn targets (D5); spawns fall at an IRREGULAR step-index across episodes (presses shift the phase-relative view) so a `step%5` clock scores worse than the +4-cell rule |
| **D5** no-spawn (¬D4) | **39** non-spawn targets (all episodes) | trivial | trivial (row 2 stays empty) | stands against the 8 clean spawns; irregular action gaps kill any fixed cadence |
| **D6** off-grid despawn | **4 clean side despawns**: ep1 s11→12 (−2, right edge), ep2 s11→12 (−2, left edge), ep5 s6→7 (−1, left edge), ep6 s15→16 (−1, right edge) · **2 masked**: ep6 s12→13, ep7 s12→13 (spawn+4 & despawn−2, net **+2**) | passive | YES — predict the count DROP at the exact edge cells | masked pairs defeat "count-up ⇒ only spawn / count-down ⇒ only despawn"; bottom-edge despawn is phase-locked to a spawn (always masked) — documented, side despawns used instead |
| **D7** `left` rule | **5** `left` targets — cause ×3: ep2 s8→9, ep5 s3→4, ep7 s9→10 (all show STRAIGHT this step); clamp ×2: ep2 s11→12, ep7 s13→14 (wind already −1 → water keeps down-left, no extra turn) | invisible at own target | via window (turn lands next `noop`) | clamp = "a 2nd `left` does NOT increase leftward drift"; cause = "`left` makes NO same-step change" |
| **D8** `right` rule | **5** `right` targets — cause ×2: ep1 s8→9, ep6 s9→10 (STRAIGHT this step); clamp ×2: ep1 s11→12, ep6 s13→14 (wind +1 → keeps down-right); **reversal ×1: ep5 s6→7 — `right` pressed while wind=−1 ⇒ water moves LEFT** | invisible at own target | via window | **anti-shortcut**: `right`→water-LEFT defeats "`right` ⇒ water moves right"; clamp defeats "each `right` adds drift" |
| **D9** `up` = no-op | **4**: ep3 s3→4,6→7; ep4 s9→10,12→13 | aliased w/ noop | YES — passive fall/spawn only | vs `noop` on identical falls ⇒ `up` adds nothing |
| **D10** `down` = no-op | **4**: ep3 s4→5,7→8; ep4 s10→11,13→14 | aliased w/ noop | YES — passive only | same |
| **D11** `click` = no-op | **4** at 4 DISTINCT locations: (11,3),(2,9),(15,15),(5,0) | aliased; location unrecoverable | YES — passive only, NOTHING at the click cell | 4 spread locations w/ identical null effect ⇒ location irrelevant (the `keep_action_params` point) |
| **D12** `noop` | **27** noop targets (every episode) | aliased | YES — carries all passive dynamics | — |

Every core dynamic is a scored target **≥4×** (D9/D10/D11 exactly 4; all others 5–39).

## 4. Action histogram of the pool (keep_action_params=TRUE)

`noop 27 · right 5 · left 5 · up 4 · down 4 · click 4` (locations 11 3, 2 9, 15 15, 5 0) = **49**.

**Why noop is the plurality (55%):** every passive dynamic — drift (D2/D3), spawn (D4),
despawn (D6) — becomes *visible* only on the tick AFTER the causing action, and because
`left`/`right`'s effect is engine-delayed while `up/down/click` are no-ops, that carrier
tick is almost always a `noop`. Loading the pool with more verbs would only pile up
ID-unsolvable items (Section 2) without adding FD signal. The verbs are held at 4–5 each —
the maximum that still (a) triggers/carries the wind changes whose effect the window must
read and (b) supplies the D9/D10/D11 no-op contrasts — and clicks use 4 spread, train-
distinct locations. This is the click-game analog of eahcw's click-heavy pool, inverted:
here the *observable* verb is `noop` (it carries the physics) and the "arrows" (left/right)
are the invisible pen-state.

## 5. How TEST50 differs from `train/` (cross-trajectory generalization)

| | train (20 targets, 8 eps) | test50 (49 targets, 8 eps) |
|---|---|---|
| Source | 8 slices of ONE clean_data2 trajectory (orig. steps 2–81) | 8 independently DRIVEN trajectories (seed 0) |
| Cohorts on screen | mostly a SINGLE early cohort (steps 2–7 etc.) | **2–3 overlapping cohorts** (all wind/updown slices sliced from steps 7–16) |
| Wind press phase | `right`@3, `left`@15/79, `right`@23 | `right`@8,9,11,13 & `left`@8,9,11,13 (+ reversal @3/6) — **later, multi-cohort, never the s3 first-cohort config** |
| Clamp positive | none (implied only) | explicit ×4: `right`@wind+1 & `left`@wind−1 with NO extra turn (ep1/2/6/7 s11→12 & s13→14) |
| Clean side despawn | 1 (masked-heavy) | **4** clean −1/−2 side despawns + 2 masked |
| up/down/click | 1 each, isolated single-cohort slices | 4 each, interleaved in dense multi-cohort rain (ep3/ep4) |
| Click positions | (8,8),(4,4) | (11,3),(2,9),(15,15),(5,0) — all distinct from train |
| Anti-shortcut | `right`@wind−1 (ep2) | `right`@wind−1 **plus a co-occurring despawn** (ep5, different water config) |

Same rules, different situations — no action sequence, water configuration, or wind phase
replicates a train slice.

## 6. Contrastive summary (what each shortcut fails on)

- **"`left`/`right` turns water THIS step"** → 5 delayed "cause" targets show STRAIGHT fall
  at the press (ep1/ep6 right, ep2/ep5/ep7 left).
- **"`right` ⇒ water moves right"** → ep5 s6→7 `right` makes water move **LEFT** (wind was −1).
- **"a direction flip ⇒ a `left`/`right` was pressed"** → every flip lands on a **`noop`**
  (ep1/ep6 first down-right, ep2/ep5/ep7 first down-left, ep5 s7→8 straighten).
- **"each `left`/`right` adds drift"** → 4 clamp targets: a 2nd press at saturated wind adds
  nothing.
- **"count-up ⇒ spawn only / count-down ⇒ despawn only"** → 2 masked spawn+despawn targets
  (net +2 with cells simultaneously leaving the edge).
- **"every step / every noop spawns"** and **`step%5` clock** → 39 no-spawn targets + presses
  that shift the phase-relative index across episodes.
- **click location matters** → 4 spread locations, identical null effect.

## 7. Uncoverable / residual gaps (deliberate, intrinsic to DGG2C)

- **Total ID-aliasing (Section 2).** No action is ID-recoverable from one `(X_t, X_t+1)`
  pair — up/down/click/noop never move a cell and left/right's wind write is invisible
  same-step. Unfixable by curation; the set is built so the *window* (`ctx_prev`) carries the
  causing left/right and the FD objective carries the physics. Under FD every target is
  correctly predictable from the window.
- **Bottom-edge despawn is phase-locked to a spawn.** Water spawned at row 2 reaches row 17
  exactly 15 (=3×5) steps later, always on a spawn step, so a *clean* bottom despawn is
  impossible under gravity. D6 is therefore carried by 4 clean **side** despawns (wind pushes
  water off the L/R edge on a non-spawn step) plus 2 masked bottom-style pairs.
- **Spawn (D4) is a pure clock** with no observable trigger, so its only contrastive defense
  is D5 (not every step spawns) + irregular press phasing; there is no deeper rule to protect
  against a `step%5` shortcut because the clock IS the rule. Documented, not data-fixable.
- **`up` on a non-spawn quiet frame vs a spawn frame** are both present, but since up/down/
  click are pure no-ops their FD content is identical to `noop`; they exist to prove
  "verb ≠ noop changes nothing", not to add new physics.
