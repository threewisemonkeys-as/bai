# dgg2c — clean_data3 coverage

Whitelist: `left,right,up,down,noop,click` ; `keep_action_params=TRUE` (click LOCATION is part
of the label: `click 8 8` and `click 4 4` are distinct labels).

dgg2c is a **rain** game. A static gray cloud sits on rows y=0,1. Lightblue water spawns at
row y=2 (cols 2,6,10,14) every 5 steps and falls under gravity every step. The ONLY player
influence is a hidden integer `wind ∈ [-1,1]` set by `left`(−1)/`right`(+1); `up/down/click/
noop` are no-ops. Water motion per step: wind 0 → straight down; wind +1 → down-right; wind
−1 → down-left. Off-grid water despawns. No win/collision/color rules.

## CORE dynamics extracted from dynamics.txt

| # | dynamic |
|---|---------|
| D1 | Gravity straight fall (wind 0): every water cell moves down 1 row, col unchanged |
| D2 | Wind-right drift (wind +1): water moves down AND right (diagonal) |
| D3 | Wind-left drift (wind −1): water moves down AND left (diagonal) |
| D4 | Spawn: when `time%5==2`, 4 water cells appear at row 2, cols 2,6,10,14 |
| D5 | No-spawn step (negative of D4): nothing new appears on the other 4/5 steps |
| D6 | Off-grid despawn: water leaving the 17×17 bounds is removed (count drops) |
| D7 | `left` → wind = max(wind−1,−1)  (incl. clamp at −1) |
| D8 | `right` → wind = min(wind+1,+1) (incl. clamp at +1) |
| D9 | `up` = no-op |
| D10 | `down` = no-op |
| D11 | `click` = no-op, LOCATION irrelevant |
| D12 | `noop` = no effect (passive clock still runs) |

## The ID/FD gap (nrdf6-style finding)

**Engine semantics:** Autumn applies all updates simultaneously from the start-of-step state.
The clock reads `wind` BEFORE the action handler's write lands, so a `left`/`right` pressed at
step *t* only changes water motion at step *t+1*. Verified on raw grids:

```
step 3 right -> 3->4 water falls STRAIGHT (wind still 0)
               4->5 (noop) water goes DOWN-RIGHT  <- the right's effect, on a NOOP transition
step 23 right (wind was -1) -> 23->24 water STILL goes DOWN-LEFT; 24->25 (noop) straightens
```

Consequences for the scored target `(X_t -> X_t+1, a_t)`:

- **ID is fundamentally weak/adversarial here.** No action has a *same-step* visible effect.
  At a `left`/`right` target the water just continues its prior direction (the wind write is
  invisible). The directional CHANGE always lands on the *next* transition, which is labelled
  `noop`. So the only visible "action-like" event (a direction flip) is attributed to a no-op,
  and `up/down/click/noop` are mutually indistinguishable always. This is the dgg2c analog of
  the nrdf6 failure (real dynamic present only as window context / on a noop).
- **FD is informative on essentially every transition** (water moves, spawns, or despawns
  every step), but predicting the *direction* requires knowing `wind`, which is recoverable
  only from the window (the prior fall direction or the last left/right in `ctx_prev`).
- **Spawn (D4) is a pure clock** (`time%5==2`) with no observable trigger, so the contrastive
  defense is only "not every step spawns" (D5) — there is no deeper rule to protect against a
  `step%5` shortcut because the clock IS the rule. Documented, not fixable by data curation.

| dynamic | TARGET under ID? | TARGET under FD? | in ORIGINAL 84-row pool? | GAP |
|---|---|---|---|---|
| D1 straight fall | weak (no action signal) | yes | yes (many noop) | none |
| D2 right drift | only via window (delayed) | yes | yes | effect lands on noop, not on `right` |
| D3 left drift | only via window (delayed) | yes | yes | effect lands on noop, not on `left` |
| D4 spawn | no (passive) | yes | yes but mostly masked by simultaneous despawn after early steps | clean +4 spawns only at steps 2/7/12; later spawns net-mixed |
| D5 no-spawn | n/a | yes (negative) | yes | none |
| D6 despawn | no (passive) | yes | yes | often co-occurs with spawn → masked |
| D7 left rule | invisible at own target | only via window | yes but sparse (8 left/right of 84) | delayed + clamp ambiguity |
| D8 right rule | invisible at own target | only via window | yes, sparse | delayed + clamp ambiguity |
| D9 up no-op | no (= other no-ops) | yes (passive) | 2 ups, BOTH on spawn steps | no clean (non-spawn) up exists |
| D10 down no-op | no | yes | yes | none |
| D11 click no-op | no | yes | 2 clicks, diff locations | none |
| D12 noop | no | yes | abundant | over-represented → balanced-20 sample could drop a unique passive-carrier noop |

The last row is the concrete curation hazard: in the original pool the passive dynamics
(spawn / drift / despawn) live almost entirely on `noop` transitions, so a balanced-20 sample
that thins noops risks dropping the only clean +4 spawn or the off-grid despawn. The curated
pool fixes this by hand-picking the clean carriers and keeping the pool at exactly 20 (≤
train-n) so balanced_split returns the whole pool unchanged.

## Curated slices (verbatim consecutive original rows) and what each target covers

Pool = **20** scored targets (verify_pool confirms; pool == train-n ⇒ all kept).
`(1.0,0.0)`=straight, `(1.0,1.0)`=down-right, `(1.0,-1.0)`=down-left, `+4/−1`=spawn/despawn.

- **episode_0 = steps [2,3,4,5,6,7]** — spawn → fall → wind-right drift → off-grid despawn:
  - `2->3`  noop  → **+4 cells** at row2 cols 2,6,10,14 — **D4 spawn POSITIVE** (clean, no despawn).
  - `3->4`  right → straight `(1.0,0.0)` — **D8 cause**; *contrastive*: `right` produces NO
    same-step motion change (wind write delayed). **D1**.
  - `4->5`  noop  → down-right `(1.0,1.0)` — **D2**; *contrastive*: the direction flip lands on a
    **noop**, not on the `right`.
  - `5->6`  noop  → down-right `(1.0,1.0)` — D2 continued (bridges to despawn).
  - `6->7`  noop  → **−1 cell** (col-16 water pushed off the right edge by wind) — **D6 despawn**.
- **episode_1 = steps [14,15,16,17]** — `left` cause→delayed effect:
  - `14->15` noop  → straight — D1 (establishes wind 0).
  - `15->16` **left** → straight `(1.0,0.0)` — **D7 cause**; *contrastive*: `left` makes no
    same-step change.
  - `16->17` noop  → down-left `(1.0,-1.0)` — **D3**; delayed effect of the `left`, again on a noop.
- **episode_2 = steps [23,24,25]** — strongest anti-shortcut pair:
  - `23->24` **right** → **down-LEFT** `(1.0,-1.0)` — `right` pressed while wind was −1; water
    keeps moving LEFT this step. Defeats "right ⇒ water moves right". **D8**.
  - `24->25` noop  → straight `(1.0,0.0)` — the `right` resolved wind −1→0; flip lands on noop. D1.
- **episode_3 = steps [27,28,29]** — `up` is a no-op:
  - `27->28` **up** → +2 net (spawn step, water falls straight) — **D9**: `up` changes nothing;
    passive spawn+fall happen regardless.
  - `28->29` noop  → straight — D1.
- **episode_4 = steps [31,32,33]** — `down` is a no-op:
  - `31->32` **down** → straight `(1.0,0.0)` — **D10**: clean (non-spawn) no-op.
  - `32->33` noop  → +1 net (spawn step) — D4 spawn at a different clock phase.
- **episode_5 = steps [35,36,37]** — `click 8 8` no-op:
  - `35->36` **click 8 8** → straight — **D11**: click does nothing; nothing appears at (8,8).
  - `36->37` noop  → straight — D1.
- **episode_6 = steps [71,72,73]** — `click 4 4` no-op (DIFFERENT location):
  - `71->72` **click 4 4** → straight — **D11 contrastive**: a click at a different location has
    the identical (null) effect ⇒ location is irrelevant (the `keep_action_params` point).
  - `72->73` noop  → +1 net (spawn) — D4.
- **episode_7 = steps [79,80,81]** — second `left` cause→effect (clean, 12 cells):
  - `79->80` **left** → straight `(1.0,0.0)` — D7 cause.
  - `80->81` noop  → down-left `(1.0,-1.0)` — D3 delayed effect on noop.

### Contrastive summary
- "Action moves water this step" is defeated by `right`@3->4 (→straight), `left`@15->16
  (→straight), and especially `right`@23->24 (→**leftward**).
- "A direction flip means a `left`/`right` was pressed" is defeated by the flips that occur on
  **noop** transitions (4->5 down-right, 16->17 / 80->81 down-left, 24->25 straighten).
- "Every step spawns" is defeated by the many straight-fall noops (D5). Pure `time%5` clock is
  intrinsic (no observable trigger) and cannot be further defended by data.
- D4 spawn POSITIVE (clean +4 at 2->3) vs no-spawn negatives; D6 despawn (6->7) vs no-despawn.
- `click 8 8` vs `click 4 4`: two locations, identical null effect ⇒ location irrelevant.

### Known residual gaps (honest)
- ID has almost no signal for ANY action in dgg2c (delayed wind + four indistinguishable
  no-ops); the value of this set is concentrated on FD (water motion + spawn/despawn) and on
  documenting that the action effect is window-only.
- No clean (non-spawn) `up` transition exists in the source trajectory; the one used coincides
  with a spawn step.
- An explicit clamp positive (`left` at wind=−1 / `right` at wind=+1 with NO change) is not
  given its own slice — every such row in the source is masked by simultaneous despawn — but
  the clamp's "no visible same-step effect" is already implied by every left/right target here.
