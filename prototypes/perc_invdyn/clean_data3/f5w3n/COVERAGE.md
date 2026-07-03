# f5w3n — clean_data3 coverage

Space-Invaders-style shooter on a 16x16 grid. Coords below are grid `(row, col) = (y, x)`
(y increases downward; hero starts at `(15,8)`). Config: whitelist `left,right,up,noop`
(no `down`/`click` — both are stated no-ops in this game), `keep_action_params=FALSE`
(movement game; verb-only labels).

## Color legend (what the diff tags mean here)
- `gray` = hero; `gray~move(0,-1)` = moved LEFT, `gray~move(0,1)` = moved RIGHT.
  `gray-1` = hero occluded/removed (a red bullet drawn on the hero cell on the fire step,
  OR hero killed -> turns black/invisible).
- `red` = hero bullet; `red+1` spawn on `up`, `red~move(-1,0)` climbs up every step,
  `red-1` despawn (hit enemy / hit enemy-bullet / flew off the top edge).
- `orange` = enemy bullet; `orange+1` spawn on the timed enemy-fire rule, `orange~move(1,0)`
  descends every step, `orange~move(-14,..)` = old one left bottom + new one spawned at top.
- `blue` = enemies; `blue~recolor` / `blue±1` = formation march (oscillates ±1 col, edge
  enemies briefly leave the grid -> count flickers), permanent `blue-1` = enemy destroyed.

## CORE dynamics extracted from dynamics.txt
1. **left** — hero moves one cell left (x-1), blocked at the grid edge.
2. **right** — hero moves one cell right (x+1), blocked at the grid edge.
3. **up = fire** — spawns a red bullet at the hero cell, **only if the hero is alive**.
4. **noop / clock** — hero unchanged; the passive clock still advances.
5. **Hero bullet travels up** — every step all red bullets move y-1 (passive).
6. **Enemy bullet travels down** — every step all orange bullets move y+1 (passive).
7. **Enemy formation march** — time-keyed (time%10==0 and ==5): the two enemy rows shift
   ±1 col in opposite directions, oscillating around their start columns (fires 2 of every
   10 steps; static the other 8).
8. **Enemy firing** — time%15==3: one new orange bullet spawns at a remaining enemy's cell.
9. **Hero bullet ∩ enemy** — both removed (enemy destroyed by being shot).
10. **Hero ∩ enemy bullet** — hero `alive`->false (gray->black/invisible); the orange bullet
    is NOT consumed.
11. **Hero bullet ∩ enemy bullet** — both removed (shots cancel).
12. **Bullet off-grid** — a bullet that leaves the 16x16 grid simply stops rendering (despawn).
13. **No win/loss/reward** — reward 0 every step; the only terminal-ish event is hero death,
    after which firing is disabled but movement still works.

## Coverage of the ORIGINAL train pool, and the GAP

The original train episode is one 106-row trajectory. A balanced-20 sample would be
dominated by `noop` (≈80 of the whitelisted rows) and the few movement/fire rows, but the
**interesting dynamics are heavily entangled** — every step simultaneously moves both bullet
types and (every 5 steps) marches the formation, so most noops look identical and the
time-keyed/conditional rules are present mostly as undifferentiated context. Specific gaps:

| Dynamic | TARGET under ID? | TARGET under FD? | Gap in original pool |
|---|---|---|---|
| 1 left | yes (gray col-1) | yes | fine, but early lefts are confounded by a climbing red bullet; clean lefts (no red) exist 43-62 |
| 2 right | yes (gray col+1) | yes | **only 2 right rows exist** (19,23) — easy to drop in a random balanced sample |
| 3 up=fire | yes (red+1) | yes | many ups, but most overlap a 2nd bullet/march; need a clean-from-rest one |
| 4 noop/clock | yes (no gray move, no red+1) | yes | abundant |
| 5 hero-bullet up | no (fires on every action) | yes | only meaningful right after a fire; pure-climb noop must be picked deliberately |
| 6 enemy-bullet down | no (every action) | yes | ubiquitous, but never isolated from other motion |
| 7 enemy march | no (passive, time-keyed) | yes | **nrdf6 risk**: fires only 2/10 steps on noop; a `step%k` clock can fake it unless paired with static near-misses |
| 8 enemy firing | no (passive, time%15==3) | yes | **nrdf6 risk**: 1 spawn / 15 noops; same step-counter shortcut risk; needs no-spawn near-misses |
| 9 hero-bullet ∩ enemy | partial | partial | **MAJOR GAP**: the only kill in the whole trajectory (col-8 enemy, steps 15-17) is entangled with a simultaneous `up` fire + a march occlusion + a 1-step prev-delay (enemy vanishes at 15->16, bullet despawns at 16->17). No clean isolated kill exists. |
| 10 hero death | no (passive collision on noop) | yes | one event only (92->93); easy to never sample as the scored pair |
| 11 bullet ∩ bullet | no | partial | only occurrence (≈70->72) is smeared across a prev-delay and overlaps a `left`; not cleanly isolable |
| 12 off-grid despawn | no | yes | present (32->33) but as a mid-trajectory noop unlikely to be the scored pair |
| 13 no reward/terminal | n/a | n/a | nothing to score |

The headline gap is the **nrdf6 pattern**: the conditional/time-keyed dynamics (march #7,
enemy-fire #8, death #10) appear only as rare events buried among visually-identical noops,
so a balanced-20 sample both (a) under-samples them as scored targets and (b) lets a
step-parity clock explain them. The curated pool fixes this by making each a scored target
**with a same-surface near-miss negative**.

## Curated slices (11 episodes -> 20 scored target pairs)

Each episode is a verbatim contiguous slice of the original train trajectory; every internal
consecutive pair is a scored target. Windows are real consecutive frames and never cross slices.

| ep | orig steps | scored target(s) | dynamic(s) covered |
|----|-----------|------------------|---------------------|
| 0 | 3,4,5 | 3->4 **up** (red+1 @hero (15,8)); 4->5 **noop** | #3 fire-spawn (ID+FD); #5 hero-bullet-up + #6 enemy-bullet-down + hero un-occludes (FD) |
| 1 | 7,8 | 7->8 **left** (15,8)->(15,7) | #1 left (ID+FD) |
| 2 | 19,20 | 19->20 **right** (15,6)->(15,7) | #2 right (ID+FD) |
| 3 | 23,24 | 23->24 **right** (15,7)->(15,8) | #2 right (2nd instance — protects the rare right class) |
| 4 | 15,16,17 | 15->16 **up** (red+1 @(15,6) + enemy at (3,8) destroyed: blue row3 loses col8); 16->17 **noop** (that red bullet despawns) | #3 fire + #9 hero-bullet∩enemy kill (FD), incl. the 1-step prev-delay despawn |
| 5 | 32,33 | 32->33 **noop** (red (0,6) flies off the top -> despawn; red (11,8) keeps climbing) | #12 off-grid despawn (FD) + #5 |
| 6 | 47,48,49 | 47->48 **left** (clean, no red bullet); 48->49 **noop** (hero static, orange descends) | #1 clean left (ID); movement near-miss (hero static on noop) + #6 clean enemy-bullet-down |
| 7 | 59,60,61,62 | 59->60 **left**; 60->61 **noop** (formation shifts, `blue~recolor`); 61->62 **noop** (formation identical) | #1 left; **#7 march POSITIVE (60->61)** vs **march NEGATIVE near-miss (61->62)** — defeats `step%k` clock |
| 8 | 77,78,79,80 | 77->78 **noop** (no orange, NO_CHANGE); 78->79 **noop** (orange spawns @(1,1)); 79->80 **noop** (orange descends, no spawn) | **#8 enemy-fire POSITIVE (78->79)** flanked by two no-spawn near-misses (77->78, 79->80) — defeats step-counter shortcut |
| 9 | 91,92,93 | 91->92 **noop** (orange descends to (14,1) toward hero); 92->93 **noop** (orange reaches (15,1), **hero dies**: gray->none) | #10 hero death (FD) with the causal descent in-window; #6 |
| 10 | 97,98,99 | 97->98 **up while DEAD** (no red+1 — fire suppressed); 98->99 **noop** | **#3 alive-guard contrast** (up fires only if alive) — near-miss vs the live ups in ep0/ep4; #6 |

## Pool composition (verified via `T.verify_pool`)

- **20 scored target transitions** (pool == train-n=20, so balanced_split keeps all of them).
- **By action:** `up:3, left:3, right:2, noop:12`.
  - 3 `up`: 2 live fires (red+1) + 1 dead (no red+1) — the alive-guard contrast.
  - 3 `left`, 2 `right` (both rights protected from sampling loss).
  - 12 `noop`, each a distinct passive/collision dynamic — NOT filler: bullet-climb,
    kill-aftermath despawn, off-grid despawn, march+, march−, enemy-fire+, two enemy-fire
    no-spawn near-misses, death-approach, death, dead-passive, clean enemy-down.

### Contrastive negatives built in (defeat shortcuts)
- **march:** 60->61 (formation shifts) vs 61->62 (identical) — same `noop`, same parity look.
- **enemy-fire:** 78->79 (spawn) vs 77->78 (NO_CHANGE) and 79->80 (descend only) — kills the
  "noop spawns orange every k steps" clock.
- **fire / alive-guard:** live up (3->4, 15->16, red+1) vs dead up (97->98, no red+1).
- **movement:** every `noop` is a hero-static near-miss for left/right; clean lefts (47->48,
  59->60) with no red bullet present prevent ID leaning on bullet motion.

### Known residual gap (documented, not fixable from this trajectory)
- #9 hero-bullet∩enemy kill is only available **entangled** (col-8, steps 15-17) with a
  simultaneous fire, a march occlusion, and a 1-step prev-delay. #11 bullet∩bullet cancel is
  similarly smeared across a prev-delay and overlaps a `left`, so it is carried only as window
  context, not as an isolated scored target. No clean isolated instance of either exists in
  the original train episode.

---
## REGENERATED trajectory (clean isolated kill + clean isolated bullet-cancel)

The residual gap above is **not fixable by re-slicing** — the original clean_data2 rollout
never produced a frame where a hero-bullet kills an enemy in isolation, nor where one hero
bullet meets exactly one enemy bullet on a quiet frame. Both events only ever occur entangled
(simultaneous fire + formation march + occlusion + 1-step prev-delay). So the trajectory was
**regenerated** by driving the real `F5W3N.sexp` with `autumn_drive.py` (seed 0), staging the
two collisions deliberately. Saved at `train_regen/episode_0/trajectory.csv`
(53 frames, filmstrip `train_regen/viz.html`); `train/` is now sliced from `train_regen`.
(`dynamics.txt` unchanged; `test/` is still the original clean_data2 copy and therefore still
lacks these regenerated states — regenerate it too for full train/test parity.)

### Timing learned from the SEXP (used to pick uncluttered frames)
- `time` starts 0, +1/step. The `on` rules evaluate on the **current** frame's `time`, then
  produce the next frame. Hero/enemy **bullets move 1 cell/step** (red up, orange down).
- **March:** enemies1 RIGHT & enemies2 LEFT on the transition out of a frame with `time%10==0`;
  enemies1 LEFT & enemies2 RIGHT out of `time%10==5`. So marches sit at transitions
  `0->1, 5->6, 10->11, 15->16, ...`; the formation is **static the other 8/10 steps**.
- **Enemy fire:** at `time%15==3` (frames 3,18,33,48,…) one orange spawns at a random remaining
  enemy's current cell (seed 0 ⇒ deterministic: 1st=col2, 2nd=col1, 3rd=col2 …, independent of
  the hero's moves, which consume no RNG).
- **Collisions are 1-step prev-delayed:** the bullet and its victim **co-locate on frame N**
  (the victim is occluded/rendered as the bullet's color), and the removal fires on
  **frame N→N+1** (`(prev …)` intersection). Cause and effect are kept in the **same slice**.
- **Structural note:** a death always lands on row15 (hero's row) exactly 14 frames after the
  killing orange spawned at row1, i.e. at `(fire+15)`, which is itself the next fire frame — so
  the death-flip unavoidably co-occurs with a fresh enemy fire at the **top** of the grid
  (row1), spatially far from the death at the bottom. The death cell stays clean.

### Action sequence (52 actions, all in whitelist `left,right,up,noop`)
```
left×6, noop, up, noop×6, up, noop×19, right, left, noop×12, noop, up, left, noop
```
Rationale, frame by frame (frame = state after that action):
- `left×6` (f0–5): hero (8,15)→(2,15). Demonstrates **left**; the t=3 enemy fire (1st draw,
  **col2**) happens here and that orange begins descending col2 — the cancel target.
- `noop, up` (f6–7): fire a red bullet up **col2** (hero alive) — the **cancel** shot.
- `noop×6` (f8–13): red climbs col2, the first orange descends col2; they **co-locate at f13
  (row10) and mutually cancel at f14** — a CLEAN ISOLATED bullet∩bullet cancel (no enemies,
  no march at that cell).
- `up` (f14): fire a second red up the now-relevant col2 — the **kill** shot.
- `noop×19` (f15–33): red climbs col2 and **kills the enemies2 enemy at row3 col2** —
  co-locate f27, both removed f28 — a CLEAN ISOLATED kill (the only other bullet, the t=18
  orange, is 6 rows away in col1). Meanwhile that t=18 orange descends col1 and **flies off the
  bottom edge at f33→34 (off-grid despawn)**, while the hero sits one column over (no death).
- `right, left` (f34–35): hero (2,15)→(3,15)→(2,15) — demonstrates the (rare) **right** move
  and a second **left**, dodging then re-entering col2 under the t=33 orange.
- `noop×13` (f36–48): the t=33 orange descends col2 onto the hero (row15) — **hero death at
  f48 (alive→false, gray→black/invisible)**.
- `up` (f49, dead): **fires nothing** (no `red+1`) — the alive-guard contrast vs the live fires.
- `left, noop` (f50–51): a dead hero still moves; trailing passive frames.

### Curated slices from train_regen (9 episodes → 20 scored targets)
| ep | regen steps | scored target(s) | dynamic(s) |
|----|-------------|------------------|------------|
| 0 | 1,2 | 1→2 **left** (clean, no bullet) | #1 left |
| 1 | 7,8,9 | 7→8 **up FIRE** (`red+1`, alive); 8→9 noop (`red↑`) | #3 fire-spawn, #5 |
| 2 | 11,12,13,14 | 11→12 approach; 12→13 **CANCEL co-locate** (`orange-1`); 13→14 **CANCEL removal** (`red-1`) | **#11 bullet∩bullet (GAP, isolated)**, #5, #6 |
| 3 | 25,26,27,28 | 25→26 **MARCH+** (`blue~recolor`); 26→27 **KILL co-locate** (`blue-1`); 27→28 **KILL removal** (`red-1`, enemy stays gone) | **#9 hero-bullet∩enemy (GAP, isolated)**, #7 march+, #5, #6 |
| 4 | 17,18,19,20 | 17→18 no-spawn; 18→19 **enemy-FIRE+** (`orange+1`); 19→20 no-spawn | **#8 enemy-fire+** with step-counter near-misses, #5 |
| 5 | 33,34,35,36 | 33→34 **OFF-GRID despawn** (`orange~move(-14,*)`); 34→35 **RIGHT**; 35→36 **LEFT** | #12 off-grid, #2 right, #1 left |
| 6 | 47,48,49 | 47→48 death-approach (orange reaches hero, `gray-1`); 48→49 **DEATH** (hero gone) | #10 hero death |
| 7 | 49,50,51 | 49→50 **up while DEAD** (no `red+1`); 50→51 dead move | **#3 alive-guard contrast** |
| 8 | 36,37 | 36→37 noop (clean `orange↓`, no march/red) | #6 clean enemy-down, #7 march− near-miss |

### Regenerated pool composition (`T.verify_pool`, train-n=20 keeps all)
- **20 scored target transitions**; by action: `noop:14, left:3, up:2, right:1`.
- The two motivating gaps are now **clean isolated scored targets**: the bullet∩bullet cancel
  (ep2, col2 row10, no enemies/march at the cell) and the hero-bullet∩enemy kill (ep3, col2
  row3, the only other bullet 6 rows away in col1), each with cause (co-locate) and effect
  (removal) in the same slice across the 1-step prev-delay.
- Contrastive negatives retained: live up `red+1` (ep1) vs dead up no-`red+1` (ep7);
  enemy-fire+ (ep4 18→19) flanked by no-spawn noops (defeats `step%k`); march+ (`blue~recolor`,
  ep3 25→26) vs march− no-blue noops (ep4, ep8); every hero-static noop is a move near-miss.
- The 14 noops are not filler — each carries a distinct passive/collision dynamic (bullet
  climb/descent, cancel, kill, off-grid despawn, march+, enemy-fire+, death-approach, death).
