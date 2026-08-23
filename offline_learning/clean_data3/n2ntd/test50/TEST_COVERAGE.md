# n2ntd — TEST50 held-out pool coverage

Game: **N2NTD**, Mario-style platformer, 12x12 grid, white background.
Config: whitelist = `left,right,up,down,noop,click`; **keep_action_params = FALSE**
(movement game — `click R C` collapses to the verb `click`; the click location is ignored
by the game itself: a fired bullet always spawns at Mario's own cell).

Scored pool (what `verify_pool(test50,'left,right,up,down,noop,click',context_k=9)` reports):
**52 scored target transitions** across **15 episodes** (curated contiguous slices of freshly
driven trajectories, seed 0). Pool size 52 ∈ 50±2. `balanced_split` returns the whole pool
when pool ≤ `--test-n 50`, so the pool IS the test set.

Sources: 6 fresh `autumn_drive.py N2NTD` runs (C, A, B, E, F, G) — action strings in §6.
Every episode is verbatim rows from one drive. `train/`, `test/`, `dynamics.txt`,
`COVERAGE.md` untouched (test50/ is additive).

## 1. Core dynamics (numbering from ../COVERAGE.md)

D1 left · D2 right · D3 up=jump(-4, only when supported) · D4 click=fire(needs ammo>0;
bullet spawns AT Mario, ammo-1, Mario nudged down 1) · D5 down=no-op · D6 noop=no-op ·
D7 gravity(falls 1/frame when unsupported) · D8 enemy patrol(1 col/frame, flips at
edges x=1/x=10) · D9 bullet rises 1/frame, FREEZES permanently below a platform ·
D10 coin pickup(+1 ammo) · D11 bullet kills enemy on intersection.

**Engine facts confirmed by driving (read from the frames, not the prose):**
- Mario **rests on platforms**: `moveDownNoCollision` is blocked by Step cells, so he
  lands ON row-8/row-6/row-10 platforms and can JUMP from them (dynamics.txt's D7 note
  "falls through steps" is wrong on this point; the original train data itself shows the
  rest at its steps 4→5). Jump condition = "cannot fall" = floor OR platform top.
- Horizontal moves override gravity for that frame (a midair left/right keeps the row).
- Via platform-hopping ALL THREE coins are collectable (train only ever took the floor
  coin (9,1)); a drive can hold ammo 2+ (drive G fires twice).
- Fire renders the bullet OVER Mario when both share a cell (red vanishes for one frame).
- Pickup fires one frame AFTER Mario enters the coin cell and overrides any movement
  action taken that frame; the nudge-down is blocked by platforms.
- Enemy patrol under seed 0: origin x=1 at t≡4 (mod 18), x=10 at t≡13 (mod 18).

## 2. Coverage map — scored TARGET pairs per dynamic

Notation: `epN a→b` = pair in `episode_N`, source-drive local steps a→b (the `Step` column).
[NEG] = contrastive negative. Change tags verified with `classify` on the actual frames.

| Dyn | Scored TARGET pairs | ID-informative? | FD-informative? |
|---|---|---|---|
| **D1 left** | ep0 4→5 ((11,2)→(11,1), enemy alive, +left-edge flip) · ep0 5→6 (→(11,0)) · ep10 24→25 (platform walk (9,1)→(9,0)) — **[NEG]** ep0 6→7 (blocked at col 0, red static) · ep10 25→26 (blocked at (9,0), full NO_CHANGE) | 3 yes (red col-1); blocked ones aliased (§4) | yes |
| **D2 right** | ep6 19→20,20→21,21→22 (floor (11,8)→(11,11)) · ep11 5→6 (midair (5,7)→(5,8), lands on row-6 platform, no fall) — **[NEG]** ep4 16→17 (blocked at col 11) | 4 yes (red col+1); blocked aliased | yes |
| **D3 up** | ep2 25→26 (floor jump col 7, row 11→7) · ep7 23→24 (floor jump col 11) · ep9 9→10 (**platform** jump (9,1)→(5,1) off the row-10 platform) — **[NEG]** ep4 12→13 (midair up at (7,11) = plain fall) | 3 yes (red row-4); midair-up aliased with noop-fall | yes (conditional on support) |
| **D4 click/fire** | ep2 26→27 (midair fire at (7,7): purple+1 at Mario's cell, red nudged to (8,7)) · ep7 24→25 (midair fire (7,11)) · ep5 19→20 (floor fire (11,9): purple+1, red hidden) · ep12 17→18 (floor fire (11,6)) — **[NEG]** ep14 34→35 (ammo-0 click at (11,11), NO_CHANGE) | fires yes (purple appears at Mario + red nudge/occlusion); ammo0 aliased | yes (needs hidden ammo state; pickups in-window for ep5/ep12 drives) |
| **D5 down** | ep4 14→15 (midair: falls, = gravity) · ep10 23→24, 26→27 (on platform, enemy dead: full NO_CHANGE) · ep13 30→31 (on floor, enemy alive + 2 frozen bullets: only enemy moves) | never (no handler; aliased with noop) | yes (predict "nothing player-side happens") |
| **D6 noop** | 29 pairs (carrier of all passive dynamics below) | as "passive" | yes |
| **D7 gravity** | falls: ep4 13→14 ((8,11)→(9,11)) · ep4 15→16 (lands floor) · ep9 7→8 (falls into coin cell) · ep11 3→4 (same, other coin) · ep2/ep7 27→28/25→26 (Mario falls while bullet rises) · ep10 20→21 (falls AND lands on row-10 platform) — **[NEG]** no-fall when supported: ep13 29→30,31→32 (floor) · ep1 38→39, ep14 32→33,33→34 (floor, fully static) | no (passive; aliased with down/midair-up) | yes (fall iff unsupported — platform landings make the support rule visible) |
| **D8 enemy patrol** | motion visible in EVERY enemy-alive pair (~35 pairs); direction FLIPS as targets: ep0 4→5 (left edge, under `left`) · ep11 4→5 (left edge, under pickup-noop) · ep3 31→32 (right edge, under rise-noop) · ep4 13→14 (right edge, under fall-noop) · ep13 31→32 (right edge, frozen-bullet decoys) | no (passive) | yes (flip only at x=1/x=10 — 5 flip targets at both edges defeat "constant drift" and step-clock rules) |
| **D9 bullet rise** | ep5 20→21,21→22 (col 9) · ep12 18→19,19→20 (col 6) · ep1 36→37 (col 3) · ep3 31→32,32→33 (col 7) · ep8 29→30,30→31 (col 11) · ep10 20→21,21→22 (col 1) · ep2/ep7 (rise+fall duals) — ~13 | no (passive) | yes (purple row-1/frame) |
| **D9 FREEZE** | ep12 20→21 (bullet reaches (9,6), does NOT move — one below the row-8 platform) — persistence **[NEG]**: ep13 29→30, 30→31 (down), 31→32 (both frozen bullets (9,6),(7,8) static forever) | no (passive) | yes (rise stops iff platform above; defeats "bullet always rises") |
| **D10 pickup** | entries: ep9 7→8 (falls into (9,1): red vanishes under gold) · ep11 3→4 (falls into (4,7)) — collections: ep9 8→9 (gold-1, red+1 at (9,1); nudge BLOCKED by row-10 platform) · ep11 4→5 (gold-1, red reappears at (5,7); nudge-down visible) | entries/collections happen on noop (passive) | yes (gold vanishes exactly where Mario is; ammo+1 explains later fire) |
| **D11 kill** | ep1 37→38 (col 3, enemy c3-5 moving LEFT) · ep3 33→34 (col 7, enemy c7-9 moving LEFT) · ep8 31→32 (col 11, enemy AT right patrol edge x=10) · ep10 22→23 (col 1, enemy AT left patrol edge x=1) — all `blue-5 mediumpurple-1` (6 enemy cells + bullet vanish; intersect frame in-slice) | no (passive) | yes (enemy+bullet vanish iff intersecting) |

### Contrastive negatives (14 / 52 = 27%, req. 20-30%)

| Negative | Pairs | Shortcut it defeats |
|---|---|---|
| blocked moves | ep0 6→7 (left@col0) · ep10 25→26 (left@(9,0)) · ep4 16→17 (right@col11) | "left/right always move Mario" |
| midair up | ep4 12→13 (falls instead of jumping) | "up always jumps" (jump needs support) |
| ammo-0 click | ep14 34→35 (NO_CHANGE) | "click always spawns a bullet" |
| down no-ops | ep10 23→24, 26→27 · ep13 30→31 | "every action changes the frame" |
| static-world noops | ep1 38→39 · ep14 32→33, 33→34 | "enemy always moves" / "noop always changes X" |
| frozen-bullet persistence | ep13 29→30, 31→32 | "bullets rise every frame" (paired with 13 rise positives) |

Step-clock resistance: slice Step ranges overlap across episodes with different labels
(e.g. Step 20 → noop (ep5), right (ep6), noop (ep10/ep12); Step 25 → up (ep2), noop (ep7),
left (ep10)); enemy flips occur under left/noop at both edges; fires at drive-times
17,19,24,26 (mod 18: 17,1,6,8). No `step % k → action` rule survives.

## 3. Action histogram (pool of 52)

```
noop   29   (all passive dynamics: falls, rises, freeze+persist, 4 kills, pickups, flips, statics)
left    5   (3 moving incl. flip-pair + 2 blocked)
right   5   (3 floor + 1 midair-landing + 1 blocked)
up      4   (2 floor jumps + 1 platform jump + 1 midair NEG)
down    4   (1 midair fall + 3 no-ops in varied decoy states)
click   5   (4 fires: 2 midair + 2 floor; 1 ammo-0 NEG)
```
Movement-verb counts are as even as the game's physics allow; noop majority mirrors the
curated train (9/20) because gravity/patrol/bullets/pickups are noop-carried.

## 4. Uncoverable / aliased — minimized & documented

- **Passive-pair ID aliasing (intrinsic).** down and noop are indistinguishable everywhere;
  midair up = fall = midair noop/down; ammo-0 click on the floor = noop. An oracle that knows
  the true rules and answers "noop" on every passive pair scores **43/52 ≈ 0.83 ID ceiling**
  (it forfeits the 4 downs, 1 midair-up, 1 ammo0-click, and 3 blocked moves). All 29 noops
  are recovered by that policy, and all 14 visible player-effect pairs (moves/jumps/fires)
  are exactly recoverable. This matches the train COVERAGE's "down vs noop fundamentally
  ID-indistinguishable" note; the aliased items are kept because they are the FD-critical
  negatives (they are what defeats "every action/frame changes something").
- **Bullet-exits-top near-miss not scored.** A full miss (bullet passes rows 1-0 with the
  enemy elsewhere and despawns off-top, enemy survives) was driven (drive D, steps 23→26)
  but cut to keep the pool ≤52; "bullet does not kill" is still contrasted by the two
  frozen bullets (never reach the enemy, ep12/ep13) and by rise pairs where bullet and
  enemy coexist without intersecting (ep1 36→37, ep8 30→31).
- **Enemy-Mario contact** has no rule in the SEXP (they never interact) — nothing to test.
- **Steps are static** under seed 0 (non-dynamic, per dynamics.txt D12) — background only.
- **Kill anti-cause caveat:** killing requires ammo, so every kill drive first collects a
  coin; the pickup is many frames outside the kill slice's window for ep1/ep3/ep8 (ammo is
  hidden state there), but each kill slice contains the rising bullet, which is sufficient.

## 5. How this TEST differs from `train/`

`train/` = 9 slices of the single original rollout (1 coin taken, 1 kill, 1 freeze).
TEST50 = 6 fresh drives exercising the same rules in situations the train never visits:

| | train | test50 |
|---|---|---|
| jumps | floor col 6, col 1 | floor col 7, col 11; **platform jump** (9,1)→(5,1); midair-up NEG at col 11 |
| fires | floor (11,4) | floor (11,9), (11,6); **midair fires** (7,7), (7,11) (bullet at old cell + Mario nudged, both visible) |
| kills | col 7, enemy c6-8 moving right, floor-fired | cols 3/7/11/1; enemy moving LEFT (ep1, ep3) and AT both patrol-edge cells (ep8 x=10, ep10 x=1); 3 of 4 midair-fired |
| freeze | (9,4) under row-8 platform | **(9,6)** under row-8 platform + second frozen bullet (7,8) under row-6 platform as persistent decoys (ep13 shows BOTH) |
| pickups | (9,1) via `right` (action overridden) | (9,1) AND (4,7) via clean `noop` pairs; nudge-blocked vs nudge-visible variants; coin (5,9) also collected in drives B/G (context) |
| coins/ammo | 1 coin, ammo ≤1 | drives collect 2 coins (G fires twice — two bullets in one frame at ep12/ep13 windows) |
| movement rows | floor + row 9 platform (9,1)-(9,3) | floor cols 8-11, platform row 9 leftward + blocked at (9,0), midair right (5,7)→(5,8) with platform landing |
| enemy state | dead for most of train (steps 29+) | ALIVE in 12/15 episodes — every movement/fire pair carries patrol motion as FD background |
| statics | quiet at (11,7), coins (5,9),(9,1) | quiet at (11,11), coins (4,7),(5,9) — different residual coin set |

No scored test pair replicates a train scored pair's (state, action): the one near-collision
(post-kill ammo-0 click at (11,7)) was detected and swapped to drive E's (11,11) version.

## 6. Reproduction

Seed 0 (default). From repo root, `<S>` = a scratch dir:
```
uv run python prototypes/perc_invdyn/autumn_drive.py N2NTD <S>/driveC --actions "left,left,left,left,left,left,left,up,right,up,noop,noop,right,right,down,noop,noop,noop,noop,noop,noop,noop,noop,noop,noop,noop,click_0_0,noop,noop,noop,noop,noop,noop,noop,noop,noop,noop,noop,noop,noop"
uv run python prototypes/perc_invdyn/autumn_drive.py N2NTD <S>/driveA --actions "up,noop,up,right,noop,noop,noop,noop,noop,noop,noop,noop,noop,noop,noop,noop,noop,noop,noop,noop,noop,noop,noop,noop,noop,up,click_0_0,noop,noop,noop,noop,noop,noop,noop,noop,click_2_2,noop"
uv run python prototypes/perc_invdyn/autumn_drive.py N2NTD <S>/driveB --actions "up,up,right,right,noop,noop,right,noop,right,right,noop,noop,up,noop,down,noop,right,left,left,click_0_0,noop,noop,noop,noop,noop,noop,left,left,noop"
uv run python prototypes/perc_invdyn/autumn_drive.py N2NTD <S>/driveE --actions "click_5_5,left,left,left,left,left,up,noop,noop,noop,right,right,down,noop,right,right,right,right,right,right,right,right,noop,up,click_0_0,noop,noop,noop,noop,noop,noop,noop,noop,noop,click_7_3,noop"
uv run python prototypes/perc_invdyn/autumn_drive.py N2NTD <S>/driveF --actions "left,left,left,left,left,up,noop,noop,noop,up,noop,noop,noop,noop,noop,noop,up,click_0_0,noop,noop,noop,noop,noop,down,left,left,down,right,right,right,noop,noop,noop"
uv run python prototypes/perc_invdyn/autumn_drive.py N2NTD <S>/driveG --actions "up,up,right,noop,noop,right,right,noop,left,left,noop,noop,noop,noop,noop,noop,left,click_0_0,noop,noop,noop,right,right,click_0_0,noop,noop,noop,noop,noop,noop,down,noop"
```
Episodes (drive, inclusive local-step range): ep0 C[4..7] · ep1 C[36..39] · ep2 A[25..28] ·
ep3 A[31..34] · ep4 B[12..17] · ep5 B[19..22] · ep6 E[19..22] · ep7 E[23..26] · ep8 E[29..32] ·
ep9 F[7..10] · ep10 F[20..27] · ep11 G[3..6] · ep12 G[17..21] · ep13 G[29..32] · ep14 E[32..35].

Verify: `verify_pool('prototypes/perc_invdyn/clean_data3/n2ntd/test50','left,right,up,down,noop,click', context_k=9)` → 52 targets, histogram §3.
Eval: `gepa_optimize.py --test-run .../n2ntd/test50 --test-n 50` (or the sweep's `--test-dir-name test50 --test-n 50`).
