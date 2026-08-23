# dq8gc — clean_data3 coverage analysis

Whitelist: `left,right,up,down,noop,click`  •  keep_action_params = FALSE (click collapsed to verb; click LOCATION is NOT the label).

Source: single original episode `clean_data2/dq8gc/train/episode_0/trajectory.csv` (60 rows, 59 transitions; ALL actions are whitelisted, so the original whitelisted pool = all 59 pairs).

## 1. Core dynamics (from dynamics.txt)

- **D1 move LEFT** — activeParticle col-1. Unconditional (no collision check).
- **D2 move RIGHT** — activeParticle col+1.
- **D3 move UP** — activeParticle row-1.
- **D4 move DOWN** — activeParticle row+1.
- **D5 overlap / no-collision** — a move onto an occupied cell merges into one rendered cell (darkgreen count drops); moving off reveals the underlying particle (count rises). Movement never blocks.
- **D6 click-swap** — clicking an inactive particle makes it the new active and releases the old active into the inactive list. Positions/colors are unchanged at the moment of swap, so the immediate frame is NO_CHANGE; the swap is only observable because a *different* particle (different color/cell) moves on the following move action.
- **D7 passive infection / contagion** — every tick (incl. noop), an inactive particle orthogonally adjacent to ANY unhealthy (darkgreen) particle in the PREVIOUS state turns darkgreen (1-step lag). Monotone (never reverts). Spreads one cell/tick along orthogonal chains.
- **D8 active-particle health rule** — the active particle, if orthogonally adjacent to an unhealthy particle, itself turns darkgreen. (dynamics.txt says this is "not directly observed in seed-0" — but it IS exercised here at 44->45 after a click-swap made a gray particle the active next to a darkgreen one.)
- **D9 noop** — does nothing except advance the clock (passive D7/D8 still run). Most ticks are NO_CHANGE noops.
- **D10 color = health** — gray = healthy, darkgreen = unhealthy. Rendered consequence of D7/D8.
- **(D11 no win / termination / reward)** — open-ended sandbox; nothing to score as a target. Not represented (correctly).

## 2. Coverage / gap table (original balanced-20 sample vs. this curated set)

| Dynamic | TARGET under ID? | TARGET under FD? | Original-pool gap |
|---|---|---|---|
| D1 left | yes (col displacement of active) | yes (cell moves) | OK (5 lefts originally) |
| D2 right | yes | yes | OK (2 rights) |
| D3 up | yes | yes | OK (3 ups) |
| D4 down | yes | yes | OK (2 downs) |
| D5 overlap | weak (active vanishes/appears) | yes (darkgreen count ±1) | **GAP**: overlap pairs are a handful of the moves; a balanced sample easily picks only clean moves and the "move can merge/reveal, count isn't conserved" rule is never scored. |
| D6 click-swap | only via the *next* move revealing a different mover | NO_CHANGE immediate frame | **GAP (nrdf6-style)**: both clicks are NO_CHANGE, so FD scores "predict nothing" and ID can only guess. Unless the click AND its identity-revealing follow-up move land in the same window, the swap semantics are unlearnable. Balanced sampling does not guarantee that pairing. |
| D7 infection | action=noop; change=adjacent recolor | yes (gray->darkgreen) | **GAP (nrdf6-style)**: infection fires on only 3 of ~44 noops (8->9, 20->21, 44->45). A balanced sample takes ~3–4 noops at random and can easily include ZERO infection noops; the dynamic appears almost only as window context. |
| D8 active-health | action=noop; active recolors | yes | **GAP**: a single noop target (44->45), almost certainly lost in a random balanced noop draw. |
| D9 noop = no-op | n/a (NO_CHANGE) | yes (predict no change) | OK in volume, but see contrastive note below. |
| D10 color=health | covered by D7/D8 targets | covered by D7/D8 | follows D7/D8. |

## 3. Curated slices (9 episodes → 20 scored targets)

Steps are verbatim consecutive original rows. Internal consecutive pairs are scored; windows stay within a slice.

| ep | original steps | scored target(s) | dynamic covered |
|---|---|---|---|
| 0 | 3,4 | 3->4 `down` (dg active, clean, move (1,0)) | D4 — clean directional, ID+FD |
| 1 | 7,8,9 | 7->8 `right` (dg clean); 8->9 `noop` gray(3,4)->dg | D2; **D7 infection positive** |
| 2 | 19,20,21 | 19->20 `left` (dg clean); 20->21 `noop` gray(5,3)->dg (chain) | D1; **D7 infection positive** |
| 3 | 23,24 | 23->24 `up` (dg active, clean) | D3 — clean directional |
| 4 | 11,12 | 11->12 `right` overlap, darkgreen 2->1 (merge) | **D5 overlap (contrastive vs clean right 7->8)** |
| 5 | 15,16 | 15->16 `down` reveal, darkgreen 1->2 | **D5 overlap (contrastive vs clean down 3->4)** |
| 6 | 27,28,29,30,31,32 | 27->28 `click 5 7` NO_CHANGE; 28->29,29->30,30->31 `noop` NC; 31->32 `up` gray active | **D6 click-swap** + window carries the gray-up that reveals the swap; 3 **D9 NO_CHANGE noop negatives** |
| 7 | 51,52,53,54,55,56 | 51->52 `click 6 6` NO_CHANGE; 52->53,53->54,54->55 `noop` NC; 55->56 `left` gray active | **D6 click-swap** (2nd, different cell) + revealing gray-left in window; 3 **D9 noop negatives** |
| 8 | 43,44,45 | 43->44 `up` gray active; 44->45 `noop` gray-active(3,5)->dg | D3 (gray active); **D8 active-health positive** |

### Verified scored pool (T.verify_pool)
- 20 scored targets. By verb: `down`×2, `right`×2, `left`×2, `up`×3, `noop`×9, `click`×2.
- Click targets carry window `next=4`, so the identity-revealing follow-up move (31->32, 55->56) is inside the click's context window — the swap is learnable, not just NO_CHANGE.

### Contrastive structure (defeats shortcuts)
- **Infection / active-health vs. inert noop:** 3 noop POSITIVES where a cell turns darkgreen (8->9, 20->21, 44->45) against 6 noop NEGATIVES that are NO_CHANGE (28->29, 29->30, 30->31, 52->53, 53->54, 54->55). The positives are at irregular spacing and depend on adjacency-to-darkgreen, so a "noop always recolors" or "recolor every k steps" clock scores strictly worse than the real adjacency rule.
- **Overlap vs. clean move (count conservation):** clean down 3->4 / clean right 7->8 (darkgreen count unchanged) against overlap down 15->16 (count 1->2) and overlap right 11->12 (count 2->1). A rule that assumes "a move just translates one cell and conserves color counts" is penalized; the no-collision/merge rule is required.
- **Click-swap vs. move:** both clicks are NO_CHANGE on the scored pair, but the in-window follow-up shows a DIFFERENT-colored particle now moving (gray, not darkgreen) — so the swap-of-control semantics is the only explanation that fits both the NO_CHANGE click and the subsequent mover identity.
- **dg-active vs. gray-active movement:** moves are present for both the original darkgreen active (D1–D4 at steps 3–24) and the post-swap gray active (up 31->32, up 43->44, left 55->56), so movement is bound to "the active particle" rather than to a fixed color/cell.

## TRAIN2 (untied-val expansion)

Purpose: the original `train/` (20 scored transitions) is ≤ `--train-n 20`, so `balanced_split`
returns the WHOLE pool for both train and val (tied). `train2/` adds a second, independently
regenerated batch so the combined pool (`train/` 20 + `train2/` 31 = 51) is large enough for a
genuine untied 30-train/30-val split. Nothing in `train/`, `test/`, `test50/`, or `dynamics.txt`
was touched.

### Regeneration

Fresh 40-transition drive, seed 0, saved verbatim to `train_regen2/episode_0/trajectory.csv`
(filmstrip: `train_regen2/viz.html`). Deterministic command (repo root):

```
uv run python prototypes/perc_invdyn/autumn_drive.py DQ8GC prototypes/perc_invdyn/clean_data3/dq8gc/train_regen2 \
  --actions "noop,down,down,down,click_6_6,noop,right,noop,left,left,left,left,noop,left,left,up,up,up,up,right,right,noop,right,noop,right,down,left,left,noop,click_3_4,up,noop,click_5_7,down,left,left,left,noop,up,noop"
```

Every event below was confirmed empirically against the printed ASCII frames / `T.classify`
tags before curation (not hand-derived from the SEXP alone) — see the row-by-row cell dump used
during construction; each POS/NEG claim matches an observed `darkgreen+1 gray-1` /
`black+1 darkgreen-1` / `NO_CHANGE` tag at the exact transition cited.

### Curated slices (6 episodes → 31 scored targets, verified via `T.verify_pool`)

| ep | train_regen2 steps | scored target(s) | dynamic covered |
|---|---|---|---|
| 0 | 2,3,4,5,6,7,8 | 2->3,3->4 clean `down`×2 (dg active); 4->5 `click 6 6` **SUPPRESSION-at-a-distance + SWAP** (D pending via the just-arrived active at (5,2), click targets the FAR particle C at (6,6) instead — NO_CHANGE); 5->6 `noop` **D7+ (D (5,3) gray→dg, fires one tick after being suppressed — the demoted old-active, not a moving active, is the adjacency cause)**; 6->7 `right` (gray C moves — reveal-of-swap); 7->8 `noop` **D7− NEG: gray-gray adjacency (C↔A(5,7)) — no infection, proves a healthy neighbor never triggers it** | D4×2, D6 (distance-suppression), D7 POS+NEG |
| 1 | 10,11,12,13 | 10->11,11->12 clean `left`×2 (gray C, arrival adjacent to D); 12->13 `noop` **D8+ #1: gray C (6,3) → dg, approached from the SOUTH** | D1×2, D8 POS |
| 2 | 18,19,20,21,22,23,24 | 18->19 clean `up`; 19->20,20->21 clean `right`×2 (arrival DIAGONAL to B(3,4)); 21->22 `noop` **D7− NEG: diagonal adjacency, dg C↔gray B — no infection (orthogonal-only rule)**; 22->23 `right` (arrival ORTHOGONAL to B); 23->24 `noop` **D7+ #2: B (3,4) gray→dg** | D3, D2×3, D7 NEG+POS (diagonal-vs-orthogonal minimal pair) |
| 3 | 24,25,26,27,28 | 24->25 clean `right`; 25->26 clean `down` (arrival next to dg B, monotone); 26->27 `left` **D5 MERGE (NOVEL DIRECTION — LEFT; unused by train/ and test50/, which only ever merged via RIGHT/UP)**, darkgreen 2→1; 27->28 `left` **D5 REVEAL (LEFT)**, darkgreen 1→2 | D2, D4, D5×2 (first-ever LEFT overlap pair) |
| 4 | 28,29,30,31 | 28->29 `noop` NC settle; 29->30 `click 3 4` **SWAP on an INFECTED (dg) inactive that caught contagion mid-drive** (NO_CHANGE); 30->31 `up` (dg B moves — reveal-of-swap) | D6 (click on a contagion-infected inactive), D3 |
| 5 | 31,32,33,34,35,36,37,38,39,40 | 31->32 `noop` NC settle; 32->33 `click 5 7` **SWAP on the LAST still-healthy particle in an otherwise 4/5-infected board, quiet/late-game** (NO_CHANGE); 33->34 `down` (gray A moves — reveal-of-swap); 34->35,35->36 clean `left`×2; 36->37 `left` (arrival DIAGONAL to D); 37->38 `noop` **D8− NEG: diagonal, A stays gray**; 38->39 `up` (arrival ORTHOGONAL to D); 39->40 `noop` **D8+ #2: A (5,4) gray→dg, approached from the EAST — final particle infected, board fully darkgreen** | D6 (click on last-healthy, late-game), D4, D1×2, D8 NEG+POS (2nd direction), full-infection closure |

### Verified scored pool (`T.verify_pool`)
- **31 scored targets.** Raw by-action: `down`×4, `right`×5, `left`×7, `up`×3, `noop`×9,
  `click 6 6`×1, `click 3 4`×1, `click 5 7`×1 (→ `click`×3 under `keep_action_params=False`).
- Every core dynamic appears ≥2 times as a scored target: D1–D4 (3–7 each), D5 overlap (2, both
  the never-before-used LEFT direction), D6 click-swap (3, three distinct swap contexts), D7
  infection (2 POS + 2 NEG), D8 active-health (2 POS + 1 NEG).

### Dynamic → target-pair coverage table

| Dynamic | POS targets in train2 | NEG (near-miss) in train2 |
|---|---|---|
| D1 left | 7 (gray C ×4, gray A ×3; incl. 2 diagonal/no-op setup moves) | — |
| D2 right | 5 (dg swap-reveal, dg clean ×2, diagonal-approach, orthogonal-approach) | — |
| D3 up | 3 (clean, reveal-of-swap, orthogonal-approach) | — |
| D4 down | 4 (clean ×2, monotone-arrival, reveal-of-swap) | — |
| D5 overlap | 2 = 1 MERGE + 1 REVEAL, **both LEFT** (unused direction in train/ and test50/, which only used RIGHT/UP/DOWN) | every clean move around it (count conserved) is the standing contrast |
| D6 click-swap | 3: distance-suppression (D pending, click far C), quiet-on-infected-inactive (B, dg), quiet-on-last-healthy (A, gray, late-game) | the 2 D7 positives are the "no click → dynamic fires" contrast; every click is itself NO_CHANGE (the game's own click-vs-noop aliasing, see below) |
| D7 infection | 2 (D via distance-suppression-then-fire; B via approach) | 2 (gray-gray adjacency: C↔A; diagonal adjacency: C↔B) |
| D8 active-health | 2 (C from the south; A from the east — two different approach directions, each preceded by its own diagonal negative) | 1 (A diagonal to D) |
| D9 noop inert | — | 5 pure/near-miss NO_CHANGE noops (7->8, 21->22, 28->29, 31->32, 37->38) vs. 4 positive-recolor noops (5->6, 12->13, 23->24, 39->40) — irregular spacing, no step%k pattern |

### Contrastive negatives (~26% of the pool)
Counting the 3 "hard" near-miss negatives (gray-gray D7−, diagonal D7−, diagonal D8−) plus the
3 click transitions (each NO_CHANGE despite being a real, effectful action — dq8gc's own
inherent click/noop aliasing, the same category TEST_COVERAGE.md documents for test50) plus 2
pure settle noops = 8 of 31 (~26%), in the target 20–30% range. The diagonal-vs-orthogonal pairs
(D7 at ep2, D8 at ep5) are placed back-to-back as minimal pairs (same actor, same general
region, only the diagonal/orthogonal geometry differs) specifically to block an
"any-proximity infects" shortcut.

### How train2 differs from train/ and test50/ (no situation is replicated)

- **LEFT-direction overlap (merge + reveal) is demonstrated for the first time.** `train/` used
  RIGHT (merge) and DOWN (reveal); `test50/` used RIGHT (dg-dg and gray-gray merge) and UP/DOWN
  (reveal). Neither ever merged or revealed via LEFT. train2's ep3 fills this gap.
- **A new suppression mechanism:** train2's click_6_6 suppresses D's pending infection where the
  adjacency cause is a **just-demoted, stationary particle** (the original active, released one
  tick earlier and sitting at (5,2)) — not a moving current active as in test50's
  suppression-at-a-distance example (ep5: pending caused by the CURRENT active's own approach).
- **Click targets in fresh health/board states:** train2 clicks (a) a particle mid-suppression
  (D, indirectly, via clicking elsewhere), (b) an inactive that turned dg **through contagion**
  (B) — distinct from test50/train, whose only dg-inactive click was the pristine ORIGINAL
  active particle at (2,2) that never moved — and (c) the LAST remaining healthy particle in an
  otherwise fully-infected board (A), late in the drive. train/ clicked (5,7) early (board only
  1/6 infected at that point); test50 never clicked (5,7) in a near-total-infection state.
- **D8 demonstrated from two fresh approach directions** (south then east), each with its own
  explicit diagonal-negative control taken immediately beforehand — a tighter minimal-pair
  ablation than either existing pool provides for this dynamic.
- **The infection is traced to full board closure** (all 5 particles end darkgreen) inside a
  single drive — neither `train/` nor `test50/` narrates the spread to totality end-to-end.
- **Irregular cadence throughout** (noop gaps of 0–3, positives/negatives interleaved,
  directional moves of varying run-length) — no fixed step%k pattern for GEPA to exploit.

### Caveats / known limits
- Some individual dynamic instances (e.g. a further RIGHT/UP merge, or another quiet click) are
  the SAME general type as something in train/ or test50/ — only 4 inactive particles + 1
  original active exist, so every click necessarily lands on one of the same 5 cells eventually;
  what differs is the surrounding board state, approach direction, and timing, which is what the
  methodology's "situation" requirement actually targets (verified distinct above per instance).
- As in test50, dq8gc's click is inherently NO_CHANGE at the pair level; all 3 clicks here rely
  on the in-window follow-up move for ID recoverability (context_k=9 keeps it inside the window
  in every case, confirmed via `T.verify_pool`'s printed win(prev,next) sizes).
- `train_regen2/` (40 transitions, 1 episode) is intentionally longer than what's curated into
  `train2/`; the un-curated remainder (steps not selected into any episode) contains only
  redundant clean directional moves and was dropped to keep `train2/` at ~30 rather than ~40.
