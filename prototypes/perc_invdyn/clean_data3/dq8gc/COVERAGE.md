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
