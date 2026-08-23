# eahcw — TEST50 held-out test pool coverage

Config: whitelist = `left,right,up,down,noop,click`; `keep_action_params=FALSE`
(movement-style game — `click` collapses to the bare verb `click`; the click LOCATION
is NOT the label). Pool = **50 scored target transitions** across 8 freshly-driven
episodes. `balanced_split` returns the whole pool when pool size ≤ `--test-n`, so run
with `gepa_optimize.py --test-run .../eahcw/test50 --test-n 50` and every pair below is
scored.

Verified: `T.verify_pool('.../eahcw/test50','left,right,up,down,noop,click',context_k=9)`
→ 50 scored transitions; every colored/red click carries its arming key(s) inside the
`ctx_prev` window (prev distances 1–6, all < context_k=9).

## 1. Core dynamics (from dynamics.txt)

Free-form 16×16 paint canvas. Hidden state: `currColor` (init "red"), `active_arrow`
(init "none"), `particles` (grows only by clicking; never moves/despawns).

- **D1 — click spawns a particle** at the clicked cell; particles persist & accumulate.
- **D2 — click color = red when no arrow armed** (`active_arrow == "none"`).
- **D3 — a direction key arms its own paint color**: up→gold, down→purple, left→green,
  right→blue (each also sets `active_arrow`). The arming is INVISIBLE in the grid; it only
  manifests in the color of the NEXT click.
- **D4 — opposite-arrow cancel**: pressing the opposite of the armed arrow resets to
  none/red (down⇄up, left⇄right). Net effect = a reset; the next click is red, not the
  armed color.
- **D5 — noop / no passive dynamics**: no `on true` rule, no clock, no goal, no collisions.
  Nothing changes without a click.

## 2. Intrinsic observability limit (read before the table)

**Only the 23 `click` transitions are ID-recoverable.** A click adds exactly one colored
cell, so its verb is identifiable and its location/color are FD-informative. The other
**27 transitions (5 up, 4 down, 4 left, 4 right, 10 noop) are all NO_CHANGE** — arrow keys
never move a cell (they only set hidden pen state), so up/down/left/right/noop are
**mutually ID-aliased**: an oracle that knows every rule still scores chance on them from a
single `(X_t, X_t+1)` pair. This is a property of EAHCW (dynamics.txt: "The arming itself is
INVISIBLE in the grid"), **not** a curation defect, and cannot be removed by data selection.
It is minimized here by making clicks 46% of the pool (the maximum reasonable observable
fraction while still carrying arm/cancel context and exercising D5). Consequently D3/D4 are
scored **indirectly**, through the COLOR of the next click, with the causing arrow key(s)
held in that click's `ctx_prev` window.

## 3. Per-dynamic coverage (positives AND negatives, under ID and FD)

| Dynamic | Scored TARGET positives | ID | FD | Contrastive negatives |
|---|---|---|---|---|
| **D1** click spawns | **23** click transitions (all episodes) | YES — new cell ⇒ verb `click` | YES — predict the new cell at the click loc | the 27 NO_CHANGE arrow/noop targets are the "nothing spawns without a click" contrast |
| **D2** red when no arm | **7** red clicks: virgin `click 2 5` (ep0, empty window), post-cancel `click 0 9` (ep2), + 5 cancel-reds (ep1/2/3/4/7) | via D1 (click) | YES — predict a RED cell | stands against 16 colored clicks; cancel-reds prove "armed key in window ≠ colored" |
| **D3** dir arms color | **16** colored clicks — up→gold ×4 (ep0 s5,s6; ep1 s8; ep5 s34), down→purple ×4 (ep2 s14; ep5 s38; ep7 s46,s48), left→green ×3 (ep3 s20,s25; ep5 s36), right→blue ×5 (ep4 s27,s32; ep6 s40,s42,s44) | indirect (click) | YES — predict the ARMED color; arm key in `ctx_prev` (prev 1–6) | **cancel** (5 reds) = "same arm key, color NOT armed"; **orthogonal re-arm** ep5 (up→gold then left→green then down→purple) = "2nd/3rd arrow SWITCHES color, does NOT reset to red" |
| **D4** opposite cancel | **5** cancel→red clicks covering all 4 directed pairs: up⇄down `click 11 11` (ep1), down⇄up `click 13 4` (ep2) & `click 15 7` (ep7), left⇄right `click 9 7` (ep3), right⇄left `click 1 1` (ep4) | indirect (click) | YES — predict RED despite an armed key in the window (BOTH arm + opposite key in prev) | **orthogonal switch** (ep5: non-opposite 2nd arrow keeps a color) + **re-arm after cancel** (ep3 s25 green, ep4 s32 blue: cancel is not permanent) + **persistence** clicks (ep0/6/7) defeat "any 2nd arrow → red" |
| **D5** noop / no passive | **10** noop NO_CHANGE targets (ep0,1,2,3,4,6,7) | trivial (NO_CHANGE) | trivial (no change) | irregular noop gaps (0/1/2) + irregular click offsets ⇒ any `step % k` "clock" shortcut scores worse; 18 arrow NO_CHANGE targets reinforce "arrows change no cell" |

Every core dynamic is a scored target ≥4× (D1 23, D2 7, D3 16 across all 4 mappings each
≥3, D4 5 across all 4 pairs, D5 10).

## 4. Action histogram of the pool (verbs; keep_action_params=False)

`click 23 · noop 10 · up 5 · down 4 · left 4 · right 4` = **50**.

Click-heavy by design: clicks are the only ID-observable verb and the only carrier of the
D2/D3/D4 color signal, so over-weighting the (invisible) arrows would only add unsolvable
ID items. Arrows are kept at 4–5 each to (a) trigger/carry D3 arming & D4 cancels in
windows and (b) supply D5 negatives.

## 5. How TEST50 differs from `train/` (cross-trajectory generalization)

| | train (30 targets, 6 eps) | test50 (50 targets, 8 eps) |
|---|---|---|
| Source | 6 slices of ONE accumulating clean_data2 trajectory | 8 independently DRIVEN episodes (seed 0) |
| Timing | rigid cadence: arm + **exactly 3 noops** + click | irregular gaps (0/1/2 noops), varied click offsets |
| Clicks/episode | exactly 1 | 2–3 (varied canvas density) |
| D4 cancel | up⇄down only (1 instance) | **all 4 directed pairs** (up⇄down, down⇄up, left⇄right, right⇄left) |
| Mechanics not in train | — | arm-**persistence** across clicks/noops (ep0/6/7), **orthogonal re-arm chains** (ep5), **re-arm after cancel** (ep3/4), **post-cancel virgin red** (ep2) |
| Click positions | (3,3)(5,7)(4,9)(10,10)(12,5)(6,14) | 23 positions, all distinct from train |
| Filler | 18 noop fillers | 10 noops (leaner; heavier on scored color clicks) |

Same rules, different situations — no action sequence, position, or state configuration
replicates a train slice.

## 6. Uncoverable / residual gaps (deliberate, intrinsic to EAHCW)

- **Arrow/noop ID-aliasing.** 27/50 transitions are NO_CHANGE and mutually
  indistinguishable under ID — arrows never move a cell (dynamics.txt). Unfixable by
  curation; minimized by the 46% click fraction. Under FD they are correct-by-"no change".
- **D3/D4 only indirectly scorable.** The arm/cancel transitions themselves are never
  ID/FD-informative; their effect is scored on the next click, whose window carries the
  causing key(s) (verified prev 1–6 < k=9). A perception module must infer the hidden pen
  color from the arm-key HISTORY in the window — the intended difficulty of the game.
- **Hidden state.** `currColor` / `active_arrow` are never rendered; no click can reveal
  the pen color except by being placed. This is the mechanic, not a data gap.
