# ntq4y — clean_data3 coverage

Sandbox water-painter. Whitelist = `noop,click`; **keep_action_params = TRUE**
(the click LOCATION is part of the label, e.g. `click 10 7`). Action/obs use (row, col);
dynamics.txt writes positions as (col, row). The trajectory also contains move verbs
(`left` at step 56) which are dropped by the whitelist — none of the chosen slices touch
step 56, so no window is truncated.

Recall the two objectives:
- **ID**: given P(X_t), P(X_t+1) (+ window) recover the hidden action. Recoverable only when
  the X_t->X_t+1 change reveals the action (placed cell at the click loc; mass remove/clear).
- **FD**: given P(X_t) **and the action** (+ window) predict P(X_t+1). Informative only when the
  action actually changes the state, and the change requires the rule (color of placed
  particle, water motion, mass deletion).

## 1. CORE dynamics

- **D1 — place particle on a free cell.** `click` on an empty cell places ONE particle of the
  currently-selected type at the clicked coords. Color = currentParticle (vessel=purple /
  plug=orange / water=blue).
- **D2 — toolbar selection.** `click` a row-0 button (vessel col2 / plug col5 / water col8)
  sets the hidden `currentParticle`. NO grid change on the click itself; the effect only shows
  up as the COLOR of the next placed particle.
- **D3 — removeButton (col 11).** `click` it deletes ALL plug (orange) cells.
- **D4 — clearButton (col 14).** `click` it deletes ALL vessels + plugs + water.
- **D5 — occupied / non-free click is a no-op.** `click` on an occupied (non-button) cell does
  nothing (isFreePos guard). Contrast to D1.
- **D6 — passive water simulation (every step).** Each water cell falls straight down if the
  cell below is free, else steps diagonally toward the nearest reachable hole; otherwise stays.
  Runs on EVERY step (noop included), independent of the action.
- **D7 — drain on unplug.** Removing the plugs (D3) opens the bottom of the funnel so water held
  above the plugs drains downward. (D3+D6 composed; precondition = plugs gone.)

(No color changes, no spawning/despawning beyond clicks/buttons, reward always 0 — nothing to
score there.)

## 2. Is each dynamic SCORED as a target in the ORIGINAL train pool? Gaps

Original train = 89 rows, 20 clicks + ~68 noops (51 of them NO_CHANGE). With keep_action_params,
`noop` is ONE giant bucket and almost every click is a unique label, so a balanced-20 sample
(round-robin by label) grabs ~one of each click but only **~1–2 noop targets**.

| Dynamic | ID target? | FD target? | In a balanced-20 of ORIGINAL? | Gap |
|---|---|---|---|---|
| D1 place (purple/orange/blue) | yes (cell @ click loc) | yes (color) | likely (clicks well sampled) | OK, but color-FD needs the selection in-window — random sampling rarely keeps the select->place adjacency |
| D2 selection | no (own step is NO_CHANGE) | only via downstream placement color | button clicks present but as NO_CHANGE targets; select->place pairing not guaranteed | **GAP** — selection is only learnable when its placement is a target AND the button click sits in that target's window; balanced sampling does not preserve this |
| D3 removeButton | yes | yes | one instance; ~50% kept | thin (single occurrence) |
| D4 clearButton | yes | yes | one instance; ~50% kept | thin (single occurrence) |
| D5 occupied-click no-op | no (NO_CHANGE) | no (NO_CHANGE) | one instance (`click 2 2`@52), ~uncertain | **GAP** — the key D1 *negative* is a lone NC click easily dropped |
| D6 water sim | partial (noop recoverable when water moved) | **yes, the crux** | **~1–2 noop targets, most likely a NO_CHANGE one** | **GAP (the nrdf6 failure)** — the only passive dynamic is squeezed into the single under-sampled `noop` bucket; the sampled noop is probably a no-motion one, so the rule is seen only as window context + invites a step-clock shortcut |
| D7 drain on unplug | partial | yes | as above (noop bucket) | **GAP** — same noop-starvation; the cause(removeButton)->effect(drain) adjacency is not guaranteed in one window |

**Summary of gaps (nrdf6-shaped):** the active click dynamics (D1/D3/D4) are fine, but everything
that lives on `noop` — the water simulation (D6), the drain (D7) — is collapsed into one balanced
bucket and almost never scored as the *target*, while the hidden selection (D2) and the D5 negative
are fragile single instances. Fix: hand-pick contiguous slices so every dynamic is a scored
target under BOTH objectives, with select->place and unplug->drain causes kept inside the target's
window, and with near-miss negatives so a clock / "noop always moves water" / "click always places"
shortcut scores worse.

## 3. Curated slices (verbatim original rows; one slice = one episode)

Verified pool = **20 scored targets** (`verify_pool`), 8 click + 12 noop, pool == train_n so all kept.

| Episode (orig steps) | target pair | action | change | dynamic / role |
|---|---|---|---|---|
| EP0 [3,4] | 3->4 | click 2 2 | purple+1 @(2,2) | **D1 place vessel** (currentParticle defaults to vessel; ID=loc, FD=color via default) |
| EP1 [15..20] | 15->16 | click 0 5 | NO_CHANGE | **D2 select plug** (cause event; sits in window of the placement below) |
| | 16->17,17->18,18->19 | noop | NO_CHANGE | **D6 negative** (noop, no water → nothing moves; defeats "noop moves water"/clock) |
| | 19->20 | click 4 6 | orange+1 @(4,6) | **D1 place plug + D2** (FD must read plug-selected from window=prev4 to predict orange) |
| EP2 [27..34] | 27->28 | click 0 8 | NO_CHANGE | **D2 select water** (cause; window of placement) |
| | 28->29,29->30,30->31 | noop | NO_CHANGE | **D6 negative** ×3 |
| | 31->32 | click 10 7 | blue+1 @(10,7) | **D1 place water + D2** (FD must read water-selected from window=prev4 → blue) |
| | 32->33 | noop | blue down 1 | **D6 straight fall** (FD positive; ID=noop since no placed cell) |
| | 33->34 | noop | blue down 1 | **D6 straight fall** (FD positive) |
| EP3 [38..42] | 38->39 | noop | blue diagonal | **D6 sideways spread** (FD positive) |
| | 39->40 | noop | NO_CHANGE | **D6/D7 negative** — water present but confined by plugs → static (contrast to drain below) |
| | 40->41 | click 0 11 | orange-8 | **D3 removeButton** (ID=loc col11 + all plugs vanish; FD=plugs→black) |
| | 41->42 | noop | blue moves down | **D7 drain on unplug** (same noop, but now plugs gone → water moves; contrast with 39->40) |
| EP4 [75,76,77] | 75->76 | noop | blue moves | **D6** water motion |
| | 76->77 | click 0 14 | purple-16 orange-1 blue-3 | **D4 clearButton** (mass deletion; ID+FD) |
| EP5 [52,53] | 52->53 | click 2 2 | NO_CHANGE | **D5 negative for D1** — same `click 2 2` as EP0 but cell now occupied → no placement (defeats "click always places") |

### Contrastive negatives (the shortcut each one defeats)
- **D1 vs D5:** `click 2 2` places a vessel when free (EP0) but does nothing when occupied (EP5) —
  identical action label, opposite outcome → "click always places" scores worse than the isFreePos rule.
- **D6 clock/"noop moves water":** water-motion noops at varied step offsets (32,33 / 38 / 41 / 75)
  vs static noops (16–18, 28–30, 39, 52) → a step%k clock or "noop always moves water" mispredicts.
- **D7 precondition:** noop 39->40 (water static, plugs IN) vs noop 41->42 (water drains, plugs OUT) —
  the only difference is the intervening removeButton; forces the drain rule to be conditional on unplug.
- **D2:** plug/water selection clicks (EP1 15->16, EP2 27->28) are NO_CHANGE on their own step and live
  in the *window* of their placement target, so the only way to predict the placed COLOR (orange/blue)
  under FD is to track the last toolbar button — the selection rule, not a guess.

### Notes / known limits
- D2 selection is **not directly ID-scorable** (its own transition is NO_CHANGE, indistinguishable from
  noop or any other NC click). It is covered under **FD via the downstream placement color**, with the
  button click guaranteed inside the placement target's window (prev=4 reaches it — confirmed by
  verify_pool). This is the intended mechanism, not a gap.
- A handful of NO_CHANGE targets (selection clicks, bridge noops, the D5 occupied click) are mutually
  unidentifiable under ID by construction — they are the contrastive negatives, not the signal-bearing
  targets. The signal-bearing half (3 placements + remove + clear + 5 water-motion noops) keeps ID/FD
  informative.
