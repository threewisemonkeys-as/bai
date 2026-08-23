# eahcw — clean_data3 coverage

Config: whitelist = `left,right,up,down,noop,click`; `keep_action_params=FALSE`
(movement-style game — `click` collapsed to the verb `click`; the click LOCATION is
NOT the label). Pool = **20 scored targets** in 4 curated episodes → fits the default
`--train-n 20` so `balanced_split` returns the whole pool deterministically.

## 1. Core dynamics (from dynamics.txt)

eahcw is a free-form paint canvas (16x16). State: `particles` (grows only by clicking,
never despawns/moves), `currColor` (init "red"), `active_arrow` (init "none").

- **D1 — click spawns a particle** at the clicked cell; particles persist & accumulate.
- **D2 — click color = red when no arrow armed** (`active_arrow == "none"`).
- **D3 — a direction key arms its own paint color**: up→gold, down→purple, left→green,
  right→blue (each also sets `active_arrow`). The arming itself is INVISIBLE in the grid;
  it only manifests in the color of the NEXT click.
- **D4 — opposite-arrow cancel**: pressing the opposite of the armed arrow resets to
  none/red (down⇄up, left⇄right). Net effect = a reset; the next click is red, not the
  armed color.
- **D5 — noop / no passive dynamics**: no `on true` rule, no clock, no goal/termination,
  no collisions. Nothing changes without a click.

## 2. Scorability under the objectives + gaps in the ORIGINAL train pool

| Dynamic | ID target? | FD target? | In a balanced-20 of the ORIGINAL 85-row pool? | Gap |
|---|---|---|---|---|
| D1 click spawns | YES — a new cell appears ⇒ verb `click` recoverable | YES — predict a new colored cell at click loc | clicks exist (10/85) but a random `rest[:20]` is noop-dominated | clicks under-sampled, not paired with cause |
| D2 click=red (no arrow) | via D1 (click) | YES — red cell appears | maybe 1 red click, no contrast | uncontrolled |
| D3 dir arms color | **NO — dir keys are grid-invisible ⇒ NO_CHANGE ⇒ unidentifiable** | INDIRECT — only via the next click's color, and ONLY if the dir key sits in that click's ctx_prev window (k=9) | dir key & its click are 4 rows apart (3 noops) → in original they usually land in DIFFERENT balanced-sample slots, so the rule appears as window context, not a scored target | **the nrdf6 gap: dynamic only as context / passively unidentifiable** |
| D4 opposite-cancel | NO (cancel is invisible) | INDIRECT — needs BOTH arm + opposite key in the click's window | only up(63)/down(67) are <=9 rows apart from a click; left/right cancels have the first key >9 rows back → window shows only the 2nd key → cancel indistinguishable from a fresh arm | only the up/down cancel is cleanly scorable |
| D5 noop / no passive | trivial (NO_CHANGE) | trivial (no change) | noops are 74/84 of the original → over-represented; a "predict no-change / guess noop" shortcut scores high | no clock shortcut (good — unlike nrdf6 there is no `step%k` rule), but NC majority can game both objectives |

**Headline gap:** every non-click action (up/down/left/right/noop) leaves the grid
unchanged, so under ID they are mutually indistinguishable — only `click` is recoverable.
The arming rules (D3) and the cancel (D4) are therefore scorable ONLY indirectly, through
the COLOR of a subsequent click, and only when the causing key(s) are carried in that
click's window. The original wide spacing (3 noops between every event) means a default
balanced-20 sample rarely pairs a click with its cause as a controlled target — exactly
the nrdf6 "dynamic present only as context" failure.

## 3. Curated slices (verbatim contiguous original rows) and what each target covers

| Episode | Original steps | Targets | Covers | Contrastive role |
|---|---|---|---|---|
| 0 | [3,4] | `click@(3,3)`→red | D1, D2 (red, `active_arrow=none` initial) | baseline red (empty window = no arrow) |
| 1 | [7..12] | up→(3×noop)→`click@(5,7)`→**gold** | D1, D3(up→gold), D5 | gold click; `up` carried in click window (prev=4) |
| 2 | [31..36] | down→(3×noop)→`click@(10,10)`→**purple** | D1, D3(down→purple), D5 | purple≠gold ⇒ defeats "any dir→one color" |
| 3 | [63..72] | up→(3×noop)→down→(3×noop)→`click@(6,14)`→**red** | D4 cancel, D2, D5 | **near-miss negative**: same `up`-arm as ep1 but the opposite `down` cancels ⇒ click is RED not gold (both keys in window, prev=8) |

Verified pool (`verify_pool`): **20 targets** — by verb: `click`=4 (red, gold, purple,
red), `up`=2, `down`=2, `noop`=12. Key windows: gold-click prev=4 (sees `up`),
purple-click prev=4 (sees `down`), cancel-click prev=8 (sees BOTH `up` and `down`).

### Contrastive negatives (defeat the shortcuts)
- **"up always → gold"** is defeated by ep3: `up` then `down` → red. Same arm, cancelled.
- **"down always → purple"** is defeated by ep3: `down` here CANCELS (→red) vs ep2 where
  `down` (from none) ARMS purple. Same key, state-dependent outcome.
- **"click is always red"** / **"click color is fixed"** is defeated by the gold & purple
  clicks coexisting with the two red clicks.
- **"predict no-change / always noop"** (the NC-majority shortcut) is countered by the 4
  click targets, which are the only ID-recoverable and FD-informative transitions.

## 4. Known residual gaps (deliberate, to keep the pool deterministic at 20)
- **left→green and right→blue are not directly scored.** Adding each as its own windowed
  click target costs +5 (dir + 3 noop + click), and covering all four colors + the cancel
  would push the pool to ~29 (> `train-n 20` ⇒ random drops). D3 is instead scored via
  up→gold and down→purple (two distinct mappings establishing "direction selects a
  direction-specific color"); green/blue rely on belief generalization. To score them too,
  add episodes `[15..20]` (left→green) and `[47..52]` (right→blue) and run with
  `--train-n 30`.
- **Dir keys / noop remain ID-unidentifiable** (no grid change) — intrinsic to the game,
  not fixable by data curation; they earn no ID signal and are correct-by-default under FD.

---

## REVISED (full-coverage rebuild)

The initial build scored only 2 of the 4 arrow→color arming mappings (up→gold, down→purple).
Since arrow→color arming (D3) is the game's defining mechanic, the train set was rebuilt to
cover **all four** mappings as scored FD targets, each with its arming key inside the click's
`ctx_prev` window:

- baseline `click`→**red** (no arrow armed)        — ep0 [3,4]
- `up`→**gold**    — ep1 [7..12]   (up in click prev=4)
- `left`→**green** — ep2 [15..20]  (left in click prev=4)
- `down`→**purple**— ep3 [31..36]  (down in click prev=4)
- `right`→**blue** — ep4 [47..52]  (right in click prev=4)
- `up` then opposite `down` **cancel**→**red** (D4) — ep5 [63..72] (both keys in click prev=8)

Final pool = **30 scored targets** (by verb: click 6, up 2, down 2, left 1, right 1, noop 18;
the 18 NO_CHANGE noops are the arm/cancel-episode fillers and double as "noop = nothing" /
"click is not periodic" negatives). The arm-color FD signal lives in the window (the arming
key is invisible in the grid), so windows were preserved by using contiguous slices.

**Run note:** this game's pool is 30 (> the default 20), so eahcw must be run with
`--train-n 30 --val-n 30` for `balanced_split` to keep every curated target. All other games
fit the default `--train-n 20`.
