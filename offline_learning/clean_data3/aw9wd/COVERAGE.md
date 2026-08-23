# aw9wd — clean_data3 coverage

Config: whitelist=`noop,click`, `keep_action_params=TRUE` (click LOCATION is part of the
label, e.g. `click 12 6`). The original trajectory also contains `up`/`left` move rows
(steps 15, 39) which are NOT whitelisted and are dropped. None of those steps fall inside
the curated slices, so no window is truncated.

## 1. Core dynamics (from dynamics.txt)

- **D1 — click breaks an eggshell.** Clicking the cell of an (unbroken) eggshell sets its
  `broken=true`. Click LOCATION matters; clicking empty space or a feather does nothing.
  The break is INVISIBLE the step it happens (color stays tan, the flag isn't rendered).
- **D2 — broken eggshells fall like liquid (passive, every step on noop).** A broken
  eggshell moves down one cell if the cell below is free, else drains sideways/diagonally
  toward the nearest hole; piles up on the floor / other settled shells.
- **D3 — despawn-on-feather → uncover (passive, on noop). THE KEY MECHANIC.** A broken
  eggshell occupying a feather cell is removed; the feather underneath (orange/yellow)
  becomes visible. Two routes: (a) click an eggshell sitting directly on a feather → it is
  removed the next step; (b) a draining eggshell that lands on a feather cell.
- **D4 — statics / no-op.** Unbroken eggshells and feathers never move; once broken shells
  settle, noop → NO_CHANGE. Arrows/noop have no handlers.

## 2. How each dynamic scores under ID / FD, and the GAPS in the original pool

| Dynamic | FD-informative? | ID-informative? | In original balanced-20? | Gap |
|---|---|---|---|---|
| D1 click breaks shell | NO at the click pair (break invisible → `click X Y` pair is NO_CHANGE) | **NO** — location unrecoverable from X_t→X_t+1 (pair is NO_CHANGE); only the ctx_next window reveals it | clicks present but every one is a NO_CHANGE pair | **GAP-1** |
| D2 liquid fall | yes (tan cell shifts; predict new position) | only as noop (passive) | yes, plentiful | minor |
| D3 despawn/uncover | yes (tan-1 + feather+1) | only as noop (passive) | present but mostly entangled w/ other falling shells, or as the lone NO_CHANGE neighbor | **GAP-2, GAP-3, GAP-4** |
| D4 static / no-op | yes (predict NO_CHANGE) | weak (no change) | abundant | — |

**GAP-1 (click ID is intrinsically unidentifiable).** Every `click X Y` transition is
NO_CHANGE: the break is invisible and its visible effect (fall / uncover) is delayed one
step to the FOLLOWING noop. So under ID neither the click nor its LOCATION can be recovered
from the scored pair `X_t→X_t+1`; it is recoverable only from the ctx_next window (the shell
falls / the feather uncovers AT the clicked cell on the next noop). Intrinsic to this game —
mitigated only by keeping cause+effect in the same slice so the window carries the cause.

**GAP-2 (nrdf6 step-counter shortcut).** The meaningful mechanics (D2, D3) fire only on
NOOP (passively). In the original trajectory clicks are issued on a rigid 4-step cadence, so
EVERY despawn lands at Step≡0 mod 4 and at the same within-window offset after a click. A
clock rule ("change at step%4==0" / "the noop right after a click always uncovers a
feather") explains the despawns without ever learning the real on-feather condition — the
exact nrdf6 failure.

**GAP-3 (despawn buried in noise mid-trajectory).** Mid-run despawn steps (e.g. 32→33,
56→57) co-occur with other shells still draining, so the uncover is not an isolated scored
signal; a random balanced-20 sample would mostly draw multi-change noops or settled
NO_CHANGE noops rather than a clean uncover.

**GAP-4 (drain-onto-feather route not isolable).** D3 route (b) only ever appears entangled
with other motion in this trajectory; only route (a), the direct click-on-feather uncover,
can be isolated as a clean target. Same underlying rule, covered via route (a).

## 3. Curated slices (verbatim original rows, one slice = one episode)

19 scored targets total; pool ≤ train-n=20 ⇒ all 19 are used. Action mix: 6 clicks (each a
unique location label) + 13 noops.

| ep | original steps | scored target pairs | dynamic(s) / role |
|---|---|---|---|
| 0 | 71,72,73,74 | `click 12 6`(NO_CHANGE); noop→**tan-1 yellow+1**; noop→NO_CHANGE | D1 cause; **D3 despawn YELLOW (clean)**; D4 settled |
| 1 | 63,64,65,66 | `click 11 11`(NO_CHANGE); noop→**orange+1 tan-1**; noop→NO_CHANGE | D1 cause; **D3 despawn ORANGE (clean, 2nd color)**; D4 |
| 2 | 82,83,84,85,86 | noop→NO_CHANGE; `click 12 9`(NO_CHANGE); noop→**tan-1 yellow+1**; noop→NO_CHANGE | D4; D1 cause; **D3 despawn at internal idx2** (varies position); D4 |
| 3 | 67,68,69,70 | `click 0 0`(NO_CHANGE); noop→NO_CHANGE; noop→NO_CHANGE | **N1 click-empty = no-op**; **N2 NO_CHANGE @ Step68≡0 mod4** (clock negative); D4 |
| 4 | 3,4,5,6 | `click 9 7`(NO_CHANGE); noop→**tan~move**; noop→**tan~move** | D1 cause (non-feather shell); **D2 fall = N3 contrast (fall, NOT despawn, after click)** |
| 5 | 7,8,9,10 | `click 9 8`(NO_CHANGE); noop→**tan~move**; noop→**tan~move** | D1 cause; **D2 clean multi-step single-cell drain** |

### Contrastive negatives (defeat the shortcuts)

- **N1 (ep3) — click on empty cell `0 0` does nothing.** Same surface cue (a click) as the
  D1/D3 clicks but zero effect → forces the rule to reference the clicked cell's contents,
  not "a click happened."
- **N2 (ep3) — NO_CHANGE noop at Step 68 (≡0 mod 4)** and at the post-click within-episode
  offset where ep0/ep1 have a despawn → kills the "change at step%4==0" / "noop after click
  always uncovers" clock.
- **N3 (ep4/ep5) — click on a NON-feather eggshell drains (D2 fall) instead of despawning
  (D3 uncover).** Same `click → noop` structure, but the next-step result is a moving tan
  cell, not tan-1+feather+1 → forces the conditional (on a feather cell vs not) rather than
  "the noop after any click uncovers a feather."
- **Position variation:** despawn appears at internal index 1 (ep0, ep1) AND index 2 (ep2),
  and the fall appears at the post-click offset (ep4/ep5), so within-episode position cannot
  proxy for the mechanic.

### Verification

`T.verify_pool('prototypes/perc_invdyn/clean_data3/aw9wd/train','noop,click')` →
19 scored targets, `{click 12 6:1, click 11 11:1, click 12 9:1, click 0 0:1, click 9 7:1,
click 9 8:1, noop:13}`; all windows full within their episode (no move-row truncation);
despawn tags `tan-1 yellow+1` / `orange+1 tan-1`, fall tags `tan~move`, negatives `NO_CHANGE`.

### Residual limitation

GAP-1 cannot be closed by data selection: in this game the click action and its location are
never recoverable from the scored center pair (always NO_CHANGE); ID for clicks must rely on
the ctx_next window, which the slices preserve by keeping each click's cause and its delayed
effect in the same episode.
