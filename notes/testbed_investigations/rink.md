# rink — testbed assessment (2026-08-23)

Engine-level investigation of whether `rink` (zip-sourced, 28x28) is a valid testbed, prompted
by the 55-game atlas rating it Tier X ("484 static cells for a 1-cell agent; skater can walk
off the grid; momentum already covered by ricochet_robots"). Every number verified on the
interpreter (render after every step, seeds >= 1) and cross-checked against a python mirror of
the .sexp (0 mismatches / 3000 engine transitions). Probe scripts:
`scripts/testbed_probes/rink/`.

## VERDICT: USABLE WITH FIXES (small, mechanical fixes)

Every atlas objection against `rink` is real *as measured*, but two of the three are artefacts
of the atlas probe's unguarded uniform-random policy, not properties of the game. The skater
starts at the corner `(0,0)`, so a uniform random drive walks off-grid on step 1–2 and spends
**68–78 % of ticks invisible**; every "nothing happens" number in the atlas (`vis 0.25`,
`distinct 1.88`, `passive 0.00`, `noop-change 0.00`) is dominated by those blank frames. Under
a 5-line render-only guard the same game measures `vis h1 = 0.99`, `distinct 4.83/5`, and under
an ice-seeking drive `noop-change 0.80` — momentum is the most active passive dynamic in our
catalogue. The repo's own `game_profile.py` rates rink **HIDDEN(4/297), QUIET-AT-RESET, no
DRIFT, no RNG, no OCCLUSION** — *cleaner than DQ8GC* (HIDDEN 10/430). The one genuine cost is
observation size: 2837 tokens/frame vs 769 for ice. That is a cost issue, not a validity issue,
and it is precisely the regime where a learned perception module should pay (9046 chars
carrying ~10 bits).

## Rules, verified on the engine

- 28x28, black bg, no RNG (frames identical across seeds 1,2,3,7,42), no click handler
  (`click_changes_state: false`). Action set = `left,right,up,down,noop`.
- Static `rink` object: 484 lightblue cells, `x,y ∈ [3,24]` (verified constant over 300
  steps). One red `skater`, init `(0,0)`. Skater renders *over* the rink cell it occupies
  (invertible: rink is static and complete).
- Off the ice, an arrow walks 1 cell. Pressing an arrow with `slide=="none"` never starts a
  slide (the gate `(! (in slide (list "none" OPP)))` blocks it).
- Sliding starts **only** via the entry rule: stepping from a non-rink cell onto a rink cell
  sets `slide = prevSlide` = the arrow just pressed. Verified the entry rule *never* fires on a
  noop (exhaustive over 2324 reachable states).
- While sliding, position moves **2 cells/tick from the previous position**; the arrow's
  1-cell walk is overwritten. The slide stops the tick after the previous position was off the
  rink → the skater rests exactly 1–2 cells outside the rink.
- **No 180° reversal** mid-slide (`left` while sliding right is a pure no-op, frame-identical
  to `noop`). **Perpendicular turns work**: sliding right at `(9,3)`, `up` → `(9,1)` (2 cells,
  from the *previous* position). 3696 perpendicular turns reachable.
- **Re-entry reversal** (verified): parked at `(25,3)` after crossing, `left` → `(24,3)`
  re-enters and slides back `22,20,…,2`.
- **The skater can never rest on the ice** — 0 of 2324 reachable states are at rest on a rink
  cell. All rest states are absorbing.

## Concern 1 — off-grid (REAL, fully fixable)

- Only *walking* leaves the grid, and only from the border pointing outward. Sliding provably
  cannot: max excursion is 2 cells past the rink (`[2,26]`), inside `[0,27]`.
- While off-grid the rendered frame is **byte-identical for every off-grid position** (484
  lightblue, no red; verified at x = −1,−2,−3,−5,−9). Concrete forward alias: `(blank frame,
  right)` → red at `(0,0)` from x=−1, still blank from x=−3.
- Recoverable in the engine (9 x `right` from x=−9 restores it), but the position is unbounded
  hidden state from frames alone.
- **Rate:** unguarded 20 % noop / 80 % arrow, seeds 1–20 x 100 steps: **20/20 drives go
  off-grid**, first fall at step 1–14 (median 2), **78.4 off-grid ticks per 100** (max 100).
- **Guard:** reject an arrow that would move the red cell out of `[0,28)²` — computable from
  the rendered frame alone, no privileged state. Measured: **0/20 drives, 0 off-grid ticks**,
  and on-ice occupancy triples (7.4 → 27.6 ticks/100). **Exhaustive proof:** BFS closure under
  this guard = 2324 states, **0 escapes**.

## Concern 2 — observation size (REAL, cost only)

| game | grid | chars | tokens (o200k) |
|---|---|---|---|
| rink | 28 | 9046 | **2837** |
| QQM74 (current largest) | 21 | 4010 | 1324 |
| BT3GB/ice | 16 | 2329 | 769 |
| N2NTD | 12 | 1354 | 442 |

Curated planning prompts carry **no history** (`eval_curated_plan.py`, CONTEXT_K=0) — current
+ goal grid = **~5.7 k tokens**, trivially within budget (`PLAN_CAP=50`). Online MPC replans
<=20x → ~114 k tokens/problem/arm, ~2x the current worst. Learning-side ID also uses the K=0
path for `raw_mode`, so 2 frames.

**One real config trap:** `--reflect-raw-prefix` defaults to 1500 chars — that is the top ~4
rows of a rink frame, so the P-proposer's orientation hint would never contain the skater.
Raise it (~9200) and drop `--reflect-max-failures` to 4 to stay cost-neutral.

## Concern 3 — visibility / hidden state (NOT a blocker)

Measured, 4 seeds x 30 branch points, 5 actions x h=3:

| policy | arrow vis h1 | distinct/5 | noop-change | off-grid ticks |
|---|---|---|---|---|
| unguarded (= atlas probe) | 0.14 | 1.53 | 0.09 | 110/120 |
| guarded | **0.99** | **4.83** | 0.29 | 0/120 |
| ice-seeking (guarded) | 0.71 | 3.80 | **0.80** | 0/120 |

On-ice `distinct = 3.00` exactly: `{noop, left, right}` alias while sliding horizontally;
`up`/`down` are unique.

**Markov depth:** over 4000 logged transitions, a K=1 key is ambiguous **495/1886 (26.2 %)** —
e.g. `((3,3), noop)` → `(1,3)` / `(3,5)` / `(5,3)`; at **K=2 it is 0/3028**. Our pipeline runs
`context_k=9`, so rink is fully Markov in-window. This is *better* than the 17/23 catalogue
games whose HIDDEN latents are unbounded.

**Credited-ID ceiling** (the number to report against): **0.682** over all 2324 states (0.600
on-ice, 0.998 off-ice); 0.688 on a realistic ice-seeking drive. Healthy, non-vacuous.

## Concern 4 — goal expressiveness (STRONG)

All plans below were found by BFS on a python mirror of the .sexp (validated 0 mismatches /
3000 engine transitions) and then **re-executed in the interpreter**: 8/8 end positions match,
7/8 are absorbing under +10 noops (the exception is "enter the ice", which is transient by
definition). Exact-frame goals are therefore well-posed: deterministic, QUIET-AT-RESET, and
rest frames do not stamp the tick (no free-running clock).

Starts: **A** = reset `(0,0)`; **B** = `down x3, right x2` → `(2,3)`; **C** = `down x10,
right x2` → `(2,10)`.

| tier | NL goal | start | h | floor@h | floor@50 |
|---|---|---|---|---|---|
| L1 | "step onto the ice" | B | 1 | 0.24 | 0.68 |
| L1 | "ride across and stop on the black margin to the right" | B | 12 | 0.00 | 0.08 |
| L2 | "enter from the left, then leave through the TOP margin" | C | 5 | 0.04 | 0.32 |
| L2 | "stop exactly at (25,10)" | C | 12 | 0.00 | 0.00 |
| L3 | "stop exactly at (11,1)" (one mid-slide turn) | C | 10 | 0.00 | 0.00 |
| L3 | "leave through the BOTTOM margin" | A | 17 | 0.00 | 0.00 |
| L4 | "stop exactly at (26,26)" (two turns) | A | 31 | 0.00 | 0.00 |
| L4 | "stop exactly at (27,0)" | A | 22 | 0.00 | 0.00 |
| — | "come to rest ON an ice cell" | A | ∞ | — | — |

**Dumbest-route check passes for L4:** the shortest ice-free walk `(0,0)→(26,26)` is **52
actions > PLAN_CAP 50**, so the ice mechanic is *necessary*, not a shortcut. Plans are
noop-heavy by construction (the slide needs ticks), so predicate-based greedy compression is
well-behaved.

## Required fixes

1. ~~Copy the program into the harness dir~~ — done 2026-08-23 via `autumn_programs/` +
   `tools/install_autumn_programs.py`.
2. **On-grid guard in the automatic sampler** —
   `offline_learning/scripts/eval_multistep_fd_plan.py::generate_drive` (and `random_plan`): a
   per-game `(grid, verb) → bool` filter with resampling. **~15 lines.** Optional
   `--on-grid-guard` flag on `offline_learning/autumn_drive.py` (**~20 lines**); hand-authored
   `--actions` lists need no code, just discipline.
3. ~~Register the game in `offline_learning/clean_sweep.py::GAMES`~~ — done 2026-08-23
   (`"rink": ("left,right,up,down,noop", False, 120, 6)`). Add to
   `offline_learning/human_replay.py::GAMES` only if human/test pools are built.
4. **Curated ladder** — a `_problems()` block in `offline_learning/curated_plan.py` using the 8
   problems above (no hidden-state tracker needed; dedup on the 2-frame key). **~60 lines.**
5. **NL checkers** — `offline_learning/nl_goals.py`. **~40 lines.** *Authoring note:* every
   checker must require the red cell to be **present**; "the skater has left the rink" is
   silently satisfied by an off-grid (invisible) skater.
6. **Config** — `--reflect-raw-prefix ≈9200 --reflect-max-failures 4` for rink.

## Surprises in the .sexp

- `(= slide (initnext "none" (prev "slide")))` and `(prev "prevSlide")` pass **string
  literals** to `prev`. It happens to work, but it is not the idiomatic `(prev slide)`.
- `prevSlide` is a dead latent: it is only read by the entry rule, where it always equals the
  arrow pressed that same tick (proved exhaustively — the entry rule never fires on a noop).
- The 180°-reversal block is implemented by the gate list, yet reversal is still achievable via
  **re-entry from the overshoot cell** — almost certainly unintended, and it makes the game
  strictly more interesting.
- The bounce-back means the ejection cell (`x=25`) is a legitimate "wall" the player can use;
  combined with perpendicular turns, the reachable state space is 2324 states — large enough
  for L4 composition, small enough to BFS exhaustively.
