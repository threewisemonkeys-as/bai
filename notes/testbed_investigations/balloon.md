# balloon — testbed assessment (2026-08-23)

Engine-level investigation of whether `balloon` (zip-sourced, 16x16, skyblue background) is a
valid testbed, prompted by the 55-game atlas rating it Tier B ("36-cell sprite, sky-blue
background, 15-cell click surface") and by the known black-background bug that broke mario's
planning goal. Every number verified on the interpreter (render after every step, seeds >= 1).
Probe scripts: `scripts/testbed_probes/balloon/`.

## VERDICT: USABLE WITH FIXES (small, mostly generic ones) — but only with authored drives and a click-at-rest discipline

The count-threshold physics is real, clean, fully deterministic, seed-independent, and
**zero-aliased** (0 of 2,195 `(frame, action)` pairs over 1,290 distinct states had two
successors). That makes it a genuinely good "abstract discovery" testbed: the rule is a
*count* over a moving region, not a pixel table, and the exact-frame planning ladder bottoms
out in two absorbing states. Three things must be fixed first: (1) three real
black-background biters, (2) random exploration produces literally nothing (95% static frames,
0 frames ever reach the threshold), (3) a click-during-motion quirk that strands rocks and can
hide them by stacking. All three are avoidable/fixable; none is intrinsic to the game.

## Verified rules (engine, seeds 1–99, all identical)

- 16x16, `(= background "skyblue")`. `interpreter.get_background()` returns `"skyblue"`.
  `render_all()` emits **only object entries + `GRID_SIZE`** — no background entry; empty
  cells are simply absent.
- Sprite = **35 cells** (not 36): 21 `mediumpurple` (rows −2..+2), 5 `tan`, 9 `brown` basket.
  Column fixed at **x = 7**; there is no horizontal motion.
- **Arrow keys are exact no-ops** — `up/down/left/right` produce byte-identical frames to
  `noop` (4/4 verified).
- Origin y ∈ **[2, 7]** only — 6 positions, 5 rows of travel (sprite is 11 tall).
- **Click surface = 6 cells**, not 15: `x ∈ {6,7,8} x y ∈ {oy+6, oy+7}` (brute-forced all 256
  cells). The `in_baloon` rect is 15 cells but 9 are brown sprite and fail `isFreePos`. It
  **moves with the balloon**.
- Click on an empty basket cell adds a `gray` rock at exactly that cell; click on a rock
  removes it.
- Threshold: count rocks in `x ∈ [5,9] x y ∈ [oy, oy+7]` (a 5x8 rect, **larger than the
  basket**). >=3 → sink 1/tick; <3 → rise 1/tick.
- **Settle = 5 ticks** each way. Absorbing: top (`oy=2`, <3 rocks) and bottom (`oy=7`, >=3).
  Verified stable over 25 noops. Nothing hovers mid-air — the only stationary rows are 2 and 7.
- Rocks ride the basket on clean (noop) ticks; a rock removed at the bottom reverses the
  balloon on the next tick (1-tick lag: the click tick itself still moves in the old
  direction).
- Nothing leaves the grid; no stochasticity; `(print num_contained)` is silenced by
  `set_verbose(False)`.

## Background audit (only the biters; ~30 other hits verified harmless)

| file:line | bites balloon? | fix |
|---|---|---|
| `MARAProtocol/python_examples/autumnbench/concrete_envs.py:476-481` (CD) and `:618-623` (Planning), image mode — `render_grid_matplotlib(...)` called **without** `background_color=` | **YES** in `--f-image`/image mode: black sky in the image, `skyblue` in the text grid | add `background_color=self.interpreter.get_background(),` (1 line each). The text twin at `:609` and the interactive image path at `:138` already do it. |
| `MARAProtocol/.../env_utils.py:121,147,178,312` — `background_color: str = "black"` defaults | root enabler of the above and of the historical mario goal bug | make the parameter **required** so omission is a `TypeError`, not a wrong grid |
| `offline_learning/mechanics.py:36-42` `_BG` table + `:269` unguarded `_BG[game]`; `mechanics_rules.py:33-35` | **YES** — hard `KeyError` on `balloon`; `coverage_exam.py:86` uses `.get(...,set())` so it silently treats all 256 cells as foreground | replace the literal table with a memoised `get_background()` / `GRID_SIZE` lookup off the installed `.sexp` |
| `offline_learning/invdyn_core.py:2702-2706` `_SCHEMA_AUTUMN` — instruction says "dominant/background colour" (good) but both examples and the whole vocabulary list lead with `black`; balloon's palette absent. `DEFAULT_KNOWLEDGE:252-256` makes no colour claim. | **YES, silently.** 8 of 9 shipped learned perception modules in `sweep_results/clean_sweep_gepa_padded_ctxk9/` hardcode `== "black"`; only `bt3gb_seed1/...py:71` uses the modal colour. A `!= "black"` filter emits all 256 cells → blows the ~2000-char perception budget → truncation → the ID signal collapses and it looks like "balloon is hard". | de-bias the examples; inject `"Empty space in this world renders as {background}."` into `DEFAULT_KNOWLEDGE` from `get_background()`. Interim: pass `offline_learning/scripts/start_perc_autumn_robust.py:36-40` (already modal) as `--start-perception`. Modal detection is safe here: 221/256 skyblue at reset, sprite bounded at 35 + <=15 rocks. |
| `offline_learning/autumn_drive.py:35-38` `COLORS` map and `:71` `if c != "black"` | cosmetic but blocks the very workflow balloon needs — every cell prints `?` and the legend lists 221 background cells as objects | modal background → `.`, auto-assign glyphs |

**Clean by construction:** `autumn_env.py:294,318`, `offline_learning/curated_plan.py:90,97`
(`self.bg = self.it.get_background()`), `program_runtime.py:417`, `game_profile.py:63`, all
`scripts/gen_*_seed1.py`. `rexpure_optimize.py`, `worldcoder_optimize.py`, `nl_goals.py`,
`forward_objective.py` have **zero** background hardcodes. The historical patcher is at
`cc_autumn/autumn-code/tools/fix_planning_goals.py` (not `tools/`); balloon has **no**
`balloon_planning.json`/`_mfp`/`_cd`, so that bug cannot recur — but it also means only
`task_type="interactive"` + curated goals work today.

**End-to-end check passed:** driving `balloon.sexp` through
`AutumnBenchEnvWrapper(task_type="interactive")` renders `[["skyblue", ...]]` correctly, and
`click 13 7` (ROW COL) lands a rock at `(col 7, row 13)` — no transposition bug.

## Other concerns, measured

**Data generation is the real problem.** 20 random drives x 60 steps (seeds 1–20): **12/607
random clicks hit (2.0%)**, 114/1200 steps changed the frame (9.5%), and only **11 unique
states** total. A 120-step flat-random drive: **114/120 static, 0 frames ever reach >=3 rocks**
— the threshold rule is never once exercised. An object-targeted policy (click a cell at
`origin + (dx∈{−1,0,1}, dy∈{6,7})`) hits **100%** and spans rock counts 0–11. An authored
58-step drive (click only while at rest) gives 43% static and 42 frames at >=3 rocks. Fix:
`offline_learning/clean_sweep.py` already has `"balloon": ("noop,click", True, 120, 6)`; the
drive must be authored via `autumn_drive.py --actions`, using the recipe *rise 5 noops → 3
clicks at `(6/7/8, oy+6)` → 5 noops → remove → 5 noops*. Do **not** add a random-click
generator (`gen_clickmove.py:62` is the wrong template — hardwired to DQ8GC and to `"black"`;
copy `curated_plan.Sim` instead).

**Two genuine quirks (both avoidable, both worth documenting in the drive spec):**
1. *Click ticks freeze rock motion.* On a tick where a rock is added or removed, the balloon
   moves but existing rocks do not. A rock clicked into the **lower** interior row while rising
   ends up co-located with the basket floor and is **frozen forever** — the balloon flies off
   and leaves it floating in mid-air. Stranded rocks still count while inside the 5x8 rect, so
   the intuitive rule ("rocks in the basket") is wrong in those configurations.
2. *Rocks can stack invisibly.* 268/1200 frames (22%) under mid-flight clicking had more rock
   objects than visible gray cells — two rocks in one cell. This is **real hidden state**: 1 of
   1,769 noop transitions (0.06%) had a direction that the visible gray count cannot predict
   (2 visible, 3 actual → sank).

Both vanish entirely if clicks happen only while the balloon is at rest (top or bottom), which
is exactly when it is stationary. Recommend adding that as a drive-authoring rule and as a
constraint on curated plan alphabets.

## NL goal ladder (plans verified on the engine; floors = 25 rollouts x 50 actions, any-step)

| | goal | start | plan | len | flat-random floor | typed-random floor |
|---|---|---|---|---|---|---|
| L1 | rise until the balloon touches the ceiling | init | 5 noops | 5 | **1.00** | **1.00** |
| L2 | put exactly one rock in the basket and keep it | init | `click 13 8` | 1 | 0.72 | 0.08 |
| L3 | land the basket on the ground | 6 noops (at top) | 3 clicks + 5 noops | 8 | 0.04 | 0.00 |
| L4a | *exact frame:* on the ground, rocks at (6,13),(7,13),(8,13) | at top | 3 clicks + 5 noops | 8 | 0.00 | 0.00 |
| L4b | land, then lighten and return to the ceiling carrying 2 rocks | at top | 3 clicks + 5 noops + 1 click + 5 noops | 14 | 0.00 | 0.00 |
| L5 | sink to the ground, then return to the ceiling with an empty basket | at top | 3 clicks + 5 noops + 3 clicks + 5 noops | 16 | 0.00 | 0.00 |

**Drop L1** — the floor is 1.00; passive drift solves it. Use L2 as the L1 rung (typed floor
0.08), L3 as L2, L4a as the exact-frame L3, and L4b/L5 as L4 (true composition: add → wait →
remove → wait, and the agent must track the *moving* click surface). L4a is absorbing, so it
satisfies the curated-set requirement.

## Required fixes, effort

1. `_SCHEMA_AUTUMN` + `DEFAULT_KNOWLEDGE` background injection (`invdyn_core.py:252-256,
   2702-2706`) — ~10 lines, **highest leverage**; interim workaround exists
   (`--start-perception start_perc_autumn_robust.py`).
2. Two missing `background_color=` kwargs (`concrete_envs.py:481, 622`) — 2 lines.
3. Make `background_color` required in `env_utils.py:121,147,178,312` — 4 lines, converts the
   whole class from silent-wrong to loud-fail.
4. Derive `_BG`/`SIZE` from the interpreter (`mechanics.py:36-42,269`;
   `mechanics_rules.py:33-35`) — needed before balloon enters coverage/compose.
5. `autumn_drive.py:35-38,71` modal-background ASCII — needed for the authoring workflow itself.
6. Author the drive + the L2–L5 curated ladder in `curated_plan.py` (background-safe already).

## Surprises in the .sexp

- The counting rect (`origin+(−2,0)` … `+(2,7)`) is **not** the basket (`+(−2,6)` … `+(2,8)`)
  — it is a 5x8 column covering the balloon body too. The abstract rule is "gray cells in the
  5-wide column below the balloon's mid-row", not "rocks in the basket".
- `weight` is a `Bool` that looks like hidden state but is a pure function of the visible rock
  count (modulo the stacking case) — the note's "hidden" tag is over-stated.
- The rise branch carries an `(intersects obj (nextSolid obj))` support test that the sink
  branch lacks — asymmetric physics; a rock floating in the basket will not rise but will sink.
- `(print num_contained)` on every tick (silenced by `set_verbose(False)`, no stdout pollution
  measured).
