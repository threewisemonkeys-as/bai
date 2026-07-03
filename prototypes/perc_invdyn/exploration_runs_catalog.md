# Good-exploration stepwise_eb_learn runs (per game)

Catalog of `stepwise_eb_learn.py` runs whose first ≤50 steps explore the map
well — i.e. high distinct-state coverage, most actions actually change the grid,
and no long "stuck" stretch (consecutive identical states). Intended as a source
of clean trajectories for inverse-dynamics / forward-objective perception work.

Generated 2026-06-15 by scanning `logs/**/*stepwise_eb_learn/episode_0/trajectory_buffer.json`.

## How to read the metrics
Computed over the first ≤50 steps of `episode_0`:
- **dStates** — distinct grid states (hash of bracketed grid lines in `obs_text`).
- **chng** — transitions where the grid actually changed (action had a visible effect).
- **stuck** — longest run of consecutive identical states. Large `stuck` = pathological (agent frozen / clicking dead coords).
- **cellΔ** — avg # of grid cells that changed per step (ARC-AGI-3 only; AutumnBench obs isn't a bracketed grid so this reads 0 and is not meaningful there).

A clean exploration profile looks like: `dStates ≈ steps`, `chng ≈ steps-1`, `stuck = 1`.

## Multi-game batches
A single invocation only plays one game (extra `episode_*` = repeats of the same
game). These batch dirs launch one run per game, covering all 5 ARC-AGI-3 games:
- `logs/matrix_v6/arc5_noperc_50steps/` (06-04) — ft09, ls20, sp80, tn36, vc33
- `logs/matrix_v5/v5_50steps/` (06-03) — ft09, ls20, sp80, tn36, vc33
- `logs/dev/may29/20260529-114550/` (05-29) — same 5
- `logs/arc_parallel/20260528-184557/` (05-28) — same 5

AutumnBench batches: `logs/dynamics_full/` (7WWW9, ADA85, DQ8GC), `logs/seed_autumn/` (7WWW9, DQ8GC, ice).

## ARC-AGI-3 — recommended run per game

| Game | Recommended run (relative to `logs/`) | Steps | dStates | chng | stuck | cellΔ |
|------|----------------------------------------|-------|---------|------|-------|-------|
| ls20 | `matrix_v6/arc5_noperc_50steps/eb_learn__arc_agi__gemini-2p5-flash__ls20/2026-06-04_11-49-26_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn` | 50 | 50 | 49 | 1 | 38.4 |
| vc33 | `matrix_v6/arc5_noperc_50steps/eb_learn__arc_agi__gemini-2p5-flash__vc33/2026-06-04_11-49-26_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn` | 50 | 50 | 49 | 1 | 1.3 |
| tn36 | `matrix_v5/v5_50steps/eb_learn__arc_agi__gemini-2p5-flash__tn36/2026-06-03_16-24-58_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn` | 50 | 50 | 49 | 1 | 1.0 |
| sp80 | `matrix_v5/v5_50steps/eb_learn__arc_agi__gemini-2p5-flash__sp80/2026-06-03_16-24-57_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn` | 44 | 44 | 43 | 1 | 89.2 |
| ft09 | `matrix_v6/arc5_noperc_50steps/eb_learn__arc_agi__gemini-2p5-flash__ft09/2026-06-04_11-49-26_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn` | 42 | 33 | 32 | 7 | 29.7 |

Notes:
- **ls20 / vc33 / tn36 / sp80** all have near-ideal exploration profiles. tn36/vc33
  have tiny per-step cell diffs (cursor-style moves); sp80 even earned reward 0.1.
- **ft09 is the weak spot.** It's a coordinate-click game (`ACTION6 x= y=`) so most
  runs are dominated by no-op clicks. There is **no clean 50-step ft09 run**. Best
  alternatives: the `matrix_v6` run above (42 steps, 33 distinct states, one 7-step
  stall), or `arc_parallel/20260528-184557/eb_learn__arc_agi__gemini-2p5-flash__ft09/2026-05-28_18-46-01_...`
  (only 34 steps but nearly no-op-free, stuck=2). If a genuinely good ~50-step
  ft09 exploration trajectory is needed, it should be re-run.

### ARC-AGI-3 — pathological runs to AVOID
- `dev/may29/20260529-114550/...__ft09` — 50 steps, stuck=21.
- `matrix_v5/v5_50steps/...__ft09` — 50 steps, stuck=14.
- `arc_parallel/20260528-184557/...__tn36` — 50 steps, stuck=50 (fully frozen).
- `arc_parallel/20260528-184557/...__vc33` — 50 steps, stuck=50 (fully frozen).

## AutumnBench — recommended run per game
(cellΔ unavailable for Autumn; judged on dStates/chng/stuck. None reward-bearing.)

| Game  | Recommended run (relative to `logs/`) | Steps | dStates | chng | stuck |
|-------|----------------------------------------|-------|---------|------|-------|
| ice   | `seed_autumn/ice/2026-06-05_14-07-07_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn` | 50 | 16 | 42 | 2 |
| DQ8GC | `seed_autumn/DQ8GC/2026-06-05_14-07-07_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn` | 50 | 30 | 38 | 5 |
| 7WWW9 | `seed_autumn/7WWW9/2026-06-05_14-07-07_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn` | 50 | 14 | 23 | 8 (weak — others cycle ≤4 states) |
| ADA85 | `dynamics_full/ADA85/2026-06-08_12-49-01_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn` | 50 | 6 | 29 | 6 (only viable alt has a 30-step stall) |
