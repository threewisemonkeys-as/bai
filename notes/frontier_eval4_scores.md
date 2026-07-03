# gepa_frontier eval (run `_frontier_eval4`) — learned dynamics scored vs. ground truth

**Date:** 2026-06-17
**Mode:** `question_scoring_method=gepa_frontier` (new experience-grounded B/P learning in `stepwise_eb_learn.py`)
**Task model / reflection model:** `google/gemini-2.5-flash`
**Steps:** 30 each · **Forward scoring:** textdiff enabled (`frontier_fd_scorer=textdiff`, `frontier_fd_weight=0.5`)
**Other knobs:** `frontier_relearn_interval=8`, `frontier_min_buffer=8`, `frontier_train_n=8`, `frontier_val_n=4`, `frontier_test_n=4`, `frontier_max_metric_calls=40`, `frontier_concurrency=4`
**Status:** all 4 runs exited 0 (`logs/_frontier_eval4/_status.txt`)

## Summary

| Game | Type | id_acc (best) | Perception | **Score /10** |
|------|------|--------------|-----------|---------------|
| ls20 (ARC)   | 4-dir maze nav      | 0.5  | empty | **5** |
| ft09 (ARC)   | pure-click Lights-Out | 0.0  | empty | **4** |
| ice (autumn) | day/night + water   | 0.75 | empty | **3** |
| DQ8GC (autumn)| infection spread   | 0.25 | empty | **3** |

## Per-game

### ls20 — 5/10
**Correct:** player identified, grid model, `ACTION1=North/up`, `ACTION3=West/left`, gray = walls that block movement, camera scrolls to follow the player, "move 1 cell if path is clear".
**Wrong/missing:** `ACTION4` labeled "no effect" (it's East — was simply wall-blocked in the sampled transitions); `ACTION2` (down) never tried; step-budget decrement, lives/respawn, patrolling hazards, fog mechanic, and the real win condition all missing (only learned the negative "goal is not the blue square at top").

### ft09 — 4/10
**Correct:** identified a click-only game over multiple 3×3 grids, the highlighted/selected grid, and the blue↔red color toggle.
**Wrong/missing:** modeled clicks as purely local — missed the 3×3 neighborhood propagation (Lights-Out pattern), the flash penalty on clicking empty tiles, and the win condition.

### ice — 3/10
**Correct:** exact initial state (2×2 gold "sun" + 3 gray "cloud" cells), reset behavior, and a precise model of clicking the 2×2 block flipping gold↔gray (which *is* the day/night toggle).
**Wrong/missing:** entirely missed cloud movement (`left`/`right`), rain (`down`), and water freezing to ice (`lightblue`) — the game's namesake mechanic. Everything framed as abstract grid-color flips with no semantics, because the agent only ever clicked.

### DQ8GC — 3/10
**Correct:** exact initial state (darkgreen active particle at (2,2) + 4 gray particles), black = inert, clicking a gray particle → darkgreen (selection), and it *sensed* autonomous dynamics ("changes not permanent", "hidden conditions").
**Wrong/missing:** never modeled the infection-spreads-to-adjacent core mechanic (only sensed it as "hidden state"), missed 4-directional movement entirely, and some muddled gray↔black claims.

## Cross-cutting findings

1. **Perception came out empty in all four runs.** With the small budget (~40 metric calls, 4–8 train transitions), GEPA never validated a perception that beat the empty seed, so the best-val candidate kept `P=""` and only `world_knowledge` evolved. Growing a real P would need more steps / higher `frontier_max_metric_calls`.
2. **Both autumn agents fixated on `click`** and never moved, despite `left/right/up/down` being available — so they only learned the click-subset of each game's dynamics. The discriminating-experiment loop did not drive directional exploration.
3. **Coordinate-click handling worked** end-to-end (the point of the run): autumn `click R C` and ARC `ACTION6 x y` both produced distinct-coordinate pools and hard-negative MCQ choice sets.
4. **ft09 click inverse-dynamics scored id_acc=0.0** — the color-cycle effect was too subtle (and clicks too clustered) to recover the click coordinate from features; the **textdiff forward term carried that run's learning** (validates enabling it).
5. **Gate bug found & fixed mid-run:** relearn was gated on ≥2 distinct *verbs*, which silently skipped all learning for pure-click ft09 (one verb). Now gated on ≥2 distinct *actions* (coordinates count). ft09 was re-run as `arc_ft09_fix` after the fix.

## Artifacts

- Run dirs: `logs/_frontier_eval4/hydra_<game>/logs/_frontier_eval4/<game>/<timestamp>/`
  (`<game>` ∈ `arc_ls20`, `arc_ft09_fix`, `autumn_ice`, `autumn_DQ8GC`)
- Per game: `episode_0/frontier.json` (ranked B/P candidates + metric), `episode_0/beliefs.txt`, per-step `step_NNN/frontier.json`.
- Logs: `logs/_frontier_eval4/<game>.log`; status: `logs/_frontier_eval4/_status.txt`.
