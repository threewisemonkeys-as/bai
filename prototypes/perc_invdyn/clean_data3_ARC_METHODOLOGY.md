# clean_data3 — REGENERATING + curating ARC-AGI-3 games

Read `clean_data3_METHODOLOGY.md` (slicing/objectives) and `clean_data3_REGEN_METHODOLOGY.md`
(regeneration via a live env) FIRST. This file covers what's DIFFERENT for ARC games.

The goal (per the request): produce a high-quality regenerated trajectory and a curated train
set where **every scored transition exposes a core game mechanic that is learnable through the
inverse-prediction (ID) or forward-prediction (FD) objective** — no filler.

## How ARC differs from autumn

1. **Observation format.** ARC obs contain integer grids in `<grid_N>...</grid_N>` blocks
   (cell values 0..15 are ARC color indices), NOT autumn `[["color",...]]` arrays. The shared
   helpers already auto-detect this — `clean_data3_tools.grid_at/classify/dump_transitions/
   verify_pool` work on ARC unchanged (cell values print as their integer index).
2. **Driver = arc_drive.py** (not autumn_drive.py):
   ```
   uv run python prototypes/perc_invdyn/arc_drive.py <game> <OUTDIR> --actions "<csv>"
   ```
   - `<game>` is the lowercase code (e.g. `ls20`). Actions: `ACTION1..ACTION5` pass through;
     `click_X_Y` becomes `ACTION6 x=X y=Y` (the complex/click action). It prints an ASCII grid
     (chars `.123456789ABCDEF`) + legend + available-actions per step and writes a
     load_transitions-compatible trajectory.csv. Deterministic — same actions reproduce states.
   - Only the whitelisted actions for the game are meaningful targets (see per-game config).
3. **ARC caveats to design around:**
   - **Camera scroll / large grids (64x64).** In nav games the whole viewport can shift when the
     player moves, so `classify` reports a big diff; the real signal is the player object's
     RELATIVE motion. Read the ASCII to confirm what actually moved. A move is still ID-learnable
     (recover direction from the player's displacement) and FD-learnable.
   - **Level transitions.** When a level/puzzle is solved the entire board can change at once.
     That transition is NOT a clean single-mechanic target — don't make it a scored target
     (slice around it), or include it only as a deliberate "level-advance" mechanic if the
     dynamics.txt treats it as one.
   - **`keep_action_params`:** for click-only games (ACTION6) the click LOCATION is the label, so
     ID must recover x,y from WHERE the toggle/effect happened — make sure clicks land in places
     where the effect reveals the location.

## Per-transition bar (enforce this)

For each scored target ask: does the action's effect show in `X_t -> X_t+1` such that
- **ID:** the action (incl. ACTION6 x,y) is recoverable from the change, AND/OR
- **FD:** `X_t+1` is predictable from `X_t` + the rule.
If a transition changes nothing decision-relevant, or its action is unrecoverable AND its next
state isn't rule-predictable, it's filler — drop it or replace with a contrastive case.
Cover each mechanic AND its near-miss negative (move-succeeds vs move-blocked-by-wall;
click-toggles vs click-no-effect; etc.).

## Workflow (same shape as the autumn regen)

1. Read `clean_data2/<game>/dynamics.txt` (the mechanics description) and inspect the original:
   `import clean_data3_tools as T; T.dump_transitions('<game>','<whitelist>')`.
2. Explore with arc_drive (to /tmp), reading the ASCII, until you understand every mechanic and
   can drive the game INTO each one (including the ones the original never showed — e.g. an
   untried direction, a wall-blocked move, a win/level-advance, every distinct click outcome).
3. Author the final trajectory -> `clean_data3/<game>/train_regen/` (arc_drive OUTDIR).
4. Filmstrip: `build_dataset_viz.py clean_data3/<game>/train_regen --out .../train_regen/viz.html`
   (it renders ARC grids with the real palette).
5. Curate `clean_data3/<game>/train/episode_*` = short contiguous slices OF train_regen, every
   target a mechanic under ID/FD, contrastive, ~18-22 targets (state which mechanic each covers).
6. Verify: `T.verify_pool('prototypes/perc_invdyn/clean_data3/<game>/train','<whitelist>')`.
7. Selection viz: `uv run python prototypes/perc_invdyn/build_clean_data3_viz.py <game>`.
8. Copy `dynamics.txt` verbatim if not present; leave `test/` the original verbatim copy.
   Write `COVERAGE.md` (mechanics list; mechanic×{ID,FD} table; the gap the original had; slice
   table with the mechanic each target exposes + contrastive negatives; any inherent limits).

## Per-game config (the 5 stepwise-eb-learn ARC games)

| game | whitelist | keep_action_params | what it is (from the frontier notes) |
|------|-----------|--------------------|--------------------------------------|
| ls20 | `ACTION1,ACTION2,ACTION3,ACTION4` | FALSE | 4-direction maze nav; gray=walls block; camera follows player; goal/levels |
| sp80 | `ACTION1,ACTION2,ACTION3,ACTION4,ACTION5,ACTION6` | TRUE | mixed move + click game (read dynamics.txt/explore) |
| tn36 | `ACTION6` | TRUE | click-only (read dynamics.txt/explore) |
| vc33 | `ACTION6` | TRUE | click-only (read dynamics.txt/explore) |
| ft09 | `ACTION6` | TRUE | click = Lights-Out toggle; clicks flip a cell pattern |

Reference: `clean_data3/bt3gb/` and `clean_data3/7xf97/` are finished autumn regen examples
(same recipe). dynamics.txt for each ARC game is the ground-truth mechanics description.
