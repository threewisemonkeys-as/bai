# ARC clean_data2 authoring process (per game)

This mirrors the AutumnBench clean_data2 process, adapted for ARC-AGI-3.
Everything runs **locally/offline** — the game source is on disk and the engine
(`arcengine`) executes it with no network. Always set `OPERATION_MODE=offline`.

You are authoring clean data for **one** game. Produce three deliverables under
`prototypes/perc_invdyn/clean_data2/<PREFIX>/`:

```
<PREFIX>/dynamics.txt              # reverse-engineered true mechanics
<PREFIX>/train/episode_0/trajectory.csv
<PREFIX>/test/episode_0/trajectory.csv
<PREFIX>/train/viz.html            # filmstrip for review
<PREFIX>/test/viz.html
```

`<PREFIX>` is the 4-char game prefix (e.g. `ls20`, `ft09`).

All commands below are run from the bai repo root: `/home/ays57/bai`.

---

## Phase 1 — Discover the true dynamics -> dynamics.txt

1. **Read the source.** The game program is real, readable Python at
   `environment_files/<PREFIX>/<HASH>/<PREFIX>.py`. This is the ground truth
   (the ARC analog of an Autumn `.sexp`). Read it fully. Find:
   - grid size & how many `<grid_N>` layers are emitted,
   - the objects/sprites and what each color index means,
   - what each action (ACTION1..ACTION7) does — movement, rotate, select, shoot, etc.,
   - ACTION6 click semantics (it carries `x`,`y`),
   - how the set of *available actions* changes by state,
   - the level / score / win / game-over logic.

2. **Confirm empirically by playing it locally** (free, offline). Two tools:

   - ASCII + available-actions per step (fast iteration):
     ```
     OPERATION_MODE=offline uv run python prototypes/perc_invdyn/arc_drive.py \
       <GAME_ID> /tmp/scratch/<PREFIX>_probe --actions ACTION1,ACTION2,click_30_20
     ```
     It prints the 64x64 grid, the reward, term flag, and `avail` (available
     actions) after every step, and writes a trajectory.csv (you can ignore the
     probe csv — it's just for inspection).

   - True-palette PNGs (to actually SEE the puzzle — ASCII hides structure):
     ```
     OPERATION_MODE=offline uv run python prototypes/perc_invdyn/render_arc.py \
       <GAME_ID> /tmp/scratch/<PREFIX>_png --actions ACTION1,ACTION2,click_30_20
     ```
     Then Read the PNGs (step00_reset.png, step01_*.png, ...) to inspect frames.

   Iterate: form a hypothesis from the source, test it with a probe, refine.

3. **Write `dynamics.txt`** in the same prose style as the reference example
   `prototypes/perc_invdyn/clean_data/dq8gc/dynamics.txt`. Cover, with concrete
   coordinates/colors: GRID & LAYERS, OBJECTS/PALETTE, each ACTION's effect,
   CLICK semantics + coordinate convention, AVAILABLE-ACTION transitions
   (which actions are offered in which states), LEVEL / GOAL structure, and
   WIN / GAME-OVER / SCORE / REWARD conditions. Be precise and empirical — state
   only what the source + your probes confirm.

### ARC specifics & gotchas
- Grids are 64x64, 16-color palette (index 0..15). Color 0 is usually background.
- **Coordinate convention: `click_X_Y` on the CLI -> `ACTION6 x=X y=Y`, where
  x = COLUMN, y = ROW.** (This is the OPPOSITE order from Autumn's row-major
  `click_ROW_COL`. Do not transpose by accident.)
- **Available actions are state-dependent.** Only emit actions that are in the
  current `avail` list — emitting an unavailable action is rejected/no-ops.
  Some games only ever offer `ACTION6` (click-only); some only movement.
- **No noop padding.** ARC state changes only on action (no passive clock). Each
  row's Action is taken FROM that row's Observation, producing the next row.
- An action may trigger a multi-layer animation cascade; only the final settled
  layer is rendered. That's expected.
- Reaching a WIN ends the episode (term=True). Keep the trajectory going up to
  but not necessarily past the first win/level transition.

---

## Phase 2 — Author DISTINCT train + test trajectories

Hand-pick two **different** action sequences that each demonstrate the game's
key mechanics. They must share mechanics but NOT be the same frames — different
routes, different click locations, the same mechanic shown from a different
state (see `prototypes/perc_invdyn/clean_data/notes.md` for the kind of
diversity wanted: "show approach from below too", "diverse click locations",
"include clicks, not just movement").

Guidance:
- Demonstrate every meaningful action at least once, plus at least one click if
  ACTION6 is available, plus a level/score transition or win if reachable.
- Keep them focused — roughly 12-40 steps each. No filler.
- Generate each with `arc_drive.py` writing to the train/test dirs:
  ```
  OPERATION_MODE=offline uv run python prototypes/perc_invdyn/arc_drive.py \
    <GAME_ID> prototypes/perc_invdyn/clean_data2/<PREFIX>/train \
    --actions <comma-separated-train-actions>

  OPERATION_MODE=offline uv run python prototypes/perc_invdyn/arc_drive.py \
    <GAME_ID> prototypes/perc_invdyn/clean_data2/<PREFIX>/test \
    --actions <comma-separated-test-actions>
  ```
  (arc_drive writes `<dir>/episode_0/trajectory.csv`.)

---

## Phase 3 — Validate

1. Build filmstrips and Read them / spot-check (they render the real ARC palette
   and show level/score per frame):
   ```
   OPERATION_MODE=offline uv run python prototypes/perc_invdyn/build_dataset_viz.py \
     prototypes/perc_invdyn/clean_data2/<PREFIX>/train \
     --out prototypes/perc_invdyn/clean_data2/<PREFIX>/train/viz.html

   OPERATION_MODE=offline uv run python prototypes/perc_invdyn/build_dataset_viz.py \
     prototypes/perc_invdyn/clean_data2/<PREFIX>/test \
     --out prototypes/perc_invdyn/clean_data2/<PREFIX>/test/viz.html
   ```
2. Walk through the trajectories (re-run render_arc.py on the exact action
   sequences and Read the PNGs) and confirm: every transition matches
   dynamics.txt; train != test framewise; both cover the key mechanics; any
   win/level transition actually fired.
3. If something is wrong, fix the action sequence and regenerate. Iterate until
   both trajectories are clean.

## Final report
Return a concise summary: the grid/objects, what each action does, the
level/win structure, the train and test action sequences you chose and what
each demonstrates, and any caveats or anomalies you hit.
