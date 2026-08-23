# clean_data3 — TEST50: large held-out test pools (~50 scored transitions per game)

Read `clean_data3_METHODOLOGY.md` (slicing recipe, objectives, verify_pool) and
`clean_data3_REGEN_METHODOLOGY.md` (autumn_drive workflow, hard-won driving lessons) first.
This file specifies the LARGE HELD-OUT TEST SET: `clean_data3/<game>/test50/`.

## Why

The GEPA sweeps (`logs/clean_sweep_gepa_cd3_*`) score learned beliefs/perception by
inverse-dynamics accuracy on a held-out test carve of only `--test-n 10` transitions from
the small original test trajectory. 10 items is far too noisy to (a) rank runs or (b) tell
WHICH dynamics a learned belief actually captures. TEST50 replaces this with a deliberate
~50-transition pool per game that exercises every core dynamic, contrastively.

## What to produce

```
clean_data3/<game>/test50/
  episode_0/trajectory.csv     # curated contiguous slices of freshly DRIVEN trajectories
  episode_1/trajectory.csv     # (verbatim rows; one slice per episode dir)
  ...
  viz.html                     # filmstrip of the whole pool (build_dataset_viz.py)
  TEST_COVERAGE.md             # dynamics -> target-pair coverage map (see below)
```

The SCORED POOL — what `verify_pool(<...>/test50, <whitelist>, context_k=9)` reports —
must be **exactly ~50 transitions** (50 +/- 2). The pool is consumed by
`gepa_optimize.py --test-run .../test50 --test-n 50` (balanced_split returns the whole
pool when pool size <= test-n), so the pool IS the test set: every pair you include gets
scored, every pair you exclude doesn't exist.

## Requirements on the pool

1. **Freshly driven, distinct from train.** Generate NEW trajectories with
   `autumn_drive.py <ENV_NAME> <outdir> --actions "..."` (seed 0, the default). Action
   sequences, object positions and state configurations must NOT replicate the train
   slices in `clean_data3/<game>/train/` (read them first) — this is a cross-trajectory
   generalization test. Same rules, different situations.
2. **Every core dynamic in `dynamics.txt` is a scored TARGET >= 4 times**, in varied
   situations (different positions/objects/timing), and each time:
   - **FD-informative:** the dynamic visibly changes `X_t -> X_t+1`;
   - **ID-informative:** the action is recoverable from the change (click location = where
     the effect appears; a move = the displacement). Passive (noop-triggered) dynamics
     still count — pair them with the negatives below.
3. **Contrastive negatives (~20-30% of the pool).** For each conditional dynamic include
   near-miss pairs — same action/surface cue, dynamic does NOT fire — so shortcut rules
   (step-parity clocks, "noop always moves X") score WORSE than the true rule. Randomize
   action timing; never a rigid every-k-steps cadence.
4. **Observable transitions.** Avoid pairs where the acting change is invisible or
   aliased (nothing moves, or two actions would produce the identical frame): an oracle
   who knows the true rules should get ~every ID item right. If a game's dynamics make
   some aliasing unavoidable, minimize it and document it in TEST_COVERAGE.md.
5. **Action balance.** Movement games: roughly even counts across the whitelist verbs.
   Click games (keep_action_params=True): diverse click locations (the full `click R C`
   string is the label — spread them over the grid, including repeated clicks on
   meaningful vs empty cells).
6. **Slices 4-12 rows long** (prefer longer): windows (`ctx_prev/ctx_next`, context_k=9)
   stop at episode boundaries, and the decoder needs context to be fairly evaluated.
   Include the cause step and delayed effect in the SAME slice. Rows are copied VERBATIM
   from the driven trajectory.csv (every column unchanged).

## Workflow per game

1. Read `clean_data3/<game>/dynamics.txt`, `clean_data3/<game>/COVERAGE.md`, the train
   slices, and the SEXP
   (`MARAProtocol/python_examples/autumnbench/example_benchmark/programs/<ENV_NAME>.sexp`)
   until you can predict every transition.
2. Draft action sequences covering req. 2-5; drive them to a scratch dir with
   `uv run python offline_learning/autumn_drive.py <ENV_NAME> <scratch> --actions "..."`,
   READ the printed ASCII frames, refine. Multiple drives (one per theme) are fine —
   each becomes its own source trajectory to slice.
3. Curate slices into `clean_data3/<game>/test50/episode_*/trajectory.csv` (build-script
   template in the base methodology; adapt `SRC` to your driven files).
4. Verify: `uv run python -c "import sys; sys.path.insert(0,'offline_learning'); import clean_data3_tools as T; T.verify_pool('offline_learning/clean_data3/<game>/test50','<whitelist>')"`
   — confirm pool size ~50, action histogram, and that each intended pair shows the
   intended change tag.
5. Visualize: `uv run python offline_learning/build_dataset_viz.py offline_learning/clean_data3/<game>/test50 --out offline_learning/clean_data3/<game>/test50/viz.html`
6. Write `TEST_COVERAGE.md`: (a) the core dynamics list; (b) a table dynamic -> which
   episode/pairs target it (positives AND negatives) under ID and FD; (c) action
   histogram of the pool; (d) anything uncoverable and why.

Do NOT modify `train/`, `test/`, `dynamics.txt`, or `COVERAGE.md`. `test50/` is additive.

## Eval hookup (for reference)

`clean_sweep.py --cross-traj` currently points at `<data-root>/<game>/test`; evaluating
against TEST50 uses `--test-run .../clean_data3/<game>/test50 --test-n 50` on
gepa_optimize.py (or the sweep's `--test-dir-name test50 --test-n 50` once added).
