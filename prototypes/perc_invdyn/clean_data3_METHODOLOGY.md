# clean_data3 — curating train sets that cover the dynamics UNDER the objectives

## Why this exists (the nrdf6 finding)

GEPA learns a perception module `P` and a beliefs block `B` by optimizing two
objectives over **scored target transitions** `(X_t -> X_t+1, a_t)`:

- **Inverse-dynamics (ID):** given features of `X_t` and `X_t+1`, identify the hidden action `a_t`.
- **Forward-prediction (FD):** given features of `X_t`, predict the features of `X_t+1`.

The composite metric is `0.5*ID + 0.5*FD`.

For `nrdf6` the train set technically *contained* the key dynamic (a crate that sinks
when a rock enters it) but only ever as **window context**, almost never as the **scored
target pair**. Worse, the one place it was a target was a `noop`, and the action cadence
let GEPA explain the crate motion with a spurious `step % 4 == 1` clock. Result: the
learned belief never expressed the real rule.

**Your job, per game:** make sure every CORE dynamic in `dynamics.txt` is exercised as a
**scored target transition** under BOTH objectives — not merely visible in surrounding
context — and do it CONTRASTIVELY so a shortcut can't fool the objective.

## Key facts about how GEPA consumes the data (do not guess — this is the mechanism)

- `load_transitions` builds ONE transition for **every consecutive CSV row pair** `i,i+1`
  in each `episode_*/trajectory.csv`, using `rows[i].Action` as `a_t`.
- Transitions whose action verb is not in the game's **whitelist** are dropped (so
  movement-only games filter to their real action set; see your assigned whitelist).
- Each transition gets a temporal **window** (`ctx_prev`/`ctx_next`, `context_k=9`) gathered
  from the SAME episode CSV; the window **stops at the episode boundary**. The window is
  CONTEXT shown to the decoder — it is NOT itself scored.
- The train set is a **balanced-by-action** sample of `--train-n` (=20) transitions
  (`balanced_split`). With `--tie-train-val` train==val.
  ⇒ If your curated train dir contains **exactly the transitions you want** and the pool
    size ≤ `train-n`, ALL of them are used (balanced_split returns the whole pool).
- `keep_action_params`: for click games the full action string (`click 0 3`) is the label
  (click LOCATION is part of the target); for movement games it is collapsed to the verb.
  Use the whitelist/keep flag you are given.

## The construction recipe (this is how you control what gets scored)

Build `clean_data3/<game>/train/` out of **short contiguous slices of the ORIGINAL
train trajectory**, one slice per `episode_<n>/trajectory.csv`, copied **verbatim**
(every CSV column unchanged — the `Observation` text, `Step`, everything).

- Each internal consecutive pair of a slice becomes a **scored target**.
- Because each slice is its own episode, windows are REAL consecutive frames and never
  bleed across slices.
- Choose slice boundaries so the pairs you want are targets and you don't drag in junk.
  A slice `[s, s+1, s+2]` yields targets `s->s+1` and `s+1->s+2`. A 2-row slice yields one.
- Keep the TOTAL pool at ~**18–22** transitions so `--train-n 20` keeps them all. (If you
  need a few more to cover everything, that's fine — set the pool to exactly N and the run
  uses N; just don't balloon it with redundant noops.)

### Coverage requirement — for EACH core dynamic in dynamics.txt

Include at least one target pair where:
- **FD is informative:** applying the dynamic visibly changes `X_t -> X_t+1` (so predicting
  `X_t+1` REQUIRES the rule). A pair where nothing happens does not test it.
- **ID is informative:** the action is recoverable from the `X_t -> X_t+1` change (e.g. the
  click LOCATION is where the new object appears; a move is recoverable from the object's
  displacement). If a dynamic is only triggered passively (on `noop`), it still belongs —
  but pair it with the contrast below so ID/FD can't be gamed.

### Contrastive principle (defeat shortcuts)

For a dynamic that fires only under certain conditions, ALSO include the **near-miss
negative**: same action / same surface cue but the dynamic does NOT fire, so a lazy rule
(e.g. "this happens every k steps", "noop always moves X") scores WORSE than the true
conditional rule. Example for nrdf6: include `noop`s that sink the crate AND `noop`s at the
same step-parity that do NOT (rock outside the crate / crate already full).

### Delayed effects

If an action's effect shows up one step later (engine `prev`-delay), put the cause step and
the effect step in the SAME slice so the window carries the cause and the effect pair is the
target.

## What to produce per game

```
clean_data3/<game>/
  dynamics.txt                      # copy verbatim from clean_data2/<game>/
  test/ ...                         # copy verbatim from clean_data2/<game>/test/
  train/episode_0/trajectory.csv    # curated slices (verbatim rows from original train)
  train/episode_1/trajectory.csv
  ...
  COVERAGE.md                       # your analysis (see below)
```

`COVERAGE.md` must contain:
1. The list of CORE dynamics you extracted from `dynamics.txt`.
2. A table: dynamic -> is it tested as a TARGET under ID? under FD? in the ORIGINAL train
   pool (i.e. would a balanced-20 sample plausibly score it), and the GAP you found.
3. The curated slices you chose and which dynamic(s) each target pair covers, incl. the
   contrastive negatives.

## Tools (run via `uv run python` from repo root `/home/ays57/bai`)

```python
import sys; sys.path.insert(0,'prototypes/perc_invdyn')
import clean_data3_tools as T
# 1. inspect EVERY transition of the original train trajectory + what changes:
T.dump_transitions('<game>', '<whitelist>')          # data_root defaults to clean_data2
# 2. after building, CONFIRM the scored pool is what you intend (THIS is what GEPA scores):
T.verify_pool('prototypes/perc_invdyn/clean_data3/<game>/train', '<whitelist>', context_k=9)
```

`T.classify(g0,g1)` / `T.diff(g0,g1)` summarize per-color cell add/remove/move between two
grids; `T.grid_at(obs)` parses the `[[...]]` color grid out of an Observation.

## Build-script template (adapt per game)

```python
import csv, shutil
from pathlib import Path
G = "<game>"
SRC  = Path(f"prototypes/perc_invdyn/clean_data2/{G}/train/episode_0/trajectory.csv")
OUT  = Path(f"prototypes/perc_invdyn/clean_data3/{G}")
rows = list(csv.DictReader(SRC.open())); fields = list(rows[0].keys())
by_step = {int(r["Step"]): r for r in rows}
EPISODES = [           # each inner list = consecutive ORIGINAL step numbers -> one episode
    [35,36,37,38],     # e.g. setup-click -> effect -> settle  (annotate which dynamic!)
    [27,28,29],
    # ...
]
if OUT.exists(): shutil.rmtree(OUT)
(OUT/"train").mkdir(parents=True)
for ei, steps in enumerate(EPISODES):
    d = OUT/"train"/f"episode_{ei}"; d.mkdir()
    w = csv.DictWriter((d/"trajectory.csv").open("w", newline=""), fieldnames=fields)
    w.writeheader()
    for s in steps: w.writerow(by_step[s])
shutil.copytree(f"prototypes/perc_invdyn/clean_data2/{G}/test", OUT/"test")
shutil.copy(f"prototypes/perc_invdyn/clean_data2/{G}/dynamics.txt", OUT/"dynamics.txt")
```

Steps must be **consecutive** within a slice (verbatim original rows). Verify the original
has a row for each step you list and that no movement-action row inside a slice gets dropped
by the whitelist in a way that truncates a window you rely on.

## Reference example

`clean_data2/nrdf6_key/` is a finished example of this recipe (12 episodes, exactly 20
scored targets, 4 crate-sink positives + 3 step-parity-matched negatives). Inspect it with
`T.verify_pool('prototypes/perc_invdyn/clean_data2/nrdf6_key/train','noop,click')`.
