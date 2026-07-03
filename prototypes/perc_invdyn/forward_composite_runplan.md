# Forward-dynamics composite in GEPA: run plan (textdiff vs LLM scoring)

## What got wired (gepa_optimize.py)

The GEPA adapter now supports an optional **composite objective**:

```
score = (1 - w)·ID + w·FD          # w = --fd-weight, default 0.5
ID = 1[action recovered]            # inverse dynamics (unchanged)
FD = score of Fwd(P(X_t), A_t, B) vs the TRUE P(X_t+1)
```

- `Fwd` = new `predict_next_state()` — the frozen task_lm run FORWARD: given current
  features + the true action + B, predict next features. Self-supervised; label is the
  logged next frame through the SAME P (`z_t1`); Fwd never sees it.
- `--fd-scorer textdiff` → deterministic `textdiff_delta_f1` (free).
  `--fd-scorer judge`   → LLM `judge_score` (extra call/instance).
  `--fd-scorer none`    → pure ID, **identical to the validated path** (default).
- `--fd-reflect` also feeds forward mispredictions to the proposer (so P can be pushed
  toward Markov sufficiency, not just selected on it).
- Secondary readout: clean-test `FD[scorer]` printed alongside the headline test ID acc.

Purity holds: Fwd and both scorers see only P's own emitted features + the logged next
frame. No raw-grid parser, no game facts. (See memory invdyn-no-external-knowledge.)

## The comparison

**Primary question:** does an LLM judge as the FD term produce a better-generalizing P
than the free deterministic `textdiff` term — enough to justify ~2x the eval cost?

**Headline metric:** clean-test **inverse-dynamics accuracy** (deployable; identical
eval across arms). **Secondary:** clean-test FD score, $ cost, wall time.

### Arms (same train/val/test split per seed)

| arm | flags | LLM calls / instance | what it isolates |
|-----|-------|---------------------:|------------------|
| **ID** (control) | `--fd-scorer none` | 1 (inverse) | the current 0.85–0.95 baseline |
| **+FD textdiff** | `--fd-scorer textdiff --fd-weight 0.5 --fd-reflect` | 2 (inverse + forward) | free FD pressure |
| **+FD judge** | `--fd-scorer judge --fd-weight 0.5 --fd-reflect` | 3 (inverse + forward + judge) | LLM FD pressure |

Cost note: judge ~3x the per-instance F cost of ID, textdiff ~2x. `cache_evaluation=True`
dedupes repeated candidates, so realized cost is lower.

### Games (pick for headroom, not ceiling)

DQ8GC is near-solved by ID alone (~0.95) → use it as a **do-no-harm / cost** check, not a
discriminator. Put the real weight on games where ID leaves room AND P actually moves
(FD needs movement to carry signal — same observability floor as ID):

- **DQ8GC** — regression guard (FD must not drag 0.95 down).
- **ls20** — P emits counts + coords, moves on ~88% of transitions → most FD headroom.
- **7WWW9** — `@r,c` format, but P moved on only ~12% (mostly unobservable) → expect FD
  to be near-inert; include as the "low-observability → FD can't help" control.

3 seeds each (1,2,3). Low-data regime to match the prior sweep:
`--start empty --tie-train-val --train-n 5 --test-n 20 --max-metric-calls 120`.

### Hypotheses / what each outcome means

- **+FD ≈ ID on test acc, FD-on-test rises** → FD is a sound auxiliary that doesn't hurt;
  prefer `textdiff` (free). Weak evidence for the second objective.
- **+FD > ID on a headroom game (ls20)** → FD breaks an ID plateau; the second objective
  earns its place. Then `judge` vs `textdiff` decides the scorer.
- **judge > textdiff on test acc** by more than the cost premium → LLM scoring worth it;
  else `textdiff` is the default and judge is a fallback for unstructured P.
- **Everything flat on 7WWW9** → confirms FD is bounded by observability, not the scorer
  (consistent with the 64%-unobservable finding in gepa-sweep-failure-attribution).

### Knobs to sweep if the first pass is ambiguous

- `--fd-weight` ∈ {0.3, 0.5, 0.7} — too high starves ID and invites the blind-P
  degeneracy (FD alone is gameable); too low and FD never bites.
- `--fd-reflect` on/off — separates "FD as a *selection* pressure" from "FD as a
  *proposal* gradient".

## Commands

Driver (runs all arms × seeds; mirrors run_sweep.py's per-game wiring):

```bash
GAMES_ID="DQ8GC ls20 7WWW9"
for g in $GAMES_ID; do
  # (resolve --run / --actions from run_sweep.py GAMES[g])
  for seed in 1 2 3; do
    for arm in "none" "textdiff" "judge"; do
      flags="--fd-scorer $arm"
      [ "$arm" != "none" ] && flags="$flags --fd-weight 0.5 --fd-reflect"
      uv run prototypes/perc_invdyn/gepa_optimize.py \
        --run "<RUNS_g>" --actions "<ACTIONS_g>" \
        --start empty --tie-train-val --train-n 5 --test-n 20 \
        --max-metric-calls 120 --seed $seed $flags \
        --out-dir logs/perc_invdyn/fd_sweep/$g/$arm/seed$seed
    done
  done
done
```

Single-cell sanity (DQ8GC, textdiff, one seed):

```bash
uv run prototypes/perc_invdyn/gepa_optimize.py \
  --run "logs/dynamics_full/DQ8GC/...,logs/seed_autumn/DQ8GC/..." \
  --actions left,right,up,down,noop --start empty --tie-train-val \
  --train-n 5 --test-n 20 --max-metric-calls 120 --seed 1 \
  --fd-scorer textdiff --fd-weight 0.5 --fd-reflect
```

A `run_fd_sweep.py` analogous to `run_sweep.py` (looping arms × seeds, writing a
results table of test ID-acc / FD / cost) is the clean way to execute — build it once the
single-cell sanity confirms the arms behave.
```
