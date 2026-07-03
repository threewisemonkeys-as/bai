# Forward-objective scoring: two methods, validated and compared

`forward_objective.py` — metric-validation harness for scoring a predicted next-state
`Ẑ_{t+1}` against the true `P(X_{t+1})`. No GEPA wiring yet (that's the composite-scalar
step). Driven by a fixed learned P (DQ8GC sweep seed1, 0.95 ID acc), 24 transitions,
task_lm = gemini-2.5-flash.

We don't have a learned forward predictor yet, so we simulate `Ẑ` at three qualities and
check each metric ranks them: **perfect** (=true frame), **stale** (=start frame, the
no-op baseline), **wrong** (=another transition's true frame).

## Results (DQ8GC seed1, n=24, P moves on 24/24)

| metric    | perfect | stale | wrong | perfect−stale | perfect−wrong |
|-----------|--------:|------:|------:|--------------:|--------------:|
| frame_f1  |   1.000 | 0.915 | 0.895 |     **0.085** |         0.105 |
| delta_f1  |   1.000 | 0.000 | 0.549 |     **1.000** |         0.451 |
| llm_judge |   0.958 | 0.142 | 0.315 |     **0.817** |     **0.644** |

Spearman(delta_f1, llm_judge) over all variants = **0.778** (strong agreement).

## Degeneracy probe (BLIND constant P, moves on 0/24)

| metric    | perfect | stale | wrong | perfect−stale |
|-----------|--------:|------:|------:|--------------:|
| frame_f1  |   1.000 | 1.000 | 1.000 |         0.000 |
| delta_f1  |   1.000 | 1.000 | 1.000 |         0.000 |
| llm_judge |   0.958 | 0.896 | 0.921 |         0.062 |

## Findings

1. **Drop `frame_f1`.** Whole-frame token overlap is swamped by the static background:
   it can't separate the true next-state from "nothing moved" (margin +0.085). Useless
   as a gradient.
2. **`delta_f1` is an excellent free deterministic signal.** Grading the predicted
   *change* (coord-atom symmetric difference vs the start frame, over P's OWN output)
   gives perfect=1.0 / stale=0.0 — a clean, total rejection of the no-op baseline.
3. **The LLM judge wins on the WRONG case** (perfect−wrong 0.644 vs delta_f1's 0.451).
   `delta_f1` scores an unrelated next-frame 0.55 because random single-cell moves share
   overlapping coords; the judge recognizes an unrelated state as bad more reliably. The
   judge's edge is semantic robustness (reordering / paraphrase) — barely exercised here
   because this P emits canonically sorted coords, so expect a larger gap on messier P.
4. **Both need inverse-dynamics pairing.** A blind constant P scores ~max under EVERY
   metric with margin ≈0 — FD-alone is gameable by a P that never moves. This is a
   property of P (no movement → no signal), not fixable by the scorer. Confirms FD must
   be a *composite* term with ID, never standalone.
5. **They agree** (Spearman 0.778), so `delta_f1` is a sound cheap proxy and the judge a
   semantic backstop.

## Recommendation for the GEPA composite (next step)

Use `score = α·ID_acc + (1−α)·FD`, with **FD = `delta_f1`** as the default (free,
deterministic, total no-op rejection), and `llm_judge` as an opt-in for games where P's
output is unstructured/reordered. α<1 keeps ID in the mix so the blind-P degeneracy can't
win. α=1 recovers pure ID for the A/B. Purity holds: every scorer touches only P's
emitted symbols and the logged next frame — no raw-grid parser, no game facts.
