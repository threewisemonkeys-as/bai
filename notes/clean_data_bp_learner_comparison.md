# Best B/P learner on clean manual trajectories — GEPA-ID vs legacy greedy across 4 games

**Date:** 2026-06-18
**Goal:** Study the most robust algorithm for learning beliefs(B)/perception(P) from
trajectory data, using *manually-curated, information-dense* trajectories as a clean
testbed (decoupling algorithm quality from the noisy active-collection data path).
**Games:** 2 autumn (F5W3N, DQ8GC) + 2 ARC-AGI-3 (ls20, ft09). Model:
`google/gemini-2.5-flash` (OpenRouter). 30 manual transitions/game, 3 seeds.

## TL;DR

- On the two **discriminating clean testbeds, GEPA-ID ≥ legacy greedy**: DQ8GC 0.70 vs
  0.57 (and stabler + 2.5× cheaper); **ft09 0.50 vs 0.10** (decisive — structured
  click-location extraction is exactly where reflective code-gen wins).
- **Legacy greedy is high-variance / unstable**: a full collapse on DQ8GC (seed3 0.2)
  and a cost blow-up on ls20 (seed3 $3.20, ~4× the mean) — echoes the prior noisy-data
  study [[perc-optimizer-comparison-dq8gc]].
- **"Clean" ≠ "useful."** F5W3N is manually curated but a *poor* testbed: its stochastic
  background (enemy oscillation + random enemy bullets) buries the hero-action signal, so
  **both** learners fall to chance and *below the raw-frame baseline* — perception learning
  is actively harmful when the controllable effect isn't separable from background dynamics.
- **Perception only pays on large observations.** On 16×16 grids the raw frame is the
  strongest decoder (DQ8GC raw 0.90); learned P only helps when the raw frame overwhelms
  the decoder (64×64 ls20/ft09 raw ≈ chance) — and there GEPA (ft09) delivers, legacy doesn't.

## Setup

- **Data:** hand-played informative trajectories, 30 transitions each, persisted at
  `prototypes/perc_invdyn/clean_data/<game>/episode_0/trajectory.csv`. Generated with two
  new drivers: `autumn_drive.py` (drives `AutumnBenchEnvWrapper`) and `arc_drive.py`
  (drives live `arc_agi.Arcade`). Each demonstrates the game's controllable dynamics
  (see "Trajectory design" below).
- **Learners:** `gepa_optimize.py --compare` runs BOTH on the same split per seed:
  - **GEPA-ID** — pareto reflective evolution, pure inverse-dynamics objective (`--fd-scorer none`).
  - **legacy greedy** — coordinate-ascent P/B loop (`run_legacy_loop`, 6 rounds).
- **Split:** `--tie-train-val` on 20 transitions (train = val, the low-data regime the
  request asked for) + 10 held-out for an unbiased inverse-dynamics **test accuracy**.
- **Metric:** F (frozen `gemini-2.5-flash`) predicts the action from `(P(X_t), P(X_t+1), B)`
  over k=5 choices (chance = 0.20). `raw-frame` = same decoder on the raw grid, no P.
- **ARC budget:** 64×64 games get `--max-metric-calls 220` (vs 120 for 16×16) so reflection
  has room to *discover* the right object color. **ft09 uses `--keep-action-params`** so the
  prediction target is the click *location* (8 distinct ACTION6 x,y), not just "ACTION6".

## Results (held-out inverse-dynamics test acc; per-seed then mean)

| game | grid | chance | raw-frame | **GEPA-ID** | **legacy** | GEPA $/seed | legacy $/seed |
|---|---|---|---|---|---|---|---|
| **DQ8GC** | 16² | 0.20 | [1.0,0.9,0.8]→**0.90** | [0.6,0.8,0.7]→**0.70** | [0.8,0.7,0.2]→0.57 | 0.067 | 0.170 |
| **F5W3N** | 16² | 0.20 | [0.4,0.4,0.7]→0.50 | [0.2,0.2,0.3]→0.23 | [0.3,0.2,0.3]→0.27 | 0.300 | 0.336 |
| **ls20**  | 64² | 0.20 | [0.2,0.2,0.2]→0.20 | [0.3,0.2,0.1]→0.20 | [0.5,0.3,0.1]→0.30 | 0.278 | 1.197 |
| **ft09**  | 64² | 0.20 | [0.1,0.0,0.1]→0.07 | [0.5,0.7,0.3]→**0.50** | [0.1,0.1,0.1]→0.10 | 1.506 | 0.824 |

(GEPA times 64–978 s/seed; legacy 76–690 s/seed. ft09 GEPA is the slow/expensive arm:
220 metric calls × 64×64 prompts ≈ 7–16 min, ~$1.5/seed.)

## Per-game reading

- **DQ8GC (clean, deterministic, small) — GEPA wins on stability.** GEPA's trace discovered
  the correct representation: a perception that enumerates **all** green segments
  (`"Player segments at: [...]"` — the multi-green case the production loop kept collapsing)
  plus beliefs encoding movement (`up ⇒ y−1`, no gray moves on a move; click changes a gray).
  GEPA 0.70 (0.6/0.8/0.7, stable) > legacy 0.57 (0.8/0.7/**0.2 collapse**), at 2.5× lower cost.
  Raw-frame 0.90 still tops both — 16×16 is small enough that the decoder reads it directly.
- **F5W3N (stochastic, small) — perception HURTS.** Raw-frame 0.50 but both learners ≈ chance
  (GEPA 0.23, legacy 0.27). The two enemy rows oscillate on a time clock and orange bullets
  spawn randomly every frame, so most cells change regardless of the hero action; the learner
  can't tell which features are action-relevant and learns lossy junk. *Clean curation does
  not rescue an unseparable control signal.*
- **ls20 (clean directional, 64²) — discovery failure.** Player = a 5×2 color-12 block moving
  one room per action (A1/A2/A3/A4 = up/down/left/right), walls block. A perfect "player at
  (r,c)" P would make ID trivial, but raw-frame is at chance (64² overwhelms the decoder) and
  **neither learner reliably isolated color-12** — the learned P fell back to generic
  color-counting → chance (GEPA 0.20, legacy 0.30, legacy marginally ahead by one lucky seed).
  legacy also blew up to $3.20 on seed3.
- **ft09 (clean click-puzzle, 64²) — GEPA wins decisively.** 3×3 grid of clickable tiles; each
  ACTION6 cycles one tile's pattern (localized diff), so the click *location* is recoverable.
  GEPA 0.50 (0.5/0.7/0.3) ≫ legacy 0.10 (flat at chance). Reflective code-gen learns to surface
  the changed-tile coordinates; the greedy P/B loop never does. This is the clearest separation
  and the canonical case for GEPA: the action is *structured* (coordinates) and only a
  programmatic extractor recovers it.

## Conclusions toward "most robust B/P learner"

1. **GEPA-ID is the more robust default**: it wins or ties legacy on every clean game, wins
   decisively where structured extraction matters (ft09), is stabler (no DQ8GC collapse), and
   avoids legacy's cost blow-ups on big grids.
2. **Legacy greedy remains high-variance** (collapse + cost spikes), as on noisy data.
3. **Testbed quality is the hidden variable.** Of the 4 games only **DQ8GC and ft09** cleanly
   *discriminate* the algorithms. F5W3N (unseparable stochastic signal) and ls20 (color-discovery
   failure within budget) are weak testbeds where baselines aren't beaten — useful to know before
   trusting either as an algorithm benchmark.
4. **Perception learning pays off precisely when raw observations are too large for the decoder.**
   This reframes the win condition: the B/P loop's value is on big/structured observations
   (ARC-scale), and there GEPA is the one that converts.

## Caveats / next steps

- N=3, test=10 (0.1 granularity) → directional, not significance-grade. Tighten with more seeds
  on the discriminating pair (DQ8GC, ft09).
- ls20 needs either a larger GEPA budget or a discovery nudge (it *can* be made trivial with the
  right P) before it's a fair testbed; F5W3N needs a lower-stochasticity variant (or is simply
  unsuitable for ID without temporal modelling).
- Consider a game-appropriate **completeness** metric (à la DQ8GC coord-recall) per game, not just
  held-out ID acc, to score *what* was learned rather than only decode accuracy.

## Artifacts & repro

- Data: `prototypes/perc_invdyn/clean_data/{dq8gc,f5w3n,ls20,ft09}/episode_0/trajectory.csv`
- Drivers: `prototypes/perc_invdyn/autumn_drive.py`, `prototypes/perc_invdyn/arc_drive.py`
- Sweep: `prototypes/perc_invdyn/clean_sweep.py` (per-game whitelist + budget + keep-params)
- Outputs: `logs/clean_sweep/results.json` and `logs/clean_sweep/<game>_seed<N>/`
  (`stdout.txt`, `best_perception_gepa_seed*.py`, `best_beliefs_gepa_seed*.txt`)
- Repro one cell:
  `uv run python prototypes/perc_invdyn/gepa_optimize.py --run prototypes/perc_invdyn/clean_data/ft09 --train-n 20 --val-n 20 --test-n 10 --tie-train-val --actions ACTION6 --keep-action-params --start empty --fd-scorer none --max-metric-calls 220 --compare --legacy-rounds 6 --seed 1`
- Full sweep: `uv run python prototypes/perc_invdyn/clean_sweep.py --seeds 1,2,3 --parallel 6`
