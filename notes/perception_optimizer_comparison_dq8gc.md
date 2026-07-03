# Why the B/P learner under-learns perception on DQ8GC — optimizer & objective comparison

**Date:** 2026-06-18
**Task / data:** AutumnBench **DQ8GC**, first 15 transitions of episode_1 from the crashed
`_frontier_pop` run
(`logs/_frontier_pop/hydra_autumn_DQ8GC/.../2026-06-17_12-41-45_..._stepwise_eb_learn/`).
Model: `google/gemini-2.5-flash` via OpenRouter. Data dir: `/tmp/dq8gc_first15/`.

## TL;DR

- **GEPA + inverse-dynamics-only is the best and cheapest** perception learner here:
  green recall **1.00**, **0/5** agent-collapse, **$0.12/seed**. Stable across two runs.
- The **greedy** (legacy/joint) optimizer is **high-variance and unreliable** — collapsed
  the multi-cell agent to a single cell in 4/5 seeds on the matched split (and 0/5 on an
  earlier, luckier split). Its apparent parity with GEPA was not robust.
- **Adding a forward-prediction term HURT** both optimizers (more collapse, ~1.5–2× cost),
  via the known blind-P/stale degeneracy. True for the LLM **judge** *and* the deterministic
  **textdiff_delta_f1** scorer.
- Root cause of all of it: **inverse dynamics under-determines perception.** A lossy P
  (single "agent" + gray *count*) is action-sufficient, so the objective is indifferent
  between lossy and complete P; the optimizer and data path decide the outcome.

## Ground-truth DQ8GC mechanics (from the trajectory)

- Grid of `black` (bg), `darkgreen` (agent cells), `gray` (targets). Coords (row, col).
- One **mobile agent** cell moves with up/down/left/right; **clicking a gray cell converts
  it to green** (gray 4→0 over steps 8–17); converted greens stay **static** (agent + trail).
- "Correct & complete" perception must: (1) parse the grid, (2) enumerate **all** darkgreen
  and **all** gray coordinates, (3) label green as agent. The agent-vs-converted-green
  distinction is **cross-frame** (only the moving cell is the true agent) → belongs in B,
  not single-frame P.

## Why the production `stepwise_eb_learn` run under-learned (investigation)

- Its `legacy_pop` learner used **pure inverse dynamics** (`frontier_fd_scorer: none` in the
  saved config) — confirmed; no forward objective was active.
- Buffer is **cumulative across episodes** (code: returned from episode N, passed into N+1,
  never reset). At the ep1/step_015 relearn the buffer held **21 transitions** (episode_0's
  6 degenerate ones + episode_1's first 15), not "the first 15". The 6 extra are zero-signal
  (constant grid), so they dilute but don't mislead — *not* the main cause.
- Per-relearn reasoning (improve.log) shows an active **belief↔perception co-adaptation
  collapse**: once B asserted "agent is a single entity," the step_015 G1 gradient explicitly
  said *"Agent features should not include multiple positions … Agent is a single entity"* →
  perception was rewritten to a single agent + gray **count**. ID never penalized it (single
  agent decodes movement; gray count decodes clicks). It recovered to enumerate all greens
  only by step_020 once merges forced the issue.
- **Greedy updates alternate** P,B,P,B (coordinate ascent; `legacy_pop.run_legacy_pop` and
  `validate_beliefs.run_legacy_loop`: `active = "P" if rnd%2==1 else "B"`). Each component is
  optimized against the other's *frozen* state → the ratchet into the single-agent basin.

## Experiments

All on `/tmp/dq8gc_first15` (15 transitions), empty start, 6 rounds (greedy) /
max_metric_calls 150 (GEPA), `google/gemini-2.5-flash`. Completeness = run final P on all 15
ground-truth states, measure fraction of true green/gray coords reported (format-robust
coordinate matcher); "collapse" = final P reports <3 greens on the 3-green state.

### 1. Alternating vs joint greedy (split 12 train / 3 holdout)

| mode | green recall | gray recall | collapse |
|---|---|---|---|
| alternating (P,B,P,B) | 0.72 | 0.58 | **2/5** |
| joint (rewrite P+B together) | 0.99 | 0.60 | **0/5** |

Joint avoids the collapse by rewriting B and P **together** in one coordinated call (it can
assign "P enumerates all greens, B explains the agent head"); one joint belief even recovered
the agent-head + trail + gray-is-clickable model. *But see matched split — this was not
robust.*

### 2. MATCHED split (train=val=15 tied, test=0; 5 seeds) — the clean comparison

| config | green recall | gray recall | **collapse** | cost/seed |
|---|---|---|---|---|
| **GEPA — ID** | **1.00** | 0.60 | **0/5** | **$0.116** |
| GEPA — ID+FD(judge) | 0.78 | 0.60 | 2/5 | $0.191 |
| greedy — ID | 0.62 | 0.82 | 4/5 | $0.144 |
| greedy — ID+FD(judge) | 0.51 | 0.37 | 5/5 | $0.259 |
| greedy — ID+FD(textdiff) | 0.51 | 0.37 | 3/5 | $0.236 |

(Earlier unmatched runs, for reference: GEPA-ID 1.00/0.80/0-collapse/$0.10; greedy-joint-ID
0.99/0.60/0-collapse/$0.15 — the greedy number did **not** reproduce on the matched split.)

## Findings

1. **GEPA-ID dominates and is stable** (green 1.00, 0/5 collapse across both runs, cheapest).
2. **Greedy is high-variance** — 0/5 collapse one run, 4/5 the next (same config). Matching the
   split exposed that its earlier parity with GEPA was luck, not robustness.
3. **Forward prediction hurt every config**: GEPA 0/5→2/5 collapse, greedy stayed bad; ~1.5–2×
   cost. Cause = blind-P/stale degeneracy (a lossy/empty P "predicts no change" and scores
   fine on FD; one GEPA-FD seed even kept the empty-start P as best). ID pairing at w=0.5 did
   not suppress it on this small/noisy data.
4. **textdiff vs judge (FD scorer):** textdiff collapsed less (3/5 vs 5/5) and was marginally
   cheaper, but **not** the cost win expected — FD cost is dominated by the `predict_next_state`
   LLM call (needed by both), not the scorer. textdiff still lost to plain ID.
5. **Inverse dynamics under-determines P** (the root): lossy P is action-optimal, so the
   objective gives no completeness pressure; data path + optimizer decide. GEPA's pareto
   exploration + (empty) belief avoids the single-agent co-adaptation that the greedy ratchets
   into.

## Caveats

- N=5; greedy/FD arms are visibly high-variance (wide error bars). The *stable* signal is
  GEPA-ID. Tighter estimate would need ~15–20 seeds on GEPA-ID vs GEPA-FD (cheap, low-variance).
- Completeness uses a format-robust coordinate matcher (validated against actual P outputs;
  earlier naive matcher gave false-negatives on `p_r3c4` / `3_4` formats — fixed).
- Held-out signal is weak (≤15 tied transitions, no test set); the reliable metric is
  ground-truth coordinate recall, not the loops' own accuracy.

## Artifacts & repro

- Data: `/tmp/dq8gc_first15/episode_0/trajectory.csv` (15 transitions).
- Scripts: `prototypes/perc_invdyn/compare_alt_joint.py` (alt/joint),
  `greedy_matched.py` (greedy none/judge/textdiff, matched split),
  `greedy_fd.py`, `gepa_optimize.py` (`--fd-scorer none|judge|textdiff --tie-train-val`).
- Outputs: `logs/greedy_matched/`, `logs/matched_gepa_id_seed*/`, `logs/matched_gepa_fd_seed*/`,
  `logs/alt_vs_joint/`, `logs/gepa_id_seed*/`, `logs/gepa_fd_seed*/`.
- GEPA matched repro:
  `uv run prototypes/perc_invdyn/gepa_optimize.py --run /tmp/dq8gc_first15 --train-n 15 --val-n 15 --test-n 0 --tie-train-val --actions left,right,up,down,noop,click --start empty --fd-scorer none --max-metric-calls 150 --seed S`
- Greedy matched repro: `uv run prototypes/perc_invdyn/greedy_matched.py`
