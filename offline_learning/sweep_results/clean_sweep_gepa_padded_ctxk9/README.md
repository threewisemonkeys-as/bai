# clean_sweep_gepa_padded_ctxk9 — essentials

Curated results from the GEPA-only sweep (antimemo config, `context_k=9`,
`deepseek/deepseek-v4-flash`) over 21 games, seed 1. Produced by
`clean_sweep.py` (see `../../HOW_IT_WORKS.md` §5).

This is the **essentials subset** of the original ~1.1 GB run directory
(`logs/clean_sweep_gepa_padded_ctxk9/`, gitignored). The raw traces
(`predictions.jsonl`, `*_calls.jsonl`, `process_log.jsonl`), the large
`optim_viz.html` dashboards, and `gepa_state.bin` are **not** included here —
they are regenerable from a rerun.

## Layout

- `results.json`, `scores.txt`, `criterion.txt` — sweep-level summary and scores.
- `<game>_seed1/`
  - `best_perception_gepa_seed1.py` — the learned perception module P.
  - `best_beliefs_gepa_seed1.txt` — the learned beliefs block B.
  - `test_trace_gepa_seed1.json` — per-item inverse-dynamics trace of the learned
    P/B on the clean test split.
  - `test_trace_raw_seed1.json` — the same for the `raw-frame` reference baseline.
  - `gepa_run_seed1/`
    - `candidates.json` — every candidate P/B GEPA proposed during search.
    - `candidate_tree.html` — the GEPA pareto candidate tree.
    - `run_log.json` — structured GEPA run log.
