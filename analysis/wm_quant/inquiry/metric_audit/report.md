# Metric and serialization audit

The information-extraction figure measures **per-frame compressed description length**, not information retained or useful information extracted. There are real changes in emitted text, but rising and falling gzip curves alone cannot establish richer or poorer state abstractions.

## Reproduction and scope

`audit.py` executes the exact `run_perceive` function extracted from `offline_learning/validate.py` on cached frames, with `PYTHONHASHSEED=0`, no model/client imports, and no network calls. All **79 selected incumbent nodes** reproduce both the original framewise gzip ratio and corpus gzip bytes exactly. The source data are `analysis/perception_metrics/metrics.csv`, its `manifest.json`, cached frames under `cache/`, and candidates under `logs/2026-08-24/human_curated/rexpure/<game>_s1/rexpure_run_seed1/candidates.jsonl`. There is one saved learning run per game, 15 games, 30 candidate nodes each; these data do not measure seed-to-seed uncertainty.

Reproducibility artifacts in this directory:

- `audit.py`, `incumbents.csv`: all 79 plotted incumbent nodes and alternative compressor/serialization measures.
- `games.csv`: first **plotted** node and shipped node; Ice's first plotted node is invalid, so use `incumbents.csv` filtered to `status == ok` for first-valid comparisons.
- `examples.json`: one exact raw observation and actual output from each incumbent.
- `ablation.py`, `ablation.json`: formatting-only counterfactuals on every training frame for SET and Colour Lines.
- `flat_stretches.json`: causes of flat consecutive plotted intervals.

## What the metric measures

`offline_learning/scripts/perception_metrics.py:346` selects the unique raw frames appearing in scored transitions. Training decoys are used for a separate collapse metric, but excluded from this ratio (`:356`). At `:412`, the plotted value is the mean of `gzip(P(x))/gzip(x)` over those fixed frames. Gzip uses level 6 and a zero timestamp (`:267`). Raw denominators are fixed across candidate nodes within each game and split. Thus within-game changes are changes to compressed output size, weighted by the inverse compressed size of each frame; they are not caused by changing raw states sampled over learning. Games have 13–114 unique training frames; average gzip(raw frame) ranges from 70.52 to 163.46 bytes.

The function executes each module afresh on a one-element history (`offline_learning/validate.py:257`), so it measures stateless behavior on an offline corpus, including any constant `step=1` produced from history length. It does not measure temporal information extraction during planning.

For fixed P, a deterministic output does not create additional Shannon information about its input. A ratio above 1 therefore means this serialization is costlier for this compressor, not that P extracted more than all of the available information. Constant prose, derived features, labels, formatting, ordering, and inability of gzip to exploit grid structure can all raise the ratio.

## Two direct counterexamples to semantic interpretation

**Colour Lines:** first learned and shipped outputs have the same background and the same coordinate/color lists on **109/109 frames**. Separator formatting changes the ratio from **1.098176 to 1.112048**, a 1.26% increase. Mean compressed output grows from 119.47 to 120.90 bytes despite shorter literal output. No varying cell facts were added.

**SET:** first learned and shipped outputs have the same coordinate/color lists on **99/99 frames**. The final program adds a constant prefix, `shape=20x20; cursor:unknown;`, and changes cell ordering. The ratio rises **2.416065 → 2.618581**, or 8.38%, without additional varying grid facts. A controlled counterfactual adding only the final constant prefix to the first output gives **2.559339**: this accounts for **70.75%** of the increase. Ordering accounts for the remainder. Mean gzip sizes are **327.24 → 346.15 → 354.05 bytes**. All grid dimensions were verified to be 20×20.

These statements are empirical equivalence on the measured frames, not universal equivalence claims for unseen states.

## Robustness and general patterns

From each game's **first valid incumbent** to its shipped node, gzip bytes per emitted character (`mean gzip(P)/mean len(P)`) rise in **11/15 games**, median **1.24676×**. Every one of the eight games whose text becomes shorter has a higher compressed-byte/character ratio. Subtracting the 18-byte gzip wrapper first gives increases in 13/15 games, median 1.31083×. This supports a descriptive trend toward text that is less repetitive for gzip, not a conclusion about semantic information density.

Examples: Grow shrinks **283.49 → 194.25 characters**, while mean gzip grows **116.23 → 136.74 bytes**. Sand shrinks **363.82 → 216.53 characters**, while mean gzip grows **139.38 → 152.62 bytes**. Grouping cell coordinates under color headings removes repeated color names; metadata and feature labels still have a cost when each frame is compressed alone.

The signs of all **50 nonflat incumbent changes** survive subtracting the gzip wrapper and switching raw grids to compact JSON. Per-frame LZMA and bzip2 each reverse only two small moves. However, Diffusion's tiny first-to-final rise (**1.233411 → 1.237173**) reverses under a ratio of mean compressed lengths, bzip2, and gzip level 1. Its literal output halves, and even mean compressed output decreases **103.36 → 102.11 bytes**; the plotted mean of framewise ratios gives different weights.

Absolute levels and the threshold of 1 are less robust: the shipped ratio exceeds 1 for **12/15 games** with the published encoding, **13/15** with compact JSON, and **15/15** with a lossless palette-plus-character-grid encoding that includes the color palette and row boundaries. Gzip level 1 instead gives 7/15. A threshold of 1 has coding-cost meaning only under the named encoding and compressor.

Corpus and per-frame measures ask different questions. Shipped per-frame median is **1.237**, versus **0.599** for the corpus ratio. Concatenating outputs gives a median **7.23×** size discount relative to compressing each output independently; raw frames get **3.38×**. First-to-shipped directions disagree across these two metrics in five games (Grow, SET, Colour Lines, Diffusion, Sand). This explains why an aggregate compression figure can suggest a different trend from the information-extraction figure.

## Plot annotations to correct before publication

- `fig_perception_metrics.py:103` excludes the seed from the plotted trajectory, despite the adjacent docstring and the summary table's `seed` label. All 15 true seeds emit the empty string. Their nonzero ratio, 0.122–0.284, comes from `gzip("") == 20 bytes`; it is not extracted information.
- The plateau explanation in `fig_perception_compression_per_game.py:91` is too strong. Among **370 flat consecutive plotted intervals**, **356** reflect a proposal that did not beat the incumbent; only **14** are score-improving replacements with unchanged P. Of the 356 rejected proposals, 282 changed P relative to their parent. Flatness therefore does not identify K-only optimization.
- The marker is the creation iteration of the eventually selected node (`wm_panel_grid.py:51`), not the search stopping iteration as claimed by its module docstring. It precedes the final logged iteration in 14/15 games.
- Each panel has its own y-axis range (`wm_panel_grid.py:10`); visual slope magnitudes are not comparable across panels.

## Paper framing

Use “per-frame gzip description-length ratio” or “relative compressed output size.” The figure supports heterogeneous restructuring of state descriptions under the dynamics objective. Some successful candidates add explicit features; others simplify redundant text or change formatting. It does not alone validate the claim in `paper/main.tex:79` that the objective discards decision-irrelevant information, nor does an upward curve establish the “get richer” interpretation in `paper/main.tex:136`. Those claims need semantic feature analysis and task/ablation evidence alongside this diagnostic.
