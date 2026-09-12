# How the perception module changes over the learning run

Plan for a measurement over the `P` (abstraction program) parameter of the NLWM runs the
paper reports: one metric per learning iteration per game, plotted as a curve. Two metrics
were asked for — (1) the size of the perception module's code, and (2) a compression ratio
`|{P(X)}| / |{X}|`, the number of distinct feature strings divided by the number of distinct
observations. Everything under "What the probe found" was measured on the real runs before
this file was written; the probe is quoted where a number is.

---

## 0. Scope: which runs

**Only the 15 learning runs behind the paper's `NLWM (Plain)` column.** That excludes the
`nofd` / `nobeliefs` ablation training arms under `logs/2026-09-09/ablations/` (their column
in `tab:ablations` is still em-dashes) and, by decision, the Opus-5 reflector runs behind the
appendix's `NLWM (SL)` column.

The chain from a paper column back to a training run is recorded in the evaluation runs
themselves, not inferred:

| paper column | evaluation run | how the artifacts were named | training runs |
|---|---|---|---|
| `NLWM (Plain)`, `fig:autumn-results` + `tab:autumn-results` | `logs/2026-09-03/planning_v2_online_ds_percap_nl` | `eval_curated_online.py --artifact-root logs/2026-08-24/human_curated` (in `driver_nl.log`, every game) | `logs/2026-08-24/human_curated/rexpure/<game>_s1` |
| `NLWM (Agentic)`, same table | `logs/2026-09-08/agent_wm_full` | `nlwm_manifest.json`: `root` = `logs/2026-08-24/human_curated`, plus a per-game sha256 of each `best_perception_rexpure_seed1.py` it copied into a workspace | *the same 15 runs* |
| `NLWM (SL)`, `tab:nlwm-sl` — **out of scope** | `logs/2026-09-02/planning_v2_online_opus5_nl` | `--artifact-root logs/2026-08-24/human_curated_opus5` | `logs/2026-08-24/human_curated_opus5/rexpure/<game>_s1` |

So the analysis is one arm over 15 runs: **Plain**, reflector `deepseek-v4-flash`. Agentic
plans with Plain's artifacts byte-for-byte, so it contributes no separate learning run, and
the SL row above is recorded only so the exclusion is explicit rather than an oversight.
Games (15): `7www9`,
`7xf97`, `bt3gb`, `colour_lines`, `diffusion`, `dino`, `dq8gc`, `eahcw`, `egg`, `f5w3n`,
`logic_gates`, `n2ntd`, `s2kt7`, `SET`, `va6fq`.

Checked, all 15 runs: 30 pool nodes each, `best_perception_rexpure_seed1.py` +
`best_beliefs_rexpure_seed1.txt` present, and the shipped `(P, K)` pair matches **exactly one**
node of `candidates.jsonl` — which is also the `argmax train_score` node. Zero mismatches.
That join is what lets a point on the curve be labelled "this is the one that shipped".

## 1. What a learning run stores

Under `<run>/rexpure_run_seed1/`:

* `candidates.jsonl` — the REx pool, one line per accepted node:
  `{idx, parents, train_score, expansions, perception, world_knowledge}`. `perception` is the
  full source of `P` at that node, so the *entire* trajectory of the parameter is on disk; no
  re-running anything.
* `process_log.jsonl` — one line per search iteration `i` (40–44 of them, for 30 nodes):
  `{i, selected, selected_score, components, verdict, new_idx, new_score, parent_ids}`.
  `verdict` is `accepted` (→ `new_idx`) or `skipped` (proposal produced nothing / duplicate).
  This is the exact iteration ↔ node map, and it is what puts the x-axis in *search time*
  rather than in pool order.
* `resume_state.json` — carries `train_fingerprint`, which makes the split rebuild checkable.

The corpus `X` is rebuilt, not re-derived: `launch.json` holds the run's own argv, and
`rexpure_optimize.build_parser()` + `build_data()` replays it into the byte-identical 60-train
/ 50-test split (the recipe already in `scripts/eval_heldout_cfd.py::rebuild_split`, including
the fingerprint assertion). `build_data` also bakes the contrastive-FD decoy frames onto each
instance, so the hard negatives are available for free.

`P` is measured through `validate.run_perceive` — the same entry point the learner scores
through. It `exec`s the module fresh and calls `perceive([obs])` with a **one-element** history,
so module-level globals reset per frame and any `len(observation_history)` step counter is
constant. That is a real limitation of the measurement, and it is the right one: it is exactly
how `invdyn_core.build_window` feeds `P` when it computes the score that drives the search.

## 2. What the probe found

Both metrics were computed over all 30 nodes of `dq8gc_s1` and `bt3gb_s1` against those runs'
own rebuilt train splits (92 and 94 unique frames, mean raw frame ≈ 2.3 KB).

**Metric (1), code size, works and is graded.** AST node count runs 164 at the seed to
1842 (dq8gc) / 1888 (bt3gb) at the largest pool node. It already shows something: dq8gc ships
node 29 at **557** AST nodes and bt3gb ships node 29 at **1120**, both far below their pool's
maximum. The proposer inflates; selection does not keep the inflation.

**Metric (2) as specified is binary, not graded.** On dq8gc, `|{P(X)}|/|{X}|` is `1.000` for
every one of the 25 working nodes and `1/92` for the 5 broken ones. On bt3gb, `1.000` for 26
and `1/94` for 4. The reason is structural: raw frames in these pools are essentially all
distinct, and any `P` that lists non-background cells is **injective**, so `|{P(X)}| = |{X}|`
by construction. Tightening the input set to the *confusable* frames does not rescue it —
restricted to each instance's hard contrastive-FD decoy set, collapse is `0.0%` for every
working node and `100%` for every broken one.

So the ratio is a **validity indicator**, and a good one: it recovers precisely the partition
the learner's own constant-output gate enforces, and it separates four failure modes that are
worth counting in their own right. On bt3gb, 4 of 30 pool nodes are dead: idx 1 and 7 do not
even parse (`... elss None`; a malformed f-string), idx 11 parses and raises at call time,
idx 0 is the blank seed. It is not a learning curve.

**What is graded is description length.** Mean `|P(x)| / |x|` in bytes moves across nodes —
`0.028 → 0.052` on dq8gc, `0.023 → 0.090` on bt3gb. `P` compresses a 2.3 KB observation to
2–9% of its bytes, and how far it does so is a property of the node.

## 3. The metrics to compute

Per `(arm, game, node)`, on the train corpus and again on the held-out test corpus:

*Size of `P`*
* `ast_nodes` — `len(list(ast.walk(ast.parse(code))))`. The headline: immune to comments,
  docstrings and formatting, all of which the reflector varies freely.
* `code_chars`, `code_lines`, `sloc` — reported alongside, because they are what a reader
  expects and they let anyone check `ast_nodes` is not doing something strange.

*Compression*
* `set_ratio` = `|{P(X)}| / |{X}|` — as asked, kept, and read as the collapse indicator.
* `decoy_collapse` — the same quantity over each instance's `{true next frame} ∪ hard decoys`;
  the task-relevant version of the same question.
* `dl_ratio` = `mean|P(x)| / mean|x|` — the graded one.
* `gzip_ratio` = `gzip(join(P(X))) / gzip(join(X))` — description length with redundancy
  priced in, so a `P` that emits a long but repetitive string is not rewarded for verbosity.
  Shipped as `norm_diversity`, with its numerator broken out as `diversity_bytes` and a
  per-frame twin `info_extraction_ratio`; see §7a for why the corpus version needed one.
* `out_vocab` — distinct whitespace-separated tokens across `{P(x)}`, a crude read on how many
  *kinds* of feature the module emits rather than how many bytes.

*What survives into features* — the part of "compression" that carries meaning here, since
frame-level injectivity is free
* `static_rate` — fraction of scored transitions with `P(x_t) == P(x_t+1)`; `P` blind to the
  step it is being scored on. (dq8gc 18%, bt3gb 3% — it varies by game, which is the point.)
* `change_ratio` — token-level symmetric difference between `P(x_t)` and `P(x_t+1)`, divided by
  the number of grid cells that actually changed between `x_t` and `x_t+1`. Below 1, `P` is
  dropping real change; far above 1, one moved cell is churning many features.

*Status and joins*
* `status` ∈ `{ok, syntax, runtime, collapsed}`, `err_rate`.
* `train_score`, `iteration_i`, `parents`, `depth`, `is_ship`, `on_ship_lineage`,
  `n_frames`, `n_unique_frames`.

## 4. `offline_learning/scripts/perception_metrics.py`

Walks the two arm roots, and for each game:

1. rebuild the split from `launch.json` (fingerprint-asserted), cache the frame corpus to
   `analysis/perception_metrics/cache/<game>.json`. Both arms were launched on the same data
   paths and seed, so the corpus is built **once per game** and reused — the rebuild is the
   slow step (17–33 s/game, dominated by reading the source drives for the K=9 context
   backfill), everything after it is milliseconds.
2. for each of the 30 nodes, run `P` over every unique frame under a per-call timeout, and
   emit one row.

Output: `analysis/perception_metrics/metrics.csv` (tidy, one row per node) plus a
`manifest.json` recording the arm roots, the eval runs they feed, the artifact sha256s, and
the rebuilt fingerprints — so a number in the figure can be traced to a training run and from
there to a paper column.

No LLM calls, no simulator. About 8 min wall for the 15 runs, most of it corpus rebuilds
(cached thereafter).

## 5. `offline_learning/scripts/fig_perception_metrics.py`

Palette imported from `analyze_planning_difficulty` (`COLOR`, `INK`, `GRID`, `SURFACE`), as in
the other figure scripts, so this lands in the paper's own ink. Writes
`analysis/wm_quant/perception_metrics.pdf` + `.png` at ICLR text width.

x-axis is the search iteration `i` from `process_log.jsonl`. Two series per panel:

* **incumbent** — the best-so-far node by `train_score` at each `i`; the `P` the run would have
  shipped had it stopped there. Solid. Flat segments are real: a node that revised only `K`
  inherits its parent's `P`.
* **pool mean** — the mean over all nodes accepted up to `i`, i.e. what the proposer is writing
  rather than what selection keeps. Dashed.

Panels:

* **A** `ast_nodes`, normalised to the shared seed node (all 15 runs start from
  `autumn_seed_perception.py`, so `1.0` means the same thing on every line) — one thin line per
  game, bold median.
* **B** `dl_ratio`, same layout.
* **C** status composition per game: stacked bars of `ok / syntax / runtime / collapsed`, which
  is how much of the search budget is spent on code that never ran.
* **D** `ast_nodes` and `dl_ratio` against `train_score`, scattered, ship node marked — the test
  of whether a bigger or a less compressive `P` buys anything at all.

## 6. Gates

Each is an assertion in the script, not a thing to remember to check:

* the rebuilt train fingerprint equals `resume_state.json::train_fingerprint` (already in
  `rebuild_split`);
* the node flagged `is_ship` reproduces, metric for metric, the numbers computed directly from
  `best_perception_rexpure_seed1.py`, and its sha256 matches `nlwm_manifest.json` where that
  manifest covers the game;
* node 0 is byte-identical across all 15 runs and equals `offline_learning/autumn_seed_perception.py`;
* a second run of the script reproduces `metrics.csv` exactly (`P` is deterministic; if it is
  not, the metric is not a property of the node). This one bit on the first pass: two
  `norm_diversity` values moved between runs because a handful of the learned modules build their
  feature string by iterating a **set**, so the byte order of their output -- though not its
  length, and not which observations it tells apart -- follows the interpreter's hash seed.
  The script now re-execs under `PYTHONHASHSEED=0` and gzips with `mtime=0`. The learner had
  the same exposure; pinning it changes nothing about what was learned, only whether the
  measurement of it repeats.

## 7. Built

* `offline_learning/scripts/perception_metrics.py` -- the measurement. `--check` runs the
  gates alone; `--games`, `--arms`, `--jobs`, `--rebuild-corpus` narrow or refresh it.
* `offline_learning/scripts/fig_perception_metrics.py` -- the summary figure and
  `analysis/perception_metrics/REPORT.md`, both from `metrics.csv`.
* `offline_learning/scripts/fig_perception_compression_per_game.py` -- the compression
  ratios drawn per game rather than medianed, one FIGURE per ratio, each panel on its own
  linear scale, shipped node marked. The summary panels state the trend; these show that the
  games disagree about it. Separate figures rather than two y-scales in one panel: the
  corpus series differ by ~10x for a real reason (the raw frames compress ~100x and the
  features do not), and a dual axis would hide that behind an arbitrary alignment.
  `--metric named` (the default) draws the three named scores; `all` adds the two
  uncompressed-numerator companions.
* Outputs: `analysis/perception_metrics/{metrics.csv, manifest.json, REPORT.md, cache/}` and
  `analysis/wm_quant/perception_metrics.{pdf,png}`. The figure is six cells: A size, B bytes
  emitted, C bytes after compression, D dead nodes by failure mode, E compression against
  train score, F the key; `analysis/wm_quant/perception_{diversity,norm_diversity,info_extraction}_per_game`
  are the 15-panel companions (`--metric all` adds `perception_dl_per_game` and
  `perception_pf_dl_gz_per_game`). `REPORT.md` carries the per-arm medians, the per-game shipped
  nodes, and the rank-correlation table behind the paragraph above.

### 7a. The three named scores

The compression story is told by three columns, named after what they measure rather than
after the compressor:

| column | definition | what it is for |
|---|---|---|
| `diversity_bytes` | `gz([P(X) for all X])` | output diversity -- how many irreducible bytes P's features carry over the whole corpus |
| `norm_diversity` | `gz([P(X)]) / gz([X])` | the same against what the observations themselves cost (was `gzip_ratio`) |
| `info_extraction_ratio` | mean over frames of `gz(P(X)) / gz(X)` | the per-frame version, priced without the cross-frame discount (was `pf_gz_gz_ratio`) |

`dl_ratio` (uncompressed feature bytes per raw observation byte), `pf_dl_gz_ratio` (per
COMPRESSED observation byte, frame by frame), `lzma_ratio` and `twopart_ratio` stay in
`metrics.csv` as checks, and are drawable with `--metric <name>`.

**(1) and (2) are the same curve within a game.** `gz([X])` is a constant per game and
split, so across the 30 nodes of one run the two rank identically -- Spearman exactly
+1.000 on all 15 games. The normalisation buys comparability ACROSS games and nothing else;
the shape inside a panel is the same in either figure.

**(2) and (3) disagree, and (3) is the honest one.** Both corpus scores gzip each side as a
single stream, which is not a fair fight: the raw grids carry their redundancy INSIDE a
frame (the 390 repeats of `"black"`), which one frame's dictionary already eats, so they
gain only ~3.4x from being compressed together, while P prints nearly the same line every
frame and gains ~7.2x. A sub-1.0 `norm_diversity` is therefore partly a reward for P being
repetitive. Median over the 15 shipped modules: `norm_diversity` 0.599, but
`info_extraction_ratio` 1.237 and `pf_dl_gz` 1.536, with 12 of 15 games above 1.0 on each.
Frame by frame P's output is typically LARGER than the observation it replaces; only Dino,
Magnets and Space Invaders come in under. Not a gzip-header artifact either: the envelope is
20 bytes, and subtracting it from both sides moves the median from 1.237 to 1.318 with the
count still 12/15 (checked directly against the 15 artifacts, not through the CSV).

Two cautions on reading any of them:

* **`pf_dl_gz` is a verbosity curve within a game** -- its denominator does not depend on
  the node, so across one run's nodes it is rank-identical to `mean_out_chars` (median
  Spearman +1.000, min +0.958). `info_extraction_ratio` is not (median +0.80), because
  compressing P's output prices its entropy rather than its length.
* **None has a fidelity term, and the direction is not toward compression.** The seed module
  emits the empty string, so 0 is the degenerate optimum on all three. Against `train_score`
  within a game, `dl_ratio` and `pf_dl_gz` are flat (median rho +0.037, positive on 8/15,
  sign p = 1.0), `norm_diversity` nearly so (+0.073, 9/15), and `info_extraction_ratio`
  leans the WRONG way for a compression story (+0.228, 12/15, p = 0.035, with
  `twopart_ratio` behaving the same) -- the nodes the objective prefers cost more compressed
  bytes per frame, not fewer. Six scores were tested against one outcome with no correction,
  so that is a hint, not a result. First-incumbent to ship, the per-frame ratio rises on 8 of
  15 games and falls on 7.

## 8. What this cannot say

* **n = 1 per game.** The spread across 15 games is the only dispersion there is; no seed
  variance, so no per-game error bar. Say "15 games" and not "15 samples".
* `P` is measured the way the learner scores it — one frame, fresh module state. A `P` that
  would behave differently given real history is measured on the behaviour that was optimised,
  which is the honest choice but is not the same as its behaviour inside the planner.
* The x-axis is search iterations of one optimiser under one budget (`--max-nodes 30`). A curve
  that is still rising at `i = 40` says the budget bound, not that the parameter converged.
