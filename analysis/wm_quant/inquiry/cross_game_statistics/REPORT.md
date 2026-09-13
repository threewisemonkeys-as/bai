# Cross-game statistical inquiry

The main descriptive tendency is upward compressed output size, with frequent reversals. From the first valid incumbent to the shipped model, the per-frame gzip ratio rises in 13/15 training curves (median +18.12%), falls in Mario and Space Invaders, and changes by less than 5% in Colour Lines and Diffusion. Nine of the fifteen curves include both upward and downward nonzero incumbent changes. Thus there is no monotonic cross-game law: endpoint growth and downward segments coexist.

The corresponding rendering measurements on the test frame set give 12/15 increases, median +15.83%; Diffusion changes sign (+0.31% train, -2.01% test). Both splits have 11 increases greater than 5%, two decreases greater than 5%, and two near-flat changes. There are 35 upward, 14 downward and 14 unchanged replacements between valid incumbent nodes. These replacements are dependent events and must not be counted as 63 independent experiments. All 14 unchanged replacements keep exactly the same perception source; rejected/lower-scoring proposals also cause flat curve stretches and are not represented in this replacement count.

The start excludes the common blank seed, and excludes Ice's initial collapsed proposal: its first valid node is #2 at iteration 2. Ice rises 46.54% from that valid start. The first plotted point is not always a useful baseline.

| Game | First valid train ratio | Shipped train ratio | Train change | Test change | Up/down valid incumbent steps |
|---|---:|---:|---:|---:|---:|
| Ants | 0.673 | 1.588 | +136.0% | +138.7% | 2/0 |
| Colour Lines | 1.098 | 1.112 | +1.3% | +1.5% | 2/1 |
| Diffusion | 1.233 | 1.237 | +0.3% | -2.0% | 2/3 |
| Dino | 0.690 | 0.910 | +31.8% | +31.8% | 4/1 |
| Disease | 0.624 | 1.045 | +67.6% | +70.3% | 3/0 |
| Egg | 1.266 | 1.962 | +54.9% | +54.9% | 3/0 |
| Grow | 1.256 | 1.483 | +18.1% | +15.7% | 1/1 |
| Ice | 0.854 | 1.251 | +46.5% | +47.2% | 3/1 |
| Logic Gates | 1.531 | 1.773 | +15.8% | +15.8% | 2/1 |
| Magnets | 0.620 | 0.947 | +52.9% | +52.3% | 2/0 |
| Mario | 1.298 | 1.160 | -10.6% | -12.1% | 1/2 |
| Paint | 0.834 | 1.212 | +45.3% | +56.6% | 3/1 |
| SET | 2.416 | 2.619 | +8.4% | +8.4% | 1/0 |
| Sand | 1.448 | 1.605 | +10.8% | +12.5% | 3/0 |
| Space Invaders | 0.999 | 0.765 | -23.4% | -23.1% | 3/3 |

## Robustness to the baseline and uncertainty

Using the incumbent at iteration 3 gives 12 increases, two decreases, one unchanged curve, and a median change of +10.75% on train (+10.96% on test). Starting at the first incumbent with nonempty world knowledge K gives the same 12/2/1 sign split and a median +8.38% (+8.44% on test). The broad upward tendency is not solely the first invalid proposal or the initial absence of K. However, its magnitude is baseline dependent and the later trajectories often reverse.

The exact two-sided sign-test p for the first-valid training endpoints is 0.007385 (13/15). A game-resampling percentile bootstrap gives a 95% interval of +1.26% to +52.92% for the median training change. Both are exploratory: these are only fifteen selected games with one training seed each, not a random sample of independent seeds. They do not quantify training-run variability. Multiple metrics, baseline choices and relationships were inspected. Descriptive effect sizes are more defensible in the paper than a thresholded significance claim.

Test uses the same selected P and a partially overlapping frame corpus, not a new trained model or an independent replication. For example, Logic Gates' 13/13 unique test frames also occur in training, Egg overlaps 35/60, and Magnets overlaps 22/50. The CSV contains frame-rendering metrics on the test set but still the original *training* objective score for each node; it is not a table of per-node held-out objective scores.

## Does larger output track a better training objective?

The existing REPORT's all-working-node calculation is reproduced: median per-game Spearman rho +0.228, positive in 12/15 games. Its 388 working nodes are dependent descendants, and repeated identical P source strings inherit different K variants. Several sensitivity analyses show a modest, control-dependent positive association:

| Training-frame metric versus train score | Median per-game association | Positive games |
|---|---:|---:|
| All working nodes | +0.228 | 12/15 |
| Nonempty K nodes | +0.240 | 13/15 |
| Identical P source grouped, mean score over K variants | +0.353 | 12/15 |
| Identical P source grouped, nonempty K only | +0.309 | 11/15 |
| Rank correlation centered within identical K groups | +0.200 | 11/15 |
| Parent-child P changes with K unchanged, 250 valid edges | +0.197 | 9/15 |
| Same edge contrast with nonempty K, 192 edges | +0.317 | 12/15 |

Grouped P means deduplication of exact source strings, not semantic equivalence. The within-K row de-means rank features and scores within each exact K string, after grouping identical P. Parent-child contrasts retain only edges with both endpoints valid and unchanged exact K. Search choice and ancestry still confound these observational comparisons. For example, the P-deduplicated/nonempty-K and within-K 11/15 sign patterns each have exploratory p=0.118; the nonempty-K edge comparison is 12/15, p=0.035. These controls do not establish that adding bytes improves predictions, and they do not support a universal compression objective. The older REPORT also overstates the converse when it says that a metric that mattered would have a consistent sign across games: task-specific or nonlinear optima can produce different signs. Mixed signs alone do not show that representation size is unconstrained or irrelevant.

## Possible explanation involving initial descriptions and game complexity

The initial gzip ratio predicts proportional change negatively (rho=-0.568), while initial and final ratios correlate positively (+0.675). This alone does not establish convergence: the initial value appears in the proportional-change definition, inducing mathematical coupling and sensitivity to noisy initial guesses.

Raw-frame compressed size has a weaker negative association with proportional change (rho=-0.471) and little relationship to the final ratio (+0.136). A post hoc visual-complexity probe, mean distinct grid colors including background, correlates strongly with proportional change (-0.771 train; -0.732 test). But color count is also strongly associated with the initial ratio (+0.800). The partial rank association controlling initial ratio is -0.642, and leave-one-game-out color/growth correlations range from -0.833 to -0.719. This is a hypothesis worth testing on new games/seeds: simple visual palettes may induce initially terse descriptions that later expand. It is not a core paper finding, and neither causal direction nor a universal game-type rule is established. Colors, density and serialization interact; several exploratory comparisons were made.

## Relation to the paper's planning results

Using exact per-task scores, final training-frame gzip ratio versus planning pass@1 gives rho=-0.219 for Plain and +0.044 for Agentic. Proportional ratio growth versus pass@1 is -0.057 and -0.058 respectively. Thus these runs provide no clear association between gzip growth and final planning success. With only fifteen games, coarse task averages and different game difficulty, this is not proof of no association.

Planning improvement over Raw has a negative association with final ratio (rho=-0.596 Plain, -0.416 Agentic), but that contrast also incorporates Raw performance and ceiling effects. It should not be presented as evidence that compression causes improvement. The change-in-size versus improvement-over-Raw associations are only +0.153/+0.088.

Sources: Plain and Raw are computed from each `logs/2026-09-03/planning_v2_online_ds_percap_nl/<game>/online.json`, selecting evaluated rows and averaging `lmwm.pass_rate`/`raw.pass_rate`. Agentic comes from `logs/2026-09-08/agent_wm_full/rows.jsonl`, grouped by game and averaging `agent.pass_rate` for status `done`. Each has 86 tasks; their per-game values reproduce paper/main.tex's table up to printed rounding. `planning.csv` records the joined values; `input_sha256.json` records input hashes.

## Defensible paper interpretation

Suggested wording: “The size of the learned descriptions changes in a task-dependent manner. The per-frame compressed output-to-input ratio increases from the first valid candidate to the selected model in 13 of 15 games (median relative increase 18%), although 9 games exhibit changes in both directions during search. This measure characterizes the coding cost of the descriptions; it does not directly measure task-relevant information. The results are consistent with selecting different levels of descriptive detail under the dynamics objective, rather than with a universal decrease in representation size.”

The paper's statement that the joint objectives encourage discarding irrelevant information is a hypothesis not established by these gzip plots. The abstract's usefulness claim should be supported by objective and planning evaluations/ablations; gzip growth is not equivalent to information gained, and lower gzip size is not equivalent to better abstraction. P is a deterministic transformation, and its fixed program/labels can make its serialized output longer without creating new information about X.

## Reproduction

Run `.venv/bin/python analysis/wm_quant/inquiry/cross_game_statistics/analyze.py` from the repository root. This reads the existing metrics, candidate pools, process logs, frame caches and planning outcomes without external/model calls. It writes only within this inquiry directory. `endpoints.csv`, `incumbent_steps.csv`, `correlations.csv`, `parent_child_edges.csv`, `baseline_sensitivity.csv`, `planning.csv`, and `summary.json` contain the exact underlying calculations. Existing scripts, plots and the paper were not changed.
