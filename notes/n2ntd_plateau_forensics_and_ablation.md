# Why the n2ntd GEPA run plateaued — forensics, the rejected-better-beliefs counterfactual, and the 3×3 P/B ablation

**Date:** 2026-08-04
**Run under study:** `logs/aug4_mixed/n2ntd_seed1` (mixed config: gpt-oss-120b task/decoder pinned cerebras,
deepseek-v4-flash reflection/analysis pinned deepseek,baidu,fireworks; new falsification-first prompts;
min(ID, cFD-hard), set-ID eps 0.1, REx C=5, stratified 30/30, context-k 9, test50). Shipped test set-ID **0.625**
(raw baseline 0.618). Ground truth: Mario-style — coins→ammo, click fires a bullet only if ammo>0 (location
ignored), up = 4-row jump when supported, gravity 1/step otherwise, bullets freeze forever under darkorange
platforms, bullet kills the patrolling blue enemy.

## TL;DR

1. **The run's best beliefs were written at iteration 4** (of 47) and shipped inside candidate 8 (found iter 14,
   val 0.6096). Nothing later beat it on val.
2. **A strictly better belief revision was proposed and rejected.** Iter-28 (a child of the shipped candidate)
   correctly added bullet-freeze-under-platform and bullet-dies-on-blue, bundled with one false rule. It **tied
   exactly** on the 15-row minibatch (10.59 = 10.59) and the strict-`>` accept rule discarded it. Counterfactual
   eval (same perception/decoder/test50): **0.679 vs 0.625** single-step; **FD-exact ×4 at h=2/4** (0.40 vs 0.10);
   **planning h=8 0.60 vs 0.40** — matching the deepseek-decoded shipped arm at gpt-oss price.
3. **The loop's objective cannot judge components at all**: two-way variance decomposition over the pool's cached
   600 val scores → row difficulty 51%, beliefs 1.3%, perception 0.5%, **noise 46%**.
4. **3×3 ablation grid** (P × B through the multistep eval): on n2ntd the **beliefs carry the value** (+0.4–0.6
   planning at h≥2, even over raw frames); perception adds ~+0.1–0.2 at h≥4 *only in combination with beliefs*;
   belief margin ≈ 0 at h=1 — the exact horizon the optimizer measures at.

## 1. Trajectory anatomy

47 iterations, 20 accepts, 21 candidates. Shipped = candidate 8 = cand-2's beliefs (written at **iter 4**, the
first belief reflection) + a perception update at iter 14. Val plateau from iter 14 onward. 11 belief iterations
total: 5 accepted (iters 4, 8, 12, 16, 40), 6 rejected (20, 24, 28, 32, 36, 44). Gate deltas of the accepts:
+4.83, +4.20, **+0.28, +0.32, +0.83** — the last three are decoder-noise-sized; of the rejects: −2.52, −1.18,
**0.00 (iter 28)**, −0.37, −1.57, **−0.17 (iter 44)**.

Only 5 distinct belief texts exist across 21 candidates:

| text (iter written) | candidates | jump=4 | gravity | bullet freeze | bullet kills blue | ammo |
|---|---|---|---|---|---|---|
| iter-4 | 2, **8 (ship)** | y | y | – | – | – |
| iter-8 "branch B" | 4,5,9,11,19 | **regressed to 1 cell** | – | – | – | – |
| iter-12 | 6,7,12–16,18 | – | – | – | – | proxy: "click no-op at row 11" |
| iter-16 | 10, 20 | y (restored) | – | – | – | proxy |
| iter-40 | 17 | **dropped again** | – | – | – | – |
| *rej. iter-28* (on cand 8) | — | y | y | **y** | **y** | – |
| *rej. iter-44* | — | – | – | y | – | – |

Branch B exists because iter-8's reflection ran on candidate 3 — whose belief slot was **empty** — so the
reflector wrote a fresh worldview never having seen branch A's text. 16/20 candidates descend from branch B.
Branch B invented `L=(s−4) mod 18` for the blue block — a Step-header side channel (this run predates the
metadata strip).

**Data coverage was NOT the problem**: the 61-transition train+val pool contains 5 bullet-spawning clicks,
2 ammo-empty clicks, 6 jumps (incl. from platform tops — `dynamics.txt` is wrong that only row 11 supports
jumping; platforms do support Mario), 3 coin pickups, 2 enemy kills, 22 frozen-bullet frames. All in reach of
the analyzer (the kill row was drawn into 24 minibatches).

## 2. Why reflection couldn't improve the score (mechanisms, with receipts)

1. **Gate blind to passive dynamics.** min(ID, cFD) rewards only action-coupled structure. The freeze rule is
   evidenced in 22 transitions and moved the minibatch by exactly 0.00. Worse, the objective incentivizes
   *inventing* false action-couplings: iter-28's analyzer produced "the projectile does not move when the action
   is 'down'" purely to disambiguate down vs noop.
2. **Noise ≥ signal at the gate.** Candidates 2 and 8 (identical beliefs) disagree on ~1/3 of val rows; train-row
   scores oscillate 0↔1 across draws. Belief edits move ≤1–2 of 15 rows; decoder sampling moves ±1–2.
3. **Statelessness.** The reflector sees parent text + one fresh minibatch; rejected children vanish. The freeze
   rule had to be rediscovered from scratch at iter 44 (rejected again). The ammo-proxy "click no-op at row 11"
   was ADDed (12), reinforced (16), DELETEd (40) as minibatches disagreed — churn, not convergence.
4. **No latent-variable induction.** Zero mentions of ammo/coins-as-counter/enemy/kill in all 47+47
   analysis+reflection responses. Single-transition mistake explanation can't form a hidden-state hypothesis.
5. **REx starvation of the val-argmax.** Cand 8 received 3 pulls (iters 25/28/29: −0.6, tie, −1.2 → all
   rejections), each incrementing its failure count in `Beta(1+C·h, 1+C·(1−h)+N_fail)`; posterior mean fell
   ~0.58→~0.40 and it was never selected again. Gate blindness poisons REx's reward.
6. **The pool solved the task; shipping discarded it.** Per-row pareto oracle over the 21 candidates = **0.873**
   val vs 0.610 shipped. No merge/crossover; ship = val-argmax of a single candidate.

Note the iter-28 rejection is NOT a REx decision: the accept gate sits *below* REx (which only picks parents
among accepted candidates). Pure REx (Tang et al. 2024, arXiv:2405.17503) admits every refinement as an arm;
our gate is a GEPA-ism kept for budget (accept = 45 metric calls vs reject = 15).

## 3. Counterfactual: the rejected iter-28 beliefs, evaluated properly

Recipe: `--max-metric-calls 1 --start-perception <shipped P> --start-beliefs <iter-28 text> --keep-obs-metadata`
(headers kept to match the original run's data condition). Run: `logs/aug4_n2ntd_counterfactual/iter28_wk_on_cand8`.

| metric | shipped | iter-28 counterfactual |
|---|---|---|
| test50 set-ID | 0.625 | **0.679** |
| strict singleton | 0.44 | **0.52** |
| FD-exact h=1/2/4/8 | 0.30/0.10/0.10/0.00 | 0.30/**0.40**/**0.40**/0.00 |
| plan h=1/2/4/8 | 0.90/1.00/0.80/0.40 | 1.00/0.90/0.80/**0.60** |

The FD-exact ×4 is the freeze rule stopping phantom bullet drift in rollouts. Planning h=8 matches the
deepseek-decoded shipped arm (0.60) with the cheap decoder. The h=1 test delta (+0.05) sits inside the noise
band — single-transition set-ID cannot see this improvement; multistep can.

## 4. Can we judge the value of perception vs beliefs?

**Free, from the run's own cache — answer: the loop can't.** Regression of the 600 cached (candidate × val-row)
scores on row + belief-identity + perception-identity: rows 51%, beliefs +1.3%, perception +0.5%, unexplained
(decoder sampling) 46%. Any component valuation built on h=1 val is reading noise.

**3×3 ablation grid** (P ∈ {shipped, seed, identity=raw-frames} × B ∈ {shipped, iter-28, empty}), multistep eval,
paired windows, gpt-oss decoder. Cells: `logs/aug4_n2ntd_ablation/`, results
`logs/multistep_shards_aug4_n2ntd_abl_*.{json,md}` (+ the two pre-existing cells). Planning success is the
cross-cell metric (engine-executed; FD-partial scorers differ raw-vs-learned mode — do not compare across modes).

Planning success h=1/2/4/8:

| P \ B | shipped B | iter-28 B | empty B |
|---|---|---|---|
| shipped P | 0.90/1.00/0.80/0.40 | 1.00/0.90/0.80/**0.60** | 0.90/0.50/0.20/**0.00** |
| identity P (raw frames) | 1.00/0.60/0.60/0.30 | 0.90/1.00/0.70/0.40 | 0.50/0.40/0.30/0.00 |
| seed P (emits "" on n2ntd) | 0.20/0.10/0.00/0.00 | 0.00/0.00/0.00/0.00 | 0.00/0.00/0.00/0.00 |

Raw-arm baseline (9 independent replicates): 0.68/0.42/0.20/0.07, per-horizon sd 0.05–0.14 (= noise gauge).

Findings:
- **Beliefs dominate**: +0.4–0.6 planning at h≥2 (vs empty-B), robust with shipped P *and* with raw frames.
  identity-P + iter-28-B (0.90/1.00/0.70/0.40) nearly matches the full shipped artifact.
- **Perception is real but secondary and interactive**: ~+0.1–0.2 planning at h≥4 and large within-row FD
  fidelity gains, but ≈ 0 with empty beliefs. P makes belief rules applicable; it doesn't replace them.
- **Beliefs need state**: seed P returns empty features on every n2ntd obs (why iter-1 was gate-zeroed);
  beliefs + no state plans ≈ 0.
- **The blind spot reproduces**: with shipped P the belief margin at h=1 is ≈ 0 — the optimizer measures at the
  one horizon where beliefs don't matter.

## Implications / levers (not yet implemented)

1. **Multistep term in the loop's objective** (minibatch and/or val) — the deepest fix; gate, REx-h, and ship are
   all blind without it. (Corroborates the earlier multistep-objective log-mining evidence.)
2. **Admit-on-tie / noise-aware gate for belief edits**, or faithful-REx admission with lazy val funding —
   preserves score-neutral-but-true content for further refinement.
3. **Value-driven component selection** via ablation probes (score accepted candidates once with beliefs blanked
   on the same rows, cached) — replaces the fixed 75/25 schedule; on n2ntd it currently points the wrong way.
4. **Latent-variable prompt affordance** for the analyzer (hypothesize hidden counters/state spanning
   transitions) — nothing in 94 calls ever named ammo.
5. **Compose the pool** (per-row pareto winners → merge/ensemble) — 0.873 oracle vs 0.610 shipped.

## Artifacts

- Forensics inputs: `logs/aug4_mixed/n2ntd_seed1/gepa_run_seed1/{process_log,analysis_calls,reflection_calls}.jsonl`, `gepa_state.bin`, `candidates.json`
- Counterfactual run: `logs/aug4_n2ntd_counterfactual/iter28_wk_on_cand8` (+ roster `logs/id_eval_aug4_n2ntd_counterfactual.json`)
- Counterfactual multistep: `logs/multistep_shards_aug4_n2ntd_counterfactual.{json,md}`
- Ablation: `logs/aug4_n2ntd_ablation/<cell>/` (artifact + roster + eval.log), `logs/multistep_shards_aug4_n2ntd_abl_<cell>.{json,md}`
- Comparison arms: `logs/multistep_shards_aug4mixed_n2ntd{,_dsdecode}.json`
