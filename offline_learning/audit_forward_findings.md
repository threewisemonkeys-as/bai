# Manual audit: does textdiff score how close Ẑ is to the true next state?

Went through all 137 real forward predictions (`diag_forward_results.json`) item by item
(`audit_forward.py` renders each as z_t / z_t1 / z_hat + token diffs), then quantified
against a direct state-similarity reference (`analyze_audit.py`).

## Verdict: textdiff measures CHANGE-fidelity, not STATE-closeness

Spearman(td_real, state_f1) = **0.344** over 100 moved items, where
`state_f1 = token-multiset F1(z_hat, z_t1)` is direct "how alike are these two states".
textdiff scores whether z_hat reproduced the true EDIT (z_t→z_t1), NOT whether z_hat≈z_t1.

| env | td_real | state_f1 | gap |
|-----|--------:|---------:|----:|
| DQ8GC | 0.793 | 0.971 | 0.18 |
| 7WWW9 | 0.277 | 0.893 | 0.62 |
| ice   | 0.193 | 0.938 | 0.75 |
| sp80  | 0.031 | 0.974 | 0.94 |
| ls20  | 0.056 | 0.929 | 0.87 |

The gap grows with state size. Sharpest on ARC: true change is one count (e.g. 28→26),
the LLM changes a DIFFERENT count (256→254), so z_hat is ~98% identical to z_t1 yet
textdiff=0.00 (#89–108).

## ...and that harshness is correct for the PURPOSE

state_f1 is exactly the background-swamped frame metric we rejected as a signal (a P that
predicts "nothing changed" scores ~0.97 while learning no dynamics). textdiff=0 on a
wrong/missed change is the right call for a dynamics objective. **Key reading: td=0
almost always means "missed the change," NOT "z_hat is garbage"** — most td=0 items have
z_hat ≈ z_t1 except for the (mis-predicted) change. This is the forward-MODEL failing to
predict the change, consistent with the env-complexity bottleneck found earlier.

## The one genuine defect: order-sensitivity (difflib)

| # | env | td_real | td_tok | state_f1 | judge |
|---|-----|--------:|-------:|---------:|------:|
| 48 | 7WWW9 | 0.00 | 1.00 | 1.00 | 1.00 |
| 51 | 7WWW9 | 0.00 | 1.00 | 1.00 | 1.00 |

z_hat is an exact PERMUTATION of z_t1 (identical state) but difflib textdiff=0.00.
Wrong by any criterion. Order-invariant `tok_delta_f1` (multiset symdiff) gets both 1.00.
Across 100 moved items **Spearman(td_real, td_tok)=0.923** — difflib's locality buys
almost nothing over the multiset, while carrying this reorder liability.

Secondary (both change-metrics share, minor): on 7WWW9 wrong-DIRECTION predictions bare
digit tokens coincide and grant spurious partial credit (0.2–0.33 at ~0 real overlap);
td_tok 0.286 vs td_real 0.277 mean — no variant is favored.

## Recommendation

- **Switch the GEPA FD scorer from difflib `textdiff_delta_f1` to order-invariant
  `tok_delta_f1`** (or sort/canonicalize P output before diffing). Near-identical ranking
  (ρ=0.92), removes the reorder false-negative. (Reverses the earlier "textdiff for
  locality" lean — real data shows locality is marginal, the order bug is real.)
- Accept FD = change-fidelity, not state-closeness (that's what we want). Read FD=0 as
  "forward model missed the change," which on complex envs is the model's limit, not P's.
- Per-item the score is reliable on DQ8GC (single-cell, gap 0.18); noisy on multi-cell
  @r,c (7WWW9) and near-binary on ARC counts — matching the forward-predictability
  collapse from the prior diagnostic.
