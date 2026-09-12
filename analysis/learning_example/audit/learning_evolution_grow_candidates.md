# Grow: candidates for a perception/belief learning figure

The strongest Grow candidate is **rain components pausing, falling, and leaving the grid**, using `train_d0` frames **174–177** and learning nodes **2 → 10 → 11 → 16**. It provides an explicit link between learned perception output and a subsequent belief update. The qualification is that perception learns connected components and their extents, rather than a game-specific relation such as contact or occlusion. Under the stricter requirement of a new interaction predicate that beliefs learn to use, **Grow still has no clean example comparable to Magnets**.

This review covers all **30 candidates**, **8 distinct nonempty beliefs**, **832 transitions across four training drives**, and **1,800 cached prediction records** in `logs/2026-08-24/human_curated/rexpure/7xf97_s1/`. Perception was replayed on individual recorded frames using the training runner. No new model evaluations were made. Node IDs and learning iterations are distinct.

[Frame comparison](../learning_evolution_grow_candidates_frames.png) · [Full evidence](learning_evolution_grow_candidates.json) · [Reproduction script](../../../offline_learning/scripts/audit_learning_evolution_grow.py)

| Candidate | Recorded sequence | Learning checkpoints | Assessment |
|---|---|---|---|
| A. Rain pauses, falls, and exits | `train_d0`, 174–177; `noop → down → noop` | 2 (iteration 3), 10 (12), 11 (13), 16 (20) | Best explicit P-to-B connection; component grouping is the semantic change. |
| B. Rain is absorbed by a flower | `train_d0`, 69–72; `noop → noop → noop` | 2 (3), 7 (8), 17 (21), 20 (24) | New contact rule works for this window; P has no contact predicate, and there is no growth. |
| C. Cloud returns from the boundary | `train_d0`, 26–29; `left → left → left` | 19 (23) → 23 (27); compare B at node 28 (32) on another branch | Stronger perception predicate; incomplete evidence of beliefs learning to use it. |
| D. Uncovering the sun enables growth | `train_d0`, 218–221; `noop → left → noop` | Inspect 11, 20, 23, 24 | Excellent game dynamic, but unsupported as a successful joint-learning example. |

## A. Recommended: rain components and boundary disappearance

All relevant rain is in column 12. The upper component moves as a group; the lower component loses cells at the bottom boundary. A `down` step pauses the existing rain in this particular recorded window.

| Frame | Action to next frame | Relevant output at P11 and P16 |
|---|---|---|
| 174 | `noop` | `blue:10-12,12-12\|blue:14-15,12-12` |
| 175 | `down` | `blue:11-13,12-12\|blue:15,12` |
| 176 | `noop` | `blue:11-13,12-12\|blue:15,12` |
| 177 | — | `blue:12-14,12-12` |

These are literal excerpts; the rest of the grid can be gray text or omitted in a figure. At frame 176, the earlier P2/P10 output instead lists `11,12:blue`, `12,12:blue`, `13,12:blue`, and `15,12:blue` as individual cells, interspersed with other colors.

The four learning rows would show:

1. **Node 2 / iteration 3:** individual colored cells; B is empty.
2. **Node 10 / iteration 12:** the same P; B learns downward drift, the `down` exception, and disappearance below row 15.
3. **Node 11 / iteration 13:** P groups connected same-color cells into components and represents rectangular components by row/column extents. B is unchanged from node 10.
4. **Node 16 / iteration 20:** P is unchanged from node 11. B now explicitly describes `blue:rowStart-rowEnd, col-col`, explains which numbers are the inclusive row range, and says **“Drift adds 1 to both row numbers”**, with an explicit bottom-boundary rule.

The chosen checkpoints all belong to `0 → 2 → 3 → 7 → 10 → 11 → 16`. B16 is a real stored belief paired with P11, not a manually combined program. [Candidate source](../../../logs/2026-08-24/human_curated/rexpure/7xf97_s1/rexpure_run_seed1/candidates.jsonl)

**Evidence of useful adaptation:** for 176→177, node 11’s cached forward reasoning reads rows as columns and selects the wrong next state. Node 16’s forward reasoning correctly shifts rows 11–13 to 12–14 and removes the cell at row 15. The cached forward-selection score changes **0 → 1**. Node 16’s inverse-action answer is also correct, but its inverse reasoning still confuses coordinates; use the forward explanation as the supporting evidence. [Node 11 prediction, line 664](../../../logs/2026-08-24/human_curated/rexpure/7xf97_s1/rexpure_run_seed1/predictions.jsonl:664) · [Node 16 prediction, line 1024](../../../logs/2026-08-24/human_curated/rexpure/7xf97_s1/rexpure_run_seed1/predictions.jsonl:1024)

The local B16 rain-update rule reproduces **all cells in all three selected transitions**. This supports the selected sequence, not general correctness: B16 misses rain creation, contact behavior, and growth elsewhere. Perception supplies spatial components, not persistent raindrop identities. If component grouping is considered too close to compression for the paper’s intended claim, reject this candidate rather than relabeling it as learned collision perception.

**Figure emphasis:** highlight the change from separate cells to `blue:11-13,12-12`, then highlight the matching row-range update in B16. Show the bottom cell disappearing and the pause under `down`. Keep unchanged sun/cloud/plant descriptions subdued.

## B. Secondary: absorption by a fully grown flower

At frames 69–72, a single blue cell moves `(10,1) → (11,1) → absent → absent`; a mediumpurple flower occupies `(12,1)` throughout. This is a compact, readable approach/contact/disappearance sequence. [Trajectory](../../../offline_learning/human_data/7xf97/informative_curated/drives/train_d0/episode_0/trajectory.csv)

Along `0 → 2 → 3 → 7 → 17 → 20`, P17 introduces connected components. B20 subsequently adds the rule that a blue cell disappears if its attempted destination is outside the grid or occupied by green or mediumpurple. Earlier B7 describes downward drift and disappearing off-grid but does not include this collision rule. P17 is unchanged at node 20. [B20, candidate line 21](../../../logs/2026-08-24/human_curated/rexpure/7xf97_s1/rexpure_run_seed1/candidates.jsonl:21)

The B20 local rule reproduces all three transitions and the whole visible grid in this window. However:

- The highlighted blue cell is a singleton, so its P change is weak visually. There is **no learned contact/above-flower predicate**.
- B20 works from component coordinates; it does not explicitly reference a new feature name as B16 does.
- There is **no cached evaluation for this exact window**, so do not claim an observed evaluation improvement here.
- Nearby multi-droplet contact examples expose an error: B20 moves the surviving droplets, while the recorded state keeps them stationary during absorption. On 203→204, the forward-selection score is 0. On 65→66, it scores 1 despite predicting the surviving rain at the wrong rows. These are not clean successes. [Line 1220](../../../logs/2026-08-24/human_curated/rexpure/7xf97_s1/rexpure_run_seed1/predictions.jsonl:1220) · [Line 1244](../../../logs/2026-08-24/human_curated/rexpure/7xf97_s1/rexpure_run_seed1/predictions.jsonl:1244)

Use this only to illustrate learning a limited contact rule, not plant growth or a reliable complete rain model.

## C. Secondary: cloud clipping and a learned edge predicate

In frames 26–29, the visible gray block changes columns **13–15 → 12–15 → 11–14 → 10–13** under three `left` actions. It first becomes fully visible, then translates at its full width.

P23 introduces the meaningful predicate **`gcre`**, computed as whether the rightmost visible gray column equals 15. Its values in these frames are **1 → 1 → 0 → 0**. P19, the direct parent, has no such flag. B is unchanged between nodes 19 and 23 and was learned at node 10.

There is some real evidence of use: a cached P23/B10 forward response explicitly interprets `gcre` as contact with the rightmost column and correctly updates it from 1 to 0 on 27→28. This is stronger evidence than the earlier `gare` example, which is not mentioned in any cached inverse or forward reasoning. However, the same `gcre` transition also has a cached response that associates the flag with absence of blue cells; the usage is not consistent. [Correct flag interpretation, line 1431](../../../logs/2026-08-24/human_curated/rexpure/7xf97_s1/rexpure_run_seed1/predictions.jsonl:1431) · [Incorrect interpretation, line 1411](../../../logs/2026-08-24/human_curated/rexpure/7xf97_s1/rexpure_run_seed1/predictions.jsonl:1411)

**The missing link:** B10 does not correctly model the initial clipped-cloud entry. B28 learns a relevant rule: when gray touches the right boundary and its visible width is less than four, `left` increases its visible width without moving the right edge. That local rule fits all three displayed transitions, but B28 lies on `0 → 2 → 3 → 7 → 10 → 11 → 14 → 19 → 21 → 28`. It uses P21’s individual-cell representation, **not P23’s flags**. Do not show P23 paired with B28 as an observed learned agent. [B28, candidate line 29](../../../logs/2026-08-24/human_curated/rexpure/7xf97_s1/rexpure_run_seed1/candidates.jsonl:29)

No stored belief names any of P23’s flags. The actual P23 descendant B24 also has a cached response that incorrectly treats `gcre` as gray existence. This candidate can illustrate separate discoveries, or occasional downstream use of a semantic feature, but does not establish a clean joint-learning progression. [B24 failure, line 1551](../../../logs/2026-08-24/human_curated/rexpure/7xf97_s1/rexpure_run_seed1/predictions.jsonl:1551)

## D. Attractive sequence to exclude from a learning-success claim

Frames 218–221 show the desired conditional-growth dynamic especially well:

- **218→219, `noop`:** rain touches the plant at column 11 and disappears; the plant does not grow while the sun is partially covered.
- **219→220, `left`:** the cloud moves left; the sun’s visible width increases from two columns to three.
- **220→221, `noop`:** another contacting drop disappears, and a green cell appears at `(14,11)`.

But all nine P23 flags are **identical throughout these four frames** (`gale=1`, `gare=0`, and the same presence/edge flags). Visible adjacency is not an occlusion test. None of the eight nonempty beliefs learns that rain-induced growth depends on the sun not overlapping the cloud; the beliefs continue to describe green/purple cells as static. The local game source contains that missing conjunction. [Game source](../../../autumn_programs/grow.sexp)

This is useful as a diagnostic or failure example. It should not be used to claim that the approach learned the sunlight condition.

## Verification and recommended choice

The accompanying script validates **72 perception/frame outputs**, **23 selected cached records**, **15 additional supporting/counterexample records**, and **9 full-grid local transitions** for A, B, and B28’s limited boundary rule in C. The additional records overlap some selected records; the counts are not a claim of 38 distinct evaluations. Figure previews show complete 16×16 observations, with orange boxes marking the relevant regions.

For a Grow figure, choose **A** and describe the learning as **cells → spatial components → beliefs that operate on component extents**. If the figure must demonstrate a new game-specific perceptual relation that beliefs learn to use, retain Magnets and treat these Grow cases as near-matches.
