# Joint perception–belief learning: stricter follow-up audit

**Recommendation: Magnets (`7www9`), `train_d1`, frames 32–35, learning nodes 1 → 6 → 8.** It satisfies all three requested conditions, with unusually direct evidence: P introduces spatial predicates; the next B update explicitly names and branches on them; and the recorded model changes from the wrong action/next-frame choice to the correct choices on the illustrated magnetic snap.

Dino is a useful backup. Its floor-contact predicate and subsequent stopping rule are semantically aligned, but B expresses the condition in coordinates rather than naming the predicate. Magnets has the stronger evidence of explicit use.

[Recorded-frame preview](../learning_evolution_joint_candidates_frames.png) · [PDF](../learning_evolution_joint_candidates_frames.pdf) · [Full evidence JSON](learning_evolution_joint_candidates.json) · [Reproduction script](../../../offline_learning/scripts/audit_learning_evolution_joint.py)

## 1. Magnets: learn relations, then use them to explain a snap

The fixed red magnet occupies `(7,7), (8,7)`. The blue magnet initially occupies `(7,5), (8,5)`. A **down** action makes the blue magnet move **down and right**, into contact with red. A noop leaves it stable, then another down moves it clear of the red rows. This is an actual attraction dynamic, not an ordinary directional translation; see the [simulator program](../../../autumn_programs/magnets.sexp).

All coordinates below are `(row, column)`. The action on row t produces observation t+1.

| Recorded step | Blue cells | New P features on those cells | Action to next frame |
|---|---|---|---|
| 32 | `(7,5), (8,5)` | `rDist2`, `rDist2` | `down` |
| 33 | `(8,6), (9,6)` | `rDist1`, `noRedRow` | `noop` |
| 34 | `(8,6), (9,6)` | `rDist1`, `noRedRow` | `down` |
| 35 | `(9,6), (10,6)` | `noRedRow`, `noRedRow` | — |

The [full training drive](../../../offline_learning/human_data/7www9/informative_curated/drives/train_d1/episode_0/trajectory.csv) contains all four frames. The snap is also present in the [curated training pair](../../../offline_learning/human_data/7www9/informative_curated/train_d1/episode_2/trajectory.csv).

### Learning checkpoints

| Node / learning iteration | Perception | World knowledge B | Whole-training score |
|---|---|---|---:|
| **1 / 1** | Non-background cell coordinates and colors | Empty | 0.8426 |
| **6 / 6** | Adds same-row relation and distance to the nearest red cell | Still empty | 0.7755 |
| **8 / 8** | Identical to node 6 | Introduces rules explicitly using `rDist` and `noRedRow` | 0.7947 |

This is the exact ancestry `0 → 1 → 6 → 8`; all three updates were accepted into the candidate pool. A diagram can use three learning rows, or prepend empty node 0 for four rows. There is no need to combine components from unrelated branches. The [candidate records](../../../logs/2026-08-24/human_curated/rexpure/7www9_s1/rexpure_run_seed1/candidates.jsonl) and [process log](../../../logs/2026-08-24/human_curated/rexpure/7www9_s1/rexpure_run_seed1/process_log.jsonl) provide the complete provenance; exact line numbers are in the evidence JSON.

**Why the P change is semantic.** Node 6 computes a relation between different objects. `rDistN` means that a blue cell shares a row with at least one red cell, with horizontal distance N to the nearest such red. `noRedRow` means no red cell shares its row. This is computed from the current frame and works under the actual evaluator. It is additional relational information, not a renamed coordinate list.

For example, the full outputs at step 32 are:

```text
P1: 7,5,blue; 7,7,red; 8,5,blue; 8,7,red
P6: 7,5,blue,rDist2; 7,7,red; 8,5,blue,rDist2; 8,7,red
P8: 7,5,blue,rDist2; 7,7,red; 8,5,blue,rDist2; 8,7,red
```

At step 33, P6/P8 emit:

```text
7,7,red; 8,6,blue,rDist1; 8,7,red; 9,6,blue,noRedRow
```

**Why B really works with P.** Node 8 introduces these verbatim clauses:

> If a blue cell is in row 7 or row 8, its label is "rDistN", where N = |col - 7| (horizontal distance to nearest red).

> Otherwise, its label is "noRedRow".

For down with no `noRedRow` blue in the same column, it states:

> The rDist blue in row 7 (upper red row) moves down 2 rows to row 9, also moves horizontally 1 step toward column 7, and becomes "noRedRow".

> The rDist blue in row 8 (lower red row) moves horizontally 1 step toward column 7 and remains "rDist" with updated N.

These clauses predict the correct next **set of visible cells** for the snap. Its noop rule correctly preserves the next frame; its same-column `noRedRow` clause correctly accounts for the subsequent separation. The reproduction script checks all three local transitions against recorded coordinates.

### The prediction logs show actual use and a local improvement

For the same scored transition, step **32 → 33**, the existing cached predictions are:

| Node | Inferred action | Correct action | Contrastive next-frame selection |
|---|---|---|---|
| 1 | `right` | `down` | Wrong, score 0 |
| 6 | `right` | `down` | Wrong, score 0 |
| 8 | `down` | `down` | Correct, score 1 |

Sources: [node 1 prediction, line 154](../../../logs/2026-08-24/human_curated/rexpure/7www9_s1/rexpure_run_seed1/predictions.jsonl#L154), [node 6 prediction, line 334](../../../logs/2026-08-24/human_curated/rexpure/7www9_s1/rexpure_run_seed1/predictions.jsonl#L334), [node 8 prediction, line 574](../../../logs/2026-08-24/human_curated/rexpure/7www9_s1/rexpure_run_seed1/predictions.jsonl#L574). They were matched using the repository's hashes of P/B and `(observation, next observation, action)`, and their text features agree with freshly executed P.

The node 8 inverse-dynamics trace explicitly applies the `rDist` clauses to obtain `(9,6), noRedRow` and `(8,6), rDist1`, then concludes `down`. Its contrastive forward trace uses the same clauses to select the correct next frame. This is stronger evidence than just overlapping vocabulary in P and B. It is **contrastive next-frame selection**, not a new free-form forward rollout. The two other illustrated transitions are verified against the written rules; they have no corresponding cached scores for these nodes in the inspected prediction log.

### What the figure can claim

Suggested narrative: **“Learning a relational representation and a compatible belief makes a magnetic snap interpretable.”** Highlight `rDist2 → rDist1/noRedRow`, then the newly learned B clauses, and annotate the snap with the logged wrong → correct action and next-frame decisions.

Limits that matter:

- **Explored branch, not saved best.** Saved node 14 scores 0.9504 and has different ancestry (`0 → 3 → 9 → 11 → 14`). Scores on the illustrated branch fall when P changes and recover only partly when B changes. This supports a local learning example, not monotonic overall improvement or final-model superiority.
- **Local rule, not a complete magnet theory.** Node 8 still incorrectly says horizontal actions always move one column. Immediately before this window, step 31's `right` is blocked, contradicting that claim. Other vertical configurations are also not modeled correctly.
- **No recovered magnet identity or pole labels.** P describes colored cells and spatial relations. B obtains the correct output cell set for the snap while attributing different motions to its two cells; it does not learn that a rigid magnet translates as one object.
- **No learned distance threshold.** B explicitly uses the distinction `rDist` versus `noRedRow` and updates N, but does not learn a general attraction law conditioned on distance N.

These limits do not invalidate the three requested conditions, but they should bound the figure caption.

## 2. Dino: floor contact becomes a condition for falling

Use **`train_d1`, frames 100–103**, with `noop → noop → up`. The dinosaur falls one row, lands, remains at the floor while obstacles continue moving, then jumps six rows. See the [recorded drive](../../../offline_learning/human_data/dino/informative_curated/drives/train_d1/episode_0/trajectory.csv) and [simulator program](../../../autumn_programs/dino.sexp).

| Step | Red row range | Relevant learned flag | What happens next |
|---|---|---|---|
| 100 | 15–18 | `red:bottom` absent | Falls to floor |
| 101 | 16–19 | `red:bottom` present | Stays grounded on noop |
| 102 | 16–19 | `red:bottom` present | Jumps on up |
| 103 | 10–13 | `red:bottom` absent | — |

| Node / learning iteration | Change | Whole-training score |
|---|---|---:|
| **12 / 13** | Cell coordinates; B says noop always increases every red cell's row by 1 | 0.8763 |
| **19 / 30** | P adds computed `color:bottom` and `color:top` flags; B unchanged | 0.8781 |
| **20 / 32** | P unchanged; B adds the floor stopping condition | 0.9272 |

The full ancestry is `0 → 3 → 4 → 8 → 10 → 12 → 19 → 20`. The flags are calculated from per-color minimum and maximum visible rows. In this game `red:bottom` is a useful floor-contact predicate, computed entirely from one frame. For the selected window, all four exact P outputs and both B versions are in the evidence JSON.

Node 20 changes the noop rule to:

> On `noop`: if the maximum row among red cells is 19, then red does not move; otherwise every red cell’s row increases by 1 (downward).

This condition is **semantically equivalent to `red:bottom`**, but B never names the flag. The subsequent model traces reason using maximum row 19; the logs do not establish whether it reads the flag or recomputes the maximum from the retained coordinates. Therefore this is weaker than Magnets for a strict requirement of explicit predicate use.

There is nevertheless supporting logged improvement on **separate** floor-contact transitions: at `train_d1` step 1, node 19's contrastive next-frame selection fails and node 20's succeeds (prediction lines 1158 and 1278). At step 8, the inferred action changes from incorrect `up` to correct `noop` (lines 1162 and 1282). Those are not the 100–103 preview frames. In the selected preview, the scored jump at 102 → 103 is already correct before these updates, so it should not be presented as a newly solved jump.

This is also a side branch: saved node 18 scores 0.9360. Node 20 still has incorrect general beliefs about disappearance and scrolling; the figure should focus on floor contact, not claim complete collision or obstacle-avoidance learning. The `step` and temporal-delta machinery in this P is inactive/constant under evaluation and should not be highlighted.

## Other games screened

The combined survey covers all 15 candidate pools, 450 candidate records. This follow-up specifically inspected descendant B updates after potentially semantic P changes, including branches outside each saved lineage.

| Game | Result under the stricter criteria |
|---|---|
| `7www9` / Magnets | Strongest: active relations, explicit B references, and logged failure → success on the shown snap. |
| `dino` | Active boundary predicate plus equivalent B condition. Good local mechanic; literal flag use is unproven. |
| `egg` | P10 adds button/blob roles and geometry; descendant B14 explicitly refers to `agent:red`/`agent:pink` and corrects breakup's spurious horizontal shift. However, B still lacks the height condition, and does not explicitly use the new bounds/centroid in that rule. Weaker than Magnets. |
| `logic_gates` | P7 extracts six blocks; B21 refines conditional block toggles. P19 names blocks A–F, but B24 still refers to coordinates. These branches omit or misdescribe wire/background dynamics and do not show equally direct new-predicate use. |
| `SET` | P17 groups tiles; B25 revises clearing to “exactly three tiles containing gold.” P does not learn card attributes or a set-validity predicate. Weaker semantic change and no general SET-rule recovery. |
| `n2ntd` / Mario | Later `red_adj` is meaningful but has no subsequent B update using it. Other branches label an agent but do not solve the missing support/gravity coupling. |
| `bt3gb` / Ice | Later B14 corrects click behavior while using structured object roles, but also introduces a false persistent-covering rule. The existing touching-object P repair has no subsequent matching B update on the saved lineage. |
| `7xf97` / Grow | `gare` works as a visible adjacency predicate, but B never adopts it. No learned B gives the sun-overlap condition that suppresses growth. The prior geometry example is not evidence for this joint-learning claim. |
| `dq8gc` / Disease | B26 explicitly uses `sel`, but P's selection inference requires history/state persistence that evaluation does not supply. It cannot demonstrate a learned, functioning selection feature. |
| `colour_lines` | Movement-status proposals need history; every current cell is marked new in single-frame evaluation. B29 even misreads compact tokens as colors/statuses. |
| `diffusion` | Mainly encodings, background choices, or inactive temporal fields; no comparably clear relation-to-rule learning chain. |
| `eahcw` / Paint | Active-color recovery requires unavailable history; cursor/brush metadata is absent. |
| `f5w3n` / Space Invaders | Orange-agent role is useful; advertised off-screen memory does not execute. No stronger demonstrated semantic feature/B/dynamic chain. |
| `s2kt7` | Predominantly formatting and unavailable temporal deltas. |
| `va6fq` / Sand | Predominantly formatting, counters, and unavailable temporal deltas. |

## Verification and reproduction

The evaluator [strips metadata](../../../offline_learning/validate.py#L211) and [calls `perceive([raw_obs])` in a fresh namespace](../../../offline_learning/validate.py#L257). All outputs here use that exact function. Supplying the full frame history would falsely activate many attractive-looking learned fields and misrepresent this run.

Run:

```bash
.venv/bin/python offline_learning/scripts/audit_learning_evolution_joint.py
```

The script audits candidate metadata, checks ancestry, executes 24 P/frame pairs, verifies the six selected local transitions, matches cached prediction hashes, and regenerates the evidence JSON and PNG/PDF/SVG frame preview. No models were called or retrained. The paper's main TeX and existing diagrams were not changed.
