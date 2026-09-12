Learning-evolution candidates from the 2026-08-24 human-curated REx runs

**Follow-up correction:** For the stricter requirement that beliefs actually use a newly learned perceptual feature, see [the joint-learning audit](learning_evolution_joint_candidates.md). Magnets is the stronger verified example. The Grow geometry recommendation below does not establish belief use of `gare` or learning the sun-overlap growth condition.

I recommend **Ice’s touching gray objects** for the clearest substantive perception repair, **Grow** for explicit relational features, and **Egg** for a new, visually clear game example. Mario also has a useful later adjacency program, but that candidate was not the saved best model.

Reviewed the candidate pools for all 15 games (450 records), traced the ancestry of each saved P/B pair, and executed the shortlisted programs on their recorded frames. Here B means the `world_knowledge` field, called K in the current paper. Learning nodes/iterations and gameplay steps are different axes: each proposed diagram applies several learned P/B snapshots to the **same four recorded frames**.

The [frame contact sheet](../learning_evolution_candidates_frames.png) shows the four main sequences; a [PDF copy](../learning_evolution_candidates_frames.pdf) is also available. The [evidence JSON](learning_evolution_candidates.json) contains exact observations, actions, candidate code, verbatim beliefs, outputs, parent IDs, learning iterations, scores, and hashes. All 68 shortlisted node/frame executions completed without errors. The paper and existing figures were not edited.

| Priority | Game and frame window | Actions between frames | Suggested learning nodes | Substantive P change |
|---|---|---|---|---|
| 1 | Ice / `bt3gb`, `train_d0`, **205–208** | `down → noop → down` | **3 → 6 → 8 → 12** | Flat cells → fixed/movable roles → repair same-color object confusion |
| 2 | Grow / `7xf97`, `train_d0`, **34–37** | `left → left → left` | **2 → 10 → 11 → 23** | Flat cells → connected rectangles → explicit adjacency/edge predicates |
| 3 | Egg, `train_d1`, **213–216** | `click 0 0 → noop → noop` | **0 → 2 → 3 → 8** | Flat cells → explicit button state and per-color blob bounds |
| 4, conditional | Mario / `n2ntd`, `train_d0`, **22–25** | `right → noop → noop` | **2 → 3 → 16 → 29** | Explicit neighbors of the red player, including support and grid boundary; node 29 is not the saved best |

**A constraint that changes the shortlist.** The evaluator [executes P in a fresh namespace and calls `perceive([raw_obs])`](../../../offline_learning/validate.py#L257). [Observation metadata is stripped](../../../offline_learning/validate.py#L211). Therefore the apparent velocity, trajectory-delta, remembered-position, active-color, and step-counter improvements in many programs do not operate as their comments suggest. A figure must use the actual single-frame outputs. Supplying the full drive history just for the figure would show behavior that was not evaluated during learning. Several programs still make substantive *single-frame* changes alongside their inactive temporal code.

**1. Ice: learning which gray cells belong to the movable cloud.**

Source: [candidate pool](../../../logs/2026-08-24/human_curated/rexpure/bt3gb_s1/rexpure_run_seed1/candidates.jsonl), [training drive](../../../offline_learning/human_data/bt3gb/informative_curated/drives/train_d0/episode_0/trajectory.csv).

At steps 205–208, the fixed 2×2 object occupies rows 0–1, columns 0–1, and the cloud occupies row 0, columns 2–4. Both are gray, so their visible cells form one connected region. `down` creates a lightblue drop at **(1,3)**, directly under the cloud’s center. `noop` moves it to **(2,3)**; the next `down` creates another drop at **(1,3)**. This makes correct cloud identity relevant to a concrete transition: where water appears.

| Node | Learning iteration | Train score | P/B at this point |
|---|---:|---:|---|
| 3 | 3 | 0.100 | P lists individual colored cells; B empty. |
| 6 | 8 | 0.583 | Same P. B distinguishes the fixed 2×2 block and three-cell movable block, and describes spawning below the movable block’s middle column. |
| 8 | 14 | 0.602 | P introduces `fixed:`, `movable:`, `other:`. However, touching gray objects defeat its three-cell-run detector. |
| 12 | 22 | 0.725 | P excludes fixed columns 0–1 before identifying the movable run. B is unchanged from node 6. This is the saved best pair. |

The relevant **actual outputs on step 205** are:

```text
node 8:  movable:gray_row0:0,1,2,3,4
node 12: movable:gray_row0:2-4
```

Node 8 also duplicates columns 2–4 under `other:`; node 12 removes that misclassification. This is an object-assignment correction, beyond shorter formatting. A four-panel learning narrative can show **cells → beliefs naming objects → imperfect object extraction → corrected object extraction**. The complete ancestry is `0 → 3 → 4 → 6 → 8 → 12`.

An alternate four-frame window, **202–205**, uses `left → noop → noop` and shows the cloud moving from columns 3–5 into contact at columns 2–4. That window makes the perceptual ambiguity visually explicit; 205–208 better explains why resolving it matters. For the paper, a small enlargement of the top-left region would help because accumulated water occupies the lower half of the full frame.

Limits: this is a successful correction for **touching**, not a general solution to occlusion. At step 211, when the cloud actually overlaps the fixed block, node 12 reports only visible columns `2,3`. Its `step=0` field is constant and should not receive learning credit. Node 6 also introduced an incorrect inside-versus-outside click distinction, already exposed by the existing `ice_click` figure. The new sequence supports cloud identity and spawning; it does not establish that every final belief is correct.

**2. Grow: learn the spatial condition that distinguishes separation from contact.**

Source: [candidate pool](../../../logs/2026-08-24/human_curated/rexpure/7xf97_s1/rexpure_run_seed1/candidates.jsonl), [training drive](../../../offline_learning/human_data/7xf97/informative_curated/drives/train_d0/episode_0/trajectory.csv). This sequence is already represented by `learning_evolution_grow_cover`; it is stronger than Mario under the requested criterion.

The gray cloud moves left three times. Its columns change **5–8 → 4–7 → 3–6 → 2–5**. Visible gold initially spans columns 0–2 and finally spans 0–1 because the cloud covers its right edge. All these blocks occupy rows 0–2.

| Node | Iteration | Score | P/B evolution |
|---|---:|---:|---|
| 2 | 3 | 0.333 | Individual colored-cell coordinates; B empty. |
| 10 | 12 | 0.625 | Same P; B now describes gray/gold blocks and rules conditioned on their relative position. |
| 11 | 13 | 0.578 | P extracts same-color connected components and represents solid rectangles. B unchanged. |
| 23 | 27 | 0.652 | P explicitly computes existence, edge contact, and gold–gray adjacency flags. B unchanged. Saved best. |

Node 23’s **`gare` changes `0 → 0 → 1 → 1`** across these frames. The implemented condition is `gold_right + 1 == gray_left`: visible gold immediately left of gray. This is a derived relational feature a reader can connect to the image. Use the implemented condition to explain the abbreviation; the program’s prose naming of left/right adjacency is confusing.

The full lineage is `0 → 2 → 3 → 7 → 10 → 11 → 14 → 19 → 23`. Node 19 returns to coordinate ranges before node 23 adds flags. Thus the representation does not evolve monotonically toward object abstraction; the table deliberately shows the logged score decrease at node 11. If four stages including a blank seed are preferred, the existing choice `0 → 2 → 10 → 23` is already defensible.

Limits: `uc`, intended to mean unchanged from the preceding observation, is always zero under the evaluator. The useful new features are the spatial flags. Describe the last transition as **visible gold being covered**, not physical destruction or shrinkage of the sun. The learned B contains imperfect conversion/expansion explanations; the flags do not demonstrate recovery of hidden object extent.

**3. Egg: a new example with a visible change of object dynamics.**

Source: [candidate pool](../../../logs/2026-08-24/human_curated/rexpure/egg_s1/rexpure_run_seed1/candidates.jsonl), [training drive](../../../offline_learning/human_data/egg/informative_curated/drives/train_d1/episode_0/trajectory.csv).

At **213**, the red button is at (0,0), and the tan egg occupies a 5×5 bounding box at rows 7–11, columns 6–10. Clicking the button turns it pink and replaces the egg with gold particles one row lower. The next two noops spread those particles sideways and downward. The original shape and its breakup are easy to see in four frames.

Use `0 → 2 → 3 → 8`, or the three substantive checkpoints `2 → 3 → 8`:

| Node | Iteration | Score | P/B evolution |
|---|---:|---:|---|
| 0 | 0 | 0.000 | Empty P and B. |
| 2 | 3 | 0.633 | Flat non-background cell list. |
| 3 | 4 | 0.704 | Same P; B distinguishes the special button, controllable tan blob, and autonomously changing gold blob. |
| 8 | 13 | 0.865 | Explicit `(0,0)` color plus per-color blob bounding boxes, followed by cell coordinates. B unchanged from node 3. Saved best. |

Node 8 outputs these useful geometric summaries:

```text
213: (0,0)=red;  tan_blob:  rows 7-11, cols 6-10
214: (0,0)=pink; gold_blob: rows 8-12, cols 6-10
215: (0,0)=pink; gold_blob: rows 8-13, cols 5-11
216: (0,0)=pink; gold_blob: rows 8-14, cols 5-12
```

The complete lineage is `0 → 2 → 3 → 5 → 8`. The immediate P update is **node 5 → 8: 0.787 → 0.865**. Node 5 merely adds dimensions/reformats cells; node 8 introduces the substantive geometry.

This supports a narrative of **learning a description at the scale used by the beliefs**. It does not recover the full particle dynamics. Node 8 groups *all cells of a color* into one blob, including disconnected gold particles. Its centroid-delta code never executes with a single-frame history. B correctly distinguishes controlled tan behavior from autonomous gold behavior, but wrongly describes universal down-right motion and misses the height condition for breakup. The chosen frames themselves refute the claimed one-column-right shift. If the diagram must show a fully corrected final B, Egg is a weaker choice.

**4. Mario can be improved, with an explicit selection caveat.**

Source: [candidate pool](../../../logs/2026-08-24/human_curated/rexpure/n2ntd_s1/rexpure_run_seed1/candidates.jsonl), [training drive](../../../offline_learning/human_data/n2ntd/informative_curated/drives/train_d0/episode_0/trajectory.csv).

The existing ledge window **22–25** is particularly well matched to **node 29**, proposed at iteration 35. It adds the player’s four neighboring colors while retaining coordinates. Across the four frames, its actual downward-neighbor feature is:

```text
step 22: d=darkorange   # supported by platform
step 23: d=white        # stepped off the ledge
step 24: d=white        # still falling
step 25: d=none         # bottom boundary
```

A diagram using **2 → 3 → 16 → 29** would show cell perception, initial beliefs, repaired gravity beliefs, and then explicit perceptual support/boundary information. Node 29 is a direct child of node 16 and inherits its beliefs unchanged. This is a substantive P change aligned with the gravity conditions.

However, **node 29 scores 0.762 versus node 16’s 0.790**. It is an explored descendant of the saved best model, not the shipped artifact. This is suitable for a search-evolution figure only if that fact is explicit; it cannot be presented as the final best model improving.

There is also a substantive P on the saved model’s ancestry that the current figure omits: **node 6**, iteration 7, score **0.751**. It replaces cells with same-color connected-component bounding boxes, e.g. `blue(0,0)3x2` and `darkorange(10,0)3x1`. A sequence `2 → 3 → 6 → 16` can show that exploration, but node 15 later returns to cell coordinates, inherited by node 16. Its velocity fields never activate. This is evidence of a transient object abstraction, not a retained final one.

**Coverage of the remaining games.** These were screened for substantive P changes both on and outside the saved lineage. They are lower-priority choices for this particular figure.

| Game | Finding |
|---|---|
| `7www9` | Saved lineage mainly adds metadata/hash to cells. Side node 6 computes same-row blue-to-red distance, but scores 0.776 versus parent 1’s 0.843; tracking proposals have single-frame limitations. |
| `SET` | Saved P removes then restores white tile cells, plus an unavailable cursor field. Tile-structured side branches exist (e.g. nodes 19/22), but are not the saved pair and have a less direct four-frame learning story. |
| `colour_lines` | Saved node 12 advertises moves/additions/removals, but those require previous observations and never appear under the evaluator. |
| `diffusion` | Saved lineage changes which background cells are represented and how cells are grouped; the motion proposals require history. Weaker semantic progression than the shortlist. |
| `dino` | Saved changes are predominantly encoding/grouping. Side node 28 learns connected components and relative shape offsets (0.885), below saved node 18 (0.936). A reserve object-abstraction example. |
| `dq8gc` | Saved P adds background/hash/dimensions. Active-cell/stack proposals largely require unavailable history; the existing Disease example is principally belief evolution. |
| `eahcw` | Saved late P advertises active-color inference, but outputs `AC:?` with one observation. It does not demonstrate recovery of active color. |
| `f5w3n` | Orange-agent labeling is useful, but the later advertised off-grid memory never runs. No demonstrated remembered off-screen state in the evaluated P. |
| `logic_gates` | Saved node 17 retains the initial cell P. Side branches extract blocks, but score below the saved pair; e.g. node 22 scores 0.249 versus parent 17’s 0.285. |
| `s2kt7` | Deltas/history counters/hash dominate the revisions. Temporal fields are inactive or constant. |
| `va6fq` | Similar delta/counter proposals; no compelling retained object/relational extraction under single-frame execution. |

Scores above are the logged whole-training-objective values, not accuracies on these four frames. Their changes support candidate selection, but do not isolate the causal effect of any one feature. No learner or game code was modified or retrained.
