**Sequential audit of learning-evolution examples — 2026-09-11**

**SET is the cleanest additional example.** It combines a real card abstraction, a subsequent belief update that counts selected cards, and a visible invalid-selection reset. Magnets remains the clearest explicit relational-predicate example. Disease and Logic Gates also demonstrate operational use of learned perception, but their beliefs retain important errors. Mario has a meaningful later perception change, with belief learning preceding it.

This review restarted from the logs and proceeded one game at a time, without subagents. All 15 games and all 450 candidate records were examined. The review order was Disease, Egg, Logic Gates, SET, Mario, Ice, Space Invaders, Colour Lines, Diffusion, Paint, Ants, Sand, Magnets, Dino, Grow.

[Frame preview (PNG)](/home/ays57/bai/analysis/learning_example/learning_evolution_sequential_candidates.png) · [PDF](/home/ays57/bai/analysis/learning_example/learning_evolution_sequential_candidates.pdf) · [SVG](/home/ays57/bai/analysis/learning_example/learning_evolution_sequential_candidates.svg) · [Complete audit index](/home/ays57/bai/analysis/learning_example/audit/learning_evolution_sequential_audit_index.json)

**How to interpret the evidence.** A learning checkpoint is an actual stored `(P, B)` pair, where B is `world_knowledge`. Candidate numbers and learning iterations differ; iterations come from `process_log.jsonl`'s `i` field. Each proposed sequence follows one ancestry, so condensing updates does not combine sibling candidates. All selected checkpoints were accepted into the search; most are explored branches rather than the saved best branch. Scores below are whole-training scores, not accuracies on the illustrated frames.

P was executed exactly as in [run_perceive](/home/ays57/bai/offline_learning/validate.py:257): a fresh namespace and `perceive([current_frame])`, after removing harness metadata. History-dependent movement, cursor, selection memory, and counters therefore do not establish active semantic learning. Frame t's action leads to frame t+1. Cached CFD means **contrastive next-frame selection**, not unconstrained forward simulation. A correct option alone is insufficient: the reasoning must also match the claimed mechanism. Repeated transition hashes can have different surrounding contexts and cached results.

The reproducible verification reran 101 selected P/frame pairs, matched 35 cached records to their source lines, and checked 23 local transitions across the main and supplementary windows. Local checks formalize only the stated belief clauses; they do not validate entire natural-language world models or prove that a feature is causally necessary.

**SET — recommended new example: object segmentation enables a count-based reset rule.**

Use `train_d1`, frames **187–190**, with actions `noop → click 4 10 → noop`. Two cards are already selected; the click selects a third card in a different row/column arrangement; the following noop clears all gold borders while retaining all nine cards. The selected triple has two seagreen cards and one coral card, so it is invalid.

| Candidate | Learning iteration | Training score | Parent |
|---|---:|---:|---:|
| [candidate 15](/home/ays57/bai/logs/2026-08-24/human_curated/rexpure/SET_s1/rexpure_run_seed1/candidates.jsonl:16) | 22 | 0.2520 | 10 |
| [candidate 17](/home/ays57/bai/logs/2026-08-24/human_curated/rexpure/SET_s1/rexpure_run_seed1/candidates.jsonl:18) | 27 | 0.2728 | 15 |
| [candidate 25](/home/ays57/bai/logs/2026-08-24/human_curated/rexpure/SET_s1/rexpure_run_seed1/candidates.jsonl:26) | 44 | 0.3173 | 17 |

At node 15, P lists individual cells. Node 17 infers occupied row/column groups and emits nine 5×3 card records, each as `T(row-range,col-range):15-character-pattern`. This is a card-level abstraction; it does not yet compute card attributes, validity, or a selected-count predicate. Gold is encoded as `g`, white as `w`, seagreen as `s`, and coral as `c`.

For example, the card at `T(1-5,8-10)` changes from `wwwwswwswwswwww` at frames 187/188 to `ggggsggsggsgggg` at 189 and back at 190. Counting the emitted card groups containing `g` gives **2 → 2 → 3 → 0**. That count is an interpretation of P's output, not a field emitted by P.

Node 25 retains P17 and replaces the old rule requiring a complete gold row/column with this actual learned statement:

> A noop action does nothing unless exactly three tiles on the grid contain at least one gold cell. In that case, every gold cell on the grid becomes white (all gold is removed).

At the reset transition 189→190, cached CFD changes **0 → 0 → 1** at nodes 15, 17, 25. The final trace identifies the three `T(...)` groups and applies the count condition; inverse-action prediction was already correct at all three checkpoints. See [prediction line 941](/home/ays57/bai/logs/2026-08-24/human_curated/rexpure/SET_s1/rexpure_run_seed1/predictions.jsonl:941), [prediction line 1061](/home/ays57/bai/logs/2026-08-24/human_curated/rexpure/SET_s1/rexpure_run_seed1/predictions.jsonl:1061), and [prediction line 1541](/home/ays57/bai/logs/2026-08-24/human_curated/rexpure/SET_s1/rexpure_run_seed1/predictions.jsonl:1541). The final rule matches every cell of all three displayed transitions.

For a paper matrix, use three learning rows: nodes **15 → 17 → 25**. Show the three relevant card records in full, abbreviate the other six as unchanged, and highlight the new `T(...)` grouping and the revised “exactly three” clause. Do not add a fictitious `valid_set` or `selected_count` feature. This example demonstrates invalid-selection reset, not learning the complete SET validity rule: actual valid triples remove cards, which B25 does not describe. Saved best is node 12 on a different branch. [Full P/B, frame outputs, and cached evidence](/home/ays57/bai/analysis/learning_example/audit/learning_evolution_sequential_audit/set.json)

**Magnets — strongest explicit predicate-to-rule connection.**

Use `train_d1`, frames **32–35**, actions `down → noop → down`. Blue cells start at `(7,5),(8,5)`, move diagonally to `(8,6),(9,6)`, remain there, then move straight down to `(9,6),(10,6)`. Red cells remain at `(7,7),(8,7)`.

| Candidate | Learning iteration | Training score | Parent |
|---|---:|---:|---:|
| [candidate 1](/home/ays57/bai/logs/2026-08-24/human_curated/rexpure/7www9_s1/rexpure_run_seed1/candidates.jsonl:2) | 1 | 0.8426 | 0 |
| [candidate 6](/home/ays57/bai/logs/2026-08-24/human_curated/rexpure/7www9_s1/rexpure_run_seed1/candidates.jsonl:7) | 6 | 0.7755 | 1 |
| [candidate 8](/home/ays57/bai/logs/2026-08-24/human_curated/rexpure/7www9_s1/rexpure_run_seed1/candidates.jsonl:9) | 8 | 0.7947 | 6 |

P6 adds the distance to the nearest red cell in the same row (`rDistN`) or `noRedRow` when there is none. The blue-cell tags evolve **rDist2/rDist2 → rDist1/noRedRow → rDist1/noRedRow → noRedRow/noRedRow**. B8 explicitly branches on those tags and describes their updates. At the diagonal transition 32→33, nodes 1 and 6 infer `right` and fail CFD; node 8 infers `down`, succeeds at CFD, and explicitly reasons with `rDist` and `noRedRow`. See [prediction line 154](/home/ays57/bai/logs/2026-08-24/human_curated/rexpure/7www9_s1/rexpure_run_seed1/predictions.jsonl:154), [prediction line 334](/home/ays57/bai/logs/2026-08-24/human_curated/rexpure/7www9_s1/rexpure_run_seed1/predictions.jsonl:334), and [prediction line 574](/home/ays57/bai/logs/2026-08-24/human_curated/rexpure/7www9_s1/rexpure_run_seed1/predictions.jsonl:574).

Use three learning rows, **1 → 6 → 8**. All three visible transitions match B8's local clauses. However, B8 assigns motion to individual blue cells in a way that does not preserve the true rigid pair identity, lacks a general distance/pole model, and incorrectly predicts some nearby blocked horizontal moves. Claim learned relational tags and locally useful rules, not correct magnet physics. Saved best is node 14 on another branch. [Full P/B, frame outputs, and cached evidence](/home/ays57/bai/analysis/learning_example/audit/learning_evolution_sequential_audit/magnets.json)

**Disease — direct neighbor-feature adoption, with a mistaken interpretation.**

Use `train_d0`, frames **20–23**, actions `noop → right → noop`. The darkgreen cell moves from `(2,3)` to `(2,4)`, becoming adjacent to gray `(3,4)`. On the next noop, `(3,4)` becomes darkgreen. The initial noop demonstrates that separated cells do not change.

| Candidate | Learning iteration | Training score | Parent |
|---|---:|---:|---:|
| [candidate 1](/home/ays57/bai/logs/2026-08-24/human_curated/rexpure/dq8gc_s1/rexpure_run_seed1/candidates.jsonl:2) | 2 | 0.5584 | 0 |
| [candidate 8](/home/ays57/bai/logs/2026-08-24/human_curated/rexpure/dq8gc_s1/rexpure_run_seed1/candidates.jsonl:9) | 13 | 0.4742 | 1 |
| [candidate 10](/home/ays57/bai/logs/2026-08-24/human_curated/rexpure/dq8gc_s1/rexpure_run_seed1/candidates.jsonl:11) | 16 | 0.7153 | 8 |
| [candidate 17](/home/ays57/bai/logs/2026-08-24/human_curated/rexpure/dq8gc_s1/rexpure_run_seed1/candidates.jsonl:18) | 25 | 0.7106 | 15 |
| [candidate 23](/home/ays57/bai/logs/2026-08-24/human_curated/rexpure/dq8gc_s1/rexpure_run_seed1/candidates.jsonl:24) | 32 | 0.6484 | 21 |

P1 gives bare coordinates/colors. P8 adds counts of orthogonally adjacent colors: at frame 22, `2,4:darkgreen[1gray]` and `3,4:gray[1darkgreen]`. B10 uses those new bracket fields to predict both cells becoming darkgreen on noop. Its cached infection prediction becomes correct, but it incorrectly says the approach step exchanges the cells' base colors.

P17 adds directional neighbors, such as `2,4:darkgreen S:gray` and `3,4:gray N:darkgreen`. Node 23 uses a descendant P with the same relevant directional outputs, and its updated B corrects the approach rule to retain base colors. At frame 23, the corresponding excerpts become `2,4:darkgreen S:darkgreen` and `3,4:darkgreen N:darkgreen`. B23's local clauses match all three visible transitions. Infection CFD is **0, 0, 1, 1, 1** at the five listed checkpoints; see [prediction line 94](/home/ays57/bai/logs/2026-08-24/human_curated/rexpure/dq8gc_s1/rexpure_run_seed1/predictions.jsonl:94), [prediction line 514](/home/ays57/bai/logs/2026-08-24/human_curated/rexpure/dq8gc_s1/rexpure_run_seed1/predictions.jsonl:514), [prediction line 754](/home/ays57/bai/logs/2026-08-24/human_curated/rexpure/dq8gc_s1/rexpure_run_seed1/predictions.jsonl:754), [prediction line 1054](/home/ays57/bai/logs/2026-08-24/human_curated/rexpure/dq8gc_s1/rexpure_run_seed1/predictions.jsonl:1054), and [prediction line 1354](/home/ays57/bai/logs/2026-08-24/human_curated/rexpure/dq8gc_s1/rexpure_run_seed1/predictions.jsonl:1354).

For a four-row matrix, use **1 → 8 → 10 → 23**. The final row condenses later P and B changes while preserving the actual stored pair and ancestry. Show only the interacting cells and abbreviate the three distant gray cells. Crucially, B calls the neighbor fields **“stacks”**, even though P computes adjacency. Preserve that wording in quoted beliefs and explain the error; presenting them as a correct physical stack model or a fully correct disease theory would be misleading. P's `active:none` and history fields do not supply active-object tracking under evaluation. Saved best is node 29 on another branch. [Full P/B, frame outputs, and cached evidence](/home/ays57/bai/analysis/learning_example/audit/learning_evolution_sequential_audit/disease.json)

**Logic Gates — secondary object-abstraction case.**

Use `train_d0`, frames **58–61**, actions `noop → click 13 4 → noop`.

| Candidate | Learning iteration | Training score | Parent |
|---|---:|---:|---:|
| [candidate 3](/home/ays57/bai/logs/2026-08-24/human_curated/rexpure/logic_gates_s1/rexpure_run_seed1/candidates.jsonl:4) | 4 | 0.2242 | 2 |
| [candidate 7](/home/ays57/bai/logs/2026-08-24/human_curated/rexpure/logic_gates_s1/rexpure_run_seed1/candidates.jsonl:8) | 9 | 0.2034 | 3 |
| [candidate 21](/home/ays57/bai/logs/2026-08-24/human_curated/rexpure/logic_gates_s1/rexpure_run_seed1/candidates.jsonl:22) | 24 | 0.2223 | 7 |

P7 replaces the cells belonging to six 2×2 blocks with named block colors, e.g. `b12_4:pink`, `b12_19:red`, `b4_12:darkblue`. B21 updates the left-switch rule: when the right switch is red, toggle blocks `(4,12),(16,12),(20,12)`; when it is pink, toggle `(8,12),(16,12),(20,12)`. The clicked switch also toggles itself. The illustrated click occurs with the right switch red and correctly produces `b12_4:red`, `b4_12:orange`, `b16_12:darkblue`, `b20_12:darkblue`, with `b8_12:orange` unchanged.

Use **3 → 7 → 21**. At 59→60, cached CFD is **0 → 0 → 1** and the final trace applies the correct block condition: [prediction line 260](/home/ays57/bai/logs/2026-08-24/human_curated/rexpure/logic_gates_s1/rexpure_run_seed1/predictions.jsonl:260), [prediction line 440](/home/ays57/bai/logs/2026-08-24/human_curated/rexpure/logic_gates_s1/rexpure_run_seed1/predictions.jsonl:440), [prediction line 1220](/home/ays57/bai/logs/2026-08-24/human_curated/rexpure/logic_gates_s1/rexpure_run_seed1/predictions.jsonl:1220). The final inverse-action answer is a different cell in the same switch and is scored incorrect, so do not claim an inverse improvement. All six block colors match across the window, but the wire cells at 60→61 are not explained correctly by B21's global-toggle rule. Neither P nor B learns general AND/OR gate semantics. Saved best is node 17 on a different branch. [Full P/B, frame outputs, and cached evidence](/home/ays57/bai/analysis/learning_example/audit/learning_evolution_sequential_audit/logic_gates.json)

**Mario — meaningful P change after the belief update.**

Use `train_d2`, frames **223–226**, actions `left → noop → noop`, nodes **15 (iteration 18) → 16 (20) → 29 (35)**. Red moves `(8,3) → (8,2) → (9,2) → (9,2)`: left, fall, then stay on a platform. B16 learns that noop moves red down only when the cell below is empty. P29 subsequently adds current neighbor colors in `red_adj`; the downward neighbor changes **white → white → darkorange → darkorange**. This is a meaningful alternative to the earlier formatting-only Mario example.

The ordering matters: B16 already solves the landing before P29 is added. Cached landing CFD is **0 → 1 → 1** at nodes 15, 16, 29; node 29's trace constructs the new `red_adj` output, but B never names that field and can use retained cell coordinates. This supports later perception exposing a useful existing condition, not proof that a later B learned to consume the new predicate. Node 16 is the saved best; node 29 is its lower-scoring child. [Full P/B, frame outputs, and cached evidence](/home/ays57/bai/analysis/learning_example/audit/learning_evolution_sequential_audit/mario.json)

**Dino — secondary semantic match without explicit flag adoption.**

Use `train_d1`, frames **100–103**, actions `noop → noop → up`, nodes **12 (iteration 13) → 19 (30) → 20 (32)**. Red falls to the floor, stays grounded while obstacles scroll, then jumps six rows upward. P19 adds `red:bottom`; B20 changes unconditional noop falling to no movement when the maximum red row is 19. This is the same geometric condition, but B and its traces use coordinates rather than naming the flag.

The only cached transition in this main window is the jump, already correct at all three nodes. Separate cached grounding records improve: at frame 1→2, CFD **0 → 0 → 1** ([prediction line 798](/home/ays57/bai/logs/2026-08-24/human_curated/rexpure/dino_s1/rexpure_run_seed1/predictions.jsonl:798), [prediction line 1158](/home/ays57/bai/logs/2026-08-24/human_curated/rexpure/dino_s1/rexpure_run_seed1/predictions.jsonl:1158), [prediction line 1278](/home/ays57/bai/logs/2026-08-24/human_curated/rexpure/dino_s1/rexpure_run_seed1/predictions.jsonl:1278)); this identical transition occurs in all four training drives. A supplementary `train_d1` window **8–11** includes grounding, jumping, and falling; the inverse answer for 8→9 improves from `up` at node 19 to `noop` at node 20, while CFD was already correct there. Saved best is node 18 on another branch. [Full P/B, frame outputs, and cached evidence](/home/ays57/bai/analysis/learning_example/audit/learning_evolution_sequential_audit/dino.json)

**Egg — weaker role abstraction, without the needed height condition.**

Use `train_d1`, frames **212–214**, actions `noop → click 0 0`, nodes **9 (iteration 14) → 10 (19) → 14 (24)**. P10 separates the red/pink button as `agent` from the tan/gold `blob`, with blob bounds and centroid. B14 adopts the agent representation and corrects breakup motion from down-and-right to down only. The red button becomes pink and the tan shape becomes gold one row lower. The two visible transitions match the local rule.

There are no cached predictions for these two transitions. B does not use the new bounds/centroid to learn the height-dependent breakup condition, and its model of the subsequent liquid spreading remains wrong. This is a role-recognition example, not a learned conditional physics example. Saved best is node 8 on another branch. [Full P/B, frame outputs, and cached evidence](/home/ays57/bai/analysis/learning_example/audit/learning_evolution_sequential_audit/egg.json)

**Remaining games and exclusion reasons.**

The following findings are limited to these runs and the actual evaluation contract; they do not assert that the games lack interesting dynamics.

| Game | Why it does not provide a stronger qualifying example | Evidence |
|---|---|---|
| Ice | P8 separates fixed celestial block and movable cloud; B14 uses the roles and improves a click prediction. Its permanent-overlap interpretation is wrong, and the P12 repair is on a sibling branch. No learned liquid/solid interaction rule. | [review and source records](/home/ays57/bai/analysis/learning_example/audit/learning_evolution_sequential_audit/bt3gb_screen.json) |
| Space Invaders | P incorrectly calls orange the agent; orange is an enemy bullet, while the controlled player is gray. Temporal tracking is inactive; later blue/static and red/falling labels are semantically wrong. | [review and source records](/home/ays57/bai/analysis/learning_example/audit/learning_evolution_sequential_audit/f5w3n_screen.json) |
| Colour Lines | Motion/identity fields have zero velocities and all cells marked new under single-frame evaluation. B29 misdecodes the status/color symbols. Active color counts have no useful descendant B adoption. | [review and source records](/home/ays57/bai/analysis/learning_example/audit/learning_evolution_sequential_audit/colour_lines_screen.json) |
| Diffusion | P11/P27 compute active white-row features, but their branch has empty B throughout. Membrane rules occur on different branches with flat coordinates. An actual open-membrane crossing exists, but not the required joint evolution. | [review and source records](/home/ays57/bai/analysis/learning_example/audit/learning_evolution_sequential_audit/diffusion_screen.json) |
| Paint | Brush/cursor fields are missing, unknown, or constant: P11 defaults to red; later AC/PH fields are unknown and STEP is zero. Beliefs do not acquire a useful active P feature. | [review and source records](/home/ays57/bai/analysis/learning_example/audit/learning_evolution_sequential_audit/eahcw_screen.json) |
| Ants | P6 grid hash is referenced as a state ID by B8, but this is a fingerprint rather than a learned direction/neighborhood relation. Temporal deltas are inactive. | [review and source records](/home/ays57/bai/analysis/learning_example/audit/learning_evolution_sequential_audit/s2kt7_screen.json) |
| Sand | P changes concern serialization, color/row grouping, dimensions, and constant counters. B24 learns some falling/blockage rules from coordinates; no new support or material predicate is learned in P. | [review and source records](/home/ays57/bai/analysis/learning_example/audit/learning_evolution_sequential_audit/va6fq_screen.json) |
| Grow | P23 adds real gold/gray adjacency and boundary flags, but no B names gare or the boundary flags. P23/B10 and P23/B24 cached reasoning never mentions gare. Other branches adopt component/range notation but treat leaves as static; no belief learns sun-overlap suppression of rain-induced growth. | [review and source records](/home/ays57/bai/analysis/learning_example/audit/learning_evolution_sequential_audit/7xf97_screen.json) |

For Grow specifically, `gare` is **gold's right edge + 1 = gray's left edge**, based on visible cells. It is not a sun-covered predicate. The game program requires both rain/green-leaf contact and **no sun/cloud overlap** for growth; the learned beliefs do not acquire that conjunction. A B clause about adjacent gold/gray columns or a copied representation is not evidence of learning the growth condition. See [Grow game program](/home/ays57/bai/autumn_programs/grow.sexp).

**Reproduction and figure scope.**

Run:

```bash
.venv/bin/python offline_learning/scripts/verify_learning_evolution_sequential.py
```

The [verification script](/home/ays57/bai/offline_learning/scripts/verify_learning_evolution_sequential.py) reads the original candidates, process records, predictions, and trajectories. It verifies stored evidence and regenerates the index and frame preview. The JSON files retain full P code, full B text, all displayed P(X) outputs, exact source-line references, and candidate ancestry. The preview is a selection aid; it is not yet a final paper matrix with every learning row. The existing Ice and Grow diagrams and `paper/main.tex` were left as they were.
