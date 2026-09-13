# Why incumbent information-extraction curves rise and fall

The metric conflates semantic structure, textual encoding, redundant fields, and output length. The observed changes do not support a universal direction of abstraction or information gain. Some rising curves add useful-looking spatial features, some add only formatting or inactive-history artifacts, and some falling curves preserve more distinctions than the earlier program.

## Verification and provenance

I traced actual incumbent records, rather than arbitrary candidate proposals. Source records are in `logs/2026-08-24/human_curated/rexpure/<game>_s1/rexpure_run_seed1/candidates.jsonl` (node N is JSONL line N+1). Iterations below were cross-checked in `process_log.jsonl`. A newly best candidate need not descend from the previously best candidate, and the plot can change when a K-only proposal brings a different branch's P into the incumbent position.

`reproduce.py` executes 33 selected programs from 11 games on both cached training and test corpora. All 66 computed ratios agree with `metrics.csv` within 1e-12. `semantic_outputs.json` stores measured lengths, ratios, and same-frame output examples. `field_checks.py` and `field_checks.json` provide explicit field-removal ablations, collision examples, and exact grid decoding tests. These are byte-level/local-program checks, not rescoring experiments with a language model.

The scoring runtime executes each program afresh and calls `perceive([raw_obs])`, a single-frame history (`offline_learning/validate.py:257`); global counters reset and most history-based delta code is inactive. The metric scores distinct frames belonging to the train/test transition pairs; decoy frames are excluded from the ratio (`offline_learning/scripts/perception_metrics.py:360`).

## Six informative cases

| Game | Actual incumbent checkpoints | Train ratio | What the source and same-frame execution show |
|---|---|---|---|
| Mario | node 1, iteration 1 → node 3, iteration 4 → node 6, iteration 7 → node 16, iteration 20 | 1.2976 → 1.4104 → 1.2256 → 1.1602 | Initial agent heuristic discards the chosen cell's color. Final grouped coordinates preserve all colors while removing repeated color names. Intermediate node 6 uses connected-component boxes; final node 16 returns to individual cells grouped by color. |
| Space Invaders | node 1, iteration 3 → node 16, iteration 25 → node 20, iteration 30 | 0.9989 → 1.3600 → 0.7651 | A verbose intermediate header and counts are removed; final code uses abbreviated agent notation and one-character colors. Exact raw grids remain reconstructable for every scored train/test frame given fixed grid dimensions/background. |
| Ants | node 1, iteration 2 → node 4, iteration 12 → node 23, iteration 35 | 0.6730 → 0.9465 → 1.5881 | Final rise comes from a state hash, constant step=0, a duplicate cell list under `added`, and constant empty delta fields. It does not add object or relational predicates. |
| Egg | node 5, iteration 7 → node 8, iteration 13 | 1.4504 → 1.9621 | Final code retains every cell and additionally prints the control cell `(0,0)` and per-color bounding boxes. This is a real derived spatial description, redundant with the full cell list. |
| Grow | node 1, iteration 1 → node 23, iteration 27 | 1.2557 → 1.4831 | Final code groups contiguous coordinates into row ranges and adds existence, edge-contact, and bounding-column-adjacency flags. Output gets shorter while compressed output gets larger. |
| Diffusion | node 2, iteration 3 → node 18, iteration 28 → node 21, iteration 31 → node 24, iteration 34 → node 28, iteration 38 | 1.2082 → 2.3908 → 1.1708 → 1.0199 → 1.2372 | Spike enumerates all background coordinates. Subsequent incumbents omit background cells; final rise adds background identity and dimensions. |

### Mario: lower ratio, more distinctions

Initial node 1 gives the same `agent:9,1` output for recorded frames 1 and 12, although that cell changes from red to mediumpurple. It also conflates three frames whose chosen cell is red, mediumpurple, or gold at (5,9). Final node 16 emits the specific color, e.g. `red: 9,1` versus `mediumpurple: 9,1`.

Across training frames, distinct outputs increase from 99/103 to 103/103; test frames increase from 90/91 to 91/91. Meanwhile mean output characters decrease from 242.85 to 121.49 and mean compressed bytes decrease from 122.77 to 109.51. This is direct evidence against reading a falling curve as loss of information. Final grouped coordinates are implemented in `n2ntd/node_16.py:50`; exact collision pairs are in `field_checks.json`.

The final node 16 is a K-only update of parent node 15. Its representation was already present in node 15; the final incumbent score increase cannot be attributed solely to a new P edit. K grows from 832 characters at node 6 to 1696 at node 16.

### Space Invaders: compact recoding, not loss of state

For the same first recorded training frame, node 16 emits:

`bg=black rows=16 cols=16 | agent at (15,14) | colors: blue:1 gray:1 | others: (1,13,blue); (15,8,gray)`

Node 20 emits:

`a:15,14 | cells:1,13,u;15,8,g`

Mean output characters fall from 193.53 to 77.93 between those incumbents, and compressed bytes from 129.04 to 72.99. Node 20 shortens color labels and omits invariant headers and redundant counts (`f5w3n/node_20.py:143`). My independent decoder recovers all 110 training and 94 test grids exactly, using the known 16×16 black-background convention and its color dictionary; no truncation or unknown colors occur. Its history-based off-grid recovery branch is inactive under single-frame evaluation. Thus this curve can fall through recoding rather than removal of decision-relevant information. K also differs between these two incumbents, so score changes do not isolate encoding utility.

### Ants: a large rise explained entirely by artifacts

Node 23 on every scored frame emits `step:0`, a SHA-256-derived 8-digit state ID, `cells:...`, `added:...` containing an exact second copy of those cells, `removed:empty`, and `changed:empty`. Since `prev_set` is empty without history, every current cell is treated as added (`s2kt7/node_23.py:82`, `:94`, `:113`). The labels do not report genuine transitions.

Removing step, hash, and delta fields yields output byte-identical to node 4 on every train and test frame. Training ratio returns exactly from 1.588080 to 0.946501. Removing only the hash changes it to 1.430104; removing only delta fields gives 1.228395. These effects need not add linearly under gzip.

Node 4 and node 23 carry the same 912-character K; scores are 0.3451 and 0.3768. The presence of a somewhat higher objective score does not establish that hash or duplicate fields supply useful world semantics.

### Egg: explicit spatial summaries can increase bytes

Node 5 emits dimensions, background, and all cells. Node 8 adds `(0,0)=red` and, for example, `red_blob: rows 0-0, cols 0-0; tan_blob: rows 0-4, cols 0-4` (`egg/node_8.py:98`). The same full coordinate list remains. Removing these additions yields node 5 exactly on all scored frames.

The `(0,0)` control and tan/gold blob vocabulary correspond to concepts already described by K, which is unchanged (1313 characters). The higher ratio has a plausible abstraction interpretation: readily usable summaries of a role and spatial extent are made explicit, at extra code length. However, same-color cells are merged even when disconnected; these are per-color bounding boxes, not general object tracking. No centroid or temporal delta is ever emitted in this evaluation, despite substantial source code for that feature. Utility of each added field has not been causally tested, and K's mechanics retain errors.

### Grow: shorter text can cost more after compression

From node 1 to node 23, mean output falls from 283.49 to 194.25 characters while mean compressed output rises from 116.23 to 136.74 bytes. Cell sequences such as `0,0:gold 0,1:gold 0,2:gold ...` become `gold:0,0-2;1,0-2;2,0-2`, alongside a new header and flags such as `gx`, `gle`, `gare` (`7xf97/node_23.py:122`). Repeated color strings were inexpensive for gzip; the new field names and derived values have different coding costs.

All eight spatial/existence flags take both 0 and 1 on the training corpus. `uc`, the unchanged-history flag, is always 0. The adjacency predicates compare extreme columns and do not check vertical overlap, so their operational definition should be stated carefully. K is unchanged between prior incumbent node 10 (iteration 12, ratio 1.2467) and final node 23, and contains spatial rules, but does not explicitly reference the abbreviated new flag names. This is evidence that the program materializes derived spatial features; their contribution to predictive performance remains unmeasured.

### Diffusion: the spike is background enumeration

Node 18 lists every coordinate of every color, including the dominant background (`diffusion/node_18.py:45`). For a 9×9 frame it begins with a long `black:0,0 0,2 0,5 ...` list. It emits a mean 347 characters and 190.72 gzip bytes. Node 21 removes the background and groups columns by row, reducing these to 99.02 characters and 96.36 bytes.

The final 24→28 increase from 1.0199 to 1.2372 is exactly the addition of `bg:<color>; dim:9x9;`: stripping this header reproduces node 24 on every train/test frame. Background identity is not constant here: it is black or blue on the training corpus and black or red on test. Thus the header can disambiguate an implicit reconstruction convention, but its predictive utility is not established by the ratio alone. Node 18 itself is a K-only proposal from node 13, so the visual spike is a branch entering the incumbent position, not background cells being added at iteration 28.

## Additional checks useful for the paper

SET's rise from 2.4161 to 2.6186 is wholly explained by an added shape header, `cursor:unknown` on every frame, and lexicographic-to-numeric sorting of identical cell strings. Removing the new headers yields ratio 2.4690; also restoring old sorting reproduces node 1 exactly. The shipped P does not build card-level entities, even though other explored SET nodes do. Those other nodes cannot explain this incumbent curve.

Colour Lines endpoints emit identical cell triples with changed separators: mean characters decline 187.27→175.54 while ratio rises slightly 1.09818→1.11205. Ice offers another useful-looking change: individual gray and blue cells become labeled `fixed`, `movable:gray_row0:5-7`, and `blue_row15:3-10` fields, aligned with K's fixed/movable vocabulary. Mean characters decrease 163.80→146.21, but gzip bytes increase 82.22→118.95. Its step counter is constant zero.

Suggested paper interpretation: the learner searches over task-adapted textual encodings, sometimes explicitly computing spatial summaries and sometimes compacting coordinate descriptions. A gzip ratio measures the coding cost of those representations. It does not by itself establish greater semantic content, abstraction quality, temporal understanding, or usefulness. The clearest supporting qualitative cases are Egg/Grow/Ice; Ants and SET are counterexamples to interpreting every rise as richer learned abstraction, while Mario shows that a fall can accompany restored state distinctions.
