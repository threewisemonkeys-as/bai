# Learned beliefs vs. ground-truth dynamics — Plan A (stepwise_eb_learn) vs RGB-Agent baseline

Date: 2026-06-17
Setup (both methods, identical): 10 AutumnBench envs, 50 env-step budget, gemini-2.5-flash,
free exploration (stepwise = dynamics mode / theory_disagreement+perception;
RGB = `run_autumn.py --task interactive`).

Ground truth read from the `.sexp` programs in
`MARAProtocol/python_examples/autumnbench/example_benchmark/programs/`.
Stepwise belief source: final episode `beliefs.txt`. RGB belief source: final analyzer
strategic briefings in `logs_analyzer.txt` (RGB keeps no explicit belief store).

Scoring is a manual 1–10 judgment of how well each method's FINAL articulated world-model
matches the true mechanics (core mechanic identified? cause→effect right? completeness?).

| Game  | Ground-truth dynamics (core mechanic)                                   | Stepwise | RGB |
|-------|--------------------------------------------------------------------------|:--------:|:---:|
| ice   | day/night sun toggled by *any* click; cloud moves L/R; `down` rains a drop; click also freezes/melts water; liquid flows, ice stays | 3 | 2 |
| DQ8GC | arrow-movable "active" particle; gray particles get infected→darkgreen and spread to neighbors; click swaps active particle | 4 | 5 |
| 7WWW9 | mobile blue magnet vs fixed red magnet; opposite poles attract (auto-move), like poles repel/block | 5 | 3 |
| ADA85 | click Suzie/Billy spawns a rock that travels to the bottle spot; "breaking" rocks break the bottle; click bottle to repair | 3 | 2 |
| 27VWC | BBQ: click toggles fire (needs gas), fill button adds gas, meat cooks over time, click meat feeds person (health ± by doneness) | 3 | 3 |
| F5W3N | Space Invaders: hero moves L/R, `up` shoots; enemy rows oscillate on a timer; bullets/enemy-bullets + collisions | 4 | 2 |
| EAHCW | click paints a cell with current color; arrows set color (up=gold,down=purple,left=green,right=blue); opposite arrow resets to red | 3 | 1 |
| QQM74 | velocity/momentum: arrows change a capped ±1 velocity, blob moves by velocity each tick; click adds a blob | 4 | 2 |
| VQJH6 | 4 edge buttons set gravity direction; 2×2 blobs fall that way each tick; click empty cell adds a blob | 4 | 3 |
| AW9WD | tan eggshell shape hides a chick; clicking an eggshell breaks it; broken shells dissolve to reveal feathers underneath | 2 | 2 |
| **Mean** | | **3.5** | **2.5** |

Tally: stepwise higher on 7, RGB higher on 1 (DQ8GC), tie on 2 (27VWC, AW9WD).

## Per-game notes

- **ice** — Stepwise found the 2×2 gold/gray toggle but modeled it as a list of specific
  "trigger cells" instead of "any click toggles day/night," and missed cloud/rain/water
  entirely. RGB invented a phantom moving "blue cursor / lightblue cursor" (it misread the
  water drops / ice) — wrong frame, no day-night/cloud/rain.
- **DQ8GC** — *RGB's best result.* It stated the conversion mechanic: move the darkgreen
  agent adjacent to a gray particle, then click → gray turns darkgreen (adjacency + click).
  Stepwise got arrow-movement of the darkgreen "player" correctly but explicitly said click
  does nothing and never captured the infection/spread or click-swap.
- **7WWW9** — *Stepwise's best edge.* It captured that the red entity *influences* blue's
  motion as a function of distance ("up shifts column right when distance==2", "right blocked
  when adjacent") — a phenomenological capture of magnetic attraction/repulsion. RGB declared
  red "purely static, clicks have no effect," missed magnetism, and quit to test.
- **ADA85** — Both poor. Stepwise tracked gray (rock) spawns + motion but wrapped it in a
  baroque, wrong positional rule-system ("ICGCS", column-10 cycles) and missed bottle/break.
  RGB only got "clicking changes cells," declared done, went to test.
- **27VWC** — Tie. Both found the same two surface click-effects (click(3,3) clears row-0
  blues; click(3,4) toggles a cell orange/white) and both missed the fire/gas/cook/feed
  semantics and the time-based cooking.
- **F5W3N** — Stepwise captured enemy oscillation ("blue cells alternate between two configs")
  and falling orange projectiles — a real phenomenological read of the shooter, though it
  misattributed the timer to click/noop. RGB was in a wrong "click white→green / black→red"
  frame and declared "Score 100, done."
- **EAHCW** — Stepwise got "clicking creates a red cell" (initial color is indeed red) but
  transposed row/col and missed the arrow→color mechanic. RGB never articulated any rule
  (kept re-reading the log) — essentially no learned dynamics.
- **QQM74** — Stepwise noticed the momentum ("noop causes passive continued movement, direction
  influenced by prior arrow actions") — the core velocity mechanic, fuzzily; but mixed with
  wrong teleport-on-click claims. RGB described green/purple/red traversal mechanics that don't
  exist in this game (hallucinated/mismatched).
- **VQJH6** — Stepwise got click→spawn a 2×2 blue blob and that blobs then move/rearrange over
  time (the falling), but missed that buttons set gravity direction. RGB sensed clicks near the
  gold/darkorange buttons affect blue patterns but framed it as a fill puzzle, no gravity.
- **AW9WD** — Both failed identically: concluded the tan shape is static and "no action changes
  the grid," missing the click-to-break-eggshell / reveal-chick mechanic.

## Caveats
- 1–10 scores are my manual judgment, not a calibrated metric.
- RGB has no belief artifact; I scored its final analyzer briefings, which are exploration-time
  reasoning, not a distilled model.
- Several RGB runs self-terminated early by calling `go-to-test` (7WWW9 @14, F5W3N @10,
  EAHCW @1, QQM74 @22, DQ8GC @36) — the analyzer repeatedly declared premature "task complete"
  rather than continuing to explore, which capped its evidence.
- Stepwise's failure mode is the opposite: over-fitting — long, internally-elaborate rule
  systems (ADA85, F5W3N) that track real surface changes but in a distorted, perception-coupled
  frame.
