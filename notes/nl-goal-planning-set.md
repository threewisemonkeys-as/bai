# NL-goal planning set — build plan

Turn the 30 curated planning problems (`logs/2026-08-18/curated/problems.json`) into a
second set where **the goal is a natural-language sentence** and success is decided by a
**predicate over the executed trajectory**, not by exact-frame equality.

Everything else is held fixed — same games, same seeds, same start states, same budget,
same arms, same 5 attempts — so every NL result is *paired* with the frame-goal result
already in `logs/2026-08-18/curated/eval/offline.json`. The headline number is
Δ(NL − frame) per game and per tier, not a new absolute score.

---

## 0. What is actually changing

The curated set was deliberately built so that **nothing predicate-shaped ships**
(`curated_plan.py` docstring): predicates existed only as solver subgoals and as the
anti-padding compression screen, and the artifact was one exact frame. This work reverses
that decision *for a parallel set*, and inherits the exact hazard the original design was
avoiding — a loose predicate is exploitable, and the 83wkq `spawn-two` incident (all three
eval arms scored 5/5 on a collapsed goal) is the documented proof.

So the build is not "expose `Problem.goal`". It is:

1. re-author each goal as a sentence + a checker that is *the denotation of that sentence*;
2. re-run the whole validation ladder against the checker instead of the frame;
3. add screens that only matter once goals accept a *set* of states.

**The existing `Problem.goal` lambdas are not usable as checkers.** They were tuned to make
BFS land on one route, so they pin coordinates the sentence never mentions (`coin-ground`
requires Mario at exactly `(9,1)`; `ice-tower` pins column 4; `gather` pins the 2×2 block at
rows 5–6, cols 5–6). As checkers they are systematically **too tight**: they would fail a
plan that satisfies the English perfectly. A few are also too loose in the other direction.

---

## 1. Decisions to settle before authoring

These change the work materially; recommendations given.

### D1. Which sentence — perceptual or intentional? (**recommend: ship both**)

`DEFAULT_KNOWLEDGE` (`invdyn_core.py:252`) tells the agent only the action space. It sees
JSON grids of CSS colour names, with **no object vocabulary at all**. So

> "Collect all three coins"

is ungroundable — nothing tells the agent that `gold` is a coin, or that `red` is it.
Learned beliefs won't rescue this: they were induced from the same colour grids and speak
of "gold cells". Two phrasings per problem, authored together, define a clean ablation:

| field | example | what it tests |
|---|---|---|
| `nl_perceptual` | "No gold cells remain anywhere on the grid, and the red cell is resting on a surface." | planning to an abstract (set-valued) goal |
| `nl_intentional` | "Collect all three coins." | that + grounding game language into a learned world model |

The same checker serves both. Run `nl_perceptual` as the primary condition (it is the fair
comparison against the frame-goal arm) and `nl_intentional` as the harder second condition;
optionally a third with a per-game glossary to separate vocabulary from abstraction.

### D2. Scoring — any-step or terminal? (**recommend: report both, headline `terminal`**)

Exact frames made this moot: every curated goal but one is absorbing, so "matched at step h"
and "matched at any step" coincide. Predicates break that — most relaxed checkers are
monotone (coin collected stays collected) but some are transient. Compute an `absorbing`
flag per problem (checker still true after +k noops, §3 N7), report `satisfied_final` and
`satisfied_any`, and headline `satisfied_final` so a random flailer cannot collect credit
for passing through an accepting state.

### D3. The `wc` arm. (**recommend: drop it from the NL set, or run it labelled as a skyline**)

`prt.plan_search` (`program_runtime.py:431`) takes a goal *grid* and uses it twice — as the
termination test and as the beam heuristic (cells matching goal). A predicate goal gives it
neither. Options: (a) drop `wc`, compare `raw` vs `lmwm` only; (b) pass a `goal_test`
callable with BFS-only search — feasible but strictly weaker, and it hands the program arm a
*formal* goal while the LLM arms get English, which is a confound that has to be stated;
(c) let an LLM translate the sentence into a goal grid first, making grounding part of the
arm. Recommend (a) for the first run and (c) later as its own ablation.

### D4. Two problems are probably not NL-expressible. (**recommend: cut, and say so**)

- `s2kt7 / intercept` — "Stop the clock mid-forage" is a *frame*, not a goal; its content is
  "the ants are exactly here at exactly this tick". Any faithful sentence is a coordinate
  dump.
- `83wkq / spawn-two` — the real content is a timing fact ("one particle has drifted, the
  other has not"), and the note already calls it close to unsolvable.

Cutting leaves 28. Alternative: keep them with deliberately coordinate-laden sentences and
report them separately as an "NL is the wrong interface here" bucket.

---

## 2. Build: files and data

New code, all importing the existing machinery rather than restating it:

```
offline_learning/nl_goals.py                 registry: pid -> NLGoal
offline_learning/scripts/build_nl_goals.py   -> logs/<date>/curated_nl/problems_nl.json
offline_learning/scripts/validate_nl_goals.py  N1..N8 (§3)
offline_learning/scripts/audit_nl_goals.py   acceptance-set render + judge agreement (§4)
offline_learning/scripts/eval_curated_nl.py        offline, imports eval_curated_plan
offline_learning/scripts/eval_curated_nl_online.py online, imports eval_curated_online
```

```python
@dataclass
class NLGoal:
    pid: str
    nl_perceptual: str
    nl_intentional: str
    check: Callable[[list[St], list[str]], bool]   # states[0] = start; actions[i]: states[i]->states[i+1]
    positives: list[list[str]]    # action sequences that MUST be accepted (>= 1 alternate route)
    negatives: list[list[str]]    # action sequences that MUST be rejected (near misses)
    note: str = ""
```

**Checker signature is over the trajectory, not the state.** Three of the s2kt7 goals
("feed the ants twice/three times") are genuinely trajectory properties — no single frame
distinguishes round 2 from round 3, which is why the original predicates read the hidden
`clicks` counter. Most checkers will only touch `states[-1]`.

**Checking at eval time needs no new state machinery**: replay the agent's actions through
`curated_plan.trace(game, seed, actions)` to get `St` objects (frame + tracked hidden state)
and apply the checker. ~0.6 ms/step on the raw interpreter, negligible against an LLM call.

**Serialization**: lambdas don't go in JSON. Ship `checker_id` in the JSON plus
`inspect.getsource(check)` as a `checker_src` string so the artifact is auditable
standalone, and keep the registry as the single executable definition used by builder,
validator and eval alike.

`problems_nl.json` rows = the existing curated row (unchanged: `start`, `plan`, `h`,
`mechanics`, and the reference `goal` frame, kept for the paired comparison) plus:
`nl_perceptual`, `nl_intentional`, `checker_id`, `checker_src`, `random_success_pred`,
`noop_success_pred`, `shortest_accepting_h`, `absorbing_pred`, `accept_sample` (rendered
alternates), `nl_status` ∈ {ok, coordinate-laden, cut}.

### Authoring rule

> Relax every coordinate the sentence does not mention; tighten the sentence wherever the
> checker must stay exact. The checker accepts exactly the set of states a competent reader
> of the sentence would accept — no more, no less.

Concrete first pass over the 30 (the tight ones, from `curated_plan.py:370-627`):

| problem | current predicate pins | NL-faithful checker |
|---|---|---|
| n2ntd platform | `red == (7,6)` | Mario at rest on any cell directly above the middle platform |
| n2ntd coin-ground/-air | Mario's exact landing cell | the named coin is gone, Mario at rest, others untouched |
| n2ntd kill-one-coin | *which* coin | enemy gone, exactly two coins remain, Mario at rest |
| bt3gb one-drop/one-ice | column 2 / 9 (named in the sentence — keep) | keep, drop the incidental `not _water(other)` if daylight already forces it |
| bt3gb ice-tower | column 4 | a 3-high ice column at *any* column, nothing else |
| bt3gb staircase | cols 8/9/10, one chirality | a 3-2-1 ice staircase anywhere, either direction |
| dq8gc walk | corner `(0,0)` + all healthy at start cells | the infected particle on any corner cell |
| dq8gc infect-one | exact positions of all five | the particle that started nearest is now infected |
| dq8gc swap-drive | `(5,0)` exactly | some initially-healthy particle now on a border cell, having moved |
| dq8gc chain | exact final positions | ≥3 infected and the third one was infected *by* the second (needs the trajectory) |
| dq8gc gather / infect-all-gather | block at rows 5-6 cols 5-6 | any 2×2 block of healthy particles / then all infected |
| s2kt7 rounds | hidden `clicks == k` | k spawn events in the action trace, board empty at the end |

Everything else (`nightfall`, `all-coins`, `infect-all`, `pool`, `freeze-pool`,
`one-eaten`, `all-eaten`, `spawn`, `park-cloud`, `high-ground`) is already close to faithful
and needs only a sentence.

### Hazard: dead-reckoned hidden state

`_track_dq8gc` / `_track_bt3gb` dead-reckon `active` and `cloud` from the action string, not
from the frame. That is sound along the *reference* plan the builder searched, and may
desync on an agent's arbitrary sequence (a blocked move, a click on empty space, a clamp).
A checker that reads `_h(s, "cloud")` would then score the wrong thing. Two fixes, do both:
prefer frame-derived quantities in checkers wherever the object is rendered, and fuzz the
trackers — 1000 random drives per game, dead-reckoned value vs a frame-derived reader, must
agree everywhere.

---

## 3. Validation — `validate_nl_goals.py`

Re-runs through `AutumnBenchEnvWrapper` (the *other* engine driver), as `validate_curated.py`
does, so a driver disagreement surfaces as a failure rather than a silently wrong dataset.

| check | statement |
|---|---|
| N1 | the reference plan's trajectory satisfies the checker, and does so at exactly step `h` (agreement with the shipped frame) |
| N2 | the checker is false at t=0 and at every earlier step of the reference plan |
| N3 | `noop^h` from the start does not satisfy it |
| N4 | **task-incompressible under the new checker**: re-run `compress()` against the checker; if the plan shrinks, the checker is too loose (this is the 83wkq trap, automated) |
| N5 | random plans of length `h` satisfy it at ≤ 5% over 200 trials (h ≥ 2 only; report honestly at h=1) — recompute, do not inherit `random_success`, relaxation moves the floor |
| N6 | every authored positive accepted, every authored negative rejected (≥1 positive and ≥3 negatives per problem; negatives drawn from the reference prefix, from other problems in the same game, and hand-perturbed) |
| N7 | `absorbing_pred`: still true after +1..+8 noops — records whether any-step and terminal scoring coincide (D2) |
| N8 | **shortcut search**: BFS over the full action alphabet to depth `h` collecting the *shortest* accepting sequence. If it is much shorter than `h`, the checker admits a route the sentence does not. Cheap at L1/L2; cap the node budget at L3/L4 and log the cap rather than silently truncating |

N4, N5 and N8 are the three that only exist because goals became set-valued. N8 is the real
anti-gaming screen; N5 is its cheap stochastic shadow.

---

## 4. Faithfulness audit — `audit_nl_goals.py`

Validation proves the checker is hard to game. It does not prove the checker *means the
sentence*. Two passes:

1. **Acceptance-set render.** Enumerate/sample accepting states within the budget (reuse the
   N8 search) and render them next to the sentence with `viz_curated.py`. Read them. Any
   accepted frame a human would call "not what the sentence says" is a checker bug.
2. **Independent judge.** An LLM that sees *only* the sentence, `DEFAULT_KNOWLEDGE` and one
   rendered frame labels satisfied / not, over the accepting sample plus the N6 negatives.
   Report agreement with the checker per problem. Disagreements go back to authoring.

The judge is an **auditor, not a scorer** — LLM-judge scoring has been noisy here before
([[forward-objective-scoring]]). It never decides an eval result.

---

## 5. Eval — `eval_curated_nl.py`

Import `eval_curated_plan` and change exactly two things, so nothing else can drift:

- **Prompt**: new `PLAN_NL_RAW_TMPL` / `PLAN_NL_WIN_TMPL`, identical to the existing pair
  but with the `=== GOAL raw grid ===` / `=== GOAL state features ===` block replaced by
  `=== GOAL (natural language) ===`. Note the asymmetry this creates and state it: in the
  frame condition the `lmwm` arm received the goal *through its own perception module*
  (`_z_goal`); NL removes that channel from both arms, which is part of what makes the
  comparison interesting.
- **Scorer**: `reached(grids, goal_frame)` → `satisfied(states, actions, checker)`,
  returning `(final, any, first_step)`. Keep `frame_hit` alongside — the gap between
  "satisfied the sentence" and "hit the reference frame" measures how much slack the
  relaxation bought, per problem.

Arms `raw` / `lmwm`, 5 attempts, `PLAN_CAP = 50`, per-game and per-tier reporting with the
recomputed random floor carried alongside — unchanged. Online/MPC variant is a one-line
substitution (the goal string is constant across rounds, where the goal frame was too).

Reported table: for each game × tier, frame-goal pass@1/pass@5 (from the existing run) vs
NL-perceptual vs NL-intentional, with the floor.

---

## 6. Order of work

| phase | output |
|---|---|
| 0 | settle D1–D4 |
| 1 | `nl_goals.py` — 28–30 sentences ×2 phrasings + checkers + positives/negatives; tracker fuzz |
| 2 | `build_nl_goals.py` + `validate_nl_goals.py`, iterate until N1–N8 pass (expect 2–3 rounds: N4/N8 will catch loose checkers) |
| 3 | `audit_nl_goals.py` — acceptance render + judge agreement |
| 4 | `eval_curated_nl.py` (+ online), paired run against the existing frame-goal results |
| 5 | write-up: Δ(NL − frame), the perceptual/intentional gap, and the problems cut as NL-inexpressible |

Phases 1–2 are the bulk and are hand work; 4 is mostly compute. The honest failure mode to
watch for is phase 2 dragging because relaxed checkers keep failing N8 — if a problem cannot
be relaxed to match its sentence without admitting a shortcut, that problem's sentence was
underspecified and needs tightening, not the checker loosening.

---

# Pilot — built 2026-08-19

Scope set by the user: **intentional phrasing only** (D1), **any-step scoring** (D2), **no
`wc` arm, `lmwm` only** (D3), and **one problem per game** to start. Five goals, one per
game, chosen as the closest thing each game has to "what the game is about" (83wkq has only
two problems, so its L1 is the pick).

| game | problem | sentence | h → nl_h | rand@50 |
|---|---|---|--:|--:|
| n2ntd | all-coins | "Collect all three coins." | 32 → 26 | 0.000 |
| bt3gb | ice-tower | "Stack three blocks of ice into a tower three cells tall, standing on the ground." | 20 → 20 | 0.000 |
| dq8gc | infect-all | "Infect every particle, leaving no healthy ones." | 12 → 12 | **0.052** |
| s2kt7 | all-eaten | "Put out a single round of food and let the ants eat all of it, without putting out any more." | 15 → 15 | 0.000 |
| 83wkq | spawn-one | "Create exactly one particle, at row 8, column 8." | 1 → 1 | 0.005 |

`nl_h` is the horizon the SENTENCE needs (the reference plan compressed against the checker);
`h` is the horizon the exact frame needed. They differ on n2ntd by exactly the six trailing
noops that existed only to bring mario to rest for the frame.

Code: `offline_learning/nl_goals.py` (registry — sentences, checkers, fixtures),
`offline_learning/scripts/validate_nl_goals.py` (N1–N8 + cross-driver),
`offline_learning/scripts/eval_curated_nl.py` (offline eval, `lmwm` by default).
Results: `logs/2026-08-19/nl_pilot/`.

## What any-step scoring changed, measured not assumed

Any-step is the reason four of the five checkers needed a guard clause the frame goal never
did. A frame goal is checked once, against a snapshot; a set-valued goal checked after every
prefix of a 50-action rollout will find anything that is briefly true anywhere.

- **n2ntd — one-frame occlusion.** A bullet renders above a coin. Collect two coins, fire
  along the third's column, and for exactly one frame the board has no gold on it. A
  42-action witness (inside the 50-action budget) is in the fixtures. The general guard —
  require the empty board to persist two frames — works, but it also **fails the most
  natural correct plan there is**, one that ends on the action collecting the third coin.
  The checker names the occluder instead (no gold cells *and* no bullet on the board), which
  rejects the exploit without rejecting correct play.
- **dq8gc — permanent occlusion.** Drive the infected particle on top of the last healthy
  one and there is no gray cell anywhere, for ever: contagion crosses orthogonal adjacency,
  which distance 0 is not, so the survivor is never infected and never re-emerges. The
  witness is 12 actions — exactly as long as the real solution. `len(darkgreen) == 5` rejects
  it. Cost of the clause: the checker demands a visible demonstration, so five infected
  particles stacked on four cells is rejected until the agent separates them.
- **s2kt7 — trivially reachable by flailing.** The ants clear the board ~15 ticks after a
  click almost regardless of how much food is out, so "put food out and let them eat it" is
  satisfied somewhere in **75%** of random 50-action drives. Neither is `two-rounds` (0.47)
  or `three-rounds` (0.25) any better. The only version that survives names the quantity —
  one round, checked against the click count — which floors at 0.000. **The lesson
  generalises: under any-step scoring a goal whose completion the environment reaches on its
  own is not a planning problem.**
- **83wkq — coincident objects.** Two particles on one cell render as one. The click count
  distinguishes them. Not exercised by a fixture (particles all take the same random step
  each tick, so they never collide, and none walks off the grid within these horizons) —
  kept because it is free.
- **s2kt7 also breaks the state-predicate assumption**: the board is empty at t=0, so a
  state predicate hands out a free pass at the first frame. Checkers take
  `(frames, actions)`, not a state.

## Open, and deliberately not fixed

`dq8gc/infect-all` fails the 0.05 random-floor gate at **0.052** (21/400 drives): 50 actions
of random walking infect all five particles about one time in twenty, which is a **23%
pass@5 floor**. The sentence cannot be tightened without changing the task, so the number is
reported in the eval table (`rand@50` column) rather than gated away. Every dq8gc score has
to be read against it.

Checkers read rendered frames and actions only — no hidden-state trackers — so the
dead-reckoning desync hazard in the plan above does not arise for these five, and the
cross-driver check (X) is a straight frame comparison.

## Smoke run

`google/gemini-3.7-flash`, `lmwm` only, 5 attempts, 5 problems: **1 minute, $0.13** (the
default `deepseek-v4-flash` hung on provider latency — `llm_call` allows 600 s per attempt
× 4, so a stalled provider costs 40 minutes before it gives up).

| problem | @1 | @5 | frame@1 | rand@50 |
|---|--:|--:|--:|--:|
| n2ntd/all-coins | 0.00 | 0.00 | 0.00 | 0.000 |
| bt3gb/ice-tower | 0.20 | 1.00 | 0.00 | 0.000 |
| dq8gc/infect-all | 0.00 | 0.00 | 0.00 | 0.052 |
| s2kt7/all-eaten | 0.00 | 0.00 | 0.00 | 0.000 |
| 83wkq/spawn-one | 1.00 | 1.00 | 1.00 | 0.005 |

Failures are genuine, not scoring artifacts, and each game fails its own way:
n2ntd collects two of three coins; dq8gc emits four-action plans that **click** the healthy
particles (it has learned that click selects, not that contagion needs adjacency) and infects
nobody; s2kt7 clicks correctly every single time and then **waits four to nine ticks when the
ants need fourteen** — the plan is right and the duration is wrong.

**The relaxation earns its keep on bt3gb.** The successful run built its tower at column 6
(`right, right, down, ...`), not the reference's column 4 — it satisfies the sentence and
would have scored zero against the exact frame. That is the `frame@1 = 0.00` against
`@1 = 0.20`.

**Five attempts is noisy.** Across three runs of the same config s2kt7 scored 0.80 / 0.40 /
0.00, the whole spread coming from how long the planner chose to wait. Any comparison on
this pilot needs more attempts than five.

Asking for a numbered rationale per action (needed for the viz) also needs the plan block to
say **actions only, no explanations** — without that line the rationale leaks into the plan
and the parser rejects it.

The paired comparison against `logs/2026-08-18/curated/eval/offline.json` still needs a run
with **the same planner that baseline used** (`deepseek-v4-flash`); the gemini numbers above
are not comparable to it.

## Viz

`offline_learning/scripts/viz_nl_goals.py` → `logs/2026-08-19/nl_pilot/viz.html`. One card
per problem: start frame + the sentence (all the agent was told — an NL goal shows no target
frame), then one row per attempt with the plan replayed frame by frame, the action above each
frame and the planner's own line for that action below it. The first accepting frame is
ringed; everything after it is dimmed, since any-step scoring means the run had already
succeeded. The reference solution rides along as a labelled row — two problems have no
successful attempt, and five failures with nothing to compare against say very little.
Rationales are keyed by the model's OWN numbering, so a step it never numbered shows
"(no line)" rather than borrowing its neighbour's.

## Online (receding horizon), same day

`eval_curated_nl_online.py` — same five sentences, same 50-action budget, same 5 attempts,
`lmwm` only, warm start on. Each round the planner writes a plan of at most `50 − n` actions,
**only the first executes**, and it replans from the observed state. Success is the same
any-step checker, evaluated on the frames executed so far. gemini-3.7-flash: **13 min, $3.27**
(25 rollouts × up to 50 sequential LLM calls, vs 25 calls total offline).

| problem | online @1 | offline @1 | mean actions used | replanned | frame@1 |
|---|--:|--:|--:|--:|--:|
| n2ntd/all-coins | 0.80 | 0.00 | 39.6 | 12% | 0.00 |
| bt3gb/ice-tower | 0.80 | 0.20 | 38.4 | 3% | 0.60 |
| dq8gc/infect-all | 1.00 | 0.00 | 20.8 | 25% | 0.00 |
| s2kt7/all-eaten | 1.00 | 0.00 | 15.0 | 0% | 1.00 |
| 83wkq/spawn-one | 1.00 | 1.00 | 1.0 | — | 1.00 |

Closing the loop moves three problems from zero to near-perfect. That confirms the diagnosis
the open-loop run suggested: **the failures were forward-model errors, not planning errors.**
The plans were mostly right; the agent could not predict how long they take or whether they
had worked.

But the two big gains have completely different characters, and only the per-round view
separates them:

- **dq8gc is real closed-loop error correction.** Round 0 executes `click 3 4` with the tail
  `click 5 3, click 5 7, click 6 6` — the same wrong theory that scored 0.00 offline, that
  clicking infects. It then SEES the click did not infect, and from round 1 switches to
  moving the driven particle into contact. 25% of its rounds are replans, the highest in the
  set, and it goes 5/5.
- **s2kt7 is the harness noticing on the agent's behalf.** All five rollouts are `click`
  followed by 14 `noop`s, 0% replanned, mean 15.0 actions — exactly the reference horizon.
  The agent never decides the food is gone; the checker accepts and the harness stops the
  rollout. This is the "no submit action" caveat becoming load-bearing: **on any goal the
  environment completes by itself, an online agent that noop-pads scores 1.00 without
  demonstrating it knows anything.** A `<done/>` token would separate the two readings, at
  the cost of pairing with the offline arm unless both get it.

n2ntd and bt3gb sit in between — high `followed` rates (88%, 97%) mean the planner mostly
executes its own plan one action at a time, using the loop to re-anchor rather than to
change its mind. bt3gb's `frame@1` of 0.60 against `on1` 0.80 says most but not all of its
towers landed on the reference column.

Cost note: online is ~25× the LLM calls of offline for the same 25 attempts. Two n2ntd
rollouts hit `budget-exhausted` at 50 actions.

Viz: `offline_learning/scripts/viz_nl_online.py` → `logs/2026-08-19/nl_pilot/viz_online.html`.
Per round it draws the action that ran, the frame it produced, the planner's line for it, and
the **tail it discarded**; each round is marked `followed` or `replanned` by comparing what
executed against what the previous round had queued next.

---

# Phase 2 — the remaining 25, planned 2026-08-20

The pilot shipped one goal per game. This is the plan for the other 25: **6 n2ntd, 7 bt3gb,
6 dq8gc, 5 s2kt7, 1 83wkq**. Same decisions as the pilot (intentional phrasing, any-step
scoring, `lmwm` only, 50-action budget) unless a problem forces the question.

Everything below is authored against the ladder the pilot paid for, plus five engine facts
checked today rather than assumed (§2). The random floors in §3 are **measured** on draft
checkers, 400 random drives each, and they decide the disposition of eight problems.

## 1. What phase 1 established, restated as authoring rules

1. **The checker is the denotation of the sentence.** Relax every coordinate the sentence
   does not mention; tighten the sentence wherever the checker must stay exact.
2. **Any-step scoring expresses achievement, never timing.** A predicate that is only
   transiently true is passed through by any run that eventually does the thing.
3. **Count from the action trace, not from pixels.** Occlusion and coincidence make cell
   counts lie; `n_clicks`/`n_downs` do not.
4. **Name only what is stable under the dynamics.** A description that stops being unique
   after the first tick is not a goal, it is a coordinate dump in disguise.
5. **The floor is set by the shortest accepting prefix, not by the reference horizon** — see
   §3, where `rand@h` and `rand@50` differ by 0.7 on the same checker.

## 2. Engine facts checked today

Each of these decides a checker, and each was measured, not read off the `.sexp`:

- **n2ntd platforms do not move.** Six noops, and the three `Step` objects stay at rows 10
  (cols 0–2), 8 (cols 4–6) and 6 (cols 8–10); only the enemy patrols. So "the middle
  platform", "the lowest platform" and "the highest platform" are stable names, and a
  platform's ROW identifies it for ever — which is what lets the checker drop the column the
  curated predicate pinned.
- **n2ntd ammo gating is real, and a blocked bullet does not die — it stalls for ever.**
  Clicking with zero bullets produces nothing, so "collect a coin and shoot" is a genuine
  two-phase task. Firing into a platform is not a wasted tick but a permanent change to the
  board: a shot from column 6 sits at row 9 under the middle platform and was still there 13
  ticks later. From the floor only columns 3, 7 and 11 are clear; from a platform the
  geometry changes (a shot from (9,1), standing on the lowest platform, flies free).
- **bt3gb's cloud clamps at the left wall** (origin x=1, cells 0–2), so "against the wall" is
  an absorbing position. At night the celestial body is gray too, but it spans rows 0 AND 1
  while the cloud is row 0 only, so `g[0][2]=="gray" and g[0][3]!="gray"` reads the parked
  cloud correctly in both day and night (checked on both).
- **dq8gc contagion is orthogonal only, and the driven particle can leave the world.**
  Diagonal contact does not infect; orthogonal contact infects one tick later. Three `left`s
  from the start and the green cell is simply *gone* — off-grid objects stop rendering. Any
  checker of the form "all five are X" needs a visible-count clause, and "drive it to the
  wall" must not be read as "past the wall".
- **An s2kt7 click does not always put two food cells on the board.** At seed 1 the fifth
  click spawns both foods on one cell and the board shows one. `one-eaten` therefore cannot
  be "2 became 1"; it has to be "this round's peak became peak−1".

## 3. Measured random floors, draft checkers, 400 drives

`rand@h` is the floor at the reference horizon, `rand@50` at the eval budget the agent
actually gets. The gap between the columns is rule 5, quantified.

| game | problem | tier | h | rand@h | rand@50 | disposition |
|---|---|---|--:|--:|--:|---|
| n2ntd | platform | L1 | 1 | 0.133 | **0.838** | floor-dominated |
| n2ntd | high-ground | L1 | 6 | 0.000 | 0.055 | ship, flagged |
| n2ntd | coin-ground | L2 | 9 | 0.000 | 0.152 | ship, flagged |
| n2ntd | coin-air | L2 | 11 | 0.030 | 0.122 | ship, flagged |
| n2ntd | kill-one-coin | L3 | 22 | 0.007 | 0.048 | ship |
| n2ntd | all-coins-kill | L4 | 34 | 0.000 | 0.000 | ship |
| bt3gb | nightfall | L1 | 1 | 0.195 | **1.000** | floor-dominated |
| bt3gb | park-cloud | L1 | 3 | 0.007 | **0.450** | ship after tightening (0.020) |
| bt3gb | one-drop | L2 | 17 | 0.000 | 0.000 | ship |
| bt3gb | one-ice | L2 | 21 | 0.000 | 0.000 | ship |
| bt3gb | pool | L2 | 20 | 0.000 | 0.003 | ship |
| bt3gb | freeze-pool | L3 | 21 | 0.000 | 0.005 | ship |
| bt3gb | staircase | L4 | 33 | 0.000 | 0.003 | ship |
| dq8gc | walk | L1 | 4 | 0.013 | 0.138 | ship, flagged |
| dq8gc | infect-one | L2 | 3 | 0.070 | **0.608** | flagged after tightening (0.175) |
| dq8gc | swap-drive | L2 | 4 | 0.000 | 0.010 | ship |
| dq8gc | chain | L3 | 7 | 0.000 | 0.022 | ship |
| dq8gc | gather | L4 | 10 | 0.000 | 0.000 | ship |
| dq8gc | infect-all-gather | L4 | 19 | 0.000 | 0.000 | ship |
| s2kt7 | spawn | L1 | 1 | 0.517 | **1.000** | floor-dominated |
| s2kt7 | one-eaten | L2 | 13 | 0.000 | 0.000 | ship |
| s2kt7 | two-rounds | L3 | 17 | 0.000 | 0.000 | ship |
| s2kt7 | three-rounds | L4 | 21 | 0.000 | 0.000 | ship |
| s2kt7 | intercept | L3 | 10 | 0.000 | 0.000 | ship, see §5 |
| 83wkq | spawn-two | L2 | 3 | 0.000 | 0.000 | ship (tightened) |

Before tightening: 16 ship, 4 ship-flagged, 5 floor-dominated. Three results are worth
stating separately, and the third moves two of those five (§3.1).

**Every floor-dominated problem has h <= 3, and the budget is most of the reason.**
`platform` is one jump and `nightfall` is one click; over 50 actions a random drive passes
through the goal almost surely. `park-cloud` is 0.007 at h=3 and 0.450 at 50 — nothing about
that sentence or checker is loose, the budget is simply 16x the task. The one h<=3 problem
that escapes is `spawn-two`, and it escapes because its sentence names exact cells.

**The floor is a function of the budget, and the curve is steep.** Same drives, same
checkers, scored at the first accepting step:

| problem | h | rand@h | rand@2h | rand@3h | rand@50 |
|---|--:|--:|--:|--:|--:|
| n2ntd/platform | 1 | 0.133 | 0.247 | 0.333 | 0.838 |
| n2ntd/high-ground | 6 | 0.000 | 0.007 | 0.018 | 0.055 |
| n2ntd/coin-ground | 9 | 0.000 | 0.010 | 0.045 | 0.152 |
| n2ntd/coin-air | 11 | 0.030 | 0.077 | 0.098 | 0.122 |
| n2ntd/kill-one-coin | 22 | 0.007 | 0.040 | 0.048 | 0.048 |
| bt3gb/nightfall | 1 | 0.195 | 0.310 | 0.432 | 1.000 |
| bt3gb/park-cloud | 3 | 0.007 | 0.037 | 0.083 | 0.450 |
| dq8gc/walk | 4 | 0.013 | 0.037 | 0.062 | 0.138 |
| dq8gc/infect-one | 3 | 0.070 | 0.228 | 0.320 | 0.608 |
| dq8gc/chain | 7 | 0.000 | 0.000 | 0.005 | 0.022 |
| s2kt7/spawn | 1 | 0.517 | 0.745 | 0.865 | 1.000 |
| 83wkq/spawn-two (loose) | 3 | 0.128 | 0.385 | 0.505 | 0.510 |

(Rows not shown are 0.000 at every budget.)  Two readings: `park-cloud`, `coin-ground`,
`walk` and `high-ground` are **budget artifacts** — at 2h they are at or below 0.04. The
three h=1 problems are **not**: the goal is one correct action out of a handful of verbs
(one in six for `platform`, one in two for `spawn`, whose only verbs are noop and click), so
no budget makes them hard. They are easy problems, and
the only honest thing to do is say so.

**Naming the cells is what saves 83wkq/spawn-two.** "Two particles, one drifted" floors at
0.510; "a particle at row 4 column 4, then one at row 10 column 10 a tick later" floors at
0.000. This is authoring rule 1 in its tightening direction, and it also **rescues a problem
the frame set had written off**: the exact frame was underivable (the first particle's drift
is pure RNG), but "the cell you clicked first is now empty and there are two particles" is a
consequence of the rules, so the problem goes from close-to-unsolvable to solvable.

### 3.1 Can a tightened sentence rescue the floor-dominated five?

Rule 1 says tighten the sentence where the checker must stay exact. Tested, same 400 drives:

| variant | sentence adds | rand@50 |
|---|---|--:|
| park-cloud | — | 0.450 |
| park-cloud | "without making it rain" | 0.055 |
| park-cloud | "without making it rain or changing the time of day" | **0.020** |
| infect-one | — | 0.608 |
| infect-one | "...the one nearest the green particle" | 0.477 |
| infect-one | "without ever taking control of another particle" | **0.175** |
| nightfall | "with a single click, and without making it rain" | 0.515 |
| platform | "without collecting a coin" | 0.838 |
| spawn | "and let none of it be eaten" | 1.000 |

**`park-cloud` is rescued outright** — 0.450 → 0.020, and the added clause is ordinary
English, not a coordinate dump. `infect-one` is half-rescued: 0.175 puts it alongside `walk`
(0.138) in the flagged bucket. The other three are irreducible, and the failed attempts say
why: a clause only lowers the floor if random play *violates* it often. Random drives rain and
click constantly (so the bt3gb and dq8gc clauses bite) but almost never touch a coin (so
`platform` does not move at all) and always eventually click (so `spawn` cannot be saved).

Final disposition: **17 ship, 5 flagged, 3 floor-dominated** (`platform`, `nightfall`,
`spawn` — all h=1, all one correct action).

## 4. The problems, per game

Sentence + checker + what it relaxes. Checkers read frames and the action trace only.

### n2ntd — 6

| problem | sentence | checker | relaxes |
|---|---|---|---|
| platform | "Stand on the middle platform." | mario is the one red cell, at row 7, darkorange directly below | column 6 |
| high-ground | "Stand on the highest platform." | same at row 5 | column 8 |
| coin-ground | "Collect the coin resting on the lowest platform, and leave the other two where they are." | gold == {(4,7),(5,9)} ∧ no bullet on (9,1) | mario at (9,1) |
| coin-air | "Collect the coin floating in mid-air, and leave the other two where they are." | gold == {(9,1),(5,9)} ∧ no bullet on (4,7) | mario at (11,7) |
| kill-one-coin | "Collect a single coin and use it to shoot the enemy dead." | no blue ∧ \|gold\| == 2 ∧ no bullet on a coin cell | *which* coin |
| all-coins-kill | "Collect all three coins and shoot the enemy dead." | no gold ∧ no blue ∧ no bullet on a coin cell | mario at rest |

- **The occluder guard has to be narrowed, and this is a latent defect in the shipped
  pilot goal.** `all-coins` currently rejects any frame with a bullet anywhere on the board.
  Since a blocked bullet stalls for ever (§2), a plan that wastes one shot into a platform
  and then collects all three coins can never be accepted. It never bit — the pilot's ten
  n2ntd rollouts, offline and online, contain zero clicks — but `kill-one-coin` and
  `all-coins-kill` *require* firing, so phase 2 cannot leave it. The fix is to guard only the
  cells the checker asserts are empty of gold: **no bullet on a coin cell**, rather than no
  bullet anywhere. It still rejects the 42-action occlusion exploit (that bullet is sitting
  exactly on (4,7)), and it stops rejecting correct play. Stall cells are (11,0..2),
  (9,4..6), (7,8..10) and no coin sits on one, so the two clauses only ever differ on
  legitimate plans.
- **The sentence must not say "then".** "Collect all three coins **then** shoot" is an order
  the end state cannot show; collect-one/shoot/collect-two satisfies the English just as
  well. Where order really is the content (`infect-all-gather`, `two-rounds`) the checker
  reads the prefix and enforces it — the machinery is there, it is just not wanted here.
- Mario is invisible on the tick he overlaps a coin (coins render over him), which matters
  for the two platform checkers because (5,9) is both a coin cell and a standing cell on the
  highest platform. One fixture each.

### bt3gb — 7

| problem | sentence | checker | relaxes |
|---|---|---|---|
| nightfall | "Make it night." | `g[1][0] == "gray"` | — |
| park-cloud | "Push the cloud all the way to the left wall without making it rain or changing the time of day." | `g[0][2]=="gray"` ∧ `g[0][3]!="gray"` ∧ #down == 0 ∧ #clicks == 0 | reads the frame instead of the dead-reckoned `cloud` tracker; the two clauses are the floor fix (§3.1) |
| one-drop | "Release a single drop of rain and let it land on the floor at column 2." | #down == 1 ∧ blue == {(15,2)} ∧ no ice | — (the column is in the sentence) |
| one-ice | "Make it night so the drops freeze, then land a single block of ice on the floor at column 9." | #down == 1 ∧ ice == {(15,9)} ∧ no liquid | the explicit `night` clause (ice implies it) |
| pool | "Rain exactly three drops of water and let them spread into a flat pool three cells wide on the floor." | #down == 3 ∧ 3 blue cells, all row 15, contiguous ∧ no ice | the `not night` clause |
| freeze-pool | "Let three drops of water settle into a flat pool on the floor, then freeze the pool solid." | #down == 3 ∧ 3 ice cells, all row 15, contiguous ∧ no liquid | — |
| staircase | "Build a staircase out of ice: three blocks tall in one column, two in the next, one in the next." | #down == 6 ∧ ice is three adjacent columns of heights 3-2-1 or 1-2-3, all resting on the floor | the exact columns 8/9/10 **and the chirality** |

- `#down`, not the cell count, is what makes "a single drop" mean one drop: two `down`s on
  consecutive ticks land on the same cell and travel as one for ever. Same guard shape as the
  pilot's 83wkq click count, and it is needed on five of the seven.
- `ice-tower` (shipped) and `freeze-pool` stay the discriminating pair after relaxation — a
  stacked column can only be frozen before the drops, a flat run only after they settle — and
  the relaxed shapes still cannot be confused: one is a column, the other is a row.
- `park-cloud` is the only checker in the whole set that would otherwise want a hidden-state
  tracker. The frame reader above replaces it, which is why the day/night ambiguity had to be
  checked in the engine.

### dq8gc — 6

| problem | sentence | checker | relaxes |
|---|---|---|---|
| walk | "Drive the green particle into the top-left corner without infecting anyone." | green == {(0,0)} ∧ 4 gray | — |
| infect-one | "Infect exactly one of the four healthy particles and leave the other three healthy, without ever taking control of another particle." | 2 green ∧ 3 gray (5 distinct cells) ∧ #clicks == 0 | *which* particle, and every particle's position |
| swap-drive | "Take control of one of the healthy particles and drive it to the left wall, keeping it healthy." | some gray cell in column 0 ∧ 5 distinct cells | *which* particle, and the other four positions |
| chain | "Infect a healthy particle, take control of that one, and use it to infect a third." | ≥3 green ∧ 5 distinct ∧ some click landed on a cell that was green in the frame the agent acted on | the exact final layout |
| gather | "Herd the four healthy particles into a 2×2 block, without infecting any of them." | the 4 gray cells form a 2×2 block ∧ exactly 1 green | the block's location (rows 5–6, cols 5–6) |
| infect-all-gather | "Herd the four healthy particles into a 2×2 block, then infect the whole block." | an earlier prefix satisfied `gather` ∧ now no gray ∧ 5 green | the block's location; **keeps the order**, which is the content |

- `5 distinct cells` is the pilot's `infect-all` overlap guard, generalised: the driven
  particle renders over anything it stands on, so any "N healthy remain" claim can be faked
  by parking on one. It also catches the off-grid case measured in §2.
- Colour is **health, not control** — click a healthy particle and you now drive a gray one.
  So no sentence in this game may refer to the controlled particle by colour; `swap-drive`
  and `gather` are phrased as "take control of" / "herd" for exactly that reason.
- `chain` is the one checker that reads what the agent *saw*: a handoff is a click on a cell
  that was green at that moment. Nothing in the end state records it.

### s2kt7 — 5

| problem | sentence | checker | relaxes |
|---|---|---|---|
| spawn | "Put out one round of food." | #clicks == 1 ∧ ≥1 red | the count 2 (a click can render as one cell) |
| one-eaten | "Put out a single round of food and let the ants eat exactly one piece of it." | #clicks == 1 ∧ red == this round's peak − 1 | "2 became 1" |
| two-rounds | "Feed the ants twice: put out food, wait until every piece is eaten, then put out a second round and let them finish that too." | #clicks == 2 ∧ board empty now ∧ food appeared and was fully eaten between the clicks | the hidden `clicks` counter → the action trace |
| three-rounds | same, three rounds | same with 3 | same |
| intercept | "Put out a single round of food and stop while an ant is right next to a piece it has not eaten yet." | #clicks == 1 ∧ 2 red ∧ some ant orthogonally adjacent to a food | replaces a named mid-forage FRAME with a mid-forage RELATION |

- The rounds checkers no longer read `_h(s, "clicks")`; they count clicks and verify from the
  frames that each round actually emptied before the next click. That is what separates
  "feed them twice" from "click twice".
- `intercept` was marked NL-inexpressible in the original plan. It is expressible — but see
  §5 before shipping it.

### 83wkq — 1

| problem | sentence | checker | relaxes |
|---|---|---|---|
| spawn-two | "Create a particle at row 4, column 4; wait a tick; then create a second at row 10, column 10." | exactly 2 clicks, at those two cells, ≥1 non-click between them, 2 blue cells, (4,4) empty and (10,10) blue | replaces the RNG-dependent exact frame with the rule-derivable consequence |

The load-bearing noop survives the rewrite: without a gap the second click suppresses
diffusion, the first particle stays where it was clicked, and `(4,4) empty` fails.

## 5. Two things the sentence cannot fix on its own

**A goal the environment completes on its own is free.** The pilot found this on
`all-eaten` (online: click, then 14 noops, 0% replanned, 1.00). `intercept` is the same shape
and worse: an ant *must* pass through adjacency on its way to the food, so "click once and
wait" satisfies it without predicting anything. Its floor is 0.000 only because random drives
click too often to ever hold at one. **A low floor is evidence that flailing does not find
the goal, not that the goal is hard.** Ship `intercept`, and report it in the same bucket as
`all-eaten` — or give both a terminal-scoring variant, which is the only scoring rule that
can tell "I knew it was done" from "I waited and the harness noticed".

**Short problems are floor-dominated, and only some of that is the budget.** Two responses,
both cheap, and they compose:

- *Free, no re-run:* the eval already records `satisfied_at`, so a second column
  `pass@1 within 2h actions` costs five lines and no compute, and it comes with a floor
  already measured (§3). It rescues the budget artifacts — `park-cloud` 0.450 → 0.037,
  `coin-ground` 0.152 → 0.010, `walk` 0.138 → 0.037 — and, being derived from the same
  rollouts, it cannot disagree with the primary metric about what happened.
- *Honest labelling:* it does **not** rescue `platform` (0.247 at 2h), `nightfall` (0.310) or
  `spawn` (0.745), and neither does a tightened sentence (§3.1), and no budget will: at h=1
  the floor is just "did a random drive pick the right verb". Keep those three at cap 50 for
  pairing with the frame-goal run and keep them out of headline averages.

## 6. Code

The eval, the online eval and both vizzes are pid-driven — they iterate `GOALS` and join to
`problems.json` by id — so **phase 2 needs no changes to any of them**. What does change:

| file | change |
|---|---|
| `offline_learning/nl_goals.py` | +25 `NLGoal` entries, ~20 checkers, shared helpers (`only`, `resting`, `flat_run`, `tower`, `block2x2`, `nverb`, `clicks_at`), ≥1 positive and ≥3 negatives each |
| `nl_goals._n2ntd_all_coins` | narrow the occluder guard to "no bullet on a coin cell" (§4). Provably does not move any pilot number — all ten n2ntd rollouts have zero clicks — but the shipped checker is wrong and the two kill problems walk straight into it |
| `nl_goals.NLGoal` | new optional `naive: Check` field — the guard-free variant used by N6 |
| `scripts/validate_nl_goals.py` | delete the hardcoded `naive_check` if-chain (it raises `KeyError` on any pid it does not know); cache traced random drives per `(game, seed)` instead of re-tracing per goal — the pilot's validator would take ~40 min at 30 goals, the shared-drive version ~12 |
| `scripts/eval_curated_nl.py` | +5 lines for the `within 2h` column and the floor-dominated bucket |

## 7. Order and cost

| step | output | cost |
|---|---|---|
| 0 | the four checker fixes this analysis already forces: narrow the n2ntd bullet guard, add the `park-cloud` / `infect-one` clauses, tighten `spawn-two` to named cells | — |
| 1 | author bt3gb (7) + s2kt7 (5) — the two games whose checkers are pure shape/count, no cross-object reasoning | — |
| 2 | author dq8gc (6) + n2ntd (6) + 83wkq (1) | — |
| 3 | validate all 30, iterate on N6/N8 failures | ~15 min CPU/round |
| 4 | offline run, 30 problems, `lmwm`, **10 attempts** (the pilot's 5 gave 0.80/0.40/0.00 on the same config three times) | ~10 min, ~$1.50 |
| 5 | online run, 30 problems, 5 attempts | ~90 min, ~$25 |
| 6 | write-up: per-tier NL vs frame, and the offline→online delta that phase 1 found | — |

Author by game, not by tier: the helpers and the hazards are per game, and every mistake
found in one problem of a game applies to its siblings.

## 8. Open decisions

- **`raw` arm?** The pilot ran `lmwm` only. Adding `raw` doubles the offline cost (~$3) and
  answers "does the learned world model help when the goal is words" — the natural headline.
- **Terminal-scoring variant** for `intercept` / `all-eaten` / the toggles: cheap offline
  (rescore the same rollouts at the final frame), and it is the only way to separate knowing
  from waiting.
- **Which planner.** The pilot used `google/gemini-3.7-flash` (the default `deepseek-v4-flash`
  hung). The paired comparison against `logs/2026-08-18/curated/eval/offline.json` still needs
  one run on that baseline's planner.

---

# Phase 2 built — 2026-08-20

Decisions taken by the user before the build: **add the `raw` arm**, **no terminal-scoring
variant**, **plan with `deepseek-v4-flash`** (the frame-goal baseline's planner, so the
comparison is paired), and **one attempt per problem per arm** in both evals.

One attempt is a deliberate trade, and its cost is known: no pass@n, every per-problem cell
is a single Bernoulli draw, and the five floor-flagged problems cannot be told apart from
their floor on one sample. What buys it back is that the online checkpoint keys on
`pid|arm|attempt`, so a later run at `--attempts 5` re-runs only the four missing rollouts.
The 1-attempt run is a first sample, not a replacement for a 5-attempt one.

## What was built

| file | change |
|---|---|
| `nl_goals.py` | 25 new goals, 20 checkers, shared helpers (`nverb`, `click_cells`, `flat_run`, `staircase`, `block2x2`, `mario`, `standing_on`, `no_bullet_on`, `five_visible`, `peak_food`, `night`); `NLGoal.ref` (a goal's own reference plan) and `NLGoal.naive` (the guard-free variant, for N6) |
| `nl_goals._n2ntd_all_coins` | occluder guard narrowed to the coin cells — the shipped version rejects any board with a bullet on it, and a blocked bullet stalls for ever |
| `validate_nl_goals.py` | `naive_check`'s hardcoded pid chain replaced by `goal.naive`; random drives traced once per `(game, seed)` and shared by that game's goals; the floor at every budget derived from one drive set by truncation; three-way verdict (PASS / FLOOR / FAIL) |
| `eval_curated_nl.py` | `--attempts` |
| `eval_curated_nl_online.py` | `--arms` (the online script was lmwm-only), `--attempts`, arm in the checkpoint key, per-arm reporting |
| `viz_nl_online.py` | per-arm rollouts, chips and summary rows |

Every fixture was replayed through the engine as it was written. Three were wrong on the
first try and the engine said so:

- **The mirrored staircase did not build.** `down,noop,down,noop,down` merges the last two
  drops into one cell when they are ICE: a new drop appears at row 1, and the previous one
  cannot move on a `down` tick (the handler assigns `water`, which suppresses the fall
  clause that tick), so it is still there. Liquid hides this because it slides sideways.
  The curated route uses `down,noop,down,noop,noop,down`, and so does the fixture now.
- **`infect-one`'s "two infected" negative was accepted**, because a run that infects two
  passes THROUGH one. The negative that works infects two in the same tick: (5,6) is
  orthogonally adjacent to both (5,7) and (6,6), so the count goes 1 → 3.
- **The bullet guard is vacuous on three of the four coin problems.** Naming the survivors
  as a set already excludes occlusion, and `kill-one-coin` needs ammo, which needs a
  collection, so the count can never be faked either. Only the two "no gold at all" forms
  need the clause. Dropped from the other three rather than left in as decoration.

## Validation, all 30

`21 PASS / 9 FLOOR / 0 FAIL` (`logs/2026-08-20/nl_full/validation.json`). FLOOR means the
checker is sound — reference satisfied, false at the start, noop-proof, every fixture
correct, cross-driver agreement — but random play finds the goal often enough that the score
has to be read against `rand@50`. The verdict was split three ways for this run precisely so
that the one real bug did not hide behind nine easy problems.

**The real bug: `coin-ground` accepted "collect all three coins" as a negative.** Under
any-step scoring a run that collects the low coin FIRST and then takes the others passes
through the accepting state, and the prefix earns the credit. So `leave the other two alone`
is enforceable only at the accepting instant, never for the rest of the run. The fixture is
now "collected the mid-air coin instead", and the limitation is recorded on the goal. It does
not arise for `coin-air`, whose coin is not the one a three-coin route takes first.

The three tightenings from §3.1 all held at full scale: `park-cloud` 0.450 → **0.020**
(PASS), `infect-one` 0.608 → **0.175** (FLOOR), `spawn-two` 0.510 → **0.000** (PASS).

`nl_h < h` on four goals — `all-coins` 32→26, `all-coins-kill` 34→28, `staircase` 33→27,
`coin-air` 11→5. That gap is what the frame goal was charging for trailing settle-noops and
one exact landing cell.

## Offline results — 30 problems, raw + lmwm, 1 attempt, deepseek-v4-flash

9 minutes, **$0.09** (`logs/2026-08-20/nl_full/eval/offline.md`). Paired against the
frame-goal run in `logs/2026-08-18/curated/eval/offline.json`, which used the same planner and
provider pin but **5** attempts — so the NL column is one Bernoulli draw per cell and the
frame column is a mean of five. SE on a 30-problem mean at one attempt is ≈ 0.09.

| subset | n | raw NL | raw frame | lmwm NL | lmwm frame |
|---|--:|--:|--:|--:|--:|
| all | 30 | 0.267 | 0.220 | 0.367 | 0.473 |
| sound only (PASS) | 21 | 0.190 | 0.152 | 0.286 | 0.429 |
| floor-flagged | 9 | 0.444 | 0.378 | 0.556 | 0.578 |
| L3 + L4 | 14 | 0.143 | 0.029 | 0.071 | 0.143 |

**The two arms move in opposite directions when the goal becomes a sentence.** `raw` is flat
to slightly up (0.220 → 0.267 overall, 0.152 → 0.190 on the sound subset); `lmwm` drops
(0.473 → 0.367, and 0.429 → 0.286 where it matters). That is about 1.5 SE, so suggestive
rather than settled — but it is the direction the design predicted, and for a mechanical
reason: **in the frame condition the `lmwm` arm received the goal through its own perception
module** (`_z_goal` renders the goal frame into the same feature language as the state).
An English goal removes that channel. The raw arm never had it, and loses nothing.

Per-problem, the relaxation pays where the sentence is genuinely looser than the frame:
`high-ground` lmwm 0.60 → 1.00, `platform` raw 0.40 → 1.00, and `three-rounds` / `all-eaten`
go 0.00 → 1.00 for raw. It costs where the frame was doing the grounding work:
`swap-drive` 0.60 → 0.00 on both arms (the goal frame showed *which* particle had moved),
`one-ice` lmwm 1.00 → 0.00, `coin-ground` lmwm 0.60 → 0.00.

**Caveat on dq8gc/lmwm**: the Aug-18 baseline ran against `dq8gc_s1.RETIRED_empty_beliefs`,
swapped on 2026-08-19 for the belief-carrying artifact this run uses. Those four dq8gc lmwm
cells mix goal format with artifact change and should not be read as a paired result.

## Online results — 60 receding-horizon rollouts, raw + lmwm, 1 attempt

649 minutes, **$2.79** (`logs/2026-08-20/nl_full/eval/online.md`). Same sentences, same
50-action budget, warm start on; each round the planner writes a plan, only the first action
executes, then it replans from the observed state.

| subset | n | raw NL off | raw NL on | raw frame off | raw frame on | lmwm NL off | lmwm NL on | lmwm frame off | lmwm frame on |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| all | 30 | 0.267 | 0.333 | 0.220 | 0.460 | 0.367 | 0.433 | 0.473 | 0.553 |
| sound only | 21 | 0.190 | **0.190** | 0.152 | **0.362** | 0.286 | **0.286** | 0.429 | **0.429** |
| L3 + L4 | 14 | 0.143 | 0.143 | 0.029 | 0.214 | 0.071 | 0.143 | 0.143 | 0.243 |

**The headline is the sound-only row: closing the loop bought nothing on an NL goal, and a
lot on a frame goal.** Frame-goal `raw` goes 0.152 → 0.362 when the loop closes; NL `raw`
goes 0.190 → 0.190, and NL `lmwm` 0.286 → 0.286. On the full 30 the NL arms do gain (+0.07
each), but every one of those gains is on a floor-flagged problem — `coin-ground` (0.152),
`coin-air` (0.122), `infect-one` (0.175).

The mechanism is not mysterious, and it is the same missing channel that cost `lmwm` offline.
**A frame goal gives a receding-horizon agent a progress signal**: every round it can compare
the state it is looking at against the picture it is aiming for. A sentence gives it the same
observations and the same budget and no way to measure distance to the goal, so replanning
re-derives the plan rather than correcting it. That is a claim about MPC, not about
comprehension — the NL agent knows what it is being asked, it just cannot tell whether the
last action helped.

Read with the noise in mind: n=21 at one attempt is SE ≈ 0.10, so the frame Δ of +0.21 is
about 2 SE and the NL Δ of 0.00 is consistent with anything from −0.2 to +0.2. The direction
is worth another run at 5 attempts; the magnitude is not yet established.

**Closing the loop wins and loses in equal numbers**, which the 5-problem pilot never showed:

| flip | arm | |
|---|---|---|
| `coin-air` 0 → 1 | lmwm | forward-model error corrected |
| `coin-ground` 0 → 1 | both | " |
| `all-coins` 0 → 1 | lmwm | reached the goal on action 50 of 50 |
| `infect-one` 0 → 1 | raw | " |
| `chain` 0 → 1 | raw | " |
| `gather` 0 → 1 | raw | " |
| `all-eaten` 1 → 0 | raw | correct open-loop plan, then churn |
| `three-rounds` 1 → 0 | raw | " |
| `spawn-two` 1 → 0 | lmwm | " |

`spawn-two` is the sharpest loss: the task is *click, wait a tick, click* — three actions —
and the closed-loop agent spent fifty without ever committing to the second click. The three
losses are all problems whose open-loop plan was already right, which is the failure mode
one-action-per-round invites: **an agent that re-decides every tick can procrastinate on a
plan it has already got.**

Four rollouts ended on `invalid-plan` rather than budget (`staircase|lmwm` at round 0,
`infect-all-gather|raw` at 3, `gather|lmwm` at 9, `pool|raw` at 26), so those cells are
parse failures, not planning failures.

Vizzes: `logs/2026-08-20/nl_full/viz_offline.html` (30 problems, 60 attempts) and
`viz_online.html` (60 rollouts, 1920 rounds), both arm-aware.
