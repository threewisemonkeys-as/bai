# logic_gates — testbed assessment (2026-08-23)

Engine-level investigation of whether `logic_gates` (zip-sourced, 24x24) is a valid testbed,
prompted by the 55-game atlas rating it Tier X ("only 4 reachable states; 99 static cells;
8-cell click surface"). Every number below was verified on the interpreter (render after every
step, seed >= 1). Probe scripts: `scripts/testbed_probes/logic_gates/` (p1 census, p2 delay,
p3 BFS, p4 obs size, p5 click surface, p6 truth-table delta, p7 lights_new, p10 variants,
p11 end-to-end, p12 drive, p13 hold-out). Variant programs: `autumn_programs/variants/`.

## VERDICT: USABLE WITH FIXES (keep it; ~1 day of work)

The atlas's Tier-X call is factually wrong on two counts and right on one. It is **not** "4
reachable states" — there are exactly **13 distinct rendered frames (12 recurrent)**, because a
real **1-tick propagation delay** makes the wire colours lag the switches. And the 99 "static"
cells are not static: **up to 60 of them change per transition**. But the concern *underneath*
the tier call is real and blocking: the entire game is **36 distinct transitions**, and a
**51-action drive covers 100% of them**. Any train/test split drawn from the same program is a
memorisation exam — a 12-row state table scores identically to "these are AND/OR/NOT/XOR" on
ID and FD. For the abstract-discovery purpose that is fatal *as-is* and cheaply fixable,
because this game has an unusually clean abstraction probe hiding in it: **on every state
except "both switches on", OR and XOR render identically and AND is constant-dark.** Hold out
that one state and the memoriser is provably wrong on 2 of 4 gate blocks while a learner that
named the gates is right zero-shot. That is exactly the experiment the group was designed to
run.

## Verified rules

24x24, black bg, **99 non-black cells at fixed positions, always** (nothing moves; only
colours change). 6 colours.

| element | cells (col,row) | off / on |
|---|---|---|
| switch1 | 4–5 x 12–13 | pink / red |
| switch2 | 19–20 x 12–13 | pink / red |
| AND out | 12–13 x 4–5 | darkblue / orange |
| OR out | 12–13 x 8–9 | darkblue / orange |
| NOT out | 12–13 x 16–17 | darkblue / orange |
| XOR out | 12–13 x 20–21 | darkblue / orange |
| wires (75 cells) | cols 5 & 19 + connectors | grey / yellow |

- **Clickable surface: exactly 8 of 576 cells** (exhaustively verified): `(4,12) (4,13) (5,12)
  (5,13)` → switch1; `(19,12) (19,13) (20,12) (20,13)` → switch2. All 568 other clicks and
  **all four arrows are exact noops** (no arrow handlers exist).
- **Gates verified exact on all 4 settled states**: AND(s1,s2), OR(s1,s2), **NOT(s1) — switch2
  has no path to NOT**, XOR(s1,s2). **No composition; wiring depth 1.**
- **Propagation delay is real.** Outputs read the *current* switch (`(.. switch1 powered)`) →
  update at t+1. Wires read `(prev switch1)` → update at t+2. **Settle time = 2 ticks.** So the
  destination lights up *before* the signal travels down the wire.
- Interpreter click args are `(col,row)`; `autumn_drive.py` CLI is `click_ROW_COL`, so
  switch1 = `click_12_4`, switch2 = `click_12_19`.

## Exact reachable-state graph

**13 distinct frames; identical at seed 1 and seed 2** (fully deterministic, seed-independent).
State = `(s1, s2, w1, w2)` with `w` = previous `(s1,s2)`; reachable iff Hamming(s,w) <= 1 →
4 + 8 = **12 recurrent states**, plus `S0` (the t=0 frame, never re-entered). Three effective
actions (noop ≡ arrow ≡ non-switch click; click_s1; click_s2); **each state's 3 successors are
distinct frames → 36 distinct transitions total, inverse dynamics 100% identifiable.** Max
shortest-path depth = 3. Changed-cell counts per transition: {0, 12, 16, 31, 43, 44, 47, 56,
60} of 99 — noop changes 0 cells in 4 states and 31–44 cells in the other 8 (that is the
atlas's "noop-change 0.06").

## Concerns, measured

| concern | measured verdict |
|---|---|
| "only 4 reachable states" | **False.** 13 frames / 12 recurrent, 36 transitions, settle 2 ticks. |
| "99 static cells" | **False as stated** — up to 60/99 change per step. But obs is verbose: **5197 chars raw grid, 5539 through `AutumnBenchEnvWrapper` ≈ 1450 tok/frame**, vs ice(16) 2329, wind(17) 2601, mario(12) 1354, blicket(11) 1054. ~2.2x an ice frame; an ID prompt carries ~2900 tok of grid. |
| "8-cell click surface / random finds nothing" | **True and severe.** Random 60-action drives (20 trials): **2.6/60 frames change (4%)**, mean **0.85 switch clicks**, **4.2/36** (state,action) coverage. |
| ID/FD well-posed? | **Yes**, but nearly free: signal is a fixed-position 2x2 pink↔red block. Expect raw-frame P near ceiling → little headroom for P to earn its keep. |
| **generalisation gap** | **The real blocker.** Greedy covering walk: **51 actions cover 36/36 transitions.** In-distribution held-out is impossible. |

## Required fixes

**Fix 1 — held-out-state split (highest value, zero new code).** Train on drives that never
turn both switches on; test on the 5 states with `s=(1,1)` or `w=(1,1)`. Verified: **7 train /
5 test states, 17 train / 19 test transitions.** On the train region: `AND=(0,0,0)`
constant-dark, `NOT=(1,0,1)`, and **`OR=(0,1,1)` is byte-identical to `XOR=(0,1,1)`**. Only the
held-out state separates them: AND=1, OR=1, **XOR=0**, NOT=0. A memoriser is wrong on AND and
XOR; a learner whose belief says "canonical 2-input logic gates" is right zero-shot. Effort:
~1–2 h, reuses `offline_learning/clean_data3_TEST50_METHODOLOGY.md`.

**Fix 2 — authored drive** (`offline_learning/autumn_drive.py` already takes explicit
`--actions`; this is the native path used for clean_data3). A 58-action authored drive
(`p12_drive.py`) gives **38/58 (66%) changed frames and 28/36 coverage vs 2.6/60 and 4.2/36
random.** Effort: ~30 min.

**Fix 3 — variant programs (both built and verified).** `InteractiveEnvironment` just reads
`{data_dir}/programs/{env_name}.sexp` (`MARAProtocol/python_examples/autumnbench/concrete_envs.py:41`),
so integration is a file copy (`tools/install_autumn_programs.py`). Verified
`AutumnBenchEnvWrapper("logic_gates", data_dir=<mirror>)` loads and renders end-to-end.
- `logic_gates_v1` (3-line change: NAND / NOR / BUFFER(s2) / XNOR, same layout, wiring,
  delay): runs, 13 frames, settle 2; **14 of 16 output bits differ from base**. Tests that P
  transfers 100% while B's *content* must be relearned.
- `logic_gates_v2` (two-stage `prev`-composition: O3 = NOT(prev AND) = NAND, O4 = AND(prev OR,
  prev NAND) = XOR): runs, **22 distinct frames, planning depth 4**, truth tables verified. This
  is the only version that can carry a real L1–L4 ladder.
Effort: ~30 min to move + validate. Files: `autumn_programs/variants/logic_gates_v{1,2}.sexp`.

**Fix 4 — score NL goals at the endpoint and never at t=0.** The t=0 frame violates the game's
own truth table (see surprises), so "all outputs dark" is trivially true at t=0 and unreachable
afterwards.

## Proposed NL goals (100 rollouts x 50 random actions, t >= 1)

| goal | reachable | min actions | endpoint floor | any-step floor |
|---|---|---|---|---|
| **light AND while the left wire is still grey** | yes | 2 (`s2, s1`) | **0.00** | 0.05 |
| **light XOR while every wire is still grey** | yes | 1 | **0.00** | 0.51 |
| light the AND output | yes | 2 | 0.06 | 0.07 |
| light AND with XOR dark | yes | 2 | 0.06 | 0.07 |
| light AND with every wire yellow (settled) | yes | 3 | 0.05 | 0.06 |
| light exactly three outputs | yes | 1 | 0.18 | 0.25 |
| light exactly two outputs | yes | 1 | 0.28 | 0.33 |

The two 0.00-floor goals are the good ones: both need the truth table **and** the propagation
delay. Two verified impossible goals make good negatives: "make all four outputs dark" (NOT=0
needs s1=1; OR=0 needs s1=0) and "light AND without ever lighting XOR" (every path to (1,1)
passes through (1,0) or (0,1), where XOR is lit). **Honest ladder answer: base logic_gates
supports L1–L2 and a stretch L3 (max shortest path = 3). L4 needs v2.**

## `lights_new` (sibling)

Verified: 8 clickable cells at (9,4),(9,5),(9,11),(9,12),(10,14),(10,15),(19,11),(19,12) —
four 2-cell switches; **24 distinct frames within depth 6**; 103 cells, 5148 chars. It is
**complementary, not a replacement**: analog 0–4 bar readout, a hidden master-enable
(switch3), a non-monotone light3, and **switch4 is an irreversible breaker that kills the
power supply permanently** — one wrong click bricks the episode, which is a genuine planning
hazard. Its abstraction is an enable-hierarchy, not a truth table. Lower priority.

## Surprising things in the .sexp

1. **Outputs lead wires.** Outputs use the current switch, wires use `(prev switch)` — the
   lamp lights before the signal reaches it.
2. **The t=0 frame violates its own truth table**: `notOutput` is initialised dark although
   NOT(false)=true. Any first action (including noop) corrects it. This is the atlas's
   "passive 0.03".
3. **NOT ignores switch2 entirely** — a learner must discover per-gate fan-in, not a global
   "both switches drive everything".
4. **Duplicate Wire objects** at (5,8) (andWire1/orWire1) and (19,8) (andWire2/orWire2): 13
   objects / 116 cell records rendering 99 distinct cells.
5. **No composition at all**, despite the "logic gates" framing.
6. The single most informative state (both switches on — the only one that separates OR from
   XOR) is the one random exploration essentially never reaches: P(both on in 60 random
   actions) ≈ 0.

**One thing to check before running:** the game's *name* must not reach any prompt.
`invdyn_core.py` prompts carry only `DEFAULT_KNOWLEDGE` (action vocabulary), and the
`Task:`/`Step:` obs header is already stripped, so this looks safe — but a data-dir path named
`logic_gates` leaking into a proposer prompt would hand over the entire abstraction for free.
