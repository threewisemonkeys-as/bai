# Previous gemini low-data runs — my own one-by-one failure attribution

**Setting:** gemini-2.5-flash (matched decode + reflection), low-data (5 train==val / 20 clean
test), `--start empty`, GEPA optimizer. AutumnBench (DQ8GC, 7WWW9, ice) text-mode; ARC
(ft09, sp80, ls20) image-mode. Seed 1.

**Method:** reconstructed each run's GEPA *best* candidate (P + B) from the log's candidate
lineage, replayed the same clean 20-item test split, and read every failure's learned belief,
P's output on both frames, the deterministic raw change, and F's reasoning. Reconstructed
accuracy matches the logged accuracy on all six games, so this is the pipeline that actually ran.

Classification is **mine**, not an automated judge. Buckets:
- **PERCEPTION** — learned P incorrect/incomplete: it gave F nothing usable (empty/broken output,
  or an undifferentiated dump that doesn't isolate the decision-relevant change).
- **BELIEF** — P faithfully showed the change, but learned B was missing/wrong for the rule F needed.
- **F_REASONING** — P faithful AND B adequate, but F still decoded the wrong action.
- **NO_SIGNAL** — the action produced no observable change in the frames; not recoverable, no component's fault.

## Summary

| game | mode | acc | learned P | learned B | dominant cause |
|------|------|----:|-----------|-----------|----------------|
| DQ8GC | text | 1.00 | good (per-cell lister) | correct (darkgreen, row/col dirs) | — (no failures) |
| 7WWW9 | text | 0.25 | good (per-cell lister) | **EMPTY** | BELIEF (+ NO_SIGNAL) |
| ice   | text | 0.35 | good (per-cell lister) | **EMPTY** | BELIEF |
| ft09  | image| 0.25 | **EMPTY** | correct-ish (X/Y convention) | PERCEPTION |
| sp80  | image| 0.35 | raw cell dump, agent not isolated | correct (row/col N/E/S/W) | PERCEPTION |
| ls20  | image| 0.15 | **broken** (outputs "none" for all elements) | EMPTY | PERCEPTION |

### Failure tally (73 failures total)

| cause | count | share |
|-------|------:|------:|
| PERCEPTION | 41 | 56% |
| BELIEF | 15 | 21% |
| NO_SIGNAL | 16 | 22% |
| F_REASONING | 1 | 1% |

## The headline finding

In this low-data regime GEPA almost always **locks in improvement on ONE component and leaves the
other at its empty seed**, and *which* component it abandons is effectively arbitrary per game:

- **7WWW9 / ice** — GEPA accepted only a *perception* update; **B stayed empty**. P faithfully shows
  the coordinate change, but with no belief F has to invent the axis/direction convention (and the
  environment dynamics), and it guesses wrong. → BELIEF.
- **ft09** — the opposite: GEPA accepted only *belief* updates; **P stayed empty** (`--start empty`).
  F literally sees `""` for both frames and says "impossible to determine." → PERCEPTION.
- **ls20** — B empty *and* the accepted P is broken: it searches for element ids 11/12/9/3 and reports
  `none` for all of them in every frame even though those elements are present in the raw grid. F
  sees no change → guesses. → PERCEPTION.
- **sp80** — GEPA learned both P and a *correct* B, but P is a naive dump of every cell of every type
  on a 64×64 multi-type scene (with a per-step counter bar at row 0 that ticks regardless of action).
  P never identifies the agent or diffs the frames, so F can't separate the agent's move from
  environmental animation and latches onto the wrong type. B is right; P is the bottleneck. → PERCEPTION.
- **DQ8GC** — the success case: GEPA got *both* the per-cell-lister P and the correct belief. 1.00.

Two corollaries:
1. **The decoder F is essentially never the problem** (1/73). When P surfaces the change and B has the
   convention (DQ8GC), F decodes correctly. Failures are upstream — an unlearned or broken component.
2. **NO_SIGNAL is a real floor (~22%)**: actions that hit a wall / are noop-equivalent (7WWW9, ice) or
   transitions where only an action-invariant counter changes. No optimizer can fix these from the
   observation alone; they cap achievable accuracy below 1.0.

## Per-failure detail

### DQ8GC (text, acc 1.00) — no failures.

### 7WWW9 (text, acc 0.25, B EMPTY) — 15 failures: 8 NO_SIGNAL, 6 BELIEF, 1 F_REASONING
P is good (lists blue/red cell coords; the change is always visible when there is one).
- **NO_SIGNAL ×8** (f0,1,4,5,7,9,10,11): raw frame identical, truth ∈ {click,up,right,left}; the
  action had no visible effect, F reasonably says noop/click. Unrecoverable.
- **BELIEF ×6** (f2,3,6,8,13,14): P shows the blue object shift, but with empty B, F mislabels the
  axes/direction — e.g. f2 col+1 is `right` but F calls it "down"; f6 row−1 is `up` but F calls it
  "left"; f3/f8 (2-cell object shifting 1) F reads as a +2 jump and says "right" for a vertical move.
  A correct convention (row+→down, col+→right) fixes all of these.
- **F_REASONING ×1** (f12): col+1 change clearly in P's output, yet F predicted noop — it ignored a
  visible delta.

### ice (text, acc 0.35, B EMPTY) — 13 failures: 4 NO_SIGNAL, 9 BELIEF
ice has non-trivial dynamics (the object drifts/falls even on `noop`; a controlled gray block shifts).
- **NO_SIGNAL ×4** (f0,4,9,10): identical frames, action had no effect.
- **BELIEF ×9** (f1,2,3,5,6,7,8,11,12): P faithfully shows the motion, but with no belief F cannot
  know that (a) downward drift on `noop` is gravity, not "down" (f1,f6), (b) the *gray block* shift —
  not the falling blue cell — encodes left/right (f3,f5,f12), or (c) a newly spawned cell corresponds
  to `down` (f11). All need a dynamics/convention belief F doesn't have.

### ft09 (image, acc 0.25, P EMPTY) — 15 failures: 14 PERCEPTION, 1 NO_SIGNAL
Every `z_t`/`z_t1` is `""`; F says "states are empty, impossible to determine" and defaults to ACTION4.
- **PERCEPTION ×14** (all with a real raw change): P produced nothing, F is blind.
- **NO_SIGNAL ×1** (f3): truth ACTION0 with no raw change — unrecoverable even with a good P.
- *Caveat:* the only logged raw change is a bottom row-63 bar that grows +2 every step regardless of
  action, so the recoverable signal here looks weak even in principle — but the immediate cause is the empty P.

### sp80 (image, acc 0.35, P+B present, B correct) — 13 failures: 13 PERCEPTION
P dumps every cell of ~15 types on a 64×64 grid; `z_t` and `z_t1` are near-identical walls of coords.
- **PERCEPTION ×13**: P never identifies the single-cell agent nor diffs the frames, and a per-step
  counter bar (Type 0, row 0) plus environmental animations (Type 9 sliding) move every step. F
  cannot isolate the agent's move from the noise — e.g. f1 (truth noop) F sees Type 9 slide +4 cols
  and wrongly reports east. B's N/E/S/W mapping is correct; the fix is a P that isolates the agent /
  reports the frame diff. (Secondary: F mis-attributes motion, but that's enabled by P's dump.)

### ls20 (image, acc 0.15, B empty, P broken) — 17 failures: 14 PERCEPTION, 3 NO_SIGNAL
P outputs `element_11/12/9/3 = none` for **every** frame, although the raw grid clearly contains those
elements (changes at rows 61–62 and a Type-12 block). P is broken — wrong element encoding / parse.
- **PERCEPTION ×14**: real raw change, but P reports `none` → F sees nothing.
- **NO_SIGNAL ×3** (f9,f12,f15): truth ACTION1 with no raw change.

## Implication for the optimizer

The dominant lever is not F and not "more reflection on the decoder." It is making GEPA reliably
improve **both** components in low data instead of abandoning one:
- guard against accepting an *empty/degenerate* component (empty P, `none`-everything P, empty B);
- on complex scenes (ARC) push P toward **isolating the agent / emitting a frame-diff** rather than a
  full cell dump;
- treat the ~22% NO_SIGNAL floor as the realistic accuracy ceiling when reporting.
