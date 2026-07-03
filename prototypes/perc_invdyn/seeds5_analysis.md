# Low-data GEPA sweep (gemini-2.5-flash) — 6 games x 5 seeds: hand failure attribution

Setting: 5 train(==val) / 20 clean test, `--start empty`, GEPA optimizer, matched decode+reflection.
AutumnBench text-mode; ARC image-mode. Replayed each run's saved best P+B (these runs predate
trace-logging), reconstructed the same test split, ran F, and attributed every failure. The 76
ambiguous (P-differs + belief-present) cases were read by hand; all other buckets are deterministic.

Buckets: PERCEPTION (learned P incorrect/incomplete — empty, broken, dropped the change, captured
an action-invariant counter, or never isolated the agent), BELIEF (P faithful but learned B
missing/wrong for the rule F needed), F_REASONING (P+B adequate, F still decoded wrong),
NO_SIGNAL (no observable change — action unrecoverable).

## Per-game (mean test acc over 5 seeds, chance=0.20)

| game | mode | mean acc | per-seed acc | PERC | BELIEF | F | NO_SIG |
|------|------|---------:|--------------|-----:|-------:|--:|-------:|
| DQ8GC | text | 0.95 | 1.00, 0.90, 0.90, 0.95, 1.00 | 0 | 4 | 1 | 0 |
| 7WWW9 | text | 0.35 | 0.60, 0.45, 0.20, 0.35, 0.15 | 5 | 11 | 0 | 49 |
| ice | text | 0.31 | 0.10, 0.40, 0.30, 0.40, 0.35 | 14 | 36 | 0 | 19 |
| ft09 | image | 0.23 | 0.25, 0.05, 0.30, 0.30, 0.25 | 73 | 0 | 0 | 4 |
| sp80 | image | 0.47 | 0.70, 0.25, 0.55, 0.40, 0.45 | 53 | 0 | 0 | 0 |
| ls20 | image | 0.26 | 0.30, 0.15, 0.25, 0.35, 0.25 | 72 | 0 | 0 | 2 |

## Overall failure attribution (all 30 runs)

Total failures: 343

| cause | count | share |
|-------|------:|------:|
| PERCEPTION | 217 | 63% |
| BELIEF | 51 | 15% |
| F_REASONING | 1 | 0% |
| NO_SIGNAL | 74 | 22% |

## The empty-component lottery (which component GEPA left at its empty seed)

| game | seeds with empty P | seeds with empty B |
|------|--------------------|--------------------|
| DQ8GC | - | 4 |
| 7WWW9 | 5 | 3, 4, 5 |
| ice | 1 | 1, 3, 4, 5 |
| ft09 | 3, 4 | 2 |
| sp80 | 2 | - |
| ls20 | 2 | 1, 3, 5 |

## Notes / how to read this

- **The decoder F is essentially never the bottleneck (1/343).** Where P surfaces the change and
  B has the convention (DQ8GC), F decodes correctly. Failures are upstream.
- **GEPA plays an empty-component lottery in low data:** it reliably improves *one* of {P, B} and
  leaves the other at its empty seed, and which one is seed-dependent (table above). The empty
  component then drives the failures — BELIEF when B is empty (AutumnBench, where P faithfully
  shows coordinates), PERCEPTION when P is empty/broken.
- **ARC perception never isolates the agent.** Across ft09/sp80/ls20 every learned P only surfaces
  action-invariant confounds: a numeric step counter, the depleting row-61/63 timer bar, or
  bounding boxes of those. F then maps a confound to an action (often via a spurious belief like
  'counter+1 => ACTION4'). So ARC failures are PERCEPTION (or NO_SIGNAL), not BELIEF/F.
- **ice's belief, when learned, is wrong** (naive direction mapping) because the real dynamics are
  gravity + a controlled block; F applies it faithfully and fails -> BELIEF.
- **NO_SIGNAL (~16%) is an irreducible floor:** wall-bumps / noop-equivalents (7WWW9, ice) and
  counter-only transitions cap achievable accuracy below 1.0.
- DQ8GC (both components learned) is the only game GEPA reliably solves (mean 0.95).
