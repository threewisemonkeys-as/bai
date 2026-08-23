# Forward-prediction metric diagnostic (real LLM predictions, all envs)

Per env: learned P from the sweep, real Fwd(P(X_t),A,B)->Ẑ, scored vs true
P(X_t+1). `real`=score of the LLM prediction, `stale`=score of copying z_t
(predict no change), `lift`=real-stale (did the prediction beat copying?).
Grouped by whether P moved in P-space.

| env | grp | n | textdiff real | td stale | td lift | judge real | judge stale | judge lift |
|-----|-----|--:|--------------:|---------:|--------:|-----------:|------------:|-----------:|
| DQ8GC | moved | 20 | 0.793 | 0.000 | +0.793 | 0.815 | 0.135 | +0.680 |
| DQ8GC | static | 5 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| 7WWW9 | moved | 20 | 0.277 | 0.000 | +0.277 | 0.325 | 0.110 | +0.215 |
| 7WWW9 | static | 8 | 0.250 | 1.000 | -0.750 | 0.688 | 1.000 | -0.312 |
| ice | moved | 20 | 0.193 | 0.000 | +0.193 | 0.148 | 0.095 | +0.053 |
| ice | static | 8 | 0.000 | 1.000 | -1.000 | 0.188 | 1.000 | -0.812 |
| ft09 | static | 8 | 0.875 | 1.000 | -0.125 | 1.000 | 1.000 | +0.000 |
| sp80 | moved | 20 | 0.031 | 0.000 | +0.031 | 0.205 | 0.037 | +0.168 |
| ls20 | moved | 20 | 0.056 | 0.000 | +0.056 | 0.140 | 0.135 | +0.005 |
| ls20 | static | 8 | 0.875 | 1.000 | -0.125 | 1.000 | 1.000 | +0.000 |

Spearman(textdiff_real, judge_real) over 100 MOVED items = 0.636

## sampling
| env | seed | P-moved / total |
|-----|------|-----------------|
| DQ8GC | seed1 | 99/104 |
| 7WWW9 | seed1 | 33/138 |
| ice | seed1 | 53/89 |
| ft09 | seed1 | 0/49 |
| sp80 | seed1 | 49/49 |
| ls20 | seed1 | 42/50 |

## Findings

1. **Forward predictability collapses with env complexity — FD is bottlenecked by the
   forward MODEL (weak frozen task_lm) + metric strictness on verbose P, not only by P
   sufficiency.** DQ8GC (single-cell coords) real=0.79 is the only env F can predict;
   multi-cell drops to 0.19–0.28; ARC (counts/bboxes) is 0.03–0.06. So even a
   Markov-sufficient P earns ~0 FD on hard envs. **FD has real teeth only on DQ8GC,
   where ID is already ~0.95** — that tension is the headline.
2. **textdiff is the stricter, better-calibrated scorer; the judge is systematically
   lenient.** On MOVED, `td_stale`≡0.000 (hard floor: any nonzero real score is genuine
   signal), whereas judge gives stale 0.10–0.14 (compresses dynamic range, shrinks every
   lift). On STATIC the LLM HALLUCINATES motion; textdiff punishes it hard (ice −1.000,
   7WWW9 −0.750) while the judge excuses it (−0.81, −0.31). A scorer that doesn't
   penalize invented motion is a weaker gradient.
3. **Agreement moderate** (Spearman 0.636 over 100 moved items) — lower than the
   simulated harness's 0.778, the gap being the stale-leniency + hallucination-penalty
   divergence.

## Implications for the composite run

- **Default FD = textdiff** (stricter floor, penalizes hallucination, wider range, free).
  The judge's leniency would let a low-information easy-to-copy P score deceptively well
  — the degeneracy ID exists to suppress.
- **Keep `--fd-weight` modest**: FD is near-noise on 4/6 envs, so it must not dominate ID
  there. DQ8GC is the only env where FD currently bites (and ID is near-ceiling).
- **Knob to consider**: run the forward predictor with a STRONGER model than F so FD can
  carry signal on multi-cell/ARC envs (adds cost + a design choice).