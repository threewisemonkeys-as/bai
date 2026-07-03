"""Forward-dynamics SCORING prototype: two ways to evaluate a predicted next-state
Z_hat_{t+1} against the TRUE next-state P(X_{t+1}), plus a harness to compare them.

Context: inverse dynamics (validate_beliefs.py / gepa_optimize.py) scores P by
"recover the action A between P(X_t) and P(X_t+1)". We want a SECOND, dual objective
-- forward dynamics: "given P(X_t) and A, predict P(X_t+1)" -- to pressure P toward
Markov sufficiency. The open question is how to SCORE the predicted next-state. This
file prototypes and compares two scorers, WITHOUT yet wiring either into GEPA.

  Method 1 (deterministic structural diff vs the actual P(X_t+1)):
      frame_f1 : F1 over a content-agnostic token multiset of P's OWN output.
      delta_f1 : F1 over the predicted CHANGE vs the true change (set symmetric
                 difference from the start frame) -- insensitive to the large
                 unchanged background, so it actually grades dynamics.
  Method 2 (LLM judge): a delta-aware judge sees start / predicted / true features
      (all in P's vocabulary) and a GENERIC rubric, returns a [0,1] similarity.

PURITY (see memory invdyn-no-external-knowledge): every scorer operates ONLY on
P's emitted symbols and the LOGGED next frame run through the SAME P. There is no
raw-grid parser, no hand-coded "background", no game facts in the judge rubric.
delta_f1's symmetric-difference is NOT raw_delta: raw_delta parsed the raw grid with
a hand-coded background detector; here we set-diff whatever P chose to emit.

We do NOT have a learned forward predictor Fwd yet, and we don't need one to validate
a SCORE. We simulate Z_hat at controlled quality levels and check each metric ranks
them correctly:
  - perfect : Z_hat = true next frame            (should score ~max)
  - stale   : Z_hat = start frame ("nothing changed")  -- the no-op baseline a metric
              MUST beat 'perfect' against, else it gives no gradient toward dynamics
  - wrong   : Z_hat = some OTHER transition's next frame (an unrelated state)
A good scorer: perfect >> wrong, and perfect > stale by a real margin. The
perfect-minus-stale margin is the usable forward-dynamics signal and is upper-bounded
by how much P moves between frames (the blind-P degeneracy, probed explicitly).

Usage:
  uv run prototypes/perc_invdyn/forward_objective.py \
      --run "logs/dynamics_full/DQ8GC/...,logs/seed_autumn/DQ8GC/..." \
      --perception logs/perc_invdyn/sweep_20260615-110303/DQ8GC/seed1/best_perception_gepa_seed1.py \
      --n 24 --task-model google/gemini-2.5-flash
"""

import argparse
import asyncio
import difflib
import random
import re
from collections import Counter
from pathlib import Path

from validate import (  # noqa: E402  (same dir; run via uv from repo root)
    Transition,
    _llm_call,
    _parse_tag,
    load_transitions,
    make_config,
    run_perceive,
)

# A deliberately BLIND perception: constant output regardless of state. Used only to
# probe the degeneracy (FD-alone rewards a P that never moves).
BLIND_P = "def perceive(h):\n    return 'state'\n"


# ---------------------------------------------------------------------------
# Content-agnostic atomizers over P's OWN output (no game semantics).
# ---------------------------------------------------------------------------
def tok_multiset(s: str) -> Counter:
    """All words and integers in the string, as a multiset. Reorder-invariant,
    format-agnostic: works on any text P emits. A single-cell move shows up as a
    couple of changed integer tokens."""
    return Counter(re.findall(r"[A-Za-z]+|\d+", s or ""))


def coord_atoms(s: str) -> set:
    """The set of (r,c)-style coordinate tuples P emitted. Slightly more structured
    than tok_multiset but still generic: any grid P that reports positions. Returns a
    set of normalized '(d,d)' strings."""
    return {f"({a},{b})" for a, b in re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", s or "")}


def _f1_multiset(pred: Counter, true: Counter) -> float:
    if not pred and not true:
        return 1.0
    inter = sum((pred & true).values())
    p = inter / max(1, sum(pred.values()))
    r = inter / max(1, sum(true.values()))
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def _f1_set(pred: set, true: set) -> float:
    if not pred and not true:
        return 1.0
    inter = len(pred & true)
    p = inter / max(1, len(pred))
    r = inter / max(1, len(true))
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# Method 1 scorers (deterministic).
# ---------------------------------------------------------------------------
def frame_f1(start: str, pred: str, true: str) -> float:
    """Whole-frame token-multiset F1 of predicted vs true next state. Dominated by the
    unchanged background, so it is a LENIENT, change-insensitive metric (kept as a
    baseline to expose exactly that weakness)."""
    return _f1_multiset(tok_multiset(pred), tok_multiset(true))


def delta_f1(start: str, pred: str, true: str) -> float:
    """F1 of the PREDICTED change vs the TRUE change, over COORDINATE atoms. Change =
    coord atoms that differ from the start frame (symmetric difference). Grades whether
    the prediction moved the RIGHT cells, ignoring the static background. BUT it assumes
    P writes positions as '(r,c)': on games where P uses '@r,c' (7WWW9), natural language
    (ft09), or counts (sp80/ls20) it extracts nothing/partial -> brittle. Kept to expose
    that. If neither moved, change sets are empty -> 1.0 (the blind-P reward)."""
    a0 = coord_atoms(start)
    true_delta = coord_atoms(true) ^ a0
    pred_delta = coord_atoms(pred) ^ a0
    return _f1_set(pred_delta, true_delta)


def _ms_symdiff(a: Counter, b: Counter) -> Counter:
    """Multiset symmetric difference: tokens that changed count between a and b."""
    return (a - b) + (b - a)


def tok_delta_f1(start: str, pred: str, true: str) -> float:
    """FORMAT-AGNOSTIC change F1. Change = multiset symmetric difference of all
    words+integers vs the start frame, so it catches whatever moved: '(r,c)', '@r,c',
    a 'Count: 894 -> 896', a renamed word. Order-invariant (multiset) -> robust to P
    reordering its output, but loses locality (sees '5' gone / '6' added, not which cell
    moved). No assumption about how P encodes position."""
    s = tok_multiset(start)
    return _f1_multiset(_ms_symdiff(tok_multiset(pred), s),
                        _ms_symdiff(tok_multiset(true), s))


def _word_toks(s: str) -> list:
    """Fine tokenization for diffing: words, integers, and individual punctuation, so
    'blue@7,5;blue@8,5' splits into alignable units rather than one blob."""
    return re.findall(r"\d+|[A-Za-z]+|[^\s\w]", s or "")


def _edit_tokens(a: str, b: str) -> Counter:
    """Multiset of tokens inside the non-equal spans of a plain difflib diff a->b."""
    ta, tb = _word_toks(a), _word_toks(b)
    sm = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
    changed: list = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "equal":
            changed += ta[i1:i2] + tb[j1:j2]
    return Counter(changed)


def textdiff_delta_f1(start: str, pred: str, true: str) -> float:
    """FORMAT-AGNOSTIC change F1 via a PLAIN TEXT DIFF (difflib). The 'true edit' is the
    diff start->true; the 'pred edit' is start->pred; F1 over the changed-token
    multisets. Sequence-aligned, so it keeps locality/structure, but is order-sensitive
    (P reordering its list registers as spurious change)."""
    return _f1_multiset(_edit_tokens(start, pred), _edit_tokens(start, true))


# ---------------------------------------------------------------------------
# Method 2 scorer (LLM judge) -- delta-aware, generic rubric, no game facts.
# ---------------------------------------------------------------------------
JUDGE_TMPL = """You grade how well a PREDICTED next-state matches the TRUE next-state of some
environment. You are given three feature summaries, all in the same opaque feature
language; you do NOT know what the features mean and must not assume any domain facts.

=== START features (the state before) ===
{start}
=== PREDICTED next-state features ===
{pred}
=== TRUE next-state features ===
{true}

Grade ONLY by comparison. Give the most weight to whether PREDICTED reproduced the
CHANGES from START to TRUE (the features that differ between START and TRUE), rather
than merely copying START. A prediction that just repeats START, when TRUE differs from
START, is a POOR prediction. Penalize features PREDICTED invented that are not in TRUE,
and changes in TRUE that PREDICTED missed.

Respond as:
<reasoning>what changed START->TRUE, and whether PREDICTED captured it</reasoning>
<score>a number from 0.0 (unrelated) to 1.0 (matches TRUE, including the changes)</score>"""


async def judge_score_reasoned(cfg, start: str, pred: str, true: str, sem) -> tuple[float, str, float]:
    """Like judge_score but also returns the judge's <reasoning> (for inspection/viz)."""
    prompt = JUDGE_TMPL.format(start=start or "(empty)", pred=pred or "(empty)",
                               true=true or "(empty)")
    async with sem:
        text, cost = await _llm_call(cfg, prompt)
    raw = _parse_tag(text or "", "score")
    m = re.search(r"[01](?:\.\d+)?|\.\d+", raw)
    val = float(m.group()) if m else 0.0
    reasoning = _parse_tag(text or "", "reasoning") or ""
    return max(0.0, min(1.0, val)), reasoning, cost


async def judge_score(cfg, start: str, pred: str, true: str, sem) -> tuple[float, float]:
    val, _reasoning, cost = await judge_score_reasoned(cfg, start, pred, true, sem)
    return val, cost


# ---------------------------------------------------------------------------
# Harness.
# ---------------------------------------------------------------------------
def perceive_pair(code: str, tr: Transition) -> tuple[str, str]:
    return run_perceive(code, tr.x_t)[0], run_perceive(code, tr.x_t1)[0]


def spearman(xs, ys) -> float:
    """Rank correlation, no scipy. Ties get average ranks."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return float("nan") if vx == 0 or vy == 0 else cov / (vx * vy)


async def run(cfg, code, transitions, use_judge, concurrency, label):
    """For every transition, build (start, true) in P's vocabulary and three simulated
    Z_hat qualities, then score each under every metric. Returns per-metric dicts of
    quality -> list of scores."""
    sem = asyncio.Semaphore(concurrency)
    pairs = [perceive_pair(code, tr) for tr in transitions]          # (start, true)
    n = len(pairs)
    # 'wrong' = a different transition's true next frame (deranged so never itself)
    perm = list(range(n))
    rng = random.Random(0)
    rng.shuffle(perm)
    perm = [(j if j != i else (j + 1) % n) for i, j in enumerate(perm)]
    variants = {
        "perfect": [true for _, true in pairs],
        "stale":   [start for start, _ in pairs],
        "wrong":   [pairs[perm[i]][1] for i in range(n)],
    }

    det_scorers = {
        "frame_f1": frame_f1,                  # lenient baseline (background swamps it)
        "delta_f1": delta_f1,                  # coord-atom change -- BRITTLE to P format
        "tok_delta_f1": tok_delta_f1,          # format-agnostic, order-invariant
        "textdiff_delta": textdiff_delta_f1,   # format-agnostic, plain difflib, order-sensitive
    }
    det = {name: {} for name in det_scorers}
    for name, fn in det_scorers.items():
        for q in variants:
            det[name][q] = [fn(s, z, t) for (s, t), z in zip(pairs, variants[q])]

    judge = {q: [] for q in variants}
    if use_judge:
        for q in variants:
            tasks = [judge_score(cfg, s, z, t, sem) for (s, t), z in zip(pairs, variants[q])]
            judge[q] = [r[0] for r in await asyncio.gather(*tasks)]

    metrics = dict(det)
    if use_judge:
        metrics["llm_judge"] = judge
    return metrics, pairs


def report(metrics, pairs, label):
    mean = lambda xs: sum(xs) / max(1, len(xs))
    print(f"\n=== {label} (n={len(pairs)}) ===")
    # how much does P actually move? (caps the achievable forward signal) -- compared
    # format-agnostically on P's raw text, NOT coord atoms (which would miss @r,c / counts)
    moved = sum(1 for s, t in pairs if s.strip() != t.strip())
    print(f"P output changed on {moved}/{len(pairs)} transitions "
          f"({moved/max(1,len(pairs))*100:.0f}%)")
    print(f"{'metric':<12} {'perfect':>8} {'stale':>8} {'wrong':>8} "
          f"{'P-S margin':>11} {'P-W margin':>11}")
    margins = {}
    for m, qs in metrics.items():
        p, s, w = mean(qs["perfect"]), mean(qs["stale"]), mean(qs["wrong"])
        margins[m] = (p - s, p - w)
        print(f"{m:<12} {p:>8.3f} {s:>8.3f} {w:>8.3f} {p - s:>11.3f} {p - w:>11.3f}")
    # agreement between the deterministic delta metric and the judge, on the pooled
    # variants (only where both ran)
    if "llm_judge" in metrics:
        det = sum((metrics["tok_delta_f1"][q] for q in ("perfect", "stale", "wrong")), [])
        jdg = sum((metrics["llm_judge"][q] for q in ("perfect", "stale", "wrong")), [])
        print(f"Spearman(tok_delta_f1, llm_judge) over all variants = {spearman(det, jdg):.3f}")
    return margins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="comma-separated run dirs (pooled)")
    ap.add_argument("--perception", required=True, help="path to a learned best_perception_*.py")
    ap.add_argument("--n", type=int, default=24, help="number of transitions to score")
    ap.add_argument("--actions", default="left,right,up,down,noop")
    ap.add_argument("--task-model", default="google/gemini-2.5-flash")
    ap.add_argument("--client", default="openrouter")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--no-judge", action="store_true", help="skip Method 2 (LLM judge)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cfg = make_config(args.task_model, args.client)
    whitelist = set(filter(None, args.actions.split(","))) or None
    run_dirs = [Path(p) for p in args.run.split(",") if p.strip()]
    transitions = load_transitions(run_dirs, whitelist)
    for t in transitions:
        t.action = t.action.split()[0]
    rng.shuffle(transitions)
    transitions = transitions[: args.n]
    code = Path(args.perception).read_text()
    print(f"transitions={len(transitions)} | P={args.perception}")
    print(f"task_lm={args.task_model} | judge={'OFF' if args.no_judge else 'ON'}")

    use_judge = not args.no_judge
    metrics, pairs = asyncio.run(run(cfg, code, transitions, use_judge, args.concurrency,
                                     "learned P"))
    learned_margins = report(metrics, pairs, "LEARNED P")

    # ---- degeneracy probe: a BLIND (constant) P should be rewarded by FD-alone,
    # which is exactly why FD must be paired with inverse dynamics. ----
    bmetrics, bpairs = asyncio.run(run(cfg, BLIND_P, transitions, use_judge,
                                       args.concurrency, "blind P"))
    blind_margins = report(bmetrics, bpairs, "BLIND P (degeneracy probe)")

    print("\n=== takeaways ===")
    for m in learned_margins:
        lp_s, lp_w = learned_margins[m]
        bp_s, _ = blind_margins[m]
        print(f"- {m}: learned perfect-vs-stale margin {lp_s:+.3f}, perfect-vs-wrong "
              f"{lp_w:+.3f}; blind-P perfect-vs-stale {bp_s:+.3f} "
              f"(near 0 = can't tell blind P moved -> needs ID pairing)")


if __name__ == "__main__":
    main()
