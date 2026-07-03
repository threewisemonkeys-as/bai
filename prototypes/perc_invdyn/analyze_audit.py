"""Quantify whether textdiff (a CHANGE-fidelity score) tracks STATE closeness of Ẑ to
the true next state. For every forward prediction compute, alongside td_real:
  state_f1  : token-multiset F1 of z_hat vs z_t1 DIRECTLY (global state similarity,
              includes the static background -- the naive 'how close are these states')
  td_tok    : order-INVARIANT change score (multiset symdiff) -- isolates how much of
              td_real's behavior is the difflib order-SENSITIVITY
Then report per-env means, the textdiff-vs-state divergence, and the worst offenders in
each direction (harsh = td<<state, lenient = td>>... not possible since td is change).
"""
import difflib
import json
import re
from collections import Counter
from pathlib import Path

rows = json.load(open(Path(__file__).resolve().parent / "diag_forward_results.json"))["rows"]


def toks(s):
    return Counter(re.findall(r"\d+|[A-Za-z]+|[^\s\w]", s or ""))


def f1(a, b):
    if not a and not b:
        return 1.0
    inter = sum((a & b).values())
    p = inter / max(1, sum(a.values()))
    r = inter / max(1, sum(b.values()))
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def symdiff(a, b):
    return (a - b) + (b - a)


def td_tok(z_t, z_hat, z_t1):
    s = toks(z_t)
    return f1(symdiff(toks(z_hat), s), symdiff(toks(z_t1), s))


def state_f1(z_hat, z_t1):
    return f1(toks(z_hat), toks(z_t1))


def reordered(z_hat, z_t1):
    """z_hat is a pure PERMUTATION of z_t1 (same multiset of tokens, different order)."""
    return toks(z_hat) == toks(z_t1) and (z_hat or "").strip() != (z_t1 or "").strip()


for r in rows:
    r["state_f1"] = state_f1(r["z_hat"], r["z_t1"])
    r["td_tok"] = td_tok(r["z_t"], r["z_hat"], r["z_t1"])
    r["reorder"] = reordered(r["z_hat"], r["z_t1"])

envs = ["DQ8GC", "7WWW9", "ice", "ft09", "sp80", "ls20"]
mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")

print("Per-env means (MOVED transitions only):")
print(f"{'env':7} {'n':>3} {'td_real':>8} {'td_tok':>7} {'state_f1':>9} {'|td-state|':>10}")
for e in envs:
    g = [r for r in rows if r["env"] == e and r["moved"]]
    if not g:
        print(f"{e:7} {'0':>3}  (no moved)")
        continue
    print(f"{e:7} {len(g):>3} {mean([r['td_real'] for r in g]):>8.3f} "
          f"{mean([r['td_tok'] for r in g]):>7.3f} {mean([r['state_f1'] for r in g]):>9.3f} "
          f"{mean([abs(r['td_real']-r['state_f1']) for r in g]):>10.3f}")

allm = [r for r in rows if r["moved"]]
import statistics
def spear(xs, ys):
    rx = {v: i for i, v in enumerate(sorted(range(len(xs)), key=lambda k: xs[k]))}
    ry = {v: i for i, v in enumerate(sorted(range(len(ys)), key=lambda k: ys[k]))}
    a = [rx[i] for i in range(len(xs))]
    b = [ry[i] for i in range(len(ys))]
    return statistics.correlation(a, b) if len(set(a)) > 1 and len(set(b)) > 1 else float("nan")

print(f"\nOver {len(allm)} MOVED items:")
print(f"  Spearman(td_real, state_f1) = {spear([r['td_real'] for r in allm], [r['state_f1'] for r in allm]):.3f}")
print(f"  Spearman(td_real, td_tok)   = {spear([r['td_real'] for r in allm], [r['td_tok'] for r in allm]):.3f}")

# reorder defect: same tokens, different order -> td_real penalizes wrongly
ro = [r for r in rows if r["reorder"]]
print(f"\nREORDER cases (z_hat == z_t1 as a multiset, only order differs): {len(ro)}")
for r in ro:
    print(f"  #{rows.index(r)} [{r['env']}] td_real={r['td_real']:.2f} td_tok={r['td_tok']:.2f} "
          f"state_f1={r['state_f1']:.2f} judge={r['judge_real']:.2f}  <- td_real should be 1.0")

# worst HARSH divergences: td_real far below state_f1 (looks far, states actually close)
allr = [r for r in rows if r["moved"]]
allr.sort(key=lambda r: r["state_f1"] - r["td_real"], reverse=True)
print("\nTop 12 HARSH divergences (state close but td_real low):")
print(f"{'#':>4} {'env':7} {'td_real':>7} {'td_tok':>7} {'state':>6} {'judge':>6}")
for r in allr[:12]:
    print(f"{rows.index(r):>4} {r['env']:7} {r['td_real']:>7.2f} {r['td_tok']:>7.2f} "
          f"{r['state_f1']:>6.2f} {r['judge_real']:>6.2f}")
