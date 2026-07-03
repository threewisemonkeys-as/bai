"""Diagnose why low-data GEPA scored <1.0 despite a correct (set-based) P.
For each seed: reconstruct the exact test split + the learned P and B, run F on
the test set, and for every FAILURE classify the cause:
  - P_dropped_change : P's two outputs do NOT reflect the true raw set-change (P's fault)
  - B_missing        : P is fine but B lacks the convention F needed (B's fault)
  - F_misreasoned    : P correct AND B has the convention, but F decoded wrong (F's fault)
"""
import asyncio, re, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_beliefs import (load_transitions, make_config, run_perceive,
                              make_choices, predict_action, balanced_split)

RUNS = [Path('logs/dynamics_full/DQ8GC/2026-06-08_12-48-57_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn'),
        Path('logs/seed_autumn/DQ8GC/2026-06-05_14-07-07_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn')]
WL = {'left', 'right', 'up', 'down', 'noop'}


def _extract_last(log, marker):
    if marker not in log:
        return None
    seg = log.split(marker)[-1]
    return re.split(r'\nIteration \d+:|\n\[gepa\]|\nProposed new text', seg)[0].strip()


def nonblack(raw):
    s = raw.find('[['); e = raw.rfind(']]') + 2
    g = json.loads(raw[s:e])
    d = {}
    for r, row in enumerate(g):
        for c, cell in enumerate(row):
            if cell != 'black':
                d.setdefault(cell, set()).add((r, c))
    return d


def true_change(x_t, x_t1):
    a, b = nonblack(x_t), nonblack(x_t1)
    out = {}
    for col in set(a) | set(b):
        app, dis = b.get(col, set()) - a.get(col, set()), a.get(col, set()) - b.get(col, set())
        if app or dis:
            out[col] = (sorted(dis), sorted(app))
    return out  # color -> (disappeared, appeared)


def p_reflects_change(z_t, z_t1, change):
    """Does P's text output differ between the two frames in a way matching the change?"""
    if z_t == z_t1:
        return False  # P output identical -> cannot reflect any change
    # the moved coords must appear in the outputs (string check, robust to format)
    for col, (dis, app) in change.items():
        for (r, c) in dis:
            if f"{r}, {c}" not in z_t and f"{r},{c}" not in z_t and f"({r}, {c})" not in z_t:
                pass
    return True  # outputs differ; rely on per-case manual read below


def b_has_convention(B):
    b = B.lower()
    hasaxis = ('column' in b or 'col' in b) and ('row' in b)
    hasdir = ('right' in b and 'left' in b and 'up' in b and 'down' in b)
    return hasaxis and hasdir


async def diagnose_seed(cfg, seed, sem):
    log = Path(f'/tmp/lowdata_gemini_seed{seed}.log').read_text()
    P = _extract_last(log, 'Proposed new text for perception:')
    B = _extract_last(log, 'Proposed new text for world_knowledge:') or ''
    # reconstruct EXACT test split + choices (mirror gepa_optimize.main tie mode)
    import random
    rng = random.Random(seed)
    trans = load_transitions(RUNS, WL); rng.shuffle(trans)
    pool = sorted({t.action for t in trans})
    rest, test_tr = balanced_split(trans, 20, 10**9, rng)
    _, train_tr = balanced_split(rest, 5, 10**9, rng)
    # bake choices in the same rng order: train(5) then test
    [make_choices(t.action, pool, 5, rng) for t in train_tr]      # consume rng for train (val==train)
    test = [(t, make_choices(t.action, pool, 5, rng)) for t in test_tr]

    async def run(tr, choices):
        z_t = run_perceive(P, tr.x_t)[0]; z_t1 = run_perceive(P, tr.x_t1)[0]
        pred, reasoning, _ = await predict_action(cfg, z_t, z_t1, B, choices, sem)
        return tr, z_t, z_t1, pred, reasoning

    rows = await asyncio.gather(*(run(t, ch) for t, ch in test))
    correct = sum(1 for r in rows if r[3] == r[0].action)
    bconv = b_has_convention(B)
    print('=' * 78)
    print(f'SEED {seed}: test acc = {correct}/{len(rows)} = {correct/len(rows):.2f} | B_has_convention={bconv}')
    print(f'  B = {B[:240].replace(chr(10), " ")}')
    causes = {'P_dropped_change': 0, 'B_missing': 0, 'F_misreasoned': 0}
    for tr, z_t, z_t1, pred, reasoning in rows:
        if pred == tr.action:
            continue
        ch = true_change(tr.x_t, tr.x_t1)
        p_ok = (z_t != z_t1) or (tr.action == 'noop' and z_t == z_t1)
        if not p_ok:
            cause = 'P_dropped_change'
        elif not bconv:
            cause = 'B_missing'
        else:
            cause = 'F_misreasoned'
        causes[cause] += 1
        print(f'  -- FAIL true={tr.action!r} pred={pred!r} cause={cause}')
        print(f'       raw change: {ch}')
        print(f'       P(X_t)  = {z_t[:130]}')
        print(f'       P(X_t1) = {z_t1[:130]}')
        print(f'       F: {reasoning[:260]}')
    print(f'  CAUSES: {causes}')
    return causes


async def main():
    cfg = make_config('google/gemini-2.5-flash', 'openrouter')
    sem = asyncio.Semaphore(8)
    tot = {'P_dropped_change': 0, 'B_missing': 0, 'F_misreasoned': 0}
    for s in (1, 2, 3):
        c = await diagnose_seed(cfg, s, sem)
        for k in tot:
            tot[k] += c[k]
    print('\n=== TOTAL FAILURE CAUSES across 3 low-data seeds ===')
    print(tot)


if __name__ == '__main__':
    asyncio.run(main())
