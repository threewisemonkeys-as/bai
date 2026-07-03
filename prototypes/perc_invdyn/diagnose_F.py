"""Isolate F: feed F a PROVABLY-correct P and a correct B, dump its reasoning on
every failure. Answers: given correct features + correct convention, does F
still reason wrong, and exactly where?"""
import asyncio, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate import make_config, run_perceive, make_choices, GOOD_P, load_transitions
from validate_beliefs import predict_action, balanced_split

# The convention B we learned (verified against data: right: (2,2)->(2,3) => +col).
CORRECT_B = """- The environment is a 2D grid. Coordinates are (row, column).
- The agent is the single green cell; only it moves between consecutive states.
- 'up' decreases the row index; 'down' increases the row index.
- 'left' decreases the column index; 'right' increases the column index.
- If the agent's position is unchanged between the two states, the action was noop."""


def green_coord(raw):
    s = raw.find('[["'); e = raw.rfind(']]') + 2
    if s == -1 or e <= 1:
        return None
    try:
        g = json.loads(raw[s:e])
    except Exception:
        return None
    for r, row in enumerate(g):
        for c, cell in enumerate(row):
            if "green" in cell:
                return (r, c)
    return None


async def main():
    rng = random.Random(3)
    cfg = make_config("google/gemini-2.5-flash", "openrouter")
    sem = asyncio.Semaphore(8)
    runs = [Path(p) for p in [
        "logs/dynamics_full/DQ8GC/2026-06-08_12-48-57_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn",
        "logs/seed_autumn/DQ8GC/2026-06-05_14-07-07_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn",
    ]]
    trans = load_transitions(runs, {"left", "right", "up", "down", "noop"})
    rng.shuffle(trans)
    _, holdout = balanced_split(trans, 18, 18, rng)
    pool = sorted({t.action for t in trans})

    async def one(tr):
        z_t = run_perceive(GOOD_P, tr.x_t)[0]
        z_t1 = run_perceive(GOOD_P, tr.x_t1)[0]
        choices = make_choices(tr.action, pool, 5, rng)
        pred, reasoning, _ = await predict_action(cfg, z_t, z_t1, CORRECT_B, choices, sem)
        return tr, z_t, z_t1, choices, pred, reasoning

    rows = await asyncio.gather(*(one(t) for t in holdout))
    n = len(rows); correct = sum(1 for r in rows if r[4] == r[0].action)
    print(f"holdout={n} correct={correct} acc={correct/n:.2f}\n")
    for tr, z_t, z_t1, choices, pred, reasoning in rows:
        gt = green_coord(tr.x_t); gt1 = green_coord(tr.x_t1)
        dr = (gt1[0]-gt[0]) if gt and gt1 else None
        dc = (gt1[1]-gt[1]) if gt and gt1 else None
        ok = pred == tr.action
        # does P's reported coord match independent ground truth?
        p_ok = (f"({gt[0]},{gt[1]})" in z_t) if gt else None
        if ok:
            continue
        print("="*80)
        print(f"TRUE={tr.action!r}  PRED={pred!r}   green {gt}->{gt1}  (dr={dr},dc={dc})  P_coord_matches_truth={p_ok}")
        print(f"  P(X_t)  : {z_t}")
        print(f"  P(X_t+1): {z_t1}")
        print(f"  choices : {choices}")
        print(f"  F reasoning: {reasoning}")
    print("\n(only failures shown above)")


if __name__ == "__main__":
    asyncio.run(main())
