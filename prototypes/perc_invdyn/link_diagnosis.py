"""Link the learned-P parsing bug to the sub-par action-prediction scores.
Runs the ACTUAL gemini seed-3 belief-run P (+B) over seed-3's holdout, and for
every failure shows: P's reported agent cell vs the true darkgreen SET change."""
import asyncio, json, random, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate import make_config, run_perceive, make_choices, load_transitions
from validate_beliefs import predict_action, balanced_split

P_CODE = Path("/tmp/gem_seed3_P.py").read_text()
B_TEXT = Path("/tmp/gem_seed3_B.txt").read_text()


def darkgreen_set(raw):
    s = raw.find('[["'); e = raw.rfind(']]') + 2
    try: g = json.loads(raw[s:e])
    except Exception: return None
    return {(r, c) for r, row in enumerate(g) for c, cell in enumerate(row) if cell == "darkgreen"}


def reported_cell(z):  # pull the (r,c) P reported as the agent
    m = re.search(r"agent_position=\((\d+),\s*(\d+)\)", z) or re.search(r"agent_pos=\((\d+),\s*(\d+)\)", z)
    return (int(m.group(1)), int(m.group(2))) if m else None


async def main():
    rng = random.Random(3)
    cfg = make_config("google/gemini-2.5-flash", "openrouter")
    sem = asyncio.Semaphore(8)
    runs = [Path(p) for p in [
        "logs/dynamics_full/DQ8GC/2026-06-08_12-48-57_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn",
        "logs/seed_autumn/DQ8GC/2026-06-05_14-07-07_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn"]]
    trans = load_transitions(runs, {"left", "right", "up", "down", "noop"})
    rng.shuffle(trans)
    _, holdout = balanced_split(trans, 18, 18, rng)
    pool = sorted({t.action for t in trans})

    async def one(tr):
        z_t = run_perceive(P_CODE, tr.x_t)[0]; z_t1 = run_perceive(P_CODE, tr.x_t1)[0]
        choices = make_choices(tr.action, pool, 5, rng)
        pred, reasoning, _ = await predict_action(cfg, z_t, z_t1, B_TEXT, choices, sem)
        return tr, z_t, z_t1, pred, reasoning

    rows = await asyncio.gather(*(one(t) for t in holdout))
    n = len(rows); correct = sum(1 for r in rows if r[3] == r[0].action)
    multi = sum(1 for tr, *_ in rows if (darkgreen_set(tr.x_t) or set()) and len(darkgreen_set(tr.x_t)) > 1)
    print(f"holdout={n}  acc={correct/n:.2f}  frames_with_multiple_darkgreen={multi}/{n}\n")

    cause = {"P_tracked_static_cell": 0, "P_tracked_wrong_cell": 0, "P_missing": 0, "other": 0}
    for tr, z_t, z_t1, pred, reasoning in rows:
        if pred == tr.action:
            continue
        st, st1 = darkgreen_set(tr.x_t) or set(), darkgreen_set(tr.x_t1) or set()
        removed, added = st - st1, st1 - st            # the true change (agent left A, arrived B)
        pc_t, pc_t1 = reported_cell(z_t), reported_cell(z_t1)
        if pc_t is None or pc_t1 is None:
            tag = "P_missing"
        elif pc_t == pc_t1 and (removed or added):
            tag = "P_tracked_static_cell"              # P saw no change -> F says noop
        elif (pc_t not in removed) or (pc_t1 not in added):
            tag = "P_tracked_wrong_cell"
        else:
            tag = "other"
        cause[tag] += 1
        print("="*88)
        print(f"TRUE={tr.action!r} PRED={pred!r} | #darkgreen {len(st)}->{len(st1)} | "
              f"true change: left {sorted(removed)} arrived {sorted(added)}")
        print(f"  P reported agent: {pc_t} -> {pc_t1}   (from last-darkgreen-in-scan)")
        print(f"  => CAUSE: {tag}")
        print(f"  F reasoning: {reasoning[:220]}")
    print(f"\nFAILURE CAUSES: {cause}")
    print("(P_tracked_static_cell / P_tracked_wrong_cell = directly caused by the multi-darkgreen parse bug)")


if __name__ == "__main__":
    asyncio.run(main())
