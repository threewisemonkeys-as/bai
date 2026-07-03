"""Regenerate the clean_data2/<game>/train viz.html with the EXACT 20 scored train
transitions highlighted (gold border + star TRAIN badge).

It reproduces gepa_optimize.py's cross-trajectory seed split bit-for-bit -- same shared
RNG, same consumption order (shuffle train_pool -> shuffle test_pool -> balanced_split
test -> balanced_split train) -- so the highlighted frames are precisely the ones the
sweep optimizes over. As a guard it compares the reproduced TEST split's action balance
against the "test action balance:" line logged by the real run (when available).

    uv run python prototypes/perc_invdyn/mark_train_split_viz.py            # all DEFAULT_GAMES
    uv run python prototypes/perc_invdyn/mark_train_split_viz.py --games dq8gc,aw9wd
"""
from __future__ import annotations
import argparse, csv, importlib.util, random, re, subprocess, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA2 = HERE / "clean_data2"
sys.path.insert(0, str(HERE))
from validate_beliefs import load_transitions, balanced_split  # noqa: E402

# pull GAMES/DEFAULT_GAMES + the split hyperparams straight from the sweep config
spec = importlib.util.spec_from_file_location("clean_sweep", HERE / "clean_sweep.py")
cs = importlib.util.module_from_spec(spec); spec.loader.exec_module(cs)
SEED, TRAIN_N, TEST_N = 1, 20, 10  # clean_sweep defaults (seeds="1", train-n 20, test-n 10)


def row_indices(csvp: Path, whitelist: set[str]) -> list[int]:
    """CSV row index (== build_dataset_viz card index == load_transitions order) of every
    transition that load_transitions keeps, mirroring its exact filter."""
    rows = list(csv.DictReader(csvp.open()))
    out = []
    for i in range(len(rows) - 1):
        r, nxt = rows[i], rows[i + 1]
        action = (r.get("Action") or "").strip()
        obs, obs_next = r.get("Observation") or "", nxt.get("Observation") or ""
        done = (r.get("Done") or "").strip().lower() in ("true", "1")
        if done or not action or not obs.strip() or not obs_next.strip():
            continue
        if whitelist is not None and action.split()[0] not in whitelist:
            continue
        out.append(i)
    return out


def selected_train_rows(game: str, whitelist: set[str], collapse: bool):
    train_dir, test_dir = DATA2 / game / "train", DATA2 / game / "test"
    tr = load_transitions([train_dir], whitelist, context_k=0)
    te = load_transitions([test_dir], whitelist, context_k=0)
    ridx = row_indices(train_dir / "episode_0" / "trajectory.csv", whitelist)
    assert len(ridx) == len(tr), f"{game}: row-index/transition count mismatch ({len(ridx)} vs {len(tr)})"
    for t, ri in zip(tr, ridx):
        t._row = ri
    if collapse:  # gepa collapses action params before bucketing
        for t in tr + te:
            t.action = t.action.split()[0]
    rng = random.Random(SEED)
    train_pool = list(tr); rng.shuffle(train_pool)
    test_pool = list(te); rng.shuffle(test_pool)
    _, test_sel = balanced_split(test_pool, TEST_N, 10 ** 9, rng)
    _, train_sel = balanced_split(train_pool, TRAIN_N, 10 ** 9, rng)
    return sorted(t._row for t in train_sel), Counter(t.action for t in test_sel), len(tr)


def logged_test_balance(game: str) -> Counter | None:
    p = HERE.parent.parent / "logs" / "clean_sweep_gepa_padded_ctxk9" / f"{game}_seed1" / "stdout.txt"
    if not p.exists():
        return None
    m = re.search(r"test action balance: (\{.*\})", p.read_text())
    if not m:
        return None
    return Counter({k: int(v) for k, v in re.findall(r"'([^']+)':\s*(\d+)", m.group(1))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default=cs.DEFAULT_GAMES)
    args = ap.parse_args()
    games = [g for g in args.games.split(",") if g.strip()]
    print(f"{'game':<7} {'pool':>4} {'sel':>3}  {'valid?':<8} selected train rows")
    for g in games:
        wl, keep, *_ = cs.GAMES[g]
        whitelist = set(wl.split(","))
        rows, test_bal, pool = selected_train_rows(g, whitelist, collapse=not keep)
        logged = logged_test_balance(g)
        if logged is None:
            valid = "n/a"
        else:
            valid = "OK" if Counter(test_bal) == Counter(logged) else "MISMATCH"
        d = DATA2 / g / "train"
        subprocess.run([sys.executable, str(HERE / "build_dataset_viz.py"), str(d),
                        "--out", str(d / "viz.html"),
                        "--highlight", ",".join(map(str, rows))], check=True,
                       stdout=subprocess.DEVNULL)
        print(f"{g:<7} {pool:>4} {len(rows):>3}  {valid:<8} {rows}")
    print(f"\nhighlighted train viz -> {DATA2}/<game>/train/viz.html")


if __name__ == "__main__":
    main()
