#!/usr/bin/env python3
"""Build a MECHANIC-BALANCED training variant from an existing human dataset.

`human_replay.pick_targets` round-robins over action VERBS, which leaves the mechanic
mix at whatever the human drives happened to contain — on ice that is 20% cloud-move
but 2% liquid-slide, and on mario the whole bullet subsystem is absent. This script
re-picks the SAME number of train targets from the SAME train drives, round-robining
over MECHANIC (mechanics.classify) instead, so every mechanic present in the drives
gets equal weight.

Only the train slices are re-picked. The test slices, and both context-source drive
dirs, are copied verbatim from the source variant, so a rebalanced run stays
comparable to the original on an identical held-out set. Nothing is drawn from the
test drives or from the coverage drives the planning eval replays.

    uv run python offline_learning/scripts/build_mechbal_dataset.py --game bt3gb
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "offline_learning")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

csv.field_size_limit(10_000_000)

from human_replay import (  # noqa: E402
    GAMES, candidates, noop_counterfactual, replay, write_episode,
)
from mechanics import classify  # noqa: E402


def read_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def pick_by_mechanic(pools: list[tuple[int, list[dict]]], n_want: int,
                     rng: random.Random) -> tuple[list[tuple[int, int]], Counter]:
    """Round-robin over mechanic labels; within a mechanic, spread over drives first.

    Mirrors pick_targets' shape (shuffle, then round-robin, then top up) so the only
    intended difference from the shipped variant is the balancing key.
    """
    by_mech: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for di, cands in pools:
        for c in cands:
            if not c["informative"] or c["mechanic"] is None:
                continue
            by_mech[c["mechanic"]].append((di, c["i"]))
    for m in by_mech:
        rng.shuffle(by_mech[m])
        by_mech[m].sort(key=lambda t: t[0])          # spread over drives before reuse

    mechs = sorted(by_mech)
    picked: list[tuple[int, int]] = []
    mi = 0
    while len(picked) < n_want and any(by_mech[m] for m in mechs):
        m = mechs[mi % len(mechs)]
        if by_mech[m]:
            picked.append(by_mech[m].pop(0))
        mi += 1
    got = Counter()
    for di, i in picked:
        for dj, cands in pools:
            if dj != di:
                continue
            got[next(c["mechanic"] for c in cands if c["i"] == i)] += 1
    return picked, got


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="bt3gb")
    ap.add_argument("--src-variant", default="informative_unified")
    ap.add_argument("--out-variant", default="informative_mechbal")
    ap.add_argument("--n-train", type=int, default=None, help="default: match the source")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    prog, human_name, whitelist = GAMES[args.game]
    root = REPO / "offline_learning/human_data" / args.game
    src, dst = root / args.src_variant, root / args.out_variant
    man = json.loads((src / "MANIFEST.json").read_text())
    n_train = args.n_train or man["stats"]["train"]["n_targets"]
    rng = random.Random(args.seed)

    # ---- label every informative train transition with its mechanic
    pools, drive_rows_by_di = [], {}
    for di, dinfo in enumerate(man["drives"]["train"]):
        rows = read_rows(src / "drives" / f"train_d{di}" / "episode_0" / "trajectory.csv")
        acts = [r["Action"] for r in rows if r["Action"]]
        rep = replay(prog, dinfo["seed"], acts)
        assert rep["grids"][0] is not None, f"replay produced no grid for train_d{di}"
        idx = [i for i, a in enumerate(rep["actions"]) if a.split()[0] != "noop"]
        cf = noop_counterfactual(prog, dinfo["seed"], rep["actions"], idx)
        cands = candidates(rep, cf)
        for c in cands:
            c["mechanic"] = classify(args.game, rep["grids"][c["i"]], c["action"],
                                     cf.get(c["i"]), rep["grids"][c["i"] + 1])
        pools.append((di, cands))
        drive_rows_by_di[di] = rows
        lab = Counter(c["mechanic"] for c in cands if c["informative"])
        print(f"  train_d{di} seed={dinfo['seed']}: {len(cands)} steps, "
              f"{sum(1 for c in cands if c['informative'])} informative, "
              f"{len([m for m in lab if m])} mechanics")

    picks, got = pick_by_mechanic(pools, n_train, rng)
    print(f"\npicked {len(picks)} train targets (want {n_train})")
    for m, c in got.most_common():
        print(f"   {m:24s} {c}")

    # ---- write the variant: new train slices, everything else copied verbatim
    if dst.exists():
        shutil.rmtree(dst)
    by_drive: dict[int, list[int]] = defaultdict(list)
    for di, i in picks:
        by_drive[di].append(i)

    run_dirs, src_dirs = [], []
    balance = Counter()
    for di in sorted(by_drive):
        rows = drive_rows_by_di[di]
        slice_root = dst / f"train_d{di}"
        for n, i in enumerate(sorted(by_drive[di])):
            write_episode([dict(rows[i]), dict(rows[i + 1])], slice_root / f"episode_{n}")
            balance[rows[i]["Action"]] += 1
        run_dirs.append(str(slice_root.resolve()))
        src_dirs.append(str((dst / "drives" / f"train_d{di}").resolve()))
    shutil.copytree(src / "drives", dst / "drives")
    test_runs, test_srcs = [], []
    for di in range(len(man["drives"]["test"])):
        if (src / f"test_d{di}").exists():
            shutil.copytree(src / f"test_d{di}", dst / f"test_d{di}")
            test_runs.append(str((dst / f"test_d{di}").resolve()))
            test_srcs.append(str((dst / "drives" / f"test_d{di}").resolve()))

    out_man = dict(man)
    out_man["variant"] = args.out_variant
    out_man["selection"] = "informative+mechanic-balanced"
    out_man["derived_from"] = args.src_variant
    out_man["mechanic_balance"] = dict(got.most_common())
    out_man["stats"] = dict(man["stats"])
    out_man["stats"]["train"] = {
        "n_targets": len(picks), "n_drives": len(by_drive),
        "balance": dict(balance.most_common()),
        "verbs": dict(Counter(a.split()[0] for a in balance.elements())),
        "mechanics": dict(got.most_common()),
    }
    (dst / "MANIFEST.json").write_text(json.dumps(out_man, indent=2) + "\n")
    (dst / "dataset_paths.json").write_text(json.dumps({
        "run": ",".join(run_dirs),
        "context_source_run": ",".join(src_dirs),
        "test_run": ",".join(test_runs),
        "test_context_source_run": ",".join(test_srcs),
        "actions": ",".join(whitelist),
    }, indent=2) + "\n")
    print(f"\nverbs in the rebalanced train set: "
          f"{dict(Counter(a.split()[0] for a in balance.elements()))}")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
