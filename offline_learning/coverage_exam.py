"""Coverage-driven inverse-dynamics exam: one test set that exercises EVERY core
mechanic of each game (the buckets in `mechanics.MECHANICS`), instead of the
frequency/heuristic sample the informative pools use.

Pipeline:
  census   sweep the human drives, replay through the Autumn engine, classify every
           transition into a mechanic bucket, and report how many examples exist per
           bucket (which core dynamics human play actually reaches).
  build    pick K human transitions per POPULATED bucket (STRATIFIED by board population,
           then spread over users / drives / click locations), engine-SYNTHESIZE a
           transition for every EMPTY bucket, and
           write the whole thing in the id_protocol on-disk shape:
               <out>/<game>/coverage/
                   drives/test_d<k>/episode_0/trajectory.csv    (context source drive)
                   test_d<k>/episode_*/trajectory.csv           (2-row scored slices)
                   mechanics.json                               (per-slice mechanic label)
                   MANIFEST.json / dataset_paths.json
           Then `id_protocol.py --spec coverage` verifies s_true + ceiling and
           `scripts/score_coverage.py` reports per-mechanic ID.

Leakage: coverage test transitions are drawn only from users NOT in the matching
`informative_unified` TRAIN split (the data the learners saw), and the builder asserts
zero shared (x_t, action, x_t1) triples against that train pool.

    uv run python offline_learning/coverage_exam.py census --all
    uv run python offline_learning/coverage_exam.py build --game bt3gb
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_BAI_ROOT = Path(__file__).resolve().parents[1]
if str(_BAI_ROOT) not in sys.path:
    sys.path.insert(0, str(_BAI_ROOT))

import csv
import hashlib
import random
import shutil

from offline_learning.human_replay import (  # noqa: E402
    CSV_FIELDS, DEFAULT_ZIP, GAMES, MIN_SEG, _grid, _nonnoop_count, drive_rows,
    load_sessions, noop_counterfactual, replay, segment, write_episode,
)
from offline_learning.mechanics import _BG, MECHANICS, classify, mechanic_ids  # noqa: E402
from offline_learning.synth import synth_candidates  # noqa: E402

DATA = _BAI_ROOT / "offline_learning/human_data"
# how many drives to scan for the census / selection (nonnoop-ranked). Bounded so the
# O(n^2) noop-counterfactual probe stays cheap; rare buckets are filled by synthesis.
SCAN_DRIVES = 30
K_PER_BUCKET = 4          # scored transitions to select per populated mechanic
K_NULL = 2               # fewer for the static-noop "nothing happened" bucket


def _train_users(game: str) -> set[str]:
    """Users whose drives fed the learners (informative_unified TRAIN split) -- excluded
    from the coverage test set to keep the exam off the training distribution."""
    man = DATA / game / "informative_unified" / "MANIFEST.json"
    if not man.exists():
        return set()
    m = json.loads(man.read_text())
    return {d["user_id"] for d in m["drives"]["train"]}


def _population(grid: str | None, bg: set) -> int:
    """Board 'fullness': number of non-background cells in a grid (JSON 2D list)."""
    if not grid:
        return 0
    return sum(1 for row in json.loads(grid) for c in row if c not in bg)


def _scan_drive(prog: str, d: dict) -> list[dict]:
    """Replay one segment and label every transition with its mechanic bucket."""
    rep = replay(prog, d["seed"], d["actions"])
    acts, grids = rep["actions"], rep["grids"]
    idx = [i for i, a in enumerate(acts) if a.split()[0] != "noop"]
    cf = noop_counterfactual(prog, d["seed"], acts, idx)
    d["rep"] = rep
    code = GAMES_CODE(prog)
    bg = _BG.get(code, set())
    recs = []
    for i, a in enumerate(acts):
        if i + 1 >= len(grids):
            break
        mech = classify(code, grids[i], a, cf.get(i), grids[i + 1])
        recs.append({"i": i, "action": a, "verb": a.split()[0], "mechanic": mech,
                     "changed": grids[i + 1] != grids[i],
                     "pop": _population(grids[i], bg)})  # board fullness BEFORE the action
    return recs


def GAMES_CODE(prog: str) -> str:
    for code, (p, _n, _wl) in GAMES.items():
        if p == prog:
            return code
    raise KeyError(prog)


def scan_game(game: str, data_zip: Path, n_drives: int) -> dict:
    """Rank drives by non-noop activity, scan the top n_drives, tally per-mechanic."""
    prog, human, wl = GAMES[game]
    t0 = time.time()
    sessions = load_sessions(game, data_zip, DATA / "_cache")
    segs = [s for sess in sessions for s in segment(sess, set(wl))]
    segs.sort(key=lambda s: -_nonnoop_count(s))
    train_users = _train_users(game)

    scanned, per_bucket = [], defaultdict(list)
    for d in segs:
        if len(scanned) >= n_drives:
            break
        if d["user_id"] in train_users:      # keep the exam off the learners' train dist
            continue
        recs = _scan_drive(prog, d)
        for r in recs:
            if r["mechanic"]:
                per_bucket[r["mechanic"]].append(
                    {"user": d["user_id"], "seed": d["seed"], "seg": d["seg_idx"],
                     "i": r["i"], "action": r["action"], "pop": r["pop"]})
        scanned.append(d)
        print(f"  [{game}] scanned u={d['user_id'][:8]} seed={d['seed']} "
              f"steps={len(recs)} ({time.time() - t0:.0f}s)", flush=True)

    census = {}
    for m in MECHANICS[game]:
        ex = per_bucket.get(m["id"], [])
        census[m["id"]] = {
            "kind": m["kind"], "n": len(ex),
            "users": len({e["user"] for e in ex}),
            "drives": len({(e["user"], e["seed"], e["seg"]) for e in ex}),
            "locations": len({e["action"] for e in ex}),
        }
    return {"game": game, "human": human, "program": prog,
            "n_drives_scanned": len(scanned),
            "excluded_train_users": len(train_users),
            "census": census, "_examples": dict(per_bucket), "_scanned": scanned}


def print_census(res: dict) -> None:
    g = res["game"]
    print(f"\n===== {g} / {res['human']}   "
          f"({res['n_drives_scanned']} drives scanned, "
          f"{res['excluded_train_users']} train-users excluded) =====")
    print(f"  {'mechanic':<22} {'kind':<8} {'n':>5} {'drives':>7} {'users':>6} {'locs':>5}")
    for mid in mechanic_ids(g):
        c = res["census"][mid]
        flag = "" if c["n"] else "   <-- EMPTY (synthesize)"
        print(f"  {mid:<22} {c['kind']:<8} {c['n']:>5} {c['drives']:>7} "
              f"{c['users']:>6} {c['locations']:>5}{flag}")


# --------------------------------------------------------------------- selection
def _dkey(e: dict) -> tuple:
    return (e["user"], e["seed"], e["seg"])


def _spread(exs: list[dict], k: int, rng: random.Random) -> list[dict]:
    """Pick k examples STRATIFIED by board population (# non-background cells BEFORE the
    action), so a bucket spans empty -> dense boards instead of collapsing to the earliest
    (empty-board) instance per drive -- e.g. particles click-spawn was always the opening
    click into an empty grid, never a click added to an evolving pattern. Population is the
    primary stratum: exs are split into k population quantile bins (low->high) and one pick
    is drawn per bin. Within a bin, prefer a fresh source drive AND a fresh action label
    (click location), so the picks also stay spread over drives / cells as before."""
    if len(exs) <= k:
        return list(exs)
    ordered = sorted(exs, key=lambda e: (e["pop"], rng.random()))  # low->high pop, ties random
    n = len(ordered)
    bins = [[] for _ in range(k)]
    for idx, e in enumerate(ordered):
        bins[min(k - 1, idx * k // n)].append(e)   # k equal-count quantile bins
    picks, seen_loc, seen_drive, bi = [], set(), set(), 0
    while len(picks) < k and any(bins):
        b = bins[bi % k]
        bi += 1
        if not b:
            continue
        pick = (next((e for e in b if _dkey(e) not in seen_drive and e["action"] not in seen_loc), None)
                or next((e for e in b if e["action"] not in seen_loc), None)
                or b[0])
        b.remove(pick)
        seen_loc.add(pick["action"])
        seen_drive.add(_dkey(pick))
        picks.append(pick)
    return picks


def _gsha(obs: str) -> str:
    return hashlib.sha1((_grid(obs) or "").encode()).hexdigest()[:16]


def _synthesize(prog: str, game: str, mid: str) -> dict | None:
    """Try each scripted sequence; keep the first step the detector labels `mid`."""
    for seed, acts in synth_candidates(game, mid):
        rep = replay(prog, seed, acts)
        ra, grids = rep["actions"], rep["grids"]
        idx = [i for i, a in enumerate(ra) if a.split()[0] != "noop"]
        cf = noop_counterfactual(prog, seed, ra, idx)
        for i, a in enumerate(ra):
            if i + 1 >= len(grids):
                break
            if classify(game, grids[i], a, cf.get(i), grids[i + 1]) == mid:
                return {"seed": seed, "rep": rep, "target_i": i}
    return None


def _train_triples(game: str) -> set[tuple[str, str, str]]:
    """(grid_t, action, grid_t+1) triples the learners TRAINED on (informative_unified
    train drives), for a leakage check against the coverage test set."""
    root = DATA / game / "informative_unified" / "drives"
    trips = set()
    for p in sorted(root.glob("train_d*/episode_0/trajectory.csv")):
        rows = list(csv.DictReader(p.open()))
        for i in range(len(rows) - 1):
            a = (rows[i].get("Action") or "").strip()
            if not a:
                continue
            g0, g1 = _grid(rows[i]["Observation"]), _grid(rows[i + 1]["Observation"])
            if g0 and g1:
                trips.add((g0, a, g1))
    return trips


# ------------------------------------------------------------------------ build
def build_game(game: str, data_zip: Path, k: int, scan_drives: int,
               out_root: Path, seed: int = 1) -> dict:
    prog, human, wl = GAMES[game]
    res = scan_game(game, data_zip, scan_drives)
    examples = res["_examples"]
    drives_by_key = {_dkey({"user": d["user_id"], "seed": d["seed"], "seg": d["seg_idx"]}): d
                     for d in res["_scanned"]}
    rng = random.Random(seed)
    train_trips = _train_triples(game)   # learners' training triples, to keep the exam off

    def _leaks(e: dict) -> bool:
        d = drives_by_key.get(_dkey(e))
        g = d["rep"]["grids"]
        i = e["i"]
        return i + 1 >= len(g) or (g[i], e["action"], g[i + 1]) in train_trips

    picks_by_drive: dict[tuple, list] = defaultdict(list)
    coverage: dict[str, dict] = {}
    synth: list[dict] = []
    for m in MECHANICS[game]:
        mid, kind = m["id"], m["kind"]
        if kind == "null":
            # "nothing happened" is not a core dynamic; also the main source of
            # train-leak / context-ambiguity, so it is left out of the scored exam.
            coverage[mid] = {"kind": kind, "n": 0, "source": "excluded"}
            continue
        exs = [e for e in examples.get(mid, []) if not _leaks(e)]   # drop training triples
        if exs:
            for e in _spread(exs, k, rng):
                picks_by_drive[_dkey(e)].append((e["i"], mid))
            coverage[mid] = {"kind": kind, "n": min(k, len(exs)), "source": "human"}
        else:
            got = _synthesize(prog, game, mid)
            if got is None:
                coverage[mid] = {"kind": kind, "n": 0, "source": "unreachable"}
                print(f"  [{game}] {mid}: UNREACHABLE by scripted synthesis", flush=True)
            else:
                synth.append({"mechanic": mid, **got})
                coverage[mid] = {"kind": kind, "n": 1, "source": "synth"}

    # ONE scored slice per test dir (single episode_0), so protocol-item order == the
    # test_dirs order the loader iterates == this labels order (glob within a dir is
    # unsorted, so a dir must never hold more than one scored episode). Slices from the
    # same drive share one source dir for context backfill.
    root = out_root / game / "coverage"
    if root.exists():
        shutil.rmtree(root)
    test_dirs, src_dirs, labels = [], [], []
    source_seeds: dict[str, int] = {}     # source-dir name -> replay seed (for id_protocol)
    di, shared = 0, 0

    def emit(rows, i, mid, synthetic, src_dir, prov):
        nonlocal di, shared
        slice_dir = root / f"test_d{di}"
        write_episode([dict(rows[i]), dict(rows[i + 1])], slice_dir / "episode_0")
        if (_grid(rows[i]["Observation"]), rows[i]["Action"],
                _grid(rows[i + 1]["Observation"])) in train_trips:
            shared += 1
        labels.append({"mechanic": mid, "synthetic": synthetic, "action": rows[i]["Action"],
                       "x_t1_sha": _gsha(rows[i + 1]["Observation"]), "src": prov})
        test_dirs.append(str(slice_dir.resolve()))
        src_dirs.append(str(src_dir.resolve()))
        di += 1

    for h, (key, picks) in enumerate(picks_by_drive.items()):
        rows = drive_rows(drives_by_key[key]["rep"])
        src = root / "drives" / f"human_{h}"
        write_episode(rows, src / "episode_0")
        source_seeds[f"human_{h}"] = key[1]
        for i, mid in sorted(picks):
            emit(rows, i, mid, False, src,
                 {"user": key[0], "seed": key[1], "seg": key[2], "step": i})
    for s in synth:
        rows = drive_rows(s["rep"])
        src = root / "drives" / f"synth_{s['mechanic']}"
        write_episode(rows, src / "episode_0")
        source_seeds[f"synth_{s['mechanic']}"] = s["seed"]
        emit(rows, s["target_i"], s["mechanic"], True, src,
             {"synthetic": True, "seed": s["seed"], "step": s["target_i"]})

    (root / "mechanics.json").write_text(json.dumps(labels, indent=1) + "\n")
    (root / "dataset_paths.json").write_text(json.dumps({
        "run": ",".join(test_dirs), "context_source_run": ",".join(src_dirs),
        "test_run": ",".join(test_dirs), "test_context_source_run": ",".join(src_dirs),
        "actions": ",".join(wl)}, indent=2) + "\n")
    manifest = {"game": game, "human_game": human, "program": prog,
                "n_items": len(labels), "n_synth": len(synth),
                "n_drives_scanned": res["n_drives_scanned"],
                "excluded_train_users": res["excluded_train_users"],
                "shared_with_train": shared, "k_per_bucket": k,
                "source_seeds": source_seeds, "coverage": coverage}
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")

    n_h = sum(1 for m in coverage.values() if m["source"] == "human")
    n_s = sum(1 for m in coverage.values() if m["source"] == "synth")
    n_u = sum(1 for m in coverage.values() if m["source"] == "unreachable")
    print(f"[{game}] {len(labels)} items across {len(MECHANICS[game])} buckets "
          f"({n_h} human, {n_s} synth, {n_u} unreachable); "
          f"leak={shared} -> {root}", flush=True)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["census", "build"])
    ap.add_argument("--game", choices=sorted(GAMES))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--data-zip", default=str(DEFAULT_ZIP))
    ap.add_argument("--scan-drives", type=int, default=SCAN_DRIVES)
    ap.add_argument("--k", type=int, default=K_PER_BUCKET,
                    help="scored transitions to select per populated mechanic")
    ap.add_argument("--out", default=str(DATA))
    args = ap.parse_args()
    games = sorted(GAMES) if args.all else [args.game]
    if not games or games == [None]:
        ap.error("pass --game or --all")

    if args.cmd == "census":
        summary = {}
        for g in games:
            res = scan_game(g, Path(args.data_zip), args.scan_drives)
            print_census(res)
            summary[g] = {"census": res["census"],
                          "n_drives_scanned": res["n_drives_scanned"]}
        out = Path(args.out) / "coverage_census.json"
        out.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\ncensus -> {out}")
    else:
        for g in games:
            build_game(g, Path(args.data_zip), args.k, args.scan_drives, Path(args.out))


if __name__ == "__main__":
    main()
