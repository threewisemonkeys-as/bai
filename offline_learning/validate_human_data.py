"""Validate a human_replay.py dataset before it is handed to a learner.

Checks, per game/variant:

  1. ROUND-TRIP   `validate.load_transitions` sees exactly the pool we intended
                  (size, action balance) on both the train and test run-dirs.
  2. BACKFILL     `backfill_context_from_source` succeeds -- every emitted target
                  occurs exactly once, with one unambiguous context, in its drive.
                  rexpure dies at startup if this fails, so it is a hard gate.
  3. VERBATIM     each slice row is byte-identical to the drive row it came from.
  4. LEAKAGE      train and test drives come from disjoint users, and no
                  (X_t, action, X_t+1) triple appears in both pools.
  5. ORACLE       per-target aliasing: re-run the engine from the target state under
                  every whitelist verb (click at the true location) and count how many
                  produce the identical next frame. >1 means even an oracle that knows
                  the dynamics cannot identify the action -- the ID ceiling.

Usage:
    uv run python offline_learning/validate_human_data.py --out offline_learning/human_data
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

_BAI_ROOT = Path(__file__).resolve().parents[1]
if str(_BAI_ROOT) not in sys.path:
    sys.path.insert(0, str(_BAI_ROOT))

from autumn_env import AutumnBenchEnvWrapper  # noqa: E402
from offline_learning.human_replay import GAMES, _grid, _obs_cell  # noqa: E402
from offline_learning.validate import (  # noqa: E402
    backfill_context_from_source,
    load_transitions,
)

csv.field_size_limit(10_000_000)


def read_rows(p: Path) -> list[dict]:
    return list(csv.DictReader(p.open()))


def check_verbatim(root: Path, split: str) -> tuple[int, list[str]]:
    """Every slice row must equal the drive row with the same Step."""
    errs, n = [], 0
    for slice_dir in sorted(root.glob(f"{split}_d*")):
        di = slice_dir.name.split("_d")[-1]
        drive = read_rows(root / "drives" / f"{split}_d{di}" / "episode_0" / "trajectory.csv")
        by_step = {r["Step"]: r for r in drive}
        for ep in sorted(slice_dir.glob("episode_*")):
            for r in read_rows(ep / "trajectory.csv"):
                n += 1
                src = by_step.get(r["Step"])
                if src is None:
                    errs.append(f"{ep.name}: Step {r['Step']} not in drive {di}")
                elif any(src[k] != r[k] for k in r):
                    errs.append(f"{ep.name}: Step {r['Step']} differs from drive {di}")
    return n, errs


def oracle_ceiling(root: Path, split: str, prog: str, whitelist: list[str],
                   limit: int) -> dict:
    """Fraction of targets whose action is uniquely recoverable from the frame pair."""
    verbs = [v for v in whitelist if v != "click"]
    alias_hist, checked = Counter(), 0
    for slice_dir in sorted(root.glob(f"{split}_d*")):
        di = slice_dir.name.split("_d")[-1]
        drive_p = root / "drives" / f"{split}_d{di}" / "episode_0" / "trajectory.csv"
        drive = read_rows(drive_p)
        seed = int(json.loads((root / "MANIFEST.json").read_text())
                   ["drives"][split][int(di)]["seed"])
        acts = [r["Action"] for r in drive if r["Action"]]
        for ep in sorted(slice_dir.glob("episode_*")):
            if checked >= limit:
                break
            rows = read_rows(ep / "trajectory.csv")
            i = int(rows[0]["Step"])
            truth_action = rows[0]["Action"]
            truth_grid = _grid(rows[1]["Observation"])
            cands = list(verbs)
            if "click" in whitelist:
                cands.append(truth_action if truth_action.startswith("click")
                             else "click 0 0")
            matches = 0
            for a in cands:
                env = AutumnBenchEnvWrapper(env_name=prog, task_type="interactive",
                                            max_episode_steps=i + 8, seed=seed,
                                            render_mode="text")
                obs, _ = env.reset(seed=seed)
                dead = False
                for pa in acts[:i]:
                    obs, _r, term, _t, _in = env.step(pa)
                    if term:
                        dead = True
                        break
                if not dead:
                    obs, _r, _t, _tr, _in = env.step(a)
                    if _grid(_obs_cell(obs)) == truth_grid:
                        matches += 1
                env.close()
            alias_hist[matches] += 1
            checked += 1
    unique = alias_hist.get(1, 0)
    return {"checked": checked, "unique": unique,
            "ceiling": round(unique / checked, 3) if checked else None,
            "alias_hist": dict(sorted(alias_hist.items()))}


def validate(game: str, variant: str, out_root: Path, oracle_n: int) -> dict:
    prog, _human, whitelist = GAMES[game]
    root = out_root / game / variant
    paths = json.loads((root / "dataset_paths.json").read_text())
    wl = set(whitelist)
    report: dict = {"game": game, "variant": variant, "errors": []}

    for split, rk, ck in (("train", "run", "context_source_run"),
                          ("test", "test_run", "test_context_source_run")):
        run_dirs = [Path(p) for p in paths[rk].split(",")]
        src_dirs = [Path(p) for p in paths[ck].split(",")]
        trs = []
        for rd, sd in zip(run_dirs, src_dirs):
            tt = load_transitions([rd], wl, context_k=9)
            try:
                backfill_context_from_source(tt, [sd], wl, context_k=9)
            except Exception as e:                      # hard gate: rexpure would die
                report["errors"].append(f"{split} backfill {rd.name}: {e}")
            trs.extend(tt)
        ctx_prev = sum(len(t.ctx_prev) for t in trs) / max(1, len(trs))
        ctx_next = sum(len(t.ctx_next) for t in trs) / max(1, len(trs))
        report[split] = {
            "n": len(trs),
            "verbs": dict(Counter(t.action.split()[0] for t in trs).most_common()),
            "distinct_actions": len({t.action for t in trs}),
            "avg_ctx_prev": round(ctx_prev, 1), "avg_ctx_next": round(ctx_next, 1),
            "triples": {(t.x_t, t.action, t.x_t1) for t in trs},
        }
        n_rows, errs = check_verbatim(root, split)
        report[split]["rows_checked"] = n_rows
        report["errors"] += errs

    overlap = report["train"]["triples"] & report["test"]["triples"]
    report["leak_triples"] = len(overlap)
    man = json.loads((root / "MANIFEST.json").read_text())
    tr_u = {d["user_id"] for d in man["drives"]["train"]}
    te_u = {d["user_id"] for d in man["drives"]["test"]}
    report["leak_users"] = sorted(tr_u & te_u)
    for s in ("train", "test"):
        del report[s]["triples"]

    if oracle_n:
        report["oracle_test"] = oracle_ceiling(root, "test", prog, whitelist, oracle_n)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_BAI_ROOT / "offline_learning/human_data"))
    ap.add_argument("--games", default=",".join(sorted(GAMES)))
    ap.add_argument("--variants", default="informative,raw")
    ap.add_argument("--oracle-n", type=int, default=50,
                    help="targets to run the aliasing check on (0 disables)")
    args = ap.parse_args()
    out_root = Path(args.out)
    reports, bad = [], 0
    for g in args.games.split(","):
        for v in args.variants.split(","):
            r = validate(g, v, out_root, args.oracle_n)
            reports.append(r)
            ok = not r["errors"] and not r["leak_users"] and not r["leak_triples"]
            bad += 0 if ok else 1
            print(f"\n=== {g}/{v} === {'OK' if ok else 'PROBLEMS'}")
            print(f"  train n={r['train']['n']:3d} verbs={r['train']['verbs']} "
                  f"distinct={r['train']['distinct_actions']} "
                  f"ctx={r['train']['avg_ctx_prev']}/{r['train']['avg_ctx_next']}")
            print(f"  test  n={r['test']['n']:3d} verbs={r['test']['verbs']} "
                  f"distinct={r['test']['distinct_actions']} "
                  f"ctx={r['test']['avg_ctx_prev']}/{r['test']['avg_ctx_next']}")
            print(f"  leakage: users={r['leak_users']} shared_triples={r['leak_triples']}")
            if r.get("oracle_test"):
                o = r["oracle_test"]
                print(f"  oracle ID ceiling (test): {o['ceiling']} "
                      f"({o['unique']}/{o['checked']} uniquely identifiable) "
                      f"alias_hist={o['alias_hist']}")
            for e in r["errors"][:5]:
                print(f"  ERROR {e}")
    # merge with any earlier pass so a cheap gate-only run cannot drop oracle results
    path = out_root / "VALIDATION.json"
    merged = {}
    if path.exists():
        for r in json.loads(path.read_text()):
            merged[(r["game"], r["variant"])] = r
    for r in reports:
        merged[(r["game"], r["variant"])] = r
    path.write_text(json.dumps([merged[k] for k in sorted(merged)], indent=2) + "\n")
    print(f"\n{len(reports) - bad}/{len(reports)} datasets clean "
          f"-> {out_root / 'VALIDATION.json'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
