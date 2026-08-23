"""Oracle inverse-dynamics ceiling of the ARTIFICIAL reference test sets.

`validate_human_data.py` reports, for each human test target, how many whitelist actions
reproduce the recorded next frame when re-run in the engine. >1 means the action is not
identifiable even by an oracle that knows the dynamics perfectly, so it caps the ID score
any learner can reach. That number is only interpretable next to the same measurement on
the artificial test sets the reference runs were scored on -- this computes those.

The reference datasets do not record which seed drove them, so the seed is INFERRED: the
drive's own action sequence is replayed under each candidate seed and the one that
reproduces the drive's recorded observations wins. A game where no candidate reproduces
the CSV is reported as unreproducible rather than silently scored.

    uv run python offline_learning/scripts/oracle_ceiling_ref.py --games bt3gb,dq8gc
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autumn_env import AutumnBenchEnvWrapper  # noqa: E402
from offline_learning.human_replay import GAMES, _grid, _obs_cell  # noqa: E402

csv.field_size_limit(10_000_000)
REF = ROOT / "logs/batch3_consolidated/{game}_s1_batch3/launch.json"
CANDIDATE_SEEDS = [0, 1, 5]


def rows_of(d: Path) -> list[dict]:
    eps = sorted(d.glob("episode_*/trajectory.csv"))
    return [r for p in eps for r in csv.DictReader(p.open())]


def slices_of(d: Path) -> list[list[dict]]:
    return [list(csv.DictReader(p.open()))
            for p in sorted(d.glob("episode_*/trajectory.csv"))]


def infer_seed(prog: str, drive: list[dict]) -> int | None:
    """Seed under which replaying the drive's actions reproduces its recorded frames."""
    acts = [r["Action"] for r in drive if r["Action"]]
    want = [_grid(r["Observation"]) for r in drive]
    probe = min(len(acts), len(want) - 1, 12)
    for seed in CANDIDATE_SEEDS:
        env = AutumnBenchEnvWrapper(env_name=prog, task_type="interactive",
                                    max_episode_steps=probe + 8, seed=seed,
                                    render_mode="text")
        obs, _ = env.reset(seed=seed)
        ok = _grid(_obs_cell(obs)) == want[0]
        for i in range(probe):
            if not ok:
                break
            obs, _r, term, _t, _in = env.step(acts[i])
            ok = _grid(_obs_cell(obs)) == want[i + 1]
            if term:
                break
        env.close()
        if ok:
            return seed
    return None


def ceiling(prog: str, whitelist: list[str], targets: list[Path], sources: list[Path],
            limit: int) -> dict:
    verbs = [v for v in whitelist if v != "click"]
    hist, checked, unreproducible = Counter(), 0, 0
    for tdir, sdir in zip(targets, sources):
        drive = rows_of(sdir)
        seed = infer_seed(prog, drive)
        if seed is None:
            unreproducible += 1
            continue
        acts = [r["Action"] for r in drive if r["Action"]]
        by_step = {r["Step"]: i for i, r in enumerate(drive)}
        for sl in slices_of(tdir):
            for a, b in zip(sl, sl[1:]):
                if checked >= limit:
                    break
                if not a["Action"] or (a.get("Done") or "").lower() in ("true", "1"):
                    continue
                i = by_step.get(a["Step"])
                if i is None or i >= len(acts):
                    continue
                truth_grid = _grid(b["Observation"])
                cands = list(verbs)
                if "click" in whitelist:
                    cands.append(a["Action"] if a["Action"].startswith("click")
                                 else "click 0 0")
                matches = 0
                for act in cands:
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
                        obs, _r, _t, _tr, _in = env.step(act)
                        if _grid(_obs_cell(obs)) == truth_grid:
                            matches += 1
                    env.close()
                hist[matches] += 1
                checked += 1
    return {"checked": checked, "unique": hist.get(1, 0),
            "ceiling": round(hist.get(1, 0) / checked, 3) if checked else None,
            "alias_hist": dict(sorted(hist.items())),
            "unreproducible_drives": unreproducible}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="bt3gb,dq8gc,n2ntd,83wkq")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--out", default=str(ROOT / "logs/aug10_human_origin/ref_ceiling.json"))
    args = ap.parse_args()
    out = {}
    for g in args.games.split(","):
        p = Path(str(REF).format(game=g))
        if not p.exists():
            out[g] = {"error": "no reference launch.json"}
            print(f"{g}: no reference launch.json")
            continue
        cmd = json.loads(p.read_text())["cmd"]
        f = {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1) if cmd[i].startswith("--")}
        targets = [Path(x) for x in f["--test-run"].split(",")]
        sources = [Path(x) for x in f.get("--test-context-source-run",
                                          f["--test-run"]).split(",")]
        prog, _h, _wl = GAMES[g]
        r = ceiling(prog, f["--actions"].split(","), targets, sources, args.limit)
        out[g] = r
        print(f"{g}: ceiling={r['ceiling']} ({r['unique']}/{r['checked']}) "
              f"alias={r['alias_hist']} unreproducible_drives={r['unreproducible_drives']}",
              flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
