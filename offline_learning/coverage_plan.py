"""Turn the coverage exam (coverage_exam.py) into MULTI-HORIZON PLANNING problems.

The coverage set pins, per game, one labelled transition per core mechanic: a source
drive (replayable under a study seed) plus the exact step `i` where that mechanic fires.
`census` reuses those anchors to build goal-conditioned planning windows and reports how
many NON-DEGENERATE problems each (mechanic x horizon) cell can supply, so we can see
which core dynamics are testable by planning (and at which horizon) before spending any
LLM budget.

Construction ("anchor-terminal"): for a mechanic anchored at step i of drive D (seed s),
and horizon h, the window is
    start = grid[i-(h-1)]     (the state h steps before the consequence)
    goal  = grid[i+1]         (the state right AFTER the mechanic fires)
    gt    = actions[i-(h-1) : i+1]   (the human's own h actions -- reach goal by construction)
so the goal is fixed at the post-mechanic frame and h only slides the start back, giving a
clean difficulty ladder (same target, longer lead-in). A plan of <= h fully-parameterised
actions is executed in the Autumn engine from the prefix-replayed state at i-(h-1); success
iff the final grid equals goal.

Degeneracy screens (a problem is kept only if ALL hold):
  D1  goal != start                              (something must change)
  D2  noop^h from the start does NOT reach goal  (doing nothing must fail)
  D3  random length-h plans rarely reach goal    (not trivially hittable)
and a per-kind COVERAGE/necessity screen tying the window to its mechanic:
  action anchor   ablate the mechanic action (i -> noop) in gt; must NOT reach goal
                  (the action's effect is load-bearing for THIS solution)
  passive anchor  the window has a non-noop setup action (implied by D2) and spans the
                  live passive step -- passive dynamics run every tick, so any correct plan
                  exercises the rule.

A mechanic whose defining outcome is a NO-OP (cloud-clamp, disease click, jump-blocked,
shoot-no-ammo) is UNTESTABLE by planning: its goal is reachable by noop, so every window
built on it is degenerate by D2/necessity. The census flags those rather than shipping a
trivial problem -- planning covers the observably-consequential dynamics.

    uv run python offline_learning/coverage_plan.py census --game bt3gb
    uv run python offline_learning/coverage_plan.py census --all
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

_BAI_ROOT = Path(__file__).resolve().parents[1]
if str(_BAI_ROOT) not in sys.path:
    sys.path.insert(0, str(_BAI_ROOT))

from autumn_env import AutumnBenchEnvWrapper  # noqa: E402
from offline_learning.human_replay import GAMES, _grid, _obs_cell  # noqa: E402
from offline_learning.mechanics import MECHANICS  # noqa: E402

DATA = _BAI_ROOT / "offline_learning/human_data"
csv.field_size_limit(10_000_000)


# --------------------------------------------------------------------- data loading
def _kind(game: str, mid: str) -> str:
    for m in MECHANICS[game]:
        if m["id"] == mid:
            return m["kind"]
    return "?"


def load_coverage(game: str) -> dict:
    """Return {anchors, drives_by_seed} for a game's coverage exam.

    anchors: [{mechanic, kind, synthetic, seed, step}]  (step = index into the drive)
    drives_by_seed[seed] = {"actions": [...], "grids": [...]}  full engine-truth drive
    """
    root = DATA / game / "coverage"
    labels = json.loads((root / "mechanics.json").read_text())

    drives_by_seed: dict[int, dict] = {}
    for ep in sorted((root / "drives").glob("*/episode_0/trajectory.csv")):
        rows = list(csv.DictReader(ep.open()))
        actions = [(r.get("Action") or "").strip() for r in rows]
        grids = [_grid(r["Observation"]) for r in rows]
        # seed: read from MANIFEST source_seeds by dir name
        drives_by_seed.setdefault(ep.parent.parent.name, {})  # placeholder, fixed below
        drives_by_seed[ep.parent.parent.name] = {"actions": actions, "grids": grids}

    man = json.loads((root / "MANIFEST.json").read_text())
    seed_by_name = man["source_seeds"]
    by_seed = {seed_by_name[name]: d for name, d in drives_by_seed.items() if name in seed_by_name}

    anchors = []
    for lab in labels:
        src = lab["src"]
        anchors.append({"mechanic": lab["mechanic"], "kind": _kind(game, lab["mechanic"]),
                        "synthetic": lab["synthetic"], "seed": src["seed"], "step": src["step"]})
    return {"anchors": anchors, "drives_by_seed": by_seed, "program": GAMES[game][0]}


# ------------------------------------------------------------------------ engine
def exec_plan(prog: str, seed: int, prefix: list[str], plan: list[str]) -> list[str | None]:
    """Rendered grid after each plan action, executed from the drive state after `prefix`
    (prefix replay from a reset at this drive's study seed). None once terminated."""
    env = AutumnBenchEnvWrapper(env_name=prog, task_type="interactive",
                                max_episode_steps=len(prefix) + len(plan) + 8,
                                seed=seed, render_mode="text")
    env.reset(seed=seed)
    term = False
    for a in prefix:
        _o, _r, term, _t, _i = env.step(a)
        if term:
            break
    out: list[str | None] = []
    for a in plan:
        if term:
            out.append(None)
            continue
        o, _r, term, _t, _i = env.step(a)
        out.append(_grid(_obs_cell(o)))
    env.close()
    return out


def _dims(grid: str) -> tuple[int, int]:
    g = json.loads(grid)
    return len(g), len(g[0])


def random_plan(game: str, h: int, dims: tuple[int, int], rng: random.Random) -> list[str]:
    verbs = [v for v in GAMES[game][2]]
    plan = []
    for _ in range(h):
        v = rng.choice(verbs)
        plan.append(f"click {rng.randrange(dims[0])} {rng.randrange(dims[1])}"
                    if v == "click" else v)
    return plan


# ------------------------------------------------------------------------ census
def evaluate_window(game: str, prog: str, seed: int, actions: list[str], grids: list[str],
                    anchor: dict, h: int, rng: random.Random, n_random: int) -> dict | None:
    """Build the anchor-terminal window at horizon h and run the degeneracy + coverage
    screens. Returns a verdict dict, or None if the window is out of range."""
    i = anchor["step"]
    t = i - (h - 1)
    if t < 0 or i + 1 >= len(grids) or grids[t] is None or grids[i + 1] is None:
        return None
    start, goal = grids[t], grids[i + 1]
    gt = actions[t:i + 1]
    if len(gt) != h or any(not a for a in gt):
        return None
    dims = _dims(start)

    v = {"mechanic": anchor["mechanic"], "kind": anchor["kind"], "h": h,
         "seed": seed, "t": t, "i": i, "synthetic": anchor["synthetic"],
         "start_grid": start, "goal_grid": goal, "gt_actions": gt}

    # sanity: the human's own actions reach the goal (engine truth)
    gt_end = exec_plan(prog, seed, actions[:t], gt)[-1]
    v["gt_ok"] = (gt_end == goal)

    v["d1_changed"] = (goal != start)
    noop_end = exec_plan(prog, seed, actions[:t], ["noop"] * h)[-1]
    v["d2_noop_solvable"] = (noop_end == goal)
    v["noop_success"] = 1.0 if v["d2_noop_solvable"] else 0.0

    hits = 0
    for _ in range(n_random):
        r_end = exec_plan(prog, seed, actions[:t], random_plan(game, h, dims, rng))[-1]
        hits += (r_end == goal)
    v["random_success"] = hits / n_random if n_random else None

    if anchor["kind"] == "action":
        ablated = [("noop" if k == i else a) for k, a in zip(range(t, i + 1), gt)]
        abl_end = exec_plan(prog, seed, actions[:t], ablated)[-1]
        v["pivotal"] = (abl_end != goal)
        v["necessity_ok"] = v["pivotal"]
    else:  # passive
        v["has_setup"] = any(a.split()[0] != "noop" for a in gt)
        v["necessity_ok"] = v["has_setup"]

    v["nondegenerate"] = bool(v["gt_ok"] and v["d1_changed"] and not v["d2_noop_solvable"]
                              and v["necessity_ok"])
    return v


def census_game(game: str, horizons: list[int], n_random: int, rng: random.Random,
                max_anchors_per_mech: int | None) -> dict:
    cov = load_coverage(game)
    prog = cov["program"]
    by_mech: dict[str, list[dict]] = defaultdict(list)
    for a in cov["anchors"]:
        by_mech[a["mechanic"]].append(a)

    t0 = time.time()
    rows = []
    for mid in [m["id"] for m in MECHANICS[game]]:
        anchors = by_mech.get(mid, [])
        if max_anchors_per_mech:
            anchors = anchors[:max_anchors_per_mech]
        for a in anchors:
            drive = cov["drives_by_seed"].get(a["seed"])
            if drive is None:
                continue
            for h in horizons:
                v = evaluate_window(game, prog, a["seed"], drive["actions"],
                                    drive["grids"], a, h, rng, n_random)
                if v is not None:
                    rows.append(v)
        print(f"  [{game}] {mid}: {len(anchors)} anchors done ({time.time() - t0:.0f}s)",
              flush=True)
    return {"game": game, "human": GAMES[game][1], "rows": rows,
            "n_anchors": len(cov["anchors"])}


# ------------------------------------------------------------------------- report
def report_game(res: dict, horizons: list[int]) -> None:
    game = res["game"]
    rows = res["rows"]
    print(f"\n===== {game} / {res['human']}  "
          f"({res['n_anchors']} mechanic anchors) =====")
    hdr = f"  {'mechanic':<22} {'kind':<8} " + " ".join(f"h={h:<7}" for h in horizons)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for m in MECHANICS[game]:
        mid, kind = m["id"], m["kind"]
        if kind == "null":
            continue
        cells = []
        for h in horizons:
            sel = [r for r in rows if r["mechanic"] == mid and r["h"] == h]
            if not sel:
                cells.append("   -   ")
                continue
            nd = [r for r in sel if r["nondegenerate"]]
            rf = sum(r["random_success"] for r in sel) / len(sel)
            cells.append(f"{len(nd)}/{len(sel)} r{rf:.2f}")
        print(f"  {mid:<22} {kind:<8} " + " ".join(f"{c:<9}" for c in cells))
    # per-game coverage: mechanics with >=1 nondegenerate window at ANY horizon
    consequential = [m["id"] for m in MECHANICS[game] if m["kind"] != "null"]
    covered = {r["mechanic"] for r in rows if r["nondegenerate"]}
    missing = [mid for mid in consequential if mid not in covered]
    print(f"  -> coverable mechanics: {len(covered)}/{len(consequential)}"
          + (f"; NOT coverable by planning: {missing}" if missing else ""))


# ------------------------------------------------------------------------- build
# Passive-consequence mechanics need a longer horizon (the setup action + the delayed
# effect must both fit in the window): enemy-hit needs the bullet-travel time, food-eaten
# the ant walk time. Ice/disease/particles are fully covered by h<=8.
BUILD_HORIZONS = {"bt3gb": [1, 2, 4, 8], "dq8gc": [1, 2, 4, 8], "83wkq": [1, 2, 4, 8],
                  "n2ntd": [1, 2, 4, 8, 12], "s2kt7": [1, 2, 4, 8, 12]}


def _record(game: str, bucket: str, v: dict) -> dict:
    """Serialise a window into a portable problem (drive seed + t + h reconstructs the
    K-history context; grids/gt kept for standalone verification & reporting)."""
    return {"game": game, "bucket": bucket, "mechanic": v["mechanic"], "kind": v["kind"],
            "h": v["h"], "seed": v["seed"], "t": v["t"], "i": v.get("i"),
            "synthetic": v["synthetic"], "gt_actions": v["gt_actions"],
            "start_grid": v["start_grid"], "goal_grid": v["goal_grid"],
            "random_success": v["random_success"], "noop_success": v["noop_success"]}


def _select(cands: list[dict], cap: int) -> list[dict]:
    """Pick up to `cap`, hardest first (lowest random floor), one-per-seed before reuse."""
    cands = sorted(cands, key=lambda r: (r["random_success"], r["seed"], r["t"]))
    picked, seen = [], set()
    for r in cands:
        if len(picked) >= cap:
            break
        if r["seed"] not in seen:
            picked.append(r)
            seen.add(r["seed"])
    for r in cands:
        if len(picked) >= cap:
            break
        if r not in picked:
            picked.append(r)
    return picked


def scan_maintain(game: str, cov: dict, horizons: list[int], cap: int, probe: int,
                  rng: random.Random) -> list[dict]:
    """goal==start 'maintain' windows: the state departs & returns (motion), random plans
    reliably break it. Bounded: at most `probe` engine-evaluated candidates per game."""
    prog = cov["program"]
    raw = []
    for seed, d in cov["drives_by_seed"].items():
        grids, acts = d["grids"], d["actions"]
        N = len(acts)
        for h in horizons:
            for t in range(0, N - h):
                if t > 190 or grids[t] is None or grids[t + h] is None:
                    continue
                if grids[t] != grids[t + h] or any(not acts[t + j] for j in range(h)):
                    continue
                if not any(grids[t + j] is not None and grids[t + j] != grids[t]
                           for j in range(1, h)):
                    continue                                     # static stretch: skip
                raw.append((seed, t, h))
    rng.shuffle(raw)
    cands = []
    for seed, t, h in raw[:probe]:
        d = cov["drives_by_seed"][seed]
        acts, grids = d["actions"], d["grids"]
        dims = _dims(grids[t])
        noop_ret = exec_plan(prog, seed, acts[:t], ["noop"] * h)[-1] == grids[t]
        hits = sum(exec_plan(prog, seed, acts[:t], random_plan(game, h, dims, rng))[-1]
                   == grids[t] for _ in range(4))
        v = {"mechanic": "maintain", "kind": "maintain", "h": h, "seed": seed, "t": t,
             "i": None, "synthetic": False, "gt_actions": acts[t:t + h],
             "start_grid": grids[t], "goal_grid": grids[t + h],
             "random_success": hits / 4, "noop_success": 1.0 if noop_ret else 0.0}
        cands.append(v)
    cands = [c for c in cands if c["random_success"] < 0.5]     # keep the discriminating ones
    return [_record(game, "maintain", c) for c in _select(cands, cap)]


def build_game(game: str, rng: random.Random, n_random: int,
               act_cap: int, wait_cap: int, maintain_cap: int) -> list[dict]:
    cov = load_coverage(game)
    prog = cov["program"]
    horizons = BUILD_HORIZONS[game]
    t0 = time.time()

    act: dict[tuple, list] = defaultdict(list)
    wait: dict[tuple, list] = defaultdict(list)
    for a in cov["anchors"]:
        drive = cov["drives_by_seed"].get(a["seed"])
        if drive is None:
            continue
        for h in horizons:
            v = evaluate_window(game, prog, a["seed"], drive["actions"], drive["grids"],
                                a, h, rng, n_random)
            if v is None or not v["gt_ok"]:
                continue
            if v["nondegenerate"]:
                act[(v["mechanic"], h)].append(_record(game, "act", v))
            elif v["kind"] == "passive" and v["d1_changed"] and v["d2_noop_solvable"]:
                wait[(v["mechanic"], h)].append(_record(game, "wait", v))
    problems = []
    for cell in act.values():
        problems += _select(cell, act_cap)
    for cell in wait.values():
        problems += _select(cell, wait_cap)
    problems += scan_maintain(game, cov, horizons, maintain_cap, probe=80, rng=rng)
    print(f"  [{game}] built {len(problems)} problems ({time.time() - t0:.0f}s)", flush=True)
    return problems


def report_composition(problems: list[dict], horizons_by_game: dict) -> None:
    from collections import Counter
    games = ["bt3gb", "dq8gc", "n2ntd", "s2kt7", "83wkq"]
    human = {g: GAMES[g][1] for g in games}
    total = len(problems)
    print("\n" + "=" * 66)
    print(f"PROBLEM SET COMPOSITION  ({total} problems)")
    print("=" * 66)

    print(f"\n  {'bucket':<12} {'count':>6} {'share':>7}   what it tests")
    tests = {"act": "cause a change (action pivotal)",
             "wait": "predict passive evolution & wait (noop reaches goal)",
             "maintain": "recognise stability, don't flail (goal==start)"}
    for b in ("act", "wait", "maintain"):
        n = sum(p["bucket"] == b for p in problems)
        print(f"  {b:<12} {n:>6} {n/total:>6.0%}   {tests[b]}")

    print(f"\n  per game x bucket")
    print(f"  {'game':<16} {'act':>5} {'wait':>5} {'maint':>6} {'total':>6}")
    for g in games:
        gp = [p for p in problems if p["game"] == g]
        c = Counter(p["bucket"] for p in gp)
        print(f"  {g+'/'+human[g]:<16} {c['act']:>5} {c['wait']:>5} "
              f"{c['maintain']:>6} {len(gp):>6}")

    print(f"\n  act+wait by horizon")
    hs = sorted({p["h"] for p in problems})
    print("  " + "bucket".ljust(10) + " ".join(f"h={h:<4}" for h in hs))
    for b in ("act", "wait", "maintain"):
        row = [sum(p["bucket"] == b and p["h"] == h for p in problems) for h in hs]
        print("  " + b.ljust(10) + " ".join(f"{n:<5}" for n in row))

    # mechanic coverage across act+wait
    print(f"\n  mechanic coverage (act|wait buckets)")
    covered = Counter()
    for p in problems:
        if p["bucket"] in ("act", "wait"):
            covered[(p["game"], p["mechanic"])] += 1
    for g in games:
        cons = [m["id"] for m in MECHANICS[g] if m["kind"] != "null"]
        hit = [m for m in cons if covered[(g, m)]]
        miss = [m for m in cons if not covered[(g, m)]]
        print(f"  {g+'/'+human[g]:<16} {len(hit)}/{len(cons)}"
              + (f"  (no problem: {miss})" if miss else ""))

    # sanity baselines
    noop = sum(p["noop_success"] for p in problems) / total
    randf = sum(p["random_success"] for p in problems) / total
    print(f"\n  baselines over whole set: always-noop {noop:.2f} | random {randf:.2f}")
    print("  (act punishes always-noop; wait/maintain punish always-act -> self-balancing)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["census", "build"])
    ap.add_argument("--game", choices=sorted(GAMES))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--horizons", default="1,2,4,8")
    ap.add_argument("--n-random", type=int, default=4)
    ap.add_argument("--max-anchors", type=int, default=0,
                    help="cap anchors scanned per mechanic (0 = all; for a quick probe)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--act-cap", type=int, default=3, help="build: kept per (mechanic,h)")
    ap.add_argument("--wait-cap", type=int, default=2, help="build: kept per passive (mechanic,h)")
    ap.add_argument("--maintain-cap", type=int, default=5, help="build: kept per game")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    horizons = [int(h) for h in args.horizons.split(",") if h]
    games = sorted(GAMES) if args.all else ([args.game] if args.game else [])
    if not games:
        ap.error("pass --game or --all")
    rng = random.Random(args.seed)

    if args.cmd == "census":
        out = Path(args.out or (_BAI_ROOT / "logs/coverage_plan_census.json"))
        summary = {}
        for g in games:
            res = census_game(g, horizons, args.n_random, rng, args.max_anchors or None)
            report_game(res, horizons)
            summary[g] = res
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"horizons": horizons, "n_random": args.n_random,
                                   "games": summary}, indent=1) + "\n")
        print(f"\nwrote {out}")
        return

    # build
    out = Path(args.out or (_BAI_ROOT / "logs/coverage_plan_problems.json"))
    problems = []
    for g in games:
        problems += build_game(g, rng, args.n_random, args.act_cap,
                               args.wait_cap, args.maintain_cap)
    report_composition(problems, BUILD_HORIZONS)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "buckets": ["act", "wait", "maintain"],
        "horizons_by_game": BUILD_HORIZONS, "n_random": args.n_random,
        "caps": {"act": args.act_cap, "wait": args.wait_cap, "maintain": args.maintain_cap},
        "n": len(problems), "problems": problems}, indent=1) + "\n")
    print(f"\nwrote {out} ({len(problems)} problems)")


if __name__ == "__main__":
    main()
