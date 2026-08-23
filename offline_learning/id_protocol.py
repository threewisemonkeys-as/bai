"""A frozen, learner-agnostic inverse-dynamics exam.

Today every learner grades itself: REx-pure bakes its own k=5 choice sets from an
uncollapsed action pool and scores an LLM-emitted set with `id_set_metrics`; WorldCoder
bakes different choice sets, collapses click parameters on some games, relaxes bare
`click` via click_enum, and reports its own strict/credit pair. The numbers land in the
same column and are not commensurable.

A PROTOCOL fixes the exam as data. Per (game, pool) it pins:

  * the ordered test transitions (by the dirs + loader that produced them, plus a
    fingerprint so silent drift is caught),
  * one k=5 choice set per item, drawn once from a canonical action pool,
  * `s_true` -- the subset of those five choices that the REAL ENGINE confirms
    reproduce X_t+1, found by replaying the drive prefix and stepping each choice.

`s_true` is what makes scores comparable across pools. Some items are aliased: two
actions genuinely yield the identical next frame, so no model can identify the action
and the best attainable score is below 1. Dividing by the ceiling
`mean_i score(s_true_i)` removes that, and a pool where 20% of items are unidentifiable
stops looking worse than one where none are.

Labels are NEVER collapsed here: the full `click R C` is the label. That keeps engine
verification exact and cheap (no 256-cell click_enum search), and where click location
genuinely does not matter -- s2kt7 spawns food at random positions and ignores the
clicked cell -- the engine simply reports every click choice as consistent, and the
ceiling absorbs it. That is the mechanism working, not a bug.

    uv run python offline_learning/id_protocol.py --spec human --game bt3gb
    uv run python offline_learning/id_protocol.py --spec synthetic --all
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

_BAI_ROOT = Path(__file__).resolve().parents[1]
if str(_BAI_ROOT) not in sys.path:
    sys.path.insert(0, str(_BAI_ROOT))

from autumn_env import AutumnBenchEnvWrapper  # noqa: E402
from offline_learning.human_replay import GAMES, _grid, _obs_cell  # noqa: E402
from offline_learning.invdyn_core import make_choices  # noqa: E402
from offline_learning.validate import (  # noqa: E402
    backfill_context_from_source,
    load_transitions,
)

csv.field_size_limit(10_000_000)

CONTEXT_K = 9
K_CHOICES = 5
INFER_SEEDS = list(range(16))   # the synthetic datasets do not record their seed, and
                                # the directory name can disagree with it: the pool in
                                # clean_data3/s2kt7_seed5 actually replays under seed 9
OUT_ROOT = _BAI_ROOT / "logs/aug10_human_origin/protocols"


# --------------------------------------------------------------------- pool specs
def human_spec(game: str) -> dict:
    """The human pool: test slices + the full drives they were sliced from."""
    root = _BAI_ROOT / "offline_learning/human_data" / game / "informative"
    paths = json.loads((root / "dataset_paths.json").read_text())
    man = json.loads((root / "MANIFEST.json").read_text())
    seeds = {f"test_d{i}": d["seed"] for i, d in enumerate(man["drives"]["test"])}
    return {"test_dirs": paths["test_run"].split(","),
            "source_dirs": paths["test_context_source_run"].split(","),
            "actions": paths["actions"].split(","),
            "seed_by_source_name": seeds}


def unified_spec(game: str) -> dict:
    """The unified pool: informative_unified test slices + the full drives they came
    from (flat 60/50, --drive-rank nonnoop; both learners trained on this same data)."""
    root = _BAI_ROOT / "offline_learning/human_data" / game / "informative_unified"
    paths = json.loads((root / "dataset_paths.json").read_text())
    man = json.loads((root / "MANIFEST.json").read_text())
    seeds = {f"test_d{i}": d["seed"] for i, d in enumerate(man["drives"]["test"])}
    return {"test_dirs": paths["test_run"].split(","),
            "source_dirs": paths["test_context_source_run"].split(","),
            "actions": paths["actions"].split(","),
            "seed_by_source_name": seeds}


def synthetic_spec(game: str) -> dict:
    """The artificial pool the reference REx-pure runs were scored on. s2kt7 has no
    launch.json (its flags were reconstructed); its test50 episodes each start at
    Step 0, so every episode is its own drive and needs no separate source."""
    p = _BAI_ROOT / f"logs/batch3_consolidated/{game}_s1_batch3/launch.json"
    if not p.exists():
        d = _BAI_ROOT / "offline_learning/clean_data3/s2kt7_seed5/test50"
        return {"test_dirs": [str(d)], "source_dirs": [str(d)],
                "actions": ["noop", "click"], "seed_by_source_name": {}}
    cmd = json.loads(p.read_text())["cmd"]
    f = {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1) if cmd[i].startswith("--")}
    test = f["--test-run"].split(",")
    src = f.get("--test-context-source-run", f["--test-run"]).split(",")
    return {"test_dirs": test, "source_dirs": src,
            "actions": f["--actions"].split(","), "seed_by_source_name": {}}


def coverage_spec(game: str) -> dict:
    """The coverage exam (coverage_exam.py): one scored slice per core-mechanic bucket
    (human-selected + engine-synthesized), each in its own single-episode test dir so the
    item order is deterministic. `source_seeds` is carried through because the human
    drives replay under large study seeds that _infer_seed's 0..15 probe would miss."""
    root = _BAI_ROOT / "offline_learning/human_data" / game / "coverage"
    paths = json.loads((root / "dataset_paths.json").read_text())
    man = json.loads((root / "MANIFEST.json").read_text())
    return {"test_dirs": paths["test_run"].split(","),
            "source_dirs": paths["test_context_source_run"].split(","),
            "actions": paths["actions"].split(","),
            "seed_by_source_name": man.get("source_seeds", {})}


SPECS = {"human": human_spec, "synthetic": synthetic_spec, "unified": unified_spec,
         "coverage": coverage_spec}


# ------------------------------------------------------------------ engine replay
def _episodes(dirs: list[str]) -> list[tuple[Path, list[dict]]]:
    out = []
    for d in dirs:
        for p in sorted(Path(d).glob("episode_*/trajectory.csv")):
            out.append((p, list(csv.DictReader(p.open()))))
    return out


def _infer_seed(prog: str, rows: list[dict], known: int | None) -> int | None:
    """Seed under which replaying the episode's actions reproduces its own frames."""
    acts = [r["Action"] for r in rows if r["Action"]]
    want = [_grid(r["Observation"]) for r in rows]
    probe = min(len(acts), len(want) - 1, 10)
    if probe <= 0:
        return known
    for seed in ([known] if known is not None else []) + INFER_SEEDS:
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


def _step_matches(prog: str, seed: int, acts: list[str], i: int,
                  action: str, want_next: str) -> bool | None:
    """Replay this drive's first i actions, apply `action`, compare to want_next.
    None if the episode terminated before step i (the state is unreachable).

    `action` must be in the ENV-facing (row-major) form the CSV stores. Protocol
    labels are already row-major, so no swap is needed."""
    env = AutumnBenchEnvWrapper(env_name=prog, task_type="interactive",
                                max_episode_steps=i + 8, seed=seed, render_mode="text")
    obs, _ = env.reset(seed=seed)
    try:
        for pa in acts[:i]:
            obs, _r, term, _t, _in = env.step(pa)
            if term:
                return None
        obs, _r, _t, _tr, _in = env.step(action)
        return _grid(_obs_cell(obs)) == want_next
    finally:
        env.close()


def _engine_consistent(prog: str, seed: int, acts: list[str], i: int,
                       choices: list[str], truth_next: str) -> list[str] | None:
    """Which of `choices`, applied at step i of this drive, reproduce truth_next."""
    out = []
    for c in choices:
        ok = _step_matches(prog, seed, acts, i, c, truth_next)
        if ok is None:
            return None
        if ok:
            out.append(c)
    return out


# ---------------------------------------------------------------------- build
def build(game: str, pool: str, seed: int, out_root: Path, verify: bool = True) -> dict:
    prog, _human, _wl = GAMES[game]
    spec = SPECS[pool](game)
    whitelist = set(spec["actions"])

    # transitions, in the exact order the learners' loaders produce them
    trs = []
    test_dirs, src_dirs = spec["test_dirs"], spec["source_dirs"]
    if len(src_dirs) == len(test_dirs):
        for td, sd in zip(test_dirs, src_dirs):
            tt = load_transitions([Path(td)], whitelist, context_k=CONTEXT_K)
            if Path(sd).resolve() != Path(td).resolve():
                try:
                    backfill_context_from_source(tt, [Path(sd)], whitelist, CONTEXT_K)
                except ValueError as e:
                    # a recurring transition (e.g. a static noop) can appear in the source
                    # drive with ambiguous context; the coverage exam includes such frames,
                    # so fall back to the slice-local context rather than crash. Other specs
                    # stay strict.
                    if pool != "coverage":
                        raise
                    print(f"[{game}/coverage] backfill skipped for {Path(td).name}: {e}",
                          flush=True)
            trs.extend(tt)
    else:
        trs = load_transitions([Path(p) for p in test_dirs], whitelist, CONTEXT_K)

    action_pool = sorted({t.action for t in trs})
    rng = random.Random(seed)
    items = [{"i": n, "truth": t.action,
              "choices": make_choices(t.action, action_pool, K_CHOICES, rng),
              "x_t_sha": hashlib.sha1(t.x_t.encode()).hexdigest()[:16],
              "x_t1_sha": hashlib.sha1(t.x_t1.encode()).hexdigest()[:16]}
             for n, t in enumerate(trs)]

    # attach the mechanic bucket to each coverage item. mechanics.json is written in the
    # same order the loader yields items (one scored slice per test dir); the target-frame
    # grid sha is cross-checked so a drift in ordering is caught rather than mislabelled.
    if pool == "coverage":
        labels = json.loads((_BAI_ROOT / "offline_learning/human_data" / game
                             / "coverage" / "mechanics.json").read_text())
        if len(labels) != len(items):
            raise SystemExit(f"coverage: {len(labels)} labels vs {len(items)} items")
        for it, t, lab in zip(items, trs, labels):
            gsha = hashlib.sha1((_grid(t.x_t1) or "").encode()).hexdigest()[:16]
            if gsha != lab["x_t1_sha"]:
                raise SystemExit(f"coverage: label/item mismatch at item {it['i']} "
                                 f"({gsha} != {lab['x_t1_sha']})")
            it["mechanic"] = lab["mechanic"]
            it["synthetic"] = lab["synthetic"]

    # locate each transition in a replayable drive, then ask the engine
    n_ok = n_alias = 0
    if verify:
        eps = _episodes(src_dirs if src_dirs != test_dirs else test_dirs)
        seeds: dict[Path, int | None] = {}
        by_obs: dict[str, list[tuple[Path, list[dict], int]]] = {}
        for p, rows in eps:
            for idx, r in enumerate(rows):
                by_obs.setdefault(r["Observation"], []).append((p, rows, idx))
        for it, tr in zip(items, trs):
            # A frame can repeat within a drive (an all-black grid before the first
            # click, a static stretch), so (x_t, x_t1) alone does not pin the step.
            # Require the recorded action to match too, then CONFIRM by replaying the
            # truth: only a step whose replay reproduces the recorded next frame is a
            # sound place to ask the counterfactual questions from.
            cands = [c for c in by_obs.get(tr.x_t, [])
                     if c[2] + 1 < len(c[1])
                     and c[1][c[2] + 1]["Observation"] == tr.x_t1
                     and (c[1][c[2]].get("Action") or "").strip() == tr.action]
            it["s_true"] = None
            for p, rows, idx in cands:
                if p not in seeds:
                    known = spec["seed_by_source_name"].get(p.parent.parent.name)
                    seeds[p] = _infer_seed(prog, rows, known)
                sd = seeds[p]
                if sd is None:
                    continue
                acts = [r["Action"] for r in rows if r["Action"]]
                nxt = _grid(rows[idx + 1]["Observation"])
                if _step_matches(prog, sd, acts, idx, rows[idx]["Action"], nxt) is not True:
                    continue                      # wrong occurrence, or non-reproducible
                s = _engine_consistent(prog, sd, acts, idx, it["choices"], nxt)
                if s is None or tr.action not in s:
                    continue
                it["s_true"] = s
                it["src"] = {"episode": str(p.relative_to(_BAI_ROOT)),
                             "step": idx, "seed": sd}
                n_ok += 1
                n_alias += (len(s) > 1)
                break

    fp = hashlib.sha1(json.dumps(
        [[t.x_t, t.action, t.x_t1] for t in trs], sort_keys=True).encode()).hexdigest()[:16]
    proto = {
        "game": game, "pool": pool, "program": prog, "k": K_CHOICES,
        "collapse": False, "context_k": CONTEXT_K, "choice_seed": seed,
        "actions": spec["actions"], "action_pool": action_pool,
        "test_dirs": test_dirs, "source_dirs": src_dirs,
        "fingerprint": fp, "n": len(items),
        "verified": n_ok, "aliased": n_alias,
        "ceiling": _ceiling(items), "items": items,
    }
    out = out_root / f"{game}_{pool}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(proto, indent=1) + "\n")
    print(f"[{game}/{pool}] n={len(items)} verified={n_ok} aliased={n_alias} "
          f"ceiling={proto['ceiling']} pool={len(action_pool)} -> {out}", flush=True)
    return proto


def score_set(truth: str, pred: list[str] | None) -> float:
    """The one scoring rule: 1/|pred| when the truth is in it, else 0."""
    if not pred or truth not in pred:
        return 0.0
    return 1.0 / len(pred)


def _ceiling(items: list[dict]) -> float | None:
    v = [score_set(it["truth"], it["s_true"]) for it in items if it.get("s_true")]
    return round(sum(v) / len(v), 4) if v else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", choices=sorted(SPECS), required=True)
    ap.add_argument("--game", choices=sorted(GAMES))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    games = sorted(GAMES) if args.all else [args.game]
    for g in games:
        build(g, args.spec, args.seed, Path(args.out), verify=not args.no_verify)


if __name__ == "__main__":
    main()
