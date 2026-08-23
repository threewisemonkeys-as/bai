"""Build clean_data3-shaped training data out of the BASIS human gameplay logs.

The human study (`basis_data.zip` -> `Human and AI baselines data/
cleaned_gameplay_data_filtered.json`, 517 users / 2580 sessions / 43 games) records
ONLY the action stream -- `{"actionType": "noop"|"up"|...}`, `{"actionType":"click",
"x":C,"y":R}`, `{"actionType":"reset","seed":"183702"}`. There are no grids anywhere,
so every observation has to be RECONSTRUCTED by replaying the action stream through
the Autumn interpreter (`AutumnBenchEnvWrapper`, exactly as `autumn_drive.py` does).

Human game name -> repo program (established by content hash, GRID_SIZE vs the clicks'
max (x,y), object names and the input-handler fingerprint):

    ice=BT3GB  disease=DQ8GC  ants=S2KT7  mario=N2NTD  particles=83WKQ

Output layout per game (mirrors what rexpure/worldcoder already consume):

    <out>/<game>/<variant>/
        drives/train_d<k>/episode_0/trajectory.csv    full replayed session (context source)
        drives/test_d<k>/episode_0/trajectory.csv
        train_d<k>/episode_*/trajectory.csv           2-row slices == the scored targets
        test_d<k>/episode_*/trajectory.csv
        MANIFEST.json                                 provenance + pool stats
        dataset_paths.json                            ready-made --run/--test-run strings

Slices are 2 rows == exactly ONE scored target each, copied VERBATIM from the drive, so
the pool size and action balance are controlled exactly (`clean_data3_METHODOLOGY.md`:
"A 2-row slice yields one"). Temporal context (`context_k=9`) is restored by
`backfill_context_from_source` from the matching full drive, which is why every emitted
target is checked to occur exactly once in its drive.

Two selection variants:

  informative  the analogue of clean_data3's hand curation, done mechanically. A target
               is kept only if the action is DISTINGUISHABLE: for a non-noop action the
               resulting frame must differ from the counterfactual frame the same state
               would have reached under `noop` (computed by re-replaying the prefix --
               replay is deterministic); for `noop` the frame must actually change, so
               passive dynamics are visible. Then round-robin over action verbs, and
               spread click locations.
  raw          uniform random sample over every in-whitelist transition -- human data as
               logged, ~80-90% of it noop. The control arm.

Usage:
    uv run python offline_learning/human_replay.py --game bt3gb --out offline_learning/human_data
    uv run python offline_learning/human_replay.py --all --out offline_learning/human_data
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

_BAI_ROOT = Path(__file__).resolve().parents[1]
if str(_BAI_ROOT) not in sys.path:
    sys.path.insert(0, str(_BAI_ROOT))

from autumn_env import AutumnBenchEnvWrapper  # noqa: E402

DEFAULT_ZIP = _BAI_ROOT / "basis_data.zip"
ZIP_MEMBER = "Human and AI baselines data/cleaned_gameplay_data_filtered.json"

MOVES = ("left", "right", "up", "down", "noop", "click")

# game -> (autumn program, human study name, action whitelist used by the learners)
GAMES = {
    "bt3gb": ("BT3GB", "ice", ["left", "right", "up", "down", "noop", "click"]),
    "dq8gc": ("DQ8GC", "disease", ["left", "right", "up", "down", "noop", "click"]),
    "n2ntd": ("N2NTD", "mario", ["left", "right", "up", "down", "noop", "click"]),
    "s2kt7": ("S2KT7", "ants", ["noop", "click"]),
    "83wkq": ("83WKQ", "particles", ["noop", "click"]),
}

CSV_FIELDS = ["Step", "Action", "Reasoning", "Observation",
              "Auxiliary_Observation", "Reward", "Done"]

MIN_SEG = 40        # a segment must have at least this many in-whitelist core actions
MAX_DRIVE = 240     # cap a drive (keeps the O(n^2) counterfactual probe cheap)
N_TRAIN_DRIVES = 3  # mirrors the reference runs' 3 train run-dirs
N_TEST_DRIVES = 3

# Scored-pool sizes of the ARTIFICIAL reference runs, measured with load_transitions on
# their own launch.json paths. The human pools are built to the same size so the only
# variable between the human and artificial arms is where the transitions came from.
#   rexpure    logs/batch3_consolidated   (--train-n 30 then samples the pool)
#   worldcoder logs/wc_seed1_consolidated (no --train-n: pool size sets train AND
#              val, since val_n = min(30, n_total//3))
REF_TRAIN_POOL = {
    "rexpure":    {"83wkq": 30, "bt3gb": 62, "dq8gc": 30, "n2ntd": 61, "s2kt7": 104},
    "worldcoder": {"83wkq": 104, "bt3gb": 20, "dq8gc": 30, "n2ntd": 30, "s2kt7": 104},
}
REF_TEST_POOL = 50


# --------------------------------------------------------------------------- data
def load_sessions(game: str, data_zip: Path, cache_dir: Path) -> list[dict]:
    """Per-game session list, cached (the source JSON is 200MB and slow to parse)."""
    _, human_name, _ = GAMES[game]
    cache = cache_dir / f"{human_name}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    print(f"[data] extracting {human_name} sessions from {data_zip} ...", flush=True)
    with zipfile.ZipFile(data_zip) as z, z.open(ZIP_MEMBER) as fh:
        blob = json.load(fh)
    out = []
    for u in blob["users"]:
        for tid, task in (u.get("tasks") or {}).items():
            if not tid.startswith(human_name + "_"):
                continue
            out.append({"user_id": u.get("user_id"), "task_id": tid,
                        "events": task.get("interactive_phase") or []})
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    print(f"[data] {len(out)} {human_name} sessions -> {cache}", flush=True)
    return out


def segment(session: dict, whitelist: set[str]) -> list[dict]:
    """Split one interactive phase into replayable segments, one per `reset`.

    A reset restarts the program with a NEW seed, so a segment is the largest run of
    actions that shares an initial state. Out-of-whitelist actions (inputs the program
    has no handler for, e.g. arrows in a click-only game) are dropped: they are exactly
    the actions `load_transitions` would drop anyway.
    """
    segs, cur = [], None
    for e in session["events"]:
        at = e.get("actionType")
        if at == "reset":
            if cur and len(cur["actions"]) >= MIN_SEG:
                segs.append(cur)
            cur = {"seed": int(e["seed"]), "actions": [],
                   "user_id": session["user_id"], "task_id": session["task_id"]}
        elif cur is not None and at in MOVES and at in whitelist:
            if at == "click":
                # log stores (x=col, y=row); the wrapper's interface is ROW-major and
                # transposes to the column-first interpreter (autumn_env ~line 469).
                cur["actions"].append(f"click {e['y']} {e['x']}")
            else:
                cur["actions"].append(at)
    if cur and len(cur["actions"]) >= MIN_SEG:
        segs.append(cur)
    for i, s in enumerate(segs):
        s["seg_idx"] = i
        s["actions"] = s["actions"][:MAX_DRIVE]
    return segs


# ----------------------------------------------------------------------- replaying
def _obs_cell(obs: dict) -> str:
    st = obs["text"]["short_term_context"]
    lt = obs["text"]["long_term_context"]
    return f"{st}\n\n========== Start of Direct Observation ==========\n{lt}"


def _grid(cell: str):
    i = cell.find("[[")
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(cell)):
        if cell[j] == "[":
            depth += 1
        elif cell[j] == "]":
            depth -= 1
            if depth == 0:
                return cell[i:j + 1]
    return None


def replay(prog: str, seed: int, actions: list[str]) -> dict:
    """Drive the program through `actions`; return frames/grids/termination index."""
    env = AutumnBenchEnvWrapper(env_name=prog, task_type="interactive",
                                max_episode_steps=len(actions) + 8, seed=seed,
                                render_mode="text")
    obs, _ = env.reset(seed=seed)
    cells, grids, taken = [_obs_cell(obs)], [_grid(_obs_cell(obs))], []
    for a in actions:
        obs, _r, term, _tr, _i = env.step(a)
        cell = _obs_cell(obs)
        cells.append(cell)
        grids.append(_grid(cell))
        taken.append(a)
        if term:
            break
    env.close()
    return {"cells": cells, "grids": grids, "actions": taken}


def noop_counterfactual(prog: str, seed: int, actions: list[str], idx: list[int],
                        horizon: int = 1) -> dict:
    """The WAITED-INSTEAD branch for each i in idx: replay actions[:i], then `noop` IN
    PLACE OF actions[i], then the human's own later actions actions[i+1:i+horizon].
    Returns {i: [grid_after_noop, grid_after_next_action, ...]} (up to `horizon` frames).

    Replay is deterministic, so a prefix re-run reproduces the exact state the human was
    in at step i. O(sum(idx) + horizon*len(idx)) steps; only probed steps are replayed.

    horizon=1 keeps exactly the historical behaviour (one frame: the noop successor).
    Larger horizons let a LATENT effect express itself: an action that changes only
    hidden state (selection, mode, arming) renders identically at t+1, but the SAME
    subsequent actions then drive the two branches apart a few steps later. Continuing
    with the human's real later actions is what makes the comparison meaningful --
    substituting the noop and stopping would just re-read the t+1 frame.
    """
    horizon = max(1, int(horizon))
    out = {}
    for i in sorted(idx):
        plan = actions[:i] + ["noop"] + actions[i + 1:i + horizon]
        env = AutumnBenchEnvWrapper(env_name=prog, task_type="interactive",
                                    max_episode_steps=len(plan) + 8, seed=seed,
                                    render_mode="text")
        obs, _ = env.reset(seed=seed)
        frames, dead = [], False
        for k, a in enumerate(plan):
            obs, _r, term, _tr, _inf = env.step(a)
            if k >= i:                       # only the branch itself, not the prefix
                frames.append(_grid(_obs_cell(obs)))
            if term:
                dead = k < i                 # terminating inside the prefix = unusable
                break
        if not dead and frames:
            out[i] = frames
        env.close()
    return out


def drive_rows(rep: dict) -> list[dict]:
    """autumn_drive-compatible rows: Action on row i leads to row i+1's Observation."""
    rows = []
    for i, a in enumerate(rep["actions"]):
        rows.append({"Step": i, "Action": a, "Reasoning": "human",
                     "Observation": rep["cells"][i], "Auxiliary_Observation": "",
                     "Reward": 0.0, "Done": False})
    rows.append({"Step": len(rep["actions"]), "Action": "", "Reasoning": "human",
                 "Observation": rep["cells"][len(rep["actions"])],
                 "Auxiliary_Observation": "", "Reward": 0.0, "Done": True})
    return rows


def write_episode(rows: list[dict], ep_dir: Path) -> None:
    ep_dir.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(10_000_000)
    with (ep_dir / "trajectory.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


# ----------------------------------------------------------------------- selection
def candidates(rep: dict, cf: dict) -> list[dict]:
    """Per-step target descriptors with the observability verdicts.

    A non-noop step is INFORMATIVE iff the real trajectory diverges from the
    waited-instead branch (see noop_counterfactual) anywhere inside the branch -- i.e.
    the action did something waiting would not have, within the horizon the branch was
    built to. With a 1-frame branch this is the historical t+1 test; with an H-frame
    branch it additionally keeps actions whose ONLY effect is on hidden state, which is
    invisible at t+1 but expressed a few steps later. Widening can only ADMIT steps:
    divergence at t+1 implies divergence within H.
    """
    out = []
    grids, acts = rep["grids"], rep["actions"]
    for i, a in enumerate(acts):
        if i + 1 >= len(grids):
            break
        verb = a.split()[0]
        changed = grids[i + 1] != grids[i]
        if verb == "noop":
            # the counterfactual for `noop` IS the action, so the only question that
            # makes sense is whether the world moved at all.
            informative = changed
        else:
            branch = cf.get(i) or []
            informative = any(
                grids[i + 1 + d] != branch[d]
                for d in range(min(len(branch), len(grids) - i - 1))
            )
        out.append({"i": i, "action": a, "verb": verb,
                    "changed": changed, "informative": bool(informative)})
    return out


def pick_targets(pools: list[tuple[int, list[dict]]], n_want: int, mode: str,
                 rng: random.Random) -> list[tuple[int, int]]:
    """Choose (drive_index, step_index) targets, round-robin over action verbs.

    Verb round-robin is what keeps a 86%-noop corpus from collapsing the pool onto noop
    (and, on click games, keeps a few hundred distinct `click R C` labels from crowding
    everything else out). Click picks are additionally spread over distinct locations
    before any location is reused.
    """
    if mode == "raw":
        # human data exactly as logged: uniform sample, so the pool keeps the natural
        # ~80-90% noop composition (no filtering, no balancing).
        allc = [(di, c["i"]) for di, cands in pools for c in cands]
        rng.shuffle(allc)
        return allc[:n_want]

    by_verb: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for di, cands in pools:
        for c in cands:
            if mode == "informative" and not c["informative"]:
                continue
            by_verb[c["verb"]].append((di, c["i"], c["action"]))
    for v in by_verb:
        rng.shuffle(by_verb[v])
    # clicks first by unseen location, so the label set is spread not clustered
    if "click" in by_verb:
        seen, first, rest = set(), [], []
        for item in by_verb["click"]:
            (first if item[2] not in seen else rest).append(item)
            seen.add(item[2])
        by_verb["click"] = first + rest

    verbs = sorted(by_verb)
    picked, vi = [], 0
    while len(picked) < n_want and any(by_verb[v] for v in verbs):
        v = verbs[vi % len(verbs)]
        if by_verb[v]:
            di, i, _a = by_verb[v].pop(0)
            picked.append((di, i))
        vi += 1
    return picked


# ---------------------------------------------------------------------- generation
def build_variant(game: str, name: str, mode: str, drives: dict, out_root: Path,
                  n_train: int, n_test: int, rng: random.Random,
                  informative_horizon: int = 1) -> dict:
    prog, human_name, whitelist = GAMES[game]
    root = out_root / game / name
    if root.exists():
        shutil.rmtree(root)

    stats = {}
    paths = {}
    for split, want in (("train", n_train), ("test", n_test)):
        pools = [(k, drives[split][k]["cands"]) for k in range(len(drives[split]))]
        picks = pick_targets(pools, want, mode, rng)
        by_drive: dict[int, list[int]] = defaultdict(list)
        for di, i in picks:
            by_drive[di].append(i)

        run_dirs, src_dirs = [], []
        balance = Counter()
        n_targets = 0
        for di in sorted(by_drive):
            drv = drives[split][di]
            rows = drv["rows"]
            slice_root = root / f"{split}_d{di}"
            for n, i in enumerate(sorted(by_drive[di])):
                # 2-row slice == exactly one scored target, rows copied verbatim
                write_episode([dict(rows[i]), dict(rows[i + 1])],
                              slice_root / f"episode_{n}")
                balance[rows[i]["Action"]] += 1
                n_targets += 1
            src = root / "drives" / f"{split}_d{di}"
            write_episode(rows, src / "episode_0")
            run_dirs.append(str(slice_root.resolve()))
            src_dirs.append(str(src.resolve()))

        paths[split] = {"run": run_dirs, "source": src_dirs}
        stats[split] = {"n_targets": n_targets,
                        "n_drives": len(by_drive),
                        "balance": dict(balance.most_common()),
                        "verbs": dict(Counter(a.split()[0] for a in balance.elements()))}

    manifest = {
        "game": game, "program": prog, "human_game": human_name,
        "variant": name, "selection": mode, "whitelist": whitelist,
        "informative_horizon": informative_horizon,
        "drives": {s: [{k: d[k] for k in ("user_id", "task_id", "seed", "seg_idx",
                                          "n_steps", "n_informative", "n_changed")}
                       for d in drives[s]] for s in ("train", "test")},
        "stats": stats,
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "dataset_paths.json").write_text(json.dumps({
        "run": ",".join(paths["train"]["run"]),
        "context_source_run": ",".join(paths["train"]["source"]),
        "test_run": ",".join(paths["test"]["run"]),
        "test_context_source_run": ",".join(paths["test"]["source"]),
        "actions": ",".join(whitelist),
    }, indent=2) + "\n")
    return manifest


def _nonnoop_count(s: dict) -> int:
    """Non-noop actions in a segment, straight from the action list (no replay)."""
    return sum(1 for a in s["actions"] if a.split()[0] != "noop")


def rank_drives(prog: str, segs: list[dict], mode: str, t0: float, game: str) -> list[dict]:
    """Order segments best-first for the train/test split.

    `score`   the original proxy: replay the 24 longest and rank by
              nonnoop + 6*distinct_nonnoop_verbs + 0.2*n_changed (needs a replay to rank).
    `nonnoop` rank every segment by how much the human actually did -- computed from the
              action list with no replay; the selected drives are replayed afterwards.
    `length`  rank by raw in-whitelist action count (simplest; can pick an idle drive).
    """
    if mode == "score":
        segs = sorted(segs, key=lambda s: -len(s["actions"]))
        probe_n = min(len(segs), 24)
        ranked = []
        for s in segs[:probe_n]:
            rep = replay(prog, s["seed"], s["actions"])
            acts, grids = rep["actions"], rep["grids"]
            n_changed = sum(grids[i + 1] != grids[i] for i in range(len(acts)))
            verbs = Counter(a.split()[0] for a in acts)
            nonnoop = sum(v for k, v in verbs.items() if k != "noop")
            ranked.append({**s, "rep": rep, "n_steps": len(acts), "n_changed": n_changed,
                           "score": nonnoop + 6 * len(set(verbs) - {"noop"}) + 0.2 * n_changed})
        ranked.sort(key=lambda d: -d["score"])
        print(f"[{game}] replayed {probe_n} segments in {time.time() - t0:.0f}s", flush=True)
        return ranked
    key = _nonnoop_count if mode == "nonnoop" else (lambda s: len(s["actions"]))
    return sorted(segs, key=lambda s: -key(s))


def generate(game: str, out_root: Path, data_zip: Path, seed: int,
             n_train: int, n_test: int, variants: list[str],
             drive_rank: str = "score", informative_horizon: int = 1) -> None:
    prog, human_name, whitelist = GAMES[game]
    wl = set(whitelist)
    rng = random.Random(seed)
    t0 = time.time()

    sessions = load_sessions(game, data_zip, out_root / "_cache")
    segs = [s for sess in sessions for s in segment(sess, wl)]
    print(f"[{game}] {len(sessions)} sessions -> {len(segs)} replayable segments "
          f"(>={MIN_SEG} in-whitelist actions)", flush=True)

    ranked = rank_drives(prog, segs, drive_rank, t0, game)

    # train/test drives from DISJOINT users (no cross-split leakage)
    train, test, used_users = [], [], set()
    for d in ranked:
        if len(train) < N_TRAIN_DRIVES:
            train.append(d)
            used_users.add(d["user_id"])
    for d in ranked:
        if len(test) < N_TEST_DRIVES and d["user_id"] not in used_users:
            test.append(d)
    if len(test) < N_TEST_DRIVES:
        raise SystemExit(f"[{game}] not enough distinct-user segments for a clean test split")

    # replay-free ranks (nonnoop/length) haven't replayed the chosen drives yet
    for d in train + test:
        if "rep" not in d:
            rep = replay(prog, d["seed"], d["actions"])
            d["rep"] = rep
            d["n_steps"] = len(rep["actions"])
            d["n_changed"] = sum(rep["grids"][i + 1] != rep["grids"][i]
                                 for i in range(len(rep["actions"])))

    for split, ds in (("train", train), ("test", test)):
        for d in ds:
            idx = [i for i, a in enumerate(d["rep"]["actions"]) if a.split()[0] != "noop"]
            t1 = time.time()
            cf = noop_counterfactual(prog, d["seed"], d["rep"]["actions"], idx,
                                     horizon=informative_horizon)
            d["cands"] = candidates(d["rep"], cf)
            d["rows"] = drive_rows(d["rep"])
            d["n_informative"] = sum(c["informative"] for c in d["cands"])
            print(f"[{game}] {split} drive u={d['user_id'][:8]} seed={d['seed']} "
                  f"steps={d['n_steps']} changed={d['n_changed']} "
                  f"informative={d['n_informative']} ({time.time() - t1:.0f}s probe)",
                  flush=True)

    drives = {"train": train, "test": test}
    for spec in variants:
        name, _, mode = spec.partition(":")
        mode = mode or name
        m = build_variant(game, name, mode, drives, out_root, n_train, n_test, rng,
                          informative_horizon=informative_horizon)
        print(f"[{game}/{name}] train pool={m['stats']['train']['n_targets']} "
              f"{m['stats']['train']['verbs']} | test pool={m['stats']['test']['n_targets']} "
              f"{m['stats']['test']['verbs']}", flush=True)
    print(f"[{game}] done in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", choices=sorted(GAMES))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=str(_BAI_ROOT / "offline_learning/human_data"))
    ap.add_argument("--data-zip", default=str(DEFAULT_ZIP))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--profile", choices=sorted(REF_TRAIN_POOL),
                    help="size the train pool to that learner's artificial reference run")
    ap.add_argument("--train-pool", type=int,
                    help="explicit train-pool size (overrides --profile)")
    ap.add_argument("--test-pool", type=int, default=REF_TEST_POOL,
                    help="scored target transitions in the test pool (== --test-n)")
    ap.add_argument("--variants", default="informative,raw",
                    help="comma-separated; 'name:mode' to name a variant dir separately "
                         "from its selection mode (informative|raw)")
    ap.add_argument("--informative-horizon", type=int, default=1,
                    help="steps of the waited-instead branch the informativeness test may "
                         "look at (default 1 = the historical t+1 test). Set to the "
                         "learner's forward context width (8 for context_k=9) to keep "
                         "actions whose only effect is on hidden state -- selection, mode "
                         "toggles, arming -- which render identically at t+1 and are "
                         "otherwise dropped wholesale. Widening only ever ADMITS steps")
    ap.add_argument("--drive-rank", choices=("score", "nonnoop", "length"), default="score",
                    help="how to pick the 3 train / 3 test drives: score=original "
                         "activity+variety proxy (replays 24 to rank); nonnoop=rank by "
                         "non-noop action count (no replay); length=raw action count")
    args = ap.parse_args()
    if not args.game and not args.all:
        ap.error("pass --game or --all")
    if args.train_pool is None and not args.profile:
        ap.error("pass --train-pool or --profile")
    games = sorted(GAMES) if args.all else [args.game]
    for g in games:
        pool = args.train_pool or REF_TRAIN_POOL[args.profile][g]
        generate(g, Path(args.out), Path(args.data_zip), args.seed,
                 pool, args.test_pool, args.variants.split(","), args.drive_rank,
                 informative_horizon=args.informative_horizon)


if __name__ == "__main__":
    main()
