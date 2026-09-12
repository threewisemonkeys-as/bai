#!/usr/bin/env python3
"""How the abstraction program P changes over a learning run, measured node by node.

The paper's NLWM columns plan with one perception module per game -- the node the search
shipped. This reads the search that produced it and measures *every* node of the pool, so
the shipped module can be seen as the endpoint of a trajectory rather than as a given.

Scope is the 15 learning runs behind the paper's NLWM (Plain) column, and only those (see
`notes/perception-metrics-plan.md` for the chain from column to run dir):

    Plain  logs/2026-08-24/human_curated/rexpure/<game>_s1         reflector deepseek-v4-flash
           -> logs/2026-09-03/planning_v2_online_ds_percap_nl  (NLWM (Plain))
           -> logs/2026-09-08/agent_wm_full                    (NLWM (Agentic), same artifacts)

The Opus-5 reflector runs under `human_curated_opus5` (the paper's NLWM (SL) appendix
column) are deliberately OUT of scope.

    uv run python offline_learning/scripts/perception_metrics.py
    uv run python offline_learning/scripts/perception_metrics.py --games dq8gc,bt3gb
    uv run python offline_learning/scripts/perception_metrics.py --check      # gates only

Writes `analysis/perception_metrics/`: `metrics.csv` (one row per node), `manifest.json`
(what was read, with hashes and fingerprints), and a gzipped frame cache under `cache/`.

Nothing here calls a model or the simulator. The corpus X is the run's OWN train/test
split, rebuilt byte-identically by replaying `launch.json` through the optimiser's own
parser (`rexpure_optimize.build_parser` + `build_data`) and asserted against the
`train_fingerprint` the run checkpointed. Both arms were launched on the same data paths
and seed, so the corpus is built once per game and the second arm's fingerprint is checked
against it rather than rebuilt.

The process is re-exec'd under `PYTHONHASHSEED=0`. A few of the learned modules build
their feature string by iterating a set, so their bytes -- though not their length, and not
which observations they tell apart -- depend on the interpreter's hash seed. The learner had
the same exposure; pinning it here only makes the measurement reproducible, and a rerun is
expected to reproduce `metrics.csv` byte for byte.

P is run through `validate.run_perceive`, the same entry point the learner scores through:
a fresh `exec` per frame and a ONE-element observation history. Module-level globals
therefore reset per frame and a `len(observation_history)` step counter is constant --
which is a real limitation, and the right one, because it is exactly how
`invdyn_core.build_window` feeds P when computing the score that drives the search.

The metrics, and what each is for:

  size          `ast_nodes` (headline -- immune to comments and formatting, which the
                reflector varies freely), plus chars / lines / sloc as a sanity check.
  compression   `set_ratio` = |{P(X)}|/|{X}|, the quantity as originally posed. It is a
                COLLAPSE INDICATOR, not a learning curve: raw frames are essentially all
                distinct and any P that lists non-background cells is injective, so a
                working node scores 1.000 and a broken one scores 1/|X| with nothing in
                between (measured on dq8gc and bt3gb; see the plan note). `decoy_collapse`
                asks the same question restricted to each instance's hard contrastive-FD
                negatives. What IS graded is description length: three named
                scores. `diversity_bytes` = gz([P(X) for all X]), the compressed size of
                everything P emits over the corpus -- output diversity, in bytes.
                `norm_diversity` = that over gz([X for all X]), the same against what the
                observations themselves cost. `info_extraction_ratio` = mean over frames of
                gz(P(X))/gz(X), which prices each observation on its own; the two corpus
                scores flatter P, whose outputs repeat frame to frame and so dedupe harder
                than the grids do, and this is the version without that discount. Kept
                alongside: `dl_ratio` and `pf_dl_gz_ratio` (uncompressed feature bytes over
                raw and over compressed observation bytes) and `lzma_ratio` /
                `twopart_ratio` as checks.
  change        `static_rate`, the fraction of scored transitions P renders identically at
                t and t+1 -- P blind to the step it is scored on; and `change_ratio`, the
                feature-token churn per grid cell that actually moved.
  status        ok / syntax / runtime / collapsed. Dead nodes are part of the story (4 of
                30 on bt3gb), so they are counted rather than dropped.
"""
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import io
import json
import lzma
import os
import random
import signal
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO, REPO / "offline_learning", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from offline_learning.validate import run_perceive  # noqa: E402

csv.field_size_limit(10**7)

# arm -> (training-run root, the evaluation runs whose paper column it feeds)
ARMS = {
    "Plain": ("logs/2026-08-24/human_curated/rexpure",
              ["logs/2026-09-03/planning_v2_online_ds_percap_nl",
               "logs/2026-09-08/agent_wm_full"]),
}
CORPUS_ARM = "Plain"          # the arm that builds each game's frame corpus
OUT = REPO / "analysis/perception_metrics"
CACHE = OUT / "cache"
SEED_PERCEPTION = REPO / "offline_learning/autumn_seed_perception.py"
PERCEIVE_TIMEOUT_S = 10

FIELDS = [
    "arm", "game", "idx", "iteration", "parents", "depth", "train_score", "is_ship",
    "on_ship_lineage", "status", "err_rate", "split",
    "code_chars", "code_lines", "sloc", "ast_nodes",
    "n_frames", "n_unique_outputs", "set_ratio", "decoy_collapse",
    "mean_raw_chars", "mean_out_chars", "dl_ratio",
    "diversity_bytes", "raw_gzip_bytes", "norm_diversity", "info_extraction_ratio",
    "lzma_ratio", "twopart_ratio", "pf_dl_gz_ratio", "code_gzip_bytes", "out_vocab",
    "n_transitions", "static_rate", "change_ratio",
]


# --------------------------------------------------------------------- corpus
def run_argv(run_dir: Path) -> list[str]:
    """The flags the run was launched with, from its own launch.json."""
    cmd = json.loads((run_dir / "launch.json").read_text())["cmd"]
    i = next(i for i, x in enumerate(cmd) if str(x).endswith(".py"))
    return [str(x) for x in cmd[i + 1:]]


def _fingerprint(run_dir: Path, seed: int) -> str | None:
    p = run_dir / f"rexpure_run_seed{seed}" / "resume_state.json"
    return json.loads(p.read_text()).get("train_fingerprint") if p.is_file() else None


def build_corpus(game: str, force: bool = False) -> dict:
    """{frames, train_pairs, test_pairs, decoy_sets, fingerprint} for one game.

    Frames are interned into one list and everything else indexes into it, which keeps the
    cache small (a frame is ~2.3 KB and most appear in several roles) and makes P's output
    computable once per distinct frame rather than once per use.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{game}.json.gz"
    if cache.is_file() and not force:
        with gzip.open(cache, "rt") as fh:
            return json.load(fh)

    from invdyn_core import _train_fingerprint  # noqa: PLC0415
    from rexpure_optimize import build_data, build_parser  # noqa: PLC0415

    run_dir = REPO / ARMS[CORPUS_ARM][0] / f"{game}_s1"
    args = build_parser().parse_args(run_argv(run_dir))
    buf = io.StringIO()
    stdout, sys.stdout = sys.stdout, buf          # build_data narrates; not our output
    try:
        train, test, _pool, _ck, _wl, _trs, _idn = build_data(args, random.Random(args.seed))
    finally:
        sys.stdout = stdout

    want = _fingerprint(run_dir, args.seed)
    got = _train_fingerprint(train)
    if want and want != got:
        raise RuntimeError(
            f"{game}: rebuilt train fingerprint {got} != checkpointed {want} -- the data or "
            "the flags moved since the run, so this is a different corpus")

    frames: list[str] = []
    index: dict[str, int] = {}

    def fid(x: str) -> int:
        x = x or ""
        if x not in index:
            index[x] = len(frames)
            frames.append(x)
        return index[x]

    def pairs(batch):
        return [[fid(i["tr"].x_t), fid(i["tr"].x_t1)] for i in batch]

    corpus = {
        "game": game, "fingerprint": got, "seed": args.seed,
        "train_pairs": pairs(train), "test_pairs": pairs(test),
        # [true_next, decoy, decoy, ...] per train instance, decoys distinct from the truth
        "decoy_sets": [
            [fid(i["tr"].x_t1)] + [fid(o) for o in i.get("cfd_options", [])
                                   if (o or "").strip() != (i["tr"].x_t1 or "").strip()]
            for i in train
        ] if train and "cfd_options" in train[0] else [],
        "frames": frames,
    }
    with gzip.open(cache, "wt") as fh:
        json.dump(corpus, fh)
    return corpus


# -------------------------------------------------------------------- running P
class _Timeout(Exception):
    pass


def _alarm(_sig, _frm):
    raise _Timeout("perceive() exceeded the per-call timeout")


def perceive_all(code: str, frames: list[str]) -> tuple[list[str], int]:
    """P over every frame, with a per-call wall-clock guard. Returns (outputs, n_errors).

    A crash yields "" -- the same value the learner's scorer sees for a raising module --
    so an error is counted AND its output still participates in the set metrics.
    """
    outs, errs = [], 0
    prev = signal.signal(signal.SIGALRM, _alarm)
    try:
        for x in frames:
            signal.setitimer(signal.ITIMER_REAL, PERCEIVE_TIMEOUT_S)
            try:
                z, err = run_perceive(code, x)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            errs += bool(err)
            outs.append(z)
    finally:
        signal.signal(signal.SIGALRM, prev)
    return outs, errs


# --------------------------------------------------------------------- metrics
def _parse_grid(obs: str):
    """The colour grid as list[list[str]], or None. Same extraction the modules use."""
    if not obs:
        return None
    start, end = obs.find("[["), obs.rfind("]]")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        grid = json.loads(obs[start:end + 2])
    except Exception:  # noqa: BLE001
        return None
    return grid if grid and isinstance(grid[0], list) else None


def cell_diff(a: str, b: str) -> int | None:
    """Number of grid cells that differ between two raw observations."""
    ga, gb = _parse_grid(a), _parse_grid(b)
    if ga is None or gb is None or len(ga) != len(gb):
        return None
    n = 0
    for ra, rb in zip(ga, gb):
        if len(ra) != len(rb):
            return None
        n += sum(x != y for x, y in zip(ra, rb))
    return n


def gzip_len(parts: list[str]) -> int:
    # mtime=0: the header carries a timestamp otherwise, and this is compared across runs
    return len(gzip.compress("\n".join(parts).encode(), 6, mtime=0))


def lzma_len(parts: list[str]) -> int:
    """Same measure with a window big enough for the whole corpus. gzip sees 32 KB, and a
    game's frames are ~200 KB, so gzip prices only local cross-frame redundancy; lzma prices
    all of it. Measured on three games, the two rank nodes the same -- lzma is the check
    that says so, not a second opinion to argue with."""
    return len(lzma.compress("\n".join(parts).encode(), preset=6))


_RAW_COMPRESSED: dict = {}          # (game, split) -> (gzip bytes, lzma bytes) of raw X


def raw_compressed(game: str, split: str, raws: list[str]) -> tuple[int, int]:
    key = (game, split)
    if key not in _RAW_COMPRESSED:
        _RAW_COMPRESSED[key] = (gzip_len(raws), lzma_len(raws))
    return _RAW_COMPRESSED[key]


_RAW_PERFRAME: dict = {}            # (game, split) -> gzip bytes of each raw X on its own


def raw_perframe(game: str, split: str, raws: list[str]) -> list[int]:
    """gzip(X) frame by frame, with no cross-frame dictionary.

    `raw_compressed` gzips the whole corpus as one stream, which lets every frame after the
    first be coded against its predecessors. That discount is not symmetric between the two
    sides of the ratio -- P's outputs are near-identical frame to frame and dedupe ~7x, the
    raw grids ~3x -- so a corpus-level ratio partly rewards P for repeating itself. These
    per-frame lengths price each observation on its own, which is the comparison to make
    when the question is whether P compresses AN OBSERVATION rather than a whole corpus."""
    key = (game, split)
    if key not in _RAW_PERFRAME:
        _RAW_PERFRAME[key] = [gzip_len([r]) for r in raws]
    return _RAW_PERFRAME[key]


def code_shape(code: str) -> dict:
    lines = code.splitlines()
    try:
        nast = sum(1 for _ in ast.walk(ast.parse(code)))
        syntax_ok = True
    except SyntaxError:
        nast, syntax_ok = -1, False
    return {
        "code_chars": len(code), "code_lines": len(lines), "ast_nodes": nast,
        "sloc": sum(1 for ln in lines if ln.strip() and not ln.strip().startswith("#")),
        "_syntax_ok": syntax_ok,
    }


def node_metrics(code: str, corpus: dict, split: str) -> dict:
    """Every metric for one perception module against one split of one game's corpus."""
    frames = corpus["frames"]
    pairs = corpus[f"{split}_pairs"]
    used = sorted({i for p in pairs for i in p})
    if split == "train":
        used = sorted(set(used) | {i for s in corpus["decoy_sets"] for i in s})

    outs, errs = perceive_all(code, [frames[i] for i in used])
    P = dict(zip(used, outs))
    scored = sorted({i for p in pairs for i in p})          # frames X, decoys excluded
    zs = [P[i] for i in scored]
    raws = [frames[i] for i in scored]

    shape = code_shape(code)
    n_out = len(set(zs))
    status = ("syntax" if not shape.pop("_syntax_ok")
              else "runtime" if errs
              else "collapsed" if n_out <= 1
              else "ok")
    if status == "runtime" and n_out <= 1:
        status = "runtime"                                   # a crash reads as a crash

    decoy = None
    if split == "train" and corpus["decoy_sets"]:
        n = hit = 0
        for s in corpus["decoy_sets"]:
            if len(s) < 2:
                continue
            n += 1
            hit += any(P[d] == P[s[0]] for d in s[1:])
        decoy = hit / n if n else None

    static = sum(P[a] == P[b] for a, b in pairs) / len(pairs)
    churn = []
    for a, b in pairs:
        cells = cell_diff(frames[a], frames[b])
        if not cells:                                        # None, or a no-op transition
            continue
        ta, tb = set(P[a].split()), set(P[b].split())
        churn.append(len(ta ^ tb) / cells)

    mean_raw = sum(map(len, raws)) / len(raws)
    mean_out = sum(map(len, zs)) / len(zs)
    raw_gz, raw_xz = raw_compressed(corpus["game"], split, raws)
    rpf = raw_perframe(corpus["game"], split, raws)
    out_gz = gzip_len(zs)
    code_gz = gzip_len([code])
    return dict(
        shape, split=split, err_rate=errs / len(used), status=status,
        n_frames=len(scored), n_unique_outputs=n_out, set_ratio=n_out / len(scored),
        decoy_collapse=decoy, mean_raw_chars=mean_raw, mean_out_chars=mean_out,
        dl_ratio=mean_out / mean_raw if mean_raw else None,
        # OUTPUT DIVERSITY: gz([P(X) for all X]). How much irreducible content P's features
        # carry over the whole corpus -- an absolute byte count, so it grows both with what
        # P says per frame and with how much that varies frame to frame. A P that prints the
        # same line every time compresses to almost nothing however long the line is.
        diversity_bytes=out_gz, raw_gzip_bytes=raw_gz,
        # NORMALISED DIVERSITY: the same against the observations' own compressed size.
        norm_diversity=out_gz / raw_gz, lzma_ratio=lzma_len(zs) / raw_xz,
        # two-part code: the module plus the features it emits, against the data itself.
        # Above 1, describing X through P costs more than just compressing X.
        twopart_ratio=(code_gz + out_gz) / raw_gz,
        # per frame, no cross-frame dictionary on either side (see raw_perframe).
        # INFO EXTRACTION RATIO: mean of gz(P(X))/gz(X) -- the dedup-neutral twin of
        # norm_diversity, and the one that asks whether P compresses AN OBSERVATION.
        info_extraction_ratio=sum(gzip_len([z]) / g for z, g in zip(zs, rpf)) / len(zs),
        # pf_dl_gz: literal feature bytes against the observation's compressed size --
        # what P costs to write down, against what X actually contains.
        pf_dl_gz_ratio=sum(len(z) / g for z, g in zip(zs, rpf)) / len(zs),
        code_gzip_bytes=code_gz,
        out_vocab=len({t for z in zs for t in z.split()}),
        n_transitions=len(pairs), static_rate=static,
        change_ratio=sum(churn) / len(churn) if churn else None,
    )


# ------------------------------------------------------------------- run reading
def read_run(arm: str, game: str) -> dict:
    """Pool, iteration map and shipped-node join for one learning run."""
    d = REPO / ARMS[arm][0] / f"{game}_s1"
    rd = d / "rexpure_run_seed1"
    pool = [json.loads(l) for l in (rd / "candidates.jsonl").open()]
    iters = {r["new_idx"]: r["i"] for r in map(json.loads, (rd / "process_log.jsonl").open())
             if r.get("new_idx") is not None}
    best_p = (d / "best_perception_rexpure_seed1.py").read_text()
    best_k = (d / "best_beliefs_rexpure_seed1.txt").read_text()
    ship = [c["idx"] for c in pool
            if c["perception"].strip() == best_p.strip()
            and c["world_knowledge"].strip() == best_k.strip()]
    if len(ship) != 1:
        raise RuntimeError(f"{arm}/{game}: shipped (P,K) matches {len(ship)} pool nodes, want 1")
    top = max(pool, key=lambda c: c["train_score"])["idx"]
    if ship[0] != top:
        raise RuntimeError(f"{arm}/{game}: shipped node {ship[0]} is not argmax train_score {top}")

    by_idx = {c["idx"]: c for c in pool}
    lineage, cur = set(), ship[0]                   # walk the ship back to the root
    while cur is not None:
        lineage.add(cur)
        ps = by_idx[cur]["parents"]
        cur = ps[0] if ps else None
    depth = {}

    def d_of(i):
        if i not in depth:
            ps = by_idx[i]["parents"]
            depth[i] = 0 if not ps else 1 + d_of(ps[0])
        return depth[i]

    return {"dir": d, "pool": pool, "iters": iters, "ship": ship[0], "lineage": lineage,
            "depth": {c["idx"]: d_of(c["idx"]) for c in pool},
            "best_p": best_p, "best_p_sha": hashlib.sha256(best_p.encode()).hexdigest()}


def score_run(arm: str, game: str) -> tuple[list[dict], dict]:
    corpus = build_corpus(game)
    run = read_run(arm, game)
    want = _fingerprint(run["dir"], corpus["seed"])
    if want and want != corpus["fingerprint"]:
        raise RuntimeError(
            f"{arm}/{game}: this arm checkpointed fingerprint {want} but the shared corpus "
            f"is {corpus['fingerprint']} -- the arms did not train on the same split")

    rows = []
    for c in run["pool"]:
        base = dict(arm=arm, game=game, idx=c["idx"],
                    iteration=run["iters"].get(c["idx"]), parents=";".join(map(str, c["parents"])),
                    depth=run["depth"][c["idx"]], train_score=c["train_score"],
                    is_ship=int(c["idx"] == run["ship"]),
                    on_ship_lineage=int(c["idx"] in run["lineage"]))
        for split in ("train", "test"):
            rows.append({**base, **node_metrics(c["perception"], corpus, split)})
    return rows, {"run_dir": str(run["dir"].relative_to(REPO)), "ship_idx": run["ship"],
                  "best_perception_sha256": run["best_p_sha"],
                  "train_fingerprint": corpus["fingerprint"], "n_nodes": len(run["pool"])}


# ------------------------------------------------------------------------ gates
def gates(rows: list[dict], runs: dict) -> list[str]:
    """Assertions that would make a number in the figure wrong. Returns failures."""
    bad = []
    seed_src = SEED_PERCEPTION.read_text().strip()
    for (arm, game), info in runs.items():
        d = REPO / ARMS[arm][0] / f"{game}_s1"
        pool = [json.loads(l) for l in (d / "rexpure_run_seed1/candidates.jsonl").open()]
        if pool[0]["perception"].strip() != seed_src:
            bad.append(f"{arm}/{game}: node 0 is not autumn_seed_perception.py")
        # the shipped node's metrics must equal the artifact file's own, metric for metric
        corpus = build_corpus(game)
        art = node_metrics((d / "best_perception_rexpure_seed1.py").read_text(), corpus, "train")
        ship = next(r for r in rows if r["arm"] == arm and r["game"] == game
                    and r["is_ship"] and r["split"] == "train")
        for k, v in art.items():
            if ship[k] != v:
                bad.append(f"{arm}/{game}: ship node {k}={ship[k]} but the artifact file says {v}")

    # every artifact the agentic run copied must be the file we measured
    man = REPO / "logs/2026-09-08/agent_wm_full/nlwm_manifest.json"
    if man.is_file():
        for game, files in json.loads(man.read_text())["games"].items():
            info = runs.get(("Plain", game))
            want = files["perception.py"]["sha256"]
            if info and info["best_perception_sha256"] != want:
                bad.append(f"Plain/{game}: measured perception sha256 != agentic manifest's")
    return bad


# ------------------------------------------------------------------------- main
def main():
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"       # must be set before the interpreter starts
        os.execv(sys.executable, [sys.executable, *sys.argv])

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", default="", help="comma-separated subset (default: all 15)")
    ap.add_argument("--arms", default=",".join(ARMS), help="comma-separated subset of arms")
    ap.add_argument("--jobs", type=int, default=8, help="parallel (arm, game) workers")
    ap.add_argument("--rebuild-corpus", action="store_true", help="ignore the frame cache")
    ap.add_argument("--check", action="store_true", help="run the gates and stop")
    a = ap.parse_args()

    arms = [x for x in a.arms.split(",") if x.strip()]
    games = ([g.strip() for g in a.games.split(",") if g.strip()]
             or sorted(p.name[:-3] for p in (REPO / ARMS[arms[0]][0]).glob("*_s1")))
    OUT.mkdir(parents=True, exist_ok=True)

    for g in games:                      # serial, and the slow part of the whole run
        build_corpus(g, force=a.rebuild_corpus)
        print(f"[corpus] {g}", flush=True)

    tasks = [(arm, g) for arm in arms for g in games]
    rows, runs = [], {}
    with ProcessPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        for (arm, g), (r, info) in zip(tasks, ex.map(_score_one, tasks)):
            rows += r
            runs[(arm, g)] = info
            ship = next(x for x in r if x["is_ship"] and x["split"] == "train")
            print(f"[{arm:5s}] {g:12s} nodes={info['n_nodes']:2d} ship=#{info['ship_idx']:2d} "
                  f"ast={ship['ast_nodes']:5d} dl={ship['dl_ratio']:.4f} "
                  f"set={ship['set_ratio']:.3f} status={ship['status']}", flush=True)

    failures = gates(rows, runs)
    for f in failures:
        print(f"[GATE] {f}", file=sys.stderr)
    if a.check:
        print(f"\n{len(failures)} gate failure(s)")
        return 1 if failures else 0

    rows.sort(key=lambda r: (r["arm"], r["game"], r["idx"], r["split"]))
    with (OUT / "metrics.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (OUT / "manifest.json").write_text(json.dumps({
        "arms": {k: {"train_root": v[0], "evaluated_by": v[1]} for k, v in ARMS.items()
                 if k in arms},
        "games": games, "n_rows": len(rows), "gate_failures": failures,
        "runs": [{"arm": k[0], "game": k[1], **v} for k, v in sorted(runs.items())],
    }, indent=1))
    print(f"\nwrote {OUT/'metrics.csv'} ({len(rows)} rows) + manifest.json")
    return 1 if failures else 0


def _score_one(t):
    return score_run(*t)


if __name__ == "__main__":
    raise SystemExit(main())
