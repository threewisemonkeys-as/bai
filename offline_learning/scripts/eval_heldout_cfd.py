#!/usr/bin/env python3
"""Held-out contrastive-FD (cFD) for rexpure artifacts that have ALREADY been trained.

Why this exists
---------------
The training loop bakes contrastive options over the TRAIN split only
(`rexpure_optimize.build_data` calls `bake_decoys(train, ...)`), and the end-of-run
summary reports a held-out number for the inverse term alone -- a run under
`--fd-scorer none` writes `"forward_score": null`. That is fine while every arm optimises
ID, because test ID is then everyone's own metric. It stops being fine for the `--no-id`
objective ablation, which trains on cFD and would otherwise be judged only on the term it
deliberately dropped.

This scores the SHIPPED candidate of a finished run on the clean test split, using the
same `eval_cfd_on` the in-run `--cfd-test` path uses and the same `bake_test_decoys`
convention, so a number produced here and one produced in-run are interchangeable.

How the split is reproduced
---------------------------
Not re-derived by hand: the run's own `launch.json` argv is re-parsed through
`rexpure_optimize.build_parser()` and fed back to `build_data()`, which is the function
that produced the split in the first place. The rebuild is then CHECKED against the
`train_fingerprint` the run checkpointed in `rexpure_run_seed<seed>/resume_state.json`;
a mismatch aborts that game rather than reporting a number off a different split.

Both target renderings are scored, and both belong in an ablation table:
  perceived -- options shown as P(option). Measures whether THIS P preserves what
      separates the true next state from a near miss. Each arm answers the question in
      its own feature language, so cross-arm comparison is not quite like-for-like.
  raw       -- options shown as raw frames. Candidate-independent all the way to the
      prompt, so the 1/(n+1) chance floor is a constant for any P and every arm answers
      the identical question. Report this one as the cross-arm column.

Usage
-----
    # one artifact tree (the reference, or an ablation arm)
    uv run python offline_learning/scripts/eval_heldout_cfd.py \
        --artifact-root logs/2026-08-24/human_curated

    # several arms at once, into one comparison table
    uv run python offline_learning/scripts/eval_heldout_cfd.py \
        --artifact-root "NLWM=logs/2026-08-24/human_curated" \
        --artifact-root "-ID=logs/2026-09-XX/ablations/noid" \
        --out logs/2026-09-XX/heldout_cfd

Results are written per game as `heldout_cfd_seed<seed>.json` next to the artifacts (so a
game is scored once and re-runs resume for free) plus a combined table at --out.
"""
from __future__ import annotations

import os as _bos, sys as _bsys  # offline_learning/ on sys.path (flat-import the kept libs)
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(_bos.path.abspath(__file__)))))
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))

import argparse
import json
import random
from pathlib import Path

from invdyn_core import (  # noqa: E402
    _train_fingerprint,
    bake_test_decoys,
    eval_cfd_on,
    make_config,
    run_async,
)
from rexpure_optimize import build_data, build_parser  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MODES = (("perceived", False), ("raw", True))


def run_argv(run_dir: Path) -> list[str]:
    """The flags the run was launched with, from its own launch.json."""
    p = run_dir / "launch.json"
    if not p.is_file():
        raise FileNotFoundError(f"{p} missing -- cannot reproduce the split")
    cmd = json.loads(p.read_text())["cmd"]
    i = next(i for i, x in enumerate(cmd) if str(x).endswith(".py"))
    return [str(x) for x in cmd[i + 1:]]


def rebuild_split(run_dir: Path):
    """(args, test, transitions, context_k) for a finished run, fingerprint-checked."""
    args = build_parser().parse_args(run_argv(run_dir))
    train, test, _pool, context_k, _wl, transitions, _idn = build_data(
        args, random.Random(args.seed))
    state = run_dir / f"rexpure_run_seed{args.seed}" / "resume_state.json"
    if state.is_file():
        want = json.loads(state.read_text()).get("train_fingerprint")
        got = _train_fingerprint(train)
        if want and want != got:
            raise RuntimeError(
                f"split mismatch for {run_dir.name}: rebuilt train fingerprint {got} != "
                f"checkpointed {want}. The data or the flags moved since the run; "
                "scoring would be against a different test set")
    return args, test, transitions, context_k


def score_game(run_dir: Path, *, concurrency: int, force: bool) -> dict:
    args, test, transitions, context_k = rebuild_split(run_dir)
    out_path = run_dir / f"heldout_cfd_seed{args.seed}.json"
    if out_path.is_file() and not force:
        return json.loads(out_path.read_text())

    pp = run_dir / f"best_perception_rexpure_seed{args.seed}.py"
    bp = run_dir / f"best_beliefs_rexpure_seed{args.seed}.txt"
    for f in (pp, bp):
        if not f.is_file():
            raise FileNotFoundError(f"{f} missing -- run not finished")
    code, beliefs = pp.read_text(), bp.read_text()

    # Same convention as the in-run --cfd-test path: hard decoys iff the run trained with
    # them, same count, own rng offset. bake_decoys mutates `test` in place.
    bake_test_decoys(test, transitions, args.cfd_decoys, args.seed,
                     hard=args.cfd_hard_decoys)
    res = {
        "run_dir": str(run_dir.relative_to(REPO) if run_dir.is_relative_to(REPO)
                       else run_dir),
        "seed": args.seed, "n_test": len(test), "context_k": context_k,
        "n_decoys": args.cfd_decoys, "hard_decoys": args.cfd_hard_decoys,
        "chance": 1.0 / (args.cfd_decoys + 1),
        "task_model": args.task_model,
        "trained_with": {"no_id": getattr(args, "no_id", False),
                         "no_beliefs": getattr(args, "no_beliefs", False),
                         "no_perception": args.no_perception,
                         "contrastive_fd": args.contrastive_fd,
                         "fd_scorer": args.fd_scorer},
        "cost": 0.0,
    }
    cfg = make_config(args.task_model, args.client,
                      provider_order=args.task_provider_order,
                      reasoning_json=args.task_reasoning_json)
    for mode, raw in MODES:
        s, c = run_async(eval_cfd_on(
            cfg, code, beliefs, test, concurrency=concurrency, context_k=context_k,
            raw_targets=raw,
            log_path=run_dir / f"test_trace_cfd_{mode}_rexpure_seed{args.seed}.json"))
        res[mode] = s
        res["cost"] += c
        print(f"    {mode:>9} targets: cFD {s:.2f}  (chance {res['chance']:.2f})",
              flush=True)
    out_path.write_text(json.dumps(res, indent=2) + "\n")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact-root", action="append", required=True,
                    metavar="[LABEL=]PATH",
                    help="a tree holding rexpure/<game>_s1/ (repeatable). LABEL names the "
                         "arm in the comparison table; defaults to the dir name")
    ap.add_argument("--games", default="",
                    help="comma-separated subset (default: every <game>_s1 found)")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--force", action="store_true",
                    help="rescore games that already have heldout_cfd_seed*.json")
    ap.add_argument("--out", default="",
                    help="write the combined table here (.json + .md)")
    a = ap.parse_args()

    arms = []
    for spec in a.artifact_root:
        label, _, path = spec.partition("=")
        if not path:
            label, path = Path(label).name, label
        root = Path(path) if Path(path).is_absolute() else REPO / path
        rex = root / "rexpure"
        if not rex.is_dir():
            raise SystemExit(f"no rexpure/ under {root}")
        arms.append((label, rex))

    want = [g for g in a.games.split(",") if g]
    table: dict[str, dict[str, dict]] = {}
    for label, rex in arms:
        dirs = sorted(d for d in rex.iterdir() if d.is_dir() and d.name.endswith("_s1"))
        if want:
            dirs = [d for d in dirs if d.name[:-3] in want]
        print(f"\n=== {label}: {len(dirs)} game(s) under {rex}", flush=True)
        for d in dirs:
            game = d.name[:-3]
            print(f"  {game}", flush=True)
            try:
                table.setdefault(game, {})[label] = score_game(
                    d, concurrency=a.concurrency, force=a.force)
            except Exception as exc:  # noqa: BLE001
                print(f"    SKIP {game}: {exc}", flush=True)
                table.setdefault(game, {})[label] = {"error": str(exc)}

    labels = [lb for lb, _ in arms]
    lines = ["# Held-out contrastive FD", "",
             "`raw` targets are candidate-independent (constant chance floor for any P) "
             "and are the cross-arm column; `perceived` targets ask each arm the question "
             "in its own feature language.", "",
             "| game | chance | " + " | ".join(
                 f"{lb} raw | {lb} perc" for lb in labels) + " |",
             "|" + "---|" * (2 + 2 * len(labels))]
    means = {lb: {"raw": [], "perceived": []} for lb in labels}
    for game in sorted(table):
        row, chance = [], "--"
        for lb in labels:
            r = table[game].get(lb) or {}
            if "error" in r or "raw" not in r:
                row += ["--", "--"]
                continue
            chance = f"{r['chance']:.2f}"
            row += [f"{r['raw']:.2f}", f"{r['perceived']:.2f}"]
            means[lb]["raw"].append(r["raw"])
            means[lb]["perceived"].append(r["perceived"])
        lines.append(f"| {game} | {chance} | " + " | ".join(row) + " |")
    macro = []
    for lb in labels:
        for m in ("raw", "perceived"):
            v = means[lb][m]
            macro.append(f"{sum(v)/len(v):.3f}" if v else "--")
    lines.append("| **macro** | | " + " | ".join(macro) + " |")
    md = "\n".join(lines) + "\n"
    print("\n" + md)
    if a.out:
        out = Path(a.out) if Path(a.out).is_absolute() else REPO / a.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".json").write_text(json.dumps(table, indent=2) + "\n")
        out.with_suffix(".md").write_text(md)
        print(f"wrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
