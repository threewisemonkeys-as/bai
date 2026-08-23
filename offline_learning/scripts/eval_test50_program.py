#!/usr/bin/env python3
"""Program-world-model (WorldCoder arm) ID + FD on the EXACT test50 protocol.

Zero LLM calls. Reuses the LLM arm's persisted eval (logs/id_eval_test50_raw_vs_
learned.json) as the source of truth for the config, the deterministic split
(re-derived via efm.reproduce_test_split and asserted row-for-row against the
stored rows), and the per-item CHOICE SETS (read directly from the stored rows),
so program numbers land in the same table as raw/learned.

Program-ID = forward-simulation inverse dynamics: run T-hat on each choice; the
consistent set S = {choices whose predicted grid == recorded s_t+1}. A bare
'click' choice (parameter-collapsed games) is consistent iff ANY cell's click
reproduces s_t+1 (click_enum semantics). exact/strict = 1 iff S == {truth};
set_credit = 1/|S| if truth in S (id_set_metrics semantics). Rows are written in
the eval_test50_idfd row schema with mode="program", so rescore_test50_id_sim.py
can sim-ground them unchanged (--id-json <this output>).

Program-FD = T-hat(prev, grid, truth action) vs the recorded next grid, scored
with the same exact/partial metrics as the raw-mode LLM FD rows (both predict
full canonical grids) plus per-cell match; stale (identity) baselines included.
On parameter-collapsed games the truth action is the bare verb -- the same
information the LLM arm's FD prompt gets.

    uv run python offline_learning/scripts/eval_test50_program.py \
        --artifact dq8gc=logs/aug6_wc/dq8gc_seed1/best_transition_wc_seed1.py
"""
from __future__ import annotations

import os as _bos, sys as _bsys  # offline_learning/ on sys.path (flat-import the kept libs)
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import eval_forward_modes as efm
import program_runtime as prt
from forward_objective import textdiff_delta_f1
from invdyn_core import exact_match_f1
from worldcoder_optimize import _clean_program, choice_consistent


def grid_or_none(raw: str):
    try:
        return prt.parse_grid_strict(raw)
    except ValueError:
        return None


def eval_game_program(game: str, stored: dict, config: dict, code: str,
                      timeout_s: float) -> dict:
    data_root = Path(config["data_root"])
    transitions, train_total, test_total = efm.reproduce_test_split(
        game, data_root, config["test_dir"], config["seed"],
        config["context_k"], config["test_n"],
    )
    # stored rows: {idx: representative row} (raw + learned share choices/truth).
    # Stored actions are already row-major, so truth/choices need no swap here.
    by_idx = {}
    for row in stored["rows"]:
        row = dict(row)
        by_idx.setdefault(row["idx"], row)
    if len(by_idx) != len(transitions):
        raise AssertionError(
            f"{game}: split size {len(transitions)} != stored rows {len(by_idx)}")
    for idx, tr in enumerate(transitions):
        if by_idx[idx]["truth"] != tr.action:
            raise AssertionError(f"{game} idx {idx}: truth drifted from stored rows")
        if json.loads(by_idx[idx]["raw_grid_start"]) != json.loads(efm.canonical_grid(tr.x_t)):
            raise AssertionError(f"{game} idx {idx}: start grid drifted from stored rows")

    items = prt.prepare_transitions(transitions, config["context_k"])
    for i, it in enumerate(items):
        it.idx = i
    rt = prt.ProgramRuntime(code, timeout_s=timeout_s)
    memo: dict = {}
    id_rows, fd_rows = [], []
    try:
        for idx, (tr, it) in enumerate(zip(transitions, items)):
            choices = by_idx[idx]["choices"]
            consistent = [c for c in choices if choice_consistent(rt, it, c, memo)]
            strict = 1.0 if consistent == [tr.action] else 0.0
            credit = (1.0 / len(consistent)) if tr.action in consistent else 0.0
            id_rows.append({
                "idx": idx, "truth": tr.action, "choices": choices,
                "raw_grid_start": efm.canonical_grid(tr.x_t),
                "raw_grid_target": efm.canonical_grid(tr.x_t1),
                "mode": "program", "z_t": None, "z_t1": None,
                "pred": consistent[0] if consistent else None,
                "prompt": "", "response": "(program forward simulation)",
                "correct": bool(strict), "exact": strict,
                "consistent_set": consistent, "set_credit": credit,
                "cost": 0.0, "retry_errors": [], "perception_error": None,
            })
            pred_grid, err = rt.transition(it.prev, it.grid, it.action)
            pred_canon = prt.canon_grid(pred_grid) if pred_grid is not None else ""
            start, target = efm.canonical_grid(tr.x_t), efm.canonical_grid(tr.x_t1)
            fd_rows.append({
                "mode": "program", "action": tr.action, "idx": idx,
                "start": start, "target": target, "pred": pred_canon,
                "prompt": "", "response": "(program transition)",
                "changed": it.changed,
                "exact": 1.0 if (pred_grid is not None
                                 and prt.canon_grid(pred_grid) == it.next_c) else 0.0,
                "partial": textdiff_delta_f1(start, pred_canon, target),
                "cell_f1": (prt.cell_f1(pred_grid, it.next_grid)
                            if pred_grid is not None else 0.0),
                "stale_exact": exact_match_f1(start, target),
                "stale_partial": textdiff_delta_f1(start, start, target),
                "program_error": err, "perception_error": None, "cost": 0.0,
                "retry_errors": [],
            })
    finally:
        rt.close()

    n = max(1, len(id_rows))
    id_summary = {
        "n": len(id_rows),
        "exact": sum(r["exact"] for r in id_rows) / n,
        "set_credit": sum(r["set_credit"] for r in id_rows) / n,
        "mean_set_size": sum(len(r["consistent_set"]) for r in id_rows) / n,
        "no_pred": sum(1 for r in id_rows if r["pred"] is None),
        "cost": 0.0,
    }
    nf = max(1, len(fd_rows))
    changed_rows = [r for r in fd_rows if r["changed"]]
    fd_summary = {
        "n": len(fd_rows), "changed_n": len(changed_rows),
        "exact": sum(r["exact"] for r in fd_rows) / nf,
        "partial": sum(r["partial"] for r in fd_rows) / nf,
        "cell_f1": sum(r["cell_f1"] for r in fd_rows) / nf,
        "stale_exact": sum(r["stale_exact"] for r in fd_rows) / nf,
        "stale_partial": sum(r["stale_partial"] for r in fd_rows) / nf,
        "dynamic_exact": (sum(r["exact"] for r in changed_rows) / len(changed_rows)
                          if changed_rows else 0.0),
        "program_errors": sum(1 for r in fd_rows if r["program_error"]),
        "cost": 0.0,
    }
    return {
        "game": game, "seed": config["seed"],
        "train_total": train_total, "test_total": test_total,
        "test_n": len(transitions),
        "summary": {"program": id_summary},
        "fd_summary": {"program": fd_summary},
        "rows": id_rows, "fd_rows": fd_rows,
    }


def parse_artifacts(args) -> dict[str, Path]:
    out = {}
    for spec in args.artifact or []:
        game, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--artifact expects game=path, got {spec!r}")
        out[game] = Path(path)
    if args.artifact_root:
        root = Path(args.artifact_root)
        for d in sorted(root.glob(f"*_seed{args.seed}")):
            game = d.name.rsplit("_seed", 1)[0]
            p = d / f"best_transition_wc_seed{args.seed}.py"
            if p.exists():
                out.setdefault(game, p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id-json", type=Path,
                    default=REPO / "logs/id_eval_test50_raw_vs_learned.json",
                    help="LLM-arm eval: source of config, split asserts, choice sets")
    ap.add_argument("--artifact", action="append", metavar="GAME=PATH",
                    help="program artifact per game (repeatable)")
    ap.add_argument("--artifact-root", default=None,
                    help="root with <game>_seed<N>/best_transition_wc_seed<N>.py")
    ap.add_argument("--out", type=Path, default=REPO / "logs/id_eval_test50_program")
    ap.add_argument("--seed", type=int, default=1,
                    help="artifact seed suffix for --artifact-root discovery")
    ap.add_argument("--program-timeout", type=float, default=1.0)
    args = ap.parse_args()

    payload_in = json.loads(args.id_json.read_text())
    config = dict(payload_in["config"])
    # the eval-of-record predates the perc_invdyn -> offline_learning move;
    # remap its stale data_root so split reproduction (and the downstream sim
    # rescore, which reads THIS output's config) resolves on the new tree.
    if not Path(config["data_root"]).exists():
        remapped = config["data_root"].replace("/prototypes/perc_invdyn/",
                                               "/offline_learning/")
        if Path(remapped).exists():
            config["data_root"] = remapped
    stored_by_game = {r["game"]: r for r in payload_in["results"]}
    artifacts = parse_artifacts(args)
    if not artifacts:
        raise SystemExit("no artifacts given (--artifact / --artifact-root)")

    started = time.time()
    results = []
    for game, path in artifacts.items():
        if game not in stored_by_game:
            print(f"[skip] {game}: not in {args.id_json}", flush=True)
            continue
        code = _clean_program(path.read_text())
        t0 = time.time()
        res = eval_game_program(game, stored_by_game[game], config, code,
                                args.program_timeout)
        res["artifact"] = str(path)
        results.append(res)
        s, f = res["summary"]["program"], res["fd_summary"]["program"]
        print(f"[done] {game}: ID strict={s['exact']:.3f} set={s['set_credit']:.3f} "
              f"(sets~{s['mean_set_size']:.2f}, no_pred={s['no_pred']}) | "
              f"FD exact={f['exact']:.3f} partial={f['partial']:.3f} "
              f"cell_f1={f['cell_f1']:.3f} (stale {f['stale_exact']:.3f}) | "
              f"{time.time()-t0:.1f}s", flush=True)

    payload = {
        "config": {
            **{k: config[k] for k in ("games", "seed", "test_n", "test_dir",
                                      "context_k", "k_choices", "data_root")},
            "id_source": str(args.id_json),
            "program_artifacts": {g: str(p) for g, p in artifacts.items()},
            "protocol": "same split + choice sets as id_source; program forward-"
                        "simulation ID (click_enum for bare click) + grid FD",
        },
        "elapsed_seconds": time.time() - started,
        "results": results,
        "aggregate": {"program": {
            "id_exact": (sum(r["summary"]["program"]["exact"] for r in results)
                         / max(1, len(results))),
            "fd_exact": (sum(r["fd_summary"]["program"]["exact"] for r in results)
                         / max(1, len(results))),
        }},
    }
    out_json = args.out.with_suffix(".json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
