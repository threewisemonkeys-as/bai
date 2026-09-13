#!/usr/bin/env python3
"""Independent audit of the wm_quant findings; no model or simulator calls.

Run from the repository root with:
    PYTHONHASHSEED=0 .venv/bin/python analysis/wm_quant/inquiry/verify_findings.py

Reads the original CSV, cached observations and candidate logs. Rebuilds
incumbents independently, executes their P programs with the production
single-frame runner, checks the reported metrics, and writes audit artifacts.
"""
from __future__ import annotations

import ast
import collections
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics as st
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
METRICS = ROOT / "analysis/perception_metrics/metrics.csv"
RUNS = ROOT / "logs/2026-08-24/human_curated/rexpure"
NAMES = {
    "7www9": "Magnets", "7xf97": "Grow", "SET": "SET", "bt3gb": "Ice",
    "colour_lines": "Colour Lines", "diffusion": "Diffusion", "dino": "Dino",
    "dq8gc": "Disease", "eahcw": "Paint", "egg": "Egg",
    "f5w3n": "Space Invaders", "logic_gates": "Logic Gates", "n2ntd": "Mario",
    "s2kt7": "Ants", "va6fq": "Sand",
}


def gz(text):
    return len(gzip.compress(text.encode(), compresslevel=6, mtime=0))


def timeout(_sig, _frame):
    raise TimeoutError("perception call exceeded 2 seconds")


def production_runner():
    # Extract this self-contained function to avoid importing LLM clients and
    # loading credentials through validate.py's unrelated module-level imports.
    path = ROOT / "offline_learning/validate.py"
    source = ast.parse(path.read_text())
    fn = next(n for n in source.body if isinstance(n, ast.FunctionDef)
              and n.name == "run_perceive")
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["run_perceive"]


def write_csv(path, rows):
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execve(sys.executable, [sys.executable, *sys.argv],
                  dict(os.environ, PYTHONHASHSEED="0"))
    from scipy.stats import binomtest, spearmanr

    raw_rows = list(csv.DictReader(METRICS.open()))
    assert len(raw_rows) == 900
    rows = {}
    for r in raw_rows:
        for k in ("idx", "iteration", "is_ship", "ast_nodes", "k_chars", "n_frames"):
            r[k] = int(r[k]) if r[k] else None
        for k in ("info_extraction_ratio", "norm_diversity", "mean_out_chars",
                  "mean_raw_chars", "train_score", "dl_ratio", "set_ratio"):
            r[k] = float(r[k]) if r[k] else None
        rows[r["game"], r["split"], r["idx"]] = r

    manifest = json.loads((ROOT / "analysis/perception_metrics/manifest.json").read_text())
    manifest_runs = {r["game"]: r for r in manifest["runs"]}
    runner = production_runner()
    signal.signal(signal.SIGALRM, timeout)
    endpoint_rows, verified_rows, step_rows, examples, correlations = [], [], [], [], []
    audit_counts = collections.Counter()

    for game in sorted(NAMES, key=NAMES.get):
        run = RUNS / f"{game}_s1"
        pool = list(map(json.loads, (run / "rexpure_run_seed1/candidates.jsonl").open()))
        events = list(map(json.loads, (run / "rexpure_run_seed1/process_log.jsonl").open()))
        by_idx = {c["idx"]: c for c in pool}
        iters = {e["new_idx"]: e["i"] for e in events if e.get("new_idx") is not None}
        assert len(pool) == 30
        ship = max(pool, key=lambda c: c["train_score"])
        best_p = (run / "best_perception_rexpure_seed1.py").read_text()
        best_k = (run / "best_beliefs_rexpure_seed1.txt").read_text()
        assert ship["perception"].strip() == best_p.strip()
        assert ship["world_knowledge"].strip() == best_k.strip()
        assert hashlib.sha256(best_p.encode()).hexdigest() == manifest_runs[game]["best_perception_sha256"]
        assert ship["idx"] == manifest_runs[game]["ship_idx"]
        for c in pool:
            r = rows[game, "train", c["idx"]]
            assert c["train_score"] == r["train_score"]
            assert iters.get(c["idx"]) == r["iteration"]

        seq, best = [], -math.inf
        for idx in sorted(iters, key=iters.get):
            c = by_idx[idx]
            if c["train_score"] > best:
                best = c["train_score"]
                seq.append(c)
            else:
                audit_counts["admitted_nonseed_candidates_not_improving_best"] += 1
        assert seq[-1]["idx"] == ship["idx"]
        first = next(c for c in seq if rows[game, "train", c["idx"]]["status"] == "ok")
        for before, after in zip(seq, seq[1:]):
            a, b = (rows[game, "train", c["idx"]] for c in (before, after))
            step_rows.append({
                "game": game, "previous_idx": before["idx"], "next_idx": after["idx"],
                "iteration": iters[after["idx"]],
                "previous_is_direct_parent": before["idx"] in after["parents"],
                "same_P": before["perception"] == after["perception"],
                "same_K": before["world_knowledge"] == after["world_knowledge"],
                "info_change": b["info_extraction_ratio"] - a["info_extraction_ratio"],
            })

        corpus = json.load(gzip.open(ROOT / f"analysis/perception_metrics/cache/{game}.json.gz", "rt"))
        checkpoint = json.loads((run / "rexpure_run_seed1/resume_state.json").read_text())
        assert corpus["fingerprint"] == checkpoint["train_fingerprint"] == manifest_runs[game]["train_fingerprint"]
        frame_ids = {split: sorted({i for p in corpus[f"{split}_pairs"] for i in p})
                     for split in ("train", "test")}
        shared = len(set(frame_ids["train"]) & set(frame_ids["test"]))
        output_cache = {}
        for c in seq:
            code = c["perception"]
            if code not in output_cache:
                outputs = []
                for x in corpus["frames"]:
                    signal.setitimer(signal.ITIMER_REAL, 2)
                    try:
                        z, _err = runner(code, x)
                    finally:
                        signal.setitimer(signal.ITIMER_REAL, 0)
                    outputs.append(z)
                output_cache[code] = outputs
            outputs = output_cache[code]
            for split, ids in frame_ids.items():
                xs = [corpus["frames"][i] for i in ids]
                zs = [outputs[i] for i in ids]
                info = sum(gz(z) / gz(x) for z, x in zip(zs, xs)) / len(xs)
                norm = gz("\n".join(zs)) / gz("\n".join(xs))
                chars = sum(map(len, zs)) / len(zs)
                ref = rows[game, split, c["idx"]]
                assert math.isclose(info, ref["info_extraction_ratio"], abs_tol=1e-12)
                assert math.isclose(norm, ref["norm_diversity"], abs_tol=1e-12)
                assert math.isclose(chars, ref["mean_out_chars"], abs_tol=1e-12)
                assert len(ids) == ref["n_frames"]
                assert len(set(zs)) / len(zs) == ref["set_ratio"]
                verified_rows.append({"game": game, "split": split, "idx": c["idx"],
                                      "info_ratio": info, "corpus_ratio": norm,
                                      "mean_output_chars": chars, "n_frames": len(ids)})
            if game in ("s2kt7", "egg", "n2ntd", "f5w3n", "diffusion", "logic_gates", "SET"):
                i = frame_ids["train"][0]
                examples.append({"game": game, "idx": c["idx"], "iteration": iters[c["idx"]],
                                 "raw_frame_id": i, "output": outputs[i]})

        for split in ("train", "test"):
            a = rows[game, split, first["idx"]]
            b = rows[game, split, ship["idx"]]
            endpoint_rows.append({
                "game": game, "name": NAMES[game], "split": split,
                "first_valid_idx": first["idx"], "first_valid_iteration": iters[first["idx"]],
                "ship_idx": ship["idx"], "ship_iteration": iters[ship["idx"]],
                "first_info_ratio": a["info_extraction_ratio"], "ship_info_ratio": b["info_extraction_ratio"],
                "info_change_pct": 100 * (b["info_extraction_ratio"] / a["info_extraction_ratio"] - 1),
                "chars_change_pct": 100 * (b["mean_out_chars"] / a["mean_out_chars"] - 1),
                "corpus_change_pct": 100 * (b["norm_diversity"] / a["norm_diversity"] - 1),
                "ast_change_pct": 100 * (b["ast_nodes"] / a["ast_nodes"] - 1),
                "first_train_score": a["train_score"], "ship_train_score": b["train_score"],
                "first_out_chars": a["mean_out_chars"], "ship_out_chars": b["mean_out_chars"],
                "train_test_shared_frames": shared, "test_frames": len(frame_ids["test"]),
            })
        valid = [rows[game, "train", c["idx"]] for c in pool
                 if rows[game, "train", c["idx"]]["status"] == "ok"]
        nonempty = [r for r in valid if r["k_chars"] > 0]
        for label, rs in (("all_working", valid), ("nonempty_K", nonempty)):
            rho = float(spearmanr([r["info_extraction_ratio"] for r in rs],
                                  [r["train_score"] for r in rs]).statistic)
            correlations.append({"game": game, "control": label, "n": len(rs), "rho": rho})
        print(f"verified {game}: {len(seq)} incumbents, {len(output_cache)} programs", flush=True)

    summary = {"input_csv_sha256": hashlib.sha256(METRICS.read_bytes()).hexdigest(),
               "verified_incumbent_split_rows": len(verified_rows),
               "incumbent_updates": len(step_rows),
               "updates_from_other_parent": sum(not r["previous_is_direct_parent"] for r in step_rows),
               "updates_with_same_P": sum(r["same_P"] for r in step_rows),
               **audit_counts, "split_summaries": {}}
    for split in ("train", "test"):
        es = [r for r in endpoint_rows if r["split"] == split]
        vals = [r["info_change_pct"] for r in es]
        pos = sum(v > 0 for v in vals)
        summary["split_summaries"][split] = {
            "rises": pos, "falls": len(vals) - pos,
            "median_info_change_pct": st.median(vals),
            "rises_above_5pct": sum(v > 5 for v in vals),
            "falls_below_minus_5pct": sum(v < -5 for v in vals),
            "two_sided_sign_p_exploratory": float(binomtest(pos, len(vals)).pvalue),
            "median_ship_info_ratio": st.median(r["ship_info_ratio"] for r in es),
            "ship_info_ratio_above_1": sum(r["ship_info_ratio"] > 1 for r in es),
            "median_chars_change_pct": st.median(r["chars_change_pct"] for r in es),
            "median_corpus_change_pct": st.median(r["corpus_change_pct"] for r in es),
            "median_ast_change_pct": st.median(r["ast_change_pct"] for r in es),
        }
    for control in ("all_working", "nonempty_K"):
        cs = [r["rho"] for r in correlations if r["control"] == control]
        summary[f"info_score_correlation_{control}"] = {
            "median_rho": st.median(cs), "positive_games": sum(c > 0 for c in cs)}
    write_csv(OUT / "verified_endpoints.csv", endpoint_rows)
    write_csv(OUT / "verified_incumbents.csv", verified_rows)
    write_csv(OUT / "verified_steps.csv", step_rows)
    write_csv(OUT / "verified_correlations.csv", correlations)
    (OUT / "verified_output_examples.json").write_text(json.dumps(examples, indent=2) + "\n")
    (OUT / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    draw_endpoints([r for r in endpoint_rows if r["split"] == "train"])


def draw_endpoints(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = sorted(rows, key=lambda r: r["info_change_pct"])
    fig, ax = plt.subplots(figsize=(8.0, 5.5), layout="constrained")
    for y, r in enumerate(rows):
        x = r["info_change_pct"]
        color = "#13795b" if x >= 0 else "#a84421"
        ax.plot([0, x], [y, y], color=color, linewidth=2)
        ax.scatter([x], [y], color=color, s=28, zorder=3)
        ax.text(max(0, x) + 2, y, f"{x:+.1f}%", va="center", fontsize=9, color=color)
    ax.set_yticks(range(len(rows)), [r["name"] for r in rows])
    ax.axvline(0, color="#727272", linewidth=.8)
    ax.set_xlim(-30, 160)
    ax.set_xlabel("Change in per-frame compressed output/input ratio (%)")
    ax.set_title("First working candidate → selected final model", loc="left", fontsize=13, pad=15)
    ax.text(0, 1.015, "15 games · one learning run per game · train frames · common horizontal scale",
            transform=ax.transAxes, fontsize=9, color="#555555")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color="#e8e8e8")
    ax.set_axisbelow(True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"info_extraction_endpoint_change.{ext}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
