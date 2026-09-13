#!/usr/bin/env python3
"""Cross-check the three agents' semantic examples and statistical controls.

Run: PYTHONHASHSEED=0 .venv/bin/python analysis/wm_quant/inquiry/verify_examples.py
"""
import collections
import csv
import gzip
import json
import math
import re
import statistics as st

from scipy.stats import spearmanr

from verify_findings import ROOT, OUT, RUNS, NAMES, gz, production_runner

runner = production_runner()
checks = {}
for game in ["s2kt7", "SET", "colour_lines", "n2ntd", "f5w3n", "egg", "diffusion"]:
    pool = {r["idx"]: r for r in map(json.loads,
            (RUNS / f"{game}_s1/rexpure_run_seed1/candidates.jsonl").open())}
    corpus = json.load(gzip.open(ROOT / f"analysis/perception_metrics/cache/{game}.json.gz", "rt"))
    checks[game] = {}
    for split in ("train", "test"):
        ids = sorted({i for p in corpus[f"{split}_pairs"] for i in p})
        xs = [corpus["frames"][i] for i in ids]
        def outputs(idx):
            results = [runner(pool[idx]["perception"], x) for x in xs]
            assert all(err is None for _, err in results)
            return [z for z, _ in results]
        def ratio(zs):
            return st.mean(gz(z) / gz(x) for z, x in zip(zs, xs))
        if game == "s2kt7":
            old, new = outputs(4), outputs(23)
            cleaned = []
            for z in new:
                parts = z.split("; ")
                d = dict(p.split(":", 1) for p in parts)
                assert d["step"] == "0"
                assert d["added"] == d["cells"]
                assert d["removed"] == d["changed"] == "empty"
                cleaned.append("; ".join(p for p in parts if p.split(":", 1)[0]
                                         not in {"step", "sid", "added", "removed", "changed"}))
            assert cleaned == old
            result = {"frames": len(xs), "same_after_removing_redundant_fields": True,
                      "ratio_before": ratio(old), "ratio_after": ratio(new),
                      "ratio_with_fields_removed": ratio(cleaned)}
        elif game == "SET":
            old, new = outputs(1), outputs(12)
            cleaned = []
            for a, b in zip(old, new):
                old_cells = sorted(re.findall(r"\d+,\d+:[a-z]+", a))
                new_cells = sorted(re.findall(r"\d+,\d+:[a-z]+", b))
                assert old_cells == new_cells
                assert "shape=20x20; cursor:unknown; " in b
                cleaned.append(b.split("; ")[0] + "; cells=" + (" ".join(new_cells) or "none"))
            assert cleaned == old
            result = {"frames": len(xs), "same_cell_facts": True,
                      "identical_after_header_removal_and_sort_restoration": True}
        elif game == "colour_lines":
            old, new = outputs(1), outputs(12)
            for a, b in zip(old, new):
                assert sorted(re.findall(r"\(\d+,\d+,[a-z]+\)", a)) == sorted(re.findall(r"\(\d+,\d+,[a-z]+\)", b))
                assert re.search(r"bg=([a-z]+)", a).group(1) == re.search(r"bg=([a-z]+)", b).group(1)
            result = {"frames": len(xs), "same_background_and_cell_facts": True}
        elif game == "n2ntd":
            old, new = outputs(1), outputs(16)
            assert len(set(new)) == len(xs)
            assert len(set(old)) < len(xs)
            result = {"frames": len(xs), "initial_unique_outputs": len(set(old)),
                      "final_unique_outputs": len(set(new))}
            if split == "train":
                a, b = ids.index(1), ids.index(12)
                assert old[a] == old[b] and new[a] != new[b]
                grids = [json.loads(x) for x in (xs[a], xs[b])]
                diffs = [(r, c, v, grids[1][r][c]) for r, row in enumerate(grids[0])
                         for c, v in enumerate(row) if v != grids[1][r][c]]
                assert diffs == [(9, 1, "red", "mediumpurple")]
                result["collision_cell_difference"] = diffs
        elif game == "f5w3n":
            new = outputs(20)
            namespace = {}
            exec(pool[20]["perception"], namespace)
            shorten = namespace["COLOUR_SHORT"]
            for x, z in zip(xs, new):
                grid = json.loads(x)
                assert len(grid) == 16 and all(len(row) == 16 for row in grid)
                assert collections.Counter(v for row in grid for v in row).most_common(1)[0][0] == "black"
                expected = {(r, c, shorten[v]) for r, row in enumerate(grid)
                            for c, v in enumerate(row) if v not in ("black", "orange")}
                agent, cell_text = z.split(" | cells:")
                got = set()
                if cell_text != "empty":
                    for t in cell_text.split(";"):
                        r, c, col = t.split(",")
                        got.add((int(r), int(c), col))
                assert got == expected
                orange = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == "orange"]
                assert len(orange) <= 1
                assert agent == (f"a:{orange[0][0]},{orange[0][1]}" if orange else "a:?")
            result = {"frames": len(xs), "all_nonbackground_cells_preserved": True,
                      "assumptions": "Known 16x16 black grid and observed palette/color-code mapping"}
        elif game == "egg":
            old, new = outputs(5), outputs(8)
            cleaned = []
            for z in new:
                pieces = z.split("; ")
                cleaned.append("; ".join(p for p in pieces
                                         if not p.startswith("(0,0)=") and "_blob:" not in p))
            assert cleaned == old
            result = {"frames": len(xs), "extra_fields_are_corner_color_and_blob_bounds": True,
                      "ratio_before": ratio(old), "ratio_after": ratio(new)}
        else:
            old, new = outputs(24), outputs(28)
            cleaned = [z.split("; ", 2)[2] for z in new]
            assert cleaned == old
            result = {"frames": len(xs), "final_increase_adds_background_and_dimensions": True,
                      "background_values": sorted({z.split(";")[0] for z in new})}
        checks[game][split] = result

# Reconstruct the nonempty-K / unique-P control directly from original candidates.
metric_rows = list(csv.DictReader((ROOT / "analysis/perception_metrics/metrics.csv").open()))
dedup_rhos = []
for game in NAMES:
    rr = {int(r["idx"]): r for r in metric_rows if r["game"] == game and r["split"] == "train"}
    pool = list(map(json.loads, (RUNS / f"{game}_s1/rexpure_run_seed1/candidates.jsonl").open()))
    groups = collections.defaultdict(list)
    for c in pool:
        r = rr[c["idx"]]
        if r["status"] == "ok" and c["world_knowledge"].strip():
            groups[c["perception"]].append(r)
    xs = [float(v[0]["info_extraction_ratio"]) for v in groups.values()]
    ys = [st.mean(float(r["train_score"]) for r in v) for v in groups.values()]
    dedup_rhos.append(float(spearmanr(xs, ys).statistic))
assert math.isclose(st.median(dedup_rhos), .3088235294117647)
assert sum(r > 0 for r in dedup_rhos) == 11
checks["unique_P_nonempty_K"] = {"median_rho": st.median(dedup_rhos),
                                  "positive_games": sum(r > 0 for r in dedup_rhos)}

endpoints = {r["game"]: r for r in csv.DictReader((OUT / "verified_endpoints.csv").open()) if r["split"] == "train"}
agentic = list(map(json.loads, (ROOT / "logs/2026-09-08/agent_wm_full/rows.jsonl").open()))
planning_rows = []
for game in NAMES:
    raw = json.loads((ROOT / f"logs/2026-09-03/planning_v2_online_ds_percap_nl/{game}/online.json").read_text())["rows"]
    plain = st.mean(r["lmwm"]["pass_rate"] for r in raw if r.get("lmwm", {}).get("status") == "evaluated")
    agent = st.mean(r["agent"]["pass_rate"] for r in agentic if r["game"] == game and r["agent"].get("status") == "done")
    planning_rows.append({"game": game, "plain": plain, "agentic": agent,
                          "info": float(endpoints[game]["ship_info_ratio"]),
                          "growth": float(endpoints[game]["info_change_pct"])})
checks["planning_correlations"] = {}
for metric in ("info", "growth"):
    for arm in ("plain", "agentic"):
        rho = float(spearmanr([r[metric] for r in planning_rows], [r[arm] for r in planning_rows]).statistic)
        checks["planning_correlations"][f"{metric}_{arm}"] = rho

(OUT / "example_verification.json").write_text(json.dumps(checks, indent=2) + "\n")
print(json.dumps(checks, indent=2))
