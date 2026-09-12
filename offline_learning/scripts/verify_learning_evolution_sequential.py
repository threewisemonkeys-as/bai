"""Verify the manually reviewed learning examples and render a frame preview.

Run from the repo:
    .venv/bin/python offline_learning/scripts/verify_learning_evolution_sequential.py

Uses original logs and recorded frames only. No model calls. The local checks
below formalize the limited natural-language rules discussed in the report;
they do not execute or validate the complete learned belief programs.
"""

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from offline_learning.validate import run_perceive, strip_autumn_obs_metadata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np

csv.field_size_limit(10**7)
EVIDENCE = ROOT / "analysis/learning_example/audit/learning_evolution_sequential_audit"
BASE = ROOT / "logs/2026-08-24/human_curated/rexpure"
NAMES = {
    "dq8gc": "Disease", "egg": "Egg", "logic_gates": "Logic Gates",
    "SET": "SET", "n2ntd": "Mario", "bt3gb": "Ice",
    "f5w3n": "Space Invaders", "colour_lines": "Colour Lines",
    "diffusion": "Diffusion", "eahcw": "Paint", "s2kt7": "Ants",
    "va6fq": "Sand", "7www9": "Magnets", "dino": "Dino", "7xf97": "Grow",
}


def content_hash(*parts):
    digest = hashlib.md5()
    for part in parts:
        digest.update((part or "").encode())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def ancestry(pool, node):
    chain = [node]
    while pool[chain[-1]]["parents"]:
        assert len(pool[chain[-1]]["parents"]) == 1
        chain.append(pool[chain[-1]]["parents"][0])
    return chain[::-1]


def cells(grid, color):
    return {(r, c) for r, row in enumerate(grid) for c, value in enumerate(row)
            if value == color}


def local_rules(data):
    """Assert only the quoted local rule consequences against recorded grids."""
    game = data["game"]
    frames = data["frames"]
    for before, after in zip(frames, frames[1:]):
        grid = np.array(json.loads(before["observation"]))
        nxt = np.array(json.loads(after["observation"]))
        action = before["action"]
        if game == "SET":
            expected = grid.copy()
            tiles = [(r, c) for r in (1, 7, 13) for c in (4, 8, 12)]
            selected = [(r, c) for r, c in tiles
                        if "gold" in grid[r:r+5, c:c+3]]
            assert len(selected) in (2, 3)
            if action == "noop" and len(selected) == 3:
                expected[expected == "gold"] = "white"
                # This triple is invalid: two seagreen cards and one coral.
                assert Counter(grid[r+1, c+1] for r, c in selected) == {
                    "seagreen": 2, "coral": 1}
            elif action.startswith("click "):
                _, row, col = action.split()
                r, c = next((r, c) for r, c in tiles
                            if r <= int(row) < r+5 and c <= int(col) < c+3)
                tile = expected[r:r+5, c:c+3]
                tile[tile == "white"] = "gold"
            else:
                assert action == "noop"
            assert np.array_equal(expected, nxt)
        elif game == "dq8gc":
            expected = grid.copy()
            if action == "right":
                assert cells(grid, "darkgreen") == {(2, 3)}
                assert grid[2, 4] == "black"
                expected[2, 3], expected[2, 4] = "black", "darkgreen"
            else:
                assert action == "noop"
                for r, c in cells(grid, "gray"):
                    if any((rr, cc) in cells(grid, "darkgreen")
                           for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))):
                        expected[r, c] = "darkgreen"
            assert np.array_equal(expected, nxt)
        elif game == "7www9":
            blue, red = cells(grid, "blue"), cells(grid, "red")
            assert red == cells(nxt, "red") == {(7, 7), (8, 7)}
            predicted = set()
            for r, c in blue:
                if action == "noop":
                    predicted.add((r, c))
                else:
                    assert action == "down"
                    outside = [(rr, cc) for rr, cc in blue if rr not in (7, 8)]
                    if r not in (7, 8) or any(cc == c for rr, cc in outside):
                        predicted.add((r+1, c))
                    else:
                        predicted.add((9 if r == 7 else 8, c+1 if c < 7 else c-1))
            assert predicted == cells(nxt, "blue")
            assert set(np.unique(grid)) == set(np.unique(nxt)) == {"black", "red", "blue"}
        elif game == "logic_gates":
            positions = [(4, 12), (8, 12), (12, 4), (12, 19), (16, 12), (20, 12)]
            expected = [grid[r, c] for r, c in positions]
            if action.startswith("click "):
                assert action == "click 13 4"
                flip = {"pink": "red", "red": "pink", "orange": "darkblue", "darkblue": "orange"}
                changed = [2, 4, 5, 1 if expected[3] == "pink" else 0]
                for i in changed:
                    expected[i] = flip[expected[i]]
            else:
                assert action == "noop"
            for (r, c), color in zip(positions, expected):
                assert np.all(nxt[r:r+2, c:c+2] == color)
        elif game == "egg":
            expected = grid.copy()
            if action == "click 0 0":
                assert grid[0, 0] == "red"
                expected[0, 0] = "pink"
                expected[expected == "tan"] = "gray"
                for r, c in cells(grid, "tan"):
                    expected[r+1, c] = "gold"
            else:
                assert action == "noop"
            assert np.array_equal(expected, nxt)
        elif game == "n2ntd":
            red = cells(grid, "red")
            assert len(red) == 1
            r, c = next(iter(red))
            if action == "left":
                predicted = {(r, c-1)}
            else:
                assert action == "noop"
                predicted = {(r+1, c)} if r < 11 and grid[r+1, c] == "white" else red
            assert predicted == cells(nxt, "red")
        elif game == "dino":
            red = cells(grid, "red")
            assert action in ("noop", "up")
            shift = -6 if action == "up" else (0 if max(r for r, c in red) == 19 else 1)
            assert {(r+shift, c) for r, c in red} == cells(nxt, "red")
            for color in ("green", "yellow"):
                assert {(r, c-1) for r, c in cells(grid, color)} == cells(nxt, color)
        else:
            raise AssertionError(game)
    scope = {
        "SET": "All cells; invalid triple selection/reset only.",
        "dq8gc": "All visible cells in approach/infection window; no physical stack interpretation.",
        "7www9": "All visible cells; not cell identity or general magnet dynamics.",
        "logic_gates": "Six 2x2 blocks only; wire model is incorrect and excluded.",
        "egg": "All cells for pre-break noop and button click; excludes liquid spreading and height generalization.",
        "n2ntd": "Red motion only; excludes other moving objects and other mechanics.",
        "dino": "Red, green and yellow cells in local window; no general collision or wrap claim.",
    }[game]
    return {"transitions": len(frames)-1, "scope": scope}


def verify_scenario(data, pool, events, prediction_lines):
    trajectory = ROOT / data["trajectory_path"]
    with trajectory.open() as handle:
        rows = {int(r["Step"]): r for r in csv.DictReader(handle)}
    steps = data["steps"]
    assert steps == list(range(steps[0], steps[-1]+1))
    assert [f["step"] for f in data["frames"]] == steps
    frames = {f["step"]: f for f in data["frames"]}
    for frame in frames.values():
        row = rows[frame["step"]]
        assert frame["observation"] == strip_autumn_obs_metadata(row["Observation"])
        assert frame["action"] == row["Action"]
    nodes = [s["node"] for s in data["snapshots"]]
    assert data["lineage"] == ancestry(pool, nodes[-1])
    assert nodes == [n for n in data["lineage"] if n in nodes]
    p_runs = 0
    for snap in data["snapshots"]:
        node = snap["node"]
        candidate, event = pool[node], events[node]
        assert snap["perception"] == candidate["perception"]
        assert snap["belief"] == candidate["world_knowledge"]
        assert snap["parents"] == candidate["parents"]
        assert snap["iteration"] == event["i"]
        assert snap["candidate_line"] == candidate["source_line"]
        assert snap["process_line"] == event["source_line"]
        assert snap["train_score"] == candidate["train_score"]
        assert [o["step"] for o in snap["outputs"]] == steps
        for out in snap["outputs"]:
            actual, error = run_perceive(candidate["perception"], frames[out["step"]]["observation"])
            assert error is None and actual == out["output"]
            p_runs += 1
    for pred in data["predictions"] + data.get("supplementary_predictions", []):
        recorded = prediction_lines[pred["source_line"]-1]
        assert all(recorded[k] == v for k, v in pred["record"].items())
        candidate = pool[pred["node"]]
        assert recorded["cand_hash"] == content_hash(candidate["perception"], candidate["world_knowledge"])
        if "step" in pred:
            step = pred["step"]
            before, after = frames[step], frames[step+1]
            assert recorded["tr_hash"] == content_hash(before["observation"], after["observation"], before["action"])
            snap = next(s for s in data["snapshots"] if s["node"] == pred["node"])
            outputs = {o["step"]: o["output"] for o in snap["outputs"]}
            assert recorded["z_t"] == outputs[step] and recorded["z_t1"] == outputs[step+1]
    return p_runs, local_rules(data)


def preview(data_by_game):
    specs = [
        ("SET", "SET: select a third card, then clear an invalid selection",
         "P17 groups cells into cards; B25 replaces row/column alignment with exactly three selected cards.",
         ["2 selected cards", "2 selected cards", "3 selected cards", "0 selected; all 9 remain"]),
        ("dq8gc", "Disease: approach, then infection",
         "P8 adds neighbor counts; P17 adds directions; B23 uses them but wrongly calls neighbors 'stacks'.",
         ["2,3:darkgreen", "2,3:darkgreen", "2,4:darkgreen S:gray", "2,4:darkgreen S:darkgreen"]),
        ("7www9", "Magnets: down causes diagonal motion, then ordinary downward motion",
         "P6 adds rDist / noRedRow; B8 branches on those exact tags. Rules are only locally correct.",
         ["rDist2 / rDist2", "rDist1 / noRedRow", "rDist1 / noRedRow", "noRedRow / noRedRow"]),
        ("logic_gates", "Logic Gates: a switch changes a conditional group of blocks",
         "P7 extracts six blocks; B21 uses the right switch's color to determine the left click's effects.",
         ["b12_4:pink\nb4_12:darkblue", "b12_4:pink\nb4_12:darkblue",
          "b12_4:red\nb4_12:orange", "blocks unchanged\nwires change (B incorrect)"]),
    ]
    fig = plt.figure(figsize=(12.8, 16.1), facecolor="white")
    fig.text(.04, .975, "Learning-evolution candidates: recorded frame sequences", fontsize=17, weight="bold")
    fig.text(.04, .954, "Frame preview; full P/B checkpoints and exact outputs are in the accompanying report and evidence JSON.", fontsize=9, color="#4b5563")
    for row, (game, title, subtitle, labels) in enumerate(specs):
        data = data_by_game[game]
        top = .924-row*.223
        fig.text(.04, top, title, fontsize=12, weight="bold", color="#17212b")
        fig.text(.04, top-.015, subtitle, fontsize=8.4, color="#4b5563")
        nodes = ", ".join(f"{s['node']} (i={s['iteration']})" for s in data["snapshots"])
        fig.text(.04, top-.029, f"{data['drive']}  |  checkpoints: {nodes}", fontsize=8, color="#4b5563")
        for col, frame in enumerate(data["frames"]):
            ax = fig.add_axes([.043+col*.24, top-.183, .205, .138])
            grid = json.loads(frame["observation"])
            rgb = np.array([[to_rgb(color) for color in line] for line in grid])
            ax.imshow(rgb, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticks(np.arange(-.5, len(grid[0]), 1), minor=True)
            ax.set_yticks(np.arange(-.5, len(grid), 1), minor=True)
            ax.grid(which="minor", color="#a9b7c0", alpha=.17, linewidth=.35)
            ax.tick_params(which="minor", length=0)
            ax.set_title(f"Frame {frame['step']}", fontsize=9, pad=5)
            ax.set_xlabel(labels[col], fontsize=8, labelpad=6,
                          fontfamily="monospace" if game != "SET" else "sans-serif")
            if col < 3:
                action = frame["action"].replace("click ", "click\n")
                fig.text(.263+col*.24, top-.110, action+"\n\u2192", ha="center", va="center", fontsize=8)
    fig.text(.04, .027, "SET counts are derived from P's gold-containing card groups; other captions are excerpts or summaries of learned P.", fontsize=8, color="#4b5563")
    fig.text(.04, .014, "These are explored learning branches. Correct local examples do not establish complete game models or causal necessity of a feature.", fontsize=8, color="#4b5563")
    prefix = ROOT / "analysis/learning_example/learning_evolution_sequential_candidates"
    for extension in ("png", "pdf", "svg"):
        fig.savefig(prefix.with_suffix("."+extension), dpi=155, facecolor="white")
    plt.close(fig)


def main():
    reports = [json.loads(path.read_text()) for path in sorted(EVIDENCE.glob("*.json"))]
    by_game = {d["game"]: d for d in reports}
    assert len(reports) == len(by_game) == 15
    assert set(by_game) == {p.name.removesuffix("_s1") for p in BASE.glob("*_s1")}
    index = []
    total_runs = total_predictions = total_transitions = 0
    for game, name in NAMES.items():
        data = by_game[game]
        pool_path = ROOT / data["candidate_path"]
        pool = {c["idx"]: dict(c, source_line=i) for i, c in enumerate(
            map(json.loads, pool_path.read_text().splitlines()), 1)}
        events = {e["new_idx"]: dict(e, source_line=i) for i, e in enumerate(
            map(json.loads, (ROOT/data["process_path"]).read_text().splitlines()), 1)
            if e.get("new_idx") is not None}
        assert len(pool) == 30
        saved = data["saved_node"]
        directory = pool_path.parent.parent
        assert pool[saved]["perception"].strip() == (directory/"best_perception_rexpure_seed1.py").read_text().strip()
        assert pool[saved]["world_knowledge"].strip() == (directory/"best_beliefs_rexpure_seed1.txt").read_text().strip()
        assert ancestry(pool, saved) == data["saved_lineage"]
        item = dict(game=game, name=name, candidates=len(pool), classification=data["classification"],
                    saved_node=saved, saved_lineage=data["saved_lineage"],
                    candidates_metadata=[dict(node=n, parents=c["parents"], iteration=events.get(n, {}).get("i"),
                                              candidate_line=c["source_line"], process_line=events.get(n, {}).get("source_line"))
                                         for n, c in pool.items()])
        if "snapshots" in data:
            prediction_lines = list(map(json.loads, pool_path.with_name("predictions.jsonl").read_text().splitlines()))
            runs, check = verify_scenario(data, pool, events, prediction_lines)
            total_runs += runs
            total_predictions += len(data["predictions"])+len(data.get("supplementary_predictions", []))
            total_transitions += check["transitions"]
            item["local_rule_verification"] = check
            item["selected_lineage"] = data["lineage"]
            item["selected_nodes"] = [s["node"] for s in data["snapshots"]]
            item["steps"] = data["steps"]
            item["drive"] = data["drive"]
            if "alternative_window" in data:
                runs, check = verify_scenario(data["alternative_window"], pool, events, prediction_lines)
                total_runs += runs
                total_predictions += len(data["alternative_window"]["predictions"])
                total_transitions += check["transitions"]
                item["alternative_local_rule_verification"] = check
        else:
            assert data["candidates_reviewed"] == len(pool)
            for record in data["selected_records"]:
                assert record["perception"] == pool[record["node"]]["perception"]
                assert record["belief"] == pool[record["node"]]["world_knowledge"]
        index.append(item)
    summary = dict(games=len(index), candidate_records=sum(g["candidates"] for g in index),
                   selected_perception_frame_evaluations=total_runs,
                   cached_prediction_records_verified=total_predictions,
                   local_transitions_checked=total_transitions,
                   method="Manual sequential review of all candidates; exact P reruns on selected recorded frames, fresh namespace and single-frame history. Local checks cover only specified B clauses.",
                   score_note="Cached CFD is contrastive next-frame selection, not free-form simulation. Whole-training scores are not window accuracies.",
                   review_order=[NAMES[g] for g in NAMES], games_detail=index)
    (ROOT/"analysis/learning_example/audit/learning_evolution_sequential_audit_index.json").write_text(json.dumps(summary, indent=2)+"\n")
    preview(by_game)
    print(f"Verified {len(index)} game records / {sum(g['candidates'] for g in index)} candidate metadata records.")
    print(f"Re-executed {total_runs} selected P/frame pairs; checked {total_predictions} cached records and {total_transitions} local transitions.")
    print("Saved audit index and PNG/PDF/SVG frame preview.")


if __name__ == "__main__":
    main()
