#!/usr/bin/env python3
"""Render exact Magnets P(X) at three learning checkpoints and abridged B.

Run: .venv/bin/python offline_learning/scripts/fig_learning_evolution_magnets.py
Outputs PDF, SVG, PNG, LaTeX caption, full-output TXT, and provenance JSON.
No model calls. P uses the training evaluator's fresh, single-frame namespace.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from offline_learning.validate import run_perceive, strip_autumn_obs_metadata

RUN = ROOT / "logs/2026-08-24/human_curated/rexpure/7www9_s1/rexpure_run_seed1"
TRAJECTORY = ROOT / "offline_learning/human_data/7www9/informative_curated/drives/train_d1/episode_0/trajectory.csv"
STEM = ROOT / "analysis/learning_example/learning_evolution_magnets_relations"
STEPS, NODES = (32, 33, 34, 35), (0, 4, 8)
CROP = dict(row_start=5, row_stop=12, col_start=4, col_stop=10)
BG, INK, MUTED, RULE = "#fcfcfb", "#17212b", "#717b83", "#dce2e5"
BLUE, BLUE_BG = "#2168b5", "#e9f2fc"
GREEN, GREEN_BG = "#147151", "#eaf5ef"
ORANGE = "#a75a21"
SANS, MONO = "DejaVu Sans", "DejaVu Sans Mono"
W, PAD, GAP = 5.5, .16, .13
CW = (W-2*PAD-3*GAP)/4
X = [PAD+i*(CW+GAP) for i in range(4)]
FS_P, LH_P = 6.1, .105
SUBS = "₀₁₂₃"

# Ordered, literal excerpts. Ellipses mark source omissions, including the up
# branch. Normalization below removes only whitespace and Markdown list/style.
B8_LINES = [
    '… Vertical actions (up/down): … For blue cells with label "rDist":',
    'If there is a "noRedRow" blue in the same column, then:',
    '  The rDist blue moves 1 row toward that noRedRow blue …',
    '  The noRedRow blue moves 1 row further in the same vertical direction.',
    'Else (no "noRedRow" in the same column): … For action "down":',
    '  The rDist blue in row 7 … moves down 2 rows to row 9,',
    '  also moves horizontally 1 step toward column 7, and becomes "noRedRow".',
    '  The rDist blue in row 8 … moves horizontally 1 step toward column 7 …',
    '… The action "noop" changes no feature. …',
]
B4_LINES = [
    '… A pair of blue cells exists. … the top cell’s coordinates (r, c). …',
    '… down: (r, c) -> (r+1, c); if c == 5 then additionally c -> c+1;',
    'if c == 9 then additionally c -> c-1. …',
    '… noop: no change. …',
]
B_EXCERPTS = {0: [], 4: B4_LINES, 8: B8_LINES}
RELATION = r"rDist(?:N|\d+)?|noRedRow"
EMPHASIS = RELATION + r"|moves 1 row toward|moves 1 row further|moves down 2 rows to row 9|moves horizontally 1 step toward column 7|changes no feature|same column|c == 5|c == 9|c -> c\+1|c -> c-1|no change"


def compact(text):
    text = re.sub(r"(?m)^\s*[-*]\s+", "", text)
    return re.sub(r"\s+", "", text.replace("**", "").replace("`", ""))


def verify_excerpt(source, excerpt):
    source, cursor = compact(source), 0
    for fragment in excerpt.split("…"):
        fragment = compact(fragment)
        if not fragment:
            continue
        start = source.find(fragment, cursor)
        assert start >= 0, f"Excerpt does not match source: {fragment!r}"
        cursor = start+len(fragment)


def digest(*parts):
    result = hashlib.md5()
    for part in parts:
        result.update((part or "").encode())
        result.update(b"\0")
    return result.hexdigest()[:16]


def positions(grid, color):
    return {(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == color}


def load():
    csv.field_size_limit(10**7)
    pool = {c["idx"]: dict(c, source_line=i) for i, c in enumerate(
        map(json.loads, (RUN/"candidates.jsonl").read_text().splitlines()), 1)}
    events = {e["new_idx"]: dict(e, source_line=i) for i, e in enumerate(
        map(json.loads, (RUN/"process_log.jsonl").read_text().splitlines()), 1)
        if e.get("new_idx") is not None}
    with TRAJECTORY.open() as handle:
        rows = {int(r["Step"]): r for r in csv.DictReader(handle)}
    frames = []
    for step in STEPS:
        obs = strip_autumn_obs_metadata(rows[step]["Observation"])
        frames.append(dict(step=step, action=rows[step]["Action"], observation=obs,
                           grid=json.loads(obs), observation_sha256=hashlib.sha256(obs.encode()).hexdigest()))
    assert [f["action"] for f in frames[:-1]] == ["down", "noop", "down"]
    snapshots = []
    for node in NODES:
        candidate, event = pool[node], events.get(node, {})
        outputs = []
        for frame in frames:
            output, error = run_perceive(candidate["perception"], frame["observation"])
            assert error is None and (output or node == 0)
            outputs.append(output)
        snapshots.append(dict(node=node, iteration=event.get("i"), parents=candidate["parents"],
                              accepted=event.get("accepted"), train_score=candidate["train_score"],
                              candidate_line=candidate["source_line"], process_line=event.get("source_line"),
                              perception=candidate["perception"], world_knowledge=candidate["world_knowledge"],
                              outputs=outputs))
    assert [s["iteration"] for s in snapshots] == [None, 4, 8]
    assert [s["parents"] for s in snapshots] == [[], [3], [6]]
    assert snapshots[0]["world_knowledge"] == ""
    assert snapshots[0]["outputs"] == [""]*4
    assert snapshots[1]["perception"] == pool[3]["perception"]
    assert snapshots[2]["perception"] == pool[6]["perception"]
    for snap in snapshots:
        chain = [snap["node"]]
        while pool[chain[-1]]["parents"]:
            chain.append(pool[chain[-1]]["parents"][0])
        snap["lineage"] = chain[::-1]
        verify_excerpt(snap["world_knowledge"], "\n".join(B_EXCERPTS[snap["node"]]))
    assert [snap["lineage"] for snap in snapshots] == [[0], [0, 3, 4], [0, 1, 6, 8]]
    # All occupied cells fit inside the fixed crop. No colored content is hidden.
    for frame in frames:
        grid = frame["grid"]
        assert len(grid) == len(grid[0]) == 16
        assert positions(grid, "red") == {(7, 7), (8, 7)}
        assert len(positions(grid, "blue")) == 2
        for color in ("red", "blue"):
            assert all(CROP["row_start"] <= r < CROP["row_stop"] and
                       CROP["col_start"] <= c < CROP["col_stop"] for r, c in positions(grid, color))
        assert {v for row in grid for v in row} == {"black", "red", "blue"}
    # Literal local B8 clauses, evaluated as visible cell sets (not identities).
    for frame, after in zip(frames, frames[1:]):
        blue = positions(frame["grid"], "blue")
        predicted = set()
        outside = {(r, c) for r, c in blue if r not in (7, 8)}
        for r, c in blue:
            if frame["action"] == "noop":
                predicted.add((r, c))
            elif r not in (7, 8) or any(cc == c for rr, cc in outside):
                predicted.add((r+1, c))
            else:
                predicted.add((9 if r == 7 else 8, c+1 if c < 7 else c-1))
        assert predicted == positions(after["grid"], "blue")
        # B4 already predicts this window using its coordinate-based pair rule.
        assert len({c for r, c in blue}) == 1
        column = next(iter(blue))[1]
        dc = 1 if column == 5 else -1 if column == 9 else 0
        predicted4 = blue if frame["action"] == "noop" else {(r+1, c+dc) for r, c in blue}
        assert predicted4 == positions(after["grid"], "blue")
    hashes = {digest(s["perception"], s["world_knowledge"]): s for s in snapshots}
    transition = digest(frames[0]["observation"], frames[1]["observation"], frames[0]["action"])
    cached = []
    for line, record in enumerate(map(json.loads, (RUN/"predictions.jsonl").read_text().splitlines()), 1):
        if record["cand_hash"] not in hashes or record["tr_hash"] != transition:
            continue
        snap = hashes[record["cand_hash"]]
        assert record["z_t"] == snap["outputs"][0] and record["z_t1"] == snap["outputs"][1]
        assert record["cfd_score"] == (0.0 if snap["node"] == 0 else 1.0)
        cached.append(dict(node=snap["node"], source_line=line, record=record))
    assert {c["node"] for c in cached} == set(NODES) and len(cached) == 3
    assert all(token in next(c for c in cached if c["node"] == 8)["record"]["cfd_response"]
               for token in ("rDist", "noRedRow"))
    return frames, snapshots, cached


def p_lines(snapshot, column):
    output = snapshot["outputs"][column]
    if snapshot["node"] == 0:
        assert output == ""
        return ['""']  # Display notation for the actual empty string.
    if snapshot["node"] == 4:
        header, body = output.split("cells: ")
        result = [header+"cells:"] + body.split(" ")
        assert len(result) == 5
    else:
        entries = output.split("; ")
        assert len(entries) == 4
        result = [entry+(";" if i < len(entries)-1 else "") for i, entry in enumerate(entries)]
    assert compact("\n".join(result)) == compact(output)
    return result


def draw(frames, snapshots):
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    height = 6.22
    fig = plt.figure(figsize=(W, height), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, W), ylim=(height, 0))
    ax.set_axis_off()
    checks = []

    def text(x, y, value, size=6, color=INK, mono=False, weight="normal", ha="left", maxw=None):
        artist = ax.text(x, y, value, fontsize=size, color=color, family=MONO if mono else SANS,
                         weight=weight, ha=ha, va="top", zorder=5)
        if maxw is not None:
            checks.append((artist, maxw))
        return artist

    def advance(value, size, mono=False, weight="normal"):
        prop = FontProperties(family=MONO if mono else SANS, size=size, weight=weight)
        width, _, _ = fig.canvas.get_renderer().get_text_width_height_descent(value, prop, ismath=False)
        return width/fig.dpi

    def panel(x, y, width, h, fill, edge=None):
        ax.add_patch(Rectangle((x, y), width, h, facecolor=fill, edgecolor=edge or fill, lw=.45, zorder=0))

    def rule(y):
        ax.plot([PAD, W-PAD], [y, y], color=RULE, lw=.6, zorder=1)

    def rich(x, y, value, size, mono=False, inherited=False, belief=False, maxw=None):
        start = x
        pattern = EMPHASIS if belief else RELATION
        parts = re.split("("+pattern+")", value)
        for part in parts:
            if not part:
                continue
            relation = re.fullmatch(RELATION, part) is not None
            emphasis = re.fullmatch(EMPHASIS, part) is not None
            color = BLUE if relation else GREEN if belief and emphasis else MUTED if mono else INK
            weight = "bold" if emphasis else "normal"
            width = advance(part, size, mono, weight)
            if relation:
                panel(x-.008, y-.009, width+.016, size/72+.033, BLUE_BG)
                if inherited:
                    color = "#4c779e"
            text(x, y, part, size, color, mono, weight)
            x += width
        if maxw is not None:
            assert x-start < maxw+.012, value

    text(PAD, .08, "Game sequence  →", 6.2, weight="bold")
    grid_top, cell = .36, .12
    grid_width, grid_height = 6*cell, 7*cell
    labels = ["Aligned with red rows", "Moves down + right", "No change", "Moves straight down"]
    for i, frame in enumerate(frames):
        cx = X[i]+CW/2
        gx = cx-grid_width/2
        text(cx, .225, f"X{SUBS[i]}  ·  frame {frame['step']}", 6.4, weight="bold", ha="center")
        for r in range(CROP["row_start"], CROP["row_stop"]):
            for c in range(CROP["col_start"], CROP["col_stop"]):
                ax.add_patch(Rectangle((gx+(c-CROP["col_start"])*cell, grid_top+(r-CROP["row_start"])*cell),
                                       cell, cell, facecolor=to_rgb(frame["grid"][r][c]),
                                       edgecolor="#343b40", lw=.25))
        ax.add_patch(Rectangle((gx, grid_top), grid_width, grid_height, fill=False, edgecolor=INK, lw=.55))
        for r in (5, 7, 8, 10, 11):
            text(gx-.035, grid_top+(r-5+.27)*cell, str(r), 4.7, color=MUTED, ha="right")
        for c in range(4, 10):
            text(gx+(c-4+.5)*cell, grid_top+grid_height+.025, str(c), 4.7, color=MUTED, ha="center")
        text(cx, 1.345, labels[i], 5.7, ha="center")
        if i < 3:
            left, right = gx+grid_width+.055, X[i+1]+CW/2-grid_width/2-.085
            ax.add_patch(FancyArrowPatch((left, .805), (right, .805), arrowstyle="-|>",
                                        mutation_scale=7, color=INK, lw=.8))
            text((left+right)/2, .655, frame["action"], 6.2, mono=True, ha="center")
    rule(1.5)
    text(PAD, 1.56, "Search checkpoints  ↓", 6.3, weight="bold")
    starts = (1.81, 2.36, 3.90)
    titles = ("P and B are empty", "P describes cells; B learns coordinate rules",
              "P adds relations; B uses those relations")
    for ri, (snap, y) in enumerate(zip(snapshots, starts)):
        node = snap["node"]
        color = MUTED if node == 0 else GREEN
        if ri:
            rule(y-.07)
        label = f"Iteration {0 if node == 0 else snap['iteration']}"
        text(PAD, y, label, 6.7, color=color, weight="bold")
        text(W-PAD, y, titles[ri], 6.1, color=color, ha="right")
        definition_height = .17 if node == 8 else 0
        if node == 8:
            rich(PAD, y+.155, "rDistN: distance to red in the same row.  noRedRow: no red in that row.",
                 5.7, maxw=W-2*PAD)
        py = y+definition_height
        body_count = max(len(p_lines(snap, i)) for i in range(4))
        for i in range(4):
            if node == 0:
                text(X[i], y+.18, f'P(X{SUBS[i]}) = ""', 6, color=MUTED)
            else:
                text(X[i], py+.18, f"P(X{SUBS[i]})", 5.8, color=MUTED, weight="bold")
                for j, line in enumerate(p_lines(snap, i)):
                    rich(X[i], py+.26+j*LH_P, line, FS_P, mono=True, maxw=CW)
                if i < 3:
                    ax.plot([X[i]+CW+GAP/2]*2, [py+.19, py+.25+body_count*LH_P], color=RULE, lw=.45)
        by = y+.32 if node == 0 else py+.79 if node == 4 else py+.705
        excerpt = B_EXCERPTS[node]
        if not excerpt:
            panel(PAD-.025, by-.012, W-2*PAD+.05, .17, "#f0f2f2")
            text(PAD+.035, by+.012, "World knowledge B = ∅"+("  ·  no learned rules" if node == 0 else "  ·  unchanged"), 6, color=MUTED)
        else:
            bh = .22+len(excerpt)*.112
            panel(PAD-.025, by-.02, W-2*PAD+.05, bh, GREEN_BG)
            title = "World knowledge B — newly learned coordinate rules (excerpts)" if node == 4 else "World knowledge B — rules using P’s labels (excerpts)"
            text(PAD+.035, by+.012, title, 6.3, color=GREEN, weight="bold")
            for j, line in enumerate(excerpt):
                rich(PAD+.07, by+.18+j*.112, line, 6.1, belief=True, maxw=W-2*PAD-.11)
    rule(6.055)
    text(PAD, 6.12, '"" denotes empty P; … marks omitted belief text.', 5.3, color=MUTED, maxw=W-2*PAD)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist, maxw in checks:
        assert artist.get_window_extent(renderer).width/fig.dpi <= maxw+.02, artist.get_text()
    for artist in ax.texts:
        bounds = artist.get_window_extent(renderer)
        assert bounds.x0 >= -.5 and bounds.x1 <= W*fig.dpi+.5, artist.get_text()
        assert bounds.y0 >= -.5 and bounds.y1 <= height*fig.dpi+.5, artist.get_text()
    for ext in ("pdf", "svg", "png"):
        fig.savefig(STEM.with_suffix("."+ext), dpi=300, facecolor=BG)
    plt.close(fig)
    return dict(width_inches=W, height_inches=height)


def save(frames, snapshots, cached, size):
    evidence = dict(game="7www9", drive="train_d1", steps=STEPS, nodes=NODES,
                    lineages={str(s["node"]): s["lineage"] for s in snapshots},
                    checkpoints_share_one_lineage=False, saved_node=14,
                    candidate_path=str((RUN/"candidates.jsonl").relative_to(ROOT)),
                    process_path=str((RUN/"process_log.jsonl").relative_to(ROOT)),
                    prediction_path=str((RUN/"predictions.jsonl").relative_to(ROOT)),
                    trajectory_path=str(TRAJECTORY.relative_to(ROOT)), crop=CROP, size=size,
                    method='Exact run_perceive on metadata-stripped observations; fresh namespace and single-frame history. Full P output retained, with "" denoting empty output. B excerpts are verified in source order with ellipses and whitespace/list-style reflow.',
                    frames=frames, snapshots=snapshots, cached_predictions=cached,
                    local_rule_verification="Both B4 and B8 match all three visible-state transitions. B8 checks concern cell sets, not identity-preserving motion or general magnet physics.",
                    limitations=["Nodes4 and8 belong to different branches; node8 is not a descendant or direct refinement of node4.",
                                 "B4 already predicts the illustrated diagonal transition correctly using coordinates.",
                                 "B8 does not preserve the true blue pair's cell identities in its explanation of the first down.",
                                 "Some movement rules are wrong; these checkpoints are explored candidates.",
                                 "Stored cached results are contrastive next-state selection for 32→33 only, not free-form simulation or full-window accuracy. These scores are omitted from the diagram."])
    lines = ["Magnets: exact P(X) and world knowledge for the diagram", "",
             "train_d1, steps 32–35; actions down → noop → down.",
             "Displayed iterations: 0 (initialization), 4, and 8; candidate nodes 0, 4, and 8.",
             "Different branches: 0 → 3 → 4 and 0 → 1 → 6 → 8; the figure does not depict a single lineage.",
             "The fixed crop is rows 5–11, columns 4–9 of the 16×16 grid; only black background is omitted.", ""]
    for snap in snapshots:
        snap["displayed_p"] = ["\n".join(p_lines(snap, i)) for i in range(4)]
        snap["displayed_b"] = B_EXCERPTS[snap["node"]]
        snap["empty_output_notation"] = '\"\"' if snap["node"] == 0 else None
        lines += ["="*75, f"Node {snap['node']}; iteration {snap['iteration']}; lineage {snap['lineage']}", ""]
        for i, out in enumerate(snap["outputs"]):
            lines += [f"P(X{i}), frame {STEPS[i]}:", out if out else '(empty string)', ""]
        lines += ["B (complete, verbatim):", snap["world_knowledge"] or "(empty)", ""]
        if snap["world_knowledge"]:
            lines += ["B (figure excerpts):", *snap["displayed_b"], ""]
    STEM.with_name(STEM.name+"_features.txt").write_text("\n".join(lines)+"\n")
    STEM.with_name(STEM.name+"_evidence.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False)+"\n")
    STEM.with_suffix(".tex").write_text(r"""% Generated by offline_learning/scripts/fig_learning_evolution_magnets.py.
\begin{figure}[p]
  \centering
  \includegraphics[width=\linewidth]{figures/learning_evolution_magnets_relations.pdf}
  \caption{Learning in Magnets. Top: recorded states
  (\texttt{train\_d1}, frames 32--35) and actions; the fixed crop
  (rows 5--11, columns 4--9) retains all colored cells.
  Below: actual $P(X)$ and excerpted beliefs $B$ ($K$ in the text) at
  iterations 0 (initialization), 4, and 8. Initially, both components are
  empty; iteration 4 has cell descriptions and coordinate rules;
  iteration 8 has same-row relations and rules using them.
  The checkpoints at iterations 4 and 8 come from different search
  branches; the latter is not a direct refinement of the former.
  Ellipses mark omitted belief text. Both learned beliefs explain these
  visible states, though the final belief's individual-cell motion attribution
  and other movement rules remain imperfect.}
  \label{fig:learning-evolution-magnets-relations}
\end{figure}
""")


def main():
    frames, snapshots, cached = load()
    size = draw(frames, snapshots)
    save(frames, snapshots, cached, size)
    print(f"Wrote {STEM.relative_to(ROOT)}.{{pdf,svg,png,tex}}")
    print("Verified 12 complete P(X) outputs, ordered B excerpts, candidate ancestry/iterations,")
    print("all 3 local transitions under both B4 and B8, and the 3 cached first-transition results.")
    print("Full outputs and provenance saved to _features.txt and _evidence.json.")


if __name__ == "__main__":
    main()
