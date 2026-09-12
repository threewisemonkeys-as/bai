#!/usr/bin/env python3
"""Draw Ice learning checkpoints with P(X) for every displayed game frame.

Run from the repository root:
    .venv/bin/python offline_learning/scripts/fig_learning_evolution_ice.py

Writes PDF, SVG, PNG, full-output TXT, evidence JSON, and a LaTeX figure snippet
under analysis/learning_example/learning_evolution_ice_identity*. Source observations, scores,
iterations, programs, and world knowledge are loaded from the original run.
Every displayed excerpt is checked against the corresponding actual output;
whitespace is reflowed and ellipses mark elisions. P is executed through the same
single-frame runner as training. No history or inferred state is supplied.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.textpath import TextPath

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from offline_learning.validate import run_perceive, strip_autumn_obs_metadata

GAME = 'bt3gb'
DRIVE = 'train_d0'
STEPS = (205, 206, 207, 208)
NODES = (3, 6, 8, 12)
SUBS = '₀₁₂₃'
RUN = ROOT / 'logs/2026-08-24/human_curated/rexpure/bt3gb_s1/rexpure_run_seed1'
TRAJECTORY = ROOT / 'offline_learning/human_data/bt3gb/informative_curated/drives/train_d0/episode_0/trajectory.csv'
STEM = ROOT / 'analysis/learning_example/learning_evolution_ice_identity'

# Match the existing paper palette, with green for the repaired assignment.
BG, INK, MUTED, RULE = '#fcfcfb', '#141718', '#626769', '#dce0e0'
BLUE, AMBER, GREEN = '#2a78d6', '#c45b18', '#148058'
BLUE_BG, AMBER_BG, GREEN_BG = '#edf4fc', '#fff0e4', '#e6f4ec'
SANS, MONO = 'DejaVu Sans', 'DejaVu Sans Mono'
W, H = 5.5, 8.0
PAD, GAP = .15, .13
CW = (W - 2 * PAD - 3 * GAP) / 4
X = [PAD + i * (CW + GAP) for i in range(4)]
FS_P, LH_P = 5.7, .092
CHECKS = []


def compact(s):
    return re.sub(r'\s+', '', s)


def verify_excerpt(full, excerpt):
    """All non-elided text must occur in order in the original string."""
    source = compact(full)
    cursor = 0
    for fragment in excerpt.split('…'):
        fragment = compact(fragment)
        if not fragment:
            continue
        pos = source.find(fragment, cursor)
        if pos < 0:
            raise AssertionError(f'Excerpt not in source: {fragment!r}')
        cursor = pos + len(fragment)


def load():
    csv.field_size_limit(10**7)
    candidates = {c['idx']: c for c in map(json.loads, (RUN / 'candidates.jsonl').read_text().splitlines())}
    process = list(map(json.loads, (RUN / 'process_log.jsonl').read_text().splitlines()))
    iterations = {r['new_idx']: r['i'] for r in process if r.get('new_idx') is not None}
    with TRAJECTORY.open() as stream:
        raw = {int(r['Step']): r for r in csv.DictReader(stream)}
    frames = []
    for step in STEPS:
        obs = strip_autumn_obs_metadata(raw[step]['Observation'])
        frames.append(dict(step=step, action=raw[step]['Action'], observation=obs, grid=json.loads(obs)))
    snapshots = []
    for n in NODES:
        c = candidates[n]
        outputs = []
        for frame in frames:
            output, error = run_perceive(c['perception'], frame['observation'])
            if error:
                raise RuntimeError(f'node {n}, step {frame["step"]}: {error}')
            outputs.append(output)
        snapshots.append(dict(node=n, iteration=iterations[n], score=c['train_score'],
                              parents=c['parents'], perception=c['perception'],
                              world_knowledge=c['world_knowledge'], outputs=outputs))
    assert snapshots[0]['outputs'] == snapshots[1]['outputs']
    assert snapshots[1]['world_knowledge'] == snapshots[2]['world_knowledge'] == snapshots[3]['world_knowledge']
    assert candidates[12]['perception'].strip() == (RUN.parent / 'best_perception_rexpure_seed1.py').read_text().strip()
    assert candidates[12]['world_knowledge'].strip() == (RUN.parent / 'best_beliefs_rexpure_seed1.txt').read_text().strip()
    lineage, c = [], candidates[12]
    while True:
        lineage.append(c['idx'])
        if not c['parents']:
            break
        c = candidates[c['parents'][0]]
    assert lineage[::-1] == [0, 3, 4, 6, 8, 12]
    return frames, snapshots


def p_lines(snapshot, col):
    """Reflow actual output fragments. Tuple: text, color, background, weight."""
    node = snapshot['node']
    body = []
    if node in (3, 6):
        body = [
            ('bg=black;', MUTED, None, 'normal'),
            ('gray(0,0),gray(0,1),…', INK, None, 'normal'),
            ('gray(0,4),gray(1,0),', INK, None, 'normal'),
            ('gray(1,1),', INK, None, 'normal'),
        ]
        # First two water entries, always verbatim and consecutive in the source.
        water = re.findall(r'lightblue\(\d+,\d+\)', snapshot['outputs'][col])[:2]
        body.extend((part + ',', INK, None, 'normal') for part in water)
        line, color, fill, weight = body[-1]
        body[-1] = (line + '…', color, fill, weight)
    elif node == 8:
        body = [
            ('… fixed:gray(0,0),…', BLUE, None, 'normal'),
            ('movable:gray_row0:', BLUE, None, 'normal'),
            ('0,1,2,3,4;', AMBER, AMBER_BG, 'bold'),
            ('other:gray_row0:2-4;', AMBER, AMBER_BG, 'normal'),
        ]
        water = re.findall(r'lightblue_row\d+:[\d,\-]+', snapshot['outputs'][col])[:2]
        body.extend((part + ';', INK, None, 'normal') for part in water)
        line, color, fill, weight = body[-1]
        body[-1] = (line + '…', color, fill, weight)
    else:
        body = [
            ('… fixed:gray(0,0),…', MUTED, None, 'normal'),
            ('movable:gray_row0:', MUTED, None, 'normal'),
            ('2-4;', GREEN, GREEN_BG, 'bold'),
        ]
        water = re.findall(r'lightblue_row\d+:[\d,\-]+', snapshot['outputs'][col])[:2]
        body.append(('other:' + water[0] + ';', INK, None, 'normal'))
        body.append((water[1] + ';', INK, None, 'normal'))
        line, color, fill, weight = body[-1]
        body[-1] = (line + '…', color, fill, weight)
    if node == 6:
        body = [(s, MUTED, bg, weight) for s, color, bg, weight in body]
    verify_excerpt(snapshot['outputs'][col], '\n'.join(s for s, *_ in body))
    return body


# These excerpts follow the order in the raw B; no summary is presented as a quote.
B_LINES = [
    '… fixed 2x2 block at cells (0,0),(0,1),(1,0),(1,1). …',
    '… a movable block: three contiguous gray cells on row 0. …',
    'On noop: every blue/lightblue cell not on row 15 attemps to move down one row. …',
    'On down: if the cell (1, c) is empty … where c is the middle column of the movable block …',
    'a new blue/lightblue cell appears at (1, c) …',
]


B_REPEAT = [
    '… a movable block: three contiguous gray cells on row 0. …',
    'On down: if the cell (1, c) is empty … where c is the middle column of the movable block …',
    'a new blue/lightblue cell appears at (1, c) …',
]


def width_in(s, size, mono=False, weight='normal'):
    prop = FontProperties(family=MONO if mono else SANS, size=size, weight=weight)
    return TextPath((0, 0), s, prop=prop).get_extents().width / 72 if s.strip() else 0


def draw(frames, snapshots):
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(W, H), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')
    ax.set_facecolor(BG)

    def text(x, y, s, size=6, color=INK, mono=False, weight='normal', ha='left', maxw=None):
        artist = ax.text(x, y, s, fontsize=size, color=color, family=MONO if mono else SANS,
                         weight=weight, ha=ha, va='top', zorder=5)
        if maxw is not None:
            CHECKS.append((artist, x, y, maxw))
        return artist

    def rule(y, color=RULE, lw=.6):
        ax.plot([PAD, W-PAD], [y, y], color=color, lw=lw, zorder=1)

    def panel(x, y, width, height, fill, edge=None):
        ax.add_patch(Rectangle((x, y), width, height, facecolor=fill,
                               edgecolor=edge or fill, linewidth=.5, zorder=0))

    text(PAD, .10, 'Learning to distinguish touching objects in Ice', 9.2, weight='bold', maxw=W-2*PAD)
    text(PAD, .29, 'Four game frames, evaluated by the same learned P and B at each checkpoint.', 6.1, color=MUTED)
    text(PAD, .43, 'Gameplay  →', 6.2, color=MUTED, weight='bold')
    grid_top, grid_size = .65, .78
    note = ['Touching gray objects', 'A drop appears at (1,3)', 'The drop falls to (2,3)', 'Another drop at (1,3)']
    for i, frame in enumerate(frames):
        cx = X[i] + CW/2
        text(cx, .53, f'X{SUBS[i]}  ·  step {frame["step"]}', 6.4, weight='bold', ha='center')
        gx = cx - grid_size/2
        cell = grid_size/16
        for r, row in enumerate(frame['grid']):
            for c, color in enumerate(row):
                ax.add_patch(Rectangle((gx+c*cell, grid_top+r*cell), cell, cell,
                                       facecolor=to_rgb(color), edgecolor='#35393b', linewidth=.13))
        ax.add_patch(Rectangle((gx, grid_top), grid_size, grid_size, fill=False, edgecolor=INK, lw=.5))
        # Pale outline around the relevant region, not an inferred object boundary.
        ax.add_patch(Rectangle((gx, grid_top), 6*cell, 4*cell, fill=False, edgecolor='#ffffff', lw=.75))
        for r in (0, 3, 15):
            text(gx-.025, grid_top+(r+.2)*cell, str(r), 4.3, color=MUTED, ha='right')
        for c in (0, 3, 15):
            text(gx+(c+.5)*cell, grid_top+grid_size+.018, str(c), 4.3, color=MUTED, ha='center')
        text(cx, 1.54, note[i], 5.5, color=INK, ha='center')
        if i < 3:
            start, end = gx+grid_size+.028, X[i+1]+CW/2-grid_size/2-.028
            mid = (start+end)/2
            ax.add_patch(FancyArrowPatch((start,1.05),(end,1.05),arrowstyle='-|>',mutation_scale=7,lw=.7,color=MUTED))
            text(mid, .90, frame['action'], 5.5, mono=True, ha='center')
    rule(1.66, MUTED)
    text(PAD, 1.72, 'Learning  ↓', 6.3, color=MUTED, weight='bold')
    text(W-PAD, 1.72, 'blue: added   ·   orange: error   ·   green: repair   ·   gray: unchanged', 5.5, color=MUTED, ha='right')
    y = 1.91
    titles = [
        'P lists colored cells',
        'B names objects and their dynamics',
        'P introduces object roles, but merges the gray objects',
        'P separates the fixed object from the three-cell cloud',
    ]
    # Inherited B repeats just the rule needed to interpret the P repair.
    for ri, snapshot in enumerate(snapshots):
        node, iteration = snapshot['node'], snapshot['iteration']
        if ri:
            rule(y-.055)
        title_color = GREEN if node == 12 else INK
        text(PAD, y, f'Iteration {iteration}  ·  node {node}', 6.7, color=title_color, weight='bold')
        text(W-PAD, y, f'min(ID, cFD) = {snapshot["score"]:.3f}', 6.1, color=title_color, ha='right')
        text(PAD, y+.145, titles[ri], 6, color=MUTED if node in (3,6) else (AMBER if node==8 else GREEN))
        py = y+.28
        body_count = max(len(p_lines(snapshot, i)) for i in range(4))
        for i in range(4):
            text(X[i], py, f'P(X{SUBS[i]})', 5.7, color=MUTED, weight='bold')
            if node == 6:
                text(X[i]+CW, py, 'unchanged', 4.6, color=MUTED, ha='right')
            body_top = py+.11
            for j, (line, color, fill, weight) in enumerate(p_lines(snapshot,i)):
                ly = body_top+j*LH_P
                if fill:
                    panel(X[i]-.018, ly-.007, min(CW+.03,width_in(line,FS_P,True,weight)+.08),LH_P+.012,fill)
                text(X[i], ly, line, FS_P, color=color, mono=True, weight=weight, maxw=CW)
            if i<3:
                ax.plot([X[i]+CW+GAP/2]*2,[py,py+.11+body_count*LH_P],color=RULE,lw=.4,zorder=1)
        body_end = py + .11 + body_count * LH_P
        if node == 3:
            text(W-PAD,y+.145,'B = ∅  (no world knowledge)',5.7,color=MUTED,ha='right')
            y = body_end + .09
        else:
            by = body_end + .035
            bcolor = BLUE if node==6 else MUTED
            excerpt = B_LINES if node==6 else B_REPEAT
            bh = .16 + len(excerpt)*.094
            panel(PAD-.035,by-.025,W-2*PAD+.07,bh,BLUE_BG if node==6 else '#f2f4f3')
            text(PAD,by,'B — updated since iteration 3' if node==6 else 'B — unchanged from iteration 8 (shortened excerpt)',5.8,color=bcolor,weight='bold')
            for j,line in enumerate(excerpt):
                text(PAD+.04,by+.13+j*.094,line,5.65,color=INK if node==6 else MUTED,maxw=W-2*PAD-.04)
            y = by + bh + .06
    # Footer generated after the rows; its location determines the final canvas height.
    footer_y = y+.005
    rule(footer_y)
    text(PAD,footer_y+.065,'… marks omitted text; whitespace is reflowed. P(X) values and B excerpts are taken from the run.',5.3,color=MUTED)
    text(PAD,footer_y+.17,'BT3GB · train_d0 · lineage 0 → 3 → 4 → 6 → 8 → 12; selected checkpoints shown; node 12 is saved best.',5.2,color=MUTED)
    final_h = footer_y+.26
    assert final_h <= 7.8, 'Keep room for a caption on the 9-inch paper text area.'
    # Preserve authored font sizes and inch geometry when expanding the canvas.
    fig.set_size_inches(W, final_h)
    ax.set_ylim(final_h,0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist,x,y,maxw in CHECKS:
        width = artist.get_window_extent(renderer).width / fig.dpi
        if width > maxw+.025:
            raise AssertionError(f'Text overflows by {width-maxw:.3f} in: {artist.get_text()}')
    for artist in ax.texts:
        bounds = artist.get_window_extent(renderer)
        assert bounds.x0 >= -.5 and bounds.x1 <= W*fig.dpi+.5, artist.get_text()
        assert bounds.y0 >= -.5 and bounds.y1 <= final_h*fig.dpi+.5, artist.get_text()
    for ext in ('pdf','svg','png'):
        fig.savefig(STEM.with_suffix('.'+ext),dpi=300,facecolor=BG)
    plt.close(fig)
    return final_h


def save_evidence(frames,snapshots):
    meta = dict(game=GAME,drive=DRIVE,steps=STEPS,nodes=NODES,
                candidate_path=str((RUN/'candidates.jsonl').relative_to(ROOT)),
                trajectory_path=str(TRAJECTORY.relative_to(ROOT)),
                method='run_perceive(code, strip_autumn_obs_metadata(obs)); fresh single-frame namespace',
                frames=frames,snapshots=snapshots)
    lines = ['Ice: every P(X) and B in learning_evolution_ice_identity.pdf',
             'Frames: train_d0/episode_0/trajectory.csv, steps 205–208.',
             'Actions: down → noop → down.',
             'Iteration is the learning proposal index; node is its candidate-pool ID.',
             'Perception is evaluated on one frame at a time, exactly as during training.',
             'Figure quotes reflow whitespace and use … for omitted source text.', '']
    for snap in snapshots:
        snap['perception_sha256'] = hashlib.sha256(snap['perception'].encode()).hexdigest()
        snap['displayed_p'] = ['\n'.join(l for l,*_ in p_lines(snap,i)) for i in range(4)]
        snap['displayed_b'] = (B_LINES if snap['node']==6 else B_REPEAT) if snap['world_knowledge'] else []
        lines.extend(['='*80,f'Iteration {snap["iteration"]}; node {snap["node"]}; score {snap["score"]:.9f}', ''])
        for i,out in enumerate(snap['outputs']):
            lines.extend([f'P(X{i}), game step {STEPS[i]}:',out,''])
        lines.extend(['B / world_knowledge (complete, verbatim):',snap['world_knowledge'] or '(empty)', ''])
    STEM.with_name(STEM.name+'_features.txt').write_text('\n'.join(lines)+'\n')
    STEM.with_name(STEM.name+'_evidence.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n')
    STEM.with_suffix('.tex').write_text(r'''% Generated by offline_learning/scripts/fig_learning_evolution_ice.py.
% The figure is intentionally not inserted into main.tex automatically.
\begin{figure}[p]
  \centering
  \includegraphics[width=\linewidth]{figures/learning_evolution_ice_identity.pdf}
  \caption{Learning object identity in Ice. Four game frames show water
  spawning beneath the cloud and falling on \texttt{noop}. Each checkpoint
  displays its $P(X)$ for every frame and world knowledge $B$ ($K$ in the text).
  Orange marks the five-cell misassignment; green marks the corrected
  three-cell cloud. World knowledge is unchanged after iteration 8.
  White outlines mark the relevant image region. Ellipses denote omissions;
  excerpts are learned claims. Scores measure the whole training objective.}
  \label{fig:learning-evolution-ice-identity}
\end{figure}
''')


def main():
    CHECKS.clear()
    frames,snapshots = load()
    for snap in snapshots:
        if snap['world_knowledge']:
            verify_excerpt(snap['world_knowledge'],'\n'.join(B_LINES))
            verify_excerpt(snap['world_knowledge'],'\n'.join(B_REPEAT))
        for i in range(4):
            p_lines(snap,i)
    height = draw(frames,snapshots)
    save_evidence(frames,snapshots)
    print(f'Wrote {STEM.relative_to(ROOT)}.{{pdf,svg,png,tex}} ({W:.2f} × {height:.2f} in).')
    print('Verified all 16 P(X) outputs, displayed source excerpts, score/iteration metadata, and saved-best lineage.')
    print('Full P(X), B, and provenance saved to _features.txt and _evidence.json.')


if __name__ == '__main__':
    main()
