#!/usr/bin/env python3
"""Draw Grow learning checkpoints with P(X) for every displayed game frame.

Run from the repository root:
    .venv/bin/python offline_learning/scripts/fig_learning_evolution_grow.py

Writes PDF, SVG, PNG, full-output TXT, evidence JSON, and a LaTeX figure snippet
under analysis/learning_example/learning_evolution_grow_relations*. Source observations, scores,
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

GAME = '7xf97'
DRIVE = 'train_d0'
STEPS = (34, 35, 36, 37)
NODES = (2, 10, 11, 23)
SUBS = '₀₁₂₃'
RUN = ROOT / 'logs/2026-08-24/human_curated/rexpure/7xf97_s1/rexpure_run_seed1'
TRAJECTORY = ROOT / 'offline_learning/human_data/7xf97/informative_curated/drives/train_d0/episode_0/trajectory.csv'
STEM = ROOT / 'analysis/learning_example/learning_evolution_grow_relations'

# Match the existing paper palette, with green for detected contact.
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
    return re.sub(r'\s+', '', s.replace('**', '').replace('`', ''))


def verify_excerpt(full, excerpt):
    """Non-elided text occurs in order; whitespace and Markdown styling are normalized."""
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
    assert candidates[23]['perception'].strip() == (RUN.parent / 'best_perception_rexpure_seed1.py').read_text().strip()
    assert candidates[23]['world_knowledge'].strip() == (RUN.parent / 'best_beliefs_rexpure_seed1.txt').read_text().strip()
    lineage, c = [], candidates[23]
    while True:
        lineage.append(c['idx'])
        if not c['parents']:
            break
        c = candidates[c['parents'][0]]
    assert lineage[::-1] == [0, 2, 3, 7, 10, 11, 14, 19, 23]
    assert all(frame['action'] == 'left' for frame in frames[:3])
    for i, frame in enumerate(frames):
        gold = [(r, c) for r, row in enumerate(frame['grid']) for c, color in enumerate(row) if color=='gold']
        gray = [(r, c) for r, row in enumerate(frame['grid']) for c, color in enumerate(row) if color=='gray']
        contact = max(c for _, c in gold) + 1 == min(c for _, c in gray)
        assert contact == (i >= 2)
        assert f'gare={int(contact)}' in snapshots[-1]['outputs'][i]
    assert 'gold_right + 1 == gray_left' in candidates[23]['perception']
    assert snapshots[2]['score'] < snapshots[1]['score']
    return frames, snapshots


def p_lines(snapshot, col):
    """Quote actual P output, highlighting object grouping and adjacency."""
    node = snapshot['node']
    output = snapshot['outputs'][col]
    if node in (2, 10):
        # Keep all row-0 cells, then examples from the bottom of each block.
        row0 = [part for part in output.split('|') if part.startswith('0,')]
        body = []
        for start in range(0, len(row0), 2):
            part = '|'.join(row0[start:start+2]) + '|'
            if start + 2 >= len(row0):
                part += '…'
            body.append((part, INK, None, 'normal'))
        for color in ('gold', 'gray'):
            part = next(part for part in output.split('|')
                        if part.startswith('2,') and part.endswith(':'+color))
            body.append((part+'|…', INK, None, 'normal'))
        if node == 10:
            body = [(line, MUTED, fill, weight) for line, _, fill, weight in body]
    elif node == 11:
        gold = re.search(r'gold:[^|]+', output).group()
        gray = re.search(r'gray:[^|]+', output).group()
        body = [
            ('rows=16,cols=16,', MUTED, None, 'normal'),
            ('bg=black;', MUTED, None, 'normal'),
            (gold+'|', BLUE, BLUE_BG, 'bold'),
            (gray+'|', BLUE, BLUE_BG, 'bold'),
            ('green:15,1|…', INK, None, 'normal'),
        ]
    else:
        flag = re.search(r'gare=([01])', output).group()
        contact = flag.endswith('1')
        gold = re.search(r'gold:[^;|]+', output).group()
        gray = re.search(r'gray:[^;|]+', output).group()
        body = [
            ('… f:gx=1,ax=1,gle=1,', BLUE, None, 'normal'),
            ('… '+flag+', …', GREEN if contact else BLUE,
             GREEN_BG if contact else BLUE_BG, 'bold'),
            ('|'+gold+';…', INK, None, 'normal'),
            ('|'+gray+';…', INK, None, 'normal'),
            ('|green:15,1;…', INK, None, 'normal'),
        ]
    verify_excerpt(output, '\n'.join(line for line, *_ in body))
    return body


# Belief excerpts are in source order, including the explicitly labelled right rule.
# That rule is a learned hypothesis about relative position, not the action shown above.
B_LINES = [
    'Gold and gray cells appear only in rows 0–2. They form two contiguous blocks …',
    '… actions can bring them together or change their widths and positions. …',
    '… left: … If a gray block exists, it shifts left by one column …',
    '… right: … The column vacated on the left becomes empty unless it is immediately right of a gold block, …',
    'in which case that column becomes gold (the gold block expands rightwards into it). …',
]

B_REPEAT = [
    '… left: … If a gray block exists, it shifts left by one column …',
    '… right: … The column vacated on the left becomes empty unless it is immediately right of a gold block, …',
    'in which case that column becomes gold …',
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

    text(PAD, .08, 'Gameplay  →', 6.2, color=MUTED, weight='bold')
    grid_top, grid_size = .30, .78
    note = ['Two empty columns', 'One empty column', 'The blocks touch', 'Gray covers gold’s edge']
    for i, frame in enumerate(frames):
        cx = X[i] + CW/2
        text(cx, .18, f'X{SUBS[i]}  ·  step {frame["step"]}', 6.4, weight='bold', ha='center')
        gx = cx - grid_size/2
        cell = grid_size/16
        for r, row in enumerate(frame['grid']):
            for c, color in enumerate(row):
                ax.add_patch(Rectangle((gx+c*cell, grid_top+r*cell), cell, cell,
                                       facecolor=to_rgb(color), edgecolor='#35393b', linewidth=.13))
        ax.add_patch(Rectangle((gx, grid_top), grid_size, grid_size, fill=False, edgecolor=INK, lw=.5))
        for r in (0, 2, 15):
            text(gx-.025, grid_top+(r+.2)*cell, str(r), 4.3, color=MUTED, ha='right')
        for c in (0, 3, 8, 15):
            text(gx+(c+.5)*cell, grid_top+grid_size+.018, str(c), 4.3, color=MUTED, ha='center')
        text(cx, 1.19, note[i], 5.5, color=INK, ha='center')
        if i < 3:
            start, end = gx+grid_size+.028, X[i+1]+CW/2-grid_size/2-.028
            mid = (start+end)/2
            ax.add_patch(FancyArrowPatch((start,.70),(end,.70),arrowstyle='-|>',mutation_scale=7,lw=.7,color=MUTED))
            text(mid, .55, frame['action'], 5.5, mono=True, ha='center')
    rule(1.31, MUTED)
    text(PAD, 1.37, 'Learning  ↓', 6.3, color=MUTED, weight='bold')
    y = 1.56
    titles = [
        'P lists colored cells',
        'B describes blocks and rules that depend on their relative position',
        'P groups connected cells into rectangles (rows, columns)',
        'P adds spatial predicates; gare = 1 when gold is immediately left of gray',
    ]
    # Inherited B repeats just the rule needed to interpret the new spatial features.
    for ri, snapshot in enumerate(snapshots):
        node, iteration = snapshot['node'], snapshot['iteration']
        if ri:
            rule(y-.055)
        title_color = GREEN if node == 23 else INK
        text(PAD, y, f'Iteration {iteration}', 6.7, color=title_color, weight='bold')
        text(W-PAD, y, f'min(ID, cFD) = {snapshot["score"]:.3f}', 6.1, color=title_color, ha='right')
        text(PAD, y+.145, titles[ri], 6, color=MUTED if node in (2,10) else (BLUE if node==11 else GREEN))
        py = y+.28
        body_count = max(len(p_lines(snapshot, i)) for i in range(4))
        for i in range(4):
            text(X[i], py, f'P(X{SUBS[i]})', 5.7, color=MUTED, weight='bold')
            if node == 10:
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
        if node == 2:
            text(W-PAD,y+.145,'B = ∅  (no world knowledge)',5.7,color=MUTED,ha='right')
            y = body_end + .09
        else:
            by = body_end + .035
            bcolor = BLUE if node==10 else MUTED
            excerpt = B_LINES if node==10 else B_REPEAT
            bh = .16 + len(excerpt)*.094
            panel(PAD-.035,by-.025,W-2*PAD+.07,bh,BLUE_BG if node==10 else '#f2f4f3')
            text(PAD,by,'B — updated since iteration 3' if node==10 else 'B — unchanged from iteration 12 (shortened excerpt)',5.8,color=bcolor,weight='bold')
            for j,line in enumerate(excerpt):
                text(PAD+.04,by+.13+j*.094,line,5.65,color=INK if node==10 else MUTED,maxw=W-2*PAD-.04)
            y = by + bh + .06
    # Footer generated after the rows; its location determines the final canvas height.
    footer_y = y+.005
    rule(footer_y)
    text(PAD,footer_y+.065,'… marks omitted text; whitespace is reflowed. P(X) values and B excerpts are taken from the run.',5.3,color=MUTED)
    final_h = footer_y+.16
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
                lineage=[0,2,3,7,10,11,14,19,23],
                notes=['Node 11 rectangles are replaced by coordinate ranges at omitted node 19.',
                       'Node 23 adds spatial flags; uc is inactive under single-frame evaluation.',
                       'gare uses visible extent and is not a latent overlap/occlusion detector.',
                       'Knowledge excerpts are hypotheses; the right-action rule is labelled explicitly.'],
                frames=frames,snapshots=snapshots)
    lines = ['Grow: every P(X) and B in learning_evolution_grow_relations.pdf',
             'Frames: train_d0/episode_0/trajectory.csv, steps 34–37.',
             'Actions: left → left → left.',
             'Iteration is the learning proposal index; node is its candidate-pool ID.',
             'Perception is evaluated on one frame at a time, exactly as during training.',
             'Figure quotes reflow whitespace/Markdown styling and use … for omitted source text.',
             'The excerpt labelled right is a belief about right actions, not the left actions pictured.',
             'The frames show visible occlusion, not physical destruction of the gold object.',
             'Omitted node 19 returns from rectangles to coordinate ranges before node 23 adds flags.', '']
    for snap in snapshots:
        snap['perception_sha256'] = hashlib.sha256(snap['perception'].encode()).hexdigest()
        snap['displayed_p'] = ['\n'.join(l for l,*_ in p_lines(snap,i)) for i in range(4)]
        snap['displayed_b'] = (B_LINES if snap['node']==10 else B_REPEAT) if snap['world_knowledge'] else []
        lines.extend(['='*80,f'Iteration {snap["iteration"]}; node {snap["node"]}; score {snap["score"]:.9f}', ''])
        for i,out in enumerate(snap['outputs']):
            lines.extend([f'P(X{i}), game step {STEPS[i]}:',out,''])
        lines.extend(['B / world_knowledge (complete, verbatim):',snap['world_knowledge'] or '(empty)', ''])
    STEM.with_name(STEM.name+'_features.txt').write_text('\n'.join(lines)+'\n')
    STEM.with_name(STEM.name+'_evidence.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n')
    STEM.with_suffix('.tex').write_text(r'''% Generated by offline_learning/scripts/fig_learning_evolution_grow.py.
% The figure is intentionally not inserted into main.tex automatically.
\begin{figure}[p]
  \centering
  \includegraphics[width=\linewidth]{figures/learning_evolution_grow_relations.pdf}
  \caption{Learning spatial relations in Grow (7XF97). Three left actions
  close the gap and then cover the gold block's edge. Each checkpoint shows
  its $P(X)$ for all four frames and world knowledge $B$ ($K$ in the text).
  Perception evolves from cells to rectangles, then adds spatial predicates.
  The new \texttt{gare} flag detects visible gold immediately left of gray:
  $0,0,1,1$. Knowledge is unchanged after iteration 12. Ellipses mark omissions;
  knowledge excerpts are learned hypotheses. Scores measure the whole training
  objective, including the temporary decrease at iteration 13.}
  \label{fig:learning-evolution-grow-relations}
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
