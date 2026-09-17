#!/usr/bin/env python3
"""Three game-example presentations of the recorded planning evaluation.

Adds a filmstrip, per-round decision cards, and an aligned planning-horizon
table to the existing native Figma file. Existing figure options are retained.
"""
from __future__ import annotations

import argparse
import collections
import copy
import datetime
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from offline_learning.scripts.fig_planning_evaluation_variants import (
    DiagramBuilder, load_example, SOURCE, IO, BG, BLUE, BLUE_BG, GREEN,
    GREEN_BG, INK, MUTED, RULE, WHITE, LIGHT, GRAY, gid, matrix, render,
)
from matplotlib.transforms import Affine2D
from PIL import Image, ImageDraw, ImageFont

OUT = SOURCE.parent / 'planning_game_examples'
PREFIX = 'planning_game_example_'
VARIANTS = ['filmstrip', 'decision_cards', 'horizon_table']
TITLES = ['Filmstrip with plans', 'One decision per card', 'Plans aligned by future step']
W, H = 1480, 812
SUBS = ['₀', '₁', '₂', '₃']


class ExampleBuilder(DiagramBuilder):
    def __init__(self, source, font_dir):
        super().__init__(source, font_dir)
        self.plan_records, self.outcome_records = [], []

    def action_chip(self, parent, action, x, y, mode='future', w=78, h=42):
        fill, stroke, color = {
            'now': (BLUE, BLUE, WHITE),
            'future': (WHITE, GRAY, MUTED),
            'past': (LIGHT, LIGHT, MUTED),
        }[mode]
        n = self.block(parent, f'Action / {mode} / {action}', x, y, w, h, fill, stroke)
        self.label(n, action, w / 2, 8, 22 if mode != 'past' else 18,
                   color, mode == 'now', align='center', maxw=w - 8)
        return n

    def recorded_plan(self, parent, example, i, x, y, spacing=87):
        actions = example['rounds'][i]['plan']
        nodes = [self.action_chip(parent, a, x + j * spacing, y,
                                 'now' if j == 0 else 'future')
                 for j, a in enumerate(actions)]
        self.plan_records.append(dict(parent=gid(parent['guid']), round=i,
                                      actions=actions, executed=actions[0],
                                      nodes=[gid(n['guid']) for n in nodes]))
        return nodes

    def outcome(self, parent, example, i, x, y, w=250, h=46, prefix='Goal? '):
        reached = example['rounds'][i]['reached_goal']
        fill, color = (GREEN_BG, GREEN) if reached else (LIGHT, INK)
        n = self.block(parent, f'Goal check after action {i + 1}', x, y, w, h, fill,
                       GREEN if reached else RULE)
        label = prefix + ('Yes · stop' if reached else 'No · replan')
        self.label(n, label, w / 2, 10, 22, color, reached, align='center', maxw=w - 12)
        self.outcome_records.append(dict(node=gid(n['guid']), round=i,
                                         state=i + 1, reached=reached))
        return n

    def footer(self, root, first_line):
        self.label(root, first_line, 30, 751, 21, maxw=1420)
        self.label(root, 'Recorded rollout · cropped game views. Exact full-state goal checks after every action; stop at success or the task step limit.',
                   30, 784, 18, MUTED, maxw=1420)


def filmstrip(b, page, x, y, ex):
    root = b.frame(page, PREFIX + VARIANTS[0], x, y, W, H, BG)
    b.heading(root, 'Filmstrip with plans')
    b.label(root, 'Task: move the blue magnet three cells left', 30, 108, 27, bold=True)
    b.label(root, 'Learned perception P + dynamics K', 30, 151, 24, BLUE, True)
    b.label(root, 'Each plan uses P(current state), P(target state), and K.', 30, 186, 21, BLUE)
    b.label(root, 'Target X{G}', 1080, 116, 26, bold=True, align='center')
    b.label(root, 'Fixed goal', 1080, 157, 22, MUTED, align='center')
    b.grid(root, ex['goal'], 1215, 111, 20, False, 'Given target state')
    b.line(root, 'Task / rollout divider', (30, 234), (1445, 234), RULE, 1.2)

    # Plans sit above the real trajectory. Only the first token feeds a transition.
    for i in range(3):
        xx = 60 + 350 * i
        b.label(root, f'Plan at t = {i}', xx + 126, 266, 24, BLUE, True, align='center')
        b.recorded_plan(root, ex, i, xx, 307)
        b.route(root, f'Round {i} / first proposed action feeds real transition',
                [(xx + 39, 349), (xx + 39, 391), (xx + 305, 391), (xx + 305, 548)], BLUE, 2.1)
        b.label(root, 'execute first', xx + 305, 361, 19, BLUE, align='center')

    b.label(root, 'Observed game states', 30, 409, 19, MUTED)
    for i, state in enumerate(ex['states']):
        xx = 60 + 350 * i
        label = f'Start X{SUBS[i]}' if i == 0 else f'Observe X{SUBS[i]}'
        b.label(root, label, xx + 130, 444, 24, INK, True, align='center')
        b.grid(root, state, xx, 483, 26, True, f'Real state X{SUBS[i]}', row_axes=False)
        if i < 3:
            b.route(root, f'Executed action {i + 1} / left', [(xx + 260, 548), (xx + 350, 548)], BLUE, 2.4)
            b.label(root, 'left', xx + 305, 552, 20, BLUE, True, align='center')
        if i:
            b.outcome(root, ex, i - 1, xx + 5, 651, 250, 47)
        else:
            b.label(root, 'Initial observation', xx + 130, 664, 22, MUTED, align='center')
    b.footer(root, 'Blue: execute now. Outlined actions: proposed remainder, reconsidered after the next observation.')
    return root


def decision_cards(b, page, x, y, ex):
    root = b.frame(page, PREFIX + VARIANTS[1], x, y, W, H, BG)
    b.heading(root, 'One decision per card')
    b.grid(root, ex['goal'], 30, 108, 20, False, 'Given target state')
    b.label(root, 'Goal: blue magnet at column 1', 257, 109, 26, bold=True)
    b.label(root, 'Shown to the planner as P(X{G})', 257, 151, 22, MUTED)
    b.label(root, 'Checked against each resulting raw state', 257, 185, 21, MUTED)
    fixed = b.block(root, 'Fixed learned model', 958, 108, 487, 100, BLUE_BG, BLUE)
    b.label(fixed, 'Same P and K in every round', 24, 16, 25, BLUE, True, maxw=440)
    b.label(fixed, 'Perceive → plan → act → check', 24, 58, 21, BLUE, maxw=440)

    for i in range(3):
        xx = 30 + 495 * i
        card = b.block(root, f'Planning round {i + 1}', xx, 225, 430, 516, WHITE, RULE)
        b.label(card, f'Round {i + 1}', 22, 14, 26, INK, True)
        b.label(card, f't = {i}', 408, 18, 21, MUTED, align='right')
        b.label(card, f'Current state X{SUBS[i]}', 215, 54, 21, align='center')
        b.grid(card, ex['states'][i], 115, 85, 20, False, f'Current state X{SUBS[i]}')
        b.label(card, f'P(X{SUBS[i]}) → LLM + K + goal', 215, 196, 21, BLUE, align='center')
        b.label(card, 'Plan', 22, 238, 20, MUTED)
        b.recorded_plan(card, ex, i, 93, 230)
        b.route(card, f'Round {i} / execute first action only',
                [(132, 272), (132, 291), (65, 291), (65, 389), (115, 389)], BLUE, 2.2)
        b.label(card, 'Execute first: left', 244, 286, 22, BLUE, True, align='center')
        b.label(card, f'Observe X{SUBS[i+1]}', 215, 319, 20, align='center')
        b.grid(card, ex['states'][i+1], 115, 349, 20, False, f'Resulting state X{SUBS[i+1]}')
        b.outcome(card, ex, i, 84, 460, 262, 43)
    for i in range(2):
        xx = 30 + 495 * i
        b.route(root, f'Failed goal check / replan from next observed state',
                [(xx + 346, 706.5), (xx + 462.5, 706.5),
                 (xx + 462.5, 360), (xx + 610, 360)], BLUE, 2.1)
        b.label(root, 'replan', xx + 397, 675, 18, BLUE, align='center')
    b.footer(root, 'Each card executes one action. A fresh observation starts the next round; a satisfied goal ends the attempt.')
    return root


def horizon_table(b, page, x, y, ex):
    root = b.frame(page, PREFIX + VARIANTS[2], x, y, W, H, BG)
    b.heading(root, 'Plans aligned by future step')
    b.label(root, 'Given task', 30, 117, 26, bold=True)
    b.label(root, 'Start X₀', 288, 104, 22, bold=True, align='center')
    b.grid(root, ex['start'], 213, 140, 15, False, 'Given start state')
    b.label(root, 'Target X{G}', 640, 104, 22, bold=True, align='center')
    b.grid(root, ex['goal'], 565, 140, 15, False, 'Given target state')
    b.route(root, 'Task / reach target state', [(391, 177), (537, 177)], MUTED, 1.8)
    b.label(root, 'reach', 464, 144, 20, MUTED, align='center')
    fixed = b.block(root, 'Learned model inputs at every round', 810, 108, 635, 106, BLUE_BG, BLUE)
    b.label(fixed, 'Planning inputs: P(Xₜ), P(X{G}), K', 24, 18, 26, BLUE, True, maxw=590)
    b.label(fixed, 'Observe again, then confirm or revise the plan.', 24, 61, 22, BLUE, maxw=590)

    b.label(root, 'Round', 80, 263, 21, MUTED, align='center')
    b.label(root, 'Current state', 260, 263, 22, INK, True, align='center')
    b.label(root, 'Plan at this round', 598, 235, 22, BLUE, True, align='center')
    for j in range(3):
        b.label(root, f'Step {j+1}', 488 + 110 * j, 269, 19, MUTED, align='center')
    b.label(root, 'Execute first', 855, 263, 21, BLUE, True, align='center')
    b.label(root, 'Next observation', 1040, 263, 21, INK, True, align='center')
    b.label(root, 'Goal check', 1303, 263, 22, INK, True, align='center')
    b.line(root, 'Table header rule', (30, 300), (1445, 300), RULE, 1.2)

    for i in range(3):
        yy = 316 + 146 * i
        b.label(root, str(i + 1), 80, yy + 18, 30, INK, True, align='center')
        b.label(root, f't = {i}', 80, yy + 60, 19, MUTED, align='center')
        b.grid(root, ex['states'][i], 165, yy, 19, False, f'Round {i+1} / current state')
        b.label(root, f'X{SUBS[i]}', 260, yy + 99, 21, MUTED, align='center')
        for j in range(i):
            b.action_chip(root, 'done', 449 + 110*j, yy + 25, 'past')
        b.recorded_plan(root, ex, i, 449 + 110 * i, yy + 25, 110)
        b.label(root, 'left', 855, yy + 8, 23, BLUE, True, align='center')
        b.route(root, f'Round {i+1} / first action causes observed result',
                [(793, yy + 49), (934, yy + 49)], BLUE, 2.3)
        b.grid(root, ex['states'][i+1], 945, yy, 19, False, f'Round {i+1} / next state')
        b.label(root, f'X{SUBS[i+1]}', 1040, yy + 99, 21, MUTED, align='center')
        b.outcome(root, ex, i, 1180, yy + 26, 245, 45, prefix='')
        b.line(root, f'Round {i+1} separator', (30, yy + 132), (1445, yy + 132), RULE, 1)
    b.footer(root, 'Gray: already executed. Blue: execute now. Outlined actions: proposed for later steps, then reconsidered.')
    return root


def remove_previous_options(source):
    children = collections.defaultdict(list)
    for n in source['nodeChanges']:
        if 'parentIndex' in n:
            children[gid(n['parentIndex']['guid'])].append(n)
    removed = set()
    def mark(n):
        removed.add(gid(n['guid']))
        for c in children[gid(n['guid'])]:
            mark(c)
    for n in source['nodeChanges']:
        if n.get('name') in [PREFIX + v for v in VARIANTS]:
            mark(n)
    result = copy.deepcopy(source)
    result['nodeChanges'] = [n for n in result['nodeChanges'] if gid(n['guid']) not in removed]
    return result, len(removed)


def validate(saved, source, b, roots, ex):
    byid = {gid(n['guid']): n for n in saved['nodeChanges']}
    assert len(byid) == len(saved['nodeChanges'])
    for n in source['nodeChanges']:
        assert byid[gid(n['guid'])] == n, ('Original node changed', n['name'])
    assert saved['blobs'][:len(source['blobs'])] == source['blobs']
    refs = []
    def scan(v):
        if isinstance(v, dict):
            for k, x in v.items():
                if k.endswith('Blob') and isinstance(x, int):
                    refs.append(x)
                else:
                    scan(x)
        elif isinstance(v, list):
            for x in v:
                scan(x)
    scan(saved['nodeChanges'])
    assert all(0 <= i < len(saved['blobs']) for i in refs)
    for n in saved['nodeChanges']:
        if 'parentIndex' in n:
            assert gid(n['parentIndex']['guid']) in byid
    root_ids = {gid(r['guid']) for r in roots}
    texts = 0
    for original in b.nodes:
        n = byid[gid(original['guid'])]
        if gid(n['guid']) in root_ids:
            continue
        chain, current = [], n
        while gid(current['guid']) not in root_ids:
            chain.append(current)
            current = byid[gid(current['parentIndex']['guid'])]
        tf = Affine2D()
        for a in chain:
            tf += matrix(a)
        ww, hh = n['size']['x'], n['size']['y']
        corners = tf.transform([(0, 0), (ww, 0), (ww, hh), (0, hh)])
        assert min(p[0] for p in corners) >= -.01, (n['name'], corners)
        assert max(p[0] for p in corners) <= W + .01, (n['name'], corners)
        assert min(p[1] for p in corners) >= -.01, (n['name'], corners)
        assert max(p[1] for p in corners) <= H + .01, (n['name'], corners)
        if n['type'] == 'TEXT':
            texts += 1
            s, d = n['textData']['characters'], n['derivedTextData']
            assert len(d['glyphs']) == len(s)
            assert [g['firstCharacter'] for g in d['glyphs']] == list(range(len(s)))

    assert len(b.plan_records) == len(b.outcome_records) == 9
    assert [r['round'] for r in b.plan_records] == [0, 1, 2] * 3
    assert [r['round'] for r in b.outcome_records] == [0, 1, 2] * 3
    for r in b.plan_records:
        assert r['actions'] == ex['rounds'][r['round']]['plan']
        assert r['executed'] == ex['rounds'][r['round']]['executed']
    for r in b.outcome_records:
        assert r['reached'] == ex['rounds'][r['round']]['reached_goal']
    assert len(b.grid_records) == 20
    children = collections.defaultdict(list)
    for n in saved['nodeChanges']:
        if 'parentIndex' in n:
            children[gid(n['parentIndex']['guid'])].append(n)
    grids_per_root = collections.Counter()
    colors = {'black': (0, 0, 0), 'blue': (0, 0, 1), 'red': (1, 0, 0)}
    for r in b.grid_records:
        assert r['full_grid'] in ex['states']
        node = byid[tuple(r['node'])]
        cells = children[gid(node['guid'])]
        assert len(cells) == 50
        for cell in cells:
            # Cell names and native fills are independently matched to the source grid.
            import re
            m = re.match(r'Cell \((\d+), (\d+)\) / (\w+)', cell['name'])
            assert m
            rr, cc, color = int(m[1]), int(m[2]), m[3]
            assert r['full_grid'][rr][cc] == color
            fill = cell['fillPaints'][0]['color']
            assert tuple(fill[k] for k in ['r', 'g', 'b']) == colors[color]
        while gid(node['guid']) not in root_ids:
            node = byid[gid(node['parentIndex']['guid'])]
        grids_per_root[node['name']] += 1
    assert [grids_per_root[r['name']] for r in roots] == [5, 7, 8]
    return dict(original_nodes_preserved=len(source['nodeChanges']),
                original_blobs_preserved=len(source['blobs']),
                new_nodes=len(b.nodes), editable_text_layers=texts,
                new_elements_within_bounds=len(b.nodes) - 3,
                valid_blob_references=len(refs), variants=3,
                grids_per_variant=dict(grids_per_root), native_game_cells=1000,
                recorded_plans_actions_and_outcomes_verified=True,
                every_visible_cell_matches_recorded_state=True,
                native_round_trip='passed', figma_application_import_tested=False)


def package(source, canvas, destination, preview, root):
    img = Image.open(preview).convert('RGBA')
    img.thumbnail((800, 800))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    with zipfile.ZipFile(source) as original:
        meta = json.loads(original.read('meta.json'))
        meta['exported_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        meta['client_meta']['thumbnail_size'] = dict(width=img.width, height=img.height)
        meta['client_meta']['render_coordinates'] = dict(
            x=root['transform']['m02'], y=root['transform']['m12'], width=W, height=H)
        with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_STORED) as z:
            for item in original.infolist():
                if item.filename not in ['canvas.fig', 'thumbnail.png', 'meta.json']:
                    z.writestr(item, original.read(item.filename))
            z.writestr('canvas.fig', canvas.read_bytes())
            z.writestr('thumbnail.png', buf.getvalue())
            z.writestr('meta.json', json.dumps(meta, separators=(',', ':')))
    with zipfile.ZipFile(destination) as z:
        assert z.testzip() is None
        assert z.read('canvas.fig') == canvas.read_bytes()


def comparison(out):
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 25)
    width, slot = W + 32, H + 65
    sheet = Image.new('RGB', (width, slot * 3 + 16), '#e8eaec')
    draw = ImageDraw.Draw(sheet)
    for i, (name, title) in enumerate(zip(VARIANTS, TITLES)):
        y = 14 + slot * i
        draw.text((17, y), f'{i + 1}. {title}', fill=INK, font=font)
        sheet.paste(Image.open(out / f'{PREFIX}{name}.png').convert('RGB'), (16, y + 44))
    sheet.save(out / 'planning_game_examples_three_options.png')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=SOURCE)
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--modules', type=Path, default=Path('/tmp/planning_game_examples/node_modules'))
    ap.add_argument('--font-dir', type=Path, default=Path('/tmp/planning_game_examples/fonts'))
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    bun = shutil.which('bun') or '/home/ays57/.bun/bin/bun'
    before = hashlib.sha256(args.source.read_bytes()).hexdigest()
    ex = load_example()
    with tempfile.TemporaryDirectory(prefix='planning-game-fig-') as directory:
        tmp = Path(directory)
        with zipfile.ZipFile(args.source) as z:
            (tmp / 'reference.canvas').write_bytes(z.read('canvas.fig'))
        subprocess.run([bun, str(IO), 'decode', str(tmp / 'reference.canvas'),
                        str(tmp / 'source.json'), '-', str(args.modules)], check=True)
        source, removed = remove_previous_options(json.loads((tmp / 'source.json').read_text()))
        page = next(n for n in source['nodeChanges'] if n['type'] == 'CANVAS' and n['name'] == 'Figures')
        top = [n for n in source['nodeChanges'] if n.get('parentIndex', {}).get('guid') == page['guid']]
        x = min(n['transform']['m02'] for n in top)
        y = max(n['transform']['m12'] + n['size']['y'] for n in top) + 160
        b = ExampleBuilder(source, args.font_dir)
        roots = [filmstrip(b, page, x, y, ex),
                 decision_cards(b, page, x + W + 120, y, ex),
                 horizon_table(b, page, x + 2 * (W + 120), y, ex)]
        message = {k: copy.deepcopy(v) for k, v in source.items()
                   if k in ['type', 'sessionID', 'ackID', 'nodeChangeOrder', 'nodeChanges', 'blobs']}
        message['nodeChanges'] += b.nodes
        message['blobs'] = b.blobs
        (tmp / 'combined.json').write_text(json.dumps(message))
        subprocess.run([bun, str(IO), 'encode', str(tmp / 'reference.canvas'),
                        str(tmp / 'combined.json'), str(tmp / 'combined.canvas'), str(args.modules)], check=True)
        saved = json.loads((tmp / 'combined.canvas.json').read_text())
        checks = validate(saved, source, b, roots, ex)
        for root, name in zip(roots, VARIANTS):
            render(saved, gid(root['guid']), args.out / f'{PREFIX}{name}.png',
                   args.out / f'{PREFIX}{name}.pdf', args.out / f'{PREFIX}{name}.svg')
        comparison(args.out)
        staged = args.out / 'learning_evolution_fig_with_game_example_options.fig'
        package(args.source, tmp / 'combined.canvas', staged,
                args.out / f'{PREFIX}{VARIANTS[0]}.png', roots[0])
        report = dict(source=str(args.source.relative_to(ROOT)), source_sha256_before=before,
                      staged_file=str(staged.relative_to(ROOT)), checks=checks,
                      replaced_previous_option_nodes=removed, frame_size=dict(width=W, height=H),
                      variants=[dict(name=r['name'], id=gid(r['guid']), position=r['transform']) for r in roots],
                      example=ex, new_native_node_ids=[gid(n['guid']) for n in b.nodes],
                      plans=b.plan_records, outcomes=b.outcome_records,
                      grids=b.grid_records, connectors=b.connectors,
                      procedure_sources=['paper/main.tex:156-173',
                          'offline_learning/scripts/eval_curated_online.py:111-237',
                          'offline_learning/scripts/eval_curated_plan.py:465-514'],
                      applied_to_source=False)
        if args.apply:
            assert hashlib.sha256(args.source.read_bytes()).hexdigest() == before
            backup = args.out / 'learning_evolution_fig_before_game_examples.fig'
            if not backup.exists():
                shutil.copy2(args.source, backup)
            pending = args.source.with_name(args.source.name + '.tmp')
            shutil.copy2(staged, pending)
            pending.replace(args.source)
            report.update(applied_to_source=True, backup=str(backup.relative_to(ROOT)),
                          source_sha256_after=hashlib.sha256(args.source.read_bytes()).hexdigest())
        (args.out / 'planning_game_examples_evidence.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: report[k] for k in ['source', 'checks', 'variants', 'applied_to_source']}, indent=2))


if __name__ == '__main__':
    main()
