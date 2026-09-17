#!/usr/bin/env python3
"""Append three editable planning-evaluation figures to the local native .fig.

The paper and online evaluator define the protocol. The storyboard uses an
existing learned-model rollout; no plans, states, or outcomes are synthesized.
Native geometry and Inter glyph caches are reused from the existing figures.
"""
from __future__ import annotations

import argparse
import base64
import collections
import copy
import datetime
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from offline_learning.scripts.fig_magnets_rex_tree_variants import TreeBuilder, command
from offline_learning.scripts.fig_grow_learning_compact import (
    BG, BLUE, BLUE_BG, GREEN, GREEN_BG, INK, MUTED, RULE,
    gid, matrix, paint, render,
)
from matplotlib.transforms import Affine2D
from PIL import Image, ImageDraw, ImageFont

SOURCE = ROOT / 'analysis/learning_example/learning_evolution_fig.fig'
OUT = SOURCE.parent / 'planning_evaluation'
IO = ROOT / 'offline_learning/scripts/figma_export/fig_kiwi_io.cjs'
PROBLEMS = ROOT / 'logs/2026-08-29/planning_v2/problems.json'
ROLLOUT = ROOT / 'logs/2026-08-30/planning_v2_online_ds/7www9/online.json'
TASK = '7www9:move-away:s101'
PREFIX = 'planning_evaluation_'
VARIANTS = ['feedback_loop', 'recorded_rollout', 'protocol_lanes']
TITLES = ['Feedback loop', 'Recorded rollout', 'Planner / environment / evaluator']
W, H = 1480, 704
WHITE, LIGHT, GRAY = '#ffffff', '#f1f3f3', '#a8b0b6'


class DiagramBuilder(TreeBuilder):
    def __init__(self, source, font_dir):
        super().__init__(source, font_dir)
        self.connectors = []
        self.grid_records = []

    def label(self, parent, s, x, y, size=23, color=INK, bold=False,
              align='left', maxw=None):
        """Inline subscripts use braces, e.g. X{G}; each run stays editable."""
        chunks = [(p[1:-1], True) if p.startswith('{') else (p, False)
                  for p in re.split(r'(\{[^}]+\})', s) if p]
        width = sum(self.measure(t, size * (.68 if sub else 1), bold)
                    for t, sub in chunks)
        if maxw is not None:
            assert width <= maxw, (s, width, maxw)
        if align == 'center':
            x -= width / 2
        elif align == 'right':
            x -= width
        for t, sub in chunks:
            sz = size * (.68 if sub else 1)
            self.text(parent, t, x, y + (size * .49 if sub else 0), sz,
                      color, bold, name=s + (' / subscript' if sub else ''))
            x += self.measure(t, sz, bold)

    def block(self, parent, name, x, y, w, h, fill=WHITE, stroke=RULE):
        n = self.frame(parent, name, x, y, w, h)
        self.rect(n, name + ' / background', 0, 0, w, h, fill, stroke, 1.3)
        return n

    def route(self, parent, name, points, color=INK, width=2, dashed=False, head=8):
        for i, (a, z) in enumerate(zip(points, points[1:])):
            if i == len(points) - 2 and not dashed:
                self.arrow(parent, name, a, z, color, width, head)
            else:
                self.line(parent, name, a, z, color, width, dashed)
        if dashed:
            a, z = points[-2:]
            dx, dy = z[0] - a[0], z[1] - a[1]
            length = (dx * dx + dy * dy) ** .5
            self.arrow(parent, name + ' / arrow',
                       (z[0] - dx / length * 3, z[1] - dy / length * 3),
                       z, color, width, head)
        self.connectors.append(dict(name=name, parent=gid(parent['guid']),
                                    points=points, dashed=dashed))

    def heading(self, root, descriptor):
        self.text(root, 'Planning evaluation', 30, 22, 32, INK, True)
        self.text(root, descriptor, W - 30, 32, 21, MUTED, align='right')
        self.line(root, 'Header rule', (30, 77), (W - 30, 77), RULE, 1.2)

    def grid(self, parent, grid, x, y, cell=24, axes=True, name='Observed state', row_axes=True):
        # Common crop contains every colored cell in all four recorded states.
        r0, r1, c0, c1 = 6, 11, 0, 10
        assert len(grid) == len(grid[0]) == 16
        assert all(c == 'black' or (r0 <= r < r1 and c0 <= col < c1)
                   for r, row in enumerate(grid) for col, c in enumerate(row))
        n = self.frame(parent, name, x, y, (c1 - c0) * cell, (r1 - r0) * cell)
        colors = {'black': '#000000', 'blue': '#0000ff', 'red': '#ff0000'}
        for r in range(r0, r1):
            for col in range(c0, c1):
                self.rect(n, f'Cell ({r}, {col}) / {grid[r][col]}',
                          (col - c0) * cell, (r - r0) * cell, cell, cell,
                          colors[grid[r][col]], '#394047', .65)
        if axes:
            for col in range(c0, c1):
                self.text(parent, str(col), x + (col - c0 + .5) * cell,
                          y + (r1 - r0) * cell + 3, 15, MUTED, align='center')
            if row_axes:
                for r in range(r0, r1):
                    self.text(parent, str(r), x - 7, y + (r - r0 + .13) * cell,
                              15, MUTED, align='right')
        self.grid_records.append(dict(node=gid(n['guid']), crop=[r0, r1, c0, c1],
                                      full_grid=grid, visible_cells=50))
        return n

    def plan(self, parent, actions, x, y, first_color=BLUE):
        for i, action in enumerate(actions):
            box = self.block(parent, f'Proposed action {i + 1} / {action}',
                             x + 87 * i, y, 78, 42,
                             first_color if i == 0 else WHITE,
                             first_color if i == 0 else GRAY)
            self.text(box, action, 39, 7, 22,
                      WHITE if i == 0 else MUTED, i == 0, align='center', maxw=70)


def feedback_loop(b, page, x, y):
    root = b.frame(page, PREFIX + VARIANTS[0], x, y, W, H, BG)
    b.heading(root, 'Feedback loop')

    init = b.block(root, 'Task / start state', 30, 115, 210, 92)
    b.label(init, 'Start state', 105, 13, 23, bold=True, align='center')
    b.label(init, 'X₀', 105, 46, 28, align='center')
    b.route(root, 'Initialize environment at start state', [(135, 207), (135, 306)])

    k = b.block(root, 'Learned dynamics K', 560, 115, 260, 92, BLUE_BG, BLUE)
    b.label(k, 'Dynamics K', 130, 13, 24, BLUE, True, align='center')
    b.label(k, 'Learned rules', 130, 49, 21, BLUE, align='center')
    b.route(root, 'Dynamics K conditions the LLM', [(685, 207), (685, 306)], BLUE)

    goal = b.block(root, 'Task / goal', 880, 115, 260, 92)
    b.label(goal, 'Goal G', 130, 13, 24, bold=True, align='center')
    b.label(goal, 'Text or target state', 130, 49, 21, align='center')
    b.route(root, 'Goal input to planner', [(880, 161), (846, 161), (846, 267), (760, 267), (760, 306)])
    b.label(root, 'State goals via P', 859, 233, 18, MUTED)
    b.route(root, 'Task criterion used by evaluator', [(1140, 161), (1300, 161), (1300, 306)], MUTED)
    b.label(root, 'Goal criterion', 1281, 208, 19, MUTED, align='right')

    blocks = [
        ('Current observation', 30, 210, WHITE, RULE),
        ('Learned perception P', 295, 210, BLUE_BG, BLUE),
        ('LLM planner', 560, 260, WHITE, RULE),
        ('Real environment', 880, 260, WHITE, RULE),
        ('Goal checker', 1195, 250, LIGHT, RULE),
    ]
    panels = {name: b.block(root, name, xx, 306, ww, 126, fill, stroke)
              for name, xx, ww, fill, stroke in blocks}
    p = panels['Current observation']
    b.label(p, 'Observe', 105, 18, 25, bold=True, align='center')
    b.label(p, 'Current state Xₜ', 105, 64, 22, align='center')
    p = panels['Learned perception P']
    b.label(p, 'Perception P', 105, 18, 25, BLUE, True, align='center')
    b.label(p, 'Oₜ = P(Xₜ)', 105, 64, 25, BLUE, align='center')
    p = panels['LLM planner']
    b.label(p, 'LLM planner', 130, 18, 25, bold=True, align='center')
    b.label(p, 'Plan aₜ, aₜ₊₁, …', 130, 64, 24, align='center')
    p = panels['Real environment']
    b.label(p, 'Execute only aₜ', 130, 18, 25, bold=True, align='center')
    b.label(p, 'Observe Xₜ₊₁', 130, 64, 24, align='center')
    p = panels['Goal checker']
    b.label(p, 'Goal satisfied?', 125, 18, 25, bold=True, align='center')
    b.label(p, 'Check real trajectory', 125, 66, 20, align='center')
    b.label(root, 'Real environment', 1010, 273, 20, MUTED, align='center')
    for a, z in [(240, 295), (505, 560), (820, 880), (1140, 1195)]:
        b.route(root, f'Forward flow {a}', [(a, 369), (z, 369)])

    b.route(root, 'Yes / stop on first goal satisfaction', [(1270, 432), (1270, 495)], GREEN)
    b.label(root, 'Yes', 1282, 451, 22, GREEN, True)
    success = b.block(root, 'Success / stop', 1180, 495, 265, 60, GREEN_BG, GREEN)
    b.label(success, 'Success · stop', 132.5, 15, 25, GREEN, True, align='center')

    b.route(root, 'No / feed observed state into next planning round',
            [(1395, 432), (1395, 470), (1463, 470), (1463, 592), (135, 592), (135, 432)], BLUE, 2.2)
    b.label(root, 'No', 1406, 439, 22, BLUE, True)
    b.label(root, 'Use Xₜ₊₁ and replan while steps remain; stop at the task step limit',
            690, 553, 23, BLUE, align='center', maxw=1110)

    b.label(root, 'Learned model', 295, 115, 24, BLUE, True, maxw=250)
    b.label(root, 'P and K stay fixed', 295, 154, 21, BLUE, maxw=250)
    b.label(root, 'during evaluation', 295, 183, 21, BLUE, maxw=250)
    b.label(root, 'Recent observations and actions also enter the planner.', 295, 646, 20, MUTED)
    b.label(root, 'Text goal: trajectory checker. Target state: exact raw-state match.',
            295, 675, 20, MUTED)
    return root


def recorded_rollout(b, page, x, y, example):
    root = b.frame(page, PREFIX + VARIANTS[1], x, y, W, H, BG)
    b.heading(root, 'A recorded magnets rollout')

    b.grid(root, example['goal'], 30, 112, 18, False, 'Task / target state crop')
    b.label(root, 'Target state  X{G}', 235, 110, 25, bold=True)
    b.label(root, 'Blue magnet at column 1', 235, 148, 23)
    b.label(root, 'Planner receives P(X{G})', 235, 184, 21, MUTED)
    model = b.block(root, 'Fixed learned model across the episode', 810, 111, 635, 106, BLUE_BG, BLUE)
    b.label(model, 'Same learned model (P, K) at every step', 24, 16, 25, BLUE, True, maxw=590)
    b.label(model, 'P encodes the state; K guides the LLM plan.', 24, 57, 22, BLUE, maxw=590)
    b.line(root, 'Task / episode divider', (30, 241), (1445, 241), RULE, 1.2)

    subs = ['₀', '₁', '₂', '₃']
    for i, grid in enumerate(example['states']):
        xx = 30 + 365 * i
        b.label(root, f't = {i}' + (' · start' if i == 0 else ''), xx + 158, 259, 25,
                GREEN if i == 3 else INK, True, align='center')
        b.grid(root, grid, xx + 39, 302, 24, True, f'Real observation X{subs[i]}', row_axes=i == 0)
        if i == 0:
            b.label(root, 'Initial observation X₀', xx + 158, 450, 22, align='center')
        elif i < 3:
            b.label(root, 'Goal? No → replan', xx + 158, 450, 22, align='center')
        else:
            b.label(root, 'Goal? Yes → stop', xx + 158, 450, 24, GREEN, True, align='center')
        if i < 3:
            b.label(root, f'P(X{subs[i]}) → LLM + K + G', xx + 158, 490, 21, BLUE, align='center')
            b.plan(root, example['rounds'][i]['plan'], xx + 19, 531)
            b.route(root, f'Round {i} / execute first action only',
                    [(xx + 58, 573), (xx + 58, 603), (xx + 334, 603),
                     (xx + 334, 362), (xx + 404, 362)], BLUE, 2.3)
        else:
            success = b.block(root, 'Recorded outcome / success at step 3',
                              xx + 19, 511, 286, 93, GREEN_BG, GREEN)
            b.label(success, 'X₃ = X{G}', 143, 12, 27, GREEN, True, align='center')
            b.label(success, 'Success at step 3', 143, 53, 23, GREEN, align='center')

    b.label(root, 'Blue action: executed now. Outlined actions: proposed remainder, reconsidered next round.',
            30, 637, 22, INK, maxw=1420)
    b.label(root, 'Recorded target-state task · cropped views; goal checks compare the full 16 × 16 states after every action.',
            30, 676, 19, MUTED, maxw=1420)
    return root


def protocol_lanes(b, page, x, y):
    root = b.frame(page, PREFIX + VARIANTS[2], x, y, W, H, BG)
    b.heading(root, 'Planner, environment, evaluator')
    b.label(root, 'Task: start state X₀ + goal G (text or target state)', 30, 103, 25, bold=True)
    b.label(root, 'Learned P and K stay fixed', 1445, 106, 22, BLUE, align='right')
    b.label(root, 'Target-state goals are encoded by P before planning.', 30, 139, 18, MUTED)

    columns = [(340, 'Planner', 'P + LLM using K and G', BLUE, BLUE_BG),
               (830, 'Environment', 'Real state transitions', INK, LIGHT),
               (1265, 'Evaluator', 'Task goal criterion', INK, LIGHT)]
    for cx, title, subtitle, color, fill in columns:
        head = b.block(root, title + ' / lane heading', cx - 170, 164, 340, 83, fill, RULE)
        b.label(head, title, 170, 10, 26, color, True, align='center')
        b.label(head, subtitle, 170, 47, 20, color, align='center')
        b.line(root, title + ' / time line', (cx, 247), (cx, 619), GRAY, 1.2, True)

    b.route(root, 'Environment supplies current observation', [(830, 287), (340, 287)])
    b.label(root, 'Observe Xₜ (initially X₀)', 585, 254, 22, align='center')
    planner = b.block(root, 'Planner / encode and propose sequence', 140, 308, 400, 76, BLUE_BG, BLUE)
    b.label(planner, 'LLM plans from P(Xₜ), G, K', 200, 10, 23, BLUE, True, align='center')
    b.label(planner, 'aₜ, aₜ₊₁, …', 200, 43, 24, BLUE, align='center')

    b.route(root, 'Planner sends only first action', [(340, 422), (830, 422)], BLUE, 2.3)
    b.label(root, 'Execute only aₜ', 600, 390, 24, BLUE, True, align='center')
    env = b.block(root, 'Environment / real execution', 720, 438, 220, 49)
    b.label(env, 'Real transition', 110, 10, 22, align='center')
    b.route(root, 'Evaluator receives observed state and trajectory', [(830, 519), (1265, 519)])
    b.label(root, 'Xₜ₊₁ + trajectory', 1048, 486, 22, align='center')
    checker = b.block(root, 'Evaluator / check after every action', 1140, 535, 250, 51, LIGHT, RULE)
    b.label(checker, 'Goal satisfied?', 125, 11, 24, bold=True, align='center')

    b.route(root, 'No / continue to next planning round', [(1265, 586), (1265, 619), (340, 619)], BLUE, 2.2)
    b.label(root, 'No · budget remains → replan from Xₜ₊₁', 783, 584, 22, BLUE, align='center')
    b.label(root, 'Next planning round', 340, 634, 23, BLUE, True, align='center')
    b.route(root, 'Yes / terminate with success', [(1390, 560), (1445, 560), (1445, 649), (1425, 649)], GREEN)
    b.label(root, 'Yes', 1404, 587, 20, GREEN, True)
    success = b.block(root, 'Evaluator / success', 1140, 628, 285, 44, GREEN_BG, GREEN)
    b.label(success, 'Success · stop', 142.5, 7, 24, GREEN, True, align='center')

    b.label(root, 'Text: trajectory checker. State: exact raw-state match. Stop if the task step limit is exhausted.',
            30, 679, 18, MUTED, maxw=1420)
    return root


def load_example():
    p = next(p for p in json.loads(PROBLEMS.read_text())['problems'] if p['task_uid'] == TASK)
    data = json.loads(ROLLOUT.read_text())
    row = next(r for r in data['rows'] if r['task_uid'] == TASK)
    attempt = row['lmwm']['attempts'][0]
    assert data['config']['goal_presentation'] == 'frame'
    assert attempt['success'] and attempt['reached_at'] == 3
    rounds = [{k: r[k] for k in ['n', 'plan', 'executed', 'reached_goal', 'z_after']}
              for r in attempt['rounds']]
    assert [r['plan'] for r in rounds] == [['left'] * 3, ['left'] * 2, ['left']]
    assert [r['executed'] for r in rounds] == ['left'] * 3
    assert [r['reached_goal'] for r in rounds] == [False, False, True]
    states = [p['start']] + [json.loads(r['grid_after']) for r in attempt['rounds']]
    assert [s == p['goal'] for s in states] == [False, False, False, True]
    for i, s in enumerate(states):
        cells = [(r, c, v) for r, row in enumerate(s) for c, v in enumerate(row) if v != 'black']
        assert cells == [(7, 4-i, 'blue'), (7, 7, 'red'), (8, 4-i, 'blue'), (8, 7, 'red')]
    return dict(task_uid=TASK, presentation='frame', arm='lmwm', attempt=0,
                task_description=p['nl_goal'], start=p['start'], goal=p['goal'],
                states=states, rounds=rounds, success=True, reached_at=3,
                action_cap=data['config']['max_actions'],
                warm_start=data['config']['warm_start'],
                problems=str(PROBLEMS.relative_to(ROOT)), rollout=str(ROLLOUT.relative_to(ROOT)))


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


def validate(saved, source, b, roots):
    byid = {gid(n['guid']): n for n in saved['nodeChanges']}
    assert len(byid) == len(saved['nodeChanges'])
    assert saved['blobs'][:len(source['blobs'])] == source['blobs']
    for n in source['nodeChanges']:
        assert byid[gid(n['guid'])] == n, ('Original node changed', n['name'])
    refs = []
    def scan(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.endswith('Blob') and isinstance(v, int):
                    refs.append(v)
                else:
                    scan(v)
        elif isinstance(obj, list):
            for v in obj:
                scan(v)
    scan(saved['nodeChanges'])
    assert all(0 <= i < len(saved['blobs']) for i in refs)
    for n in saved['nodeChanges']:
        if 'parentIndex' in n:
            assert gid(n['parentIndex']['guid']) in byid
    root_ids = {gid(r['guid']) for r in roots}
    texts, bounded = 0, 0
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
        points = tf.transform([(0, 0), (ww, 0), (ww, hh), (0, hh)])
        assert min(p[0] for p in points) >= -.01, (n['name'], points)
        assert max(p[0] for p in points) <= W + .01, (n['name'], points)
        assert min(p[1] for p in points) >= -.01, (n['name'], points)
        assert max(p[1] for p in points) <= H + .01, (n['name'], points)
        bounded += 1
        if n['type'] == 'TEXT':
            texts += 1
            s, d = n['textData']['characters'], n['derivedTextData']
            assert len(d['glyphs']) == len(s)
            assert [g['firstCharacter'] for g in d['glyphs']] == list(range(len(s)))
    # All visible game cells are native rectangles, generated from recorded states.
    assert len(b.grid_records) == 5
    assert all(g['visible_cells'] == 50 for g in b.grid_records)
    return dict(original_nodes_preserved=len(source['nodeChanges']),
                original_blobs_preserved=len(source['blobs']), variants=3,
                new_nodes=len(b.nodes), editable_text_layers=texts,
                new_elements_within_bounds=bounded, valid_blob_references=len(refs),
                recorded_plans_actions_states_and_goal_checks_verified=True,
                native_game_cells=250, native_round_trip='passed',
                figma_application_import_tested=False)


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
    width, slot = 1512, 767
    sheet = Image.new('RGB', (width, slot * 3 + 18), '#e8eaec')
    draw = ImageDraw.Draw(sheet)
    for i, (name, title) in enumerate(zip(VARIANTS, TITLES)):
        y = 14 + slot * i
        draw.text((17, y), f'{i + 1}. {title}', fill=INK, font=font)
        sheet.paste(Image.open(out / f'{PREFIX}{name}.png').convert('RGB'), (16, y + 44))
    sheet.save(out / 'planning_evaluation_three_options.png')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=SOURCE)
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--modules', type=Path, default=Path('/tmp/grow_figma_compact/node_modules'))
    ap.add_argument('--font-dir', type=Path, default=Path('/tmp/rex_magnets_figma'))
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    bun = shutil.which('bun') or '/home/ays57/.bun/bin/bun'
    before = hashlib.sha256(args.source.read_bytes()).hexdigest()
    example = load_example()
    with tempfile.TemporaryDirectory(prefix='planning-eval-fig-') as directory:
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
        b = DiagramBuilder(source, args.font_dir)
        roots = [feedback_loop(b, page, x, y),
                 recorded_rollout(b, page, x + W + 120, y, example),
                 protocol_lanes(b, page, x + 2 * (W + 120), y)]
        message = {k: copy.deepcopy(v) for k, v in source.items()
                   if k in ['type', 'sessionID', 'ackID', 'nodeChangeOrder', 'nodeChanges', 'blobs']}
        message['nodeChanges'] += b.nodes
        message['blobs'] = b.blobs
        (tmp / 'combined.json').write_text(json.dumps(message))
        subprocess.run([bun, str(IO), 'encode', str(tmp / 'reference.canvas'),
                        str(tmp / 'combined.json'), str(tmp / 'combined.canvas'), str(args.modules)], check=True)
        saved = json.loads((tmp / 'combined.canvas.json').read_text())
        checks = validate(saved, source, b, roots)
        for root, name in zip(roots, VARIANTS):
            render(saved, gid(root['guid']), args.out / f'{PREFIX}{name}.png',
                   args.out / f'{PREFIX}{name}.pdf', args.out / f'{PREFIX}{name}.svg')
        comparison(args.out)
        staged = args.out / 'learning_evolution_fig_with_planning_options.fig'
        package(args.source, tmp / 'combined.canvas', staged,
                args.out / f'{PREFIX}{VARIANTS[0]}.png', roots[0])
        report = dict(source=str(args.source.relative_to(ROOT)), source_sha256_before=before,
                      staged_file=str(staged.relative_to(ROOT)), checks=checks,
                      replaced_previous_option_nodes=removed,
                      frame_size=dict(width=W, height=H),
                      variants=[dict(name=r['name'], id=gid(r['guid']), position=r['transform']) for r in roots],
                      procedure_sources=['paper/main.tex:156-173',
                          'offline_learning/scripts/eval_curated_online.py:111-237',
                          'offline_learning/scripts/eval_curated_plan.py:465-514'],
                      notation=dict(P='learned perception program', K='learned dynamics knowledge',
                                    O_t='P(X_t)', X_t='raw current state'),
                      example=example, new_native_node_ids=[gid(n['guid']) for n in b.nodes],
                      connectors=b.connectors, grid_records=b.grid_records,
                      applied_to_source=False)
        if args.apply:
            assert hashlib.sha256(args.source.read_bytes()).hexdigest() == before
            backup = args.out / 'learning_evolution_fig_before_planning.fig'
            if not backup.exists():
                shutil.copy2(args.source, backup)
            pending = args.source.with_name(args.source.name + '.tmp')
            shutil.copy2(staged, pending)
            pending.replace(args.source)
            report.update(applied_to_source=True, backup=str(backup.relative_to(ROOT)),
                          source_sha256_after=hashlib.sha256(args.source.read_bytes()).hexdigest())
        (args.out / 'planning_evaluation_evidence.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: report[k] for k in ['source', 'checks', 'variants', 'applied_to_source']}, indent=2))


if __name__ == '__main__':
    main()
