#!/usr/bin/env python3
"""Clean the selected rollout schema and append three recorded game examples.

All game states, plans, executed actions, and goal checks come from saved lmwm
evaluations. This generates native editable Figma geometry, PNG, PDF, and SVG.
The original document is updated only with --apply.
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
from offline_learning.scripts.fig_planning_evaluation_variants import (
    DiagramBuilder, SOURCE, IO, BG, BLUE, BLUE_BG, GREEN, GREEN_BG,
    INK, MUTED, RULE, WHITE, GRAY, gid, matrix, render,
)
from offline_learning.scripts.fig_grow_learning_compact import rect_commands
from matplotlib.colors import to_rgba
from matplotlib.transforms import Affine2D
from PIL import Image, ImageDraw, ImageFont

OUT = SOURCE.parent / 'planning_visual_rollouts'
PROBLEMS = ROOT / 'logs/2026-08-29/planning_v2/problems.json'
RUN = ROOT / 'logs/2026-08-30/planning_v2_online_ds'
PREFIX = 'planning_visual_rollout_'
REFERENCE = 'planning_evaluation_recorded_rollout'
W, H = 1480, 696
SUBS = '₀₁₂₃₄'
SPECS = [
    dict(key='grow', game='Grow', task='7xf97:move-cloud:s101',
         description='Move the cloud three cells left.',
         expected_plans=[['left'] * n for n in (3, 2, 1)]),
    dict(key='sand', game='Sand', task='va6fq:sand-rectangle:s101',
         description='Fill both holes in the sand block.',
         expected_plans=[['click 6 3', 'click 5 3', 'click 6 6', 'click 5 6'][i:]
                         for i in range(4)]),
    dict(key='egg', game='Egg', task='egg:carry-left:s101',
         description='Move the intact egg three cells left.',
         expected_plans=[['left'] * n for n in (3, 2, 1)]),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_examples():
    problems = {p['task_uid']: p for p in json.loads(PROBLEMS.read_text())['problems']}
    result = []
    for spec in SPECS:
        p = problems[spec['task']]
        path = RUN / p['game'] / 'online.json'
        data = json.loads(path.read_text())
        row = next(r for r in data['rows'] if r['task_uid'] == p['task_uid'])
        presentation = row.get('goal_presentation') or row.get('eval_goal_mode')
        assert presentation in ('frame', 'exact_frame'), presentation
        a = row['lmwm']['attempts'][0]
        rounds = [{k: r[k] for k in ('n', 'plan', 'executed', 'reached_goal', 'z_after')}
                  for r in a['rounds']]
        states = [p['start']] + [json.loads(r['grid_after']) for r in a['rounds']]
        count = len(rounds)
        assert a['success'] and a['reached_at'] == count == a['actions_used']
        assert [r['n'] for r in rounds] == list(range(count))
        assert [r['plan'] for r in rounds] == spec['expected_plans']
        assert all(r['executed'] == r['plan'][0] for r in rounds)
        assert [r['reached_goal'] for r in rounds] == [False] * (count - 1) + [True]
        assert [s == p['goal'] for s in states] == [False] * count + [True]
        assert all(len(s) == len(s[0]) for s in states)
        result.append(dict(**spec, task_uid=p['task_uid'], arm='lmwm', attempt=0,
                           presentation='frame', task_description=p['nl_goal'],
                           prefix=p.get('prefix', []), start=p['start'], goal=p['goal'],
                           states=states, rounds=rounds, success=True, reached_at=count,
                           action_cap=data['config']['max_actions'],
                           warm_start=data['config']['warm_start'],
                           problems=str(PROBLEMS.relative_to(ROOT)),
                           problems_sha256=sha(PROBLEMS),
                           rollout=str(path.relative_to(ROOT)), rollout_sha256=sha(path)))
    return result


def prepare_source(original):
    source = copy.deepcopy(original)
    children = collections.defaultdict(list)
    for n in source['nodeChanges']:
        if 'parentIndex' in n:
            children[gid(n['parentIndex']['guid'])].append(n)
    removed_options = set()
    def mark(n):
        removed_options.add(gid(n['guid']))
        for child in children[gid(n['guid'])]:
            mark(child)
    for n in source['nodeChanges']:
        if n.get('name') in [PREFIX + s['key'] for s in SPECS]:
            mark(n)
    source['nodeChanges'] = [n for n in source['nodeChanges']
                             if gid(n['guid']) not in removed_options]

    root = next(n for n in source['nodeChanges'] if n.get('name') == REFERENCE)
    direct = children[gid(root['guid'])]
    remove = [n for n in direct if n['name'] in
              ('Planning evaluation', 'A recorded magnets rollout', 'Header rule')
              or n['name'].startswith(('Blue action:', 'Recorded target-state task'))]
    cleaned_ids = {gid(n['guid']) for n in remove}
    changed = []
    if remove:
        assert len(remove) == 5, [n['name'] for n in remove]
        assert root['size'] == dict(x=1480, y=704)
        for n in direct:
            if gid(n['guid']) not in cleaned_ids:
                n['transform']['m12'] -= 81
                changed.append(gid(n['guid']))
        root['size']['y'] = 554
        root['fillGeometry'][0]['commandsBlob'] = len(source['blobs'])
        source['blobs'].append({'bytes': {'$bytes': base64.b64encode(rect_commands(0, 0, W, 554)).decode()}})
        changed.append(gid(root['guid']))
        source['nodeChanges'] = [n for n in source['nodeChanges']
                                 if gid(n['guid']) not in cleaned_ids]
    else:
        assert root['size']['y'] == 554
    return source, root, dict(removed_previous_options=sorted(removed_options),
                              removed_reference_nodes=sorted(cleaned_ids),
                              repositioned_or_resized_reference_nodes=changed)


class RolloutBuilder(DiagramBuilder):
    def __init__(self, source, font_dir):
        super().__init__(source, font_dir)
        self.plan_records, self.outcome_records = [], []

    def full_grid(self, parent, grid, x, y, width, task, role, axes=False, rows=False):
        dim = len(grid)
        cell = width / dim
        n = self.frame(parent, role, x, y, width, width)
        bg = collections.Counter(c for row in grid for c in row).most_common(1)[0][0]
        stroke = '#73777a' if bg == 'gray' else '#343b40'
        for r, row in enumerate(grid):
            for c, color in enumerate(row):
                self.rect(n, f'Cell ({r}, {c}) / {color}',
                          c * cell, r * cell, cell, cell, color, stroke, .42)
        if axes:
            size = 11 if dim == 16 else 13
            for c in range(dim):
                self.text(parent, str(c), x + (c + .5) * cell, y + width + 4,
                          size, MUTED, align='center')
            if rows:
                for r in range(dim):
                    self.text(parent, str(r), x - 7, y + r * cell + (cell - size * 1.2) / 2,
                              size, MUTED, align='right')
        self.grid_records.append(dict(node=gid(n['guid']), task_uid=task,
                                      role=role, full_grid=grid, crop=None,
                                      visible_cells=dim * dim))
        return n

    def proposed_plan(self, root, example, i, x, y):
        actions = example['rounds'][i]['plan']
        clicks = all(a.startswith('click ') for a in actions)
        width, height, gap = (58, 56, 6) if clicks else (78, 42, 9)
        nodes = []
        for j, action in enumerate(actions):
            n = self.block(root, f'Round {i} / proposed action {j + 1} / {action}',
                           x + j * (width + gap), y, width, height,
                           BLUE if j == 0 else WHITE, BLUE if j == 0 else GRAY)
            color = WHITE if j == 0 else MUTED
            if clicks:
                _, rr, cc = action.split()
                self.label(n, 'click', width / 2, 5, 16, color, j == 0, align='center')
                self.label(n, f'{rr}, {cc}', width / 2, 28, 18, color, j == 0, align='center')
            else:
                self.label(n, action, width / 2, 7, 22, color, j == 0, align='center')
            nodes.append(gid(n['guid']))
        self.plan_records.append(dict(task_uid=example['task_uid'], round=i,
                                      actions=actions, executed=actions[0], nodes=nodes))
        return x + width / 2, y + height


def draw_rollout(b, page, x, y, ex):
    root = b.frame(page, PREFIX + ex['key'], x, y, W, H, BG)
    b.full_grid(root, ex['goal'], 30, 26, 128, ex['task_uid'], 'Given target state X_G')
    b.label(root, 'Target state  X{G}', 185, 29, 25, bold=True)
    b.label(root, ex['description'], 185, 72, 23, maxw=590)
    b.label(root, 'Planner receives P(X{G})', 185, 114, 21, MUTED)
    model = b.block(root, 'Fixed learned model across the episode',
                    810, 31, 635, 106, BLUE_BG, BLUE)
    b.label(model, 'Same learned model (P, K) at every step',
            24, 16, 25, BLUE, True, maxw=590)
    b.label(model, 'P encodes the state; K guides the LLM plan.',
            24, 57, 22, BLUE, maxw=590)
    b.line(root, 'Task / episode divider', (30, 176), (1445, 176), RULE, 1.2)

    five = len(ex['states']) == 5
    first, spacing, width = (162, 292, 220) if five else (188, 365, 240)
    centers = [first + i * spacing for i in range(len(ex['states']))]
    grid_y = 357 - width / 2
    final = len(ex['rounds'])
    for i, (center, state) in enumerate(zip(centers, ex['states'])):
        b.label(root, f't = {i}' + (' · start' if i == 0 else ''), center, 194, 25,
                GREEN if i == final else INK, True, align='center')
        b.full_grid(root, state, center - width / 2, grid_y, width,
                    ex['task_uid'], f'Real observation X{SUBS[i]}', True, i == 0)
        if i == 0:
            b.label(root, 'Initial observation X₀', center, 508,
                    20 if five else 22, align='center')
        else:
            reached = ex['rounds'][i - 1]['reached_goal']
            b.label(root, 'Goal? Yes → stop' if reached else 'Goal? No → replan',
                    center, 508, (22 if reached else 20) if five else (24 if reached else 22),
                    GREEN if reached else INK, reached, align='center')
            b.outcome_records.append(dict(task_uid=ex['task_uid'], round=i - 1,
                                           state=i, reached=reached))
        if i < final:
            b.label(root, f'P(X{SUBS[i]}) → LLM + K + G', center, 548,
                    18 if five else 21, BLUE, align='center')
            start_x, start_y = b.proposed_plan(root, ex, i, center - (128 if five else 139), 588)
            riser = center + spacing / 2
            b.route(root, f'Round {i} / execute first action only',
                    [(start_x, start_y), (start_x, 666), (riser, 666),
                     (riser, 357), (centers[i + 1] - width / 2, 357)], BLUE, 2.3)
        else:
            w = 252 if five else 286
            success = b.block(root, f'Recorded success at step {final}',
                              center - w / 2, 571, w, 94, GREEN_BG, GREEN)
            b.label(success, f'X{SUBS[final]} = X{{G}}', w / 2, 12, 27,
                    GREEN, True, align='center')
            b.label(success, f'Success at step {final}', w / 2, 53,
                    21 if five else 23, GREEN, align='center')
    return root


def validate(saved, original, source, cleanup, b, roots, reference, examples):
    byid = {gid(n['guid']): n for n in saved['nodeChanges']}
    assert len(byid) == len(saved['nodeChanges'])
    assert saved['blobs'][:len(original['blobs'])] == original['blobs']
    for n in source['nodeChanges']:
        assert byid[gid(n['guid'])] == n, ('Unexpected change to source', n['name'])
    allowed_changes = {tuple(i) for i in cleanup['repositioned_or_resized_reference_nodes']}
    removed = {tuple(i) for k in ('removed_previous_options', 'removed_reference_nodes')
               for i in cleanup[k]}
    for n in original['nodeChanges']:
        if gid(n['guid']) not in allowed_changes | removed:
            assert byid[gid(n['guid'])] == n
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
    children = collections.defaultdict(list)
    for n in saved['nodeChanges']:
        if 'parentIndex' in n:
            assert gid(n['parentIndex']['guid']) in byid
            children[gid(n['parentIndex']['guid'])].append(n)
    root_ids = {gid(n['guid']) for n in roots + [reference]}
    texts, bounded = 0, 0
    def bounded_node(n):
        nonlocal texts, bounded
        if gid(n['guid']) in root_ids:
            return
        chain, current = [], n
        while gid(current['guid']) not in root_ids:
            chain.append(current)
            current = byid[gid(current['parentIndex']['guid'])]
        tf = Affine2D()
        for item in chain:
            tf += matrix(item)
        w, h = n['size']['x'], n['size']['y']
        points = tf.transform([(0, 0), (w, 0), (w, h), (0, h)])
        for axis, size in [(0, current['size']['x']), (1, current['size']['y'])]:
            assert min(p[axis] for p in points) >= -.02, (n['name'], points)
            assert max(p[axis] for p in points) <= size + .02, (n['name'], points)
        bounded += 1
        if n['type'] == 'TEXT':
            texts += 1
            s, d = n['textData']['characters'], n['derivedTextData']
            assert len(d['glyphs']) == len(s)
            assert [g['firstCharacter'] for g in d['glyphs']] == list(range(len(s)))
            assert s not in ('Planning evaluation', 'A recorded magnets rollout')
            assert not s.startswith(('Blue action:', 'Recorded target-state task'))
    for n in b.nodes:
        bounded_node(byid[gid(n['guid'])])
    def visit(n):
        bounded_node(n)
        for c in children[gid(n['guid'])]:
            visit(c)
    visit(byid[gid(reference['guid'])])

    example_by_task = {e['task_uid']: e for e in examples}
    for p in b.plan_records:
        r = example_by_task[p['task_uid']]['rounds'][p['round']]
        assert p['actions'] == r['plan'] and p['executed'] == r['executed']
    for outcome in b.outcome_records:
        assert outcome['reached'] == example_by_task[outcome['task_uid']]['rounds'][outcome['round']]['reached_goal']
    assert len(b.plan_records) == len(b.outcome_records) == 10
    cells = 0
    for record in b.grid_records:
        grid = record['full_grid']
        assert grid in example_by_task[record['task_uid']]['states']
        actual = children[tuple(record['node'])]
        assert len(actual) == record['visible_cells']
        for n in actual:
            m = re.fullmatch(r'Cell \((\d+), (\d+)\) / (\w+)', n['name'])
            assert m
            r, c, color = int(m[1]), int(m[2]), m[3]
            assert grid[r][c] == color
            expected = to_rgba(color)
            native = n['fillPaints'][0]['color']
            assert all(abs(native[k] - v) < 1e-7 for k, v in zip('rgba', expected))
        cells += len(actual)
    assert cells == 3160 and len(b.grid_records) == 16
    return dict(original_nodes=len(original['nodeChanges']),
                original_blobs_preserved=len(original['blobs']),
                unrelated_nodes_unchanged=True, reference_frame_cleaned=True,
                new_nodes=len(b.nodes), editable_text_layers_checked=texts,
                elements_within_frame_bounds=bounded, valid_blob_references=len(refs),
                recorded_plans_and_actions_checked=10, goal_checks_verified=10,
                full_recorded_grids=16, native_game_cells=cells,
                every_visible_cell_matches_recording=True,
                native_round_trip='passed', figma_application_import_tested=False)


def package(source, canvas, destination, preview, root):
    img = Image.open(preview).convert('RGBA')
    img.thumbnail((800, 800))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    with zipfile.ZipFile(source) as old:
        meta = json.loads(old.read('meta.json'))
        meta['exported_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        meta['client_meta']['thumbnail_size'] = dict(width=img.width, height=img.height)
        meta['client_meta']['render_coordinates'] = dict(
            x=root['transform']['m02'], y=root['transform']['m12'], width=W, height=H)
        with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_STORED) as z:
            for item in old.infolist():
                if item.filename not in ('canvas.fig', 'thumbnail.png', 'meta.json'):
                    z.writestr(item, old.read(item.filename))
            z.writestr('canvas.fig', canvas.read_bytes())
            z.writestr('thumbnail.png', buf.getvalue())
            z.writestr('meta.json', json.dumps(meta, separators=(',', ':')))
    with zipfile.ZipFile(destination) as z:
        assert z.testzip() is None
        assert z.read('canvas.fig') == canvas.read_bytes()


def comparison(out):
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 25)
    sheet = Image.new('RGB', (W + 32, (H + 66) * 3 + 16), '#e8eaec')
    draw = ImageDraw.Draw(sheet)
    for i, spec in enumerate(SPECS):
        y = 14 + i * (H + 66)
        # Option labels belong only to the comparison sheet, outside the figures.
        draw.text((17, y), f'{i + 1}. {spec["game"]} — {spec["description"]}', fill=INK, font=font)
        sheet.paste(Image.open(out / f'{PREFIX}{spec["key"]}.png').convert('RGB'), (16, y + 44))
    sheet.save(out / 'planning_visual_rollouts_three_options.png')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=SOURCE)
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--modules', type=Path, default=Path('/tmp/planning_game_examples/node_modules'))
    ap.add_argument('--font-dir', type=Path, default=Path('/tmp/planning_game_examples/fonts'))
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    before, examples = sha(args.source), load_examples()
    bun = shutil.which('bun') or '/home/ays57/.bun/bin/bun'
    with tempfile.TemporaryDirectory(prefix='planning-visual-rollouts-') as directory:
        tmp = Path(directory)
        with zipfile.ZipFile(args.source) as z:
            (tmp / 'reference.canvas').write_bytes(z.read('canvas.fig'))
        subprocess.run([bun, str(IO), 'decode', str(tmp / 'reference.canvas'),
                        str(tmp / 'source.json'), '-', str(args.modules)], check=True)
        original = json.loads((tmp / 'source.json').read_text())
        source, reference, cleanup = prepare_source(original)
        page = next(n for n in source['nodeChanges'] if n['type'] == 'CANVAS' and n['name'] == 'Figures')
        top = [n for n in source['nodeChanges'] if n.get('parentIndex', {}).get('guid') == page['guid']]
        x = min(n['transform']['m02'] for n in top)
        y = max(n['transform']['m12'] + n['size']['y'] for n in top) + 160
        b = RolloutBuilder(source, args.font_dir)
        roots = [draw_rollout(b, page, x + i * (W + 120), y, ex) for i, ex in enumerate(examples)]
        message = {k: copy.deepcopy(v) for k, v in source.items()
                   if k in ('type', 'sessionID', 'ackID', 'nodeChangeOrder', 'nodeChanges', 'blobs')}
        message['nodeChanges'] += b.nodes
        message['blobs'] = b.blobs
        (tmp / 'combined.json').write_text(json.dumps(message))
        subprocess.run([bun, str(IO), 'encode', str(tmp / 'reference.canvas'),
                        str(tmp / 'combined.json'), str(tmp / 'combined.canvas'), str(args.modules)], check=True)
        saved = json.loads((tmp / 'combined.canvas.json').read_text())
        checks = validate(saved, original, source, cleanup, b, roots, reference, examples)
        for root in roots + [reference]:
            name = root['name'] if root != reference else 'planning_recorded_rollout_magnets_clean'
            render(saved, gid(root['guid']), args.out / f'{name}.png',
                   args.out / f'{name}.pdf', args.out / f'{name}.svg')
        comparison(args.out)
        staged = args.out / 'learning_evolution_fig_with_visual_rollouts.fig'
        package(args.source, tmp / 'combined.canvas', staged,
                args.out / f'{roots[0]["name"]}.png', roots[0])
        report = dict(source=str(args.source.relative_to(ROOT)), source_sha256_before=before,
                      staged_file=str(staged.relative_to(ROOT)), staged_sha256=sha(staged),
                      frame_size=dict(width=W, height=H), checks=checks, cleanup=cleanup,
                      variants=[dict(name=r['name'], id=gid(r['guid']), position=r['transform']) for r in roots],
                      cleaned_reference=dict(name=reference['name'], id=gid(reference['guid']), size=reference['size']),
                      examples=examples, plans=b.plan_records, outcomes=b.outcome_records,
                      grids=b.grid_records, connectors=b.connectors,
                      new_native_node_ids=[gid(n['guid']) for n in b.nodes],
                      procedure_sources=['paper/main.tex:156-173',
                          'offline_learning/scripts/eval_curated_online.py:111-237',
                          'offline_learning/scripts/eval_curated_plan.py:465-514'],
                      applied_to_source=False)
        if args.apply:
            assert sha(args.source) == before
            backup = args.out / 'learning_evolution_fig_before_visual_rollouts.fig'
            if not backup.exists():
                shutil.copy2(args.source, backup)
            pending = args.source.with_name(args.source.name + '.tmp')
            shutil.copy2(staged, pending)
            pending.replace(args.source)
            for ext in ('png', 'pdf', 'svg'):
                shutil.copy2(args.out / f'planning_recorded_rollout_magnets_clean.{ext}',
                             args.source.parent / f'planning_evaluation/{REFERENCE}.{ext}')
            report.update(applied_to_source=True, backup=str(backup.relative_to(ROOT)),
                          source_sha256_after=sha(args.source))
        (args.out / 'planning_visual_rollouts_evidence.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: report[k] for k in ('checks', 'variants', 'cleaned_reference', 'applied_to_source')}, indent=2))


if __name__ == '__main__':
    main()
