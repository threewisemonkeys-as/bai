#!/usr/bin/env python3
"""Add a recorded Egg rollout with an explicit plan revision to the native .fig."""
from __future__ import annotations

import argparse
import collections
import copy
import datetime
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
from offline_learning.scripts.fig_planning_visual_rollouts import (
    RolloutBuilder, SOURCE, PROBLEMS, IO, sha, gid, matrix, render,
    BG, BLUE, BLUE_BG, GREEN, GREEN_BG, INK, MUTED, RULE, WHITE, GRAY,
)
from offline_learning.planning_nl_goals import get_python_goal, freeze_grid
from matplotlib.colors import to_rgba
from matplotlib.transforms import Affine2D
from PIL import Image

OUT = SOURCE.parent / 'planning_replanning'
ROLLOUT = ROOT / 'logs/2026-09-01/planning_v2_online_ds_nl/egg/online.json'
TASK = 'egg:controlled-gravity-drop:s101'
NAME = 'planning_replanning_egg'
W, H = 1480, 774
AMBER, AMBER_BG = '#a35d0b', '#fff7e8'
SUBS = '₀₁₂₃₄'


def load_example():
    p = next(p for p in json.loads(PROBLEMS.read_text())['problems'] if p['task_uid'] == TASK)
    data = json.loads(ROLLOUT.read_text())
    row = next(r for r in data['rows'] if r['task_uid'] == TASK)
    assert row['goal_presentation'] == data['config']['goal_presentation'] == 'nl'
    a = row['lmwm']['attempts'][0]
    assert a['success'] and a['reached_at'] == a['actions_used'] == 4
    rounds = [{k: r[k] for k in ('n', 'plan', 'executed', 'reached_goal', 'z_after')}
              for r in a['rounds']]
    assert [r['n'] for r in rounds] == list(range(4))
    assert [r['plan'] for r in rounds] == [
        ['click 0 0', 'noop', 'click 0 0'], ['click 0 0'] * 3,
        ['click 0 0'] * 2, ['click 0 0']]
    assert [r['executed'] for r in rounds] == ['click 0 0'] * 4
    states = [p['start']] + [json.loads(r['grid_after']) for r in a['rounds']]
    checker = get_python_goal(p['nl_checker'])
    checks = [bool(checker.check([freeze_grid(s) for s in states[:i + 1]],
                                 [r['executed'] for r in rounds[:i]])) for i in range(5)]
    assert checks == [False, False, False, False, True]
    assert [r['reached_goal'] for r in rounds] == checks[1:]
    revisions = [i for i in range(1, 4) if rounds[i]['plan'] != rounds[i - 1]['plan'][1:]]
    assert revisions == [1]
    assert rounds[0]['plan'][1] == 'noop' and rounds[1]['executed'] == 'click 0 0'
    # Verify each concise scene annotation against the raw observations.
    assert [s[0][0] for s in states] == ['red', 'pink', 'red', 'pink', 'red']
    assert [max(r for r, row in enumerate(s) if 'tan' in row) for s in states] == [13, 13, 14, 14, 15]
    assert all(sum(c == 'tan' for row in s for c in row) == 21 for s in states)
    return dict(task_uid=TASK, game='Egg', arm='lmwm', attempt=0, presentation='nl',
                goal_text=p['nl_goal'], goal_checker=p['nl_checker'],
                states=states, rounds=rounds, revisions=revisions,
                revision=dict(round=1, old_remaining=rounds[0]['plan'][1:],
                              new_plan=rounds[1]['plan'], old_next_action='noop',
                              new_next_action='click 0 0'),
                success=True, reached_at=4, independently_recomputed_goal_checks=checks,
                quiescence_waived=a['quiescence_waived'], prefix=p['prefix'],
                action_cap=data['config']['max_actions'], warm_start=data['config']['warm_start'],
                problems=str(PROBLEMS.relative_to(ROOT)), problems_sha256=sha(PROBLEMS),
                rollout=str(ROLLOUT.relative_to(ROOT)), rollout_sha256=sha(ROLLOUT),
                scene_annotation_source='autumn_programs/egg.sexp:20-36')


class ReplanBuilder(RolloutBuilder):
    def recorded_plan(self, root, ex, i, x, y):
        actions = ex['rounds'][i]['plan']
        if i == 1:
            self.rect(root, 'Annotation / revised plan outline', x - 6, y - 6,
                      226, 68, AMBER_BG, AMBER, 1.5)
            self.label(root, 'Revised plan', x + 107, y - 31, 18,
                       AMBER, True, align='center')
        nodes = []
        for j, action in enumerate(actions):
            replaced = i == 0 and j == 1
            stroke = BLUE if j == 0 else AMBER if replaced else GRAY
            node = self.block(root, f'Round {i} / proposed action {j + 1} / {action}',
                              x + 74 * j, y, 66, 56, BLUE if j == 0 else WHITE, stroke)
            color = WHITE if j == 0 else AMBER if replaced else MUTED
            if action.startswith('click '):
                _, rr, cc = action.split()
                self.label(node, 'click', 33, 5, 17, color, j == 0, align='center')
                self.label(node, f'{rr}, {cc}', 33, 29, 18, color, j == 0, align='center')
            else:
                assert action == 'noop'
                self.label(node, action, 33, 17, 18, color, align='center')
            nodes.append(gid(node['guid']))
            if replaced:
                self.label(root, 'replaced', x + 74 * j + 33, y + 63,
                           15, AMBER, align='center')
        if i == 1:
            self.label(root, 'noop → click 0, 0', x + 107, y + 64,
                       17, AMBER, align='center')
        self.plan_records.append(dict(round=i, actions=actions,
                                      executed=ex['rounds'][i]['executed'], nodes=nodes))
        return x + 33, y + 56


def draw_figure(b, page, x, y, ex):
    root = b.frame(page, NAME, x, y, W, H, BG)
    b.label(root, 'Goal  G{NL}', 30, 27, 25, bold=True)
    lines = ['From the raised position, turn gravity on,',
             'land the egg, then turn gravity off.']
    assert ' '.join(lines) == ex['goal_text']
    for j, line in enumerate(lines):
        b.label(root, line, 30, 69 + 32 * j, 23, maxw=720)
    b.label(root, 'Planner receives this text.', 30, 139, 19, MUTED)
    model = b.block(root, 'Fixed learned model across the episode',
                    810, 31, 635, 106, BLUE_BG, BLUE)
    b.label(model, 'Same learned model (P, K) at every step', 24, 16,
            25, BLUE, True, maxw=590)
    b.label(model, 'P encodes the state; K guides the LLM plan.', 24, 57,
            22, BLUE, maxw=590)
    b.line(root, 'Task / episode divider', (30, 178), (1445, 178), RULE, 1.2)

    centers = [162 + 292 * i for i in range(5)]
    descriptions = ['Raised · gravity off', 'Still raised · gravity on',
                    'Lower · gravity off', 'Above floor · gravity on', 'Landed · gravity off']
    for i, (center, state) in enumerate(zip(centers, ex['states'])):
        b.label(root, f't = {i}' + (' · start' if i == 0 else ''), center, 194,
                25, GREEN if i == 4 else INK, True, align='center')
        b.full_grid(root, state, center - 110, 239, 220, TASK,
                    f'Real observation X{SUBS[i]}', True, i == 0)
        b.label(root, descriptions[i], center, 491, 16,
                GREEN if i == 4 else MUTED, align='center', maxw=264)
        if i == 0:
            b.label(root, 'Initial observation X₀', center, 526, 20, align='center')
        else:
            reached = ex['rounds'][i - 1]['reached_goal']
            b.label(root, 'Goal? Yes → stop' if reached else 'Goal? No → replan',
                    center, 526, 22 if reached else 20, GREEN if reached else INK,
                    reached, align='center')
            b.outcome_records.append(dict(round=i - 1, state=i, reached=reached))
        if i < 4:
            b.label(root, f'P(X{SUBS[i]}) → LLM + K + G{{NL}}', center, 565,
                    18, BLUE, align='center', maxw=268)
            sx, sy = b.recorded_plan(root, ex, i, center - 128, 632)
            b.route(root, f'Round {i} / execute only {ex["rounds"][i]["executed"]}',
                    [(sx, sy), (sx, 744), (center + 146, 744),
                     (center + 146, 349), (centers[i + 1] - 110, 349)], BLUE, 2.3)
        else:
            success = b.block(root, 'Goal checker / success at step 4',
                              center - 126, 625, 252, 94, GREEN_BG, GREEN)
            b.label(success, 'Goal satisfied', 126, 12, 26, GREEN, True, align='center')
            b.label(success, 'Success at step 4', 126, 53, 21, GREEN, align='center')
    return root


def remove_previous(source):
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
        if n.get('name') == NAME:
            mark(n)
    result = copy.deepcopy(source)
    result['nodeChanges'] = [n for n in result['nodeChanges'] if gid(n['guid']) not in removed]
    return result, sorted(removed)


def validate(saved, source, b, root, ex):
    byid = {gid(n['guid']): n for n in saved['nodeChanges']}
    assert len(byid) == len(saved['nodeChanges'])
    assert saved['blobs'][:len(source['blobs'])] == source['blobs']
    for n in source['nodeChanges']:
        assert byid[gid(n['guid'])] == n
    children = collections.defaultdict(list)
    for n in saved['nodeChanges']:
        if 'parentIndex' in n:
            assert gid(n['parentIndex']['guid']) in byid
            children[gid(n['parentIndex']['guid'])].append(n)
    refs = []
    def scan(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k.endswith('Blob') and isinstance(v, int):
                    refs.append(v)
                else:
                    scan(v)
        elif isinstance(o, list):
            for v in o:
                scan(v)
    scan(saved['nodeChanges'])
    assert all(0 <= r < len(saved['blobs']) for r in refs)
    text_count = 0
    for old in b.nodes:
        n = byid[gid(old['guid'])]
        if gid(n['guid']) == gid(root['guid']):
            continue
        tf, current = Affine2D(), n
        while gid(current['guid']) != gid(root['guid']):
            tf += matrix(current)
            current = byid[gid(current['parentIndex']['guid'])]
        w, h = n['size']['x'], n['size']['y']
        corners = tf.transform([(0, 0), (w, 0), (w, h), (0, h)])
        assert min(p[0] for p in corners) >= -.02 and max(p[0] for p in corners) <= W + .02
        assert min(p[1] for p in corners) >= -.02 and max(p[1] for p in corners) <= H + .02
        if n['type'] == 'TEXT':
            text_count += 1
            s = n['textData']['characters']
            assert len(n['derivedTextData']['glyphs']) == len(s)
            assert [g['firstCharacter'] for g in n['derivedTextData']['glyphs']] == list(range(len(s)))
            assert s != 'Planning evaluation' and not s.startswith('Blue action:')
    assert len(b.plan_records) == len(b.outcome_records) == 4
    for p, o, r in zip(b.plan_records, b.outcome_records, ex['rounds']):
        assert p['round'] == o['round'] == r['n']
        assert p['actions'] == r['plan'] and p['executed'] == r['executed']
        assert o['reached'] == r['reached_goal']
    assert len(b.grid_records) == 5
    for record, state in zip(b.grid_records, ex['states']):
        assert record['full_grid'] == state
        cells = children[tuple(record['node'])]
        assert len(cells) == 256
        for cell in cells:
            m = re.fullmatch(r'Cell \((\d+), (\d+)\) / (\w+)', cell['name'])
            assert m and state[int(m[1])][int(m[2])] == m[3]
            color = cell['fillPaints'][0]['color']
            assert all(abs(color[k] - v) < 1e-7 for k, v in zip('rgba', to_rgba(m[3])))
    return dict(original_nodes_preserved=len(source['nodeChanges']),
                original_blobs_preserved=len(source['blobs']), new_nodes=len(b.nodes),
                editable_text_layers=text_count, all_new_elements_within_bounds=True,
                valid_blob_references=len(refs), native_game_cells=1280,
                all_states_plans_actions_and_goal_checks_verified=True,
                goal_checks_independently_recomputed=True, revision_verified_at_t=1,
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
        assert z.testzip() is None and z.read('canvas.fig') == canvas.read_bytes()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=SOURCE)
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--modules', type=Path, default=Path('/tmp/planning_game_examples/node_modules'))
    ap.add_argument('--font-dir', type=Path, default=Path('/tmp/planning_game_examples/fonts'))
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    before, ex = sha(args.source), load_example()
    bun = shutil.which('bun') or '/home/ays57/.bun/bin/bun'
    with tempfile.TemporaryDirectory(prefix='planning-replanning-') as directory:
        tmp = Path(directory)
        with zipfile.ZipFile(args.source) as z:
            (tmp / 'reference.canvas').write_bytes(z.read('canvas.fig'))
        subprocess.run([bun, str(IO), 'decode', str(tmp / 'reference.canvas'),
                        str(tmp / 'source.json'), '-', str(args.modules)], check=True)
        source, removed = remove_previous(json.loads((tmp / 'source.json').read_text()))
        page = next(n for n in source['nodeChanges'] if n['type'] == 'CANVAS' and n['name'] == 'Figures')
        top = [n for n in source['nodeChanges'] if n.get('parentIndex', {}).get('guid') == page['guid']]
        x = min(n['transform']['m02'] for n in top)
        y = max(n['transform']['m12'] + n['size']['y'] for n in top) + 160
        b = ReplanBuilder(source, args.font_dir)
        root = draw_figure(b, page, x, y, ex)
        message = {k: copy.deepcopy(v) for k, v in source.items()
                   if k in ('type', 'sessionID', 'ackID', 'nodeChangeOrder', 'nodeChanges', 'blobs')}
        message['nodeChanges'] += b.nodes
        message['blobs'] = b.blobs
        (tmp / 'combined.json').write_text(json.dumps(message))
        subprocess.run([bun, str(IO), 'encode', str(tmp / 'reference.canvas'),
                        str(tmp / 'combined.json'), str(tmp / 'combined.canvas'), str(args.modules)], check=True)
        saved = json.loads((tmp / 'combined.canvas.json').read_text())
        checks = validate(saved, source, b, root, ex)
        render(saved, gid(root['guid']), args.out / f'{NAME}.png',
               args.out / f'{NAME}.pdf', args.out / f'{NAME}.svg')
        staged = args.out / 'learning_evolution_fig_with_replanning.fig'
        package(args.source, tmp / 'combined.canvas', staged, args.out / f'{NAME}.png', root)
        report = dict(source=str(args.source.relative_to(ROOT)), source_sha256_before=before,
                      staged_file=str(staged.relative_to(ROOT)), staged_sha256=sha(staged),
                      frame=dict(name=NAME, id=gid(root['guid']), position=root['transform'], size=root['size']),
                      checks=checks, example=ex, replaced_previous_option_nodes=removed,
                      plans=b.plan_records, outcomes=b.outcome_records, grids=b.grid_records,
                      connectors=b.connectors, new_native_node_ids=[gid(n['guid']) for n in b.nodes],
                      applied_to_source=False)
        if args.apply:
            assert sha(args.source) == before
            backup = args.out / 'learning_evolution_fig_before_replanning.fig'
            if not backup.exists():
                shutil.copy2(args.source, backup)
            pending = args.source.with_name(args.source.name + '.tmp')
            shutil.copy2(staged, pending)
            pending.replace(args.source)
            report.update(applied_to_source=True, backup=str(backup.relative_to(ROOT)),
                          source_sha256_after=sha(args.source))
        (args.out / 'planning_replanning_evidence.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: report[k] for k in ('frame', 'checks', 'applied_to_source')}, indent=2))


if __name__ == '__main__':
    main()
