#!/usr/bin/env python3
"""Add SET, Mario, and Sand excerpts with recorded plan revisions to Figma."""
from __future__ import annotations

import argparse
import collections
import copy
import difflib
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
from offline_learning.scripts.fig_planning_replanning import AMBER, AMBER_BG, package
from offline_learning.planning_nl_goals import (
    get_python_goal, freeze_grid, set_card_count, set_selected_count,
)
from matplotlib.colors import to_rgba
from matplotlib.transforms import Affine2D
from PIL import Image, ImageDraw, ImageFont

OUT = SOURCE.parent / 'planning_replanning_rich'
PREFIX = 'planning_replanning_rich_'
W, H = 1480, 774
SPECS = [
    dict(key='set', game='SET', task='SET:remove-valid-set-s211:s211',
         rollout='logs/2026-08-30/planning_v2_online_ds/SET/online.json',
         start=1, end=5, focus=2, focus_note='Choose different cards',
         expected_plans=[['click 1 4', 'click 1 8', 'click 7 4'],
                         ['click 8 12', 'click 14 8'], ['click 14 8'], ['noop']]),
    dict(key='mario', game='Mario', task='n2ntd:high-ground:s0',
         rollout='logs/2026-09-03/planning_v2_online_ds_percap_nl/n2ntd/online.json',
         start=7, end=11, focus=9, focus_note='Future noop → up',
         expected_plans=[['right', 'right', 'noop', 'noop'],
                         ['right', 'noop', 'noop'], ['noop', 'up'], ['up']]),
    dict(key='sand', game='Sand', task='va6fq:sand-rectangle:s101',
         rollout='logs/2026-09-03/planning_v2_online_ds_percap_nl/va6fq/online.json',
         start=2, end=6, focus=3, focus_note='Click 5, 6 → 6, 6',
         expected_plans=[['click 5 6', 'click 5 6', 'click 5 3'],
                         ['click 6 6'], ['click 5 3', 'click 5 6'], ['click 5 6']]),
]


def subscript(i):
    return str(i).translate(str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉'))


def load_examples():
    problems = {p['task_uid']: p for p in json.loads(PROBLEMS.read_text())['problems']}
    examples = []
    for spec in SPECS:
        p = problems[spec['task']]
        path = ROOT / spec['rollout']
        data = json.loads(path.read_text())
        row = next(r for r in data['rows'] if r['task_uid'] == spec['task'])
        mode = row.get('goal_presentation') or row.get('eval_goal_mode')
        assert mode in ('nl', 'python')
        attempt = row['lmwm']['attempts'][0]
        action_cap = data['config'].get('action_caps', {}).get(spec['task'], data['config']['max_actions'])
        assert attempt['rounds'][0]['remaining'] == action_cap
        rounds = [{k: r[k] for k in ('n', 'plan', 'executed', 'reached_goal', 'z_after')}
                  for r in attempt['rounds']]
        assert [r['n'] for r in rounds] == list(range(len(rounds)))
        assert attempt['success'] and attempt['reached_at'] == attempt['actions_used'] == spec['end']
        states = [p['start']] + [json.loads(r['grid_after']) for r in attempt['rounds']]
        checker = get_python_goal(p['nl_checker'])
        checks = [bool(checker.check([freeze_grid(s) for s in states[:i + 1]],
                                    [r['executed'] for r in rounds[:i]]))
                  for i in range(len(states))]
        assert checks == [False] * spec['end'] + [True]
        assert checks[1:] == [r['reached_goal'] for r in rounds]
        assert all(r['executed'] == r['plan'][0] for r in rounds)
        assert [r['plan'] for r in rounds[spec['start']:spec['end']]] == spec['expected_plans']
        revisions = []
        for t in range(spec['start'] + 1, spec['end']):
            old, new = rounds[t - 1]['plan'][1:], rounds[t]['plan']
            if old != new:
                revisions.append(dict(t=t, old_remaining=old, new_plan=new,
                                      kind='revised' if old else 'extended'))
        focus = next(r for r in revisions if r['t'] == spec['focus'])
        if spec['key'] == 'mario':
            assert focus['old_remaining'] == ['noop', 'noop']
            assert focus['new_plan'] == ['noop', 'up'] and rounds[10]['executed'] == 'up'
        else:
            assert focus['old_remaining'][0] != focus['new_plan'][0]
        examples.append(dict(**spec, task_uid=spec['task'], arm='lmwm', attempt=0,
                             presentation='nl', original_mode=mode, goal_text=p['nl_goal'],
                             goal_checker=p['nl_checker'], states=states, rounds=rounds,
                             displayed_steps=list(range(spec['start'], spec['end'] + 1)),
                             revisions=revisions, checks=checks, success=True,
                             reached_at=spec['end'], prefix=p.get('prefix', []),
                             quiescence_waived=attempt.get('quiescence_waived', row.get('quiescence_waived', False)),
                             warm_start=data['config']['warm_start'],
                             action_cap=action_cap,
                             run_config=data['config'], problems=str(PROBLEMS.relative_to(ROOT)),
                             problems_sha256=sha(PROBLEMS), rollout_sha256=sha(path)))
    return examples


def scene_caption(ex, t):
    state = ex['states'][t]
    if ex['key'] == 'set':
        cards, selected = set_card_count(state), set_selected_count(state)
        if t == ex['end']:
            assert cards == 6 and selected == 0
            return 'Three cards removed'
        assert cards == 9
        return ('Nine cards · none selected' if selected == 0 else
                f'{selected} card' + ('' if selected == 1 else 's') + ' selected')
    if ex['key'] == 'mario':
        red = [(r, c) for r, row in enumerate(state) for c, color in enumerate(row) if color == 'red']
        assert len(red) == 1
        if t == ex['end']:
            assert red[0] == (5, 8)
            return 'On the highest platform'
        return f'Player at ({red[0][0]}, {red[0][1]})'
    missing = sum(state[r][c] != 'tan' for r in range(5, 10) for c in range(2, 8))
    if t == ex['end']:
        assert missing == 0
        return 'Rectangle complete'
    return f'{missing} empty cell' + ('' if missing == 1 else 's')


class RichBuilder(RolloutBuilder):
    def snapshot(self, root, ex, t, x, y, width, row_axes=False):
        grid = ex['states'][t]
        node = self.full_grid(root, grid, x, y, width, ex['task_uid'],
                              f'Real observation X{subscript(t)}')
        self.grid_records[-1]['t'] = t
        size = 9.3 if len(grid) == 20 else 12
        cell = width / len(grid)
        for c in range(len(grid)):
            self.label(root, str(c), x + (c + .5) * cell, y + width + 4,
                        size, MUTED, align='center')
        if row_axes:
            for r in range(len(grid)):
                self.label(root, str(r), x - 7, y + r * cell + (cell - size * 1.2) / 2,
                            size, MUTED, align='right')
        return node

    def plan(self, root, ex, t, center, y):
        actions = ex['rounds'][t]['plan']
        width, gap, height = 58, 6, 56
        span = len(actions) * width + (len(actions) - 1) * gap
        x = center - span / 2
        revision = next((r for r in ex['revisions'] if r['t'] == t), None)
        if revision:
            self.rect(root, f'Annotation / {revision["kind"]} plan at t={t}',
                      x - 6, y - 6, span + 12, height + 12, AMBER_BG, AMBER, 1.5)
            self.label(root, 'Revised plan' if revision['kind'] == 'revised' else 'Plan extended',
                        center, y - 31, 18, AMBER, True, align='center')
        # Amber on the old proposal is a retrospective annotation of changes
        # in the next displayed planning round. All original actions stay visible.
        changed_old = set()
        if t + 1 < ex['end']:
            next_actions = ex['rounds'][t + 1]['plan']
            for tag, a0, a1, _, _ in difflib.SequenceMatcher(
                    a=actions[1:], b=next_actions, autojunk=False).get_opcodes():
                if tag in ('replace', 'delete'):
                    changed_old.update(range(a0 + 1, a1 + 1))
        nodes = []
        for j, action in enumerate(actions):
            marked = j in changed_old
            stroke = BLUE if j == 0 else AMBER if marked else GRAY
            color = WHITE if j == 0 else AMBER if marked else MUTED
            node = self.block(root, f't={t} / proposed action {j + 1} / {action}',
                              x + j * (width + gap), y, width, height,
                              BLUE if j == 0 else WHITE, stroke)
            if action.startswith('click '):
                _, rr, cc = action.split()
                self.label(node, 'click', width / 2, 5, 16, color, j == 0, align='center')
                self.label(node, f'{rr}, {cc}', width / 2, 29, 16,
                            color, j == 0, align='center', maxw=width - 5)
            else:
                self.label(node, action, width / 2, 17, 17,
                            color, j == 0, align='center', maxw=width - 5)
            nodes.append(gid(node['guid']))
        if changed_old and revision is None:
            self.label(root, 'Revised next round', center, y - 31, 16,
                        AMBER, align='center', maxw=264)
        self.plan_records.append(dict(task_uid=ex['task_uid'], t=t, actions=actions,
                                      executed=ex['rounds'][t]['executed'], nodes=nodes,
                                      changed_old_action_indices=sorted(changed_old)))
        return x + width / 2, y + height


def draw_figure(b, page, x, y, ex):
    root = b.frame(page, PREFIX + ex['key'], x, y, W, H, BG)
    b.label(root, 'Goal  G{NL}', 30, 27, 25, bold=True)
    lines, line = [], ''
    for word in ex['goal_text'].split():
        trial = (line + ' ' + word).strip()
        if b.measure(trial, 23) > 720:
            lines.append(line)
            line = word
        else:
            line = trial
    lines.append(line)
    if ex['key'] == 'sand':
        lines = ['Make the block of sand look like a', 'complete rectangle without holes.']
    assert len(lines) <= 2 and ' '.join(lines) == ex['goal_text']
    for j, line in enumerate(lines):
        b.label(root, line, 30, 69 + 32 * j, 23, maxw=720)
    b.label(root, f'Text goal · recorded excerpt t = {ex["start"]}–{ex["end"]}',
            30, 139, 19, MUTED)
    model = b.block(root, 'Fixed learned model across the episode',
                    810, 31, 635, 106, BLUE_BG, BLUE)
    b.label(model, 'Same learned model (P, K) at every step', 24, 16,
            25, BLUE, True, maxw=590)
    b.label(model, 'P encodes the state; K guides the LLM plan.', 24, 57,
            22, BLUE, maxw=590)
    b.line(root, 'Task / episode divider', (30, 178), (1445, 178), RULE, 1.2)

    centers = [162 + 292 * i for i in range(5)]
    for i, (t, center) in enumerate(zip(ex['displayed_steps'], centers)):
        reached = ex['checks'][t]
        b.label(root, f't = {t}', center, 194, 25, GREEN if reached else INK,
                True, align='center')
        b.snapshot(root, ex, t, center - 115, 234, 230, i == 0)
        b.label(root, scene_caption(ex, t), center, 491, 16,
                GREEN if reached else MUTED, align='center', maxw=264)
        b.label(root, 'Goal? Yes → stop' if reached else 'Goal? No → replan',
                center, 526, 22 if reached else 20, GREEN if reached else INK,
                reached, align='center')
        b.outcome_records.append(dict(task_uid=ex['task_uid'], t=t, reached=reached))
        if i < 4:
            b.label(root, f'P(X{subscript(t)}) → LLM + K + G{{NL}}',
                    center, 565, 18, BLUE, align='center', maxw=268)
            sx, sy = b.plan(root, ex, t, center, 632)
            b.route(root, f't={t} / execute only {ex["rounds"][t]["executed"]}',
                    [(sx, sy), (sx, 744), (center + 146, 744),
                     (center + 146, 349), (centers[i + 1] - 115, 349)], BLUE, 2.3)
        else:
            success = b.block(root, f'Goal checker / success at step {t}',
                              center - 126, 625, 252, 94, GREEN_BG, GREEN)
            b.label(success, 'Goal satisfied', 126, 12, 26, GREEN, True, align='center')
            b.label(success, f'Success at step {t}', 126, 53, 21, GREEN, align='center')
    return root


def remove_previous(source):
    names = {PREFIX + s['key'] for s in SPECS}
    children, removed = collections.defaultdict(list), set()
    for n in source['nodeChanges']:
        if 'parentIndex' in n:
            children[gid(n['parentIndex']['guid'])].append(n)
    def mark(n):
        removed.add(gid(n['guid']))
        for child in children[gid(n['guid'])]:
            mark(child)
    for n in source['nodeChanges']:
        if n.get('name') in names:
            mark(n)
    result = copy.deepcopy(source)
    result['nodeChanges'] = [n for n in result['nodeChanges'] if gid(n['guid']) not in removed]
    return result, sorted(removed)


def validate(saved, source, b, roots, examples):
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
    root_ids, text_count = {gid(r['guid']) for r in roots}, 0
    for original in b.nodes:
        n = byid[gid(original['guid'])]
        if gid(n['guid']) in root_ids:
            continue
        tf, current = Affine2D(), n
        while gid(current['guid']) not in root_ids:
            tf += matrix(current)
            current = byid[gid(current['parentIndex']['guid'])]
        w, h = n['size']['x'], n['size']['y']
        corners = tf.transform([(0, 0), (w, 0), (w, h), (0, h)])
        assert min(p[0] for p in corners) >= -.02 and max(p[0] for p in corners) <= W + .02
        assert min(p[1] for p in corners) >= -.02 and max(p[1] for p in corners) <= H + .02
        if n['type'] == 'TEXT':
            text_count += 1
            s, glyphs = n['textData']['characters'], n['derivedTextData']['glyphs']
            assert len(glyphs) == len(s) and [g['firstCharacter'] for g in glyphs] == list(range(len(s)))
            assert s != 'Planning evaluation' and not s.startswith('Blue action:')
    ex_by_task = {e['task_uid']: e for e in examples}
    assert len(b.plan_records) == 12 and len(b.outcome_records) == len(b.grid_records) == 15
    for p in b.plan_records:
        r = ex_by_task[p['task_uid']]['rounds'][p['t']]
        assert p['actions'] == r['plan'] and p['executed'] == r['executed']
    for outcome in b.outcome_records:
        assert outcome['reached'] == ex_by_task[outcome['task_uid']]['checks'][outcome['t']]
    count = 0
    for record in b.grid_records:
        state = ex_by_task[record['task_uid']]['states'][record['t']]
        assert record['full_grid'] == state
        cells = children[tuple(record['node'])]
        assert len(cells) == len(state) ** 2
        for cell in cells:
            m = re.fullmatch(r'Cell \((\d+), (\d+)\) / (\w+)', cell['name'])
            assert m and state[int(m[1])][int(m[2])] == m[3]
            color = cell['fillPaints'][0]['color']
            assert all(abs(color[k] - v) < 1e-7 for k, v in zip('rgba', to_rgba(m[3])))
        count += len(cells)
    assert count == 3220
    return dict(original_nodes_preserved=len(source['nodeChanges']),
                original_blobs_preserved=len(source['blobs']), new_nodes=len(b.nodes),
                editable_text_layers=text_count, all_new_elements_within_bounds=True,
                valid_blob_references=len(refs), native_game_cells=count,
                displayed_plans_and_actions_verified=12, displayed_goal_checks_verified=15,
                all_episode_goal_checks_independently_recomputed=True,
                three_focus_revisions_verified=True, original_step_numbers_preserved=True,
                native_round_trip='passed', figma_application_import_tested=False)


def comparison(out):
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 25)
    sheet = Image.new('RGB', (W + 32, (H + 66) * 3 + 16), '#e8eaec')
    draw = ImageDraw.Draw(sheet)
    titles = ['SET — change which cards to select', 'Mario — revise a future action',
              'Sand — change click locations and extend the plan']
    for i, (spec, title) in enumerate(zip(SPECS, titles)):
        y = 14 + i * (H + 66)
        draw.text((17, y), f'{i + 1}. {title}', fill=INK, font=font)
        sheet.paste(Image.open(out / f'{PREFIX}{spec["key"]}.png').convert('RGB'), (16, y + 44))
    sheet.save(out / 'planning_replanning_rich_three_options.png')


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
    with tempfile.TemporaryDirectory(prefix='planning-replanning-rich-') as directory:
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
        b = RichBuilder(source, args.font_dir)
        roots = [draw_figure(b, page, x + i * (W + 120), y, ex) for i, ex in enumerate(examples)]
        message = {k: copy.deepcopy(v) for k, v in source.items()
                   if k in ('type', 'sessionID', 'ackID', 'nodeChangeOrder', 'nodeChanges', 'blobs')}
        message['nodeChanges'] += b.nodes
        message['blobs'] = b.blobs
        (tmp / 'combined.json').write_text(json.dumps(message))
        subprocess.run([bun, str(IO), 'encode', str(tmp / 'reference.canvas'),
                        str(tmp / 'combined.json'), str(tmp / 'combined.canvas'), str(args.modules)], check=True)
        saved = json.loads((tmp / 'combined.canvas.json').read_text())
        checks = validate(saved, source, b, roots, examples)
        for r in roots:
            render(saved, gid(r['guid']), args.out / f'{r["name"]}.png',
                   args.out / f'{r["name"]}.pdf', args.out / f'{r["name"]}.svg')
        comparison(args.out)
        staged = args.out / 'learning_evolution_fig_with_rich_replanning.fig'
        package(args.source, tmp / 'combined.canvas', staged,
                args.out / f'{roots[0]["name"]}.png', roots[0])
        report = dict(source=str(args.source.relative_to(ROOT)), source_sha256_before=before,
                      staged_file=str(staged.relative_to(ROOT)), staged_sha256=sha(staged),
                      frames=[dict(name=r['name'], id=gid(r['guid']), position=r['transform'], size=r['size']) for r in roots],
                      checks=checks, examples=examples, replaced_previous_option_nodes=removed,
                      plans=b.plan_records, outcomes=b.outcome_records, grids=b.grid_records,
                      connectors=b.connectors, new_native_node_ids=[gid(n['guid']) for n in b.nodes],
                      applied_to_source=False)
        if args.apply:
            assert sha(args.source) == before
            backup = args.out / 'learning_evolution_fig_before_rich_replanning.fig'
            if not backup.exists():
                shutil.copy2(args.source, backup)
            pending = args.source.with_name(args.source.name + '.tmp')
            shutil.copy2(staged, pending)
            pending.replace(args.source)
            report.update(applied_to_source=True, backup=str(backup.relative_to(ROOT)),
                          source_sha256_after=sha(args.source))
        (args.out / 'planning_replanning_rich_evidence.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: report[k] for k in ('frames', 'checks', 'applied_to_source')}, indent=2))


if __name__ == '__main__':
    main()
