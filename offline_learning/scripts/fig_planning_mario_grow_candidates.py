#!/usr/bin/env python3
"""Build two editable comparison boards: five Mario and five Grow rollouts.

Each option contains consecutive recorded observations and exact logged plans.
The original native Figma file is changed only with --apply.
"""
from __future__ import annotations

import argparse
import collections
import copy
import datetime
import difflib
import io
import itertools
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
    RolloutBuilder, SOURCE, IO, sha, gid, matrix, render,
    BG, BLUE, BLUE_BG, GREEN, GREEN_BG, INK, MUTED, RULE, WHITE, GRAY,
)
from offline_learning.scripts.fig_planning_replanning import AMBER, AMBER_BG
from offline_learning.scripts.fig_planning_replanning_rich import subscript
from offline_learning.planning_nl_goals import get_python_goal, freeze_grid
from matplotlib.colors import to_rgba
from matplotlib.transforms import Affine2D
from PIL import Image

OUT = SOURCE.parent / 'planning_mario_grow_candidates'
PREFIX = 'planning_candidates_'
W, H = 1840, 832
PAD, BAND, GAP = 24, 44, 28
BOARD_W = W + PAD * 2
BOARD_H = PAD * 2 + 5 * (H + BAND) + 4 * GAP
RUNS = {
    'frame': '2026-08-30/planning_v2_online_ds',
    'nl': '2026-09-01/planning_v2_online_ds_nl',
    'opus': '2026-09-02/planning_v2_online_opus5_nl',
    'cap': '2026-09-03/planning_v2_online_ds_percap_nl',
}
SPECS = [
    dict(key='mario_1', game='Mario', task='n2ntd:high-ground:s0', run='opus',
         start=0, end=6, focus=1, title='Highest platform: insert another up',
         old=['right', 'right', 'noop'], new=['up', 'right', 'right', 'noop', 'noop'],
         recommendation='Best complete example: an immediate plan change and visible ascent, ending in success.'),
    dict(key='mario_2', game='Mario', task='n2ntd:coin-air:s0', run='cap',
         start=1, end=5, focus=4, title='Airborne coin: add one more step',
         old=[], new=['noop'],
         recommendation='A compact example of checking the goal after a plan is exhausted and adding a noop.'),
    dict(key='mario_3', game='Mario', task='n2ntd:coin-ground:s0', run='cap',
         start=5, end=9, focus=8, title='Lowest coin: wait, move, then check again',
         old=[], new=['noop'],
         recommendation='The player descends toward the lowest coin; one extra round completes collection.'),
    dict(key='mario_4', game='Mario', task='n2ntd:all-coins:s0', run='cap',
         start=21, end=25, focus=23, title='All coins: replace noop with down',
         old=['noop'], new=['down'],
         recommendation='A particularly simple one-action replacement, followed by removal of the last coin.'),
    dict(key='mario_5', game='Mario', task='n2ntd:all-coins-kill:s0', run='opus',
         start=46, end=50, focus=47, title='Enemy: replace waiting with a shot',
         old=['noop', 'noop'], new=['click 2 8'],
         recommendation='Most visually varied: a click, purple bullets, a moving enemy, and success.'),
    dict(key='grow_1', game='Grow', task='7xf97:move-cloud:s101', run='opus',
         start=0, end=3, focus=2, title='Cloud goal: shorten the proposed remainder',
         old=['left', 'left'], new=['left'],
         recommendation='The only successful Grow task with a changed plan; shows both lengthening and shortening.'),
    dict(key='grow_2', game='Grow', task='7xf97:bloom-purple:s101', run='frame',
         start=41, end=45, focus=42, title='Falling water: replace a wait with a click',
         old=['noop'] * 8, new=['click 13 13'] + ['noop'] * 7,
         recommendation='Richest Grow scene: four falling water cells, the sun, the cloud, and eight plants.'),
    dict(key='grow_3', game='Grow', task='7xf97:water-left-plant:s101', run='cap',
         start=16, end=20, focus=17, title='Watering task: change movement to a click',
         old=['right'], new=['click 0 12', 'left'],
         recommendation='A clear change in action type, followed by further cloud movement; the goal remains unmet.'),
    dict(key='grow_4', game='Grow', task='7xf97:bloom-purple:s101', run='nl',
         start=28, end=32, focus=29, title='Bloom task: reverse the cloud direction',
         old=['right'], new=['left', 'left'],
         recommendation='The easiest Grow direction change to read: a proposed right becomes two left actions.'),
    dict(key='grow_5', game='Grow', task='7xf97:shower-drain:s101', run='opus',
         start=34, end=38, focus=35, title='Shower task: change waiting to movement',
         old=['noop'] * 3, new=['right'] * 3,
         recommendation='Shows the alternative stopping condition: the goal is still false when the action limit is reached.'),
]


def inventory():
    """Audit all available game-specific online logs, including excluded runs."""
    result = []
    for path in sorted((ROOT / 'logs').glob('*/planning_v2_online*/*/online.json')):
        if path.parent.name not in ('n2ntd', '7xf97'):
            continue
        d = json.loads(path.read_text())
        excluded = any(s in str(path) for s in ('_icl', '_abl_'))
        for row in d['rows']:
            for i, attempt in enumerate(row.get('lmwm', {}).get('attempts', [])):
                rr = attempt['rounds']
                changes = [t for t in range(1, len(rr))
                           if rr[t]['plan'] != rr[t - 1]['plan'][1:]]
                immediate = [t for t in changes if rr[t - 1]['plan'][1:]
                             and rr[t]['plan'][0] != rr[t - 1]['plan'][1]]
                result.append(dict(rollout=str(path.relative_to(ROOT)), task_uid=row['task_uid'],
                                   attempt=i, success=attempt['success'], steps=len(rr),
                                   revision_steps=changes, changed_next_action_steps=immediate,
                                   selected_pool=not excluded,
                                   exclusion='ICL comparison or ablation run' if excluded else None))
    return result


def load_examples():
    examples = []
    for spec in SPECS:
        path = ROOT / 'logs' / RUNS[spec['run']] / spec['task'].split(':')[0] / 'online.json'
        d = json.loads(path.read_text())
        problem_path = ROOT / d['config']['problems']
        problem = next(p for p in json.loads(problem_path.read_text())['problems']
                       if p['task_uid'] == spec['task'])
        row = next(r for r in d['rows'] if r['task_uid'] == spec['task'])
        mode = row.get('goal_presentation') or row.get('eval_goal_mode')
        assert mode in ('nl', 'python', 'frame', 'exact_frame'), mode
        presentation = 'frame' if mode in ('frame', 'exact_frame') else 'nl'
        attempt = row['lmwm']['attempts'][0]
        assert attempt['start_match'] and d['config']['warm_start']
        rounds = [{k: r[k] for k in ('n', 'plan', 'executed', 'reached_goal', 'z_after')}
                  for r in attempt['rounds']]
        assert [r['n'] for r in rounds] == list(range(len(rounds)))
        assert all(r['plan'] and r['executed'] == r['plan'][0] for r in rounds)
        states = [problem['start']] + [json.loads(r['grid_after']) for r in attempt['rounds']]
        if presentation == 'frame':
            checks = [s == problem['goal'] for s in states]
        else:
            checker = get_python_goal(problem['nl_checker'])
            checks = [bool(checker.check([freeze_grid(s) for s in states[:t + 1]],
                                         [r['executed'] for r in rounds[:t]]))
                      for t in range(len(states))]
        assert checks[1:] == [r['reached_goal'] for r in rounds]
        assert checks == [False] * len(rounds) + [bool(attempt['success'])]
        cap = d['config'].get('action_caps', {}).get(spec['task'], d['config']['max_actions'])
        assert attempt['rounds'][0]['remaining'] == cap
        assert attempt['actions_used'] == len(rounds) <= cap
        if attempt['success']:
            assert attempt['reached_at'] == len(rounds) == spec['end']
        else:
            assert attempt['failed_reason'] == 'budget-exhausted' and len(rounds) == cap
        revisions = []
        for t in range(spec['start'] + 1, spec['end']):
            old, new = rounds[t - 1]['plan'][1:], rounds[t]['plan']
            if old != new:
                kind = ('extended' if not old else
                        'shortened' if len(new) < len(old) and new == old[:len(new)] else
                        'lengthened' if len(new) > len(old) and old == new[:len(old)] else 'revised')
                revisions.append(dict(t=t, old_remaining=old, new_plan=new, kind=kind))
        focus = next(r for r in revisions if r['t'] == spec['focus'])
        assert focus['old_remaining'] == spec['old'] and focus['new_plan'] == spec['new']
        examples.append(dict(**spec, task_uid=spec['task'], arm='lmwm', attempt=0,
                             presentation=presentation, original_mode=mode,
                             goal_text=problem['nl_goal'], goal_checker=problem['nl_checker'],
                             goal=problem['goal'], states=states, rounds=rounds, checks=checks,
                             displayed_steps=list(range(spec['start'], spec['end'] + 1)),
                             revisions=revisions, success=attempt['success'],
                             reached_at=attempt['reached_at'], total_steps=len(rounds),
                             failed_reason=attempt['failed_reason'], action_cap=cap,
                             prefix=problem.get('prefix', []), warm_start=True,
                             quiescence_waived=attempt.get('quiescence_waived'), run_config=d['config'],
                             problems=str(problem_path.relative_to(ROOT)), problems_sha256=sha(problem_path),
                             rollout=str(path.relative_to(ROOT)), rollout_sha256=sha(path)))
    return examples


def caption(ex, t):
    state = ex['states'][t]
    count = collections.Counter(c for row in state for c in row)
    if ex['key'] == 'mario_1':
        pos = [(r, c) for r, row in enumerate(state) for c, color in enumerate(row) if color == 'red']
        assert len(pos) == 1
        return 'Highest platform reached' if ex['checks'][t] else f'Player at ({pos[0][0]}, {pos[0][1]})'
    if ex['key'] == 'mario_5':
        assert count['gold'] == 0
        return 'Enemy gone' if count['blue'] == 0 else f'Enemy visible · {count["mediumpurple"]} bullet' + ('s' if count['mediumpurple'] != 1 else '')
    if ex['game'] == 'Mario':
        return 'No coins remain' if count['gold'] == 0 else f'{count["gold"]} coin' + (' remains' if count['gold'] == 1 else 's remain')
    if ex['key'] == 'grow_2':
        blue = [(r, c) for r, row in enumerate(state) for c, color in enumerate(row) if color == 'blue']
        assert len(blue) == 4 and all(c == 13 for r, c in blue)
        return f'Lowest water cell: row {max(r for r, c in blue)}'
    if ex['key'] == 'grow_4':
        assert count['mediumpurple'] == 0 and count['green'] == 8
        return 'No purple bloom'
    if ex['key'] == 'grow_5':
        assert count['blue'] == 0 and count['green'] == 8
        return 'No water · no growth'
    if ex['key'] == 'grow_3':
        assert state[14][11] != 'green' and count['green'] == 8
        return 'Plant has not grown'
    cols = [c for row in state for c, color in enumerate(row) if color == 'gray']
    return f'Cloud left edge: column {min(cols)}'


class CandidateBuilder(RolloutBuilder):
    def snapshot(self, root, ex, t, center, width, row_axes):
        x, y = center - width / 2, 368 - width / 2
        state = ex['states'][t]
        self.full_grid(root, state, x, y, width, ex['task_uid'], f'Recorded X{subscript(t)}')
        self.grid_records[-1].update(key=ex['key'], t=t)
        cell = width / len(state)
        fs = 10.5 if len(state) == 16 else 11.5
        for c in range(len(state)):
            self.label(root, str(c), x + (c + .5) * cell, y + width + 4, fs, MUTED, align='center')
        if row_axes:
            for r in range(len(state)):
                self.label(root, str(r), x - 7, y + r * cell + (cell - fs * 1.2) / 2,
                           fs, MUTED, align='right')

    def plan(self, root, ex, t, center, available):
        actions = ex['rounds'][t]['plan']
        changed_old = set()
        next_revision = next((r for r in ex['revisions'] if r['t'] == t + 1), None)
        if next_revision:
            old, new = actions[1:], ex['rounds'][t + 1]['plan']
            if len(old) == len(new):
                # Compare equal-length plans by position: repeated noops must
                # not make a first-action replacement look like a last-action edit.
                changed_old.update(j + 1 for j, (a, z) in enumerate(zip(old, new)) if a != z)
            else:
                for tag, a0, a1, _, _ in difflib.SequenceMatcher(
                        a=old, b=new, autojunk=False).get_opcodes():
                    if tag in ('replace', 'delete'):
                        changed_old.update(range(a0 + 1, a1 + 1))
                if old and new and old[0] != new[0]:
                    changed_old.add(1)
        # Keep the executed first action separate; exact repeat counts preserve
        # every action in long plans without tiny chips or omitted continuations.
        tokens = [(actions[0], [0])]
        if len(actions) > 5 or len(actions) * 48 + (len(actions) - 1) * 6 > available:
            for (action, _marked), group in itertools.groupby(
                    enumerate(actions[1:], 1), key=lambda p: (p[1], p[0] in changed_old)):
                tokens.append((action, [i for i, _ in group]))
        else:
            tokens += [(action, [i]) for i, action in enumerate(actions[1:], 1)]
        assert [action for action, ids in tokens for _ in ids] == actions
        gap, height = 6, 56
        width = min(66, (available - gap * (len(tokens) - 1)) / len(tokens))
        assert width >= 42, (ex['key'], t, width)
        span = len(tokens) * width + (len(tokens) - 1) * gap
        x, y = center - span / 2, 664
        revision = next((r for r in ex['revisions'] if r['t'] == t), None)
        if revision:
            self.rect(root, f'Annotation / changed plan at t={t}', x - 6, y - 6,
                      span + 12, height + 12, AMBER_BG, AMBER, 1.5)
            names = dict(extended='Plan extended', shortened='Plan shortened',
                         lengthened='Plan lengthened', revised='Revised plan')
            self.label(root, names[revision['kind']], center, 628, 18, AMBER, True, align='center')
        elif next_revision:
            self.label(root, 'Plan extended next' if next_revision['kind'] == 'extended' else 'Revised next round',
                       center, 628, 17, AMBER, align='center')
        rendered = []
        for i, (action, ids) in enumerate(tokens):
            first, marked = i == 0, any(j in changed_old for j in ids)
            color = WHITE if first else AMBER if marked else MUTED
            stroke = BLUE if first else AMBER if marked else GRAY
            node = self.block(root, f't={t} / plan positions {ids} / {action}',
                              x + i * (width + gap), y, width, height,
                              BLUE if first else WHITE, stroke)
            if action.startswith('click '):
                _, r, c = action.split()
                self.label(node, 'click', width / 2, 5, 16, color, first, align='center')
                self.label(node, f'{r}, {c}', width / 2, 29, 16, color, first, align='center')
                assert len(ids) == 1
            else:
                fs = 17 if width >= 54 else 16
                self.label(node, action, width / 2, 6 if len(ids) > 1 else 17,
                           fs, color, first, align='center', maxw=width - 4)
                if len(ids) > 1:
                    self.label(node, f'×{len(ids)}', width / 2, 30, 16, color, align='center')
            rendered.append(dict(node=gid(node['guid']), action=action, positions=ids,
                                 count=len(ids), executed=first))
        self.plan_records.append(dict(key=ex['key'], task_uid=ex['task_uid'], t=t,
                                      actions=actions, executed=actions[0], tokens=rendered,
                                      changed_old_action_indices=sorted(changed_old)))
        return x + width / 2, y + height


def wrap(b, text, width, size=23):
    lines, line = [], ''
    for word in text.split():
        trial = (line + ' ' + word).strip()
        if b.measure(trial, size) > width:
            lines.append(line)
            line = word
        else:
            line = trial
    lines.append(line)
    assert len(lines) <= 2 and ' '.join(lines) == text
    return lines


def draw_option(b, board, y, ex):
    root = b.frame(board, PREFIX + ex['key'], PAD, y, W, H, BG)
    gx = 30
    if ex['presentation'] == 'frame':
        b.full_grid(root, ex['goal'], 30, 24, 128, ex['task_uid'], 'Given target X_G')
        b.grid_records[-1].update(key=ex['key'], t='goal')
        gx = 185
    b.label(root, 'Target state  X{G}' if ex['presentation'] == 'frame' else 'Goal  G{NL}',
            gx, 27, 25, bold=True)
    for i, line in enumerate(wrap(b, ex['goal_text'], 1020 - gx)):
        b.label(root, line, gx, 69 + i * 32, 23)
    mode = 'Planner receives P(X{G})' if ex['presentation'] == 'frame' else 'Planner receives this text'
    extent = 'full episode' if ex['start'] == 0 and ex['end'] == ex['total_steps'] else 'excerpt'
    b.label(root, f'{mode} · {extent} t = {ex["start"]}–{ex["end"]}', gx, 139, 18, MUTED)
    model = b.block(root, 'Fixed learned model', 1080, 31, 725, 106, BLUE_BG, BLUE)
    b.label(model, 'Same learned model (P, K) at every step', 24, 16, 25, BLUE, True)
    b.label(model, 'P encodes the state; K guides the LLM plan.', 24, 57, 22, BLUE)
    b.line(root, 'Task / trajectory divider', (30, 178), (1805, 178), RULE, 1.2)
    steps = ex['displayed_steps']
    spacing = (W - 80) / len(steps)
    centers = [40 + (i + .5) * spacing for i in range(len(steps))]
    grid_width = min(260, spacing - 42)
    for i, (t, center) in enumerate(zip(steps, centers)):
        reached, at_limit = ex['checks'][t], t == ex['action_cap'] and not ex['checks'][t]
        color = GREEN if reached else AMBER if at_limit else INK
        b.label(root, f't = {t}' + (' · start' if t == 0 else ''), center, 198,
                24, color, True, align='center')
        b.snapshot(root, ex, t, center, grid_width, i == 0)
        b.label(root, caption(ex, t), center, 525, 16 if len(steps) == 7 else 18,
                GREEN if reached else MUTED, align='center', maxw=spacing - 14)
        goal_label = ('Initial observation X₀' if t == 0 else 'Goal? Yes → stop' if reached
                      else 'Goal? No → limit' if at_limit else 'Goal? No → replan')
        b.label(root, goal_label, center, 560, 19 if len(steps) == 7 else 22,
                color, reached, align='center')
        b.outcome_records.append(dict(key=ex['key'], t=t, reached=reached,
                                      at_limit=at_limit, initial=t == 0))
        if i < len(steps) - 1:
            goal = 'P(X{G})' if ex['presentation'] == 'frame' else 'G{NL}'
            b.label(root, f'P(X{subscript(t)}) → LLM + K + {goal}', center, 597,
                    15.5 if len(steps) == 7 else 19, BLUE, align='center', maxw=spacing - 10)
            sx, sy = b.plan(root, ex, t, center, spacing - 28)
            b.route(root, f'{ex["key"]} / t={t} / execute {ex["rounds"][t]["executed"]}',
                    [(sx, sy), (sx, 802), (center + spacing / 2, 802),
                     (center + spacing / 2, 368), (centers[i + 1] - grid_width / 2, 368)], BLUE, 2.3)
        else:
            cw = min(300, spacing - 22)
            title, detail = ('Goal satisfied', f'Success at step {t}') if reached else (
                ('Step limit reached', f'Unsolved at step {t}') if at_limit else
                ('Continue planning', f'Run ends unmet at t = {ex["total_steps"]}'))
            card = b.block(root, f'Outcome / {title}', center - cw / 2, 658, cw, 100,
                           GREEN_BG if reached else AMBER_BG, GREEN if reached else AMBER)
            b.label(card, title, cw / 2, 13, 23 if len(steps) == 7 else 25,
                    GREEN if reached else AMBER, True, align='center', maxw=cw - 10)
            b.label(card, detail, cw / 2, 58, 18 if len(steps) == 7 else 19,
                    GREEN if reached else AMBER, align='center', maxw=cw - 10)
    return root


def remove_previous(original):
    children, removed = collections.defaultdict(list), set()
    for n in original['nodeChanges']:
        if 'parentIndex' in n:
            children[gid(n['parentIndex']['guid'])].append(n)
    def mark(n):
        removed.add(gid(n['guid']))
        for c in children[gid(n['guid'])]:
            mark(c)
    for n in original['nodeChanges']:
        if n.get('name') in (PREFIX + 'mario_five_options', PREFIX + 'grow_five_options'):
            mark(n)
    result = copy.deepcopy(original)
    result['nodeChanges'] = [n for n in result['nodeChanges'] if gid(n['guid']) not in removed]
    return result, sorted(removed)


def validate(saved, source, b, boards, options, examples):
    byid = {gid(n['guid']): n for n in saved['nodeChanges']}
    assert len(byid) == len(saved['nodeChanges'])
    for n in source['nodeChanges']:
        assert byid[gid(n['guid'])] == n
    assert saved['blobs'][:len(source['blobs'])] == source['blobs']
    children, refs = collections.defaultdict(list), []
    for n in saved['nodeChanges']:
        if 'parentIndex' in n:
            assert gid(n['parentIndex']['guid']) in byid
            children[gid(n['parentIndex']['guid'])].append(n)
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
    assert all(0 <= r < len(saved['blobs']) for r in refs)
    roots = {gid(n['guid']): n for n in boards + options}
    text_count = 0
    for original in b.nodes:
        n = byid[gid(original['guid'])]
        if gid(n['guid']) in roots:
            continue
        current, tf = n, Affine2D()
        while gid(current['guid']) not in roots:
            tf += matrix(current)
            current = byid[gid(current['parentIndex']['guid'])]
        width, height = current['size']['x'], current['size']['y']
        nw, nh = n['size']['x'], n['size']['y']
        corners = tf.transform([(0, 0), (nw, 0), (nw, nh), (0, nh)])
        assert min(p[0] for p in corners) >= -.03 and max(p[0] for p in corners) <= width + .03, n['name']
        assert min(p[1] for p in corners) >= -.03 and max(p[1] for p in corners) <= height + .03, n['name']
        if n['type'] == 'TEXT':
            text_count += 1
            s, glyphs = n['textData']['characters'], n['derivedTextData']['glyphs']
            assert len(glyphs) == len(s) and [g['firstCharacter'] for g in glyphs] == list(range(len(s)))
            assert s != 'Planning evaluation' and not s.startswith('Blue action:')
    for option in options:
        assert option['transform']['m02'] == PAD
        assert option['transform']['m12'] + H <= BOARD_H - PAD
    lookup = {e['key']: e for e in examples}
    for record in b.plan_records:
        expected = lookup[record['key']]['rounds'][record['t']]
        assert record['actions'] == expected['plan'] and record['executed'] == expected['executed']
        assert [token['action'] for token in record['tokens'] for _ in range(token['count'])] == expected['plan']
        assert sum(token['executed'] for token in record['tokens']) == 1
        assert record['tokens'][0]['count'] == 1
    for record in b.outcome_records:
        assert record['reached'] == lookup[record['key']]['checks'][record['t']]
    cell_count = 0
    for record in b.grid_records:
        ex = lookup[record['key']]
        state = ex['goal'] if record['t'] == 'goal' else ex['states'][record['t']]
        assert record['full_grid'] == state and record['crop'] is None
        cells = children[tuple(record['node'])]
        assert len(cells) == len(state) ** 2
        for n in cells:
            m = re.fullmatch(r'Cell \((\d+), (\d+)\) / (\w+)', n['name'])
            assert m and state[int(m[1])][int(m[2])] == m[3]
            color = n['fillPaints'][0]['color']
            assert all(abs(color[k] - v) < 1e-7 for k, v in zip('rgba', to_rgba(m[3])))
        cell_count += len(cells)
    assert len(options) == 10 and len(boards) == 2
    assert len(b.plan_records) == sum(e['end'] - e['start'] for e in examples)
    assert len(b.outcome_records) == sum(len(e['displayed_steps']) for e in examples)
    return dict(original_nodes_preserved=len(source['nodeChanges']), original_blobs_preserved=len(source['blobs']),
                new_nodes=len(b.nodes), editable_text_layers=text_count, native_game_cells=cell_count,
                valid_blob_references=len(refs), all_new_elements_within_bounds=True,
                displayed_plans_and_actions_verified=len(b.plan_records),
                displayed_states_and_goal_checks_verified=len(b.outcome_records),
                full_episode_goal_checks_independently_recomputed=True, ten_focus_revisions_verified=True,
                repeat_count_expansion_verified=True, original_step_numbers_preserved=True,
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
        meta['client_meta']['render_coordinates'] = dict(x=root['transform']['m02'], y=root['transform']['m12'],
                                                        width=root['size']['x'], height=root['size']['y'])
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
    before, examples, audited = sha(args.source), load_examples(), inventory()
    bun = shutil.which('bun') or '/home/ays57/.bun/bin/bun'
    with tempfile.TemporaryDirectory(prefix='planning-mario-grow-') as directory:
        tmp = Path(directory)
        with zipfile.ZipFile(args.source) as z:
            (tmp / 'reference.canvas').write_bytes(z.read('canvas.fig'))
        subprocess.run([bun, str(IO), 'decode', str(tmp / 'reference.canvas'), str(tmp / 'source.json'), '-', str(args.modules)], check=True)
        source, removed = remove_previous(json.loads((tmp / 'source.json').read_text()))
        page = next(n for n in source['nodeChanges'] if n['type'] == 'CANVAS' and n['name'] == 'Figures')
        top = [n for n in source['nodeChanges'] if n.get('parentIndex', {}).get('guid') == page['guid']]
        x = min(n['transform']['m02'] for n in top)
        y = max(n['transform']['m12'] + n['size']['y'] for n in top) + 160
        b = CandidateBuilder(source, args.font_dir)
        # Keep new page-level ordering positions distinct from existing layers.
        positions = [int(n['parentIndex']['position'][1:], 16) for n in top
                     if re.fullmatch(r'![0-9a-f]+', n['parentIndex']['position'])]
        b.positions[gid(page['guid'])] = max(positions, default=-1) + 1
        boards, options = [], []
        for j, game in enumerate(('Mario', 'Grow')):
            board = b.frame(page, PREFIX + game.lower() + '_five_options',
                            x + j * (BOARD_W + 120), y, BOARD_W, BOARD_H, '#e8eaec')
            boards.append(board)
            for i, ex in enumerate(e for e in examples if e['game'] == game):
                top_y = PAD + i * (H + BAND + GAP)
                b.label(board, f'{i + 1}. {ex["title"]}', PAD + 6, top_y + 3, 25, INK, True)
                status = f'Success at step {ex["total_steps"]}' if ex['success'] else f'Goal unmet after {ex["total_steps"]} steps'
                b.label(board, status, BOARD_W - PAD - 6, top_y + 8, 20,
                        GREEN if ex['success'] else AMBER, align='right')
                options.append(draw_option(b, board, top_y + BAND, ex))
        message = {k: copy.deepcopy(v) for k, v in source.items()
                   if k in ('type', 'sessionID', 'ackID', 'nodeChangeOrder', 'nodeChanges', 'blobs')}
        message['nodeChanges'] += b.nodes
        message['blobs'] = b.blobs
        (tmp / 'combined.json').write_text(json.dumps(message))
        subprocess.run([bun, str(IO), 'encode', str(tmp / 'reference.canvas'), str(tmp / 'combined.json'),
                        str(tmp / 'combined.canvas'), str(args.modules)], check=True)
        saved = json.loads((tmp / 'combined.canvas.json').read_text())
        checks = validate(saved, source, b, boards, options, examples)
        for root in options + boards:
            print('Rendering ' + root['name'], flush=True)
            render(saved, gid(root['guid']), args.out / (root['name'] + '.png'),
                   args.out / (root['name'] + '.pdf'), args.out / (root['name'] + '.svg'))
        staged = args.out / 'learning_evolution_fig_with_mario_grow_candidates.fig'
        package(args.source, tmp / 'combined.canvas', staged, args.out / (boards[0]['name'] + '.png'), boards[0])
        report = dict(source=str(args.source.relative_to(ROOT)), source_sha256_before=before,
                      staged_file=str(staged.relative_to(ROOT)), staged_sha256=sha(staged),
                      boards=[dict(name=n['name'], id=gid(n['guid']), position=n['transform'], size=n['size']) for n in boards],
                      options=[dict(name=n['name'], id=gid(n['guid']), parent=gid(n['parentIndex']['guid']),
                                    position=n['transform'], size=n['size']) for n in options],
                      checks=checks, examples=examples, inventory=audited,
                      replaced_previous_nodes=removed, plans=b.plan_records, outcomes=b.outcome_records,
                      grids=b.grid_records, connectors=b.connectors,
                      new_native_node_ids=[gid(n['guid']) for n in b.nodes], applied_to_source=False)
        if args.apply:
            assert sha(args.source) == before
            backup = args.out / 'learning_evolution_fig_before_mario_grow_candidates.fig'
            if not backup.exists():
                shutil.copy2(args.source, backup)
            pending = args.source.with_name(args.source.name + '.tmp')
            shutil.copy2(staged, pending)
            pending.replace(args.source)
            report.update(applied_to_source=True, backup=str(backup.relative_to(ROOT)), source_sha256_after=sha(args.source))
        (args.out / 'planning_mario_grow_candidates_evidence.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: report[k] for k in ('boards', 'checks', 'applied_to_source')}, indent=2))


if __name__ == '__main__':
    main()
