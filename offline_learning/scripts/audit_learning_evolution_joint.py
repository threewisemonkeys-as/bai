"""Reproduce evidence and a frame preview for joint P/B learning examples.

Run from the repo: .venv/bin/python offline_learning/scripts/audit_learning_evolution_joint.py
Reads existing logs only; performs no model calls or simulator modifications.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from offline_learning.validate import run_perceive, strip_autumn_obs_metadata

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np

csv.field_size_limit(10**7)
BASE = ROOT / 'logs/2026-08-24/human_curated/rexpure'
OUT = ROOT / 'analysis/learning_example/audit'
SPECS = [
    dict(game='7www9', title='Magnets: a downward action produces a diagonal snap',
         drive='train_d1', steps=[32, 33, 34, 35], nodes=[1, 6, 8],
         labels=['two aligned cells\nrDist2 / rDist2',
                 'snapped down and right\nrDist1 / noRedRow',
                 'stable under noop\nrDist1 / noRedRow',
                 'moved clear of red rows\nnoRedRow / noRedRow']),
    dict(game='dino', title='Dino: fall, land, stay grounded, then jump',
         drive='train_d1', steps=[100, 101, 102, 103], nodes=[12, 19, 20],
         labels=['above the floor\nflags:green:bottom',
                 'landed\nflags:green:bottom|red:bottom',
                 'red stays; obstacles scroll\nflags:green:bottom|red:bottom',
                 'jumped six rows\nflags:green:bottom']),
]


def content_hash(*parts):
    h = hashlib.md5()
    for part in parts:
        h.update((part or '').encode())
        h.update(b'\x00')
    return h.hexdigest()[:16]


def load_pool(game):
    directory = BASE / f'{game}_s1'
    pool_path = directory / 'rexpure_run_seed1/candidates.jsonl'
    pool = {c['idx']: dict(c, source_line=i) for i, c in
            enumerate(map(json.loads, pool_path.read_text().splitlines()), 1)}
    process_path = pool_path.with_name('process_log.jsonl')
    events = {r['new_idx']: dict(r, source_line=i) for i, r in
              enumerate(map(json.loads, process_path.read_text().splitlines()), 1)
              if r.get('new_idx') is not None}
    best_p = (directory / 'best_perception_rexpure_seed1.py').read_text().strip()
    best_b = (directory / 'best_beliefs_rexpure_seed1.txt').read_text().strip()
    saved = next(c for c in pool.values() if c['perception'].strip() == best_p
                 and c['world_knowledge'].strip() == best_b)
    return pool_path, pool, events, saved


def lineage(pool, node):
    chain = [node]
    while pool[chain[-1]]['parents']:
        assert len(pool[chain[-1]]['parents']) == 1
        chain.append(pool[chain[-1]]['parents'][0])
    return chain[::-1]


def positions(frame, color):
    return sorted((r, c) for r, row in enumerate(frame['grid'])
                  for c, value in enumerate(row) if value == color)


def verify_selected_rules(scenario):
    """Check only the quoted rules on this window, not the full learned model."""
    frames = scenario['frames']
    for frame, nxt in zip(frames, frames[1:]):
        action = frame['action']
        if scenario['game'] == '7www9':
            blue = positions(frame, 'blue')
            red = positions(frame, 'red')
            assert red == positions(nxt, 'red') == [(7, 7), (8, 7)]
            if action == 'noop':
                predicted = blue
            else:
                assert action == 'down'
                outside = [(r, c) for r, c in blue if r not in (7, 8)]
                predicted = []
                for r, c in blue:
                    same_col = [(rr, cc) for rr, cc in outside if cc == c]
                    if r not in (7, 8):
                        predicted.append((r + 1, c))
                    elif same_col:
                        assert len(same_col) == 1 and same_col[0][0] > r
                        predicted.append((r + 1, c))
                    elif r == 7:
                        predicted.append((9, c + (1 if c < 7 else -1)))
                    else:
                        predicted.append((8, c + (1 if c < 7 else -1)))
            assert sorted(predicted) == positions(nxt, 'blue')
        else:
            red = positions(frame, 'red')
            shift = -6 if action == 'up' else (0 if max(r for r, c in red) == 19 else 1)
            assert action in ('noop', 'up')
            assert sorted((r + shift, c) for r, c in red) == positions(nxt, 'red')
            for color in ('green', 'yellow'):
                assert sorted((r, c - 1) for r, c in positions(frame, color)) == positions(nxt, color)
    return dict(transitions_checked=len(frames) - 1,
                scope='Literal local rules quoted in report, checked against recorded coordinates; not an execution of natural-language B.')


def build():
    evidence = dict(
        method='Exact candidate programs executed by run_perceive on metadata-stripped observations: fresh namespace, single-frame history. No model reruns.',
        scoring_note='Whole-training scores are not four-frame accuracies. Cached CFD scores refer to contrastive next-frame selection, not free-form forward simulation.',
        audit=[], scenarios=[])
    for directory in sorted(BASE.glob('*_s1')):
        game = directory.name.removesuffix('_s1')
        path, pool, _, saved = load_pool(game)
        evidence['audit'].append(dict(game=game, candidates=len(pool), saved_node=saved['idx'],
                                      saved_score=saved['train_score'], saved_lineage=lineage(pool, saved['idx'])))

    fig, axes = plt.subplots(2, 4, figsize=(12, 8.5))
    fig.patch.set_facecolor('#fcfcfb')
    fig.subplots_adjust(top=.91, bottom=.08, left=.045, right=.97, wspace=.30, hspace=.78)
    for ri, spec in enumerate(SPECS):
        path, pool, events, saved = load_pool(spec['game'])
        traj = ROOT / f"offline_learning/human_data/{spec['game']}/informative_curated/drives/{spec['drive']}/episode_0/trajectory.csv"
        rows = {int(r['Step']): r for r in csv.DictReader(traj.open())}
        scenario = {k: v for k, v in spec.items() if k != 'labels'}
        scenario.update(candidate_path=str(path.relative_to(ROOT)),
                        process_path=str(path.with_name('process_log.jsonl').relative_to(ROOT)),
                        trajectory_path=str(traj.relative_to(ROOT)),
                        lineage=lineage(pool, spec['nodes'][-1]),
                        saved_node=saved['idx'], saved_score=saved['train_score'],
                        saved_lineage=lineage(pool, saved['idx']),
                        on_saved_lineage=spec['nodes'][-1] in lineage(pool, saved['idx']),
                        frames=[], snapshots=[], cached_predictions=[])
        for ci, step in enumerate(spec['steps']):
            row = rows[step]
            obs = strip_autumn_obs_metadata(row['Observation'])
            grid = json.loads(obs)
            frame = dict(step=step, action=row['Action'], observation=obs, grid=grid,
                         observation_sha256=hashlib.sha256(obs.encode()).hexdigest())
            scenario['frames'].append(frame)
            ax = axes[ri, ci]
            ax.imshow(np.array([[to_rgb(c) for c in row] for row in grid]), interpolation='nearest')
            ax.set_xticks(range(0, len(grid[0]), 2))
            ax.set_yticks(range(0, len(grid), 2))
            ax.tick_params(length=0, labelsize=6)
            ax.set_xticks(np.arange(-.5, len(grid[0]), 1), minor=True)
            ax.set_yticks(np.arange(-.5, len(grid), 1), minor=True)
            ax.grid(which='minor', color='#97a6b0', alpha=.24, linewidth=.4)
            ax.tick_params(which='minor', length=0)
            ax.set_title(f'Step {step}', fontsize=10, pad=7)
            ax.set_xlabel(spec['labels'][ci], fontsize=8, labelpad=9)
            if ci < 3:
                ax.text(1.15, .5, row['Action'] + ' →', transform=ax.transAxes,
                        ha='center', va='center', fontsize=8, rotation=90)
        top = axes[ri, 0].get_position().y1
        fig.text(.045, top + .056, spec['title'], fontsize=13, weight='bold', color='#141718')
        fig.text(.045, top + .037,
                 f"Recorded {spec['drive']} frames; annotations from learned P. Coordinates are row, column.",
                 fontsize=8, color='#626769')
        hashes = {}
        for node in spec['nodes']:
            candidate = pool[node]
            event = events[node]
            parent = pool[candidate['parents'][0]]
            outputs = []
            for frame in scenario['frames']:
                out, err = run_perceive(candidate['perception'], frame['observation'])
                assert err is None, (spec['game'], node, frame['step'], err)
                assert out and 'error' not in out.lower()
                outputs.append(dict(step=frame['step'], output=out))
            snapshot = dict(node=node, iteration=event['i'], parents=candidate['parents'],
                            candidate_line=candidate['source_line'], process_line=event['source_line'],
                            accepted=event['accepted'], train_score=candidate['train_score'],
                            perception_changed=candidate['perception'] != parent['perception'],
                            belief_changed=candidate['world_knowledge'] != parent['world_knowledge'],
                            perception_code=candidate['perception'], belief=candidate['world_knowledge'],
                            outputs=outputs)
            scenario['snapshots'].append(snapshot)
            hashes[content_hash(candidate['perception'], candidate['world_knowledge'])] = node
        assert pool[spec['nodes'][-2]]['perception'] == pool[spec['nodes'][-1]]['perception']
        assert pool[spec['nodes'][-2]]['world_knowledge'] != pool[spec['nodes'][-1]]['world_knowledge']
        scenario['local_rule_verification'] = verify_selected_rules(scenario)
        frames = scenario['frames']
        transitions = {content_hash(f['observation'], nxt['observation'], f['action']): f['step']
                       for f, nxt in zip(frames, frames[1:])}
        pred_path = path.with_name('predictions.jsonl')
        fields = ('cand_hash', 'tr_hash', 'truth', 'pred', 'id_score', 'cfd_score', 'score',
                  'z_t', 'z_t1', 'reasoning', 'cfd_response')
        for li, line in enumerate(pred_path.open(), 1):
            record = json.loads(line)
            if record['cand_hash'] not in hashes:
                continue
            selected = record['tr_hash'] in transitions
            floor_extra = spec['game'] == 'dino' and record['tr_hash'] in ('e05abf99c4f8096b', '58aba89fca1a9fde')
            if not (selected or floor_extra):
                continue
            scenario['cached_predictions'].append(dict(
                node=hashes[record['cand_hash']], source_path=str(pred_path.relative_to(ROOT)),
                source_line=li, step=transitions.get(record['tr_hash']),
                in_selected_window=selected,
                note=None if selected else 'Separate floor-contact transition; not one of the four illustrated frames.',
                record={k: record.get(k) for k in fields}))
        if spec['game'] == '7www9':
            records = {r['node']: r['record'] for r in scenario['cached_predictions'] if r['step'] == 32}
            assert set(records) == {1, 6, 8}
            for node in (1, 6):
                assert records[node]['pred'] == ['right'] and records[node]['cfd_score'] == 0
            assert records[8]['pred'] == ['down'] and records[8]['cfd_score'] == 1
            assert 'rDist' in records[8]['reasoning'] and 'noRedRow' in records[8]['reasoning']
        evidence['scenarios'].append(scenario)

    fig.text(.045, .025,
             'Candidate preview. Both are explored branches; see learning_evolution_joint_candidates.md for P/B checkpoints and limitations.',
             fontsize=8, color='#626769')
    prefix = OUT.parent / 'learning_evolution_joint_candidates_frames'
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(prefix.with_suffix('.' + ext), dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    (OUT / 'learning_evolution_joint_candidates.json').write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + '\n')
    print(f"Audited {len(evidence['audit'])} pools / {sum(a['candidates'] for a in evidence['audit'])} candidates.")
    print('Executed 24 exact P/frame pairs; verified six local transitions against recorded states.')
    print('Verified cached Magnet snap: nodes 1 and 6 fail; node 8 succeeds and names the learned predicates.')
    for scenario in evidence['scenarios']:
        print(scenario['game'], 'lineage', scenario['lineage'], 'snapshots',
              [(s['node'], s['iteration']) for s in scenario['snapshots']])


if __name__ == '__main__':
    build()
