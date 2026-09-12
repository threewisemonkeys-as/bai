"""Reproduce the focused Grow candidate audit from saved logs, without model calls.

Writes exact perception replays, belief excerpts, cached prediction evidence,
and a four-sequence frame preview. Local checks formalize only the specific
rules discussed in analysis/learning_example/audit/learning_evolution_grow_candidates.md.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from offline_learning.validate import run_perceive, strip_autumn_obs_metadata

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np

RUN = ROOT / 'logs/2026-08-24/human_curated/rexpure/7xf97_s1/rexpure_run_seed1'
DATA = ROOT / 'offline_learning/human_data/7xf97/informative_curated/drives'
OUT = ROOT / 'analysis/learning_example/audit/learning_evolution_grow_candidates.json'
STEM = ROOT / 'analysis/learning_example/learning_evolution_grow_candidates_frames'
FLAGS = ('gx', 'ax', 'gle', 'gre', 'gcle', 'gcre', 'gare', 'gale', 'uc')
SPECS = [
    dict(key='rain_ranges', title='A. Rain pauses, falls, and exits the grid',
         drive='train_d0', steps=[174, 175, 176, 177], nodes=[2, 10, 11, 16],
         focus=(9, 16, 10, 15),
         notes=['rows 10–12 + 14–15', 'rows 11–13 + 15', 'same under down', 'rows 12–14 only'],
         status='Strongest explicit P-to-B use; perception change is component grouping, not a new interaction predicate.'),
    dict(key='flower_contact', title='B. Rain reaches a flower and disappears',
         drive='train_d0', steps=[69, 70, 71, 72], nodes=[2, 7, 17, 20],
         focus=(9, 16, 0, 3),
         notes=['blue at (10,1)', 'blue at (11,1)', 'blue disappears', 'flower unchanged'],
         status='Local contact rule is correct; perception adds components but no contact predicate; no cached evaluation for this exact window.'),
    dict(key='cloud_boundary', title='C. A clipped cloud returns into view',
         drive='train_d0', steps=[26, 27, 28, 29], nodes=[2, 10, 19, 23, 28],
         focus=(0, 3, 10, 16),
         notes=['gray columns 13–15', 'gray columns 12–15', 'gray columns 11–14', 'gray columns 10–13'],
         status='P23 learns an edge predicate used correctly in one cached prediction. B28 learns the entry rule on another branch without that predicate.'),
    dict(key='sunlight_growth', title='D. Rain contact: covered sun versus uncovered sun',
         drive='train_d0', steps=[218, 219, 220, 221], nodes=[2, 11, 20, 23, 24],
         focus=(9, 16, 10, 14),
         notes=['contact; sun partly covered', 'no growth; cloud moves next', 'sun uncovered; rain contacts', 'plant gains (14,11)'],
         status='Excellent game dynamic, but no stored belief learns the sunlight-dependent growth rule; P23 flags do not distinguish these states.'),
]


def digest(*parts):
    result = hashlib.md5()
    for part in parts:
        result.update((part or '').encode())
        result.update(b'\0')
    return result.hexdigest()[:16]


def ancestry(pool, node):
    chain = [node]
    while pool[chain[-1]]['parents']:
        assert len(pool[chain[-1]]['parents']) == 1
        chain.append(pool[chain[-1]]['parents'][0])
    return chain[::-1]


def cells(grid, color):
    return {(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == color}


def flags(output):
    return {key: int(value) for key, value in re.findall(r'\b(' + '|'.join(FLAGS) + r')=([01])', output)}


def load():
    csv.field_size_limit(10**7)
    pool = {c['idx']: dict(c, source_line=i) for i, c in enumerate(
        map(json.loads, (RUN / 'candidates.jsonl').read_text().splitlines()), 1)}
    events = {r['new_idx']: dict(r, source_line=i) for i, r in enumerate(
        map(json.loads, (RUN / 'process_log.jsonl').read_text().splitlines()), 1)
        if r.get('new_idx') is not None}
    lookup = {digest(c['perception'], c['world_knowledge']): n for n, c in pool.items()}
    records = []
    for line, raw in enumerate((RUN / 'predictions.jsonl').open(), 1):
        rec = json.loads(raw)
        if rec['cand_hash'] in lookup:
            records.append(dict(node=lookup[rec['cand_hash']], source_line=line, record=rec))
    drives = {}
    transition_index = {}
    for path in sorted(DATA.glob('train_d*/episode_0/trajectory.csv')):
        drive = path.parent.parent.name
        with path.open() as stream:
            drives[drive] = [dict(step=int(row['Step']), action=row['Action'],
                observation=strip_autumn_obs_metadata(row['Observation'])) for row in csv.DictReader(stream)]
        for a, b in zip(drives[drive], drives[drive][1:]):
            key = digest(a['observation'], b['observation'], a['action'])
            transition_index.setdefault(key, []).append(dict(drive=drive, step=a['step'], action=a['action']))
    return pool, events, records, drives, transition_index


def replay(pool, events, records, drives, spec):
    data = dict(spec)
    source = {r['step']: r for r in drives[spec['drive']]}
    data['trajectory_path'] = str((DATA / spec['drive'] / 'episode_0/trajectory.csv').relative_to(ROOT))
    data['frames'] = [dict(source[t], grid=json.loads(source[t]['observation'])) for t in spec['steps']]
    data['snapshots'] = []
    for node in spec['nodes']:
        cand = pool[node]
        outputs = []
        for frame in data['frames']:
            result, error = run_perceive(cand['perception'], frame['observation'])
            assert error is None, (node, frame['step'], error)
            outputs.append(result)
        data['snapshots'].append(dict(node=node, iteration=events.get(node, {}).get('i'),
            parents=cand['parents'], lineage=ancestry(pool, node),
            candidate_line=cand['source_line'], process_line=events.get(node, {}).get('source_line'),
            perception=cand['perception'], belief=cand['world_knowledge'], outputs=outputs))
    by_node = {s['node']: s for s in data['snapshots']}
    transitions = {}
    for i, (a, b) in enumerate(zip(data['frames'], data['frames'][1:])):
        transitions.setdefault(digest(a['observation'], b['observation'], a['action']), []).append(i)
    data['cached_predictions'] = []
    fields = ('cand_hash', 'tr_hash', 'truth', 'pred', 'id_score', 'id_hit', 'cfd_score',
              'cfd_pred', 'cfd_ambiguous', 'z_t', 'z_t1', 'reasoning', 'cfd_response')
    for item in records:
        n, rec = item['node'], item['record']
        if n not in by_node or rec['tr_hash'] not in transitions:
            continue
        for i in transitions[rec['tr_hash']]:
            assert rec['z_t'] == by_node[n]['outputs'][i]
            assert rec['z_t1'] == by_node[n]['outputs'][i+1]
        data['cached_predictions'].append(dict(node=n, source_line=item['source_line'],
            steps=[data['frames'][i]['step'] for i in transitions[rec['tr_hash']]],
            record={k: rec.get(k) for k in fields}))
    return data


def verify_local(data, pool):
    grids = [f['grid'] for f in data['frames']]
    key = data['key']
    results = []
    if key in ('rain_ranges', 'flower_contact'):
        for before, after in zip(data['frames'], data['frames'][1:]):
            grid, nxt = before['grid'], after['grid']
            expected = [row.copy() for row in grid]
            blue = cells(grid, 'blue')
            for r, c in blue:
                expected[r][c] = 'black'
            for r, c in blue:
                rr = r if before['action'] == 'down' else r+1
                if rr < 16 and grid[rr][c] not in ('green', 'mediumpurple'):
                    expected[rr][c] = 'blue'
            assert expected == nxt, (key, before['step'])
            results.append(dict(step=before['step'], all_visible_cells_match=True))
        if key == 'rain_ranges':
            assert pool[11]['perception'] == pool[16]['perception']
            assert pool[10]['world_knowledge'] == pool[11]['world_knowledge']
            assert 'blue:rowStart-rowEnd, col-col' in pool[16]['world_knowledge']
            assert 'Drift adds 1 to both row numbers' in pool[16]['world_knowledge']
        else:
            assert pool[17]['perception'] == pool[20]['perception']
            assert 'occupied by a static cell (green or mediumpurple)' in pool[20]['world_knowledge']
    elif key == 'cloud_boundary':
        ranges = [(min(c for r, c in cells(g, 'gray')), max(c for r, c in cells(g, 'gray'))) for g in grids]
        assert ranges == [(13, 15), (12, 15), (11, 14), (10, 13)]
        p23 = next(s for s in data['snapshots'] if s['node'] == 23)
        assert [flags(s)['gcre'] for s in p23['outputs']] == [1, 1, 0, 0]
        assert 23 not in ancestry(pool, 28)
        assert pool[19]['world_knowledge'] == pool[23]['world_knowledge'] == pool[10]['world_knowledge']
        for before, after in zip(data['frames'], data['frames'][1:]):
            expected = [row.copy() for row in before['grid']]
            gray = cells(expected, 'gray')
            lo, hi = min(c for r, c in gray), max(c for r, c in gray)
            for r, c in gray: expected[r][c] = 'black'
            new_hi = hi if hi == 15 and hi-lo+1 < 4 else hi-1
            for r in range(3):
                for c in range(lo-1, new_hi+1): expected[r][c] = 'gray'
            assert expected == after['grid']
            results.append(dict(step=before['step'], all_visible_cells_match_B28_local_rule=True))
    else:
        p23 = next(s for s in data['snapshots'] if s['node'] == 23)
        fs = [flags(s) for s in p23['outputs']]
        assert all(f == fs[0] for f in fs)
        assert (14, 11) in cells(grids[0], 'blue')
        assert cells(grids[1], 'green') == cells(grids[0], 'green')
        assert cells(grids[3], 'green') - cells(grids[2], 'green') == {(14, 11)}
        assert (14, 11) in cells(grids[2], 'blue')
        assert [len({c for r, c in cells(g, 'gold')}) for g in grids] == [2, 2, 3, 3]
        results.append(dict(P23_flags_identical_for_all_frames=True,
                            observed_growth_after_uncovering=[14, 11],
                            no_claim_of_learned_growth_rule=True))
    data['local_checks'] = results


def draw(scenarios):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(10.2, 11.7), facecolor='#fcfcfb')
    fig.text(.04, .982, 'Grow: candidate state sequences', size=15, weight='bold', va='top')
    fig.text(.04, .958, 'Recorded frames and actions. A–C are candidates with different limitations; D shows an unsupported learning claim.',
             size=9, color='#555d62', va='top')
    for row, data in enumerate(scenarios):
        top = .916 - row*.225
        fig.text(.04, top+.023, data['title'], size=11, weight='bold', va='top')
        fig.text(.04, top+.006, data['drive'] + '  ·  frames ' + str(data['steps'][0]) + '–' + str(data['steps'][-1]),
                 size=8, color='#555d62', va='top')
        y = top-.169
        for i, frame in enumerate(data['frames']):
            x = .055+i*.24
            ax = fig.add_axes([x, y, .17, .15])
            rgb = np.array([[to_rgb(c) for c in rr] for rr in frame['grid']])
            ax.imshow(rgb, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_edgecolor('#c6cccf')
            r0, r1, c0, c1 = data['focus']
            ax.add_patch(Rectangle((c0-.5, r0-.5), c1-c0, r1-r0, fill=False,
                                   edgecolor='#ff8c42', linewidth=1.2))
            if data['key'] == 'sunlight_growth':
                ax.add_patch(Rectangle((6.5, -.5), 7, 3, fill=False, edgecolor='#ff8c42', linewidth=1.2))
            ax.set_title('t = '+str(frame['step']), size=9, pad=4)
            fig.text(x+.085, y-.009, data['notes'][i], size=8, ha='center', va='top')
            if i < 3:
                start, end = x+.18, x+.229
                fig.add_artist(FancyArrowPatch((start,y+.074),(end,y+.074), transform=fig.transFigure,
                                               arrowstyle='-|>', mutation_scale=11, color='#65717a'))
                fig.text((start+end)/2,y+.090,frame['action'],size=8,ha='center')
    fig.text(.04,.02,'Orange boxes mark the relevant regions; all 16 × 16 cells are shown. Exact P/B and limitations are in the accompanying report.',
             size=8,color='#555d62')
    for suffix in ('.png', '.pdf', '.svg'):
        fig.savefig(STEM.with_suffix(suffix), dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    pool, events, records, drives, transition_index = load()
    assert len(pool) == 30
    scenarios = [replay(pool, events, records, drives, spec) for spec in SPECS]
    for s in scenarios: verify_local(s, pool)
    flag_uses = []
    for item in records:
        if item['node'] not in (23, 24): continue
        for field in ('reasoning', 'cfd_response'):
            text = item['record'].get(field, '') or ''
            hits = [f for f in FLAGS if re.search(r'\b'+f+r'\b', text)]
            if hits:
                flag_uses.append(dict(node=item['node'], source_line=item['source_line'], field=field,
                    flags=hits, transitions=transition_index[item['record']['tr_hash']], text=text))
    assert not any(re.search(r'\b(?:gare|gale|gcle|gcre|gle|gre|gx|ax|uc)\b', c['world_knowledge']) for c in pool.values())
    assert not any('gare' in use['flags'] for use in flag_uses)
    # Retain nearby cached failures as counterevidence, not just selected successes.
    supporting_lines = {664, 1024, 704, 1004, 1064, 1244, 980, 1220, 990, 1230, 1411, 1431, 1551, 1392, 1518}
    supporting = []
    for item in records:
        if item['source_line'] not in supporting_lines: continue
        rec = item['record']; first = transition_index[rec['tr_hash']][0]
        source = {r['step']:r for r in drives[first['drive']]}
        for off, field in [(0, 'z_t'), (1, 'z_t1')]:
            out, error = run_perceive(pool[item['node']]['perception'], source[first['step']+off]['observation'])
            assert error is None and out == rec[field]
        supporting.append(dict(node=item['node'], source_line=item['source_line'],
            transitions=transition_index[rec['tr_hash']],
            record={k: rec.get(k) for k in ('cand_hash','tr_hash','truth','pred','id_score','cfd_score','reasoning','cfd_response','z_t','z_t1')}))
    metadata = []
    for n, c in pool.items():
        par = pool[c['parents'][0]] if c['parents'] else None
        metadata.append(dict(node=n, iteration=events.get(n, {}).get('i'), parents=c['parents'],
            candidate_line=c['source_line'], process_line=events.get(n, {}).get('source_line'),
            perception_changed=par is None or par['perception'] != c['perception'],
            belief_changed=par is None or par['world_knowledge'] != c['world_knowledge']))
    result = dict(game='7xf97', run=str(RUN.relative_to(ROOT)), candidates_reviewed=len(pool),
        unique_nonempty_beliefs=len({c['world_knowledge'] for c in pool.values() if c['world_knowledge']}),
        training_frames=sum(len(fs) for fs in drives.values()),
        training_transitions=sum(len(fs)-1 for fs in drives.values()),
        cached_predictions_scanned=len(records), candidates=metadata,
        saved_node=23, saved_lineage=ancestry(pool,23), scenarios=scenarios,
        P23_P24_flag_references_in_cached_reasoning=flag_uses,
        supporting_and_counterexample_records=supporting,
        validation=dict(perception_frame_replays=sum(len(s['nodes'])*len(s['steps']) for s in scenarios),
                        selected_cached_records_verified=sum(len(s['cached_predictions']) for s in scenarios),
                        additional_cached_records_verified=len(supporting),
                        full_grid_local_transition_checks=sum(len(s['local_checks']) for s in scenarios[:3])))
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    draw(scenarios)
    print(json.dumps({k:result[k] for k in ('candidates_reviewed','unique_nonempty_beliefs','training_frames',
        'training_transitions','cached_predictions_scanned','validation')}, indent=2))
    for s in scenarios:
        print(s['key'], [(x['node'],x['iteration']) for x in s['snapshots']],
              'cached records',len(s['cached_predictions']))


if __name__ == '__main__':
    main()
