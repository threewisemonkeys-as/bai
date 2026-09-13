#!/usr/bin/env python3
"""Reproduce exploratory statistics without changing input metrics or figures.

Run with .venv/bin/python analysis/wm_quant/inquiry/cross_game_statistics/analyze.py
from the repo root. Inference is descriptive across 15 purposively selected games,
each with one dependent search tree; node-level p-values are not evidence of replication.
"""
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, pearsonr, rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
METRIC = 'info_extraction_ratio'
NAMES = dict(zip(['eahcw', 'egg', 'bt3gb', 'dq8gc', '7xf97', 'n2ntd', 'va6fq',
                 's2kt7', 'colour_lines', 'SET', 'diffusion', 'dino', 'f5w3n',
                 'logic_gates', '7www9'],
                ['Paint', 'Egg', 'Ice', 'Disease', 'Grow', 'Mario', 'Sand', 'Ants',
                 'Colour Lines', 'SET', 'Diffusion', 'Dino', 'Space Invaders',
                 'Logic Gates', 'Magnets']))
TRAIN = ROOT / 'logs/2026-08-24/human_curated/rexpure'
inputs = {}

def read(path):
    data = path.read_bytes()
    inputs[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return data.decode()

def mean(x):
    return float(np.mean(x))

def corr(x, y):
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    a, p = spearmanr(x, y)
    return {'rho': float(a), 'p_unadjusted': float(p), 'n': len(x)}

def write_csv(name, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open('w') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)

def signed_summary(xs):
    xs = np.asarray([x for x in xs if x is not None], dtype=float)
    nz = xs[abs(xs) > 1e-12]
    return dict(n=len(xs), median=float(np.median(xs)), mean=mean(xs),
                positive=int(sum(xs > 1e-12)), negative=int(sum(xs < -1e-12)),
                zero=int(sum(abs(xs) <= 1e-12)),
                sign_p_two_sided=float(binomtest(int(sum(nz > 0)), len(nz), 0.5).pvalue)
                if len(nz) else None)

def bootstrap_median(xs):
    rng = np.random.default_rng(5731)
    xs = np.asarray(xs)
    z = np.median(rng.choice(xs, (50000, len(xs))), axis=1)
    return [float(a) for a in np.quantile(z, [.025, .975])]

raw = list(csv.DictReader(read(ROOT / 'analysis/perception_metrics/metrics.csv').splitlines()))
rows = []
for r in raw:
    for k, v in r.items():
        if v in ['', 'None']:
            r[k] = None
        else:
            try:
                r[k] = float(v)
            except ValueError:
                pass
    for k in ['idx', 'iteration', 'is_ship', 'depth', 'on_ship_lineage']:
        if r[k] is not None:
            r[k] = int(r[k])
    rows.append(r)

games = sorted(NAMES)
endpoints, correlations, edges, incumbent_steps, planning, baselines = [], [], [], [], [], []
for game in games:
    run = TRAIN / f'{game}_s1' / 'rexpure_run_seed1'
    pool = {r['idx']: r for r in map(json.loads, read(run / 'candidates.jsonl').splitlines())}
    logs = list(map(json.loads, read(run / 'process_log.jsonl').splitlines()))
    logs_by_idx = {r['new_idx']: r for r in logs if r.get('new_idx') is not None}
    cache_path = ROOT / f'analysis/perception_metrics/cache/{game}.json.gz'
    inputs[str(cache_path.relative_to(ROOT))] = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    cache = json.loads(gzip.decompress(cache_path.read_bytes()))
    train_used = {i for p in cache['train_pairs'] for i in p}
    test_used = {i for p in cache['test_pairs'] for i in p}
    for split in ['train', 'test']:
        rs = sorted([r for r in rows if r['game'] == game and r['split'] == split],
                    key=lambda r: r['iteration'] if r['iteration'] is not None else -1)
        by_idx = {r['idx']: r for r in rs}
        plotted = [r for r in rs if r['iteration'] is not None and r[METRIC] is not None]
        best = -1
        inc = []
        for r in plotted:
            if r['train_score'] > best:
                best = r['train_score']
                inc.append(r)
        valid_inc = [r for r in inc if r['status'] == 'ok']
        first, ship = valid_inc[0], next(r for r in rs if r['is_ship'])
        assert ship['idx'] == inc[-1]['idx']
        used = sorted(train_used if split == 'train' else test_used)
        frames = [cache['frames'][i] for i in used]
        gz = [len(gzip.compress(s.encode(), 6, mtime=0)) for s in frames]
        grids = [json.loads(s[s.find('[['):s.rfind(']]') + 2]) for s in frames]
        counts = [Counter(c for row in grid for c in row) for grid in grids]
        d = dict(game=game, name=NAMES[game], split=split,
                 first_plotted_idx=inc[0]['idx'], first_plotted_status=inc[0]['status'],
                 first_plotted_info=inc[0][METRIC],
                 first_valid_idx=first['idx'], first_valid_it=first['iteration'],
                 first_valid_info=first[METRIC], first_train_score=first['train_score'],
                 ship_idx=ship['idx'], ship_it=ship['iteration'], ship_info=ship[METRIC],
                 ship_train_score=ship['train_score'], ship_mean_out_chars=ship['mean_out_chars'],
                 ship_diversity=ship['norm_diversity'], first_diversity=first['norm_diversity'],
                 absolute_change=ship[METRIC] - first[METRIC],
                 relative_change=ship[METRIC] / first[METRIC] - 1,
                 log_change=math.log(ship[METRIC] / first[METRIC]),
                 valid_incumbents=len(valid_inc), n_frames=len(used),
                 mean_raw_gz=mean(gz), mean_grid_cells=mean([c.total() for c in counts]),
                 mean_nonmodal_cells=mean([c.total() - c.most_common(1)[0][1] for c in counts]),
                 mean_colors=mean([len(c) for c in counts]),
                 raw_mean_chars=mean(list(map(len, frames))),
                 train_test_shared_frames=len(train_used & test_used),
                 test_frame_overlap_fraction=len(train_used & test_used) / len(test_used))
        jumps = []
        for a, b in zip(valid_inc, valid_inc[1:]):
            delta = b[METRIC] - a[METRIC]
            p_same = pool[a['idx']]['perception'] == pool[b['idx']]['perception']
            k_same = pool[a['idx']]['world_knowledge'] == pool[b['idx']]['world_knowledge']
            incumbent_steps.append(dict(game=game, split=split, previous=a['idx'], idx=b['idx'],
                                        iteration=b['iteration'], delta_info=delta,
                                        delta_score=b['train_score'] - a['train_score'],
                                        p_same=int(p_same), k_same=int(k_same),
                                        component=','.join(logs_by_idx[b['idx']]['components'])))
            if abs(delta) > 1e-12:
                jumps.append(delta)
        d['up_jumps'], d['down_jumps'] = sum(v > 0 for v in jumps), sum(v < 0 for v in jumps)
        d['flat_incumbent_updates'] = len(valid_inc) - 1 - len(jumps)
        d['monotonic'] = int(not jumps or all(v > 0 for v in jumps) or all(v < 0 for v in jumps))
        endpoints.append(d)
        starts = {'first_valid_incumbent': first,
                  'first_nonempty_k_incumbent': next(r for r in valid_inc if r['k_chars'] > 0)}
        for t in [3, 10, 20]:
            before = [r for r in valid_inc if r['iteration'] <= t]
            if before:
                starts[f'incumbent_at_iteration_{t}'] = before[-1]
        for label, start in starts.items():
            baselines.append(dict(game=game, split=split, baseline=label,
                                  first_idx=start['idx'], first_iteration=start['iteration'],
                                  first_info=start[METRIC], ship_info=ship[METRIC],
                                  relative_change=ship[METRIC] / start[METRIC] - 1))
        ok = [r for r in rs if r['status'] == 'ok']
        has_k = [r for r in ok if pool[r['idx']]['world_knowledge'].strip()]
        def pcorr(sub):
            c = corr([r[METRIC] for r in sub], [r['train_score'] for r in sub])
            return c['rho'] if c else None
        def dedup(sub):
            groups = defaultdict(list)
            for r in sub:
                groups[pool[r['idx']]['perception']].append(r)
            return [dict(**{METRIC: vs[0][METRIC]}, train_score=mean([r['train_score'] for r in vs]))
                    for vs in groups.values()]
        du, duk = dedup(ok), dedup(has_k)
        # Demean rank scores within identical K strings; P varies but K is held fixed.
        k_groups = defaultdict(list)
        for r in ok:
            k_groups[pool[r['idx']]['world_knowledge']].append(r)
        cx, cy = [], []
        for ks in k_groups.values():
            ks = dedup(ks)
            if len(ks) < 2:
                continue
            x, y = rankdata([r[METRIC] for r in ks]), rankdata([r['train_score'] for r in ks])
            cx.extend(x - np.mean(x))
            cy.extend(y - np.mean(y))
        within_k = float(pearsonr(cx, cy).statistic) if len(cx) >= 3 and len(set(cx)) > 1 and len(set(cy)) > 1 else None
        correlations.append(dict(game=game, name=NAMES[game], split=split,
                                 n_ok=len(ok), n_nonempty_k=len(has_k),
                                 n_unique_p=len(du), n_unique_p_nonempty_k=len(duk),
                                 rho_all_ok=pcorr(ok), rho_nonempty_k=pcorr(has_k),
                                 rho_unique_p_mean_score=pcorr(du),
                                 rho_unique_p_nonempty_k_mean_score=pcorr(duk),
                                 within_k_centered_rank_corr=within_k,
                                 within_k_n_centered_points=len(cx)))
        for r in ok:
            for parent in pool[r['idx']]['parents']:
                a = by_idx[parent]
                if a['status'] != 'ok':
                    continue
                c1, c2 = pool[parent], pool[r['idx']]
                p_same = c1['perception'] == c2['perception']
                k_same = c1['world_knowledge'] == c2['world_knowledge']
                edges.append(dict(game=game, split=split, parent=parent, idx=r['idx'],
                                  iteration=r['iteration'], p_same=int(p_same), k_same=int(k_same),
                                  nonempty_k=int(bool(c1['world_knowledge'].strip())),
                                  delta_info=r[METRIC] - a[METRIC],
                                  delta_score=r['train_score'] - a['train_score']))
    plan_file = ROOT / f'logs/2026-09-03/planning_v2_online_ds_percap_nl/{game}/online.json'
    ps = json.loads(read(plan_file))['rows']
    pr = dict(game=game, name=NAMES[game], n_tasks=len(ps))
    for key in ['raw', 'lmwm']:
        rates = [p[key]['pass_rate'] for p in ps if p.get(key, {}).get('status') == 'evaluated']
        pr[key] = mean(rates)
    pr['plain_lift_over_raw'] = pr['lmwm'] - pr['raw']
    planning.append(pr)

agent = list(map(json.loads, read(ROOT / 'logs/2026-09-08/agent_wm_full/rows.jsonl').splitlines()))
for r in planning:
    ar = [a['agent']['pass_rate'] for a in agent if a['game'] == r['game'] and a['agent'].get('status') == 'done']
    r['agentic'] = mean(ar)
    r['n_agentic_tasks'] = len(ar)
    r['agentic_lift_over_raw'] = r['agentic'] - r['raw']

summary = {}
for split in ['train', 'test']:
    es = [r for r in endpoints if r['split'] == split]
    cs = [r for r in correlations if r['split'] == split]
    inc = [r for r in incumbent_steps if r['split'] == split]
    change = [r['relative_change'] for r in es]
    s = dict(relative_change=signed_summary(change),
             median_relative_change_bootstrap_95pct=bootstrap_median(change),
             median_first_info=float(np.median([r['first_valid_info'] for r in es])),
             median_ship_info=float(np.median([r['ship_info'] for r in es])),
             first_compressed_below_1=sum(r['first_valid_info'] < 1 for r in es),
             ship_compressed_below_1=sum(r['ship_info'] < 1 for r in es),
             monotonic_games=sum(r['monotonic'] for r in es),
             incumbent_jumps=signed_summary([r['delta_info'] for r in inc]),
             flat_same_p=sum(abs(r['delta_info']) < 1e-12 and r['p_same'] for r in inc))
    for threshold in [.01, .05, .10, .20]:
        s[f'relative_change_threshold_{threshold}'] = dict(up=sum(c > threshold for c in change),
                                                        down=sum(c < -threshold for c in change),
                                                        within=sum(abs(c) <= threshold for c in change))
    s['correlations'] = {k: signed_summary([r[k] for r in cs]) for k in cs[0] if 'rho_' in k or 'centered_rank_corr' in k}
    s['baseline_sensitivity'] = {label: signed_summary([r['relative_change'] for r in baselines
                                   if r['split'] == split and r['baseline'] == label])
                               for label in sorted({r['baseline'] for r in baselines})}
    s['cross_game_correlations'] = {}
    for x in ['first_valid_info', 'mean_raw_gz', 'mean_nonmodal_cells', 'mean_colors', 'raw_mean_chars']:
        for y in ['relative_change', 'absolute_change', 'ship_info']:
            s['cross_game_correlations'][x + '__' + y] = corr([r[x] for r in es], [r[y] for r in es])
    s['colors_versus_initial_info'] = corr([r['mean_colors'] for r in es], [r['first_valid_info'] for r in es])
    # Descriptive sensitivity, without incorrectly calibrated partial-correlation p-values.
    cc, yy = rankdata([r['mean_colors'] for r in es]), rankdata([r['relative_change'] for r in es])
    zz = np.column_stack([np.ones(len(es)), rankdata([r['first_valid_info'] for r in es])])
    cr = cc - zz @ np.linalg.lstsq(zz, cc, rcond=None)[0]
    yr = yy - zz @ np.linalg.lstsq(zz, yy, rcond=None)[0]
    s['colors_growth_partial_rank_corr_controlling_initial_info'] = float(pearsonr(cr, yr).statistic)
    loo = [spearmanr(np.delete(cc, i), np.delete(yy, i)).statistic for i in range(len(es))]
    s['colors_growth_leave_one_game_out_rho_range'] = [float(min(loo)), float(max(loo))]
    s['planning_correlations'] = {}
    plan_map = {r['game']: r for r in planning}
    for x in ['ship_info', 'relative_change', 'ship_train_score', 'mean_raw_gz']:
        for y in ['lmwm', 'agentic', 'plain_lift_over_raw', 'agentic_lift_over_raw']:
            s['planning_correlations'][x + '__' + y] = corr([r[x] for r in es], [plan_map[r['game']][y] for r in es])
    s['fixed_k_p_edges'] = {}
    for subset in ['all', 'nonempty_k']:
        ee = [r for r in edges if r['split'] == split and r['k_same'] and not r['p_same']
              and (subset == 'all' or r['nonempty_k'])]
        ge = defaultdict(list)
        for e in ee:
            ge[e['game']].append(e)
        rr = {g: corr([r['delta_info'] for r in rs], [r['delta_score'] for r in rs]) for g, rs in ge.items()}
        qualified = [r['rho'] for r in rr.values() if r]
        s['fixed_k_p_edges'][subset] = dict(n_edges=len(ee), per_game_corr=rr,
                                          per_game_corr_summary=signed_summary(qualified),
                                          pooled_corr_descriptive_only=corr([r['delta_info'] for r in ee], [r['delta_score'] for r in ee]),
                                          positive_score_and_up=sum(r['delta_score'] > 0 and r['delta_info'] > 1e-12 for r in ee),
                                          positive_score_and_down=sum(r['delta_score'] > 0 and r['delta_info'] < -1e-12 for r in ee))
    summary[split] = s

summary['train_test_direction_agreement'] = sum(np.sign(a['relative_change']) == np.sign(b['relative_change'])
    for a, b in zip([r for r in endpoints if r['split'] == 'train'], [r for r in endpoints if r['split'] == 'test']))
summary['notes'] = [
    'First valid incumbent means earliest strict best-so-far train_score node with status ok, excluding the blank seed and the initial Ice collapsed proposal.',
    'Test rendering uses the same training-selected program and overlapping raw frames; it is not independent replication or held-out dynamics/planning validation.',
    'All correlation p-values are exploratory, unadjusted and across dependent node sequences or only 15 selected games.',
    'Deduplication uses exact P source strings and averages train scores over all K variants paired with that P.',
    'Within-K rank correlations center rank features and scores within identical K strings after deduplicating P; groups with fewer than two P variants contribute no data.',
    'Parent-child edge correlations condition on unchanged exact K string and both endpoints status ok; they are observational search-tree contrasts, not randomized P interventions.',
    'Raw gzip size shares the denominator of the info metric, and initial ratio occurs in the change score: apparent predictors are mathematically coupled.',
    'Game-level bootstrap resamples 15 games, not nodes or frames; it summarizes this selected collection and does not estimate seed-to-seed uncertainty.',
]
for name, data in [('endpoints.csv', endpoints), ('correlations.csv', correlations), ('parent_child_edges.csv', edges),
                   ('incumbent_steps.csv', incumbent_steps), ('planning.csv', planning), ('baseline_sensitivity.csv', baselines)]:
    write_csv(name, data)
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x)) + '\n')
(OUT / 'input_sha256.json').write_text(json.dumps(inputs, indent=2) + '\n')
for split in ['train', 'test']:
    s = summary[split]
    print(split, json.dumps({k:s[k] for k in ['relative_change', 'median_relative_change_bootstrap_95pct',
          'median_first_info', 'median_ship_info', 'monotonic_games', 'incumbent_jumps', 'correlations']}, indent=2))
print('Wrote artifacts to', OUT)
