"""Read-only recomputation of saved perception outputs; writes this inquiry's results only."""
import ast, bz2, csv, gzip, hashlib, json, lzma, os, statistics as st, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
if os.environ.get('PYTHONHASHSEED') != '0':
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable, *sys.argv])
# Run the exact standalone helper without importing unrelated LLM/client plumbing.
tree = ast.parse((ROOT / 'offline_learning/validate.py').read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_perceive')
ns = {}
exec(compile(ast.Module(body=[fn], type_ignores=[]), 'validate.run_perceive', 'exec'), ns)
run_perceive = ns['run_perceive']
source = list(csv.DictReader((ROOT / 'analysis/perception_metrics/metrics.csv').open()))
root = ROOT / 'logs/2026-08-24/human_curated/rexpure'

def gz(s, level=6):
    return len(gzip.compress(s.encode(), compresslevel=level, mtime=0))

def compact(s):
    return json.dumps(json.loads(s), separators=(',',':'), ensure_ascii=False)

def lossless_grid(s):
    g = json.loads(s)
    colors = sorted({c for r in g for c in r})
    palette = {c: chr(65+i) for i,c in enumerate(colors)}
    # The palette text and matrix row boundaries are included: lossless for these color grids.
    return ','.join(colors) + '\n' + '\n'.join(''.join(palette[c] for c in r) for r in g)

nodes, games, output_examples = [], [], {}
flat = {'intervals': 0, 'unchanged_P_no_new_incumbent': 0, 'changed_P_no_new_incumbent': 0,
        'same_P_new_incumbent': 0, 'changed_P_new_incumbent_same_metric': 0}
for game in sorted({r['game'] for r in source}):
    rows = [r for r in source if r['game']==game and r['split']=='train']
    rs = sorted([r for r in rows if r['iteration']], key=lambda r: int(r['iteration']))
    cs = [json.loads(x) for x in (root / f'{game}_s1/rexpure_run_seed1/candidates.jsonl').read_text().splitlines()]
    cs = {c['idx']: c for c in cs}
    corpus = json.load(gzip.open(ROOT / f'analysis/perception_metrics/cache/{game}.json.gz', 'rt'))
    scored = sorted({i for pair in corpus['train_pairs'] for i in pair})
    raws = [corpus['frames'][i] for i in scored]
    raw_gz = [gz(x) for x in raws]
    raw_compact = [gz(compact(x)) for x in raws]
    raw_grid = [gz(lossless_grid(x)) for x in raws]
    raw_xz = [len(lzma.compress(x.encode(), preset=6)) for x in raws]
    raw_bz = [len(bz2.compress(x.encode(), compresslevel=9)) for x in raws]
    raw_l1 = [gz(x,1) for x in raws]
    raw_l9 = [gz(x,9) for x in raws]
    best = -1
    incs = []
    previous = None
    for r in rs:
        is_new = float(r['train_score']) > best
        if is_new:
            best = float(r['train_score'])
            incs.append(r)
        if previous:
            oldr = previous
            if float(r['info_extraction_ratio'] if is_new else oldr['info_extraction_ratio']) == float(oldr['info_extraction_ratio']):
                flat['intervals'] += 1
                candidate = cs[int(r['idx'])]
                candidate_parent = cs[candidate['parents'][0]]
                same_parent_p = candidate['perception'] == candidate_parent['perception']
                same_old_p = candidate['perception'] == cs[int(oldr['idx'])]['perception']
                if not is_new:
                    flat['unchanged_P_no_new_incumbent' if same_parent_p else 'changed_P_no_new_incumbent'] += 1
                elif same_old_p:
                    flat['same_P_new_incumbent'] += 1
                else:
                    flat['changed_P_new_incumbent_same_metric'] += 1
        previous = r if is_new else previous
    output_examples[game] = {'raw': raws[0], 'raw_gzip': raw_gz[0], 'compact_raw_gzip':raw_compact[0],
                            'lossless_grid_raw_gzip':raw_grid[0], 'incumbents': {}}
    pergame = []
    for r in incs:
        idx = int(r['idx']); code = cs[idx]['perception']
        result = [run_perceive(code, x) for x in raws]
        zs = [z for z,err in result]
        nerr = sum(err is not None for z,err in result)
        zg = [gz(z) for z in zs]
        val = st.fmean(z/g for z,g in zip(zg, raw_gz))
        assert abs(val-float(r['info_extraction_ratio'])) < 1e-12, (game,idx,val,r['info_extraction_ratio'])
        zjoin='\n'.join(zs)
        assert gz(zjoin) == int(r['diversity_bytes']), (game,idx,'corpus')
        d = dict(game=game, idx=idx, iteration=int(r['iteration']), ship=int(r['is_ship']),
                 status=r['status'], n_frames=len(raws), n_errors=nerr,
                 original=val, stripped_header=st.fmean((z-18)/(g-18) for z,g in zip(zg,raw_gz)),
                 ratio_of_means=st.fmean(zg)/st.fmean(raw_gz),
                 compact_raw=st.fmean(z/g for z,g in zip(zg,raw_compact)),
                 palette_grid_raw=st.fmean(z/g for z,g in zip(zg,raw_grid)),
                 frame_lzma=st.fmean(len(lzma.compress(z.encode(),preset=6))/g for z,g in zip(zs,raw_xz)),
                 frame_bzip=st.fmean(len(bz2.compress(z.encode(),compresslevel=9))/g for z,g in zip(zs,raw_bz)),
                 gzip_level1=st.fmean(gz(z,1)/g for z,g in zip(zs,raw_l1)),
                 gzip_level9=st.fmean(gz(z,9)/g for z,g in zip(zs,raw_l9)),
                 mean_out_chars=st.fmean(map(len,zs)), mean_out_gzip=st.fmean(zg),
                 mean_raw_gzip=st.fmean(raw_gz), mean_raw_chars=st.fmean(map(len,raws)),
                 corpus_gzip=float(r['norm_diversity']), corpus_lzma=float(r['lzma_ratio']),
                 output_reuse=sum(zg)/gz(zjoin), raw_reuse=sum(raw_gz)/gz('\n'.join(raws)),
                 code_sha=hashlib.sha256(code.encode()).hexdigest())
        nodes.append(d); pergame.append(d)
        output_examples[game]['incumbents'][str(idx)] = {'output':zs[0], 'gzip':zg[0]}
    first, last = pergame[0], pergame[-1]
    games.append(dict(game=game, first_idx=first['idx'], ship_idx=last['idx'], n_incumbents=len(incs),
                      mean_raw_gzip=st.fmean(raw_gz), min_raw_gzip=min(raw_gz), max_raw_gzip=max(raw_gz),
                      mean_compact_raw_gzip=st.fmean(raw_compact),
                      mean_palette_grid_raw_gzip=st.fmean(raw_grid),
                      **{f'{k}_{suffix}': d[k] for suffix,d in [('first',first),('ship',last)]
                         for k in ['original','stripped_header','ratio_of_means','compact_raw','palette_grid_raw',
                                   'frame_lzma','frame_bzip','gzip_level1','gzip_level9',
                                   'mean_out_chars','corpus_gzip','corpus_lzma','output_reuse','raw_reuse']}))
    print(game, 'nodes', len(incs), 'first', round(first['original'],4), 'ship',round(last['original'],4), flush=True)
for name, data in [('incumbents.csv',nodes),('games.csv',games)]:
    with (OUT/name).open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
(OUT/'examples.json').write_text(json.dumps(output_examples,indent=2))
(OUT/'flat_stretches.json').write_text(json.dumps(flat,indent=2))
print('FLAT',flat)
print('verified incumbents',len(nodes))
