"""Counterfactual serialization changes on saved train frames; no model calls."""
import ast, csv, gzip, json, os, re, statistics as st, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
if os.environ.get('PYTHONHASHSEED')!='0':
    os.environ['PYTHONHASHSEED']='0';os.execv(sys.executable,[sys.executable,*sys.argv])
fn=next(n for n in ast.parse((ROOT/'offline_learning/validate.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run_perceive')
ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'validate.run_perceive','exec'),ns)
run=ns['run_perceive'];ans={}
for game,ship in [('colour_lines',12),('SET',12)]:
    cs=[json.loads(x) for x in (ROOT/f'logs/2026-08-24/human_curated/rexpure/{game}_s1/rexpure_run_seed1/candidates.jsonl').read_text().splitlines()]
    c=json.load(gzip.open(ROOT/f'analysis/perception_metrics/cache/{game}.json.gz','rt'))
    raws=[c['frames'][i] for i in sorted({i for p in c['train_pairs'] for i in p})]
    zs=[[run(cs[k]['perception'],x)[0] for x in raws] for k in [1,ship]]
    G=lambda s:len(gzip.compress(s.encode(),6,mtime=0))
    metric=lambda z:st.fmean(G(t)/G(x) for t,x in zip(z,raws))
    canon=(lambda z:sorted(re.findall(r'\((\d+),(\d+),([^()]+)\)',z))) if game=='colour_lines' else (lambda z:sorted(re.findall(r'(\d+),(\d+):(\w+)',z)))
    d={'n_frames':len(raws),'equal_coordinate_color_lists':sum(canon(a)==canon(b) for a,b in zip(*zs)),
       'ratio_first':metric(zs[0]),'ratio_ship':metric(zs[1]),
       'mean_gzip_first':st.fmean(map(G,zs[0])),'mean_gzip_ship':st.fmean(map(G,zs[1]))}
    assert d['equal_coordinate_color_lists']==d['n_frames']
    assert all(z.startswith('bg=black;') for arr in zs for z in arr)
    if game=='SET':
        newprefix='bg=black; shape=20x20; cursor:unknown; '
        assert all(z.startswith(newprefix) for z in zs[1])
        assert all(len(json.loads(x))==20 and all(len(row)==20 for row in json.loads(x)) for x in raws)
        mid=[z.replace('bg=black; cells=',newprefix) for z in zs[0]]
        d['ratio_after_constant_prefix_only']=metric(mid)
        d['mean_gzip_after_constant_prefix_only']=st.fmean(map(G,mid))
        d['constant_prefix_share_of_ratio_increase']=(metric(mid)-metric(zs[0]))/(metric(zs[1])-metric(zs[0]))
        d['constant_ship_prefix']=newprefix
    ans[game]=d
(OUT/'ablation.json').write_text(json.dumps(ans,indent=2)+'\n')
print(json.dumps(ans,indent=2))
