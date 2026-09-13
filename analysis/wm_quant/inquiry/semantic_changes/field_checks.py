import gzip,json,re,statistics,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
BASE=ROOT/'logs/2026-08-24/human_curated/rexpure'
def gz(s):return len(gzip.compress(s.encode(),6,mtime=0))
def run(code,frame):
 ns={};exec(code,ns);return ns['perceive']([frame])
def ratio(outs,frames):return statistics.mean(gz(z)/gz(f) for z,f in zip(outs,frames))
def parse(f):return json.loads(f[f.find('[['):f.rfind(']]')+2])
results={}
for game in ['SET','egg','diffusion','n2ntd','f5w3n','7xf97']:
 pool={n['idx']:n for n in map(json.loads,(BASE/f'{game}_s1/rexpure_run_seed1/candidates.jsonl').read_text().splitlines())}
 with gzip.open(ROOT/f'analysis/perception_metrics/cache/{game}.json.gz','rt') as f:corp=json.load(f)
 results[game]={}
 for split in ['train','test']:
  used=sorted({i for p in corp[f'{split}_pairs'] for i in p}); frames=[corp['frames'][i] for i in used]
  grids=[parse(f) for f in frames]
  if game=='SET':
   old=[run(pool[1]['perception'],f) for f in frames];new=[run(pool[12]['perception'],f) for f in frames]
   ab=[re.sub(r'; shape=[^;]+; cursor:unknown; ','; cells=',z) for z in new]
   resort=[z.split('; cells=')[0]+'; cells='+' '.join(sorted(z.split('; cells=')[1].split())) for z in ab]
   results[game][split]={'old_ratio':ratio(old,frames),'new_ratio':ratio(new,frames),'after_remove_headers_ratio':ratio(ab,frames),'remove_headers_and_restore_sort_same_as_old':resort==old,'mean_sort_only_effect_gzip_bytes':statistics.mean(gz(a)-gz(o) for a,o in zip(ab,old))}
  elif game=='egg':
   old=[run(pool[5]['perception'],f) for f in frames];new=[run(pool[8]['perception'],f) for f in frames]
   ab=[re.sub(r'; \(0,0\)=[^;]+(?:; [^;]+_blob: [^;]+)+; cells:', '; cells:',z) for z in new]
   results[game][split]={'remove_00_and_blobs_same_as_node5':ab==old,'after_remove_ratio':ratio(ab,frames)}
  elif game=='diffusion':
   old=[run(pool[24]['perception'],f) for f in frames];new=[run(pool[28]['perception'],f) for f in frames]
   ab=[re.sub(r'^bg:[^;]+; dim:[^;]+; ','',z) for z in new]
   results[game][split]={'remove_bg_dim_same_as_node24':ab==old,'after_remove_ratio':ratio(ab,frames),'backgrounds':sorted({re.match('bg:([^;]+);',z).group(1) for z in new})}
  elif game=='n2ntd':
   old=[run(pool[1]['perception'],f) for f in frames];new=[run(pool[16]['perception'],f) for f in frames]
   groups=collections.defaultdict(list)
   for j,z in enumerate(old):groups[z].append(j)
   collisions=[]
   for z,js in groups.items():
    if len(js)>1:
     a,b=js[:2]
     diffs=[(r,c,grids[a][r][c],grids[b][r][c]) for r,row in enumerate(grids[a]) for c,cell in enumerate(row) if cell!=grids[b][r][c]]
     collisions.append({'frame_ids':[used[j] for j in js],'old_output':z,'new_outputs':[new[j] for j in js],'first_two_cell_differences':diffs})
   results[game][split]={'n_frames':len(frames),'old_unique':len(groups),'new_unique':len(set(new)),'collisions':collisions}
  elif game=='f5w3n':
   new=[run(pool[20]['perception'],f) for f in frames]
   ns={};exec(pool[20]['perception'],ns)
   aliases={v:k for k,v in ns['COLOUR_SHORT'].items()};aliases['g']='gray'
   exact=[]
   for z,g in zip(new,grids):
    bg=collections.Counter(c for row in g for c in row).most_common(1)[0][0]
    assert bg=='black' and len(g)==16 and len(g[0])==16
    dec=[['black']*16 for _ in range(16)]
    a,c=z.split(' | cells:')
    if a!='a:?':
     rr,cc=map(int,a[2:].split(','));dec[rr][cc]='orange'
    if c!='empty':
     for s in c.split(';'):
      rr,cc,col=s.split(',');dec[int(rr)][int(cc)]=aliases[col]
    exact.append(dec==g)
   results[game][split]={'n_frames':len(frames),'exact_raw_grid_reconstruction_all':all(exact),'n_exact':sum(exact),'assumed_invariants':'16x16 black background and colour-short dictionary'}
  elif game=='7xf97':
   new=[run(pool[23]['perception'],f) for f in frames]
   flag_values=collections.defaultdict(set)
   for z in new:
    ff=z.split(';f:')[1].split('|')[0]
    for kv in ff.split(','):
     k,v=kv.split('=');flag_values[k].add(v)
   results[game][split]={'flag_values':{k:sorted(v) for k,v in flag_values.items()},'n_frames':len(frames)}
(HERE/'field_checks.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
