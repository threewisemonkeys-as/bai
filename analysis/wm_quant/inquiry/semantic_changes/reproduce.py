import csv,gzip,json,statistics,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
BASE=ROOT/'logs/2026-08-24/human_curated/rexpure'
selected={'n2ntd':[1,3,6,15,16],'f5w3n':[1,10,16,20],'s2kt7':[1,4,23],'egg':[2,5,8],'SET':[1,12],'diffusion':[1,2,18,21,24,28],'7xf97':[1,23],'bt3gb':[2,12],'dino':[1,18],'va6fq':[1,24],'colour_lines':[1,12]}
rows={(r['game'],int(r['idx']),r['split']):r for r in csv.DictReader((ROOT/'analysis/perception_metrics/metrics.csv').open())}
def gz(s):return len(gzip.compress(s.encode(),6,mtime=0))
def run(code,frame):
 ns={};exec(code,ns);return ns['perceive']([frame])
summary={}
for game,ids in selected.items():
 pool={n['idx']:n for n in map(json.loads,(BASE/f'{game}_s1/rexpure_run_seed1/candidates.jsonl').read_text().splitlines())}
 with gzip.open(ROOT/f'analysis/perception_metrics/cache/{game}.json.gz','rt') as f:corp=json.load(f)
 by_split={}
 for split in ['train','test']:
  used=sorted({i for p in corp[f'{split}_pairs'] for i in p});frames=[corp['frames'][i] for i in used]
  allouts={idx:[run(pool[idx]['perception'],f) for f in frames] for idx in ids}
  computed={}
  for idx,outs in allouts.items():
   ratio=statistics.mean(gz(z)/gz(f) for z,f in zip(outs,frames))
   expected=float(rows[game,idx,split]['info_extraction_ratio'])
   assert abs(ratio-expected)<1e-12,(game,idx,split,ratio,expected)
   computed[idx]={'info_ratio':ratio,'mean_gzip_out':statistics.mean(map(gz,outs)),'mean_out_chars':statistics.mean(map(len,outs)),'unique_outputs':len(set(outs)),'n_frames':len(frames)}
  examples=[]
  for j in sorted(set([0,len(frames)//2,len(frames)-1])):
   examples.append({'frame_id':used[j],'raw':frames[j],'raw_gzip':gz(frames[j]),'outputs':{idx:{'text':outs[j],'chars':len(outs[j]),'gzip_bytes':gz(outs[j])} for idx,outs in allouts.items()}})
  extra={}
  if game=='s2kt7':
   z=allouts[23]
   ops={
    'remove_sid':lambda o:re.sub(r'sid:[^;]+; ','',o),
    'remove_step':lambda o:re.sub(r'^step:[^;]+; ','',o),
    'remove_delta':lambda o:o.split('; added:')[0],
    'remove_sid_step_delta':lambda o:re.sub(r'^step:[^;]+; sid:[^;]+; ','',o.split('; added:')[0]),
   }
   for name,op in ops.items():
    ab=[op(o) for o in z]
    extra[name]={'ratio':statistics.mean(gz(z)/gz(f) for z,f in zip(ab,frames)),'same_as_node4':all(a==b for a,b in zip(ab,allouts[4]))}
   extra['step_zero_all']=all(o.startswith('step:0;') for o in z)
   extra['added_duplicates_cells_all']=all(o.split('cells:',1)[1].split('; added:')[0]==o.split('; added:',1)[1].split('; removed:')[0] for o in z)
  if game=='egg':
   extra['deltas_activated']=sum('delta:' in o for o in allouts[8])
  if game=='SET':
   extra['cursor_unknown_all']=all('cursor:unknown' in o for o in allouts[12])
  if game=='f5w3n':
   ns={};exec(pool[20]['perception'],ns)
   grids=[json.loads(f[f.find('[['):f.rfind(']]')+2]) for f in frames]
   extra['unknown_colors']=sorted({c for grid in grids for row in grid for c in row if c not in ns['COLOUR_SHORT']})
   extra['max_orange_count']=max(sum(c=='orange' for row in grid for c in row) for grid in grids)
   extra['hash_truncation_count']=sum('##' in o for o in allouts[20])
  by_split[split]={'metrics':computed,'examples':examples,'ablations':extra}
 summary[game]=by_split
(HERE/'semantic_outputs.json').write_text(json.dumps(summary,indent=2))
for game,sp in summary.items():
 print(game)
 print(json.dumps(sp['train']['metrics']))
 if sp['train']['ablations']:print(json.dumps(sp['train']['ablations']))
 print(json.dumps(sp['train']['examples'][0]['outputs']))
