import os
import sys, random
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from fast import *
from drv import new
ARROWS=['left','right','up','down']; ACTS=ARROWS+['noop']
D={'left':(-1,0),'right':(1,0),'up':(0,-1),'down':(0,1)}
def legal(p,a):
    if a=='noop': return True
    dx,dy=D[a]; return 0<=p[0]+dx<28 and 0<=p[1]+dy<28

def branch(seed, prefix, h=3):
    out={}
    for a in ACTS:
        it=replay(seed,prefix); p1=obs(step(it,a))
        seq=[p1]
        for _ in range(h-1): seq.append(obs(step(it,'noop')))
        out[a]=(p1, tuple(seq))
    return out

vis1=vis3=dist=0; n=0; on_dist=[]; off_dist=[]; alias1=0
for seed in range(1,6):
    it=new(seed); prefix=[]; p=spos(it)
    for t in range(40):
        b=branch(seed,prefix)
        base1=b['noop'][0]; base3=b['noop'][1]
        vis1+=sum(1 for a in ARROWS if b[a][0]!=base1)/4
        vis3+=sum(1 for a in ARROWS if b[a][1]!=base3)/4
        d=len(set(v[0] for v in b.values())); dist+=d; n+=1
        onice = 3<=p[0]<=24 and 3<=p[1]<=24
        (on_dist if onice else off_dist).append(d)
        cand=[a for a in ARROWS if legal(p,a)]
        a='noop' if random.Random(seed*1000+t).random()<0.2 else random.Random(seed*97+t).choice(cand)
        p=step(it,a); prefix.append(a)
print(f'GUARDED drive, 5 seeds x 40 branch points (n={n})')
print(f'  arrow visibility h1={vis1/n:.2f}  h3={vis3/n:.2f}')
print(f'  distinct next frames / 5 actions: {dist/n:.2f}')
print(f'    on-ice  points {len(on_dist):3d}: {sum(on_dist)/max(1,len(on_dist)):.2f}')
print(f'    off-ice points {len(off_dist):3d}: {sum(off_dist)/max(1,len(off_dist)):.2f}')
