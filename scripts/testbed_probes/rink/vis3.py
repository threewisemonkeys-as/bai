import os
import sys, random
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from fast import spos, obs
from drv import new
ARROWS=['left','right','up','down']; ACTS=ARROWS+['noop']
D={'left':(-1,0),'right':(1,0),'up':(0,-1),'down':(0,1)}
def legal(p,a):
    if a=='noop': return True
    dx,dy=D[a]; return 0<=p[0]+dx<28 and 0<=p[1]+dy<28
def fstep(it,a):           # no render (rink uses no collision primitive; A/B verified)
    if a!='noop': getattr(it,a)()
    it.step()
def rstep(it,a):
    fstep(it,a); return spos(it)
def fast_replay(seed,acts):
    it=new(seed)
    for a in acts: fstep(it,a)
    return it

def branch(seed,prefix,h=3):
    out={}
    for a in ACTS:
        it=fast_replay(seed,prefix)
        seq=[obs(rstep(it,a))]
        for _ in range(h-1): seq.append(obs(rstep(it,'noop')))
        out[a]=(seq[0],tuple(seq))
    return out

vis1=vis3=dist=0.0; n=0; on_d=[]; off_d=[]
for seed in range(1,5):
    rng=random.Random(seed)
    it=new(seed); prefix=[]; p=spos(it)
    for t in range(30):
        b=branch(seed,prefix)
        base1=b['noop'][0]; base3=b['noop'][1]
        vis1+=sum(1 for a in ARROWS if b[a][0]!=base1)/4
        vis3+=sum(1 for a in ARROWS if b[a][1]!=base3)/4
        d=len(set(v[0] for v in b.values())); dist+=d; n+=1
        onice = 3<=p[0]<=24 and 3<=p[1]<=24
        (on_d if onice else off_d).append(d)
        cand=[a for a in ARROWS if legal(p,a)]
        a='noop' if rng.random()<0.2 else rng.choice(cand)
        p=rstep(it,a); prefix.append(a)
print(f'GUARDED drive, 4 seeds x 30 branch points (n={n})')
print(f'  arrow visibility h1={vis1/n:.2f}  h3={vis3/n:.2f}')
print(f'  distinct next frames / 5 actions: {dist/n:.2f}')
print(f'    on-ice  {len(on_d):3d} pts: {sum(on_d)/max(1,len(on_d)):.2f}')
print(f'    off-ice {len(off_d):3d} pts: {sum(off_d)/max(1,len(off_d)):.2f}')
