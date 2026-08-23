import os
import sys, random
from collections import defaultdict
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from fast import spos, obs
from drv import new
ARROWS=['left','right','up','down']
D={'left':(-1,0),'right':(1,0),'up':(0,-1),'down':(0,1)}
def legal(p,a):
    if a=='noop': return True
    dx,dy=D[a]; return 0<=p[0]+dx<28 and 0<=p[1]+dy<28
def onice(p): return 3<=p[0]<=24 and 3<=p[1]<=24
def rstep(it,a):
    if a!='noop': getattr(it,a)()
    it.step(); return spos(it)
def iceseek(rng,p):
    if onice(p):
        if rng.random()<0.35: return 'noop'
        return rng.choice([a for a in ARROWS if legal(p,a)])
    if rng.random()<0.1: return 'noop'
    want=[]
    if p[0]<3: want.append('right')
    if p[0]>24: want.append('left')
    if p[1]<3: want.append('down')
    if p[1]>24: want.append('up')
    if want and rng.random()<0.8: return rng.choice(want)
    return rng.choice([a for a in ARROWS if legal(p,a)])

T=[]
for seed in range(1,21):
    rng=random.Random(seed*31); it=new(seed); p=spos(it); hist=[obs(p)]
    for t in range(200):
        a=iceseek(rng,p); p=rstep(it,a); hist.append(obs(p))
        T.append((hist[-3] if len(hist)>=3 else None, hist[-2], a, hist[-1], onice(p)))
print('transitions:', len(T))
for K,keyf in [(1, lambda r:(r[1],r[2])), (2, lambda r:(r[0],r[1],r[2]))]:
    g=defaultdict(set)
    for r in T: g[keyf(r)].add(r[3])
    bad=[k for k,v in g.items() if len(v)>1]
    cov=sum(len(v) for k,v in g.items() if len(v)>1)
    print(f'  window K={K}: {len(g)} distinct (window,action) keys, {len(bad)} ambiguous '
          f'({100*len(bad)/len(g):.1f}%)')
    if K==1 and bad:
        k=bad[0]; print('    example alias:', k, '->', sorted(g[k]))
