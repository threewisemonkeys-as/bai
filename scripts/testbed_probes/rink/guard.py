import os
import sys, random, json
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from drv import *
ARROWS=['left','right','up','down']
D={'left':(-1,0),'right':(1,0),'up':(0,-1),'down':(0,1)}

def legal(p, a):
    """reject an arrow that would walk the skater off the grid (only matters off the ice)."""
    if a=='noop': return True
    dx,dy=D[a]; x,y=p[0]+dx,p[1]+dy
    return 0<=x<28 and 0<=y<28

def run(seed,n=100,p_noop=0.2,guard=True):
    rng=random.Random(seed); it=new(seed)
    off=0; onice=0; slides=0; prev=skater(it); acts=[]
    for t in range(n):
        cand=['noop'] if rng.random()<p_noop else ARROWS[:]
        if guard:
            cand=[a for a in (cand if cand!=['noop'] else ARROWS) if legal(prev,a)] or ['noop']
            a = 'noop' if rng.random()<p_noop else rng.choice(cand)
            if not legal(prev,a): a='noop'
        else:
            a = 'noop' if rng.random()<p_noop else rng.choice(ARROWS)
        act(it,a); p=skater(it); acts.append(a)
        if not (0<=p[0]<28 and 0<=p[1]<28): off+=1
        if 3<=p[0]<=24 and 3<=p[1]<=24: onice+=1
        if abs(p[0]-prev[0])+abs(p[1]-prev[1])==2: slides+=1
        prev=p
    return off,onice,slides

for guard in (False,True):
    O=[];I=[];S=[]
    for s in range(1,21):
        o,i,sl=run(s,100,guard=guard); O.append(o);I.append(i);S.append(sl)
    print(f'guard={guard}: off-grid ticks/100 mean {sum(O)/20:.1f} (drives w/ any off: {sum(1 for o in O if o)}/20), on-ice ticks/100 mean {sum(I)/20:.1f}, slide-moves/100 mean {sum(S)/20:.1f}')
