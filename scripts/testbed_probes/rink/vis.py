import os
import sys, random, json
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from drv import *
ARROWS=['left','right','up','down']; ACTS=ARROWS+['noop']
D={'left':(-1,0),'right':(1,0),'up':(0,-1),'down':(0,1)}
def legal(p,a):
    if a=='noop': return True
    dx,dy=D[a]; x,y=p[0]+dx,p[1]+dy; return 0<=x<28 and 0<=y<28
def gridkey(it):
    # what the PIPELINE sees: off-grid cells clipped
    c,G=frame(it)
    return tuple(sorted((x,y,col) for (x,y),col in c.items() if 0<=x<G and 0<=y<G))
def replay(seed,acts):
    it=new(seed)
    for a in acts: act(it,a)
    return it

# guarded random drive; at each state branch all 5 actions
rng=random.Random(7)
vis1=[];vis3=[];dist=[];onice_dist=[];office_dist=[]
for seed in range(1,6):
    it=new(seed); prefix=[]; p=skater(it)
    for t in range(60):
        # branch
        nxt={}
        for a in ACTS:
            j=replay(seed,prefix); act(j,a); k1=gridkey(j)
            act(j,'noop'); act(j,'noop'); k3=gridkey(j)
            nxt[a]=(k1,k3)
        n1=len(set(v[0] for v in nxt.values()))
        base1=nxt['noop'][0]; base3=nxt['noop'][1]
        v1=sum(1 for a in ARROWS if nxt[a][0]!=base1)/4
        v3=sum(1 for a in ARROWS if nxt[a][1]!=base3)/4
        vis1.append(v1); vis3.append(v3); dist.append(n1)
        onice = 3<=p[0]<=24 and 3<=p[1]<=24
        (onice_dist if onice else office_dist).append(n1)
        cand=[a for a in ARROWS if legal(p,a)]
        a='noop' if rng.random()<0.2 else rng.choice(cand)
        act(it,a); prefix.append(a); p=skater(it)
print(f'guarded drive, 5 seeds x 60 branch points ({len(vis1)} points)')
print(f'  arrow visibility h1 = {sum(vis1)/len(vis1):.2f}   h3 = {sum(vis3)/len(vis3):.2f}')
print(f'  distinct next frames among 5 actions: mean {sum(dist)/len(dist):.2f}  (on-ice {sum(onice_dist)/max(1,len(onice_dist)):.2f} over {len(onice_dist)} pts, off-ice {sum(office_dist)/max(1,len(office_dist)):.2f} over {len(office_dist)} pts)')
