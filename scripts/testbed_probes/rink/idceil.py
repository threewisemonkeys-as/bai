import os
import sys, random
from collections import deque
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from goals import *
import model
from model import tick, onice, INIT
ACTS=['noop','left','right','up','down']
def obs(st): return (st[0],st[1]) if (0<=st[0]<28 and 0<=st[1]<28) else None

def credited(st):
    """1/|alias class| averaged over the 5 actions, using ONLY the rendered next frame."""
    nx={a: obs(tick(st,a)) for a in ACTS}
    from collections import Counter
    c=Counter(nx.values())
    return sum(1.0/c[nx[a]] for a in ACTS)/len(ACTS), len(set(nx.values()))

# over the whole reachable on-grid state space
seen={INIT[:3]}; q=deque([INIT]); tot=0; n=0; on=[]; off=[]
while q:
    st=q.popleft(); c,_=credited(st); tot+=c; n+=1
    (on if onice(st[0],st[1]) else off).append(c)
    for a in ACTS:
        ns=tick(st,a)
        if not ongrid(ns) or ns[:3] in seen: continue
        seen.add(ns[:3]); q.append(ns)
print(f'credited-ID ceiling over ALL {n} reachable on-grid states: {tot/n:.3f}')
print(f'   on-ice states  {len(on):4d}: {sum(on)/len(on):.3f}')
print(f'   off-ice states {len(off):4d}: {sum(off)/len(off):.3f}')

# same but along an ice-seeking drive (the realistic data distribution)
def iceseek(rng,st):
    x,y=st[0],st[1]
    cand=[a for a in ACTS if ongrid(tick(st,a))]
    if onice(x,y): return rng.choice(cand)
    want=[]
    if x<3: want.append('right')
    if x>24: want.append('left')
    if y<3: want.append('down')
    if y>24: want.append('up')
    want=[a for a in want if a in cand]
    if want and rng.random()<0.8: return rng.choice(want)
    return rng.choice(cand)
tot=0;n=0;ic=0
for s in range(20):
    rng=random.Random(s); st=INIT
    for t in range(200):
        c,_=credited(st); tot+=c; n+=1; ic+= onice(st[0],st[1])
        st=tick(st,iceseek(rng,st))
print(f'credited-ID ceiling on a guarded ice-seeking drive (20x200): {tot/n:.3f}  (on-ice fraction {ic/n:.2f})')

# walking-only shortest path (0,0)->(26,26) avoiding the ice
q=deque([((0,0),0)]); seen2={(0,0)}
best=None
while q:
    (x,y),d=q.popleft()
    if (x,y)==(26,26): best=d; break
    for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
        nx,ny=x+dx,y+dy
        if not(0<=nx<28 and 0<=ny<28) or onice(nx,ny) or (nx,ny) in seen2: continue
        seen2.add((nx,ny)); q.append(((nx,ny),d+1))
print(f'shortest ICE-FREE walk (0,0)->(26,26): {best} actions  (PLAN_CAP=50 curated / 20 coverage)')
