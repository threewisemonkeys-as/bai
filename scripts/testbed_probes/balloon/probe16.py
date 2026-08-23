import sys, random
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
from collections import defaultdict
def key(c): return tuple(sorted(c.items()))
alias=defaultdict(set); states=set()
for s in range(1,41):
    rng=random.Random(500+s); it=new(s); f=frame(it)
    for t in range(80):
        o=origin(f); r=rng.random()
        if r<0.4: a="noop"
        elif r<0.75:
            dx,dy=rng.choice([(-1,6),(0,6),(1,6),(-1,7),(0,7),(1,7)]); a=(o[0]+dx,o[1]+dy)
        elif r<0.9 and rocks(f): a=rng.choice(rocks(f))
        else: a=(rng.randrange(16),rng.randrange(16))
        k0=key(f); f=act(it,a)
        alias[(k0, a if isinstance(a,str) else ("c",)+a)].add(key(f)); states.add(key(f))
bad=[k for k,v in alias.items() if len(v)>1]
print("targeted drives: pairs=",len(alias)," unique states=",len(states)," ALIASED pairs=",len(bad))
# is the rock count always fully visible (never occluded)?
occl=0; tot=0
for s in range(1,21):
    rng=random.Random(900+s); it=new(s); f=frame(it)
    for t in range(60):
        o=origin(f); dx,dy=rng.choice([(-1,6),(0,6),(1,6),(-1,7),(0,7),(1,7)])
        a=(o[0]+dx,o[1]+dy) if rng.random()<0.5 else "noop"
        f=act(it,a)
        import json as J
        d=J.loads(it.render_all()); nr=len(d.get("rocks",[]))
        vis=sum(1 for v in f.values() if v in ("gray","grey"))
        tot+=1
        if nr!=vis: occl+=1
print(f"frames where #rock objects != #visible gray cells: {occl}/{tot}")
