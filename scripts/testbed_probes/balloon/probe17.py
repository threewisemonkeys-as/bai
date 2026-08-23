import sys, random, json
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
from collections import Counter
cause=Counter(); example=None
for s in range(1,21):
    rng=random.Random(900+s); it=new(s); f=frame(it)
    hist=[]
    for t in range(60):
        o=origin(f); dx,dy=rng.choice([(-1,6),(0,6),(1,6),(-1,7),(0,7),(1,7)])
        a=(o[0]+dx,o[1]+dy) if rng.random()<0.5 else "noop"
        hist.append(a); f=act(it,a)
        d=json.loads(it.render_all()); rl=[(e["position"]["x"],e["position"]["y"]) for e in d.get("rocks",[])]
        vis=sorted(p for p,c in f.items() if c in ("gray","grey"))
        if len(rl)!=len(vis):
            dup=[p for p,c in Counter(rl).items() if c>1]
            off=[p for p in rl if not(0<=p[0]<16 and 0<=p[1]<16)]
            cause["duplicate cell"]+= 1 if dup else 0
            cause["off grid"]+= 1 if off else 0
            cause["other"]+= 1 if not dup and not off else 0
            if example is None: example=(s,t,sorted(rl),vis,dup,off)
print(cause)
print("example seed,t:",example[0],example[1]); print(" rock objects:",example[2]); print(" visible gray:",example[3]); print(" dups:",example[4]," offgrid:",example[5])
# Does a duplicate change the threshold reading? construct one deliberately
print("\n=== deliberate stack test ===")
it=new(1); f=frame(it)
f=act(it,(7,14))       # lower interior row while rising -> stranded at (7,14)
for _ in range(6): f=act(it,"noop")
print(" after strand:", origin(f), rocks(f))
o=origin(f); f=act(it,(o[0],o[1]+7)); print(" add rock lower row:", origin(f), rocks(f))
for _ in range(8): f=act(it,"noop"); print("  ", origin(f), rocks(f), "n_obj=",len(json.loads(it.render_all()).get("rocks",[])))
