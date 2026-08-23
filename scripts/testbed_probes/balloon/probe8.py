import sys, random, json
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
from collections import Counter, defaultdict

def key(cells): return tuple(sorted(cells.items()))

print("=== J) stdout pollution from (print num_contained)? ===")
import io, contextlib
buf=io.StringIO()
with contextlib.redirect_stdout(buf):
    it=new(1)
    for _ in range(5): act(it,"noop")
print(" captured python-level stdout chars:", len(buf.getvalue()), repr(buf.getvalue()[:80]))

print("\n=== K) random-drive click hit rate (20 drives x 60 steps, seeds 1..20) ===")
tot_click=0; hit_click=0; tot_steps=0; changed=0; nonpassive=0
uniq_states=set(); alias=defaultdict(set)
for s in range(1,21):
    rng=random.Random(s)
    it=new(s); f=frame(it)
    for t in range(60):
        a = rng.choice(["noop","up","down","left","right"] + ["click"]*5)
        if a=="click": a=(rng.randrange(16), rng.randrange(16))
        before_rocks=set(rocks(f)); k0=key(f)
        f2=act(it,a)
        tot_steps+=1
        if isinstance(a,tuple):
            tot_click+=1
            if set(rocks(f2))!=before_rocks: hit_click+=1
        if f2!=f: changed+=1
        alias[(k0, a if isinstance(a,str) else ("click",)+a)].add(key(f2))
        uniq_states.add(key(f2))
        f=f2
print(f" clicks={tot_click} hits(rock-set changed)={hit_click} rate={hit_click/max(1,tot_click):.4f}")
print(f" steps={tot_steps} frame-changed={changed} ({changed/tot_steps:.3f})  unique states={len(uniq_states)}")
bad=[k for k,v in alias.items() if len(v)>1]
print(f" (frame,action) pairs with >1 successor (ALIASING): {len(bad)} / {len(alias)}")

print("\n=== L) informative content of a random drive: rock counts seen ===")
cnt=Counter()
for s in range(1,21):
    rng=random.Random(100+s); it=new(s); f=frame(it)
    for t in range(60):
        a = rng.choice(["noop"]*3 + ["click"]*7)
        if a=="click": a=(rng.randrange(16), rng.randrange(16))
        f=act(it,a); cnt[len(rocks(f))]+=1
print(" rock-count histogram over 1200 random-drive frames:", dict(sorted(cnt.items())))

print("\n=== M) OBJECT-TARGETED click policy (click a basket-interior cell) ===")
tot=0; hit=0; cnts=Counter()
for s in range(1,21):
    rng=random.Random(200+s); it=new(s); f=frame(it)
    for t in range(60):
        if rng.random()<0.5:
            o=origin(f); dx,dy=rng.choice([(-1,6),(0,6),(1,6),(-1,7),(0,7),(1,7)])
            a=(o[0]+dx,o[1]+dy); before=set(rocks(f)); f=act(it,a); tot+=1
            if set(rocks(f))!=before: hit+=1
        else:
            f=act(it,"noop")
        cnts[len(rocks(f))]+=1
print(f" targeted clicks={tot} hits={hit} rate={hit/tot:.3f}")
print(" rock-count histogram:", dict(sorted(cnts.items())))
