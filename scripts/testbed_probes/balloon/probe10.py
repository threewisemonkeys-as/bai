import sys, random
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
from collections import deque

def replay(prefix, seed=1):
    it=new(seed); f=frame(it)
    for a in prefix: f=act(it,a)
    return it,f

def st(f):
    o=origin(f); return (o[1], frozenset(rocks(f)))

def acts(f):
    o=origin(f)
    A=["noop"]+[(o[0]+dx,o[1]+dy) for dx,dy in [(-1,6),(0,6),(1,6),(-1,7),(0,7),(1,7)]]
    A+= [r for r in rocks(f)]
    out=[];seen=set()
    for a in A:
        if a not in seen: seen.add(a); out.append(a)
    return out

def bfs(start_prefix, goalfn, maxdepth=14, cap=40000):
    it,f0=replay(start_prefix)
    if goalfn(f0, [f0]): return [], 0
    seen={st(f0)}; q=deque([([], f0)]); n=0
    while q:
        pre, f = q.popleft()
        if len(pre)>=maxdepth: continue
        for a in acts(f):
            n+=1
            if n>cap: return None,n
            it2,_=replay(start_prefix+pre)
            f2=act(it2,a)
            k=st(f2)
            if k in seen: continue
            seen.add(k)
            npre=pre+[a]
            # any-step trajectory check
            _,ftrace = replay(start_prefix)
            if goalfn(f2, None): return npre, n
            q.append((npre,f2))
    return None,n

TOP=6*["noop"]
def yy(f): return origin(f)[1]
GOALS = {
 "G1 float to ceiling (from bottom start)":      ([],       lambda f,_=None: yy(f)==2),
 "G2 land on the ground (from top start)":       (TOP,      lambda f,_=None: yy(f)==7),
 "G3 carry exactly 3 rocks (from bottom start)": ([],       lambda f,_=None: len(rocks(f))==3),
 "G4 exact frame: bottom + rocks (6,13)(7,13)(8,13)": ([], lambda f,_=None: yy(f)==7 and set(rocks(f))=={(6,13),(7,13),(8,13)}),
 "G5 empty basket at ceiling (from 3-rock bottom)": (None, None),
 "G6 hover row 4 with 1 rock":                   ([],       lambda f,_=None: yy(f)==4 and len(rocks(f))==1),
}
for name,(pre,g) in GOALS.items():
    if pre is None: continue
    p,n = bfs(pre,g,maxdepth=12)
    print(f"{name}: plan_len={len(p) if p is not None else 'UNREACHABLE'} nodes={n} plan={p}")

# G5 start: sunk with 3 rocks
S5 = TOP+[(6,8),(7,8),(8,8)]+5*["noop"]
it,f=replay(S5); print("\nG5 start state: origin",origin(f),"rocks",rocks(f))
p,n=bfs(S5, lambda f,_=None: yy(f)==2 and len(rocks(f))==0, maxdepth=14)
print("G5 remove all rocks and return to ceiling: plan_len=", len(p) if p else 'UNREACH', p)
