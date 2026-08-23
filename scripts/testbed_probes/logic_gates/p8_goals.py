import sys, random; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
ON={"red","orange","yellow"}
def rd(c):
    g=lambda p: c.get(p,"black")
    return dict(s1=g((4,12)) in ON, s2=g((19,12)) in ON,
                AND=g((12,4)) in ON, OR=g((12,8)) in ON, NOT=g((12,16)) in ON, XOR=g((12,20)) in ON,
                w1=g((5,4)) in ON, w2=g((14,4)) in ON)
def nlit(r): return sum(r[k] for k in ("AND","OR","NOT","XOR"))
GOALS = {
 "G1 light the AND output":            lambda r: r["AND"],
 "G2 light the XOR output":            lambda r: r["XOR"],
 "G3 make all four outputs dark":      lambda r: nlit(r)==0,
 "G4 light exactly two outputs":       lambda r: nlit(r)==2,
 "G5 light exactly three outputs":     lambda r: nlit(r)==3,
 "G6 make the NOT output dark":        lambda r: not r["NOT"],
 "G7 light AND while all wires yellow":lambda r: r["AND"] and r["w1"] and r["w2"],
 "G8 exactly one output lit":          lambda r: nlit(r)==1,
 "G9 AND lit and XOR dark":            lambda r: r["AND"] and not r["XOR"],
 "G10 all wires grey but some output lit": lambda r: (not r["w1"]) and (not r["w2"]) and nlit(r)>0,
}
# 1) exhaustive reachability + min actions via BFS over rendered frames (any-step)
from collections import deque
ACT=["noop","s1","s2"]
def canon(pre):
    it=load("logic_gates",1); frame(it)
    for a in pre:
        if a=="s1": it.click(4,12)
        elif a=="s2": it.click(19,12)
        it.step(); c,_=frame(it)
    if not pre: c,_=frame(it)
    return c
seen={}; q=deque([[]]); f0=canon([]); seen[key(f0)]=([],f0)
while q:
    p=q.popleft()
    if len(p)>7: continue
    for a in ACT:
        n=p+[a]; c=canon(n); k=key(c)
        if k not in seen: seen[k]=(n,c); q.append(n)
print(f"reachable frames incl. t=0: {len(seen)}")
print(f"{'goal':42s} {'reach':6s} {'min_actions':11s} example")
for name,f in GOALS.items():
    best=None
    for k,(p,c) in seen.items():
        if f(rd(c)) and (best is None or len(p)<len(best)): best=p
    print(f"{name:42s} {'YES' if best is not None else 'NO ':6s} {str(len(best)) if best is not None else '-':11s} {best}")
# 2) random floor: 25 rollouts x 50 uniform actions over {noop,l,r,u,d} + all 576 clicks
random.seed(7)
N,H=25,50
hits={k:0 for k in GOALS}; switch_hits=[]
for t in range(N):
    it=load("logic_gates",1); frame(it)
    grids=[frame(it)[0]]; sh=0
    for _ in range(H):
        a=random.randrange(581)
        if a==0: pass
        elif a==1: it.left()
        elif a==2: it.right()
        elif a==3: it.up()
        elif a==4: it.down()
        else:
            cx=(a-5)%24; cy=(a-5)//24; it.click(cx,cy)
            if (cx,cy) in {(4,12),(4,13),(5,12),(5,13),(19,12),(19,13),(20,12),(20,13)}: sh+=1
        it.step(); grids.append(frame(it)[0])
    switch_hits.append(sh)
    for name,f in GOALS.items():
        if any(f(rd(g)) for g in grids): hits[name]+=1
print(f"\nrandom floor ({N} rollouts x {H} actions, any-step). mean switch-cell hits/rollout = {sum(switch_hits)/N:.2f}, rollouts with >=1 hit = {sum(1 for s in switch_hits if s>0)}/{N}")
for name in GOALS: print(f"  {name:42s} floor={hits[name]/N:.2f}")
