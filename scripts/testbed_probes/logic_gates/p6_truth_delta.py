import sys, itertools; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
ON={"red","orange","yellow"}
def rd(c):
    g=lambda p: c.get(p,"black")
    return dict(s1=g((4,12)) in ON, s2=g((19,12)) in ON, AND=g((12,4)) in ON, OR=g((12,8)) in ON,
                NOT=g((12,16)) in ON, XOR=g((12,20)) in ON, w1=g((5,4)) in ON, w2=g((14,4)) in ON)
def run(pre):
    it=load("logic_gates",1); frame(it)
    fs=[frame(it)[0]]
    for a in pre:
        if a=="s1": it.click(4,12)
        elif a=="s2": it.click(19,12)
        it.step(); fs.append(frame(it)[0])
    return fs
# truth table over settled states
print("SETTLED TRUTH TABLE (2 noops after each click to settle):")
for tgt in [(0,0),(1,0),(0,1),(1,1)]:
    pre=[]
    if tgt[0]: pre.append("s1")
    if tgt[1]: pre.append("s2")
    pre += ["noop","noop","noop"]
    f=run(pre)[-1]; r=rd(f)
    print(f"  s1={int(r['s1'])} s2={int(r['s2'])} -> AND={int(r['AND'])} OR={int(r['OR'])} NOT={int(r['NOT'])} XOR={int(r['XOR'])} wires1={int(r['w1'])} wires2={int(r['w2'])}"
          f"   [check AND={int(r['s1'] and r['s2'])} OR={int(r['s1'] or r['s2'])} NOT={int(not r['s1'])} XOR={int(r['s1']!=r['s2'])}]")
# delta magnitudes across all 12x3 transitions
import collections
prefixes = {}
from collections import deque
def canon(pre):
    it=load("logic_gates",1); frame(it)
    for a in pre:
        if a=="s1": it.click(4,12)
        elif a=="s2": it.click(19,12)
        it.step(); c,_=frame(it)
    if not pre: c,_=frame(it)
    return c
seen={}; q=deque([[]])
while q:
    p=q.popleft()
    if len(p)>6: continue
    for a in ["noop","s1","s2"]:
        n=p+[a]; c=canon(n); k=key(c)
        if k not in seen: seen[k]=(n,c); q.append(n)
states=sorted(seen.items(), key=lambda kv: len(kv[1][0]))
print(f"\nreachable-from-t=1 closed set: {len(states)}")
deltas=collections.Counter()
for k,(p,c) in states:
    for a in ["noop","s1","s2"]:
        c2=canon(p+[a])
        d=sum(1 for pos in set(c)|set(c2) if c.get(pos,"black")!=c2.get(pos,"black"))
        deltas[(a,d)]+=1
print("changed-cell counts per (action, ncells):")
for (a,d),n in sorted(deltas.items()): print(f"   action={a:5s} changed_cells={d:3d}  x{n}")
