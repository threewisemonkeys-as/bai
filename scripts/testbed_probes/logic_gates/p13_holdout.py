import sys; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
from collections import deque
ON={"red","orange","yellow"}
def lat(c):
    g=lambda p: c.get(p,"black")
    return (int(g((4,12)) in ON), int(g((19,12)) in ON), int(g((5,4)) in ON), int(g((14,4)) in ON))
def outs(c):
    g=lambda p: c.get(p,"black")
    return tuple(int(g(p) in ON) for p in [(12,4),(12,8),(12,16),(12,20)])
def canon(pre):
    it=load("logic_gates",1); frame(it)
    for a in pre:
        if a=="s1": it.click(4,12)
        elif a=="s2": it.click(19,12)
        it.step(); c,_=frame(it)
    return frame(it)[0] if not pre else c
seen={}; q=deque([[]]); f=canon([]); seen[key(f)]=([],f)
while q:
    p=q.popleft()
    if len(p)>6: continue
    for a in ["noop","s1","s2"]:
        n=p+[a]; c=canon(n); k=key(c)
        if k not in seen: seen[k]=(n,c); q.append(n)
closed={k:v for k,v in seen.items() if len(v[0])>0}
def is_test(c):
    s1,s2,w1,w2 = lat(c)
    return (s1 and s2) or (w1 and w2)
tr=[k for k,v in closed.items() if not is_test(v[1])]
te=[k for k,v in closed.items() if is_test(v[1])]
print(f"closed set {len(closed)} states -> TRAIN(no both-on) {len(tr)}  TEST(both-on) {len(te)}")
ntr=nte=0
for k,(p,c) in closed.items():
    for a in ["noop","s1","s2"]:
        c2=canon(p+[a])
        if is_test(c) or is_test(c2): nte+=1
        else: ntr+=1
print(f"transitions: train-only(both endpoints in train region) {ntr}, test {nte}, total {ntr+nte}")
print("\nWhat the 4 gates look like on the TRAIN states only (settled):")
for tgt in [(0,0),(1,0),(0,1)]:
    pre=(["s1"] if tgt[0] else [])+(["s2"] if tgt[1] else [])+["noop"]*4
    print(f"  s=({tgt[0]},{tgt[1]}): AND,OR,NOT,XOR = {outs(canon(pre))}")
print("  HELD-OUT s=(1,1):        AND,OR,NOT,XOR =", outs(canon(["s1","s2"]+["noop"]*4)))
cols={}
for i,nm in enumerate(["AND","OR","NOT","XOR"]):
    col=tuple(outs(canon((["s1"] if t[0] else [])+(["s2"] if t[1] else [])+["noop"]*4))[i] for t in [(0,0),(1,0),(0,1)])
    cols.setdefault(col,[]).append(nm)
print("\n  gates INDISTINGUISHABLE on the train region:", {k:v for k,v in cols.items() if len(v)>1})
