import sys; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
from collections import deque
base=load("lights_new",1); frame(base); base.step(); nf,_=frame(base)
hits=[]
for y in range(24):
    for x in range(24):
        it=load("lights_new",1); frame(it); it.click(x,y); it.step(); c,_=frame(it)
        if key(c)!=key(nf): hits.append((x,y))
print("lights_new clickable:",len(hits),sorted(hits))
# group into switches by effect
ACT=[("noop",None)]+[(f"click_{x}_{y}",(x,y)) for (x,y) in hits]
def canon(pre):
    it=load("lights_new",1); frame(it)
    for a,arg in pre:
        if arg: it.click(*arg)
        it.step(); c,_=frame(it)
    if not pre: c,_=frame(it)
    return c
seen={key(canon([])):([],canon([]))}; q=deque([[]])
while q:
    p=q.popleft()
    if len(p)>5: continue
    for a in ACT:
        n=p+[a]; c=canon(n); k=key(c)
        if k not in seen: seen[k]=(n,c); q.append(n)
print("lights_new distinct frames (depth<=6):",len(seen))
