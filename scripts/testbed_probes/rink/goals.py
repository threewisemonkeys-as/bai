import os
import sys, random
from collections import deque
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import model
from model import tick, onice, INIT
ACTS=['noop','left','right','up','down']
D=model.D

def ongrid(st): return 0<=st[0]<28 and 0<=st[1]<28
def run(prefix, st=INIT):
    for a in prefix: st=tick(st,a)
    return st

def bfs(start, pred, cap=50, guard=True):
    """shortest action sequence whose ANY intermediate state satisfies pred"""
    seen={start[:3]}; q=deque([(start,[])])
    while q:
        st,path=q.popleft()
        if len(path)>=cap: continue
        for a in ACTS:
            ns=tick(st,a)
            if guard and not ongrid(ns): continue
            if pred(ns): return path+[a], ns
            k=ns[:3]
            if k in seen: continue
            seen.add(k); q.append((ns,path+[a]))
    return None, None

def rand_floor(start, pred, trials=25, horizon=50, guard=True, seed0=0):
    hit=0
    for i in range(trials):
        rng=random.Random(seed0*1000+i); st=start; ok=False
        for t in range(horizon):
            cand=[a for a in ACTS if (not guard) or ongrid(tick(st,a))] or ['noop']
            st=tick(st,rng.choice(cand))
            if pred(st): ok=True; break
        hit+=ok
    return hit/trials

def rest(st):
    """absorbing: a noop leaves the frame unchanged"""
    return tick(st,'noop')[:2]==st[:2]
