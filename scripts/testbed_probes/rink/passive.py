import os
import sys, random
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from goals import *
from model import tick, onice, INIT
ACTS=['noop','left','right','up','down']
def ob(st): return (st[0],st[1]) if (0<=st[0]<28 and 0<=st[1]<28) else None
def iceseek(rng,st):
    x,y=st[0],st[1]; cand=[a for a in ACTS if ongrid(tick(st,a))]
    if onice(x,y): return rng.choice(cand)
    want=[a for a,c in (('right',x<3),('left',x>24),('down',y<3),('up',y>24)) if c and a in cand]
    if want and rng.random()<0.8: return rng.choice(want)
    return rng.choice(cand)
for name,pol in [('unguarded uniform', lambda rng,st: rng.choice(ACTS)),
                 ('guarded uniform',   lambda rng,st: rng.choice([a for a in ACTS if ongrid(tick(st,a))])),
                 ('ice-seeking',       iceseek)]:
    nc=0; n=0; changed=0; tot=0; offg=0
    for s in range(20):
        rng=random.Random(s*7); st=INIT
        for t in range(200):
            # noop-change at this state
            if ob(tick(st,'noop'))!=ob(st): nc+=1
            n+=1
            a=pol(rng,st); ns=tick(st,a)
            if ob(ns)!=ob(st): changed+=1
            tot+=1; offg += (ob(ns) is None)
            st=ns
    print(f'{name:20s} noop-change-in-play {nc/n:.2f} | any-frame-change {changed/tot:.2f} | off-grid ticks {offg/tot:.2f}')
