import sys; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
# baseline: from t=0, noop
base = load("logic_gates",1); frame(base); base.step(); noop_f,_ = frame(base)
hits=[]
for y in range(24):
    for x in range(24):
        it = load("logic_gates",1); frame(it)
        it.click(x,y); it.step(); c,_ = frame(it)
        if key(c)!=key(noop_f): hits.append((x,y))
print("clickable (x=col,y=row) cells that differ from noop:", len(hits), "/576")
print(sorted(hits))
# which switch
for (x,y) in hits:
    it = load("logic_gates",1); frame(it); it.click(x,y); it.step(); c,_=frame(it)
    s1 = c.get((4,12)); s2 = c.get((19,12))
    print(f"  click(col={x},row={y}) -> s1={s1} s2={s2}")
