import sys, random
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
def y(f): return origin(f)[1]

def stats(name, gen):
    tot=0; static=0; thr=0
    it=new(1); f=frame(it)
    for a in gen(lambda: (f, it)):
        pass
    return
# simpler: explicit drives
def run_drive(actions):
    it=new(1); f=frame(it); recs=[]
    for a in actions:
        f2=act(it,a); recs.append((f,a,f2)); f=f2
    static=sum(1 for s,a,s2 in recs if s==s2)
    ge3=sum(1 for s,a,s2 in recs if len(rocks(s))>=3)
    moving=sum(1 for s,a,s2 in recs if origin(s)[1]!=origin(s2)[1])
    return len(recs), static, ge3, moving

# (a) random flat
rng=random.Random(7); A=[]
for _ in range(120):
    A.append(rng.choice([(rng.randrange(16),rng.randrange(16))]*5+["noop","up","down","left","right"]))
print("random-flat   n,static,rockcount>=3,balloon-moved =", run_drive(A))

# (b) curated: click ONLY when the balloon is at rest, upper interior row
it=new(1); f=frame(it); C=[]
def push(a):
    global f
    C.append(a); f=act(it,a)
for _ in range(6): push("noop")                     # rise to ceiling
for c in (6,7,8): push((c, origin(f)[1]+6))         # 3 rocks -> sink
for _ in range(6): push("noop")
push((6,13))                                        # remove 1 -> rise
for _ in range(6): push("noop")
for c in (6,7): push((c, origin(f)[1]+6))           # back to 4 -> sink
for _ in range(6): push("noop")
for c in (6,7): push((c,13))
for _ in range(6): push("noop")
push((8,origin(f)[1]+6)); push((7,origin(f)[1]+6)); push((6,origin(f)[1]+6))
for _ in range(6): push("noop")
for c in (6,7,8): push((c,13))
for _ in range(8): push("noop")
print("curated n=",len(C)," ->", run_drive(C))
print("curated actions:", C)
