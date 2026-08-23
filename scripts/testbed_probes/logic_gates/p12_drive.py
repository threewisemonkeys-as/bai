import sys, random; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
SW1=(4,12); SW2=(19,12)
def new():
    it=load("logic_gates",1); frame(it); return it
def do(it,a):
    if a=="s1": it.click(*SW1)
    elif a=="s2": it.click(*SW2)
    elif a=="empty": it.click(0,0)
    elif a=="left": it.left()
    elif a=="right": it.right()
    elif a=="up": it.up()
    elif a=="down": it.down()
    it.step(); return frame(it)[0]
EFF=["noop","s1","s2"]
# greedy covering walk over (state,eff-action)
it=new(); cur=frame(it)[0]; seen=set(); walk=[]
def peek(prefix,a):
    j=new()
    for x in prefix: do(j,x)
    return key(do(j,a))
for step in range(200):
    best=None
    for a in EFF:
        k=(key(cur),a)
        if k not in seen: best=a; break
    if best is None:
        # move to a state with uncovered actions: try each action, pick one leading somewhere useful
        best=random.choice(EFF)
    seen.add((key(cur),best)); walk.append(best); cur=do(it,best)
    if len(seen)>=36: break
print(f"greedy covering walk: {len(walk)} actions to cover {len(seen)}/36 (state,action) pairs")
print("walk:", ",".join(walk))
# authored 60-action drive
DRIVE = ("s1,noop,noop,s1,noop,noop,s2,noop,noop,s2,noop,noop,"     # single-switch settled, both directions
         "s1,noop,noop,s2,noop,noop,s1,noop,noop,s2,noop,noop,"     # walk all 4 combos settled
         "s1,s2,s1,s2,s1,s2,noop,noop,"                             # back-to-back, never settle
         "empty,empty,left,right,up,down,noop,noop,"                # null actions
         "s2,s1,noop,s2,noop,s1,s1,noop,noop,s2,"                   # mixed
         "s2,noop,s1,noop,s2,noop,noop,noop").split(",")
print(f"\nauthored drive length {len(DRIVE)}")
it=new(); cur=frame(it)[0]; cov=set(); changed=0; nonnoop_visible=0
for a in DRIVE:
    eff = a if a in EFF else "noop"
    cov.add((key(cur),eff))
    nxt=do(it,a)
    if key(nxt)!=key(cur): changed+=1
    if a in ("s1","s2"): nonnoop_visible+=1
    cur=nxt
print(f"  distinct (state,eff-action) pairs covered: {len(cov)}/36")
print(f"  transitions where frame changed: {changed}/{len(DRIVE)} ({changed/len(DRIVE):.0%})")
print(f"  switch clicks: {nonnoop_visible}")
# random 60-action drive for comparison
random.seed(3); tot_c=0; tot_cov=[]; tot_sw=0
ALL=["noop","left","right","up","down"]+[f"c_{x}_{y}" for y in range(24) for x in range(24)]
for trial in range(20):
    it=new(); cur=frame(it)[0]; cov2=set(); ch=0; sw=0
    for _ in range(60):
        a=random.choice(ALL)
        if a.startswith("c_"):
            _,x,y=a.split("_"); x,y=int(x),int(y); it.click(x,y)
            eff = "s1" if (x,y) in {(4,12),(4,13),(5,12),(5,13)} else ("s2" if (x,y) in {(19,12),(19,13),(20,12),(20,13)} else "noop")
            if eff!="noop": sw+=1
        elif a in ("left","right","up","down"): getattr(it,a)(); eff="noop"
        else: eff="noop"
        cov2.add((key(cur),eff)); it.step(); nxt=frame(it)[0]
        if key(nxt)!=key(cur): ch+=1
        cur=nxt
    tot_c+=ch; tot_cov.append(len(cov2)); tot_sw+=sw
print(f"\nrandom 60-action drives (20 trials): mean changed {tot_c/20:.1f}/60, mean (s,a) coverage {sum(tot_cov)/20:.1f}/36, mean switch clicks {tot_sw/20:.2f}")
