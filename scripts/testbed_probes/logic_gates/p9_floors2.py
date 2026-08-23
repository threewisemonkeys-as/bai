import sys, random; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
ON={"red","orange","yellow"}
def rd(c):
    g=lambda p: c.get(p,"black")
    return dict(AND=g((12,4)) in ON, OR=g((12,8)) in ON, NOT=g((12,16)) in ON, XOR=g((12,20)) in ON,
                w1=g((5,4)) in ON, w2=g((14,4)) in ON)
def nlit(r): return sum(r[k] for k in ("AND","OR","NOT","XOR"))
GOALS={
 "A  light the AND output":                     lambda r: r["AND"],
 "B  light XOR while every wire is still grey": lambda r: r["XOR"] and not r["w1"] and not r["w2"],
 "C  light exactly two outputs":                lambda r: nlit(r)==2,
 "D  light exactly three outputs":              lambda r: nlit(r)==3,
 "E  AND lit AND every wire yellow (settled)":  lambda r: r["AND"] and r["w1"] and r["w2"],
 "F  light AND with XOR dark":                  lambda r: r["AND"] and not r["XOR"],
 "G  only the NOT output lit, nothing else":    lambda r: r["NOT"] and nlit(r)==1,
 "H  AND lit but the left wire still grey":     lambda r: r["AND"] and not r["w1"],
}
random.seed(11); N,H=100,50
anyst={k:0 for k in GOALS}; endp={k:0 for k in GOALS}; sh=[]
SW={(4,12),(4,13),(5,12),(5,13),(19,12),(19,13),(20,12),(20,13)}
for t in range(N):
    it=load("logic_gates",1); frame(it); grids=[]; n=0
    for _ in range(H):
        a=random.randrange(581)
        if a==0: pass
        elif a<5: [it.left,it.right,it.up,it.down][a-1]()
        else:
            cx=(a-5)%24; cy=(a-5)//24; it.click(cx,cy)
            if (cx,cy) in SW: n+=1
        it.step(); grids.append(frame(it)[0])
    sh.append(n)
    for k,f in GOALS.items():
        if any(f(rd(g)) for g in grids): anyst[k]+=1
        if f(rd(grids[-1])): endp[k]+=1
print(f"random floor: {N} rollouts x {H} actions, t>=1 only. mean switch hits {sum(sh)/N:.2f}; >=1 hit in {sum(1 for x in sh if x>0)}/{N}; >=2 hits in {sum(1 for x in sh if x>1)}/{N}")
print(f"{'goal':46s} {'any-step':9s} endpoint")
for k in GOALS: print(f"  {k:44s} {anyst[k]/N:8.2f} {endp[k]/N:8.2f}")
