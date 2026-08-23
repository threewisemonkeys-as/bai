import sys, random
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *

def rr(f): return set(rocks(f))
def y(f): return origin(f)[1]
TOP=6*["noop"]

# predicates take (f, hist) where hist = list of frames seen so far (any-step semantics)
G = {
 "L1 rise until the balloon's top row is row 0 (ceiling)":
   ([],  lambda f,h: y(f)==2,  ["noop"]*5),
 "L2 put exactly one rock in the basket (and keep it)":
   ([],  lambda f,h: len(rr(f))==1, [(8,13)]),
 "L3 land the basket on the ground (bottom row)":
   (TOP, lambda f,h: y(f)==7, [(6,8),(7,8),(8,8)]+["noop"]*5),
 "L4a exact frame: on the ground with rocks at (6,13),(7,13),(8,13)":
   (TOP, lambda f,h: y(f)==7 and rr(f)=={(6,13),(7,13),(8,13)}, [(6,8),(7,8),(8,8)]+["noop"]*5),
 "L4b land, then lighten and return to the ceiling with 2 rocks":
   (TOP, lambda f,h: y(f)==2 and len(rr(f))==2 and any(y(g)==7 for g in h),
    [(6,8),(7,8),(8,8)]+["noop"]*5+[(6,13)]+["noop"]*5),
 "L5 sink to the ground then return to the ceiling with an EMPTY basket":
   (TOP, lambda f,h: y(f)==2 and len(rr(f))==0 and any(y(g)==7 for g in h),
    [(6,8),(7,8),(8,8)]+["noop"]*5+[(6,13),(7,13),(8,13)]+["noop"]*5),
}
print("=== T) verified plans ===")
for name,(pre,g,plan) in G.items():
    it=new(1); f=frame(it); h=[f]
    for a in pre: f=act(it,a); h.append(f)
    h=[f]
    ok_at=None
    for i,a in enumerate(plan):
        f=act(it,a); h.append(f)
        if g(f,h) and ok_at is None: ok_at=i+1
    print(f" {name}\n   plan_len={len(plan)} reached_at_step={ok_at} final_ok={g(f,h)}  plan={plan}")

print("\n=== U) random-policy floors (25 rollouts x 50 actions, any-step) ===")
ALL=[(x,yy) for yy in range(16) for x in range(16)]+["noop","up","down","left","right"]
def floor(pre,g,mode,N=25,H=50,seed0=1000):
    hit=0
    for s in range(N):
        rng=random.Random(seed0+s)
        it=new(1); f=frame(it)
        for a in pre: f=act(it,a)
        h=[f]; ok=g(f,h)
        for t in range(H):
            if mode=="flat": a=rng.choice(ALL)
            else:
                k=rng.choice(["noop","up","down","left","right","click"])
                a=(rng.randrange(16),rng.randrange(16)) if k=="click" else k
            f=act(it,a); h.append(f)
            if g(f,h): ok=True
        if ok: hit+=1
    return hit/N
for name,(pre,g,plan) in G.items():
    a=floor(pre,g,"flat"); b=floor(pre,g,"typed")
    print(f" {name[:60]:62s} flat={a:.2f} typed={b:.2f}")
