import os
import sys, random
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from fast import spos, obs
from drv import new
ARROWS=['left','right','up','down']; ACTS=ARROWS+['noop']
D={'left':(-1,0),'right':(1,0),'up':(0,-1),'down':(0,1)}
def legal(p,a):
    if a=='noop': return True
    dx,dy=D[a]; return 0<=p[0]+dx<28 and 0<=p[1]+dy<28
def onice(p): return 3<=p[0]<=24 and 3<=p[1]<=24
def fstep(it,a):
    if a!='noop': getattr(it,a)()
    it.step()
def rstep(it,a):
    fstep(it,a); return spos(it)
def rep(seed,acts):
    it=new(seed)
    for a in acts: fstep(it,a)
    return it
def branch(seed,prefix,h=3):
    out={}
    for a in ACTS:
        it=rep(seed,prefix); seq=[obs(rstep(it,a))]
        for _ in range(h-1): seq.append(obs(rstep(it,'noop')))
        out[a]=(seq[0],tuple(seq))
    return out

def policy_unguarded(rng,p):
    return 'noop' if rng.random()<0.2 else rng.choice(ARROWS)
def policy_guarded(rng,p):
    if rng.random()<0.2: return 'noop'
    return rng.choice([a for a in ARROWS if legal(p,a)])
def policy_iceseek(rng,p):
    """guarded + when off the ice, bias toward the rink centre"""
    if onice(p):
        if rng.random()<0.35: return 'noop'
        return rng.choice([a for a in ARROWS if legal(p,a)])
    if rng.random()<0.1: return 'noop'
    want=[]
    if p[0]<3: want.append('right')
    if p[0]>24: want.append('left')
    if p[1]<3: want.append('down')
    if p[1]>24: want.append('up')
    if want and rng.random()<0.8: return rng.choice(want)
    return rng.choice([a for a in ARROWS if legal(p,a)])

def measure(policy,name,seeds=range(1,5),T=30):
    vis1=vis3=dist=0.0; n=0; on_d=[]; off_d=[]; offgrid=0; ice=0; tot=0
    for seed in seeds:
        rng=random.Random(seed*17); it=new(seed); prefix=[]; p=spos(it)
        for t in range(T):
            b=branch(seed,prefix)
            base1=b['noop'][0]; base3=b['noop'][1]
            vis1+=sum(1 for a in ARROWS if b[a][0]!=base1)/4
            vis3+=sum(1 for a in ARROWS if b[a][1]!=base3)/4
            d=len(set(v[0] for v in b.values())); dist+=d; n+=1
            (on_d if onice(p) else off_d).append(d)
            a=policy(rng,p); p=rstep(it,a); prefix.append(a); tot+=1
            if obs(p) is None: offgrid+=1
            if onice(p): ice+=1
    print(f'{name:12s} n={n:3d} | arrow vis h1={vis1/n:.2f} h3={vis3/n:.2f} | distinct/5 = {dist/n:.2f} '
          f'| on-ice branchpts {len(on_d):3d} (d={sum(on_d)/max(1,len(on_d)):.2f}) '
          f'| off-grid ticks {offgrid}/{tot} | on-ice ticks {ice}/{tot}')

measure(policy_unguarded,'unguarded')
measure(policy_guarded,'guarded')
measure(policy_iceseek,'ice-seeking')
