import os
import sys, random
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from drv import *

ARROWS=['left','right','up','down']
def run(seed, n=100, policy='uniform', p_noop=0.2):
    rng=random.Random(seed); it=new(seed)
    off_first=None; off_ticks=0; poss=[]
    for t in range(n):
        if policy=='uniform':
            a = 'noop' if rng.random()<p_noop else rng.choice(ARROWS)
        act(it,a); p=skater(it); poss.append(p)
        if not (0<=p[0]<28 and 0<=p[1]<28):
            off_ticks+=1
            if off_first is None: off_first=t+1
    return off_first, off_ticks, poss

tot=0; firsts=[]; offt=[]
for s in range(1,21):
    f,o,ps = run(s,100)
    firsts.append(f); offt.append(o)
    if f is not None: tot+=1
print(f'random drives (20 seeds x 100 steps, 20% noop / 80% uniform arrow):')
print(f'  drives that went off-grid at least once: {tot}/20')
print(f'  first off-grid step: {[f for f in firsts]}')
print(f'  off-grid ticks per 100: mean {sum(offt)/20:.1f}, max {max(offt)}')
