import os
import sys, random
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from fast import spos
from drv import new
import model
ACTS=['left','right','up','down','noop']
bad=0; n=0
for seed in range(1,16):
    rng=random.Random(seed*13); it=new(seed); st=model.INIT
    for t in range(200):
        a=rng.choice(ACTS)
        if a!='noop': getattr(it,a)()
        it.step(); p=spos(it)
        st=model.tick(st,a); n+=1
        if (st[0],st[1])!=p:
            bad+=1
            if bad<4: print('MISMATCH t',t,'act',a,'engine',p,'model',st)
            st=(p[0],p[1],st[2],st[3])
print(f'model vs engine: {n} transitions, {bad} mismatches')
