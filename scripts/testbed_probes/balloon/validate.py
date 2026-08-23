import sys, random
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
import model as M
bad=0; tot=0
for s in range(1,31):
    rng=random.Random(s); it=new(s); f=frame(it)
    y=origin(f)[1]; rk=frozenset(rocks(f))
    for t in range(80):
        o=origin(f)
        r=rng.random()
        if r<0.35: a="noop"
        elif r<0.45: a=rng.choice(["up","down","left","right"])
        elif r<0.75:
            dx,dy=rng.choice([(-1,6),(0,6),(1,6),(-1,7),(0,7),(1,7)]); a=(o[0]+dx,o[1]+dy)
        elif r<0.85 and rocks(f): a=rng.choice(rocks(f))
        else: a=(rng.randrange(16),rng.randrange(16))
        am = a if isinstance(a,tuple) else "noop"
        py,prk = M.step(y,rk,am)
        f=act(it,a); tot+=1
        ey=origin(f)[1]; erk=frozenset(rocks(f))
        if (py,prk)!=(ey,erk):
            bad+=1
            if bad<=5: print("MISMATCH",t,"a=",a,"model",(py,sorted(prk)),"engine",(ey,sorted(erk)),"from",(y,sorted(rk)))
        y,rk=ey,erk
        if M.render(y,rk)!=f and bad<=5: print("RENDER MISMATCH at t",t)
print(f"validated {tot} transitions, mismatches={bad}")
