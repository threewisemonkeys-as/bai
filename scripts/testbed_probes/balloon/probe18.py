import sys, random, json
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
def rect_visible(f):
    o=origin(f)
    return [p for p,c in f.items() if c in ("gray","grey") and o[0]-2<=p[0]<=o[0]+2 and o[1]<=p[1]<=o[1]+7]
found=None; checked=0; viol=0
for s in range(1,61):
    rng=random.Random(1234+s); it=new(s); f=frame(it)
    for t in range(80):
        o=origin(f); r=rng.random()
        if r<0.35: a="noop"
        elif r<0.8:
            dx,dy=rng.choice([(-1,6),(0,6),(1,6),(-1,7),(0,7),(1,7)]); a=(o[0]+dx,o[1]+dy)
        elif rocks(f): a=rng.choice(rocks(f))
        else: a="noop"
        y0=origin(f)[1]; vis=len(rect_visible(f))
        f2=act(it,a); y1=origin(f2)[1]
        if a=="noop":              # motion determined purely by pre-state
            checked+=1
            pred_down = vis>=3
            act_down  = y1>y0
            if y1!=y0 and pred_down!=act_down:
                viol+=1
                if found is None: found=(s,t,y0,y1,vis,sorted(rocks(f)),len(json.loads(it.render_all()).get("rocks",[])))
        f=f2
print(f"noop transitions checked={checked}  direction mispredicted from VISIBLE gray count: {viol} ({viol/max(1,checked):.3f})")
print("example:",found)
