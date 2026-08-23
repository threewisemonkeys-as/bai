import sys, json; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *

def obs_text(it):
    d = raw(it); G = d.pop("GRID_SIZE",0)
    try: bg = it.get_background()
    except Exception: bg = "black"
    m = [[bg]*G for _ in range(G)]
    for name, elems in d.items():
        for e in elems:
            x,y,c = e["position"]["x"], e["position"]["y"], e["color"].lower()
            if 0<=y<G and 0<=x<G: m[y][x]=c
    return json.dumps(m), G

for g in ["logic_gates","ice","mario","blicket","wind","lights_new"]:
    try: it = load(g,1)
    except Exception as e: print(f"{g:12s} LOAD FAIL {e}"); continue
    t,G = obs_text(it); n=len(t)
    print(f"{g:12s} grid={G:3d} chars={n:6d} ~tokens={n/3.6:6.0f}  nonbg_cells={sum(1 for r in json.loads(t) for c in r if c!='black')}")
it = load("logic_gates",1)
t,G = obs_text(it)
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "obs_sample.txt"),"w").write(t)
# ascii view
M = json.loads(t)
CH = {"black":".","grey":"-","pink":"p","red":"R","darkblue":"b","orange":"O","yellow":"Y"}
print("\n   " + "".join(str(x%10) for x in range(G)))
for y,row in enumerate(M):
    print(f"{y:2d} " + "".join(CH.get(c,"?") for c in row))
