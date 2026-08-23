import json, sys, io, contextlib
ROOT = "/home/ays57/bai"
sys.path.insert(0, f"{ROOT}/Autumn.cpp/build")
sys.path.insert(0, f"{ROOT}/MARAProtocol/python_examples/autumnbench")
import interpreter_module as im
from autumnstdlib import autumnstdlib
prog = open(f"{ROOT}/.cache/autumn55/programs/balloon.sexp").read()

def new(seed=1):
    it = im.Interpreter(); it.run_script(prog, autumnstdlib, "", seed)
    try: it.set_verbose(False)
    except Exception: pass
    return it

def raw(it):
    return json.loads(it.render_all())

def frame(it):
    d = raw(it); G = d.pop("GRID_SIZE", 0)
    cells = {}
    for name, elems in d.items():
        for e in elems:
            cells[(e["position"]["x"], e["position"]["y"])] = e["color"].lower()
    return cells, G

it = new(1)
r = raw(it)
print("RAW KEYS:", list(r.keys()))
print("GRID_SIZE:", r.get("GRID_SIZE"))
for k,v in r.items():
    if k=="GRID_SIZE": continue
    print(f"  {k}: n={len(v)}")
    if v: print("   sample:", v[0])
cells,G = frame(it)
from collections import Counter
print("ncells:", len(cells), "colors:", Counter(cells.values()))
# print grid
def show(cells,G=16):
    m = {"mediumpurple":"P","tan":"t","brown":"b","gray":"R","grey":"R"}
    out=[]
    for y in range(G):
        out.append("".join(m.get(cells.get((x,y),"."),"?") for x in range(G)))
    return "\n".join(out)
print(show(cells))
# derive balloon origin: purple block center
purple=[p for p,c in cells.items() if c=="mediumpurple"]
print("purple x range", min(p[0] for p in purple), max(p[0] for p in purple),
      "y range", min(p[1] for p in purple), max(p[1] for p in purple))
