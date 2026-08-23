import json, sys
from collections import Counter
ROOT = "/home/ays57/bai"
sys.path.insert(0, f"{ROOT}/Autumn.cpp/build")
sys.path.insert(0, f"{ROOT}/MARAProtocol/python_examples/autumnbench")
import interpreter_module as im
from autumnstdlib import autumnstdlib
PROG = open(f"{ROOT}/.cache/autumn55/programs/balloon.sexp").read()
M = {"mediumpurple":"P","tan":"t","brown":"b","gray":"R","grey":"R"}

def new(seed=1):
    it = im.Interpreter(); it.run_script(PROG, autumnstdlib, "", seed)
    try: it.set_verbose(False)
    except Exception: pass
    return it

def frame(it):
    d = json.loads(it.render_all()); d.pop("GRID_SIZE", 0)
    cells = {}
    for name, elems in d.items():
        for e in elems:
            cells[(e["position"]["x"], e["position"]["y"])] = e["color"].lower()
    return cells

def show(cells, G=16):
    return "\n".join("".join(M.get(cells.get((x,y)), ".") for x in range(G)) for y in range(G))

def origin(cells):
    p = [q for q,c in cells.items() if c=="mediumpurple"]
    if not p: return None
    return (min(q[0] for q in p)+2, min(q[1] for q in p)+2)

def rocks(cells):
    return sorted(q for q,c in cells.items() if c in ("gray","grey"))

def act(it, a):
    if a=="noop": pass
    elif a=="up": it.up()
    elif a=="down": it.down()
    elif a=="left": it.left()
    elif a=="right": it.right()
    else: it.click(a[0], a[1])
    it.step(); return frame(it)
