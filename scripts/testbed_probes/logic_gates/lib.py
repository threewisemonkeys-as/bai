import json, sys
ROOT = "/home/ays57/bai"
sys.path.insert(0, f"{ROOT}/Autumn.cpp/build")
sys.path.insert(0, f"{ROOT}/MARAProtocol/python_examples/autumnbench")
import interpreter_module as im
from autumnstdlib import autumnstdlib

def load(prog_name="logic_gates", seed=1):
    prog = open(f"{ROOT}/.cache/autumn55/programs/{prog_name}.sexp").read()
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
            x,y,c = e["position"]["x"], e["position"]["y"], e["color"].lower()
            cells[(x,y)] = c
    return cells, G

def key(cells):
    return tuple(sorted(cells.items()))
