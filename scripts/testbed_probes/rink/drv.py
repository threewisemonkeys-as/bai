import json, sys, random
ROOT = "/home/ays57/bai"
sys.path.insert(0, f"{ROOT}/Autumn.cpp/build")
sys.path.insert(0, f"{ROOT}/MARAProtocol/python_examples/autumnbench")
import interpreter_module as im
from autumnstdlib import autumnstdlib

PROG = open(f"{ROOT}/.cache/autumn55/programs/rink.sexp").read()

def new(seed=1, prog=None):
    it = im.Interpreter(); it.run_script(prog or PROG, autumnstdlib, "", seed)
    try: it.set_verbose(False)
    except Exception: pass
    return it

def raw(it):
    d = json.loads(it.render_all()); G = d.pop("GRID_SIZE", 0)
    return d, G

def frame(it):
    d, G = raw(it)
    cells = {}
    for name, elems in d.items():
        for e in elems:
            cells[(e["position"]["x"], e["position"]["y"])] = e["color"].lower()
    return cells, G

def objs(it):
    """return per-object list of (x,y,color)"""
    d, G = raw(it)
    return {k: [(e["position"]["x"], e["position"]["y"], e["color"].lower()) for e in v] for k, v in d.items()}, G

def act(it, a):
    if a == "left": it.left()
    elif a == "right": it.right()
    elif a == "up": it.up()
    elif a == "down": it.down()
    elif a == "noop": pass
    elif a.startswith("click"):
        _, c, r = a.split(); it.click(int(c), int(r))
    it.step()
    return frame(it)

def skater(it):
    """position of the red cell (may be off-grid)."""
    d, G = raw(it)
    for name, elems in d.items():
        for e in elems:
            if e["color"].lower() == "red":
                return (e["position"]["x"], e["position"]["y"])
    return None
