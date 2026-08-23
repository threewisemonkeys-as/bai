import sys; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *

def state_desc(it):
    d = raw(it); d.pop("GRID_SIZE",None)
    out = {}
    for name, elems in d.items():
        cols = sorted(set(e["color"].lower() for e in elems))
        out[name] = "/".join(cols)
    return out

def show(tag, it):
    s = state_desc(it)
    print(f"{tag:22s} s1={s['switch1']:5s} s2={s['switch2']:5s} | AND={s['andOutput']:8s} OR={s['orOutput']:8s} NOT={s['notOutput']:8s} XOR={s['xorOutput']:8s} | aw1={s['andWire1']:6s} aw2={s['andWire2']:6s} ow1={s['orWire1']:6s} nw={s['notWire']:6s} xw1={s['xorWire1']:6s} xw2={s['xorWire2']:6s}")

it = load("logic_gates", 1)
show("t=0 init", it)
# click switch1 at (4,12)
it.click(4,12); it.step(); show("t=1 after click s1", it)
for i in range(2,6):
    it.step(); show(f"t={i} noop", it)
print()
it.click(19,12); it.step(); show("t=6 after click s2", it)
for i in range(7,11):
    it.step(); show(f"t={i} noop", it)
print()
it.click(4,12); it.step(); show("t=11 unclick s1", it)
for i in range(12,15):
    it.step(); show(f"t={i} noop", it)
