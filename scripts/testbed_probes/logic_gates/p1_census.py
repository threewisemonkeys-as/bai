import sys; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
it = load("logic_gates", 1)
d = raw(it)
print("GRID_SIZE", d.get("GRID_SIZE"))
for name, elems in d.items():
    if name=="GRID_SIZE": continue
    print(f"  {name}: {len(elems)} cells, colors={sorted(set(e['color'].lower() for e in elems))}")
cells,G = frame(it)
print("distinct cells:", len(cells), "colors:", sorted(set(cells.values())))
