import sys; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *

# 1) which cells are clickable (add a rock) at init? brute force all 256
print("=== clickable scan at init (origin y=7) ===")
adds=[]
for y in range(16):
    for x in range(16):
        it=new(1); f=frame(it)
        f2=act(it,(x,y))
        if rocks(f2):
            adds.append((x,y))
print("cells that ADD a rock:", adds)
print("count:", len(adds))

# 2) does the rock appear at the click position?
it=new(1); f2=act(it,(7,13)); print("click (7,13) -> rocks:", rocks(f2))
print(show(f2))
