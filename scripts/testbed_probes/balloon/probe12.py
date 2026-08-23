import sys; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
print("=== R) does a rock in the LOWER basket row block rising? ===")
it=new(1); f=frame(it)   # y=7 at bottom, 0 rocks
# click lower interior row (rel 7 = row 14)
f=act(it,(8,14)); print(" click (8,14):", origin(f), rocks(f))
for t in range(6): f=act(it,"noop"); print(f"  noop{t+1}", origin(f), rocks(f))
print(show(f))
print("\n=== S) same but rock in UPPER basket row (rel 6 = row 13) ===")
it=new(1); f=frame(it)
f=act(it,(8,13)); print(" click (8,13):", origin(f), rocks(f))
for t in range(6): f=act(it,"noop"); print(f"  noop{t+1}", origin(f), rocks(f))
