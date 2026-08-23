import sys; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *

def rise_to_top(it):
    f=frame(it)
    for _ in range(6): f=act(it,"noop")
    return f

print("=== C) 3 rocks placed at top, then pure noops ===")
it=new(1); f=rise_to_top(it); print("origin",origin(f))
for tgt in [(6,8),(7,8),(8,8)]:
    f=act(it,tgt); print(f" click {tgt} origin={origin(f)} rocks={rocks(f)}")
for t in range(10):
    f=act(it,"noop"); print(f"  noop{t+1} origin={origin(f)} rocks={rocks(f)}")
print(show(f))

print("\n=== D) same but place rocks in LOWER basket row first ===")
it=new(1); f=rise_to_top(it)
for tgt in [(6,9),(7,9),(8,9)]:
    f=act(it,tgt); print(f" click {tgt} origin={origin(f)} rocks={rocks(f)}")
for t in range(10):
    f=act(it,"noop"); print(f"  noop{t+1} origin={origin(f)} rocks={rocks(f)}")
print(show(f))

print("\n=== E) click on an existing rock -> removal? ===")
it=new(1); f=rise_to_top(it)
f=act(it,(7,8)); print("after add:", rocks(f), origin(f))
f=act(it,(7,8)); print("click same cell again:", rocks(f), origin(f))
