import sys, random
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *

print("=== N) remove a rock MID-SINK -> does it reverse? ===")
it=new(1); f=frame(it)
for _ in range(6): f=act(it,"noop")           # top
for tgt in [(6,9),(7,9),(8,9)]: f=act(it,tgt) # 3 rocks
print(" after 3 clicks:", origin(f), rocks(f))
f=act(it,"noop"); print(" noop:", origin(f), rocks(f))
f=act(it,"noop"); print(" noop:", origin(f), rocks(f))
r=rocks(f)[0]
f=act(it,r); print(f" click rock {r}:", origin(f), rocks(f), " <- reversal?")
for t in range(6):
    f=act(it,"noop"); print(f"  noop{t+1}", origin(f), rocks(f))
print(show(f))

print("\n=== O) absorbing states ===")
def settle(it, f, n=25):
    seq=[]
    for _ in range(n):
        f2=act(it,"noop"); seq.append(origin(f2)[1]); 
        f=f2
    return f, seq
it=new(1); f=frame(it); f,seq=settle(it,f); print(" 0 rocks from init, noop rows:", seq)
it=new(1); f=frame(it)
for _ in range(6): f=act(it,"noop")
for tgt in [(6,8),(7,8),(8,8)]: f=act(it,tgt)
f,seq=settle(it,f); print(" 3 rocks from top,  noop rows:", seq, "final rocks", rocks(f))
