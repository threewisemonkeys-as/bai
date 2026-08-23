import sys; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
def top(it):
    f=frame(it)
    for _ in range(6): f=act(it,"noop")
    return f

print("=== F) click DURING motion: do existing rocks lag? ===")
it=new(1); f=top(it)
for tgt in [(6,9),(7,9),(8,9)]: f=act(it,tgt)
print("start sink:", origin(f), rocks(f))
f=act(it,"noop"); print(" noop -> ",origin(f), rocks(f))
o=origin(f)
tgt=(o[0]-1,o[1]+6)   # click upper-basket-interior of CURRENT position
f=act(it,tgt); print(f" click {tgt} -> origin={origin(f)} rocks={rocks(f)}  <-- do old rocks move?")
for t in range(6):
    f=act(it,"noop"); print(f"  noop{t+1} origin={origin(f)} rocks={rocks(f)}")
print(show(f))

print("\n=== G) arrows/other keys have any effect? ===")
for a in ["up","down","left","right"]:
    it=new(1); f0=frame(it); f1=act(it,a)
    it2=new(1); g0=frame(it2); g1=act(it2,"noop")
    print(f" {a}: identical to noop? {f1==g1}")

print("\n=== H) is the game deterministic across seeds? ===")
base=None
for s in [1,2,3,7,12,99]:
    it=new(s); f=frame(it)
    seq=[]
    for t in range(8): f=act(it,"noop"); seq.append(origin(f))
    if base is None: base=seq
    print(f" seed{s}: {seq} same_as_seed1={seq==base}")

print("\n=== I) reachable balloon rows / travel range ===")
it=new(1); f=frame(it); rows=set()
for t in range(40):
    f=act(it,"noop"); rows.add(origin(f)[1])
print(" rows reachable rising:", sorted(rows))
