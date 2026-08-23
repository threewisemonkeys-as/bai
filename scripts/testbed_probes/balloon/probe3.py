import sys; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *

print("=== A) 0 rocks, 12 noops: rise + settle ===")
it=new(1); f=frame(it)
print("t0 origin", origin(f))
for t in range(12):
    f=act(it,"noop"); print(f" t{t+1} origin={origin(f)} rocks={rocks(f)} ncells={len(f)}")

print("\n=== B) from top, add rocks one by one, watch threshold ===")
it=new(1); f=frame(it)
for t in range(5): f=act(it,"noop")   # rise to top
o=origin(f); print("at top origin", o)
# basket interior rows o[1]+6, o[1]+7 ; x 6,7,8
for i,(dx,dy) in enumerate([(-1,6),(0,6),(1,6),(-1,7),(0,7),(1,7)]):
    o=origin(f); tgt=(o[0]+dx,o[1]+dy)
    f=act(it,tgt)
    print(f" click{i+1} {tgt} -> origin={origin(f)} nrocks={len(rocks(f))} rocks={rocks(f)}")
print(show(f))
print("\n then 12 noops:")
for t in range(12):
    f=act(it,"noop"); print(f"  t{t+1} origin={origin(f)} rocks={rocks(f)}")
print(show(f))
