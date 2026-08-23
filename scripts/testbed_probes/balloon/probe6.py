import sys,json; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
it=new(1); f=frame(it)
for _ in range(6): f=act(it,"noop")
for tgt in [(6,9),(7,9),(8,9)]: f=act(it,tgt)
f=act(it,"noop")
o=origin(f); f=act(it,(o[0]-1,o[1]+6))
for _ in range(6): f=act(it,"noop")
d=json.loads(it.render_all())
print("KEY ORDER:", list(d.keys()))
# does a rock share a cell with a balloon cell?
bal={(e["position"]["x"],e["position"]["y"]) for e in d["balloon"]}
rk=[(e["position"]["x"],e["position"]["y"]) for e in d.get("rocks",[])]
print("rocks:",rk)
print("rock cells also in balloon sprite:", [p for p in rk if p in bal])
print("total elems:", sum(len(v) for k,v in d.items() if k!="GRID_SIZE"))
print("raw rocks entries:", d.get("rocks"))
