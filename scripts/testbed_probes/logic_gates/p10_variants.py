import sys, json; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT="/home/ays57/bai"; sys.path.insert(0,f"{ROOT}/Autumn.cpp/build"); sys.path.insert(0,f"{ROOT}/MARAProtocol/python_examples/autumnbench")
import interpreter_module as im
from autumnstdlib import autumnstdlib
from collections import deque
import os; D=os.path.dirname(os.path.abspath(__file__))
ON={"red","orange","yellow"}
def mk(path,seed=1):
    it=im.Interpreter(); it.run_script(open(path).read(), autumnstdlib, "", seed)
    try: it.set_verbose(False)
    except Exception: pass
    return it
def frame(it):
    d=json.loads(it.render_all()); d.pop("GRID_SIZE",None); c={}
    for n,es in d.items():
        for e in es: c[(e["position"]["x"],e["position"]["y"])]=e["color"].lower()
    return c
def key(c): return tuple(sorted(c.items()))
def rd(c):
    g=lambda p: c.get(p,"black")
    return f"s1={int(g((4,12)) in ON)} s2={int(g((19,12)) in ON)} | O1={int(g((12,4)) in ON)} O2={int(g((12,8)) in ON)} O3={int(g((12,16)) in ON)} O4={int(g((12,20)) in ON)} | w1={int(g((5,4)) in ON)} w2={int(g((14,4)) in ON)}"
def analyse(path,label,maxd=8):
    def canon(pre):
        it=mk(path); frame(it)
        for a in pre:
            if a=="s1": it.click(4,12)
            elif a=="s2": it.click(19,12)
            it.step(); c=frame(it)
        return frame(it) if not pre else c
    f0=canon([]); seen={key(f0):[]}; q=deque([[]])
    while q:
        p=q.popleft()
        if len(p)>maxd: continue
        for a in ["noop","s1","s2"]:
            n=p+[a]; c=canon(n); k=key(c)
            if k not in seen: seen[k]=n; q.append(n)
    print(f"--- {label}: {len(seen)} distinct frames (BFS depth<={maxd}), max depth used {max(len(v) for v in seen.values())}")
    # settle time from a click
    it=mk(path); frame(it)
    for _ in range(4): it.step(); frame(it)
    it.click(4,12)
    hist=[]
    for i in range(6):
        it.step(); hist.append(frame(it))
    st=None
    for i in range(1,6):
        if key(hist[i])==key(hist[i-1]) and st is None: st=i
    print(f"    settle after click_s1: {st} ticks   trace:")
    for i,h in enumerate(hist): print(f"      t+{i+1}: {rd(h)}")
    # truth table settled
    print("    settled truth table:")
    for tgt in [(0,0),(1,0),(0,1),(1,1)]:
        pre=(["s1"] if tgt[0] else [])+(["s2"] if tgt[1] else [])+["noop"]*5
        print("      "+rd(canon(pre)))
analyse(f"{D}/logic_gates_v1.sexp","V1 (NAND/NOR/BUF-s2/XNOR)")
analyse(f"{D}/logic_gates_v2.sexp","V2 (composed, prev-staged)")
