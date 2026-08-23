import sys; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
from collections import deque

# replay-based BFS: state address = action prefix
ACTIONS = [("noop",None), ("click_s1",(4,12)), ("click_s2",(19,12)), ("click_empty",(0,0))]

def replay(seed, prefix):
    it = load("logic_gates", seed)
    frame(it)   # render at t=0
    for a,arg in prefix:
        if a.startswith("click"): it.click(arg[0],arg[1])
        it.step(); c,_ = frame(it)
    if not prefix:
        c,_ = frame(it)
    return c

def summarize(cells):
    # human-readable latent readout
    def col(p): return cells.get(p,"black")
    s1 = col((4,12)); s2 = col((19,12))
    AND=col((12,4)); OR=col((12,8)); NOT=col((12,16)); XOR=col((12,20))
    aw1=col((5,4)); aw2=col((14,4)); nw=col((6,12))
    on = lambda c: c in ("red","orange","yellow")
    return f"s1={int(on(s1))} s2={int(on(s2))} | AND={int(on(AND))} OR={int(on(OR))} NOT={int(on(NOT))} XOR={int(on(XOR))} | w1={int(on(aw1))} w2={int(on(aw2))}"

for seed in (1,2):
    start = replay(seed, [])
    seen = {key(start): ([], start)}
    q = deque([[]])
    edges = []
    while q:
        pre = q.popleft()
        if len(pre) > 8: continue
        for a,arg in ACTIONS:
            nxt = pre+[(a,arg)]
            c = replay(seed, nxt)
            k = key(c)
            src = key(replay(seed,pre))
            if k not in seen:
                seen[k] = (nxt, c)
                q.append(nxt)
            edges.append((src, a, k))
    ids = {k:i for i,(k,_) in enumerate(sorted(seen.items(), key=lambda kv: len(kv[1][0])))}
    print(f"=== seed {seed}: {len(seen)} distinct rendered frames ===")
    for k,(pre,c) in sorted(seen.items(), key=lambda kv: len(kv[1][0])):
        print(f"  S{ids[k]:<2d} depth={len(pre):<2d} {summarize(c)}  via {[a for a,_ in pre]}")
    # dedup transition table
    tab = {}
    for s,a,d in edges:
        tab[(ids[s],a)] = ids[d]
    print("  transitions:")
    for i in range(len(ids)):
        row = " ".join(f"{a}->S{tab.get((i,a),'?')}" for a,_ in ACTIONS)
        print(f"    S{i}: {row}")
