import os
import sys, re, json, random
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from drv import new
RE = re.compile(r'"skater": \[\{"position": \{"x": (-?\d+), "y": (-?\d+)\}')
def spos(it):
    m = RE.search(it.render_all())
    return (int(m.group(1)), int(m.group(2)))
def step(it, a):
    if a != 'noop': getattr(it, a)()
    it.step()
    return spos(it)
def obs(p):
    """what the pipeline text grid shows: None if the skater is clipped away"""
    return p if (0 <= p[0] < 28 and 0 <= p[1] < 28) else None
def replay(seed, acts):
    it = new(seed)
    for a in acts: step(it, a)
    return it
