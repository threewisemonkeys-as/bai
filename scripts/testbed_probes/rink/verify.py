import os
import sys, json
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from goals import *
from drv import new
from fast import spos
sys.path.insert(0,'/home/ays57/bai/MARAProtocol')

A=[]; B=['down','down','down','right','right']; C=['down']*10+['right','right']
def P_onice(st): return onice(st[0],st[1])
def P_rest_right(st): return rest(st) and st[0]>=25 and 3<=st[1]<=24
def P_rest_top(st):  return rest(st) and st[1]<=2 and 3<=st[0]<=24
def P_rest_bottom(st): return rest(st) and st[1]>=25 and 3<=st[0]<=24
def P_cell(c): return lambda st: rest(st) and (st[0],st[1])==c
LAD=[
 ("L1 enter the ice",                       B, P_onice),
 ("L1 cross to the right margin",           B, P_rest_right),
 ("L2 leave through the TOP margin",        C, P_rest_top),
 ("L2 stop exactly at (25,10)",             C, P_cell((25,10))),
 ("L3 stop exactly at (11,1) [1 turn]",     C, P_cell((11,1))),
 ("L3 leave through the BOTTOM margin",     A, P_rest_bottom),
 ("L4 stop exactly at (26,26) [2 turns]",   A, P_cell((26,26))),
 ("L4 stop exactly at (27,0)",              A, P_cell((27,0))),
]
print(f'{"goal":42s} {"h":>3} {"floor25":>8} {"floor50":>8}  plan')
rows=[]
for name,pre,pred in LAD:
    st0=run(pre); plan,end=bfs(st0,pred,cap=50)
    f25=rand_floor(st0,pred,25,len(plan),guard=False,seed0=3)
    f50=rand_floor(st0,pred,25,50,guard=False,seed0=3)
    rows.append((name,pre,plan,end,f25,f50))
    print(f'{name:42s} {len(plan):3d} {f25:8.2f} {f50:8.2f}  {",".join(plan)} -> {end[:2]}')

# ---- engine verification of every plan, incl. absorbing check ----
print('\nengine verification (prefix + plan replayed in the interpreter):')
for name,pre,plan,end,_,_ in rows:
    it=new(1)
    for a in pre+plan:
        if a!='noop': getattr(it,a)()
        it.step(); it.render_all()
    p=spos(it)
    it2=new(1)
    for a in pre+plan+['noop']*10:
        if a!='noop': getattr(it2,a)()
        it2.step(); it2.render_all()
    p10=spos(it2)
    print(f'  {name:42s} engine end {p}  model end {end[:2]}  match={p==tuple(end[:2])}  absorbing(+10 noop)={p10==p}')
