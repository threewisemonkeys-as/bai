import os
import sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from goals import *
A=[]                                   # start (0,0)
B=['down','down','down','right','right']          # (2,3) left margin, top row of rink
C=['down']*10+['right','right']                   # (2,10) left margin, mid rink
print('start A', run(A), ' B', run(B), ' C', run(C))

def P_onice(st): return onice(st[0],st[1])
def P_rest_right(st): return rest(st) and st[0]>=25 and 3<=st[1]<=24
def P_rest_top(st):  return rest(st) and st[1]<=2 and 3<=st[0]<=24
def P_cell(c):       return lambda st: rest(st) and (st[0],st[1])==c
def P_rest_bottom(st): return rest(st) and st[1]>=25 and 3<=st[0]<=24

LAD=[
 ("L1  step onto the ice",                      B, P_onice),
 ("L1  exit right margin (any row)",            B, P_rest_right),
 ("L2  exit TOP margin after entering left",    C, P_rest_top),
 ("L2  rest exactly at (25,10)",                C, P_cell((25,10))),
 ("L3  rest exactly at (11,1)  (turn mid-slide)",C, P_cell((11,1))),
 ("L3  exit BOTTOM margin from start (0,0)",    A, P_rest_bottom),
 ("L4  rest exactly at (26,26)",                A, P_cell((26,26))),
 ("L4  rest exactly at (27,0) top-right corner",A, P_cell((27,0))),
 ("--  rest ON an ice cell (impossible?)",      A, lambda st: rest(st) and onice(st[0],st[1])),
]
for name,pre,pred in LAD:
    st0=run(pre)
    plan,end=bfs(st0,pred,cap=50)
    f_g=rand_floor(st0,pred,25,50,guard=True,seed0=1)
    f_u=rand_floor(st0,pred,25,50,guard=False,seed0=1)
    h='--' if plan is None else len(plan)
    print(f'{name:46s} start={st0[:2]} h={h:>3}  rand_floor(guarded)={f_g:.2f} (unguarded)={f_u:.2f}')
    if plan: print(f'{"":46s} plan={",".join(plan)} -> end {end[:2]}')
