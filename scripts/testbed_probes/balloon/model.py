"""Exact python model of balloon.sexp, validated against the engine."""
PURPLE=[(-1,-2),(0,-2),(1,-2)]+[(dx,dy) for dy in (-1,0,1) for dx in (-2,-1,0,1,2)]+[(-1,2),(0,2),(1,2)]
TAN=[(-1,3),(0,4),(1,3),(-1,5),(1,5)]
BROWN=[(-2,6),(-2,7),(-2,8),(2,6),(2,7),(2,8),(-1,8),(0,8),(1,8)]
SPRITE=PURPLE+TAN+BROWN
X0=7
def sprite_cells(y): return {(X0+dx, y+dy) for dx,dy in SPRITE}
def in_bounds(y): return all(0<=X0+dx<16 and 0<=y+dy<16 for dx,dy in SPRITE)
def count(y, rocks): return sum(1 for (rx,ry) in rocks if X0-2<=rx<=X0+2 and y<=ry<=y+7)
def in_basket(p,y): return X0-2<=p[0]<=X0+2 and y+6<=p[1]<=y+8
def step(y, rocks, action):
    rocks=set(rocks)
    w = count(y,rocks)>=3
    ny = y+1 if (w and in_bounds(y+1)) else (y-1 if (not w and in_bounds(y-1)) else y)
    clicked=False; nr=set(rocks)
    if isinstance(action,tuple):
        if action in rocks:
            nr = rocks-{action}; clicked=True
        elif in_basket(action,y) and action not in rocks and action not in sprite_cells(y):
            nr = rocks|{action}; clicked=True
    if not clicked:
        sc=sprite_cells(y)
        moved=set(); d = 1 if ny>y else (-1 if ny<y else 0)
        for r in rocks:
            rel = r[1]-y
            below=(r[0], r[1]+1)
            sup = below in sc or below in (rocks-{r})
            interior = (X0-2<=r[0]<=X0+2) and rel in (6,7)
            if r in sc: moved.add(r)
            elif d==1 and interior: moved.add((r[0],r[1]+1))
            elif d==-1 and interior and sup: moved.add((r[0],r[1]-1))
            elif d==0 and (not w) and in_basket(r,y):
                p=r
                while True:
                    b=(p[0],p[1]+1)
                    if b in sc or b in (rocks-{r}) or b[1]>15: break
                    p=b
                moved.add(p)
            else: moved.add(r)
        nr=moved
    return ny, frozenset(nr)
def render(y, rocks):
    c={}
    for dx,dy in PURPLE: c[(X0+dx,y+dy)]="mediumpurple"
    for dx,dy in TAN:    c[(X0+dx,y+dy)]="tan"
    for dx,dy in BROWN:  c[(X0+dx,y+dy)]="brown"
    for r in rocks: c[r]="gray"
    return c
