"""Pure-python mirror of rink.sexp, to be validated against the engine."""
D={'left':(-1,0),'right':(1,0),'up':(0,-1),'down':(0,1)}
OPP={'left':'right','right':'left','up':'down','down':'up'}
def onice(x,y): return 3<=x<=24 and 3<=y<=24
def tick(st, a):
    x,y,slide,prevSlide = st
    px,py = x,y           # prev skater
    pslide = slide        # prev slide
    if a in D:
        prevSlide = a
        x += D[a][0]; y += D[a][1]
        # gate: (on (& A (! (in slide (list "none" OPP[A]))))) (= slide A)
        if slide not in ("none", OPP[a]):
            slide = a
    # stop
    if pslide != "none" and not onice(px,py):
        slide = "none"
    # slide move (2 cells from prev pos)
    if slide in D:
        x = px + 2*D[slide][0]; y = py + 2*D[slide][1]
    # enter
    if (not onice(px,py)) and onice(x,y):
        slide = prevSlide
    return (x,y,slide,prevSlide)
INIT=(0,0,"none","none")
