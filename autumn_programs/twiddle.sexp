(program
(= GRID_SIZE 3)
(object Twiddle (: color String) (Cell 0 0 color))

(= rotate_clockwise (fn (ts c) (let
(= pos1 (if (== c 1) then (Position 0 0)
else (if (== c 2) then (Position 1 0)
else (if (== c 3) then (Position 0 1)
else (Position 1 1)))))

(= pos2 (if (== c 1) then (Position 1 0)
else (if (== c 2) then (Position 2 0)
else (if (== c 3) then (Position 1 1)
else (Position 2 1)))))

(= pos3 (if (== c 1) then (Position 1 1)
else (if (== c 2) then (Position 2 1)
else (if (== c 3) then (Position 1 2)
else (Position 2 2)))))

(= pos4 (if (== c 1) then (Position 0 1)
else (if (== c 2) then (Position 1 1)
else (if (== c 3) then (Position 0 2)
else (Position 1 2)))))

(= t1 (head (filter (--> o (== (.. o origin) pos1)) ts)))
(= t2 (head (filter (--> o (== (.. o origin) pos2)) ts)))
(= t3 (head (filter (--> o (== (.. o origin) pos3)) ts)))
(= t4 (head (filter (--> o (== (.. o origin) pos4)) ts)))

(= ts (updateObj ts (--> t (moveRight t)) (--> t (== t t1))))
(= ts (updateObj ts (--> t (moveDown t)) (--> t (== t t2))))
(= ts (updateObj ts (--> t (moveLeft t)) (--> t (== t t3))))
(= ts (updateObj ts (--> t (moveUp t)) (--> t (== t t4))))
ts
)))

(= shuffle_twiddles (fn (ts) (let
(= num_shuffles (uniformChoice (list 5 6 7 8 9)))
(print num_shuffles)
(= cs (uniformChoice (list 1 2 3 4) num_shuffles))
(print cs)
(= ts (head (map (--> c (rotate_clockwise ts c)) cs)))
ts
)))

(: twiddles (List Twiddle))
(= twiddles (initnext (shuffle_twiddles (list (Twiddle "pink" (Position 0 0))
(Twiddle "purple" (Position 0 1))
(Twiddle "blue" (Position 1 0))
(Twiddle "gold" (Position 1 1))
(Twiddle "orangered" (Position 1 2))
(Twiddle "cyan" (Position 2 0))
(Twiddle "green" (Position 2 1))
(Twiddle "magenta" (Position 2 2))
(Twiddle "gray" (Position 0 2))
)) (prev twiddles)))

(: currentset Number)
(= currentset (initnext 0 (prev currentset)))

(on clicked
; (= currentset (if (in (Position (.. click x) (.. click y)) (list (Position 0 0) (Position 0 1) (Position 1 1)))
; then 1
; else (if (in (Position (.. click x) (.. click y)) (list (Position 1 0) (Position 2 0)))
; then 2
; else (if (in (Position (.. click x) (.. click y)) (list (Position 0 2) (Position 1 2)))
; then 3
; else 4
;))))
(let
(= pos (Position (.. click x) (.. click y)))
(= currentset (if (== pos (Position 0 0)) then
1
else (if (== pos (Position 2 0)) then
2
else (if (== pos (Position 0 2)) then
3
else (if (== pos (Position 2 2)) then
4
else 0)))))
true
)
)

(on (== currentset 1) (let
(= twiddles (rotate_clockwise twiddles 1))
(= currentset 0)
))

(on (== currentset 2) (let
(= twiddles (rotate_clockwise twiddles 2))
(= currentset 0)
))

(on (== currentset 3) (let
(= twiddles (rotate_clockwise twiddles 3))
(= currentset 0)
))

(on (== currentset 4) (let
(= twiddles (rotate_clockwise twiddles 4))
(= currentset 0)
))
)
