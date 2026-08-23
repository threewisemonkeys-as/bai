(program
(= GRID_SIZE 25)

(: CENTER Number)
(= CENTER (/ (+ 1 GRID_SIZE) 2))

(object Glitter (: color String) (list (Cell 0 0 color) (Cell 0 1 color) (Cell 1 0 color)))

(: COLORS (List String))
(= COLORS (list "gold" "silver" "pink" "mediumpurple" "lightgreen"))

(= nextReflGlitter (fn (glitterList) (let
; quadrant 1
(= quadrant1 (filter (--> g (& (& (>= (.. (.. g origin) x) 0) (< (.. (.. g origin) x) CENTER)) (& (>= (.. (.. g origin) y) 0) (< (.. (.. g origin) y) CENTER)))) glitterList))
; quadrant 2
(= quadrant2 (filter (--> g (& (& (>= (.. (.. g origin) x) CENTER) (< (.. (.. g origin) x) GRID_SIZE)) (& (>= (.. (.. g origin) y) 0) (< (.. (.. g origin) y) CENTER)))) glitterList))
; quadrant 3
(= quadrant3 (filter (--> g (& (& (>= (.. (.. g origin) x) 0) (< (.. (.. g origin) x) CENTER)) (& (>= (.. (.. g origin) y) CENTER) (< (.. (.. g origin) y) GRID_SIZE)))) glitterList))
; quadrant 4
(= quadrant4 (filter (--> g (& (& (>= (.. (.. g origin) x) CENTER) (< (.. (.. g origin) x) GRID_SIZE)) (& (>= (.. (.. g origin) y) CENTER) (< (.. (.. g origin) y) GRID_SIZE)))) glitterList))

; reflect quadrant 1
(= reflected1 (map (--> g (let
; reflect to 2
(= relfGlitter2 (rotate (Glitter (.. g color) (Position (- GRID_SIZE (.. (.. g origin) x)) (.. (.. g origin) y)))))
; reflect to 3
(= relfGlitter3 (rotate (rotate (rotate (Glitter (.. g color) (Position (.. (.. g origin) x) (- GRID_SIZE (.. (.. g origin) y))))))))
; reflect to 4
(= relfGlitter4 (rotate (rotate (Glitter (.. g color) (Position (- GRID_SIZE (.. (.. g origin) x)) (- GRID_SIZE (.. (.. g origin) y)))))))
(list relfGlitter2 relfGlitter3 relfGlitter4)
)) quadrant1))
; reflect quadrant 2
(= reflected2 (map (--> g (let
; reflect to 1
(= relfGlitter1 (rotate (Glitter (.. g color) (Position (- GRID_SIZE (.. (.. g origin) x)) (.. (.. g origin) y)))))
; reflect to 3
(= relfGlitter3 (rotate (rotate (Glitter (.. g color) (Position (- GRID_SIZE (.. (.. g origin) x)) (- GRID_SIZE (.. (.. g origin) y)))))))
; reflect to 4
(= relfGlitter4 (rotate (rotate (rotate (Glitter (.. g color) (Position (.. (.. g origin) x) (- GRID_SIZE (.. (.. g origin) y))))))))
(list relfGlitter1 relfGlitter3 relfGlitter4)
)) quadrant2))
; reflect quadrant 3
(= reflected3 (map (--> g (let
; reflect to 1
(= relfGlitter1 (rotate (rotate (rotate (Glitter (.. g color) (Position (.. (.. g origin) x) (- GRID_SIZE (.. (.. g origin) y))))))))
; reflect to 2
(= relfGlitter2 (rotate (rotate (Glitter (.. g color) (Position (- GRID_SIZE (.. (.. g origin) x)) (- GRID_SIZE (.. (.. g origin) y)))))))
; reflect to 4
(= relfGlitter4 (rotate (Glitter (.. g color) (Position (- GRID_SIZE (.. (.. g origin) x)) (.. (.. g origin) y)))))
(list relfGlitter1 relfGlitter2 relfGlitter4)
)) quadrant3))
; reflect quadrant 4
(= reflected4 (map (--> g (let
; reflect to 1
(= relfGlitter1 (rotate (rotate (rotate (Glitter (.. g color) (Position (- GRID_SIZE (.. (.. g origin) x)) (- GRID_SIZE (.. (.. g origin) y))))))))
; reflect to 2
(= relfGlitter2 (rotate (rotate (Glitter (.. g color) (Position (- GRID_SIZE (.. (.. g origin) x)) (.. (.. g origin) y))))))
; reflect to 3
(= relfGlitter3 (rotate (Glitter (.. g color) (Position (.. (.. g origin) x) (- GRID_SIZE (.. (.. g origin) y))))))
(list relfGlitter1 relfGlitter2 relfGlitter3)
)) quadrant4))

; combine all reflected glitter
(vcat (list reflected1 reflected2 reflected3 reflected4))
)))

(: original_glitter (List Glitter))
(= original_glitter (initnext (list) (prev original_glitter)))

(: refl_glitter (List Glitter))
(= refl_glitter (initnext (list) (nextReflGlitter original_glitter)))

; add glitter to original_glitter and rotate it randomly
(on clicked (let
(= glitter (Glitter (uniformChoice COLORS) (Position (.. click x) (.. click y))))
; (= original_glitter (addObj original_glitter (uniformChoice (list glitter (rotate glitter) (rotate (rotate glitter)) (rotate (rotate (rotate glitter)))))))
(= original_glitter (addObj original_glitter glitter))
))
)
