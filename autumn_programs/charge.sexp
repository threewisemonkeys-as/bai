(program
(= GRID_SIZE 7)
(= background "black")

(object Jumper (map (--> pos (Cell (.. pos x) (.. pos y) "red")) (rect (Position 0 0) (Position 2 2))))
(object Stop (: toggle Bool) (list (Cell 0 0 (if toggle then "gold" else "blue")) (Cell 1 0 (if toggle then "gold" else "blue"))))

(: jumper Jumper)
(= jumper (initnext (Jumper (Position 0 (- GRID_SIZE 2))) (moveDownNoCollision (prev jumper))))

(: stop Stop)
(= stop (initnext (Stop false (Position 3 0)) (prev "stop")))

(: energy Int)
(= energy (initnext 0 (prev energy)))

(: time Int)
(= time (initnext 0 (+ (prev time) 1)))

(on ((left)) (= jumper (moveLeftNoCollision jumper)))
(on ((right)) (= jumper (moveRightNoCollision jumper)))

(on (& (& (== (.. (.. (prev jumper) origin) y) (- GRID_SIZE 2)) (<= energy 3)) (== (% time 3) 0))
(let (= energy (+ energy 1))
true
)
)

(on (& clicked (
let (print (.. jumper origin))
(print (- GRID_SIZE 2))
(print energy)
(& (== (.. (.. (prev jumper) origin) y) (- GRID_SIZE 2)) (> energy 0))
))
(let (= jumper (move (prev jumper) (Position 0 (- 0 (prev energy)))))
(= energy 0)
true
))

(on (adjacentTwoObjs (prev stop) (prev jumper) 1)
(= stop (updateObj stop "toggle" (! (.. stop toggle)))))

; (on true (print (list (.. (.. jumper origin) y) (.. (.. jumper origin) x) (.. stop toggle) energy )))
)
