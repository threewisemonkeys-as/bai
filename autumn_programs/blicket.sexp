(program
(= GRID_SIZE 11)

; shape 1 is up/down/left/right+middle and shape 2 is diagonals+middle
(object Blicket (: selected Bool) (: colour String) (: shape1 Bool) (: visible Bool) (map
(--> pos (Cell (.. pos x) (.. pos y)
(if visible then
(if (and shape1 (in pos (list (Position 0 0) (Position 0 1) (Position 1 0) (Position 0 -1) (Position -1 0)))) then colour
else (if (and (! shape1) (in pos (list (Position 0 0) (Position 1 1) (Position -1 -1) (Position -1 1) (Position 1 -1)))) then colour else "black")) else "black")))
(rect (Position -1 -1) (Position 2 2))))
(object Machine (: onOff Bool) (list (Cell -4 -2 "gray") (Cell -3 -2 "gray") (Cell -2 -2 "gray") (Cell -1 -2 "gray") (Cell 0 -2 "gray") (Cell 1 -2 "gray") (Cell 2 -2 "gray") (Cell 3 -2 "gray") (Cell 4 -2 "gray")
(Cell -4 -1 "gray") (Cell 4 -1 "gray")
(Cell -4 0 "gray") (Cell 4 0 "gray")
(Cell -4 1 "gray") (Cell 4 1 "gray")
(Cell -4 2 "gray") (Cell -3 2 "gray") (Cell -2 2 "gray") (Cell -1 2 "gray") (Cell 0 2 "gray") (Cell 1 2 "gray") (Cell 2 2 "gray") (Cell 3 2 "gray") (Cell 4 2 "gray")
(Cell -4 3 "gray") (Cell -3 3 "gray") (Cell -2 3 (if onOff then "green" else "red")) (Cell -1 3 (if onOff then "green" else "red")) (Cell 0 3 (if onOff then "green" else "red")) (Cell 1 3 (if onOff then "green" else "red")) (Cell 2 3 (if onOff then "green" else "red")) (Cell 3 3 "gray") (Cell 4 3 "gray")
(Cell -4 4 "gray") (Cell -3 4 "gray") (Cell -2 4 "gray") (Cell -1 4 "gray") (Cell 0 4 "gray") (Cell 1 4 "gray") (Cell 2 4 "gray") (Cell 3 4 "gray") (Cell 4 4 "gray")
))

(: blicket1 Blicket)
(= blicket1 (initnext (Blicket false "blue" false true (Position 3 5)) (updateObj blicket1 "visible" (| (.. (prev blicket1) selected) (.. (prev blicket1) visible)))))

(: blicket2 Blicket)
(= blicket2 (initnext (Blicket false "gold" true true (Position 7 5)) (updateObj blicket2 "visible" (| (.. (prev blicket2) selected) (.. (prev blicket2) visible)))))

(: machine Machine)
(= machine (initnext (Machine false (Position 5 5)) (prev machine)))

; up/down switches the shape of the blicket
(on (and (.. blicket1 selected) ((up))) (= blicket1 (updateObj blicket1 "shape1" true)))
(on (and (.. blicket2 selected) ((up))) (= blicket2 (updateObj blicket2 "shape1" true)))

(on (and (.. blicket1 selected) ((down))) (= blicket1 (updateObj blicket1 "shape1" false)))
(on (and (.. blicket2 selected) ((down))) (= blicket2 (updateObj blicket2 "shape1" false)))

(on (clicked blicket1) (let
(= blicket1 (updateObj (updateObj blicket1 "visible" (! (.. (prev blicket1) selected))) "selected" (! (.. blicket1 selected))))
(= blicket2 (updateObj blicket2 "selected" false))
true
))
(on (clicked blicket2) (let
(= blicket2 (updateObj (updateObj blicket2 "visible" (! (.. (prev blicket2) selected))) "selected" (! (.. blicket2 selected))))
(= blicket1 (updateObj blicket1 "selected" false))
true
))

(on (or (and (.. blicket1 shape1) (.. blicket1 visible)) (and (! (.. blicket2 shape1)) (.. blicket2 visible))) (= machine (updateObj machine "onOff" true)))
(on (! (or (and (.. blicket1 shape1) (.. blicket1 visible)) (and (! (.. blicket2 shape1)) (.. blicket2 visible)))) (= machine (updateObj machine "onOff" false)))
)
