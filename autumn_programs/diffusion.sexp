(program
(= GRID_SIZE 9)
(= FRAME_RATE 6)

(object RedCell (Cell 0 0 "red"))
(object BlueCell (Cell 0 0 "blue"))
(object Membrane (map (--> p (Cell 0 p "white")) (filter (--> y (== 0 (% y 3))) (range 0 GRID_SIZE))))

(: density Number)
(= density (initnext 2 (prev density)))

(: all_pos (List Position))
(= all_pos (list (Position (/ GRID_SIZE 2) 0) (Position (/ GRID_SIZE 2) 1) (Position (/ GRID_SIZE 2) 2)))

(: membranes (List Membrane))
(= membranes (initnext (map (--> p (Membrane p)) all_pos) (map (--> p (Membrane p)) (filter (--> p (<= (.. p y) density)) all_pos))))

(: redCells (List RedCell))
(= redCells (initnext (list (RedCell (Position 1 1))) (updateObj (prev redCells) (--> r (uniformChoice (list (moveDownNoCollision r) (moveUpNoCollision r) (moveLeftNoCollision r) (moveRightNoCollision r)))))))

(: blueCells (List BlueCell))
(= blueCells (initnext (list (BlueCell (Position (- GRID_SIZE 1) (- GRID_SIZE 1)))) (updateObj (prev blueCells) (--> b (uniformChoice (list (moveDownNoCollision b) (moveUpNoCollision b) (moveLeftNoCollision b) (moveRightNoCollision b)))))))

(on clicked (if (<= (.. click x) (/ GRID_SIZE 2)) then (= redCells (addObj redCells (RedCell (Position (.. click x) (.. click y))))) else (= blueCells (addObj blueCells (BlueCell (Position (.. click x) (.. click y)))))))

(on up (= density (% (+ (prev density) 1) 3)))

(on true (print density))
)
