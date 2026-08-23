(program
(= GRID_SIZE 10)

(object Grid (: is_mine Bool) (: is_revealed Bool) (Cell 0 0 (if is_revealed then (if is_mine then "red" else "green") else "gray")))
(object Border (: mine_active Bool) (map (--> p (Cell (.. p x) (.. p y) (if mine_active then "red" else "gray"))) (rect (Position 0 0) (Position GRID_SIZE GRID_SIZE))))

(: grids (List Grid))
(= grids (initnext (map (--> p (Grid (== 1 (uniformChoice (list 1 2 3 4))) false p)) (rect (Position 0 0) (Position GRID_SIZE GRID_SIZE))) (let
(= revealed_mines (filter (--> g (and (.. g is_mine) (.. g is_revealed))) (prev grids)))
(= areas_to_reveal (vcat (map (--> g (rect (Position (- (.. (.. g origin) x) 1) (- (.. (.. g origin) y) 1)) (Position (+ (.. (.. g origin) x) 2) (+ (.. (.. g origin) y) 2)))) revealed_mines)))
(updateObj (prev grids) (--> grid (updateObj grid "is_revealed" true)) (--> grid (in (.. grid origin) areas_to_reveal)))
)))

(on (clicked grids) (= grids (updateObj grids (--> grid (updateObj grid "is_revealed" true)) (--> grid (== (.. grid origin) (Position (.. click x) (.. click y)))))))
)
