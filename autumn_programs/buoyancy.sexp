(program
    (= GRID_SIZE 7)

    (object Crate (: addWeight Number) (list (Cell -2 -1 "brown") (Cell 2 -1 "brown") (Cell -2 0 "brown") (Cell 2 0 "brown") 
    (Cell -2 1 "brown") (Cell -1 1 "brown") (Cell 0 1 "brown") (Cell 1 1 "brown") (Cell 2 1 "brown")))
    
    (object Rock (list (Cell 0 0 "silver")))
    (object Water (Cell 0 0 "blue"))

    ; rock in crate sinks the crate
    (= inCrate (--> (rock) (let 
        (= crateOrigin (.. crate origin))
        ; (= box (rect (Position (- (.. crateOrigin x) 2) 0) (Position (+ (.. crateOrigin x) 3) (+ (.. crateOrigin y) 1))))
        (= rock_x (.. (.. rock origin) x))
        (= rock_y (.. (.. rock origin) y))
        (= res (and (and (>= rock_x (- (.. crateOrigin x) 2)) (< rock_x (+ (.. crateOrigin x) 3))) (and (>= rock_y 0) (< rock_y (+ (.. crateOrigin y) 1)))))
        res
    )))

    (= nextCrate (--> (crate rocks) (let
        (= rocksInCrate (filter (--> r (inCrate r)) rocks))
        (= weight (length rocksInCrate))
        (if (& (< (.. crate addWeight) weight) (< weight 5)) then (moveDown (updateObj crate "addWeight" (+ (.. crate addWeight) 1))) else crate)
    )))

    (: water (List Water))
    (= water (initnext (map (--> p (Water p)) (rect (Position 0 3) (Position GRID_SIZE GRID_SIZE))) (updateObj water (--> w (if (intersectsPosPoss (.. w origin) (map (--> r (.. r origin)) rocks)) then (moveUp w) else (nextLiquid w))))))

    ; move down the rock if it is within bounds and does not collide with other rocks or the crate
    (= nextRock (--> (rock) (let
        (= next_rock (moveDown rock))
        (= crate_origin (.. crate origin))
        (= crate_pos (list (Position (- (.. crate_origin x) 2) (- (.. crate_origin y) 1)) (Position (+ (.. crate_origin x) 2) (- (.. crate_origin y) 1))
        (Position (- (.. crate_origin x) 2) (.. crate_origin y)) (Position (+ (.. crate_origin x) 2) (.. crate_origin y))
        (Position (- (.. crate_origin x) 2) (+ (.. crate_origin y) 1)) (Position (- (.. crate_origin x) 1) (+ (.. crate_origin y) 1))
        (Position (.. crate_origin x) (+ (.. crate_origin y) 1)) (Position (+ (.. crate_origin x) 1) (+ (.. crate_origin y) 1))
        (Position (+ (.. crate_origin x) 2) (+ (.. crate_origin y) 1))))
        (= noObjBelow (if (in (.. next_rock origin) (vcat (list crate_pos (map (--> r (.. r origin)) rocks)))) then false else true))
        (= res (if (& (isWithinBounds next_rock) noObjBelow) then next_rock else rock))
        res
    )))

    (: rocks (List Rock))
    (= rocks (initnext (list) (updateObj rocks nextRock)))

    (: crate Crate)
    (= crate (initnext (Crate 0 (Position 3 1)) (nextCrate (prev crate) (prev rocks))))

    (on clicked (if (isFreePos (Position (.. click x) (.. click y))) then (= rocks (addObj rocks (Rock (Position (.. click x) (.. click y))))) else true))
)