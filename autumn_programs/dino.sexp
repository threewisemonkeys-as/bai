(program
(= GRID_SIZE 20)
(= background "black")
(= FRAME_RATE 7)

(object Dino (: visible Bool) (list (Cell 0 0 (if visible then (if visible then "red" else background) else background)) (Cell 0 1 (if visible then "red" else background)) (Cell 1 0 (if visible then "red" else background)) (Cell 0 2 (if visible then "red" else background)) (Cell 0 3 (if visible then "red" else background)) (Cell -1 2 (if visible then "red" else background))))
(object Cactus (list (Cell 0 0 "green") (Cell 0 1 "green") (Cell 0 2 "green")))
(object Bird (Cell 0 0 "yellow"))

(: dino Dino)
(= dino (initnext (Dino true (Position 2 (- GRID_SIZE 4))) (prev dino)))

(: cactus Cactus)
(= cactus (initnext (Cactus (Position (- GRID_SIZE 1) (- GRID_SIZE 3))) (moveLeft (prev cactus))))

(: bird Bird)
(= bird (initnext (Bird (Position (+ GRID_SIZE 7) (- GRID_SIZE 5))) (moveLeft (prev bird))))

; when bird hits wall, put bird in random position
(on (== (.. (.. bird origin) x) 0) (= bird (updateObj (prev bird) "origin" (Position (- GRID_SIZE 1) (uniformChoice (list (- GRID_SIZE 6) (- GRID_SIZE 7) (- GRID_SIZE 5)))))))

; when cactus hits wall, put cactus initial position
(on (== (.. (.. cactus origin) x) 0) (= cactus (updateObj (prev cactus) "origin" (Position (- GRID_SIZE 1) (- GRID_SIZE 3)))))

; jump when up is pressed then move automatically down
(on up (= dino (move dino (Position 0 -7))))

; move dino down automatically (until it reaches the bottom)
(on true (= dino (moveDownNoCollision dino)))

; on collision with cactus or bird, make dino invisible
(on (intersects (prev dino) (prev cactus)) (= dino (updateObj dino "visible" false)))
(on (intersects (prev dino) (prev bird)) (= dino (updateObj dino "visible" false)))

)
