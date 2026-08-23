(program
(= GRID_SIZE 10)
(= FRAME_RATE 4)
(= background "black") ; optional background color

(: timestep Number)
(= timestep (initnext 0 (+ (prev timestep) 1)))

;; -------------------------
;; Object Definitions
;; -------------------------
(object Pacman (: alive Bool) (Cell 0 0 (if alive then "yellow" else "black")))
(object Ghost (: color String) (Cell 0 0 color))
(object Wall (Cell 0 0 "gray"))
(object Pellet (Cell 0 0 "white"))

;; -------------------------
;; Global State
;; -------------------------
(: pacman Pacman)
(= pacman
(initnext
;; Initial state (center-ish of the grid)
(Pacman true (Position 5 5))
;; Each timestep, Pac-Man remains in its current position (unless moved by arrow key events)
(prev pacman)
)
)

;; A small set of ghost objects with different colors
(: ghosts (List Ghost))
(= ghosts
(initnext
(list
(Ghost "red" (Position 1 1))
(Ghost "cyan" (Position 8 8))
)
;; Example ghost AI: move them randomly or chase pacman
(updateObj (prev ghosts) (--> g (if (& (.. pacman alive) (== (% timestep 3) 0)) then (chasePacman g (prev pacman)) else g)))
)
)

;; A few walls around the edges or in the center
(: walls (List Wall))
(= walls
(initnext
(list
(Wall (Position 0 0)) (Wall (Position 0 1)) ; top-left corner example
(Wall (Position 4 5)) ; a single wall in the middle
(Wall (Position 9 9)) ; bottom-right corner example
)
(prev walls)
)
)

;; Pellets scattered around
(: pellets (List Pellet))
(= pellets
(initnext
(list
(Pellet (Position 2 2))
(Pellet (Position 5 4))
(Pellet (Position 7 5))
(Pellet (Position 8 2))
)
(prev pellets)
)
)

;; Keep track of player score
(: score Number)
(= score (initnext 0 (prev score)))

;; -------------------------
;; Movement "AI" for Ghosts
;; -------------------------
(= chasePacman (fn (ghost pacman)
;; Very simplified chase: if ghost is left of pacman, move right; if above, move down, etc.
(let
(= pacPos (.. pacman origin))
(= new_ghost (move ghost (unitVectorObjPos ghost pacPos)))
(if (isWithinBounds (.. new_ghost origin)) then
new_ghost
else
ghost
)
)
))

;; -------------------------
;; Player Controls (Pac-Man)
;; -------------------------
(on (& ((left)) (.. pacman alive)) (let
(= new_pacman (moveNoCollisionWithBlocker pacman -1 0 walls))
(if (in (.. new_pacman origin) (rect (Position 0 0) (Position GRID_SIZE GRID_SIZE))) then
(= pacman new_pacman)
else
pacman
)
))
(on (& ((right)) (.. pacman alive)) (let
(= new_pacman (moveNoCollisionWithBlocker pacman 1 0 walls))
(if (in (.. new_pacman origin) (rect (Position 0 0) (Position GRID_SIZE GRID_SIZE))) then
(= pacman new_pacman)
else
pacman
)
))
(on (& ((up)) (.. pacman alive)) (let
(= new_pacman (moveNoCollisionWithBlocker pacman 0 -1 walls))
(if (in (.. new_pacman origin) (rect (Position 0 0) (Position GRID_SIZE GRID_SIZE))) then
(= pacman new_pacman)
else
pacman
)
))
(on (& ((down)) (.. pacman alive)) (let
(= new_pacman (moveNoCollisionWithBlocker pacman 0 1 walls))
(if (in (.. new_pacman origin) (rect (Position 0 0) (Position GRID_SIZE GRID_SIZE))) then
(= pacman new_pacman)
else
pacman
)
))

(= moveNoCollisionWithBlocker (fn (agent dx dy blockers)
(let
(= newPos (Position
(+ (.. (.. agent origin) x) dx)
(+ (.. (.. agent origin) y) dy)))
(= renderedBlockers (renderValue blockers))
(if
(intersectsPosElems newPos renderedBlockers) ; Collides with a wall
then agent
else (move agent (Position dx dy))
)
)))

(= intersectsPos (fn (pos objects)
;; True if 'pos' intersects any object in 'objects'
(some (--> o (== (.. o origin) pos)) objects)
))

;; -------------------------
;; Collision Events
;; -------------------------

;; 1. Pac-Man eats pellets
(on (& (.. pacman alive) (intersects (prev pacman) (prev pellets)))
(let
;; Remove any pellets Pac-Man is intersecting
(= pellets (removeObj (prev pellets) (--> p (intersects p (prev pacman)))))
;; Increase score
(= score (+ (prev score) 10))
true
)
)

;; 2. Pac-Man hits a ghost => Game Over (or reduce life, etc.)
(on (& (.. pacman alive) (intersects (prev pacman) (prev ghosts)))
(let
(print "GAME OVER! Pac-Man got caught by a ghost.")
;; Optionally remove pacman, or freeze movement, etc.
(updateObj pacman "alive" false)
))
)

