(program
(= GRID_SIZE 8)

(object Peg (: player Bool) (: active Bool) (Cell 0 0 (if player then "lightgreen" else "lightblue")))
(object Highlight (Cell 0 0 "yellow"))

(: player1 (List Peg))
(= player1 (initnext (map (--> p (Peg true false p))
(list (Position 0 0) (Position 0 1) (Position 1 0) (Position 1 1) (Position 2 0) (Position 2 1) (Position 1 2) (Position 0 2) (Position 0 3) (Position 3 0)))
(prev player1)))

(: player2 (List Peg))
(= player2 (initnext (map (--> p (Peg false false p))
(list (Position 7 7) (Position 7 6) (Position 6 7) (Position 6 6) (Position 5 7) (Position 5 6) (Position 6 5) (Position 7 5) (Position 7 4) (Position 4 7)))
(prev player2)))

(: highlights (List Highlight))
(= highlights (list))

(= nextPossPegs (fn (peg) (let
; next pos is 2 cells away where the middle cell is not free
(= poss (list (Position (+ (.. (.. peg origin) x) 2) (.. (.. peg origin) y)) (Position (.. (.. peg origin) x) (+ (.. (.. peg origin) y) 2)) (Position (- (.. (.. peg origin) x) 2) (.. (.. peg origin) y)) (Position (.. (.. peg origin) x) (- (.. (.. peg origin) y) 2))))
(filter (--> p (&& (&& (&& (< (.. p x) GRID_SIZE) (>= (.. p x) 0)) (&& (< (.. p y) GRID_SIZE) (>= (.. p y) 0))) (&& (isFreePos p) (! (== (length (filter (--> peg_ (== (.. peg_ origin) (Position (/ (+ (.. (.. peg origin) x) (.. p x)) 2) (/ (+ (.. (.. peg origin) y) (.. p y)) 2)))) (vcat (list player1 player2)))) 0))))) poss)
)))

; on clicking a peg, set it to active and set all other pegs to inactive and create a new list of active pos
(on (clicked player1) (let
(= activePos (Position (.. click x) (.. click y)))
; Clear previous highlights
(= highlights (list))
; Deactivate all other pegs first
(= player1 (updateObj player1 (--> p (if (== (.. p origin) activePos) then (updateObj p "active" true) else (updateObj p "active" false)))))
(= player2 (updateObj player2 (--> p (if (== (.. p origin) activePos) then (updateObj p "active" true) else (updateObj p "active" false)))))
; Get possible moves for the active peg
(= possible_moves (nextPossPegs (head (objClicked player1))))
; Create highlight objects for possible move positions
(= highlights (map (--> pos (Highlight pos)) possible_moves))
))

(on (clicked player2) (let
(= activePos (Position (.. click x) (.. click y)))
; Clear previous highlights
(= highlights (list))
; Deactivate all other pegs first
(= player1 (updateObj player1 (--> p (if (== (.. p origin) activePos) then (updateObj p "active" true) else (updateObj p "active" false)))))
(= player2 (updateObj player2 (--> p (if (== (.. p origin) activePos) then (updateObj p "active" true) else (updateObj p "active" false)))))
; Get possible moves for the active peg
(= possible_moves (nextPossPegs (head (objClicked player2))))
; Create highlight objects for possible move positions
(= highlights (map (--> pos (Highlight pos)) possible_moves))
))

(on (clicked highlights) (let
(= clicked_pos (Position (.. click x) (.. click y)))
; clear the highlights
(= highlights (list))
; move the active peg to the clicked position
(= player1 (updateObj player1 (--> p (if (.. p active) then
(updateObj p "origin" clicked_pos)
else p))))
(= player2 (updateObj player2 (--> p (if (.. p active) then
(updateObj p "origin" clicked_pos)
else p))))
; add highlights for the next possible moves after the move and activate the new active peg
(= possible_moves (nextPossPegs (head (filter (--> peg (== (.. peg origin) clicked_pos)) (vcat (list player1 player2))))))
(= highlights (map (--> pos (Highlight pos)) possible_moves))
))
)
