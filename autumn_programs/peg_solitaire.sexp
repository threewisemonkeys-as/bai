(program
(= GRID_SIZE 5)

; Define game objects
(object Peg (: empty Bool) (: active Bool) (: possible_next_move Bool) (Cell 0 0 (if active then "maroon" else (if (! empty) then "orangered" else (if possible_next_move then "pink" else "black")))))

(: initial_active_position Position)
(= initial_active_position (uniformChoice (list (Position 2 4) (Position 4 2) (Position 0 2) (Position 2 0))))

(: pegs (List Peg))
(= pegs (initnext (map (--> pos (Peg (== pos (Position 2 2)) (== pos initial_active_position) (== pos (Position 2 2)) pos))
(rect (Position 0 0) (Position GRID_SIZE GRID_SIZE))) (prev pegs)))

; Jump condition
(= check_jump_possible (--> (p1 p2) (
let (= delta (deltaPos p1 p2))
(and (in delta (list (Position 2 0) (Position -2 0) (Position 0 2) (Position 0 -2))) (== 1 (length (filter (--> peg (and (! (.. peg empty)) (== (.. peg origin) (Position (/ (+ (.. p1 x) (.. p2 x)) 2) (/ (+ (.. p1 y) (.. p2 y)) 2))))) (prev pegs)))))
)
))

(on (clicked pegs) (if (in (head (objClicked (prev pegs))) (filter (--> obj (and (.. obj empty) (.. obj possible_next_move))) (prev pegs))) then (let
; Find the peg that is active
(= active_pegs (filter (--> peg (.. peg active)) (prev pegs)))
; if no peg is active, then do nothing
(if (== 0 (length active_pegs)) then true else
(let
(= active_peg (head active_pegs))
; Find the peg in between
(= mid_pos (Position (/(+ (.. (.. active_peg origin) x) (.. click x)) 2) (/(+ (.. (.. active_peg origin) y) (.. click y)) 2)))
; Remove the peg in between if jump is possible
(= pegs (updateObj pegs (--> peg (updateObj peg "empty" true)) (--> peg (== (.. peg origin) mid_pos))))
; Make the clicked peg non-empty
(= pegs (updateObj pegs (--> peg (updateObj peg "empty" false)) (--> peg (== (.. peg origin) (Position (.. click x) (.. click y))))))
; Set the active peg to empty
(= pegs (updateObj pegs (--> peg (updateObj peg "empty" true)) (--> peg (== peg active_peg))))
; Set the next possible jumps to false and active to false
(= pegs (updateObj pegs (--> peg (updateObj (updateObj peg "possible_next_move" false) "active" false))))
))
true
)
else (if (in (head (objClicked (prev pegs))) (filter (--> obj (and (! (.. obj empty)) (! (.. obj active)))) (prev pegs))) then (let
(print "HERE")
; Set the clicked peg to active
(= pegs (updateObj pegs (--> peg (if (== (.. peg origin) (Position (.. click x) (.. click y))) then (updateObj peg "active" true) else (updateObj (updateObj peg "active" false) "possible_next_move" false)))))
; set the next possible jumps to true
(= pegs (updateObj pegs (--> peg (updateObj peg "possible_next_move" true)) (--> peg (check_jump_possible (Position (.. click x) (.. click y)) (.. peg origin)))))
true
) else true)
))
)
