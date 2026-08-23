(program
(= GRID_SIZE 12)

(object Goal (Cell 0 0 "green"))
(: goal Goal)
(= goal (initnext (Goal (Position 0 10)) (prev goal)))

(object Ball (: direction Number) (: color String) (Cell 0 0 color))

(object Wall (: visible Bool)
(list (Cell 0 0 (if visible then "white" else "gray"))
(Cell 0 1 (if visible then "white" else "gray"))
(Cell 0 2 (if visible then "white" else "gray"))))

(: wall Wall)
(= wall (initnext (Wall false (Position 4 9)) (prev wall)))

(= nextBall
(fn (ball)
(if (== (.. ball direction) 1)
then (updateObj ball "origin" (Position (.. (.. ball origin) x) (- (.. (.. ball origin) y) 1)))
else (if (== (.. ball direction) 2)
then (updateObj ball "origin" (Position (+ (.. (.. ball origin) x) 1) (- (.. (.. ball origin) y) 1)))
else (if (== (.. ball direction) 3)
then (updateObj ball "origin" (Position (+ (.. (.. ball origin) x) 1) (.. (.. ball origin) y)))
else (if (== (.. ball direction) 4)
then (updateObj ball "origin" (Position (+ (.. (.. ball origin) x) 1) (+ (.. (.. ball origin) y) 1)))
else (if (== (.. ball direction) 5)
then (updateObj ball "origin" (Position (.. (.. ball origin) x) (+ (.. (.. ball origin) y) 1)))
else (if (== (.. ball direction) 6)
then (updateObj ball "origin" (Position (- (.. (.. ball origin) x) 1) (+ (.. (.. ball origin) y) 1)))
else (if (== (.. ball direction) 7)
then (updateObj ball "origin" (Position (- (.. (.. ball origin) x) 1) (.. (.. ball origin) y)))
else (if (== (.. ball direction) 8)
then (updateObj ball "origin" (Position (- (.. (.. ball origin) x) 1) (- (.. (.. ball origin) y) 1)))
else ball))))))))))

(on (clicked wall) (= wall (updateObj wall "visible" (! (.. wall visible)))))

;; Existing wallintersect function for grid edge bounces
(= wallintersect (fn (ball)
(if (& (== (.. ball direction) 4) (== (.. (.. ball origin) y) GRID_SIZE)) then 2 else
(if (& (== (.. ball direction) 5) (== (.. (.. ball origin) y) GRID_SIZE)) then 1 else
(if (& (== (.. ball direction) 6) (== (.. (.. ball origin) y) GRID_SIZE)) then 8 else
(if (& (== (.. ball direction) 6) (== (.. (.. ball origin) x) 0)) then 4 else
(if (& (== (.. ball direction) 7) (== (.. (.. ball origin) x) 0)) then 3 else
(if (& (== (.. ball direction) 8) (== (.. (.. ball origin) x) 0)) then 2 else
(if (& (== (.. ball direction) 2) (== (.. (.. ball origin) x) GRID_SIZE)) then 8 else
(if (& (== (.. ball direction) 3) (== (.. (.. ball origin) x) GRID_SIZE)) then 7 else
(if (& (== (.. ball direction) 4) (== (.. (.. ball origin) x) GRID_SIZE)) then 6 else
(if (& (== (.. ball direction) 8) (== (.. (.. ball origin) y) 0)) then 6 else
(if (& (== (.. ball direction) 1) (== (.. (.. ball origin) y) 0)) then 5 else
(if (& (== (.. ball direction) 2) (== (.. (.. ball origin) y) 0)) then 4 else

(.. ball direction)))))))))))))))

;; New function to reflect the horizontal component when bouncing off a vertical wall
(= wallBounce
(fn (ball)
(if (== (.. ball direction) 2)
then 8
else (if (== (.. ball direction) 3)
then 7
else (if (== (.. ball direction) 4)
then 6
else (if (== (.. ball direction) 6)
then 4
else (if (== (.. ball direction) 7)
then 3
else (if (== (.. ball direction) 8)
then 2
else (.. ball direction)))))))))

;; Update ball_a to include a wall collision check if the wall is visible.
(: ball_a Ball)
(= ball_a
(initnext (Ball 7 "blue" (Position GRID_SIZE (- GRID_SIZE 2)))
(if (intersects ball_a ball_b)
then (nextBall (updateObj (prev ball_a) "direction" 6))
else (if (and (.. wall visible) (intersects (prev ball_a) wall))
then (nextBall (updateObj (prev ball_a) "direction" (wallBounce (prev ball_a))))
else (nextBall (updateObj (prev ball_a) "direction" (wallintersect (prev ball_a))))))))

;; Similarly update ball_b with a wall collision check.
(: ball_b Ball)
(= ball_b
(initnext (Ball 6 "red" (Position GRID_SIZE 5))
(if (intersects (prev ball_a) ball_b)
then (nextBall (updateObj (prev ball_b) "direction" 2))
else (if (and (.. wall visible) (intersects (prev ball_b) wall))
then (nextBall (updateObj (prev ball_b) "direction" (wallBounce (prev ball_b))))
else (nextBall (updateObj (prev ball_b) "direction" (wallintersect (prev ball_b))))))))

(on ((left)) (= wall (moveLeftNoCollision wall)))
(on ((right)) (= wall (moveRightNoCollision wall)))
)
