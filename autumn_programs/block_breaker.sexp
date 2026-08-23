(program
(= GRID_SIZE 13)

(object Ball (: direction Number) (Cell 0 0 "red"))

(object Paddle (list (Cell 0 0 "gold") (Cell 1 0 "gold") (Cell -1 0 "gold")))

(object Balloon (: color String) (Cell 0 0 color))

(: paddle Paddle)
(= paddle (initnext (Paddle (Position 4 6)) (prev paddle)))

(= randBalloon (fn () (Balloon (uniformChoice (list "skyblue" "mediumpurple" "pink")) (uniformChoice (vcat (list (rect (Position 0 0) (Position GRID_SIZE (/ GRID_SIZE 2))) (rect (Position 0 (+ (/ GRID_SIZE 2) 1)) (Position GRID_SIZE GRID_SIZE))))))))

(= nextBall
(fn (ball)
(if (== (.. ball direction) 1)
then (moveNoCollision ball 0 -1)
else (if (== (.. ball direction) 2)
then (moveNoCollision ball 1 -1)
else (if (== (.. ball direction) 3)
then (moveNoCollision ball 1 0)
else (if (== (.. ball direction) 4)
then (moveNoCollision ball 1 1)
else (if (== (.. ball direction) 5)
then (moveNoCollision ball 0 1)
else (if (== (.. ball direction) 6)
then (moveNoCollision ball -1 1)
else (if (== (.. ball direction) 7)
then (moveNoCollision ball -1 0)
else (if (== (.. ball direction) 8)
then (moveNoCollision ball -1 -1)
else ball))))))))))

(= nextBalloons (--> (local_ball local_balloons) (let
(= near_ball (rect (Position (- (.. (.. local_ball origin) x) 1) (- (.. (.. local_ball origin) y) 1)) (Position (+ (.. (.. local_ball origin) x) 2) (+ (.. (.. local_ball origin) y) 2))))
(filter (--> obj2 (! (in (.. obj2 origin) near_ball))) local_balloons)
)))

;; Existing wallintersect function for grid edge bounces
(= wallintersect (fn (ball)
(if (& (== (.. ball direction) 4) (>= (.. (.. ball origin) y) (- GRID_SIZE 1))) then 2 else
(if (& (== (.. ball direction) 5) (>= (.. (.. ball origin) y) (- GRID_SIZE 1))) then 1 else
(if (& (== (.. ball direction) 6) (>= (.. (.. ball origin) y) (- GRID_SIZE 1))) then 8 else
(if (& (== (.. ball direction) 6) (<= (.. (.. ball origin) x) 0)) then 4 else
(if (& (== (.. ball direction) 7) (<= (.. (.. ball origin) x) 0)) then 3 else
(if (& (== (.. ball direction) 8) (<= (.. (.. ball origin) x) 0)) then 2 else
(if (& (== (.. ball direction) 2) (>= (.. (.. ball origin) x) (- GRID_SIZE 1))) then 8 else
(if (& (== (.. ball direction) 3) (>= (.. (.. ball origin) x) (- GRID_SIZE 1))) then 7 else
(if (& (== (.. ball direction) 4) (>= (.. (.. ball origin) x) (- GRID_SIZE 1))) then 6 else
(if (& (== (.. ball direction) 8) (<= (.. (.. ball origin) y) 0)) then 6 else
(if (& (== (.. ball direction) 1) (<= (.. (.. ball origin) y) 0)) then 5 else
(if (& (== (.. ball direction) 2) (<= (.. (.. ball origin) y) 0)) then 4 else
(.. ball direction)))))))))))))))

;; New function to reflect the vertical component when bouncing off a horizontal paddle
(= paddleBounce
(fn (ball)
(if (== (.. ball direction) 2)
then 4
else (if (== (.. ball direction) 4)
then 2
else (if (== (.. ball direction) 6)
then 8
else (if (== (.. ball direction) 8)
then 6
else (.. ball direction)))))))

(= paddleBounceCorner (fn (ball)
(if (== (.. ball direction) 2)
then 8
else (if (== (.. ball direction) 4)
then 6
else (if (== (.. ball direction) 6)
then 4
else (if (== (.. ball direction) 8)
then 2
else (.. ball direction)))))))

(: ball Ball)
(= ball
(initnext (Ball 6 (Position (- GRID_SIZE 1) 5)) (let
(= delta (deltaPos (.. (prev ball) origin) (.. paddle origin)))
(= ballPaddleCornerIntersect (in delta (list (Position 2 1) (Position 2 -1) (Position -2 -1) (Position -2 1))))
(= ballPaddleIntersect (in delta (list (Position 0 1) (Position 0 -1) (Position -1 1) (Position 1 1) (Position -1 -1) (Position 1 -1))))
(if ballPaddleCornerIntersect then (nextBall (updateObj (prev ball) "direction" (paddleBounceCorner (prev ball))))
else (if ballPaddleIntersect
then (nextBall (updateObj (prev ball) "direction" (paddleBounce (prev ball))))
else (nextBall (updateObj (prev ball) "direction" (wallintersect (prev ball))))))
)))

(: balloons (List Balloon))
(= balloons (initnext (map (--> i ((randBalloon))) (range 0 20)) (nextBalloons (prev ball) (prev balloons))))

(on ((left)) (= paddle (moveLeftNoCollision paddle)))
(on ((right)) (= paddle (moveRightNoCollision paddle)))
)
