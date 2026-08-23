(program
(= GRID_SIZE 13)
(: UPDATE_RATE Number)
(= UPDATE_RATE 10)

(object Car (: speed Number) (Cell 0 0 "blue"))
(object Track (: solid Bool) (Cell 0 0 (if solid then "white" else "black")))
(object Obstacle (Cell 0 0 "red"))
(object FinishLine (: color String) (Cell 0 0 color))
(object Score (: color String) (Cell 0 0 color))

(: frameCount Number)
(= frameCount (initnext 0 (% (+ (prev frameCount) 1) UPDATE_RATE)))

(: playedNumber Number)
(= playedNumber (initnext 0 (prev playedNumber)))

(: playerCar Car)
(= playerCar
(initnext
(Car 1 (Position 5 10)) ; Start near the bottom
(let
;; Only move when frameCount is 0 (every UPDATE_RATE frames)
(= shouldUpdate (== (prev frameCount) 0))
(if shouldUpdate
then (moveUpNoCollision (prev "playerCar"))
else (prev "playerCar"))
)
)
)

(: scores (List Score))
(= scores (initnext (list ) (prev scores)))

;; Define the track as a list of track cells (white lines or black empty road)
(: tracks (List Track))
(= tracks
(initnext
(list
;; A few lines marking track boundaries or lanes
(Track true (Position 0 0)) (Track true (Position 0 1))
(Track true (Position 11 0)) (Track true (Position 11 1))
)
(prev tracks)
)
)
(: obstacle1 Obstacle)
(= obstacle1 (initnext (Obstacle (Position 5 2)) (if (== (prev frameCount) 0) then (moveDown obstacle1) else (prev obstacle1))))

(: obstacle2 Obstacle)
(= obstacle2 (initnext (Obstacle (Position 3 7)) (if (== (prev frameCount) 0) then (moveDown obstacle2) else (prev obstacle2))))

(: obstacle3 Obstacle)
(= obstacle3 (initnext (Obstacle (Position 7 5)) (if (== (prev frameCount) 0) then (moveDown obstacle3) else (prev obstacle3))))

(: finish FinishLine)
(= finish (initnext (FinishLine "gold" (Position 5 0)) (prev finish)))

(: raceOver Bool)
(= raceOver false)

(on left (= playerCar (moveLeftNoCollision (prev "playerCar"))))
(on right (= playerCar (moveRightNoCollision (prev "playerCar"))))

(on up (= UPDATE_RATE (max (- UPDATE_RATE 1) 3)))
(on down (= UPDATE_RATE (min (+ UPDATE_RATE 1) 18)))

(= check_crash (fn (obs c)
(| (== (deltaPos (.. obs origin) (.. c origin)) (Position 0 1)) (== (deltaPos (.. obs origin) (.. c origin)) (Position 0 0)))))
(on (| (| (check_crash obstacle1 playerCar) (check_crash obstacle2 playerCar)) (check_crash obstacle3 playerCar))
(let
(= raceOver true)
; (= finish (updateObj (prev finish) "color" "red"))
(print "CRASH! Game Over.")
;; reset the game after a delay
(if (== (prev frameCount) 0)
then (let
(= playerCar (updateObj (prev "playerCar") "origin" (Position 5 10)))
(= obstacle1 (updateObj (prev obstacle1) "origin" (Position 5 2)))
(= obstacle2 (updateObj (prev obstacle2) "origin" (Position 3 7)))
(= obstacle3 (updateObj (prev obstacle3) "origin" (Position 7 5)))
(= finish (updateObj (prev finish) "color" "gold"))
(= scores (addObj (prev scores) (Score "red" (Position (- GRID_SIZE 1) (- 12 (prev playedNumber))))))
(= playedNumber (+ (prev playedNumber) 1))
(= raceOver false)
)
else true
)
)
)

(on (adjacentTwoObjs (prev "playerCar") finish 1)
(let
(= raceOver true)
; (= finish (updateObj (prev finish) "color" "green"))
(print "You reached the finish line! You win!")
;; reset the game after a delay
(if (== (prev frameCount) 0)
then (let
(= playerCar (updateObj (prev "playerCar") "origin" (Position 5 10)))
(= obstacle1 (updateObj (prev obstacle1) "origin" (Position 5 2)))
(= obstacle2 (updateObj (prev obstacle2) "origin" (Position 3 7)))
(= obstacle3 (updateObj (prev obstacle3) "origin" (Position 7 5)))
(= finish (updateObj (prev finish) "color" "gold"))
(= scores (addObj (prev scores) (Score "green" (Position (- GRID_SIZE 1) (- 12 (prev playedNumber))))))
(= playedNumber (+ (prev playedNumber) 1))
(= raceOver false)
)
else true
)
)
)
)
