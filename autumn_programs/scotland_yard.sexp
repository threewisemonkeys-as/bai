(program
(= GRID_SIZE 20)

(object Station (Cell 0 0 "lightblue"))

(object Connection (Cell 0 0 "yellow"))

(object MrX (: moving Bool) (: direction String) (Cell 0 0 "red"))

(: connections (List Connection))
(= connections (initnext (list
(Connection (Position 5 5)) (Connection (Position 6 5)) (Connection (Position 7 5)) (Connection (Position 8 5)) (Connection (Position 9 5)) (Connection (Position 10 5)) (Connection (Position 11 5)) (Connection (Position 12 5)) (Connection (Position 13 5)) (Connection (Position 14 5))
(Connection (Position 14 5)) (Connection (Position 14 6)) (Connection (Position 14 7)) (Connection (Position 14 8)) (Connection (Position 14 9)) (Connection (Position 14 10)) (Connection (Position 14 11)) (Connection (Position 14 12)) (Connection (Position 14 13)) (Connection (Position 14 14))
(Connection (Position 14 14)) (Connection (Position 13 14)) (Connection (Position 12 14)) (Connection (Position 11 14)) (Connection (Position 10 14)) (Connection (Position 9 14)) (Connection (Position 8 14)) (Connection (Position 7 14)) (Connection (Position 6 14)) (Connection (Position 5 14))
(Connection (Position 5 14)) (Connection (Position 5 13)) (Connection (Position 5 12)) (Connection (Position 5 11)) (Connection (Position 5 10)) (Connection (Position 5 9)) (Connection (Position 5 8)) (Connection (Position 5 7)) (Connection (Position 5 6)) (Connection (Position 5 5))
(Connection (Position 2 10)) (Connection (Position 3 10)) (Connection (Position 4 10)) (Connection (Position 5 10)) (Connection (Position 6 10)) (Connection (Position 7 10)) (Connection (Position 8 10)) (Connection (Position 9 10)) (Connection (Position 10 10)) (Connection (Position 11 10)) (Connection (Position 12 10)) (Connection (Position 13 10)) (Connection (Position 14 10)) (Connection (Position 15 10)) (Connection (Position 16 10)) (Connection (Position 17 10))
(Connection (Position 8 2)) (Connection (Position 8 3)) (Connection (Position 8 4)) (Connection (Position 8 5)) (Connection (Position 8 6)) (Connection (Position 8 7)) (Connection (Position 8 8)) (Connection (Position 8 9)) (Connection (Position 8 10)) (Connection (Position 8 11)) (Connection (Position 8 12)) (Connection (Position 8 13)) (Connection (Position 8 14)) (Connection (Position 8 15)) (Connection (Position 8 16))
(Connection (Position 12 4)) (Connection (Position 12 5)) (Connection (Position 12 6)) (Connection (Position 12 7)) (Connection (Position 12 8)) (Connection (Position 12 9)) (Connection (Position 12 10)) (Connection (Position 12 11)) (Connection (Position 12 12)) (Connection (Position 12 13)) (Connection (Position 12 14)) (Connection (Position 12 15)) (Connection (Position 12 16))
(Connection (Position 6 17)) (Connection (Position 7 17)) (Connection (Position 8 17)) (Connection (Position 9 17)) (Connection (Position 10 17)) (Connection (Position 11 17)) (Connection (Position 12 17)) (Connection (Position 13 17))
(Connection (Position 5 3)) (Connection (Position 5 4)) (Connection (Position 5 5))
(Connection (Position 14 3)) (Connection (Position 14 4)) (Connection (Position 14 5))
(Connection (Position 5 14)) (Connection (Position 5 15)) (Connection (Position 5 16))
(Connection (Position 14 14)) (Connection (Position 14 15)) (Connection (Position 14 16))
(Connection (Position 2 2)) (Connection (Position 3 2)) (Connection (Position 4 2)) (Connection (Position 5 2)) (Connection (Position 6 2)) (Connection (Position 7 2)) (Connection (Position 8 2)) (Connection (Position 9 2)) (Connection (Position 10 2)) (Connection (Position 11 2)) (Connection (Position 12 2)) (Connection (Position 13 2)) (Connection (Position 14 2)) (Connection (Position 15 2)) (Connection (Position 16 2))
(Connection (Position 4 18)) (Connection (Position 5 18)) (Connection (Position 6 18)) (Connection (Position 7 18)) (Connection (Position 8 18)) (Connection (Position 9 18)) (Connection (Position 10 18)) (Connection (Position 11 18)) (Connection (Position 12 18)) (Connection (Position 13 18)) (Connection (Position 14 18)) (Connection (Position 15 18))
(Connection (Position 2 4)) (Connection (Position 2 5)) (Connection (Position 2 6)) (Connection (Position 2 7)) (Connection (Position 2 8)) (Connection (Position 2 9)) (Connection (Position 2 10)) (Connection (Position 2 11)) (Connection (Position 2 12)) (Connection (Position 2 13)) (Connection (Position 2 14)) (Connection (Position 2 15))
(Connection (Position 17 5)) (Connection (Position 17 6)) (Connection (Position 17 7)) (Connection (Position 17 8)) (Connection (Position 17 9)) (Connection (Position 17 10)) (Connection (Position 17 11)) (Connection (Position 17 12)) (Connection (Position 17 13)) (Connection (Position 17 14)) (Connection (Position 17 15))
(Connection (Position 4 16)) (Connection (Position 5 15)) (Connection (Position 6 14)) (Connection (Position 7 13)) (Connection (Position 8 12)) (Connection (Position 9 11)) (Connection (Position 10 10)) (Connection (Position 11 9)) (Connection (Position 12 8)) (Connection (Position 13 7)) (Connection (Position 14 6)) (Connection (Position 15 5)) (Connection (Position 16 4))
) (prev connections)))

; Stations at key intersections, endpoints, and strategic locations
(: stations (List Station))
(= stations (initnext (list
(Station (Position 5 5)) (Station (Position 14 5)) (Station (Position 14 14)) (Station (Position 5 14))
(Station (Position 10 5)) (Station (Position 14 10)) (Station (Position 10 14)) (Station (Position 5 10))
(Station (Position 2 10)) (Station (Position 8 10)) (Station (Position 12 10)) (Station (Position 17 10))
(Station (Position 8 2)) (Station (Position 8 8)) (Station (Position 8 16))
(Station (Position 12 4)) (Station (Position 12 12)) (Station (Position 12 16))
(Station (Position 6 17)) (Station (Position 10 17)) (Station (Position 13 17))
(Station (Position 5 3)) (Station (Position 5 2)) (Station (Position 2 4)) (Station (Position 2 2)) (Station (Position 14 3)) (Station (Position 5 16)) (Station (Position 14 16)) (Station (Position 2 15)) (Station (Position 5 18)) (Station (Position 8 13))
) (prev stations)))

; MrX starts at the central hub station
(: mrX MrX)
(= mrX (initnext (MrX false "none" (Position 10 10)) (prev mrX)))

; Helper functions for MrX's movement
(= getAdjacentConnections (fn (pos)
(filter (--> conn
(let
(= connPos (.. conn origin))
; Check if connection is adjacent (horizontally or vertically, no diagonals)
(| (& (== (.. connPos x) (.. pos x)) (== (abs (- (.. connPos y) (.. pos y))) 1))
(& (== (.. connPos y) (.. pos y)) (== (abs (- (.. connPos x) (.. pos x))) 1)))
)
) connections)
))

(= isAtStation (fn (pos)
(any (--> station (== (.. station origin) pos)) stations)
))

(= getValidNextPositions (fn (currentPos prevPos)
(let
(= adjConnections (getAdjacentConnections currentPos))
(= allPositions (map (--> conn (.. conn origin)) adjConnections))
; Filter out the previous position to avoid oscillation
(= filteredPositions (filter (--> pos (!= pos prevPos)) allPositions))
; If no valid positions after filtering, return all positions (including previous)
(if (> (length filteredPositions) 0) then
filteredPositions else
allPositions
)
)
))

(= getNextPosition (fn (currentPos prevPos)
(let
(= validPositions (getValidNextPositions currentPos prevPos))
(if (> (length validPositions) 0) then
(uniformChoice validPositions) else
currentPos
)
)
))

(= getNextPosInDirection (fn (currentPos direction)
(if (== direction "left") then
(Position (- (.. currentPos x) 1) (.. currentPos y))
else (if (== direction "right") then
(Position (+ (.. currentPos x) 1) (.. currentPos y))
else (if (== direction "up") then
(Position (.. currentPos x) (- (.. currentPos y) 1))
else (if (== direction "down") then
(Position (.. currentPos x) (+ (.. currentPos y) 1))
else currentPos))))))

; Helper function to calculate Manhattan distance between two positions
(= manhattanDistance (fn (pos1 pos2)
(+ (abs (- (.. pos1 x) (.. pos2 x))) (abs (- (.. pos1 y) (.. pos2 y)))
)))

; Find the nearest station to a given position
(= findNearestStation (fn (currentPos)
(let
(= stationPositions (map (--> station (.. station origin)) stations))
(= distances (map (--> pos (manhattanDistance currentPos pos)) stationPositions))
(= minDist (foldl (--> (d1 d2) (min d1 d2)) (head distances) distances))
(= nearestStation (head (filter (--> pos (== (manhattanDistance currentPos pos) minDist)) stationPositions)))
nearestStation
)
))

; Determine the best direction to move towards a target position
(= getDirectionToTarget (fn (currentPos targetPos)
(let
(= dx (- (.. targetPos x) (.. currentPos x)))
(= dy (- (.. targetPos y) (.. currentPos y)))
(if (> (abs dx) (abs dy)) then
(if (> dx 0) then "right" else "left")
else
(if (> dy 0) then "down" else "up")
)
)
))

; MrX's movement dynamics
(on left
(let
(= currentPos (.. mrX origin))
(= nextPos (getNextPosInDirection currentPos "left"))
(if (isValidMove nextPos) then
(let
(= mrX (updateObj (updateObj mrX "origin" nextPos) "direction" "left"))
(print "Moving left")
true
) else
(let
(print "Cannot move left")
false
)
)
)
)

(on right
(let
(= currentPos (.. mrX origin))
(= nextPos (getNextPosInDirection currentPos "right"))
(if (isValidMove nextPos) then
(let
(= mrX (updateObj (updateObj mrX "origin" nextPos) "direction" "right"))
(print "Moving right")
true
) else
(let
(print "Cannot move right")
false
)
)
)
)

(on up
(let
(= currentPos (.. mrX origin))
(= nextPos (getNextPosInDirection currentPos "up"))
(if (isValidMove nextPos) then
(let
(= mrX (updateObj (updateObj mrX "origin" nextPos) "direction" "up"))
(print "Moving up")
true
) else
(let
(print "Cannot move up")
false
)
)
)
)

(on down
(let
(= currentPos (.. mrX origin))
(= nextPos (getNextPosInDirection currentPos "down"))
(if (isValidMove nextPos) then
(let
(= mrX (updateObj (updateObj mrX "origin" nextPos) "direction" "down"))
(print "Moving down")
true
) else
(let
(print "Cannot move down")
false
)
)
)
)

; Continuous movement in the chosen direction
(on (!= (.. mrX direction) "none")
(let
(= currentPos (.. mrX origin))
(= atStation (isAtStation currentPos))
(if atStation then
(let
(= mrX (updateObj mrX "direction" "none"))
(print "Stopped at station")
true
) else
(let
(= nextPos (getNextPosInDirection currentPos (.. mrX direction)))
(if (isValidMove nextPos) then
(let
(= mrX (updateObj mrX "origin" nextPos))
(print "Continuing movement")
true
) else
(let
(= nearestStation (findNearestStation currentPos))
(= newDirection (getDirectionToTarget currentPos nearestStation))
(= mrX (updateObj mrX "direction" newDirection))
(print "Finding path to nearest station")
true
)
)
)
)
)
)

(= isValidMove (fn (pos)
(let
(= adjConnections (getAdjacentConnections (.. mrX origin)))
(= validPositions (map (--> conn (.. conn origin)) adjConnections))
(any (--> validPos (== validPos pos)) validPositions)
)
))
)
