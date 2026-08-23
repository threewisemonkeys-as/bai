(program
(= GRID_SIZE 10)
(: COLORS (List String))
(= COLORS (list "red" "blue" "green" "yellow" "purple"))

(object Ball (: color String) (: moving Bool) (Cell 0 0 color))

(: destFlag Bool)
(= destFlag false)

(: destPos Position)
(= destPos (Position 0 0))

(: balls (List Ball))
(= balls (initnext (list (Ball "red" true (Position 0 0)) (Ball "blue" false (Position 4 5)) (Ball "green" false (Position 7 8))) (prev balls)))

; When the user clicks on the canvas, set the active ball position to the click position
(on (clicked balls) (let
; set active ball to moving
(= balls (updateObj balls (--> b (updateObj b "moving" true)) (--> b (== (.. b origin) click))))
; set others to false
(= balls (updateObj balls (--> b (updateObj b "moving" false)) (--> b (!= (.. b origin) click))))
))

; When the user clicks on the canvas and the click is on a free position, add a new ball at a random position and create a destination at the click position
(on (& ((clicked)) (isFreePos click)) (let
(= balls (addObj balls (Ball (uniformChoice COLORS) false (head (randomPositions GRID_SIZE 1)))))
(= destPos (Position (.. click x) (.. click y)))
(= destFlag true)
))

; When the user clicks on the canvas, move the active ball to the destination. If the ball is at the destination, remove the destination.
(on destFlag (let
(= activeBall (head (filter (--> b (.. b moving)) balls)))
(= balls (updateObj balls (--> b (move b (unitVectorObjPos b destPos))) (--> b (.. b moving))))
(= destFlag (if (== (.. activeBall origin) destPos) then false else destFlag))
))
)
