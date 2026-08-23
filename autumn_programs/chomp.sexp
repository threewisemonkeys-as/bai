(program
; The Chop game
(= GRID_SIZE 7)
(= background "black")

(object PlayerTracker (: currentPlayer Number) (Cell -1 -1 background))
(: playerTracker PlayerTracker)
(= playerTracker (initnext (PlayerTracker 0 (Position -1 -1)) (prev "playerTracker")))

(object Block (: selected Bool) (: game_over Bool) (Cell 0 0 (if game_over then background else
(if selected then (if (== (.. playerTracker currentPlayer) 0) then "blue" else "green") else "brown"))))


(: blocks (List Block))
(= blocks (initnext
(concat
(map (--> i
(map (--> j (Block false false (Position i j)))
(range (if (== i 1) then 2 else 1) (- GRID_SIZE 2))))
(range 1 ( - GRID_SIZE 1))))
(prev "blocks")))

(object Button (: color String) (Cell 0 0 color))

(: removeButton Button)
(= removeButton (initnext (Button "orange" (Position 3 (- GRID_SIZE 1))) (prev "removeButton")))


(object Player (: which Number) (Cell 0 0
(if (== which 0) then (if (== (.. playerTracker currentPlayer) which) then "blue" else "lightblue") else (if (== (.. playerTracker currentPlayer) which) then "green" else "lightgreen") )))


(: player2 Player)
(= player2 (initnext (Player 1 (Position (- GRID_SIZE 1) (- GRID_SIZE 1))) (prev "player2")))

(: player1 Player)
(= player1 (initnext (Player 0 (Position 0 (- GRID_SIZE 1))) (prev "player1")))

(object PoisonedBlock (: game_over Bool) (Cell 0 0 (if game_over then background else "purple")))

(: poisoned_block PoisonedBlock)
(= poisoned_block (initnext (PoisonedBlock false (Position 1 1)) (prev "poisoned_block")))


; Set all blocks to black when poisoned block is clicked
(on (clicked poisoned_block) (= blocks (updateObj blocks (--> b (updateObj b "game_over" true)))))
(on (clicked poisoned_block) (= poisoned_block (updateObj poisoned_block "game_over" true)))
(on (clicked poisoned_block) (= removeButton (updateObj removeButton "color" (if (== (.. playerTracker currentPlayer) 0) then "blue" else "green"))))

; When a block is clicked, select all blocks below and to the right of it
(on (& (clicked blocks) (! (.. poisoned_block game_over)))
(= blocks
(updateObj
blocks
(--> b (updateObj b "selected"
(& (>= (.. (.. b origin) x) (.. click x))
(>= (.. (.. b origin) y) (.. click y)))
)))))

; Remove all selected blocks when clicked
(on (clicked removeButton)
(= playerTracker
(if (any (--> b (.. b selected)) blocks)
then (updateObj playerTracker "currentPlayer" (- 1 (.. (prev "playerTracker") currentPlayer)))
else (prev "playerTracker"))))
(on (clicked removeButton)
(= blocks (if (any (--> b (.. b selected)) blocks) then (removeObj blocks (--> b (.. b selected))) else (prev "blocks"))))
; switch turns only if there were selected blocks

)
