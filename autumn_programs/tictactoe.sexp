(program
;; --------------------------------
;; Grid & Game Settings
;; --------------------------------
(= GRID_SIZE 5) ; We're still using a 16×16 grid
(= background "gray") ; Optional board background color

;; Track the current player's symbol: 1 or 2
(: currentPlayer Number)
(= currentPlayer (initnext 1 (prev currentPlayer)))

;; Track if the game has ended (win or draw)
(: gameOver Bool)
(= gameOver (initnext false (prev "gameOver")))

;; --------------------------------
;; TTTCell Definition
;; occupant can be 1, 2, or 0 (empty)
;; We'll color the cell based on occupant
;; (e.g. 1 => red, 2 => blue, empty => white)
;; --------------------------------
(object TTTCell (: occupant Number)
(Cell 0 0 (if (== occupant 1)
then "red"
else (if (== occupant 2) then "blue" else "white"))))

;; --------------------------------
;; Board Initialization
;; We lay out 9 cells in a 3x3 arrangement on the grid
;; using positions (1, 2, 3) for x and y so it's centered.
;; --------------------------------
(: board (List TTTCell))
(= board
(initnext
;; Initial creation of 3×3 TTTCells, occupant = 0
(map
(--> pos (TTTCell 0 (Position (+ (.. pos x) (/ GRID_SIZE 2)) (+ (.. pos y) (/ GRID_SIZE 2)))))
(rect (Position -1 -1) (Position 2 2))
)
;; No automatic changes each step, so we just keep the same board
(prev "board")
)
)

(= winnerSymbol 0)

(: winnerLine (List TTTCell))
(= winnerLine
(initnext list (prev winnerLine))
)

(on (> winnerSymbol 0)
(let
(= winnerLine (map
(--> pos (TTTCell winnerSymbol (Position (.. pos x) (.. pos y))))
(concat (list (rect (Position 0 4) (Position 5 5)) (rect (Position 0 0) (Position 5 1))
(rect (Position 0 0) (Position 1 4)) (rect (Position 4 0) (Position 5 5))
))
))
(winnerLine)
)
)

;; --------------------------------
;; Click to Place Symbol
;; --------------------------------
;; If gameOver == true, we ignore clicks.
;; Otherwise, if occupant == 0, place currentPlayer symbol,
;; then check for winner or toggle currentPlayer.
(on (clicked board)
(let
(= isOver gameOver)
(if isOver
;; If the game is already over, do nothing
then true else
(let
;; Identify which cell was clicked
(= clickedCells (filter (--> cell (clicked cell)) board))
(= isClickedEmpty (and (> (length clickedCells) 0) (== (.. (head clickedCells) occupant) 0)))
(if isClickedEmpty then
(let
(= clickedCell (head clickedCells))
;; Cell is empty, so place the current symbol
(= board (updateOccupant board (.. clickedCell origin) (currentPlayer)))
(= winnerSymbol (checkWinner board))
(print currentPlayer)
(= currentPlayer (- 3 currentPlayer))
(print currentPlayer)
(if (| (!= winnerSymbol 0) (isBoardFull board)) then
(let
(= gameOver true)
true
) else
true
)
true
) else
;; If occupant != 0, cell is already taken; do nothing
true
)
)
)
)
)

(= updateOccupant (fn (theBoard pos newOccupant)
;; Update occupant for the cell at 'pos'
(map
(--> cell (if (== (.. cell origin) pos) then
(updateObj cell "occupant" newOccupant) else
cell))
theBoard
)
))


;; --------------------------------
;; Check for Winner
;; --------------------------------
;; Return 1 or 2 if there's a winner, else 0
(= checkWinner (fn (theBoard)
(let
(= lines (winningLines theBoard))
;; lines is a list of occupant triplets. If any triplet is 1 1 1 or 2 2 2 => winner
(= winnerSymList (filter (--> trip (!= (allSame trip) 0)) lines))
(if (> (length winnerSymList) 0) then (let
(head (head winnerSymList))
) else 0)
)
))

(= allSame (fn (triple)
(let
;; triple is like (list occupant1 occupant2 occupant3)
;; If occupant1 == occupant2 == occupant3 and not 0, return occupant
(= a (head triple))
(= b (at triple 1))
(= c (tail triple))
(if (and (== a b) (and (== b c) (!= a 0)))
then a
else 0
)
)))

(= winningLines (fn (theBoard)
;; We gather occupant numbers for rows, columns, and 2 diagonals
(list
(getRow theBoard 1) (getRow theBoard 2) (getRow theBoard 3)
(getCol theBoard 1) (getCol theBoard 2) (getCol theBoard 3)
(getDiag theBoard (Position 2 2) 1 1) ; left-top to right-bottom
(getDiag theBoard (Position 2 2) -1 1) ; right-top to left-bottom
)
))

;; Return occupant numbers for a row (fixed y, x in 5..7).
(= getRow (fn (theBoard y)
(list
(occupantAt theBoard (Position 1 y))
(occupantAt theBoard (Position 2 y))
(occupantAt theBoard (Position 3 y))
)
))

;; Return occupant numbers for a column (fixed x, y in 5..7).
(= getCol (fn (theBoard x)
(list
(occupantAt theBoard (Position x 1))
(occupantAt theBoard (Position x 2))
(occupantAt theBoard (Position x 3))
)
))

;; Return occupant numbers along a diagonal.
;; Starting from startPos, move (dx, dy) for 3 steps
(= getDiag (fn (theBoard startPos dx dy)
(list
(occupantAt theBoard startPos)
(occupantAt theBoard (movePos startPos (Position dx dy)))
(occupantAt theBoard (movePos startPos (Position (- dx) (- dy))))
)
))

(= occupantAt (fn (theBoard pos)
;; Return occupant number of the cell in theBoard at pos
(let
(= found (filter (--> c (== (.. c origin) pos)) theBoard))
(if (== (length found) 1)
then (.. (head found) occupant)
else 0 ; fallback if no occupant found
)
)
))

;; --------------------------------
;; Check for Draw (board is full)
;; --------------------------------
(= isBoardFull (fn (theBoard)
;; If there's no cell occupant == 0, board is full
(! (any (--> c (== (.. c occupant) 0)) theBoard))
))
)

