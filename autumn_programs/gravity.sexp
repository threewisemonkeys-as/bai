(program
    (= GRID_SIZE 17)
      
    (object Button (: color String) (Cell 0 0 color))
    (object Blob (list (Cell 0 -1 "blue") (Cell 0 0 "blue") (Cell 1 -1 "blue") (Cell 1 0 "blue")))
    
    (: leftButton Button)
    (= leftButton (initnext (Button "red" (Position 0 (/ GRID_SIZE 2))) (prev leftButton)))
    
    (: rightButton Button)
    (= rightButton (initnext (Button "darkorange" (Position (- GRID_SIZE 1) (/ GRID_SIZE 2))) (prev rightButton)))
      
    (: upButton Button)
    (= upButton (initnext (Button "gold" (Position (/ GRID_SIZE 2) 0)) (prev upButton)))
    
    (: downButton Button)
    (= downButton (initnext (Button "green" (Position (/ GRID_SIZE 2) (- GRID_SIZE 1))) (prev downButton)))
    
    (: blobs (List Blob))
    (= blobs (initnext (list) (prev blobs)))
    
    (: gravity String)
    (= gravity (initnext "down" (prev "gravity")))
    
    (on (== gravity "left") (= blobs (updateObj blobs (--> obj (moveLeft obj)))))
    (on (== gravity "right") (= blobs (updateObj blobs (--> obj (moveRight obj)))))
    (on (== gravity "up") (= blobs (updateObj blobs (--> obj (moveUp obj)))))
    (on (== gravity "down") (= blobs (updateObj blobs (--> obj (moveDown obj)))))
    
    (on (& ((clicked)) (isFreePos click)) (= blobs (addObj blobs (Blob (Position (.. click x) (.. click y))))) )
    
    (on (clicked leftButton) (= gravity "left"))
    
    (on (clicked rightButton) (= gravity "right"))
    
    (on (clicked upButton) (= gravity "up"))
    
    (on (clicked downButton) (= gravity "down"))
)
