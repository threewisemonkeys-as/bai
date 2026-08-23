(program
  (= GRID_SIZE 12)
  (= background "white")
 (= FRAME_RATE 7)
  
  (object Mario (: bullets Number) (Cell 0 0 "red"))
  (object Step (: movingLeft Bool) (list (Cell -1 0 "darkorange") (Cell 0 0 "darkorange") (Cell 1 0 "darkorange")))
  (object Coin (Cell 0 0 "gold"))
  (object Enemy (: movingLeft Bool) (: lives Number) (list (Cell -1 0 "blue") (Cell 0 0 "blue") (Cell 1 0 "blue")
                                      (Cell -1 1 "blue") (Cell 0 1 "blue") (Cell 1 1 "blue")))
  (object Bullet (Cell 0 0 "mediumpurple"))
  
  (: mario Mario)
  (= mario (initnext (Mario 0 (Position 6 11)) (if (intersects (moveDown (prev mario)) (prev coins)) then (moveDown (prev mario)) else (moveDownNoCollision (prev mario)))))
  
  (: steps (List Step))
  (= steps (initnext (list (Step true (Position 1 10)) (Step true (Position 5 8)) (Step true (Position 9 6))) (updateObj (prev steps) (--> step (if (.. step movingLeft) then (moveLeft step) else (moveRight step))))))
  
  (: coins (List Coin))
  (= coins (initnext (list (Coin (Position 1 9)) (Coin (Position 7 4)) (Coin (Position 9 5))) (prev coins)))
  
  (: enemy Enemy)
  (= enemy (initnext (Enemy true 1 (Position 5 0)) (if (.. (prev enemy) movingLeft) then (moveLeft (prev enemy)) else (moveRight (prev enemy)))))
  
  (: bullets (List Bullet))
  (= bullets (initnext (list) (updateObj (prev bullets) (--> obj (if (intersects (moveUp obj) (prev steps)) then (removeObj obj) else (moveUp obj))))))
  
  (: enemyLives Number)
  (= enemyLives (initnext 1 (prev enemyLives)))
  
  (on (& (defined "enemy") ( == (.. (.. (prev enemy) origin) x) 1)) (= enemy (moveRight (updateObj (prev enemy) "movingLeft" false))))
  (on (& (defined "enemy") (== (.. (.. (prev enemy) origin) x) 10)) (= enemy (moveLeft (updateObj (prev enemy) "movingLeft" true))))
  
  (on left (= mario (if (intersects (moveLeft (prev mario)) (prev coins)) then (moveLeft (prev mario)) else (moveLeftNoCollision (prev mario)))))
  (on right (= mario (if (intersects (moveRight (prev mario)) (prev coins)) then (moveRight (prev mario)) else (moveRightNoCollision (prev mario)))))
  (on (& ((up)) (== (moveDownNoCollision (prev mario)) (prev mario))) (= mario (moveNoCollision (prev mario) 0 -4)))

  (on true (= steps (updateObj (prev steps) (--> step (updateObj step "movingLeft" false)) (--> step (== (.. (.. step origin) x) 1)))))
  (on true (= steps (updateObj (prev steps) (--> step (updateObj step "movingLeft" true)) (--> step (== (.. (.. step origin) x) 10)))))
  
  (on (intersects (prev mario) (prev coins)) 
    (let (= coins (removeObj (prev coins) (--> (obj) (intersects obj (prev mario))))) 
          (= mario (moveDownNoCollision (updateObj (prev mario) "bullets" (+ (.. (prev mario) bullets) 1))))) )
  
  (on (& ((clicked)) (> (.. (prev mario) bullets) 0)) 
    (let (= bullets (addObj (prev bullets) (Bullet (.. (prev mario) origin)))) 
          (= mario (moveDownNoCollision (updateObj (prev mario) "bullets" (- (.. (prev mario) bullets) 1))))
          true
          ))
  
  (on (& (defined "enemy") (intersects (prev enemy) (prev bullets)))
    (let (= bullets (removeObj (prev bullets) (--> obj (intersects obj (prev enemy))))) 
          (= enemy (if (== (prev enemyLives) 1) then (removeObj (prev enemy)) else (if (.. (prev enemy) movingLeft) then (moveLeft (prev enemy)) else (moveRight (prev enemy))) ))
          (= enemyLives (- (prev enemyLives) 1)))
          true
    )
)
