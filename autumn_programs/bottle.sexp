(program 
  (= GRID_SIZE 11)

  (object Suzie (Cell 0 0 "blue"))
  (object Billy (Cell 0 0 "red"))

  (object Bottle (: broken Bool) (list (Cell 0 0 (if broken then "yellow" else "white"))
                                        (Cell 0 1 (if broken then "white" else "yellow"))
                                        (Cell 0 2 (if broken then "gold" else "yellow"))
                                        (Cell 0 3 (if broken then "white" else "yellow"))
                                        (Cell 0 4 (if broken then "yellow" else "white"))))

  (object BottleSpot (Cell 0 0 "white"))
  (object Rock (: breaksBottle Bool) (Cell 0 0 "gray"))

  (: suzie Suzie)
  (= suzie (Suzie (Position 0 0)))

  (: billy Billy)
  (= billy (Billy (Position 0 (- GRID_SIZE 1))))

  (: bottleSpot BottleSpot)
  (= bottleSpot (initnext (BottleSpot (Position (- GRID_SIZE 1) (/ GRID_SIZE 2))) (BottleSpot (Position (- GRID_SIZE 1) (/ GRID_SIZE 2)))))

  (: rocks (List Rock))
  (= rocks
    (initnext (list)
              (updateObj (prev rocks) (--> obj
                              (if (intersects bottleSpot obj) then (removeObj obj)
                                else (move obj (unitVector obj bottleSpot)))))))
  
  (= nextBottle (fn (bot rockst bottleSpott) (if 
  (> (length (filter (--> r (& (.. r breaksBottle) (intersects bottleSpott r))) rockst)) 0)
  then (updateObj bot "broken" true) 
  else bot)))

  (: bottle Bottle)
  (= bottle (initnext (Bottle false (Position (- GRID_SIZE 1) (- (/ GRID_SIZE 2) 2))) (nextBottle (prev bottle) (prev rocks) (prev bottleSpot))))

  (on (clicked suzie) (= rocks (addObj (prev rocks) (Rock (uniformChoice (list true false)) (Position 0 0)))))
  (on (clicked billy) (= rocks (addObj (prev rocks) (Rock (uniformChoice (list true false)) (Position 0 (- GRID_SIZE 1))))))
  (on (clicked bottle) (if (.. bottle broken) then (= bottle (updateObj bottle "broken" false)) else true))
  )