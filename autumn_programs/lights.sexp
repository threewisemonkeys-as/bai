(program
  (= GRID_SIZE 16)

  (object Lights (: turnedOn Bool) (: dir Number) 
  (map
    (--> (i)
      (Cell 
        (* i 
           (if (== dir 0) then 1 else
           (if (== dir 1) then 1 else
           (if (== dir 2) then 0 else
           (if (== dir 3) then -1 else
           (if (== dir 4) then -1 else
           (if (== dir 5) then -1 else
           (if (== dir 6) then 0 else
           (if (== dir 7) then 1 else
               0)))))))))
        (* i 
           (if (== dir 0) then 0 else
           (if (== dir 1) then 1 else
           (if (== dir 2) then 1 else
           (if (== dir 3) then 1 else
           (if (== dir 4) then 0 else
           (if (== dir 5) then -1 else
           (if (== dir 6) then -1 else
           (if (== dir 7) then -1 else
               0)))))))))
        (if turnedOn then "yellow" else "white")
      ))
    (range -2 3)))
  
  (: lights Lights)
  (= lights (initnext (Lights false 1 (Position 7 7))
                      (prev lights)))

  (on clicked (= lights (updateObj lights "turnedOn" (! (.. lights turnedOn)))))

  ; Move the lights up, down, left, and right if they are turned off
  (on (& up (! (.. lights turnedOn))) (= lights (updateObj lights moveUp)))
  (on (& down (! (.. lights turnedOn))) (= lights (updateObj lights moveDown)))
  (on (& left (! (.. lights turnedOn))) (= lights (updateObj lights moveLeft)))
  (on (& right (! (.. lights turnedOn))) (= lights (updateObj lights moveRight)))

  ;; ; rotate the lights if they are turned on
  (on (& (.. lights turnedOn) up) (= lights  (updateObj lights "dir" (% (+ (.. lights dir) 1) 8))))
  (on (& (.. lights turnedOn) down) (= lights  (updateObj lights "dir" (% (+ (.. lights dir) 7) 8))))
  (on (& (.. lights turnedOn) left) (= lights  (updateObj lights "dir" (% (+ (.. lights dir) 2) 8))))
  (on (& (.. lights turnedOn) right) (= lights  (updateObj lights "dir" (% (+ (.. lights dir) 6) 8))))

)