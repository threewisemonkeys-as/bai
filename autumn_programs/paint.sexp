(program
    (= GRID_SIZE 16)
    
    (object Particle (: color String) (Cell 0 0 color))
    
    (: particles (List Particle))
    (= particles (initnext (list) (prev "particles")))
    
    (: currColor String)
    (= currColor (initnext "red" (prev currColor)))

    (: active_arrow String)
    (= active_arrow (initnext "none" (prev active_arrow)))
    
    (on clicked (= particles (addObj (prev "particles") (Particle (if (== active_arrow "none") then "red" else currColor) (Position (.. click x) (.. click y))))))
    (on up (let (= currColor "gold") (= active_arrow "up")))
    (on down (let (= currColor "purple") (= active_arrow "down")))
    (on left (let (= currColor "green") (= active_arrow "left")))
    (on right (let (= currColor "blue") (= active_arrow "right")))

    (on (and down (== (prev active_arrow) "up")) (let (= active_arrow "none") (= currColor "red")))
    (on (and up (== (prev active_arrow) "down")) (let (= active_arrow "none") (= currColor "red")))
    (on (and left (== (prev active_arrow) "right")) (let (= active_arrow "none") (= currColor "red")))
    (on (and right (== (prev active_arrow) "left")) (let (= active_arrow "none") (= currColor "red")))
)
