(program
(= GRID_SIZE 16)
(= FRAME_RATE 2)

(object Particle (Cell 0 0 "blue"))

(: particles (List Particle))
(= particles (initnext (list (Particle (Position 7 7)))
(updateObj (prev particles)
(--> obj (let
(= new_par (Particle (uniformChoice (adjPositions (.. obj origin)))))
(if (isWithinBounds new_par) then new_par else obj))))))

(on clicked (= particles (vcat (list (prev particles) (map (--> p (Particle (.. p origin))) (prev particles))))))
)
