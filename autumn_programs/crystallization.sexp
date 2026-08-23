(program
(= GRID_SIZE 10)
(= FRAME_RATE 7)
(object Particle (: crystal Bool) (Cell 0 0 (if crystal then "red" else "blue")))

(= nextParticle (fn (p) (let
(= adjObjs (filter (--> o (.. o crystal)) (adjacentObjs p 1)))
(if (== (length adjObjs) 0) then
(updateObj p "origin" (uniformChoice (adjPositions (.. p origin))))
else
(updateObj p "crystal" true)
)
)))

(: particles (List Particle))
(= particles
(initnext (map (--> p (Particle false p)) (rect (Position 2 2) (Position 6 6)))
(updateObj (prev particles) (--> obj (nextParticle obj)))))

(on clicked (= particles (addObj particles (Particle true (Position (.. click x) (.. click y))))))
)
