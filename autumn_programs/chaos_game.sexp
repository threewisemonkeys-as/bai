(program
(= GRID_SIZE 50)

(object Particle (: color String) (Cell 0 0 color))

(: particles_initial (List Particle))
(= particles_initial (list (Particle "red" (Position (/ GRID_SIZE 2) 0)) (Particle "blue" (Position 0 (- GRID_SIZE 1))) (Particle "green" (Position (- GRID_SIZE 1) (- GRID_SIZE 1)))))
(: particles (List Particle))
(= particles (initnext (list (Particle "black" (uniformChoice (rect (Position 0 0) (Position (- GRID_SIZE 1) (- GRID_SIZE 1))))))
(prev particles)
))

(on clicked (let
; Get the last particle from the initial list
(= lastParticle (tail particles))
(= lastPos (.. lastParticle origin))
; Randomly choose one of the initial particles
(= randomParticle (uniformChoice particles_initial))
(= color (.. randomParticle color))
(= randomPos (.. randomParticle origin))
; Get the midpoint between the last particle and the clicked position
(= midpoint (Position (/ (+ (.. lastPos x) (.. randomPos x)) 2) (/ (+ (.. lastPos y) (.. randomPos y)) 2)))
; Add the midpoint to the particles
(= particles (addObj particles (Particle color midpoint)))
))
)
