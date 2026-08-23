(program
(= GRID_SIZE 28)
(= FRAME_RATE 3)
(= background "black")

(object Rink (map (--> pos (Cell (.. pos x) (.. pos y) "lightblue")) (rect (Position 0 0) (Position 22 22))))
(object Skater (Cell 0 0 "red"))

(: rink Rink)
(= rink (initnext (Rink (Position 3 3)) (prev rink)) )

(: skater Skater)
(= skater (initnext (Skater (Position 0 0)) (prev skater)))

(: slide String)
(= slide (initnext "none" (prev "slide")))

(: prevSlide String)
(= prevSlide (initnext "none" (prev "prevSlide")))

(on left (let (= prevSlide "left") (= skater (moveLeft skater))))
(on right (let (= prevSlide "right") (= skater (moveRight skater))))
(on up (let (= prevSlide "up") (= skater (moveUp skater))))
(on down (let (= prevSlide "down") (= skater (moveDown skater))))

(on (& left (! (in slide (list "none" "right")))) (= slide "left"))
(on (& right (! (in slide (list "none" "left")))) (= slide "right"))
(on (& up (! (in slide (list "none" "down")))) (= slide "up"))
(on (& down (! (in slide (list "none" "up")))) (= slide "down"))

(on (& (!= (prev "slide") "none") (! (intersects (prev skater) rink)))
(= slide "none"))

(on (== slide "left") (= skater (move (prev skater) (Position -2 0))))
(on (== slide "right") (= skater (move (prev skater) (Position 2 0))))
(on (== slide "up") (= skater (move (prev skater) (Position 0 -2))))
(on (== slide "down") (= skater (move (prev skater) (Position 0 2))))


(on (& (! (in (.. (prev skater) origin) (rect (Position 3 3) (Position 25 25)))) (in (.. skater origin) (rect (Position 3 3) (Position 25 25))))
(= slide prevSlide))
)
