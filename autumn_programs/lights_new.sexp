(program
(= GRID_SIZE 24)

(object Wire (Cell 0 0 "gray"))

; (: lights (List Light))
; (= lights (initnext (map (--> pos (Light false pos)) (filter (--> pos (== (.. pos x) (.. pos y))) (allPositions GRID_SIZE))) (prev lights)))

(: wires (List Wire))
(= wires (initnext
(list
(Wire (Position 2 5))
(Wire (Position 2 6))
(Wire (Position 2 7))
(Wire (Position 2 8))
(Wire (Position 2 9))
(Wire (Position 2 10))
(Wire (Position 2 11))
(Wire (Position 2 12))

(Wire (Position 3 5))
(Wire (Position 4 5))
(Wire (Position 5 5))
(Wire (Position 6 5))
(Wire (Position 7 5))
(Wire (Position 8 5))
(Wire (Position 9 5))
(Wire (Position 10 5))
(Wire (Position 11 5))
(Wire (Position 12 5))
(Wire (Position 12 5))
(Wire (Position 12 6))
(Wire (Position 12 7))
(Wire (Position 12 8))
(Wire (Position 12 9))
(Wire (Position 12 10))
(Wire (Position 12 11))
(Wire (Position 12 12))

(Wire (Position 3 12))
(Wire (Position 4 12))
(Wire (Position 5 12))
(Wire (Position 6 12))
(Wire (Position 7 12))
(Wire (Position 8 12))
(Wire (Position 9 12))
(Wire (Position 10 12))
(Wire (Position 11 12))

(Wire (Position 13 12))
(Wire (Position 14 12))
(Wire (Position 15 12))
(Wire (Position 16 12))
(Wire (Position 17 12))
(Wire (Position 18 12))
(Wire (Position 19 12))
(Wire (Position 20 12))
(Wire (Position 21 12))

(Wire (Position 21 13))
(Wire (Position 21 14))
(Wire (Position 21 15))
(Wire (Position 21 16))
(Wire (Position 21 17))
(Wire (Position 21 18))

(Wire (Position 2 13))
(Wire (Position 2 14))
(Wire (Position 2 15))
(Wire (Position 2 16))
(Wire (Position 2 17))
(Wire (Position 2 18))
(Wire (Position 2 19))

(Wire (Position 3 19))
(Wire (Position 4 19))
(Wire (Position 5 19))
(Wire (Position 6 19))
(Wire (Position 7 19))
(Wire (Position 8 19))
(Wire (Position 9 19))
(Wire (Position 10 19))
(Wire (Position 11 19))
(Wire (Position 12 19))
(Wire (Position 13 19))
(Wire (Position 14 19))
(Wire (Position 15 19))
(Wire (Position 16 19))
(Wire (Position 17 19))
(Wire (Position 18 19))
(Wire (Position 19 19))
(Wire (Position 20 19))
(Wire (Position 21 19))

(Wire (Position 3 15))
(Wire (Position 4 15))
(Wire (Position 5 15))
(Wire (Position 6 15))
(Wire (Position 7 15))
(Wire (Position 8 15))
(Wire (Position 9 15))
(Wire (Position 10 15))
(Wire (Position 11 15))
(Wire (Position 12 15))
(Wire (Position 13 15))
(Wire (Position 14 15))
(Wire (Position 15 15))
(Wire (Position 16 15))
(Wire (Position 17 15))
(Wire (Position 18 15))
(Wire (Position 19 15))
(Wire (Position 20 15))

)
(prev wires)
)
)

(object Light (: power Number)
(addObj (map (--> (i) (Cell 0 (- 1 i) (if (>= power i) then "yellow" else "black"))) (range 1 5)) (Cell 0 1 "orange")))


(: light1 Light)
(= light1 (initnext (Light 0 (Position 5 4)) (prev light1)))

(: light2 Light)
(= light2 (initnext (Light 0 (Position 5 11)) (prev light2)))

(: light3 Light)
(= light3 (initnext (Light 0 (Position 15 11)) (prev light3)))

(object Switch (: onOff Bool)
(list (Cell 0 1 (if onOff then "grey" else "brown"))
(Cell 0 0 (if onOff then "brown" else "grey")))
)

(: switch1 Switch)
(= switch1 (initnext (Switch false (Position 9 4)) (prev switch1)))

(: switch2 Switch)
(= switch2 (initnext (Switch false (Position 9 11)) (prev switch2)))

(: switch3 Switch)
(= switch3 (initnext (Switch false (Position 19 11)) (prev switch3)))

(: switch4 Switch)
(= switch4 (initnext (Switch false (Position 10 14)) (prev switch4)))

(: switches (List Switch))
(= switches (initnext (list "switch1" "switch2" "switch3") (prev switches)))

(object PowerSupply (: working Bool)
(concat
(map (--> j
(map (--> i (Cell i j (if working then "green" else "red"))) (range 0 3)))
(range 0 3))
)
)

(: powersupply PowerSupply)
(= powersupply (initnext (PowerSupply true (Position 16 18)) (prev powersupply)))

(on (clicked switch1) (= switch1 (updateObj switch1 "onOff" (! (.. (prev switch1) onOff)))))
(on (clicked switch2) (= switch2 (updateObj switch2 "onOff" (! (.. (prev switch2) onOff)))))
(on (clicked switch3) (= switch3 (updateObj switch3 "onOff" (! (.. (prev switch3) onOff)))))
(on (clicked switch4) (= switch4 (updateObj switch4 "onOff" (! (.. (prev switch4) onOff)))))

(on (clicked switch4) (= powersupply (updateObj powersupply "working" false)))

(on true (let
(= light1 (updateObj light1 "power" (if (! (.. powersupply working)) then 0
else (if (! (.. switch3 onOff)) then 0
else (if (& (.. switch1 onOff) (.. switch2 onOff)) then 4
else (if (.. switch1 onOff) then 3 else 0))))))
(= light2 (updateObj light2 "power" (if (! (.. powersupply working)) then 0
else (if (! (.. switch3 onOff)) then 0
else (if (& (.. switch1 onOff) (.. switch2 onOff)) then 4
else (if (.. switch2 onOff) then 3 else 0))))))
(= light3 (updateObj light3 "power" (if (! (.. powersupply working)) then 0
else (if (! (.. switch3 onOff)) then 0
else (if (& (.. switch1 onOff) (.. switch2 onOff)) then 2
else (if (or (.. switch1 onOff) (.. switch2 onOff)) then 3 else 0)))))
0
)))
)
