(program
(= GRID_SIZE 24)

; Define enhanced objects
(object Switch (: powered Bool) (list
(Cell 0 0 (if powered then "red" else "pink"))
(Cell 0 1 (if powered then "red" else "pink"))
(Cell 1 0 (if powered then "red" else "pink"))
(Cell 1 1 (if powered then "red" else "pink"))))

(object Output (: powered Bool) (list
(Cell 0 0 (if powered then "orange" else "darkblue"))
(Cell 0 1 (if powered then "orange" else "darkblue"))
(Cell 1 0 (if powered then "orange" else "darkblue"))
(Cell 1 1 (if powered then "orange" else "darkblue"))))

; Wire object - just a single cell
(object Wire (: powered Bool) (: dir Bool) (if dir then (list (Cell 0 0 (if powered then "yellow" else "grey")) (Cell 0 1 (if powered then "yellow" else "grey")) (Cell 0 2 (if powered then "yellow" else "grey")) (Cell 0 3 (if powered then "yellow" else "grey")))
else (list (Cell 0 0 (if powered then "yellow" else "grey")) (Cell 1 0 (if powered then "yellow" else "grey")) (Cell 2 0 (if powered then "yellow" else "grey")) (Cell 3 0 (if powered then "yellow" else "grey")) (Cell 4 0 (if powered then "yellow" else "grey")) (Cell 5 0 (if powered then "yellow" else "grey")))))

; Logic functions
(= evaluateAND (--> (a b) (& a b)))
(= evaluateOR (--> (a b) (| a b)))
(= evaluateNOT (--> (a) (! a)))
(= evaluateXOR (--> (a b) (| (& a (! b)) (& (! a) b))))

; Create input switches with better positioning
(: switch1 Switch)
(= switch1 (initnext (Switch false (Position 4 12)) (prev switch1)))

(: switch2 Switch)
(= switch2 (initnext (Switch false (Position 19 12)) (prev switch2)))

; === AND Gate (Row 1) ===
; Wires from switch1 to AND
(: andWire1 (List Wire))
(= andWire1 (initnext (list (Wire false true (Position 5 8)) (Wire false true (Position 5 4)) (Wire false false (Position 6 4)))
(updateObj (prev andWire1) (--> w (updateObj w "powered" (.. (prev switch1) powered))))
))

; Wires from switch2 to AND
(: andWire2 (List Wire))
(= andWire2 (initnext (list (Wire false true (Position 19 8)) (Wire false true (Position 19 4)) (Wire false false (Position 14 4)))
(updateObj (prev andWire2) (--> w (updateObj w "powered" (.. (prev switch2) powered))))
))

; AND gate output
(: andOutput Output)
(= andOutput (initnext
(Output false (Position 12 4))
(Output (evaluateAND (.. switch1 powered) (.. switch2 powered)) (Position 12 4))))

; === OR Gate (Row 2) ===
; Wires from switch1 to OR
(: orWire1 (List Wire))
(= orWire1 (initnext (list (Wire false true (Position 5 8)) (Wire false false (Position 6 8)))
(updateObj (prev orWire1) (--> w (updateObj w "powered" (.. (prev switch1) powered))))
))

; Wires from switch2 to OR
(: orWire2 (List Wire))
(= orWire2 (initnext (list (Wire false true (Position 19 8)) (Wire false false (Position 14 8)))
(updateObj (prev orWire2) (--> w (updateObj w "powered" (.. (prev switch2) powered))))
))

; OR gate output
(: orOutput Output)
(= orOutput (initnext
(Output false (Position 12 8))
(Output (evaluateOR (.. switch1 powered) (.. switch2 powered)) (Position 12 8))))

; === NOT Gate (Row 3) ===
; Wires from switch1 to NOT
(: notWire (List Wire))
(= notWire (initnext (list (Wire false false (Position 6 12)) (Wire false true (Position 12 12)))
(updateObj (prev notWire) (--> w (updateObj w "powered" (.. (prev switch1) powered))))
))

; NOT gate output
(: notOutput Output)
(= notOutput (initnext
(Output false (Position 12 16))
(Output (evaluateNOT (.. (prev andOutput) powered)) (Position 12 16))))

; === XOR Gate (Row 4) ===
; Wires from switch1 to XOR
(: xorWire1 (List Wire))
(= xorWire1 (initnext (list (Wire false true (Position 5 14)) (Wire false true (Position 5 18)) (Wire false false (Position 6 21)))
(updateObj (prev xorWire1) (--> w (updateObj w "powered" (.. (prev switch1) powered))))
))

; Wires from switch2 to XOR
(: xorWire2 (List Wire))
(= xorWire2 (initnext (list (Wire false true (Position 19 14)) (Wire false true (Position 19 18)) (Wire false false (Position 14 21)))
(updateObj (prev xorWire2) (--> w (updateObj w "powered" (.. (prev switch2) powered))))
))

; XOR gate output
(: xorOutput Output)
(= xorOutput (initnext
(Output false (Position 12 20))
(Output (evaluateAND (.. (prev orOutput) powered) (.. (prev notOutput) powered)) (Position 12 20))))

; Click handlers for all switches
(on (clicked switch1) (= switch1 (updateObj switch1 "powered" (! (.. (prev switch1) powered)))))
(on (clicked switch2) (= switch2 (updateObj switch2 "powered" (! (.. (prev switch2) powered)))))
)
