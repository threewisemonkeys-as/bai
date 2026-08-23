(program
(= GRID_SIZE 9)

(object StormCloud (: charge Number) (list
(Cell 0 2 (if (< charge 8) then "skyblue" else (if (< charge 15) then "darkblue" else "purple")))
(Cell 1 2 (if (< charge 8) then "skyblue" else (if (< charge 15) then "darkblue" else "purple")))
(Cell 2 2 (if (< charge 8) then "skyblue" else (if (< charge 15) then "darkblue" else "purple")))
(Cell 3 2 (if (< charge 8) then "skyblue" else (if (< charge 15) then "darkblue" else "purple")))
(Cell 4 2 (if (< charge 8) then "skyblue" else (if (< charge 15) then "darkblue" else "purple")))
(Cell 0 1 (if (< charge 8) then "skyblue" else (if (< charge 15) then "darkblue" else "purple")))
(Cell 1 1 (if (< charge 8) then "skyblue" else (if (< charge 15) then "darkblue" else "purple")))
(Cell 2 1 (if (< charge 8) then "skyblue" else (if (< charge 15) then "darkblue" else "purple")))
(Cell 3 1 (if (< charge 8) then "skyblue" else (if (< charge 15) then "darkblue" else "purple")))
(Cell 4 1 (if (< charge 8) then "skyblue" else (if (< charge 15) then "darkblue" else "purple")))
(Cell 5 1 (if (< charge 8) then "skyblue" else (if (< charge 15) then "darkblue" else "purple")))
(Cell 1 0 (if (< charge 8) then "skyblue" else (if (< charge 15) then "darkblue" else "purple")))
(Cell 2 0 (if (< charge 8) then "skyblue" else (if (< charge 15) then "darkblue" else "purple")))
(Cell 3 0 (if (< charge 8) then "skyblue" else (if (< charge 15) then "darkblue" else "purple")))
(Cell 4 0 (if (< charge 8) then "skyblue" else (if (< charge 15) then "darkblue" else "purple")))
(Cell 2 -1 (if (< charge 8) then "skyblue" else (if (< charge 15) then "darkblue" else "purple")))
(Cell 3 -1 (if (< charge 8) then "skyblue" else (if (< charge 15) then "darkblue" else "purple")))
))

(object LightningRod (: active Number) (list
(Cell 0 0 (if (> active 0) then "gold" else "silver"))
(Cell 0 1 (if (> active 0) then "gold" else "silver"))
(Cell 0 2 (if (> active 0) then "gold" else "silver"))
(Cell 1 2 (if (> active 0) then "gold" else "silver"))
))

(: time Number)
(= time (initnext 0 (+ (prev time) 1)))

(: cloud StormCloud)
(= cloud (initnext (StormCloud 0 (Position 2 1)) (updateObj cloud "charge" (if (== (% time 3) 0) then (+ (.. cloud charge) 1) else (.. cloud charge)))))

; gets inactive after 20 steps of remaining active
(: rod LightningRod)
(= rod (initnext (LightningRod 0 (Position 4 6)) (updateObj rod "active" (if (== (.. rod active) 0) then 0 else (- (.. rod active) 1)))))

(on left (= cloud (moveLeftNoCollision cloud)))
(on right (= cloud (moveRightNoCollision cloud)))

; Click on cloud to discharge it
(on (clicked cloud) (let
(= charge (.. cloud charge))
(if (>= charge 15) then
(= rod (updateObj rod "active" (+ (.. rod active) 20)))
else true)
(= cloud (updateObj cloud "charge" 0))
))
)
