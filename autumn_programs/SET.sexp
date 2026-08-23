(program
(= GRID_SIZE 20)

(= pick_color (fn (local_shade local_color) (if (== local_shade 1) then (if (== local_color 1) then "coral" else (if (== local_color 2) then "blue" else (if (== local_color 3) then "seagreen" else "black")))
else (if (== local_shade 2) then (if (== local_color 1) then "orangered" else (if (== local_color 2) then "darkblue" else (if (== local_color 3) then "darkgreen" else "black")))
else (if (== local_shade 3) then (if (== local_color 1) then "red" else (if (== local_color 2) then "lightblue" else (if (== local_color 3) then "lightgreen" else "black")))
else "black")))))


(object Card (: selected Bool) (: num Number) (: color Number) (: shade Number) (list
(Cell -1 -2 (if selected then "gold" else "white")) (Cell -1 -1 (if selected then "gold" else "white")) (Cell -1 0 (if selected then "gold" else "white")) (Cell -1 1 (if selected then "gold" else "white")) (Cell -1 2 (if selected then "gold" else "white"))
(Cell 0 -2 (if selected then "gold" else "white")) (Cell 0 -1 (if (>= num 1) then (pick_color shade color) else "black")) (Cell 0 0 (if (>= num 2) then (pick_color shade color) else "black")) (Cell 0 1 (if (== num 3) then (pick_color shade color) else "black")) (Cell 0 2 (if selected then "gold" else "white"))
(Cell 1 -2 (if selected then "gold" else "white")) (Cell 1 -1 (if selected then "gold" else "white")) (Cell 1 0 (if selected then "gold" else "white")) (Cell 1 1 (if selected then "gold" else "white")) (Cell 1 2 (if selected then "gold" else "white"))))

(: positions (List Position))
(= positions (list (Position 5 3) (Position 5 9) (Position 5 15) (Position 9 3) (Position 9 9) (Position 9 15) (Position 13 3) (Position 13 9) (Position 13 15)))

(: all_cards (List Card))
(= all_cards (list (Card false 3 1 1 (Position -10 0)) (Card false 3 2 1 (Position -10 0)) (Card false 3 3 1 (Position -10 0)) ; (Card false 2 1 1 (Position -10 0)) (Card false 2 2 1 (Position -10 0)) (Card false 2 3 1 (Position -10 0)) (Card false 1 1 1 (Position -10 0)) (Card false 1 2 1 (Position -10 0)) (Card false 1 3 1 (Position -10 0))
))

; (= random_props_all (fn (cards_local) (let
; (= proposed_card (Card false (uniformChoice (list 1 2 3)) (uniformChoice (list 1 2 3)) 1 (Position -10 0)))
; (if (any (--> c (& (== (.. c num) (.. proposed_card num)) (== (.. c shade) (.. proposed_card shade)))) cards) then (random_props_all cards_local) else proposed_card)
; )))

(= random_card (fn (cards) (let
; (= proposed_card_ (random_props_all cards))
(= proposed_card_ (uniformChoice all_cards))
(= pos (head (removeObj positions (--> pos (any (--> c (== pos (.. c origin))) cards)))))
(updateObj proposed_card_ "origin" pos)
)))

(: cards (List Card))
(= cards (initnext (list ) (prev cards)))

(on (< (length cards) 9) (= cards (addObj (prev cards) (random_card (prev cards)))))

(= remove_cards (fn (selected_cards) (& (& (== (% (sum (map (--> c (.. c num)) selected_cards)) 3) 0) (== (% (sum (map (--> c (.. c shade)) selected_cards)) 3) 0)) (== (% (sum (map (--> c (.. c color)) selected_cards)) 3) 0))))

(on (== (length (filter (--> card (.. card selected)) cards)) 3) (let
(= selected_cards_ (filter (--> card (.. card selected)) cards))
(if (remove_cards selected_cards_) then
(let
(= cards (removeObj cards (--> card (.. card selected))))
)
else (= cards (updateObj cards (--> card (updateObj card "selected" false)))))))

; select the clicked card
(on (clicked cards) (= cards (updateObj cards (--> card (updateObj card "selected" true)) (--> card (clicked card)))))
)
