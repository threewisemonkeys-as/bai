(program
(= GRID_SIZE 17)

(object Match (: selected Bool) (Cell 0 0 (if selected then "brown" else "maroon")))
(object RemoveButton (Cell 0 0 "red"))

(: row1_match (List Match))
(= row1_match (initnext (list (Match false (Position 8 4))) (prev row1_match)))

(: remove_button_1 RemoveButton)
(= remove_button_1 (initnext (RemoveButton (Position 16 4)) (prev remove_button_1)))

(on (clicked remove_button_1) (= row1_match (removeObj row1_match (--> x (.. x selected)))))
(on (clicked row1_match) (= row1_match (updateObj row1_match (--> x (updateObj x "selected" (! (.. x selected)))) (--> x (== (.. x origin) (Position (.. click x) (.. click y)))))))

(: row2_match (List Match))
(= row2_match (initnext (list (Match false (Position 6 7)) (Match false (Position 8 7)) (Match false (Position 10 7))) (prev row2_match)))

(: remove_button_2 RemoveButton)
(= remove_button_2 (initnext (RemoveButton (Position 16 7)) (prev remove_button_2)))

(on (clicked remove_button_2) (= row2_match (removeObj row2_match (--> x (.. x selected)))))
(on (clicked row2_match) (= row2_match (updateObj row2_match (--> x (updateObj x "selected" (! (.. x selected)))) (--> x (== (.. x origin) (Position (.. click x) (.. click y)))))))

(: row3_match (List Match))
(= row3_match (initnext (list (Match false (Position 4 10)) (Match false (Position 6 10)) (Match false (Position 8 10)) (Match false (Position 10 10)) (Match false (Position 12 10))) (prev row3_match)))

(: remove_button_3 RemoveButton)
(= remove_button_3 (initnext (RemoveButton (Position 16 10)) (prev remove_button_3)))

(on (clicked remove_button_3) (= row3_match (removeObj row3_match (--> x (.. x selected)))))
(on (clicked row3_match) (= row3_match (updateObj row3_match (--> x (updateObj x "selected" (! (.. x selected)))) (--> x (== (.. x origin) (Position (.. click x) (.. click y)))))))

(: row4_match (List Match))
(= row4_match (initnext (list (Match false (Position 2 13)) (Match false (Position 4 13)) (Match false (Position 6 13)) (Match false (Position 8 13)) (Match false (Position 10 13)) (Match false (Position 12 13)) (Match false (Position 14 13))) (prev row4_match)))

(: remove_button_4 RemoveButton)
(= remove_button_4 (initnext (RemoveButton (Position 16 13)) (prev remove_button_4)))

(on (clicked remove_button_4) (= row4_match (removeObj row4_match (--> x (.. x selected)))))
(on (clicked row4_match) (= row4_match (updateObj row4_match (--> x (updateObj x "selected" (! (.. x selected)))) (--> x (== (.. x origin) (Position (.. click x) (.. click y)))))))
)
