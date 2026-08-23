(program
(= GRID_SIZE 12)

(object Guess (: col Number) (Cell 0 0 (if (== col 0) then "gray" else (if (== col 1) then "tan" else (if (== col 2) then "coral" else (if (== col 3) then "cyan" else (if (== col 4) then "steelblue" else (if (== col 5) then "mediumpurple" else "goldenrod"))))))))
(object Button (Cell 0 0 "maroon"))
(object Hint (: color String) (Cell 0 0 color))
(object CorrectAnswer (: show Bool) (: col Number) (Cell 0 0 (if (! show) then "silver" else (if (== col 1) then "tan" else (if (== col 2) then "coral" else (if (== col 3) then "cyan" else (if (== col 4) then "steelblue" else (if (== col 5) then "mediumpurple" else "goldenrod"))))))))

(: col_list (List Number))
(= col_list (list 1 2 3 4 5 6))

(= gen_answer (fn (col_list) (let
(= answer1 (uniformChoice col_list))
(= col_list (filter (--> o (!= o answer1)) col_list))
(= answer2 (uniformChoice col_list))
(= col_list (filter (--> o (!= o answer2)) col_list))
(= answer3 (uniformChoice col_list))
(= col_list (filter (--> o (!= o answer3)) col_list))
(= answer4 (uniformChoice col_list))
(list (CorrectAnswer true answer1 (Position 8 0)) (CorrectAnswer true answer2 (Position 9 0)) (CorrectAnswer true answer3 (Position 10 0)) (CorrectAnswer true answer4 (Position 11 0)))
)))

(: correct_answer (List CorrectAnswer))
(= correct_answer (initnext (gen_answer col_list) (prev correct_answer)))

(= is_correct (fn (guess) (== (map (--> o (.. o col)) guess) (map (--> o (.. o col)) correct_answer))))

(: enterButton Button)
(= enterButton (initnext (Button (Position 0 0)) (prev enterButton)))

(: curr_guess (List Guess))
(= curr_guess (list (Guess 0 (Position 3 2)) (Guess 0 (Position 4 2)) (Guess 0 (Position 5 2)) (Guess 0 (Position 6 2))))

(: prev_guesses (List Guess))
(= prev_guesses (initnext (list ) (prev prev_guesses)))

(: hints (List Hint))
(= hints (initnext (list ) (prev hints)))

(= create_hint (fn (guess) (let
(= correct_cols (map (--> o (.. o col)) correct_answer))
(= num_correct (length (filter (--> o (in (.. o col) correct_cols)) guess)))
(= num_correct_and_position (length (filter (--> o o) (arrayEqual (map (--> o (.. o col)) guess) correct_cols))))

(= hintColor (fn ()
(let
(= hc (if (> num_correct_and_position 0) then "green" else (if (> num_correct 0) then "yellow" else "grey")))
(= num_correct_and_position (- num_correct_and_position 1))
(= num_correct (- num_correct 1))
hc
)
))
(= positions (list (Position 8 3) (Position 9 3) (Position 8 4) (Position 9 4)))
(= pos1 (uniformChoice positions))
(= positions (removeObj positions pos1))
(= pos2 (uniformChoice positions))
(= positions (removeObj positions pos2))
(= pos3 (uniformChoice positions))
(= positions (removeObj positions pos3))
(= pos4 (uniformChoice positions))
; find random position for each hint
(list (Hint ((hintColor)) pos1) (Hint ((hintColor)) pos2) (Hint ((hintColor)) pos3) (Hint ((hintColor)) pos4))
)))

(on (clicked curr_guess) (= curr_guess
(updateObj
curr_guess
(--> obj (updateObj obj "col" (% (+ (.. obj col) 1) 7)))
(--> obj (== (.. obj origin) (Position (.. click x) (.. click y))))
)))

(on (& (clicked enterButton) (foldl (--> (acc, g) (& acc (> (.. g col) 0))) true curr_guess)) (let
; (print (map (--> o (.. o col)) correct_answer))
(= prev_guesses (concat (list prev_guesses curr_guess)))
(= hints (map (--> h (moveDown h)) hints))
(= hints (map (--> h (moveDown h)) hints))
(= prev_guesses (map (--> o (moveDown o)) prev_guesses))
(= prev_guesses (map (--> o (moveDown o)) prev_guesses))
(= curr_hint (create_hint curr_guess))
(= hints (vcat (list hints curr_hint)))
(if (is_correct curr_guess) then (= correct_answer (updateObj correct_answer (--> o (updateObj o "show" true)))) else true)
(= curr_guess (list (Guess 0 (Position 3 2)) (Guess 0 (Position 4 2)) (Guess 0 (Position 5 2)) (Guess 0 (Position 6 2))))
))
)

