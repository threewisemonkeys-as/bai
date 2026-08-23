# The Autumn Language

Autumn is a Lisp-like language with S-expressions. All code is wrapped in parentheses, with the first element being the operator or function name. The language is designed for game programming with built-in support for physics and collision detection.

## Formal Grammar

Below is a formal presentation of the grammar for both **expressions** (`Expr`) and **statements** (`Stmt`), integrating the relevant token types (e.g., $\texttt{IDENTIFIER}$, $\texttt{PLUS}$, $\texttt{MINUS}$, $\texttt{STRING}$, etc.). In this grammar:

- $\textit{Expr}$ denotes an expression node.
- $\textit{Stmt}$ denotes a statement node.
- $\textit{Token}[X]$ indicates a token whose $\texttt{TokenType}$ is $X$.
- $\textit{T*}$ indicates a list of elements of type $T$.
- Braces $( \dots )$ enclose parameter lists for each construct.

```text
Expressions:

  Assign       ::= "=" (name : Token[IDENTIFIER]) (value : Expr)

  Binary       ::= op : { PLUS, MINUS, STAR, SLASH, MODULO } (left : Expr) (right : Expr)

  Call         ::= (callee : Expr) (arguments : Expr)* | ((callee))

  Grouping     ::= (expression : Expr)

  Literal      ::= Token[NUMBER | STRING | TRUE | FALSE | NIL]

  Logical      ::= (op : { AND, OR }) (left : Expr) (right : Expr)

  Get          ::= ".." (object : Expr) (name : Token[IDENTIFIER])

  Unary        ::= (op : { BANG, MINUS }) (right : Expr)

  Lambda       ::= "-->" (params : Token[IDENTIFIER]*) (right : Expr)

  Variable     ::= (name : Token[IDENTIFIER])

  TypeExpr ::= (name : Token[IDENTIFIER])

  TypeDecl     ::= ":" (name : Token[IDENTIFIER]) (typeexpr : TypeExpr)

  ListTypeExpr ::= "List" (typeexpr : TypeExpr)

  ListVarExpr  ::= "list" (varExprs : Expr)*

  IfExpr       ::= "if" (condition : Expr) "then" (elseBranch : Expr)

  Let          ::= "Let" (exprs : Expr)*

  InitNext     ::= "initnext" (initializer : Expr) (nextExpr : Expr)


Statements:

  Object       ::= "object" (name : Token[IDENTIFIER]) (fields : TypeExpr*) (Cell : Expr)*

  Expression   ::= (expression : Expr)

  OnStmt       ::= "on" (condition : Expr) (expr : Expr)
```

We provide the script in [tools/generate_ast.cpp](https://github.com/BasisResearch/Autumn.cpp/blob/main/tools/generate_ast.cpp) to generate and modify this grammar quickly.

### Token Types and Usage

- **Single-character tokens**: 
  $\texttt{LEFT\\_PAREN}$ $($, 
  $\texttt{RIGHT\\_PAREN}$ $)$,
  $\texttt{LEFT\\_BRACE}$ $\\{$,
  $\texttt{RIGHT\\_BRACE}$ $\\}$,
  $\texttt{COMMA}$ $\texttt{,}$, 
  $\texttt{DOT}$ $.$, 
  $\texttt{MINUS}$ $-$,
  $\texttt{PLUS}$ $+$, 
  $\texttt{SLASH}$ $/$,
  $\texttt{STAR}$ $*$,
  $\texttt{COLON}$ $:$,
  $\texttt{MODULO}$ $\%$.

- **Multi-character tokens**:
  $\texttt{BANG\\_EQUAL}$ $!=$,
  $\texttt{EQUAL\\_EQUAL}$ $==$,
  $\texttt{GREATER\\_EQUAL}$ $\ge$,
  $\texttt{LESS\\_EQUAL}$ $\le$,
  $\texttt{MAPTO}$ $\rightarrow$.

- **Literals**: 
  $\texttt{IDENTIFIER}$ (variable names),
  $\texttt{STRING}$, 
  $\texttt{NUMBER}$,
  $\texttt{TRUE}$, 
  $\texttt{FALSE}$, 
  $\texttt{NIL}$.

- **Keywords**:
  $\texttt{PROGRAM}$ $\text{program}$, $\texttt{AND}$ (either & or $\texttt{and}$, $\texttt{OBJECT}$ $\texttt{object}$, $\texttt{ELSE}$ $\texttt{else}$, $\texttt{FUN}$ $\texttt{fn}$, $\texttt{IF}$ $\texttt{if}$, $\texttt{THEN}$ $\texttt{then}$, $\texttt{ON}$ $\texttt{on}$, $\texttt{OR}$ (either $\texttt{or}$ or |), $\texttt{LET}$ $\texttt{let}$, $\texttt{INITNEXT}$.

For modifying Token (adding token types, etc. please look at [TokenType header file](https://github.com/BasisResearch/Autumn.cpp/blob/main/include/lexer/TokenType.hpp)

### Examples
Below is a compact set of examples illustrating each expression type and statement type from the grammar. Every snippet is wrapped in a (program ...) form for completeness, but in real usage, you would mix and match these constructs as needed.

Expression Examples

1. Assign
```sexp
(program
  (= x 42) ; Assigning the literal 42 to variable x
)
```

- Explanation: The expression `(= x 42)` assigns the value 42 to a variable named x.

2. Binary
```sexp
(program
  (on true
    (let
      (= sum (+ 2 3)) ; Binary expr: (+ 2 3)
      (print sum)
      true
    )
  )
)
```
- Explanation: `(+ 2 3)` is a binary expression using the + operator.

3. Call
```sexp
(program
  (: foo (fn (a b) (* a b))) ; A simple lambda function for multiplication
  (= result (foo 10 5))      ; Call expr: (foo 10 5)
)
```
- Explanation: `(foo 10 5)` calls the function stored in foo with arguments 10 and 5.

4. Get
```sexp
(program
  (object Player (: health Number) (Cell 0 0 "blue"))
  (: player Player)
  (= player (Player (Position 5 5)))
  
  (= currentHealth (.. player health)) ; Get expr: (.. player health)
)
```
- Explanation: `(.. player health)` accesses the health field from the object player.

5. Grouping
```sexp
(program
  (= grouped ( ( + 1 (* 2 3) ) )) ; Grouping expr around (+ 1 (* 2 3))
)
```
- Explanation: Grouping is demonstrated by wrapping expressions in parentheses (though in Autumn’s Lisp-like syntax, parentheses are already used heavily). The outer parentheses around `(+ 1 (* 2 3))` illustrate grouping.

6. Literal
```sexp
(program
  (= str "hello world") ; Literal string
  (= num 123)           ; Literal number
  (= boolValue true)    ; Literal boolean
)
```
- Explanation: Literal values can be strings, numbers, booleans, or nil.

7. Logical
```sexp
(program
  (= x 3)
  (= y 10)
  (= check (& (> x 2) (== y 10))) ; Logical expr: (& (> x 2) (== y 10))
)
```
- Explanation: This checks the conjunction (&) of two comparisons: x > 2 and y == 10.

8. Set
```sexp
(program
  (object Agent (: speed Number) (Cell 0 0 "red"))
  (: myAgent Agent)
  (= myAgent (Agent (Position 4 4)))
  
  (= (.. myAgent speed) 99) ; Set expr: updates the 'speed' field of 'myAgent'
)
```
- Explanation: (= (.. myAgent speed) 99) modifies the speed field of myAgent.

9. Unary
```sexp
(program
  (: value Number)
  (= value 5)
  
  (= negValue (- value)) ; Unary minus
  (= notAlive (! alive)) ; Unary bang (if alive is a bool)
)
```
- Explanation:
  - (- value) is the unary negation of value.
  - (! alive) would be a unary logical NOT, assuming alive is a boolean variable.

10. Lambda
```sexp
(program
  (= adder (fn (a b) (+ a b))) ; A lambda function taking two params a, b
  (= result (adder 7 8))       ; Calls the lambda -> 15
)
```
- Explanation: (fn (a b) (+ a b)) is a lambda expression assigned to adder.

11. Variable
```sexp
(program
  (: x Number)  ; Declare variable x
  (= x 10)      ; Variable usage
  (on true (print x)) ; x is used directly
)
```
- Explanation: x is a simple variable referred to by name.

12. TypeDecl
```sexp
(program
  (: MyCustomType Bool)
)
```
- Explanation: Declares a new type MyCustomType as a list of some type T (In this case, Bool). (Requires having typevariable Bool declared somewhere.)

13. ListTypeExpr
```sexp
(program
  (: MyType (List Bool))  ; Interpreted as "List of Number"
)

- Explanation: (listtypeexpr Number) means a type expression for a list of Number.

14. ListVarExpr
```sexp
(program
  (= items (list 1 2 3 4)) ; Creates a list of literal values
)
```
- Explanation: (list 1 2 3 4) is a typical “ListVarExpr” usage, returning a list expression of values.

15. IfExpr
```sexp
(program
  (= x 5)
  (= result (if (> x 3) then "x is large" else "x is small")) ; IfExpr
)
```
- Explanation: (if (> x 3) ... ...) is the conditional expression returning one of two branches.
16. Let
```sexp
(program
  (on true
    (let
      (= a 1)
      (= b 2)
      (print (+ a b)) ; -> prints 3
      true
    )
  )
)
```
- Explanation: A let expression creates local bindings (a, b) and then executes sub-expressions, and return value of the final expression evaluation

17. InitNext
```sexp
(program
  (: counter Number)
  (= counter (initnext 0 ( + (prev counter) 1 )))
)
```
- Explanation: On initialization, counter is 0. Each subsequent timestep, it sets counter to (prev counter) + 1.

#### Statement Examples
1. Object
```sexp
(program
  (object Player (: health Number) (Cell 0 0 "blue"))
  (object Enemy (: damage Number) (Cell 0 0 "red"))
)
```
- Explanation: Defines two object types, Player and Enemy, each with fields and a base Cell.

2. Expression (Statement)
```sexp
(program
  (+ 1 2)
  (print "Hello from an expression stmt")
)
```
- Explanation: An Expression statement simply evaluates an expression where a statement is allowed (e.g., top-level or inside a block).

3. OnStmt
```sexp
(program
  (= x 0)
  (on (< x 5)
    (= x (+ x 1))
  )
)
```
- Explanation: onStmt triggers the expression body whenever the condition (< x 5) evaluates to true. Here, it increments x until it reaches 

# Documentation for All Functions (Including the native ones.)

Below is a comprehensive set of documentation for every function **defined** in the [Autumn Stdlib](https://github.com/BasisResearch/Autumn.cpp/blob/main/autumnstdlib/stdlib.sexp), as well as brief documentation for functions/calls **referenced** but not defined within the snippet. 

## Defined Functions

### 1. `(move obj dir)`
- **Description**: Moves the given object `obj` by the vector `dir`.
- **Parameters**:
  - `obj`: The object to move (must have an `origin` property).
  - `dir`: A `Position` object specifying the displacement in x and y.
- **Returns**: A new version of `obj` whose `origin` has been updated to `origin + dir`.

---

### 2. `(moveRight obj)`
- **Description**: Moves the given object `obj` exactly one unit to the right (x + 1).
- **Parameters**:
  - `obj`: The object to move.
- **Returns**: The updated object with the x-coordinate of `origin` incremented by 1.

---

### 3. `(moveLeft obj)`
- **Description**: Moves the given object `obj` exactly one unit to the left (x - 1).
- **Parameters**:
  - `obj`: The object to move.
- **Returns**: The updated object with the x-coordinate of `origin` decremented by 1.

---

### 4. `(moveUp obj)`
- **Description**: Moves the given object `obj` exactly one unit up (y - 1).
- **Parameters**:
  - `obj`: The object to move.
- **Returns**: The updated object with the y-coordinate of `origin` decremented by 1.

---

### 5. `(moveDown obj)`
- **Description**: Moves the given object `obj` exactly one unit down (y + 1).
- **Parameters**:
  - `obj`: The object to move.
- **Returns**: The updated object with the y-coordinate of `origin` incremented by 1.

---

### 6. `(moveNoCollision obj x y)`
- **Description**: Moves `obj` by `(x, y)` **only if** the resulting position is free, excluding any collision checks with the original `obj` itself.
- **Parameters**:
  - `obj`: The object to move.
  - `x`: The change in the x-direction.
  - `y`: The change in the y-direction.
- **Returns**: A new version of `obj` if the position is free, otherwise returns the original `obj` without movement.

---

### 7. `(moveCanCollision obj x y obj2)`
- **Description**: Moves `obj` by `(x, y)` if the resulting position is free, excluding collisions with `obj` and `obj2`.
- **Parameters**:
  - `obj`: The primary object to move.
  - `x`: The change in the x-direction.
  - `y`: The change in the y-direction.
  - `obj2`: A second object to exclude from collision checks.
- **Returns**: The updated object if the position is free, otherwise the original object.

---

### 8. `(rotateNoCollision obj)`
- **Description**: Rotates `obj` (using an external function `rotate`) only if the rotated object:
  1. Remains within bounds.
  2. Does not collide with anything except the original `obj`.
- **Parameters**:
  - `obj`: The object to rotate.
- **Returns**: The rotated object if rotation is valid, otherwise the original object.

---

### 9. `(movePos pos dir)`
- **Description**: Computes a new `Position` by adding `dir` to `pos`.
- **Parameters**:
  - `pos`: A `Position` object.
  - `dir`: A `Position` specifying how much to move in x and y.
- **Returns**: A new `Position` with updated coordinates.

---

### 10. `(moveRightPos pos)`
- **Description**: Returns a new `Position` moved one unit to the right from `pos`.
- **Parameters**:
  - `pos`: The initial `Position`.
- **Returns**: `Position(pos.x + 1, pos.y)`.

---

### 11. `(moveLeftPos pos)`
- **Description**: Returns a new `Position` moved one unit to the left.
- **Parameters**:
  - `pos`: The initial `Position`.
- **Returns**: `Position(pos.x - 1, pos.y)`.

---

### 12. `(moveUpPos pos)`
- **Description**: Returns a new `Position` moved one unit up.
- **Parameters**:
  - `pos`: The initial `Position`.
- **Returns**: `Position(pos.x, pos.y - 1)`.

---

### 13. `(moveDownPos pos)`
- **Description**: Returns a new `Position` moved one unit down.
- **Parameters**:
  - `pos`: The initial `Position`.
- **Returns**: `Position(pos.x, pos.y + 1)`.

---

### 14. `(abs x)`
- **Description**: Returns the absolute value of `x`.
- **Parameters**:
  - `x`: A number.
- **Returns**: Absolute value of `x`.

---

### 15. `(sign x)`
- **Description**: Determines the sign of `x`.
- **Parameters**:
  - `x`: A number (integer, as Autumn does not have float yet).
- **Returns**: 
  - `-1` if `x < 0`
  - `0` if `x == 0`
  - `1` if `x > 0`

---

### 16. `(vcat ...)`
- **Description**: An alias for `concat`. In this code, `vcat` is set to `concat`.
- **Parameters**: Varies depending on usage, typically a list of lists to concatenate.
- **Returns**: A single concatenated list.

---

### 17. `(deltaPos pos1 pos2)`
- **Description**: Computes the difference in coordinates between `pos2` and `pos1`.
- **Parameters**:
  - `pos1`: The first `Position`.
  - `pos2`: The second `Position`.
- **Returns**: A new `Position` = `(pos2 - pos1)`, i.e. `(pos2.x - pos1.x, pos2.y - pos1.y)`.

---

### 18. `(rect pos1 pos2)`
- **Description**: Generates a list of `Position` objects for the rectangular region spanning from `(pos1.x, pos1.y)` to `(pos2.x, pos2.y)`, **excluding** the upper bound if the language follows typical `range` conventions.
- **Parameters**:
  - `pos1`: The top-left `Position`.
  - `pos2`: The bottom-right `Position`.
- **Returns**: A flattened list of `Position` objects representing each coordinate in the rectangle.

---

### 19. `(displacement pos1 pos2)`
- **Description**: An alias for `deltaPos`. Computes the displacement between two positions.
- **Parameters**:
  - `pos1`: The first `Position`.
  - `pos2`: The second `Position`.
- **Returns**: A `Position` representing `(pos2 - pos1)`.

---

### 20. `(deltaElem e1 e2)`
- **Description**: Computes the coordinate difference between two elements by comparing their `position` property.
- **Parameters**:
  - `e1`: An element with a `position`.
  - `e2`: Another element with a `position`.
- **Returns**: The result of `deltaPos(e1.position, e2.position)`.

---

### 21. `(deltaObj obj1 obj2)`
- **Description**: Computes the difference between the `origin` of two objects.
- **Parameters**:
  - `obj1`: First object with an `origin`.
  - `obj2`: Second object with an `origin`.
- **Returns**: A `Position` representing `(obj2.origin - obj1.origin)`.

---

### 22. `(adjacentElem e1 e2)`
- **Description**: Checks if two elements (`e1` and `e2`) are adjacent on a grid, meaning the sum of the absolute difference in x plus y coordinates is `1`.
- **Parameters**:
  - `e1`: An element with a `position`.
  - `e2`: Another element with a `position`.
- **Returns**: Boolean `true` if they are horizontally or vertically adjacent, otherwise `false`.

---

### 23. `(adjacentPoss p1 p2 unitSize)`
- **Description**: Checks if two positions are within `unitSize` adjacency in a grid sense. Here, adjacency is defined by `abs(delta.x) + abs(delta.y) <= unitSize`.
- **Parameters**:
  - `p1`: A `Position`.
  - `p2`: Another `Position`.
  - `unitSize`: The threshold for adjacency.
- **Returns**: Boolean `true` if within `unitSize`, else `false`.

---

### 24. `(adjacentTwoObjs obj1 obj2 unitSize)`
- **Description**: Checks if `obj1` and `obj2` are adjacent, using the same logic as `adjacentPoss` on their `origin`.
- **Parameters**:
  - `obj1`: First object.
  - `obj2`: Second object.
  - `unitSize`: Adjacency threshold.
- **Returns**: Boolean `true` if within `unitSize`, else `false`.

---

### 25. `(adjacentPossDiag p1 p2)`
- **Description**: Checks if two positions are adjacent **including diagonals**, i.e. `abs(delta.x) <= 1` and `abs(delta.y) <= 1`.
- **Parameters**:
  - `p1`: A `Position`.
  - `p2`: A `Position`.
- **Returns**: Boolean `true` if they are adjacent in any of the 8 surrounding squares, else `false`.

---

### 26. `(adjacentTwoObjsDiag obj1 obj2)`
- **Description**: Uses `adjacentPossDiag` to check diagonal adjacency between two objects.
- **Parameters**:
  - `obj1`: First object.
  - `obj2`: Second object.
- **Returns**: Boolean `true` if diagonally (or directly) adjacent, else `false`.

---

### 27. `(adjacentObjs obj1 unitSize)`
- **Description**: Finds all objects in the global `allObjs()` list that are adjacent to `obj1` (or each object in `obj1`, if `obj1` is a list) given `unitSize`.
- **Parameters**:
  - `obj1`: The object or list of objects.
  - `unitSize`: The threshold for adjacency.
- **Returns**: A list of objects from `allObjs()` that are adjacent.

---

### 28. `(adjacentObjsDiag obj)`
- **Description**: Finds all objects in the global `allObjs()` list that are adjacent to `obj` (or each object in `obj`, if `obj` is a list) **including diagonals**.
- **Parameters**:
  - `obj`: The object or list of objects.
- **Returns**: A list of objects considered adjacent diagonally or orthogonally.

---

### 29. `(adj obj objs unitSize)`
- **Description**: Checks if `obj` is adjacent to **any** object in `objs` given `unitSize`.
- **Parameters**:
  - `obj`: The object to check.
  - `objs`: A list of objects.
  - `unitSize`: Adjacency threshold.
- **Returns**: Boolean `true` if adjacency is found with at least one in `objs`, else `false`.

---

### 30. `(objClicked objs)`
- **Description**: Filters a list of objects, returning only those that have been "clicked" (based on an external `clicked` function).
- **Parameters**:
  - `objs`: A list of objects to check.
- **Returns**: A list of objects for which `clicked(obj) == true`.

---

### 31. `(sqdist pos1 pos2)`
- **Description**: Computes the squared distance between two positions.
- **Parameters**:
  - `pos1`: A `Position`.
  - `pos2`: Another `Position`.
- **Returns**: `(delta.x * delta.x + delta.y * delta.y)`, where `delta = pos2 - pos1`.

---

### 32. `(unitVector obj target)`
- **Description**: Computes a "unit vector" (in grid sense) from `obj`'s origin to `target`'s origin. If both x and y need to move and are both nonzero, it defaults to moving only along x first (i.e. `(sign_x, 0)`).
- **Parameters**:
  - `obj`: The "source" object.
  - `target`: The "destination" object.
- **Returns**: A `Position` `(sign_x, sign_y)` representing direction, with a special rule if both are nonzero.

---

### 33. `(unitVectorObjPos obj pos)`
- **Description**: Similar to `unitVector`, but the target is a bare position `pos` instead of an object.
- **Parameters**:
  - `obj`: The "source" object (has an `origin`).
  - `pos`: A direct `Position`.
- **Returns**: A `Position` `(sign_x, sign_y)` using the same logic as `unitVector`.

---

### 34. `(max a b)`
- **Description**: Returns the maximum of `a` and `b`.
- **Parameters**:
  - `a`: A number.
  - `b`: A number.
- **Returns**: The larger of `a` and `b`.

---

### 35. `(min a b)`
- **Description**: Returns the minimum of `a` and `b`.
- **Parameters**:
  - `a`: A number.
  - `b`: A number.
- **Returns**: The smaller of `a` and `b`.

---

### 36. `(closest obj listObjs)`
- **Description**: From a list of objects, finds the one with the smallest squared distance to `obj`.
- **Parameters**:
  - `obj`: The reference object.
  - `listObjs`: A list of objects.
- **Returns**: The single closest object in `listObjs`. Returns `obj` if `listObjs` is empty.

---

### 37. `(closestPos obj listPoss)`
- **Description**: From a list of positions, finds the position closest (in squared distance) to `obj`'s origin.
- **Parameters**:
  - `obj`: The reference object (with an `origin`).
  - `listPoss`: A list of `Position`s.
- **Returns**: The closest `Position` by squared distance. Returns `obj.origin` if `listPoss` is empty.

---

### 38. `(renderValue obj)`
- **Description**: Produces a list of renderable elements for `obj`. If `obj` is a list, concatenates all of their renderValues. If a single object, returns `(.. obj render)`.
- **Parameters**:
  - `obj`: An object or list of objects that can be rendered.
- **Returns**: A list of rendered elements (or concatenated lists if `obj` is a list).

---

### 39. `(intersectsElems elems1 elems2)`
- **Description**: Checks if any element in `elems1` shares the same position as any element in `elems2`.
- **Parameters**:
  - `elems1`: A list of elements (each with a `position`).
  - `elems2`: Another list of elements.
- **Returns**: Boolean `true` if at least one position overlaps, otherwise `false`.

---

### 40. `(intersectsPosElems pos elems)`
- **Description**: Checks if a given `pos` coincides with the `position` of any element in `elems`.
- **Parameters**:
  - `pos`: A `Position`.
  - `elems`: A list of elements (each with a `position`).
- **Returns**: Boolean `true` if any element matches `pos`, else `false`.

---

### 41. `(intersectsPosPoss pos poss)`
- **Description**: Checks if `pos` is contained in the list of positions `poss`.
- **Parameters**:
  - `pos`: A `Position`.
  - `poss`: A list of `Position` objects.
- **Returns**: Boolean `true` if `pos` is in `poss`, else `false`.

---

### 42. `(intersects obj1 obj2)`
- **Description**: Determines if `obj1` and `obj2` have any intersecting elements by comparing their rendered values.
- **Parameters**:
  - `obj1`: An object (or list of objects).
  - `obj2`: Another object (or list of objects).
- **Returns**: Boolean `true` if their rendered elements overlap.

---

### 43. `(isFree obj)`
- **Description**: Checks if all rendered elements of `obj` occupy free positions (i.e., no collisions).
- **Parameters**:
  - `obj`: An object or list of objects to check for collisions.
- **Returns**: Boolean `true` if none of the positions are blocked, else `false`.

---

### 44. `(in e l)`
- **Description**: Checks if element `e` is in list `l`.
- **Parameters**:
  - `e`: Element to find.
  - `l`: A list.
- **Returns**: Boolean `true` if `e` is found in `l`, otherwise `false`.

---

### 45. `(isFreeExcept obj prev_obj)`
- **Description**: Checks if all rendered elements of `obj` are free, ignoring collisions with `prev_obj`’s rendered elements.
- **Parameters**:
  - `obj`: The object to check.
  - `prev_obj`: The object whose elements we ignore in collision checks.
- **Returns**: Boolean `true` if none of the new positions (beyond those occupied by `prev_obj`) are blocked, otherwise `false`.

---

### 46. `(isFreePosExceptObj pos obj)`
- **Description**: Checks if a single position `pos` is free or is intersecting only with `obj`.
- **Parameters**:
  - `pos`: A `Position`.
  - `obj`: The object to ignore in collision checks.
- **Returns**: Boolean `true` if `pos` is free or only occupied by `obj`.

---

### 47. `(isFreeRangeExceptObj start end obj)`
- **Description**: Checks positions in the range `[start, end)` along the x-axis (y=0 in the code snippet) to see if they are free, excluding the `obj`'s positions.
- **Parameters**:
  - `start`: Start of the x-range.
  - `end`: End of the x-range (exclusive if using typical `range`).
  - `obj`: The object to exclude from collisions.
- **Returns**: Boolean `true` if all those positions are free except for where `obj` is, else `false`.

---

### 48. `(moveLeftNoCollision obj)`
- **Description**: Moves `obj` one unit to the left if it is within bounds and free (excluding the object’s own current space).
- **Parameters**:
  - `obj`: The object to move.
- **Returns**: The updated `obj` if valid, otherwise returns `obj` unchanged.

---

### 49. `(moveRightNoCollision obj)`
- **Description**: Moves `obj` one unit to the right if it is within bounds and free.
- **Parameters**:
  - `obj`: The object to move.
- **Returns**: The updated `obj` if valid, otherwise `obj`.

---

### 50. `(moveUpNoCollision obj)`
- **Description**: Moves `obj` one unit up if it is within bounds and free.
- **Parameters**:
  - `obj`: The object to move.
- **Returns**: The updated `obj` if valid, otherwise `obj`.

---

### 51. `(moveDownNoCollision obj)`
- **Description**: Moves `obj` one unit down if it is within bounds and free.
- **Parameters**:
  - `obj`: The object to move.
- **Returns**: The updated `obj` if valid, otherwise `obj`.

---

### 52. `(nextSolid obj)`
- **Description**: This is simply an alias set to `moveDownNoCollision`. Typically used in contexts where an object moves like a “solid” block.
- **Parameters**:
  - `obj`: The object to move down.
- **Returns**: The updated `obj` if valid, otherwise the original `obj`.

---

### 53. `(nextLiquidClosestHole obj holes)`
- **Description**: Finds the closest "hole" (position) among `holes` and attempts to move `obj` toward it using `nextLiquidMoveClosestHole`.
- **Parameters**:
  - `obj`: The object representing a liquid droplet (or similar).
  - `holes`: A list of positions representing available holes.
- **Returns**: The moved `obj` if a hole is found and the path is free, or the original `obj` otherwise.

---

### 54. `(nextLiquid obj)`
- **Description**: Simulates liquid movement:
  1. If `obj` can move down freely and isn’t at the bottom, it moves down.
  2. Else, it scans the next row for "holes" and attempts to move toward the closest one.
- **Parameters**:
  - `obj`: The liquid-like object to move.
- **Returns**: The updated `obj` after applying liquid movement logic.

---

### 55. `(nextLiquidMoveClosestHole obj closestHole)`
- **Description**: Moves `obj` toward `closestHole` by computing a direction via `unitVectorObjPos` and verifying free space. 
- **Parameters**:
  - `obj`: The object representing a liquid droplet.
  - `closestHole`: A `Position` representing the hole’s location.
- **Returns**: The updated `obj` if movement is valid, otherwise the original `obj`.

---

## Native functions and classes

Below are brief descriptions of native functions or constructs:

- **`updateObj(obj, propertyName, newValue)`**  
  Updates the object `obj` by setting its `propertyName` to `newValue`. Returns a new or updated version of `obj`.

- **`Position(x, y)`**  
  Represents a coordinate pair with fields `x` and `y`. Used extensively for movement and geometry.

- **`isFreePos(pos)`**  
  Checks if the position `pos` is free (i.e., not occupied or blocked) in the game/grid world.

- **`isWithinBounds(obj)`**  
  Checks if `obj` (or possibly its rendered elements) lies within the valid boundaries of the grid/board.

- **`rotate(obj)`**  
  Rotates `obj` in some manner (e.g., rotating the shape or positions). Used by `rotateNoCollision`.

- **`isList(value)`**  
  Tests whether `value` is recognized as a list in this language.

- **`allObjs()`**  
  Returns a list of all objects that exist in the current environment or game state.

- **`addObj oldList newObj`**
  Add the `newObj` to the `oldList`.


- **`removeObj oldList oldObj`**
  Remove the `oldObj` from the `oldList`.



- **`clicked(obj)`**  
  Checks if `obj` was clicked (mouse/touch event). Used by `objClicked`.

- **`map(function, list)`**  
  Applies `function` to each element of `list`, returning a new list of results.

- **`filter(function, list)`**  
  Retains only those elements from `list` for which `function(element)` returns `true`.

- **`any(function, list)`**  
  Returns `true` if `function(element)` is `true` for at least one element in `list`, otherwise `false`.

- **`foldl(function, initValue, list)`**  
  Left-fold (reduce): iterates through `list`, applying `function(acc, elem)`, carrying forward an accumulator.

- **`head(list)`**  
  Returns the first element of `list`.

- **`at(list, idx)`**  
  Returns the `idx`th element in the list.

- **`tail(list)`**  
  Returns the last element of `list`.


- **`concat(listOfLists)`**  
  Flattens or concatenates a list of lists into a single list.

- **`range(start, end)`**  
  Returns a list of integers from `start` up to but not including `end` (based on typical usage in the code).

- **`print(value)`**  
  Outputs `value` for debugging or logging purposes.

- **Logical Operators** (used in the code):
  - **`and`**, **`or`**, **`not`** (shown as `(! ...)`)  
    Standard logical operations for combining boolean expressions.
  - **`if (condition) then (expr1) else (expr2)`**  
    Standard conditional expression.
