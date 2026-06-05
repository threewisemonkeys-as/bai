## 1. Objects

### `Ant`
- Defined as `(object Ant (Cell 0 0 "gray"))`.
- Visual/physical footprint: a single gray cell at local offset `(0, 0)` from the ant’s origin.
- No custom fields are declared.
- Its position is controlled by its built-in origin, supplied when constructing an ant, e.g. `(Ant (Position 5 5))`.

### `Food`
- Defined as `(object Food (Cell 0 0 "red"))`.
- Visual/physical footprint: a single red cell at local offset `(0, 0)` from the food’s origin.
- No custom fields are declared.
- Its position is controlled by its built-in origin, supplied when constructing food, e.g. `(Food pos)`.

### Other state
- `GRID_SIZE` is set to `16`.
- It is used only when generating random food positions: `(randomPositions GRID_SIZE 2)`.

## 2. Initial state

### Ants
The variable `ants` is a list of `Ant` objects:

```lisp
(= ants
  (initnext
    (list (Ant (Position 5 5)) (Ant (Position 1 14)))
    (prev ants)))
```

Initial ants:
- One gray ant at position `(5, 5)`.
- One gray ant at position `(1, 14)`.

The default next value is `(prev ants)`, so ants persist from frame to frame unless changed by an `on` handler.

### Foods
The variable `foods` is a list of `Food` objects:

```lisp
(= foods (initnext (list) (prev foods)))
```

Initial foods:
- The food list starts empty.

The default next value is `(prev foods)`, so foods persist unless removed or added by handlers.

## 3. Dynamics

### Food removal every tick

```lisp
(on true
  (= foods
    (filter
      (--> obj (! (intersects obj (prev ants))))
      (prev foods))))
```

This runs every tick because the condition is `true`.

Effect:
- Start from the previous frame’s food list, `(prev foods)`.
- Keep only foods that do **not** intersect any ant from the previous frame, `(prev ants)`.
- Since both ants and foods are single-cell objects, a food is removed when its cell overlaps an ant’s cell.

Important detail:
- The removal test uses `prev ants`, so it checks where ants were in the previous state, not necessarily after the current movement update.

### Ant movement every tick

```lisp
(on true
  (= ants
    (updateObj
      (prev ants)
      (--> obj
        (move obj
          (unitVector obj (closest obj foods)))))))
```

This also runs every tick.

Effect:
- Each ant from `(prev ants)` is updated.
- For each ant `obj`:
  1. Find the closest food to that ant using `(closest obj foods)`.
  2. Compute a one-step direction toward that food using `(unitVector obj ...)`.
  3. Move the ant by that direction using `(move obj ...)`.

Movement behavior:
- Ants move one grid step per tick toward their closest food.
- If the target food differs in both x and y, `unitVector` moves horizontally first according to the standard-library behavior.
- If there are no foods, `closest` returns the ant itself, so the direction is zero and the ant does not move.

There is no collision avoidance in this movement: the code uses `move`, not `moveNoCollision`.

## 4. Player actions

The program responds to one input event:

### Global click

```lisp
(on clicked
  (= foods
    (addObj foods
      (map
        (--> pos (Food pos))
        (randomPositions GRID_SIZE 2)))))
```

When `clicked` is true:
- Generate 2 random positions using `(randomPositions GRID_SIZE 2)`.
- Convert each position into a `Food` object with `(Food pos)`.
- Add those new food objects to the existing `foods` list.

The click is not tied to clicking a particular ant or food object; the handler uses the global `clicked` condition.

No arrow keys, dragging, object-specific clicking, or other inputs are handled.

## 5. Edge cases / boundaries

- `GRID_SIZE = 16` is only passed to `randomPositions`; the program itself does not explicitly clamp ant movement to the grid.
- Ants use plain `(move ...)`, so there is no explicit bounds check, wall check, or collision check.
- Ants can potentially overlap each other; there is no rule preventing this.
- Food spawning has no explicit check against ants, other foods, or duplicate positions in the game code.
- Foods that overlap ants are removed by the every-tick filter.
- If an ant moves onto a food during a tick, that food is removed on a subsequent food-removal update because removal checks intersections against `prev ants`.
- If the food list is empty, ants remain stationary because their closest target is effectively themselves.
- There is no modular arithmetic, wrapping, health value, gas/energy limit, or other clamped resource.

## 6. Win/lose / progression

There is no explicit win condition, lose condition, score, reward, level transition, or terminal state.

The implicit progression loop is:
1. The player clicks to create red food at random positions.
2. Gray ants move one step per tick toward the closest food.
3. When an ant reaches/overlaps food, that food is removed.
4. The simulation continues indefinitely.
