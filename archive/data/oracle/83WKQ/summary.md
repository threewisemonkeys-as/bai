## 1. Objects

### `Particle`
Defined as:

`(object Particle (Cell 0 0 "blue"))`

- A `Particle` has no custom fields declared by the program.
- It uses the built-in object position/origin field supplied when it is constructed, e.g. `(Particle (Position x y))`.
- Its visual representation is one blue cell:
  - `Cell 0 0 "blue"` means the particle renders as a blue square/cell at local offset `(0, 0)` from its origin.
- The particle’s `origin` controls where that blue cell appears on the grid.

### Global state
- `GRID_SIZE` is set to `16`, indicating a 16×16 grid.
- `particles` is declared as a list of `Particle` objects:

`(: particles (List Particle))`

This list is the main game state.

---

## 2. Initial state

The `particles` list is initialized with:

`(initnext (list) ...)`

So the game starts with:

- A 16×16 grid.
- No particles on the board.
- No score, timer, target, or other state variables.

Particles only appear after the player clicks.

---

## 3. Dynamics

### Per-timestep particle motion

The `particles` variable is defined with an `initnext` expression:

```lisp
(= particles
  (initnext
    (list)
    (updateObj
      (prev particles)
      (--> obj
        (updateObj
          obj
          "origin"
          (uniformChoice (adjPositions (.. obj origin))))))))
```

This means:

- On initialization, `particles` is the empty list.
- On later updates, the new `particles` list is computed from `(prev particles)`.
- Each existing particle is updated by changing its `"origin"` field.
- The new origin is chosen by:

`(uniformChoice (adjPositions (.. obj origin)))`

In plain terms: each particle performs a random walk, moving to one uniformly chosen adjacent position relative to its previous origin.

The exact set of possible adjacent positions is whatever the built-in/native `adjPositions` returns. The program itself does not spell out whether this includes only orthogonal neighbors, diagonal neighbors, in-bounds filtering, or staying still.

### Click handler

There is one `on` handler:

```lisp
(on clicked
  (= particles
    (addObj
      (prev particles)
      (Particle (Position (.. click x) (.. click y))))))
```

When the global `clicked` condition is true:

- A new `Particle` is created.
- Its origin is set to the click position:

`(Position (.. click x) (.. click y))`

- That particle is added to the previous particle list using `addObj`.

The handler uses `(prev particles)`, so the append operation is based on the particle list from the previous state.

---

## 4. Player actions

The only player input handled by the program is clicking/tapping.

### Click anywhere
Condition:

`clicked`

Effect:

- Adds a new blue `Particle` at the clicked grid coordinate.
- The x coordinate comes from `(.. click x)`.
- The y coordinate comes from `(.. click y)`.

There are no keyboard controls, no arrow-key movement, no object-specific click checks, and no dragging behavior.

---

## 5. Edge cases and boundaries

- `GRID_SIZE` is set to `16`, but the program does not explicitly clamp particle positions.
- Random movement has no explicit collision check.
- Click placement has no explicit bounds check.
- Multiple particles may be added at the same position; the program does not prevent overlap.
- Particles are never removed.
- There is no explicit rule preventing particles from moving onto each other.
- Boundary behavior for random movement depends on the implementation of `adjPositions` and the engine’s grid handling. The source itself does not add extra guards.
- If the player clicks outside the valid grid, the source code itself does not say what happens; it simply uses the provided click coordinates.

---

## 6. Win/lose and progression

The program defines no win condition, lose condition, reward, score, level progression, or terminal state.

The only progression is emergent:

- The player can keep adding particles by clicking.
- Existing particles continue randomly moving each update.
- The number of particles grows over time unless no more clicks occur.
