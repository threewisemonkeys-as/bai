## 1. Objects

### `Particle`
- Fields:
  - `health : Bool`
- Appearance:
  - A `Particle` renders as one cell at local offset `(0, 0)`.
  - Its color depends entirely on `health`:
    - `health = true` → `"gray"`
    - `health = false` → `"darkgreen"`
  - This is defined by:
    ```lisp
    (Cell 0 0 (if health then "gray" else "darkgreen"))
    ```
- Interpretation:
  - The program treats `health=false` particles as the spreading source: many rules check `(! (.. obj health))`.

There are two state variables holding particles:
- `inactiveParticles : List Particle` — the non-controlled particle list.
- `activeParticle : Particle` — the currently controlled/selected particle.

`GRID_SIZE` is set to `16`, but the program itself does not use it for movement limits or collision checks.

---

## 2. Initial state

### `inactiveParticles`
Initially contains four gray particles, all with `health=true`:

| Position | Health | Color |
|---|---:|---|
| `(7, 5)` | `true` | gray |
| `(4, 3)` | `true` | gray |
| `(6, 6)` | `true` | gray |
| `(3, 5)` | `true` | gray |

Defined by:
```lisp
(list
  (Particle true (Position 7 5))
  (Particle true (Position 4 3))
  (Particle true (Position 6 6))
  (Particle true (Position 3 5)))
```

### `activeParticle`
Initially one darkgreen particle:

| Position | Health | Color |
|---|---:|---|
| `(2, 2)` | `false` | darkgreen |

Defined by:
```lisp
(Particle false (Position 2 2))
```

---

## 3. Dynamics

### Default update for `inactiveParticles`

`inactiveParticles` is defined with `initnext`. After initialization, each timestep updates the previous inactive list:

```lisp
(updateObj
  (prev inactiveParticles)
  updater
  predicate)
```

For each inactive particle from the previous frame, the predicate checks whether it is adjacent to any `health=false` particle among:

1. the previous `activeParticle`, and
2. all previous `inactiveParticles`.

That source list is built by:
```lisp
(filter
  (--> o2 (! (.. o2 health)))
  (vcat (list (list (prev activeParticle)) (prev inactiveParticles))))
```

If an inactive particle is adjacent to any such `health=false` particle within distance `1`, it is updated to:

```lisp
(updateObj obj "health" false)
```

So, in effect:
- Darkgreen / `health=false` status spreads from the previous active particle and previous inactive particles to neighboring inactive particles.
- The spread uses the previous timestep’s state via `prev`.
- The update only ever sets `health` to `false`; there is no rule that restores `health` to `true`.

Adjacency is tested with:
```lisp
(adj obj unhealthyParticles 1)
```
Using the standard library, this is Manhattan-style adjacency with unit size `1`: orthogonal neighbors count; diagonals do not.

### Default update for `activeParticle`

`activeParticle` is also defined with `initnext`:

```lisp
(initnext (Particle false (Position 2 2)) (prev activeParticle))
```

So by default, the active particle simply persists from the previous frame unless an `on` handler changes it.

### Active particle becoming `health=false`

There is an `on` rule:

```lisp
(on
  (any (--> obj (! (.. obj health))) (adjacentObjs activeParticle 1))
  (= activeParticle (updateObj activeParticle "health" false)))
```

If any object adjacent to the current `activeParticle` has `health=false`, then the active particle is set to `health=false`.

This can turn a gray selected particle darkgreen if it gets next to a darkgreen particle.

---

## 4. Player actions

### Clicking an inactive particle

The program responds to clicks on particles in `prev inactiveParticles`:

```lisp
(on (clicked (prev inactiveParticles)) ...)
```

When an inactive particle is clicked, the handler performs a selection swap:

1. Remove all clicked particles from the previous inactive list:
   ```lisp
   (filter (--> obj (! (clicked obj))) (prev inactiveParticles))
   ```

2. Add the current `activeParticle` into the inactive list:
   ```lisp
   (addObj filteredInactive activeParticle)
   ```

3. Set `activeParticle` to the first clicked inactive particle:
   ```lisp
   (= activeParticle (head (objClicked (prev inactiveParticles))))
   ```

So, normally, clicking an inactive particle makes that particle the new active/controlled particle, while the old active particle becomes inactive.

Clicking the active particle itself has no special handler.

### Arrow-key movement

The active particle can be moved one grid cell using arrow inputs:

```lisp
(on left  (= activeParticle (move (prev activeParticle) (Position -1 0))))
(on right (= activeParticle (move (prev activeParticle) (Position 1 0))))
(on up    (= activeParticle (move (prev activeParticle) (Position 0 -1))))
(on down  (= activeParticle (move (prev activeParticle) (Position 0 1))))
```

Effects:
- `left` moves active particle by `(-1, 0)`.
- `right` moves active particle by `(1, 0)`.
- `up` moves active particle by `(0, -1)`.
- `down` moves active particle by `(0, 1)`.

Movement uses plain `move`, not a collision-checked or bounds-checked movement function.

---

## 5. Edge cases and boundaries

- `GRID_SIZE` is set to `16`, but the movement rules do not clamp the active particle to the grid.
- The active particle can be moved outside the nominal grid unless the runtime enforces bounds externally; this program does not.
- Movement does not check collisions. The active particle may move onto the same cell as another particle.
- There is no rule preventing particles from overlapping.
- Diagonal adjacency does not cause spreading under the explicit `adj ... 1` rules.
- `health=false` is absorbing: no rule changes a particle from `false` back to `true`.
- If multiple inactive particles are clicked at once, the code removes all clicked particles from `inactiveParticles` but only makes the first clicked particle, via `head`, the new `activeParticle`.
- The program does not define custom priority rules for simultaneous inputs or multiple `on` handlers assigning `activeParticle`.

---

## 6. Win / lose / progression

There is no explicit win condition, lose condition, score, reward, or terminal state.

The only progression-like mechanic is the spread of `health=false` / darkgreen status through adjacency. Over time, particles near darkgreen particles can become darkgreen themselves, but the program does not declare this as a victory, failure, or end state.
