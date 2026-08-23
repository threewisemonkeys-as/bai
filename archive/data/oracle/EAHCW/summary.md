This program is a simple particle-placing/color-selection toy on a grid.

## 1. Objects

### `Particle`
Defined as:

`(object Particle (: color String) (Cell 0 0 color))`

- Fields:
  - `color : String` — controls the rendered color of the particle.
- Appearance:
  - A `Particle` renders as a single `Cell` at local offset `(0, 0)`.
  - The cell’s color is whatever the particle’s `color` field contains.
- Position:
  - Each particle has an implicit origin/position supplied when it is constructed, e.g. `(Particle "red" (Position x y))`.

There are no other object types.

## 2. Initial state

- `GRID_SIZE` is set to `16`.
  - The code itself does not use `GRID_SIZE` in any condition or calculation.
- `particles` starts as an empty list:

  `(= particles (initnext (list) (prev "particles")))`

- `currColor` starts as `"red"`:

  `(= currColor (initnext "red" (prev currColor)))`

- `active_arrow` starts as `"none"`:

  `(= active_arrow (initnext "none" (prev active_arrow)))`

So initially, there are no particles, the current color is red, and no arrow direction is active.

## 3. Dynamics

### Persistent state via `initnext`

Each state variable persists from frame to frame unless changed by an `on` handler:

- `particles`:
  - Initial value: empty list.
  - Default next value: previous `particles` list.
- `currColor`:
  - Initial value: `"red"`.
  - Default next value: previous `currColor`.
- `active_arrow`:
  - Initial value: `"none"`.
  - Default next value: previous `active_arrow`.

### Click handler

On any `clicked` event:

```lisp
(on clicked
  (= particles
    (addObj
      (prev "particles")
      (Particle
        (if (== active_arrow "none") then "red" else currColor)
        (Position (.. click x) (.. click y))))))
```

This adds a new `Particle` to the particle list.

- The new particle is placed at the click coordinate:
  - x-coordinate: `(.. click x)`
  - y-coordinate: `(.. click y)`
- Its color is:
  - `"red"` if `active_arrow` is `"none"`.
  - Otherwise, `currColor`.
- Existing particles are preserved by adding to `(prev "particles")`.

### Arrow key handlers

Pressing an arrow key normally selects that arrow and sets the current color:

- `up`:

  ```lisp
  (on up (let (= currColor "gold") (= active_arrow "up")))
  ```

  Sets `currColor` to `"gold"` and `active_arrow` to `"up"`.

- `down`:

  ```lisp
  (on down (let (= currColor "purple") (= active_arrow "down")))
  ```

  Sets `currColor` to `"purple"` and `active_arrow` to `"down"`.

- `left`:

  ```lisp
  (on left (let (= currColor "green") (= active_arrow "left")))
  ```

  Sets `currColor` to `"green"` and `active_arrow` to `"left"`.

- `right`:

  ```lisp
  (on right (let (= currColor "blue") (= active_arrow "right")))
  ```

  Sets `currColor` to `"blue"` and `active_arrow` to `"right"`.

### Opposite-direction reset handlers

If the player presses the opposite of the previously active arrow, the program resets the selection:

- If previous active arrow was `"up"` and `down` is pressed:

  ```lisp
  (on (and down (== (prev active_arrow) "up"))
    (let (= active_arrow "none") (= currColor "red")))
  ```

- If previous active arrow was `"down"` and `up` is pressed:

  ```lisp
  (on (and up (== (prev active_arrow) "down"))
    (let (= active_arrow "none") (= currColor "red")))
  ```

- If previous active arrow was `"right"` and `left` is pressed:

  ```lisp
  (on (and left (== (prev active_arrow) "right"))
    (let (= active_arrow "none") (= currColor "red")))
  ```

- If previous active arrow was `"left"` and `right` is pressed:

  ```lisp
  (on (and right (== (prev active_arrow) "left"))
    (let (= active_arrow "none") (= currColor "red")))
  ```

Thus, pressing the opposite direction of the previously active arrow is intended to deselect the active arrow and return the color to red.

## 4. Player actions

The program responds to these input events:

- `clicked`
  - Adds a particle at the click position.
  - If no arrow is active, the particle is red.
  - If an arrow is active, the particle uses the current selected arrow color.

- `up`
  - Selects the up arrow and sets the current color to gold.
  - If the previous active arrow was down, it also matches the reset condition and resets to no active arrow/red.

- `down`
  - Selects the down arrow and sets the current color to purple.
  - If the previous active arrow was up, it also matches the reset condition and resets to no active arrow/red.

- `left`
  - Selects the left arrow and sets the current color to green.
  - If the previous active arrow was right, it also matches the reset condition and resets to no active arrow/red.

- `right`
  - Selects the right arrow and sets the current color to blue.
  - If the previous active arrow was left, it also matches the reset condition and resets to no active arrow/red.

Clicks are not restricted to clicking on particles or any particular object; the handler responds to the global `clicked` event and uses the global `click` position.

## 5. Edge cases / boundaries

- The program defines `GRID_SIZE` as `16`, but it does not explicitly clamp click positions or check whether a click is within bounds.
- There is no collision checking.
- Multiple particles may be placed on the same coordinate.
- Particles never move, disappear, merge, or change color after creation.
- The particle list only grows when clicks occur.
- The arrow reset handlers overlap with the unconditional arrow handlers. For example, pressing `down` while the previous active arrow was `"up"` satisfies both:
  - the normal `down` handler, and
  - the opposite-direction reset handler.
  
  The code explicitly includes the reset assignment to `"none"`/`"red"` for such cases, but exact conflict resolution depends on Autumn’s handling of multiple simultaneous `on` assignments.

## 6. Win/lose / progression

There is no win condition, lose condition, score, reward, terminal state, timer, health, or level progression defined in the program.

The only progression is that the player can keep placing colored particles over time, while changing or clearing the active color selection with the arrow keys.
