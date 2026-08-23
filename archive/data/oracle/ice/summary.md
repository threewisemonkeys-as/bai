## Overview
This is a 16-grid simulation with a movable gray cloud, a 2×2 celestial body that toggles between “day” and “not day,” and falling water drops. The player can move the cloud left/right, spawn drops below it, and click to toggle the world state and all existing drops.

## 1. Objects

### `CelestialBody`
Defined as:

`(object CelestialBody (: day Bool) ...)`

Fields:
- `day : Bool`
  - Controls the celestial body’s color.
  - Also controls the type of newly spawned `Water` drops.

Appearance:
- A 2×2 block of cells at local offsets:
  - `(0,0)`, `(0,1)`, `(1,0)`, `(1,1)`
- Each cell is:
  - `"gold"` if `day` is `true`
  - `"gray"` if `day` is `false`

So when `day = true`, it appears as a gold 2×2 square; when `day = false`, it appears as a gray 2×2 square.

---

### `Cloud`
Defined as:

`(object Cloud ...)`

Fields:
- No custom fields.

Appearance:
- A horizontal 3-cell gray cloud at local offsets:
  - `(-1,0)`, `(0,0)`, `(1,0)`
- All cells are `"gray"`.

The cloud’s origin is its center cell.

---

### `Water`
Defined as:

`(object Water (: liquid Bool) ...)`

Fields:
- `liquid : Bool`
  - Controls both the drop’s color and its movement behavior.
  - If `liquid = true`, the drop behaves as liquid using `nextLiquid`.
  - If `liquid = false`, the drop behaves as a solid using `nextSolid`.

Appearance:
- A single cell at local offset `(0,0)`.
- Color:
  - `"blue"` if `liquid` is `true`
  - `"lightblue"` if `liquid` is `false`

## 2. Initial state

### Grid
The program sets:

`(= GRID_SIZE 16)`

So the game is intended to run on a 16-cell grid, with boundary checks handled through `isWithinBounds`.

### Celestial body
Initialized by:

`(CelestialBody true (Position 0 0))`

Initial state:
- `day = true`
- origin at `(0,0)`
- appears as a gold 2×2 block occupying world cells `(0,0)`, `(0,1)`, `(1,0)`, and `(1,1)`.

Its `initnext` clause is:

`(initnext (CelestialBody true (Position 0 0)) (prev celestialBody))`

So after initialization, it simply persists from frame to frame unless changed by an input handler.

### Cloud
Initialized by:

`(Cloud (Position 4 0))`

Initial state:
- origin at `(4,0)`
- appears as three gray cells at `(3,0)`, `(4,0)`, and `(5,0)`.

Its `initnext` clause is:

`(initnext (Cloud (Position 4 0)) (prev cloud))`

So it persists from frame to frame unless moved by input.

### Water list
Initialized by:

`(initnext (list) (updateObj (prev water) nextWater))`

Initial state:
- no water drops exist.

On later frames:
- the previous water list is updated by applying `nextWater` to each drop.
- The list is not reset; spawned drops persist and continue updating.

## 3. Dynamics

### Automatic water update
Each frame, the water list updates using:

`(updateObj (prev water) nextWater)`

The helper function is:

`(= nextWater (--> (drop) (if (.. drop liquid) then (nextLiquid drop) else (nextSolid drop))))`

Meaning:
- If a drop’s `liquid` field is `true`, it is updated with `nextLiquid`.
  - Per the standard library, liquid tries to move downward if possible, and otherwise may flow toward available holes.
- If a drop’s `liquid` field is `false`, it is updated with `nextSolid`.
  - `nextSolid` is the standard-library falling-solid behavior, equivalent to moving down if the downward move is in bounds and collision-free; otherwise the drop stays put.

### Cloud movement helper
The helper function:

`(= nextCloud (--> (cloud position) (if (isWithinBounds (move cloud position)) then (move cloud position) else cloud)))`

tries to move the cloud by a given vector. The move only succeeds if the moved cloud remains within bounds. If the move would place any part of the cloud outside the grid, the cloud remains unchanged.

This helper checks bounds only; it does not check collision with water or the celestial body.

## 4. Player actions

The program responds to four input events: `left`, `right`, `down`, and `clicked`.

### Left input
Handler:

`(on left (= cloud (nextCloud cloud (Position -1 0))))`

When `left` is active:
- the cloud attempts to move one cell left.
- if the resulting cloud would still be within bounds, it moves.
- otherwise it stays in place.

### Right input
Handler:

`(on right (= cloud (nextCloud cloud (Position 1 0))))`

When `right` is active:
- the cloud attempts to move one cell right.
- if the resulting cloud would still be within bounds, it moves.
- otherwise it stays in place.

### Down input
Handler:

`(on down (= water (addObj water (Water (.. celestialBody day) (movePos (.. cloud origin) (Position 0 1))))))`

When `down` is active:
- a new `Water` drop is added to the water list.
- Its `liquid` field is set to the current value of `celestialBody.day`.
  - If the celestial body is in day mode, the new drop is liquid/blue.
  - If the celestial body is not in day mode, the new drop is solid/lightblue.
- The new drop spawns one cell below the cloud origin:
  - spawn position = `cloud.origin + (0,1)`
  - since the cloud origin is its center, drops spawn below the center of the cloud, not below all three cloud cells.

There is no explicit collision or bounds check when spawning a new drop.

### Click input
Handler:

```lisp
(on clicked
  (let 
    (= celestialBody (updateObj celestialBody "day" (! (.. celestialBody day))))
    (= water (updateObj water (--> drop (updateObj drop "liquid" (! (.. drop liquid))))))
  ))
```

When a generic `clicked` event occurs:
- the celestial body’s `day` field is toggled.
  - `true` becomes `false`
  - `false` becomes `true`
- every existing water drop has its `liquid` field toggled.
  - liquid drops become solid
  - solid drops become liquid

The code does not check which object was clicked. It responds to the general `clicked` condition, so no object-specific click target is required by the program.

## 5. Edge cases and boundaries

- Cloud movement is guarded by `isWithinBounds`.
  - The cloud cannot be moved so that any of its three cells leave the board.
  - There is no wrapping and no modular movement.
  - If the move is invalid, the cloud simply stays in place.

- Because the cloud is three cells wide with offsets `(-1,0)`, `(0,0)`, and `(1,0)`, on a normal 16-wide grid its center cannot move all the way to the extreme left or right if that would push part of the cloud outside the board.

- Cloud movement does not check collisions.
  - The cloud may overlap other objects if bounds allow it.
  - The only condition in `nextCloud` is `isWithinBounds`.

- Water spawning does not check whether the spawn cell is free.
  - Pressing `down` repeatedly can add multiple drops.
  - A drop may be added even if another object already occupies the target position.

- Water movement uses the standard-library collision/bounds-aware movement functions through `nextLiquid` and `nextSolid`.
  - Drops that cannot move according to those rules remain in place.
  - The program does not explicitly delete drops at the bottom or when blocked.

- Clicking toggles all existing water drops, but if the water list is empty, only the celestial body changes.

- The program defines no response to the `up` key or other input events.

## 6. Win/lose/progression

There is no explicit win condition, lose condition, score, reward, level progression, or terminal state.

Progression is purely simulated:
- the player moves the cloud left/right,
- spawns drops with `down`,
- drops fall or flow according to their `liquid` state,
- clicking toggles the celestial body and converts all existing drops between liquid and solid behavior.
