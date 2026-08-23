<!--
Ground-truth beliefs for AutumnBench task 'S2KT7'.
Drafted by an LLM; REVIEW AND EDIT before relying on this file.
Lines starting with `- ` are parsed as individual beliefs by
judge_beliefs_against_gt. Everything else is treated as commentary.
-->

- The only object types defined by the program are `Ant` and `Food`.
- An `Ant` occupies one gray cell at its origin.
- A `Food` occupies one red cell at its origin.
- `GRID_SIZE` has the value `16`.
- The initial `ants` list contains an `Ant` at `(5, 5)`.
- The initial `ants` list contains an `Ant` at `(1, 14)`.
- The initial `foods` list is empty.
- No rule creates additional `Ant` objects after initialization.
- No rule removes `Ant` objects.
- The food-removal rule runs every tick because its condition is `true`.
- The food-removal rule filters the previous tick’s `foods` list.
- A previous `Food` is removed when it intersects any previous `Ant`.
- A previous `Food` is kept when it intersects no previous `Ant`.
- For an `Ant` and a `Food`, intersection means their single occupied cells are the same grid cell.
- Food removal uses previous ant positions.
- Food removal does not use ant positions after the current tick’s movement.
- Food is not removed merely because it is adjacent to an ant.
- `Food` objects have no movement rule.
- The ant-movement rule runs every tick because its condition is `true`.
- The ant-movement rule updates ants from the previous tick’s `ants` list.
- Each previous `Ant` is updated by the same movement function.
- Each ant chooses a target using `(closest obj foods)`.
- The closest-food calculation uses squared distance.
- If `foods` is empty, `closest` returns the ant itself.
- If `foods` is empty, an ant receives a zero movement vector.
- If `foods` is empty, ants remain in their previous cells.
- An ant moves at most one grid cell in a tick.
- If the selected food is in the same row, the ant moves one horizontal step toward it.
- If the selected food is in the same column, the ant moves one vertical step toward it.
- If the selected food differs in both x and y, the ant moves one horizontal step toward it.
- If the selected food differs in both x and y, the ant does not change y on that tick.
- Ant movement uses plain `move`.
- Ant movement does not check for collisions before moving.
- Ant movement does not check grid bounds before moving.
- Ants can overlap each other if their moves put them on the same cell.
- Ants can move onto food cells.
- Moving onto a food cell does not itself delete the food.
- A food reached by an ant’s new position can be removed by a later tick’s previous-ant filter.
- The only player input condition used by the program is global `clicked`.
- When `clicked` is true, the program requests `2` random positions using grid size `16`.
- Each random position produced on click is converted into a `Food`.
- Click-created `Food` objects are added to the existing `foods` list.
- The click handler does not directly modify the `ants` list.
- The click handler does not use the clicked screen or grid coordinate.
- The program performs no spawn-time occupancy check before adding clicked food.
- The program has no score variable.
- The program has no terminal win or lose condition.
