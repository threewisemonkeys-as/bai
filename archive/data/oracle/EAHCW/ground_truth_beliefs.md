<!--
Ground-truth beliefs for AutumnBench task 'EAHCW'.
Drafted by an LLM; REVIEW AND EDIT before relying on this file.
Lines starting with `- ` are parsed as individual beliefs by
judge_beliefs_against_gt. Everything else is treated as commentary.
-->

- `Particle` is the only object type defined by the program.
- Each `Particle` has a `color` string field.
- Each `Particle` renders exactly one `Cell`.
- The rendered cell of a `Particle` is at local offset `(0, 0)` from the particle origin.
- The rendered cell color of a `Particle` equals that particle’s `color` field.
- A constructed `Particle` appears at the `Position` passed to the `Particle` constructor.
- `GRID_SIZE` is set to `16`.
- No rule reads `GRID_SIZE` to decide whether an action is allowed.
- The initial `particles` list is empty.
- The initial `currColor` is `"red"`.
- The initial `active_arrow` is `"none"`.
- On a tick with no `clicked` event, `particles` remains the previous particle list.
- On a tick with no arrow input, `currColor` remains its previous value.
- On a tick with no arrow input, `active_arrow` remains its previous value.
- No program rule moves an existing `Particle`.
- No program rule removes an existing `Particle`.
- No program rule changes the `color` field of an existing `Particle`.
- A tick with `clicked` true adds a new `Particle` to `particles`.
- The click-created `Particle` is added to the previous particle list.
- The click-created `Particle` has x-coordinate equal to `click.x`.
- The click-created `Particle` has y-coordinate equal to `click.y`.
- If `active_arrow` is `"none"` when the click color is evaluated, the click-created `Particle` is `"red"`.
- If `active_arrow` is not `"none"` when the click color is evaluated, the click-created `Particle` has color `currColor`.
- The click handler does not require the click to be on an existing object.
- The click handler does not test whether another particle already occupies the clicked cell.
- The click handler has no explicit coordinate bounds check.
- A click does not assign a new value to `currColor`.
- A click does not assign a new value to `active_arrow`.
- For a tick in which `up` is the only arrow input and previous `active_arrow` is not `"down"`, the tick ends with `currColor` equal to `"gold"`.
- For a tick in which `up` is the only arrow input and previous `active_arrow` is not `"down"`, the tick ends with `active_arrow` equal to `"up"`.
- For a tick in which `down` is the only arrow input and previous `active_arrow` is not `"up"`, the tick ends with `currColor` equal to `"purple"`.
- For a tick in which `down` is the only arrow input and previous `active_arrow` is not `"up"`, the tick ends with `active_arrow` equal to `"down"`.
- For a tick in which `left` is the only arrow input and previous `active_arrow` is not `"right"`, the tick ends with `currColor` equal to `"green"`.
- For a tick in which `left` is the only arrow input and previous `active_arrow` is not `"right"`, the tick ends with `active_arrow` equal to `"left"`.
- For a tick in which `right` is the only arrow input and previous `active_arrow` is not `"left"`, the tick ends with `currColor` equal to `"blue"`.
- For a tick in which `right` is the only arrow input and previous `active_arrow` is not `"left"`, the tick ends with `active_arrow` equal to `"right"`.
- For a tick in which `down` is the only arrow input and previous `active_arrow` is `"up"`, the tick ends with `currColor` equal to `"red"`.
- For a tick in which `down` is the only arrow input and previous `active_arrow` is `"up"`, the tick ends with `active_arrow` equal to `"none"`.
- For a tick in which `up` is the only arrow input and previous `active_arrow` is `"down"`, the tick ends with `currColor` equal to `"red"`.
- For a tick in which `up` is the only arrow input and previous `active_arrow` is `"down"`, the tick ends with `active_arrow` equal to `"none"`.
- For a tick in which `left` is the only arrow input and previous `active_arrow` is `"right"`, the tick ends with `currColor` equal to `"red"`.
- For a tick in which `left` is the only arrow input and previous `active_arrow` is `"right"`, the tick ends with `active_arrow` equal to `"none"`.
- For a tick in which `right` is the only arrow input and previous `active_arrow` is `"left"`, the tick ends with `currColor` equal to `"red"`.
- For a tick in which `right` is the only arrow input and previous `active_arrow` is `"left"`, the tick ends with `active_arrow` equal to `"none"`.
- A tick with arrow input but no `clicked` event does not add any particle.
- The program defines no score variable.
- The program defines no rule that ends the game.
