<!--
Ground-truth beliefs for AutumnBench task '83WKQ'.
Drafted by an LLM; REVIEW AND EDIT before relying on this file.
Lines starting with `- ` are parsed as individual beliefs by
judge_beliefs_against_gt. Everything else is treated as commentary.
-->

- The game’s `GRID_SIZE` variable is `16`.
- The only object type defined by the source is `Particle`.
- A `Particle` renders with one `Cell`.
- A `Particle`’s rendered `Cell` has local x offset `0`.
- A `Particle`’s rendered `Cell` has local y offset `0`.
- A `Particle`’s rendered `Cell` is blue.
- A `Particle`’s visible cell appears at the particle’s `origin`.
- The game state variable `particles` is a list of `Particle` objects.
- The initial value of `particles` is the empty list.
- If `(prev particles)` is empty and `clicked` is false, the next `particles` list is empty.
- The default update for `particles` reads from `(prev particles)`.
- The default update applies an update function to each `Particle` in `(prev particles)`.
- The default update sets each previous `Particle`’s `"origin"` field.
- The default update selects each new origin using `uniformChoice`.
- The choices passed to `uniformChoice` are `(adjPositions (.. obj origin))`.
- A particle’s random-motion rule does not use the click position.
- A particle’s random-motion rule does not inspect the positions of other particles.
- On a no-click tick, the default update preserves the number of particles.
- Particles can change position without player input.
- No source rule removes a `Particle` from `particles`.
- The source contains exactly one event handler.
- The event handler’s condition is the global `clicked` condition.
- When `clicked` is true, the click handler constructs one new `Particle`.
- The new clicked `Particle` has origin x coordinate equal to `(.. click x)`.
- The new clicked `Particle` has origin y coordinate equal to `(.. click y)`.
- The click handler inserts the new `Particle` using `addObj`.
- The click handler inserts the new `Particle` into `(prev particles)`.
- The click handler’s output contains one more particle than `(prev particles)`.
- Repeated clicked ticks can accumulate multiple `Particle` objects.
- The click handler does not require the click to be on an existing `Particle`.
- The click handler has no check for whether the clicked position is occupied.
- The click handler has no explicit bounds check against `GRID_SIZE`.
- The click handler has no maximum-particle-count guard.
- The random-motion rule has no explicit boundary comparison against `GRID_SIZE`.
- The random-motion rule has no collision check.
- The random-motion rule has no source-level rule preventing two particles from having the same origin.
- The program has no score variable.
- The program has no win-condition rule.
- The program has no lose-condition rule.
- The program has no terminal-state rule.
