<!--
Ground-truth beliefs for AutumnBench task 'DQ8GC'.
Drafted by an LLM; REVIEW AND EDIT before relying on this file.
Lines starting with `- ` are parsed as individual beliefs by
judge_beliefs_against_gt. Everything else is treated as commentary.
-->

- `GRID_SIZE` is set to `16`.
- The movement rules do not use `GRID_SIZE`.
- A `Particle` renders as one cell at its own origin.
- A `Particle` with `health=true` appears gray.
- A `Particle` with `health=false` appears darkgreen.
- The initial inactive list contains a `health=true` particle at `(7, 5)`.
- The initial inactive list contains a `health=true` particle at `(4, 3)`.
- The initial inactive list contains a `health=true` particle at `(6, 6)`.
- The initial inactive list contains a `health=true` particle at `(3, 5)`.
- The initial active particle is at `(2, 2)`.
- The initial active particle has `health=false`.
- By default, `activeParticle` keeps its previous value on the next tick.
- By default, `inactiveParticles` is recomputed from `prev inactiveParticles`.
- The inactive-particle spread rule uses `prev activeParticle` as a possible darkgreen source.
- The inactive-particle spread rule uses `prev inactiveParticles` as possible darkgreen sources.
- Particles with `health=true` are ignored as spread sources.
- The inactive-particle spread rule uses Manhattan distance at most `1`.
- A diagonally adjacent particle does not satisfy the spread adjacency test.
- A particle on the same cell as a `health=false` particle satisfies the unit-distance adjacency test.
- A previous inactive particle satisfying the spread predicate is assigned `health=false`.
- A previous inactive particle not satisfying the spread predicate keeps its previous health under the default inactive update.
- The default inactive update does not move inactive particles.
- No rule changes an inactive particle from `health=false` back to `health=true`.
- A newly darkgreen inactive particle cannot spread darkgreen status to another inactive particle until a later tick.
- The initial active particle is not unit-adjacent to any initial inactive particle.
- The active-particle health rule triggers when some unit-adjacent object has `health=false`.
- The active-particle health rule sets `activeParticle.health` to `false`.
- The active-particle health rule does not change `activeParticle`'s position.
- A selected gray active particle can become darkgreen by being unit-adjacent to a darkgreen object.
- No rule directly mutates `activeParticle.health` to `true`.
- Clicking is checked only against `prev inactiveParticles`.
- Clicking a previous inactive particle triggers the selection handler.
- Clicking only the active particle does not trigger the selection handler.
- The selection handler removes clicked previous inactive particles from `inactiveParticles`.
- The selection handler adds the previously controlled particle to `inactiveParticles`.
- The selection handler sets `activeParticle` to the first clicked previous inactive particle.
- If multiple previous inactive particles are clicked, only the first clicked one becomes `activeParticle`.
- If multiple previous inactive particles are clicked, the non-selected clicked particles are removed from `inactiveParticles`.
- Selecting a `health=true` inactive particle can make the controlled active particle gray.
- Pressing `left` sets `activeParticle` to `prev activeParticle` shifted by `(-1, 0)`.
- Pressing `right` sets `activeParticle` to `prev activeParticle` shifted by `(1, 0)`.
- Pressing `up` sets `activeParticle` to `prev activeParticle` shifted by `(0, -1)`.
- Pressing `down` sets `activeParticle` to `prev activeParticle` shifted by `(0, 1)`.
- Arrow-key movement uses plain `move`.
- Arrow-key movement does not check collisions.
- The program contains no rule preventing the active particle from moving onto an inactive particle's cell.
- The program contains no rule preventing the active particle from moving outside the nominal `16` by `16` grid.
- Arrow-key movement is not conditional on the active particle's health.
- Arrow-key handlers assign only `activeParticle`.
- No rule removes a particle solely because its `health` is `false`.
- The program defines no win condition.
- The program defines no lose condition.
- The program defines no score variable.
