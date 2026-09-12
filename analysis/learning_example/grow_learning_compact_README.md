# Compact Grow learning figure

`grow_learning_compact.fig` is a native Figma `.fig` document containing one editable frame, `grow_learning_compact`.
`learning_evolution_fig_with_grow_compact.fig` contains the original document plus the new Grow frame to the right of `magnets_learning_compact`. The source `learning_evolution_fig.fig` is unchanged.

The layout follows the supplied `magnets_learning_compact`: Inter text, four states with action arrows, perception columns underneath, horizontal learning checkpoints, and a right-hand dynamics-model panel. It preserves the full 16×16 Grow observations, with no white focus boxes. The new figure is 1829×1050 px; its additional height accommodates Grow's four checkpoints.

The frame contains editable text, rectangles, vectors, and named state/iteration groups. Preview exports are `grow_learning_compact.png`, `.pdf`, and `.svg`. The PDF/SVG previews use the saved glyph outlines; the `.fig` preserves live text.

## Content

Frames 34–37 from Grow's `train_d0` show three `left` actions: the gray block approaches the gold block, touches it, then covers its edge. All 16 perception outputs are checked by running the saved perception programs. Excerpts retain their source order; ellipses identify omitted text.

- Iteration 3: colored-cell coordinates; no learned world knowledge.
- Iteration 12: perception is unchanged; learned world knowledge describes blocks and their relative-position rules.
- Iteration 13: perception groups connected cells into rectangles; world knowledge remains unchanged.
- Iteration 27: perception adds `gare`, which changes from 0 to 1 in the illustrated sequence; world knowledge remains unchanged and does **not** reference `gare`.

`Dynamics model D` follows the reference's display terminology for the run's `world_knowledge`, previously labelled B. The figure does not claim that Grow's beliefs adopt `gare` or learn conditional plant growth. The `right` rule is explicitly labelled; the illustrated actions are `left`.

## Validation and provenance

The native files were generated locally by decoding and re-encoding the exact Kiwi schema embedded in the supplied `.fig`, using [Evan Wallace's Kiwi implementation](https://github.com/evanw/kiwi). The container preserves the reference's format version 106 and compression types. No Figma MCP or cloud editing service was used. Import in the Figma application has not been tested.

Checks cover the native encode/decode round trip, all parent/blob references, text bounds, 1024 grid cells and their colors, 16 perception panels, four belief panels, original candidate/trajectory provenance, and exact preservation of every original node in the combined file. The previews render the saved native geometry and text glyphs and were visually reviewed. Single-character glyphs are reused only when their source text spans exactly one character, avoiding substitution of `->` ligatures for range hyphens.

Reference: `analysis/learning_example/learning_evolution_fig.fig`, frame `magnets_learning_compact`.
Evidence: `analysis/learning_example/learning_evolution_grow_relations_evidence.json` and the original candidate/trajectory logs it references.
Detailed checks: `grow_learning_compact_evidence.json`.

## Regenerate

From the repository root, with Bun available:

```sh
bun add --cwd /tmp/grow_figma_compact kiwi-schema@0.5.0
.venv/bin/python offline_learning/scripts/fig_grow_learning_compact.py
```

For dependencies installed elsewhere, pass `--modules /absolute/path/to/node_modules`. The helper is `offline_learning/scripts/figma_export/fig_kiwi_io.cjs`. Preview rendering uses the repository's existing Matplotlib installation and the reference's cached font outlines; it does not require downloading fonts.
