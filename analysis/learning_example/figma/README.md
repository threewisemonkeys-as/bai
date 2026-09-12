# Editable Magnets and Grow diagrams for Figma

Import **learning_evolution_magnets_and_grow.sketch** for both diagrams on separate
pages, or import the individual `.sketch` files. These use Figma's supported
Sketch import format; they are not native `.fig` files or published Figma links.
No Sketch application or custom Figma plugin is required.

1. Install the four fonts in `fonts/` if they are not already available in Figma.
2. In Figma's file browser, choose **Create → Import → From your computer** and
   select the combined or individual `.sketch` file. Dragging it onto the file
   browser also works.
3. Open the imported design. Each page contains an editable diagram frame.

The layers are grouped into the game sequence, individual states, action arrows,
and iterations. Each iteration has separate perception groups for X0–X3 and a
world knowledge group. Grid cells, arrows, highlights, and text remain editable.
Colored text runs remain separate text layers to retain their exact positioning.
The frames are 1188 px wide (the paper SVG geometry scaled uniformly by 3).

Magnets retains iterations 0, 4, and 8, with relation definitions in iteration 8.
Grow retains iterations 3, 12, 13, and 27. Both retain the latest formatting edits.
Source programs, beliefs, and full-output evidence remain in the parent figures
directory; their content has not been changed for this export.

The DejaVu fonts and their license are included. Figma's desktop app can use
installed fonts; browser users may need Figma's font installer. See the official
import instructions for platform-specific font support.

The SVG files are alternate vector imports. Use the `.sketch` files when editable
text is important. PNG previews were rendered locally from the exported native
layers. Archive structure, the official JSON schema, exact text preservation,
and every displayed grid cell were checked. The Figma application importer was
not run in this session because a Figma connection was not available.

Regenerate after updating the paper SVGs:

    .venv/bin/python offline_learning/scripts/export_learning_evolution_figma.py

Official documentation:
- [Figma: Import Sketch files](https://help.figma.com/hc/en-us/articles/360040514273-Import-Sketch-files)
- [Sketch document format](https://developer.sketch.com/file-format/)
- [Official JSON schemas](https://github.com/sketch-hq/sketch-document)
