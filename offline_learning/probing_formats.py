"""Wire formats for raw-grid answers, with tolerant parsers.

The frozen probing protocol asks the evaluator to write a raw grid as a JSON array of
colour strings and scores anything unparseable as zero. On the final-selected run that
rule zeroed 72.6% of raw forward answers and 76.3% of reconstruction answers, while the
same model writing the short native feature string failed on 0.4%. The failures were
overwhelmingly syntax, not content: backslash-escaped quotes, one redundant outer
bracket, and the prompt's own placeholder echoed back between the answer tags.

This module makes the wire format a choice rather than a constant, so the probe measures
world modelling instead of JSON hygiene. Each GridFormat renders a grid for the prompt's
worked examples, states its own output contract, and parses an answer back.

Parsers are deliberately tolerant. They accept the answer with or without its tags, with
an unclosed tag, wrapped in a code fence, preceded by prose, or with the instruction's
placeholder echoed ahead of the real answer, and each format additionally absorbs its own
common corruptions. Tolerance never invents cell content: a parsed answer must supply
exactly the target's cells or it fails, so a repaired answer is still scored against the
untouched target by the usual metrics.
"""
from __future__ import annotations

import json
import re

# Legend characters for the palette format: unambiguous, never quoted or escaped.
PALETTE_CHARS = ".#o+x*=~abcdefghijklmnpqrstuvwyz0123456789"
PLACEHOLDER = re.compile(r"^(?:[A-Z][A-Z ]*[A-Z]|[A-Z]+)$")

# Sparse-format notations observed from the evaluator, all unambiguous: a cell as
# "r,c,colour" / "r c colour" / "(r, c, colour)" / "r,c:colour", a colour-grouped run
# "colour:r,c r,c", and redundant "dim:9x9" metadata.
COLOUR = r"[A-Za-z][A-Za-z_-]*"
CELL_ENTRY = re.compile(rf"(\d+)\s*[,\s]\s*(\d+)\s*[,:\s]\s*({COLOUR})")
GROUPED_CELLS = re.compile(rf"({COLOUR})\s*[:=]\s*((?:\d+\s*[,\s]\s*\d+\s*[,\s]*)+)")
COORDINATE_PAIR = re.compile(r"(\d+)\s*[,\s]\s*(\d+)")
BACKGROUND = re.compile(rf"\bbg\s*=\s*({COLOUR})")
DIMENSIONS = re.compile(r"\b(?:dim|dims|size|shape|grid)\s*[:=]\s*\d+\s*[xX*]\s*\d+", re.I)


def _strip_fence(text: str) -> str:
    match = re.fullmatch(r"```(?:[a-zA-Z]*)?\s*\n?(.*?)\n?```", text.strip(), re.DOTALL)
    return match[1].strip() if match else text.strip()


def answer_candidates(response: str, tag: str) -> list[str]:
    """Every plausible answer body, best-first; parsers try each in turn."""
    text = (response or "").strip()
    out: list[str] = []
    for body in re.findall(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL):
        body = _strip_fence(body)
        if body and not PLACEHOLDER.match(body):
            out.append(body)
    if f"</{tag}>" in text:
        out.append(_strip_fence(text.rsplit(f"</{tag}>", 1)[-1]))
    if f"<{tag}>" in text:
        out.append(_strip_fence(text.rsplit(f"<{tag}>", 1)[-1].replace(f"</{tag}>", "")))
    out.append(_strip_fence(re.sub(rf"</?{tag}>", " ", text)))
    seen, ordered = set(), []
    for body in out:
        body = body.strip()
        if body and body not in seen:
            seen.add(body)
            ordered.append(body)
    return ordered


def _shape(grid: list[list[str]]) -> tuple[int, int]:
    return len(grid), len(grid[0])


def _reflow(cells: list[str], shape: tuple[int, int]) -> list[list[str]]:
    """Regroup a flat cell sequence into the target shape; never pads or truncates."""
    rows, cols = shape
    if len(cells) != rows * cols:
        raise ValueError(f"expected {rows * cols} cells, got {len(cells)}")
    return [cells[r * cols:(r + 1) * cols] for r in range(rows)]


def _check(grid: list[list[str]], shape: tuple[int, int]) -> list[list[str]]:
    if not grid or not grid[0]:
        raise ValueError("expected a nonempty grid")
    width = len(grid[0])
    if any(len(row) != width or any(not isinstance(c, str) or not c for c in row)
           for row in grid):
        raise ValueError("expected a rectangular colour-name grid")
    if _shape(grid) != shape:
        raise ValueError("predicted grid dimensions differ from target")
    return grid


class GridFormat:
    """One wire format: how examples are rendered, requested, and read back."""

    name = "base"
    schema_line = ""

    def render(self, grid: list[list[str]], background: str,
               colours: list[str] | None = None) -> str:
        raise NotImplementedError

    def contract(self, shape: tuple[int, int], background: str,
                 colours: list[str]) -> str:
        """The output contract, restated next to the answer tag."""
        raise NotImplementedError

    def legend(self, colours: list[str]) -> dict[str, str]:
        return {}

    def parse_body(self, body: str, shape: tuple[int, int], background: str,
                   colours: list[str]) -> list[list[str]]:
        raise NotImplementedError

    def parse(self, response: str, tag: str, shape: tuple[int, int], *,
              background: str = "black", colours: list[str] | None = None) -> list[list[str]]:
        colours = colours or []
        errors = []
        for body in answer_candidates(response, tag):
            try:
                return _check(self.parse_body(body, shape, background, colours), shape)
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                errors.append(str(exc))
        raise ValueError(errors[0] if errors else "empty prediction")


class JsonFormat(GridFormat):
    """The frozen baseline: a JSON array of rows of colour strings."""

    name = "json"
    schema_line = ("The raw output format is a JSON array of rows of colour-name strings; "
                   "coordinates, when present, are zero-based (row, column).")

    def render(self, grid, background, colours=None):
        # Compact separators, matching probing_common.json_text and the frozen prompts.
        return json.dumps(grid, ensure_ascii=False, separators=(",", ":"))

    def contract(self, shape, background, colours):
        rows, cols = shape
        return (f"a JSON array of {rows} rows of {cols} colour-name strings, "
                f'like [["{background}","{background}"],["{background}","{background}"]]')

    def parse_body(self, body, shape, background, colours):
        attempts = [body]
        if '\\"' in body:  # the model wrote the array as an escaped JSON string
            attempts.append(body.replace('\\"', '"'))
        opener = re.search(r"\[\s*\[", body)  # tolerate prose before a pretty-printed grid
        if opener and opener.start() > 0:
            attempts.append(body[opener.start():].replace('\\"', '"'))
        errors = []
        for attempt in attempts:
            attempt = re.sub(r",\s*([\]}])", r"\1", attempt.strip())  # trailing commas
            try:
                value, _ = json.JSONDecoder().raw_decode(attempt)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            # A redundant outer bracket: [[[...]]] instead of [[...]].
            if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list) \
                    and value[0] and isinstance(value[0][0], list):
                value = value[0]
            if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                return _reflow(value, shape)  # rows flattened into one array
            if not isinstance(value, list):
                errors.append("expected a nonempty grid")
                continue
            return [list(row) if isinstance(row, list) else row for row in value]
        raise ValueError(errors[0] if errors else "expected a nonempty grid")


class JsonRowsFormat(GridFormat):
    """One JSON row per line, each labelled with its row index.

    Dropping exactly one row is the dominant JSON failure: 16 of 21 tolerant-parse
    failures in the bake-off were an N-1 row answer at full width. No parser can repair
    that without inventing cells, so the index is a counting scaffold -- writing "15:"
    is a harder thing to skip than silently emitting one row too few.
    """

    name = "json_rows"
    schema_line = ("The raw output format is one line per grid row, each written as "
                   "ROW: followed by a JSON array of colour-name strings; coordinates, "
                   "when present, are zero-based (row, column).")

    def render(self, grid, background, colours=None):
        return "\n".join(f"{r}: " + json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                         for r, row in enumerate(grid))

    def contract(self, shape, background, colours):
        rows, cols = shape
        return (f"{rows} lines, each ROW: followed by a JSON array of {cols} colour-name "
                f'strings, e.g. 0: ["{background}","{background}"] -- with the row indices '
                f"running 0 to {rows - 1} in order and every index present")

    def parse_body(self, body, shape, background, colours):
        indexed: dict[int, list] = {}
        positional: list[list] = []
        for line in body.splitlines():
            label = re.match(r"\s*(\d+)\s*[:.)]\s*(?=[\[\"'])", line)
            text = line[label.end():] if label else line.strip()
            text = re.sub(r",\s*$", "", text.strip()).replace('\\"', '"')
            if not text.startswith("["):
                continue
            try:
                row, _ = json.JSONDecoder().raw_decode(re.sub(r",\s*\]", "]", text))
            except ValueError:
                continue
            if not isinstance(row, list) or not row:
                continue
            if label:
                indexed[int(label[1])] = row
            else:
                positional.append(row)
        if indexed:
            if set(indexed) != set(range(shape[0])):  # a labelled row is missing
                missing = sorted(set(range(shape[0])) - set(indexed))
                raise ValueError(f"missing grid rows {missing}")
            return [indexed[r] for r in range(shape[0])]
        if not positional:
            raise ValueError("expected a nonempty grid")
        return positional


class LinesFormat(GridFormat):
    """One row per line, bare colour names separated by spaces. No quotes or brackets."""

    name = "lines"
    schema_line = ("The raw output format is one line per grid row, with the row's colour "
                   "names separated by single spaces; coordinates, when present, are "
                   "zero-based (row, column).")

    def render(self, grid, background, colours=None):
        return "\n".join(" ".join(row) for row in grid)

    def contract(self, shape, background, colours):
        rows, cols = shape
        return (f"{rows} lines of {cols} colour names separated by single spaces, "
                "with no quotes, commas or brackets")

    def parse_body(self, body, shape, background, colours):
        rows = []
        for line in body.splitlines():
            line = re.sub(r"^\s*(?:row\s*)?\d+\s*[:.)]\s*", "", line, flags=re.I)  # "row 3:"
            cells = [c for c in re.split(r"[\s,;|]+", line.strip(" \t[]\"'")) if c]
            cells = [c.strip("\"'[],;") for c in cells]
            cells = [c for c in cells if c and not c.isdigit()]
            if cells:
                rows.append(cells)
        if not rows:
            raise ValueError("expected a nonempty grid")
        if _shape(rows) != shape and sum(len(r) for r in rows) == shape[0] * shape[1]:
            return _reflow([c for row in rows for c in row], shape)  # wrapped differently
        return rows


class PaletteFormat(GridFormat):
    """A one-character-per-cell picture over a stated legend: the shortest answer."""

    name = "palette"
    schema_line = ("The raw output format is a picture of the grid, one character per "
                   "cell over the supplied LEGEND; coordinates, when present, are "
                   "zero-based (row, column).")

    def legend(self, colours):
        """Character -> colour, fixed per game so every example shares one legend."""
        if not colours:
            raise ValueError("the palette format needs the game's colour vocabulary")
        if len(colours) > len(PALETTE_CHARS):
            raise ValueError("palette larger than the legend alphabet")
        return {PALETTE_CHARS[i]: colour for i, colour in enumerate(colours)}

    def legend_text(self, colours) -> str:
        return " ".join(f"{ch}={colour}" for ch, colour in self.legend(colours).items())

    def render(self, grid, background, colours=None):
        codes = {v: k for k, v in self.legend(colours).items()}
        missing = {cell for row in grid for cell in row} - set(codes)
        if missing:
            raise ValueError(f"colours outside the legend: {sorted(missing)}")
        return "\n".join("".join(codes[cell] for cell in row) for row in grid)

    def contract(self, shape, background, colours):
        rows, cols = shape
        return (f"{rows} lines of exactly {cols} legend characters, with no spaces, "
                f"quotes or colour names, using LEGEND {self.legend_text(colours)}")

    def parse_body(self, body, shape, background, colours):
        table = self.legend(colours)
        table.update({ch: colour for ch, colour in  # a legend restated in the answer
                      re.findall(r"([^\s=])\s*=\s*([A-Za-z]+)", body)})
        allowed = set(table)
        rows = []
        for line in body.splitlines():
            line = re.sub(r"^\s*(?:row\s*)?\d+\s*[:.)]\s*", "", line, flags=re.I)
            if "=" in line:  # a legend line, not a picture line
                continue
            chars = [c for c in line.strip() if not c.isspace()]
            if chars and all(c in allowed for c in chars):
                rows.append([table[c] for c in chars])
        if not rows:
            raise ValueError("expected a nonempty grid")
        if _shape(rows) != shape and sum(len(r) for r in rows) == shape[0] * shape[1]:
            return _reflow([c for row in rows for c in row], shape)
        return rows


class SparseFormat(GridFormat):
    """A background plus the non-background cells: the native format's shape."""

    name = "sparse"
    schema_line = ("The raw output format names the background colour and then lists only "
                   "the cells that differ from it as row,column,colour; coordinates are "
                   "zero-based (row, column).")

    def render(self, grid, background, colours=None):
        cells = [f"{r},{c},{cell}" for r, row in enumerate(grid)
                 for c, cell in enumerate(row) if cell != background]
        return f"bg={background} cells:" + ";".join(cells)

    def contract(self, shape, background, colours):
        rows, cols = shape
        return (f"bg=COLOUR followed by cells:ROW,COL,COLOUR;... for every cell of the "
                f"{rows}x{cols} grid that differs from the background, separated by "
                "semicolons and ordered by row then column")

    def parse_body(self, body, shape, background, colours):
        match = BACKGROUND.search(body)
        fill = match[1] if match else background
        rest = body.split("cells:", 1)[1] if "cells:" in body else body
        rest = BACKGROUND.sub(" ", DIMENSIONS.sub(" ", rest))
        grid = [[fill] * shape[1] for _ in range(shape[0])]
        found = False
        for entry in re.split(r"[;\n|]+", rest):
            entry = entry.strip().strip("()[] ,")
            if not entry:
                continue
            # "blue:0,8 4,8 5,5" -- one colour followed by its coordinate pairs.
            group = GROUPED_CELLS.fullmatch(entry)
            if group:
                for row, col in COORDINATE_PAIR.findall(group[2]):
                    _place(grid, int(row), int(col), group[1], shape)
                    found = True
                continue
            # "7,7,red", "(7, 7, red)", "7 7 red" or "7,7:red", possibly several in a row.
            for row, col, colour in CELL_ENTRY.findall(entry):
                _place(grid, int(row), int(col), colour, shape)
                found = True
        if not found and "cells:" not in body:
            raise ValueError("expected a nonempty grid")
        return grid


def _place(grid, row: int, col: int, colour: str, shape: tuple[int, int]) -> None:
    if not (0 <= row < shape[0] and 0 <= col < shape[1]):
        raise ValueError(f"cell ({row},{col}) is outside the {shape[0]}x{shape[1]} grid")
    grid[row][col] = colour


def colour_vocabulary(grids, background: str) -> list[str]:
    """The game's colour list, background first so it always takes the same legend char."""
    seen = {cell for grid in grids for row in grid for cell in row} | {background}
    return [background] + sorted(seen - {background})


FORMATS: dict[str, GridFormat] = {fmt.name: fmt for fmt in
                                  (JsonFormat(), JsonRowsFormat(), LinesFormat(),
                                   PaletteFormat(), SparseFormat())}
# Chosen by analysis/probing/format_bakeoff.py on matched frozen queries. `sparse` scores
# higher overall (0.950/0.625 against 0.817/0.618) and costs a third of the output tokens,
# but it is a coordinate list like the learned representations themselves, so restating the
# input is cheap to write -- and half of the forward targets are static, where restating the
# input scores 1.0. On targets that actually moved, the part that needs a world model,
# `json_rows` leads: 0.537 against sparse's 0.483 and json's 0.387. It is a dense format, so
# it has no such shortcut, and it still answers validly 98.3%/96.7% of the time against
# json's 88.3%/76.7%. We would rather pay the tokens than score an echo.
DEFAULT_FORMAT = "json_rows"


def get_format(name: str) -> GridFormat:
    if name not in FORMATS:
        raise ValueError(f"unknown grid format {name!r}; known: {sorted(FORMATS)}")
    return FORMATS[name]


class GridContract:
    """A wire format bound to one game's grid shape, background and colour vocabulary.

    Prompt builders and the scorer share this object so the contract shown to the model
    and the parser applied to its answer can never drift apart.
    """

    def __init__(self, name: str, shape, background: str, colours):
        self.format = get_format(name)
        self.name = name
        self.shape = (int(shape[0]), int(shape[1]))
        self.background = background
        self.colours = list(colours)

    @classmethod
    def for_grids(cls, name: str, grids, background: str) -> "GridContract":
        grids = list(grids)
        if not grids:
            raise ValueError("a grid contract needs at least one grid")
        return cls(name, _shape(grids[0]), background,
                   colour_vocabulary(grids, background))

    def render(self, grid) -> str:
        return self.format.render(grid, self.background, self.colours)

    @property
    def schema_line(self) -> str:
        return self.format.schema_line

    @property
    def contract_line(self) -> str:
        return self.format.contract(self.shape, self.background, self.colours)

    def parse(self, response: str, tag: str) -> list[list[str]]:
        return self.format.parse(response, tag, self.shape,
                                 background=self.background, colours=self.colours)

    def as_dict(self) -> dict:
        return {"format": self.name, "shape": list(self.shape),
                "background": self.background, "colours": self.colours}

    @classmethod
    def from_dict(cls, value: dict) -> "GridContract":
        return cls(value["format"], value["shape"], value["background"], value["colours"])
