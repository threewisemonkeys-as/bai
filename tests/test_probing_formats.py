"""Round-trip and tolerance tests for the raw-grid wire formats.

The tolerance cases are the corruptions actually observed in the final-selected probing
run (analysis/probing/final_selected/results.jsonl), which the frozen JSON-only parser
scored as zero: escaped quotes, a redundant outer bracket, an echoed instruction
placeholder, unclosed tags, and code fences.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "offline_learning"))

from probing_common import forward_prompt, grid_scores, reconstruction_prompt  # noqa: E402
from probing_formats import (  # noqa: E402
    DEFAULT_FORMAT, FORMATS, GridContract, JsonFormat, LinesFormat, PaletteFormat,
    SparseFormat, colour_vocabulary, get_format,
)

GRID = [
    ["black", "black", "black"],
    ["black", "red", "black"],
    ["gold", "black", "blue"],
]
SHAPE = (3, 3)
COLOURS = colour_vocabulary([GRID], "black")
TAG = "next_state"


def parse(fmt, text):
    return fmt.parse(text, TAG, SHAPE, background="black", colours=COLOURS)


@pytest.mark.parametrize("name", sorted(FORMATS))
def test_round_trip(name):
    """Rendering a grid and parsing it back is the identity, for every format."""
    fmt = get_format(name)
    body = fmt.render(GRID, "black", COLOURS)
    assert parse(fmt, f"<{TAG}>{body}</{TAG}>") == GRID


@pytest.mark.parametrize("name", sorted(FORMATS))
def test_tag_damage(name):
    """A missing, unclosed or fenced tag still yields the grid."""
    fmt = get_format(name)
    body = fmt.render(GRID, "black", COLOURS)
    assert parse(fmt, body) == GRID
    assert parse(fmt, f"<{TAG}>{body}") == GRID
    assert parse(fmt, f"Here it is:\n<{TAG}>\n{body}\n</{TAG}>\nDone.") == GRID
    assert parse(fmt, f"<{TAG}>\n```\n{body}\n```\n</{TAG}>") == GRID


@pytest.mark.parametrize("name", sorted(FORMATS))
def test_placeholder_echo(name):
    """The observed failure where the contract is echoed and the answer follows it."""
    fmt = get_format(name)
    body = fmt.render(GRID, "black", COLOURS)
    text = f"<{TAG}>COMPLETE JSON GRID</{TAG}>\n{body}"
    assert parse(fmt, text) == GRID


@pytest.mark.parametrize("name", ["json", "lines", "palette"])
def test_wrong_shape_is_rejected(name):
    """Tolerance never invents cells: a short answer must fail, not be padded."""
    fmt = get_format(name)
    short = fmt.render(GRID[:2], "black", COLOURS)
    with pytest.raises(ValueError):
        parse(fmt, f"<{TAG}>{short}</{TAG}>")


def test_sparse_takes_its_shape_from_the_target():
    """The sparse format cannot fail on shape, so a dropped cell is wrong, not invalid.

    That removes the shape-mismatch failure mode the row-based formats have, at the cost
    of turning a truncated answer into a scored wrong answer instead of a parse error.
    """
    fmt = SparseFormat()
    truncated = fmt.render(GRID, "black")[: -len(";2,2,blue")]
    parsed = parse(fmt, truncated)
    assert parsed != GRID
    assert parsed[2][2] == "black"  # the dropped cell falls back to the background


@pytest.mark.parametrize("name", sorted(FORMATS))
def test_empty_answer_is_rejected(name):
    fmt = get_format(name)
    with pytest.raises(ValueError):
        parse(fmt, "")


def test_json_escaped_quotes():
    """1456 forward and 609 reconstruction answers were escaped exactly like this."""
    body = json.dumps(GRID).replace('"', '\\"')
    assert parse(JsonFormat(), f"<{TAG}>{body}</{TAG}>") == GRID


def test_json_extra_outer_bracket():
    body = json.dumps([GRID])
    assert parse(JsonFormat(), f"<{TAG}>{body}</{TAG}>") == GRID


def test_json_trailing_comma_and_flat_rows():
    assert parse(JsonFormat(), f'<{TAG}>{json.dumps(GRID)[:-1]},]</{TAG}>') == GRID
    flat = json.dumps([cell for row in GRID for cell in row])
    assert parse(JsonFormat(), f"<{TAG}>{flat}</{TAG}>") == GRID


def test_json_pretty_printed_with_prose():
    body = json.dumps(GRID, indent=2)
    assert parse(JsonFormat(), f"The grid is:\n{body}") == GRID


def test_lines_separator_and_row_labels():
    fmt = LinesFormat()
    assert parse(fmt, "black, black, black\nblack, red, black\ngold, black, blue") == GRID
    labelled = "row 0: black black black\nrow 1: black red black\nrow 2: gold black blue"
    assert parse(fmt, labelled) == GRID
    quoted = '"black" "black" "black"\n"black" "red" "black"\n"gold" "black" "blue"'
    assert parse(fmt, quoted) == GRID


def test_lines_reflow_when_wrapped_differently():
    """All nine cells on one line still regroups to 3x3; no cell is invented."""
    assert parse(LinesFormat(), " ".join(c for row in GRID for c in row)) == GRID


def test_palette_legend_is_stable_and_tolerant():
    fmt = PaletteFormat()
    assert fmt.legend(COLOURS)["."] == "black"  # background always takes the first char
    body = fmt.render(GRID, "black", COLOURS)
    assert parse(fmt, f"LEGEND: {fmt.legend_text(COLOURS)}\n{body}") == GRID
    assert parse(fmt, "\n".join(" ".join(line) for line in body.splitlines())) == GRID


def test_palette_rejects_colours_outside_the_legend():
    with pytest.raises(ValueError):
        PaletteFormat().render([["green"] * 3] * 3, "black", COLOURS)


def test_sparse_coordinate_tolerance():
    fmt = SparseFormat()
    assert parse(fmt, "bg=black cells:1,1,red;2,0,gold;2,2,blue") == GRID
    assert parse(fmt, "bg=black cells:\n(1, 1, red);\n(2, 0, gold);\n(2, 2, blue)") == GRID
    assert parse(fmt, "bg=black cells:1 1 red\n2 0 gold\n2 2 blue") == GRID


def test_sparse_empty_foreground_and_out_of_range():
    fmt = SparseFormat()
    blank = [["black"] * 3 for _ in range(3)]
    assert parse(fmt, "bg=black cells:") == blank
    with pytest.raises(ValueError):
        parse(fmt, "bg=black cells:9,9,red")


def test_sparse_falls_back_to_the_known_background():
    assert parse(SparseFormat(), "cells:1,1,red;2,0,gold;2,2,blue") == GRID


def test_sparse_notation_variants_seen_from_the_evaluator():
    """Notations the model actually used in the bake-off instead of the asked-for one."""
    fmt = SparseFormat()
    assert parse(fmt, "bg=black | 1,1:red; 2,0:gold; 2,2:blue") == GRID  # colon, pipes
    assert parse(fmt, "bg=black; red:1,1; gold:2,0; blue:2,2") == GRID  # colour-grouped
    assert parse(fmt, "bg=black; dim:3x3; red:1,1; gold:2,0; blue:2,2") == GRID  # metadata
    grouped = "bg=black; blue:2,2; red:1,1; gold:2,0"
    assert parse(fmt, f"<{TAG}>{grouped}</{TAG}>") == GRID


def test_sparse_grouped_run_of_pairs():
    """One colour covering several cells: "colour:r,c r,c r,c"."""
    expected = [["black", "black", "black"], ["red", "red", "black"], ["black"] * 3]
    assert parse(SparseFormat(), "bg=black; red:1,0 1,1") == expected


def test_colour_vocabulary_orders_background_first():
    assert colour_vocabulary([GRID], "black") == ["black", "blue", "gold", "red"]
    assert colour_vocabulary([GRID], "white")[0] == "white"


def test_unknown_format_name():
    with pytest.raises(ValueError):
        get_format("yaml")


def test_contract_round_trips_through_json():
    contract = GridContract.for_grids("sparse", [GRID], "black")
    same = GridContract.from_dict(json.loads(json.dumps(contract.as_dict())))
    assert (same.name, same.shape, same.background, same.colours) == (
        contract.name, contract.shape, contract.background, contract.colours)
    assert same.parse(contract.render(GRID), TAG) == GRID


def test_default_format_is_the_bakeoff_winner():
    """`sparse` wins the pooled score, but most of its forward margin is on static targets
    it can echo. The default is the format that wins where a world model is required."""
    assert DEFAULT_FORMAT == "json_rows"


EXAMPLES = [("bg=black cells:1,1,red", json.dumps(GRID))]


def test_frozen_prompt_wording_is_unchanged_without_a_contract():
    """Saved runs must keep reproducing their published numbers, so the frozen
    JSON-only wording -- placeholder included -- has to survive untouched."""
    text = reconstruction_prompt("z", EXAMPLES)
    assert "<reconstruction>COMPLETE JSON GRID</reconstruction>" in text
    assert "a JSON array of rows of colour-name strings" in text
    assert json.dumps(GRID) in text  # the example grid is passed through verbatim
    forward = forward_prompt(["s0", "s1"], ["noop"], ["left"], "k", EXAMPLES)
    assert "Return only <next_state>YOUR PREDICTION</next_state>." in forward
    assert "as a JSON array of rows of colour strings" in forward


def test_contract_rewrites_both_prompts_and_re_renders_examples():
    contract = GridContract.for_grids("sparse", [GRID], "black")
    text = reconstruction_prompt("z", EXAMPLES, contract)
    assert "COMPLETE JSON GRID" not in text
    assert json.dumps(GRID) not in text  # the example is now in the wire format
    assert contract.render(GRID) in text
    assert "row,column,colour" in text
    forward = forward_prompt(["s0", "s1"], ["noop"], ["left"], "k", EXAMPLES,
                             contract=contract)
    assert contract.render(GRID) in forward
    assert "nothing else between the tags" in forward


def test_native_forecasts_reject_a_raw_grid_contract():
    contract = GridContract.for_grids("sparse", [GRID], "black")
    with pytest.raises(ValueError):
        forward_prompt(["s0"], [], ["left"], "k", [], native=True, contract=contract)


def test_deterministic_baseline_scores_perfectly_in_every_format():
    """A copy/constant control answers with a rendered grid, not raw JSON; if the
    contract did not render it too, every deterministic baseline would score zero."""
    target = json.dumps(GRID)
    for name in sorted(FORMATS):
        contract = GridContract.for_grids(name, [GRID], "black")
        preset = contract.render(GRID)  # what probing_eval stores as preset_response
        scores = grid_scores(preset, target, background="black", contract=contract)
        assert scores["parse_error"] is None, name
        assert scores["exact"] == 1.0, name


def test_grid_scores_uses_the_contract_parser():
    contract = GridContract.for_grids("sparse", [GRID], "black")
    target = json.dumps(GRID)
    answer = f"<{TAG}>{contract.render(GRID)}</{TAG}>"
    assert grid_scores(answer, target, background="black", contract=contract)["exact"] == 1.0
    # The same answer is unreadable to the frozen strict JSON parser.
    assert grid_scores(answer, target, background="black")["parse_error"] is not None


def test_json_rows_round_trip_and_tolerance():
    fmt = get_format("json_rows")
    assert parse(fmt, fmt.render(GRID, "black")) == GRID
    escaped = fmt.render(GRID, "black").replace('"', '\\"')
    assert parse(fmt, escaped) == GRID
    assert parse(fmt, fmt.render(GRID, "black").replace(":", ".")) == GRID  # "0." labels
    unlabelled = "\n".join(json.dumps(row) for row in GRID)  # indices dropped entirely
    assert parse(fmt, unlabelled) == GRID


def test_json_rows_names_the_missing_row_instead_of_padding():
    """The dominant JSON failure is a dropped row; labelling makes it explicit."""
    fmt = get_format("json_rows")
    short = "\n".join(l for l in fmt.render(GRID, "black").splitlines() if not l.startswith("1:"))
    with pytest.raises(ValueError, match=r"missing grid rows \[1\]"):
        parse(fmt, short)
