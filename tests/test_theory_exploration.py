"""Unit tests for theory_exploration: MI math, weighting, and lenient parsing."""

import math

from explore.theory_exploration import (
    Theory,
    assign_rank_weights,
    mutual_information,
    parse_crux_questions,
    parse_theories,
)


def _approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_mi_zero_when_theories_agree():
    # All theories predict YES -> answer is uninformative about theory identity.
    w = [0.5, 0.3, 0.2]
    assert _approx(mutual_information(w, [1.0, 1.0, 1.0]), 0.0)
    assert _approx(mutual_information(w, [0.0, 0.0, 0.0]), 0.0)


def test_mi_max_for_even_split_two_theories():
    # Two equally-weighted theories that disagree -> answer fully determines
    # which theory holds -> MI == H(prior) == 1 bit.
    w = [0.5, 0.5]
    assert _approx(mutual_information(w, [1.0, 0.0]), 1.0)


def test_mi_split_less_than_prior_entropy():
    # A question that separates {T1} from {T2,T3} cannot exceed prior entropy
    # and should be positive.
    w = [1 / 3, 1 / 3, 1 / 3]
    h_prior = math.log2(3)
    mi = mutual_information(w, [1.0, 0.0, 0.0])
    assert 0.0 < mi < h_prior


def test_mi_unknown_is_agnostic():
    # A theory that is agnostic (0.5) contributes no separation by itself.
    w = [0.5, 0.5]
    assert _approx(mutual_information(w, [0.5, 0.5]), 0.0)


def test_rank_weights_normalize_and_decay():
    ts = [Theory(world_knowledge="a", rank=1),
          Theory(world_knowledge="b", rank=2),
          Theory(world_knowledge="c", rank=3)]
    assign_rank_weights(ts, decay=0.5)
    assert _approx(sum(t.weight for t in ts), 1.0)
    # Monotonically decreasing with rank.
    assert ts[0].weight > ts[1].weight > ts[2].weight
    # decay=0.5 -> raw 1, .5, .25 -> normalized 4/7, 2/7, 1/7.
    assert _approx(ts[0].weight, 4 / 7)


def test_rank_weights_uniform_when_decay_one():
    ts = [Theory(world_knowledge="a", rank=1), Theory(world_knowledge="b", rank=2)]
    assign_rank_weights(ts, decay=1.0)
    assert _approx(ts[0].weight, 0.5) and _approx(ts[1].weight, 0.5)


def test_parse_theories_lenient_with_malformed_tags():
    # Mirrors the malformed closing tags actually emitted by gemini in
    # scripts/simulate_theories.py output (e.g. </rationate>, </ration-ale>).
    text = """
<theories>
<theory rank="1" likelihood="Very High">
<world_knowledge>
- Match each grid to the center pattern.
- Completing all grids wins.
</world_knowledge>
<rationale>Pattern matching is common.</rationate>
</theory>
<theory rank="2" likelihood="High">
<world_knowledge>
- Turn all perimeter squares red.
</world_knowledge>
<rationale>Alternative.</ration-ale>
</theory>
</theories>
"""
    ts = parse_theories(text)
    assert len(ts) == 2
    assert ts[0].rank == 1 and "center pattern" in ts[0].world_knowledge
    assert "Pattern matching" in ts[0].rationale  # recovered despite bad close tag
    assert ts[1].rank == 2 and "perimeter squares red" in ts[1].world_knowledge


def test_parse_theories_skips_block_without_world_knowledge():
    text = "<theory rank='1'><rationale>no wk here</rationale></theory>"
    assert parse_theories(text) == []


def test_parse_crux_questions():
    text = """
<crux_questions>
<q>Does matching a grid to its center pattern lock it as solved?</q>
<q>Does turning the whole perimeter red complete a level?</q>
</crux_questions>
"""
    qs = parse_crux_questions(text)
    assert len(qs) == 2
    assert qs[0].startswith("Does matching")


def test_parse_crux_questions_with_nested_question_tag():
    text = "<q><question>Is the center a target?</question></q>"
    qs = parse_crux_questions(text)
    assert qs == ["Is the center a target?"]


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
