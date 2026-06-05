from stepwise_eb_learn_improve import (
    _iter_q_blocks_with_refs,
    _normalize_question_index,
    _parse_1_based_indices,
    _parse_q_tag_indices,
)


def test_question_refs_must_be_exact_bank_labels():
    assert _normalize_question_index("Q4", 10) == 3
    assert _normalize_question_index("4", 10) == 3
    assert _normalize_question_index(" Q4 ", 10) == 3

    assert _normalize_question_index("Q_new4", 10) is None
    assert _normalize_question_index("Q4_new", 10) is None
    assert _normalize_question_index("new question 4", 10) is None


def test_q_tag_parser_rejects_generated_labels():
    text = """
    <selected_experiments>
    <q n="Q_new1"><experiment_plan>bad</experiment_plan></q>
    <q n="Q4"><experiment_plan>good</experiment_plan></q>
    <q n="Q4_new"><experiment_plan>bad</experiment_plan></q>
    </selected_experiments>
    """

    assert _parse_q_tag_indices(text, 10) == [3]
    assert _iter_q_blocks_with_refs(text, 10) == [
        (None, "Q_new1", "<experiment_plan>bad</experiment_plan>"),
        (3, "Q4", "<experiment_plan>good</experiment_plan>"),
        (None, "Q4_new", "<experiment_plan>bad</experiment_plan>"),
    ]


def test_freeform_index_parser_rejects_digits_inside_generated_labels():
    text = "Prefer Q_new1 and Q4_new, then Q6 or 7."
    assert _parse_1_based_indices(text, 10) == [5, 6]
