"""Tests for diff_utils."""

from diff_utils import (
    apply_unified_diff,
    looks_like_unified_diff,
    parse_unified_diff,
    strip_code_fences,
)


def test_strip_code_fences_diff():
    text = "```diff\n@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n```"
    assert strip_code_fences(text) == "@@ -1,3 +1,3 @@\n a\n-b\n+B\n c"


def test_strip_code_fences_python():
    text = "```python\ndef foo():\n    pass\n```"
    assert strip_code_fences(text) == "def foo():\n    pass"


def test_strip_code_fences_none():
    assert strip_code_fences("hello\nworld") == "hello\nworld"


def test_looks_like_unified_diff():
    assert looks_like_unified_diff("@@ -1,3 +1,3 @@\n x")
    assert not looks_like_unified_diff("def perceive():\n    pass")
    assert not looks_like_unified_diff("")


def test_parse_single_hunk():
    patch = "@@ -1,3 +1,3 @@\n a\n-b\n+B\n c"
    hunks = parse_unified_diff(patch)
    assert len(hunks) == 1
    assert hunks[0].before == "a\nb\nc"
    assert hunks[0].after == "a\nB\nc"


def test_parse_multiple_hunks():
    patch = "@@ -1 +1 @@\n-old1\n+new1\n@@ -10 +10 @@\n-old2\n+new2"
    hunks = parse_unified_diff(patch)
    assert len(hunks) == 2
    assert hunks[0].before == "old1"
    assert hunks[0].after == "new1"
    assert hunks[1].before == "old2"
    assert hunks[1].after == "new2"


def test_parse_with_file_headers():
    patch = (
        "--- a/perception.py\n"
        "+++ b/perception.py\n"
        "@@ -1,3 +1,3 @@\n a\n-b\n+B\n c"
    )
    hunks = parse_unified_diff(patch)
    assert len(hunks) == 1
    assert hunks[0].before == "a\nb\nc"


def test_apply_simple_replacement():
    source = "alpha\nbeta\ngamma\n"
    patch = "@@ -1,3 +1,3 @@\n alpha\n-beta\n+BETA\n gamma"
    result, err = apply_unified_diff(source, patch)
    assert err is None
    assert result == "alpha\nBETA\ngamma\n"


def test_apply_ignores_wrong_line_numbers():
    """Hunk header line numbers are ignored; matching is by content."""
    source = "alpha\nbeta\ngamma\n"
    patch = "@@ -999,3 +999,3 @@\n alpha\n-beta\n+BETA\n gamma"
    result, err = apply_unified_diff(source, patch)
    assert err is None
    assert result == "alpha\nBETA\ngamma\n"


def test_apply_pure_insertion():
    source = "alpha\nbeta\ngamma"
    patch = "@@ -3 +3,2 @@\n gamma\n+delta"
    result, err = apply_unified_diff(source, patch)
    assert err is None
    assert "delta" in result
    assert result.startswith("alpha\nbeta\ngamma")


def test_apply_addition_only_hunk_appends():
    source = "alpha\nbeta"
    patch = "@@ -2 +2,2 @@\n+gamma"
    result, err = apply_unified_diff(source, patch)
    assert err is None
    assert result.endswith("gamma")


def test_apply_no_match():
    source = "alpha\nbeta\ngamma\n"
    patch = "@@ -1,3 +1,3 @@\n alpha\n-DELTA\n+EPSILON\n gamma"
    result, err = apply_unified_diff(source, patch)
    assert result is None
    assert err is not None
    assert "not found" in err


def test_apply_ambiguous_match():
    source = "x\nx\nx\n"
    patch = "@@ -1 +1 @@\n-x\n+y"
    result, err = apply_unified_diff(source, patch)
    assert result is None
    assert err is not None
    assert "ambiguous" in err.lower() or "matches" in err.lower()


def test_apply_empty_patch():
    source = "alpha\n"
    result, err = apply_unified_diff(source, "")
    assert result is None
    assert err is not None
    assert "No hunks" in err


def test_apply_multiple_hunks_in_order():
    source = "line1\nline2\nline3\nline4\nline5\n"
    patch = (
        "@@ -1,2 +1,2 @@\n"
        " line1\n"
        "-line2\n"
        "+LINE2\n"
        "@@ -4,2 +4,2 @@\n"
        " line4\n"
        "-line5\n"
        "+LINE5"
    )
    result, err = apply_unified_diff(source, patch)
    assert err is None
    assert "LINE2" in result and "LINE5" in result


def test_apply_blank_context_line_without_leading_space():
    """Some LLMs drop the leading space on blank context lines."""
    source = "alpha\n\nbeta\n"
    patch = "@@ -1,3 +1,3 @@\n alpha\n\n-beta\n+BETA"
    result, err = apply_unified_diff(source, patch)
    assert err is None
    assert result == "alpha\n\nBETA\n"


if __name__ == "__main__":
    import sys

    failed = 0
    test_fns = [obj for name, obj in globals().items() if name.startswith("test_")]
    for fn in test_fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {fn.__name__}: {e}")
            failed += 1
    print(f"\n{len(test_fns) - failed}/{len(test_fns)} passed")
    sys.exit(0 if failed == 0 else 1)
